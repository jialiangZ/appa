r"""Latent diffusion (denoiser) training as a Lightning module."""

__all__ = [
    "DenoiserTrainingModule",
]

import torch

from einops import rearrange
from functools import partial
from omegaconf import DictConfig, open_dict
from torch import Tensor

from appa.diagnostics.plots import plot_image_grid, plot_spheres_grid
from appa.models.diffusion import DenoiserLoss, create_denoiser, create_schedule
from appa.models.sampling import DDIMSampler
from appa.training.loggers import log_images, logger_image
from appa.training.modules import AppaTrainingModule
from appa.training.optim import get_optimizer
from appa.utils.grids import create_icosphere, latlon_to_xyz


class DenoiserTrainingModule(AppaTrainingModule):
    r"""Lightning module for latent diffusion (denoiser) training.

    The loss is :py:class:`appa.models.diffusion.DenoiserLoss` with random
    noise levels :math:`\sigma_t \sim \mathcal{U}(0, 1)` passed through the
    noise schedule. Quantitative validation logs the loss at random noise
    levels; qualitative validation generates DDIM samples for a few
    validation dates and logs per-channel snapshot grids.

    Arguments:
        cfg: The training configuration (holding the ``backbone`` section).
        ae_cfg: The configuration of the autoencoder run that generated the
            latents.
        runid: The run id.
        valid_dataset: The validation dataset (for qualitative snapshots).
        snapshot_indices: Indices of the validation dates used for snapshots.
    """

    def __init__(
        self,
        cfg: DictConfig,
        ae_cfg: DictConfig,
        runid: str = "",
        valid_dataset=None,
        snapshot_indices=None,
    ):
        super().__init__(cfg, runid=runid)

        self.ae_cfg = ae_cfg
        self.valid_dataset = valid_dataset
        self.snapshot_indices = list(snapshot_indices or [])

        if "graph" in cfg.backbone.name:
            grid, _ = create_icosphere(ae_cfg.ae.ico_divisions[-1])
            x, y, z = latlon_to_xyz(grid).unbind(-1)
        else:
            x, y, z = None, None, None

        self.register_buffer("x", x)
        self.register_buffer("y", y)
        self.register_buffer("z", z)

        self.schedule = create_schedule(cfg.train, device="cpu")

        with open_dict(cfg):
            cfg.backbone["blanket_size"] = cfg.train.blanket_size

        self.denoise = create_denoiser(
            diffusion_cfg=cfg,
            ae_cfg=ae_cfg,
            distributed=False,
            device="cpu",
        )

        self.model = self.denoise.backbone
        self.model_loss = DenoiserLoss(self.denoise, std_min_penalty=cfg.loss.sigma_min_penalty)

    def configure_optimizers(self):
        optimizer, scheduler = get_optimizer(
            params=self.model.parameters(),
            update_steps=self.cfg.train.update_steps,
            **self.cfg.optim,
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    #
    # Training / validation
    #

    def compute_loss(self, batch) -> Tensor:
        data, context = batch

        data = data.flatten(1)
        sigma_t = self.schedule(torch.rand(data.shape[0], device=data.device))

        return self.model_loss(data, sigma_t, date=context)

    #
    # Qualitative validation
    #

    @torch.no_grad()
    def log_snapshots(self, logger) -> None:
        r"""Logs DDIM sample snapshot grids for a few validation dates."""

        import matplotlib.pyplot as plt

        if self.valid_dataset is None or not self.snapshot_indices:
            return

        cfg = self.cfg
        snapshots, dates = [], []

        for idx in self.snapshot_indices:
            gt, context = self.valid_dataset[idx]
            gt, context = gt[None], context[None]
            context = context.to(self.device)

            conditioned_denoiser = partial(self.denoise, date=context)
            sampler = DDIMSampler(
                denoiser=conditioned_denoiser,
                steps=cfg.valid.denoising_steps,
                schedule=self.schedule,
            )

            generated_samples = [gt.to(self.device)]

            for _ in range(cfg.valid.n_valid_samples_per_date):
                sampled_state = sampler(
                    sampler.schedule(torch.ones(1, device=self.device))
                    * torch.randn(
                        1,
                        cfg.train.blanket_size * torch.tensor(gt.shape[-2:]).prod(),
                        device=self.device,
                    )
                ).reshape((-1, cfg.train.blanket_size, *gt.shape[-2:]))

                generated_samples.append(sampled_state)

            snapshots.append(torch.cat(generated_samples).cpu())
            dates.append(context.cpu())

        num_channels = cfg.valid.n_valid_channels
        channels = torch.linspace(0, self.valid_dataset.get_dim() - 1, num_channels).int()
        row_titles = ["GT"] + ["Sample"] * cfg.valid.n_valid_samples_per_date

        metrics = {}

        for i, (valid_blankets, date) in enumerate(zip(snapshots, dates, strict=False)):
            for channel_id in channels:
                valid_blankets_c = valid_blankets[..., channel_id]
                vmin = valid_blankets_c[0].min().item()
                vmax = valid_blankets_c[0].max().item()

                if "graph" in cfg.backbone.name:
                    fig = plot_spheres_grid(
                        blankets=valid_blankets_c,
                        date=date,
                        coordinates=(self.x, self.y, self.z),
                        vmin=vmin,
                        vmax=vmax,
                        row_titles=row_titles,
                    )
                else:
                    H, W = self.model.shape
                    B, K = valid_blankets_c.shape[:2]
                    valid_blankets_c = rearrange(
                        valid_blankets_c, "B K (H W) -> (B K) H W", H=H, W=W
                    )

                    fig = plot_image_grid(
                        valid_blankets_c,
                        shape=(B, K),
                        vmin=vmin,
                        vmax=vmax,
                        fontsize=24,
                        tex_font=False,
                    )

                metrics[f"valid/snapshot_{i}/channel_{channel_id}"] = logger_image(logger, fig)
                plt.close(fig)

        log_images(logger, metrics)
