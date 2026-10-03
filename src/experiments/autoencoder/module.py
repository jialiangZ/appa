r"""Autoencoder training as a Lightning module."""

__all__ = [
    "AETrainingModule",
]

import torch

from einops import rearrange
from omegaconf import DictConfig, open_dict
from torch import Tensor

from appa.data.const import (
    CONTEXT_VARIABLES,
    ERA5_ATMOSPHERIC_VARIABLES,
    ERA5_PRESSURE_LEVELS,
    ERA5_SURFACE_VARIABLES,
    SUB_PRESSURE_LEVELS,
)
from appa.diagnostics.plots import plot_image_grid
from appa.models.losses import AELoss
from appa.training.loggers import log_images, logger_image
from appa.training.modules import AppaTrainingModule
from appa.training.optim import get_optimizer
from appa.utils.io import select_ae_architecture


class AETrainingModule(AppaTrainingModule):
    r"""Lightning module for autoencoder training.

    The loss is :py:class:`appa.models.losses.AELoss` plus an optional
    :py:`loss.l2_weight` penalty on the latents. Quantitative validation
    logs the loss and the per-variable RMSE; qualitative validation logs
    ground-truth/reconstruction snapshot grids.

    Arguments:
        cfg: The training configuration (holding the ``ae`` section).
        runid: The run id.
        quali_dataset: The dataset used for qualitative validation snapshots.
    """

    def __init__(self, cfg: DictConfig, runid: str = "", quali_dataset=None):
        super().__init__(cfg, runid=runid)

        pressure_levels = (
            SUB_PRESSURE_LEVELS if cfg.train.sub_pressure_levels else ERA5_PRESSURE_LEVELS
        )
        self.pressure_levels = pressure_levels

        with open_dict(cfg):
            num_surface = len(ERA5_SURFACE_VARIABLES) if ERA5_SURFACE_VARIABLES else 0
            num_atmos = (
                (len(ERA5_ATMOSPHERIC_VARIABLES) * len(pressure_levels))
                if ERA5_ATMOSPHERIC_VARIABLES and pressure_levels
                else 0
            )
            num_context = len(CONTEXT_VARIABLES) if CONTEXT_VARIABLES else 0

            cfg.ae["in_channels"] = num_surface + num_atmos
            cfg.ae["context_channels"] = num_context

        ae_arch = select_ae_architecture(cfg.ae.name)
        self.ae = ae_arch(**cfg.ae)

        self.model_loss = AELoss(
            criterion=cfg.loss.error,
            latitude_weighting=cfg.loss.latitude_weighting,
            level_weighting=cfg.loss.level_weighting,
            levels=pressure_levels,
            N_lat=cfg.ae.shape[0],
            N_lon=cfg.ae.shape[1],
        )

        self.quali_dataset = quali_dataset
        self._valid_rmse = []

    def configure_optimizers(self):
        optimizer, scheduler = get_optimizer(
            params=self.ae.parameters(),
            update_steps=self.cfg.train.update_steps,
            **self.cfg.optim,
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    #
    # Training / validation
    #

    def compute_loss(self, batch) -> Tensor:
        state, context, timestamp = batch

        state = rearrange(state, "B T Z Lat Lon -> (B T) (Lat Lon) Z")
        context = rearrange(context, "B T Z Lat Lon -> (B T) (Lat Lon) Z")
        timestamp = rearrange(timestamp, "B T D -> (B T) D")

        z, x = self.ae(state, timestamp, context)
        loss = self.model_loss(x, state) + self.cfg.loss.l2_weight * (z**2).mean()

        return loss

    def compute_valid_loss(self, batch) -> Tensor:
        state, context, timestamp = batch

        state = rearrange(state, "B T Z Lat Lon -> (B T) (Lat Lon) Z")
        context = rearrange(context, "B T Z Lat Lon -> (B T) (Lat Lon) Z")
        timestamp = rearrange(timestamp, "B T D -> (B T) D")

        _, x = self.ae(state, timestamp, context)
        loss = self.model_loss(state, x)

        rmse = torch.sqrt((x - state).square().mean(dim=1)).mean(dim=0)
        # Keep on-device for sync_dist logging (NCCL does not support CPU tensors).
        self._valid_rmse.append(rmse.detach())

        return loss

    def _valid_extras(self) -> None:
        if len(self._valid_rmse) == 0:
            return

        rmse = torch.stack(self._valid_rmse).mean(0)

        n_surf = len(ERA5_SURFACE_VARIABLES)
        surf_error, atm_error = rmse[:n_surf], rmse[n_surf:]

        for channel_id, var_surf in enumerate(ERA5_SURFACE_VARIABLES):
            self.log(f"valid/surface_rmse/{var_surf}", surf_error[channel_id], sync_dist=True)

        for channel_id, var_atm in enumerate(ERA5_ATMOSPHERIC_VARIABLES):
            for level_id, level in enumerate(self.pressure_levels):
                idx = channel_id * len(self.pressure_levels) + level_id
                self.log(
                    f"valid/atmosphere_rmse/{var_atm}_{level}hPa",
                    atm_error[idx],
                    sync_dist=True,
                )

        self._valid_rmse = []

    #
    # Qualitative validation
    #

    @torch.no_grad()
    def log_snapshots(self, logger) -> None:
        r"""Logs ground-truth/reconstruction snapshot grids to the tracker."""

        import matplotlib.pyplot as plt

        if self.quali_dataset is None:
            return

        n_snapshots = self.cfg.valid.n_valid_snapshots
        snapshots, dates = [], []

        for i in range(len(self.quali_dataset)):
            data, context, timestamp = self.quali_dataset[i]

            lon = data.shape[-1]

            state = rearrange(data[None], "B T Z Lat Lon -> (B T) (Lat Lon) Z")
            state = state.nan_to_num().to(self.device)
            context = rearrange(context[None], "B T Z Lat Lon -> (B T) (Lat Lon) Z")
            context = context.nan_to_num().to(self.device)
            timestamp = rearrange(timestamp[None], "B T D -> (B T) D").to(self.device)

            autocast = self.amp_dtype != torch.float32

            with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=autocast):
                _, x = self.ae(state, timestamp, context)

            x = rearrange(x, "(B T) (Lat Lon) Z -> B T Z Lat Lon", Lon=lon, B=1)
            state = rearrange(state, "(B T) (Lat Lon) Z -> B T Z Lat Lon", Lon=lon, B=1)

            snapshot = torch.cat([state[:, 0, 0, :, :], x[:, 0, 0, :, :]])
            snapshots.append(snapshot.cpu())
            dates.append(timestamp.cpu())

        snapshots = torch.stack(snapshots)
        dates = torch.cat(dates)

        # Order snapshot gt/pred pairs by date and take the first ones.
        dates = [d.tolist() for d in dates]
        dates = [f"{d[0]:04d}/{d[1]:02d}/{d[2]:02d} {d[3]:02d}h" for d in dates]
        dates, snapshots = zip(*sorted(zip(dates, snapshots, strict=False)), strict=False)
        snapshots = torch.stack(snapshots)[:n_snapshots]
        dates = list(dates[:n_snapshots])

        # Set the same vmap for all snapshots
        vmin = snapshots[:, 0].min().item()
        vmax = snapshots[:, 0].max().item()
        snapshots = rearrange(snapshots, "B T ... -> (T B) ...")

        fig = plot_image_grid(
            snapshots,
            shape=(2, snapshots.shape[0] // 2),
            vmin=vmin,
            vmax=vmax,
            x_titles=dates,
            fontsize=24,
            tex_font=False,
        )

        log_images(logger, {"valid/snapshots": logger_image(logger, fig)})
        plt.close(fig)
