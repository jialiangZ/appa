r"""Diffusion helpers."""

import logging
import math
import torch
import torch.nn as nn

from einops import rearrange
from functools import partial
from omegaconf import DictConfig, open_dict
from torch import Tensor
from torch import distributed as dist
from typing import Callable, List, Union

from appa.utils.grids import create_icosphere, icosphere_nhops_edges
from appa.utils.math import gmres
from appa.models.nn.sptgraph import SpatioTemporalGraphDiT
from appa.models.nn.triggers import disable_checkpointing, disable_xfa
from appa.models.nn.vit import ViDiT

logger = logging.getLogger(__name__)


class Denoiser(nn.Module):
    r"""Denoiser model with EDM-style preconditioning.

    .. math:: d(x_t) \approx E[x | x_t]

    References:
        | Elucidating the Design Space of Diffusion-Based Generative Models (Karras et al., 2022)
        | https://arxiv.org/abs/2206.00364

    Arguments:
        backbone: A noise conditional network.
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()

        self.backbone = backbone

    def forward(self, xt: Tensor, sigma_t: Tensor, **kwargs) -> Tensor:
        r"""
        Arguments:
            xt: The noisy tensor, with shape :math:`(*, D)`.
            sigma_t: The noise std, with shape :math:`(*, 1)`.
            kwargs: Keyword arguments passed to the backbone.

        Returns:
            The denoised tensor :math:`d(x_t)`, with shape :math:`(*, D)`.
        """

        # [中文注释 | 论文§3.2] 不直接学 score, 而是训练 denoiser d(x_t) ≈ E[x|x_t],
        # 两者经一阶 Tweedie 公式互转: s = (d(x_t) - x_t) / sigma_t^2.
        # 下面的 EDM 预条件系数保证网络输出直接就是 E[x|x_t]; Appa 全程工作在
        # denoiser 空间, score 仅在 MMPSDenoiser 与 sampler 内部隐式出现.

        c_skip = 1 / (sigma_t**2 + 1)
        c_out = sigma_t / torch.sqrt(sigma_t**2 + 1)
        c_in = 1 / torch.sqrt(sigma_t**2 + 1)
        c_noise = 1e1 * torch.log(sigma_t)
        c_noise = c_noise.squeeze(dim=-1)

        x_out = self.backbone(c_in * xt, c_noise, **kwargs)

        return c_skip * xt + c_out * x_out


class MMPSDenoiser(nn.Module):
    r"""
    Creates an MMPS denoiser module.

    References:
        | Learning Diffusion Priors from Observations by Expectation Maximization (Rozet et al., 2024)
        | https://arxiv.org/abs/2405.13712

    Arguments:
        denoiser: A Gaussian denoiser.
        A: The forward operator :math:`x \mapsto Ax`. It should take in a vector :math:`x` of shape :math:`(B, D)` and a "blanket_id" integer, and return a vector of shape :math:`(M, B)`.
        y: An observation :math:`y \sim \mathcal{N}(Ax, \diag(\sigma_y ^ 2))` of shape :math:`(M)`, or a list thereof. If a list, corresponds to the observation for each different blanket.
        var_y: The observation variance matrix :math:`\diag(\sigma_y ^ 2)`, or a list thereof. The variance is assumed diagonal with shape :math:`()`, :math:`(D)` or :math:`(D, D)`.
        tweedie_covariance: Whether to use the Tweedie covariance formula or not.
            If :py:`False`, use :math:`\Sigma_t` instead.
        iterations: The number of solver iterations.
    """

    # [中文注释 | 公式对照, 论文 arXiv:2504.18720 (Appa)]
    #   Eq.(6): 后验 score = 先验 score + 似然 score. 本模块在 denoiser 空间隐式实现:
    #           返回后验 denoiser x_hat + σ_t²·s_obs, sampler 侧按 Tweedie 公式转回
    #           score 时两项自然相加, 故代码中看不到显式的 score 相加.
    #   Eq.(7): p(y|z) ≈ N(y; 𝒜(z), Σ_y), 其中 𝒜 = ℳ ∘ D_ψ → 即实验脚本传入的
    #           self.A (Appa 中为 observations.py 的 observator_partial).
    #   Eq.(8): MMPS 近似 p(y|z_t) ≈ N(y; 𝒜(E[z|z_t]), Σ_y + J·V[z|z_t]·J^T),
    #           J 是 𝒜 的 Jacobian (论文原文用符号 A 表示) → 由 forward 内局部
    #           函数 A/At 以免矩阵的 jvp/vjp 方式实现.

    def __init__(
        self,
        denoiser: Denoiser,
        A: Callable[[Tensor, int], Tensor],
        y: Union[Tensor, List[Tensor]],
        var_y: Union[Tensor, List[Tensor]],
        tweedie_covariance: bool = True,
        iterations: int = 1,
    ):
        super().__init__()

        self.denoiser = denoiser

        self.A = A

        # [中文注释] y 与 var_y(= Σ_y, 对角)作为 buffer 存入模块: 观测数据从此
        # "随身携带", sampler 每步调用本模块时无需再传观测. list/nested 版本支持
        # 每个 blanket 各有自己的观测序列(卫星掩膜随窗口相位变化, 长度 M 可不同).
        if isinstance(y, list):
            self.register_buffer("y", torch.nested.as_nested_tensor(y))
            self.register_buffer("var_y", torch.nested.as_nested_tensor(var_y))
        else:
            self.register_buffer("y", torch.as_tensor(y))
            self.register_buffer("var_y", torch.as_tensor(var_y))

        self.tweedie_covariance = tweedie_covariance

        self.solve = partial(gmres, iterations=iterations)

    def forward(self, x_t: Tensor, sigma_t: Tensor, blanket_id: int = 0, **kwargs):
        # [中文注释] x_t: 当前带噪 latent blanket (≈ 论文§3.2 的 z_{i:i+w}(t)).
        # var_t: 假设 p(z)=N(0,I) 时 Var[z|z_t] 的解析式, tweedie_covariance=False 时使用.
        var_t = sigma_t**2 / (1 + sigma_t**2)

        # [中文注释] selfA: 把观测算子 𝒜(Eq.7)钉住当前 blanket_id 后的单参数版本;
        # 注意与下方局部函数 A(v) 区分 — 那是 𝒜 在 x_hat 处线性化的 Jacobian 作用
        # 于向量(即 Eq.8 中的 J). selfy/selfvar_y: 本窗口的观测 y 与对角方差 Σ_y.
        selfA = partial(self.A, blanket_id=blanket_id)
        selfy = self.y[blanket_id] if self.y.is_nested else self.y
        selfvar_y = self.var_y[blanket_id] if self.var_y.is_nested else self.var_y

        if selfy is None or len(selfy) == 0:
            return self.denoiser(x_t, sigma_t, **kwargs)  # Unconditioned blanket.

        with torch.enable_grad():
            x_t = x_t.detach().requires_grad_()
            x_hat = self.denoiser(x_t, sigma_t, **kwargs)  # ① E[z|z_t] — Eq.(8) 的期望项
            y_hat = selfA(x_hat)  # ② 𝒜(E[z|z_t]) — "当前 latent 猜出来的观测值"

        # [中文注释] 下方 A/At 是 𝒜 在 x_hat 处线性化后的免矩阵 Jacobian 算子:
        #   A(v)  = J·v   (JVP, 正向): latent 空间方向 → 该方向引起的观测值变化
        #   At(v) = J^T·v (VJP, 反向): 观测空间向量 → latent 空间向量
        def A(v):
            with (
                disable_xfa(),
                disable_checkpointing(),
            ):  # jvp incompatible with xformers and checkpointing
                return torch.func.jvp(selfA, (x_hat.detach(),), (v,))[1]

        def At(v):
            return torch.autograd.grad(y_hat, x_hat, v, retain_graph=True)[0]

        # fmt: off
        # [中文注释 | Eq.(8) 的 V[z|z_t]] cov_x: 后验方差作用于向量的免矩阵实现.
        if self.tweedie_covariance:
            # Tweedie 协方差 σ_t²·∂x_hat/∂x_t: 用 denoiser 自身 Jacobian 估计后验
            # 方差, 比下面的 N(0,I) 解析式更贴近真实数据先验(MMPS 论文的核心技巧).
            def cov_x(v):
                return sigma_t**2 * torch.autograd.grad(x_hat, x_t, v, retain_graph=True)[0]
        else:
            # 退化版本: 直接用解析式 σ_t²/(1+σ_t²) (假设 p(z) = N(0, I)).
            def cov_x(v):
                return var_t * v
        # fmt: on

        # [中文注释 | Eq.(8) 的协方差整体] cov_y(v) = (Σ_y + J·V[z|z_t]·J^T)·v,
        # 完全矩阵-free: 对角 Σ_y 逐点乘 + Jacobian 往返传播(At → cov_x → A).
        def cov_y(v):
            return selfvar_y * v + A(cov_x(At(v)))

        grad = selfy - y_hat  # ⑤ 新息 y - 𝒜(E[z|z_t]): 观测数据在此参与运算
        # ⑥ w ≈ Σ_eff^{-1}·新息: GMRES 在 Krylov 子空间内的免矩阵近似解(2 次迭代)
        grad = self.solve(A=cov_y, b=grad)
        # ⑦ score = J^T·w = ∇_{z_t} log p(y|z_t), 即 Eq.(6) 的第二项(似然 score);
        # autograd 一次反向 pass 完成 VJP, 成本与观测维数 M、latent 维数 D 的大小无关.
        score = torch.autograd.grad(y_hat, x_t, grad)[0]

        # ⑧ 后验 denoiser: x_hat + σ_t²·s_obs. sampler 侧按 (d - z_t)/σ_t² 转回
        # score 时, 先验项与似然项自然相加 — 这就是 Eq.(6) 在 denoiser 空间的实现.
        return x_hat + sigma_t**2 * score


# TODO: Support for blanket_id.
class SDADenoiser(nn.Module):
    r"""Creates an sda-like guided denoiser module.

    References:
        | Score-based Data Assimiliation (Rozet et al., 2023)
        | https://arxiv.org/abs/2306.10574

    Arguments:
        denoiser: A Gaussian denoiser.
        y: An observation :math:`y \sim \mathcal{N}(A(x), \Sigma_y)`, with shape :math:`(*, D)`.
        A: The forward operator :math:`x \mapsto A(x)`.
        var_y: The noise variance :math:`\Sigma_y`.
        gamma: A coefficient :math:`\gamma \approx \diag(A A^\top)`.
    """

    def __init__(
        self,
        denoiser: nn.Module,
        y: Tensor,
        A: Callable[[Tensor], Tensor],
        var_y: Tensor,
        gamma: Tensor = 1.0,
    ):
        super().__init__()

        self.denoiser = denoiser
        self.A = A

        self.register_buffer("y", torch.as_tensor(y))
        self.register_buffer("var_y", torch.as_tensor(var_y))
        self.register_buffer("gamma", torch.as_tensor(gamma))

    def forward(self, x_t: Tensor, sigma_t: Tensor, **kwargs):
        var_t = sigma_t**2 / (1 + sigma_t**2)

        with torch.enable_grad():
            x_t = x_t.detach().requires_grad_()
            x_hat = self.denoiser(x_t, sigma_t, **kwargs)

            y_hat = self.A(x_hat)

            log_p = (self.y - y_hat) ** 2 / (self.var_y + self.gamma * var_t)

            log_p = -1 / 2 * log_p.sum()

        score = torch.autograd.grad(log_p, x_t)[0]

        return x_hat + sigma_t**2 * score


# TODO: Support for blanket_id.
class W2CDenoiser(nn.Module):
    r"""Creates an simplified sda-like guided denoiser module.

    References:
        | A Generative Framework for Probabilistic, Spatiotemporally Coherent Downscaling
        | of Climate Simulation (Schmidt et al., 2024)
        | https://arxiv.org/abs/2412.15361

    Arguments:
        denoiser: A Gaussian denoiser.
        y: An observation :math:`y \sim \mathcal{N}(A(x), \Sigma_y)`, with shape :math:`(*, D)`.
        A: The forward operator :math:`x \mapsto A(x)`.
        var_y: The noise variance :math:`\Sigma_y`.
        gamma: A coefficient :math:`\gamma \approx \diag(A A^\top)`.
    """

    def __init__(
        self,
        denoiser: nn.Module,
        y: Tensor,
        A: Callable[[Tensor], Tensor],
        var_y: Tensor,
        gamma: Tensor = 1.0,
    ):
        super().__init__()

        self.denoiser = denoiser
        self.A = A

        self.register_buffer("y", torch.as_tensor(y))
        self.register_buffer("var_y", torch.as_tensor(var_y))
        self.register_buffer("gamma", torch.as_tensor(gamma))

    def forward(self, x_t: Tensor, sigma_t: Tensor, **kwargs):
        var_t = sigma_t**2 / (1 + sigma_t**2)

        x_hat = self.denoiser(x_t, sigma_t, **kwargs)

        with torch.enable_grad():
            x_hat = x_hat.detach().requires_grad_()
            y_hat = self.A(x_hat)

            log_p = (self.y - y_hat) ** 2 / (self.var_y + self.gamma * var_t)
            log_p = -1 / 2 * log_p.sum()

        grad = torch.autograd.grad(log_p, x_hat)[0]

        return x_hat + var_t * grad


class TrajectoryDenoiser(nn.Module):
    r"""Wraps a blanket denoising model into a full trajectory model.

    Arguments:
        denoiser: A denoiser backbone that operates on blanket.
        blanket_size: The size of the backbone blanket.
        blanket_stride: The stride between two consecutive blankets in a trajectory.
        state_size: The size of an individual state.
        distributed: Whether to use distributed processing.
        pass_blanket_ids: Whether to pass the blanket id to the backbone denoiser (e.g., conditioning).
        mode: Mode of trajectory reconstruction from individual windows.
    """

    # [中文注释 | 论文§3.2 / Algorithm 2] 长轨迹的 blanket 分解与重组:
    # 轨迹按步长 Δ=13(= blanket_size 25 - overlap 12)切成重叠的 w=25 小时窗口,
    # 每个 blanket 独立过 denoiser(自带各自的观测 y), 再按 symmetrical 模式
    # 取每个窗口中间 Δ 段拼接回全轨迹 — 对应 Algorithm 2 的 a=(w-Δ)/2, b=a+Δ 截取.

    def __init__(
        self,
        denoiser: nn.Module,
        blanket_size: int,
        blanket_stride: int,
        state_size: int,
        distributed: bool = False,
        pass_blanket_ids: bool = False,
        mode: str = "symmetrical",
    ):
        super().__init__()
        self.denoiser = denoiser
        self.blanket_size = blanket_size
        self.blanket_stride = blanket_stride
        self.state_size = state_size
        self.distributed = distributed
        self.pass_blanket_ids = pass_blanket_ids
        self.mode = mode

        if distributed:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def forward(self, x_t, t, date):
        r"""Processes a trajectory blanket by blanket.

        Arguments:
            x_t: The input flattened trajectory with shape :math:`(B, T * D)`, with D the state size.
            t: The diffusion time tensor with shape :math:`()`.
            date: The date tensor of the trajectory used as a timestamp context, with shape :math:`(B, T, 4)`.

        Returns:
            The output denoised flat trajectory with shape :math:`(B, T * D)`.
        """

        B = x_t.shape[0]

        x_t, date = self.unfold(x_t, date)

        if self.distributed:
            x_t = x_t.tensor_split(self.world_size, dim=0)
            date = date.tensor_split(self.world_size, dim=0)
            sizes = [x.shape[0] for x in x_t]
            max_size = max(sizes)
            x_t, date = x_t[self.rank], date[self.rank]

        x_t_list = []
        for blanket_id, (x, d) in enumerate(
            zip(x_t.tensor_split(x_t.shape[0]), date.tensor_split(x_t.shape[0]))
        ):
            if self.pass_blanket_ids:
                x_t_list.append(self.denoiser(x, t, date=d, blanket_id=blanket_id))
            else:
                x_t_list.append(self.denoiser(x, t, date=d))
        x_t = torch.cat(x_t_list)

        if self.distributed:
            x_t = torch.cat([
                x_t,
                x_t.new_full((max_size - x_t.shape[0], *x_t.shape[1:]), 0),
            ]).contiguous()
            x_ts = [torch.zeros_like(x_t) for _ in range(self.world_size)]
            dist.all_gather(x_ts, x_t)
            x_ts = [x[: sizes[i]] for i, x in enumerate(x_ts)]
            x_t = torch.cat(x_ts, dim=0)

        x_t = self.fold(x_t, batch_size=B)

        return x_t

    def unfold(self, x, d):
        x = rearrange(x, "B (T D) -> B T D", D=self.state_size)
        T = x.shape[1]

        assert (T - self.blanket_size) % self.blanket_stride == 0, (
            f"K={self.blanket_size} and S={self.blanket_stride} do not fit in {T}."
        )
        num_segments = (T - self.blanket_size) // self.blanket_stride + 1

        x = x.unfold(1, self.blanket_size, self.blanket_stride)
        x = rearrange(x, "B N D K -> (B N) (K D)", N=num_segments)

        d = d.unfold(1, self.blanket_size, self.blanket_stride)
        d = rearrange(d, "B N F K -> (B N) K F")

        return x, d

    def fold(self, x, batch_size):
        x = rearrange(x, "(B N) (K D) -> B N K D", B=batch_size, D=self.state_size)

        segments = []
        if x.shape[1] > 1:
            if self.mode == "symmetrical":
                i = (self.blanket_size - self.blanket_stride) // 2
                j = i + self.blanket_stride
                segments.append(x[:, 0, :j])
                segments.extend(x[:, 1:-1, i:j].unbind(1))
                segments.append(x[:, -1, i:])

                x = torch.cat(segments, dim=1)
            elif self.mode == "causal":
                i = self.blanket_size - self.blanket_stride
                segments.append(x[:, 0, :])
                segments.extend(x[:, 1:, i:].unbind(1))

                x = torch.cat(segments, dim=1)
            elif self.mode == "average":
                length = (x.shape[1] - 1) * self.blanket_stride + self.blanket_size
                accum_x = torch.zeros(x.shape[0], length, x.shape[-1])
                accum_count = torch.zeros(x.shape[0], length, 1)
                for i in range(x.shape[1]):
                    accum_x[
                        :, i * self.blanket_stride : i * self.blanket_stride + self.blanket_size
                    ] += x[:, i]
                    accum_count[
                        :, i * self.blanket_stride : i * self.blanket_stride + self.blanket_size
                    ] += 1

                x = accum_x / accum_count
            else:
                raise NotImplementedError(
                    f"Trajectory reconstruction mode {self.mode} not implemented."
                )
        else:
            x = x[:, 0]

        return rearrange(x, "B T D -> B (T D)")


class DenoiserLoss(nn.Module):
    r"""Loss for a denoiser model.

    .. math:: \lambda_t || d(x_t) - x ||^2

    Arguments:
        denoiser: A denoiser model :math:`d(x_t) \approx E[x | x_t]`.
    """

    def __init__(self, denoiser: Denoiser, std_min_penalty: float = 0):
        super().__init__()

        self.denoiser = denoiser
        self.std_min_penalty = std_min_penalty

    def forward(self, x: Tensor, sigma_t: Tensor, **kwargs) -> Tensor:
        r"""
        Arguments:
            x: The clean tensor, with shape :math:`(*, D)`.
            sigma_t: The noise std, with shape :math:`(*, 1)`.
            kwargs: Keyword arguments passed to the denoiser.

        Returns:
            The reduced denoising loss, with shape :math:`()`.
        """

        lmb_t = (1 / (sigma_t + self.std_min_penalty) ** 2 + 1).squeeze(-1)

        z = torch.randn_like(x)
        xt = x + sigma_t * z

        dxt = self.denoiser(xt, sigma_t, **kwargs)

        error = dxt - x

        return torch.mean(lmb_t * torch.mean(error**2, dim=-1))


class Schedule(nn.Module):
    r"""Base class for noise schedules.

    Arguments:
        sigma_min: The minimum noise std :math:`\sigma_\min \in \mathbb{R}_+`.
        sigma_max: The maximum noise std :math:`\sigma_\max \in \mathbb{R}_+`.
        bfloat16: Whether to use bfloat16 precision or not (float32).
    """

    def __init__(self, sigma_min: float = 1e-3, sigma_max: float = 1e3, bfloat16: bool = False):
        super().__init__()

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.log_sigma_min = math.log(sigma_min)
        self.log_sigma_max = math.log(sigma_max)

        self.t_max = 1 - 1 / 256 if bfloat16 else 1.0

        if bfloat16:
            logger.warning(
                "Using bfloat16 precision, "
                "please make sure to use schedule.sigma_tmax (t_max ≠ 1.0!)."
            )

    def sigma_tmax(self) -> Tensor:
        r"""Returns the maximum noise std :math:`\sigma_{t_{max}}`.

        Returns:
            The noise std :math:`\sigma_{t_{max}}`.
        """

        return self.forward(torch.tensor([self.t_max]))


class LogLinearSchedule(Schedule):
    r"""Log-linear noise schedule.

    .. math:: \sigma_t = \exp(\log(a) (1 - t) + \log(b) t)

    Arguments:
        a: The noise lower bound.
        b: The noise upper bound.
    """

    def forward(self, t: Tensor) -> Tensor:
        r"""
        Arguments:
            t: The schedule time, with shape :math:`(*)`.

        Returns:
            The noise std :math:`\sigma_t`, with shape :math:`(*, 1)`.
        """

        return torch.exp(
            self.log_sigma_min + (self.log_sigma_max - self.log_sigma_min) * t
        ).unsqueeze(-1)


class LogLogitSchedule(Schedule):
    r"""Creates a log-logit noise schedule.

    .. math::
        \sigma_t & = \sqrt{\sigma_\min \sigma_\max} \exp(\rho \logit t)

    See also:
        :func:`torch.logit`

    Arguments:
        sigma_min: The initial noise scale :math:`\sigma_\min \in \mathbb{R}_+`.
        sigma_max: The final noise scale :math:`\sigma_\max \in \mathbb{R}_+`.
        spread: The spread factor :math:`\rho \in \mathbb{R}_+`.
    """

    def __init__(
        self,
        sigma_min: float = 1e-3,
        sigma_max: float = 1e3,
        bfloat16: bool = False,
        spread: float = 2.0,
    ):
        super().__init__(sigma_min=sigma_min, sigma_max=sigma_max, bfloat16=bfloat16)

        self.eps = math.sqrt(sigma_min / sigma_max) ** (1 / spread)
        self.log_sigma_med = math.log(sigma_min * sigma_max) / 2
        self.spread = spread

    def forward(self, t: Tensor) -> Tensor:
        return torch.exp(
            self.spread * torch.logit(t * (1 - 2 * self.eps) + self.eps) + self.log_sigma_med
        ).unsqueeze(-1)


class RectifiedSchedule(Schedule):
    r"""Creates a rectified noise schedule.

    .. math::
        \sigma_t & = \frac{t + (1 - t) \sigma_\min}{t (\sigma_\max^{-1} - 1) + 1}

    Arguments:
        sigma_min: The initial noise scale :math:`\sigma_\min \in \mathbb{R}_+`.
        sigma_max: The final noise scale :math:`\sigma_\max \in \mathbb{R}_+`.
    """

    def forward(self, t: Tensor) -> Tensor:
        return ((t + (1 - t) * self.sigma_min) / (t * (1 / self.sigma_max - 1) + 1)).unsqueeze(-1)


def create_schedule(
    train_cfg: DictConfig,
    bfloat16: bool = False,
    device: str = "cuda",
) -> Schedule:
    r"""Create a noise schedule."""

    schedules = {
        "rectified": RectifiedSchedule,
        "loglogit": LogLogitSchedule,
        "loglinear": LogLinearSchedule,
    }

    noise_schedule = train_cfg.noise_schedule
    sigma_min = train_cfg.sigma_min
    sigma_max = train_cfg.sigma_max

    if noise_schedule not in schedules:
        raise ValueError(f"Invalid noise schedule: {noise_schedule}")

    return schedules[noise_schedule](
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        bfloat16=bfloat16,
    ).to(device)


def create_denoiser(
    diffusion_cfg: DictConfig,
    ae_cfg: DictConfig,
    distributed: bool = False,
    device: str = "cuda",
    overrides: dict = None,
):
    r"""Create a denoiser model with an SPTGraphDiT backbone.

    Arguments:
        diffusion_cfg: Diffusion configuration.
        ae_cfg: Autoencoder configuration.
        distributed: Whether to use distributed data parallel.
        device: Device to use.
        overrides: Optional overrides for the score network configuration.

    Returns:
        A denoiser model.
    """

    denoiser_cfg = diffusion_cfg.backbone

    if overrides:
        with open_dict(denoiser_cfg):
            for k in overrides.keys():
                denoiser_cfg[k] = overrides[k]

    if "graphdit" in denoiser_cfg.name:
        grid, _ = create_icosphere(ae_cfg.ae.ico_divisions[-1])

        if denoiser_cfg.self_attention_hops is None or denoiser_cfg.cross_attention_hops is None:
            self_edges = None
            cross_edges = None
        else:
            self_edges = icosphere_nhops_edges(grid, grid, denoiser_cfg.self_attention_hops)
            cross_edges = icosphere_nhops_edges(grid, grid, denoiser_cfg.cross_attention_hops)

        dit = SpatioTemporalGraphDiT(
            input_grid=grid,
            input_channels=ae_cfg.ae.latent_channels,
            self_edges=self_edges,
            cross_edges=cross_edges,
            **denoiser_cfg,
        ).to(device)
    elif "vit" in denoiser_cfg.name:
        assert "cae" in ae_cfg.ae.name, f"ViDiT not compatible with {ae_cfg.ae.name}"

        H, W, C = ae_cfg.ae.latent_shape
        dit = ViDiT(
            in_channels=C,
            shape=(H, W),
            **denoiser_cfg,
        ).to(device)
    else:
        raise NotImplementedError(f"Unknown denoiser type with name {denoiser_cfg.name}")

    if distributed:
        dit = torch.nn.parallel.DistributedDataParallel(module=dit, device_ids=[device])

    return Denoiser(dit).to(device)
