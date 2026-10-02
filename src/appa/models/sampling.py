r"""Diffusion helpers."""

import logging
import torch
import torch.nn as nn

from abc import abstractmethod
from torch import Tensor
from typing import Callable

from appa.models.diffusion import Schedule
from appa.utils.math import gauss_legendre

logger = logging.getLogger(__name__)


class Sampler(nn.Module):
    r"""Base class for samplers, implementing the init API but leaving the forward to be implemetned by subclasses.

    .. math:: x_s = x_t - \tau (x_t - d(x_t)) + \sigma_s \sqrt{\tau} \epsilon

    where :math:`\tau` is determined by the noise schedule.

    Arguments:
        denoiser: A denoiser model :math:`d(x_t) \approx E[x | x_t]`.
        schedule: The noise schedule.
        steps: The number of sampling steps.
    """

    def __init__(
        self,
        denoiser: Callable[[Tensor, Tensor], Tensor],
        schedule: Schedule,
        steps: int = 256,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.denoiser = denoiser
        self.schedule = schedule
        self.steps = steps

    @abstractmethod
    def forward(self, x1: Tensor) -> Tensor:
        r"""
        Arguments:
            x1: A noise tensor from :math:`p(x_1)`, with shape :math:`(*, D)`.

        Returns:
            A data tensor from :math:`p(x_0 | x_1)`, with shape :math:`(*, D)`.
        """

        pass


class DDPMSampler(Sampler):
    r"""DDPM sampler for the reverse SDE.

    .. math:: x_s = x_t - \tau (x_t - d(x_t)) + \sigma_s \sqrt{\tau} \epsilon

    where :math:`\tau = 1 - \frac{\sigma_s^2}{\sigma_t^2}`.

    Arguments:
        denoiser: A denoiser model :math:`d(x_t) \approx E[x | x_t]`.
        schedule: The noise schedule.
        steps: The number of sampling steps.
    """

    def __init__(
        self,
        denoiser: Callable[[Tensor, Tensor], Tensor],
        schedule: nn.Module = None,
        steps: int = 256,
        silent: bool = True,
    ):
        super().__init__(denoiser=denoiser, schedule=schedule, steps=steps)

        self.silent = silent

    def forward(self, x1: Tensor) -> Tensor:
        r"""
        Arguments:
            x1: A noise tensor from :math:`p(x_1)`, with shape :math:`(*, D)`.

        Returns:
            A data tensor from :math:`p(x_0 | x_1)`, with shape :math:`(*, D)`.
        """

        dt = torch.as_tensor(1 / self.steps, device=x1.device)
        time = torch.linspace(self.schedule.t_max, dt, self.steps, device=x1.device)

        xt = x1
        for i, t in enumerate(time):
            xt = self.step(xt, t, t - dt)

            if not self.silent:
                logger.info("Diffusion step %d/%d done.", i + 1, self.steps)

        return xt

    def step(self, xt: Tensor, t: Tensor, s: Tensor) -> Tensor:
        sigma_s, sigma_t = self.schedule(s), self.schedule(t)
        tau = 1 - (sigma_s / sigma_t) ** 2
        eps = torch.randn_like(xt)

        return xt - tau * (xt - self.denoiser(xt, sigma_t)) + sigma_s * torch.sqrt(tau) * eps


class DDIMSampler(DDPMSampler):
    r"""DDIM sampler for the reverse SDE.

    .. math:: x_s = x_t - (1 - \frac{\sigma_s}{\sigma_t}) (x_t - d(x_t))

    Arguments:
        denoiser: A denoiser model :math:`d(x_t) \approx E[x | x_t]`.
        schedule: The noise schedule.
        steps: The number of sampling steps.

    """

    def step(self, xt: Tensor, t: Tensor, s: Tensor) -> Tensor:
        sigma_s, sigma_t = self.schedule(s), self.schedule(t)

        return xt - (1 - sigma_s / sigma_t) * (xt - self.denoiser(xt, sigma_t))


class RewindDDIMSampler(DDIMSampler):
    r"""DDIM sampler for the reverse SDE with forward jumps.

    .. math:: x_s = x_t - (1 - \frac{\sigma_s}{\sigma_t}) (x_t - d(x_t))

    Arguments:
        rewind_steps: Number of inner denoising steps before forward diffusion rewind to the next outer step.
        jump: Whether to perform large jumps backward (true) or small steps.

    """

    def __init__(self, rewind_after: int = 16, jump: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.rewind_after = rewind_after
        self.jump = jump

    @torch.no_grad()
    def forward(self, x1: Tensor, **kwargs) -> Tensor:
        x_t = x1

        dt = torch.as_tensor(1 / self.steps, device=x1.device)

        for i, t in enumerate(torch.linspace(1, dt, self.steps)):
            kmax = min(self.rewind_after, int(t / dt))
            s = t - kmax * dt

            if self.jump:
                x_s = self.step(x_t, t, s, **kwargs)
            else:
                x_s = x_t
                for ti in torch.linspace(t, s + dt, kmax, device=x1.device):
                    x_s = self.step(x_s, ti, ti - dt, **kwargs)

            sigma_s = self.schedule(s)
            sigma_t = self.schedule(t - dt)

            B = torch.sqrt(sigma_t**2 - sigma_s**2)
            x_t = x_s + B * torch.randn_like(x_s)

            if not self.silent:
                logger.info("Diffusion step %d/%d done.", i + 1, self.steps)

        x = x_t

        return x


class LMSSampler(Sampler):
    r"""Creates a linear multi-step (LMS) sampler.

    References:
        | k-diffusion (Katherine Crowson)
        | https://github.com/crowsonkb/k-diffusion
        | azula library (François Rozet)
        | https://github.com/francois-rozet/azula

    Arguments:
        denoiser: A denoiser model :math:`d(x_t) \approx E[x | x_t]`.
        schedule: The noise schedule.
        steps: The number of sampling steps.
        order: The order of the multi-step method.
        kwargs: Keyword arguments passed to :class:`Sampler`.
    """

    def __init__(
        self,
        denoiser: Callable[[Tensor, Tensor], Tensor],
        schedule: nn.Module = None,
        steps: int = 256,
        order: int = 3,
        silent: bool = True,
    ):
        super().__init__(denoiser=denoiser, schedule=schedule, steps=steps)

        self.order = order
        self.silent = silent

    @staticmethod
    def adams_bashforth(t: Tensor, i: int, order: int = 3) -> Tensor:
        r"""Returns the coefficients of the :math:`N`-th order Adams-Bashforth method.

        Wikipedia:
            https://wikipedia.org/wiki/Linear_multistep_method

        Arguments:
            t: The integration variable, with shape :math:`(T)`.
            i: The integration step.
            order: The method order :math:`N`.

        Returns:
            The Adams-Bashforth coefficients, with shape :math:`(N)`.
        """

        ti = t[i]
        tj = t[i - order : i]
        tk = torch.cat((tj, tj)).unfold(0, order, 1)[:order, 1:]
        tj_tk = tj[..., None] - tk

        # Lagrange basis
        def lj(t):
            return torch.prod((t[..., None, None] - tk) / tj_tk, dim=-1)

        # Adams-Bashforth coefficients
        cj = gauss_legendre(lj, tj[-1], ti, n=order // 2 + 1)

        return cj

    @torch.no_grad()
    def forward(self, x1: Tensor, **kwargs) -> Tensor:
        # Ok to go to 0 because enumerate(time[:-1])
        time = torch.linspace(self.schedule.t_max, 0, self.steps + 1, device=x1.device)

        sigmas = self.schedule(time).squeeze()
        ratio = sigmas.double()

        xt = x1

        derivatives = []

        for i, sigma_t in enumerate(sigmas[:-1]):
            q_t = self.denoiser(xt, sigma_t, **kwargs)
            z_t = (xt - q_t) / sigma_t

            derivatives.append(z_t)

            if len(derivatives) > self.order:
                derivatives.pop(0)

            coefficients = self.adams_bashforth(ratio, i + 1, order=len(derivatives))
            coefficients = coefficients.to(xt)

            delta = sum(c * d for c, d in zip(coefficients, derivatives))

            xt = xt + delta

            if not self.silent:
                logger.info("Diffusion step %d/%d done.", i + 1, self.steps)

        x0 = xt

        return x0


class PCSampler(DDPMSampler):
    r"""Creates a predictor-corrector (PC) sampler.

    References:
        | Score-Based Generative Modeling through Stochastic Differential Equations (Song et al., 2021)
        | https://arxiv.org/abs/2011.13456

    Arguments:
        denoiser: A Gaussian denoiser.
        corrections: The number of Langevin corrections per step.
        delta: The amplitude of Langevin corrections.
        kwargs: Keyword arguments passed to :class:`Sampler`.
    """

    def __init__(
        self,
        denoiser: Callable[[Tensor, Tensor], Tensor],
        schedule: nn.Module = None,
        steps: int = 256,
        corrections: int = 1,
        delta: float = 0.01,
        **kwargs,
    ):
        super().__init__(denoiser=denoiser, schedule=schedule, steps=steps, **kwargs)

        self.denoiser = denoiser
        self.corrections = corrections

        self.register_buffer("delta", torch.as_tensor(delta))

    def step(self, x_t: Tensor, t: Tensor, s: Tensor, **kwargs) -> Tensor:
        sigma_s = self.schedule(s)
        sigma_t = self.schedule(t)

        # Corrector
        for _ in range(self.corrections):
            x_hat = self.denoiser(x_t, sigma_t, **kwargs)
            x_t = (
                x_t
                + self.delta * (x_hat - x_t)
                + torch.sqrt(2 * self.delta) * sigma_t * torch.randn_like(x_t)
            )

        # Predictor
        x_hat = self.denoiser(x_t, sigma_t, **kwargs)
        x_s = x_hat + sigma_s / sigma_t * (x_t - x_hat)

        return x_s


def select_sampler(name: str):
    r"""Return a sampler class based on a short name.

    Arguments:
        name: The name of the sampler, one of "pc", "ddpm", "ddim", "rewind", or "lms".

    Returns:
        A sampler class corresponding to the given name.
    """

    if name == "pc":
        return PCSampler
    elif name == "ddpm":
        return DDPMSampler
    elif name == "ddim":
        return DDIMSampler
    elif name == "rewind":
        return RewindDDIMSampler
    elif name == "lms":
        return LMSSampler
    else:
        raise ValueError(
            f"Unknown sampler type: {name}. Options are 'pc', 'ddpm', 'ddim', 'rewind', and 'lms'."
        )
