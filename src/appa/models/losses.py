r"""Losses used for training auto encoder"""

import torch
import torch.nn as nn

from einops import rearrange
from torch import Tensor
from typing import Optional

from appa.data.const import (
    ERA5_ATMOSPHERIC_VARIABLES,
    ERA5_PRESSURE_LEVELS,
    ERA5_SURFACE_VARIABLES,
)


def graphcast_level_weights(levels: list, reduction: str = "sum") -> Tensor:
    r"""Returns the level weights of each variable following Graphcast's implementation.

    References:
        | GraphCast: Learning skillful medium-range global weather forecasting
        | https://arxiv.org/abs/2212.12794

    """
    weights = torch.zeros(
        len(ERA5_SURFACE_VARIABLES) + len(levels) * len(ERA5_ATMOSPHERIC_VARIABLES)
    )

    p_levels = torch.as_tensor(levels, dtype=torch.float32)
    if reduction == "sum":
        lin_atm_weights = p_levels / p_levels.sum()
    elif reduction == "mean":
        lin_atm_weights = p_levels / p_levels.mean()

    for i, var in enumerate(ERA5_SURFACE_VARIABLES):
        if "2m_temperature" in var:
            weights[i] = 1.0
        else:
            weights[i] = 0.1

    weights[len(ERA5_SURFACE_VARIABLES) :] = lin_atm_weights.repeat(
        len(ERA5_ATMOSPHERIC_VARIABLES)
    )

    return weights


class AELoss(nn.Module):
    r"""Auto-encoder loss.

    Arguments:
        criterion: Criterion used for the loss.
        N_lat: Number of latitude points.
        N_lon: Number of longitude points.
        latitude_weighting: Whether to weight the loss across latitudes or not.
        level_weighting: Whether to weight the loss across pressure levels or not.
        levels: List of pressure levels.
    """

    def __init__(
        self,
        criterion: str,
        N_lat: int = 721,
        N_lon: int = 1440,
        latitude_weighting: bool = True,
        level_weighting: bool = False,
        levels: Optional[list] = ERA5_PRESSURE_LEVELS,
    ):
        super().__init__()
        if criterion == "mse":
            self.loss = nn.MSELoss(reduction="none")
        elif criterion == "mae":
            self.loss = nn.L1Loss(reduction="none")
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

        self.N_lat = N_lat
        self.N_lon = N_lon

        if latitude_weighting:
            # Weight the error by the cosine of the latitude
            # and add 1 / self.N_lon to weight the poles latitudes as unique points
            lat_weights = (
                torch.linspace(torch.pi / 2, -torch.pi / 2, self.N_lat).cos() + 1 / self.N_lon
            )
        else:
            lat_weights = None
        self.register_buffer("latitude_weights", lat_weights)

        if level_weighting:
            level_weights = graphcast_level_weights(levels)
        else:
            level_weights = None
        self.register_buffer("level_weights", level_weights)

    def forward(self, input: Tensor, target: Tensor):
        loss = self.loss(input, target)

        if self.latitude_weights is not None:
            loss = rearrange(loss, "B (Lat Lon) Z -> B Lon Z Lat", Lat=self.N_lat)
            loss = loss * self.latitude_weights.to(loss)
            loss = rearrange(loss, "B Lon Z Lat -> B (Lat Lon) Z")

        if self.level_weights is not None:
            loss = loss * self.level_weights.to(loss)

        return loss.mean()
