from __future__ import annotations

from dataclasses import dataclass

import torch

LOG_VAR_MIN = -10.0
LOG_VAR_MAX = 10.0


@dataclass(frozen=True)
class LossWeights:
    """Container for regression loss weights."""

    nll: float = 1.0
    mse: float = 0.0
    mae: float = 0.0

    def validate(self) -> None:
        for name, value in (("nll", self.nll), ("mse", self.mse), ("mae", self.mae)):
            if value < 0:
                raise ValueError(f"{name} weight must be non-negative (got {value}).")
        if self.nll + self.mse + self.mae <= 0:
            raise ValueError("At least one loss weight must be greater than zero.")


def gaussian_nll_loss(pred_mean: torch.Tensor, pred_log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood under a Gaussian with predicted mean/log-variance."""
    log_var = torch.clamp(pred_log_var, min=LOG_VAR_MIN, max=LOG_VAR_MAX)
    inv_var = torch.exp(-log_var)
    loss = 0.5 * (log_var + (target - pred_mean) ** 2 * inv_var)
    return torch.mean(loss)


def mse_loss(pred_mean: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error."""
    return torch.mean((pred_mean - target) ** 2)


def mae_loss(pred_mean: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean absolute error."""
    return torch.mean(torch.abs(pred_mean - target))


def weighted_regression_loss(
    pred_mean: torch.Tensor,
    pred_log_var: torch.Tensor,
    target: torch.Tensor,
    weights: LossWeights,
) -> torch.Tensor:
    """Return weighted combination of Gaussian NLL, MSE, and MAE components."""
    # Clamp log-variance to keep variance in a stable, positive range.
    pred_log_var = torch.clamp(pred_log_var, min=-10.0, max=10.0)

    total_loss: torch.Tensor | None = None
    if weights.nll > 0:
        nll = gaussian_nll_loss(pred_mean, pred_log_var, target)
        total_loss = weights.nll * nll
    if weights.mse > 0:
        mse = mse_loss(pred_mean, target)
        total_loss = mse * weights.mse if total_loss is None else total_loss + weights.mse * mse
    if weights.mae > 0:
        mae = mae_loss(pred_mean, target)
        total_loss = mae * weights.mae if total_loss is None else total_loss + weights.mae * mae

    if total_loss is None:
        raise ValueError("weighted_regression_loss requires at least one positive weight.")
    return total_loss
