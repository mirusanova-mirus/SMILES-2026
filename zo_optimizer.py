"""
zo_optimizer.py — Zero-order optimizer skeleton (student-implemented).

Strict black-box optimizer:
  - updates only the final classifier head
  - never reads gradients, logits, labels, or intermediate features
  - uses only scalar calls to ``loss_fn()``
"""

from __future__ import annotations

from typing import Callable, List

import torch
import torch.nn as nn


class ZeroOrderOptimizer:
    """Black-box optimizer for a low-dimensional calibration of the classifier head."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 5e-3,
        eps: float = 5e-2,
        perturbation_mode: str = "gaussian",
    ) -> None:
        self.model = model
        self.lr = lr
        self.eps = eps

        for param in self.model.parameters():
            param.requires_grad_(False)

        if perturbation_mode not in ("gaussian", "uniform"):
            raise ValueError(
                f"perturbation_mode must be 'gaussian' or 'uniform', "
                f"got '{perturbation_mode}'"
            )
        self.perturbation_mode = perturbation_mode

        self.layer_names: List[str] = ["fc.weight", "fc.bias"]

        self.K: int = 1
        self.q: int = 1
        self.beta1: float = 0.9
        self.beta2: float = 0.999
        self.eps_adam: float = 1e-8
        self.scale_clip: float = 0.5
        self.bias_clip: float = 2.0
        self.accept_reject: bool = True

        fc = self._fc_layer()
        self._base_weight = fc.weight.detach().clone()
        self._base_bias = fc.bias.detach().clone()

        self._log_scale = torch.zeros(fc.out_features, device=fc.weight.device, dtype=fc.weight.dtype)
        self._bias_shift = torch.zeros_like(self._base_bias)

        self._m_scale = torch.zeros_like(self._log_scale)
        self._v_scale = torch.zeros_like(self._log_scale)
        self._m_bias = torch.zeros_like(self._bias_shift)
        self._v_bias = torch.zeros_like(self._bias_shift)
        self._t: int = 0

        self._apply_latent()

    def _fc_layer(self) -> nn.Linear:
        fc = getattr(self.model, "fc", None)
        if not isinstance(fc, nn.Linear):
            raise TypeError("ZeroOrderOptimizer expects model.fc to be nn.Linear")
        return fc

    def _apply_latent(
        self,
        log_scale: torch.Tensor | None = None,
        bias_shift: torch.Tensor | None = None,
    ) -> None:
        fc = self._fc_layer()
        log_scale = self._log_scale if log_scale is None else log_scale
        bias_shift = self._bias_shift if bias_shift is None else bias_shift

        with torch.no_grad():
            fc.weight.copy_(self._base_weight * torch.exp(log_scale).unsqueeze(1))
            fc.bias.copy_(self._base_bias + bias_shift)

    def _sample_direction(self, param: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(param).bernoulli_(0.5).mul_(2.0).sub_(1.0)

    def _estimate_grad(self, loss_fn: Callable[[], float]) -> tuple[torch.Tensor, torch.Tensor]:
        grad_scale = torch.zeros_like(self._log_scale)
        grad_bias = torch.zeros_like(self._bias_shift)

        with torch.no_grad():
            for _ in range(self.q):
                u_scale = self._sample_direction(self._log_scale)
                u_bias = self._sample_direction(self._bias_shift)

                self._apply_latent(
                    log_scale=(self._log_scale + self.eps * u_scale).clamp(-self.scale_clip, self.scale_clip),
                    bias_shift=(self._bias_shift + self.eps * u_bias).clamp(-self.bias_clip, self.bias_clip),
                )
                f_plus = loss_fn()

                self._apply_latent(
                    log_scale=(self._log_scale - self.eps * u_scale).clamp(-self.scale_clip, self.scale_clip),
                    bias_shift=(self._bias_shift - self.eps * u_bias).clamp(-self.bias_clip, self.bias_clip),
                )
                f_minus = loss_fn()

                coeff = (f_plus - f_minus) / (2.0 * self.eps)
                grad_scale.add_(u_scale, alpha=coeff)
                grad_bias.add_(u_bias, alpha=coeff)

            grad_scale.div_(self.q)
            grad_bias.div_(self.q)

        self._apply_latent()
        return grad_scale, grad_bias

    def _adam_update(self, grad_scale: torch.Tensor, grad_bias: torch.Tensor) -> None:
        self._t += 1
        bc1 = 1.0 - self.beta1 ** self._t
        bc2 = 1.0 - self.beta2 ** self._t

        with torch.no_grad():
            self._m_scale.mul_(self.beta1).add_(grad_scale, alpha=1.0 - self.beta1)
            self._v_scale.mul_(self.beta2).addcmul_(grad_scale, grad_scale, value=1.0 - self.beta2)
            self._m_bias.mul_(self.beta1).add_(grad_bias, alpha=1.0 - self.beta1)
            self._v_bias.mul_(self.beta2).addcmul_(grad_bias, grad_bias, value=1.0 - self.beta2)

            m_scale_hat = self._m_scale / bc1
            v_scale_hat = self._v_scale / bc2
            m_bias_hat = self._m_bias / bc1
            v_bias_hat = self._v_bias / bc2

            self._log_scale.sub_(self.lr * m_scale_hat / (v_scale_hat.sqrt() + self.eps_adam))
            self._bias_shift.sub_(self.lr * m_bias_hat / (v_bias_hat.sqrt() + self.eps_adam))

            self._log_scale.clamp_(-self.scale_clip, self.scale_clip)
            self._bias_shift.clamp_(-self.bias_clip, self.bias_clip)

    def step(self, loss_fn: Callable[[], float]) -> float:
        with torch.no_grad():
            loss_curr = float(loss_fn())

        for _ in range(self.K):
            snap = (
                self._log_scale.clone(),
                self._bias_shift.clone(),
                self._m_scale.clone(),
                self._v_scale.clone(),
                self._m_bias.clone(),
                self._v_bias.clone(),
                self._t,
            )

            grad_scale, grad_bias = self._estimate_grad(loss_fn)
            self._adam_update(grad_scale, grad_bias)
            self._apply_latent()

            with torch.no_grad():
                loss_new = float(loss_fn())

            if not self.accept_reject or loss_new < loss_curr:
                loss_curr = loss_new
            else:
                (
                    self._log_scale,
                    self._bias_shift,
                    self._m_scale,
                    self._v_scale,
                    self._m_bias,
                    self._v_bias,
                    self._t,
                ) = snap
                self._apply_latent()

        return loss_curr
