"""
zo_optimizer.py — Zero-order optimizer skeleton (student-implemented).

Strict black-box optimizer:
  - updates only the final classifier head
  - never reads gradients, logits, labels, or intermediate features
  - uses only scalar calls to ``loss_fn()``
"""

from __future__ import annotations

from typing import Callable, Dict, List

import torch
import torch.nn as nn


class ZeroOrderOptimizer:
    """Black-box optimizer for fine-tuning a low-dimensional head parameterization."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 5e-3,
        eps: float = 1e-2,
        perturbation_mode: str = "gaussian",
    ) -> None:
        self.model = model
        self.lr = lr
        self.eps = eps

        if perturbation_mode not in ("gaussian", "uniform"):
            raise ValueError(
                f"perturbation_mode must be 'gaussian' or 'uniform', "
                f"got '{perturbation_mode}'"
            )
        self.perturbation_mode = perturbation_mode

        self.layer_names: List[str] = ["fc.bias"]

        self.K: int = 2
        self.q: int = 32

        self.beta1: float = 0.9
        self.beta2: float = 0.999
        self.eps_adam: float = 1e-8
        self.grad_clip: float = 10.0
        self._m: Dict[str, torch.Tensor] = {}
        self._v: Dict[str, torch.Tensor] = {}
        self._t: int = 0

        self.reset_adam_per_batch: bool = True

    def _active_params(self) -> Dict[str, nn.Parameter]:
        named = dict(self.model.named_parameters())
        missing = [n for n in self.layer_names if n not in named]
        if missing:
            raise KeyError(
                f"The following layer names were not found in the model: "
                f"{missing}. Use [n for n, _ in model.named_parameters()] "
                f"to inspect valid names."
            )
        return {n: named[n] for n in self.layer_names}

    def _sample_direction(self, param: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(param).bernoulli_(0.5).mul_(2.0).sub_(1.0)

    def _estimate_grad(
        self,
        loss_fn: Callable[[], float],
        params: Dict[str, nn.Parameter],
    ) -> Dict[str, torch.Tensor]:
        grads = {n: torch.zeros_like(p) for n, p in params.items()}
        with torch.no_grad():
            for _ in range(self.q):
                us = {n: self._sample_direction(p) for n, p in params.items()}

                for n, p in params.items():
                    p.data.add_(us[n], alpha=self.eps)
                f_plus = loss_fn()

                for n, p in params.items():
                    p.data.add_(us[n], alpha=-2.0 * self.eps)
                f_minus = loss_fn()

                for n, p in params.items():
                    p.data.add_(us[n], alpha=self.eps)

                coeff = (f_plus - f_minus) / (2.0 * self.eps)
                for n, u in us.items():
                    grads[n].add_(u, alpha=coeff)

            for n in grads:
                grads[n].div_(self.q)
        return grads

    def _update_params(
        self,
        params: Dict[str, nn.Parameter],
        grads: Dict[str, torch.Tensor],
    ) -> None:
        self._t += 1
        bc1 = 1.0 - self.beta1 ** self._t
        bc2 = 1.0 - self.beta2 ** self._t
        with torch.no_grad():
            for name, p in params.items():
                g = grads[name]
                gn = g.norm()
                if gn > self.grad_clip and gn > 0:
                    g = g * (self.grad_clip / gn)

                if name not in self._m:
                    self._m[name] = torch.zeros_like(p)
                    self._v[name] = torch.zeros_like(p)
                m = self._m[name].mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
                v = self._v[name].mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)

                m_hat = m / bc1
                denom = (v / bc2).sqrt_().add_(self.eps_adam)
                p.data.addcdiv_(m_hat, denom, value=-self.lr)

    def step(self, loss_fn: Callable[[], float]) -> float:
        if self.reset_adam_per_batch:
            self._m = {}
            self._v = {}
            self._t = 0

        params = self._active_params()
        with torch.no_grad():
            loss_curr = float(loss_fn())
        loss_initial = loss_curr

        for _ in range(self.K):
            p_snap = {n: p.data.clone() for n, p in params.items()}
            m_snap = {n: self._m[n].clone() for n in params if n in self._m}
            v_snap = {n: self._v[n].clone() for n in params if n in self._v}
            t_snap = self._t

            grads = self._estimate_grad(loss_fn, params)
            self._update_params(params, grads)

            with torch.no_grad():
                loss_new = float(loss_fn())
            if loss_new < loss_curr:
                loss_curr = loss_new
            else:
                with torch.no_grad():
                    for n, p in params.items():
                        p.data.copy_(p_snap[n])
                    for n, m_old in m_snap.items():
                        self._m[n].copy_(m_old)
                    for n, v_old in v_snap.items():
                        self._v[n].copy_(v_old)
                self._t = t_snap

        return loss_initial
