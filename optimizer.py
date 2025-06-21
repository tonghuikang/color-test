"""AdamW optimizer that inherits from torch.optim.AdamW and allows negative learning rates."""

from __future__ import annotations

from typing import Iterable, Tuple, Union

import torch
from torch import Tensor
from torch.optim import AdamW


class CustomAdamW(AdamW):
    """AdamW optimizer that allows negative learning rates.

    This class inherits from torch.optim.AdamW and overrides the learning rate
    validation to allow negative values.

    Args:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-3). Can be negative.
        betas: coefficients used for computing running averages of gradient
            and its square (default: (0.9, 0.999))
        eps: term added to the denominator to improve numerical stability
            (default: 1e-8)
        weight_decay: weight decay coefficient (default: 0.01)
        amsgrad: whether to use the AMSGrad variant of this algorithm
            from the paper `On the Convergence of Adam and Beyond`
            (default: False)
        maximize: maximize the params based on the objective, instead of minimizing
            (default: False)
        foreach: whether foreach implementation of optimizer is used (default: None)
        capturable: whether this instance is safe to capture in a CUDA graph
            (default: False)
        differentiable: whether autograd should occur through the optimizer step
            (default: False)
        fused: whether fused implementation of optimizer is used (default: None)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict],
        lr: Union[float, Tensor] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
        *,
        maximize: bool = False,
        foreach: bool | None = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: bool | None = None,
    ) -> None:
        # Validate parameters except learning rate
        if isinstance(lr, Tensor):
            if foreach and not capturable:
                raise ValueError(
                    "lr as a Tensor is not supported for capturable=False and foreach=True"
                )
            if lr.numel() != 1:
                raise ValueError("Tensor lr must be 1-element")
        # Skip the lr >= 0 check to allow negative learning rates
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        # Create defaults dict before calling parent __init__
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            foreach=foreach,
            maximize=maximize,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
            decoupled_weight_decay=False,  # Add missing parameter
        )

        # Call parent Optimizer.__init__ directly to bypass AdamW's lr validation
        torch.optim.Optimizer.__init__(self, params, defaults)

        # Handle fused optimizer settings
        if fused:
            if differentiable:
                raise RuntimeError("`fused` does not support `differentiable`")
            self._step_supports_amp_scaling = True
            if foreach:
                raise RuntimeError("`fused` and `foreach` cannot be `True` together.")
