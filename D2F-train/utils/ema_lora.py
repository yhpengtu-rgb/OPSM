"""
EMA LoRA weight manager for DMD (Distribution Matching Distillation).

Only LoRA adapter parameters (~33.5M = 67MB) are shadow-copied; the 16GB
base model is shared.  The fake-model forward pass is implemented by
temporarily swapping EMA weights into the student model's LoRA layers,
running a no-grad forward, then restoring the current trainable weights.

Usage::

    ema = EMALoRA(student_model, decay=0.999)
    # after each optimizer.step():
    ema.update(student_model)
    # fake-model forward:
    with torch.no_grad():
        with ema.swap(student_model):
            logits_fake = student_model(input_ids, **kwargs).logits
"""

from contextlib import contextmanager
from typing import Dict, List
import torch
import torch.nn as nn


def _unwrap(model: nn.Module) -> nn.Module:
    """Peel off DeepSpeed / DDP wrappers to reach the PeftModel."""
    while hasattr(model, "module"):
        model = model.module
    return model


class EMALoRA:
    """Exponential moving average of LoRA adapter parameters.

    Args:
        student_model: The trainable student model (DeepSpeed / DDP / Peft
            wrapped).  Only parameters with ``requires_grad=True`` (LoRA
            adapters) are tracked.
        decay: EMA decay rate (0.999 = slow tracking, 0.9 = fast).
    """

    def __init__(self, student_model: nn.Module, decay: float = 0.999):
        base = _unwrap(student_model)
        self._param_names: List[str] = [
            n for n, p in base.named_parameters() if p.requires_grad
        ]
        self.shadow: Dict[str, torch.Tensor] = {
            n: base.get_parameter(n).data.clone() for n in self._param_names
        }
        self.decay = decay
        self._backup: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, student_model: nn.Module) -> None:
        """In-place EMA update of shadow weights from current student params."""
        base = _unwrap(student_model)
        for n in self._param_names:
            self.shadow[n].mul_(self.decay).add_(
                base.get_parameter(n).data, alpha=1.0 - self.decay
            )

    @contextmanager
    def swap(self, student_model: nn.Module):
        """Context manager: temporarily swap EMA weights into the student.

        Inside the ``with`` block, the student model's LoRA parameters are
        replaced by their EMA copies, enabling a fake-model forward pass.
        The original trainable weights are restored on exit.
        """
        base = _unwrap(student_model)
        # Backup current trainable weights
        self._backup = {
            n: base.get_parameter(n).data.clone() for n in self._param_names
        }
        # Swap in EMA weights
        for n in self._param_names:
            base.get_parameter(n).data.copy_(self.shadow[n])
        try:
            yield
        finally:
            # Restore original weights
            for n in self._param_names:
                base.get_parameter(n).data.copy_(self._backup[n])
            self._backup.clear()
