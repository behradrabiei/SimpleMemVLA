
from __future__ import annotations

_ALIASED_SUBMODULES = ("_utils", "placement_types")


def apply_fla_torch_compat() -> None:
    try:
        import torch.distributed._tensor as _priv
        import torch.distributed.tensor as _pub
    except Exception:
        return

    def _reexport(src) -> None:
        for name in dir(src):
            if not name.startswith("__") and not hasattr(_pub, name):
                setattr(_pub, name, getattr(src, name))

    _reexport(_priv)
    try:
        import torch.distributed._tensor.placement_types as _plt

        _reexport(_plt)
    except Exception:
        pass

    _alias_submodules(_pub)


def _alias_submodules(pub) -> None:
    import importlib
    import sys

    for name in _ALIASED_SUBMODULES:
        public_path = f"torch.distributed.tensor.{name}"
        try:
            importlib.import_module(public_path)
            continue
        except ImportError:
            pass
        try:
            module = importlib.import_module(f"torch.distributed._tensor.{name}")
        except ImportError:
            continue
        sys.modules[public_path] = module
        setattr(pub, name, module)


def apply_patch_embed_linear_compat() -> None:
    """Run the Qwen3.5 vision patch embedding as a linear map instead of Conv3d.

    Local fix (RTX 5090 env, torch 2.9.1+cu128): a bf16/fp16 ``nn.Conv3d`` *with a bias*
    takes ~0.7 s for one 256x256 image (256 patches) on this torch, against ~6 ms without
    the bias and ~0.02 ms as a linear layer, and the eval embeds several patches per
    decision. The patch embedding has kernel == stride, so it is exactly a linear map over
    the flattened (C, T, H, W) patch; outputs agree with the convolution to bf16 rounding.
    Set SIMPLEMEMVLA_CONV3D_PATCH_EMBED=1 to keep the original convolution.
    """
    import os

    if os.environ.get("SIMPLEMEMVLA_CONV3D_PATCH_EMBED", "0") not in ("", "0"):
        return
    try:
        import torch.nn.functional as F
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    except Exception:
        return
    cls = qwen.Qwen3_5VisionPatchEmbed
    if getattr(cls, "_linear_patch_embed", False):
        return

    def forward(self, hidden_states):
        proj = self.proj
        patch_dim = (
            self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
        )
        flat = hidden_states.reshape(-1, patch_dim).to(dtype=proj.weight.dtype)
        return F.linear(flat, proj.weight.reshape(self.embed_dim, patch_dim), proj.bias)

    cls.forward = forward
    cls._linear_patch_embed = True
