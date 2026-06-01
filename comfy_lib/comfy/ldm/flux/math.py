import torch
from einops import rearrange
from torch import Tensor

from comfy.ldm.modules.attention import optimized_attention
import comfy.model_management
import logging


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, mask=None, transformer_options={}) -> Tensor:
    if pe is not None:
        q, k = apply_rope(q, k, pe)
    heads = q.shape[1]
    x = optimized_attention(q, k, v, heads, skip_reshape=True, mask=mask, transformer_options=transformer_options)
    return x

def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    if comfy.model_management.is_device_mps(pos.device) or comfy.model_management.is_intel_xpu() or comfy.model_management.is_directml_enabled():
        device = torch.device("cpu")
    else:
        device = pos.device

    scale = torch.linspace(0, (dim - 2) / dim, steps=dim//2, dtype=torch.float64, device=device)
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32, device=device), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.to(dtype=torch.float32, device=pos.device)


def _apply_rope1(x: Tensor, freqs_cis: Tensor):
    x_ = x.to(dtype=freqs_cis.dtype).reshape(*x.shape[:-1], -1, 1, 2)
    if x_.shape[2] != 1 and freqs_cis.shape[2] != 1 and x_.shape[2] != freqs_cis.shape[2]:
        freqs_cis = freqs_cis[:, :, :x_.shape[2]]

    x_out = freqs_cis[..., 0] * x_[..., 0]
    x_out.addcmul_(freqs_cis[..., 1], x_[..., 1])

    return x_out.reshape(*x.shape).type_as(x)


def _apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor):
    return apply_rope1(xq, freqs_cis), apply_rope1(xk, freqs_cis)


def _kitchen_rope_available() -> bool:
    """comfy_kitchen rope ops (opt-in; extra VRAM). Need cu128+ and AH_COMFY_KITCHEN_ROPE=1."""
    import os

    raw = os.environ.get("AH_COMFY_KITCHEN_ROPE", "0").strip().lower()
    if raw not in ("1", "true", "yes", "on"):
        return False
    if not torch.version.cuda:
        return False
    try:
        cuda_version = tuple(map(int, str(torch.version.cuda).split(".")))
    except ValueError:
        return False
    # Match quant_ops: cu128/cu129 wheels report 12.8 / 12.9, not 13.x
    if cuda_version >= (13,):
        return True
    return len(cuda_version) >= 2 and cuda_version >= (12, 8)


try:
    import comfy.quant_ops

    if _kitchen_rope_available() and hasattr(comfy.quant_ops, "ck"):
        q_apply_rope = comfy.quant_ops.ck.apply_rope
        q_apply_rope1 = comfy.quant_ops.ck.apply_rope1

        def apply_rope(xq, xk, freqs_cis):
            if comfy.model_management.in_training:
                return _apply_rope(xq, xk, freqs_cis)
            return apply_rope1(xq, freqs_cis), apply_rope1(xk, freqs_cis)

        def apply_rope1(x, freqs_cis):
            if comfy.model_management.in_training:
                return _apply_rope1(x, freqs_cis)
            return q_apply_rope1(x, freqs_cis)
    else:
        raise ImportError("comfy_kitchen rope disabled (set AH_COMFY_KITCHEN_ROPE=1, need cu128+)")
except Exception as exc:
    if _kitchen_rope_available():
        logging.warning(
            "comfy_kitchen fast apply_rope unavailable (%s); using reference implementation.",
            exc,
        )
    else:
        logging.debug(
            "apply_rope: reference implementation (opt-in fast path: AH_COMFY_KITCHEN_ROPE=1).",
        )
    apply_rope = _apply_rope
    apply_rope1 = _apply_rope1
