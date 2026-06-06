"""PyTorch FaceEnhancer (DeepFaceLab x4 face upscaler)."""

from __future__ import annotations

import os
import threading
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from externals.face_lib.model_paths import model_path

_LOCK = threading.Lock()
_ENHANCER: FaceEnhancerNet | None = None


def _use_cpu(*, use_cpu: bool = False) -> bool:
    if use_cpu:
        return True
    raw = os.environ.get("AH_FACE_GPU", "1").strip().lower()
    return raw in ("0", "false", "no", "off", "cpu")


def _device(*, use_cpu: bool = False) -> torch.device:
    if _use_cpu(use_cpu=use_cpu):
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_conv2d(module: nn.Conv2d, weights: dict[str, np.ndarray], prefix: str) -> None:
    w = np.asarray(weights[f"{prefix}/weight:0"], dtype=np.float32)
    module.weight.data.copy_(torch.from_numpy(np.transpose(w, (3, 2, 0, 1))))
    b = np.asarray(weights[f"{prefix}/bias:0"], dtype=np.float32).reshape(-1)
    module.bias.data.copy_(torch.from_numpy(b))


def _load_linear(module: nn.Linear, weights: dict[str, np.ndarray], prefix: str) -> None:
    w = np.asarray(weights[f"{prefix}/weight:0"], dtype=np.float32)
    module.weight.data.copy_(torch.from_numpy(w.T))
    if module.bias is not None and f"{prefix}/bias:0" in weights:
        b = np.asarray(weights[f"{prefix}/bias:0"], dtype=np.float32).reshape(-1)
        module.bias.data.copy_(torch.from_numpy(b))


class FaceEnhancerNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.dense1 = nn.Linear(1, 64, bias=False)
        self.dense2 = nn.Linear(1, 64, bias=False)

        self.e0_conv0 = nn.Conv2d(64, 64, 3, padding=1)
        self.e0_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.e1_conv0 = nn.Conv2d(64, 112, 3, padding=1)
        self.e1_conv1 = nn.Conv2d(112, 112, 3, padding=1)
        self.e2_conv0 = nn.Conv2d(112, 192, 3, padding=1)
        self.e2_conv1 = nn.Conv2d(192, 192, 3, padding=1)
        self.e3_conv0 = nn.Conv2d(192, 336, 3, padding=1)
        self.e3_conv1 = nn.Conv2d(336, 336, 3, padding=1)
        self.e4_conv0 = nn.Conv2d(336, 512, 3, padding=1)
        self.e4_conv1 = nn.Conv2d(512, 512, 3, padding=1)

        self.center_conv0 = nn.Conv2d(512, 512, 3, padding=1)
        self.center_conv1 = nn.Conv2d(512, 512, 3, padding=1)
        self.center_conv2 = nn.Conv2d(512, 512, 3, padding=1)
        self.center_conv3 = nn.Conv2d(512, 512, 3, padding=1)

        self.d4_conv0 = nn.Conv2d(1024, 512, 3, padding=1)
        self.d4_conv1 = nn.Conv2d(512, 512, 3, padding=1)
        self.d3_conv0 = nn.Conv2d(848, 512, 3, padding=1)
        self.d3_conv1 = nn.Conv2d(512, 512, 3, padding=1)
        self.d2_conv0 = nn.Conv2d(704, 288, 3, padding=1)
        self.d2_conv1 = nn.Conv2d(288, 288, 3, padding=1)
        self.d1_conv0 = nn.Conv2d(400, 160, 3, padding=1)
        self.d1_conv1 = nn.Conv2d(160, 160, 3, padding=1)
        self.d0_conv0 = nn.Conv2d(224, 96, 3, padding=1)
        self.d0_conv1 = nn.Conv2d(96, 96, 3, padding=1)

        self.out1x_conv0 = nn.Conv2d(96, 48, 3, padding=1)
        self.out1x_conv1 = nn.Conv2d(48, 3, 3, padding=1)
        self.dec2x_conv0 = nn.Conv2d(96, 96, 3, padding=1)
        self.dec2x_conv1 = nn.Conv2d(96, 96, 3, padding=1)
        self.out2x_conv0 = nn.Conv2d(96, 48, 3, padding=1)
        self.out2x_conv1 = nn.Conv2d(48, 3, 3, padding=1)
        self.dec4x_conv0 = nn.Conv2d(96, 72, 3, padding=1)
        self.dec4x_conv1 = nn.Conv2d(72, 72, 3, padding=1)
        self.out4x_conv0 = nn.Conv2d(72, 36, 3, padding=1)
        self.out4x_conv1 = nn.Conv2d(36, 3, 3, padding=1)

    @classmethod
    def from_npy(cls, path: os.PathLike[str] | str) -> FaceEnhancerNet:
        raw = np.load(path, allow_pickle=True)
        weights = raw.item() if hasattr(raw, "item") else raw
        if not isinstance(weights, dict):
            raise ValueError(f"Unexpected FaceEnhancer weights format: {path}")

        model = cls()
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Conv2d):
                _load_conv2d(mod, weights, name)
            elif isinstance(mod, nn.Linear):
                _load_linear(mod, weights, name)
        return model

    def forward(
        self,
        bgr: torch.Tensor,
        param: torch.Tensor,
        param1: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv1(bgr)
        a = self.dense1(param).view(-1, 64, 1, 1)
        b = self.dense2(param1).view(-1, 64, 1, 1)
        x = F.leaky_relu(x + a + b, 0.1)

        x = F.leaky_relu(self.e0_conv0(x), 0.1)
        x = e0 = F.leaky_relu(self.e0_conv1(x), 0.1)

        x = F.avg_pool2d(x, 2)
        x = F.leaky_relu(self.e1_conv0(x), 0.1)
        x = e1 = F.leaky_relu(self.e1_conv1(x), 0.1)

        x = F.avg_pool2d(x, 2)
        x = F.leaky_relu(self.e2_conv0(x), 0.1)
        x = e2 = F.leaky_relu(self.e2_conv1(x), 0.1)

        x = F.avg_pool2d(x, 2)
        x = F.leaky_relu(self.e3_conv0(x), 0.1)
        x = e3 = F.leaky_relu(self.e3_conv1(x), 0.1)

        x = F.avg_pool2d(x, 2)
        x = F.leaky_relu(self.e4_conv0(x), 0.1)
        x = e4 = F.leaky_relu(self.e4_conv1(x), 0.1)

        x = F.avg_pool2d(x, 2)
        x = F.leaky_relu(self.center_conv0(x), 0.1)
        x = F.leaky_relu(self.center_conv1(x), 0.1)
        x = F.leaky_relu(self.center_conv2(x), 0.1)
        x = F.leaky_relu(self.center_conv3(x), 0.1)

        x = torch.cat([F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False), e4], dim=1)
        x = F.leaky_relu(self.d4_conv0(x), 0.1)
        x = F.leaky_relu(self.d4_conv1(x), 0.1)

        x = torch.cat([F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False), e3], dim=1)
        x = F.leaky_relu(self.d3_conv0(x), 0.1)
        x = F.leaky_relu(self.d3_conv1(x), 0.1)

        x = torch.cat([F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False), e2], dim=1)
        x = F.leaky_relu(self.d2_conv0(x), 0.1)
        x = F.leaky_relu(self.d2_conv1(x), 0.1)

        x = torch.cat([F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False), e1], dim=1)
        x = F.leaky_relu(self.d1_conv0(x), 0.1)
        x = F.leaky_relu(self.d1_conv1(x), 0.1)

        x = torch.cat([F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False), e0], dim=1)
        x = F.leaky_relu(self.d0_conv0(x), 0.1)
        x = d0 = F.leaky_relu(self.d0_conv1(x), 0.1)

        x = F.leaky_relu(self.out1x_conv0(x), 0.1)
        x = self.out1x_conv1(x)
        out1x = bgr + torch.tanh(x)

        x = d0
        x = F.leaky_relu(self.dec2x_conv0(x), 0.1)
        x = F.leaky_relu(self.dec2x_conv1(x), 0.1)
        d2x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        x = F.leaky_relu(self.out2x_conv0(d2x), 0.1)
        x = self.out2x_conv1(x)
        out2x = F.interpolate(out1x, scale_factor=2, mode="bilinear", align_corners=False) + torch.tanh(x)

        x = d2x
        x = F.leaky_relu(self.dec4x_conv0(x), 0.1)
        x = F.leaky_relu(self.dec4x_conv1(x), 0.1)
        d4x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        x = F.leaky_relu(self.out4x_conv0(d4x), 0.1)
        x = self.out4x_conv1(x)
        out4x = F.interpolate(out2x, scale_factor=2, mode="bilinear", align_corners=False) + torch.tanh(x)
        return out4x


class FaceEnhancer:
    """Inference wrapper with DFL-compatible patch tiling."""

    def __init__(self, *, use_cpu: bool = False) -> None:
        weights = model_path("FaceEnhancer.npy")
        if not weights.is_file():
            raise FileNotFoundError(f"Unable to load face model: {weights}")

        self.device = _device(use_cpu=use_cpu)
        self.model = FaceEnhancerNet.from_npy(weights).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def _run_patch(self, patch_hwc: np.ndarray) -> np.ndarray:
        # Input patch is RGB float HWC in [-1, 1]; model uses NCHW.
        tensor = torch.from_numpy(patch_hwc.transpose(2, 0, 1)).unsqueeze(0).to(
            self.device, dtype=torch.float32
        )
        param = torch.tensor([[0.2]], device=self.device, dtype=torch.float32)
        param1 = torch.tensor([[1.0]], device=self.device, dtype=torch.float32)
        out = self.model(tensor, param, param1)
        return out.squeeze(0).permute(1, 2, 0).cpu().numpy()

    def enhance(
        self,
        inp_img: np.ndarray,
        *,
        is_tanh: bool = False,
        preserve_size: bool = True,
    ) -> np.ndarray:
        if not is_tanh:
            inp_img = np.clip(inp_img * 2 - 1, -1, 1)

        up_res = 4
        patch_size = 192
        patch_size_half = patch_size // 2

        ih, iw, ic = inp_img.shape
        h, w, c = ih, iw, ic

        t_padding = b_padding = l_padding = r_padding = 0
        if h < patch_size:
            t_padding = (patch_size - h) // 2
            b_padding = (patch_size - h) - t_padding
        if w < patch_size:
            l_padding = (patch_size - w) // 2
            r_padding = (patch_size - w) - l_padding

        if t_padding:
            inp_img = np.concatenate(
                [np.zeros((t_padding, w, c), dtype=np.float32), inp_img], axis=0
            )
            h, w, c = inp_img.shape
        if b_padding:
            inp_img = np.concatenate(
                [inp_img, np.zeros((b_padding, w, c), dtype=np.float32)], axis=0
            )
            h, w, c = inp_img.shape
        if l_padding:
            inp_img = np.concatenate(
                [np.zeros((h, l_padding, c), dtype=np.float32), inp_img], axis=1
            )
            h, w, c = inp_img.shape
        if r_padding:
            inp_img = np.concatenate(
                [inp_img, np.zeros((h, r_padding, c), dtype=np.float32)], axis=1
            )
            h, w, c = inp_img.shape

        i_max = w - patch_size + 1
        j_max = h - patch_size + 1

        final_img = np.zeros((h * up_res, w * up_res, c), dtype=np.float32)
        final_img_div = np.zeros((h * up_res, w * up_res, 1), dtype=np.float32)

        ramp = np.concatenate(
            [
                np.linspace(0, 1, patch_size_half * up_res),
                np.linspace(1, 0, patch_size_half * up_res),
            ]
        )
        xx, yy = np.meshgrid(ramp, ramp)
        patch_mask = (xx * yy)[..., None]

        j = 0
        while j < j_max:
            i = 0
            while i < i_max:
                patch_img = inp_img[j : j + patch_size, i : i + patch_size, :]
                out = self._run_patch(patch_img)
                final_img[
                    j * up_res : (j + patch_size) * up_res,
                    i * up_res : (i + patch_size) * up_res,
                    :,
                ] += out * patch_mask
                final_img_div[
                    j * up_res : (j + patch_size) * up_res,
                    i * up_res : (i + patch_size) * up_res,
                    :,
                ] += patch_mask
                if i == i_max - 1:
                    break
                i = min(i + patch_size_half, i_max - 1)
            if j == j_max - 1:
                break
            j = min(j + patch_size_half, j_max - 1)

        final_img_div[final_img_div == 0] = 1.0
        final_img /= final_img_div

        if t_padding + b_padding + l_padding + r_padding:
            final_img = final_img[
                t_padding * up_res : (h - b_padding) * up_res,
                l_padding * up_res : (w - r_padding) * up_res,
                :,
            ]

        if preserve_size:
            final_img = cv2.resize(final_img, (iw, ih), interpolation=cv2.INTER_LANCZOS4)

        if not is_tanh:
            final_img = np.clip(final_img / 2 + 0.5, 0, 1)

        return final_img


def get_enhancer(*, use_cpu: bool = False) -> FaceEnhancer:
    global _ENHANCER
    with _LOCK:
        if _ENHANCER is None:
            _ENHANCER = FaceEnhancer(use_cpu=use_cpu)
        return _ENHANCER
