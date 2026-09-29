"""PWC-Net / PWC-DC-Net as a plain nn.Module for Vitis AI quantization.

Numerically equivalent to ptlflow's `pwcnet` (PWC-DC-Net, used by the pipeline)
and `pwcnet_nodc`, but:
  * tensor in / tensor out: input [B, 6, H, W] = two RGB images in [0, 1],
    output = quarter-resolution flow before the x20 scaling (flow2);
  * correlation implemented with nn.Unfold instead of a CUDA extension.
H and W must be multiples of 64. Use `postprocess` to get full-size flow.

Only depends on torch/numpy so it runs inside the Vitis AI docker (py3.8).
Deployment note: warping (grid_sample) and correlation (unfold) are not DPU
operators; the Vitis AI compiler places them on the CPU. See deploy/vitis_ai/README.md.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DIV_FLOW = 20.0
PTLFLOW_CHECKPOINTS = {
    ("pwcnet", "things"): "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/pwcdcnet-things-cc223701.ckpt",
    ("pwcnet", "sintel"): "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/pwcdcnet-sintel-c7d08a46.ckpt",
    ("pwcnet_nodc", "things"): "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/pwcnet-things-6a2e540b.ckpt",
    ("pwcnet_nodc", "sintel"): "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/pwcnet-sintel-533815e5.ckpt",
}


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, bias=True),
        nn.LeakyReLU(0.1),
    )


def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)


def deconv(in_planes, out_planes):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True)


class Correlation(nn.Module):
    """Local cost volume: mean over channels of f1 * shifted f2, displacements in [-md, md]^2."""

    def __init__(self, max_displacement: int = 4):
        super().__init__()
        self.side = 2 * max_displacement + 1
        self.unfold = nn.Unfold(kernel_size=self.side, padding=max_displacement)

    def forward(self, f1, f2):
        b, c, h, w = f1.shape
        patches = self.unfold(f2).view(b, c, self.side * self.side, h, w)
        return (f1.unsqueeze(2) * patches).mean(dim=1)


class PWCNetDPU(nn.Module):
    def __init__(self, dc: bool = True, md: int = 4):
        super().__init__()
        self.dc = dc
        self.conv1a, self.conv1aa, self.conv1b = conv(3, 16, stride=2), conv(16, 16), conv(16, 16)
        self.conv2a, self.conv2aa, self.conv2b = conv(16, 32, stride=2), conv(32, 32), conv(32, 32)
        self.conv3a, self.conv3aa, self.conv3b = conv(32, 64, stride=2), conv(64, 64), conv(64, 64)
        self.conv4a, self.conv4aa, self.conv4b = conv(64, 96, stride=2), conv(96, 96), conv(96, 96)
        self.conv5a, self.conv5aa, self.conv5b = conv(96, 128, stride=2), conv(128, 128), conv(128, 128)
        self.conv6aa, self.conv6a, self.conv6b = conv(128, 196, stride=2), conv(196, 196), conv(196, 196)
        self.corr = Correlation(md)
        self.leakyRELU = nn.LeakyReLU(0.1)

        nd = (2 * md + 1) ** 2
        dd = np.cumsum([128, 128, 96, 64, 32]).tolist()
        # Decoder levels 6..2; attribute names match ptlflow so checkpoints load strictly.
        for level, extra in ((6, 0), (5, 128 + 4), (4, 96 + 4), (3, 64 + 4), (2, 32 + 4)):
            od = nd + extra
            for i, out in enumerate((128, 128, 96, 64, 32)):
                setattr(self, f"conv{level}_{i}", conv(od + (dd[i - 1] if i else 0), out))
            setattr(self, f"predict_flow{level}", predict_flow(od + dd[4]))
            if level > 2:
                setattr(self, f"deconv{level}", deconv(2, 2))
                setattr(self, f"upfeat{level}", deconv(od + dd[4], 2))
        if dc:
            od = nd + 32 + 4 + dd[4]
            self.dc_conv1 = conv(od, 128)
            self.dc_conv2 = conv(128, 128, padding=2, dilation=2)
            self.dc_conv3 = conv(128, 128, padding=4, dilation=4)
            self.dc_conv4 = conv(128, 96, padding=8, dilation=8)
            self.dc_conv5 = conv(96, 64, padding=16, dilation=16)
            self.dc_conv6 = conv(64, 32)
            self.dc_conv7 = predict_flow(32)

    @staticmethod
    def warp(x, flow):
        """Backward-warp x by flow; zero where the sample falls outside the image (as ptlflow)."""
        # Sizes as Python ints and the grid built directly in [-1, 1]: XIR cannot export
        # shape arithmetic or multi-output ops (meshgrid). linspace(-1, 1, w)[x] = 2x/(w-1) - 1.
        h, w = int(x.shape[2]), int(x.shape[3])
        base_x = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype).view(1, 1, w)
        base_y = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype).view(1, h, 1)
        gx = base_x + flow[:, 0] * (2.0 / max(w - 1, 1))
        gy = base_y + flow[:, 1] * (2.0 / max(h - 1, 1))
        vgrid = torch.stack((gx, gy), dim=-1)
        output = F.grid_sample(x, vgrid, align_corners=True)
        mask = F.grid_sample(torch.ones_like(x), vgrid, align_corners=True)
        return output * (mask >= 0.9999).to(x.dtype)

    def _cost(self, f1, f2):
        return self.leakyRELU(self.corr(f1, f2))

    def _decode(self, level: int, x):
        for i in range(5):
            x = torch.cat((getattr(self, f"conv{level}_{i}")(x), x), 1)
        return x

    def _pyramid(self, im):
        c1 = self.conv1b(self.conv1aa(self.conv1a(im)))
        c2 = self.conv2b(self.conv2aa(self.conv2a(c1)))
        c3 = self.conv3b(self.conv3aa(self.conv3a(c2)))
        c4 = self.conv4b(self.conv4aa(self.conv4a(c3)))
        c5 = self.conv5b(self.conv5aa(self.conv5a(c4)))
        c6 = self.conv6b(self.conv6a(self.conv6aa(c5)))
        return {2: c2, 3: c3, 4: c4, 5: c5, 6: c6}

    def forward(self, images):
        p1, p2 = self._pyramid(images[:, :3]), self._pyramid(images[:, 3:])
        x = self._decode(6, self._cost(p1[6], p2[6]))
        flow = self.predict_flow6(x)
        up_flow, up_feat = self.deconv6(flow), self.upfeat6(x)
        for level, scale in ((5, 0.625), (4, 1.25), (3, 2.5), (2, 5.0)):
            cost = self._cost(p1[level], self.warp(p2[level], up_flow * scale))
            x = self._decode(level, torch.cat((cost, p1[level], up_flow, up_feat), 1))
            flow = getattr(self, f"predict_flow{level}")(x)
            if level > 2:
                up_flow = getattr(self, f"deconv{level}")(flow)
                up_feat = getattr(self, f"upfeat{level}")(x)
        if self.dc:
            x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
            flow = flow + self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))
        return flow


DPU_LEAKY_SLOPE = 26 / 256  # the only LeakyReLU slope DPUCZDX8G implements (0.1015625)


def set_leaky_slope(model: nn.Module, slope: float) -> nn.Module:
    """PWC-Net is trained with slope 0.1; with 0.1 every conv+activation falls back to the CPU
    on the DPU, so the deploy graph uses DPU_LEAKY_SLOPE (flow impact measured in the README)."""
    for m in model.modules():
        if isinstance(m, nn.LeakyReLU):
            m.negative_slope = slope
    return model


def load_ptlflow_weights(model: PWCNetDPU, checkpoint: str = "things") -> PWCNetDPU:
    """Load the official ptlflow checkpoint (downloaded and cached by torch.hub)."""
    name = "pwcnet" if model.dc else "pwcnet_nodc"
    ckpt = torch.hub.load_state_dict_from_url(PTLFLOW_CHECKPOINTS[(name, checkpoint)], map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model.eval()


def preprocess(prev_bgr: np.ndarray, curr_bgr: np.ndarray, to_rgb: bool = True) -> torch.Tensor:
    """Two uint8 BGR frames (H, W, 3) -> [1, 6, H, W] in [0, 1].

    ptlflow feeds PWC-DC-Net RGB but plain PWC-Net BGR, so pass to_rgb=model.dc.
    """
    order = (slice(None), slice(None), slice(None, None, -1 if to_rgb else 1))
    pair = np.concatenate([prev_bgr[order], curr_bgr[order]], axis=2)
    return torch.from_numpy(np.ascontiguousarray(pair)).permute(2, 0, 1).float().div(255.0).unsqueeze(0)


def postprocess(flow2: torch.Tensor) -> torch.Tensor:
    """Quarter-resolution network output -> full-resolution flow in pixels."""
    return F.interpolate(flow2 * DIV_FLOW, scale_factor=4, mode="bilinear", align_corners=True)
