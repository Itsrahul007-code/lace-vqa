"""Image degradations that mimic photos taken by blind and low-vision users
(blur, poor lighting, poor framing, rotation, compression, sensor noise).

Every function maps (PIL.Image, severity in 1..5) -> PIL.Image and is deterministic given `seed`.
"""
import io

import numpy as np
from PIL import Image, ImageFilter


def _rng(seed):
    return np.random.default_rng(seed)


def gaussian_blur(img, s, seed=0):
    r = [1, 2, 3, 5, 8][s - 1] * max(img.size) / 500.0
    return img.filter(ImageFilter.GaussianBlur(radius=r))


def motion_blur(img, s, seed=0):
    k = int([5, 9, 15, 21, 31][s - 1] * max(img.size) / 500.0) | 1
    k = max(k, 3)
    angle = _rng(seed).uniform(0, 180)
    kern = np.zeros((k, k), np.float32)
    c = k // 2
    dx, dy = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    for t in np.linspace(-c, c, 2 * k):
        x, y = int(round(c + t * dx)), int(round(c + t * dy))
        if 0 <= x < k and 0 <= y < k:
            kern[y, x] = 1.0
    kern /= kern.sum()
    arr = np.asarray(img, dtype=np.float32)
    out = np.empty_like(arr)
    try:
        from scipy.signal import fftconvolve
        for ch in range(3):
            out[..., ch] = fftconvolve(arr[..., ch], kern, mode="same")
    except ImportError:  # pragma: no cover - PIL fallback (kernel size limited to 5)
        return img.filter(ImageFilter.BoxBlur(k // 2))
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def low_light(img, s, seed=0):
    arr = np.asarray(img, dtype=np.float32) / 255.0
    gain = [0.6, 0.45, 0.3, 0.2, 0.12][s - 1]
    arr = (arr ** 1.5) * gain
    noise = _rng(seed).normal(0, [0.01, 0.02, 0.03, 0.04, 0.05][s - 1], arr.shape)
    return Image.fromarray((np.clip(arr + noise, 0, 1) * 255).astype(np.uint8))


def overexposure(img, s, seed=0):
    arr = np.asarray(img, dtype=np.float32) / 255.0
    gain = [1.4, 1.8, 2.3, 3.0, 4.0][s - 1]
    return Image.fromarray((np.clip(arr * gain, 0, 1) * 255).astype(np.uint8))


def gaussian_noise(img, s, seed=0):
    arr = np.asarray(img, dtype=np.float32) / 255.0
    sigma = [0.04, 0.08, 0.12, 0.18, 0.26][s - 1]
    arr = arr + _rng(seed).normal(0, sigma, arr.shape)
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))


def jpeg(img, s, seed=0):
    q = [30, 18, 12, 8, 5][s - 1]
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def poor_framing(img, s, seed=0):
    """Keep only an off-centre window: the object of interest is partly out of frame."""
    keep = [0.85, 0.7, 0.55, 0.45, 0.35][s - 1]
    w, h = img.size
    cw, ch = int(w * keep), int(h * keep)
    rng = _rng(seed)
    corner = rng.integers(4)
    x0 = 0 if corner in (0, 2) else w - cw
    y0 = 0 if corner in (0, 1) else h - ch
    return img.crop((x0, y0, x0 + cw, y0 + ch))


def rotation(img, s, seed=0):
    angle = [15, 30, 90, 135, 180][s - 1]
    return img.rotate(angle, expand=True, fillcolor=(0, 0, 0))


CORRUPTIONS = {
    "gaussian_blur": gaussian_blur,
    "motion_blur": motion_blur,
    "low_light": low_light,
    "overexposure": overexposure,
    "gaussian_noise": gaussian_noise,
    "jpeg": jpeg,
    "poor_framing": poor_framing,
    "rotation": rotation,
}


def apply(img, name, severity, seed=0):
    if not name or name == "none" or severity == 0:
        return img
    return CORRUPTIONS[name](img, int(severity), seed)
