import numpy as np
import cv2
from PIL import Image


def robust_normalize_to_u8(depth: np.ndarray, invert: bool = False, eps: float = 1e-8) -> np.ndarray:
    """
    Convert a float depth map to uint8 using robust percentile clipping.
    - Handles NaNs/Infs safely
    - Optionally inverts the mapping
    """
    d = np.asarray(depth, dtype=np.float32)

    # Replace non-finite values to avoid breaking normalization
    finite_mask = np.isfinite(d)
    if not np.any(finite_mask):
        return np.zeros(d.shape, dtype=np.uint8)

    v = d[finite_mask]
    lo = np.percentile(v, 2.0)
    hi = np.percentile(v, 98.0)
    if (hi - lo) < eps:
        hi = lo + 1.0

    d = np.clip(d, lo, hi)
    d = (d - lo) / (hi - lo + eps)

    if invert:
        d = 1.0 - d

    d_u8 = (d * 255.0).astype(np.uint8)

    # If there were NaNs originally, set them to 0 (or any neutral value)
    if not np.all(finite_mask):
        d_u8[~finite_mask] = 0

    return d_u8


def to_control_image(depth_u8: np.ndarray, size: tuple[int, int] = (1024, 1024)) -> Image.Image:
    """
    Convert a uint8 depth map into a 3-channel PIL image, resized for ControlNet.
    """
    d = np.asarray(depth_u8)
    if d.dtype != np.uint8:
        d = d.astype(np.uint8)

    # Ensure HxW
    if d.ndim == 3:
        d = d[..., 0]

    # Resize with good quality
    d_resized = cv2.resize(d, size, interpolation=cv2.INTER_CUBIC)

    # ControlNet expects RGB-like input, replicate grayscale into 3 channels
    rgb = np.stack([d_resized, d_resized, d_resized], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def add_realism_to_depth(depth_u8: np.ndarray, seed: int = 0) -> np.ndarray:
    """
    Add mild realism without stamping circular occlusions.
    - Mild blur
    - Tiny grain noise
    """
    rng = np.random.default_rng(seed)

    out = np.asarray(depth_u8)
    if out.dtype != np.uint8:
        out = out.astype(np.uint8)

    # Very mild blur to remove banding
    out = cv2.GaussianBlur(out, (0, 0), sigmaX=0.6, sigmaY=0.6)

    # Tiny grain noise (no shapes)
    noise = rng.normal(0.0, 2.0, size=out.shape).astype(np.float32)
    out_f = out.astype(np.float32) + noise
    out_f = np.clip(out_f, 0, 255)

    return out_f.astype(np.uint8)
