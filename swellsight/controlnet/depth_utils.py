import numpy as np
import cv2

def add_realism_to_depth(depth_u8: np.ndarray, seed: int = 0) -> np.ndarray:
    """
    Add mild realism without stamping circular occlusions.
    Keeps the depth usable for ControlNet but avoids visible circle artifacts.
    """
    rng = np.random.default_rng(seed)

    # Ensure uint8 grayscale
    out = depth_u8.copy()
    if out.dtype != np.uint8:
        out = out.astype(np.uint8)

    # Very mild blur to remove banding
    out = cv2.GaussianBlur(out, (0, 0), sigmaX=0.6, sigmaY=0.6)

    # Add tiny grain noise (no shapes)
    noise = rng.normal(0.0, 2.0, size=out.shape).astype(np.float32)
    out_f = out.astype(np.float32) + noise

    out_f = np.clip(out_f, 0, 255)
    return out_f.astype(np.uint8)

