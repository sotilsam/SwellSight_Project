from typing import Dict, Any, Tuple
import numpy as np
import cv2


def sample_wave_params(rng: np.random.Generator) -> Dict[str, Any]:
    # Procedural wave parameters for synthetic depth generation
    height_m = float(rng.uniform(0.2, 2.0))

    wave_type = rng.choice(["beach_break", "reef_break", "point_break", "closeout", "a_frame"]).item()
    direction = rng.choice(["left", "right", "both"]).item()

    wavelength = float(rng.uniform(10.0, 28.0))
    angle_deg = float(rng.uniform(-20.0, 20.0))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))

    return {
        "height_meters": height_m,
        "wave_type": wave_type,
        "direction": direction,
        "wavelength": wavelength,
        "angle_deg": angle_deg,
        "phase": phase,
        # Default: do not add occlusions (prevents circles in ControlNet outputs)
        "occlusion_mode": "none",  # options: "none", "soft_irregular"
    }


def _add_soft_irregular_occlusions(depth: np.ndarray, rng: np.random.Generator, max_count: int = 2) -> np.ndarray:
    """
    Optional: add very soft, irregular occlusions.
    This is designed to avoid obvious circles by using blurred ellipses and low contrast.
    Use only if you really need occlusions later.
    """
    out = depth.copy()
    H, W = out.shape

    count = int(rng.integers(0, max_count + 1))
    if count == 0:
        return out

    # Set occlusions to slightly farther depth, but not extreme
    far_val = float(out.max()) + 0.12

    for _ in range(count):
        cx = int(rng.integers(int(W * 0.20), int(W * 0.80)))
        cy = int(rng.integers(int(H * 0.20), int(H * 0.80)))

        ax = int(rng.integers(int(W * 0.03), int(W * 0.07)))
        ay = int(rng.integers(int(H * 0.03), int(H * 0.07)))
        angle = float(rng.uniform(0.0, 180.0))

        mask = np.zeros((H, W), dtype=np.float32)
        cv2.ellipse(mask, (cx, cy), (ax, ay), angle, 0, 360, 1.0, thickness=-1)

        # Heavy blur to remove hard edges
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=9.0, sigmaY=9.0)

        # Make the effect weak so it does not "stamp" artifacts into RGB
        strength = float(rng.uniform(0.08, 0.18))
        mask = mask * strength

        out = out * (1.0 - mask) + far_val * mask

    return out


def generate_param_depth_map_v2(
    params: Dict[str, Any],
    size: Tuple[int, int] = (768, 768),
    seed: int = 0,
) -> np.ndarray:
    """
    Beach-camera-like depth map for ControlNet Depth:
    - Strong depth gradient: far (top) is larger depth, near (bottom) is smaller depth
    - Perspective foreshortening: wave frequency increases toward horizon
    - Breaking line shaped by wave_type and direction
    - Run-up band near camera
    - Optional occlusions are disabled by default to prevent circles in the generated RGB
    """
    rng = np.random.default_rng(seed)
    H, W = size

    # Normalized coordinates: x in [-1, 1], y in [0, 1]
    # y=0 is near camera, y=1 is far/horizon
    x = np.linspace(-1.0, 1.0, W, dtype=np.float32)[None, :].repeat(H, axis=0)
    y = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None].repeat(W, axis=1)

    height = float(params.get("height_meters", 1.0))
    wave_type = str(params.get("wave_type", "beach_break"))
    direction = str(params.get("direction", "both"))

    # Base depth increases with y (farther is larger depth)
    gamma = 1.7
    y_p = y ** gamma
    base_depth = 0.6 + 3.5 * y_p

    # Direction controls breaker slant and approach angle
    if direction == "left":
        theta = np.deg2rad(18.0)
        slant = 0.14
    elif direction == "right":
        theta = np.deg2rad(-18.0)
        slant = -0.14
    else:
        theta = np.deg2rad(0.0)
        slant = 0.0

    # Wave frequency increases toward horizon (foreshortening)
    wavelength = float(params.get("wavelength", 18.0))
    k0 = (2.0 * np.pi) / max(wavelength, 1e-3)
    k = k0 * (1.0 + 2.8 * y_p)

    u = np.cos(theta) * x + np.sin(theta) * (y_p - 0.55)
    phase0 = float(params.get("phase", 0.0))
    phase = k * (u * 6.0) + phase0

    # Keep relief smaller than base depth
    amp = 0.08 + 0.18 * np.clip(height / 2.5, 0.0, 1.0)
    wave_relief = amp * np.sin(phase)

    # Soft spatial noise to avoid perfect stripes
    n = rng.normal(0.0, 1.0, size=(H, W)).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), sigmaX=3.0, sigmaY=3.0)
    n = (n - n.min()) / (n.max() - n.min() + 1e-8)
    noise_relief = (n - 0.5) * (0.04 + 0.03 * float(rng.random()))

    # Breaking line position and shape by wave_type
    break_y = 0.22 + 0.04 * float(rng.uniform(-1.0, 1.0))

    if wave_type == "closeout":
        curvature = 0.0
        irregular = 0.0
        slant *= 0.2
    elif wave_type == "a_frame":
        curvature = 0.10
        irregular = 0.01
        slant *= 0.3
    elif wave_type == "point_break":
        curvature = 0.03
        irregular = 0.01
        slant *= 1.2
    elif wave_type == "reef_break":
        curvature = 0.02
        irregular = 0.03
        slant *= 0.9
    else:
        curvature = 0.02
        irregular = 0.015
        slant *= 0.7

    # Breaker line equation
    if wave_type == "a_frame":
        line = break_y + slant * x + curvature * np.abs(x)
    else:
        line = break_y + slant * x + curvature * (x ** 2)

    # Irregularity along x
    if irregular > 0:
        ix = rng.normal(0.0, 1.0, size=(1, W)).astype(np.float32)
        ix = cv2.GaussianBlur(ix, (0, 0), sigmaX=10.0)
        ix = (ix - ix.min()) / (ix.max() - ix.min() + 1e-8)
        ix = (ix - 0.5) * irregular
        line = line + ix.repeat(H, axis=0)

    breaker_band = np.exp(-((y - line) ** 2) / (2.0 * (0.012 ** 2))).astype(np.float32)
    runup_band = np.exp(-((y - 0.10) ** 2) / (2.0 * (0.020 ** 2))).astype(np.float32)

    breaker_relief = 0.35 * breaker_band
    runup_relief = 0.22 * runup_band

    depth = base_depth - wave_relief - noise_relief - breaker_relief - runup_relief

    # Gentle shoreline slope near camera
    shore_slope = 0.25 * (1.0 - y) ** 2
    depth = depth - shore_slope

    # Optional occlusions, disabled by default
    occlusion_mode = str(params.get("occlusion_mode", "none"))
    if occlusion_mode == "soft_irregular":
        depth = _add_soft_irregular_occlusions(depth, rng=rng, max_count=2)

    return depth.astype(np.float32)


def generate_param_depth_map(params: Dict[str, Any], size: Tuple[int, int] = (768, 768), seed: int = 0) -> np.ndarray:
    # Backward-compatible wrapper
    return generate_param_depth_map_v2(params=params, size=size, seed=seed)
