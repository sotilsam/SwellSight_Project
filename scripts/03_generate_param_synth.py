import os
import argparse
from PIL import Image
import numpy as np

from swellsight.utils import write_jsonl, ensure_dir, set_seed
from swellsight.controlnet.param_depth_generator import sample_wave_params, generate_param_depth_map_v2
from swellsight.controlnet.depth_utils import robust_normalize_to_u8, to_control_image, add_realism_to_depth
from swellsight.controlnet.sdxl_depth_controlnet import SDXLDepthControlNet

# --- Label space ---
WAVE_TYPES = ["beach_break", "reef_break", "point_break", "closeout", "a_frame"]
DIRECTIONS = ["left", "right", "both"]

# --- Prompt templates ---
PROMPT_TEMPLATE = (
    "Ultra realistic beach camera photo, ocean water surface, waves breaking near shore, "
    "visible shoreline and run-up foam, visible horizon line, oblique angle from sand level, "
    "wave height about {height:.1f} meters, breaking type {wave_type}, peeling direction {direction}, "
    "sea spray, whitewater, foam patterns, natural daylight, sharp focus, high detail, photo"
)

NEG_PROMPT = (
    "sand, dunes, desert, ripple sand, seabed, underwater, top-down, aerial, drone, "
    "cartoon, illustration, painting, CGI, low detail, blurry, flat water, "
    "text, watermark, logo, people, surfers, boats, buildings"
    "bokeh, lens flare, circular blur, circle artifact, ring, discs, spots"
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--guidance", type=float, default=6.0)
    p.add_argument("--control_scale", type=float, default=0.75)
    p.add_argument("--use_cpu_offload", action="store_true")
    p.add_argument("--depth_size", type=int, default=768)
    p.add_argument("--out_size", type=int, default=1024)
    p.add_argument("--min_height", type=float, default=0.2)
    p.add_argument("--max_height", type=float, default=2.5)
    args = p.parse_args()

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    ensure_dir(args.out_dir)
    images_dir = os.path.join(args.out_dir, "images")
    depth_dir = os.path.join(args.out_dir, "depth")
    ensure_dir(images_dir)
    ensure_dir(depth_dir)

    gen = SDXLDepthControlNet(use_cpu_offload=args.use_cpu_offload)

    out_items = []

    for i in range(args.n):
        # Balanced labels across the dataset
        wave_type = WAVE_TYPES[i % len(WAVE_TYPES)]
        direction = DIRECTIONS[(i // len(WAVE_TYPES)) % len(DIRECTIONS)]

        # Sample physical parameters for depth generation
        params = sample_wave_params(rng)

        # Ensure height is continuous and matches our label range
        height = float(rng.uniform(args.min_height, args.max_height))
        params["height_meters"] = height
        params["wave_type"] = wave_type
        params["direction"] = direction

        # Generate depth map (v2, beach-camera-like)
        depth_float = generate_param_depth_map_v2(
            params,
            size=(args.depth_size, args.depth_size),
            seed=args.seed + i
        )

        depth_u8 = robust_normalize_to_u8(depth_float, invert=True)
        depth_u8 = add_realism_to_depth(depth_u8, seed=args.seed + 10000 + i)

        depth_path = os.path.join(depth_dir, f"depth_{i:06d}.png")
        rgb_path = os.path.join(images_dir, f"rgb_{i:06d}.png")

        Image.fromarray(depth_u8, mode="L").save(depth_path)

        control_img = to_control_image(depth_u8, size=(args.out_size, args.out_size))
        prompt = PROMPT_TEMPLATE.format(height=height, wave_type=wave_type, direction=direction)

        rgb = gen.generate(
            depth_control_image=control_img,
            prompt=prompt,
            negative_prompt=NEG_PROMPT,
            out_size=args.out_size,
            seed=args.seed + i,
            steps=args.steps,
            guidance=args.guidance,
            control_scale=args.control_scale
        )
        rgb.save(rgb_path)

        rec = {
            "image_path": rgb_path,
            "depth_path": depth_path,
            "height_meters": height,
            "wave_type": wave_type,
            "direction": direction,
            "confidence": "high",
            "notes": "synthetic_from_params_depth_controlnet",
            "data_key": 999,
            "source": "param_synth",
            "prompt": prompt
        }
        out_items.append(rec)

    out_index = os.path.join(args.out_dir, "index.jsonl")
    write_jsonl(out_items, out_index)
    print(f"Saved {len(out_items)} param synthetic samples to {out_index}")


if __name__ == "__main__":
    main()
