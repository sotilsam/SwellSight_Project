import os
import json
import argparse
from PIL import Image

from swellsight.utils import read_jsonl, write_jsonl, ensure_dir, set_seed
from swellsight.controlnet.dpt_depth import DPTDepthExtractor
from swellsight.controlnet.depth_utils import robust_normalize_to_u8, to_control_image, add_realism_to_depth
from swellsight.controlnet.sdxl_depth_controlnet import SDXLDepthControlNet


DEFAULT_PROMPT = (
    "ultra photorealistic ocean waves breaking near shore, "
    "beach camera perspective from sand level, "
    "visible shoreline run-up, realistic sea foam, detailed water surface, "
    "natural daylight, no people, no surfers, no boats"
)

DEFAULT_NEG = (
    "cartoon, illustration, painting, CGI, low detail, blurry, flat water, "
    "text, watermark, logo, people, surfers, boats, buildings"
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real_train_jsonl", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=35)
    p.add_argument("--guidance", type=float, default=7.0)
    p.add_argument("--control_scale", type=float, default=1.0)
    p.add_argument("--use_cpu_offload", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)

    images_dir = os.path.join(args.out_dir, "images")
    depth_dir = os.path.join(args.out_dir, "depth")
    ensure_dir(images_dir)
    ensure_dir(depth_dir)

    items = read_jsonl(args.real_train_jsonl)

    dpt = DPTDepthExtractor()
    gen = SDXLDepthControlNet(use_cpu_offload=args.use_cpu_offload)

    out_items = []
    uid = 0

    for r in items:
        src_img = Image.open(r["image_path"]).convert("RGB")

        for k in range(args.repeats):
            depth_float = dpt.predict_depth(src_img)
            depth_u8 = robust_normalize_to_u8(depth_float, invert=True)
            depth_u8 = add_realism_to_depth(depth_u8, seed=args.seed + uid)

            depth_path = os.path.join(depth_dir, f"depth_{uid:06d}.png")
            rgb_path = os.path.join(images_dir, f"rgb_{uid:06d}.png")

            Image.fromarray(depth_u8, mode="L").save(depth_path)

            control_img = to_control_image(depth_u8, size=(1024, 1024))

            rgb = gen.generate(
                depth_control_image=control_img,
                prompt=DEFAULT_PROMPT,
                negative_prompt=DEFAULT_NEG,
                out_size=1024,
                seed=args.seed + uid,
                steps=args.steps,
                guidance=args.guidance,
                control_scale=args.control_scale
            )
            rgb.save(rgb_path)

            nr = dict(r)
            nr["image_path"] = rgb_path
            nr["depth_path"] = depth_path
            nr["source"] = "real_aug"
            out_items.append(nr)

            uid += 1

    out_index = os.path.join(args.out_dir, "index.jsonl")
    write_jsonl(out_items, out_index)
    print(f"Saved {len(out_items)} augmented samples to {out_index}")


if __name__ == "__main__":
    main()
