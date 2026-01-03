import os
import json
import argparse
from swellsight.utils import ensure_dir


def build_real_index(images_dir: str, labels_json: str, out_jsonl: str) -> None:
    with open(labels_json, "r", encoding="utf-8") as f:
        labels = json.load(f)

    if not isinstance(labels, dict):
        raise ValueError("labels.json must be a dict: {filename: {...}}")

    ensure_dir(os.path.dirname(out_jsonl))

    records = []
    missing = 0

    for filename, ann in labels.items():
        img_path = os.path.join(images_dir, filename)
        if not os.path.exists(img_path):
            missing += 1
            continue

        rec = {
            "image_path": img_path,
            "height_meters": float(ann["height_meters"]),
            "wave_type": str(ann["wave_type"]),
            "direction": str(ann["direction"]),
            "confidence": str(ann.get("confidence", "medium")),
            "notes": str(ann.get("notes", "")),
            "data_key": int(ann.get("data_key", -1)),
            "source": "real",
        }
        records.append(rec)

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(records)} to {out_jsonl}")
    if missing:
        print(f"Warning: {missing} images missing on disk")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images_dir", required=True)
    p.add_argument("--labels_json", required=True)
    p.add_argument("--out_index", required=True)
    args = p.parse_args()
    build_real_index(args.images_dir, args.labels_json, args.out_index)


if __name__ == "__main__":
    main()
