from typing import Dict, Any, List, Tuple
import json
from PIL import Image
import torch
from torch.utils.data import Dataset


def confidence_to_weight(conf: str) -> float:
    conf = (conf or "medium").lower().strip()
    if conf == "high":
        return 1.0
    if conf == "medium":
        return 0.7
    if conf == "low":
        return 0.4
    return 0.7


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def build_vocabs(items: List[Dict[str, Any]]) -> Tuple[Dict[str, int], Dict[str, int]]:
    wave_types = sorted({x["wave_type"] for x in items})
    directions = sorted({x["direction"] for x in items})
    return {k: i for i, k in enumerate(wave_types)}, {k: i for i, k in enumerate(directions)}


class SwellSightDataset(Dataset):
    def __init__(
        self,
        index_jsonl: str,
        transform=None,
        wave_type_to_id: Dict[str, int] = None,
        direction_to_id: Dict[str, int] = None,
    ):
        self.items = read_jsonl(index_jsonl)
        self.transform = transform

        if wave_type_to_id is None or direction_to_id is None:
            self.wave_type_to_id, self.direction_to_id = build_vocabs(self.items)
        else:
            self.wave_type_to_id = wave_type_to_id
            self.direction_to_id = direction_to_id

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.items[idx]
        img = Image.open(r["image_path"]).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        height = torch.tensor([float(r["height_meters"])], dtype=torch.float32)
        wave_type = torch.tensor(self.wave_type_to_id[r["wave_type"]], dtype=torch.long)
        direction = torch.tensor(self.direction_to_id[r["direction"]], dtype=torch.long)
        weight = torch.tensor([confidence_to_weight(r.get("confidence", "medium"))], dtype=torch.float32)

        return {
            "image": img,
            "height": height,
            "wave_type": wave_type,
            "direction": direction,
            "weight": weight,
            "meta": r
        }
