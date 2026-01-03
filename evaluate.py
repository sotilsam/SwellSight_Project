import os
import json
import argparse
import torch
from torch.utils.data import DataLoader

from swellsight.data import SwellSightDataset
from swellsight.transforms import build_transforms
from swellsight.model import SwellSightNet
from swellsight.metrics import regression_metrics, accuracy, macro_f1


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test_jsonl", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--vocabs", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=64)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.vocabs, "r", encoding="utf-8") as f:
        voc = json.load(f)
    wt2id = voc["wave_type_to_id"]
    d2id = voc["direction_to_id"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SwellSightNet(len(wt2id), len(d2id)).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = SwellSightDataset(
        args.test_jsonl,
        transform=build_transforms(False, args.image_size),
        wave_type_to_id=wt2id,
        direction_to_id=d2id
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    all_h_pred, all_h_true = [], []
    all_wt_logits, all_wt_true = [], []
    all_dir_logits, all_dir_true = [], []

    for b in loader:
        x = b["image"].to(device)
        y_h = b["height"].to(device)
        y_wt = b["wave_type"].to(device).view(-1)
        y_dir = b["direction"].to(device).view(-1)

        pred_h, pred_wt, pred_dir = model(x)

        all_h_pred.append(pred_h.cpu())
        all_h_true.append(y_h.cpu())
        all_wt_logits.append(pred_wt.cpu())
        all_wt_true.append(y_wt.cpu())
        all_dir_logits.append(pred_dir.cpu())
        all_dir_true.append(y_dir.cpu())

    h_pred = torch.cat(all_h_pred, 0)
    h_true = torch.cat(all_h_true, 0)
    wt_logits = torch.cat(all_wt_logits, 0)
    wt_true = torch.cat(all_wt_true, 0)
    dir_logits = torch.cat(all_dir_logits, 0)
    dir_true = torch.cat(all_dir_true, 0)

    metrics = {}
    metrics.update(regression_metrics(h_pred, h_true))
    metrics["wave_type_acc"] = accuracy(wt_logits, wt_true)
    metrics["direction_acc"] = accuracy(dir_logits, dir_true)
    metrics["wave_type_macro_f1"] = macro_f1(wt_logits, wt_true, wt_logits.shape[1])
    metrics["direction_macro_f1"] = macro_f1(dir_logits, dir_true, dir_logits.shape[1])

    out_path = os.path.join(args.out_dir, "eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(metrics)


if __name__ == "__main__":
    main()
