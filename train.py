import os
import json
import argparse

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from swellsight.utils import set_seed, ensure_dir
from swellsight.data import SwellSightDataset, read_jsonl, build_vocabs
from swellsight.transforms import build_transforms
from swellsight.model import SwellSightNet
from swellsight.losses import MultiTaskLoss
from swellsight.metrics import regression_metrics, accuracy, macro_f1


def run_epoch(model, loader, optimizer, loss_fn, device, train: bool, scaler=None):
    model.train() if train else model.eval()

    total_loss = 0.0
    n = 0

    all_h_pred, all_h_true = [], []
    all_wt_logits, all_wt_true = [], []
    all_dir_logits, all_dir_true = [], []

    for batch in loader:
        x = batch["image"].to(device)
        y_h = batch["height"].to(device)
        y_wt = batch["wave_type"].to(device).view(-1)
        y_dir = batch["direction"].to(device).view(-1)
        w = batch["weight"].to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with autocast(enabled=(device == "cuda")):
                pred_h, pred_wt, pred_dir = model(x)
                loss, _ = loss_fn(pred_h, pred_wt, pred_dir, y_h, y_wt, y_dir, w)

            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.item() * x.size(0)
        n += x.size(0)

        all_h_pred.append(pred_h.detach().cpu())
        all_h_true.append(y_h.detach().cpu())
        all_wt_logits.append(pred_wt.detach().cpu())
        all_wt_true.append(y_wt.detach().cpu())
        all_dir_logits.append(pred_dir.detach().cpu())
        all_dir_true.append(y_dir.detach().cpu())

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

    return (total_loss / max(1, n)), metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_jsonl", required=True)
    p.add_argument("--val_jsonl", required=True)
    p.add_argument("--out_dir", required=True)

    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--dropout", type=float, default=0.35)
    p.add_argument("--resume", default="")
    args = p.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir)

    # Build vocabs from train set only
    train_items = read_jsonl(args.train_jsonl)
    wt2id, d2id = build_vocabs(train_items)

    with open(os.path.join(args.out_dir, "vocabs.json"), "w", encoding="utf-8") as f:
        json.dump({"wave_type_to_id": wt2id, "direction_to_id": d2id}, f, indent=2, ensure_ascii=False)

    train_ds = SwellSightDataset(
        args.train_jsonl,
        transform=build_transforms(True, args.image_size),
        wave_type_to_id=wt2id,
        direction_to_id=d2id
    )
    val_ds = SwellSightDataset(
        args.val_jsonl,
        transform=build_transforms(False, args.image_size),
        wave_type_to_id=wt2id,
        direction_to_id=d2id
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SwellSightNet(len(wt2id), len(d2id), dropout=args.dropout).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    loss_fn = MultiTaskLoss(1.0, 1.0, 1.0)
    scaler = GradScaler(enabled=(device == "cuda"))

    start_epoch = 1
    best_val = 1e9
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        best_val = float(ckpt.get("best_val", best_val))
        start_epoch = int(ckpt.get("epoch", 0)) + 1

    history = []
    patience = 6
    bad = 0

    for epoch in range(start_epoch, args.epochs + 1):
        tr_loss, tr_m = run_epoch(model, train_loader, optimizer, loss_fn, device, train=True, scaler=scaler)
        va_loss, va_m = run_epoch(model, val_loader, optimizer, loss_fn, device, train=False, scaler=scaler)

        scheduler.step(va_loss)

        row = {"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss, **{f"train_{k}": v for k, v in tr_m.items()}, **{f"val_{k}": v for k, v in va_m.items()}}
        history.append(row)
        print(row)

        ckpt_path = os.path.join(args.out_dir, "last.pt")
        torch.save({"model": model.state_dict(), "epoch": epoch, "best_val": best_val}, ckpt_path)

        if va_loss < best_val:
            best_val = va_loss
            bad = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "best_val": best_val}, os.path.join(args.out_dir, "best.pt"))
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping")
                break

    with open(os.path.join(args.out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"Done, best val loss {best_val}")


if __name__ == "__main__":
    main()
