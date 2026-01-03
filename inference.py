import json
import argparse
import torch
from PIL import Image
from torchvision import transforms

from swellsight.model import SwellSightNet


def build_infer_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--vocabs", required=True)
    p.add_argument("--image_size", type=int, default=224)
    args = p.parse_args()

    with open(args.vocabs, "r", encoding="utf-8") as f:
        voc = json.load(f)
    wt2id = voc["wave_type_to_id"]
    d2id = voc["direction_to_id"]
    id2wt = {v: k for k, v in wt2id.items()}
    id2d = {v: k for k, v in d2id.items()}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SwellSightNet(len(wt2id), len(d2id)).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    img = Image.open(args.image).convert("RGB")
    x = build_infer_transform(args.image_size)(img).unsqueeze(0).to(device)

    pred_h, pred_wt, pred_dir = model(x)

    height_m = float(pred_h.item())
    wave_type = id2wt[int(torch.argmax(pred_wt, dim=1).item())]
    direction = id2d[int(torch.argmax(pred_dir, dim=1).item())]

    print({"height_meters": height_m, "wave_type": wave_type, "direction": direction})


if __name__ == "__main__":
    main()
