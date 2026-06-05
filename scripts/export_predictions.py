from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "upstream" / "vendor"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data-root", default="data/raw/kvasir-seg")
    parser.add_argument("--split-file", default="data/splits/kvasir_split.json")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def split_samples(data_root: Path, split_file: Path, split: str) -> list[Path]:
    images = sorted((data_root / "images").glob("*.jpg"))
    with open(split_file) as file:
        indices = json.load(file)[split]
    return [images[idx] for idx in indices]


def add_vendor(name: str) -> Path:
    path = VENDOR / name
    sys.path.insert(0, str(path))
    return path


def torch_device(name: str):
    import torch

    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def load_torch_model(model, checkpoint: Path, device):
    import torch

    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def latest_checkpoint(paths: list[Path]) -> Path:
    matches = []
    for path in paths:
        matches.extend(path.parent.glob(path.name))
    return sorted(matches, key=lambda item: item.stat().st_mtime)[-1]


def save_mask(path: Path, mask: np.ndarray, threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.squeeze(mask)
    if mask.max(initial=0) > 1:
        mask = mask / 255.0
    mask = ((mask > threshold) * 255).astype(np.uint8)
    cv2.imwrite(str(path), mask)


def export_hardnet_mseg(args: argparse.Namespace, samples: list[Path], output_dir: Path) -> None:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision import transforms

    vendor = add_vendor("hardnet_mseg")
    from lib.HarDMSEG import HarDMSEG

    checkpoint = Path(args.checkpoint or vendor / "snapshots/hardnet_mseg_kvasir/HarD-MSEG-best.pth")
    device = torch_device(args.device)
    model = load_torch_model(HarDMSEG(), checkpoint, device)
    transform = transforms.Compose([
        transforms.Resize((352, 352)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    with torch.no_grad():
        for image_path in samples:
            image = Image.open(image_path).convert("RGB")
            shape = image.size[1], image.size[0]
            tensor = transform(image).unsqueeze(0).to(device)
            pred = model(tensor)
            pred = F.interpolate(pred, size=shape, mode="bilinear", align_corners=False)
            pred = torch.sigmoid(pred).cpu().numpy().squeeze()
            save_mask(output_dir / f"{image_path.stem}.png", pred, args.threshold)


def export_hardnet_dfus(args: argparse.Namespace, samples: list[Path], output_dir: Path) -> None:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from PIL import Image
    from torchvision import transforms

    vendor = add_vendor("hardnet_dfus")
    from utils.utils import build_model

    checkpoint = Path(args.checkpoint) if args.checkpoint else latest_checkpoint([
        vendor / "weights/hardnet_dfus_kvasir/*best*.pth",
        vendor / "weights/hardnet_dfus_kvasir/*final*.pth",
        vendor / "weights/hardnet_dfus_kvasir/*.pth",
    ])
    device = torch_device(args.device)
    model = build_model("lawinloss4", 1, 53)
    model = load_torch_model(model, checkpoint, device)
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    with torch.no_grad():
        for image_path in samples:
            image = Image.open(image_path).convert("RGB")
            shape = image.size[1], image.size[0]
            tensor = transform(image).unsqueeze(0).to(device)
            pred = model(tensor)
            if isinstance(pred, (list, tuple)):
                pred = pred[0]
            pred = F.interpolate(pred, size=shape, mode="bilinear", align_corners=False)
            pred = nn.Tanh()(pred).cpu().numpy().squeeze()
            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-16)
            save_mask(output_dir / f"{image_path.stem}.png", pred, args.threshold)


def export_cascade(args: argparse.Namespace, samples: list[Path], output_dir: Path) -> None:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision import transforms

    vendor = add_vendor("cascade")
    from lib.networks import PVT_CASCADE

    checkpoint = Path(args.checkpoint or vendor / "model_pth/PolypPVT-CASCADE-Kvasir/PolypPVT-CASCADE.pth")
    device = torch_device(args.device)
    model = load_torch_model(PVT_CASCADE(), checkpoint, device)
    transform = transforms.Compose([
        transforms.Resize((352, 352)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    with torch.no_grad():
        for image_path in samples:
            image = Image.open(image_path).convert("RGB")
            shape = image.size[1], image.size[0]
            tensor = transform(image).unsqueeze(0).to(device)
            preds = model(tensor)
            pred = preds[0] + preds[1] + preds[2] + preds[3]
            pred = F.interpolate(pred, size=shape, mode="bilinear", align_corners=False)
            pred = torch.sigmoid(pred).cpu().numpy().squeeze()
            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
            save_mask(output_dir / f"{image_path.stem}.png", pred, args.threshold)


def export_colonformer(args: argparse.Namespace, samples: list[Path], output_dir: Path) -> None:
    import torch
    import torch.nn.functional as F

    add_vendor("colonformer")
    from mmseg.models.segmentors import ColonFormer

    checkpoint = Path(args.checkpoint or VENDOR / "colonformer/snapshots/ColonFormerB3_Kvasir/last.pth")
    device = torch_device(args.device)
    model = ColonFormer(
        backbone=dict(type="mit_b3", style="pytorch"),
        decode_head=dict(
            type="UPerHead",
            in_channels=[64, 128, 320, 512],
            in_index=[0, 1, 2, 3],
            channels=128,
            dropout_ratio=0.1,
            num_classes=1,
            norm_cfg=dict(type="BN", requires_grad=True),
            align_corners=False,
            decoder_params=dict(embed_dim=768),
            loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
        ),
        neck=None,
        auxiliary_head=None,
        train_cfg=dict(),
        test_cfg=dict(mode="whole"),
        pretrained=None,
    )
    model = load_torch_model(model, checkpoint, device)

    with torch.no_grad():
        for image_path in samples:
            image = cv2.imread(str(image_path))
            shape = image.shape[:2]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (352, 352)).astype("float32") / 255.0
            tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(device)
            pred = model(tensor)[0]
            pred = F.interpolate(pred, size=shape, mode="bilinear", align_corners=False)
            pred = torch.sigmoid(pred).cpu().numpy().squeeze()
            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
            save_mask(output_dir / f"{image_path.stem}.png", pred, args.threshold)


def export_tganet(args: argparse.Namespace, samples: list[Path], output_dir: Path) -> None:
    import torch

    vendor = add_vendor("tganet")
    from model import TGAPolypSeg
    from text2embed import Text2Embed

    checkpoint = Path(args.checkpoint or vendor / "files/checkpoint.pth")
    device = torch_device(args.device)
    model = load_torch_model(TGAPolypSeg(), checkpoint, device)
    embed = Text2Embed()
    words = ["one", "multiple", "small", "medium", "large"]
    label = np.array([embed.to_embed(word)[0] for word in words])
    label = torch.from_numpy(label).unsqueeze(0).to(device, dtype=torch.float32)

    with torch.no_grad():
        for image_path in samples:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            image = cv2.resize(image, (256, 256))
            image = image.transpose(2, 0, 1) / 255.0
            image = torch.from_numpy(image).unsqueeze(0).to(device, dtype=torch.float32)
            pred, _, _ = model(image, label)
            pred = torch.sigmoid(pred).cpu().numpy().squeeze()
            save_mask(output_dir / f"{image_path.stem}.png", pred, args.threshold)


def export_meta_polyp(args: argparse.Namespace, samples: list[Path], output_dir: Path) -> None:
    vendor = add_vendor("meta_polyp")
    from model import build_model

    checkpoint = Path(args.checkpoint or vendor / "best_model.h5")
    model = build_model(256)
    model.load_weights(str(checkpoint))

    for image_path in samples:
        image = cv2.imread(str(image_path))
        image = cv2.resize(image, (256, 256))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pred = model.predict(np.expand_dims(image, axis=0), verbose=0)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        save_mask(output_dir / f"{image_path.stem}.png", pred, args.threshold)


def export_ssformer(args: argparse.Namespace, samples: list[Path], output_dir: Path) -> None:
    vendor = add_vendor("ssformer")
    from mmseg.apis import inference_segmentor, init_segmentor

    config = vendor / "configs/SSformer/kvasir_binary.py"
    checkpoint = Path(args.checkpoint or vendor / "work_dirs/kvasir_binary/latest.pth")
    model = init_segmentor(str(config), str(checkpoint), device=args.device)

    for image_path in samples:
        pred = inference_segmentor(model, str(image_path))[0]
        save_mask(output_dir / f"{image_path.stem}.png", pred.astype(np.float32), 0.5)


def export_esfpnet(args: argparse.Namespace, samples: list[Path], output_dir: Path) -> None:
    import torch
    import torch.nn.functional as F

    from data.transforms.augmentation import get_val_transforms
    from models import get_model
    from scripts.train import load_config, set_seed

    config = load_config("configs/models/esfpnet.yaml")
    set_seed(config.get("seed", 42))

    checkpoint = Path(args.checkpoint or ROOT / "checkpoints" / config["logging"]["run_name"] / "best.pth")
    device = torch_device(args.device)
    model_config = config["model"]
    model = get_model(
        model_config["name"],
        num_classes=model_config["num_classes"],
        model_type=model_config.get("model_type", "b2"),
        pretrained_path=model_config.get("pretrained_path"),
    )
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    transform = get_val_transforms(config["data"]["image_size"])

    with torch.no_grad():
        for image_path in samples:
            image = cv2.imread(str(image_path))
            shape = image.shape[:2]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = np.zeros(shape, dtype=np.float32)
            tensor = transform(image=image, mask=mask)["image"].unsqueeze(0).to(device)
            pred = model(tensor)
            if isinstance(pred, (list, tuple)):
                pred = pred[0]
            pred = F.interpolate(pred, size=shape, mode="bilinear", align_corners=False)
            pred = torch.sigmoid(pred).cpu().numpy().squeeze()
            save_mask(output_dir / f"{image_path.stem}.png", pred, args.threshold)


EXPORTERS = {
    "hardnet_mseg": export_hardnet_mseg,
    "hardnet_dfus": export_hardnet_dfus,
    "ssformer": export_ssformer,
    "tganet": export_tganet,
    "colonformer": export_colonformer,
    "esfpnet": export_esfpnet,
    "meta_polyp": export_meta_polyp,
    "cascade": export_cascade,
}


def main() -> None:
    args = parse_args()
    samples = split_samples(ROOT / args.data_root, ROOT / args.split_file, args.split)
    output_dir = Path(args.output_dir or ROOT / "outputs" / "predictions" / args.model)
    EXPORTERS[args.model](args, samples, output_dir)
    print(f"Saved {len(samples)} predictions in: {output_dir}")


if __name__ == "__main__":
    main()
