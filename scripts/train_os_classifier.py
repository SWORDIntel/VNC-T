#!/usr/bin/env python3
"""Train OS/platform classifier using pseudo-labels from visual features.

Pseudo-labels images as: windows_desktop, linux_desktop, embedded_hmi,
server_console, kiosk_ui, blank_screen
based on visual heuristics (taskbar style, color scheme, UI density, aspect ratio).
Then trains MobileNetV3 classifier on these labels.

Exports PyTorch, ONNX, OpenVINO IR for NCS2 deployment.

Usage:
  python3 scripts/train_os_classifier.py --data-dir dataset --epochs 30 --batch-size 256
"""
import argparse
import json
import os
import signal
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import cv2

from rich.console import Console

console = Console()

OS_CLASSES = ["windows_desktop", "linux_desktop", "embedded_hmi", "server_console", "kiosk_ui", "blank_screen"]


def pseudo_label_image(img_path, original_class):
    """Derive OS/platform label from visual features + original class context."""
    img = cv2.imread(str(img_path))
    if img is None:
        return "blank_screen"

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Brightness
    brightness = hsv[:, :, 2].mean()
    # UI density (edge density)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = edges.sum() / (h * w * 255) * 100
    # Color variance (more colorful = desktop, monochrome = server/embedded)
    color_std = img.std(axis=(0, 1)).mean()
    # Taskbar detection: bottom 5% of image has different brightness
    bottom = img[max(0, h - h // 20):, :, :]
    main = img[:max(0, h - h // 20), :, :]
    taskbar_diff = abs(bottom.mean() - main.mean()) if h > 50 else 0

    # Blank screen
    if brightness < 15 or edge_density < 2:
        return "blank_screen"

    # Use original class as context
    if original_class == "blank_screen":
        return "blank_screen"
    elif original_class == "embedded":
        return "embedded_hmi"
    elif original_class == "hmi_scada":
        # SCADA could be embedded HMI or Windows-based HMI
        if brightness < 80 and color_std < 40:
            return "embedded_hmi"
        else:
            return "windows_desktop"
    elif original_class == "kiosk":
        return "kiosk_ui"
    elif original_class == "server":
        # Server: dark, low color, terminal-like
        if brightness < 60 and color_std < 30:
            return "server_console"
        elif taskbar_diff > 15:
            return "windows_desktop"
        else:
            return "linux_desktop"
    elif original_class == "workstation":
        # Workstation: could be Windows or Linux desktop
        if taskbar_diff > 20 and color_std > 50:
            return "windows_desktop"
        elif brightness < 80:
            return "linux_desktop"
        else:
            return "windows_desktop"
    return "blank_screen"


def build_pseudo_labeled_dataset(data_dir, output_dir):
    """Scan dataset, copy images into OS-class subdirectories using pseudo-labels."""
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        console.print(f"[yellow]OS pseudo-labeled dataset already exists at {out}, reusing[/yellow]")
        return out

    for cls in OS_CLASSES:
        (out / cls).mkdir(parents=True, exist_ok=True)

    data_path = Path(data_dir)
    counts = {c: 0 for c in OS_CLASSES}
    images = []
    for cls_dir in sorted(data_path.iterdir()):
        if not cls_dir.is_dir():
            continue
        for img_path in cls_dir.glob("*"):
            images.append((img_path, cls_dir.name))

    console.print(f"[cyan]Pseudo-labeling {len(images)} images for OS classification...[/cyan]")
    for i, (img_path, original_class) in enumerate(images):
        label = pseudo_label_image(img_path, original_class)
        dst = out / label / f"os_{i}_{img_path.name}"
        shutil.copy2(img_path, dst)
        counts[label] += 1
        if (i + 1) % 1000 == 0:
            console.print(f"  Labeled {i+1}/{len(images)}: {counts}")

    console.print(f"[green]OS pseudo-label distribution: {counts}[/green]")
    return out


class OSDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {c: i for i, c in enumerate(OS_CLASSES)}
        for cls in OS_CLASSES:
            cls_dir = self.root / cls
            if cls_dir.is_dir():
                for img_path in cls_dir.glob("*"):
                    self.samples.append((str(img_path), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(img_size, augment=True):
    if augment:
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.RandomRotation(degrees=2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def export_onnx(model, output_dir, img_size, device):
    try:
        model.eval()
        dummy = torch.randn(1, 3, img_size, img_size).to(device)
        onnx_path = Path(output_dir) / "os-classifier.onnx"
        try:
            torch.onnx.export(model, dummy, str(onnx_path), export_params=True,
                              opset_version=18, input_names=["input"], output_names=["output"],
                              dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}, dynamo=False)
        except (TypeError, AttributeError):
            torch.onnx.export(model, dummy, str(onnx_path), export_params=True,
                              opset_version=18, input_names=["input"], output_names=["output"],
                              dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
        console.print(f"[green]✓ ONNX saved: {onnx_path}[/green]")
        return str(onnx_path)
    except Exception as e:
        console.print(f"[red]✗ ONNX export failed: {e}[/red]")
        return None


def export_openvino(onnx_path, output_dir, img_size):
    try:
        from openvino.runtime import Core
        core = Core()
        ov_model = core.read_model(str(onnx_path))
        ov_model.reshape({0: [1, 3, img_size, img_size]})
        ir_xml = Path(output_dir) / "os-classifier-static.xml"
        core.save_model(ov_model, str(ir_xml))
        console.print(f"[green]✓ OpenVINO IR saved: {ir_xml}[/green]")
        return str(ir_xml)
    except Exception as e:
        console.print(f"[yellow]⚠ OpenVINO export skipped: {e}[/yellow]")
    return None


def train(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[cyan]Device: {device}[/cyan]")
    if device.type == "cuda":
        console.print(f"[green]GPU: {torch.cuda.get_device_name(0)}[/green]")

    # Build pseudo-labeled dataset
    pseudo_dir = output_dir / "os_dataset"
    build_pseudo_labeled_dataset(args.data_dir, pseudo_dir)

    # Dataset
    train_tf = get_transforms(args.img_size, augment=True)
    val_tf = get_transforms(args.img_size, augment=False)
    full_dataset = OSDataset(str(pseudo_dir), transform=train_tf)
    num_classes = len(OS_CLASSES)
    console.print(f"[cyan]Dataset: {len(full_dataset)} images, {num_classes} classes: {OS_CLASSES}[/cyan]")

    total = len(full_dataset)
    val_size = int(total * 0.15)
    train_size = total - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(42))
    val_ds.dataset.transform = val_tf

    batch_size = args.batch_size
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=min(40, os.cpu_count() or 4), pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=min(40, os.cpu_count() or 4), pin_memory=True, persistent_workers=True)

    # Model — MobileNetV3 for OS classification
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    model = model.to(device)

    # Class weights
    counts = [0] * num_classes
    for _, label in full_dataset:
        counts[label] += 1
    weights = torch.tensor([total / (num_classes * max(1, c)) for c in counts], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # SIGTERM handler globals
    global _sigterm_model, _sigterm_output_dir
    _sigterm_model = model
    _sigterm_output_dir = output_dir

    with open(output_dir / "os-classifier-labels.json", "w") as f:
        json.dump({"classes": OS_CLASSES, "class_to_idx": {c: i for i, c in enumerate(OS_CLASSES)}}, f, indent=2)

    best_acc = 0.0
    best_epoch = 0
    no_improve = 0
    curves = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(args.epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            train_loss += loss.item() * images.size(0)
            train_correct += outputs.argmax(1).eq(labels).sum().item()
            train_total += labels.size(0)
        scheduler.step()

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                if scaler:
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_correct += outputs.argmax(1).eq(labels).sum().item()
                val_total += labels.size(0)

        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        avg_train_loss = train_loss / train_total
        avg_val_loss = val_loss / val_total

        curves["train_loss"].append(avg_train_loss)
        curves["val_loss"].append(avg_val_loss)
        curves["train_acc"].append(train_acc)
        curves["val_acc"].append(val_acc)
        with open(output_dir / "os_training_curves.json", "w") as f:
            json.dump(curves, f)

        console.print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f} train_acc={train_acc:.2f}% val_loss={avg_val_loss:.4f} val_acc={val_acc:.2f}%")

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            best_epoch = epoch + 1
            no_improve = 0
            torch.save(model.state_dict(), output_dir / "os-classifier-best.pt")
        else:
            no_improve += 1

        if no_improve >= args.patience:
            console.print(f"[yellow]Early stopping at epoch {epoch+1}[/yellow]")
            break

    # Export
    best_path = output_dir / "os-classifier-best.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    onnx_path = export_onnx(model, output_dir, args.img_size, device)
    if onnx_path:
        export_openvino(onnx_path, output_dir, args.img_size)

    report = {
        "model": "os_classifier",
        "backbone": "mobilenetv3_large",
        "classes": OS_CLASSES,
        "epochs_trained": epoch + 1,
        "best_epoch": best_epoch,
        "best_val_acc": best_acc,
        "pseudo_label_counts": counts,
    }
    with open(output_dir / "os_classifier_report.json", "w") as f:
        json.dump(report, f, indent=2)

    console.print(f"\n[green]OS classifier done! Best acc: {best_acc:.2f}% at epoch {best_epoch}[/green]")


def get_args():
    p = argparse.ArgumentParser(description="Train OS/platform classifier")
    p.add_argument("--data-dir", required=True, help="Source dataset directory")
    p.add_argument("--output-dir", default="models/os", help="Output directory")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--patience", type=int, default=10)
    return p.parse_args()


_sigterm_model = None
_sigterm_output_dir = None


def _sigterm_handler(signum, frame):
    console.print("\n[bold red]⚠ SIGTERM — preemptible VM shutdown! Saving checkpoint...[/bold red]")
    if _sigterm_model is not None:
        try:
            torch.save(_sigterm_model.state_dict(), Path(_sigterm_output_dir) / "os-classifier-sigterm.pt")
            console.print("[green]✓ Emergency checkpoint saved[/green]")
        except Exception as e:
            console.print(f"[red]Checkpoint failed: {e}[/red]")
    sys.exit(143)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm_handler)
    train(get_args())
