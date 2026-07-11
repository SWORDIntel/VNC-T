#!/usr/bin/env python3
"""Train anomaly detection autoencoder for VNC screenshots.

Unsupervised model — trains on all images, learns to reconstruct "normal" screens.
High reconstruction error = anomaly/unusual screen.

Uses a convolutional autoencoder with MobileNetV2-style encoder/decoder.
Exports ONNX for inference. Anomaly score = MSE between input and reconstruction.

Usage:
  python3 scripts/train_anomaly_ae.py --data-dir dataset --epochs 30 --batch-size 256
"""
import argparse
import json
import os
import signal
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

from rich.console import Console

console = Console()


class ConvAutoencoder(nn.Module):
    """Lightweight convolutional autoencoder for anomaly detection."""

    def __init__(self, img_size=224, latent_dim=256):
        super().__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),   # 112x112
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),   # 56x56
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 28x28
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), # 14x14
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, latent_dim, 3, stride=2, padding=1),  # 7x7
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 3, stride=2, padding=1, output_padding=1),  # 14x14
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),  # 28x28
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),   # 56x56
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),    # 112x112
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),     # 224x224
            nn.Tanh(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)


class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        for cls_dir in sorted(self.root.iterdir()):
            if cls_dir.is_dir():
                for img_path in cls_dir.glob("*"):
                    self.samples.append(str(img_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0  # dummy label


def get_transforms(img_size, augment=True):
    t = [transforms.Resize((img_size, img_size))]
    if augment:
        t.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        ])
    t.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transforms.Compose(t)


def export_onnx(model, output_dir, img_size, device):
    try:
        model.eval()
        dummy = torch.randn(1, 3, img_size, img_size).to(device)
        onnx_path = Path(output_dir) / "anomaly-autoencoder.onnx"
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


def compute_threshold(model, loader, device, percentile=95):
    """Compute anomaly threshold as the Nth percentile of reconstruction errors."""
    model.eval()
    errors = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    recon = model(images)
            else:
                recon = model(images)
            # Per-sample MSE
            mse = ((images - recon) ** 2).mean(dim=[1, 2, 3]).cpu().numpy()
            errors.extend(mse.tolist())
    errors = np.array(errors)
    threshold = np.percentile(errors, percentile)
    return float(threshold), errors


def train(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[cyan]Device: {device}[/cyan]")
    if device.type == "cuda":
        console.print(f"[green]GPU: {torch.cuda.get_device_name(0)}[/green]")

    # Dataset — all images, no labels needed
    train_tf = get_transforms(args.img_size, augment=True)
    val_tf = get_transforms(args.img_size, augment=False)
    full_dataset = ImageDataset(args.data_dir, transform=train_tf)
    console.print(f"[cyan]Dataset: {len(full_dataset)} images (unsupervised)[/cyan]")

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

    # Model
    model = ConvAutoencoder(img_size=args.img_size, latent_dim=args.latent_dim).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    console.print(f"[cyan]Autoencoder params: {param_count:,}[/cyan]")

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # SIGTERM handler globals
    global _sigterm_model, _sigterm_output_dir
    _sigterm_model = model
    _sigterm_output_dir = output_dir

    best_loss = float('inf')
    best_epoch = 0
    no_improve = 0
    curves = {"train_loss": [], "val_loss": []}

    for epoch in range(args.epochs):
        model.train()
        train_loss, train_total = 0.0, 0
        for images, _ in train_loader:
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    recon = model(images)
                    loss = criterion(recon, images)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                recon = model(images)
                loss = criterion(recon, images)
                loss.backward()
                optimizer.step()
            train_loss += loss.item() * images.size(0)
            train_total += images.size(0)
        scheduler.step()

        model.eval()
        val_loss, val_total = 0.0, 0
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device, non_blocking=True)
                if scaler:
                    with torch.cuda.amp.autocast():
                        recon = model(images)
                        loss = criterion(recon, images)
                else:
                    recon = model(images)
                    loss = criterion(recon, images)
                val_loss += loss.item() * images.size(0)
                val_total += images.size(0)

        avg_train_loss = train_loss / train_total
        avg_val_loss = val_loss / val_total
        curves["train_loss"].append(avg_train_loss)
        curves["val_loss"].append(avg_val_loss)
        with open(output_dir / "anomaly_training_curves.json", "w") as f:
            json.dump(curves, f)

        console.print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.6f} val_loss={avg_val_loss:.6f}")

        is_best = avg_val_loss < best_loss
        if is_best:
            best_loss = avg_val_loss
            best_epoch = epoch + 1
            no_improve = 0
            torch.save(model.state_dict(), output_dir / "anomaly-autoencoder-best.pt")
        else:
            no_improve += 1

        if no_improve >= args.patience:
            console.print(f"[yellow]Early stopping at epoch {epoch+1}[/yellow]")
            break

    # Load best and compute anomaly threshold
    best_path = output_dir / "anomaly-autoencoder-best.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    console.print("[cyan]Computing anomaly threshold (95th percentile)...[/cyan]")
    threshold, errors = compute_threshold(model, val_loader, device, percentile=95)
    console.print(f"[green]Anomaly threshold (MSE): {threshold:.6f}[/green]")
    console.print(f"  Error stats: mean={errors.mean():.6f} std={errors.std():.6f} min={errors.min():.6f} max={errors.max():.6f}")

    # Export
    onnx_path = export_onnx(model, output_dir, args.img_size, device)

    report = {
        "model": "anomaly_autoencoder",
        "architecture": "ConvAutoencoder",
        "latent_dim": args.latent_dim,
        "img_size": args.img_size,
        "epochs_trained": epoch + 1,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "anomaly_threshold": threshold,
        "error_stats": {
            "mean": float(errors.mean()),
            "std": float(errors.std()),
            "min": float(errors.min()),
            "max": float(errors.max()),
            "p50": float(np.percentile(errors, 50)),
            "p95": float(np.percentile(errors, 95)),
            "p99": float(np.percentile(errors, 99)),
        },
        "param_count": param_count,
    }
    with open(output_dir / "anomaly_detector_report.json", "w") as f:
        json.dump(report, f, indent=2)

    console.print(f"\n[green]Anomaly autoencoder done! Best loss: {best_loss:.6f} at epoch {best_epoch}[/green]")
    console.print(f"[green]Threshold: {threshold:.6f} (MSE above this = anomaly)[/green]")


def get_args():
    p = argparse.ArgumentParser(description="Train anomaly detection autoencoder")
    p.add_argument("--data-dir", required=True, help="Dataset directory")
    p.add_argument("--output-dir", default="models/anomaly", help="Output directory")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--patience", type=int, default=10)
    return p.parse_args()


_sigterm_model = None
_sigterm_output_dir = None


def _sigterm_handler(signum, frame):
    console.print("\n[bold red]⚠ SIGTERM — preemptible VM shutdown! Saving checkpoint...[/bold red]")
    if _sigterm_model is not None:
        try:
            torch.save(_sigterm_model.state_dict(), Path(_sigterm_output_dir) / "anomaly-autoencoder-sigterm.pt")
            console.print("[green]✓ Emergency checkpoint saved[/green]")
        except Exception as e:
            console.print(f"[red]Checkpoint failed: {e}[/red]")
    sys.exit(143)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm_handler)
    train(get_args())
