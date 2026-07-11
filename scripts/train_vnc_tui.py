#!/usr/bin/env python3
"""Train VNC screenshot classifier with Rich TUI, fallbacks, and auto-export.

Runs on NVIDIA L40s (CUDA 13) with auto-fallback to CPU.
Exports PyTorch, ONNX, and OpenVINO IR for NCS2 deployment.

Usage:
  python3 scripts/train_vnc_tui.py --data-dir dataset --epochs 50 --batch-size 256
"""
import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms, datasets, models

# Rich TUI
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich.layout import Layout


console = Console()


def get_args():
    parser = argparse.ArgumentParser(description="Train VNC classifier with TUI")
    parser.add_argument("--data-dir", default="dataset", help="Training dataset directory")
    parser.add_argument("--output-dir", default="models", help="Output directory for models")
    parser.add_argument("--backbone", default="mobilenetv2", choices=["mobilenetv2", "mobilenetv3", "efficientnet_b0"], help="Model backbone")
    parser.add_argument("--epochs", type=int, default=50, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--img-size", type=int, default=224, help="Input image size")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation split")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--checkpoint-every", type=int, default=5, help="Checkpoint interval (epochs)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if present")
    parser.add_argument("--no-tui", action="store_true", help="Disable TUI, use plain logs")
    return parser.parse_args()


def setup_device():
    """Detect GPU and return torch device, with fallback messaging."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        return device
    console.print("[yellow]⚠ CUDA not available, falling back to CPU[/yellow]")
    return torch.device("cpu")


def build_model(backbone, num_classes, device):
    """Build model with backbone fallback chain."""
    attempts = [
        ("mobilenetv2", _mobilenetv2),
        ("mobilenetv3", _mobilenetv3),
        ("efficientnet_b0", _efficientnet_b0),
    ]
    if backbone == "mobilenetv3":
        attempts = [("mobilenetv3", _mobilenetv3), ("mobilenetv2", _mobilenetv2), ("efficientnet_b0", _efficientnet_b0)]
    if backbone == "efficientnet_b0":
        attempts = [("efficientnet_b0", _efficientnet_b0), ("mobilenetv3", _mobilenetv3), ("mobilenetv2", _mobilenetv2)]

    for name, builder in attempts:
        try:
            model = builder(num_classes)
            model = model.to(device)
            console.print(f"[green]✓ Loaded backbone: {name}[/green]")
            return model
        except Exception as e:
            console.print(f"[yellow]⚠ Backbone {name} failed: {e}, trying next...[/yellow]")
    raise RuntimeError("All backbones failed to load")


def _mobilenetv2(num_classes):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def _mobilenetv3(num_classes):
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model


def _efficientnet_b0(num_classes):
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def get_transforms(img_size, augment=True):
    if augment:
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.RandomRotation(degrees=3),
            transforms.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.98, 1.02)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def find_checkpoint(output_dir):
    """Find latest checkpoint for resume."""
    cand = Path(output_dir) / "vnc-checkpoint-latest.pt"
    if cand.exists():
        return str(cand)
    return None


def save_checkpoint(output_dir, model, optimizer, scheduler, epoch, best_acc, state, batch_size):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / "vnc-checkpoint-latest.pt"
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_acc": best_acc,
        "state": state,
        "batch_size": batch_size,
    }, path)


def try_load_checkpoint(output_dir, model, optimizer, scheduler, device):
    """Attempt to resume training from checkpoint."""
    ckpt = find_checkpoint(output_dir)
    if not ckpt:
        return None
    try:
        data = torch.load(ckpt, map_location=device)
        model.load_state_dict(data["model_state"])
        optimizer.load_state_dict(data["optimizer_state"])
        scheduler.load_state_dict(data["scheduler_state"])
        console.print(f"[green]✓ Resumed from checkpoint epoch {data['epoch']}[/green]")
        return data
    except Exception as e:
        console.print(f"[yellow]⚠ Failed to load checkpoint: {e}[/yellow]")
        return None


def export_onnx(model, output_dir, img_size, device):
    """Export to ONNX with CPU fallback and PyTorch 2.5+ compatibility."""
    try:
        model.eval()
        dummy = torch.randn(1, 3, img_size, img_size).to(device)
        onnx_path = Path(output_dir) / "vnc-classifier.onnx"

        # Try torch.onnx.export with dynamo=False (old exporter, more stable)
        try:
            torch.onnx.export(
                model, dummy, str(onnx_path),
                export_params=True, opset_version=18,
                input_names=["input"], output_names=["output"],
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                dynamo=False,
            )
        except (TypeError, AttributeError):
            # Older PyTorch without dynamo kwarg
            torch.onnx.export(
                model, dummy, str(onnx_path),
                export_params=True, opset_version=18,
                input_names=["input"], output_names=["output"],
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            )
        console.print(f"[green]✓ ONNX saved: {onnx_path}[/green]")
        return str(onnx_path)
    except Exception as e:
        console.print(f"[red]✗ ONNX export failed: {e}[/red]")
        return None


def export_openvino(onnx_path, output_dir, img_size):
    """Export OpenVINO IR, fallback to CPU."""
    try:
        from openvino.runtime import Core
        core = Core()
        ov_model = core.read_model(str(onnx_path))
        ov_model.reshape({0: [1, 3, img_size, img_size]})
        ir_xml = Path(output_dir) / "vnc-classifier-static.xml"
        ir_bin = Path(output_dir) / "vnc-classifier-static.bin"
        core.save_model(ov_model, str(ir_xml))
        console.print(f"[green]✓ OpenVINO IR saved: {ir_xml}[/green]")
        try:
            compiled = core.compile_model(ov_model, "GPU")
            console.print("[green]✓ OpenVINO compiled on GPU[/green]")
        except Exception:
            console.print("[yellow]⚠ OpenVINO GPU compile skipped, IR ready for NCS2[/yellow]")
        return str(ir_xml)
    except ImportError:
        console.print("[yellow]⚠ OpenVINO not installed, skipping IR export[/yellow]")
    except Exception as e:
        console.print(f"[red]✗ OpenVINO export failed: {e}[/red]")
    return None


def make_layout(progress, stats, per_class, log_panel):
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=12),
    )
    layout["body"].split_row(
        Layout(name="left", ratio=2),
        Layout(name="right", ratio=1),
    )
    layout["header"].update(Panel("[bold cyan]VNC Classifier Training[/bold cyan]", style="cyan"))
    layout["left"].update(Panel(progress, title="Training Progress"))
    layout["right"].update(Panel(stats, title="Stats"))
    layout["footer"].split_row(
        Layout(name="per_class"),
        Layout(name="logs"),
    )
    layout["footer"]["per_class"].update(Panel(per_class, title="Per-Class Accuracy"))
    layout["footer"]["logs"].update(Panel(log_panel, title="Recent Logs"))
    return layout


def build_stats_table(epoch, epochs, train_loss, val_loss, train_acc, val_acc, best_acc, gpu_mem, lr, batch_size):
    table = Table(show_header=False, box=None)
    table.add_row("Epoch", f"{epoch}/{epochs}")
    table.add_row("Batch Size", str(batch_size))
    table.add_row("LR", f"{lr:.2e}")
    table.add_row("Train Loss", f"{train_loss:.4f}")
    table.add_row("Val Loss", f"{val_loss:.4f}")
    table.add_row("Train Acc", f"{train_acc:.2f}%")
    table.add_row("Val Acc", f"{val_acc:.2f}%")
    table.add_row("Best Val Acc", f"{best_acc:.2f}%")
    table.add_row("GPU Memory", f"{gpu_mem:.1f} MB" if gpu_mem else "N/A")
    return table


def build_per_class_table(classes, per_class_correct, per_class_total):
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Class")
    table.add_column("Acc")
    table.add_column("Count")
    for i, cls in enumerate(classes):
        total = per_class_total[i]
        correct = per_class_correct[i]
        acc = 100.0 * correct / total if total else 0.0
        color = "green" if acc > 85 else "yellow" if acc > 60 else "red"
        table.add_row(cls, f"[{color}]{acc:.1f}%[/{color}]", f"{correct}/{total}")
    return table


def train(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = setup_device()
    if device.type == "cuda":
        console.print(f"[green]✓ GPU: {torch.cuda.get_device_name(0)}[/green]")
        console.print(f"[green]✓ CUDA: {torch.version.cuda}[/green]")

    # Classes
    data_path = Path(args.data_dir)
    classes = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    num_classes = len(classes)
    console.print(f"[cyan]Classes ({num_classes}): {classes}[/cyan]")

    # Counts
    class_counts = {}
    for cls in classes:
        class_counts[cls] = len(list((data_path / cls).glob("*")))
        console.print(f"  {cls}: {class_counts[cls]} images")

    # Transforms
    train_transforms = get_transforms(args.img_size, augment=True)
    val_transforms = get_transforms(args.img_size, augment=False)

    full_dataset = datasets.ImageFolder(str(data_path), transform=train_transforms)
    class_to_idx = full_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Save labels
    labels_path = output_dir / "vnc-classifier-labels.json"
    with open(labels_path, "w") as f:
        json.dump({"classes": classes, "class_to_idx": class_to_idx}, f, indent=2)

    total_size = len(full_dataset)
    val_size = int(total_size * args.val_split)
    train_size = total_size - val_size
    console.print(f"[cyan]Train: {train_size} | Val: {val_size}[/cyan]")

    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    val_dataset.dataset.transform = val_transforms

    # Class weights for loss
    weights = torch.tensor([total_size / (num_classes * max(1, class_counts[idx_to_class[i]])) for i in range(num_classes)],
                           dtype=torch.float32).to(device)

    # Batch size with OOM retry
    batch_size = args.batch_size
    train_loader = None
    val_loader = None
    while batch_size >= 4:
        try:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                      num_workers=min(40, os.cpu_count() or 4),
                                      pin_memory=True, persistent_workers=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                    num_workers=min(40, os.cpu_count() or 4),
                                    pin_memory=True, persistent_workers=True)
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                batch_size = batch_size // 2
                console.print(f"[yellow]⚠ OOM with batch_size, trying {batch_size}[/yellow]")
            else:
                raise

    if not train_loader:
        raise RuntimeError("Could not create DataLoader even with batch_size=4")

    # Model
    model = build_model(args.backbone, num_classes, device)

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # Resume
    start_epoch = 0
    best_val_acc = 0.0
    best_epoch = 0
    no_improve = 0
    state = "training"
    if args.resume:
        ckpt = try_load_checkpoint(output_dir, model, optimizer, scheduler, device)
        if ckpt:
            start_epoch = ckpt["epoch"]
            best_val_acc = ckpt["best_acc"]
            batch_size = ckpt.get("batch_size", batch_size)

    # Scaler
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # TUI setup
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    epoch_task = progress.add_task("[cyan]Epoch", total=args.epochs)
    batch_task = progress.add_task("[magenta]Batch", total=len(train_loader))

    stats = Table(show_header=False, box=None)
    per_class = Table(title="Per-Class")
    log_panel = Table(show_header=False, box=None)
    log_lines = []

    def log(msg):
        ts = time.strftime("%H:%M:%S")
        log_lines.append(f"[{ts}] {msg}")
        if len(log_lines) > 8:
            log_lines.pop(0)

    if not args.no_tui:
        layout = make_layout(progress, stats, per_class, log_panel)
        live = Live(layout, refresh_per_second=4, console=console)
    else:
        live = None

    context = live.__enter__() if live else None

    try:
        for epoch in range(start_epoch, args.epochs):
            state = "training"
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            progress.reset(batch_task, total=len(train_loader))
            log(f"Epoch {epoch+1}/{args.epochs} started")

            for batch_idx, (images, labels) in enumerate(train_loader):
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
                _, predicted = outputs.max(1)
                train_correct += predicted.eq(labels).sum().item()
                train_total += labels.size(0)

                progress.update(batch_task, advance=1)

                if not args.no_tui and context:
                    context["left"].update(Panel(progress, title="Training Progress"))

            scheduler.step()

            # Validation
            state = "validating"
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            per_class_correct = [0] * num_classes
            per_class_total = [0] * num_classes

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
                    _, predicted = outputs.max(1)
                    val_correct += predicted.eq(labels).sum().item()
                    val_total += labels.size(0)
                    for i in range(labels.size(0)):
                        per_class_total[labels[i].item()] += 1
                        if predicted[i].item() == labels[i].item():
                            per_class_correct[labels[i].item()] += 1

            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            avg_train_loss = train_loss / train_total
            avg_val_loss = val_loss / val_total

            gpu_mem = 0
            if device.type == "cuda":
                gpu_mem = torch.cuda.max_memory_allocated(device) / 1024 / 1024
                torch.cuda.reset_peak_memory_stats(device)

            # Save best
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                no_improve = 0
                torch.save(model.state_dict(), output_dir / "vnc-classifier-best.pt")
                log(f"New best val acc: {val_acc:.2f}% (epoch {best_epoch})")
            else:
                no_improve += 1
                log(f"No improvement: {no_improve}/{args.patience} (best {best_val_acc:.2f}%)")

            # Checkpoint
            if (epoch + 1) % args.checkpoint_every == 0 or is_best:
                save_checkpoint(output_dir, model, optimizer, scheduler, epoch + 1, best_val_acc, state, batch_size)
                log(f"Checkpoint saved at epoch {epoch+1}")

            # Update TUI
            lr = optimizer.param_groups[0]["lr"]
            stats = build_stats_table(epoch + 1, args.epochs, avg_train_loss, avg_val_loss,
                                      train_acc, val_acc, best_val_acc, gpu_mem, lr, batch_size)
            per_class = build_per_class_table(classes, per_class_correct, per_class_total)
            log_panel = Table(show_header=False, box=None)
            for line in log_lines:
                log_panel.add_row(line)

            progress.update(epoch_task, advance=1)

            if not args.no_tui and context:
                context["left"].update(Panel(progress, title="Training Progress"))
                context["right"].update(Panel(stats, title="Stats"))
                context["footer"]["per_class"].update(Panel(per_class, title="Per-Class Accuracy"))
                context["footer"]["logs"].update(Panel(log_panel, title="Recent Logs"))

            console.print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f} train_acc={train_acc:.2f}% val_loss={avg_val_loss:.4f} val_acc={val_acc:.2f}% best={best_val_acc:.2f}%")

            # Early stopping
            if no_improve >= args.patience:
                log(f"Early stopping at epoch {epoch+1}")
                break

        state = "exporting"
        log("Training complete, exporting models")

        # Load best and export
        best_path = output_dir / "vnc-classifier-best.pt"
        if best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location=device))
        model.eval()

        onnx_path = export_onnx(model, output_dir, args.img_size, device)
        if onnx_path:
            export_openvino(onnx_path, output_dir, args.img_size)

        # Report
        report = {
            "backbone": args.backbone,
            "img_size": args.img_size,
            "num_classes": num_classes,
            "classes": classes,
            "class_to_idx": class_to_idx,
            "epochs_trained": epoch + 1,
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "batch_size": batch_size,
            "lr": args.lr,
            "val_split": args.val_split,
            "device": str(device),
        }
        with open(output_dir / "training_report.json", "w") as f:
            json.dump(report, f, indent=2)

        log("All done!")
        if live:
            live.__exit__(None, None, None)

        console.print(f"\n[green]Best val acc: {best_val_acc:.2f}% at epoch {best_epoch}[/green]")
        console.print(f"[green]Models saved in: {output_dir}[/green]")

    except Exception as e:
        if live:
            live.__exit__(None, None, None)
        console.print(f"\n[red]Training crashed: {e}[/red]")
        traceback.print_exc()
        # Save emergency checkpoint
        try:
            save_checkpoint(output_dir, model, optimizer, scheduler, 0, best_val_acc, "crashed", batch_size)
            console.print("[yellow]Emergency checkpoint saved[/yellow]")
        except Exception:
            pass
        raise


if __name__ == "__main__":
    args = get_args()
    train(args)
