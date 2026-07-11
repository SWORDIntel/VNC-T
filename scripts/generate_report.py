#!/usr/bin/env python3
"""Generate a fancy HTML training/evaluation report for the VNC classifier.

Runs on the L40s GPU after training. Produces:
- Full evaluation with per-class precision/recall/F1
- Confusion matrix heatmap
- Training curves (loss/accuracy over epochs)
- Inference throughput benchmark (GPU vs CPU)
- Sample predictions grid with images
- Model summary and artifact verification
- Dataset distribution analysis
- Exported as self-contained HTML with embedded base64 images
"""
import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
sns.set_theme(style="darkgrid")

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, top_k_accuracy_score,
)

sys.path.insert(0, str(Path(__file__).parent))
from train_vnc_tui import build_model, get_transforms


def get_args():
    p = argparse.ArgumentParser(description="Generate VNC classifier report")
    p.add_argument("--data-dir", required=True, help="Dataset directory")
    p.add_argument("--output-dir", default="models", help="Model output directory")
    p.add_argument("--report-dir", default="reports", help="Report output directory")
    p.add_argument("--batch-size", type=int, default=256, help="Eval batch size")
    p.add_argument("--img-size", type=int, default=224, help="Input image size")
    p.add_argument("--num-samples", type=int, default=36, help="Sample predictions to show")
    p.add_argument("--benchmark-iterations", type=int, default=100, help="Inference benchmark iterations")
    return p.parse_args()


def fig_to_base64(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def load_model(output_dir, num_classes, device, img_size):
    backbone = "mobilenetv2"
    labels_path = Path(output_dir) / "vnc-classifier-labels.json"
    if labels_path.exists():
        with open(labels_path) as f:
            meta = json.load(f)
        classes = meta["classes"]
    else:
        classes = sorted([d.name for d in Path(output_dir).iterdir() if d.is_dir()])

    report_path = Path(output_dir) / "training_report.json"
    if report_path.exists():
        with open(report_path) as f:
            train_report = json.load(f)
        backbone = train_report.get("backbone", "mobilenetv2")
    else:
        train_report = {}

    model = build_model(backbone, num_classes, device)

    best_path = Path(output_dir) / "vnc-classifier-best.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"Loaded best model from {best_path}")
    else:
        print(f"WARNING: No best model at {best_path}, using random weights")

    model.eval()
    return model, classes, train_report, backbone


def full_evaluation(model, data_dir, classes, device, img_size, batch_size):
    val_transforms = get_transforms(img_size, augment=False)
    full_dataset = datasets.ImageFolder(str(data_dir), transform=val_transforms)
    class_to_idx = full_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    total_size = len(full_dataset)
    val_size = int(total_size * 0.15)
    train_size = total_size - val_size

    gen = torch.Generator().manual_seed(42)
    _, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=gen)

    loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=min(40, os.cpu_count() or 4),
                        pin_memory=device.type == "cuda")

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = model(images)
            else:
                outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    per_class_precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    cls_report = classification_report(all_labels, all_preds, target_names=classes, zero_division=0)

    # Top-2 and Top-3 accuracy
    top2 = top_k_accuracy_score(all_labels, all_probs, k=2)
    top3 = top_k_accuracy_score(all_labels, all_probs, k=3)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'top2': top2,
        'top3': top3,
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'per_class_f1': per_class_f1,
        'confusion_matrix': cm,
        'classification_report': cls_report,
        'all_preds': all_preds,
        'all_labels': all_labels,
        'all_probs': all_probs,
        'val_size': val_size,
        'class_names': classes,
        'idx_to_class': idx_to_class,
        'val_dataset': val_dataset,
    }


def plot_confusion_matrix(cm, classes):
    fig, ax = plt.subplots(figsize=(8, 6))
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax,
                cbar_kws={'label': 'Normalized Count'})
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_per_class_metrics(classes, precision, recall, f1):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(classes))
    width = 0.25
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#2196F3', alpha=0.85)
    bars2 = ax.bar(x, recall, width, label='Recall', color='#4CAF50', alpha=0.85)
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#FF9800', alpha=0.85)
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=7)
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_training_curves(output_dir):
    curves_path = Path(output_dir) / "training_curves.json"
    if not curves_path.exists():
        return None
    with open(curves_path) as f:
        curves = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(curves['train_loss']) + 1)
    ax1.plot(epochs, curves['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, curves['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, curves['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, curves['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training & Validation Accuracy', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig_to_base64(fig)


def plot_dataset_distribution(data_dir, classes):
    counts = {}
    for cls in classes:
        cls_dir = Path(data_dir) / cls
        if cls_dir.is_dir():
            counts[cls] = len(list(cls_dir.glob("*")))
        else:
            counts[cls] = 0

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = sns.color_palette("viridis", len(classes))
    bars = ax.barh(list(counts.keys()), list(counts.values()), color=colors)
    ax.set_xlabel('Image Count', fontsize=12, fontweight='bold')
    ax.set_title('Dataset Class Distribution', fontsize=14, fontweight='bold')
    for bar in bars:
        w = bar.get_width()
        ax.annotate(f'{int(w)}', xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=10)
    plt.tight_layout()
    return fig_to_base64(fig), counts


def plot_sample_predictions(model, val_dataset, idx_to_class, classes, device, img_size, num_samples=36):
    indices = np.random.choice(len(val_dataset), min(num_samples, len(val_dataset)), replace=False)
    fig, axes = plt.subplots(6, 6, figsize=(16, 16))
    axes = axes.ravel()

    val_transforms = get_transforms(img_size, augment=False)
    # Access underlying dataset and apply transform
    for i, idx in enumerate(indices):
        img, label = val_dataset[idx]
        img_tensor = img.unsqueeze(0).to(device)
        with torch.no_grad():
            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    output = model(img_tensor)
            else:
                output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            pred = probs.argmax(dim=1).item()
            conf = probs[0][pred].item()

        # Denormalize for display
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_display = img * std + mean
        img_display = img_display.clamp(0, 1).permute(1, 2, 0).numpy()

        axes[i].imshow(img_display)
        true_name = idx_to_class.get(label, str(label))
        pred_name = idx_to_class.get(pred, str(pred))
        correct = pred == label
        color = 'green' if correct else 'red'
        axes[i].set_title(f'T: {true_name}\nP: {pred_name} ({conf:.1%})',
                          fontsize=8, color=color)
        axes[i].axis('off')

    plt.suptitle('Sample Predictions (Green=Correct, Red=Wrong)', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    return fig_to_base64(fig, dpi=120)


def benchmark_inference(model, device, img_size, batch_size=256, iterations=100):
    model.eval()
    dummy = torch.randn(batch_size, 3, img_size, img_size).to(device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    model(dummy)
            else:
                model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # GPU benchmark
    start = time.time()
    for _ in range(iterations):
        with torch.no_grad():
            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    model(dummy)
            else:
                model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start

    total_images = batch_size * iterations
    throughput = total_images / elapsed
    latency_ms = (elapsed / iterations) * 1000

    # CPU benchmark (smaller batch, fewer iterations)
    cpu_metrics = None
    if device.type == "cuda":
        cpu_model = model.cpu()
        cpu_batch = min(32, batch_size)
        cpu_dummy = torch.randn(cpu_batch, 3, img_size, img_size)
        for _ in range(5):
            with torch.no_grad():
                cpu_model(cpu_dummy)
        cpu_start = time.time()
        cpu_iters = min(20, iterations)
        for _ in range(cpu_iters):
            with torch.no_grad():
                cpu_model(cpu_dummy)
        cpu_elapsed = time.time() - cpu_start
        cpu_throughput = (cpu_batch * cpu_iters) / cpu_elapsed
        cpu_latency = (cpu_elapsed / cpu_iters) * 1000
        cpu_metrics = {
            'throughput': cpu_throughput,
            'latency_ms': cpu_latency,
            'batch_size': cpu_batch,
        }
        model.to(device)

    # Model size
    param_count = sum(p.numel() for p in model.parameters())
    model_size_mb = param_count * 4 / 1024 / 1024

    return {
        'device': str(device),
        'batch_size': batch_size,
        'iterations': iterations,
        'total_images': total_images,
        'elapsed_seconds': elapsed,
        'throughput_imgs_sec': throughput,
        'latency_ms_per_batch': latency_ms,
        'param_count': param_count,
        'model_size_mb': model_size_mb,
        'cpu': cpu_metrics,
    }


def verify_artifacts(output_dir):
    artifacts = {}
    checks = [
        ("PyTorch Model", "vnc-classifier-best.pt"),
        ("ONNX Model", "vnc-classifier.onnx"),
        ("OpenVINO IR (XML)", "vnc-classifier-static.xml"),
        ("OpenVINO IR (BIN)", "vnc-classifier-static.bin"),
        ("Labels JSON", "vnc-classifier-labels.json"),
        ("Training Report", "training_report.json"),
        ("Checkpoint", "vnc-checkpoint-latest.pt"),
    ]
    for name, filename in checks:
        path = Path(output_dir) / filename
        if path.exists():
            size = path.stat().st_size
            artifacts[name] = {"exists": True, "size_bytes": size, "size_human": format_size(size)}
        else:
            artifacts[name] = {"exists": False, "size_bytes": 0, "size_human": "N/A"}
    return artifacts


def format_size(bytes_val):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f} TB"


def generate_html(report_data, report_dir):
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VNC Classifier Training Report</title>
<style>
  body {{ font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif; margin: 0; padding: 0; background: #0f172a; color: #e2e8f0; }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
  h1 {{ font-size: 2em; background: linear-gradient(135deg, #3b82f6, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 4px; }}
  h2 {{ color: #60a5fa; border-bottom: 2px solid #334155; padding-bottom: 8px; margin-top: 32px; }}
  h3 {{ color: #93c5fd; margin-top: 20px; }}
  .header {{ text-align: center; padding: 32px 0; }}
  .subtitle {{ color: #94a3b8; font-size: 1.1em; }}
  .timestamp {{ color: #64748b; font-size: 0.9em; }}
  .card {{ background: #1e293b; border-radius: 12px; padding: 20px; margin: 16px 0; border: 1px solid #334155; }}
  .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin: 16px 0; }}
  .metric {{ background: #1e293b; border-radius: 10px; padding: 16px; text-align: center; border: 1px solid #334155; }}
  .metric-value {{ font-size: 2em; font-weight: bold; color: #60a5fa; }}
  .metric-label {{ color: #94a3b8; font-size: 0.85em; text-transform: uppercase; margin-top: 4px; }}
  .metric.good {{ border-color: #22c55e; }}
  .metric.good .metric-value {{ color: #4ade80; }}
  .metric.warn {{ border-color: #f59e0b; }}
  .metric.warn .metric-value {{ color: #fbbf24; }}
  img {{ max-width: 100%; border-radius: 8px; margin: 12px 0; border: 1px solid #334155; }}
  table {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
  th {{ background: #334155; padding: 10px; text-align: left; color: #93c5fd; border-radius: 4px; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid #334155; }}
  tr:hover {{ background: #1e293b; }}
  pre {{ background: #0f172a; padding: 16px; border-radius: 8px; overflow-x: auto; border: 1px solid #334155; font-size: 0.85em; }}
  .badge {{ display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 0.8em; font-weight: bold; }}
  .badge.ok {{ background: #22c55e20; color: #4ade80; border: 1px solid #22c55e; }}
  .badge.fail {{ background: #ef444420; color: #f87171; border: 1px solid #ef4444; }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  @media (max-width: 768px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>VNC Classifier Training Report</h1>
    <div class="subtitle">SWORD Intel — Automated Pipeline on NVIDIA L40s</div>
    <div class="timestamp">Generated: {report_data['timestamp']}</div>
  </div>

  <h2>Executive Summary</h2>
  <div class="metric-grid">
    <div class="metric {'good' if report_data['eval']['accuracy'] > 0.85 else 'warn' if report_data['eval']['accuracy'] > 0.70 else ''}">
      <div class="metric-value">{report_data['eval']['accuracy']*100:.1f}%</div>
      <div class="metric-label">Overall Accuracy</div>
    </div>
    <div class="metric">
      <div class="metric-value">{report_data['eval']['f1']*100:.1f}%</div>
      <div class="metric-label">F1 Score</div>
    </div>
    <div class="metric">
      <div class="metric-value">{report_data['eval']['precision']*100:.1f}%</div>
      <div class="metric-label">Precision</div>
    </div>
    <div class="metric">
      <div class="metric-value">{report_data['eval']['recall']*100:.1f}%</div>
      <div class="metric-label">Recall</div>
    </div>
    <div class="metric">
      <div class="metric-value">{report_data['eval']['top2']*100:.1f}%</div>
      <div class="metric-label">Top-2 Accuracy</div>
    </div>
    <div class="metric">
      <div class="metric-value">{report_data['eval']['top3']*100:.1f}%</div>
      <div class="metric-label">Top-3 Accuracy</div>
    </div>
  </div>

  <h2>Model Configuration</h2>
  <div class="card">
    <table>
      <tr><th>Parameter</th><th>Value</th></tr>
      <tr><td>Backbone</td><td>{report_data['backbone']}</td></tr>
      <tr><td>Image Size</td><td>{report_data['img_size']}×{report_data['img_size']}</td></tr>
      <tr><td>Num Classes</td><td>{len(report_data['eval']['class_names'])}</td></tr>
      <tr><td>Classes</td><td>{', '.join(report_data['eval']['class_names'])}</td></tr>
      <tr><td>Device</td><td>{report_data['device']}</td></tr>
      <tr><td>Validation Size</td><td>{report_data['eval']['val_size']} images</td></tr>
      <tr><td>Parameters</td><td>{report_data['benchmark']['param_count']:,}</td></tr>
      <tr><td>Model Size</td><td>{report_data['benchmark']['model_size_mb']:.1f} MB</td></tr>
    </table>
  </div>

  <h2>Dataset Distribution</h2>
  <div class="card">
    <img src="data:image/png;base64,{report_data['dataset_dist_plot']}" alt="Dataset Distribution">
    <table>
      <tr><th>Class</th><th>Count</th><th>Percentage</th></tr>
"""
    total = sum(report_data['dataset_counts'].values())
    for cls, count in report_data['dataset_counts'].items():
        pct = 100 * count / total if total else 0
        html += f"      <tr><td>{cls}</td><td>{count}</td><td>{pct:.1f}%</td></tr>\n"
    html += f"""    </table>
  </div>

  <h2>Confusion Matrix</h2>
  <div class="card">
    <img src="data:image/png;base64,{report_data['confusion_matrix_plot']}" alt="Confusion Matrix">
  </div>

  <h2>Per-Class Metrics</h2>
  <div class="card">
    <img src="data:image/png;base64,{report_data['per_class_plot']}" alt="Per-Class Metrics">
    <table>
      <tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th></tr>
"""
    for i, cls in enumerate(report_data['eval']['class_names']):
        html += f"      <tr><td>{cls}</td><td>{report_data['eval']['per_class_precision'][i]*100:.1f}%</td><td>{report_data['eval']['per_class_recall'][i]*100:.1f}%</td><td>{report_data['eval']['per_class_f1'][i]*100:.1f}%</td></tr>\n"
    html += f"""    </table>
  </div>

  <h2>Training Curves</h2>
  <div class="card">
"""
    if report_data.get('training_curves_plot'):
        html += f"    <img src=\"data:image/png;base64,{report_data['training_curves_plot']}\" alt=\"Training Curves\">\n"
    else:
        html += "    <p>No training curves data found (training_curves.json missing).</p>\n"
    html += f"""  </div>

  <h2>Sample Predictions</h2>
  <div class="card">
    <img src="data:image/png;base64,{report_data['sample_predictions_plot']}" alt="Sample Predictions">
  </div>

  <h2>Inference Benchmark</h2>
  <div class="card">
    <div class="metric-grid">
      <div class="metric">
        <div class="metric-value">{report_data['benchmark']['throughput_imgs_sec']:.0f}</div>
        <div class="metric-label">imgs/sec ({report_data['benchmark']['device']})</div>
      </div>
      <div class="metric">
        <div class="metric-value">{report_data['benchmark']['latency_ms_per_batch']:.1f}ms</div>
        <div class="metric-label">Latency/batch</div>
      </div>
"""
    if report_data['benchmark'].get('cpu'):
        html += f"""      <div class="metric">
        <div class="metric-value">{report_data['benchmark']['cpu']['throughput_imgs_sec']:.0f}</div>
        <div class="metric-label">imgs/sec (CPU)</div>
      </div>
      <div class="metric">
        <div class="metric-value">{report_data['benchmark']['cpu']['latency_ms']:.1f}ms</div>
        <div class="metric-label">CPU Latency/batch</div>
      </div>
"""
    if report_data['benchmark'].get('cpu') and report_data['benchmark']['cpu']['throughput_imgs_sec'] > 0:
        speedup = report_data['benchmark']['throughput_imgs_sec'] / report_data['benchmark']['cpu']['throughput_imgs_sec']
        html += f"""      <div class="metric good">
        <div class="metric-value">{speedup:.1f}×</div>
        <div class="metric-label">GPU Speedup</div>
      </div>
"""
    html += f"""    </div>
    <table>
      <tr><th>Metric</th><th>GPU ({report_data['benchmark']['device']})</th>"""
    if report_data['benchmark'].get('cpu'):
        html += f"<th>CPU</th>"
    html += "</tr>\n"
    html += f"      <tr><td>Batch Size</td><td>{report_data['benchmark']['batch_size']}</td>"
    if report_data['benchmark'].get('cpu'):
        html += f"<td>{report_data['benchmark']['cpu']['batch_size']}</td>"
    html += "</tr>\n"
    html += f"      <tr><td>Iterations</td><td>{report_data['benchmark']['iterations']}</td>"
    if report_data['benchmark'].get('cpu'):
        html += "<td>20</td>"
    html += "</tr>\n"
    html += f"      <tr><td>Total Images</td><td>{report_data['benchmark']['total_images']:,}</td>"
    if report_data['benchmark'].get('cpu'):
        html += f"<td>{report_data['benchmark']['cpu']['batch_size'] * 20:,}</td>"
    html += "</tr>\n"
    html += f"      <tr><td>Throughput (imgs/sec)</td><td>{report_data['benchmark']['throughput_imgs_sec']:.1f}</td>"
    if report_data['benchmark'].get('cpu'):
        html += f"<td>{report_data['benchmark']['cpu']['throughput_imgs_sec']:.1f}</td>"
    html += "</tr>\n"
    html += f"      <tr><td>Latency per batch (ms)</td><td>{report_data['benchmark']['latency_ms_per_batch']:.2f}</td>"
    if report_data['benchmark'].get('cpu'):
        html += f"<td>{report_data['benchmark']['cpu']['latency_ms']:.2f}</td>"
    html += "</tr>\n"
    html += """    </table>
  </div>

  <h2>Deployment Artifacts</h2>
  <div class="card">
    <table>
      <tr><th>Artifact</th><th>Status</th><th>Size</th></tr>
"""
    for name, info in report_data['artifacts'].items():
        badge = '<span class="badge ok">✓ Present</span>' if info['exists'] else '<span class="badge fail">✗ Missing</span>'
        html += f"      <tr><td>{name}</td><td>{badge}</td><td>{info['size_human']}</td></tr>\n"
    html += f"""    </table>
  </div>

  <h2>Classification Report (sklearn)</h2>
  <div class="card">
    <pre>{report_data['eval']['classification_report']}</pre>
  </div>

  <h2>Training Metadata</h2>
  <div class="card">
    <pre>{json.dumps(report_data.get('train_report', {}), indent=2)}</pre>
  </div>

  <div style="text-align: center; padding: 24px; color: #64748b; font-size: 0.85em;">
    Generated by VNC-T Pipeline · SWORD Intel · {report_data['timestamp']}
  </div>
</div>
</body>
</html>"""
    return html


def main():
    args = get_args()
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")

    data_dir = Path(args.data_dir)
    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    num_classes = len(classes)
    print(f"Classes ({num_classes}): {classes}")

    # Load model
    model, model_classes, train_report, backbone = load_model(args.output_dir, num_classes, device, args.img_size)

    # Full evaluation
    print("Running full evaluation...")
    eval_results = full_evaluation(model, args.data_dir, classes, device, args.img_size, args.batch_size)
    print(f"Accuracy: {eval_results['accuracy']*100:.2f}%")
    print(f"F1: {eval_results['f1']*100:.2f}%")

    # Plots
    print("Generating plots...")
    cm_plot = plot_confusion_matrix(eval_results['confusion_matrix'], classes)
    per_class_plot = plot_per_class_metrics(
        classes, eval_results['per_class_precision'],
        eval_results['per_class_recall'], eval_results['per_class_f1']
    )
    dist_plot, dataset_counts = plot_dataset_distribution(args.data_dir, classes)
    curves_plot = plot_training_curves(args.output_dir)
    sample_plot = plot_sample_predictions(
        model, eval_results['val_dataset'], eval_results['idx_to_class'],
        classes, device, args.img_size, args.num_samples
    )

    # Benchmark
    print("Running inference benchmark...")
    benchmark = benchmark_inference(model, device, args.img_size, args.batch_size, args.benchmark_iterations)
    print(f"Throughput: {benchmark['throughput_imgs_sec']:.1f} imgs/sec")
    if benchmark.get('cpu'):
        print(f"CPU Throughput: {benchmark['cpu']['throughput_imgs_sec']:.1f} imgs/sec")
        print(f"GPU Speedup: {benchmark['throughput_imgs_sec'] / benchmark['cpu']['throughput_imgs_sec']:.1f}x")

    # Artifacts
    artifacts = verify_artifacts(args.output_dir)

    # Assemble report
    report_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
        'backbone': backbone,
        'img_size': args.img_size,
        'device': str(device),
        'eval': eval_results,
        'benchmark': benchmark,
        'artifacts': artifacts,
        'dataset_counts': dataset_counts,
        'dataset_dist_plot': dist_plot,
        'confusion_matrix_plot': cm_plot,
        'per_class_plot': per_class_plot,
        'training_curves_plot': curves_plot,
        'sample_predictions_plot': sample_plot,
        'train_report': train_report,
    }

    # Generate HTML
    print("Generating HTML report...")
    html = generate_html(report_data, report_dir)

    html_path = report_dir / "vnc-training-report.html"
    with open(html_path, 'w') as f:
        f.write(html)
    print(f"Report saved: {html_path}")

    # Also save JSON metrics
    json_metrics = {
        'timestamp': report_data['timestamp'],
        'backbone': backbone,
        'img_size': args.img_size,
        'device': str(device),
        'accuracy': eval_results['accuracy'],
        'precision': eval_results['precision'],
        'recall': eval_results['recall'],
        'f1': eval_results['f1'],
        'top2': eval_results['top2'],
        'top3': eval_results['top3'],
        'per_class_precision': eval_results['per_class_precision'].tolist(),
        'per_class_recall': eval_results['per_class_recall'].tolist(),
        'per_class_f1': eval_results['per_class_f1'].tolist(),
        'confusion_matrix': eval_results['confusion_matrix'].tolist(),
        'benchmark': {k: v for k, v in benchmark.items() if k != 'cpu'},
        'cpu_benchmark': benchmark.get('cpu'),
        'artifacts': artifacts,
        'dataset_counts': dataset_counts,
        'train_report': train_report,
    }
    json_path = report_dir / "vnc-evaluation-metrics.json"
    with open(json_path, 'w') as f:
        json.dump(json_metrics, f, indent=2)
    print(f"Metrics saved: {json_path}")

    print("\nDone! Report ready.")


if __name__ == "__main__":
    main()
