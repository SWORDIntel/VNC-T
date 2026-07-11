#!/usr/bin/env python3
"""Train CVE vulnerability type classifier using NVD descriptions.

Text CNN that classifies CVE descriptions into vulnerability categories:
  rce, lpe, dos, auth_bypass, sqli, xss, path_traversal, info_disclosure, other

Replaces keyword-based RCE detection in kev_correlator.py with a trained model.
Downloads NVD CVE data on the instance, builds vocabulary, trains a character
+ word-level CNN, exports to ONNX for fast inference.

Usage:
  python3 scripts/train_cve_classifier.py --output-dir models/cve --epochs 15
"""
import argparse
import json
import os
import re
import signal
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from rich.console import Console

console = Console()

VULN_CLASSES = [
    "rce",               # Remote Code Execution
    "lpe",               # Local Privilege Escalation
    "dos",               # Denial of Service
    "auth_bypass",       # Authentication Bypass / Missing Auth
    "sqli",              # SQL Injection
    "xss",               # Cross-Site Scripting
    "path_traversal",    # Path Traversal / Directory Traversal
    "info_disclosure",   # Information Disclosure
    "other",             # Everything else
]
CLASS_TO_IDX = {c: i for i, c in enumerate(VULN_CLASSES)}

# CWE → vulnerability type mapping
CWE_TO_TYPE = {
    "CWE-94": "rce", "CWE-77": "rce", "CWE-78": "rce", "CWE-502": "rce",
    "CWE-119": "rce", "CWE-787": "rce", "CWE-125": "rce", "CWE-416": "rce",
    "CWE-190": "rce", "CWE-20": "rce",
    "CWE-269": "lpe", "CWE-862": "lpe", "CWE-863": "lpe", "CWE-285": "lpe",
    "CWE-287": "auth_bypass", "CWE-306": "auth_bypass",
    "CWE-89": "sqli",
    "CWE-79": "xss",
    "CWE-22": "path_traversal", "CWE-23": "path_traversal",
    "CWE-200": "info_disclosure", "CWE-441": "info_disclosure",
    "CWE-400": "dos", "CWE-770": "dos", "CWE-404": "dos",
}

# Keyword fallback for labeling
KEYWORD_MAP = {
    "rce": ["remote code execution", "arbitrary code execution", "code injection",
            "command injection", "arbitrary command", "rce", "unauthenticated rce",
            "buffer overflow", "heap overflow", "stack overflow", "use after free",
            "deserialization", "memory corruption"],
    "lpe": ["privilege escalation", "elevation of privilege", "local privilege",
            "gain privileges", "improper privilege"],
    "dos": ["denial of service", "dos", "crash", "hang", "resource exhaustion",
            "infinite loop", "amplification"],
    "auth_bypass": ["authentication bypass", "missing authentication", "bypass authentication",
                    "unauthenticated", "broken authentication", "auth bypass"],
    "sqli": ["sql injection", "sqli", "sql query"],
    "xss": ["cross-site scripting", "xss", "stored xss", "reflected xss"],
    "path_traversal": ["path traversal", "directory traversal", "path traversal",
                       "arbitrary file read", "file inclusion", "lfi", "rfi"],
    "info_disclosure": ["information disclosure", "info disclosure", "sensitive information",
                        "cleartext", "plaintext", "leak", "exposure of"],
}

MAX_VOCAB = 10000
MAX_LEN = 200  # max tokens per description


def download_nvd(year, out_dir):
    """Download NVD CVE feed for a given year."""
    url = f"https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-{year}.json"
    out_path = Path(out_dir) / f"nvdcve-1.1-{year}.json"
    if out_path.exists() and out_path.stat().st_size > 1000:
        return out_path
    console.print(f"[cyan]Downloading NVD feed for {year}...[/cyan]")
    try:
        urllib.request.urlretrieve(url, str(out_path))
        return out_path
    except Exception as e:
        console.print(f"[yellow]Failed to download NVD {year}: {e}[/yellow]")
        return None


def load_nvd_data(data_dir):
    """Load NVD CVE data and extract (description, label) pairs."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    current_year = time.localtime().tm_year

    for year in range(current_year - 4, current_year + 1):
        path = download_nvd(year, data_dir)
        if path is None:
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            console.print(f"[yellow]Failed to parse {path}: {e}[/yellow]")
            continue

        for item in data.get("CVE_Items", []):
            cve_id = item.get("cve", {}).get("CVE_data_meta", {}).get("ID", "")
            descs = item.get("cve", {}).get("description", {}).get("description_data", [])
            desc_text = ""
            for d in descs:
                if d.get("lang") == "en":
                    desc_text = d.get("value", "")
                    break
            if not desc_text or len(desc_text) < 20:
                continue

            # Extract CWEs
            cwes = []
            for problem in item.get("cve", {}).get("problemtype", {}).get("problemtype_data", []):
                for desc in problem.get("description", []):
                    cwe = desc.get("value", "")
                    if cwe.startswith("CWE-"):
                        cwes.append(cwe)

            # Determine label
            label = "other"
            # Try CWE mapping first
            for cwe in cwes:
                if cwe in CWE_TO_TYPE:
                    label = CWE_TO_TYPE[cwe]
                    break
            # Fall back to keyword matching
            if label == "other":
                desc_lower = desc_text.lower()
                for vuln_type, keywords in KEYWORD_MAP.items():
                    if any(kw in desc_lower for kw in keywords):
                        label = vuln_type
                        break

            samples.append({"cve_id": cve_id, "description": desc_text, "label": label, "cwes": cwes})

    return samples


def tokenize(text):
    """Simple word-level tokenizer."""
    text = text.lower()
    tokens = re.findall(r'[a-z0-9]+', text)
    return tokens


def build_vocab(samples, max_vocab=MAX_VOCAB):
    """Build vocabulary from tokenized descriptions."""
    counter = Counter()
    for s in samples:
        counter.update(tokenize(s["description"]))
    # Reserve: 0=pad, 1=unk
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, count in counter.most_common(max_vocab - 2):
        vocab[word] = len(vocab)
    return vocab


def encode(text, vocab, max_len=MAX_LEN):
    """Encode text to fixed-length token IDs."""
    tokens = tokenize(text)[:max_len]
    ids = [vocab.get(t, 1) for t in tokens]
    # Pad
    ids += [0] * (max_len - len(ids))
    return ids


class CVEDataset(Dataset):
    def __init__(self, samples, vocab, max_len=MAX_LEN):
        self.samples = samples
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        ids = encode(s["description"], self.vocab, self.max_len)
        label = CLASS_TO_IDX.get(s["label"], CLASS_TO_IDX["other"])
        return torch.tensor(ids, dtype=torch.long), label


class TextCNN(nn.Module):
    """1D Convolutional text classifier for CVE descriptions."""

    def __init__(self, vocab_size, embed_dim=128, num_classes=9, num_filters=128,
                 kernel_sizes=(3, 4, 5), dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        embedded = embedded.transpose(1, 2)  # (batch, embed_dim, seq_len)
        conv_outs = []
        for conv in self.convs:
            c = torch.relu(conv(embedded))  # (batch, num_filters, seq_len - k + 1)
            c = torch.max(c, dim=2)[0]  # (batch, num_filters)
            conv_outs.append(c)
        out = torch.cat(conv_outs, dim=1)  # (batch, num_filters * len(kernel_sizes))
        out = self.dropout(out)
        return self.fc(out)


def export_onnx(model, vocab_size, output_dir, device):
    try:
        model.eval()
        dummy = torch.randint(0, vocab_size, (1, MAX_LEN)).to(device)
        onnx_path = Path(output_dir) / "cve-classifier.onnx"
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


# SIGTERM handler
_sigterm_model = None
_sigterm_output_dir = None


def _sigterm_handler(signum, frame):
    console.print("\n[bold red]⚠ SIGTERM — preemptible VM shutdown! Saving checkpoint...[/bold red]")
    if _sigterm_model is not None:
        try:
            torch.save(_sigterm_model.state_dict(), Path(_sigterm_output_dir) / "cve-classifier-sigterm.pt")
            console.print("[green]✓ Emergency checkpoint saved[/green]")
        except Exception as e:
            console.print(f"[red]Checkpoint failed: {e}[/red]")
    sys.exit(143)


def train(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[cyan]Device: {device}[/cyan]")
    if device.type == "cuda":
        console.print(f"[green]GPU: {torch.cuda.get_device_name(0)}[/green]")

    # Load NVD data
    nvd_dir = output_dir / "nvd_data"
    console.print("[cyan]Loading NVD CVE data...[/cyan]")
    samples = load_nvd_data(nvd_dir)
    console.print(f"[green]Loaded {len(samples)} CVE samples[/green]")

    if len(samples) < 100:
        console.print("[red]Not enough NVD data — need at least 100 samples[/red]")
        return

    # Label distribution
    label_counts = Counter(s["label"] for s in samples)
    console.print(f"[cyan]Label distribution: {dict(label_counts)}[/cyan]")

    # Build vocab
    vocab = build_vocab(samples, MAX_VOCAB)
    console.print(f"[cyan]Vocabulary size: {len(vocab)}[/cyan]")

    # Save vocab and labels
    with open(output_dir / "cve-vocab.json", "w") as f:
        json.dump(vocab, f)
    with open(output_dir / "cve-classifier-labels.json", "w") as f:
        json.dump({"classes": VULN_CLASSES, "class_to_idx": CLASS_TO_IDX}, f, indent=2)

    # Dataset
    full_dataset = CVEDataset(samples, vocab)
    total = len(full_dataset)
    val_size = int(total * 0.15)
    train_size = total - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(42))

    batch_size = args.batch_size
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=min(20, os.cpu_count() or 4), pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=min(20, os.cpu_count() or 4), pin_memory=True)

    # Model
    model = TextCNN(vocab_size=len(vocab), num_classes=len(VULN_CLASSES)).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    console.print(f"[cyan]TextCNN params: {param_count:,}[/cyan]")

    global _sigterm_model, _sigterm_output_dir
    _sigterm_model = model
    _sigterm_output_dir = output_dir

    # Class weights for imbalanced labels
    class_counts = [0] * len(VULN_CLASSES)
    for s in samples:
        class_counts[CLASS_TO_IDX[s["label"]]] += 1
    weights = torch.tensor([total / (len(VULN_CLASSES) * max(1, c)) for c in class_counts],
                           dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    best_acc = 0.0
    best_epoch = 0
    no_improve = 0
    curves = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(args.epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for texts, labels in train_loader:
            texts, labels = texts.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(texts)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(texts)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            train_loss += loss.item() * texts.size(0)
            train_correct += outputs.argmax(1).eq(labels).sum().item()
            train_total += labels.size(0)
        scheduler.step()

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for texts, labels in val_loader:
                texts, labels = texts.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                if scaler:
                    with torch.cuda.amp.autocast():
                        outputs = model(texts)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(texts)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * texts.size(0)
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
        with open(output_dir / "cve_training_curves.json", "w") as f:
            json.dump(curves, f)

        console.print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f} train_acc={train_acc:.2f}% val_loss={avg_val_loss:.4f} val_acc={val_acc:.2f}%")

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            best_epoch = epoch + 1
            no_improve = 0
            torch.save(model.state_dict(), output_dir / "cve-classifier-best.pt")
        else:
            no_improve += 1

        if no_improve >= args.patience:
            console.print(f"[yellow]Early stopping at epoch {epoch+1}[/yellow]")
            break

    # Export
    best_path = output_dir / "cve-classifier-best.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    onnx_path = export_onnx(model, len(vocab), output_dir, device)

    # Quick benchmark
    if device.type == "cuda":
        dummy = torch.randint(0, len(vocab), (256, MAX_LEN)).to(device)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(50):
            with torch.no_grad():
                model(dummy)
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        throughput = 50 * 256 / elapsed
        console.print(f"[green]GPU inference: {throughput:.0f} CVEs/sec[/green]")
    else:
        dummy = torch.randint(0, len(vocab), (64, MAX_LEN))
        t0 = time.time()
        for _ in range(20):
            with torch.no_grad():
                model(dummy)
        elapsed = time.time() - t0
        throughput = 20 * 64 / elapsed
        console.print(f"[green]CPU inference: {throughput:.0f} CVEs/sec[/green]")

    report = {
        "model": "cve_vuln_type_classifier",
        "architecture": "TextCNN",
        "classes": VULN_CLASSES,
        "vocab_size": len(vocab),
        "max_len": MAX_LEN,
        "epochs_trained": epoch + 1,
        "best_epoch": best_epoch,
        "best_val_acc": best_acc,
        "label_distribution": dict(label_counts),
        "param_count": param_count,
        "throughput_cves_per_sec": throughput,
    }
    with open(output_dir / "cve_classifier_report.json", "w") as f:
        json.dump(report, f, indent=2)

    console.print(f"\n[green]CVE classifier done! Best acc: {best_acc:.2f}% at epoch {best_epoch}[/green]")
    console.print(f"[green]Throughput: {throughput:.0f} CVEs/sec — classifies entire KEV catalog in <1s[/green]")


def get_args():
    p = argparse.ArgumentParser(description="Train CVE vulnerability type classifier")
    p.add_argument("--output-dir", default="models/cve", help="Output directory")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=7)
    return p.parse_args()


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm_handler)
    train(get_args())
