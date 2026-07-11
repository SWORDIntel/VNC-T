# VNC Classifier Training

Public repo for auto-deploying VNC screenshot classifier training on an NVIDIA L40s GPU instance.

## Quick Start

Spin up the Nebius instance with the provided `cloud-init.yaml`, then SSH in and attach to the tmux session:

```bash
ssh john@<INSTANCE_IP>
tmux attach -t train
```

Training will auto-start via cloud-init. The TUI shows live progress, per-class accuracy, GPU memory, and best validation accuracy.

## Files

- `scripts/train_vnc_tui.py` — Training with Rich TUI, AMP, checkpointing, OOM fallback, and ONNX/OpenVINO export
- `scripts/augment_dataset.py` — Smart OpenCV augmentation using visual heuristics
- `scripts/setup.sh` — Installs PyTorch/CUDA, dev tools (Codex CLI, Google agents-cli, Antigravity SDK)
- `scripts/run.sh` — Downloads dataset, starts training in tmux
- `cloud-init.yaml` — Full cloud-init for Nebius Ubuntu 24.04 + CUDA 13
- `requirements-gpu.txt` — Python dependencies

## Dataset

The dataset is hosted as a GitHub release asset (`dataset.tar.gz`). The `run.sh` script downloads and extracts it automatically.

To package the dataset locally:

```bash
tar czf dataset.tar.gz -C /path/to/training_dataset_augmented .
```

Upload to the `v1.0` release in this repo.

## Outputs

After training completes, the following files are in `models/`:

- `vnc-classifier-best.pt` — PyTorch best model
- `vnc-classifier.onnx` — ONNX export
- `vnc-classifier-static.xml` + `.bin` — OpenVINO IR for NCS2
- `vnc-classifier-labels.json` — Class label mapping
- `training_report.json` — Final metrics
- `vnc-checkpoint-latest.pt` — Latest checkpoint

## Manual Run

If cloud-init didn't start it or you need to restart:

```bash
sudo bash /opt/vnc-training-repo/scripts/setup.sh
sudo bash /opt/vnc-training-repo/scripts/run.sh
```

## Dev Tools

The setup script also installs (non-fatal if they fail):

- OpenAI Codex CLI: `codex --help`
- Google agents-cli: `agents-cli --help`
- Google Antigravity SDK: `python3 -c "import google_antigravity"`

## License

Internal SWORD Intel use.
