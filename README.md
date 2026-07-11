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

- `scripts/train_vnc_tui.py` — Main VNC screenshot classifier (MobileNetV2, Rich TUI, AMP, checkpointing, ONNX/OpenVINO export)
- `scripts/train_alarm_detector.py` — Alarm state detector (normal/warning/alarm/critical, pseudo-labeled from color heuristics)
- `scripts/train_os_classifier.py` — OS/platform classifier (Windows/Linux/embedded/server/kiosk, MobileNetV3)
- `scripts/train_anomaly_ae.py` — Anomaly autoencoder (unsupervised, ConvAE, reconstruction error = anomaly score)
- `scripts/generate_report.py` — Fancy HTML report with benchmarks, confusion matrix, training curves, sample predictions
- `scripts/augment_dataset.py` — Smart OpenCV augmentation using visual heuristics
- `scripts/setup.sh` — Installs PyTorch/CUDA, dev tools (Codex CLI, Google agents-cli, Antigravity SDK)
- `scripts/run.sh` — Downloads dataset, runs all 4 models sequentially in tmux
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

After all training completes, models are in `models/`:

**VNC Classifier** (`models/`):
- `vnc-classifier-best.pt`, `vnc-classifier.onnx`, `vnc-classifier-static.xml` + `.bin`
- `vnc-classifier-labels.json`, `training_report.json`, `training_curves.json`

**Alarm Detector** (`models/alarm/`):
- `alarm-detector-best.pt`, `alarm-detector.onnx`, `alarm-detector-static.xml`
- `alarm-detector-labels.json`, `alarm_detector_report.json`

**OS Classifier** (`models/os/`):
- `os-classifier-best.pt`, `os-classifier.onnx`, `os-classifier-static.xml`
- `os-classifier-labels.json`, `os_classifier_report.json`

**Anomaly Autoencoder** (`models/anomaly/`):
- `anomaly-autoencoder-best.pt`, `anomaly-autoencoder.onnx`
- `anomaly_detector_report.json` (includes anomaly threshold)

**Reports** (`reports/`):
- `vnc-training-report.html` — Full HTML report with benchmarks, confusion matrix, sample predictions

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
