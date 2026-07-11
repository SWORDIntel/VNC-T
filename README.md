# VNC Classifier Training

Public repo for auto-deploying VNC screenshot classifier training on an NVIDIA L40s GPU instance.

## Quick Start

Spin up the Nebius instance with the provided `cloud-init.yaml`, then SSH in and attach to the tmux session:

```bash
ssh john@<INSTANCE_IP>
tmux attach -t train
```

Training will auto-start via cloud-init. The TUI shows live progress, per-class accuracy, GPU memory, and best validation accuracy.

## CLI Launcher

`launch.sh` in the repo root creates the VM via Nebius CLI, waits for SSH, and drops into a live progress monitor.

```bash
# Full: create VM + wait + monitor
export PROJECT_ID=proj-xxx
export PLATFORM=<platform>          # nebius compute platform list
export PRESET=<preset>              # nebius compute preset list
export SUBNET_ID=<subnet-id>        # nebius vpc subnet list
./launch.sh

# Just monitor an existing VM
./launch.sh --monitor

# Check VM status + pipeline state
./launch.sh --status

# SSH in directly
./launch.sh --ssh

# Stop / start / delete
./launch.sh --stop
./launch.sh --start
./launch.sh --delete
```

The progress monitor shows pipeline state (✅/🔄/⏳ per model), GPU utilization, tmux session status, and recent log output — refreshing every 5 seconds. Ctrl+C exits the monitor (VM keeps running).

## Files

- `scripts/train_vnc_tui.py` — Main VNC screenshot classifier (MobileNetV2, Rich TUI, AMP, checkpointing, ONNX/OpenVINO export)
- `scripts/train_alarm_detector.py` — Alarm state detector (normal/warning/alarm/critical, pseudo-labeled from color heuristics)
- `scripts/train_os_classifier.py` — OS/platform classifier (Windows/Linux/embedded/server/kiosk, MobileNetV3)
- `scripts/train_anomaly_ae.py` — Anomaly autoencoder (unsupervised, ConvAE, reconstruction error = anomaly score)
- `scripts/train_cve_classifier.py` — CVE vulnerability type classifier (TextCNN, downloads NVD data, classifies CVEs as RCE/LPE/DoS/etc.)
- `scripts/pipeline_state.py` — Pipeline state tracker for preemptible VM resume
- `scripts/generate_report.py` — Fancy HTML report with benchmarks, confusion matrix, training curves, sample predictions
- `scripts/augment_dataset.py` — Smart OpenCV augmentation using visual heuristics
- `scripts/setup.sh` — Installs PyTorch/CUDA, dev tools (Codex CLI, Google agents-cli, Antigravity SDK)
- `scripts/run.sh` — Downloads dataset, runs all 5 models sequentially in tmux (preemptible VM safe)
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

**CVE Classifier** (`models/cve/`):
- `cve-classifier-best.pt`, `cve-classifier.onnx`
- `cve-vocab.json`, `cve-classifier-labels.json`, `cve_classifier_report.json`

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

## Preemptible VM Support

The pipeline is designed for preemptible (spot) VMs that can be killed at any time with a 60-second SIGTERM notice.

**How it works:**

1. **SIGTERM handler** in every training script catches the signal and saves an emergency checkpoint before exit
2. **`pipeline_state.json`** tracks which models have completed — survives reboots
3. **`cloud-init.yaml`** only runs full setup on first boot (sentinel file), then always re-runs the pipeline
4. **`run.sh`** skips completed models and resumes from the interrupted one
5. **Dataset** is only re-downloaded if missing (not every boot)

**State management:**

```bash
python3 scripts/pipeline_state.py show    # current state
python3 scripts/pipeline_state.py reset   # reset all to pending
```

The VM can be killed and restarted multiple times — it always picks up where it left off.

## Pipeline Overview

```
cloud-init.yaml
  ├─ first boot: /opt/vnc-setup.sh  (install CUDA, PyTorch, deps)
  └─ every boot: /opt/vnc-run.sh    (resume pipeline)
       └─ tmux session "train"
            ├─ [1/5] VNC Screenshot Classifier   (MobileNetV2, 50 epochs)
            ├─ [2/5] Alarm State Detector         (MobileNetV2, 30 epochs)
            ├─ [3/5] OS/Platform Classifier       (MobileNetV3, 30 epochs)
            ├─ [4/5] Anomaly Autoencoder          (ConvAE, 30 epochs)
            └─ [5/5] CVE Vulnerability Classifier  (TextCNN, 15 epochs)
```

Each model:
- Trains with AMP (mixed precision) on the L40s GPU
- Saves best checkpoint + training curves
- Exports to ONNX (and OpenVINO IR where applicable)
- Handles SIGTERM → emergency checkpoint → resume on reboot
