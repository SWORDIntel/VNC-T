#!/bin/bash
set -euo pipefail
# Auto-run script: download dataset, start training in tmux
exec > >(tee /var/log/vnc-run.log) 2>&1

log() { echo "[VNC-RUN] $(date -Iseconds) $*"; }

REPO_URL="https://github.com/SWORDIntel/VNC-T.git"
CLONE_DIR="/opt/vnc-training-repo"
DATA_DIR="$CLONE_DIR/dataset"
VENV="/opt/vnc-training"
RELEASE_URL="https://github.com/SWORDIntel/VNC-T/releases/download/v1.0/dataset.tar.gz"

cd "$CLONE_DIR" || {
  log "Repo not found at $CLONE_DIR, cloning..."
  git clone "$REPO_URL" "$CLONE_DIR"
  cd "$CLONE_DIR"
}

# Pull latest
log "Pulling latest repo changes..."
git pull || log "No network or already latest"

# Download dataset
DATASET_ARCHIVE="/tmp/dataset.tar.gz"
if [ ! -f "$DATASET_ARCHIVE" ]; then
  log "Downloading dataset from GitHub release..."
  # Try wget then curl
  wget -q --show-progress "$RELEASE_URL" -O "$DATASET_ARCHIVE" || \
    curl -L --progress-bar "$RELEASE_URL" -o "$DATASET_ARCHIVE" || {
      log "Failed to download dataset!"
      exit 1
    }
fi

# Extract dataset
log "Extracting dataset to $DATA_DIR..."
rm -rf "$DATA_DIR"
mkdir -p "$DATA_DIR"
tar xzf "$DATASET_ARCHIVE" -C "$DATA_DIR"
rm -f "$DATASET_ARCHIVE"

# Verify dataset
CLASSES=$(find "$DATA_DIR" -maxdepth 1 -mindepth 1 -type d | wc -l)
IMAGES=$(find "$DATA_DIR" -type f | wc -l)
log "Dataset: $CLASSES classes, $IMAGES images"

# Activate venv
source "$VENV/bin/activate"

# Start training in tmux with TUI
log "Starting training in tmux session 'train'..."
if tmux has-session -t train 2>/dev/null; then
  tmux kill-session -t train
fi

tmux new-session -d -s train \
  "cd '$CLONE_DIR' && \
   source '$VENV/bin/activate' && \
   echo '=== [1/4] VNC Screenshot Classifier ===' && \
   python3 scripts/train_vnc_tui.py \
     --data-dir '$DATA_DIR' \
     --output-dir '$CLONE_DIR/models' \
     --epochs 50 \
     --batch-size 256 \
     --backbone mobilenetv2 \
     --resume \
     2>&1 | tee /tmp/train.log && \
   echo '=== [2/4] Alarm State Detector ===' && \
   python3 scripts/train_alarm_detector.py \
     --data-dir '$DATA_DIR' \
     --output-dir '$CLONE_DIR/models/alarm' \
     --epochs 30 \
     --batch-size 256 \
     2>&1 | tee /tmp/train_alarm.log && \
   echo '=== [3/4] OS/Platform Classifier ===' && \
   python3 scripts/train_os_classifier.py \
     --data-dir '$DATA_DIR' \
     --output-dir '$CLONE_DIR/models/os' \
     --epochs 30 \
     --batch-size 256 \
     2>&1 | tee /tmp/train_os.log && \
   echo '=== [4/4] Anomaly Autoencoder ===' && \
   python3 scripts/train_anomaly_ae.py \
     --data-dir '$DATA_DIR' \
     --output-dir '$CLONE_DIR/models/anomaly' \
     --epochs 30 \
     --batch-size 256 \
     2>&1 | tee /tmp/train_anomaly.log && \
   echo '=== ALL TRAINING COMPLETE ===' && \
   echo 'Models in: $CLONE_DIR/models/' && \
   echo 'Reports in: $CLONE_DIR/reports/'"

log "Training pipeline started (4 models). Attach with: tmux attach -t train"
log "Monitor: tail -f /tmp/train.log /tmp/train_alarm.log /tmp/train_os.log /tmp/train_anomaly.log"
