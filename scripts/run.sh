#!/bin/bash
set -euo pipefail
# Auto-run script: download dataset, run 5-model training pipeline in tmux
# Handles preemptible VM SIGTERM — resumes from pipeline_state.json on reboot
exec > >(tee /var/log/vnc-run.log) 2>&1

log() { echo "[VNC-RUN] $(date -Iseconds) $*"; }

REPO_URL="https://github.com/SWORDIntel/VNC-T.git"
CLONE_DIR="/opt/vnc-training-repo"
DATA_DIR="$CLONE_DIR/dataset"
VENV="/opt/vnc-training"
RELEASE_URL="https://github.com/SWORDIntel/VNC-T/releases/download/v1.0/dataset.tar.gz"
STATE_PY="$CLONE_DIR/scripts/pipeline_state.py"

cd "$CLONE_DIR" || {
  log "Repo not found at $CLONE_DIR, cloning..."
  git clone "$REPO_URL" "$CLONE_DIR"
  cd "$CLONE_DIR"
}

# Pull latest
log "Pulling latest repo changes..."
git pull || log "No network or already latest"

# Download dataset only if not already extracted
if [ ! -d "$DATA_DIR" ] || [ "$(find "$DATA_DIR" -type f | wc -l)" -lt 100 ]; then
  DATASET_ARCHIVE="/tmp/dataset.tar.gz"
  if [ ! -f "$DATASET_ARCHIVE" ]; then
    log "Downloading dataset from GitHub release..."
    wget -q --show-progress "$RELEASE_URL" -O "$DATASET_ARCHIVE" || \
      curl -L --progress-bar "$RELEASE_URL" -o "$DATASET_ARCHIVE" || {
        log "Failed to download dataset!"
        exit 1
      }
  fi
  log "Extracting dataset to $DATA_DIR..."
  rm -rf "$DATA_DIR"
  mkdir -p "$DATA_DIR"
  tar xzf "$DATASET_ARCHIVE" -C "$DATA_DIR"
  rm -f "$DATASET_ARCHIVE"
fi

# Verify dataset
CLASSES=$(find "$DATA_DIR" -maxdepth 1 -mindepth 1 -type d | wc -l)
IMAGES=$(find "$DATA_DIR" -type f | wc -l)
log "Dataset: $CLASSES classes, $IMAGES images"

# Activate venv
source "$VENV/bin/activate"

# Show current pipeline state
log "Pipeline state:"
python3 "$STATE_PY" show

# Build pipeline command — skip completed models
CMD="cd '$CLONE_DIR' && source '$VENV/bin/activate'"

# Model 1: VNC Classifier
if python3 "$STATE_PY" check vnc_classifier; then
  log "✓ vnc_classifier already done — skipping"
else
  CMD="$CMD && echo '=== [1/5] VNC Screenshot Classifier ==='"
  CMD="$CMD && python3 '$STATE_PY' mark vnc_classifier running"
  CMD="$CMD && python3 scripts/train_vnc_tui.py --data-dir '$DATA_DIR' --output-dir '$CLONE_DIR/models' --epochs 50 --batch-size 256 --backbone mobilenetv2 --resume 2>&1 | tee /tmp/train.log"
  CMD="$CMD && python3 '$STATE_PY' mark vnc_classifier done"
fi

# Model 2: Alarm Detector
if python3 "$STATE_PY" check alarm_detector; then
  log "✓ alarm_detector already done — skipping"
else
  CMD="$CMD && echo '=== [2/5] Alarm State Detector ==='"
  CMD="$CMD && python3 '$STATE_PY' mark alarm_detector running"
  CMD="$CMD && python3 scripts/train_alarm_detector.py --data-dir '$DATA_DIR' --output-dir '$CLONE_DIR/models/alarm' --epochs 30 --batch-size 256 2>&1 | tee /tmp/train_alarm.log"
  CMD="$CMD && python3 '$STATE_PY' mark alarm_detector done"
fi

# Model 3: OS Classifier
if python3 "$STATE_PY" check os_classifier; then
  log "✓ os_classifier already done — skipping"
else
  CMD="$CMD && echo '=== [3/5] OS/Platform Classifier ==='"
  CMD="$CMD && python3 '$STATE_PY' mark os_classifier running"
  CMD="$CMD && python3 scripts/train_os_classifier.py --data-dir '$DATA_DIR' --output-dir '$CLONE_DIR/models/os' --epochs 30 --batch-size 256 2>&1 | tee /tmp/train_os.log"
  CMD="$CMD && python3 '$STATE_PY' mark os_classifier done"
fi

# Model 4: Anomaly Autoencoder
if python3 "$STATE_PY" check anomaly_ae; then
  log "✓ anomaly_ae already done — skipping"
else
  CMD="$CMD && echo '=== [4/5] Anomaly Autoencoder ==='"
  CMD="$CMD && python3 '$STATE_PY' mark anomaly_ae running"
  CMD="$CMD && python3 scripts/train_anomaly_ae.py --data-dir '$DATA_DIR' --output-dir '$CLONE_DIR/models/anomaly' --epochs 30 --batch-size 256 2>&1 | tee /tmp/train_anomaly.log"
  CMD="$CMD && python3 '$STATE_PY' mark anomaly_ae done"
fi

# Model 5: CVE Vulnerability Type Classifier
if python3 "$STATE_PY" check cve_classifier; then
  log "✓ cve_classifier already done — skipping"
else
  CMD="$CMD && echo '=== [5/5] CVE Vulnerability Type Classifier ==='"
  CMD="$CMD && python3 '$STATE_PY' mark cve_classifier running"
  CMD="$CMD && python3 scripts/train_cve_classifier.py --output-dir '$CLONE_DIR/models/cve' --epochs 15 --batch-size 256 2>&1 | tee /tmp/train_cve.log"
  CMD="$CMD && python3 '$STATE_PY' mark cve_classifier done"
fi

CMD="$CMD && echo '=== ALL TRAINING COMPLETE ===' && echo 'Models in: $CLONE_DIR/models/' && echo 'Reports in: $CLONE_DIR/reports/'"

# Start training in tmux
log "Starting training pipeline in tmux session 'train'..."
if tmux has-session -t train 2>/dev/null; then
  tmux kill-session -t train
fi

tmux new-session -d -s train "$CMD"

log "Training pipeline started. Attach with: tmux attach -t train"
log "Monitor: tail -f /tmp/train.log /tmp/train_alarm.log /tmp/train_os.log /tmp/train_anomaly.log"
