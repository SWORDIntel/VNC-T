#!/bin/bash
set -euo pipefail
# One-command setup for VNC classifier training on NVIDIA L40s / CUDA 13
exec > >(tee /var/log/vnc-setup.log) 2>&1

log() { echo "[VNC-SETUP] $(date -Iseconds) $*"; }

REPO_URL="https://github.com/SWORDIntel/VNC-T.git"
CLONE_DIR="/opt/vnc-training-repo"
VENV="/opt/vnc-training"

log "Starting setup on $(hostname)"

# 1. System packages
log "Installing system packages..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -y -q \
  python3.12 python3.12-venv python3.12-dev python3-pip \
  build-essential cmake pkg-config \
  libopencv-dev libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
  git curl wget tmux htop nvtop unzip \
  nodejs npm

# 2. Verify Python
PYTHON="python3.12"
$PYTHON --version

# 3. Create venv
log "Creating Python virtual environment at $VENV..."
$PYTHON -m venv "$VENV" || rm -rf "$VENV" && $PYTHON -m venv "$VENV"
source "$VENV/bin/activate"

# 4. Upgrade base tools
pip install --upgrade pip wheel setuptools

# 5. PyTorch with CUDA (try CUDA 12.6 wheels, fall back to CPU if needed)
log "Installing PyTorch for CUDA..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 || {
  log "PyTorch cu126 failed, trying cu124..."
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 || {
    log "PyTorch CUDA failed, installing CPU version..."
    pip install torch torchvision
  }
}

# 6. Core Python deps
log "Installing Python dependencies..."
pip install opencv-python-headless pillow numpy rich tqdm openvino onnx onnxscript matplotlib seaborn scikit-learn

# 7. Dev tools — OpenAI Codex CLI (non-fatal)
log "Installing OpenAI Codex CLI..."
npm install -g @openai/codex || log "Codex CLI install failed, continuing"

# 8. Dev tools — uv (for agents-cli)
log "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh || true
export PATH="$HOME/.cargo/bin:$PATH"

# 9. Dev tools — Google agents-cli (non-fatal)
log "Installing Google agents-cli..."
uvx google-agents-cli setup --skip-auth || {
  log "agents-cli setup failed; trying pip install..."
  pip install google-agents-cli || log "agents-cli install failed, continuing"
}

# 10. Dev tools — Google Antigravity SDK (non-fatal)
log "Installing Google Antigravity SDK..."
pip install google-antigravity || log "Antigravity SDK install failed, continuing"

# 11. Verify GPU
log "GPU check:"
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')" || true

# 12. Clone repo if not already present
if [ ! -d "$CLONE_DIR/.git" ]; then
  log "Cloning VNC-T repo..."
  git clone "$REPO_URL" "$CLONE_DIR"
else
  log "Repo already exists, pulling latest..."
  (cd "$CLONE_DIR" && git pull)
fi

# 13. Make scripts executable
chmod +x "$CLONE_DIR/scripts/"*.py "$CLONE_DIR/scripts/"*.sh || true

log "Setup complete. Run /opt/vnc-training-repo/scripts/run.sh to start training."
