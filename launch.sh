#!/bin/bash
set -euo pipefail

# VNC-T Pipeline Launcher — Nebius CLI
# Creates a preemptible GPU VM, waits for it to come up, then SSHs in
# and drops into a crude progress monitor for the 5-model training pipeline.
#
# Usage:
#   ./launch.sh                          # create + monitor
#   ./launch.sh --monitor                # just monitor existing VM
#   ./launch.sh --status                 # just check VM status
#   ./launch.sh --ssh                    # just SSH in
#   ./launch.sh --stop                   # stop the VM
#   ./launch.sh --delete                 # delete the VM
#
# Requires: nebius CLI, jq, ssh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLOUD_INIT_FILE="$SCRIPT_DIR/cloud-init.yaml"

# --- Config (override via env vars) ---
VM_NAME="${VM_NAME:-vnc-training}"
PROJECT_ID="${PROJECT_ID:-}"
PLATFORM="${PLATFORM:-}"
PRESET="${PRESET:-}"
SUBNET_ID="${SUBNET_ID:-}"
DISK_SIZE_GIB="${DISK_SIZE_GIB:-250}"
DISK_TYPE="${DISK_TYPE:-network_ssd}"
IMAGE_FAMILY="${IMAGE_FAMILY:-ubuntu24.04-cuda13}"
SSH_USER="${SSH_USER:-john}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
SSH_HOST="${SSH_HOST:-}"  # set after VM creation or override manually

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log()   { echo -e "${CYAN}[launch]${NC} $(date -Iseconds) $*"; }
ok()    { echo -e "${GREEN}[launch]${NC} $*"; }
warn()  { echo -e "${YELLOW}[launch]${NC} $*"; }
err()   { echo -e "${RED}[launch]${NC} $*" >&2; }

die() { err "$*"; exit 1; }

# --- Helpers ---

check_deps() {
  local missing=()
  command -v nebius >/dev/null 2>&1 || missing+=(nebius)
  command -v jq >/dev/null 2>&1 || missing+=(jq)
  command -v ssh >/dev/null 2>&1 || missing+=(ssh)
  if [ ${#missing[@]} -gt 0 ]; then
    err "Missing dependencies: ${missing[*]}"
    err "Install with: "
    err "  nebius:  See https://docs.nebius.com/cli/"
    err "  jq:      apt install jq / brew install jq"
    err "  ssh:     apt install openssh-client / brew install openssh"
    exit 1
  fi
}

get_vm_json() {
  nebius compute instance get-by-name --name "$VM_NAME" --format json 2>/dev/null || echo "{}"
}

get_vm_status() {
  local json
  json=$(get_vm_json)
  echo "$json" | jq -r '.status.state // "not_found"'
}

get_vm_ip() {
  local json
  json=$(get_vm_json)
  echo "$json" | jq -r '.status.network_interfaces[0].public_ip_address.address // empty' | cut -d/ -f1
}

wait_for_ip() {
  log "Waiting for public IP assignment..."
  local ip=""
  for i in $(seq 1 30); do
    ip=$(get_vm_ip)
    if [ -n "$ip" ]; then
      ok "Public IP: $ip"
      echo "$ip"
      return 0
    fi
    sleep 5
  done
  err "Timed out waiting for public IP"
  return 1
}

wait_for_ssh() {
  local ip="$1"
  log "Waiting for SSH to come up at ${ip}..."
  for i in $(seq 1 60); do
    if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -o BatchMode=yes \
          -i "$SSH_KEY" "$SSH_USER@$ip" "echo ready" 2>/dev/null; then
      ok "SSH is up!"
      return 0
    fi
    printf "."
    sleep 5
  done
  echo
  err "Timed out waiting for SSH"
  return 1
}

# --- Actions ---

do_create() {
  if [ -z "$PROJECT_ID" ]; then
    die "PROJECT_ID not set. Export it: export PROJECT_ID=<your-project-id>"
  fi
  if [ -z "$PLATFORM" ] || [ -z "$PRESET" ]; then
    warn "PLATFORM/PRESET not set — listing available options..."
    warn "Platforms:"
    nebius compute platform list --format json 2>/dev/null | jq -r '.items[] | "  \(.name) (id: \(.id))"' 2>/dev/null || \
      nebius compute platform list 2>/dev/null || warn "  (could not list platforms)"
    warn "Presets:"
    nebius compute preset list --format json 2>/dev/null | jq -r '.items[] | "  \(.name) — \(.description // "")"' 2>/dev/null || \
      nebius compute preset list 2>/dev/null || warn "  (could not list presets)"
    die "Set PLATFORM and PRESET env vars, then re-run."
  fi
  if [ -z "$SUBNET_ID" ]; then
    warn "SUBNET_ID not set — listing subnets..."
    nebius vpc subnet list --format json 2>/dev/null | jq -r '.items[] | "  \(.name) (id: \(.id))"' 2>/dev/null || \
      nebius vpc subnet list 2>/dev/null || warn "  (could not list subnets)"
    die "Set SUBNET_ID env var, then re-run."
  fi

  if [ ! -f "$CLOUD_INIT_FILE" ]; then
    die "cloud-init.yaml not found at $CLOUD_INIT_FILE"
  fi

  local existing_status
  existing_status=$(get_vm_status)
  if [ "$existing_status" != "not_found" ]; then
    warn "VM '$VM_NAME' already exists (status: $existing_status)"
    read -rp "Delete and recreate? [y/N] " confirm
    [ "$confirm" = "y" ] || { ok "Keeping existing VM."; return 0; }
    do_delete
  fi

  log "Creating preemptible VM '$VM_NAME'..."
  log "  Platform: $PLATFORM"
  log "  Preset:   $PRESET"
  log "  Disk:     ${DISK_SIZE_GIB}GiB $DISK_TYPE"
  log "  Image:    $IMAGE_FAMILY"
  log "  Subnet:   $SUBNET_ID"

  nebius compute instance create \
    --name "$VM_NAME" \
    --parent-id "$PROJECT_ID" \
    --resources-platform "$PLATFORM" \
    --resources-preset "$PRESET" \
    --boot-disk-managed-disk-size-gibibytes "$DISK_SIZE_GIB" \
    --boot-disk-managed-disk-type "$DISK_TYPE" \
    --boot-disk-managed-disk-source-image-family-image-family "$IMAGE_FAMILY" \
    --boot-disk-attach-mode READ_WRITE \
    --preemptible-on-preemption stop \
    --recovery-policy recover \
    --cloud-init-user-data "$(cat "$CLOUD_INIT_FILE")" \
    --network-interfaces "[{\"name\":\"eth0\",\"subnet_id\":\"$SUBNET_ID\",\"ip_address\":{},\"public_ip_address\":{}}]"

  ok "VM created!"
}

do_monitor() {
  local ip="${SSH_HOST:-$(get_vm_ip)}"
  if [ -z "$ip" ]; then
    die "No VM found or no public IP. Set SSH_HOST or create a VM first."
  fi

  log "Connecting to ${SSH_USER}@${ip} for progress monitoring..."
  log "Press Ctrl+C to disconnect (VM keeps running)."

  # SSH in and run a crude progress monitor that refreshes every 5 seconds
  ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" "$SSH_USER@$ip" bash -s <<'REMOTE_SCRIPT'
    while true; do
      clear
      echo "╔══════════════════════════════════════════════════════════════╗"
      echo "║          VNC-T Training Pipeline — Progress Monitor          ║"
      echo "╠══════════════════════════════════════════════════════════════╣"

      # Pipeline state
      STATE_FILE="/opt/vnc-training-repo/pipeline_state.json"
      if [ -f "$STATE_FILE" ]; then
        echo "║  Pipeline State:                                             ║"
        python3 -c "
import json
state = json.load(open('$STATE_FILE'))
icons = {'done': '✅', 'running': '🔄', 'pending': '⏳'}
names = {
    'vnc_classifier': 'VNC Screenshot Classifier',
    'alarm_detector': 'Alarm State Detector',
    'os_classifier':  'OS/Platform Classifier',
    'anomaly_ae':     'Anomaly Autoencoder',
    'cve_classifier': 'CVE Vuln Type Classifier',
}
for model, status in state.items():
    name = names.get(model, model)
    icon = icons.get(status, '?')
    print(f'║  {icon} {name:30s} [{status:8s}]                   ║')
" 2>/dev/null || echo "║  (state file exists but unreadable)                          ║"
      else
        echo "║  ⏳ Pipeline not started yet (no state file)                 ║"
      fi

      echo "╠══════════════════════════════════════════════════════════════╣"

      # GPU status
      if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
        if [ -n "$GPU_UTIL" ]; then
          IFS=',' read -r UTIL MEM_USED MEM_TOTAL TEMP <<< "$GPU_UTIL"
          echo "║  GPU: ${UTIL}% util | ${MEM_USED}/${MEM_TOTAL} MiB | ${TEMP}C"
        fi
      fi

      # tmux session check
      if tmux has-session -t train 2>/dev/null; then
        echo "║  tmux [train]: ACTIVE                                        ║"
      else
        echo "║  tmux [train]: not running                                   ║"
      fi

      # Latest log lines
      echo "╠══════════════════════════════════════════════════════════════╣"
      echo "║  Recent log output:                                          ║"
      LOG="/tmp/train.log"
      [ -f "/tmp/train_alarm.log" ] && LOG="/tmp/train_alarm.log"
      [ -f "/tmp/train_os.log" ] && LOG="/tmp/train_os.log"
      [ -f "/tmp/train_anomaly.log" ] && LOG="/tmp/train_anomaly.log"
      [ -f "/tmp/train_cve.log" ] && LOG="/tmp/train_cve.log"
      if [ -f "$LOG" ]; then
        tail -8 "$LOG" 2>/dev/null | while IFS= read -r line; do
          printf "║  %.58s\n" "$line"
        done
      else
        echo "║  (no training logs yet)                                      ║"
      fi

      echo "╠══════════════════════════════════════════════════════════════╣"
      echo "║  Attach to tmux:   tmux attach -t train                      ║"
      echo "║  Pipeline state:   python3 scripts/pipeline_state.py show    ║"
      echo "║  Refresh: every 5s | Ctrl+C to exit monitor                  ║"
      echo "╚══════════════════════════════════════════════════════════════╝"
      echo "Last update: $(date)"
      sleep 5
    done
REMOTE_SCRIPT
}

do_status() {
  local status
  status=$(get_vm_status)
  local ip
  ip=$(get_vm_ip)
  echo "VM:       $VM_NAME"
  echo "Status:   $status"
  echo "Public IP: ${ip:-none}"

  if [ -n "$ip" ] && [ "$status" = "running" ]; then
    log "Checking pipeline state on VM..."
    ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -i "$SSH_KEY" "$SSH_USER@$ip" \
      "python3 /opt/vnc-training-repo/scripts/pipeline_state.py show 2>/dev/null || echo '(pipeline not started)'" 2>/dev/null || \
      warn "Could not reach VM via SSH"
  fi
}

do_ssh() {
  local ip="${SSH_HOST:-$(get_vm_ip)}"
  [ -z "$ip" ] && die "No VM found or no public IP."
  log "SSH to ${SSH_USER}@${ip}"
  ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" "$SSH_USER@$ip"
}

do_stop() {
  log "Stopping VM '$VM_NAME'..."
  nebius compute instance stop --name "$VM_NAME"
  ok "VM stopped."
}

do_start() {
  log "Starting VM '$VM_NAME'..."
  nebius compute instance start --name "$VM_NAME"
  ok "VM starting. Pipeline will auto-resume via cloud-init."
}

do_delete() {
  log "Deleting VM '$VM_NAME'..."
  nebius compute instance delete --name "$VM_NAME" 2>/dev/null || {
    warn "Could not delete by name, trying to get ID..."
    local id
    id=$(get_vm_json | jq -r '.id // empty')
    [ -n "$id" ] && nebius compute instance delete --id "$id" || die "Could not delete VM"
  }
  ok "VM deleted."
}

do_create_and_monitor() {
  do_create
  local ip
  ip=$(wait_for_ip) || die "No public IP"
  wait_for_ssh "$ip" || die "SSH never came up"
  do_monitor
}

# --- Main ---

check_deps

case "${1:-}" in
  --monitor|-m)    do_monitor ;;
  --status|-s)     do_status ;;
  --ssh)           do_ssh ;;
  --stop)          do_stop ;;
  --start)         do_start ;;
  --delete|-d)     do_delete ;;
  --create|-c)     do_create ;;
  --help|-h)
    cat <<EOF
VNC-T Pipeline Launcher — Nebius CLI

Usage: launch.sh [command]

Commands:
  (none)       Create VM + wait + drop into progress monitor
  --create     Create the VM only
  --monitor    Connect to existing VM and show progress monitor
  --status     Show VM status + pipeline state
  --ssh        SSH into the VM
  --start      Start a stopped VM
  --stop       Stop the VM (keeps disk)
  --delete     Delete the VM entirely
  --help       This help

Environment variables (override as needed):
  VM_NAME       VM name (default: vnc-training)
  PROJECT_ID    Nebius project ID (required for create)
  PLATFORM      GPU platform name (required for create)
  PRESET        GPU preset name (required for create)
  SUBNET_ID     VPC subnet ID (required for create)
  DISK_SIZE_GIB Boot disk size in GiB (default: 250)
  DISK_TYPE     Disk type (default: network_ssd)
  IMAGE_FAMILY  Boot image family (default: ubuntu24.04-cuda13)
  SSH_USER      SSH username (default: john)
  SSH_KEY       Path to SSH private key (default: ~/.ssh/id_ed25519)
  SSH_HOST      Override IP for monitor/ssh (skip VM lookup)

Examples:
  # Full create + monitor
  PROJECT_ID=proj-xxx PLATFORM=... PRESET=... SUBNET_ID=... ./launch.sh

  # Just monitor an existing VM
  ./launch.sh --monitor

  # Check status
  ./launch.sh --status
EOF
    ;;
  *)               do_create_and_monitor ;;
esac
