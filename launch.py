#!/usr/bin/env python3
"""VNC-T Pipeline Launcher — TUI dashboard with VM creation and pre-seeding.

Creates a preemptible GPU VM on Nebius, pre-seeds the repo via rsync,
then drops into a rich TUI dashboard showing live training progress.

Usage:
  python3 launch.py                          # create + pre-seed + monitor
  python3 launch.py --monitor                # just monitor existing VM
  python3 launch.py --status                 # just check VM status
  python3 launch.py --ssh                    # just SSH in
  python3 launch.py --stop                   # stop the VM
  python3 launch.py --start                  # start a stopped VM
  python3 launch.py --delete                 # delete the VM
  python3 launch.py --seed                   # just pre-seed repo to VM

Requires: nebius CLI, jq, ssh, rsync, Python rich
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
    from rich.align import Align
    from rich import box
except ImportError:
    print("Installing rich...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
    from rich.align import Align
    from rich import box

console = Console()

# --- Config (override via env vars) ---
VM_NAME = os.environ.get("VM_NAME", "vnc-training")
PROJECT_ID = os.environ.get("PROJECT_ID", "")
PLATFORM = os.environ.get("PLATFORM", "")
PRESET = os.environ.get("PRESET", "")
SUBNET_ID = os.environ.get("SUBNET_ID", "")
DISK_SIZE_GIB = os.environ.get("DISK_SIZE_GIB", "250")
DISK_TYPE = os.environ.get("DISK_TYPE", "network_ssd")
IMAGE_FAMILY = os.environ.get("IMAGE_FAMILY", "ubuntu24.04-cuda13")
SSH_USER = os.environ.get("SSH_USER", "john")
SSH_KEY = os.environ.get("SSH_KEY", os.path.expanduser("~/.ssh/id_ed25519"))
SSH_HOST = os.environ.get("SSH_HOST", "")
REPO_DIR = Path(__file__).parent.resolve()
CLOUD_INIT_FILE = REPO_DIR / "cloud-init.yaml"

MODEL_NAMES = {
    "vnc_classifier": "VNC Screenshot Classifier",
    "alarm_detector": "Alarm State Detector",
    "os_classifier": "OS/Platform Classifier",
    "anomaly_ae": "Anomaly Autoencoder",
    "cve_classifier": "CVE Vuln Type Classifier",
}
MODEL_ICONS = {"done": "✅", "running": "🔄", "pending": "⏳"}


# --- Helpers ---

def run(cmd, capture=True, check=False, timeout=30):
    """Run a shell command."""
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=capture, text=True, timeout=timeout
        )
        if check and r.returncode != 0:
            console.print(f"[red]Command failed: {cmd}[/red]")
            if r.stderr:
                console.print(r.stderr)
        return r
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, 124, "", "timeout")


def check_deps():
    missing = []
    for dep in ["nebius", "jq", "ssh", "rsync"]:
        if run(f"command -v {dep}").returncode != 0:
            missing.append(dep)
    if missing:
        console.print(f"[red]Missing: {', '.join(missing)}[/red]")
        sys.exit(1)


def get_vm_json():
    r = run(f"nebius compute instance get-by-name --name {VM_NAME} --format json")
    try:
        return json.loads(r.stdout) if r.stdout.strip() else {}
    except json.JSONDecodeError:
        return {}


def get_vm_status():
    j = get_vm_json()
    return j.get("status", {}).get("state", "not_found")


def get_vm_ip():
    j = get_vm_json()
    addr = j.get("status", {}).get("network_interfaces", [{}])[0].get("public_ip_address", {}).get("address", "")
    return addr.split("/")[0] if addr else ""


def wait_for_ip(timeout=150):
    console.print("[cyan]Waiting for public IP...[/cyan]")
    for i in range(timeout // 5):
        ip = get_vm_ip()
        if ip:
            console.print(f"[green]Public IP: {ip}[/green]")
            return ip
        time.sleep(5)
    console.print("[red]Timed out waiting for IP[/red]")
    return None


def wait_for_ssh(ip, timeout=300):
    console.print(f"[cyan]Waiting for SSH at {ip}...[/cyan]")
    for i in range(timeout // 5):
        r = run(
            f"ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -o BatchMode=yes "
            f"-i {SSH_KEY} {SSH_USER}@{ip} echo ready"
        )
        if r.returncode == 0:
            console.print("[green]SSH is up![/green]")
            return True
        time.sleep(5)
    console.print("[red]SSH never came up[/red]")
    return False


def ssh_cmd(ip, remote_cmd, timeout=15):
    """Run a command on the VM via SSH."""
    return run(
        f"ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -o BatchMode=yes "
        f"-i {SSH_KEY} {SSH_USER}@{ip} '{remote_cmd}'",
        timeout=timeout,
    )


# --- Pre-seed ---

def preseed_vm(ip):
    """Rsync the repo to the VM, install deps if needed, then kick off the pipeline."""
    console.print("[cyan]Pre-seeding repo to VM via rsync...[/cyan]")

    remote_dir = "/opt/vnc-training-repo"

    # Ensure remote dir exists
    ssh_cmd(ip, f"sudo mkdir -p {remote_dir} && sudo chown {SSH_USER}:{SSH_USER} {remote_dir}")

    # Rsync repo files (exclude .git, dataset, models, runtime artifacts)
    excludes = [
        "--exclude=.git",
        "--exclude=dataset/",
        "--exclude=models/",
        "--exclude=reports/",
        "--exclude=pipeline_state.json",
        "--exclude=nvd_data/",
        "--exclude=__pycache__/",
        "--exclude=*.pyc",
        "--exclude=*.log",
        "--exclude=*.tar.gz",
    ]
    rsync_cmd = (
        f"rsync -azP --delete {' '.join(excludes)} "
        f"-e 'ssh -o StrictHostKeyChecking=no -i {SSH_KEY}' "
        f"{REPO_DIR}/ {SSH_USER}@{ip}:{remote_dir}/"
    )
    r = run(rsync_cmd, timeout=120)
    if r.returncode != 0:
        console.print(f"[red]rsync failed: {r.stderr}[/red]")
        return False
    console.print("[green]Repo synced![/green]")

    # Check if venv exists (setup already ran via cloud-init)
    r = ssh_cmd(ip, "test -d /opt/vnc-training/bin && echo yes || echo no")
    venv_exists = "yes" in (r.stdout or "")

    if not venv_exists:
        console.print("[yellow]Venv not found — running setup.sh on VM...[/yellow]")
        ssh_cmd(
            ip,
            f"cd {remote_dir} && bash scripts/setup.sh",
            timeout=600,
        )
    else:
        console.print("[green]Venv already exists — skipping setup[/green]")

    # Make scripts executable
    ssh_cmd(ip, f"chmod +x {remote_dir}/scripts/*.py {remote_dir}/scripts/*.sh")

    # Check if pipeline is already running
    r = ssh_cmd(ip, "tmux has-session -t train 2>/dev/null && echo running || echo stopped")
    tmux_running = "running" in (r.stdout or "")

    if tmux_running:
        console.print("[green]Training pipeline already running — skipping launch[/green]")
    else:
        console.print("[cyan]Starting training pipeline...[/cyan]")
        ssh_cmd(
            ip,
            f"cd {remote_dir} && bash scripts/run.sh",
            timeout=30,
        )
        console.print("[green]Pipeline started![/green]")

    return True


# --- VM actions ---

def do_create():
    if not PROJECT_ID:
        console.print("[red]PROJECT_ID not set. export PROJECT_ID=<id>[/red]")
        sys.exit(1)
    if not PLATFORM or not PRESET:
        console.print("[yellow]PLATFORM/PRESET not set. Listing options...[/yellow]")
        r = run("nebius compute platform list --format json")
        try:
            for p in json.loads(r.stdout).get("items", []):
                console.print(f"  {p['name']} (id: {p['id']})")
        except Exception:
            console.print("  (could not list)")
        r = run("nebius compute preset list --format json")
        try:
            for p in json.loads(r.stdout).get("items", []):
                console.print(f"  {p['name']} — {p.get('description', '')}")
        except Exception:
            console.print("  (could not list)")
        sys.exit(1)
    if not SUBNET_ID:
        console.print("[yellow]SUBNET_ID not set. Listing subnets...[/yellow]")
        r = run("nebius vpc subnet list --format json")
        try:
            for s in json.loads(r.stdout).get("items", []):
                console.print(f"  {s['name']} (id: {s['id']})")
        except Exception:
            console.print("  (could not list)")
        sys.exit(1)

    if not CLOUD_INIT_FILE.exists():
        console.print(f"[red]cloud-init.yaml not found at {CLOUD_INIT_FILE}[/red]")
        sys.exit(1)

    existing = get_vm_status()
    if existing != "not_found":
        console.print(f"[yellow]VM '{VM_NAME}' already exists (status: {existing})[/yellow]")
        resp = input("Delete and recreate? [y/N] ").strip().lower()
        if resp != "y":
            console.print("[green]Keeping existing VM.[/green]")
            return
        do_delete()

    console.print(f"[cyan]Creating preemptible VM '{VM_NAME}'...[/cyan]")
    console.print(f"  Platform: {PLATFORM}")
    console.print(f"  Preset:   {PRESET}")
    console.print(f"  Disk:     {DISK_SIZE_GIB}GiB {DISK_TYPE}")
    console.print(f"  Image:    {IMAGE_FAMILY}")
    console.print(f"  Subnet:   {SUBNET_ID}")

    cloud_init = CLOUD_INIT_FILE.read_text()
    net_json = json.dumps([{
        "name": "eth0",
        "subnet_id": SUBNET_ID,
        "ip_address": {},
        "public_ip_address": {},
    }])

    cmd = (
        f"nebius compute instance create "
        f"--name {VM_NAME} "
        f"--parent-id {PROJECT_ID} "
        f"--resources-platform {PLATFORM} "
        f"--resources-preset {PRESET} "
        f"--boot-disk-managed-disk-size-gibibytes {DISK_SIZE_GIB} "
        f"--boot-disk-managed-disk-type {DISK_TYPE} "
        f"--boot-disk-managed-disk-source-image-family-image-family {IMAGE_FAMILY} "
        f"--boot-disk-attach-mode READ_WRITE "
        f"--preemptible-on-preemption stop "
        f"--recovery-policy recover "
        f"--cloud-init-user-data '{cloud_init.replace(chr(39), chr(92)+chr(39))}' "
        f"--network-interfaces '{net_json}'"
    )
    r = run(cmd, timeout=120, check=True)
    if r.returncode == 0:
        console.print("[green]VM created![/green]")


def do_create_and_monitor():
    do_create()
    ip = wait_for_ip()
    if not ip:
        sys.exit(1)
    if not wait_for_ssh(ip):
        sys.exit(1)
    preseed_vm(ip)
    run_tui(ip)


def do_delete():
    console.print(f"[cyan]Deleting VM '{VM_NAME}'...[/cyan]")
    run(f"nebius compute instance delete --name {VM_NAME}", timeout=60)
    console.print("[green]VM deleted.[/green]")


def do_stop():
    console.print(f"[cyan]Stopping VM '{VM_NAME}'...[/cyan]")
    run(f"nebius compute instance stop --name {VM_NAME}", timeout=60)
    console.print("[green]VM stopped.[/green]")


def do_start():
    console.print(f"[cyan]Starting VM '{VM_NAME}'...[/cyan]")
    run(f"nebius compute instance start --name {VM_NAME}", timeout=60)
    console.print("[green]VM starting. Pipeline will auto-resume via cloud-init.[/green]")


def do_ssh():
    ip = SSH_HOST or get_vm_ip()
    if not ip:
        console.print("[red]No VM found or no public IP.[/red]")
        sys.exit(1)
    console.print(f"[cyan]SSH to {SSH_USER}@{ip}[/cyan]")
    os.execvp("ssh", ["ssh", "-o", "StrictHostKeyChecking=no", "-i", SSH_KEY, f"{SSH_USER}@{ip}"])


def do_status():
    status = get_vm_status()
    ip = get_vm_ip()
    console.print(f"VM:        {VM_NAME}")
    console.print(f"Status:    {status}")
    console.print(f"Public IP: {ip or 'none'}")
    if ip and status == "running":
        console.print("[cyan]Checking pipeline state on VM...[/cyan]")
        r = ssh_cmd(ip, f"python3 /opt/vnc-training-repo/scripts/pipeline_state.py show 2>/dev/null || echo '(not started)'")
        if r.stdout:
            console.print(r.stdout.strip())


# --- TUI Dashboard ---

def fetch_vm_state(ip):
    """Fetch all state from VM in a single SSH call."""
    r = ssh_cmd(ip, """
        echo '===STATE==='
        cat /opt/vnc-training-repo/pipeline_state.json 2>/dev/null || echo '{}'
        echo '===GPU==='
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo 'none'
        echo '===TMUX==='
        tmux has-session -t train 2>/dev/null && echo running || echo stopped
        echo '===LOG==='
        for f in /tmp/train_cve.log /tmp/train_anomaly.log /tmp/train_os.log /tmp/train_alarm.log /tmp/train.log; do
            if [ -f "$f" ]; then tail -12 "$f"; break; fi
        done
        echo '===UPTIME==='
        uptime 2>/dev/null || echo 'unknown'
        echo '===DISK==='
        df -h /opt 2>/dev/null | tail -1 || echo 'unknown'
        echo '===DONE==='
    """, timeout=15)

    sections = {}
    current = None
    for line in (r.stdout or "").splitlines():
        if line.startswith("===") and line.endswith("==="):
            current = line.strip("= ")
            sections[current] = []
        elif current:
            sections[current].append(line)

    return sections


def build_dashboard(data, ip):
    """Build the rich TUI layout from VM state."""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )
    layout["body"].split_row(
        Layout(name="left"),
        Layout(name="right"),
    )

    # Header
    header = Panel(
        Align.center(
            Text("VNC-T Training Pipeline", style="bold cyan", justify="center")
            + Text(f"  |  {SSH_USER}@{ip}  |  {time.strftime('%H:%M:%S')}", style="dim", justify="center")
        ),
        border_style="cyan",
    )
    layout["header"].update(header)

    # Pipeline state table
    state_lines = data.get("STATE", [])
    state_json = "\n".join(state_lines).strip()
    try:
        state = json.loads(state_json) if state_json else {}
    except json.JSONDecodeError:
        state = {}

    table = Table(title="Pipeline State", box=box.ROUNDED, border_style="blue", expand=True)
    table.add_column("Model", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Bar", ratio=1)

    total_done = 0
    for model_key in ["vnc_classifier", "alarm_detector", "os_classifier", "anomaly_ae", "cve_classifier"]:
        name = MODEL_NAMES.get(model_key, model_key)
        status = state.get(model_key, "pending")
        icon = MODEL_ICONS.get(status, "?")
        if status == "done":
            total_done += 1
            bar = "[green]████████████[/green]"
        elif status == "running":
            bar = "[yellow]█████░░░░░░░░[/yellow]"
        else:
            bar = "[dim]░░░░░░░░░░░░░░[/dim]"
        table.add_row(f"{icon} {name}", status, bar)

    progress_pct = total_done * 100 // 5
    table.add_row("", "", "")
    table.add_row("[bold]Overall[/bold]", f"{total_done}/5", f"[bold green]{progress_pct}%[/bold green]")

    layout["left"].update(table)

    # Right column: GPU + system info
    gpu_lines = data.get("GPU", [])
    gpu_text = ""
    if gpu_lines and gpu_lines[0] != "none":
        parts = [p.strip() for p in gpu_lines[0].split(",")]
        if len(parts) >= 4:
            util, mem_used, mem_total, temp = parts[:4]
            gpu_text = (
                f"[bold]GPU Utilization:[/bold]  {util}%\n"
                f"[bold]GPU Memory:[/bold]      {mem_used} / {mem_total} MiB\n"
                f"[bold]GPU Temp:[/bold]        {temp}°C\n"
            )
        else:
            gpu_text = "GPU data unavailable\n"
    else:
        gpu_text = "[dim]GPU not available[/dim]\n"

    tmux_lines = data.get("TMUX", [])
    tmux_status = tmux_lines[0] if tmux_lines else "unknown"
    tmux_text = f"[bold]tmux [train]:[/bold] {'🟢 running' if tmux_status == 'running' else '🔴 stopped'}\n"

    uptime_lines = data.get("UPTIME", [])
    uptime_text = f"[bold]Uptime:[/bold] {uptime_lines[0] if uptime_lines else 'unknown'}\n"

    disk_lines = data.get("DISK", [])
    disk_text = f"[bold]Disk /opt:[/bold] {disk_lines[0] if disk_lines else 'unknown'}\n"

    sys_panel = Panel(
        Text.assemble(
            (gpu_text, ""),
            ("\n", ""),
            (tmux_text, ""),
            ("\n", ""),
            (uptime_text, ""),
            ("\n", ""),
            (disk_text, ""),
        ),
        title="System",
        border_style="green",
        box=box.ROUNDED,
    )
    layout["right"].update(sys_panel)

    # Footer: recent log output
    log_lines = data.get("LOG", [])
    log_text = "\n".join(log_lines[-10:]) if log_lines else "[dim](no training logs yet)[/dim]"
    log_panel = Panel(
        Text(log_text, style="dim"),
        title="Recent Log Output",
        border_style="yellow",
        box=box.ROUNDED,
    )
    layout["footer"].update(log_panel)

    return layout


def run_tui(ip):
    """Run the live TUI dashboard."""
    console.print(f"[cyan]Starting TUI dashboard (Ctrl+C to exit, VM keeps running)...[/cyan]")
    console.print(f"[dim]Connect manually: ssh -i {SSH_KEY} {SSH_USER}@{ip}[/dim]\n")

    try:
        with Live(build_dashboard({}, ip), console=console, refresh_per_second=0.2, screen=True) as live:
            while True:
                try:
                    data = fetch_vm_state(ip)
                    live.update(build_dashboard(data, ip))
                except Exception:
                    live.update(build_dashboard({}, ip))
                time.sleep(5)
    except KeyboardInterrupt:
        console.print("\n[yellow]Exited monitor. VM is still running.[/yellow]")
        console.print(f"[dim]Reconnect: python3 launch.py --monitor[/dim]")
        console.print(f"[dim]SSH:       ssh -i {SSH_KEY} {SSH_USER}@{ip}[/dim]")


# --- Main ---

def main():
    check_deps()

    parser = argparse.ArgumentParser(description="VNC-T Pipeline Launcher")
    parser.add_argument("--monitor", "-m", action="store_true", help="Monitor existing VM")
    parser.add_argument("--status", "-s", action="store_true", help="Show VM status")
    parser.add_argument("--ssh", action="store_true", help="SSH into VM")
    parser.add_argument("--stop", action="store_true", help="Stop VM")
    parser.add_argument("--start", action="store_true", help="Start VM")
    parser.add_argument("--delete", "-d", action="store_true", help="Delete VM")
    parser.add_argument("--seed", action="store_true", help="Pre-seed repo to VM only")
    parser.add_argument("--create", "-c", action="store_true", help="Create VM only")
    args = parser.parse_args()

    if args.status:
        do_status()
    elif args.ssh:
        do_ssh()
    elif args.stop:
        do_stop()
    elif args.start:
        do_start()
    elif args.delete:
        do_delete()
    elif args.seed:
        ip = SSH_HOST or get_vm_ip()
        if not ip:
            console.print("[red]No VM found.[/red]")
            sys.exit(1)
        preseed_vm(ip)
    elif args.create:
        do_create()
    elif args.monitor:
        ip = SSH_HOST or get_vm_ip()
        if not ip:
            console.print("[red]No VM found or no public IP.[/red]")
            sys.exit(1)
        run_tui(ip)
    else:
        do_create_and_monitor()


if __name__ == "__main__":
    main()
