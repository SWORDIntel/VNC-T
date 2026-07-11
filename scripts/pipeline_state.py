#!/usr/bin/env python3
"""Pipeline state tracker for preemptible VM resume.

Tracks which models have completed training in pipeline_state.json.
Called by run.sh to check/mark model status.

Usage:
  python3 scripts/pipeline_state.py check <model>   # exit 0 if done, 1 if not
  python3 scripts/pipeline_state.py mark <model> <status>  # set status
  python3 scripts/pipeline_state.py show             # print state
  python3 scripts/pipeline_state.py reset            # reset all to pending
"""
import json
import sys
from pathlib import Path

MODELS = ["vnc_classifier", "alarm_detector", "os_classifier", "anomaly_ae"]
STATE_FILE = Path(__file__).parent.parent / "pipeline_state.json"


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {m: "pending" for m in MODELS}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def main():
    if len(sys.argv) < 2:
        print("Usage: pipeline_state.py [check|mark|show|reset] ...")
        sys.exit(2)

    cmd = sys.argv[1]

    if cmd == "check":
        model = sys.argv[2]
        state = load_state()
        sys.exit(0 if state.get(model) == "done" else 1)

    elif cmd == "mark":
        model = sys.argv[2]
        status = sys.argv[3]  # "running", "done", "pending"
        state = load_state()
        state[model] = status
        save_state(state)
        print(f"[pipeline] {model} -> {status}")

    elif cmd == "show":
        state = load_state()
        print(json.dumps(state, indent=2))

    elif cmd == "reset":
        save_state({m: "pending" for m in MODELS})
        print("[pipeline] Reset all models to pending")

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(2)


if __name__ == "__main__":
    main()
