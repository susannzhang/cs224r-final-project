"""Poll the Modal volume for new Phase 2 checkpoints and auto-render PNGs.

Each time a new checkpoint lands (history.json's iter_at_save advances),
download the latest policy_phase2.pt and call render_phase2_ckpt.py,
overwriting the PNGs under checkpoint_output/phase2/<launch_id>/.

Usage:
    python watch_phase2_renders.py \\
        --launch-id phase2-retarget-... \\
        --memory-bank phase1-uniform-init-output \\
        --poll-seconds 60
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

VOLUME_NAME = os.environ.get(
    "PHASE2_VOLUME_NAME", "cs224r-phase2-parallel-buffer")


def parse_args():
    p = argparse.ArgumentParser(description="Auto-render Phase 2 checkpoints")
    p.add_argument("--launch-id", required=True, type=str)
    p.add_argument("--memory-bank", required=True, type=Path)
    p.add_argument("--poll-seconds", type=int, default=60)
    p.add_argument("--out-root", type=Path,
                   default=Path("checkpoint_output/phase2"))
    p.add_argument("--T", type=int, default=20,
                   help="Rollout steps for render (match the training T).")
    return p.parse_args()


def fetch_remote_iter(launch_id: str) -> int | None:
    """Download history.json from Volume and return its iter_at_save.
    None if file doesn't exist yet."""
    tmp = Path(f"/tmp/{launch_id}_history.json")
    r = subprocess.run(
        ["modal", "volume", "get", VOLUME_NAME,
         f"{launch_id}/history.json", str(tmp), "--force"],
        capture_output=True,
    )
    if r.returncode != 0:
        return None
    try:
        data = json.loads(tmp.read_text())
        return int(data.get("iter_at_save", -1))
    except Exception:
        return None


def render_checkpoint(launch_id: str, memory_bank: Path,
                      out_root: Path, T: int) -> Path:
    """Pull latest policy_phase2.pt + overwrite PNGs in out_root/launch_id/."""
    out_dir = out_root / launch_id
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        sys.executable, "render_phase2_ckpt.py",
        "--launch-id", launch_id,
        "--memory-bank", str(memory_bank),
        "--out-dir", str(out_dir),
        "--T", str(T),
    ], check=True)
    return out_dir


def main():
    args = parse_args()
    print(f"[watcher] launch_id={args.launch_id}  "
          f"memory_bank={args.memory_bank}  "
          f"poll={args.poll_seconds}s")
    print(f"[watcher] output → {args.out_root / args.launch_id}/  (overwritten each checkpoint)")

    last_iter = -1
    while True:
        cur = fetch_remote_iter(args.launch_id)
        ts = time.strftime("%H:%M:%S")
        if cur is None:
            print(f"[{ts}] no history.json yet; waiting for first checkpoint")
        elif cur > last_iter:
            print(f"[{ts}] new checkpoint at iter {cur} — rendering")
            try:
                out_dir = render_checkpoint(
                    args.launch_id, args.memory_bank, args.out_root, args.T)
                print(f"          → {out_dir}/  (PNGs refreshed)")
                last_iter = cur
            except subprocess.CalledProcessError as e:
                print(f"          ! render failed: {e}")
        else:
            print(f"[{ts}] no new checkpoint (last seen iter={last_iter})")

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
