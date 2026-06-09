"""Re-FDFD unique Phase 1 states to recover P[30] (per-receiver power vector).

Phase 1's buffer logs only the scalar reward per (state, action, goal), and
the 11-unknown / 10-equation reward inversion is underdetermined. To use the
buffer for retarget-aligned reward modeling or for learning V(s, θ), we
need the full 30-receiver P vector per state. FDFD is deterministic, so we
just re-run it on the unique ε's the buffer references.

Runs on Modal in parallel — each container handles a batch of states.

Usage (one-shot, no deploy needed):
    modal run compute_P_per_state_sampled.py::main \\
        --buffer phase1-uniform-init-output/replay_buffer.pkl \\
        --out    phase1-uniform-init-output/mean_states_P.npz \\
        --states mean

    # Expensive version: re-FDFD all candidate next_states (~500k)
    modal run compute_P_per_state_sampled.py::main --states next

Outputs an .npz with {eps: (N, 10, 10), P: (N, 30)}.
"""

import os
import pickle
import time
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent           # dynamic_beam_steering/
PROJECT_ROOT = REPO_ROOT.parent                        # cs153 repo root (for geometry, simulation)
APP_NAME = os.environ.get("PHASE1_RELABEL_APP_NAME", "cs224r-phase1-relabel")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("gcc", "g++")
    .pip_install("numpy", "scipy", "scikit-image", "autograd", "ceviche")
    .add_local_dir(
        PROJECT_ROOT, "/root/app", copy=True,
        ignore=["__pycache__", "*.pyc", "phase1_checkpoints",
                "phase1_training_output", "checkpoint_output",
                "phase2_tiny_smoke", "phase2_output", "phase2_parallel_output",
                "tests/visual_output", ".git", ".venv", ".pytest_cache",
                "*.pkl", "wandb"],
    )
)

app = modal.App(APP_NAME)


def _build_pm_env():
    """Standard 10×10 pm_setup env, 30 receivers (3 walls × 10).
    Mirrors _build_pm_env in train_phase2_parallel_modal.py."""
    import sys
    sys.path.insert(0, "/root/app"); sys.path.insert(0, "/root/app/dynamic_beam_steering")
    from geometry import (create_design_region, create_environment,
                          create_grid, create_receiver, create_source)
    from simulation import initialize_environment
    region = create_design_region(resolution=0.0005, bg_permittivity=1.0,
                                  margin_cells=20)
    grid = create_grid(num_rods_x=10, num_rods_y=10,
                       radius=0.01, distance=0.002, rod_permittivity=1.0)
    source = create_source(index=1, frequency=6e9, length=0.02, walls=True)
    receivers = []
    for i in range(1, 11):
        receivers.append(create_receiver(index=i, length=0.02, side='bottom', rod_index=i))
    for i in range(1, 11):
        receivers.append(create_receiver(index=10 + i, length=0.02, side='right', rod_index=i))
    for i in range(1, 11):
        receivers.append(create_receiver(index=20 + i, length=0.02, side='top', rod_index=11 - i))
    env = create_environment(design_region=region, grid=grid,
                             sources=[source], receivers=receivers)
    initialize_environment(env)
    return env


@app.function(image=image, cpu=2, memory=4096, timeout=600)
def fdfd_batch(eps_batch_packed: bytes) -> bytes:
    """Run FDFD on a batch of ε's; return packed (P_batch, receiver_indices).

    P_batch:           (B, 30)
    receiver_indices:  (30,) of env.receivers[j].index — the "receiver number"
                        each column corresponds to.
    """
    import sys
    sys.path.insert(0, "/root/app"); sys.path.insert(0, "/root/app/dynamic_beam_steering")
    import numpy as np
    from algorithms.agents.es_agent import apply_eps_to_canvas
    from simulation import simulate_ez_fields_per_source

    eps_batch = pickle.loads(eps_batch_packed)
    env = _build_pm_env()

    receiver_indices = np.array([int(r.index) for r in env.receivers],
                                dtype=np.int64)

    P_list = []
    for eps in eps_batch:
        eps = np.asarray(eps, dtype=np.float32)
        apply_eps_to_canvas(env, eps)
        ez = sum(simulate_ez_fields_per_source(env).values())
        intensity = np.abs(ez) ** 2
        P = np.array([float(np.sum(intensity * r._mask))
                      for r in env.receivers], dtype=np.float64)
        P_list.append(P)

    return pickle.dumps((np.stack(P_list), receiver_indices))


def _extract_unique_states(buffer: list, which: str) -> list:
    """Return unique ε's from the buffer, deduped by raw bytes, preserving order.
    which='mean'  → buffer.state  (~2,500 unique)
    which='next'  → buffer.next_state  (~500,000 unique)
    """
    import numpy as np
    field = "state" if which == "mean" else "next_state"
    seen, ordered = set(), []
    for t in buffer:
        arr = getattr(t, field)
        h = arr.tobytes() if hasattr(arr, "tobytes") else bytes(arr)
        if h not in seen:
            seen.add(h)
            ordered.append(np.asarray(arr, dtype=np.float32))
    return ordered


@app.local_entrypoint()
def main(
    buffer: str = "phase1-uniform-init-output/replay_buffer.pkl",
    out: str = "phase1-uniform-init-output/mean_states_P.npz",
    states: str = "mean",   # "mean" | "next"
    batch_size: int = 25,
):
    """Dispatch parallel FDFD over unique buffer states; save (eps, P) npz."""
    import numpy as np

    if states not in ("mean", "next"):
        raise ValueError(f"--states must be 'mean' or 'next', got {states!r}")

    buf_path = Path(buffer)
    out_path = Path(out)
    if out_path.exists():
        print(f"WARN: {out_path} already exists; will overwrite on save.")

    print(f"Loading buffer: {buf_path}")
    with open(buf_path, "rb") as f:
        trs = pickle.load(f)
    print(f"  {len(trs):,} transitions")

    print(f"\nExtracting unique '{states}' states...")
    ordered = _extract_unique_states(trs, states)
    print(f"  {len(ordered):,} unique states (shape {ordered[0].shape})")

    batches = [ordered[i:i + batch_size] for i in range(0, len(ordered), batch_size)]
    print(f"\nDispatching {len(batches)} batches × ≤{batch_size} FDFDs each "
          f"({len(ordered):,} total FDFDs)")

    batches_packed = [pickle.dumps(b) for b in batches]

    t0 = time.time()
    P_chunks = []
    receiver_indices = None
    for i, chunk_bytes in enumerate(fdfd_batch.map(batches_packed)):
        P_chunk, rx_idx = pickle.loads(chunk_bytes)
        P_chunks.append(P_chunk)
        if receiver_indices is None:
            receiver_indices = rx_idx
        if (i + 1) % max(1, len(batches) // 20) == 0:
            elapsed = time.time() - t0
            print(f"  [{i + 1:>4}/{len(batches)}]  {elapsed:.1f}s elapsed")
    P_all = np.concatenate(P_chunks, axis=0)
    elapsed = time.time() - t0
    print(f"\nFinished in {elapsed:.1f}s  ({elapsed / 60:.1f} min)")
    print(f"  P shape: {P_all.shape}")
    print(f"  P stats: mean={P_all.mean():.2f}  max={P_all.max():.2f}  "
          f"min={P_all.min():.2f}")
    print(f"  receiver_indices: {receiver_indices.tolist()}")

    states_arr = np.stack(ordered)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
             eps=states_arr,
             P=P_all,
             receiver_indices=receiver_indices)
    print(f"\nSaved → {out_path.resolve()}")
    print(f"  eps:               {states_arr.shape}")
    print(f"  P:                 {P_all.shape}   (column j ↔ receiver_indices[j])")
    print(f"  receiver_indices:  {receiver_indices.shape}")
