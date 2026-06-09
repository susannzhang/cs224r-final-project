# Dynamic Beam Steering

End-to-end Phase 1 + Phase 2 pipeline for the dynamic beam-steering task:
given a configurable rod array driven by FDFD-simulated EM fields, learn
permittivity configurations ε that concentrate transmitted power at any
target receiver angle θ.

Generic EM/FDFD infrastructure (`geometry.py`, `simulation.py`,
`visualization.py`, `converter.py`, `constants.py`, `pm_setup.py`) lives at
the project root and is imported from here via the parent-on-`sys.path`
shim each script applies on startup.

## Layout

```
dynamic_beam_steering/
├── algorithms/                          # core algorithms (Phase 1 + Phase 2)
│   ├── agents/es_agent.py               # Phase 1 ES on ε (static inverse design)
│   ├── critics/dqn_critic.py            # parked DQN critic (TD trainer in train_critic.py)
│   ├── policies/es_policy.py            # parameter-space ES on π_φ (deprecated branch)
│   ├── policies/es_state_space_policy.py  # actor-critic ε-search with M surrogate filter
│   └── infrastructure/utils.py          # ReplayBuffer, Transition
│
├── train_phase1.py                      # Phase 1 local driver
├── train_phase1_modal.py                # Phase 1 on Modal
├── train_phase1_reinit_modal.py         # Phase 1 from far-anchor warm-starts
├── train_phase2_modal.py                # Phase 2 closed-loop (deprecated)
├── train_phase2_parallel_modal.py       # Phase 2 K-parallel rollouts on Modal
├── train_phase2_state_space_modal.py    # ★ Phase 2 actor-critic on Modal (the working pipeline)
├── train_phase2_anchor_traj.py          # anchor-pair trajectory training experiment
├── train_phase2_bptt.py                 # BPTT-through-M experiment (failure)
├── train_phase2_distill_closed_loop.py  # distill trajectories into a sub-ms π_φ
├── train_V_network.py                   # train M_ψ(ε) → P[30] surrogate (the critic)
├── train_awr.py                         # AWR pretrain π_φ from Phase 1 buffer
├── train_critic.py                      # post-hoc TD(0) trainer for the DQN critic
│
├── spawn_phase1.py / spawn_phase2*.py   # Modal spawn helpers
│
├── deploy_phase2_distilled.py           # per-step FDFD eval of distilled π_φ
├── deploy_phase2_online.py              # M-only inner-loop ES at deploy time
├── deploy_phase2_fdfd_inner.py          # FDFD-inner-loop ES for the hardest goals
│
├── pretrain_phase2_from_buffer.py       # one-shot AWR pretrain
├── pretrain_phase2_policy.py            # imitation pretrain
│
├── calibrate_M_surrogate.py             # rollout-state argmax + action-ranking ρ for M
├── compute_P_per_state_sampled.py       # compute P[30] for sampled buffer states
├── reconstruct_reinit_trajectories.py   # convert reinit eps_buffer → per-iter eps_traj
├── recover_phase1_reinit.py             # rehydrate Phase 1 reinit runs from Modal Volume
├── analyze_iter4_endstate.py            # inspect a failed Phase 2 end-state
├── test_interpolation_baseline.py       # baseline: linear interp between anchors
│
├── render_*.py                          # 2x2 before/after PNG renderers
├── watch_phase2_renders.py              # live render watcher
│
├── pretrain/                            # trained M and π checkpoints (M_fdfd_surrogate_v{1..4}.pt,
│                                          policy_distilled_v2.pt, V_retarget.pt, etc.)
├── phase1-uniform-init-output/          # Phase 1 memory bank (ε* per anchor + viz)
├── phase1-reinit-output/                # Phase 1 reinit (far-anchor warm-start) per-goal results
├── phase1-reinit-anneal-output/         # σ-annealed reinit results
├── phase2_state_space_output/           # ★ Phase 2 results (state-space ES per launch_id)
├── distill_inputs/                      # reconstructed per-iter ε trajectories for distillation
training Algorithm 2
│   ├── phase2_results.tex               # 0.647 mean target_frac result
│   └── phase2_negative_result.tex       # original closed-loop ES failure mode
│
└── tests/                               # beam-steering-specific tests
    ├── test_es_agent.py
    ├── test_es_policy.py
    ├── test_dqn_critic.py
    └── test_phase2_tiny.py
```

## Running scripts

All scripts inject the project root + this directory into `sys.path` on
startup, so they work from any CWD:

```bash
# from the project root
python dynamic_beam_steering/train_phase2_state_space_modal.py ...

# or from inside the subdir
cd dynamic_beam_steering
python train_phase2_state_space_modal.py ...
```

For Modal scripts the container mounts the **project root** at `/root/app`,
and the container-side `sys.path` is set to include both `/root/app` (for
`geometry.py`, `simulation.py`) and `/root/app/dynamic_beam_steering` (for
`from algorithms... import ...`). No changes needed at the call site.

## Tests

```bash
pytest dynamic_beam_steering/tests/           # beam-steering tests
pytest tests/                                 # generic EM/FDFD tests
pytest                                        # all of them
```

A `conftest.py` in this directory adds the directory itself to `sys.path`
during test collection so `from algorithms... import ...` resolves from
test files.

## Headline result

`train_phase2_state_space_modal.py` (actor-critic state-space ES with M
surrogate filter + DAGGER refresh) achieves **mean target_frac 0.647** on
the 10 heldout goals `{1, 4, 7, 10, 13, 16, 19, 22, 25, 28}`, a 3.5× lift
over the prior best deployment result (anchored online ES with M v1, 0.185)
and 4.5× over linear interpolation (0.143). See
[`writeup/phase2_results.tex`](writeup/phase2_results.tex) for the per-goal
table and [`writeup/phase2_pipeline.tex`](writeup/phase2_pipeline.tex) for
the algorithm.
