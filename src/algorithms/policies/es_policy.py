# =============================================================================
# ES POLICY: Closed-Loop Retargeting Policy Training
# =============================================================================
"""
Phase 2 of the two-phase training curriculum trains a closed-loop, goal-conditioned
retargeting policy π_φ(δ | ε, θ_target) via Evolution Strategies (ES) on the
policy parameters φ ∈ R^d (d ≪ N_x · N_y), using the pretrained critic Q_ψ from
Phase 1 as a fitness oracle.

Inputs (from Phase 1):
- Pretrained critic Q_ψ(ε, δ; θ) and target Q_{ψ̄}
- Replay buffer B with multi-goal-labeled transitions
- Memory bank M = {ε*(θ_j)}_{j=1}^{10} of converged static configurations

Output:
- Trained policy π_φ that maps (ε, θ_target) → δ in a single forward pass,
  enabling sub-millisecond retargeting at inference (only π_φ is deployed;
  Q_ψ and M are training-time auxiliaries)

Definitions:
============
- State (s): ε ∈ [-1, 1]^{N_x × N_y}, current normalized permittivity vector.
- Goal cue (θ_target): target steering angle, sampled from the 10 training
                       angles {θ^train_j}_{j=1}^{10}.
- Action (a): a = clip(s + δ, [-1, 1]), where δ = π_{φ+σε_k}(s; θ_target) is the
              policy's bounded perturbation under ES-perturbed parameters.
- State Transition: s_{t+1} = a_t, realized by FDFD on a_t.
- Trajectory: τ_k = {(ε^k_t, δ^k_t, r^k_t, ε^k_{t+1})}_{t=0}^{T_k}, terminated
              on objective satisfaction or step budget T.
- Radiation pattern: P(s) ∈ R^{10}, signal strength at all 10 receivers, from
                     a single FDFD solve (reused for multi-goal logging, §8).

Reward function (same as Phase 1, §1 of es_agent.py):
- r(s, a ; θ) = P_θ(s) - λ_crosstalk(s) * Σ_{θ'≠ θ} P_{θ'}(s)
               - λ_loss(s, a) * P_loss(s) - λ_energy(s, a) * E_rods(s)
- Goal-independent components (P, P_loss, E_rods) enable multi-goal logging.

Design Choices:
===============

1. POLICY ARCHITECTURE (π_φ neural network)
   - Input: (ε ∈ R^{N_x × N_y}, θ_target ∈ R) — flattened permittivity + scalar
            angle cue (or one-hot over 10 training angles)
   - Output: δ ∈ R^{N_x × N_y}, perturbation in ε-space
   - Output activation: tanh-scaled to enforce δ ∈ [-1 - ε, 1 - ε] element-wise
     (combined with downstream clip to [-1, 1])
   - Reason: d ≪ N_x · N_y so ES in parameter space is far cheaper than ES
     in state space; goal conditioning enables generalization across angles

2. BEHAVIOR CLONING INITIALIZATION
   - φ ← argmin_φ E_{(ε, δ, θ) ~ B} ||π_φ(ε; θ) - δ||²
   - Supervised regression of Phase 1 ES perturbations on (ε, θ) pairs
   - Reason: ES from random init is sample-inefficient in policy space; BC on
     Phase 1's converged trajectories gives a high-quality starting policy
     that already encodes single-target inverse-design knowledge

3. POPULATION SIZE (K = 100, with K/2 mirrored pairs)
   - Reason: Same trade-off as Phase 1 — balances ES gradient stability with
     FDFD-rollout cost, which dominates compute (each rollout is up to T solves)
   - Mirrored sampling (antithetic pairs ±ε_k) halves gradient variance

4. PERTURBATION SCALE (σ in policy-parameter space)
   - σ is the standard deviation of policy-parameter Gaussian perturbations:
     ε_k ~ N(0, I_d), perturbed weights φ + σ ε_k
   - Note: this σ lives in R^d (policy space), distinct from Phase 1's σ in
     R^{N_x × N_y} (state space)
   - Tuned per problem; constant within Phase 2 (no decay schedule)

5. ROLLOUT COLLECTION (mixed cold-start + retargeting)
   - Per population member k, sample target θ_k ~ Uniform{θ^train_1, ..., θ^train_10}
   - Initial state mixture:
     ε^k_0 ~ p_rand * U([-1, 1]^{N_x × N_y}) + (1 - p_rand) * δ_{ε*(θ^k_prev)}
     where θ^k_prev ~ Uniform{θ^train_j : θ^train_j ≠ θ_k}
   - Reason: p_rand fraction trains cold-start dynamics; the rest trains
     retargeting dynamics (start at the optimum for some other angle, learn
     to reroute toward θ_k) — directly targets the deployment regime
   - Roll out π_{φ+σε_k}(·; θ_k) for at most T steps, terminating on
     objective satisfaction P_θ(ε) ≥ 1 - η

6. CRITIC-BASED FITNESS SCORING (F_k = Σ_t Q_ψ)
   - Trajectory fitness: F_k = Σ_{t=0}^{T_k} Q_ψ(ε^k_t, δ^k_t ; θ_k)
   - Replaces vanilla ES's Monte Carlo return F^MC_k = Σ_t r^k_t
   - Reason: Q_ψ aggregates information across the entire buffer (other
     population members, prior iterations, cross-goal transitions), yielding
     substantially lower variance than on-policy return — especially valuable
     in the sparse-reward regime where P_θ rises sharply only near terminal states

7. ES GRADIENT ON POLICY PARAMETERS (centered-rank fitness shaping)
   - Centered ranks: u_k = rank(F_k)/K - 1/2 ∈ [-1/2, 1/2]
   - Policy update: φ ← φ + (α_2 / (K σ)) Σ_k u_k ε_k
   - Reason: Rank shaping is invariant to monotone transformations of Q_ψ's
     output, stabilizing training as the critic's value scale drifts across
     iterations
   - α_2 is the Phase 2 ES learning rate (tuned per problem; typically smaller
     than α_1 since policy-space updates compound across rollout steps)

8. MULTI-GOAL TRANSITION LOGGING (continued from Phase 1)
   - Per rollout step: FDFD solve yields P^k_t ∈ R^{10}
   - Log 10 transitions per step, one per training angle:
     {(ε^k_t, δ^k_t, r(ε^k_t, δ^k_t ; θ^train_{j'}), ε^k_{t+1}, θ^train_{j'})}_{j'=1}^{10} → B
   - Episode reward uses rollout's actual goal: r^k_t = r^{k,(k)}_t
   - Reason: Continues Phase 1's pre-emptive goal relabeling — buffer grows
     10× faster than FDFD calls; critic continues to see broad cross-goal coverage

9. EPISODE TERMINATION
   - Objective satisfaction: P_θ_k(ε^k_t) ≥ 1 - η for small η
   - Step budget: T_k ≤ T (e.g. T = 50)
   - Reason: Caps FDFD cost per rollout; encourages the policy to converge
     quickly (short trajectories ⇒ higher cumulative Q_ψ via fewer discount steps)

10. ITERATIONS & LOGGING
    - Total iterations: N_iter (tuned to compute budget; e.g. 500)
    - Log every 10 iterations: mean / max fitness, mean rollout length,
      retargeting success rate on held-out (θ_prev, θ_target) pairs
    - Checkpoint: policy_φ.pt + metadata.json + retargeting trajectory
                  visualizations
    - Critic checkpointing handled in algorithms/critics/dqn_critic.py

11. DEPLOYMENT
    - At inference, only π_φ is required: ε_{t+1} = clip(ε_t + π_φ(ε_t; θ_target), [-1, 1])
    - Critic Q_ψ and memory bank M are discarded
    - Retargeting latency dominated by policy forward-pass cost — sub-millisecond,
      orders of magnitude faster than Phase 1 per-target inverse design (≈ 1000
      FDFD solves per angle)
    - On hardware: δ_t applied as DC bias updates to the rod array via the
      inverse Drude map ε_i → ρ_i → voltage

Critic training (Q_ψ TD(0) updates, target network, policy-bootstrap action
distribution) is documented in algorithms/critics/dqn_critic.py.
"""
