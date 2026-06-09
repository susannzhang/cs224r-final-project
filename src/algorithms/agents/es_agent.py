# =============================================================================
# ES AGENT: Phase 1 — Static Inverse Design via Evolution Strategies
# =============================================================================
"""
Phase 1 of the two-phase training curriculum solves the static inverse-design
problem independently for each of the 10 training angles {θ_j}_{j=1}^{10}
spanning [0°, 180°] via Evolution Strategies (ES) directly on the permittivity
vector ε ∈ [-1, 1]^{N_x × N_y}, without a parameterized policy.

Outputs (Phase 2 inputs):
- Memory bank M = {ε*(θ_j)}_{j=1}^{10} of converged configurations
- Replay buffer B with off-policy transitions (multi-goal labeled, 10× growth
  per FDFD solve), consumed by the critic in algorithms/critics/dqn_critic.py

Definitions:
============
- State (s): ε ∈ [-1, 1]^{N_x × N_y}, where ε_i is the normalized permittivity
             value of rod i.
- Action (a): Apply perturbation δ = σ ξ with ξ ~ N(0, I_{N_x × N_y}); resulting
              state a = clip(s + δ, [-1, 1]) keeps the permittivity in range.
- State Transition: s_{t+1} = a_t
- P_total(s, θ): P_source + P_θ(s) + Σ_{θ'≠ θ} P_{θ'}(s) + P_rods(s) + P_loss(s),
                 where:
                 →  P_source: Power from the source (fixed)
                 →  P_θ(s): Signal strength at target angle θ in state s
                 →  P_{θ'}(s): Signal strength at non-target angles θ' in state s
                 →  P_rods(s): Power absorbed by the rods in state s
                 →  P_loss(s): Power loss due to dispersion in free space in state s
- E_rods(s): Energy (V) applied to the rods in state s, computed using ε.
- Radiation pattern: P(s) ∈ R^{10}, the signal strength at all 10 receivers,
                     returned by a single FDFD solve.

Design Choices:
===============

1. REWARD FUNCTION
   - r(s, a ; θ) = P_θ(s) - λ_crosstalk(s) * Σ_{θ'≠ θ} P_{θ'}(s)
                  - λ_loss(s, a) * P_loss(s) - λ_energy(s, a) * E_rods(s)
   - Reason: Directly optimizes signal isolation at target angle; λ's give
             dynamic control over trade-offs between crosstalk, loss, and
             energy efficiency.
   - λ_crosstalk(s): 1 - (P_θ(s) / (P_θ(s) + Σ_{θ'≠ θ} P_{θ'}(s))) ;
                     penalizes for low signal-to-noise ratio at state s,
                     dynamically scaled by current state (larger penalty when
                     crosstalk is high).
   - λ_loss(s, a): ΔP_loss(s, a) = P_loss(a) - P_loss(s) ; weighting factor for
                   signal loss, dynamically adjusted based on change in signal
                   loss due to action a.
                   - If ΔP_loss > 0, then λ_loss is positive → penalize action
                   - If ΔP_loss < 0, then λ_loss is negative → reward action
   - λ_energy(s, a): ΔE_rods(s, a) = E_rods(a) - E_rods(s) ; weighting factor for
                     energy efficiency, dynamically adjusted based on change in
                     applied energy due to action a.
                     - If ΔE_rods > 0, then λ_energy is positive → penalize action
                     - If ΔE_rods < 0, then λ_energy is negative → reward action
   - Decomposition: P(s), P_loss, E_rods are goal-independent; only λ_crosstalk
     depends on θ. This enables multi-goal transition logging (see §5).

2. POPULATION SIZE (K = 200, with K/2 mirrored pairs)
   - Reason: Balances stable ES gradient estimates with reasonable FDFD-solve cost
   - Mirrored sampling (antithetic pairs ±ξ_k) halves gradient variance at
     fixed population size K; only K/2 noise vectors are drawn per iteration
   - 200 is optimal (** NEEDS EMPIRICAL VERIFICATION **) for 100-rod parameter space

3. PERTURBATION SCALE (σ)
   - σ is the standard deviation of ε-space Gaussian perturbations: ξ_k ~ N(0, I),
     δ_k = σ ξ_k
   - Larger σ explores broader regions; smaller σ refines locally
   - Constant within Phase 1 (no decay schedule)

4. ES GRADIENT (centered-rank fitness shaping)
   - Per-iteration fitness F_k = r(ε, δ_k ; θ_j) under current training angle θ_j
   - Centered ranks: u_k = rank(F_k)/K - 1/2 ∈ [-1/2, 1/2]
   - State update: ε ← clip(ε + (α_1 / (K σ)) Σ_k u_k ξ_k, [-1, 1])
   - Reason: Rank-based shaping is invariant to monotone reward transformations
     and robust to reward-scale drift across iterations
   - α_1 is the Phase 1 ES learning rate

5. MULTI-GOAL TRANSITION LOGGING (10× buffer growth, zero extra simulation)
   - Each FDFD solve produces P(ε) ∈ R^{10}, supporting reward computation
     under all 10 training angles simultaneously
   - Log 10 transitions per FDFD solve:
     {(ε, δ_k, r(ε, δ_k ; θ_{j'}), ε + δ_k, θ_{j'})}_{j'=1}^{10} → B
   - Reason: Provides natural coverage of cross-goal state-goal pairings, 
     enabled by the reward decomposition in §1—the same state evaluated under 
     every training angle — which would otherwise be absent (ES at iteration n 
     only optimizes one angle)

6. WARM-START + EARLY TERMINATION
   - Per training angle θ_j: initialize ε ~ U([-1, 1]^{N_x × N_y})
   - Retain best candidate between iterations; sample new perturbations around it
   - Early termination on objective satisfaction: P_θ(ε) ≥ 1 - η for small η
   - Reason: Preserves progress between iterations; saves compute on
     well-converged angles; mitigates local optima via independent runs
     across the 10 training angles

7. ITERATIONS & LOGGING
   - Total iterations per training angle: M = 1000 (chosen for Modal compute
     budget and design refinement)
   - Log every 10 iterations (rich dataset for downstream Phase 2 research)
   - Checkpoint per angle: metadata.json + best_rho.npy + best_eps_r.npy
                          + best_Ez.npy + visualization
   - Final output: memory bank M = {ε*(θ_j)}_{j=1}^{10} + replay buffer B
   - Critic checkpointing handled in algorithms/critics/dqn_critic.py

8. FREQUENCY (Fixed at 6 GHz)
   - Matches lab hardware and earlier analysis
   - Fixed for initial runs; can sweep later if broadband designs needed

9. PERMITTIVITY PARAMETERIZATION (ρ_i = normalized plasma frequency of rod i)
   - ρ_i ∈ [0, 1] maps to ω_{p,i} = ρ_i * OMEGA_P_MAX, with OMEGA_P_MAX = √2 * ω
   - Drude model: ε(ω_{p,i}) = 1 - (ω_{p,i} / ω_i)², where:
                  → ω_{p,i} is the plasma frequency of rod i
                  → ω_i is the applied frequency driving rod i
   - We use ε_i = ε(ω_{p,i}) as the state representation for rod i,
     which is a function of ρ_i. This is because the FDFD simulation (Ceviche)
     pipeline directly takes in ε for field solves.
   - We keep track of ρ_i because it is physically meaningful in mapping to
     hardware voltage controls, connecting designs to experimental implementation.

Critic pretraining (Q_ψ TD(0) updates, target network, bootstrap action
distribution) is documented in algorithms/critics/dqn_critic.py.
"""
