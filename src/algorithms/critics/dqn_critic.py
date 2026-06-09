# =============================================================================
# DQN CRITIC: Goal-Conditioned Q-Function Q_ψ(ε, δ ; θ)
# =============================================================================
"""
Goal-conditioned Q-function trained via TD(0) on the shared replay
buffer B. Used as a fitness oracle for ES updates in Phase 2 (es_policy.py),
where Σ_t Q_ψ(ε^k_t, δ^k_t; θ_k) replaces Monte Carlo trajectory returns.

Trained across both phases of the curriculum:
- Phase 1 (es_agent.py): interleaved updates with ES-on-ε, bootstrap action
  drawn from the ES perturbation distribution (de facto behavior policy)
- Phase 2 (es_policy.py): interleaved updates with ES-on-φ, bootstrap action
  drawn from the current closed-loop policy π_φ

Architecture:
=============
- Input: (ε ∈ R^{N_x × N_y}, δ ∈ R^{N_x × N_y}, θ ∈ R) — current permittivity,
         proposed perturbation, target angle cue
- Output: Q_ψ(ε, δ ; θ) ∈ R — scalar value estimate
- Goal conditioning is per-transition (no marginalization over θ): minibatches
  preserve the (state, action, goal) triple as drawn from B
- Target network Q_{ψ̄} maintained for TD bootstrap stability

Design Choices:
===============

1. TD(0) LOSS
   - L(ψ) = E_{(ε, δ, r, ε', θ) ~ B} [(r + γ Q_{ψ̄}(ε', δ'; θ) - Q_ψ(ε, δ; θ))²]
   - Discount γ ∈ (0, 1) (e.g. 0.99)
   - Minibatches sampled uniformly from B
   - Reason: Standard off-policy value learning; one-step bootstrap balances
     variance (lower than Monte Carlo) against bias (controlled by Q_{ψ̄} lag)

2. BOOTSTRAP ACTION DISTRIBUTION (phase-dependent)
   - Phase 1: δ' ~ N(0, σ² I) — Phase 1's ES perturbation distribution serves
     as the implicit behavior policy (no parameterized actor exists yet)
   - Phase 2: δ' ~ π_φ(· | ε', θ) — current closed-loop policy bootstraps
     against the on-policy action distribution
   - Reason: Gradually retargets Q_ψ from "value of random ES search" (Phase 1)
     to "value of policy execution" (Phase 2), tracking the action distribution
     that will actually be deployed at inference

3. TARGET NETWORK (Polyak averaging)
   - ψ̄ ← τ ψ̄ + (1 - τ) ψ after each iteration, τ ≈ 0.995
   - Reason: Soft updates stabilize the TD target across noisy ES iterations;
     hard target swaps would interact badly with the rank-shaped ES gradient

4. UPDATE SCHEDULE (interleaved with ES iterations)
   - Per ES iteration, perform G TD(0) gradient steps (e.g. G = 20)
   - Reason: Amortizes the cost of expensive FDFD rollouts across multiple
     critic updates; without this, each rollout's transitions would be visited
     only once before being diluted by subsequent population samples

5. REPLAY BUFFER (B, shared with agents)
   - Populated by both Phase 1 (es_agent.py) and Phase 2 (es_policy.py)
     rollouts via multi-goal transition logging
   - Each FDFD solve writes 10 transitions to B (one per training angle θ_{j'}),
     yielding broad cross-goal coverage that the critic relies on for
     goal-conditioned generalization
   - See es_agent.py and es_policy.py for the rollout-collection pipeline; 

6. INITIALIZATION (Phase 1) vs. CONTINUATION (Phase 2)
   - Phase 1 starts ψ from random init alongside the very first ES iteration
   - Phase 2 inherits (ψ, ψ̄) from Phase 1 — no re-initialization
   - By end of Phase 1, Q_ψ is well-calibrated as a fitness oracle on the
     joint state-goal distribution that Phase 2 will query

Deployment:
===========
- Q_ψ is a training-time auxiliary only; discarded at inference
- The deployed system uses only π_φ from es_policy.py for retargeting
"""
