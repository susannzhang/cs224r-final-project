"""
Tests for the Phase 1 ES agent.

Organization:
- Pure-logic tests (no FDFD): config validation, reward math, ranks, helpers.
- Integration tests (real FDFD on a TINY env): canvas stamping, multi-goal
  logging, end-to-end training loop. These deliberately use a 3x3 grid and
  K=4, M=2 so the whole file runs in seconds.

Run:
    pytest -q tests/test_es_agent.py
    pytest -q tests/test_es_agent.py -m "not slow"   # skip integration tests
"""

import numpy as np
import pytest

from geometry import (create_design_region, create_grid, create_source,
                      create_receiver, create_environment)
from simulation import initialize_environment

from algorithms.agents.es_agent import (
    ESAgent, ESAgentConfig, TrainingResult,
    apply_eps_to_canvas, get_receiver_powers, get_reward, centered_ranks,
    elite_centered_ranks,
)
from algorithms.infrastructure.utils import ReplayBuffer


# =============================================================================
# Fixtures
# =============================================================================

def _build_tiny_env():
    """A deliberately small env so FDFD finishes in <1s per solve.

    3x3 grid, coarse resolution, 3 receivers (one per side except left).
    """
    region = create_design_region(resolution=0.005, bg_permittivity=1.0,
                                  margin_cells=10)
    grid = create_grid(num_rods_x=3, num_rods_y=3,
                       radius=0.01, distance=0.005, rod_permittivity=1.0)
    source = create_source(index=1, frequency=6e9, length=0.02, walls=True)
    receivers = [
        create_receiver(index=1, length=0.01, side='bottom', rod_index=2),
        create_receiver(index=2, length=0.01, side='right',  rod_index=2),
        create_receiver(index=3, length=0.01, side='top',    rod_index=2),
    ]
    env = create_environment(design_region=region, grid=grid,
                             sources=[source], receivers=receivers)
    initialize_environment(env)
    return env


@pytest.fixture
def tiny_env():
    return _build_tiny_env()


@pytest.fixture
def tiny_agent(tiny_env):
    cfg = ESAgentConfig(K=4, sigma=0.1, alpha_1=0.02, M=2,
                        eta=1e-2, log_every=1, seed=0)
    return ESAgent(env=tiny_env, training_indices=[0, 1, 2],
                   config=cfg, verbose=False)


# =============================================================================
# Pure-logic tests
# =============================================================================

class TestConfig:
    def test_defaults_match_design_doc(self):
        cfg = ESAgentConfig()
        assert cfg.K == 100
        assert cfg.M == 1000
        assert cfg.sigma == 0.1
        assert cfg.alpha_1 == 0.02
        assert cfg.eta == 1e-2

    def test_rejects_odd_population(self):
        with pytest.raises(ValueError, match="even"):
            ESAgentConfig(K=7)

    def test_even_population_ok(self):
        ESAgentConfig(K=2)    # smallest even
        ESAgentConfig(K=200)


class TestCenteredRanks:
    def test_range_for_arbitrary_K(self):
        for K in [2, 4, 10, 100]:
            u = centered_ranks(np.random.default_rng(0).standard_normal(K))
            assert u.min() == pytest.approx(-0.5)
            assert u.max() == pytest.approx((K - 1) / K - 0.5)

    def test_mean_is_minus_one_over_two_K(self):
        # Sum of ranks 0..K-1 is K*(K-1)/2 → mean of u = (K-1)/(2K) - 1/2 = -1/(2K)
        for K in [4, 10, 100]:
            u = centered_ranks(np.random.default_rng(0).standard_normal(K))
            assert u.mean() == pytest.approx(-1.0 / (2 * K))

    def test_largest_fitness_gets_largest_weight(self):
        F = np.array([0.1, 0.5, 0.3, 0.9, 0.2])
        u = centered_ranks(F)
        assert int(np.argmax(F)) == int(np.argmax(u))

    def test_monotone_invariant(self):
        # Rank shaping is invariant to monotone transformations of fitness.
        F = np.array([0.1, 0.5, 0.3, 0.9, 0.2])
        u_raw = centered_ranks(F)
        u_log = centered_ranks(np.log1p(F))
        u_squared = centered_ranks(F ** 2)
        np.testing.assert_array_equal(u_raw, u_log)
        np.testing.assert_array_equal(u_raw, u_squared)


class TestEliteCenteredRanks:
    def test_only_top_k_get_nonzero_weight(self):
        # Top 3 of F = [0.1, 0.5, 0.3, 0.9, 0.2] are indices {1, 2, 3}.
        F = np.array([0.1, 0.5, 0.3, 0.9, 0.2])
        u = elite_centered_ranks(F, k_elite=3)
        assert np.count_nonzero(u) == 3
        assert set(np.flatnonzero(u).tolist()) == {1, 2, 3}
        assert u[0] == 0.0 and u[4] == 0.0

    def test_internal_centered_rank_structure(self):
        # Within the top-k elites, weights span [-1/2, (k-1)/k - 1/2].
        F = np.array([0.1, 0.5, 0.3, 0.9, 0.2])
        k = 3
        u = elite_centered_ranks(F, k_elite=k)
        nonzero = u[u != 0]
        assert nonzero.min() == pytest.approx(-0.5)
        assert nonzero.max() == pytest.approx((k - 1) / k - 0.5)
        # Best of the elites (= overall argmax) keeps the largest weight.
        assert int(np.argmax(u)) == int(np.argmax(F))

    def test_k_elite_equal_K_matches_centered_ranks(self):
        # No truncation → should match the full centered-rank weights exactly.
        F = np.random.default_rng(0).standard_normal(8)
        np.testing.assert_array_equal(
            elite_centered_ranks(F, k_elite=8),
            centered_ranks(F),
        )


class TestReward:
    def test_high_target_low_others_positive(self):
        P = np.array([10.0, 1.0, 1.0, 1.0])
        r = get_reward(P, target_idx=0, training_indices=[0, 1, 2, 3])
        assert r > 0

    def test_low_target_high_others_negative(self):
        P = np.array([1.0, 10.0, 10.0, 10.0])
        r = get_reward(P, target_idx=0, training_indices=[0, 1, 2, 3])
        assert r < 0

    def test_lambda_crosstalk_is_one_when_target_zero(self):
        # When P_target is 0, λ_crosstalk = 1 - 0/(0+2) = 1, so r = 0 - 1*2 = -2.
        P = np.array([0.0, 1.0, 1.0])
        r = get_reward(P, target_idx=0, training_indices=[0, 1, 2])
        assert r == pytest.approx(-2.0)

    def test_all_zero_power_does_not_divide_by_zero(self):
        # No-throw is the bar; specific value should be 0 - 1*0 = 0.
        P = np.zeros(3)
        r = get_reward(P, target_idx=0, training_indices=[0, 1, 2])
        assert r == pytest.approx(0.0)

    def test_singleton_training_set(self):
        # Only the target is a training receiver; others=0, crosstalk=0.
        P = np.array([5.0, 99.0, 99.0])
        r = get_reward(P, target_idx=0, training_indices=[0])
        assert r == pytest.approx(5.0)

    def test_loss_and_energy_penalize(self):
        # λ_loss and λ_energy now subtract directly (no extra multiplier).
        P = np.array([10.0, 1.0, 1.0])
        r_no_pen = get_reward(P, 0, [0, 1, 2])
        r_pen = get_reward(P, 0, [0, 1, 2], lambda_loss=2.0, lambda_energy=3.0)
        assert r_pen == pytest.approx(r_no_pen - 2.0 - 3.0)

    def test_negative_lambdas_reward(self):
        # ΔP_loss < 0 (action reduced loss) should INCREASE reward.
        P = np.array([10.0, 1.0, 1.0])
        r_no_pen = get_reward(P, 0, [0, 1, 2])
        r_bonus = get_reward(P, 0, [0, 1, 2], lambda_loss=-1.0, lambda_energy=-2.0)
        assert r_bonus == pytest.approx(r_no_pen + 1.0 + 2.0)

    def test_static_weights_scale_each_term(self):
        # Verify each w_* scales its associated term linearly.
        P = np.array([10.0, 5.0, 5.0])  # P_target=10, P_others=10
        # baseline reward, default weights all 1.0
        r0 = get_reward(P, 0, [0, 1, 2], lambda_loss=2.0, lambda_energy=3.0)

        # zero the crosstalk weight → reward gains back the crosstalk term
        r_no_xtalk = get_reward(P, 0, [0, 1, 2], lambda_loss=2.0, lambda_energy=3.0,
                                w_crosstalk=0.0)
        # λ_crosstalk = 1 - 10/20 = 0.5; contribution = 0.5 * 10 = 5
        assert r_no_xtalk == pytest.approx(r0 + 5.0)

        # zero the loss weight → reward gains back lambda_loss = 2
        r_no_loss = get_reward(P, 0, [0, 1, 2], lambda_loss=2.0, lambda_energy=3.0,
                               w_loss=0.0)
        assert r_no_loss == pytest.approx(r0 + 2.0)

        # zero the energy weight → reward gains back lambda_energy = 3
        r_no_energy = get_reward(P, 0, [0, 1, 2], lambda_loss=2.0, lambda_energy=3.0,
                                 w_energy=0.0)
        assert r_no_energy == pytest.approx(r0 + 3.0)


# =============================================================================
# Integration tests — real FDFD on a tiny env
# =============================================================================

@pytest.mark.slow
class TestApplyEpsToCanvas:
    def test_rod_cells_take_new_permittivity(self, tiny_env):
        eps = np.full((tiny_env.grid.num_rods_x, tiny_env.grid.num_rods_y), 4.2)
        apply_eps_to_canvas(tiny_env, eps)
        for rod in tiny_env.grid.rods.values():
            assert rod.permittivity == pytest.approx(4.2)
            r, c = rod._center
            assert tiny_env.design_region._canvas[r, c] == pytest.approx(4.2)

    def test_walls_unaffected(self, tiny_env):
        wall_mask_before = (tiny_env.design_region._canvas ==
                            tiny_env.design_region.plate_permittivity)
        eps = np.full((tiny_env.grid.num_rods_x, tiny_env.grid.num_rods_y), 0.5)
        apply_eps_to_canvas(tiny_env, eps)
        wall_mask_after = (tiny_env.design_region._canvas ==
                           tiny_env.design_region.plate_permittivity)
        np.testing.assert_array_equal(wall_mask_before, wall_mask_after)

    def test_indexing_uses_grid_x_grid_y(self, tiny_env):
        N_x, N_y = tiny_env.grid.num_rods_x, tiny_env.grid.num_rods_y
        eps = np.arange(N_x * N_y, dtype=float).reshape(N_x, N_y) / 10.0
        apply_eps_to_canvas(tiny_env, eps)
        for (x, y), rod in tiny_env.grid.rods.items():
            assert rod.permittivity == pytest.approx(eps[x - 1, y - 1])


@pytest.mark.slow
class TestReceiverPowers:
    def test_returns_one_value_per_receiver(self, tiny_env):
        P = get_receiver_powers(tiny_env)
        assert P.shape == (len(tiny_env.receivers),)

    def test_values_are_nonnegative_finite(self, tiny_env):
        P = get_receiver_powers(tiny_env)
        assert np.all(P >= 0)
        assert np.all(np.isfinite(P))


@pytest.mark.slow
class TestESAgentInit:
    def test_rejects_out_of_range_training_index(self, tiny_env):
        with pytest.raises(IndexError):
            ESAgent(env=tiny_env, training_indices=[999],
                    config=ESAgentConfig(K=2, M=1))

    def test_constructs_default_buffer_when_none(self, tiny_env):
        agent = ESAgent(env=tiny_env, training_indices=[0],
                        config=ESAgentConfig(K=2, M=1), verbose=False)
        assert isinstance(agent.buffer, ReplayBuffer)
        assert len(agent.buffer) == 0

    def test_uses_provided_buffer(self, tiny_env):
        b = ReplayBuffer()
        agent = ESAgent(env=tiny_env, training_indices=[0],
                        config=ESAgentConfig(K=2, M=1), buffer=b, verbose=False)
        assert agent.buffer is b


@pytest.mark.slow
class TestTrainOneAngle:
    def test_returns_training_result_with_correct_shape(self, tiny_agent):
        result = tiny_agent.train_one_angle(target_idx=0)
        assert isinstance(result, TrainingResult)
        assert result.eps_star.shape == (tiny_agent.env.grid.num_rods_x,
                                         tiny_agent.env.grid.num_rods_y)
        assert np.all(result.eps_star >= -1.0)
        assert np.all(result.eps_star <= 1.0)

    def test_iterations_within_budget(self, tiny_agent):
        result = tiny_agent.train_one_angle(target_idx=0)
        assert 1 <= result.iterations <= tiny_agent.config.M

    def test_history_logged_at_log_every_interval(self, tiny_agent):
        result = tiny_agent.train_one_angle(target_idx=0)
        # log_every=1, M=2 → expect ≤ 2 entries (fewer if converged early)
        assert 1 <= len(result.history) <= tiny_agent.config.M
        for entry in result.history:
            assert 'iteration' in entry
            assert 'pop_reward_mean' in entry
            assert 'best_ever_target_fraction' in entry

    def test_rejects_target_not_in_training(self, tiny_agent):
        with pytest.raises(ValueError, match="training_indices"):
            tiny_agent.train_one_angle(target_idx=999)


@pytest.mark.slow
class TestMultiGoalLogging:
    def test_buffer_size_equals_K_times_training_per_iter(self, tiny_env):
        cfg = ESAgentConfig(K=4, sigma=0.1, alpha_1=0.02, M=1,
                            eta=-1.0,  # never converge
                            log_every=1, seed=0)
        agent = ESAgent(env=tiny_env, training_indices=[0, 1, 2],
                        config=cfg, verbose=False)
        agent.train_one_angle(target_idx=0)
        assert len(agent.buffer) == cfg.K * len(agent.training_indices) * cfg.M

    def test_state_action_constant_across_goal_labels(self, tiny_env):
        cfg = ESAgentConfig(K=2, sigma=0.1, alpha_1=0.02, M=1,
                            eta=-1.0, log_every=1, seed=0)
        agent = ESAgent(env=tiny_env, training_indices=[0, 1, 2],
                        config=cfg, verbose=False)
        agent.train_one_angle(target_idx=0)

        n_train = len(agent.training_indices)
        for k in range(cfg.K):
            chunk = agent.buffer.transitions[k * n_train:(k + 1) * n_train]
            assert len(chunk) == n_train
            # Same state/action/next_state across the per-goal copies for this k.
            for t in chunk[1:]:
                np.testing.assert_array_equal(t.state, chunk[0].state)
                np.testing.assert_array_equal(t.action, chunk[0].action)
                np.testing.assert_array_equal(t.next_state, chunk[0].next_state)
            assert sorted(t.goal for t in chunk) == sorted(agent.training_indices)

    def test_transitions_have_independent_array_storage(self, tiny_env):
        # Buffer must .copy() — otherwise later in-place updates to ε would
        # silently mutate historical transitions.
        cfg = ESAgentConfig(K=2, sigma=0.1, alpha_1=0.02, M=2,
                            eta=-1.0, log_every=1, seed=0)
        agent = ESAgent(env=tiny_env, training_indices=[0],
                        config=cfg, verbose=False)
        agent.train_one_angle(target_idx=0)
        state_ids = {id(t.state) for t in agent.buffer.transitions}
        action_ids = {id(t.action) for t in agent.buffer.transitions}
        assert len(state_ids) == len(agent.buffer.transitions)
        assert len(action_ids) == len(agent.buffer.transitions)


@pytest.mark.slow
class TestTrainAllAngles:
    def test_returns_one_entry_per_target(self, tiny_agent):
        bank = tiny_agent.train_all_angles(target_indices=[0, 1])
        assert set(bank.keys()) == {0, 1}
        for _, result in bank.items():
            assert isinstance(result, TrainingResult)

    def test_defaults_to_all_training_indices(self, tiny_env):
        cfg = ESAgentConfig(K=2, M=1, eta=-1.0, log_every=1, seed=0)
        agent = ESAgent(env=tiny_env, training_indices=[0, 2],
                        config=cfg, verbose=False)
        bank = agent.train_all_angles()    # no arg → all training_indices
        assert set(bank.keys()) == {0, 2}


@pytest.mark.slow
class TestCriticIntegration:
    """Verify ESAgent calls critic.update(self.buffer) once per ES iteration."""

    def test_critic_update_called_per_iteration(self, tiny_env):
        # A tiny stub critic that just counts how often it's called.
        class _StubCritic:
            def __init__(self):
                self.n_updates = 0
                self.last_buffer_size = None

            def update(self, buffer):
                self.n_updates += 1
                self.last_buffer_size = len(buffer)
                return 0.42   # pretend loss

        cfg = ESAgentConfig(K=2, sigma=0.1, alpha_1=0.02, M=3,
                            eta=-1.0, log_every=1, seed=0)
        critic = _StubCritic()
        agent = ESAgent(env=tiny_env, training_indices=[0, 1, 2],
                        config=cfg, critic=critic, verbose=False)
        result = agent.train_one_angle(target_idx=0)

        # One critic update per iteration we actually ran.
        assert critic.n_updates == result.iterations
        # critic_loss surfaces into the history records.
        for entry in result.history:
            assert entry['critic_loss'] == pytest.approx(0.42)

    def test_no_critic_means_no_call_and_critic_loss_is_none(self, tiny_env):
        cfg = ESAgentConfig(K=2, sigma=0.1, alpha_1=0.02, M=2,
                            eta=-1.0, log_every=1, seed=0)
        agent = ESAgent(env=tiny_env, training_indices=[0, 1, 2],
                        config=cfg, critic=None, verbose=False)
        result = agent.train_one_angle(target_idx=0)
        for entry in result.history:
            assert entry['critic_loss'] is None


@pytest.mark.slow
class TestConvergence:
    def test_loose_eta_triggers_early_termination(self, tiny_env):
        # η = 1.0 means "≥ 0 fraction at target", trivially satisfied on iter 1.
        cfg = ESAgentConfig(K=2, sigma=0.1, alpha_1=0.02, M=10,
                            eta=1.0, log_every=1, seed=0)
        agent = ESAgent(env=tiny_env, training_indices=[0, 1, 2],
                        config=cfg, verbose=False)
        result = agent.train_one_angle(target_idx=0)
        assert result.converged is True
        assert result.iterations == 1

    def test_unreachable_eta_runs_to_max_iter(self, tiny_env):
        # η = -1.0 means "≥ 2.0 fraction" — impossible — so we exhaust M.
        cfg = ESAgentConfig(K=2, sigma=0.1, alpha_1=0.02, M=2,
                            eta=-1.0, log_every=1, seed=0)
        agent = ESAgent(env=tiny_env, training_indices=[0, 1, 2],
                        config=cfg, verbose=False)
        result = agent.train_one_angle(target_idx=0)
        assert result.converged is False
        assert result.iterations == cfg.M


# =============================================================================
# Visual sanity check — initial vs converged permittivity + field intensity
# =============================================================================

@pytest.mark.slow
@pytest.mark.parametrize("target_idx,target_label", [
    (0, "bottom"),   # -90°
    (1, "right"),    #   0°
    (2, "top"),      # +90°
])
def test_visualize_initial_vs_converged(target_idx, target_label):
    """
    Run a brief ES session on a 3x3 / 3-receiver env at FINE resolution
    (0.001 m/pixel, 5x finer than tiny_env) for EACH of the three target
    sides, and save a 2x2 figure per target to tests/visual_output/:
        top row    = initial random ε   |  initial field intensity |E_z|²
        bottom row = converged ε*        |  converged field intensity

    Output files:
        tests/visual_output/es_agent_before_after_bottom.png
        tests/visual_output/es_agent_before_after_right.png
        tests/visual_output/es_agent_before_after_top.png
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pathlib import Path
    from simulation import simulate_ez_fields_per_source

    # Fresh env at finer resolution than tiny_env. Same 3x3 grid, same
    # 3 receivers (one centered on each side: bottom, right, top).
    region = create_design_region(resolution=0.001, bg_permittivity=1.0,
                                  margin_cells=20)
    grid = create_grid(num_rods_x=3, num_rods_y=3,
                       radius=0.01, distance=0.005, rod_permittivity=1.0)
    source = create_source(index=1, frequency=6e9, length=0.02, walls=True)
    receivers = [
        create_receiver(index=1, length=0.02, side='bottom', rod_index=2),
        create_receiver(index=2, length=0.02, side='right',  rod_index=2),
        create_receiver(index=3, length=0.02, side='top',    rod_index=2),
    ]
    env = create_environment(design_region=region, grid=grid,
                             sources=[source], receivers=receivers)
    initialize_environment(env)

    training_indices = [0, 1, 2]

    cfg = ESAgentConfig(K=16, sigma=0.15, alpha_1=0.05, M=50,
                        eta=-1.0,           # don't early-stop; want a full run
                        log_every=5, seed=0)
    agent = ESAgent(env=env, training_indices=training_indices,
                    config=cfg, verbose=False)

    # ----- capture INITIAL state -----------------------------------------
    eps_initial = np.random.default_rng(123).uniform(
        -1.0, 1.0, size=(env.grid.num_rods_x, env.grid.num_rods_y),
    )
    apply_eps_to_canvas(env, eps_initial)
    canvas_initial = env.design_region._canvas.copy()
    ez_initial = sum(simulate_ez_fields_per_source(env).values())
    intensity_initial = np.abs(ez_initial) ** 2
    P_initial = np.array([
        float(np.sum(intensity_initial * r._mask)) for r in env.receivers
    ])

    # ----- run ES ---------------------------------------------------------
    result = agent.train_one_angle(target_idx=target_idx)

    # ----- capture FINAL state -------------------------------------------
    apply_eps_to_canvas(env, result.eps_star)
    canvas_final = env.design_region._canvas.copy()
    ez_final = sum(simulate_ez_fields_per_source(env).values())
    intensity_final = np.abs(ez_final) ** 2
    P_final = np.array([
        float(np.sum(intensity_final * r._mask)) for r in env.receivers
    ])

    # ----- print receiver powers (before vs after) -----------------------
    def _print_powers(label, powers):
        print(f"\n  Receiver powers — {label}:")
        for i, r in enumerate(env.receivers):
            marker = "  ← TARGET" if i == target_idx else ""
            print(f"    receiver {i} ({r.side:>6s}, rod {r.rod_index}):  "
                  f"{powers[i]:.3e}{marker}")
        total = powers.sum()
        target_frac = powers[target_idx] / total if total > 0 else 0.0
        print(f"    target / total = {target_frac:.3f}")

    print(f"\n=== target = {target_label} (receiver index {target_idx}) ===")
    _print_powers("INITIAL (random ε)", P_initial)
    _print_powers("FINAL  (converged ε*)", P_final)

    # ----- compose 2x2 figure --------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    def _draw_permittivity(ax, canvas, title):
        clipped = np.clip(canvas, 0, 10)
        im = ax.imshow(clipped, cmap='plasma', origin='lower', vmin=0, vmax=10)
        ax.set_title(title)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        plt.colorbar(im, ax=ax, label='ε')

    def _draw_intensity(ax, intensity, canvas, title, target_receiver):
        vmax = np.percentile(intensity, 98)
        im = ax.imshow(intensity, cmap='inferno', origin='lower', vmin=0, vmax=vmax)
        # overlay rod/wall outlines and the target receiver in cyan
        ax.contour(canvas, [3.0, 5e5], colors='white', alpha=0.5, linewidths=0.6)
        ax.contour(target_receiver._mask, [0.5], colors='cyan', linewidths=1.5)
        ax.set_title(title)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        plt.colorbar(im, ax=ax, label='|E_z|²')

    target_receiver = env.receivers[target_idx]
    _draw_permittivity(axes[0, 0], canvas_initial, 'Initial permittivity (random ε)')
    _draw_intensity(axes[0, 1], intensity_initial, canvas_initial,
                    'Initial field intensity', target_receiver)
    _draw_permittivity(axes[1, 0], canvas_final,
                       f'Converged ε* (target receiver {target_idx}, '
                       f'{result.iterations} iters)')
    _draw_intensity(axes[1, 1], intensity_final, canvas_final,
                    f'Converged field intensity  (reward={result.best_reward:+.2e})',
                    target_receiver)

    fig.suptitle(f'ES Agent: initial vs converged on a 3x3 grid — '
                 f'target = {target_label} (receiver {target_idx})  '
                 f'(K={cfg.K}, M={cfg.M}, σ={cfg.sigma})',
                 fontsize=13)

    out_dir = Path(__file__).parent / 'visual_output'
    out_dir.mkdir(exist_ok=True)
    save_path = out_dir / f'es_agent_before_after_{target_label}.png'
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close('all')

    print(f'\nSaved: {save_path}')
    assert save_path.exists()
    assert save_path.stat().st_size > 0
