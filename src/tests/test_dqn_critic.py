"""
Tests for the goal-conditioned DQN critic.

Pure-logic tests only (no FDFD): the critic is environment-independent. It
consumes Transitions from a ReplayBuffer and produces scalar Q estimates.
"""

import tempfile

import numpy as np
import pytest
import torch

from algorithms.critics.dqn_critic import (DQNCritic, DQNCriticConfig,
                                            QNetwork)
from algorithms.infrastructure.utils import ReplayBuffer, Transition


# =============================================================================
# Helpers
# =============================================================================

def _fake_buffer(n: int = 64, state_shape=(3, 3), n_goals: int = 3, seed: int = 0):
    """A buffer of n random transitions (uncorrelated rewards)."""
    rng = np.random.default_rng(seed)
    buf = ReplayBuffer()
    for _ in range(n):
        eps = rng.uniform(-1, 1, size=state_shape).astype(np.float32)
        delta = (rng.standard_normal(state_shape) * 0.1).astype(np.float32)
        next_eps = np.clip(eps + delta, -1, 1).astype(np.float32)
        goal = int(rng.integers(0, n_goals))
        reward = float(rng.standard_normal())
        buf.append(Transition(state=eps, action=delta, reward=reward,
                              next_state=next_eps, goal=goal))
    return buf


def _make_critic(state_shape=(3, 3), n_goals=3, **overrides):
    cfg_kwargs = dict(hidden_dim=32, n_hidden_layers=2,
                      n_goals=n_goals, batch_size=8, G=2, seed=0)
    cfg_kwargs.update(overrides)
    cfg = DQNCriticConfig(**cfg_kwargs)
    return DQNCritic(state_shape=state_shape, config=cfg)


# =============================================================================
# Config
# =============================================================================

class TestConfig:
    def test_defaults_match_design_doc(self):
        cfg = DQNCriticConfig()
        assert cfg.gamma == 0.99
        assert cfg.tau == 0.995
        assert cfg.G == 20
        assert cfg.lr == 1e-3
        assert cfg.batch_size == 256
        assert cfg.n_goals == 30


# =============================================================================
# Q-network
# =============================================================================

class TestQNetwork:
    def test_output_shape_batch(self):
        net = QNetwork(state_shape=(3, 3), n_goals=3, hidden_dim=16, n_hidden_layers=1)
        B = 5
        eps = torch.zeros(B, 3, 3)
        delta = torch.zeros(B, 3, 3)
        goal = torch.zeros(B, dtype=torch.long)
        q = net(eps, delta, goal)
        assert q.shape == (B,)

    def test_one_hot_goal_changes_output(self):
        # Two different goal indices should give different Q (with random init).
        torch.manual_seed(0)
        net = QNetwork(state_shape=(3, 3), n_goals=3, hidden_dim=16, n_hidden_layers=1)
        eps = torch.randn(1, 3, 3)
        delta = torch.randn(1, 3, 3)
        q0 = net(eps, delta, torch.tensor([0])).item()
        q1 = net(eps, delta, torch.tensor([1])).item()
        assert q0 != q1


# =============================================================================
# DQNCritic — prediction + scoring
# =============================================================================

class TestPredict:
    def test_predict_single_returns_scalar(self):
        critic = _make_critic()
        eps = np.zeros((3, 3), dtype=np.float32)
        delta = np.zeros((3, 3), dtype=np.float32)
        q = critic.predict(eps, delta, goal=0)
        assert isinstance(q, float)

    def test_predict_batch_returns_array(self):
        critic = _make_critic()
        B = 5
        eps = np.zeros((B, 3, 3), dtype=np.float32)
        delta = np.zeros((B, 3, 3), dtype=np.float32)
        goal = np.zeros(B, dtype=np.int64)
        q = critic.predict(eps, delta, goal)
        assert isinstance(q, np.ndarray)
        assert q.shape == (B,)


class TestScoreTrajectory:
    def test_sum_of_per_step_Q(self):
        critic = _make_critic()
        T = 4
        eps_traj = np.zeros((T, 3, 3), dtype=np.float32)
        delta_traj = np.zeros((T, 3, 3), dtype=np.float32)
        per_step = critic.predict(eps_traj, delta_traj, np.zeros(T, dtype=np.int64))
        score = critic.score_trajectory(eps_traj, delta_traj, goal=0)
        assert score == pytest.approx(float(per_step.sum()))


# =============================================================================
# DQNCritic — TD(0) updates
# =============================================================================

class TestUpdate:
    def test_too_small_buffer_returns_none(self):
        critic = _make_critic(batch_size=64)
        buf = _fake_buffer(n=4)  # < batch_size
        assert critic.update(buf) is None

    def test_one_update_returns_loss_scalar(self):
        critic = _make_critic(batch_size=8, G=1)
        buf = _fake_buffer(n=32)
        loss = critic.update(buf)
        assert isinstance(loss, float)
        assert np.isfinite(loss)

    def test_weights_change_after_update(self):
        critic = _make_critic(batch_size=8, G=2)
        buf = _fake_buffer(n=64)
        before = [p.detach().clone() for p in critic.q.parameters()]
        critic.update(buf, n_steps=5)
        after = list(critic.q.parameters())
        # At least one parameter should have moved.
        moved = any((not torch.allclose(b, a)) for b, a in zip(before, after))
        assert moved

    def test_loss_decreases_on_constant_target(self):
        # If we hand the critic a buffer where (s, a, g) → r is fully
        # determined (no s' randomness), TD(0) should drive loss down.
        critic = _make_critic(batch_size=8, G=1, gamma=0.0, lr=1e-2)
        rng = np.random.default_rng(0)
        buf = ReplayBuffer()
        for _ in range(64):
            eps = rng.uniform(-1, 1, size=(3, 3)).astype(np.float32)
            delta = (rng.standard_normal((3, 3)) * 0.1).astype(np.float32)
            reward = float(np.sum(eps))  # deterministic function of state
            buf.append(Transition(state=eps, action=delta, reward=reward,
                                  next_state=eps, goal=0))
        # γ=0 → target = r; pure regression on a stationary target.
        loss_first = np.mean([critic.update(buf, n_steps=1) for _ in range(5)])
        loss_last = np.mean([critic.update(buf, n_steps=1) for _ in range(50, 55)])
        assert loss_last < loss_first


# =============================================================================
# DQNCritic — bootstrap action sampler
# =============================================================================

class TestBootstrapSampler:
    def test_custom_sampler_is_called(self):
        critic = _make_critic(batch_size=8, G=1)
        buf = _fake_buffer(n=32)
        calls = []

        def fake_policy(eps_next, goal):
            calls.append((eps_next.shape, goal.shape))
            return np.zeros_like(eps_next, dtype=np.float32)

        critic.update(buf, n_steps=1, bootstrap_action_sampler=fake_policy)
        assert len(calls) == 1
        eps_shape, goal_shape = calls[0]
        assert eps_shape == (critic.config.batch_size, 3, 3)
        assert goal_shape == (critic.config.batch_size,)

    def test_default_sampler_matches_eps_shape(self):
        critic = _make_critic(batch_size=8, G=1)
        eps = np.zeros((4, 3, 3), dtype=np.float32)
        goal = np.zeros(4, dtype=np.int64)
        delta = critic._default_gaussian_bootstrap(eps, goal)
        assert delta.shape == eps.shape


# =============================================================================
# DQNCritic — target network
# =============================================================================

class TestTargetNetwork:
    def test_target_initially_equals_online(self):
        critic = _make_critic()
        for p_q, p_t in zip(critic.q.parameters(), critic.q_target.parameters()):
            assert torch.allclose(p_q, p_t)

    def test_target_frozen(self):
        critic = _make_critic()
        for p in critic.q_target.parameters():
            assert not p.requires_grad

    def test_polyak_update_moves_target_toward_online(self):
        critic = _make_critic(tau=0.5)  # large step for visibility
        buf = _fake_buffer(n=32)
        # Snapshot target before update.
        target_before = [p.detach().clone() for p in critic.q_target.parameters()]
        # update() will gradient-step the online net, then Polyak the target.
        critic.update(buf, n_steps=5)
        # Target should have moved (because online moved during the gradient steps).
        moved = any(
            (not torch.allclose(tb, ta))
            for tb, ta in zip(target_before, critic.q_target.parameters())
        )
        assert moved


# =============================================================================
# DQNCritic — persistence
# =============================================================================

class TestPersistence:
    def test_save_load_round_trip_preserves_Q(self):
        critic = _make_critic()
        buf = _fake_buffer(n=64)
        critic.update(buf, n_steps=20)  # train a bit so weights are non-default

        # Snapshot Q on a probe before save
        probe_eps = np.zeros((3, 3, 3), dtype=np.float32)
        probe_delta = np.zeros((3, 3, 3), dtype=np.float32)
        probe_goal = np.array([0, 1, 2], dtype=np.int64)
        q_before = critic.predict(probe_eps, probe_delta, probe_goal)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            critic.save(path)
            restored = DQNCritic.load(path)
            q_after = restored.predict(probe_eps, probe_delta, probe_goal)
            np.testing.assert_allclose(q_after, q_before, rtol=1e-6)
        finally:
            import os
            os.unlink(path)

    def test_load_preserves_state_shape(self):
        critic = _make_critic(state_shape=(4, 5), n_goals=3)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            critic.save(path)
            restored = DQNCritic.load(path)
            assert restored.state_shape == (4, 5)
        finally:
            import os
            os.unlink(path)
