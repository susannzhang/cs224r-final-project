# =============================================================================
# MLP policy: π_φ(ε, θ) → δ  (flatten-and-MLP baseline; same shape as QNetwork)
# =============================================================================
"""
The original Phase 2 policy: flatten ε to a length-(N_x·N_y) vector, concat the
2-D (sin, cos) goal encoding, run it through a ReLU MLP. Kept as a baseline and
for loading pre-CNN checkpoints (select via ESPolicyConfig.policy_arch="mlp").

The physics-aware convolutional policy lives in networks/cnn.py; see its module
docstring for why a CNN is the better fit for the 2-D plasma-rod lattice. Both
classes share the exact same constructor kwargs and forward(eps, goal) -> delta
contract, so ESPolicy can swap between them with no other call-site change.
"""

import math

import torch
import torch.nn as nn


class MLPPolicyNetwork(nn.Module):
    """
    Inputs (batched):
        eps:  (B, N_x, N_y)  — current permittivity
        goal: (B,) int       — receiver index k ∈ [0, n_goals-1], encoded
                               internally as
                                   angle = π · k / (n_goals - 1)
                                   goal_feat = [sin(angle), cos(angle)]
                               so the receivers map onto a SEMICIRCLE
                               (k=0 → angle 0; k=n_goals-1 → angle π).
                               This matches the physical pm_setup geometry —
                               from the source on the left wall, the 30
                               receivers spanning bottom / right / top of
                               the design region cover ~180°, not a full
                               turn. Adjacent indices share representation;
                               k=0 and k=n_goals-1 are at opposite poles.
    Output:
        delta: (B, N_x, N_y) — perturbation in ε-space, optionally tanh-squashed
    """

    def __init__(self, state_shape, n_goals, hidden_dim=256,
                 n_hidden_layers=3, tanh_output=True, tanh_output_scale=1.0):
        super().__init__()
        N_x, N_y = state_shape
        # +2 for the sin/cos goal encoding (always 2 features regardless of n_goals).
        in_dim = N_x * N_y + 2
        out_dim = N_x * N_y
        layers = []
        prev = in_dim
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.ReLU())
            prev = hidden_dim
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
        self.state_shape = state_shape
        self.n_goals = n_goals
        self.tanh_output = tanh_output
        # Caps |δ| per element. scale < 1.0 forces graduated multi-step
        # trajectories instead of saturating jumps; gives ES room to refine
        # without fighting tanh saturation.
        self.tanh_output_scale = float(tanh_output_scale)

    def forward(self, eps, goal):
        B = eps.shape[0]
        eps_flat = eps.reshape(B, -1)
        # Map discrete index k to angle π·k/(n_goals-1), then encode as (sin, cos).
        # k=0 → angle 0 → (0, 1) ; k=n_goals-1 → angle π → (0, -1).
        # Semicircle (not full circle) — matches the physical receiver layout
        # spanning ~180° around the source.
        denom = max(self.n_goals - 1, 1)   # safe for n_goals=1 corner case
        angle = (math.pi / denom) * goal.float()
        goal_encoded = torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)
        x = torch.cat([eps_flat, goal_encoded], dim=-1)
        delta_flat = self.net(x)
        if self.tanh_output:
            delta_flat = self.tanh_output_scale * torch.tanh(delta_flat)
        return delta_flat.reshape(B, *self.state_shape)


# Drop-in alias, matching networks/cnn.py's `PolicyNetwork` export.
PolicyNetwork = MLPPolicyNetwork
