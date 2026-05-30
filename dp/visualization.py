"""Visualization helpers for the DP drone rescue assignment."""

import matplotlib.pyplot as plt
import numpy as np
from dp.env import DroneRescueEnv


def plot_policy(env: DroneRescueEnv, policy: dict, battery: int, rescue_mask: int, step: int = 0):
    """Plot the greedy policy arrows for a fixed battery and rescue mask slice."""
    grid_labels = np.full((env.grid_size, env.grid_size), "", dtype=object)
    direction_symbols = {"UP": "^", "DOWN": "v", "LEFT": "<", "RIGHT": ">", "HOVER": "o"}

    for r in range(env.grid_size):
        for c in range(env.grid_size):
            state = (r, c, battery, rescue_mask, step)
            if state in policy:
                grid_labels[r, c] = direction_symbols[policy[state]]
            else:
                grid_labels[r, c] = env.grid[r, c]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np.zeros((env.grid_size, env.grid_size)), cmap="Greys", alpha=0.0)
    for r in range(env.grid_size):
        for c in range(env.grid_size):
            ax.text(c, r, grid_labels[r, c], ha="center", va="center", fontsize=16)
    ax.set_xticks(np.arange(-0.5, env.grid_size, 1))
    ax.set_yticks(np.arange(-0.5, env.grid_size, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(color="black", linestyle="-", linewidth=1)
    ax.set_title(f"Policy visualization (battery={battery})")
    plt.show()


def plot_value_heatmap(env: DroneRescueEnv, value: np.ndarray, state_to_index: dict, battery: int, rescue_mask: int, step: int = 0):
    """Plot a state-value heatmap for a fixed battery and rescue mask slice."""
    heatmap = np.full((env.grid_size, env.grid_size), np.nan, dtype=float)
    for r in range(env.grid_size):
        for c in range(env.grid_size):
            state = (r, c, battery, rescue_mask, step)
            if state in state_to_index:
                heatmap[r, c] = value[state_to_index[state]]

    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap, cmap="viridis", origin="upper")
    plt.colorbar(label="Value")
    plt.title(f"Value heatmap (battery={battery})")
    plt.xticks([])
    plt.yticks([])
    plt.show()
