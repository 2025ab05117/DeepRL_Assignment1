"""Dynamic programming utilities for the Drone Rescue environment."""

from typing import Dict, List, Tuple
import numpy as np
from dp.env import DroneRescueEnv


def value_iteration(
    env: DroneRescueEnv,
    reachable_states: List[Tuple[int, int, int, int, int]],
    theta: float = 1e-3,
    gamma: float = 1.0,
    max_iters: int = 1000,
) -> Tuple[np.ndarray, Dict[Tuple[int, int, int, int, int], str], int, float]:
    """Compute the optimal value function and policy using value iteration."""
    state_to_index = {state: idx for idx, state in enumerate(reachable_states)}
    value = np.zeros(len(reachable_states), dtype=float)
    policy = {}

    for iteration in range(1, max_iters + 1):
        delta = 0.0
        for state in reachable_states:
            if env.is_terminal(state):
                continue

            best_value = -np.inf
            best_action = env.ACTIONS[0]
            for action in env.valid_actions(state):
                next_state, reward, _ = env._step_deterministic(state, action)
                candidate = reward + gamma * value[state_to_index[next_state]]
                if candidate > best_value:
                    best_value = candidate
                    best_action = action

            idx = state_to_index[state]
            delta = max(delta, abs(value[idx] - best_value))
            value[idx] = best_value
            policy[state] = best_action

        if delta < theta:
            return value, policy, iteration, delta

    return value, policy, max_iters, delta
