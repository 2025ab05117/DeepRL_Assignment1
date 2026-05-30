"""Bandit strategy implementations for the MAB assignment."""

from typing import Dict, List, Tuple
import numpy as np
from mab.environment import MABEnvironment


StrategyHistory = Dict[str, List[float]]


def greedy_strategy(env: MABEnvironment, initial_pulls: int = 10) -> Tuple[MABEnvironment, StrategyHistory]:
    """Run the immediate exploitation strategy.

    The policy tests each arm `initial_pulls` times, then exploits the best arm.
    """
    history = {"cumulative_reward": []}
    env.reset()

    for t in range(len(env.dataset)):
        if t < env.n_arms * initial_pulls:
            arm = t // initial_pulls
        else:
            arm = int(np.argmax(env.get_average_rewards()))

        _, reward = env.step(arm)
        history["cumulative_reward"].append(env.get_cumulative_reward())

    return env, history


def epsilon_greedy_strategy(
    env: MABEnvironment,
    epsilon: float,
    initial_pulls: int = 1,
) -> Tuple[MABEnvironment, StrategyHistory]:
    """Run epsilon-greedy with a fixed exploration probability."""
    history = {
        "cumulative_reward": [],
        "epsilon": [],
    }
    env.reset()

    for t in range(len(env.dataset)):
        if np.random.rand() < epsilon and np.sum(env.pulls) > 0:
            arm = int(np.random.choice(env.n_arms))
        else:
            if np.any(env.pulls == 0):
                arm = int(np.argmin(env.pulls))
            else:
                arm = int(np.argmax(env.get_average_rewards()))

        _, reward = env.step(arm)
        history["cumulative_reward"].append(env.get_cumulative_reward())
        history["epsilon"].append(epsilon)

    return env, history


def ucb1_strategy(env: MABEnvironment) -> Tuple[MABEnvironment, StrategyHistory]:
    """Run the UCB1 exploration-exploitation strategy."""
    history = {"cumulative_reward": [], "selected_arm": []}
    env.reset()

    for t in range(len(env.dataset)):
        if t < env.n_arms:
            arm = t
        else:
            average = env.get_average_rewards()
            confidence = np.sqrt(2 * np.log(t + 1) / env.pulls)
            arm = int(np.argmax(average + confidence))

        _, reward = env.step(arm)
        history["cumulative_reward"].append(env.get_cumulative_reward())
        history["selected_arm"].append(arm)

    return env, history
