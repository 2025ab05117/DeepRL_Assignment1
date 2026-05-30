"""Reusable Multi-Armed Bandit environment wrapper."""

from typing import List, Tuple
import pandas as pd
import numpy as np
from mab.dataset import compute_reward


class MABEnvironment:
    """Bandit environment that tracks pulls, rewards, and patient assignments."""

    def __init__(self, hidden_probs: List[float], dataset: pd.DataFrame):
        self.hidden_probs = hidden_probs
        self.dataset = dataset.copy()
        self.n_arms = len(hidden_probs)
        self.reset()

    def reset(self):
        """Reset environment statistics and dataset state."""
        self.dataset["assigned_medicine"] = -1
        self.dataset["clinical_outcome"] = -1
        self.dataset["utility_score"] = 0.0
        self.pulls = np.zeros(self.n_arms, dtype=int)
        self.total_rewards = np.zeros(self.n_arms, dtype=float)
        self.successes = np.zeros(self.n_arms, dtype=int)
        self.current_step = 0

    def step(self, arm: int) -> Tuple[int, float]:
        """Assign the selected arm and return the reward for the current patient."""
        patient = self.dataset.loc[self.current_step]
        severity = int(patient["severity_score"])
        outcome = int(np.random.rand() < self.hidden_probs[arm])
        reward = compute_reward(outcome, severity)

        self.pulls[arm] += 1
        self.total_rewards[arm] += reward
        self.successes[arm] += outcome

        self.dataset.at[self.current_step, "assigned_medicine"] = arm
        self.dataset.at[self.current_step, "clinical_outcome"] = outcome
        self.dataset.at[self.current_step, "utility_score"] = reward

        self.current_step += 1
        return outcome, reward

    def get_average_rewards(self) -> np.ndarray:
        """Return average utility reward per arm, handling zero-pull arms safely."""
        return np.divide(
            self.total_rewards,
            self.pulls,
            out=np.zeros_like(self.total_rewards),
            where=self.pulls > 0,
        )

    def get_cumulative_reward(self) -> float:
        """Return sum of utility scores collected so far."""
        return float(self.dataset["utility_score"].sum())

    def get_current_step(self) -> int:
        return int(self.current_step)
