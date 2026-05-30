"""Analysis helpers and plotting utilities for MAB experiments."""

from typing import Dict, List
import matplotlib.pyplot as plt


def plot_cumulative_rewards(history_map: Dict[str, List[float]]):
    """Plot cumulative reward curves for multiple strategies."""
    plt.figure(figsize=(10, 6))
    for label, history in history_map.items():
        plt.plot(history, label=label)

    plt.xlabel("Patient Number")
    plt.ylabel("Cumulative Reward")
    plt.title("Cumulative Reward vs Number of Patients")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def summarize_results(history_map: Dict[str, List[float]]) -> Dict[str, float]:
    """Return the final cumulative reward for each strategy."""
    results = {label: history[-1] for label, history in history_map.items()}
    return results
