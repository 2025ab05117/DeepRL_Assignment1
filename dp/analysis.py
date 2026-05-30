"""Analysis helpers for the DP drone rescue workflow."""

from typing import Dict


def summarize_dp_results(metadata: Dict[str, float]) -> str:
    """Build a short analysis summary based on DP results."""
    summary_lines = [
        "Dynamic programming results summary:",
        f"- Convergence iterations: {metadata.get('iterations', 'N/A')}",
        f"- Runtime (seconds): {metadata.get('runtime', 'N/A'):.2f}",
        f"- Final delta: {metadata.get('delta', 'N/A'):.6f}",
        f"- Start state value: {metadata.get('start_value', 'N/A'):.2f}",
    ]
    return "\n".join(summary_lines)
