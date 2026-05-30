"""Synthetic dataset construction for the Multi-Armed Bandit assignment."""

import numpy as np
import pandas as pd
from parameters import GROUP_NUMBER, MAB_NUM_PATIENTS


def create_patient_dataset(group_number: int = GROUP_NUMBER, n_patients: int = MAB_NUM_PATIENTS):
    """Create the synthetic MAB patient dataset according to assignment rules.

    Args:
        group_number: The team/group number used for reproducibility.
        n_patients: The number of synthetic patients to generate.

    Returns:
        A pandas DataFrame with patient_id, severity_score, assigned_medicine,
        clinical_outcome, and utility_score columns.
    """
    np.random.seed(group_number)

    # Rule 1.1: number of medicines
    k_medicines = (group_number % 3) + 5

    # Rule 1.2: hidden success probabilities for each medicine
    hidden_probs = [0.4 + ((group_number + i) % 6) * 0.07 for i in range(k_medicines)]

    patient_ids = np.arange(n_patients)
    severity_scores = (patient_ids % 5) + 1

    dataset = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "severity_score": severity_scores,
            "assigned_medicine": -1,
            "clinical_outcome": -1,
            "utility_score": 0.0,
        }
    )

    metadata = {
        "group_number": group_number,
        "n_medicines": k_medicines,
        "hidden_probs": hidden_probs,
    }
    return dataset, metadata


def compute_reward(clinical_outcome: int, severity_score: int) -> float:
    """Compute the reward for a patient based on outcome and severity.

    The reward decreases for higher severity, and is zero when the patient does not recover.
    """
    return float(clinical_outcome) * (1.0 - 0.1 * severity_score)
