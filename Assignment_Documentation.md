# Deep Reinforcement Learning Assignment 1 Documentation

## 1. Purpose

This repository provides a structured, team-friendly implementation for the assignment divided into two parts:

- **Part 1: Multi-Armed Bandit (MAB)**
  - Solve a clinical trial recommendation problem using bandit learning strategies.
  - Implement environment setup, data generation, greedy/Epsilon-Greedy strategies, UCB1, and comparative analysis.

- **Part 2: Dynamic Programming (DP)**
  - Solve an autonomous drone rescue problem in a grid world using value iteration.
  - Implement a custom environment, state/action modeling, dynamic programming, policy visualization, and state-value analysis.

## 2. Problem Statement Summary

### MAB Problem

A hospital must recommend treatments for patients with varied disease severity. Each treatment is an arm with a hidden success probability. The goal is to learn which treatment performs best over time while maximizing cumulative reward.

Key concepts:
- Synthetic patient dataset
- Severity-based reward scaling
- Immediate exploitation
- Controlled exploration with epsilon-greedy
- Confidence-based selection using UCB1

### DP Problem

A rescue drone must navigate a disaster grid to save targets, avoid danger zones, manage battery, and recharge when needed. The problem is modeled as a finite Markov Decision Process (MDP).

Key concepts:
- Grid environment with safe, danger, wind, charging, rescue, and blocked cells
- Drone state includes position, battery, and rescue target status
- Reward structure incentivizes rescue and penalizes danger/battery exhaustion
- Value iteration computes the optimal policy

## 3. Folder Structure

```
DeepRL_Assignment1/
  README.md
  parameters.py
  requirements.txt
  Assignment_Documentation.md
  Assignment_Documentation.pdf
  mab/
    __init__.py
    analysis.py
    dataset.py
    environment.py
    run_mab.ipynb
    strategies.py
  dp/
    __init__.py
    analysis.py
    env.py
    mdp.py
    run_dp.ipynb
    visualization.py
```

### File descriptions

- `README.md` — project overview and quick start instructions.
- `parameters.py` — shared derived parameters for both MAB and DP, based on `GROUP_NUMBER = 91`.
- `requirements.txt` — Python dependencies.
- `Assignment_Documentation.md` — detailed documentation.
- `Assignment_Documentation.pdf` — generated PDF version of this documentation.
- `mab/` — MAB-specific modules and workflow notebook.
- `dp/` — DP-specific modules and workflow notebook.

## 4. Shared Parameters and Derivation

The shared parameter file is `parameters.py`.

### Derived values from group number 91

- `GROUP_NUMBER = 91`
- `LAST_DIGIT = 1`

#### MAB derived parameters

- `MAB_NUM_MEDICINES = (GROUP_NUMBER % 3) + 5 = 6`
- `MAB_HIDDEN_PROBS = [0.4 + ((GROUP_NUMBER + i) % 6) * 0.07 for i in range(6)]`
- `MAB_NUM_PATIENTS = 1000`

#### DP derived parameters

Since the last digit is `1`, the DP settings are:
- `DP_GRID_SIZE = 5`
- `DP_MAX_BATTERY = 15`
- `DP_WIND_PROB = 0.20`
- `DP_MAX_STEPS = 50`
- `DP_NUM_RESCUE = 2`
- `DP_NUM_CHARGERS = 1`
- `DP_NUM_DANGER = 3`
- `DP_NUM_BLOCKED = 2`

#### Feature toggles

- `MAB_ENABLED = 1`
- `DP_ENABLED = 1`

## 5. How to Change Parameters

1. Open `parameters.py`.
2. Change `GROUP_NUMBER` to your new team/group number.
3. The derived parameters will automatically update based on the formulas in `parameters.py`.

If you want to enable or disable a part of the assignment, set:
- `MAB_ENABLED = 0` to disable MAB flows
- `DP_ENABLED = 0` to disable DP flows

## 6. How to Run the Flow

### Setup

Install dependencies once:

```bash
pip install -r requirements.txt
```

### Run MAB workflow

Open and execute `mab/run_mab.ipynb` in Jupyter Notebook or JupyterLab.

### Run DP workflow

Open and execute `dp/run_dp.ipynb` in Jupyter Notebook or JupyterLab.

### Optional: run modules from Python

You can import functions directly from the modules in the scripts.

## 7. Purpose of MAB and DP in this Assignment

### MAB Purpose

- Demonstrate learning from sequential clinical trial data.
- Compare exploitation vs. exploration strategies.
- Show how different bandit algorithms balance risk and reward.
- Provide visual evidence of cumulative reward performance.

### DP Purpose

- Demonstrate modeling an MDP for robotic decision-making.
- Use dynamic programming to compute an optimal policy.
- Show how state representation and reward design affect convergence.
- Provide visualization of policy behavior and state values.

## 8. Making This a Team Effort

### Suggested team roles

- **Member 1:** MAB dataset and environment setup.
- **Member 2:** MAB strategy implementation and tuning.
- **Member 3:** DP environment and state/action representation.
- **Member 4:** DP value iteration, visualization, and report writing.

### How to collaborate

- Use the notebook `run_mab.ipynb` and `run_dp.ipynb` as working documents.
- Each member should fill in the placeholder sections with notes and results.
- Keep code modular by editing only the relevant module in `mab/` or `dp/`.
- Use `parameters.py` to keep group settings consistent across both parts.

## 9. Update Process

To update the parameter file and propagate changes:

1. Modify `GROUP_NUMBER` in `parameters.py`.
2. Run the notebooks again.
3. If needed, update environment-specific values in `mab/dataset.py` or `dp/env.py`.

For future extensions, add new modules under `mab/` or `dp/` and reference them from the notebooks.

## 10. Notes

- The framework is designed to be reusable and maintainable.
- The notebooks are intentionally structured with team placeholders.
- The shared `parameters.py` ensures the entire repo uses the same group-derived values.
