# Deep Reinforcement Learning Assignment 1 Framework

This repository provides a structured framework for the group assignment.

## Project Layout

- `mab/`
  - `dataset.py`: synthetic patient dataset creation and environment setup
  - `environment.py`: reusable bandit environment wrapper
  - `strategies.py`: greedy, epsilon-greedy, and UCB1 implementations
  - `analysis.py`: cumulative reward plotting and result comparison
  - `run_mab.ipynb`: notebook for running the MAB workflow

- `dp/`
  - `env.py`: drone rescue environment implementation
  - `mdp.py`: value iteration for optimal policy computation
  - `visualization.py`: policy and value heatmap plotting
  - `analysis.py`: analysis helpers and scenario summary
  - `run_dp.ipynb`: notebook for running the DP workflow

## Team Contribution Guidelines

Each module contains clearly labeled sections for team members to contribute.
Use the notebook cells to document which member is responsible for each section.

## Installation

```bash
pip install -r requirements.txt
```

## Execution

Open the notebook files in `mab/run_mab.ipynb` and `dp/run_dp.ipynb` to run the full workflow.

## Documentation

A detailed project guide is available in `Assignment_Documentation.md` and `Assignment_Documentation.pdf`.

