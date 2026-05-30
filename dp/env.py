"""Implementation of the custom Drone Rescue environment."""

from typing import List, Tuple
import numpy as np
from parameters import GROUP_NUMBER


class DroneRescueEnv:
    """Grid world environment for the rescue drone dynamic programming problem."""

    ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "HOVER"]
    DELTA = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}

    def __init__(self, group_id: int = GROUP_NUMBER):
        self.group_id = group_id
        self.grid_size = 6 if group_id % 10 >= 5 else 5
        self.max_battery = 15 if group_id % 2 == 1 else 10
        self.wind_prob = 0.30 if group_id % 10 >= 5 else 0.20
        self.max_steps = 75 if self.grid_size == 6 else 50
        self.start = (0, 0)
        self._build_grid()

    def _build_grid(self):
        n = self.grid_size
        self.grid = np.full((n, n), "F", dtype="<U1")
        self.grid[self.start] = "S"

        if self.grid_size == 6:
            self.rescue_positions = [(1, 4), (3, 1), (4, 5)]
            self.charging_positions = [(2, 2), (5, 3)]
            self.danger_positions = [(0, 4), (3, 3), (4, 1), (5, 0)]
            self.wind_positions = [(1, 2), (2, 4), (4, 3)]
            self.blocked_positions = [(1, 1), (2, 5), (4, 2)]
        else:
            self.rescue_positions = [(1, 3), (3, 1)]
            self.charging_positions = [(2, 2)]
            self.danger_positions = [(0, 4), (3, 3), (4, 1)]
            self.wind_positions = [(1, 2), (3, 4)]
            self.blocked_positions = [(1, 1), (2, 4)]

        for pos in self.rescue_positions:
            self.grid[pos] = "R"
        for pos in self.charging_positions:
            self.grid[pos] = "C"
        for pos in self.danger_positions:
            self.grid[pos] = "D"
        for pos in self.wind_positions:
            self.grid[pos] = "W"
        for pos in self.blocked_positions:
            self.grid[pos] = "X"

        self.rescue_index = {pos: idx for idx, pos in enumerate(self.rescue_positions)}

    def state_to_key(self, row: int, col: int, battery: int, rescue_mask: int, step: int) -> Tuple[int, int, int, int, int]:
        return row, col, battery, rescue_mask, step

    def is_terminal(self, state: Tuple[int, int, int, int, int]) -> bool:
        _, _, battery, rescue_mask, step = state
        all_rescued = rescue_mask == 0
        return battery <= 0 or all_rescued or step >= self.max_steps

    def valid_actions(self, state: Tuple[int, int, int, int, int]) -> List[str]:
        if self.is_terminal(state):
            return []
        return self.ACTIONS.copy()

    def _is_rescue_active(self, row: int, col: int, rescue_mask: int) -> bool:
        if (row, col) not in self.rescue_index:
            return False
        bit = 1 << self.rescue_index[(row, col)]
        return (rescue_mask & bit) != 0

    def _move(self, row: int, col: int, action: str) -> Tuple[int, int]:
        if action == "HOVER":
            return row, col
        dr, dc = self.DELTA[action]
        new_row, new_col = row + dr, col + dc
        if not (0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size):
            return row, col
        if self.grid[new_row, new_col] == "X":
            return row, col
        return new_row, new_col

    def _apply_wind(self, state: Tuple[int, int, int, int, int], action: str) -> Tuple[int, int, int, int, int, float]:
        row, col, battery, rescue_mask, step = state
        if self.grid[row, col] != "W" or action == "HOVER":
            return self._step_deterministic(state, action)

        outcomes = []
        intended = self._step_deterministic(state, action)
        outcomes.append((1.0 - self.wind_prob, intended[0], intended[1], intended[2], intended[3], intended[4]))

        random_prob = self.wind_prob / 4.0
        for random_action in ["UP", "DOWN", "LEFT", "RIGHT"]:
            next_state, reward, done = self._step_deterministic(state, random_action)
            outcomes.append((random_prob, next_state[0], next_state[1], next_state[2], next_state[3], next_state[4]))

        merged = {}
        for prob, r, c, b, m, st in outcomes:
            key = (r, c, b, m, st)
            if key not in merged:
                merged[key] = [0.0, 0.0]
            merged[key][0] += prob
            merged[key][1] = 0.0

        # For deterministic code skeleton, return the most probable outcome.
        best_state = max(merged.items(), key=lambda x: x[1][0])[0]
        return best_state, 0.0, self.is_terminal(best_state)

    def _step_deterministic(self, state: Tuple[int, int, int, int, int], action: str) -> Tuple[Tuple[int, int, int, int, int], float, bool]:
        row, col, battery, rescue_mask, step = state
        if self.is_terminal(state):
            return state, 0.0, True

        if action == "HOVER":
            if self.grid[row, col] == "C":
                battery = min(self.max_battery, battery + 2)
            else:
                battery -= 1
            new_row, new_col = row, col
        else:
            battery -= 1
            new_row, new_col = self._move(row, col, action)
            if (new_row, new_col) in self.charging_positions:
                battery = self.max_battery

        new_mask = rescue_mask
        if self._is_rescue_active(new_row, new_col, rescue_mask):
            bit = 1 << self.rescue_index[(new_row, new_col)]
            new_mask = rescue_mask & ~bit

        reward = -1
        if self.grid[new_row, new_col] == "D":
            reward -= 10
        if self.grid[new_row, new_col] == "C" and (new_row, new_col) != (row, col):
            reward += 5
        if self._is_rescue_active(new_row, new_col, rescue_mask):
            reward += 20
        if battery <= 0:
            reward -= 20

        new_state = (new_row, new_col, max(battery, 0), new_mask, step + 1)
        return new_state, reward, self.is_terminal(new_state)

    def enumerate_reachable_states(self):
        """Enumerate all reachable states from the start state."""
        from collections import deque

        start_mask = (1 << len(self.rescue_positions)) - 1
        start_state = (self.start[0], self.start[1], self.max_battery, start_mask, 0)
        queue = deque([start_state])
        visited = {start_state}

        while queue:
            state = queue.popleft()
            if self.is_terminal(state):
                continue
            for action in self.valid_actions(state):
                next_state, _, _ = self._step_deterministic(state, action)
                if next_state not in visited:
                    visited.add(next_state)
                    queue.append(next_state)
        return list(visited)
