"""Shared assignment parameters derived from the group number."""

GROUP_NUMBER = 91
LAST_DIGIT = GROUP_NUMBER % 10

# MAB derived parameters
MAB_NUM_MEDICINES = (GROUP_NUMBER % 3) + 5
MAB_HIDDEN_PROBS = [0.4 + ((GROUP_NUMBER + i) % 6) * 0.07 for i in range(MAB_NUM_MEDICINES)]
MAB_NUM_PATIENTS = 1000

# DP derived parameters
DP_GRID_SIZE = 6 if LAST_DIGIT >= 5 else 5
DP_MAX_BATTERY = 15 if LAST_DIGIT % 2 == 1 else 10
DP_WIND_PROB = 0.30 if LAST_DIGIT >= 5 else 0.20
DP_MAX_STEPS = 75 if DP_GRID_SIZE == 6 else 50
DP_NUM_RESCUE = 2 if LAST_DIGIT <= 4 else 3
DP_NUM_CHARGERS = 1 if LAST_DIGIT <= 4 else 2
DP_NUM_DANGER = 3 if LAST_DIGIT <= 4 else 4
DP_NUM_BLOCKED = 2 if LAST_DIGIT <= 4 else 3

# Environment type indicators
MAB_ENV_NAME = "Multi-Armed Bandit"
DP_ENV_NAME = "Drone Rescue DP"

# Feature toggles
MAB_ENABLED = 1
DP_ENABLED = 1
