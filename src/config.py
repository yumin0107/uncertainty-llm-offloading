# src/config.py

# Position Setting
M = 4  # number of ES
D = 500  # grid size
FIXED_ES = [
    [125.0, 125.0],
    [125.0, 375.0],
    [375.0, 125.0],
    [375.0, 375.0],
]  # es position

# Communication
BANDWIDTH = 1e7  # Hz
TRANSMIT_POWER = 0.1  # W
NOISE_POWER = 0.1
FREQUENCY = 3.5e9
LIGHTSPEED = 3e8

# Computation
LOCAL_COMPUTE_CAP = 2e9  # C_L (2 GHz)
EDGE_COMPUTE_CAP = 100e9  # C_ES (100 GHz)
MAX_COMPUTE_PER_USER = 10e9  # C_max (10 GHz)

# Model
SLM = "meta-llama/Llama-3.2-1B-Instruct"
LLM = "meta-llama/Llama-3.2-3B-Instruct"
K = 10
