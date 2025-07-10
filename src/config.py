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
BANDWIDTH = 10e6  # Hz
TRANSMIT_POWER = 0.2  # W
NOISE_POWER = 4e-21 * BANDWIDTH * 3.16
FREQUENCY = 3.5e9
LIGHTSPEED = 3e8

# Computation
LOCAL_COMPUTE_CAP = 91.06e12  # C_L (2 GHz)
EDGE_COMPUTE_CAP = 91.06e12  # C_ES (100 GHz)

# Model
SLM = "meta-llama/Llama-3.2-1B-Instruct"
LLM = "meta-llama/Llama-3.1-8B-Instruct"
K = 10
