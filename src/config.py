# src/config.py

# Communication
BANDWIDTH = 1e7  # Hz
TRANSMIT_POWER = 0.1  # W
NOISE_POWER = -174

# Computation
LOCAL_COMPUTE_CAP = 2e9  # C_L (2 GHz)
EDGE_COMPUTE_CAP = 100e9  # C_ES (100 GHz)
MAX_COMPUTE_PER_USER = 10e9  # C_max (10 GHz)

# Model
SLM = "TinyLlama/TinyLlama_v1.1"
LLM = "meta-llama/Llama-2-7b"
