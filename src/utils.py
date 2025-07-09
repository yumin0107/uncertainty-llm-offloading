import time
import re

from transformers import AutoConfig
import numpy as np
import torch
import tensorflow as tf
from sionna.channel import RayleighBlockFading


def estimate_workload(input_length: int, model_name: str) -> float:

    try:
        cfg = AutoConfig.from_pretrained(model_name)
    except Exception as e:
        raise ValueError(f"Cannot load config for '{model_name}': {e}")

    n_layers = cfg.num_hidden_layers
    d_model = cfg.hidden_size
    d_ff = getattr(cfg, "intermediate_size", 4 * d_model)

    # Compute FLOPs per layer
    flops_attn = 4 * input_length * (d_model**2)
    flops_ffn = 8 * d_model * d_ff
    flops_per_layer = flops_attn + flops_ffn

    # Total FLOPs for all layers
    total_flops = flops_per_layer * n_layers

    return total_flops


def generate_rayleigh_coeffs(M: int) -> np.ndarray:
    rayleigh = RayleighBlockFading(num_rx=1, num_tx=1, num_rx_ant=1, num_tx_ant=1)
    h, _ = rayleigh(batch_size=M, num_time_steps=1)
    h = tf.squeeze(h, axis=[1, 2, 3, 4, 5])
    return h.numpy()


def bit_size_text(text) -> int:
    return len(text.encode("utf-8"))


def is_correct(prediction: str, answer: str) -> bool:
    prediction = prediction.strip().lower()
    answer = answer.strip().lower()
    return re.search(rf"\b{re.escape(answer)}\b", prediction) is not None
