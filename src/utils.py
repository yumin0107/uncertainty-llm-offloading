import time
import re

from transformers import AutoConfig
import numpy as np
import torch
import tensorflow as tf
from sionna.channel import RayleighBlockFading, AWGN
from sionna.channel.utils import gen_single_sector_topology

from config import FREQUENCY, LIGHTSPEED, NOISE_POWER


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


def generate_rayleigh_coeffs(M: int, d_i: tf.float32) -> np.ndarray:
    fspl = LIGHTSPEED / (4 * np.pi * FREQUENCY * d_i)  # d_i: 1 x M
    large_scale_gain = tf.reshape(tf.cast(fspl, tf.complex64), [1, 1, 1, M, 1, 1, 1])

    rayleigh = RayleighBlockFading(num_rx=1, num_rx_ant=1, num_tx=M, num_tx_ant=1)
    awgn_channel = AWGN()
    h_small, _ = rayleigh(batch_size=1, num_time_steps=1)  # [1, 1, 1, M, 1, 1, 1]

    h = h_small * large_scale_gain

    h = tf.squeeze(h, axis=None)
    return h.numpy()


def bit_size_text(text) -> int:
    return len(text.encode("utf-8"))


def is_correct(prediction: str, answer: str) -> bool:
    prediction = prediction.strip().lower()
    answer = answer.strip().lower()
    return re.search(rf"\b{re.escape(answer)}\b", prediction) is not None
