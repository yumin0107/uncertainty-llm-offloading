import time

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


def measure_inference_delay(model, input: str, max_length: int = 100) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.model.to(device)
    model.tokenizer.model_max_length = max_length

    inputs = model.tokenizer(input, return_tensors="pt", truncation=True).to(device)

    _ = model.model.generate(**inputs, max_length=max_length)

    torch.cuda.synchronize()
    start = time.time()
    _ = model.model.generate(**inputs, max_length=max_length)
    torch.cuda.synchronize()
    end = time.time()

    return end - start


def generate_rayleigh_coeffs(N: int) -> np.ndarray:
    rayleigh = RayleighBlockFading(num_rx=1, num_tx=1, dtype=tf.complex64)
    x = tf.ones((N, 1, 1, 1), dtype=tf.complex64)
    h = rayleigh(x, training=False)
    h = tf.squeeze(h, axis=[1, 2, 3])
    return h.numpy()


def bit_size_text(text) -> int:
    return len(text.encode("utf-8"))
