import time
import re
from typing import Dict, List, Tuple

from transformers import AutoConfig
import numpy as np
import torch
import tensorflow as tf
from sionna.channel import RayleighBlockFading, AWGN
from sionna.channel.utils import gen_single_sector_topology

from config import FREQUENCY, LIGHTSPEED, NOISE_POWER
from basestation import User, EdgeServer


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


def is_offloading(u_id: int, decisions: Dict[int, Dict[int, int]]) -> bool:
    return any(value == 1 for value in decisions[u_id].values())


def calc_delay_accuracy(
    users: List[User],
    user_model,
    es: List[EdgeServer],
    edge_model,
    decisions: Dict[int, Dict[int, int]],
) -> Tuple[float, float]:
    delay = 0
    correct = 0
    total = 0

    for u in users:
        output_SLM, inf_delay_SLM = user_model.generate(u.input)
        output_LLM, inf_delay_LLM = edge_model.generate(u.input)
        pred_SLM = output_SLM[len(u.input) :]
        pred_LLM = output_LLM[len(u.input) :]
        user_to_server = {u.id: e for e in es for u in e.users}
        if is_offloading(u.id, decisions):
            e = user_to_server[u.id]
            delay += inf_delay_LLM / (e.C_j_ES / u.C_i_L) + e.total_comm_delay(u)
            correct += is_correct(pred_LLM, u.label)
        else:
            delay += inf_delay_SLM
            correct += is_correct(pred_SLM, u.label)
        total += 1
    return delay / len(users) * 1000, correct / total * 100
