import time
import re
from typing import Dict, List, Tuple

from transformers import AutoConfig
import numpy as np
import torch
import tensorflow as tf
from sionna.channel import RayleighBlockFading
from sionna.channel.utils import gen_single_sector_topology

from config import FREQUENCY, LIGHTSPEED, LOCAL_COMPUTE_CAP
from basestation import User, EdgeServer


def generate_rayleigh_coeffs(M: int, d_i: tf.float32) -> np.ndarray:
    fspl = LIGHTSPEED / (4 * np.pi * FREQUENCY * d_i)  # d_i: 1 x M
    large_scale_gain = tf.reshape(tf.cast(fspl, tf.complex64), [1, 1, 1, M, 1, 1, 1])

    rayleigh = RayleighBlockFading(num_rx=1, num_rx_ant=1, num_tx=M, num_tx_ant=1)
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
    es: List[EdgeServer],
    decisions: Dict[int, Dict[int, int]],
) -> Tuple[float, float]:
    delay = 0
    correct = 0
    total = 0

    for u in users:
        pred_SLM = u.output_SLM[len(u.input) :]
        pred_LLM = u.output_LLM[len(u.input) :]
        user_to_server = {u.id: e for e in es for u in e.users}
        if is_offloading(u.id, decisions):
            e = user_to_server[u.id]
            delay += u.t_comp_llm / (e.C_j_ES / LOCAL_COMPUTE_CAP) + e.total_comm_delay(
                u
            )
            correct += is_correct(pred_LLM, u.label)
        else:
            delay += u.t_comp_slm
            correct += is_correct(pred_SLM, u.label)
        total += 1
    return delay / len(users) * 1000, correct / total * 100
