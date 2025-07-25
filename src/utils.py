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


def generate_rayleigh_coeffs(N: int, M: int, d: tf.Tensor) -> np.ndarray:
    fspl = LIGHTSPEED / (4 * np.pi * FREQUENCY * d)  # d: N x M
    large_scale_gain = tf.reshape(tf.cast(fspl, tf.complex64), [1, N, 1, M, 1, 1, 1])

    rayleigh = RayleighBlockFading(num_rx=N, num_rx_ant=1, num_tx=M, num_tx_ant=1)
    h_small, _ = rayleigh(batch_size=1, num_time_steps=1)  # [1, N, 1, M, 1, 1, 1]

    h = h_small * large_scale_gain

    h = tf.squeeze(h, axis=None)
    return h.numpy()


def bit_size_text(text) -> int:
    return len(text.encode("utf-8"))


def is_correct(prediction: str, answer: str) -> bool:
    prediction = prediction.strip().lower()
    answer = answer.strip().lower()
    return re.search(rf"\b{re.escape(answer)}\b", prediction) is not None


def is_offloading(u_id: int, decisions: List[List[int]]) -> bool:
    return any(value == 1 for value in decisions[u_id])


def num_offloading(eid: int, decisions: List[List[int]]) -> int:
    return sum(row[eid] == 1 for row in decisions)


def calc_delay_accuracy(
    users: List[User],
    es: List[EdgeServer],
    decisions: List[List[int]],
) -> Tuple[float, float, float, float, float]:
    delay_edge = 0
    delay_local = 0
    t_comm_sum = 0
    t_comp_sum = 0
    correct = 0
    total = 0
    n_edge = 0
    n_local = 0

    for u in users:
        pred_SLM = u.output_SLM[len(u.input) :]
        pred_LLM = u.output_LLM[len(u.input) :]
        user_to_server = {u.id: e for e in es for u in e.users}
        if is_offloading(u.id, decisions):
            e = user_to_server[u.id]
            n = num_offloading(e.id, decisions)
            B_j = e.bandwidth_allocation(n)
            t_comm = e.total_comm_delay(u, decisions, B_j)
            t_comp = u.t_comp_llm / (e.C_j_ES / LOCAL_COMPUTE_CAP)
            delay_edge += t_comm + t_comp
            t_comm_sum += t_comm
            t_comp_sum += t_comp
            correct += is_correct(pred_LLM, u.label)
            n_edge += 1
        else:
            delay_local += u.t_comp_slm
            correct += is_correct(pred_SLM, u.label)
            n_local += 1
        total += 1
    delay_edge_avg = delay_edge / n_edge * 1000 if n_edge > 0 else 0
    delay_local_avg = delay_local / n_local * 1000 if n_local > 0 else 0
    t_comp_avg = t_comp_sum / n_edge * 1000 if n_edge > 0 else 0
    t_comm_avg = t_comm_sum / n_edge * 1000 if n_edge > 0 else 0
    accuracy = correct / total * 100
    return delay_edge_avg, delay_local_avg, t_comm_avg, t_comp_avg, accuracy


def calc_delay(u: User, e: EdgeServer, decisions: List[List[int]]) -> float:
    B_j = e.bandwidth_allocation(len(e.users) + 1)
    C_j_ES = e.compute_allocation(len(e.users) + 1)

    t_j_comm = u.comm_delay(B_j, e.id, decisions)
    t_j_comp_ES = e.edge_comp_delay(u, C_j_ES)
    t_i_comp_L = u.t_comp_slm

    return t_j_comm + t_j_comp_ES - t_i_comp_L


def calc_delta_set(
    us: List[User], es: List[EdgeServer], decisions: List[List[int]]
) -> Dict[int, Dict[int, float]]:
    delta_set: Dict[int, Dict[int, float]] = {}
    for u in us:
        delta_set[u.id] = {}
        for e in es:
            delta_set[u.id][e.id] = u.uncertainty * calc_delay(u, e, decisions)
    return delta_set


def calc_delay_set(
    us: List[User], es: List[EdgeServer], decisions: List[List[int]]
) -> Dict[int, Dict[int, float]]:
    delay_set: Dict[int, Dict[int, float]] = {}
    for u in us:
        delay_set[u.id] = {}
        for e in es:
            delay_set[u.id][e.id] = calc_delay(u, e, decisions)
    return delay_set


def add_user_to_ES(u: User, e: EdgeServer, decisions: List[List[int]]) -> None:
    decisions[u.id][e.id] = 1
    e.add_user(u)
    e.B_j = e.bandwidth_allocation(len(e.users))
    e.C_j_ES = e.compute_allocation(len(e.users))
    return
