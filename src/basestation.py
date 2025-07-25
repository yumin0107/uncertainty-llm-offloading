import copy
import math
from typing import List, Dict, Optional

from config import BANDWIDTH, TRANSMIT_POWER, NOISE_POWER

import tensorflow as tf
import numpy as np


class User:
    def __init__(
        self,
        id: int,
        D: float,
        h,  # N x M
        P: float,
        sigma2: float,
        input: any,
        output_slm: any,
        output_llm: any,
        label: any,
        t_comp_slm: float,
        t_comp_llm: float,
        C_i_L: float,
        p_k: List[float],
    ):
        self.id = id
        self.D = D
        self.h = h
        self.P = P
        self.sigma2 = sigma2
        self.input = input
        self.output_SLM = output_slm
        self.output_LLM = output_llm
        self.label = label
        self.t_comp_slm = t_comp_slm
        self.t_comp_llm = t_comp_llm
        self.C_i_L = C_i_L
        self.p_k = p_k

        # self.t_comm: Optional[float] = None
        # self.prediction: Optional[str] = None

        self.uncertainty = 1.0 - (self.p_k[0] - self.p_k[1])

    def comm_delay(self, B_j: float, j: int, decisions: List[List[int]]) -> float:

        mask = np.array(decisions, dtype=bool)

        other = mask.copy()
        other[:, j] = False
        other[self.id, :] = False

        interferers = other.any(axis=1)

        h_abs2 = np.abs(self.h[:, j]) ** 2

        interference = self.P * h_abs2[interferers].sum()

        h_ij = np.abs(self.h[self.id, j])
        sinr = (self.P * h_ij**2) / (interference + self.sigma2)
        R_ij = B_j * math.log2(1 + sinr)
        return self.D / R_ij if R_ij > 0 else float("inf")


class EdgeServer:
    def __init__(self, id: int, pos: tf.Tensor, B: float, C_ES: float, C_max: float):
        self.id = id
        self.pos = pos
        self.B = B
        self.C_ES = C_ES
        self.C_max = C_max

        self.B_j = B
        self.C_j_ES = self.C_max
        self.users: List[User] = []

    def add_user(self, user: User):
        self.users.append(user)
        return

    def bandwidth_allocation(self, n_offloaded: int) -> float:
        return self.B / n_offloaded

    def compute_allocation(self, n_offloaded: int) -> float:
        return min(self.C_max, self.C_ES / n_offloaded)

    def edge_comp_delay(self, u: User, C_j_ES: float) -> float:
        cap_ratio = C_j_ES / u.C_i_L
        return u.t_comp_slm * 8 / cap_ratio

    def total_comm_delay(
        self, u: User, decisions: List[List[int]], B_j: float
    ) -> float:

        mask = np.array(decisions, dtype=bool)

        other = mask.copy()
        other[u.id, :] = False
        other[:, self.id] = False

        interferers = other.any(axis=1)

        h_abs2 = np.abs(u.h[:, self.id]) ** 2

        interference = u.P * h_abs2[interferers].sum()

        h_ij = np.abs(u.h[u.id, self.id])
        sinr = (u.P * h_ij**2) / (interference + u.sigma2)
        R_ij = B_j * math.log2(1 + sinr)
        return u.D / R_ij if R_ij > 0 else float("inf")
