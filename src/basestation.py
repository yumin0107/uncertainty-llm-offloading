import math
from typing import List, Dict, Optional

from config import BANDWIDTH, TRANSMIT_POWER, NOISE_POWER


class User:
    def __init__(
        self,
        id: int,
        D: float,
        h: List[complex],
        P: float,
        sigma2: float,
        input: any,
        label: any,
        W_i_SLM: float,
        W_i_LLM: float,
        C_i_L: float,
        p_k: List[float],
    ):
        self.id = id
        self.D = D
        self.h = h
        self.P = P
        self.sigma2 = sigma2
        self.input = input
        self.label = label
        self.W_i_SLM = W_i_SLM
        self.W_i_LLM = W_i_LLM
        self.C_i_L = C_i_L
        self.p_k = p_k

        self.t_comm: Optional[float] = None
        self.t_comp: Optional[float] = None
        self.prediction: Optional[str] = None

        self.uncertainty = 1.0 - (self.p_k[0] - self.p_k[1])

    def comm_delay(self, B_j: float, j: int) -> float:
        R_ij = B_j * math.log2(1 + self.P * abs(self.h[j]) / self.sigma2)
        return self.D / R_ij if R_ij > 0 else float("inf")

    def local_comp_delay(self) -> float:
        return self.W_i_SLM / self.C_i_L


class EdgeServer:
    def __init__(self, id, B: float, C_ES: float, C_max: float):
        self.id = id
        self.B = B
        self.C_ES = C_ES
        self.C_max = C_max

        self.B_j = B
        self.users: List[User] = []

    def add_user(self, user: User):
        self.users.append(user)
        return

    def bandwidth_allocation(self, n_offloaded: int) -> float:
        return self.B / n_offloaded

    def compute_allocation(self, n_offloaded: int) -> float:
        return max(self.C_max, self.C_ES / n_offloaded)

    def edge_comp_delay(self, u: User, C_j_ES: float) -> float:
        return u.W_i_LLM / C_j_ES

    def total_comm_delay(self, u: User) -> float:
        R_ij = self.B_j * math.log2(1 + u.P * abs(u.h[self.id]) / u.sigma2)
        return u.D / R_ij
