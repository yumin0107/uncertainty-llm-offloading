import math
from typing import List


class User:
    def __init__(
        self,
        id: int,
        D: float,
        h: complex,
        P: float,
        sigma2: float,
        W_i_SLM: float,
        W_i_LLM: float,
        input_q: any,  # query type
        model: any,
        C_i_L: float,
        p_k: List[float],
    ):
        self.id = id
        self.D = D
        self.h = h
        self.P = P
        self.sigma2 = sigma2
        self.W_i_SLM = W_i_SLM
        self.W_i_LLM = W_i_LLM
        self.input_q = input_q
        self.model = model
        self.C_i_L = C_i_L
        self.p_k = p_k

        self.uncertainty = 1.0 - (self.p_k[0] - self.p_k[1])

    def comm_rate(self, B_i: float) -> float:
        return B_i * math.log2(1 + self.P * abs(self.h) / self.sigma2)

    def comm_delay(self, B_i: float) -> float:
        R = self.comm_rate(B_i)
        return self.D / R if R > 0 else float("inf")

    def local_comp_delay(self) -> float:
        return self.W_i_SLM / self.C_i_L


class EdgeServer:
    def __init__(self, B: float, C_ES: float, C_max: float):
        self.B = B
        self.C_ES = C_ES
        self.C_max = C_max
        self.users: List[User] = []

    def add_user(self, user: User):
        self.users.append(user)
        return

    def bandwidth_allocation(self, n_offloaded: int) -> float:
        return self.B / n_offloaded

    def compute_allocation(self, n_offloaded: int) -> float:
        return max(self.C_max, self.C_ES / n_offloaded)

    def edge_comp_delay(self, u: User, C_i_ES: float) -> float:
        return u.W_i_LLM / C_i_ES
