import copy
import math
from typing import List, Dict, Optional

from config import BANDWIDTH, TRANSMIT_POWER, NOISE_POWER

import tensorflow as tf


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

    def __deepcopy__(self, memo):
        cls = self.__class__
        new = cls(
            self.id,
            self.D,
            list(self.h).append,
            self.P,
            self.sigma2,
            copy.deepcopy(self.input, memo),
            copy.deepcopy(self.lable, memo),
            self.W_i_SLM,
            self.W_i_LLM,
            self.C_i_L,
            list(self.p_k),
        )
        new.t_comm = self.t_comm
        new.t_comp = self.t_comp
        new.prediction = self.prediction
        new.uncertainty = self.uncertainty

        memo[id(self)] = new
        return new

    def comm_delay(self, B_j: float, j: int) -> float:
        R_ij = B_j * math.log2(1 + self.P * abs(self.h[j]) ** 2 / self.sigma2)
        return self.D / R_ij if R_ij > 0 else float("inf")

    def local_comp_delay(self) -> float:
        return self.W_i_SLM / self.C_i_L


class EdgeServer:
    def __init__(self, id: int, pos: tf.Tensor, B: float, C_ES: float):
        self.id = id
        self.pos = pos
        self.B = B
        self.C_ES = C_ES

        self.B_j = B
        self.users: List[User] = []

    def __deepcopy__(self, memo):
        cls = self.__class__
        new = cls(self.id, self.pos, self.B, self.C_ES)
        new.users = []

        memo[id(self)] = new
        return new

    def add_user(self, user: User):
        self.users.append(user)
        return

    def bandwidth_allocation(self, n_offloaded: int) -> float:
        return self.B / n_offloaded

    def edge_comp_delay(self, u: User) -> float:
        return u.W_i_LLM / self.C_ES

    def total_comm_delay(self, u: User) -> float:
        R_ij = self.B_j * math.log2(1 + u.P * abs(u.h[self.id]) ** 2 / u.sigma2)
        return u.D / R_ij
