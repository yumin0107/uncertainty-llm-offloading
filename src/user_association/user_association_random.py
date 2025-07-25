import random
from typing import List, Tuple
from basestation import User, EdgeServer
from utils import add_user_to_ES


def random1_offloading(
    us: List[User], es: List[EdgeServer], n_users: int
) -> List[List[int]]:
    decisions: List[List[int]] = [[0 for _ in range(len(es))] for _ in range(len(us))]

    us_offloading = random.sample(us, n_users)
    for u in us_offloading:
        num = random.choice([0, 1, 2, 3])
        e = es[num]
        add_user_to_ES(u, e, decisions)

    return decisions
