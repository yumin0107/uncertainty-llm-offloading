import random
from typing import List, Dict
from basestation import User, EdgeServer
from utils import calc_delay_set, add_user_to_ES


def random1_offloading(
    us: List[User], es: List[EdgeServer], n_users: int
) -> List[List[int]]:
    decisions: List[List[int]] = [[0 for _ in range(len(es))] for _ in range(len(us))]

    us_offloading = random.sample(us, n_users)
    delta: Dict[int, Dict[int, float]] = calc_delay_set(us_offloading, es, decisions)
    while us_offloading:
        uid_p, eid_p, _ = min(
            ((uid, eid, val) for uid, es in delta.items() for eid, val in es.items()),
            key=lambda x: x[2],
        )
        add_user_to_ES(us[uid_p], es[eid_p], decisions)
        us_offloading.remove(us[uid_p])
        delta = calc_delay_set(us_offloading, es, decisions)

    return decisions
