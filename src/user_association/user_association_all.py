from typing import Dict, List
from basestation import User, EdgeServer
from utils import calc_delta_set, add_user_to_ES


def all_offloading(us: List[User], es: List[EdgeServer]) -> List[List[int]]:
    decisions: List[List[int]] = [[0 for _ in range(len(es))] for _ in range(len(us))]

    I: List[User] = [u for u in us]
    delta: Dict[int, Dict[int, float]] = calc_delta_set(I, es, decisions)
    while I:
        uid_p, eid_p, _ = min(
            ((uid, eid, val) for uid, es in delta.items() for eid, val in es.items()),
            key=lambda x: x[2],
        )
        add_user_to_ES(us[uid_p], es[eid_p], decisions)
        I.remove(us[uid_p])
        delta = calc_delta_set(I, es, decisions)

    return decisions
