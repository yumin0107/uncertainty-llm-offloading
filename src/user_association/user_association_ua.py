import time
from typing import Dict, List, Tuple

from basestation import User, EdgeServer
from utils import calc_delta_set, add_user_to_ES


def uncertainty_aware_offloading(
    us: List[User], es: List[EdgeServer], tau: float
) -> Tuple[List[List[int]], int, float]:
    start = time.time()
    decisions: List[List[int]] = [[0 for _ in range(len(es))] for _ in range(len(us))]

    # Step 1
    I_off: List[User] = [u for u in us if u.uncertainty > tau]
    delta_off: Dict[int, Dict[int, float]] = calc_delta_set(I_off, es, decisions)
    while I_off:
        uid_p, eid_p, _ = min(
            (
                (uid, eid, val)
                for uid, es in delta_off.items()
                for eid, val in es.items()
            ),
            key=lambda x: x[2],
        )
        add_user_to_ES(us[uid_p], es[eid_p], decisions)  # update decisions
        I_off.remove(us[uid_p])  # update I_off
        delta_off = calc_delta_set(I_off, es, decisions)

    # Step 2
    I_rem: List[User] = [u for u in us if u.uncertainty <= tau]
    delta_rem: Dict[int, Dict[int, float]] = calc_delta_set(I_rem, es, decisions)
    while I_rem:
        uid_p, eid_p, delta_min = min(
            (
                (uid, eid, val)
                for uid, es in delta_rem.items()
                for eid, val in es.items()
            ),
            key=lambda x: x[2],
        )
        if delta_min >= 0:
            break

        add_user_to_ES(us[uid_p], es[eid_p], decisions)
        I_rem.remove(us[uid_p])
        delta_rem = calc_delta_set(I_rem, es, decisions)

    end = time.time()
    return decisions, len(us) - len(I_rem), end - start
