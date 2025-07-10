import random
from typing import Dict, List, Tuple
from basestation import User, EdgeServer


def random_offloading(
    us: List[User], es: List[EdgeServer]
) -> Dict[int, Dict[int, int]]:
    decisions: Dict[int, Dict[int, int]] = {u.id: {e.id: 0 for e in es} for u in us}

    for u in us:
        if random.random() < 0.5:
            e_p = random.choice(es)
            decisions[u.id][e_p.id] = 1
            e_p.add_user(u)
            e_p.B_j = e_p.bandwidth_allocation(len(e_p.users))

    return decisions
