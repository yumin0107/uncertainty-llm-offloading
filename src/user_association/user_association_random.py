import random
from typing import Dict, List
from basestation import User, EdgeServer


def random1_offloading(
    us: List[User], es: List[EdgeServer], n_users: int
) -> Dict[int, Dict[int, int]]:
    decisions: Dict[int, Dict[int, int]] = {u.id: {e.id: 0 for e in es} for u in us}

    us_offloading = random.sample(us, n_users)
    for u in us_offloading:
        delta_p: Dict[EdgeServer, float] = {}
        for e in es:
            B_j = e.bandwidth_allocation(len(e.users) + 1)
            C_j_ES = e.compute_allocation(len(e.users) + 1)

            t_j_comm = u.comm_delay(B_j, e.id)
            t_j_comp_ES = e.edge_comp_delay(u, C_j_ES)
            t_j_comp_L = u.t_comp_slm

            delta_p[e] = t_j_comm + t_j_comp_ES - t_j_comp_L

        e_p = min(delta_p, key=lambda e: delta_p[e])

        decisions[u.id][e_p.id] = 1
        e_p.add_user(u)
        e_p.B_j = e_p.bandwidth_allocation(len(e_p.users))
        e_p.C_j_ES = e_p.compute_allocation(len(e_p.users))

    return decisions


def random2_offloading(
    us: List[User], es: List[EdgeServer]
) -> Dict[int, Dict[int, int]]:
    decisions: Dict[int, Dict[int, int]] = {u.id: {e.id: 0 for e in es} for u in us}

    n_rand2 = 0
    for u in us:
        num = random.choice([0, 1, 2, 3, 4])
        if num == 4:
            continue
        else:
            e = es[num]
            decisions[u.id][e.id] = 1
            e.add_user(u)
            e.B_j = e.bandwidth_allocation(len(e.users))
            e.C_j_ES = e.compute_allocation(len(e.users))
            n_rand2 += 1

    return decisions, n_rand2


def random3_offloading(
    us: List[User], es: List[EdgeServer]
) -> Dict[int, Dict[int, int]]:
    decisions: Dict[int, Dict[int, int]] = {u.id: {e.id: 0 for e in es} for u in us}

    n_rand3 = 0
    for u in us:
        if random.random() < 0.5:
            continue
        else:
            num = random.choice([0, 1, 2, 3])
            e = es[num]
            decisions[u.id][e.id] = 1
            e.add_user(u)
            e.B_j = e.bandwidth_allocation(len(e.users))
            e.C_j_ES = e.compute_allocation(len(e.users))
            n_rand3 += 1

    return decisions, n_rand3
