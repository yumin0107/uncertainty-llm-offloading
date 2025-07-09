from typing import Dict, List, Tuple
from basestation import User, EdgeServer
from copy import deepcopy


def uncertainty_aware_offloading(
    us: List[User], es: List[EdgeServer], tau: float
) -> Dict[int, Dict[int, int]]:
    decisions: Dict[int, Dict[int, int]] = {u.id: {e.id: 0 for e in es} for u in us}
    us_copy = deepcopy(us)

    while max(u.uncertainty for u in us_copy) > tau:
        u_p = max(us_copy, key=lambda i: i.uncertainty)
        delta_p: Dict[EdgeServer, float] = {}
        for e in es:
            B_j = e.bandwidth_allocation(len(e.users) + 1)
            C_j_ES = e.compute_allocation(len(e.users) + 1)

            t_j_comm = u_p.comm_delay(B_j, e.id)
            t_j_comp_ES = e.edge_comp_delay(u_p, C_j_ES)
            t_j_comp_L = u_p.local_comp_delay()

            delta_p[e] = t_j_comm + t_j_comp_ES - t_j_comp_L
        e_p = min(delta_p, key=lambda e: delta_p[e])

        decisions[u_p.id][e_p.id] = 1
        e_p.add_user(u_p)
        e_p.B_j = e_p.bandwidth_allocation(len(e_p.users))
        us_copy.remove(u_p)

    delta: Dict[Tuple[User, EdgeServer], float] = {}
    for u in us_copy:
        for e in es:
            B_j = e.bandwidth_allocation(len(e.users) + 1)
            C_j_ES = e.compute_allocation(len(e.users) + 1)

            t_j_comm = u.comm_delay(B_j, e.id)
            t_j_comp_ES = e.edge_comp_delay(u, C_j_ES)
            t_j_comp_L = u.local_comp_delay()

            delta[(u, e)] = t_j_comm + t_j_comp_ES - t_j_comp_L

    while True:
        candidates = [(pair, gap) for pair, gap in delta.items() if gap < 0]
        if not candidates:
            break

        (u_p, e_p), _ = min(candidates, key=lambda item: item[1])
        decisions[u_p.id][e_p.id] = 1
        e_p.add_user(u_p)
        e_p.B_j = e_p.bandwidth_allocation(len(e_p.users))
        us_copy.remove(u_p)

        for u in us_copy:
            for e in es:
                B_j = e.bandwidth_allocation(len(e.users) + 1)
                C_j_ES = e.compute_allocation(len(e.users) + 1)

                t_j_comm = u.comm_delay(B_j, e.id)
                t_j_comp_ES = e.edge_comp_delay(u, C_j_ES)
                t_j_comp_L = u.local_comp_delay()

                delta[(u, e)] = t_j_comm + t_j_comp_ES - t_j_comp_L

    return decisions
