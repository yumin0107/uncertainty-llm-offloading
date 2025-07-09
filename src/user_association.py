from typing import Dict, List
from basestation import User, EdgeServer


def uncertainty_aware_offloading(es: EdgeServer, tau: float) -> Dict[int, int]:
    decisions: Dict[int, int] = {u.id: 0 for u in es.users}
    offloaded = []
    remaining = []

    for u in es.users:
        if u.uncertainty > tau:
            decisions[u.id] = 1
            offloaded.append(u)
        else:
            remaining.append(u)

    es.B_i = es.bandwidth_allocation(len(offloaded))

    while remaining:
        B_i = es.bandwidth_allocation(len(offloaded) + 1)
        C_i_ES = es.compute_allocation(len(offloaded) + 1)

        deltas = []
        for u in remaining:
            t_i_comm = u.comm_delay(B_i)
            t_i_comp_ES = es.edge_comp_delay(u, C_i_ES)
            t_i_comp_L = u.local_comp_delay()

            delta = t_i_comm + t_i_comp_ES - t_i_comp_L
            deltas.append((u, delta))

        u_min, delta_min = min(deltas, key=lambda x: x[1])

        if delta_min >= 0:
            break
        else:
            decisions[u_min.id] = 1
            offloaded.append(u_min)
            remaining.remove(u_min)
            es.B_i = B_i

    return decisions
