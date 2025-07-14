from typing import Dict, List, Tuple
from basestation import User, EdgeServer


def all_offloading(us: List[User], es: List[EdgeServer]) -> Dict[int, Dict[int, int]]:
    decisions: Dict[int, Dict[int, int]] = {u.id: {e.id: 0 for e in es} for u in us}
    id_to_user: Dict[int, User] = {u.id: u for u in us}
    remaining_uids: List[int] = list(id_to_user.keys())

    while remaining_uids:
        u_p_id = max(remaining_uids, key=lambda uid: id_to_user[uid].uncertainty)
        u_p = id_to_user[u_p_id]

        delta_p: Dict[EdgeServer, float] = {}
        for e in es:
            B_j = e.bandwidth_allocation(len(e.users) + 1)
            C_j_ES = e.compute_allocation(len(e.users) + 1)

            t_j_comm = u_p.comm_delay(B_j, e.id)
            t_j_comp_ES = e.edge_comp_delay(u_p, C_j_ES)
            t_j_comp_L = u_p.t_comp_slm

            delta_p[e] = t_j_comm + t_j_comp_ES - t_j_comp_L

        e_p = min(delta_p, key=lambda e: delta_p[e])

        decisions[u_p.id][e_p.id] = 1
        e_p.add_user(u_p)
        e_p.B_j = e_p.bandwidth_allocation(len(e_p.users))
        e_p.C_j_ES = e_p.compute_allocation(len(e_p.users))

        remaining_uids.remove(u_p_id)

    return decisions
