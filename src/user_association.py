from typing import Dict, List, Tuple
from basestation import User, EdgeServer


def uncertainty_aware_offloading(
    us: List[User], es: List[EdgeServer], tau: float
) -> Dict[int, Dict[int, int]]:
    decisions: Dict[int, Dict[int, int]] = {u.id: {e.id: 0 for e in es} for u in us}
    id_to_user: Dict[int, User] = {u.id: u for u in us}
    remaining_uids: List[int] = list(id_to_user.keys())

    while remaining_uids:
        max_unc = max(id_to_user[uid].uncertainty for uid in remaining_uids)
        if max_unc <= tau:
            break
        u_p_id = max(remaining_uids, key=lambda uid: id_to_user[uid].uncertainty)
        u_p = id_to_user[u_p_id]

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

        remaining_uids.remove(u_p_id)

    delta: Dict[Tuple[int, int], float] = {}
    for uid in remaining_uids:
        u = id_to_user[uid]
        for e in es:
            B_j = e.bandwidth_allocation(len(e.users) + 1)
            C_j_ES = e.compute_allocation(len(e.users) + 1)

            t_j_comm = u.comm_delay(B_j, e.id)
            t_j_comp_ES = e.edge_comp_delay(u, C_j_ES)
            t_j_comp_L = u.local_comp_delay()

            delta[(uid, e.id)] = t_j_comm + t_j_comp_ES - t_j_comp_L

    while remaining_uids:
        candidates = [
            ((uid, eid), gap)
            for (uid, eid), gap in delta.items()
            if gap < 0 and uid in remaining_uids
        ]
        if not candidates:
            break

        (u_p_id, e_p_id), _ = min(candidates, key=lambda item: item[1])
        u_p = id_to_user[u_p_id]
        e_p = next(e for e in es if e.id == e_p_id)

        decisions[u_p.id][e_p.id] = 1
        e_p.add_user(u_p)
        e_p.B_j = e_p.bandwidth_allocation(len(e_p.users))
        remaining_uids.remove(u_p_id)

        for uid in remaining_uids:
            u = id_to_user[uid]
            for e in es:
                B_j = e.bandwidth_allocation(len(e.users) + 1)
                C_j_ES = e.compute_allocation(len(e.users) + 1)

                t_j_comm = u.comm_delay(B_j, e.id)
                t_j_comp_ES = e.edge_comp_delay(u, C_j_ES)
                t_j_comp_L = u.local_comp_delay()

                delta[(uid, e.id)] = t_j_comm + t_j_comp_ES - t_j_comp_L

    return decisions
