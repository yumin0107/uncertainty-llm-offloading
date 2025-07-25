from typing import List
from basestation import User, EdgeServer


def none_offloading(us: List[User], es: List[EdgeServer]) -> List[List[int]]:
    return [[0 for _ in range(len(es))] for _ in range(len(us))]
