import random
from typing import Dict, List, Tuple
from basestation import User, EdgeServer


def none_offloading(us: List[User], es: List[EdgeServer]) -> Dict[int, Dict[int, int]]:
    return {u.id: {e.id: 0 for e in es} for u in us}
