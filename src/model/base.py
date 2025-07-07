from abc import ABC, abstractmethod
from typing import List, Tuple


class BaseModel(ABC):
    @abstractmethod
    def generate(self, prompt: str, **generate_kwargs) -> str: ...

    @abstractmethod
    def topk_probs(self, prompt: str, k: int) -> List[Tuple[str, float]]: ...
