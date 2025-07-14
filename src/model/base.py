from abc import ABC, abstractmethod
from typing import List, Tuple


class BaseModel(ABC):
    @abstractmethod
    def generate(
        self, prompt: str, **generate_kwargs
    ) -> Tuple[List[Tuple[str, float]], float, str]: ...
