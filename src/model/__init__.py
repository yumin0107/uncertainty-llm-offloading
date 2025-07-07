from .base import BaseModel
from .huggingface_model import HuggingfaceModel


def get_model(model_name: str, **kwargs) -> BaseModel:
    return HuggingfaceModel(model_name=model_name, **kwargs)
