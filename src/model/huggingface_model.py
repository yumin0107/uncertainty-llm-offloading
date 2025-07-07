from .base import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Tuple


class HuggingfaceModel(BaseModel):
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        out = self.model.generate(**inputs, max_length=100)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def topk_probs(self, prompt: str, k: int) -> List[Tuple[str, float]]:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits  # (1, L, V)
        next_logits = logits[0, -1]  # (V,)
        probs = torch.softmax(next_logits, dim=-1)
        topk_p, topk_i = torch.topk(probs, k)
        topk_p = topk_p / topk_p.sum()  # normalize among top-k
        return [
            (self.tokenizer.decode([idx]), float(p))
            for idx, p in zip(topk_i.tolist(), topk_p.tolist())
        ]
