from .base import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Tuple
import sys
import time


class HuggingfaceModel(BaseModel):
    def __init__(self, model_name: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def generate(
        self, prompt: str, k: int
    ) -> Tuple[List[Tuple[str, float]], float, str]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits  # (1, L, V)
        next_logits = logits[0, -1]  # (V,)
        probs = torch.softmax(next_logits, dim=-1)
        topk_p, topk_i = torch.topk(probs, k * 3)
        prob_dict = {}
        for idx, prob in zip(topk_i.tolist(), topk_p.tolist()):
            token = self.tokenizer.decode([idx]).strip().lower()
            if token == "":
                continue
            if token in prob_dict:
                prob_dict[token] += float(prob)
            else:
                prob_dict[token] = float(prob)
        merged_topk = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:k]
        total = sum(p for _, p in merged_topk)
        normalized_topk = [(token, p / total) for token, p in merged_topk]

        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=5, do_sample=False)
        torch.cuda.synchronize()
        end = time.time()

        return (
            normalized_topk,
            end - start,
            self.tokenizer.decode(out[0], skip_special_tokens=True),
        )
