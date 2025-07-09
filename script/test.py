import sys
import os
import argparse
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# Third-party
import numpy as np
import torch
import tensorflow as tf
from datasets import load_dataset

# Communication
from basestation import User, EdgeServer
from user_association import uncertainty_aware_offloading
from utils import (
    estimate_worklad,
    measure_inference_delay,
    generate_rayleigh_coeffs,
    bit_size_text,
)
from config import (
    BANDWIDTH,
    TRANSMIT_POWER,
    NOISE_POWER,
    LOCAL_COMPUTE_CAP,
    EDGE_COMPUTE_CAP,
    MAX_COMPUTE_PER_USER,
    SLM,
    LLM,
    K,
)

# LLM
from model.huggingface_model import HuggingfaceModel
from model import get_model


def generate_users(
    N: int,
    model: HuggingfaceModel,
    dataset,
    seed: int,
    P: float,
    sigma2: float,
    C_L: float,
) -> list[User]:

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)

    ds = dataset.shuffle().select(range(N))
    h_list = generate_rayleigh_coeffs(N)

    users = []
    for i, sample in enumerate(ds):
        text = sample["text"]
        label = sample["label"]

        tokens = model.tokenizer.encode(text, truncation=True)
        D_i = bit_size_text(text)

        topk = model.topk_probs(text, k=K)
        p_topk = [prob for _, prob in topk]

        W_i_SLM = estimate_worklad(len(tokens), SLM)
        W_i_LLM = estimate_worklad(len(tokens), LLM)

        users.append(
            User(
                id=i + 1,
                D=D_i,
                h=h_list[i],
                P=P,
                sigma2=sigma2,
                input=text,
                label=label,
                W_i_SLM=W_i_SLM,
                W_i_LLM=W_i_LLM,
                C_i_L=C_L,
                p_k=p_topk,
            )
        )
    return users


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--tau", type=float, default=0.5)

    return parser


if __name__ == "__main__":
    parser = get_parser()
    main_args = parser.parse_args()

    # load model & dataset
    user_model = get_model(SLM)
    edge_model = get_model(LLM)
    ##############################################
    ################# 경로 수정 필요 ###############
    dataset = load_dataset("google-research-datasets/natural_questions", split="train")
    ##############################################
    ##############################################

    n_run = 50
    b_seed = 42
    correct_count = 0
    total_count = 0
    total_delay = 0

    for i in range(n_run):
        seed = b_seed + i
        # initailize user & edge server
        users = generate_users(
            main_args.N,
            user_model,
            dataset,
            seed,
            TRANSMIT_POWER,
            NOISE_POWER,
            LOCAL_COMPUTE_CAP,
        )
        es = EdgeServer(BANDWIDTH, EDGE_COMPUTE_CAP, MAX_COMPUTE_PER_USER)
        for u in users:
            es.add_user(u)

        # ua
        decisions = uncertainty_aware_offloading(es, main_args.tau)

        # delay & accuracy
        for u in users:
            if decisions[u.id] == 1:
                u.t_comp = measure_inference_delay(edge_model, u.input, max_length=100)
                u.t_comm = es.total_comm_delay(decisions, u)
                u.prediction = edge_model.generate(u.input)
            else:
                u.t_comp = measure_inference_delay(user_model, u.input, max_length=100)
                u.t_comm = 0
                u.prediction = user_model.generate(u.input)

            u.t_total = u.t_comp + u.t_comm
            total_delay += u.t_total

            u.is_correct = int(u.prediction.strip() == u.label.strip())
            correct_count += u.is_correct
            total_count += 1

    accuracy = correct_count / total_count * 100
    delay = total_delay / (main_args.N * n_run)

    print(f"\n=== Final Accuracy over {total_count} samples: {accuracy:.2f}% ===\n")
    print(f"\n=== Final Delay: {delay*1000:.3f} ms ===\n")
