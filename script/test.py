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
    estimate_workload,
    generate_rayleigh_coeffs,
    bit_size_text,
    is_correct,
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


def generate_es(M: int) -> list[EdgeServer]:
    return [
        EdgeServer(i, BANDWIDTH, EDGE_COMPUTE_CAP, MAX_COMPUTE_PER_USER)
        for i in range(M)
    ]


def generate_users(
    N: int,
    M: int,
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

    users = []
    for i in range(N):
        passage = ds[i]["passage"]
        question = ds[i]["question"]
        answer = ds[i]["answer"]
        text = (
            f"Instruction: Answer with only one word. No explanation."
            f"Context: {passage.strip()}\n"
            f"Question: {question.strip()}\n"
            f"Answer(only one word):"
        )

        tokens = model.tokenizer.encode(text, truncation=True)
        D_i = bit_size_text(text)

        topk = model.topk_probs(text, k=K)
        p_topk = [prob for _, prob in topk]

        W_i_SLM = estimate_workload(len(tokens), SLM)
        W_i_LLM = estimate_workload(len(tokens), LLM)

        h_list = generate_rayleigh_coeffs(M)

        users.append(
            User(
                id=i,
                D=D_i,
                h=h_list,
                P=P,
                sigma2=sigma2,
                input=text,
                label=answer,
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
    parser.add_argument("--M", type=int, default=3)
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
    dataset = load_dataset("Muennighoff/babi", split="train")
    ##############################################
    ##############################################

    n_run = 10
    b_seed = 42
    correct_count_SLM = 0
    correct_count_LLM = 0
    correct_count_offloading = 0
    total_delay_SLM = 0
    total_delay_offloading = 0
    total_delay_LLM = 0
    total_count = 0

    for i in range(n_run):
        seed = b_seed + i
        # initailize user & edge server
        es = generate_es(main_args.M)
        users = generate_users(
            main_args.N,
            main_args.M,
            user_model,
            dataset,
            seed,
            TRANSMIT_POWER,
            NOISE_POWER,
            LOCAL_COMPUTE_CAP,
        )

        # ua
        decisions = uncertainty_aware_offloading(users, es, main_args.tau)

        # delay & accuracy
        for e in es:
            for u in e.users:
                pred_SLM, inf_delay_SLM = user_model.generate(u.input)
                pred_LLM, inf_delay_LLM = edge_model.generate(u.input)
                if decisions[u.id] == 1:
                    u.t_comp = inf_delay_LLM
                    u.t_comm = e.total_comm_delay(u)
                    u.prediction = pred_LLM
                else:
                    u.t_comp = inf_delay_SLM
                    u.t_comm = 0
                    u.prediction = pred_SLM

                total_delay_SLM += inf_delay_SLM
                total_delay_offloading += u.t_comp + u.t_comm
                total_delay_LLM += inf_delay_LLM

                correct_count_SLM += is_correct(pred_SLM, u.label)
                correct_count_offloading += is_correct(u.prediction, u.label)
                correct_count_LLM += is_correct(pred_LLM, u.label)
                total_count += 1

    accuracy_SLM = correct_count_SLM / total_count * 100
    accuracy_offloading = correct_count_offloading / total_count * 100
    accuracy_LLM = correct_count_LLM / total_count * 100

    delay_SLM = total_delay_SLM / (main_args.N * n_run)
    delay_offloading = total_delay_offloading / (main_args.N * n_run)
    inf_delay_LLM = total_delay_LLM / (main_args.N * n_run)

print("\n=== Accuracy Metrics ===")
print(f"SLM Accuracy        : {accuracy_SLM:.2f}%")
print(f"Offloading Accuracy : {accuracy_offloading:.2f}%")
print(f"LLM Accuracy        : {accuracy_LLM:.2f}%")

print("\n=== Inference Delay (per sample) ===")
print(f"SLM Delay           : {delay_SLM*1000:.3f} ms")
print(f"Offloading Delay    : {delay_offloading*1000:.3f} ms")
print(f"LLM Delay           : {inf_delay_LLM*1000:.3f} ms")
