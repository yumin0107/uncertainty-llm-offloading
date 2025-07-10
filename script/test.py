import sys
import os
import argparse
import random
from pprint import pprint
from copy import deepcopy

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
import pandas as pd

# Communication
from basestation import User, EdgeServer
from user_association import uncertainty_aware_offloading
from user_association_random import random_offloading
from utils import (
    estimate_workload,
    generate_rayleigh_coeffs,
    bit_size_text,
    is_correct,
    is_offloading,
)
from config import (
    M,
    D,
    FIXED_ES,
    BANDWIDTH,
    TRANSMIT_POWER,
    NOISE_POWER,
    LOCAL_COMPUTE_CAP,
    EDGE_COMPUTE_CAP,
    SLM,
    LLM,
    K,
)

# LLM
from model.huggingface_model import HuggingfaceModel
from model import get_model


def generate_es(M: int) -> list[EdgeServer]:
    es_pos = tf.constant(FIXED_ES, dtype=tf.float32)
    return [EdgeServer(i, es_pos[i], BANDWIDTH, EDGE_COMPUTE_CAP) for i in range(M)]


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

    ds = dataset.shuffle(seed=seed).select(range(N))

    u_pos = tf.constant(np.random.uniform(0, D, (N, 2)), dtype=tf.float32)  # N x 2
    es_pos = tf.constant(FIXED_ES, dtype=tf.float32)  # M x 2
    d = tf.norm(u_pos[:, None, :] - es_pos[None, :, :], axis=-1)  # N x M

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

        h_list = generate_rayleigh_coeffs(M, d[i])

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
    correct_count_random = 0
    total_delay_SLM = 0
    total_delay_offloading = 0
    total_delay_random = 0
    total_delay_LLM = 0
    total_count = 0

    for i in range(n_run):
        print(f"iteration {i+1}")
        seed = b_seed + i

        # initailize user & edge server
        es = generate_es(M)
        es_ua = deepcopy(es)
        es_rand = deepcopy(es)
        users = generate_users(
            main_args.N,
            M,
            user_model,
            dataset,
            seed,
            TRANSMIT_POWER,
            NOISE_POWER,
            LOCAL_COMPUTE_CAP,
        )

        # ua
        decisions = uncertainty_aware_offloading(users, es_ua, main_args.tau)
        decisions_random = random_offloading(users, es_rand)

        # delay & accuracy
        for u in users:
            output_SLM, inf_delay_SLM = user_model.generate(u.input)
            output_LLM, inf_delay_LLM = edge_model.generate(u.input)
            pred_SLM = output_SLM[len(u.input) :]
            pred_LLM = output_LLM[len(u.input) :]

            # offloading
            if is_offloading(u.id, decisions):
                for e in es_ua:
                    if u in e.users:
                        u.t_comp = inf_delay_LLM
                        u.t_comm = e.total_comm_delay(u)
                        u.prediction = pred_LLM
                        break
            else:
                u.t_comp = inf_delay_SLM
                u.t_comm = 0
                u.prediction = pred_SLM

            # random
            if is_offloading(u.id, decisions_random):
                for e in es_rand:
                    if u in e.users:
                        total_delay_random += inf_delay_LLM + e.total_comm_delay(u)
                        correct_count_random += is_correct(pred_LLM, u.label)
                        break
            else:
                total_delay_random += inf_delay_SLM
                correct_count_random += is_correct(pred_SLM, u.label)

            total_delay_SLM += inf_delay_SLM
            total_delay_offloading += u.t_comp + u.t_comm
            total_delay_LLM += inf_delay_LLM + e.total_comm_delay(u)

            # print(inf_delay_SLM)
            # print(e.total_comm_delay(u))
            # print(inf_delay_LLM)

            correct_count_SLM += is_correct(pred_SLM, u.label)
            correct_count_offloading += is_correct(u.prediction, u.label)
            correct_count_LLM += is_correct(pred_LLM, u.label)
            total_count += 1

    accuracy_SLM = correct_count_SLM / total_count * 100
    accuracy_offloading = correct_count_offloading / total_count * 100
    accuracy_random = correct_count_random / total_count * 100
    accuracy_LLM = correct_count_LLM / total_count * 100

    delay_SLM = total_delay_SLM / (main_args.N * n_run)
    delay_offloading = total_delay_offloading / (main_args.N * n_run)
    delay_random = total_delay_random / (main_args.N * n_run)
    delay_LLM = total_delay_LLM / (main_args.N * n_run)

print("\n=== Accuracy Metrics ===")
print(f"SLM Accuracy        : {accuracy_SLM:.2f}%")
print(f"Offloading Accuracy : {accuracy_offloading:.2f}%")
print(f"Random Accuracy     : {accuracy_random:.2f}%")
print(f"LLM Accuracy        : {accuracy_LLM:.2f}%")

print("\n=== Inference Delay (per sample) ===")
print(f"SLM Delay           : {delay_SLM*1000:.3f} ms")
print(f"Offloading Delay    : {delay_offloading*1000:.3f} ms")
print(f"Random Delay        : {delay_random*1000:.3f} ms")
print(f"LLM Delay           : {delay_LLM*1000:.3f} ms")

df = pd.DataFrame(
    [
        {
            "accuracy_SLM": accuracy_SLM,
            "accuracy_offloading": accuracy_offloading,
            "accuracy_random": accuracy_random,
            "accuracy_LLM": accuracy_LLM,
            "delay_SLM": delay_SLM,
            "delay_offloading": delay_offloading,
            "delay_random": delay_random,
            "delay_LLM": delay_LLM,
        }
    ]
)
output_dir = os.path.join(os.path.dirname(__file__), "..", "result")
out_fname = os.path.join(output_dir, f"results_N{main_args.N}_tau{main_args.tau}.csv")
df.to_csv(out_fname, index=False)
