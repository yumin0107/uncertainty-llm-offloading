import sys
import os
import argparse
import random
import gc
from pprint import pprint

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
from user_association.user_association_ua import uncertainty_aware_offloading
from user_association.user_association_none import none_offloading
from user_association.user_association_all import all_offloading
from user_association.user_association_random import (
    random1_offloading,
    random2_offloading,
    random3_offloading,
)
from user_association.user_association_dmin import dmin_offloading
from utils import (
    generate_rayleigh_coeffs,
    bit_size_text,
    is_correct,
    is_offloading,
    calc_delay_accuracy,
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
    MAX_COMPUTE_PER_USER,
    SLM,
    LLM,
    K,
)

# LLM
from model.huggingface_model import HuggingfaceModel
from model import get_model


def generate_es(M: int) -> list[EdgeServer]:
    np.random.seed(seed)
    es_pos = tf.constant(FIXED_ES, dtype=tf.float32)
    return [
        EdgeServer(
            i,
            es_pos[i],
            BANDWIDTH,
            EDGE_COMPUTE_CAP * np.random.uniform(0.6, 1.4),
            MAX_COMPUTE_PER_USER,
        )
        for i in range(M)
    ]


def generate_users(
    N: int,
    M: int,
    user_model: HuggingfaceModel,
    edge_model: HuggingfaceModel,
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
            f"Instruction: Answer with only one word. No explanation.\n"
            f"Context: {passage.strip()}\n"
            f"Question: {question.strip()}\n"
            f"Answer(only one word):"
        )

        D_i = bit_size_text(text)
        C_i_L = C_L * np.random.uniform(0.5, 1.5)

        topk, t_comp_slm, output_slm = user_model.generate(text, k=K)
        _, t_comp_llm, output_llm = edge_model.generate(text, k=K)
        p_topk = [prob for _, prob in topk]

        t_comp_slm /= C_i_L / C_L

        h_list = generate_rayleigh_coeffs(M, d[i])

        users.append(
            User(
                id=i,
                D=D_i,
                h=h_list,
                P=P,
                sigma2=sigma2,
                input=text,
                output_slm=output_slm,
                output_llm=output_llm,
                label=answer,
                t_comp_slm=t_comp_slm,
                t_comp_llm=t_comp_llm,
                C_i_L=C_i_L,
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
    gc.collect()
    torch.cuda.empty_cache()
    user_model = get_model(SLM)
    edge_model = get_model(LLM)

    ds = load_dataset("Muennighoff/babi", split="train")
    exclude_tasks = [7, 8, 19]
    dataset = ds.filter(lambda example: example["task"] not in exclude_tasks)

    n_run = 100
    b_seed = 42
    total_accuracy_list = []  # [SLM, uao, random, LLM]
    total_delay_list = []  # [SLM, uao, random, LLM]

    for i in range(n_run):
        print(f"iteration {i+1}")
        seed = b_seed + i

        # initailize user & edge server
        es_none = generate_es(M)
        es_uao = generate_es(M)
        es_rand_1 = generate_es(M)
        es_rand_2 = generate_es(M)
        es_rand_3 = generate_es(M)
        es_all = generate_es(M)
        es_dmin = generate_es(M)
        users = generate_users(
            main_args.N,
            M,
            user_model,
            edge_model,
            dataset,
            seed,
            TRANSMIT_POWER,
            NOISE_POWER,
            LOCAL_COMPUTE_CAP,
        )

        # ua
        decisions_none = none_offloading(users, es_none)
        decisions_uao, n_uao = uncertainty_aware_offloading(
            users, es_uao, main_args.tau
        )
        decisions_rand_1 = random1_offloading(users, es_rand_1, n_uao)
        decisions_rand_2, n_rand_2 = random2_offloading(users, es_rand_2)
        decisions_rand_3, n_rand_3 = random3_offloading(users, es_rand_3)
        decisions_all = all_offloading(users, es_all)
        decisions_dmin, n_dmin = dmin_offloading(users, es_dmin)

        d_none, a_none = calc_delay_accuracy(users, es_none, decisions_none)
        d_uao, a_uao = calc_delay_accuracy(users, es_uao, decisions_uao)
        d_rand_1, a_rand_1 = calc_delay_accuracy(users, es_rand_1, decisions_rand_1)
        d_rand_2, a_rand_2 = calc_delay_accuracy(users, es_rand_2, decisions_rand_2)
        d_rand_3, a_rand_3 = calc_delay_accuracy(users, es_rand_3, decisions_rand_3)
        d_all, a_all = calc_delay_accuracy(users, es_all, decisions_all)
        d_dmin, a_dmin = calc_delay_accuracy(users, es_dmin, decisions_dmin)

        total_delay_list.append(
            [d_none, d_uao, d_rand_1, d_rand_2, d_rand_3, d_all, d_dmin]
        )
        total_accuracy_list.append(
            [a_none, a_uao, a_rand_1, a_rand_2, a_rand_3, a_all, a_dmin]
        )

    accuracy_avg = np.mean(total_accuracy_list, axis=0)
    delay_avg = np.mean(total_delay_list, axis=0)

"""
print("\n=== Accuracy Metrics ===")
print(f"{'Accuracy_Local_All (n=0)':<35}: {accuracy_avg[0]:>6.2f}%")
print(f"{'Accuracy_UAO       (n='+str(n_uao)+')':<35}: {accuracy_avg[1]:>6.2f}%")
print(f"{'Accuracy_Random_1  (n='+str(n_uao)+')':<35}: {accuracy_avg[2]:>6.2f}%")
print(f"{'Accuracy_Random_2  (n='+str(n_rand_2)+')':<35}: {accuracy_avg[3]:>6.2f}%")
print(f"{'Accuracy_Random_3  (n='+str(n_rand_3)+')':<35}: {accuracy_avg[4]:>6.2f}%")
print(f"{'Accuracy_Edge_All  (n='+str(main_args.N)+')':<35}: {accuracy_avg[5]:>6.2f}%")
print(f"{'Accuracy_dmin      (n='+str(n_dmin)+')':<35}: {accuracy_avg[6]:>6.2f}%")

print("\n=== Inference Delay (per sample) ===")
print(f"{'Delay_Local_All':<35}: {delay_avg[0]:>7.3f} ms")
print(f"{'Delay_UAO':<35}: {delay_avg[1]:>7.3f} ms")
print(f"{'Delay_Random_1':<35}: {delay_avg[2]:>7.3f} ms")
print(f"{'Delay_Random_2':<35}: {delay_avg[3]:>7.3f} ms")
print(f"{'Delay_Random_3':<35}: {delay_avg[4]:>7.3f} ms")
print(f"{'Delay_Edge_All':<35}: {delay_avg[5]:>7.3f} ms")
print(f"{'Delay_dmin':<35}: {delay_avg[6]:>7.3f} ms")
"""

df = pd.DataFrame(
    [
        {
            "accuracy_local_all": accuracy_avg[0],
            "accuracy_uao": accuracy_avg[1],
            "accuracy_random_1": accuracy_avg[2],
            "accuracy_random_2": accuracy_avg[3],
            "accuracy_random_3": accuracy_avg[4],
            "accuracy_edge_all": accuracy_avg[5],
            "accuracy_dmin": accuracy_avg[6],
            "delay_local_all": delay_avg[0],
            "delay_uao": delay_avg[1],
            "delay_random_1": delay_avg[2],
            "delay_random_2": delay_avg[3],
            "delay_random_3": delay_avg[4],
            "delay_edge_all": delay_avg[5],
            "delay_dmin": delay_avg[6],
            "N_uao": n_uao,
            "N_rand_2": n_rand_2,
            "N_rand_3": n_rand_3,
            "N_dmin": n_dmin,
            "N": main_args.N,
        }
    ]
)
output_dir = os.path.join(os.path.dirname(__file__), "..", "result/data")
out_fname = os.path.join(output_dir, f"results_N{main_args.N}_tau{main_args.tau}.csv")
df.to_csv(out_fname, index=False)
