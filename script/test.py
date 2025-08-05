import sys
import os
import argparse
import random
import gc
import json

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
from user_association.user_association_random import random1_offloading
from user_association.user_association_dmin import dmin_offloading
from user_association.user_association_ga import ga_offloading
from utils import (
    generate_rayleigh_coeffs,
    bit_size_text,
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
from model import get_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)


def generate_es(M: int) -> list[EdgeServer]:
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
    data,
) -> list[User]:

    u_pos = tf.constant(np.random.uniform(0, D, (N, 2)), dtype=tf.float32)  # N x 2
    es_pos = tf.constant(FIXED_ES, dtype=tf.float32)  # M x 2
    d = tf.norm(u_pos[:, None, :] - es_pos[None, :, :], axis=-1)  # N x M
    h = generate_rayleigh_coeffs(N, M, d)  # N x M

    users = []
    for i in range(N):
        ds = data[i]

        users.append(
            User(
                id=ds["id"],
                D=ds["D"],
                h=h,
                P=ds["P"],
                sigma2=ds["sigma2"],
                input=ds["input"],
                output_slm=ds["output_slm"],
                output_llm=ds["output_llm"],
                label=ds["label"],
                t_comp_slm=ds["t_comp_slm"],
                t_comp_llm=ds["t_comp_llm"],
                C_i_L=ds["C_i_L"],
                p_k=ds["p_k"],
            )
        )
    return users


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--tau", type=float, default=0.5)

    return parser


def main():
    parser = get_parser()
    main_args = parser.parse_args()

    with open("data.json", "r") as f:
        data = json.load(f)

    n_run = 500
    b_seed = 42
    total_delay_total_list = []
    total_delay_local_list = []
    total_t_comm_list = []
    total_t_comp_list = []
    total_accuracy_list = []
    total_n_uao = 0
    total_n_dmin = 0
    total_t_uao = 0

    for i in range(n_run):
        data_i = data[i]
        print(f"iteration {i+1}")
        seed = b_seed + i
        set_seed(seed)

        # initailize user & edge server
        es_none = generate_es(M)
        es_uao = generate_es(M)
        es_rand_1 = generate_es(M)
        es_all = generate_es(M)
        es_dmin = generate_es(M)
        users = generate_users(N=main_args.N, M=M, data=data_i)

        # ua
        decisions_none = none_offloading(users, es_none)
        decisions_uao, n_uao, t_uao = uncertainty_aware_offloading(
            users, es_uao, main_args.tau
        )
        decisions_rand_1 = random1_offloading(users, es_rand_1, n_uao)
        decisions_all = all_offloading(users, es_all)
        decisions_dmin, n_dmin = dmin_offloading(users, es_dmin)

        decision_list = [
            decisions_none,
            decisions_uao,
            decisions_rand_1,
            decisions_all,
            decisions_dmin,
        ]
        es_list = [es_none, es_uao, es_rand_1, es_all, es_dmin]

        delay_total_list = []
        delay_local_list = []
        t_comm_list = []
        t_comp_list = []
        accuracy_list = []

        for decisions, es in zip(decision_list, es_list):
            delay_total, delay_local, t_comm, t_comp, accuracy = calc_delay_accuracy(
                users, es, decisions
            )  # [ms, ms, ms, ms, %]
            delay_total_list.append(delay_total)
            delay_local_list.append(delay_local)
            t_comm_list.append(t_comm)
            t_comp_list.append(t_comp)
            accuracy_list.append(accuracy)

        total_delay_total_list.append(delay_total_list)
        total_delay_local_list.append(delay_local_list)
        total_t_comm_list.append(t_comm_list)
        total_t_comp_list.append(t_comp_list)
        total_accuracy_list.append(accuracy_list)
        total_n_uao += n_uao
        total_n_dmin += n_dmin
        total_t_uao += t_uao

    delay_total_avg = np.mean(total_delay_total_list, axis=0)
    delay_local_avg = np.mean(total_delay_local_list, axis=0)
    t_comm_avg = np.mean(total_t_comm_list, axis=0)
    t_comp_avg = np.mean(total_t_comp_list, axis=0)
    accuracy_avg = np.mean(total_accuracy_list, axis=0)
    n_uao_avg = total_n_uao / n_run
    n_dmin_avg = total_n_dmin / n_run
    t_uao_avg = total_t_uao / n_run

    methods = ["local_all", "goa", "random_1", "edge_all", "dmin"]

    data = {}

    for metric_name, values in [
        ("accuracy", accuracy_avg),
        ("delay_total", delay_total_avg),
        ("delay_local", delay_local_avg),
        ("t_comm", t_comm_avg),
        ("t_comp", t_comp_avg),
    ]:
        for method, val in zip(methods, values):
            data[f"{metric_name}_{method}"] = val
    data["N"] = main_args.N
    data["N_goa"] = n_uao_avg
    data["N_dmin"] = n_dmin_avg
    data["t_goa"] = t_uao_avg

    df = pd.DataFrame([data])

    output_dir = os.path.join(os.path.dirname(__file__), "..", "result/data")
    out_fname = os.path.join(
        output_dir, f"results_N{main_args.N}_tau{main_args.tau}.csv"
    )
    df.to_csv(out_fname, index=False)


def extract_data():
    parser = get_parser()
    main_args = parser.parse_args()

    gc.collect()
    torch.cuda.empty_cache()
    user_model = get_model(SLM)
    edge_model = get_model(LLM)

    ds = load_dataset("Muennighoff/babi", split="train")
    exclude_tasks = [7, 8, 19]
    dataset = ds.filter(lambda example: example["task"] not in exclude_tasks)

    n_run = 500
    b_seed = 42

    data = []  # [iteration user user_data]
    for i in range(n_run):
        print(f"iteration {i+1}")
        data_i = []
        seed = b_seed + i

        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        torch.manual_seed(seed)

        ds = dataset.shuffle(seed=seed).select(range(main_args.N))

        u_pos = tf.constant(
            np.random.uniform(0, D, (main_args.N, 2)), dtype=tf.float32
        )  # N x 2
        es_pos = tf.constant(FIXED_ES, dtype=tf.float32)  # M x 2
        d = tf.norm(u_pos[:, None, :] - es_pos[None, :, :], axis=-1)  # N x M

        for u in range(main_args.N):
            passage = ds[u]["passage"]
            question = ds[u]["question"]
            answer = ds[u]["answer"]
            text = (
                f"Instruction: Answer with only one word. No explanation.\n"
                f"Context: {passage.strip()}\n"
                f"Question: {question.strip()}\n"
                f"Answer(only one word):"
            )

            D_i = bit_size_text(text)
            C_i_L = LOCAL_COMPUTE_CAP * np.random.uniform(0.5, 1.5)

            topk, t_comp_slm, output_slm = user_model.generate(text, k=K)
            _, t_comp_llm, output_llm = edge_model.generate(text, k=K)
            p_topk = [prob for _, prob in topk]

            t_comp_slm /= C_i_L / LOCAL_COMPUTE_CAP

            data_i.append(
                {
                    "id": u,
                    "D": D_i,
                    "P": TRANSMIT_POWER,
                    "sigma2": NOISE_POWER,
                    "input": text,
                    "output_slm": output_slm,
                    "output_llm": output_llm,
                    "label": answer,
                    "t_comp_slm": t_comp_slm,
                    "t_comp_llm": t_comp_llm,
                    "C_i_L": C_i_L,
                    "p_k": p_topk,
                }
            )
        data.append(data_i)
    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

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
