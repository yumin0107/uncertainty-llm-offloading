import os
import pandas as pd
import matplotlib.pyplot as plt
import glob

output_dir = os.path.join(os.path.dirname(__file__), "..", "result")

csv_files = glob.glob(os.path.join(output_dir, "results_*.csv"))

dataframes = []
for file in csv_files:
    basename = os.path.basename(file)
    parts = basename.replace(".csv", "").split("_")
    N = int(parts[1][1:])
    tau = float(parts[2][3:])

    df = pd.read_csv(file)
    df["N"] = N
    df["tau"] = tau
    dataframes.append(df)

full_df = pd.concat(dataframes, ignore_index=True)

plot_dir = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(plot_dir, exist_ok=True)


def plot_accuracy_vs_tau(df, fixed_N, save=True):
    sub = df[df["N"] == fixed_N].sort_values(by="tau").reset_index(drop=True)

    plt.figure()
    for method in ["local_all", "uao", "random", "edge_all"]:
        plt.plot(sub["tau"], sub[f"accuracy_{method}"], marker="o", label=method)

    plt.xlabel("tau")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs tau (N={fixed_N})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save:
        fname = f"accuracy_vs_tau_N{fixed_N}.png"
        plt.savefig(os.path.join(plot_dir, fname))

    plt.close()


def plot_accuracy_vs_N(df, fixed_tau, save=True):
    sub = (
        df[(df["tau"] == fixed_tau) & (df["N"] != 100)]
        .sort_values(by="N")
        .reset_index(drop=True)
    )

    plt.figure()
    for method in ["local_all", "uao", "random", "edge_all"]:
        plt.plot(sub["N"], sub[f"accuracy_{method}"], marker="o", label=method)

    plt.xlabel("N")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs N (tau={fixed_tau})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(range(150, sub["N"].max() + 1, 50))
    plt.tight_layout()

    if save:
        fname = f"accuracy_vs_N_tau{fixed_tau}.png"
        plt.savefig(os.path.join(plot_dir, fname))

    plt.close()


def plot_delay_vs_tau(df, fixed_N, save=True):
    sub = df[df["N"] == fixed_N].sort_values(by="tau").reset_index(drop=True)

    plt.figure()
    for method in ["local_all", "uao", "random", "edge_all"]:
        plt.plot(sub["tau"], sub[f"delay_{method}"], marker="o", label=method)

    plt.xlabel("tau")
    plt.ylabel("Delay")
    plt.title(f"Delay vs tau (N={fixed_N})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save:
        fname = f"delay_vs_tau_N{fixed_N}.png"
        plt.savefig(os.path.join(plot_dir, fname))

    plt.close()


def plot_delay_vs_N(df, fixed_tau, save=True):
    sub = (
        df[(df["tau"] == fixed_tau) & (df["N"] != 100)]
        .sort_values(by="N")
        .reset_index(drop=True)
    )

    plt.figure()
    for method in ["local_all", "uao", "random", "edge_all"]:
        plt.plot(sub["N"], sub[f"delay_{method}"], marker="o", label=method)

    plt.xlabel("N")
    plt.ylabel("Delay")
    plt.title(f"Delay vs N (tau={fixed_tau})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save:
        fname = f"delay_vs_N_tau{fixed_tau}.png"
        plt.savefig(os.path.join(plot_dir, fname))

    plt.close()


for tau_val in sorted(full_df["tau"].unique()):
    plot_accuracy_vs_N(full_df, fixed_tau=tau_val, save=True)
    plot_delay_vs_N(full_df, fixed_tau=tau_val, save=True)

for N_val in sorted(full_df["N"].unique()):
    plot_accuracy_vs_tau(full_df, fixed_N=N_val, save=True)
    plot_delay_vs_tau(full_df, fixed_N=N_val, save=True)
