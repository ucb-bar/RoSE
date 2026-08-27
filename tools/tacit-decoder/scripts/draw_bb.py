from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

# Configure the inputs you want to compare.
INPUT_FILES: List[str] = [
    "/scratch/iansseijelly/spec-path-profiles/mcf/trace.vbb.csv",
    "/scratch/iansseijelly/spec-path-profiles/leela/trace.vbb.csv",
    "/scratch/iansseijelly/spec-path-profiles/x264/trace.vbb.csv",
    "/scratch/iansseijelly/spec-path-profiles/exchange2/trace.vbb.csv",
]
LABELS = [
    "mcf",
    "leela",
    "x264",
    "exchange2",
]
TOP_N = 10  # how many entries to take from each file
OUTPUT_FILE = "trace.multi.std.png"


def load_shares(input_path: str, top_n: int) -> Tuple[List[float], str]:
    df = pd.read_csv(input_path, skipinitialspace=True)
    if df.empty:
        raise SystemExit(f"No data found in {input_path}")

    label_column = None
    for candidate in ("path", "bb"):
        if candidate in df.columns:
            label_column = candidate
            break
    if label_column is None:
        raise SystemExit(f"{input_path}: expected a 'path' or 'bb' column.")

    if "total_time" not in df.columns:
        if {"mean", "count"}.issubset(df.columns):
            df["total_time"] = df["mean"] * df["count"]
        else:
            raise SystemExit(
                f"{input_path}: need total_time or both mean and count columns."
            )

    total_time_sum = df["total_time"].sum()
    if total_time_sum <= 0:
        raise SystemExit(f"{input_path}: total_time sums to zero; nothing to plot.")

    df["share"] = df["netvar"] / total_time_sum
    df = df.sort_values(by="share", ascending=False)
    top_df = df.head(top_n).copy()
    shares = top_df["share"].clip(lower=0).tolist()
    return shares, input_path


def plot_multi(datasets: List[Tuple[str, List[float]]]) -> None:
    plt.rc('font', size=18)
    fig, ax = plt.subplots(figsize=(12, 1.2 * len(datasets) + 1))
    colors = plt.cm.tab10(range(TOP_N))

    y_positions = list(range(len(datasets)))
    for y, (file_label, shares) in enumerate(datasets):
        left = 0.0
        for rank, share in enumerate(shares):
            if share <= 0:
                continue
            ax.barh(y, share, left=left, color=colors[rank % len(colors)])
            # ax.text(
            #     left + share / 2,
            #     y,
            #     f"{share * 100:.1f}%",
            #     ha="center",
            #     va="center",
            #     fontsize=8,
            #     color="white" if share > 0.08 else "black",
            # )
            left += share
        if left < 1.0:
            remainder = 1.0 - left
            ax.barh(y, remainder, left=left, color="#f0f0f0")
            ax.text(
                left + remainder / 2,
                y,
                f"{remainder * 100:.1f}%",
                ha="center",
                va="center",
                fontsize=18,
            )

    ax.set_xlim(0, 1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("share = net variration / total_time")
    ax.set_title(f"Net variration time share for top {TOP_N} entries across benchmarks")
    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    datasets: List[Tuple[str, List[float]]] = []
    for input_path in INPUT_FILES:
        shares, label_column = load_shares(input_path, TOP_N)
        datasets.append((input_path, shares))

    if not datasets:
        raise SystemExit("No input files configured; update INPUT_FILES.")

    plot_multi(datasets)


if __name__ == "__main__":
    main()
