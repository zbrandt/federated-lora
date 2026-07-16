from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean, stdev
from pathlib import Path

# Drop any inherited backend (e.g. Colab/Jupyter sets MPLBACKEND to an inline
# backend) so matplotlib doesn't reject it at import time in this subprocess.
import os

os.environ.pop("MPLBACKEND", None)

import matplotlib

matplotlib.use("Agg")  # headless: this script only writes PNGs
import matplotlib.pyplot as plt


COLOR_STYLES = {
    "FEDIT / IID": {"color": "#1f77b4", "linestyle": "-"},
    "FEDIT / NONIID (alpha=0.1)": {"color": "#ff7f0e", "linestyle": "--"},
    "FFA / IID": {"color": "#2ca02c", "linestyle": "-"},
    "FFA / NONIID (alpha=0.1)": {"color": "#d62728", "linestyle": "--"},
}

DEFAULT_STYLE = {"color": "#7f7f7f", "linestyle": ":"}


def style_description(label: str) -> str:
    style = COLOR_STYLES.get(label, DEFAULT_STYLE)
    if style["linestyle"] == "--":
        suffix = "dashed line"
    elif style["linestyle"] == ":":
        suffix = "dotted line"
    else:
        suffix = "solid line"
    return f"{label} ({suffix})"


@dataclass(slots=True)
class RunData:
    label: str
    config: dict[str, object]
    history: list[dict[str, object]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot metrics from main.py JSON outputs")
    parser.add_argument("inputs", nargs="+", help="One or more JSON output files from main.py")
    parser.add_argument(
        "--output-dir",
        default="figures",
        help="Directory where figures will be written",
    )
    return parser.parse_args()


def load_run(path: Path) -> RunData:
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload["config"]
    history = payload["history"]
    method = str(config.get("method", path.stem)).upper()
    partition_strategy = str(config.get("partition_strategy", "iid")).upper()
    if partition_strategy == "NONIID":
        alpha = config.get("dirichlet_alpha")
        label = f"{method} / {partition_strategy} (alpha={alpha})"
    else:
        label = f"{method} / {partition_strategy}"
    return RunData(label=label, config=config, history=history)


def group_runs_by_label(runs: list[RunData]) -> dict[str, list[RunData]]:
    grouped: dict[str, list[RunData]] = defaultdict(list)
    for run in runs:
        grouped[run.label].append(run)
    return grouped


def metric_values(run: RunData, key: str) -> list[float]:
    return [float(entry[key]) for entry in run.history]


def align_series(runs: list[RunData], key: str) -> tuple[list[int], list[float], list[float]]:
    if not runs:
        return [], [], []

    max_rounds = max(len(run.history) for run in runs)
    x_values: list[int] = []
    means: list[float] = []
    stds: list[float] = []

    for round_index in range(max_rounds):
        values = [metric_values(run, key)[round_index] for run in runs if len(run.history) > round_index]
        x_values.append(round_index + 1)
        means.append(mean(values))
        stds.append(stdev(values) if len(values) > 1 else 0.0)

    return x_values, means, stds


def align_bytes_vs_metric(runs: list[RunData], x_key: str, y_key: str) -> tuple[list[float], list[float], list[float]]:
    if not runs:
        return [], [], []

    max_rounds = max(len(run.history) for run in runs)
    x_values: list[float] = []
    means: list[float] = []
    stds: list[float] = []

    for round_index in range(max_rounds):
        xs = [float(run.history[round_index][x_key]) for run in runs if len(run.history) > round_index]
        ys = [float(run.history[round_index][y_key]) for run in runs if len(run.history) > round_index]
        x_values.append(mean(xs))
        means.append(mean(ys))
        stds.append(stdev(ys) if len(ys) > 1 else 0.0)

    return x_values, means, stds


def plot_metric_by_round(
    runs: list[RunData],
    metric_key: str,
    title: str,
    y_label: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    for label, label_runs in group_runs_by_label(runs).items():
        style = COLOR_STYLES.get(label, DEFAULT_STYLE)
        x_values, y_values, y_stds = align_series(label_runs, metric_key)
        plt.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=2.5,
            label=style_description(label),
            color=style["color"],
            linestyle=style["linestyle"],
        )
        if any(y_stds):
            lower = [value - spread for value, spread in zip(y_values, y_stds, strict=True)]
            upper = [value + spread for value, spread in zip(y_values, y_stds, strict=True)]
            plt.fill_between(x_values, lower, upper, alpha=0.12, color=style["color"])
    plt.title(title)
    plt.xlabel("Round")
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_perplexity_vs_bytes(
    runs: list[RunData],
    title: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    for label, label_runs in group_runs_by_label(runs).items():
        style = COLOR_STYLES.get(label, DEFAULT_STYLE)
        x_values, y_values, y_stds = align_bytes_vs_metric(label_runs, "cumulative_uploaded_bytes", "perplexity")
        plt.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=2.5,
            label=style_description(label),
            color=style["color"],
            linestyle=style["linestyle"],
        )
        if any(y_stds):
            lower = [value - spread for value, spread in zip(y_values, y_stds, strict=True)]
            upper = [value + spread for value, spread in zip(y_values, y_stds, strict=True)]
            plt.fill_between(x_values, lower, upper, alpha=0.12, color=style["color"])
    plt.title(title)
    plt.xlabel("Cumulative uploaded bytes")
    plt.ylabel("Perplexity")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    runs = [load_run(Path(input_path)) for input_path in args.inputs]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_metric_by_round(runs, "train_loss", "Training Loss by Round", "Train loss", output_dir / "train_loss_by_round.png")
    plot_metric_by_round(runs, "eval_loss", "Validation Loss by Round", "Validation loss", output_dir / "eval_loss_by_round.png")
    plot_metric_by_round(runs, "perplexity", "Perplexity by Round", "Perplexity", output_dir / "perplexity_by_round.png")
    plot_metric_by_round(runs, "uploaded_bytes", "Uploaded Bytes by Round", "Uploaded bytes", output_dir / "uploaded_bytes_by_round.png")
    plot_perplexity_vs_bytes(runs, "Perplexity vs Cumulative Communication", output_dir / "perplexity_vs_bytes.png")

    print(f"Wrote figures to {output_dir.resolve()}")


if __name__ == "__main__":
    main()