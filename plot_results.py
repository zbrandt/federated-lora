from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


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
    label = str(config.get("method", path.stem)).upper()
    return RunData(label=label, config=config, history=history)


def series(run: RunData, key: str) -> list[float]:
    return [float(entry[key]) for entry in run.history]


def round_numbers(run: RunData) -> list[int]:
    return [int(entry["round_index"]) for entry in run.history]


def make_line_plot(
    runs: list[RunData],
    x_getter,
    y_getter,
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    for run in runs:
        plt.plot(x_getter(run), y_getter(run), marker="o", linewidth=2, label=run.label)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
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

    make_line_plot(
        runs,
        round_numbers,
        lambda run: series(run, "train_loss"),
        "Training Loss by Round",
        "Round",
        "Train loss",
        output_dir / "train_loss_by_round.png",
    )
    make_line_plot(
        runs,
        round_numbers,
        lambda run: series(run, "eval_loss"),
        "Validation Loss by Round",
        "Round",
        "Validation loss",
        output_dir / "eval_loss_by_round.png",
    )
    make_line_plot(
        runs,
        round_numbers,
        lambda run: series(run, "perplexity"),
        "Perplexity by Round",
        "Round",
        "Perplexity",
        output_dir / "perplexity_by_round.png",
    )
    make_line_plot(
        runs,
        round_numbers,
        lambda run: series(run, "uploaded_bytes"),
        "Uploaded Bytes by Round",
        "Round",
        "Uploaded bytes",
        output_dir / "uploaded_bytes_by_round.png",
    )
    make_line_plot(
        runs,
        lambda run: series(run, "cumulative_uploaded_bytes"),
        lambda run: series(run, "perplexity"),
        "Perplexity vs Cumulative Communication",
        "Cumulative uploaded bytes",
        "Perplexity",
        output_dir / "perplexity_vs_bytes.png",
    )

    print(f"Wrote figures to {output_dir.resolve()}")


if __name__ == "__main__":
    main()