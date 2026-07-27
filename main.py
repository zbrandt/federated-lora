from __future__ import annotations

import argparse
import importlib


METHOD_PACKAGES = {
    "lalora": "la_lora",
    "flora": "flora",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Federated LoRA harness")
    sub = parser.add_subparsers(dest="command", required=True)

    # `run` forwards all remaining flags (--method, --task, --seed, ...) to specified method
    sub.add_parser("run", help="run a federated LoRA method")

    plot_p = sub.add_parser("plot", help="plot accuracy against round number from output JSON")
    plot_p.add_argument(
        "inputs",
        nargs="*",
        default=["results"],
        help="result files or a directory of them (defaults to results/)",
    )
    plot_p.add_argument("--output-dir", default="figures")

    args, run_argv = parser.parse_known_args()

    if args.command == "run":
        method_parser = argparse.ArgumentParser(add_help=False)
        method_parser.add_argument("--method", default="lalora")
        method_args, _ = method_parser.parse_known_args(run_argv)

        package = METHOD_PACKAGES.get(method_args.method)
        if package is None:
            raise SystemExit(f"Unknown method '{method_args.method}'. Available: {sorted(METHOD_PACKAGES)}")

        importlib.import_module(package).run(run_argv)
    elif args.command == "plot":
        import plot_results
        plot_results.plot(args.inputs, args.output_dir)


if __name__ == "__main__":
    main()
