from __future__ import annotations

import argparse
import importlib

METHODS = ("la_lora",)

def main() -> None:
    parser = argparse.ArgumentParser(description="Federated LoRA harness")
    sub = parser.add_subparsers(dest="command", required=True)  # TODO: understand this

    run_p = sub.add_parser("run", help="run a federated LoRA method")
    run_p.add_argument("--method", choices=METHODS, default="la_lora")

    plot_p = sub.add_parser("plot", help="plot result JSON files")
    plot_p.add_argument("inputs", nargs="+")
    plot_p.add_argument("--output-dir", default="figures")

    args, method_argv = parser.parse_known_args()

    if args.command == "run":
        method = importlib.import_module(args.method)
        method.run(method_argv)
    elif args.command =="plot":
        import plot_results
        plot_results.plot(args.inputs, args.output_dir)


if __name__ == "__main__":
    main()
