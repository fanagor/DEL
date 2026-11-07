"""DEL: Differentiable Exact Learning. One configuration per experiment."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import yaml

from .config import _merge, load_config, validate_config


def main(argv=None):
    parser = argparse.ArgumentParser(prog="del", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "data", "config", "teaching-set"):
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path, help="Small recipe YAML merged into package defaults")
        if name == "train":
            resume = p.add_mutually_exclusive_group()
            resume.add_argument(
                "--resume", action="store_true", help="Continue the exact saved experiment"
            )
            resume.add_argument(
                "--resume-auto",
                action="store_true",
                help="Start a new run or resume its checkpoint",
            )
            p.add_argument(
                "--handle-signals",
                action="store_true",
                help="Save a checkpoint on scheduler SIGTERM",
            )
            p.add_argument("--run-dir", type=Path, help="Explicit output/resume directory")
        p.add_argument("overrides", nargs="*", help="Validated dotted key=value overrides")
    p = sub.add_parser("evaluate")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--checkpoint", default="latest.pt")
    p.add_argument("--device", default="auto")
    p.add_argument("--output", type=Path)
    p.add_argument("overrides", nargs="*", help="Only evaluation.* overrides")
    p = sub.add_parser("plot", help="Plot the compact local training history")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--output", type=Path)
    from .viewer.server import add_arguments

    add_arguments(
        sub.add_parser("animate", help="Explore a controller in a local tape/grid viewer")
    )
    # argparse stops collecting a '*' positional after an intervening option.
    # Accept trailing assignments too, while rejecting unknown flags/arguments.
    args, trailing = parser.parse_known_args(argv)
    if trailing:
        if not hasattr(args, "overrides") or any(
            value.startswith("-") or "=" not in value for value in trailing
        ):
            parser.error(f"unrecognized arguments: {' '.join(trailing)}")
        args.overrides.extend(trailing)
    if args.command in {"train", "data", "config", "teaching-set"}:
        config = load_config(args.config, args.overrides)
        if args.command == "config":
            print(yaml.safe_dump(config, sort_keys=False))
        elif args.command == "teaching-set":
            from .data.teaching import teaching_manifest

            print(json.dumps(teaching_manifest(config["data"]), indent=2))
        elif args.command == "data":
            from .data import build_static_dataset

            traces = build_static_dataset(config["data"], cache_dir=config["data"]["cache_dir"])
            print(f"Prepared {len(traces)} expert traces in {config['data']['cache_dir']}")
        else:
            from .training import train

            print(
                train(
                    config,
                    resume="auto" if args.resume_auto else args.resume,
                    run_dir=args.run_dir,
                    handle_signals=args.handle_signals,
                )
            )
    elif args.command == "evaluate":
        from .checkpoint import atomic_json
        from .evaluation import evaluate
        from .training import load_run

        model, expert, config, checkpoint = load_run(args.run_dir, args.checkpoint, args.device)
        config = deepcopy(config)
        # Make the new field overridable even in older checkpoint configurations.
        config["evaluation"].setdefault("scan", "sequential")
        for value in args.overrides:
            if not value.startswith("evaluation.") or "=" not in value:
                parser.error(
                    "Evaluation accepts only evaluation.* overrides; model/data come from the checkpoint"
                )
            key, raw = value.split("=", 1)
            update = yaml.safe_load(raw)
            for part in reversed(key.split(".")[1:]):
                update = {part: update}
            _merge(config["evaluation"], update, "evaluation")
        validate_config(config)
        progress_dir = args.run_dir / "evaluation_progress"
        result = evaluate(model, expert, config, progress_dir=progress_dir)
        result["step"] = checkpoint["step"]
        output = args.output or args.run_dir / "evaluation.json"
        atomic_json(result, output)
        print(json.dumps(result, indent=2))
    elif args.command == "animate":
        from .viewer.server import run_arguments

        run_arguments(args)
    else:
        import matplotlib.pyplot as plt

        path = args.run_dir / "metrics.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rows:
            raise ValueError(f"No training metrics in {path}")
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
        axes[0].plot([row["step"] for row in rows], [row["loss"] for row in rows])
        axes[0].set(xlabel="Optimizer updates", ylabel="Masked action loss", yscale="log")
        axes[1].plot([row["step"] for row in rows], [row["accuracy"] for row in rows])
        axes[1].set(xlabel="Optimizer updates", ylabel="Joint action accuracy", ylim=(0, 1.01))
        output = args.output or args.run_dir / "training.png"
        fig.savefig(output, dpi=180)
        plt.close(fig)
        print(output)


if __name__ == "__main__":
    main()
