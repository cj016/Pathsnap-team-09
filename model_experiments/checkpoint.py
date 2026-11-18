"""Command line entry point for the team's separate feature and prep worktrees."""
import argparse
import runpy
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prep-dir", type=Path, default=root)
    parser.add_argument("--framework-dir", type=Path, default=root)
    parser.add_argument("action", choices=("prepare", "train"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    dependency = [(args.framework_dir, "features/feature_engineering.py")]
    if args.action == "prepare":
        dependency.append((args.prep_dir, "src/data_prep.py"))
    for directory, name in dependency:
        if not (directory / name).is_file():
            parser.error(f"Missing {directory / name}. See model_experiments/README.md for setup.")
        sys.path.append(str(directory.resolve()))
    module = "prepare_checkpoint" if args.action == "prepare" else "run_checkpoint"
    sys.argv = [module, *args.arguments]
    runpy.run_module(f"model_experiments.{module}", run_name="__main__")


if __name__ == "__main__":
    main()
