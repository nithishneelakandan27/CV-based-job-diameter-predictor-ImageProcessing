"""Convenience entry point for the exploratory edge experiment."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_edge_experiment import parse_args, run


def main() -> None:
    args = parse_args()
    run(args.input, args.output)


if __name__ == "__main__":
    main()
