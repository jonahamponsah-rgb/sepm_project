"""CHTC single-outer-fold driver for the uploaded TMC/SEPM comparison.

Does not alter the model selection algorithm, evaluation protocol, or dataset grids.
"""
import argparse
from pathlib import Path
import pilot_tmc_sepm as experiment


def main():
    parser = argparse.ArgumentParser(description="Run one common-protocol SEPM/TMC outer fold on CHTC")
    parser.add_argument("--dataset", choices=["cervical", "breast", "diabetes", "haberman"], required=True)
    parser.add_argument("--fold", type=int, choices=range(10), required=True)
    parser.add_argument("--mode", choices=["pilot", "benchmark"], default="pilot")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--tmc-permutations", type=int, default=None)
    parser.add_argument("--tmc-seeds", type=int, nargs="+", default=[101])
    parser.add_argument("--search-jobs", type=int, default=2)
    args = parser.parse_args()

    if args.tmc_permutations is not None and args.tmc_permutations < 1:
        parser.error("--tmc-permutations must be >= 1")
    if args.search_jobs < 1:
        parser.error("--search-jobs must be >= 1")

    experiment.DATA_DIR = args.data_dir.resolve()
    experiment.OUTPUT_ROOT = args.output_root.resolve()
    experiment.DATASETS = [args.dataset]
    experiment.OUTER_FOLD_INDEX = args.fold
    experiment.OUTER_FOLD_LIMIT = None
    experiment.SEARCH_JOBS = args.search_jobs
    experiment.TMC_SEEDS = args.tmc_seeds
    experiment.TMC_PERMUTATIONS = (args.tmc_permutations if args.tmc_permutations is not None
                                   else (10 if args.mode == "pilot" else 100))

    # Fail early and clearly for files missing from the user's CHTC data directory.
    input_path = experiment.DATA_DIR / {
        "cervical": "cervical_cancer.csv", "breast": "breast-cancer.data",
        "diabetes": "pima-indians-diabetes.csv", "haberman": "haberman.csv"
    }[args.dataset]
    if not input_path.is_file():
        raise FileNotFoundError(f"Missing {input_path}; upload the required dataset before submitting")

    print(f"Dataset={args.dataset}, outer fold={args.fold}, mode={args.mode}, "
          f"TMC permutations={experiment.TMC_PERMUTATIONS}, seeds={args.tmc_seeds}", flush=True)
    experiment.main()


if __name__ == "__main__":
    main()
