"""Run the final common protocol from an IDE or with python run_benchmark.py."""
from pathlib import Path
from benchmark import pilot_tmc_sepm as experiment

ROOT = Path(__file__).resolve().parent
DATASETS = ["cervical", "breast", "diabetes", "haberman"]

def main():
    experiment.DATA_DIR = ROOT / "data"
    experiment.OUTPUT_ROOT = ROOT / "reruns" / "benchmark"
    experiment.DATASETS = DATASETS
    experiment.OUTER_FOLD_LIMIT = None
    experiment.OUTER_FOLD_INDEX = None
    experiment.TMC_SEEDS = [101]
    experiment.TMC_PERMUTATIONS = 100
    experiment.SEARCH_JOBS = 2  # Recorded completed CHTC jobs; estimator threads = 1.
    experiment.main()

if __name__ == "__main__":
    main()
