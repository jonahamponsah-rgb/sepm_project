"""Run the full 20-scenario, 100-repetition simulation from an IDE."""
from pathlib import Path
from simulation import sepm_simulation as experiment

def main():
    experiment.OUTPUT_ROOT = Path(__file__).resolve().parent / "reruns" / "simulation"
    experiment.SEARCH_JOBS = 2  # Match the completed batch jobs.
    experiment.main()

if __name__ == "__main__":
    main()
