"""Create CHTC input bundles from the packaged source and user-supplied data."""
from pathlib import Path
import tarfile
ROOT = Path(__file__).resolve().parents[1]
def main():
    for target in ["tmc", "simulation"]:
        (ROOT/"cluster"/target/"logs").mkdir(exist_ok=True)
    with tarfile.open(ROOT/"cluster/tmc/sepm_tmc_bundle.tar.gz", "w:gz") as t:
        for name in ["pilot_tmc_sepm.py", "chtc_fold_runner.py"]:
            t.add(ROOT/"benchmark"/name, arcname="scripts/"+name)
        for name in ["cervical_cancer.csv", "breast-cancer.data", "pima-indians-diabetes.csv", "haberman.csv"]:
            t.add(ROOT/"data"/name, arcname="data/"+name)
    with tarfile.open(ROOT/"cluster/simulation/sepm_simulation_bundle.tar.gz", "w:gz") as t:
        for name in ["sepm_simulation.py", "sim_batch.py"]:
            t.add(ROOT/"simulation"/name, arcname=name)
    print("Bundles created; edit container_image in each submit file before submission.")
if __name__ == "__main__":
    main()
