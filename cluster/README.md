# CHTC execution

`tmc/` contains the supplied job wrapper and submit template, with the private container location replaced by `YOUR_CONTAINER_IMAGE_URI`. `tmc/jobs.txt` lists all 40 dataset/fold benchmark jobs. `benchmark/pilot_tmc_sepm.py` is the exact source whose SHA-256 appears in the completed jobs; its pilot defaults are overridden by `chtc_fold_runner.py --mode benchmark` (100 permutations, seed 101).

`simulation/` contains the original wrapper, submit template and full 720-job schedule for 2,000 repetitions. Scientific functions are supplied by the hash-matched `simulation/sepm_simulation.py`; `sim_batch.py` applies the recorded batch settings. Both completed CHTC workflows requested two CPU cores, 8 GB memory and 25 GB disk, with two grid-search workers and one model thread. The separate laptop clinical runs used three search workers. These are different recorded execution contexts.

1. Supply and verify the four data files using `prepare_data.py`.
2. Run `cluster/build_bundles.py` from the repository root.
3. Set `container_image` in both submit files to a container accessible on your system. Create/transfer the corresponding bundle, job list, wrapper, submit file and logs directory to the submit host.
4. From the respective cluster directory submit `condor_submit sepm_tmc.sub` or `condor_submit simulation.sub`.
5. Use `simulation/combine_results.py --archives YOUR_RETURNED_ARCHIVES --output YOUR_COMBINED_OUTPUT --mode full` to combine returned simulation jobs. The benchmark archive verifier in `reporting/reproduce.py` demonstrates aggregation without unsafe extraction.

The original `python-ml.sif` binary and its original build recipe were not supplied. `container-reconstruction.def` is a newly prepared recipe using the recorded benchmark package versions, not a claim to recreate the historical image bit for bit. Build from `cluster/` using `apptainer build python-ml-reconstructed.sif container-reconstruction.def`; local Apptainer permissions and image transfer are site-specific. Simulation batch metadata did not record a complete package lockfile, so the benchmark pins are a reconstruction environment for simulations, not a verified simulation lockfile. Original source, scheduler records, hashes and a representative simulation batch record are retained for provenance.

The metadata `pilot` boolean in individual benchmark fold jobs remains true because the legacy expression flags a single fold or one random stream. The recorded budget is 100 permutations and the supplied job list covers all 40 folds; the boolean does not mean these are the earlier 10-permutation runs.
