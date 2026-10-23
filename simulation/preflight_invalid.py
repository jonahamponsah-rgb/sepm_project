#!/usr/bin/env python3
"""Audit minority counts without refitting any models or changing any draws."""
import csv
import numpy as np
import sepm_simulation as sim

adaptive = []
invalid = []
for n in sim.TRAIN_SIZES:
    for prevalence in sim.TARGET_PREVALENCES:
        intercept = sim.calibrate_intercept(prevalence)
        scenario = int(np.random.SeedSequence(
            [n, int(round(prevalence*1000000))]).generate_state(1)[0])
        for rep in range(sim.N_REPETITIONS):
            y = sim.generate(n, intercept, sim.seed_for(scenario, rep, 0))[1]
            counts = np.bincount(y, minlength=2)
            minority = int(counts.min())
            record = dict(n_train=n, target_prevalence=prevalence, repetition=rep,
                          positive_count=int(counts[1]), negative_count=int(counts[0]),
                          minority_count=minority, requested_folds=sim.INNER_FOLDS,
                          effective_folds=min(sim.INNER_FOLDS, minority))
            if minority < 2:
                invalid.append(record)
            elif minority < sim.INNER_FOLDS:
                adaptive.append(record)
fields = ['n_train','target_prevalence','repetition','positive_count','negative_count',
          'minority_count','requested_folds','effective_folds']
for filename, rows in [('adaptive_cv_preflight.csv',adaptive),
                       ('invalid_draws_preflight.csv',invalid)]:
    with open(filename,'w',newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
print(f'Adaptive CV needed for {len(adaptive)} draws; CV impossible for {len(invalid)} draws')
for row in adaptive: print(row)
for row in invalid: print('INVALID',row)
