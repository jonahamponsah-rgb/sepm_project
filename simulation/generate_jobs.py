#!/usr/bin/env python3
"""Create deterministic CHTC job list; 5 repeats/job up to n=1000, 1 for n=5000."""
from pathlib import Path
SIZES = [100, 200, 500, 1000, 5000]
PREVALENCES = [10, 20, 30, 35]
N_REPS = 100
here = Path(__file__).resolve().parent
rows = []
for n in SIZES:
    per_job = 1 if n == 5000 else 5
    for p in PREVALENCES:
        for first in range(0, N_REPS, per_job):
            rows.append(f'{n} {p} {first} {min(per_job,N_REPS-first)} full')
(here/'jobs_full.txt').write_text('\n'.join(rows)+'\n')
(here/'jobs_smoke.txt').write_text('100 20 0 1 smoke\n')
print(f'Generated {len(rows)} full jobs, one smoke job')
