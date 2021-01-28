import os

n_trials = 50
for trial in range(n_trials):
    os.system(f"sbatch trial_{trial}.sh")

