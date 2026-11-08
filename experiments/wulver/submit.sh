#!/bin/bash
# Submits a job under the submitter's DS 677 course account. Some of us have a
# research account as our Slurm default, so never call sbatch bare. Run from
# the repo root:
#   experiments/wulver/submit.sh experiments/wulver/env_check.sbatch
#   experiments/wulver/submit.sh --job-name=sweep_mlp experiments/wulver/sweep.sbatch mlp

exec sbatch --account="2026-fall-ds-677-ikoutis-$USER" "$@"
