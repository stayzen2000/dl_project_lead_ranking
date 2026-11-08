#!/bin/bash -l
#SBATCH --job-name=sweep
#SBATCH --partition=course_gpu
#SBATCH --qos=course
#SBATCH --gres=gpu:a100_10g:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=02:00:00
#SBATCH --array=0-11
#SBATCH --output=experiments/wulver/logs/%x_%A_%a.out

# Training-size sweep (component D): one array task per (size, seed) pair for
# one model, 4 sizes x 3 seeds = 12 tasks. The account comes from submit.sh:
#   experiments/wulver/submit.sh --job-name=sweep_mlp experiments/wulver/sweep.sbatch mlp

set -euo pipefail
MODEL=${1:?usage: submit.sh experiments/wulver/sweep.sbatch <model>}
SIZES=(500 2000 10000 50000)
SEEDS=(0 1 2)
N_TRAIN=${SIZES[$((SLURM_ARRAY_TASK_ID / ${#SEEDS[@]}))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % ${#SEEDS[@]}))]}

echo "model=$MODEL n_train=$N_TRAIN seed=$SEED host=$(hostname) started: $(date)"
source .venv/bin/activate

# TODO: entry point, written once the data pipeline and model code exist.
python -m leadrank.run --model "$MODEL" --n-train "$N_TRAIN" --seed "$SEED" \
    --out "experiments/results/$MODEL/n${N_TRAIN}_s${SEED}.json"
echo "finished: $(date)"
