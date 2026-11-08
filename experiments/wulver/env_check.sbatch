#!/bin/bash
#SBATCH --job-name=env_check
#SBATCH --partition=course_gpu
#SBATCH --qos=course
#SBATCH --gres=gpu:a100_10g:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=00:15:00
#SBATCH --output=experiments/wulver/logs/%x_%j.out

# Smoke test: confirms the GPU, the project environment and PyTorch's CUDA build
# before we spend allocation on the sweep. The account comes from submit.sh;
# --qos=course makes Slurm reject the job under any non-course account.
#   experiments/wulver/submit.sh experiments/wulver/env_check.sbatch

set -euo pipefail
echo "host: $(hostname)  job: $SLURM_JOB_ID  account: $SLURM_JOB_ACCOUNT  started: $(date)"
nvidia-smi
source .venv/bin/activate
python - <<'EOF'
import sklearn, torch, xgboost

print("torch", torch.__version__, "| CUDA build", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    free, total = torch.cuda.mem_get_info()
    print(f"GPU memory: {free / 2**30:.1f} GiB free of {total / 2**30:.1f} GiB")
    x = torch.randn(4096, 4096, device="cuda")
    print("matmul ok:", tuple((x @ x).shape))
print("sklearn", sklearn.__version__, "| xgboost", xgboost.__version__)
EOF
echo "finished: $(date)"
