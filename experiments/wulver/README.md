# Running on Wulver

Batch jobs for the training-size sweep (component D). Submit every job from the repo root through `submit.sh`, which adds your course account (`2026-fall-ds-677-ikoutis-<UCID>`). Logs land in `logs/` and are committed with the scripts.

Wulver docs: [access](https://hpc.njit.edu/clusters/cluster_access/) · [course jobs](https://hpc.njit.edu/Courses/course-job-submission/) · [batch jobs](https://hpc.njit.edu/Running_jobs/batch-jobs/)

## What the course allocation gives us

| | |
|---|---|
| Account | `2026-fall-ds-677-ikoutis-<UCID>`, one per student |
| Partitions | `course_gpu` (3 nodes, 8 × A100 10 GB MIG slices each), `course` (CPU) |
| QoS | `course`: at most 1 GPU and 128 CPUs per job, 3-day walltime |
| Memory | up to 4 GB per CPU on `course_gpu` |
| Storage | `/course/2026/fall/ds/677/ikoutis/<UCID>/` (private to you) |

The main `gpu` partition (full A100s, L40s) does not accept the course QoS, so every GPU job runs on a 10 GB slice.

## One-time setup

1. Off campus, connect to the NJIT VPN first. Log in with your UCID and approve the Duo prompt:
   ```bash
   ssh <UCID>@wulver.njit.edu
   ```
2. Work in your course folder:
   ```bash
   cd /course/2026/fall/ds/677/ikoutis/$USER
   git clone https://github.com/stayzen2000/dl_project_lead_ranking.git
   cd dl_project_lead_ranking
   ```
3. Install uv, keep its cache next to the project, and build the environment on a CPU compute node:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   echo "export UV_CACHE_DIR=/course/2026/fall/ds/677/ikoutis/$USER/.uv-cache" >> ~/.bashrc && source ~/.bashrc
   srun --account=2026-fall-ds-677-ikoutis-$USER --partition=course --qos=course \
        --cpus-per-task=4 --mem-per-cpu=4000M --time=00:30:00 ~/.local/bin/uv sync
   ```
4. Put the Kaggle data in `data/raw/` as described in the main README, for example by copying it up from your laptop:
   ```bash
   rsync -avP data/raw/ <UCID>@wulver.njit.edu:/course/2026/fall/ds/677/ikoutis/<UCID>/dl_project_lead_ranking/data/raw/
   ```

## Jobs

| Script | What it does |
|--------|--------------|
| `env_check.sbatch` | 15-minute GPU smoke test: `nvidia-smi`, PyTorch CUDA build, GPU memory, a matmul |
| `sweep.sbatch <model>` | 12-task job array, one task per (training size, seed) pair for one model |

```bash
experiments/wulver/submit.sh experiments/wulver/env_check.sbatch
experiments/wulver/submit.sh --job-name=sweep_mlp experiments/wulver/sweep.sbatch mlp
squeue -u $USER               # PD = pending, R = running
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS
```
