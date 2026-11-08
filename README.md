# Ranking quote leads with little labelled data: TabPFN vs. XGBoost

> **Approved claim:** We claim that with 2,000 or fewer labelled quotes, TabPFN captures more
> converting customers in the top 10% of the ranked list than a tuned XGBoost on TabReD's Homesite
> time-based split.

NJIT DS 677 Deep Learning · Industry Application track
Group: Irteza Alif (`stayzen2000`) · Nikunj Nileshkumar Kantaria (`techy-Nik`)

## The problem

Businesses that sell through quotes (insurance, roofing, mortgages) collect many people who ask for a
quote and never buy. Ranking those people by how likely they are to convert lets a sales team call the
best ones first. A small business usually has only a few hundred to a few thousand past quotes with a
known outcome, so the question is which model ranks best when labelled data is that scarce.

## Dataset

[Homesite Quote Conversion](https://www.kaggle.com/c/homesite-quote-conversion) (Kaggle): 260,753
insurance quotes, about 19% converted. We use the time-based split from the TabReD benchmark
(Rubachev et al., ICLR 2025):

| Split | Period | Rows |
|---|---|---|
| Train | before Feb 2015 | 224,320 |
| Validation | Feb–Mar 2015 | 20,138 |
| Test | Apr–May 2015 | 16,295 |

Training on older quotes and testing on newer ones is how a business would use the model, and it is
harder than a random split.

**The data is not in this repository.** Use is governed by the Kaggle competition rules. To run the
code, download `train.csv` from Kaggle after accepting the rules and place it under `data/raw/`, which
is git-ignored.

## Models

| Role | Model |
|---|---|
| Baseline | Logistic regression |
| Baseline | XGBoost, tuned |
| Deep | MLP |
| Deep | FT-Transformer (transformer over table columns) |
| Deep | TabPFN (pretrained transformer, in-context learning across rows) |
| Deep | Transductive row-attention transformer, no pretraining (adapted from DS 677 Assignment 3) |

TabPFN and the transductive transformer both attend across rows; only TabPFN is pretrained. Comparing
them shows how much of any gain comes from pretraining.

CNNs and RNNs are left out on purpose: table columns have no spatial order and a row is not a sequence.

## Plan

| Component | What we do |
|---|---|
| A. Model and claim | TabPFN against tuned XGBoost at small training sizes (claim above) |
| B. Baseline | Logistic regression and tuned XGBoost, each given a fair tuning budget |
| C. Replication anchor | Reproduce TabReD's published Homesite ROC AUC: XGBoost 0.9601, MLP 0.9500, FT-Transformer 0.9622 (Table 3, averaged over 15 seeds; tuning spaces in [`paper/exp/<model>/homesite-insurance/`](https://github.com/yandex-research/tabred/tree/main/paper/exp)) |
| D. Perturbation study | Data-scale sweep: 500 / 2,000 / 10,000 / 50,000 training quotes, 3 seeds each, run as Wulver batch jobs |
| E. Analysis and outlook | Where the claim holds, the training size at which XGBoost catches up, and where each model breaks |

**Metrics:** top-10% buyer capture (primary), ROC AUC, log-loss. Capture is primary because models on
Homesite all score near 0.95–0.96 ROC AUC, so AUC barely separates them, while capture is the number a
sales team acts on. With a 19% conversion rate the best possible top-10% capture is about 0.53, so we
also report capture as a fraction of that ceiling ([`src/leadrank/metrics.py`](src/leadrank/metrics.py)).
Results are reported as mean and spread across seeds.

## Work split

| Member | Owns |
|---|---|
| Irteza Alif | Data pipeline and TabReD split, baselines and replication anchor, TabPFN, transductive transformer |
| Nikunj Nileshkumar Kantaria | MLP and FT-Transformer, Wulver batch setup for the sweep, seed and variance analysis |
| Both | Shared metrics code, report, demo |

## Repository layout

```
README.md             # claim, plan, how to run
report/               # report drafts and final PDF source
src/                  # code
experiments/          # scripts, configs, results
└── wulver/           # Wulver job scripts and logs
demo/                 # self-contained demo notebook
```

## How to run

We use [uv](https://docs.astral.sh/uv/) so laptops and Wulver run the same locked package versions:

```bash
uv sync    # creates .venv from uv.lock (CPU PyTorch on Windows, CUDA PyTorch on Linux)
```

**Wulver.** The course allocation runs GPU jobs only on 10 GB A100 slices (`course_gpu`, one per
job). Setup, accounts and job scripts: [`experiments/wulver/README.md`](experiments/wulver/README.md).

Training and evaluation commands arrive with the code in Milestone 3.

## References

- Rubachev et al. *TabReD: Analyzing Pitfalls and Filling the Gaps in Tabular Deep Learning
  Benchmarks.* ICLR 2025.
- Hollmann et al. *Accurate predictions on small data with a tabular foundation model.* Nature, 2025.
- Gorishniy et al. *Revisiting Deep Learning Models for Tabular Data.* NeurIPS 2021.
