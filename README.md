# ECS 170 — Artificial Intelligence Course Project

**UC Davis, Spring 2026 — Group Project (5 members)**

Implementations of classic and deep learning models across a quarter-long, multi-stage
project: classical classifiers, an MLP, a CNN, an RNN/LSTM/GRU, and a GNN, applied to
tabular, image, text, and graph data respectively.

## My Contribution

I was one of five contributors. Across stages, I typically wrote the first working
implementation of each model — generic, reusable model-builder functions and training
scaffolding — which teammates then extended with hyperparameter experiments, ablations,
and evaluation/reporting. I led **Stage 2 (MLP)** end-to-end, and contributed core
implementation work on Stages 1, 3, and 4. I reviewed but did not implement Stage 5 (GNN).

## Stages

| Stage | Task | Models | Datasets |
|-------|------|--------|----------|
| 1 | Toy classification | Decision Tree, MLP, SVM | Small synthetic dataset |
| 2 | Multiclass classification | MLP (baseline + ablation variants) | Provided tabular dataset |
| 3 | Image classification | CNN | MNIST, ORL (faces), CIFAR |
| 4 | Text classification & generation | RNN, LSTM, GRU | IMDB sentiment, Short Jokes |
| 5 | Node classification | GNN (GCN) | Cora, Pubmed, Citeseer |

## Repository Structure

```
ECS170_Spring_2026_Source_Code_Template/
├── local_code/
│   ├── base_class/       # Shared abstract classes (Dataset, Method, Evaluate, Result, Setting)
│   ├── stage_1_code/     # Decision Tree, MLP, SVM
│   ├── stage_2_code/     # MLP (baseline + ablation)
│   ├── stage_3_code/     # CNN
│   ├── stage_4_code/     # RNN classifier + generation
│   └── stage_5_code/     # GNN (teammate-led)
├── script/                # Per-stage run scripts
├── data/                  # Provided datasets
└── result/                # Saved outputs per stage

stage_4_outputs/           # Stage 4 notebooks, filled report, and generated plots
```

## Setup

```bash
pip install -r requirements.txt
```

Run any stage from the project root, e.g.:

```bash
cd ECS170_Spring_2026_Source_Code_Template
python -m script.stage_1_script.script_mlp
```

## Notes

This was a required team assignment (max team size: 5) for ECS 170. Work was
distributed across teammates per stage; the breakdown above reflects my own
contribution, not the full team's individual credits.