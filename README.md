# Contrastive Sample Representations (single-cell)

Learn **patient/sample-level embeddings** from single-cell RNA-seq with **contrastive learning**. This repo provides a lightweight **multi-head aggregator** over cells, optional **supervised heads** (classification, regression, ordinal), and **adversarial batch correction** via a Gradient Reversal Layer (GRL)—plus utilities for evaluation, plotting, and Optuna-based hyperparameter search.

---

## Features

- **Two-stage training**: contrastive pretrain → supervised/joint fine-tune (with early stopping).
- **Multi-head cell aggregation** (optionally with Gumbel Softmax and a diversity regularizer).
- Task heads for **classification**, **regression**, **ordinal regression**.
- Optional **batch-invariant** embeddings using a **GRL** discriminator.
- **InfoNCE** family of losses (default: **Cauchy** variant).
- Utilities for **KNN/NN evaluation**, **UMAP** plots, and **training curves**.

---

## Installation

```bash
# (Recommended) fresh virtual environment
python -m venv .venv && source .venv/bin/activate

# Core scientific stack
pip install numpy pandas scipy scikit-learn matplotlib seaborn umap-learn pyyaml

# Single-cell + optimization
pip install scanpy optuna

# PyTorch (choose the correct build for your OS/CUDA)
# See https://pytorch.org/get-started/locally/
pip install torch
```

- Python ≥ 3.10 recommended.
- A GPU is optional but helpful for faster training.

---

## Repository Structure

```
.
├─ contrastive_model.py     # High-level trainer (ContrastiveModel), staging, early stopping, evaluation
├─ models.py                # Aggregator, projector, classifier/regression/ordinal heads, GRL discriminator
├─ losses.py                # InfoNCE variants (Cauchy/Gaussian/Cosine), ordinal loss, batch-aware option
├─ datasets.py              # SamplesDataset (per-sample tensors), TransformedPairDataset (two random views)
├─ utils.py                 # KNN/NN eval, batch leakage score, UMAP, training/metric plots, embeddings IO
```

---

## Quickstart (Python API)

```python
import torch, scanpy as sc, pandas as pd
from contrastive_model import ContrastiveModel
from datasets import SamplesDataset
from utils import get_sample_representations

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) Load data
adata = sc.read_h5ad("combat_processed.h5ad")
sample_key = "scRNASeq_sample_ID"
layer = "X_scANVI_Pool_ID"

# 2) Define tasks (use what you need)
tasks = {
    "classification": ["Source"],
    # "regression": ["Age"],
    # "ordinal_regression": ["Outcome"],
    # "batch_correction": "Pool_ID",
}

# 3) (Optional) split by patient ids
donors = adata.obs[sample_key].unique()

result_record = {
    "num_layers": 1,
    "hidden_size": 86,
    "learning_rate_feature": 0.0027232013732692937,
    "learning_rate_discriminator": 0.0008329193828138741,
    "weight_decay": 6.8656067372346156e-06,
    "lambda_": 0.6642516866102309,
    "batch_size": 4,
    "n_aggregator_heads": 17,
    "output_dim": 146,
    "n_cells_per_sample": 2500,
    "classifier_num_layers": 2,
    "classifier_hidden_size": 217,
    "regression_num_layers": 1,
    "regression_hidden_size": 128,
    "ordinal_num_layers": 1,
    "ordinal_hidden_size": 128,
    "use_normalization": True
}


model = ContrastiveModel(
    adata=adata,
    sample_key=sample_key,
    tasks=tasks,
    layer=layer,
    device=device,
    # split_config=split_config,
    extra_covariates=["Pool_ID"],
    # extra_covariates=["Outcome","Pool_ID"],
    num_layers=result_record["num_layers"],
    hidden_size=result_record["hidden_size"],
    learning_rate_feature=result_record["learning_rate_feature"],
    learning_rate_discriminator=result_record["learning_rate_discriminator"],
    weight_decay=result_record["weight_decay"],
    lambda_=result_record["lambda_"],
    batch_size=result_record["batch_size"],
    n_aggregator_heads=result_record["n_aggregator_heads"],
    output_dim=result_record["output_dim"],
    n_cells_per_sample=result_record["n_cells_per_sample"],
    classifier_num_layers=result_record.get("classifier_num_layers", 1),
    classifier_hidden_size=result_record.get("classifier_hidden_size", 128),
    regression_num_layers=result_record.get("regression_num_layers", 1),
    regression_hidden_size=result_record.get("regression_hidden_size", 128),
    ordinal_num_layers=result_record.get("ordinal_num_layers", 1),
    ordinal_hidden_size=result_record.get("ordinal_hidden_size", 128),
    use_normalization=result_record.get("use_normalization", False),
    num_epochs_stage1=150,
    num_epochs_stage2=150,
    verbose=True,
    early_stopping_patience=20,
    train_ids=donors,
    # train_ids=train_ids,
    # val_ids=val_ids,
    # test_ids=None,
    # test_ids=test_ids,
    diversity_loss_weight=0.14,
    use_gumbel= True,
    temperature = 1.5,
    feature_normalization='BatchNorm'
)


best_state, best_val, e1, e2 = model.train(
    num_epochs_stage1=80,
    num_epochs_stage2=30,
    two_stages=True,                     # contrastive → supervised stage
    stage_2="joint",  # or "only_supervised_with_agg" 
    stage1_val_metric="loss",
    stage2_val_metric="total",
    verbose=False,
)

```
---

## Evaluate Quickly (KNN/NN + Batch Leakage)

```python

results = {}
results["train"] = model.evaluate_split(
    model.train_dataset,
    model.metadata_all.loc[model.train_dataset.unique_categories],
    model.train_dataset,
    model.metadata_all.loc[model.train_dataset.unique_categories]
) if model.train_dataset is not None else {}
results["val"] = model.evaluate_split(
    model.train_dataset,
    model.metadata_all.loc[model.train_dataset.unique_categories],
    model.val_dataset,
    model.metadata_all.loc[model.val_dataset.unique_categories]
) if model.val_dataset is not None else {}
results["test"] = model.evaluate_split(
    model.train_dataset,
    model.metadata_all.loc[model.train_dataset.unique_categories],
    model.test_dataset,
    model.metadata_all.loc[model.test_dataset.unique_categories]
) if model.test_dataset is not None else {}

results

```