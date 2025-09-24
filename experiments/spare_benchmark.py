import os
import json
import argparse
import random
from pathlib import Path

import scanpy as sc
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sampleclr.contrastive_model import ContrastiveModel
from sampleclr.utils import get_sample_representations, plot_losses
import torch

COLLAPSE_DISTANCE_THRESHOLD = 1e-2

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# Set initial seed
set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run SampleCLR benchmark for specific dataset and model")
    parser.add_argument("--dataset-config", type=str, required=True,
                        help="Path to dataset configuration JSON file")
    parser.add_argument("--model-config", type=str, required=True,
                        help="Path to model configuration JSON file")
    parser.add_argument("--output-dir", type=str, default="../results",
                        help="Output directory for results (default: ../results)")
    parser.add_argument("--prior", action="store_true",
                        help="Use prior information in the model")
    parser.add_argument("--supervised", action="store_true",
                        help="Use supervised learning tasks")
    return parser.parse_args()


def main():
    """Main function to run the benchmark."""
    args = parse_args()
    
    # Load configurations
    dataset_config = load_config(args.dataset_config)
    model_config = load_config(args.model_config)
    
    print(f"Processing dataset: {dataset_config['dataset_name']}")
    print(f"Model config: {args.model_config}")
    print(f"Output directory: {args.output_dir}")
    
    # Create output directory
    output_dir = Path(args.output_dir) / dataset_config["dataset_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    adata = sc.read_h5ad(dataset_config["data_path"])
    print(f"Loaded adata")
    print(adata)
    
    metadata = pd.read_csv(dataset_config["metadata_path"], index_col=0)
    print(f"Loaded metadata: {metadata.shape}")

    set_seed(42)

    print("Removing small samples")
    metadata = metadata[metadata["n_cells"] >= dataset_config["min_cells"]]
    adata = adata[adata.obs[dataset_config["sample_id_col"]].isin(metadata.index)].copy()

    if "split" in metadata.columns:
        print("Validation split column found in metadata, using it")
        train_ids = metadata[metadata["split"] == "train"].index.tolist()
        val_ids = metadata[metadata["split"] == "val"].index.tolist()
    else:
        print("Validation split column not found in metadata, using random split")
        train_ids = np.random.choice(metadata.index.tolist(), size=int(0.8 * len(metadata)), replace=False)
        val_ids = [sample for sample in metadata.index.tolist() if sample not in train_ids]

    print(f"Total cells: {len(adata)}, Total samples: {len(metadata)}, Train: {len(train_ids)}, Validation: {len(val_ids)}, Validation %: {round(100 * len(val_ids) / len(metadata), 2)}")

    dataset_params = {
        "adata": adata,
        "sample_key": dataset_config["sample_id_col"],
        "layer": dataset_config["layer"],
        "device": device,
        "train_ids": train_ids,
        "val_ids": val_ids,
    }

    print("Checking if dataset params are correct:")
    print("Sample key is in the data:", dataset_config["sample_id_col"] in adata.obs.columns)
    print("Layer is in the data:", dataset_config["layer"] in adata.obsm.keys())
    print("Prior sample distances is in the data:", dataset_config["prior_sample_distances"] in adata.uns.keys())

    print("Dataset params:")
    print(dataset_params)

    print("Model config:")
    print(model_config)

    # Determine model type based on command-line arguments
    model_type_parts = []
    if args.supervised:
        model_type_parts.append("supervised")
    else:
        model_type_parts.append("Unsupervised")
    
    if args.prior:
        model_type_parts.append("prior")
    
    model_name = "SampleCLR_" + "_".join(model_type_parts)
    
    print(f"Running model: {model_name}")
    
    # Prepare model parameters
    tasks = dataset_config["supervised_task"] if args.supervised else {}
    extra_params = {}
    
    if args.prior:
        prior_distance_matrix = adata.uns[dataset_config["prior_sample_distances"]]
        similarity_graph = np.exp(-prior_distance_matrix / 5)
        extra_params = {"sample_similarity_graph": similarity_graph}
    
    model_size = Path(args.model_config).stem

    # Compensate lack of supervised heads by additional layers to make #parameters more equal
    if not args.supervised:
        model_config["num_layers"] += 1

    # Combine model config with dataset params
    common_params = {**model_config, **dataset_params}

    # if dataset_config["dataset_name"] in ["stephenson", "hlca"]:
    #     common_params["xsample_clr_graph_temperature"] = 0.05

    set_seed(42)
    model = ContrastiveModel(tasks=tasks, **common_params, **extra_params)

    best_state, best_val_loss, num_epochs_stage1_trained, num_epochs_stage2_trained = model.train(
        num_epochs_stage1=300,
        num_epochs_stage2=450,
        two_stages=True,
        stage_2="joint",
        verbose=False,
        stage1_val_metric="loss",
        stage2_val_metric="total"
    )

    print("Model:", model_name)
    print("Best validation loss:", best_val_loss)

    subset_size = model.n_cells_per_sample
    if isinstance(subset_size, list):
        subset_size = subset_size[1]

    def get_top_cells(group):
        n = len(group)
        if n >= subset_size:
            return group.nsmallest(subset_size, "eigenvector_centrality")
        else:
            # Resample cells to get subset_size total
            top_cells = group.sample(n=subset_size, replace=True)
            return top_cells

    all_representations = np.zeros((len(train_ids) + len(val_ids), model.output_dim))
    with torch.no_grad():
        model.projector.eval()
        model.aggregator.eval()
        model.aggregator.return_weights = False
        # val_representations = get_sample_representations(model.projector, model.aggregator, model.val_dataset, subset_size=None, device=device)
        # train_representations = get_sample_representations(model.projector, model.aggregator, model.train_dataset, subset_size=None, device=device)

        top_cells = (adata.obs
                    .groupby(dataset_config["sample_id_col"])
                    .apply(get_top_cells)
                    .index.get_level_values(1))
        selected_cells = adata[top_cells, :]
        for i, sample in enumerate(train_ids + val_ids):
            sample_mask = selected_cells.obs[dataset_config["sample_id_col"]] == sample
            sample_cells = torch.Tensor(selected_cells[sample_mask].obsm[dataset_config["layer"]]).unsqueeze(0).to(device)
            aggregated_cells = model.aggregator(sample_cells)
            representations = model.projector(aggregated_cells)
            all_representations[i] = representations.squeeze(0).cpu().numpy()

    representation_df = pd.DataFrame(all_representations, index=list(train_ids) + list(val_ids))
    distances = pdist(representation_df, metric="cosine")
    if (distances < COLLAPSE_DISTANCE_THRESHOLD).all():
        print("Collapse detected, not saving representations")
        return
    
    model_output_dir = output_dir / model_size / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    representation_df.to_csv(model_output_dir / f"{model_name}_representations.csv")
    pd.DataFrame(squareform(distances), index=representation_df.index, columns=representation_df.index).to_csv(model_output_dir / f"{model_name}_distances.csv")

    fig = plot_losses(model)
    fig.savefig(model_output_dir / f"{model_name}_losses.png")


if __name__ == "__main__":
    main()
