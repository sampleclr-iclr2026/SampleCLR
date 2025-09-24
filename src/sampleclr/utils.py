import torch
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
import os

def get_sample_representations(projector, aggregator, dataset, subset_size=None, device='cpu'):
    """
    Extracts representations for all samples in the dataset.

    Args:
        projector: The projector network.
        aggregator: The aggregator network.
        dataset: Instance of SamplesDataset.
        subset_size: Number of cells to sample from each sample.
        device: Device to perform computations on.

    Returns:
        np.ndarray: Array of representations.
    """
    projector.eval()
    aggregator.eval()
    representations = []

    with torch.no_grad():
        for sample, *_ in dataset:
            if subset_size is None:
                indices = np.arange(sample.shape[0])
            else:
                if sample.shape[0] >= subset_size:
                    indices = np.random.choice(sample.shape[0], size=subset_size, replace=False)
                else:
                    indices = np.random.choice(sample.shape[0], size=subset_size, replace=True)

            sample_tensor = torch.Tensor(sample[indices]).unsqueeze(0).to(device)
            sample_agg = aggregator(sample_tensor)
            if isinstance(sample_agg, tuple):
                # features = sample_agg[0]
                features = projector(sample_agg[0])
            else:
                features = projector(sample_agg)
                # features = sample_agg
            representation = features.squeeze(0).cpu().numpy()
            representations.append(representation)

    return np.array(representations)


def evaluate_knn_nn_batch(
    projector,
    aggregator,
    classifiers,
    regressors,
    ordinal_regressors,
    discriminator,
    train_dataset,
    train_metadata,
    test_dataset,
    test_metadata,
    classification_tasks,
    regression_tasks,
    ordinal_regression_tasks,
    batch_col_encoded,
    device,
    extra_covariates=None
):
    """
    Evaluates the model using KNN and NN classifiers for classification, regression, ordinal regression, 
    and batch tasks. Also evaluates a set of extra covariates using KNN.

    Args:
        projector: The projector network.
        aggregator: The aggregator network.
        classifiers: List of classification heads.
        regressors: List of regression heads.
        ordinal_regressors: List of ordinal regression heads.
        discriminator: Discriminator head.
        train_dataset: Instance of SamplesDataset for training.
        train_metadata: DataFrame containing metadata for training.
        test_dataset: Instance of SamplesDataset for testing.
        test_metadata: DataFrame containing metadata for testing.
        classification_tasks: List of classification task dictionaries.
        regression_tasks: List of regression task column names.
        ordinal_regression_tasks: List of ordinal regression task dictionaries.
        batch_col_encoded: Encoded batch column name.
        device: Device to perform computations on.
        extra_covariates: List of extra covariate column names to evaluate with KNN.

    Returns:
        dict: evaluation metrics for all tasks and covariates.
    """

    train_reps = get_sample_representations(projector, aggregator, train_dataset, device=device)
    test_reps = get_sample_representations(projector, aggregator, test_dataset, device=device)
    results = {}

    # Classification tasks
    for idx, task in enumerate(classification_tasks):
        col_name = task['encoded_col']
        train_not_na = ~pd.isna(train_metadata[col_name])
        test_not_na = ~pd.isna(test_metadata[col_name])

        X_train = train_reps[train_not_na]
        y_train = train_metadata[col_name][train_not_na].astype(int).values

        X_test = test_reps[test_not_na]
        y_test = test_metadata[col_name][test_not_na].astype(int).values

        # Identify test samples with < 3 occurrences in train set
        train_counts = pd.Series(y_train).value_counts()
        valid_classes = train_counts[train_counts >= 3].index
        valid_mask = np.isin(y_test, valid_classes)

        if len(X_train) == 0 or sum(valid_mask) == 0:
            results[f'knn_{task["column"]}_score'] = np.nan
            results[f'nn_{task["column"]}_score'] = np.nan
            continue

        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)

        knn_preds = knn.predict(X_test[valid_mask])
        knn_class_score = f1_score(y_test[valid_mask], knn_preds, average='weighted')

        with torch.no_grad():
            logits = classifiers[idx](torch.Tensor(X_test[valid_mask]).to(device))
        nn_preds = logits.argmax(axis=1).cpu().numpy()
        nn_class_score = f1_score(y_test[valid_mask], nn_preds, average='weighted')

        results[f'knn_{task["column"]}_score'] = knn_class_score
        results[f'nn_{task["column"]}_score'] = nn_class_score

    # Regression tasks
    for idx, task in enumerate(regression_tasks):
        col_name = task
        train_not_na = ~pd.isna(train_metadata[col_name])
        test_not_na = ~pd.isna(test_metadata[col_name])

        X_train = train_reps[train_not_na]
        y_train = train_metadata[col_name][train_not_na].astype(float).values

        X_test = test_reps[test_not_na]
        y_test = test_metadata[col_name][test_not_na].astype(float).values

        if len(X_train) == 0 or len(X_test) == 0:
            results.update({f'knn_{task}_{m}': np.nan for m in ['mse', 'mae', 'r2']})
            results.update({f'nn_{task}_{m}': np.nan for m in ['mse', 'mae', 'r2']})
            continue

        knn_reg = KNeighborsRegressor(n_neighbors=3)
        knn_reg.fit(X_train, y_train)
        knn_preds = knn_reg.predict(X_test)

        knn_mse = mean_squared_error(y_test, knn_preds)
        knn_mae = mean_absolute_error(y_test, knn_preds)
        knn_r2 = r2_score(y_test, knn_preds)

        with torch.no_grad():
            outputs = regressors[idx](torch.Tensor(X_test).to(device)).squeeze().cpu().numpy()
        nn_mse = mean_squared_error(y_test, outputs)
        nn_mae = mean_absolute_error(y_test, outputs)
        nn_r2 = r2_score(y_test, outputs)

        results[f'knn_{task}_mse'] = knn_mse
        results[f'knn_{task}_mae'] = knn_mae
        results[f'knn_{task}_r2'] = knn_r2
        results[f'nn_{task}_mse'] = nn_mse
        results[f'nn_{task}_mae'] = nn_mae
        results[f'nn_{task}_r2'] = nn_r2


    # Ordinal regression tasks
    for idx, task in enumerate(ordinal_regression_tasks):
        col_name = task['encoded_col']
        num_classes = task['n_classes']

        train_not_na = ~pd.isna(train_metadata[col_name])
        test_not_na = ~pd.isna(test_metadata[col_name])

        X_train = train_reps[train_not_na]
        y_train = train_metadata[col_name][train_not_na].astype(int).values

        X_test = test_reps[test_not_na]
        y_test = test_metadata[col_name][test_not_na].astype(int).values

        if len(X_train) == 0 or len(X_test) == 0:
            results[f'knn_{task["column"]}_accuracy'] = np.nan
            results[f'knn_{task["column"]}_mse'] = np.nan
            results[f'nn_{task["column"]}_accuracy'] = np.nan
            results[f'nn_{task["column"]}_mse'] = np.nan
            continue

        knn_reg = KNeighborsRegressor(n_neighbors=3)
        knn_reg.fit(X_train, y_train)
        knn_preds_continuous = knn_reg.predict(X_test)
        knn_preds = np.clip(np.round(knn_preds_continuous).astype(int), 0, num_classes - 1)

        knn_accuracy = f1_score(y_test, knn_preds, average='weighted')
        knn_mse = mean_squared_error(y_test, knn_preds_continuous)

        with torch.no_grad():
            probas = ordinal_regressors[idx](torch.Tensor(X_test).to(device)).cpu().numpy()
        cumulative_probas = np.cumsum(probas, axis=1)
        nn_preds = np.clip((cumulative_probas > 0.5).sum(axis=1).astype(int), 0, num_classes - 1)
        nn_accuracy = f1_score(y_test, nn_preds, average='weighted')
        nn_mse = mean_squared_error(y_test, cumulative_probas.sum(axis=1))

        results[f'knn_{task["column"]}_accuracy'] = knn_accuracy
        results[f'knn_{task["column"]}_mse'] = knn_mse
        results[f'nn_{task["column"]}_accuracy'] = nn_accuracy
        results[f'nn_{task["column"]}_mse'] = nn_mse
        # Extra covariates evaluation
    if extra_covariates is not None:
        for cov in extra_covariates:
            if cov not in train_metadata.columns:
                results[f'knn_{cov}_score'] = np.nan
                continue

            train_not_na = ~pd.isna(train_metadata[cov])
            test_not_na = ~pd.isna(test_metadata[cov])

            X_train = train_reps[train_not_na]
            y_train = train_metadata[cov][train_not_na].astype(str).values

            X_test = test_reps[test_not_na]
            y_test = test_metadata[cov][test_not_na].astype(str).values

            if len(X_train) == 0 or len(X_test) == 0:
                results[f'knn_{cov}_score'] = np.nan
                continue

            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(X_train, y_train)
            knn_preds = knn.predict(X_test)
            knn_class_score = f1_score(y_test, knn_preds, average='weighted')
            results[f'knn_{cov}_score'] = knn_class_score

    # Batch effect evaluation
    if batch_col_encoded is not None:
        train_not_na = ~pd.isna(train_metadata[batch_col_encoded])
        test_not_na = ~pd.isna(test_metadata[batch_col_encoded])

        X_train = train_reps[train_not_na]
        y_train = train_metadata[batch_col_encoded][train_not_na].astype(int).values

        X_test = test_reps[test_not_na]
        y_test = test_metadata[batch_col_encoded][test_not_na].astype(int).values

        if len(X_train) > 0 and len(X_test) > 0:
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(X_train, y_train)
            knn_preds = knn.predict(X_test)
            knn_batch_score = f1_score(y_test, knn_preds, average='weighted')
            results['knn_batch_score'] = knn_batch_score
            
            # if discriminator is not None:
            #     with torch.no_grad():
            #         logits = discriminator(torch.Tensor(X_test).to(device), grl_lambda=0.0)
            #     nn_preds = logits.argmax(axis=1).cpu().numpy()
            #     nn_batch_score = f1_score(y_test, nn_preds, average='weighted')
            #     results['nn_batch_score'] = nn_batch_score
            # else:
            #     results['nn_batch_score'] = np.nan
        else:
            results['knn_batch_score'] = np.nan
            results['nn_batch_score'] = np.nan

    return results

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
import pandas as pd

def plot_training_losses(model, save_fig=False, save_dir='./figures', plot_only_total_loss=False):
    """
    Plots training and validation losses/metrics over epochs from the ContrastiveModel on a single figure.
    
    Args:
        model: Instance of ContrastiveModel containing loss history.
        save_fig: Boolean indicating whether to save the figure.
        save_dir: Directory where the figure will be saved.
        plot_only_total_loss: If True, only the total loss is plotted.
    """
    total_losses = [epoch_loss['total_loss'] for epoch_loss in model.losses_per_epoch]
    contr_losses = [epoch_loss['contrastive_loss'] for epoch_loss in model.losses_per_epoch]
    discr_losses = [epoch_loss['discriminator_loss'] for epoch_loss in model.losses_per_epoch] if model.discriminator else None
    class_losses = [sum(epoch_loss['classification_losses']) for epoch_loss in model.losses_per_epoch] if model.classifiers else None
    reg_losses = [sum(epoch_loss['regression_losses']) for epoch_loss in model.losses_per_epoch] if model.regressors else None
    ord_losses = [sum(epoch_loss['ordinal_regression_losses']) for epoch_loss in model.losses_per_epoch] if model.ordinal_regressors else None

    val_losses = model.val_losses_per_epoch if hasattr(model, 'val_losses_per_epoch') else []

    plt.figure(figsize=(10, 5))
    # Plot train total loss
    plt.plot(total_losses, label="Train Total Loss")

    if not plot_only_total_loss:
        # Plot other train losses
        if contr_losses and any(loss != 0 for loss in contr_losses):
            plt.plot(contr_losses, label="Train Contrastive Loss")
        if class_losses and any(loss != 0 for loss in class_losses):
            plt.plot(class_losses, label="Train Classification Loss")
        if reg_losses and any(loss != 0 for loss in reg_losses):
            plt.plot(reg_losses, label="Train Regression Loss")
        if ord_losses and any(loss != 0 for loss in ord_losses):
            plt.plot(ord_losses, label="Train Ordinal Regression Loss")
        if discr_losses and any(loss != 0 for loss in discr_losses):
            plt.plot(discr_losses, label="Train Discriminator Loss")

    # Plot val loss if available
    if val_losses and not all(np.isnan(x) for x in val_losses):
        plt.plot(val_losses, label="Val Loss", color='red')

    # Plot val metrics if available
    val_metrics = model.val_metrics_per_epoch if hasattr(model, 'val_metrics_per_epoch') else []
    if val_metrics and not all(np.isnan(x) for x in val_metrics):
        plt.plot(val_metrics, label="Val Metric", color='green')

    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Metric")
    plt.title("Training and Validation Losses/Metrics Over Epochs")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend outside
    plt.tight_layout()
    if save_fig:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, "training_losses.png"))
    plt.close()  # Close the plot to prevent display


def plot_evaluation_metrics(model, save_fig=False, save_dir='./figures'):
    """
    Plots all accuracy metrics over epochs from the ContrastiveModel.
    """
    epochs = len(model.stage_2_eval_results_per_epoch)
    epoch_indices = list(range(1, epochs + 1))

    if save_fig and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(10, 5))
    has_any_accuracy = False

    if model.classification_tasks:
        for idx, task in enumerate(model.classification_tasks):
            knn_scores = [epoch_results.get(f'knn_{task["column"]}_score', np.nan) 
                          for epoch_results in model.stage_2_eval_results_per_epoch]
            plt.plot(epoch_indices, knn_scores, label=f"{task['column']} KNN Classification Accuracy")
            has_any_accuracy = True

    if model.ordinal_regression_tasks:
        for idx, task in enumerate(model.ordinal_regression_tasks):
            knn_accuracy = [epoch_results.get(f'knn_{task["column"]}_accuracy', np.nan) 
                            for epoch_results in model.stage_2_eval_results_per_epoch]
            plt.plot(epoch_indices, knn_accuracy, label=f"{task['column']} KNN Ordinal Accuracy")
            has_any_accuracy = True

    if 'batch_correction' in model.tasks:
        knn_batch_scores = [epoch_results.get('knn_batch_score', np.nan) 
                            for epoch_results in model.stage_2_eval_results_per_epoch]
        plt.plot(epoch_indices, knn_batch_scores, label="KNN Batch Correction Accuracy")
        has_any_accuracy = True

    if has_any_accuracy:
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Model KNN Accuracies Over Epochs")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        if save_fig:
            plt.savefig(os.path.join(save_dir, "model_accuracies.png"))
    else:
        print("No accuracy metrics to plot.")

    plt.show()

def plot_umap(representations, metadata, hue_column, palette='tab20', title=None, save_fig=False, save_dir='./figures', filename='umap.png', **kwargs):
    """
    Plots a UMAP projection of the representations colored by the specified metadata column.
    
    Args:
        representations (np.ndarray): Array of sample representations.
        metadata (pd.DataFrame): DataFrame containing metadata for samples.
        hue_column (str): Column name in metadata to color the UMAP by.
        palette (str or list): Color palette for the plot.
        title (str): Title of the plot.
        save_fig (bool): Whether to save the figure.
        save_dir (str): Directory to save the figure.
        filename (str): Filename for the saved figure.
        **kwargs: Additional keyword arguments for seaborn.scatterplot.
    """
    umap_reducer = UMAP()
    umap_features = umap_reducer.fit_transform(representations)

    plt.figure(figsize=(10, 5))
    sns.scatterplot(
        x=umap_features[:, 0],
        y=umap_features[:, 1],
        hue=metadata[hue_column],
        palette=palette,
        **kwargs
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend outside
    plt.title(title if title else f"UMAP colored by {hue_column}")
    plt.tight_layout()

    if save_fig:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, filename))
    plt.close()  # Close the plot to prevent display


def plot_losses(model):
    train_contrastive_losses = [losses["contrastive_loss"] for losses in model.loss_tracker.train_losses]
    val_contrastive_losses = [losses["contrastive_loss"] for losses in model.loss_tracker.val_losses]

    n_additional_plots = 0
    titles = []
    val_losses = []

    if model.discriminator:
        n_additional_plots += 1
        val_losses.append([losses["discriminator_loss"] for losses in model.loss_tracker.val_losses])
        titles.append("Discriminator validation loss")

    for i in range(len(model.classifiers)):
        n_additional_plots += 1
        task_name = model.classification_tasks[i]["column"]
        val_losses.append([losses["classification_losses"][i] for losses in model.loss_tracker.val_losses])
        titles.append(f"{task_name} classifier validation loss")

    for i in range(len(model.regressors)):
        n_additional_plots += 1
        task_name = model.regression_tasks[i]
        val_losses.append([losses["regression_losses"][i] for losses in model.loss_tracker.val_losses])
        titles.append(f"{task_name} regression validation loss")

    for i in range(len(model.ordinal_regressors)):
        n_additional_plots += 1
        task_name = model.ordinal_regression_tasks[i]["column"]
        val_losses.append([losses["ordinal_regression_losses"][i] for losses in model.loss_tracker.val_losses])
        titles.append(f"{task_name} ordinal regression validation loss")

    fig, axes = plt.subplots(nrows=1, ncols=n_additional_plots + 1, figsize=((n_additional_plots + 1) * 5, 5))

    if n_additional_plots == 0:
        # Contrastive loss is the only plot, so the type of `axes` is different
        ax = axes
    else:
        ax = axes[0]

    ax.plot(train_contrastive_losses, label='train')
    ax.plot(val_contrastive_losses, label='val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(loc="upper right")
    # plt.xscale('log')
    # plt.yscale('log')
    ax.set_title("Contrastive loss")

    for i, loss in enumerate(val_losses):
        axes[i + 1].plot(loss)
        axes[i + 1].set_xlabel('Epoch')
        axes[i + 1].set_ylabel('Loss')
        axes[i + 1].set_title(titles[i])
        # plt.xscale('log')
        axes[i + 1].set_yscale('log')

    return fig