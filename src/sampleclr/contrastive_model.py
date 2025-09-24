import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import logging
from enum import Enum
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from sklearn.preprocessing import LabelEncoder
from .datasets import SamplesDataset, TransformedPairDataset
from .losses import InfoNCECauchy, ordinal_regression_loss,InfoNCECauchyBatchAware,InfoNCECosine, InfoNCEGaussian, XSampleCLR
from .models import (
    DynamicNetwork,
    MultiHeadAggregationNetwork,
    ClassifierHead,
    RegressionHead,
    OrdinalRegressionHead,
    DiscriminatorHead,
    UncertaintyWeightingLoss
)
from .utils import evaluate_knn_nn_batch

# Configure logging
logger = logging.getLogger(__name__)

class TrainingStage(Enum):
    """Enum for different training stages."""
    CONTRASTIVE = "contrastive"
    JOINT = "joint"
    SUPERVISED_WITH_AGG = "only_supervised_with_agg"

class LossTracker:
    """Class to manage and track all training and validation losses."""
    
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.epoch_metrics = []
        
    def add_epoch(self, train_losses, val_losses=None):
        """
        Add losses for a new epoch.
        
        Args:
            train_losses (dict): Training losses for the epoch
            val_losses (dict, optional): Validation losses for the epoch
        """
        self.train_losses.append(train_losses)
        if val_losses is not None:
            self.val_losses.append(val_losses)
            
    def get_latest_train_losses(self):
        """
        Get the latest training losses.
        
        Returns:
            dict: Latest training losses, or empty dict if no epochs recorded
        """
        return self.train_losses[-1] if self.train_losses else {}
        
    def get_latest_val_losses(self):
        """
        Get the latest validation losses.
        
        Returns:
            dict: Latest validation losses, or empty dict if no epochs recorded
        """
        return self.val_losses[-1] if self.val_losses else {}
        
    def format_loss_report(self, train_losses, val_losses=None, tasks_info=None):
        """
        Format losses in the requested format.
        
        Creates a structured report showing training and validation losses
        for all components in a consistent format. Only shows losses that
        are actually present in the loss dictionaries.
        
        Args:
            train_losses (dict): Training losses for the epoch
            val_losses (dict, optional): Validation losses for the epoch
            tasks_info (dict, optional): Task information for better naming
            
        Returns:
            list: List of formatted loss report strings, one line per component
        """
        def format_loss_value(value):
            """Helper function to format loss values safely."""
            if isinstance(value, (int, float)) and not np.isnan(value):
                return f"{value:.4f}"
            else:
                return str(value)
        
        lines = []
        
        # Contrastive loss - only show if present
        if 'contrastive_loss' in train_losses:
            train_contr = train_losses['contrastive_loss']
            val_contr = val_losses.get('contrastive_loss', 'N/A') if val_losses else 'N/A'
            lines.append(f"contrastive – train: {format_loss_value(train_contr)} – val: {format_loss_value(val_contr)}")
        
        # Classification losses - only show if present and non-empty
        classification_losses = train_losses.get('classification_losses', [])
        if classification_losses and len(classification_losses) > 0:
            for i, loss in enumerate(classification_losses):
                if tasks_info and 'classification_tasks' in tasks_info and i < len(tasks_info['classification_tasks']):
                    task_name = f"classification ({tasks_info['classification_tasks'][i]['column']})"
                else:
                    task_name = f"classification_{i}"
                train_val = loss
                val_val = val_losses.get('classification_losses', [])[i] if val_losses and i < len(val_losses.get('classification_losses', [])) else 'N/A'
                lines.append(f"{task_name} – train: {format_loss_value(train_val)} – val: {format_loss_value(val_val)}")
            
        # Regression losses - only show if present and non-empty
        regression_losses = train_losses.get('regression_losses', [])
        if regression_losses and len(regression_losses) > 0:
            for i, loss in enumerate(regression_losses):
                if tasks_info and 'regression_tasks' in tasks_info and i < len(tasks_info['regression_tasks']):
                    task_name = f"regression ({tasks_info['regression_tasks'][i]})"
                else:
                    task_name = f"regression_{i}"
                train_val = loss
                val_val = val_losses.get('regression_losses', [])[i] if val_losses and i < len(val_losses.get('regression_losses', [])) else 'N/A'
                lines.append(f"{task_name} – train: {format_loss_value(train_val)} – val: {format_loss_value(val_val)}")
            
        # Ordinal regression losses - only show if present and non-empty
        ordinal_losses = train_losses.get('ordinal_regression_losses', [])
        if ordinal_losses and len(ordinal_losses) > 0:
            for i, loss in enumerate(ordinal_losses):
                if tasks_info and 'ordinal_regression_tasks' in tasks_info and i < len(tasks_info['ordinal_regression_tasks']):
                    task_name = f"ordinal_regression ({tasks_info['ordinal_regression_tasks'][i]['column']})"
                else:
                    task_name = f"ordinal_regression_{i}"
                train_val = loss
                val_val = val_losses.get('ordinal_regression_losses', [])[i] if val_losses and i < len(val_losses.get('ordinal_regression_losses', [])) else 'N/A'
                lines.append(f"{task_name} – train: {format_loss_value(train_val)} – val: {format_loss_value(val_val)}")
            
        # Discriminator loss - only show if present
        if 'discriminator_loss' in train_losses and train_losses['discriminator_loss'] > 0:
            train_disc = train_losses['discriminator_loss']
            val_disc = val_losses.get('discriminator_loss', 'N/A') if val_losses else 'N/A'
            lines.append(f"discriminator – train: {format_loss_value(train_disc)} – val: {format_loss_value(val_disc)}")
        
        return lines

def label_encode_column(adata, col):
    """
    Helper function to encode a categorical column in adata.obs if not already encoded.
    Returns the name of the encoded column.
    """
    enc_col = f"{col}_encoded"
    if enc_col not in adata.obs.columns:
        labels_known = ~pd.isna(adata.obs[col])
        encoder = LabelEncoder().fit(adata.obs.loc[labels_known,col])
        adata.obs[enc_col] = np.nan
        adata.obs.loc[labels_known,enc_col] = encoder.transform(adata.obs.loc[labels_known,col])
    return enc_col

def _compute_classification_losses(features_1, features_2, classifiers, classification_tasks, 
                                 labels, criterion, device):
    """
    Compute classification losses for all classifiers.
    
    Args:
        features_1 (torch.Tensor): Features from first sample, shape (batch_size, feature_dim)
        features_2 (torch.Tensor): Features from second sample, shape (batch_size, feature_dim)
        classifiers (list): List of classifier models
        classification_tasks (list): List of classification task configurations
        labels (dict): Dictionary mapping task names to label tensors
        criterion (torch.nn.Module): Loss criterion (e.g., CrossEntropyLoss)
        device (torch.device): Device to compute losses on
        
    Returns:
        list: List of classification losses for each task
    """
    losses = []
    for idx, clf in enumerate(classifiers):
        col = classification_tasks[idx]['encoded_col']
        label = labels.get(col, None)
        if label is not None:
            label = label.to(device)
            known_labels = ~torch.isnan(label)
            if known_labels.sum() > 0:
                labels_batch = label[known_labels].long()
                logits_1 = clf(features_1[known_labels])
                logits_2 = clf(features_2[known_labels])
                classification_loss = criterion(logits_1, labels_batch) + criterion(logits_2, labels_batch)
                losses.append(classification_loss)
            else:
                losses.append(torch.tensor(0.0, device=device))
        else:
            losses.append(torch.tensor(0.0, device=device))
    return losses

def _compute_regression_losses(features_1, features_2, regressors, regression_tasks, 
                             regression_targets, criterion, device):
    """
    Compute regression losses for all regressors.
    
    Args:
        features_1 (torch.Tensor): Features from first sample, shape (batch_size, feature_dim)
        features_2 (torch.Tensor): Features from second sample, shape (batch_size, feature_dim)
        regressors (list): List of regressor models
        regression_tasks (list): List of regression task names
        regression_targets (dict): Dictionary mapping task names to target tensors
        criterion (torch.nn.Module): Loss criterion (e.g., MSELoss)
        device (torch.device): Device to compute losses on
        
    Returns:
        list: List of regression losses for each task
    """
    losses = []
    for idx, reg in enumerate(regressors):
        col = regression_tasks[idx]
        target = regression_targets.get(col, None)
        if target is not None:
            target = target.to(device)
            known_targets = ~torch.isnan(target)
            if known_targets.sum() > 0:
                target_batch = target[known_targets].float()
                output_1 = reg(features_1[known_targets])
                output_2 = reg(features_2[known_targets])
                regression_loss = criterion(output_1, target_batch.unsqueeze(1)) + criterion(output_2, target_batch.unsqueeze(1))
                losses.append(regression_loss)
            else:
                losses.append(torch.tensor(0.0, device=device))
        else:
            losses.append(torch.tensor(0.0, device=device))
    return losses

def _compute_ordinal_regression_losses(features_1, features_2, ordinal_regressors, 
                                     ordinal_regression_tasks, ordinal_targets, criterion, device):
    """
    Compute ordinal regression losses for all ordinal regressors.
    
    Args:
        features_1 (torch.Tensor): Features from first sample, shape (batch_size, feature_dim)
        features_2 (torch.Tensor): Features from second sample, shape (batch_size, feature_dim)
        ordinal_regressors (list): List of ordinal regressor models
        ordinal_regression_tasks (list): List of ordinal regression task configurations
        ordinal_targets (dict): Dictionary mapping task names to target tensors
        criterion (callable): Loss function for ordinal regression
        device (torch.device): Device to compute losses on
        
    Returns:
        list: List of ordinal regression losses for each task
    """
    losses = []
    for idx, ord_reg in enumerate(ordinal_regressors):
        col = ordinal_regression_tasks[idx]['encoded_col']
        target = ordinal_targets.get(col, None)
        if target is not None:
            target = target.to(device)
            known_targets = ~torch.isnan(target)
            if known_targets.sum() > 0:
                target_batch = target[known_targets].long()
                probas_1 = ord_reg(features_1[known_targets])
                probas_2 = ord_reg(features_2[known_targets])
                ordinal_loss = criterion(probas_1, target_batch) + criterion(probas_2, target_batch)
                losses.append(ordinal_loss)
            else:
                losses.append(torch.tensor(0.0, device=device))
        else:
            losses.append(torch.tensor(0.0, device=device))
    return losses

def _compute_discriminator_loss(features_1, features_2, discriminator, batch_label, 
                              criterion, device, lambda_=0.0):
    """
    Compute discriminator loss for adversarial training.
    
    Args:
        features_1 (torch.Tensor): Features from first sample, shape (batch_size, feature_dim)
        features_2 (torch.Tensor): Features from second sample, shape (batch_size, feature_dim)
        discriminator (torch.nn.Module): Discriminator model
        batch_label (torch.Tensor): Batch labels for discriminator training
        criterion (torch.nn.Module): Loss criterion (e.g., CrossEntropyLoss)
        device (torch.device): Device to compute losses on
        lambda_ (float): Gradient reversal layer lambda parameter
        
    Returns:
        torch.Tensor: Discriminator loss value
    """
    if discriminator is None:
        return torch.tensor(0.0, device=device)
    
    # Ensure batch_label is on the correct device
    batch_label = batch_label.to(device)
    known_batch_labels = ~torch.isnan(batch_label)
    if known_batch_labels.sum() > 0:
        batch_labels = batch_label[known_batch_labels].long()
        discriminator_output_1 = discriminator(features_1[known_batch_labels], grl_lambda=lambda_)
        discriminator_output_2 = discriminator(features_2[known_batch_labels], grl_lambda=lambda_)
        
        discriminator_loss_1 = criterion(discriminator_output_1, batch_labels)
        discriminator_loss_2 = criterion(discriminator_output_2, batch_labels)
        
        discriminator_loss = discriminator_loss_1 + discriminator_loss_2
        return discriminator_loss
    else:
        return torch.tensor(0.0, device=device)

def _compute_validation_losses(model, val_loader, device):
    """
    Compute validation losses for all model components.
    
    Args:
        model (ContrastiveModel): The model instance containing all components
        val_loader (torch.utils.data.DataLoader): Validation data loader
        device (torch.device): Device to compute losses on
        
    Returns:
        dict: Dictionary containing validation losses for all components:
            - 'contrastive_loss': float
            - 'classification_losses': list of floats
            - 'regression_losses': list of floats
            - 'ordinal_regression_losses': list of floats
            - 'discriminator_loss': float
    """
    model.projector.eval()
    model.aggregator.eval()
    
    total_losses = {
        'contrastive_loss': 0.0,
        'classification_losses': [0.0] * len(model.classifiers),
        'regression_losses': [0.0] * len(model.regressors),
        'ordinal_regression_losses': [0.0] * len(model.ordinal_regressors),
        'discriminator_loss': 0.0
    }
    
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            random_sample_1, random_sample_2, labels, batch_label, regression, ordinal, sample_ids = batch
            random_sample_1 = random_sample_1.to(device)
            random_sample_2 = random_sample_2.to(device)
            batch_label = batch_label.to(device)
            
            # Move label dictionaries to device
            labels = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in labels.items()}
            regression = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in regression.items()}
            ordinal = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in ordinal.items()}
            
            # Compute features
            features_1, features_2, _, _ = model._compute_sample_representation(random_sample_1, random_sample_2)

            features = torch.vstack((features_1, features_2))
            
            # Contrastive loss
            contr_loss = model.criterion(features, sample_ids=sample_ids)
            total_losses['contrastive_loss'] += contr_loss.item()
            
            # Task-specific losses
            if model.classifiers:
                class_losses = _compute_classification_losses(
                    features_1, features_2, model.classifiers, 
                    model.classification_tasks, labels, 
                    model.classification_criterion, device
                )
                for i, loss in enumerate(class_losses):
                    total_losses['classification_losses'][i] += loss.item()
                    
            if model.regressors:
                reg_losses = _compute_regression_losses(
                    features_1, features_2, model.regressors,
                    model.regression_tasks, regression,
                    model.regression_criterion, device
                )
                for i, loss in enumerate(reg_losses):
                    total_losses['regression_losses'][i] += loss.item()
                    
            if model.ordinal_regressors:
                ord_losses = _compute_ordinal_regression_losses(
                    features_1, features_2, model.ordinal_regressors,
                    model.ordinal_regression_tasks, ordinal,
                    model.ordinal_regression_criterion, device
                )
                for i, loss in enumerate(ord_losses):
                    total_losses['ordinal_regression_losses'][i] += loss.item()
                    
            if model.discriminator:
                disc_loss = _compute_discriminator_loss(
                    model.aggregator(random_sample_1), model.aggregator(random_sample_2), model.discriminator,
                    batch_label, model.discriminator_criterion, device,
                    lambda_=model.lambda_
                )
                total_losses['discriminator_loss'] += disc_loss.item()
                
            count += 1
    
    # Average losses
    if count > 0:
        for i in range(len(total_losses['classification_losses'])):
            total_losses['classification_losses'][i] /= len(val_loader)
        for i in range(len(total_losses['regression_losses'])):
            total_losses['regression_losses'][i] /= len(val_loader)
        for i in range(len(total_losses['ordinal_regression_losses'])):
            total_losses['ordinal_regression_losses'][i] /= len(val_loader)
        total_losses['discriminator_loss'] /= len(val_loader)
    
    return total_losses


class ContrastiveModel:
    def __init__(
        self,
        adata,
        sample_key,
        tasks=None,
        layer='X_raw_counts',
        device=None,
        extra_covariates=None,
        num_layers=2,
        hidden_size=234,
        learning_rate_feature=0.0016082645125362449,
        learning_rate_discriminator=6.745614777659622e-05,
        weight_decay=0.0054298503547461335,
        batch_size=32,
        n_aggregator_heads=9,
        aggregator_num_layers=1,
        aggregator_hidden_size=128,
        aggregator_normalization='BatchNorm',
        aggregator_activation='sigmoid',
        aggregator_pruning_threshold=None,
        output_dim=211,
        lambda_=0.033824397252418315,
        n_cells_per_sample=1000,
        num_epochs_stage1=10,
        num_epochs_stage2=90,
        num_warmup_epochs_stage1=10,
        num_warmup_epochs_stage2=10,
        verbose=True,
        early_stopping_patience=10,
        classifier_num_layers=1,
        classifier_hidden_size=128,
        regression_num_layers=1,
        regression_hidden_size=128,
        ordinal_num_layers=1,
        ordinal_hidden_size=128,
        use_normalization=False,
        train_ids=None,      
        val_ids=None,       
        test_ids=None,
        diversity_loss_weight=1.0,
        use_gumbel= True,
        diversity_loss_temperature = 1.0,
        discriminator_toggle_threshold = 0.5,
        discriminator_num_layers = 1 ,
        discriminator_hidden_size = 128,
        feature_normalization= 'BatchNorm',
        contrastive_loss='InfoNCECosine',
        contrastive_loss_temperature = 1.0,
        xsample_clr_graph_temperature = 0.1,
        cell_weights_col=None,
        sample_similarity_graph: pd.DataFrame | None = None,
    ):
        """
        Initializes the ContrastiveModel with data, tasks, hyperparameters, and optional splits.

        Args:
            adata: AnnData object containing the dataset.
            sample_key: Column in adata.obs indicating sample IDs.
            tasks: Dictionary specifying tasks. Example:
                   {
                       'classification': ['Sex'],
                       'regression': ['Age'],
                       'ordinal_regression': ['Outcome'],
                       'batch_correction': 'Pool_ID'
                   }
            layer: The obsm layer to use.
            device: 'cpu' or 'cuda'.
            split_config: Dict with 'split_by','shared_cov','val_frac','test_frac' for dataset splitting.
            extra_covariates: Additional classification covariates for evaluation.
            num_layers,hidden_size,...: model/training hyperparams
            early_stopping_patience: How many epochs to wait before stopping if val loss does not improve.
        """
        # self.adata = adata.copy()
        self.adata = adata
        self.sample_key = sample_key
        self.tasks = tasks if tasks is not None else {}
        self.layer = layer
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.extra_covariates = extra_covariates if extra_covariates else []
        if not isinstance(self.extra_covariates, list):
            self.extra_covariates = [self.extra_covariates]
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.learning_rate_feature = learning_rate_feature
        self.learning_rate_discriminator = learning_rate_discriminator
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.n_aggregator_heads = n_aggregator_heads
        self.aggregator_num_layers = aggregator_num_layers
        self.aggregator_hidden_size = aggregator_hidden_size
        self.aggregator_normalization = aggregator_normalization
        self.aggregator_activation = aggregator_activation
        self.aggregator_pruning_threshold = aggregator_pruning_threshold
        self.output_dim = output_dim
        self.lambda_ = lambda_
        self.n_cells_per_sample = n_cells_per_sample
        self.num_epochs_stage1 = num_epochs_stage1
        self.num_epochs_stage2 = num_epochs_stage2
        self.num_warmup_epochs_stage1 = num_warmup_epochs_stage1
        self.num_warmup_epochs_stage2 = num_warmup_epochs_stage2
        self.verbose = verbose
        self.early_stopping_patience = early_stopping_patience

        self.classifier_num_layers = classifier_num_layers
        self.classifier_hidden_size = classifier_hidden_size
        self.regression_num_layers = regression_num_layers
        self.regression_hidden_size = regression_hidden_size
        self.ordinal_num_layers = ordinal_num_layers
        self.ordinal_hidden_size = ordinal_hidden_size
        self.use_normalization = use_normalization
        
        self.feature_normalization = feature_normalization

        self.cell_weights_col = cell_weights_col
        self.sample_similarity_graph = sample_similarity_graph

        self.discriminator_toggle_threshold = discriminator_toggle_threshold ##TODO:?!
        
        self.train_ids = train_ids
        self.val_ids = val_ids
        self.test_ids = test_ids
        
        self.diversity_loss_weight = diversity_loss_weight
        self.return_weights = self.diversity_loss_weight > 0
            
        self.use_gumbel = use_gumbel
        self.temperature = diversity_loss_temperature
        
        self.discriminator_num_layers = discriminator_num_layers
        self.discriminator_hidden_size = discriminator_hidden_size

        self.seed = 42
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Initialize loss tracker
        self.loss_tracker = LossTracker()
        
        self.stage_2_eval_results_per_epoch = []
        self.val_metrics_per_epoch = []

        self.classification_tasks = []
        self.regression_tasks = []
        self.ordinal_regression_tasks = []
        self.discriminator = None
        self.classifiers = []
        self.regressors = []
        self.ordinal_regressors = []

        self.contrastive_loss = contrastive_loss
        self.contrastive_loss_temperature = contrastive_loss_temperature
        self.xsample_clr_graph_temperature = xsample_clr_graph_temperature

        self.prepare_data()
        self.initialize_models()
        self.initialize_optimizers()

    def prepare_data(self):
        """
        Prepares the data for training, validation and testing.
        Assumes that the patient-level metadata is obtained by dropping duplicates
        based on self.sample_key.
        Uses externally provided patient ID lists (self.train_ids, self.val_ids, self.test_ids)
        if available; otherwise, all deduplicated patients are used as training.
        Also, it creates the corresponding SamplesDataset and TransformedPairDataset objects.
        """

        if 'classification' in self.tasks:
            for col in self.tasks['classification']:
                enc_col = label_encode_column(self.adata, col)
                self.classification_tasks.append({
                    'column': col,
                    'encoded_col': enc_col,
                    'n_classes': len(self.adata.obs[enc_col].dropna().unique())
                })

        if 'regression' in self.tasks:
            self.regression_tasks = self.tasks['regression']

        if 'ordinal_regression' in self.tasks:
            for col in self.tasks['ordinal_regression']:
                enc_col = label_encode_column(self.adata, col)
                n_classes = len(self.adata.obs[enc_col].dropna().unique())
                self.ordinal_regression_tasks.append({
                    'column': col,
                    'encoded_col': enc_col,
                    'n_classes': n_classes
                })

        batch_col_encoded = None
        if 'batch_correction' in self.tasks:
            col = self.tasks['batch_correction']
            enc_col = label_encode_column(self.adata, col)
            batch_col_encoded = enc_col

        classification_cols = [t['encoded_col'] for t in self.classification_tasks] if self.classification_tasks else None
        regression_cols = self.regression_tasks if self.regression_tasks else None
        ordinal_cols = [t['encoded_col'] for t in self.ordinal_regression_tasks] if self.ordinal_regression_tasks else None

        # Deduplicate patient-level metadata based on sample_key.
        metadata_all = self.adata.obs.drop_duplicates(subset=[self.sample_key]).set_index(self.sample_key)
        self.metadata_all = metadata_all

        # Use externally provided patient IDs if given; otherwise, use all patients.
        train_ids = self.train_ids if self.train_ids is not None else metadata_all.index.values
        val_ids = self.val_ids if self.val_ids is not None else []
        test_ids = self.test_ids if self.test_ids is not None else []
        
        self.train_dataset = SamplesDataset(
            data=self.adata,
            unique_categories=list(train_ids),
            sample_col=self.sample_key,
            classification_cols=classification_cols,
            regression_cols=regression_cols,
            ordinal_regression_cols=ordinal_cols,
            batch_col=batch_col_encoded,
            layer=self.layer,
            cell_weights_col=self.cell_weights_col
        )

        if len(val_ids) > 0:
            self.val_dataset = SamplesDataset(
                data=self.adata,
                unique_categories=list(val_ids),
                sample_col=self.sample_key,
                classification_cols=classification_cols,
                regression_cols=regression_cols,
                ordinal_regression_cols=ordinal_cols,
                batch_col=batch_col_encoded,
                layer=self.layer,
                cell_weights_col=self.cell_weights_col
            )
        else:
            self.val_dataset = None

        if len(test_ids) > 0:
            self.test_dataset = SamplesDataset(
                data=self.adata,
                unique_categories=list(test_ids),
                sample_col=self.sample_key,
                classification_cols=classification_cols,
                regression_cols=regression_cols,
                ordinal_regression_cols=ordinal_cols,
                batch_col=batch_col_encoded,
                layer=self.layer,
                cell_weights_col=self.cell_weights_col
            )
        else:
            self.test_dataset = None

        self.train_pair_dataset = TransformedPairDataset(
            dataset=self.train_dataset,
            subset_size=self.n_cells_per_sample,
            device=self.device
        )

        if self.val_dataset is not None:
            self.val_pair_dataset = TransformedPairDataset(
                dataset=self.val_dataset,
                subset_size=self.n_cells_per_sample,
                device=self.device
            )
        else:
            self.val_pair_dataset = None

        if self.test_dataset is not None:
            self.test_pair_dataset = TransformedPairDataset(
                dataset=self.test_dataset,
                subset_size=self.n_cells_per_sample,
                device=self.device
            )
        else:
            self.test_pair_dataset = None

        self.batch_col_encoded = batch_col_encoded

    def initialize_models(self):
        """
        Initialize projector, aggregator, and heads.
        """
        n_batches=0
        if self.batch_col_encoded is not None:
            n_batches=len(self.adata.obs[self.batch_col_encoded].dropna().unique())

        classification_n_classes=[t['n_classes'] for t in self.classification_tasks]
        ordinal_n_classes=[t['n_classes'] for t in self.ordinal_regression_tasks]

        self.projector = DynamicNetwork(
            n_input_features=self.train_dataset.sample_cells[0].shape[1]*self.n_aggregator_heads,
            n_output_features=self.output_dim,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            activation='relu',
            normalization=self.feature_normalization
        ).to(self.device)

        self.aggregator = MultiHeadAggregationNetwork(
            n_heads=self.n_aggregator_heads,
            num_layers=self.aggregator_num_layers,
            hidden_size=self.aggregator_hidden_size,
            normalization=self.aggregator_normalization,
            n_input_features=self.train_dataset.sample_cells[0].shape[1],
            activation=self.aggregator_activation,
            pruning_threshold=self.aggregator_pruning_threshold,
            use_gumbel=self.use_gumbel,
            temperature=self.temperature,
            return_weights=self.return_weights,
        ).to(self.device)

        self.classifiers=[]
        for n_classes in classification_n_classes:
            self.classifiers.append(
                ClassifierHead(
                    n_input_features=self.output_dim,
                    n_classes=n_classes,
                    num_layers=self.classifier_num_layers,
                    hidden_size=self.classifier_hidden_size,
                    use_normalization=self.use_normalization
                ).to(self.device)
            )
        self.regressors = []
        for _ in (self.regression_tasks if self.regression_tasks else []):
            self.regressors.append(
                RegressionHead(
                    input_dim=self.output_dim,
                    num_layers=self.regression_num_layers,
                    hidden_size=self.regression_hidden_size,
                    use_normalization=self.use_normalization
                ).to(self.device)
            )
        self.ordinal_regressors = []
        for n_classes in ordinal_n_classes:
            self.ordinal_regressors.append(
                OrdinalRegressionHead(
                    input_dim=self.output_dim,
                    num_classes=n_classes,
                    num_layers=self.ordinal_num_layers,
                    hidden_size=self.ordinal_hidden_size,
                    use_normalization=self.use_normalization
                ).to(self.device)
            )
            

        self.discriminator=None
        if n_batches > 0:
            self.discriminator = DiscriminatorHead(
                input_dim=self.train_dataset.sample_cells[0].shape[1] * self.n_aggregator_heads,
                num_classes=n_batches,
                num_layers=self.discriminator_num_layers,
                hidden_size=self.discriminator_hidden_size,      
                use_normalization=self.use_normalization
            ).to(self.device)

        # Set the contrastive loss criterion based on the provided name
        if self.contrastive_loss == "InfoNCECosine":
            self.criterion = InfoNCECosine(temperature=self.contrastive_loss_temperature)
        elif self.contrastive_loss == "InfoNCECauchy":
            self.criterion = InfoNCECauchy(temperature=self.contrastive_loss_temperature)
        elif self.contrastive_loss == "InfoNCEGaussian":
            self.criterion = InfoNCEGaussian(temperature=self.contrastive_loss_temperature)
        elif self.contrastive_loss == "InfoNCECauchyBatchAware" :
            self.criterion = InfoNCECauchyBatchAware(temperature=self.contrastive_loss_temperature)
        elif self.contrastive_loss == "XSampleCLR":
            self.criterion = XSampleCLR(similarity_graph=self.sample_similarity_graph, temperature=self.contrastive_loss_temperature, graph_temperature=self.xsample_clr_graph_temperature)
        else:
            raise ValueError(f"Unknown contrastive loss: {self.contrastive_loss}")

        self.classification_criterion=nn.CrossEntropyLoss() if self.classifiers else None
        self.regression_criterion=nn.MSELoss() if self.regressors else None
        self.ordinal_regression_criterion=ordinal_regression_loss if self.ordinal_regressors else None
        
        self.discriminator_criterion=nn.CrossEntropyLoss() if self.discriminator else None
        num_losses = 1 + len(self.classifiers) + len(self.regressors) + len(self.ordinal_regressors)
        if self.discriminator:
            num_losses += 1
        if self.diversity_loss_weight > 0:
            num_losses += 1
        self.uncertainty_weighting_loss=UncertaintyWeightingLoss(num_losses).to(self.device)

    def initialize_optimizers(self):
        """
        Initialize optimizers for features and discriminator.
        """
        feature_params=list(self.projector.parameters())+list(self.aggregator.parameters())
        for clf in self.classifiers:
            feature_params+=list(clf.parameters())
        for reg in self.regressors:
            feature_params+=list(reg.parameters())
        for ord_reg in self.ordinal_regressors:
            feature_params+=list(ord_reg.parameters())

        self.feature_optimizer=torch.optim.Adam(feature_params,lr=self.learning_rate_feature,weight_decay=self.weight_decay)
        self.feature_optimizer.add_param_group({'params': self.uncertainty_weighting_loss.parameters()})

        if self.discriminator is not None:
            self.discriminator_optimizer=torch.optim.SGD(self.discriminator.parameters(), lr=self.learning_rate_discriminator, weight_decay=0)
        else:
            self.discriminator_optimizer=None

    def initialize_lr_schedulers(self, warmup_epochs, num_epochs, warmup_end_factor=1.0):
        warmup_scheduler = LinearLR(
            self.feature_optimizer, start_factor=1e-8, end_factor=warmup_end_factor, total_iters=warmup_epochs
        )

        # Cosine annealing: decay LR after warmup
        cosine_scheduler = CosineAnnealingLR(
            self.feature_optimizer, T_max=num_epochs, eta_min=1e-6
        )

        # Combine: warmup then cosine
        self.scheduler = SequentialLR(
            self.feature_optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
        )

    def set_heads_mode(self, heads, requires_grad, mode):
        for head in heads:
            for param in head.parameters():
                param.requires_grad = requires_grad
            if mode == 'train':
                head.train()
            elif mode == 'eval':
                head.eval()

    def train_epoch(
        self,
        dataloader,
        stage='joint',  # Options: 'contrastive', 'joint'
        lambda_=0.1
    ):
        """
        Trains the model for one epoch using the specified stage.

        Args:
            dataloader: DataLoader for training data.
            stage: 'contrastive' or 'joint'
            lambda_: Lambda parameter for the Gradient Reversal Layer (GRL).

        Returns:
            tuple: Total loss, contrastive loss, classification losses, regression losses, ordinal regression losses, discriminator loss
        """
        epoch_loss = 0
        epoch_contr_loss = 0
        epoch_class_losses = [0.0] * len(self.classifiers)
        epoch_reg_losses = [0.0] * len(self.regressors)
        epoch_ord_losses = [0.0] * len(self.ordinal_regressors)
        epoch_discr_loss = 0

        for batch in dataloader:
            self.feature_optimizer.zero_grad()
            random_sample_1, random_sample_2, labels, batch_label, regression, ordinal, sample_ids = batch
            random_sample_1 = random_sample_1.to(self.device)
            random_sample_2 = random_sample_2.to(self.device)
            batch_label = batch_label.to(self.device)
            
            # Compute features
            features_1, features_2, attn1, attn2 = self._compute_sample_representation(random_sample_1, random_sample_2)
            features = torch.vstack((features_1, features_2))

            loss_components = []

            # Contrastive and diversity losses
            if stage in ['joint', 'contrastive']:
                contrastive_loss = self.criterion(features, sample_ids=sample_ids)
                loss_components.append(contrastive_loss)
                epoch_contr_loss += contrastive_loss.item()
                
                # Diversity loss
                if self.diversity_loss_weight > 0 and attn1 is not None and attn2 is not None:
                    diversity_loss1 = self.aggregator.diversity_loss(attn1)
                    diversity_loss2 = self.aggregator.diversity_loss(attn2)
                    diversity_loss = (diversity_loss1 + diversity_loss2) / (2.0 * len(dataloader))
                    loss_components.append(self.diversity_loss_weight * diversity_loss)

            # Supervised losses
            if stage in ['joint', 'only_supervised_with_agg']:
                # Update discriminator first (if exists)
                if self.discriminator:
                    epoch_discr_loss += self._update_discriminator(random_sample_1, random_sample_2, batch_label)

                # Task-specific losses
                if self.classifiers:
                    class_losses = _compute_classification_losses(
                        features_1, features_2, self.classifiers, 
                        self.classification_tasks, labels, 
                        self.classification_criterion, self.device
                    )
                    for i, loss in enumerate(class_losses):
                        loss_components.append(loss / len(dataloader))
                        epoch_class_losses[i] += loss.item()

                if self.regressors:
                    reg_losses = _compute_regression_losses(
                        features_1, features_2, self.regressors,
                        self.regression_tasks, regression,
                        self.regression_criterion, self.device
                    )
                    for i, loss in enumerate(reg_losses):
                        loss_components.append(loss / len(dataloader))
                        epoch_reg_losses[i] += loss.item()

                if self.ordinal_regressors:
                    ord_losses = _compute_ordinal_regression_losses(
                        features_1, features_2, self.ordinal_regressors,
                        self.ordinal_regression_tasks, ordinal,
                        self.ordinal_regression_criterion, self.device
                    )
                    for i, loss in enumerate(ord_losses):
                        loss_components.append(loss / len(dataloader))
                        epoch_ord_losses[i] += loss.item()

                # Discriminator loss for feature learning
                if self.discriminator:
                    disc_loss = _compute_discriminator_loss(
                        self.aggregator(random_sample_1), self.aggregator(random_sample_2), self.discriminator,
                        batch_label, self.discriminator_criterion, self.device,
                        lambda_=lambda_
                    )
                    loss_components.append(disc_loss / len(dataloader))
                    epoch_discr_loss += disc_loss.item()

            # Backward pass
            total_loss = sum(loss_components)
            total_loss.backward()
            self.feature_optimizer.step()
            self.scheduler.step()
            epoch_loss += total_loss.item()

        return epoch_loss, epoch_contr_loss, epoch_class_losses, epoch_reg_losses, epoch_ord_losses, epoch_discr_loss

    def _set_training_stage(self, stage):
        """
        Set the appropriate training mode for all model components based on stage.
        
        This method configures which components are trainable and which are in
        evaluation mode based on the training stage (contrastive, joint, or
        supervised with aggregator).
        
        Args:
            stage (str): Training stage identifier. Must be one of:
                        - 'contrastive': Only projector and aggregator are trainable
                        - 'joint': All components are trainable
                        - 'only_supervised_with_agg': Only aggregator and heads are trainable
                        
        Raises:
            ValueError: If stage is not one of the valid options
        """
        if stage == TrainingStage.CONTRASTIVE.value:
            self.set_heads_mode(self.classifiers, False, 'eval')
            self.set_heads_mode(self.regressors, False, 'eval')
            self.set_heads_mode(self.ordinal_regressors, False, 'eval')
            if self.discriminator is not None:
                self.set_heads_mode([self.discriminator], False, 'eval')
            self.set_heads_mode([self.projector, self.aggregator], True, 'train')

        elif stage == TrainingStage.JOINT.value:
            self.set_heads_mode([self.projector, self.aggregator], True, 'train')
            self.set_heads_mode(self.classifiers, True, 'train')
            self.set_heads_mode(self.regressors, True, 'train')
            self.set_heads_mode(self.ordinal_regressors, True, 'train')
            if self.discriminator is not None:
                self.set_heads_mode([self.discriminator], True, 'train')

        elif stage == TrainingStage.SUPERVISED_WITH_AGG.value:
            self.set_heads_mode([self.projector], False, 'eval')
            self.set_heads_mode([self.aggregator], True, 'train')
            self.set_heads_mode(self.classifiers, True, 'train')
            self.set_heads_mode(self.regressors, True, 'train')
            self.set_heads_mode(self.ordinal_regressors, True, 'train')
            if self.discriminator is not None:
                self.set_heads_mode([self.discriminator], True, 'train')

        else:
            raise ValueError(f"Stage must be one of {[s.value for s in TrainingStage]}")

    def _compute_sample_representation(self, sample_1, sample_2):
        """
        Compute sample representations for both samples using cell aggregator and projector.
        
        Args:
            sample_1 (torch.Tensor): First sample tensor
            sample_2 (torch.Tensor): Second sample tensor
            
        Returns:
            tuple: (features_1, features_2, attn1, attn2) where:
                - features_1, features_2: sample representations
                - attn1, attn2: Attention weights for cells in each sample (None if diversity loss is disabled)
        """
        sample_1_out = self.aggregator(sample_1)
        sample_2_out = self.aggregator(sample_2)

        if isinstance(sample_1_out, tuple):
            sample_1_agg, attn1 = sample_1_out
            sample_2_agg, attn2 = sample_2_out
        else:
            sample_1_agg = sample_1_out
            sample_2_agg = sample_2_out
            attn1, attn2 = None, None

        if self.contrastive_loss == "InfoNCECosine":
            # Normalize features as this loss is scale-invariant
            features_1 = torch.nn.functional.normalize(self.projector(sample_1_agg), dim=-1)
            features_2 = torch.nn.functional.normalize(self.projector(sample_2_agg), dim=-1)
        else:
            features_1 = self.projector(sample_1_agg)
            features_2 = self.projector(sample_2_agg)
        
        return features_1, features_2, attn1, attn2

    def _update_discriminator(self, random_sample_1, random_sample_2, batch_label):
        """
        Update discriminator with detached features.
        
        This method updates the discriminator using features that have been
        detached from the computation graph to prevent gradient flow back to
        the feature extractor during discriminator training.
        
        Args:
            random_sample_1 (torch.Tensor): Set of cells from first sample
            random_sample_2 (torch.Tensor): Set of cells from second sample
            batch_label (torch.Tensor): Batch labels for discriminator training
            
        Returns:
            float: Discriminator loss value for logging purposes
        """
        self.discriminator_optimizer.zero_grad()
        with torch.no_grad():
            features_detached_1 = self.aggregator(random_sample_1).detach()
            features_detached_2 = self.aggregator(random_sample_2).detach()

        discriminator_output_1 = self.discriminator(features_detached_1, grl_lambda=0.0)
        discriminator_output_2 = self.discriminator(features_detached_2, grl_lambda=0.0)

        known_batch_labels = ~torch.isnan(batch_label)
        if known_batch_labels.sum() > 0:
            batch_labels = batch_label[known_batch_labels].long()
            discriminator_loss_1 = self.discriminator_criterion(discriminator_output_1[known_batch_labels], batch_labels)
            discriminator_loss_2 = self.discriminator_criterion(discriminator_output_2[known_batch_labels], batch_labels)
            discriminator_loss = discriminator_loss_1 + discriminator_loss_2

            discriminator_loss.backward()
            self.discriminator_optimizer.step()
            return discriminator_loss.item()
        return 0.0
    
    def compute_total_val_loss(self):
        """
        Compute the total validation loss as the sum of the contrastive loss and the supervised losses
        from the classification, regression, and ordinal regression heads. The losses are averaged over the validation batches.
        The discriminator loss is not included.
        Returns a dictionary with individual average losses and the total loss.
        """
        if self.val_pair_dataset is None or len(self.val_pair_dataset) == 0:
            return None
            
        val_loader = DataLoader(self.val_pair_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)

        
        val_losses = _compute_validation_losses(self, val_loader, self.device)
        
        # Calculate total loss
        total_loss = (val_losses['contrastive_loss'] + 
                     sum(val_losses['classification_losses']) + 
                     sum(val_losses['regression_losses']) + 
                     sum(val_losses['ordinal_regression_losses']) + 
                     val_losses['discriminator_loss'])
        
        return total_loss


    def compute_val_contrastive_loss(self):
        """
        Compute the validation contrastive loss (Stage 1 if 'loss' metric chosen).
        """
        if self.val_pair_dataset is None or len(self.val_pair_dataset) == 0:
            return None
            
        val_loader = DataLoader(self.val_pair_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)
        
        val_losses = _compute_validation_losses(self, val_loader, self.device)
        return val_losses['contrastive_loss']

    def compute_epoch_validation_losses(self):
        """
        Compute validation losses for the current epoch.
        
        This method computes validation losses for all model components including
        contrastive loss, classification losses, regression losses, ordinal
        regression losses, and discriminator loss.
        
        Returns:
            dict or None: Dictionary containing validation losses for all components,
                         or None if no validation dataset is available.
                         Keys: 'contrastive_loss', 'classification_losses',
                               'regression_losses', 'ordinal_regression_losses',
                               'discriminator_loss'
        """
        try:
            if self.val_pair_dataset is None or len(self.val_pair_dataset) == 0:
                return None
                
            val_loader = DataLoader(self.val_pair_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)
            
            return _compute_validation_losses(self, val_loader, self.device)
        except Exception as e:
            logger.warning(f"Error computing validation losses: {e}")
            return None

    def format_epoch_loss_report(self, train_losses, val_losses=None):
        """
        Format the loss report for the current epoch using the LossTracker.
        
        This method creates a formatted report showing training and validation
        losses for all components in a consistent format.
        
        Args:
            train_losses (dict): Dictionary containing training losses for the epoch
            val_losses (dict, optional): Dictionary containing validation losses for the epoch.
                                       If None, only training losses are shown.
        
        Returns:
            list: List of formatted loss report strings, one line per component
        """
        tasks_info = {
            'classification_tasks': self.classification_tasks,
            'regression_tasks': self.regression_tasks,
            'ordinal_regression_tasks': self.ordinal_regression_tasks
        }
        
        return self.loss_tracker.format_loss_report(train_losses, val_losses, tasks_info)

    def get_stage1_validation_metric(self, val_results, stage1_val_metric='loss'):
        """
        Compute a validation metric for Stage 1 early stopping.

        'loss': use val_contrastive_loss from compute_val_contrastive_loss()
        'knn': use knn metric on shared_cov
        """
        if stage1_val_metric=='loss':
            # val_loss = val_results.get('val_contrastive_loss', None)
            val_loss = self.compute_val_contrastive_loss()
            if val_loss is None:
                return None
            # For early stopping, we want to minimize loss, so we invert sign to maximize metric
            return -val_loss
        elif stage1_val_metric=='knn':
            # For now, return None as split_config is not implemented
            # TODO: Implement proper split_config handling
            logger.warning("KNN validation metric not implemented yet. Returning None.")
            return None

    def get_stage2_validation_metric(self, val_results, stage2_val_metric='loss'):
        """
        Compute validation metric for Stage 2 early stopping.
        'nn' or 'knn'
        """
        
        if stage2_val_metric=='loss':
            # val_loss = val_results.get('val_contrastive_loss', None)
            val_loss = self.compute_val_contrastive_loss()
            if val_loss is None:
                return None
            # For early stopping, we want to minimize loss, so we invert sign to maximize metric
            return -val_loss
        
        elif stage2_val_metric == 'total':
            total_loss_dict = self.compute_total_val_loss()
            if total_loss_dict is None:
                return None
            # Return negative total loss so that lower total loss gives a higher metric.
            return -total_loss_dict
        
        metrics_list=[]

        if stage2_val_metric == 'nn':
            # Ordinal tasks nn accuracy
            for t in self.ordinal_regression_tasks:
                col=t['column']
                acc=val_results.get(f'nn_{col}_accuracy',np.nan)
                if not np.isnan(acc):
                    metrics_list.append(acc)

            # Classification tasks nn score
            for t in self.classification_tasks:
                col=t['column']
                sc=val_results.get(f'nn_{col}_score',np.nan)
                if not np.isnan(sc):
                    metrics_list.append(sc)

            # Regression tasks nn r2
            if self.regression_tasks:
                for col in self.regression_tasks:
                    r2=val_results.get(f'nn_{col}_r2',np.nan)
                    if not np.isnan(r2):
                        metrics_list.append(r2)

            # Batch -nn_batch_score
            nn_batch=val_results.get('nn_batch_score',np.nan)
            if not np.isnan(nn_batch):
                metrics_list.append(-nn_batch)

        elif stage2_val_metric=='knn':
            # Use knn_ metrics for ordinal, classification, regression + -knn_batch_score
            for t in self.ordinal_regression_tasks:
                col=t['column']
                acc=val_results.get(f'knn_{col}_accuracy',np.nan)
                if not np.isnan(acc):
                    metrics_list.append(acc)

            for t in self.classification_tasks:
                col=t['column']
                sc=val_results.get(f'knn_{col}_score',np.nan)
                if not np.isnan(sc):
                    metrics_list.append(sc)

            if self.regression_tasks:
                for col in self.regression_tasks:
                    r2=val_results.get(f'knn_{col}_r2',np.nan)
                    if not np.isnan(r2):
                        metrics_list.append(r2)

            knn_batch=val_results.get('knn_batch_score',np.nan)
            if not np.isnan(knn_batch):
                metrics_list.append(-knn_batch)

        if len(metrics_list)==0:
            return None
        return np.mean(metrics_list)

    def evaluate_split(self,train_dataset,train_metadata,test_dataset,test_metadata):
        return evaluate_knn_nn_batch(
            projector=self.projector,
            aggregator=self.aggregator,
            classifiers=self.classifiers,
            regressors=self.regressors,
            ordinal_regressors=self.ordinal_regressors,
            discriminator=self.discriminator,
            train_dataset=train_dataset,
            train_metadata=train_metadata,
            test_dataset=test_dataset,
            test_metadata=test_metadata,
            classification_tasks=self.classification_tasks,
            regression_tasks=self.regression_tasks,
            ordinal_regression_tasks=self.ordinal_regression_tasks,
            batch_col_encoded=self.batch_col_encoded,
            device=self.device,
            extra_covariates=self.extra_covariates
        )

    def train(
        self,
        num_epochs_stage1=None,
        num_epochs_stage2=None,
        two_stages = True,
        stage_2 = "joint", # 'joint','only_supervised_with_agg'
        lambda_=None,
        verbose=True,
        stage1_val_metric='knn', # 'loss' or 'knn'
        stage2_val_metric='knn'  # 'nn' or 'knn' or 'loss'
    ):
        """
        Trains the model using the specified training strategy.
        
        If early stopping triggers in stage 1, we do NOT stop training entirely,
        we proceed to stage 2.
        
        After stage 1 finishes (early stopped or max epochs), we print evaluation on validation set.
        
        In stage 2, we print evaluation after every epoch and use chosen metric for early stopping.
        
        We store all losses and metrics for plotting.
        """
        def log(*args, **kwargs):
            if verbose:
                print(*args, **kwargs)

        num_epochs_stage1 = num_epochs_stage1 if num_epochs_stage1 else self.num_epochs_stage1
        num_epochs_stage2 = num_epochs_stage2 if num_epochs_stage2 else self.num_epochs_stage2
        original_lambda = lambda_ if lambda_ is not None else self.lambda_

        has_val = self.val_dataset is not None and len(self.val_dataset) > 0
        num_epochs_stage1_trained = 0 
        num_epochs_stage2_trained = 0

        train_loader = DataLoader(self.train_pair_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        self.initialize_lr_schedulers(self.num_warmup_epochs_stage1, num_epochs_stage1 * 2)

        # Stage 1
        if two_stages:
            log("Stage 1: Contrastive-only training")
            
            best_stage1_metric=-np.inf
            no_improve_count_stage1=0
            best_stage1_model_state=None

            for epoch in range(num_epochs_stage1):
                num_epochs_stage1_trained += 1
                epoch_loss, epoch_contr_loss, epoch_class_losses, epoch_reg_losses, epoch_ord_losses, epoch_discr_loss = self.train_epoch(train_loader, stage='contrastive', lambda_=original_lambda)
                epoch_losses={
                    'total_loss': epoch_loss,
                    'contrastive_loss': epoch_contr_loss,
                    'classification_losses': epoch_class_losses,
                    'regression_losses': epoch_reg_losses,
                    'ordinal_regression_losses': epoch_ord_losses,
                    'discriminator_loss': epoch_discr_loss,
                    'learning_rate_aggregator': self.scheduler.get_last_lr()[0],
                    'learning_rate_projector': self.scheduler.get_last_lr()[1],
                }
                
                
                # Display basic training loss info for Stage 1
                log(f"Epoch {epoch+1}/{num_epochs_stage1} - Total Loss: {epoch_loss:.4f}, Contrastive Loss: {epoch_contr_loss:.4f}")

                if has_val:
                    try:
                        # Compute validation losses for this epoch
                        val_losses = self.compute_epoch_validation_losses()
                        
                        # Add to loss tracker
                        self.loss_tracker.add_epoch(epoch_losses, val_losses)
                        
                        # Get stage 1 metric for early stopping
                        stage1_metric = self.get_stage1_validation_metric(None, stage1_val_metric=stage1_val_metric)
                                                
                        if stage1_metric is not None:
                            self.val_metrics_per_epoch.append(stage1_metric)
                            if stage1_metric > best_stage1_metric:
                                best_stage1_metric = stage1_metric
                                best_stage1_model_state = {
                                    "projector": self.projector.state_dict(),
                                    "aggregator": self.aggregator.state_dict(),
                                    "classifiers": [clf.state_dict() for clf in self.classifiers],
                                    "regressors": [reg.state_dict() for reg in self.regressors],
                                    "ordinal_regressors": [ord_reg.state_dict() for ord_reg in self.ordinal_regressors],
                                    "discriminator": self.discriminator.state_dict() if self.discriminator else None,
                                    "uncertainty_weighting_loss": self.uncertainty_weighting_loss.state_dict(),
                                    "lr_scheduler": self.scheduler.state_dict(),
                                }
                                no_improve_count_stage1 = 0
                            else:
                                no_improve_count_stage1 += 1
                                if no_improve_count_stage1 > self.early_stopping_patience:
                                    log("Early stopping triggered in Stage 1. Will proceed to Stage 2.")
                                    # restore best stage 1 model
                                    self.projector.load_state_dict(best_stage1_model_state['projector'])
                                    self.aggregator.load_state_dict(best_stage1_model_state['aggregator'])
                                    for clf, state in zip(self.classifiers, best_stage1_model_state["classifiers"]):
                                        clf.load_state_dict(state)
                                    for reg, state in zip(self.regressors, best_stage1_model_state["regressors"]):
                                        reg.load_state_dict(state)
                                    for ord_reg, state in zip(self.ordinal_regressors, best_stage1_model_state["ordinal_regressors"]):
                                        ord_reg.load_state_dict(state)
                                    if best_stage1_model_state["discriminator"] is not None and self.discriminator is not None:
                                        self.discriminator.load_state_dict(best_stage1_model_state["discriminator"])
                                    self.uncertainty_weighting_loss.load_state_dict(best_stage1_model_state["uncertainty_weighting_loss"])
                                    self.scheduler.load_state_dict(best_stage1_model_state["lr_scheduler"])
                                    break
                        else:
                            self.val_metrics_per_epoch.append(np.nan)
                        
                        # Show full loss report after validation
                        if val_losses is not None:
                            loss_report = self.format_epoch_loss_report(epoch_losses, val_losses)
                            log("Full Loss Report:")
                            for line in loss_report:
                                log(line)
                    except Exception as e:
                        logger.error(f"Error in Stage 1 validation: {e}")
                else:
                    # No validation set
                    self.loss_tracker.add_epoch(epoch_losses)
                    self.val_metrics_per_epoch.append(np.nan)

            # # After stage 1 finishes (either completed or early stopped), evaluate on val set (if available)
            # if has_val:
            #     val_results_stage1=self.evaluate_split(self.train_dataset,self.metadata_all.loc[self.train_dataset.unique_categories],self.val_dataset,self.metadata_all.loc[self.val_dataset.unique_categories])
            #     # Print evaluation result before starting Stage 2
            #     log("Evaluation after Stage 1 (before Stage 2):")
            #     eval_strs=[]
            #     for metric,value in val_results_stage1.items():
            #         if isinstance(value,float):
            #             eval_strs.append(f"{metric}: {value:.4f}")
            #     log(", ".join(eval_strs))
        else:
            # If not two_stages, just do joint training stage
            pass

        # Stage 2
        log(f"Stage {stage_2}: training")
        best_metric = -np.inf
        best_model_state = {}
        no_improve_count = 0
        self._set_training_stage(stage=stage_2)

        self.initialize_lr_schedulers(self.num_warmup_epochs_stage2, 2 * num_epochs_stage2, warmup_end_factor=1.0)
        
        if has_val:
            val_res=self.evaluate_split(self.train_dataset,self.metadata_all.loc[self.train_dataset.unique_categories],self.val_dataset,self.metadata_all.loc[self.val_dataset.unique_categories])
            self.discriminator_toggle_threshold = val_res.get('knn_batch_score', 0.7) - self.discriminator_toggle_threshold
        else:
            train_res =self.evaluate_split(self.train_dataset,self.metadata_all.loc[self.train_dataset.unique_categories],self.train_dataset,self.metadata_all.loc[self.train_dataset.unique_categories])
            self.discriminator_toggle_threshold = train_res.get('knn_batch_score', 0.7) - self.discriminator_toggle_threshold

        # Freeze the backbone while additional heads are not trained yet
        if self.classifiers or self.regressors or self.ordinal_regressors or self.discriminator is not None:
            log(f"Freezing the backbone for {self.num_warmup_epochs_stage2} epochs")
            self.set_heads_mode([self.projector, self.aggregator], requires_grad=False, mode='eval')
            is_backbone_frozen = True
        else:
            is_backbone_frozen = False

        for epoch in range(num_epochs_stage2):
            num_epochs_stage2_trained += 1
            if epoch > self.num_warmup_epochs_stage2 and is_backbone_frozen:
                is_backbone_frozen = False
                self.set_heads_mode([self.projector, self.aggregator], requires_grad=True, mode='train')
                log(f"Unfreezing the backbone")

            epoch_loss, epoch_contr_loss, epoch_class_losses, epoch_reg_losses, epoch_ord_losses, epoch_discr_loss=self.train_epoch(train_loader,stage=stage_2,lambda_=original_lambda)
            epoch_losses={
                'total_loss':epoch_loss,
                'contrastive_loss':epoch_contr_loss,
                'classification_losses':epoch_class_losses,
                'regression_losses':epoch_reg_losses,
                'ordinal_regression_losses':epoch_ord_losses,
                'discriminator_loss':epoch_discr_loss,
                'learning_rate_aggregator': self.scheduler.get_last_lr()[0],
                'learning_rate_projector': self.scheduler.get_last_lr()[1],
            }
            if has_val:
                try:
                    # Compute validation losses for this epoch
                    val_losses = self.compute_epoch_validation_losses()
                    
                    # Add to loss tracker
                    self.loss_tracker.add_epoch(epoch_losses, val_losses)
                    
                    # Get evaluation results for metrics
                    val_results = self.evaluate_split(self.train_dataset, self.metadata_all.loc[self.train_dataset.unique_categories], 
                                                   self.val_dataset, self.metadata_all.loc[self.val_dataset.unique_categories])
                    self.stage_2_eval_results_per_epoch.append(val_results)
                    
                    # Get stage 2 metric for early stopping
                    val_metric = self.get_stage2_validation_metric(None, stage2_val_metric=stage2_val_metric)
                    
                    # Store for backward compatibility
                    self.val_metrics_per_epoch.append(val_metric if val_metric is not None else np.nan)
                    
                    # Adjust lambda_ based on batch effect
                    knn_batch_score = val_results.get('knn_batch_score', np.nan)
                    if not np.isnan(knn_batch_score) and self.discriminator:
                        if knn_batch_score >= self.discriminator_toggle_threshold:
                            self.lambda_ = original_lambda
                            log(f"Batch effect high (knn_batch_score={knn_batch_score:.4f}), enabling adversarial training (lambda={self.lambda_})")
                        else:
                            self.discriminator_toggle_threshold = knn_batch_score
                            self.lambda_ = 0.0
                            log(f"Batch effect low (knn_batch_score={knn_batch_score:.4f}), disabling adversarial training (lambda={self.lambda_})")
                    
                    # Format and display loss report
                    loss_report = self.format_epoch_loss_report(epoch_losses, val_losses)
                    log(f"\nEpoch {epoch+1}/{num_epochs_stage2}")
                    for line in loss_report:
                        log(line)
                    log("-" * 30)
                except Exception as e:
                    logger.error(f"Error in Stage 2 validation: {e}")
                    # Continue training without validation for this epoch
                    self.val_metrics_per_epoch.append(np.nan)
                    log(f"\nEpoch {epoch+1}/{num_epochs_stage2} - Error in validation, continuing training")
                    log("-" * 30)
                    val_metric = None

                if val_metric is not None:
                    if val_metric>best_metric:
                        best_metric=val_metric
                        best_model_state={
                            "projector": self.projector.state_dict(),
                            "aggregator": self.aggregator.state_dict(),
                            "classifiers":[clf.state_dict() for clf in self.classifiers],
                            "regressors":[reg.state_dict() for reg in self.regressors],
                            "ordinal_regressors":[ord_reg.state_dict() for ord_reg in self.ordinal_regressors],
                            "discriminator":self.discriminator.state_dict() if self.discriminator else None,
                            "uncertainty_weighting_loss":self.uncertainty_weighting_loss.state_dict(),
                            "lr_scheduler": self.scheduler.state_dict(),
                        }
                        no_improve_count=0
                    else:
                        no_improve_count+=1
                        if no_improve_count>self.early_stopping_patience:
                            log("Early stopping triggered in Stage 2.")
                            self.projector.load_state_dict(best_model_state['projector'])
                            self.aggregator.load_state_dict(best_model_state['aggregator'])
                            for clf, state in zip(self.classifiers, best_model_state["classifiers"]):
                                clf.load_state_dict(state)
                            for reg, state in zip(self.regressors, best_model_state["regressors"]):
                                reg.load_state_dict(state)
                            for ord_reg, state in zip(self.ordinal_regressors, best_model_state["ordinal_regressors"]):
                                ord_reg.load_state_dict(state)
                            if best_model_state["discriminator"] is not None and self.discriminator is not None:
                                self.discriminator.load_state_dict(best_model_state["discriminator"])
                            self.uncertainty_weighting_loss.load_state_dict(best_model_state["uncertainty_weighting_loss"])
                            self.scheduler.load_state_dict(best_model_state["lr_scheduler"])
                            break

            else:
                # No validation set
                try:
                    self.loss_tracker.add_epoch(epoch_losses)
                    
                    # Get training results for batch effect analysis
                    train_res = self.evaluate_split(self.train_dataset, self.metadata_all.loc[self.train_dataset.unique_categories], 
                                                 self.train_dataset, self.metadata_all.loc[self.train_dataset.unique_categories])
                    knn_batch_score = train_res.get('knn_batch_score', self.discriminator_toggle_threshold)
                    
                    if not np.isnan(knn_batch_score) and self.discriminator:
                        if knn_batch_score >= self.discriminator_toggle_threshold:
                            self.lambda_ = original_lambda
                            log(f"Batch effect high (knn_batch_score={knn_batch_score:.4f}), enabling adversarial training (lambda={self.lambda_})")
                        else:
                            self.discriminator_toggle_threshold = knn_batch_score
                            self.lambda_ = 0.0
                            log(f"Batch effect low (knn_batch_score={knn_batch_score:.4f}), disabling adversarial training (lambda={self.lambda_})")
                            
                    self.val_metrics_per_epoch.append(np.nan)
                    
                    # Format and display training loss report
                    loss_report = self.format_epoch_loss_report(epoch_losses)
                    log(f"\nEpoch {epoch+1}/{num_epochs_stage2}")
                    for line in loss_report:
                        log(line)
                    log("No validation set, so no evaluation or early stopping here.")
                    log("-" * 30)
                except Exception as e:
                    logger.error(f"Error in Stage 2 training evaluation: {e}")
                    self.val_metrics_per_epoch.append(np.nan)
                    log(f"\nEpoch {epoch+1}/{num_epochs_stage2} - Error in training evaluation, continuing training")
                    log("-" * 30)

        return best_model_state, -best_metric, num_epochs_stage1_trained, num_epochs_stage2_trained

    def load_model(self, checkpoint):
        """
        Loads the model state from the specified path.
        """
        if isinstance(checkpoint, dict):
            cp = checkpoint
        else:
            cp = torch.load(checkpoint, map_location=self.device)
        self.projector.load_state_dict(cp["projector"])
        self.aggregator.load_state_dict(cp["aggregator"])
        if "classifiers" in cp and self.classifiers:
            for clf, state in zip(self.classifiers, cp["classifiers"]):
                clf.load_state_dict(state)
        if "regressors" in cp and self.regressors:
            for reg, state in zip(self.regressors, cp["regressors"]):
                reg.load_state_dict(state)
        if "ordinal_regressors" in cp and self.ordinal_regressors:
            for ord_reg, state in zip(self.ordinal_regressors, cp["ordinal_regressors"]):
                ord_reg.load_state_dict(state)
        if "discriminator" in cp and self.discriminator:
            self.discriminator.load_state_dict(cp["discriminator"])
        if "uncertainty_weighting_loss" in cp:
            self.uncertainty_weighting_loss.load_state_dict(cp["uncertainty_weighting_loss"])
        if "lr_scheduler" in cp:
            self.scheduler.load_state_dict(cp["lr_scheduler"])
        logger.info("Model loaded from checkpoint")
