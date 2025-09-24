# datasets.py

import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.sparse as sp

class SamplesDataset(Dataset):
    def __init__(
        self,
        data,
        unique_categories,
        sample_col,
        classification_cols=None,
        regression_cols=None,
        ordinal_regression_cols=None,
        batch_col=None,
        cell_weights_col=None,
        layer=None
    ):
        """
        Args:
            data: AnnData object or similar with obsm and obs attributes.
            unique_categories: List of unique sample identifiers.
            sample_col: Column in obs indicating sample IDs.
            classification_cols: List of classification task encoded column names.
            regression_cols: List of regression task column names.
            ordinal_regression_cols: List of ordinal regression task encoded column names.
            batch_col: Column in obs indicating batch information.
            layer: The layer in obsm to use for features.
        """
        self.data = data
        self.unique_categories = unique_categories
        self.sample_col = sample_col
        self.layer = layer

        # Initialize labels for classification tasks
        self.classification_cols = classification_cols if classification_cols is not None else []
        self.regression_cols = regression_cols if regression_cols is not None else []
        self.ordinal_regression_cols = ordinal_regression_cols if ordinal_regression_cols is not None else []
        self.batch_col = batch_col
        self.cell_weights_col = cell_weights_col

        self.labels = {col: [np.nan] * len(self.unique_categories) for col in self.classification_cols}
        self.regression_targets = {col: [np.nan] * len(self.unique_categories) for col in self.regression_cols}
        self.ordinal_regression_targets = {col: [np.nan] * len(self.unique_categories) for col in self.ordinal_regression_cols}
        self.batches = [np.nan] * len(self.unique_categories)
        self.sample_cells = []

        if self.cell_weights_col is not None:
            self.cell_weights = []
        else:
            self.cell_weights = None

        for i, sample_id in enumerate(self.unique_categories):
            is_sample_i = self.data.obs[self.sample_col] == sample_id
            sample = self.data.obsm[self.layer][is_sample_i]
            self.sample_cells.append(sample)

            for col in self.classification_cols:
                label_values = self.data.obs[col][is_sample_i].unique()
                self.labels[col][i] = label_values[0] if len(label_values) == 1 else np.nan

            for col in self.regression_cols:
                reg_values = self.data.obs[col][is_sample_i].unique()
                self.regression_targets[col][i] = reg_values[0] if len(reg_values) == 1 else np.nan

            for col in self.ordinal_regression_cols:
                ord_values = self.data.obs[col][is_sample_i].unique()
                self.ordinal_regression_targets[col][i] = ord_values[0] if len(ord_values) == 1 else np.nan

            if self.batch_col is not None:
                batch_values = self.data.obs[self.batch_col][is_sample_i].unique()
                self.batches[i] = batch_values[0] if len(batch_values) == 1 else np.nan

            if self.cell_weights_col is not None:
                cell_weights = self.data.obs[self.cell_weights_col][is_sample_i]
                cell_weights /= cell_weights.sum()  # Normalize to be probability distribution
                self.cell_weights.append(cell_weights)

    def __len__(self):
        return len(self.unique_categories)

    def __getitem__(self, idx):
        sample = self.sample_cells[idx]
        if sp.issparse(sample):
            sample = sample.toarray()
        labels = {col: self.labels[col][idx] for col in self.classification_cols}
        regression = {col: self.regression_targets[col][idx] for col in self.regression_cols}
        ordinal = {col: self.ordinal_regression_targets[col][idx] for col in self.ordinal_regression_cols}
        batch_label = self.batches[idx]
        if self.cell_weights_col is not None:
            cell_weights = self.cell_weights[idx]
        else:
            cell_weights = None
        return sample, labels, batch_label, regression, ordinal, cell_weights

class TransformedPairDataset(Dataset):
    def __init__(self, dataset, subset_size=1000, device='cpu'):
        """
        Args:
            dataset: Instance of SamplesDataset.
            subset_size: Number of cells to sample from each sample or a range of minimum and maximum number of cells.
            device: Device to perform computations on.
        """
        self.dataset = dataset
        self.subset_size = subset_size
        self.device = device

    def __len__(self):
        return len(self.dataset)
    
    @staticmethod
    def _sample_random_cells(sample, subset_size: int, cell_weights, device):
        if sample.shape[0] >= subset_size:
            replace = False
        else:
            replace = True
        
        random_indices = np.random.choice(sample.shape[0], size=subset_size, replace=replace, p=cell_weights)
        return torch.Tensor(sample[random_indices]).to(device)

    @staticmethod
    def _pad_sample(sample, max_size):
        padded_sample = torch.zeros(max_size, sample.shape[1], device=sample.device)
        padded_sample[:sample.shape[0]] = sample
        return padded_sample

    def __getitem__(self, idx):
        sample, labels, batch_label, regression, ordinal, cell_weights = self.dataset[idx]

        is_size_fixed = isinstance(self.subset_size, int)
        if is_size_fixed:
            subset_size_1, subset_size_2 = self.subset_size, self.subset_size
        else:
            subset_size_1 = np.random.randint(self.subset_size[0], self.subset_size[1] + 1)
            subset_size_2 = np.random.randint(self.subset_size[0], self.subset_size[1] + 1)

        random_sample_1 = self._sample_random_cells(sample, subset_size_1, cell_weights, self.device)
        random_sample_2 = self._sample_random_cells(sample, subset_size_2, cell_weights, self.device)
        
        if not is_size_fixed:
            # Pad samples to the maximum size
            random_sample_1 = self._pad_sample(random_sample_1, self.subset_size[1])
            random_sample_2 = self._pad_sample(random_sample_2, self.subset_size[1])

        return random_sample_1, random_sample_2, labels, batch_label, regression, ordinal, self.dataset.unique_categories[idx]
