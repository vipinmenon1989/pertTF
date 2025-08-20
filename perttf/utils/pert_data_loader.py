from typing import List, Tuple, Dict, Union, Optional, Literal
import time
import copy
from sklearn.model_selection import train_test_split, KFold
import torch
import numpy as np
import random
from scipy.sparse import issparse
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from anndata import AnnData
from scipy.sparse import issparse
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)
import scanpy as sc
import scgpt as scg
from scgpt.model import TransformerModel, AdversarialDiscriminator
import wandb
import matplotlib.pyplot as plt

# Assuming these functions are available from your original code
from custom_tokenizer import tokenize_and_pad_batch, random_mask_value
from perttf.model.train_function import train, eval_testdata, evaluate, define_wandb_metrcis


# STEPS TO TRAIN:

# create a PertTFDataManager First and use it generate loaders, validation and data_gen dictionary
# Pass train_loader, valid_loader and data_gen to the wrapper_train either once or as part of kfold loop



class PertTFDataset(Dataset):
    """
    A PyTorch Dataset for AnnData objects that performs next-cell sampling on the fly.
    """
    def __init__(self, adata: AnnData, indices: np.ndarray = None, 
                 cell_type_to_index: dict = None, genotype_to_index: dict = None, expr_layer: str = 'X_binned',
                 ps_columns: list = None, next_cell_pred: str = "identity"):
        """
        Args:
            adata (AnnData): The full AnnData object, may be shared by multiple objects.
            indices (np.ndarray): The indices of the adata object that belong to this dataset (e.g., train or valid).
            config (object): A configuration object with parameters like 'binned_layer_key'.
            cell_type_to_index (dict): Mapping from cell type string to integer index.
            genotype_to_index (dict): Mapping from genotype string to integer index.
            ps_columns (list, optional): List of columns in obs to use for 'ps' scores.
            next_cell_pred (str): The mode for next cell prediction ("identity" or "pert").
        """
        self.adata = adata
        self._check_anndata_content()
        self.indices = indices if indices is not None else np.arange(len(self.adata.obs.index))
        self.expr_layer = expr_layer
        self.next_cell_pred = next_cell_pred
        
        # Mappings
        self.cell_type_to_index = cell_type_to_index if cell_type_to_index is not None else {t: i for i, t in enumerate(self.adata.obs['celltype'].unique())}
        self.genotype_to_index = genotype_to_index if genotype_to_index is not None else {t: i for i, t in enumerate(self.adata.obs['genotype'].unique())}
        self.ps_columns = ps_columns or []

        # For efficient next-cell sampling, pre-compute a dictionary of valid choices
        # IMPORTANT: This dictionary only contains cells from the current data split (train/valid)
        # to prevent data leakage.
        self.next_cell_dict = self._create_next_cell_pool()


    def _check_anndata_content(self):
        assert 'genotype' in self.adata.obs.columns and 'celltype' in self.adata.obs.columns, 'no genotype or celltype column found in anndata'
        if "batch" not in self.adata.obs.columns: 
            batch_ids_0=random.choices( [0,1], k=self.adata.shape[0])
            self.adata.obs["batch"]=batch_ids_0
        if "batch_id" not in self.adata.obs.columns: 
            self.adata.obs["str_batch"] = self.adata.obs["batch"]
            self.adata.obs["str_batch"] = self.adata.obs["str_batch"].astype(str)
            self.adata.obs["batch_id"] = self.adata.obs["str_batch"].astype("category").cat.codes.values
        
    def set_new_indices(self, indices):
        self.indices = indices
        self.next_cell_dict = self._create_next_cell_pool()

    def get_adata_subset(self):
        if self.next_cell_pred != "pert":
            return self.adata[self.indices,].copy()
        else:
            adata_small = self.adata[self.indices,].copy()
            next_cell_id_list = []
            next_pert_list = []
            for i in self.indices:
                current_cell_obs = self.adata.obs.iloc[i]
                next_cell_id, next_pert_label_str = self._sample_next_cell(current_cell_obs)
                next_cell_id_list.append(next_cell_id)
                next_pert_list.append(next_pert_label_str)
            adata_small.obs['genotype_next'] = next_pert_list
            adata_small.obs['next_cell_id'] = next_cell_id_list

        return adata_small

    def __len__(self):
        return len(self.indices)

    def _create_next_cell_pool(self):
        """Pre-computes a dictionary for fast sampling of the next cell."""
        if self.next_cell_pred != "pert":
            return None
            
        next_cell_dict = {}
        # Use only the subset of adata relevant to this dataset split
        obs_subset = self.adata.obs.iloc[self.indices]
        
        for cell_type in obs_subset['celltype'].unique():
            next_cell_dict[cell_type] = {}
            for genotype in obs_subset['genotype'].unique():
                # Find cells matching the criteria within the current split
                mask = (obs_subset['celltype'] == cell_type) & (obs_subset['genotype'] == genotype)
                included_cells_indices = obs_subset[mask].index.tolist()
                if included_cells_indices:
                    next_cell_dict[cell_type][genotype] = included_cells_indices
        return next_cell_dict

    def _sample_next_cell(self, current_cell_obs):
        """Samples a 'next cell' for a given current cell."""
        current_cell_id = current_cell_obs.name
        current_cell_type = current_cell_obs['celltype']
        current_genotype = current_cell_obs['genotype']

        if self.next_cell_pred == "identity":
            return current_cell_id, current_genotype

        # Logic for perturbation prediction
        valid_genotypes = self.next_cell_dict.get(current_cell_type, {})
        if not valid_genotypes:
             return current_cell_id, current_genotype # Fallback

        if current_genotype == 'WT':
            # Randomly select a different genotype
            next_pert_value = random.choice(list(valid_genotypes.keys()))
        else:
            # For non-WT, the "next" state is the same perturbation
            next_pert_value = current_genotype

        if next_pert_value == current_genotype:
            return current_cell_id, next_pert_value
        else:
            # Sample a random cell with the target cell type and genotype
            possible_next_cells = valid_genotypes.get(next_pert_value, [current_cell_id])
            next_cell_id = random.choice(possible_next_cells)
            return next_cell_id, next_pert_value
    
    def __getitem__(self, idx: int):
        """
        Retrieves one sample from the dataset. This is where on-the-fly processing happens.
        """

        # 1. Get the index for the current cell
        current_cell_global_idx = self.indices[idx]

        current_cell_obs = self.adata.obs.iloc[current_cell_global_idx]
        
        # 2. Get expression data for the current cell
        binned_layer_key = self.expr_layer

        current_expr = self.adata.layers[binned_layer_key][current_cell_global_idx]
        if issparse(current_expr):
            current_expr = current_expr.toarray().flatten()

        # 3. Sample the next cell and its metadata
        next_cell_id, next_pert_label_str = self._sample_next_cell(current_cell_obs)
        next_cell_global_idx = self.adata.obs.index.get_loc(next_cell_id)
        
        # 4. Get expression data for the next cell
        next_expr = self.adata.layers[binned_layer_key][next_cell_global_idx]
        if issparse(next_expr):
            next_expr = next_expr.toarray().flatten()

        # 5. Get labels and PS scores
        cell_label = self.cell_type_to_index[current_cell_obs['celltype']]
        pert_label = self.genotype_to_index[current_cell_obs['genotype']]
        batch_label = current_cell_obs['batch_id']

        # Next cell labels are the same for cell type, but perturbation can change
        cell_label_next = cell_label
        pert_label_next = self.genotype_to_index[next_pert_label_str]
        
        ps_scores = current_cell_obs[self.ps_columns].values.astype(np.float32) if self.ps_columns else np.array([0.0], dtype=np.float32)
        ps_scores_next = self.adata.obs.loc[next_cell_id, self.ps_columns].values.astype(np.float32) if self.ps_columns else np.array([0.0], dtype=np.float32)

        return {
            "expr": current_expr,
            "expr_next": next_expr,
            "celltype_labels": cell_label,
            "perturbation_labels": pert_label,
            "batch_labels": batch_label,
            "celltype_labels_next": cell_label_next,
            "perturbation_labels_next": pert_label_next,
            "ps": ps_scores,
            "ps_next": ps_scores_next,
        }

class CustomCollator:
    """
    A collate function for the DataLoader that tokenizes, pads, and masks batches on the fly.
    """
    def __init__(self, config: object, vocab: object, gene_ids: np.ndarray):
        self.config = config
        self.vocab = vocab
        self.gene_ids = gene_ids

    def __call__(self, batch: list) -> dict:
        """
        Processes a list of samples from the Dataset into a single batch tensor.
        """
        # 1. Separate the components of the batch
        expr_list = [item['expr'] for item in batch]
        expr_next_list = [item['expr_next'] for item in batch]
        
        # 2. Tokenize and pad the expression data for the current batch
        tokenized_train, gene_idx_list = tokenize_and_pad_batch(
            np.array(expr_list), self.gene_ids, max_len=self.config.max_seq_len,
            vocab=self.vocab, pad_token=self.config.pad_token, pad_value=self.config.pad_value,
            append_cls=True, include_zero_gene=True
        )
        tokenized_train_next, _ = tokenize_and_pad_batch(
            np.array(expr_next_list), self.gene_ids, max_len=self.config.max_seq_len,
            vocab=self.vocab, pad_token=self.config.pad_token, pad_value=self.config.pad_value,
            append_cls=True, include_zero_gene=True, sample_indices=gene_idx_list
        )
        
        # 3. Apply random masking for this batch
        masked_values = random_mask_value(
            tokenized_train["values"], mask_ratio=self.config.mask_ratio,
            mask_value=self.config.mask_value, pad_value=self.config.pad_value,
        )

        # 4. Collate all other labels into tensors
        collated_batch = {
            "gene_ids": tokenized_train["genes"],
            "values": masked_values,
            "target_values": tokenized_train["values"],
            "target_values_next": tokenized_train_next["values"],
        }
        
        # Stack scalar or vector labels from each item in the batch
        for key in batch[0].keys():
            if key not in ["expr", "expr_next"]:
                values = [item[key] for item in batch]
                tensor = torch.from_numpy(np.array(values))
                # Ensure labels are long type and scores are float
                collated_batch[key] = tensor.long() if 'label' in key else tensor.float()

        return collated_batch
    
    

class PertTFDataManager:
    """
    Manages data loading, preprocessing, and splitting for an AnnData object.
    This class encapsulates all data-related setup, including vocab, mappings,
    and provides methods to get data loaders for training and cross-validation.
    """
    def __init__(self, adata: AnnData, config: object, ps_columns: list = None):
        self.adata = adata
        self.config = config
        self.ps_columns = ps_columns

        # --- Perform one-time data setup ---
        print("Initializing AnnDataManager: Creating vocab and mappings...")
        #if "batch_id" not in self.adata.obs.columns:
         #   self.adata.obs["str_batch"] = "batch_0"
          #  self.adata.obs["batch_id"] = self.adata.obs["str_batch"].astype("category").cat.codes
        
        # Create and store mappings and vocab as instance attributes
        self.cell_type_to_index = {t: i for i, t in enumerate(self.adata.obs['celltype'].unique())}
        self.genotype_to_index = {t: i for i, t in enumerate(self.adata.obs['genotype'].unique())}
        self.num_cell_types = len(self.cell_type_to_index)
        self.num_genotypes = len(self.genotype_to_index)
        #self.num_batch_types = len(self.adata.obs["batch_id"].unique())
        if "batch" not in self.adata.obs.columns: 
            batch_ids_0=random.choices( [0,1], k=self.adata.shape[0])
            self.adata.obs["batch"]=batch_ids_0
        if "batch_id" not in self.adata.obs.columns: 
            self.adata.obs["str_batch"] = self.adata.obs["batch"]
            self.adata.obs["str_batch"] = self.adata.obs["str_batch"].astype(str)
            self.adata.obs["batch_id"] = self.adata.obs["str_batch"].astype("category").cat.codes.values
            
        self.num_batch_types = len(self.adata.obs["batch_id"].unique())
        self.genes = self.adata.var.index.tolist()
        self.vocab = Vocab(VocabPybind(self.genes + self.config.special_tokens, None))
        self.vocab.set_default_index(self.vocab["<pad>"])
        self.gene_ids = np.array(self.vocab(self.genes), dtype=int)
        
        # The collator can be created once and reused
        self.collator = CustomCollator(self.config, self.vocab, self.gene_ids)
        print("Initialization complete.")

    def get_adata_input_dict(self):
        data_gen = { 
            'genes': self.genes,
            'gene_ids': self.gene_ids,
            'vocab': self.vocab,
            'num_batch_types': self.num_batch_types, # need to change this
            'num_cell_types': self.num_cell_types,
            'num_genotypes': self.num_genotypes,
            'cell_type_to_index': self.cell_type_to_index,
            'genotype_to_index': self.genotype_to_index
        }
        
        return data_gen

    def _create_dataset_from_indices(self, train_indices, valid_indices):
        """A helper function to create datasets and dataloaders from index arrays."""
        train_dataset = PertTFDataset(
            self.adata, indices=train_indices, cell_type_to_index=self.cell_type_to_index, genotype_to_index=self.genotype_to_index,
            ps_columns=self.ps_columns, next_cell_pred=self.config.next_cell_pred_type
        )
        valid_dataset = PertTFDataset(
            self.adata, indices=valid_indices, cell_type_to_index=self.cell_type_to_index, genotype_to_index=self.genotype_to_index,
            ps_columns=self.ps_columns, next_cell_pred=self.config.next_cell_pred_type
        )
        return train_dataset, valid_dataset

    def _create_loaders_from_dataset(self, train_dataset, valid_dataset):
        """A helper function to create datasets and dataloaders from index arrays."""     
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True,
            num_workers=4, collate_fn=self.collator, pin_memory=True
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=self.config.batch_size, shuffle=False,
            num_workers=4, collate_fn=self.collator, pin_memory=True
        )
        return train_loader, valid_loader

    def get_dataloaders(self, test_size: float = 0.1):
        """Provides a single, standard train/validation split."""
        print(f"Creating a single train/validation split (test_size={test_size})...")
        indices = np.arange(self.adata.n_obs)
        train_indices, valid_indices = train_test_split(indices, test_size=test_size, shuffle=True)
        train_data, valid_data = self._create_dataset_from_indices(train_indices, valid_indices)
        valid_adata = valid_data.get_adata_subset()
        return self._create_loaders_from_dataset(train_data, valid_data), valid_adata, self.get_adata_input_dict()

    def k_fold_split(self, n_splits: int = 5):
        """
        An iterator that yields train and validation dataloaders for each fold
        in a k-fold cross-validation setup.
        """
        print(f"Setting up K-Fold cross-validation with {n_splits} splits...")
        kf = KFold(n_splits=n_splits, shuffle=True)
        indices = np.arange(self.adata.n_obs)
        
        for fold, (train_indices, valid_indices) in enumerate(kf.split(indices)):
            print(f"--- Yielding data loaders for Fold {fold+1}/{n_splits} ---")
            train_data, valid_data = self._create_dataset_from_indices(train_indices, valid_indices)
            valid_adata = valid_data.get_adata_subset()
            yield self._create_loaders_from_dataset(train_data, valid_data), valid_adata, self.get_adata_input_dict()






