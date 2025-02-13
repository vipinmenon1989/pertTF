from typing import List, Tuple, Dict, Union, Optional

import torch
from torch import nn, Tensor

from torch.utils.data import Dataset, DataLoader




def produce_training_datasets(adata, config,
                              input_layer_key = "X_binned",
                              next_layer_key = "X_binned_next",
                              cell_type_to_index = None,
                              genotype_to_index = None,
                              logger = None):
    # add necessary columns to adata
    adata.var["gene_name"] = adata.var.index.tolist()

    # set up these random values such that they don't get errors later on during training
    adata.obs["str_batch"] = adata.obs["batch"]
    adata.obs["str_batch"] = adata.obs["str_batch"].astype(str)
    adata.obs["batch_id"] = adata.obs["str_batch"].astype("category").cat.codes.values
    #adata.obs["celltype"] = random.choices( [0,1], k=adata.shape[0])
    #adata.obs["celltype"] = adata.obs["celltype"].astype("category")

    # add next layers
    all_counts = (adata.layers[input_layer_key].A if issparse(adata.layers[input_layer_key]) else adata.layers[input_layer_key])

    # predict the next state of a cell
    next_counts = add_pred_layer(adata)
    # or just duplicate the data
    #next_counts = all_counts.copy()

    adata.layers[next_layer_key]=next_counts
    #next_counts = (adata.layers[next_layer_key].A if issparse(adata.layers[next_layer_key]) else adata.layers[next_layer_key])

    genes = adata.var["gene_name"].tolist()

    if "celltype" in adata.obs.columns and config.cell_type_classifier:
        celltypes_labels = adata.obs["celltype"].tolist()  # make sure count from 0
    else:
        celltypes_labels = random.choices( [0,1], k=adata.shape[0])
    #num_types = len(set(celltypes_labels))
    celltypes_labels = np.array(celltypes_labels)

    # prompt: number of unique values in celltypes_labels



    if "genotype" in adata.obs.columns and config.perturbation_classifier_weight > 0:
        perturbation_labels = adata.obs["genotype"].tolist()  # make sure count from 0
    else:
        perturbation_labels = random.choices( [0,1], k=adata.shape[0])
    perturbation_labels = np.array(perturbation_labels)


    if "batch_id" in adata.obs.columns: # and config.DSBN:
        batch_ids = adata.obs["batch_id"].tolist()
    else:
        batch_ids=random.choices( [0,1], k=adata.shape[0])
    num_batch_types = len(set(batch_ids))
    batch_ids = np.array(batch_ids)

    # generate cell type indexs and perturbation indexes

    if cell_type_to_index is None:
        cell_type_to_index = {cell_type: index for index, cell_type in enumerate(set(celltypes_labels))}

    celltypes_indexes = [cell_type_to_index[cell_type] for cell_type in celltypes_labels]
    celltypes_indexes = np.array(celltypes_indexes)
    # Now celltypes_indexes contains the numerical representation of cell types
    #print(celltypes_indexes[:10]) # Example: print first 10 indexes

    if genotype_to_index is None:
        genotype_to_index = {genotype: index for index, genotype in enumerate(set(perturbation_labels))}
    perturbation_indexes = [genotype_to_index[genotype] for genotype in perturbation_labels]
    perturbation_indexes = np.array(perturbation_indexes)
    #print(perturbation_indexes[:10])

    #n_cls = len(set(celltypes_labels)) if config.cell_type_classifier else 1
    #n_cls
    #n_perturb = len(set(perturbation_labels)) if config.perturbation_classifier_weight > 0 else 1

    n_cls = len(cell_type_to_index)
    n_perturb = len(genotype_to_index)
    # now, split train and test data
    (
        train_data,
        valid_data,
        train_celltype_labels,
        valid_celltype_labels,
        train_perturbation_labels,
        valid_perturbation_labels,
        train_batch_labels,
        valid_batch_labels,
        train_data_next,
        valid_data_next,
        cell_ids_train,
        cell_ids_valid
    ) = train_test_split(
        all_counts, celltypes_indexes, perturbation_indexes, batch_ids, next_counts, adata.obs.index.values, test_size=0.1, shuffle=True
    )

    adata.obs['is_in_training']=adata.obs.index.isin(cell_ids_train)

    # further expand the next layer prediction
    n_rounds = 0

    adata_small=adata[adata.obs['is_in_training']==True] # only consider training data
    for ni in range(n_rounds):
        print(f"round {ni}")

        round_all_c = adata_small.layers[input_layer_key].copy()
        round_next_c = add_pred_layer(adata_small)


        train_data = np.concatenate([train_data, round_all_c], axis=0)
        train_data_next = np.concatenate([train_data_next, round_next_c], axis=0)

    train_celltype_labels = np.concatenate([train_celltype_labels] * (n_rounds+1))
    train_perturbation_labels = np.concatenate([train_perturbation_labels] * (n_rounds+1))
    train_batch_labels = np.concatenate([train_batch_labels] * (n_rounds+1))
    cell_ids_train = np.concatenate([cell_ids_train] * (n_rounds + 1))


    # prompt: shuffle the rows of all_counts and next_counts with the same order

    # Generate a random permutation of indices
    indices = np.arange(train_data.shape[0])
    np.random.shuffle(indices)

    # Shuffle both arrays using the same indices
    train_data = train_data[indices]
    train_data_next = train_data_next[indices]
    train_celltype_labels = train_celltype_labels[indices]
    train_perturbation_labels = train_perturbation_labels[indices]
    train_batch_labels = train_batch_labels[indices]
    cell_ids_train = cell_ids_train[indices]

    if config.per_seq_batch_sample  and "batch_id" in adata.obs.columns: # and config.DSBN
        # sort the adata by batch_id in advance
        #adata_sorted = adata[adata.obs["batch_id"].argsort()].copy()
        #adata_sorted.obs
        adata_sorted = adata_small[adata_small.obs["batch_id"].argsort()].copy()
    else:
        #adata_sorted = adata.copy()
        adata_sorted = adata_small.copy()

    # construct vocab
    vocab = Vocab(
            VocabPybind(genes + config.special_tokens, None)
    )  # bidirectional lookup [gene <-> int]
    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(vocab(genes), dtype=int)

    # construct tokenized data
    tokenized_train = tokenize_and_pad_batch(
        train_data,
        gene_ids,
        max_len=config.max_seq_len,
        vocab=vocab,
        pad_token=config.pad_token,
        pad_value=config.pad_value,
        append_cls=True,  # append <cls> token at the beginning
        include_zero_gene=True,
    )
    tokenized_valid = tokenize_and_pad_batch(
        valid_data,
        gene_ids,
        max_len=config.max_seq_len,
        vocab=vocab,
        pad_token=config.pad_token,
        pad_value=config.pad_value,
        append_cls=True,
        include_zero_gene=True,
    )

    tokenized_train_next = tokenize_and_pad_batch(
        train_data_next,
        gene_ids,
        max_len=config.max_seq_len,
        vocab=vocab,
        pad_token=config.pad_token,
        pad_value=config.pad_value,
        append_cls=True,  # append <cls> token at the beginning
        include_zero_gene=True,
    )
    tokenized_valid_next = tokenize_and_pad_batch(
        valid_data_next,
        gene_ids,
        max_len=config.max_seq_len,
        vocab=vocab,
        pad_token=config.pad_token,
        pad_value=config.pad_value,
        append_cls=True,
        include_zero_gene=True,
    )
    if logger is not None:
        logger.info(
            f"train set number of samples: {tokenized_train['genes'].shape[0]}, "
            f"\n\t feature length: {tokenized_train['genes'].shape[1]}"
        )
        logger.info(
            f"valid set number of samples: {tokenized_valid['genes'].shape[0]}, "
            f"\n\t feature length: {tokenized_valid['genes'].shape[1]}"
        )


    # construct return value
    ret_dict={
        'adata': adata, # original adata
        'adata_sorted': adata_sorted, # training adata
        'vocab': vocab,
        'n_perturb': n_perturb,
        'n_cls': n_cls,
        'genes': genes,
        'gene_ids': gene_ids,
        'num_batch_types': num_batch_types,
        'cell_type_to_index': cell_type_to_index,
        'genotype_to_index': genotype_to_index,
        'train_data': train_data,
        'valid_data': valid_data,
        'train_data_next': train_data_next,
        'valid_data_next': valid_data_next,
        'train_celltype_labels': train_celltype_labels,
        'valid_celltype_labels': valid_celltype_labels,
        'train_perturbation_labels': train_perturbation_labels,
        'valid_perturbation_labels': valid_perturbation_labels,
        'train_batch_labels': train_batch_labels,
        'valid_batch_labels': valid_batch_labels,
        'cell_ids_train': cell_ids_train,
        'cell_ids_valid': cell_ids_valid,
        'tokenized_train': tokenized_train,
        'tokenized_valid': tokenized_valid,
        'tokenized_train_next': tokenized_train_next,
        'tokenized_valid_next': tokenized_valid_next,
    }
    return ret_dict

def prepare_data(
    data_dict,
    config,
    sort_seq_batch=False,
    epoch = 0) -> Tuple[Dict[str, torch.Tensor]]:

    # extract data
    tokenized_train=data_dict['tokenized_train']
    tokenized_valid=data_dict['tokenized_valid']
    tokenized_train_next=data_dict['tokenized_train_next']
    tokenized_valid_next=data_dict['tokenized_valid_next']
    train_batch_labels=data_dict['train_batch_labels']
    valid_batch_labels=data_dict['valid_batch_labels']
    train_celltype_labels=data_dict['train_celltype_labels']
    valid_celltype_labels=data_dict['valid_celltype_labels']
    train_perturbation_labels=data_dict['train_perturbation_labels']
    valid_perturbation_labels=data_dict['valid_perturbation_labels']

    masked_values_train = random_mask_value(
        tokenized_train["values"],
        mask_ratio=config.mask_ratio,
        mask_value=config.mask_value,
        pad_value=config.pad_value,
    )
    masked_values_valid = random_mask_value(
        tokenized_valid["values"],
        mask_ratio=config.mask_ratio,
        mask_value=config.mask_value,
        pad_value=config.pad_value,
    )
    print(
        f"random masking at epoch {epoch:3d}, ratio of masked values in train: ",
        f"{(masked_values_train == config.mask_value).sum() / (masked_values_train - config.pad_value).count_nonzero():.4f}",
    )

    input_gene_ids_train, input_gene_ids_valid = (
        tokenized_train["genes"],
        tokenized_valid["genes"],
    )
    input_values_train, input_values_valid = masked_values_train, masked_values_valid
    target_values_train, target_values_valid = (
        tokenized_train["values"],
        tokenized_valid["values"],
    )
    target_values_train_next, target_values_valid_next = ( # added
        tokenized_train_next["values"],
        tokenized_valid_next["values"],
    )

    tensor_batch_labels_train = torch.from_numpy(train_batch_labels).long()
    tensor_batch_labels_valid = torch.from_numpy(valid_batch_labels).long()


    tensor_celltype_labels_train = torch.from_numpy(train_celltype_labels).long()#added
    tensor_celltype_labels_valid = torch.from_numpy(valid_celltype_labels).long()#added

    tensor_perturbation_labels_train = torch.from_numpy(train_perturbation_labels).long()#added
    tensor_perturbation_labels_valid = torch.from_numpy(valid_perturbation_labels).long()#added

    if sort_seq_batch:
        train_sort_ids = np.argsort(train_batch_labels)
        input_gene_ids_train = input_gene_ids_train[train_sort_ids]
        input_values_train = input_values_train[train_sort_ids]
        target_values_train = target_values_train[train_sort_ids]
        target_values_train_next = target_values_train_next[train_sort_ids] # added
        tensor_batch_labels_train = tensor_batch_labels_train[train_sort_ids]
        tensor_celltype_labels_train = tensor_celltype_labels_train[train_sort_ids]#added
        tensor_perturbation_labels_train = tensor_perturbation_labels_train[train_sort_ids]#added

        valid_sort_ids = np.argsort(valid_batch_labels)
        input_gene_ids_valid = input_gene_ids_valid[valid_sort_ids]
        input_values_valid = input_values_valid[valid_sort_ids]
        target_values_valid = target_values_valid[valid_sort_ids]
        target_values_valid_next = target_values_valid_next[valid_sort_ids] # added
        tensor_batch_labels_valid = tensor_batch_labels_valid[valid_sort_ids]
        tensor_celltype_labels_valid = tensor_celltype_labels_valid[valid_sort_ids] #added
        tensor_perturbation_labels_valid = tensor_perturbation_labels_valid[valid_sort_ids] #added

    train_data_pt = {
        "gene_ids": input_gene_ids_train,
        "values": input_values_train,
        "target_values": target_values_train,
        "target_values_next": target_values_train_next, # added
        "batch_labels": tensor_batch_labels_train,
        "celltype_labels": tensor_celltype_labels_train, #added
        "perturbation_labels": tensor_perturbation_labels_train, #added
    }
    valid_data_pt = {
        "gene_ids": input_gene_ids_valid,
        "values": input_values_valid,
        "target_values": target_values_valid,
        "target_values_next": target_values_valid_next, # added
        "batch_labels": tensor_batch_labels_valid,
        "celltype_labels": tensor_celltype_labels_valid, #added
        "perturbation_labels": tensor_perturbation_labels_valid, #added
    }

    return train_data_pt, valid_data_pt


# dataset
class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


# data_loader
def prepare_dataloader(
    data_pt: Dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool = False,
    intra_domain_shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    dataset = SeqDataset(data_pt)

    if config.per_seq_batch_sample:
        # find the indices of samples in each seq batch
        subsets = []
        batch_labels_array = data_pt["batch_labels"].numpy()
        for batch_label in np.unique(batch_labels_array):
            batch_indices = np.where(batch_labels_array == batch_label)[0].tolist()
            subsets.append(batch_indices)
        data_loader = DataLoader(
            dataset=dataset,
            batch_sampler=SubsetsBatchSampler(
                subsets,
                batch_size,
                intra_subset_shuffle=intra_domain_shuffle,
                inter_subset_shuffle=shuffle,
                drop_last=drop_last,
            ),
            num_workers=num_workers,
            pin_memory=True,
        )
        return data_loader

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    )
    return data_loader


