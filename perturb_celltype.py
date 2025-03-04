import os
import argparse

import gc
import random
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import torch

# import scvi
import wandb

sys.path.insert(0, "../")
import scgpt as scg
from scgpt.preprocess import Preprocessor
from scgpt.utils import load_pretrained

sc.set_figure_params(figsize=(4, 4))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings("ignore")

# sys.path.insert(0, "/content/pertTF/")

from perttf.model.config_gen import generate_config
from perttf.model.pertTF import PerturbationTFModel
from perttf.model.train_data_gen import produce_training_datasets
from perttf.model.train_function import wrapper_train

"""# Step 3: hyperparameter set up"""

hyperparameter_defaults = dict(
    seed=42,
    # dataset_name="PBMC_10K", # Dataset name
    dataset_name="pancreatic",
    do_train=True,  # Flag to indicate whether to do update model parameters during training
    # load_model="/content/drive/MyDrive/Colab Notebooks/scGPT/pretrain_blood", # Path to pre-trained model
    load_model=None,
    GEPC=True,  # Gene expression modelling for cell objective
    ecs_thres=0.7,  # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
    dab_weight=0.0,  # 2000.0, # DAR objective weight for batch correction; if has batch, set to 1.0
    this_weight=1.0,  # weight for predicting the expression of current cell
    next_weight=00.0,  # weight for predicting the next cell
    n_rounds=1,  # number of rounds for generating the next cell
    next_cell_pred_type="identity",  # the method to predict the next cell, either "pert" (random next cell within the same cell type) or "identity" (the same cell). If "identity", set next_weight=0
    #
    ecs_weight=1.0,  # weight for predicting the similarity of cells
    cell_type_classifier=True,  #  do we need the trasnformer to separate cell types?
    cell_type_classifier_weight=1.0,
    perturbation_classifier_weight=10.0,
    perturbation_input=False,  # use perturbation as input?
    CCE=False,  # Contrastive cell embedding objective
    mask_ratio=0.15,  # Default mask ratio
    epochs=100,  # Default number of epochs for fine-tuning
    n_bins=51,  # Default number of bins for value binning in data pre-processing
    # lr=1e-4, # Default learning rate for fine-tuning
    lr=1e-3,  # learning rate for learning de novo
    batch_size=32,  # Default batch size for fine-tuning
    layer_size=32,  # defalut 32
    nlayers=2,
    nhead=4,  # if load model, batch_size, layer_size, nlayers, nhead will be ignored
    dropout=0.4,  # Default dropout rate during model fine-tuning
    schedule_ratio=0.99,  # Default rate for learning rate decay
    save_eval_interval=5,  # Default model evaluation interval
    log_interval=100,  # Default log interval
    fast_transformer=True,  # Default setting
    pre_norm=False,  # Default setting
    amp=True,  # # Default setting: Automatic Mixed Precision
    do_sample_in_train=False,  # sample the bernoulli in training
    ADV=False,  # Adversarial training for batch correction
    adv_weight=10000,
    adv_E_delay_epochs=2,  # delay adversarial training on encoder for a few epochs
    adv_D_delay_epochs=2,
    lr_ADV=1e-3,  # learning rate for discriminator, used when ADV is True
    DSBN=False,  # True if (config.dab_weight >0 or config.ADV ) else False  # Domain-spec batchnorm; default is True
    per_seq_batch_sample=False,  # DSBN # default True
    use_batch_label=False,  # default: equal to DSBN
    schedule_interval=1,
    explicit_zero_prob=True,  # whether explicit bernoulli for zeros
    n_hvg=3000,  # number of highly variable genes
    mask_value=-1,
    pad_value=-2,
    pad_token="<pad>",
)

"""wandb configuration"""

config, run_session = generate_config(hyperparameter_defaults, wandb_mode="online")


# the following parameters have been moved to config
# DSBN =False # True if (config.dab_weight >0 or config.ADV ) else False  # Domain-spec batchnorm; default is True
# per_seq_batch_sample =  False # DSBN # default True


dataset_name = config.dataset_name
save_dir = Path(f"./save/dev_{dataset_name}-{time.strftime('%b%d-%H-%M')}/")
save_dir.mkdir(parents=True, exist_ok=True)
print(f"save to {save_dir}")
logger = scg.logger
scg.utils.add_file_handler(logger, save_dir / "run.log")

# embsize = config.layer_size
# nhead = config.nhead
# nlayers = config.nlayers
# d_hid = config.layer_size

"""# step 4: load scrna-seq dataset"""

parser = argparse.ArgumentParser(description="Process files")
parser.add_argument('-d', '--data_path', type=str, help='Path to the input file')
parser.add_argument('-o', '--output_dir', type=str, help='Path to the output directory')

args = parser.parse_args()

# adata0 = sc.read_h5ad("D18_diabetes_merged_reduced.h5ad")
print("Data path: ", args.data_path)
print("output directory: ", args.output_dir)

data_path = args.data_path
output_dir = args.output_dir

adata0 = sc.read_h5ad(data_path)
adata0.layers["GPTin"] = adata0.X.copy()

# set up the preprocessor, use the args to config the workflow

preprocessor = Preprocessor(
    use_key="GPTin",  # the key in adata.layers to use as raw data
    # filter_gene_by_counts=3,  # step 1
    # filter_cell_by_counts=False,  # step 2
    normalize_total=None,  # 3. whether to normalize the raw data and to what sum
    # result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
    log1p=False,  # 4. whether to log1p the normalized data
    # result_log1p_key="X_log1p",
    subset_hvg=False,  # 5. whether to subset the raw data to highly variable genes; use n_hvg default
    hvg_flavor="seurat_v3",  # if data_is_raw else "cell_ranger",
    binning=config.n_bins,  # 6. whether to bin the raw data and to what number of bins
    # binning=0,  # 6. whether to bin the raw data and to what number of bins
    result_binned_key="X_binned",  # the key in adata.layers to store the binned data
)
preprocessor(adata0, batch_key=None)

# this is for integration training
if False:
    adata = adata0[
        (adata0.obs["batch"] == 0) | (adata0.obs["Diabetes_status"] == "Non-diabetic"),
        :,
    ]

    # remove unknown cell type
    adata = adata[adata.obs["celltype"] != "unknown", :]
    # only test cells where two batches share the same cell type

    adata = adata[adata.obs["celltype"].isin(["SC-beta", "SC-alpha", "SC-delta"]), :]


if True:
    adata0.obs.loc[adata0.obs["genotype"].isna(), "genotype"] = "WT"
    adata = adata0[adata0.obs["batch"] == 0, :]  # only use cell line data for training
    adata_prim = adata0[
        adata0.obs["batch"] == 1, :
    ]  # a separate primary data as a validation data
    # prompt: randomly select 1000 n_obs from adata_prim

    import random

    random_indices = random.sample(range(adata_prim.n_obs), min(3000, adata_prim.n_obs))
    adata_prim = adata_prim[random_indices]
    adata_prim
adata

adata_prim

adata.obs.Diabetes_status.value_counts(dropna=False)

adata.obs["genotype"].value_counts(dropna=False)


adata.obs["genotype"].value_counts(dropna=False)

adata.obs

# for debugging only; use top 2k genes only
if False:
    value_x = raw_data_f[3, :].toarray()
    value_y = adata.X[3, :]
    # prompt: calculate and plot the correlation between value_x and value_y

    import matplotlib.pyplot as plt
    import numpy as np

    # Assuming value_x and value_y are defined as in the provided code
    # Replace with your actual data if needed

    correlation = np.corrcoef(value_x, value_y)[0, 1]
    print(f"Correlation between value_x and value_y: {correlation}")

    plt.scatter(value_x, value_y)
    plt.xlabel("value_x")
    plt.ylabel("value_y")
    plt.title(f"Correlation: {correlation:.2f}")
    plt.show()

set(adata.obs["genotype"])

adata.obs["celltype"].value_counts(dropna=False)

adata.obs["genotype"].value_counts(dropna=False)

adata.obs.loc[adata.obs["batch"] == 0, "celltype"].value_counts(dropna=False)

adata.obs.loc[adata.obs["batch"] == 1, "celltype"].value_counts(dropna=False)

adata.obs.loc[adata.obs["batch"] == 0, "genotype"].value_counts(dropna=False)

adata.obs.loc[adata.obs["batch"] == 1, "genotype"].value_counts(dropna=False)

"""# step 5: preprocess dataset"""

# process the batch separately
# not needed, because the normalization is for each cell
# adata_batch0.layers['X_binned'][3,:].max()#
# adata=adata_batch0.concatenate(adata_batch1)

# adata.layers['GPTin'].toarray()

# adata.obs['batch_id'].value_counts(dropna=False)

"""Tokenize the input data for model fine-tuning"""

data_produced = produce_training_datasets(adata, config)

# cell_type_to_index=data_produced['cell_type_to_index']
# genotype_to_index=data_produced['genotype_to_index']


# num_batch_types=data_produced['num_batch_types']
# n_perturb=data_produced['n_perturb']
# n_cls=data_produced['n_cls']

"""save data"""

# adata.obs['is_in_training']=adata.obs.index.isin(data_produced['cell_ids_train'])
adata0.obs["is_in_training"] = adata0.obs.index.isin(data_produced["cell_ids_train"])
# sum(adata.obs['is_in_training'])
adata0.write_h5ad(save_dir / "adata_original_full.h5ad")

# train_data.shape

""" # Step 6: Build or Load the pre-trained scGPT model"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab = data_produced["vocab"]

ntokens = len(vocab)  # size of vocabulary
model = PerturbationTFModel(
    data_produced["n_perturb"],  # n_perturb, # number of perturbation labels
    3,  # layers
    ntokens,
    config.layer_size,  # embsize,
    config.nhead,  # nhead,
    config.layer_size,  # d_hid,
    config.nlayers,  # nlayers,
    vocab=vocab,
    dropout=config.dropout,
    pad_token=config.pad_token,
    pad_value=config.pad_value,
    do_mvc=config.GEPC,
    do_dab=True if config.dab_weight > 0 else False,
    use_batch_labels=config.use_batch_label,
    num_batch_labels=data_produced["num_batch_types"],  # num_batch_types,
    domain_spec_batchnorm=config.DSBN,
    n_input_bins=config.n_bins,
    ecs_threshold=config.ecs_thres,
    explicit_zero_prob=config.explicit_zero_prob,
    use_fast_transformer=config.fast_transformer,
    pre_norm=config.pre_norm,
    n_cls=data_produced["n_cls"],  # n_cls, # number of cell type labels
    nlayers_cls=3,
)

if config.load_model is not None:
    load_pretrained(model, torch.load(model_file), verbose=False)

model.to(device)
wandb.watch(model)


print(model)


"""# Step 7: train"""

if config.per_seq_batch_sample:
    adata_prim_v = adata_prim[adata_prim.obs["batch_id"].argsort()].copy()
else:
    adata_prim_v = adata_prim.copy()
eval_adata_dict = {
    "validation": data_produced["adata_sorted"],
    "primary": adata_prim_v,
}

best_model = wrapper_train(
    model, config, data_produced, eval_adata_dict=eval_adata_dict
)

if False:
    ret_adata = results["adata"]
    ret_adata.obsm["X_pert_pred"].shape
    genotype_to_index

    import numpy as np

    # Assuming ret_adata.obsm['X_pert_pred'] is a numpy array or can be converted to one
    logits = ret_adata.obsm["X_pert_pred"]

    # Convert logits to probabilities using softmax
    X_pert_pred_probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

    # Assign the probabilities back to the AnnData object
    ret_adata.obsm["X_pert_pred_probs"] = X_pert_pred_probs

    # prompt: convert X_pert_pred_probs, which is the probabilities of each label, into label predictions, whose order is defined in genotype_to_index

    # Convert probabilities to predicted labels
    label_predictions = np.argmax(X_pert_pred_probs, axis=1)

    # Map predicted indices back to genotypes using genotype_to_index
    # Assuming genotype_to_index is a dictionary where keys are indices and values are genotypes
    index_to_genotype = {v: k for k, v in genotype_to_index.items()}
    predicted_genotypes = [index_to_genotype[i] for i in label_predictions]

    # Add the predicted genotypes to the AnnData object
    ret_adata.obs["predicted_genotype"] = predicted_genotypes

# ret_adata=results_p['adata']
# ret_adata.obs[['Diabetes_status','predicted_genotype']].value_counts(dropna=False)

# ret_adata.obs


# sc.pl.umap( ret_adata,color=["celltype"])

# sc.pl.umap( ret_adata,color=["predicted_genotype"])


# adata_prim.obs['celltype'].isin(cell_type_to_index)

# adata_prim.obs['genotype'].value_counts(dropna=False)

# adata_prim.obs.columns


artifact = wandb.Artifact("best_model", type="model")
glob_str = os.path.join(save_dir, "best_model.pt")
artifact.add_file(glob_str)
run_session.log_artifact(artifact)

run_session.finish()
wandb.finish()
gc.collect()

# save the best model
# already incorporated into wrapper_train
if False:
    torch.save(best_model.state_dict(), save_dir / "best_model.pt")
    torch.save(vocab, save_dir / "vocab.pt")
    running_parameters = {
        "cell_type_to_index": data_produced["cell_type_to_index"],
        "genotype_to_index": data_produced["genotype_to_index"],
        "genes": data_produced["genes"],  # genes,
        "gene_ids": data_produced["gene_ids"],  # gene_ids,
    }
    running_parameters

    torch.save(running_parameters, save_dir / "running_parameters.pt")

# save the dataset
# torch.save(adata, save_dir / "adata.pt")
# torch.save(tokenized_train, save_dir / "tokenized_train.pt")
# torch.save(tokenized_valid, save_dir / "tokenized_valid.pt")
# torch.save(tokenized_train_next, save_dir / "tokenized_train_next.pt")
# torch.save(tokenized_valid_next, save_dir / "tokenized_valid_next.pt")

# results['celltype_umap'].savefig(save_dir / f"test.png", dpi=300,bbox_inches='tight')

# !cp -r $save_dir /content/drive/MyDrive/Colab\ Notebooks/scGPT/pancreatic/
