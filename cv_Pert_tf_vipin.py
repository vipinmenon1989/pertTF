import gpu_pin
import copy
import json
import time
import os
import logging
import numpy as np
import resource
import psutil
os.environ["WANDB_API_KEY"]= "3f42c1f651e5c0658618383b0a787f06656bd550"
os.environ["KMP_WARNINGS"] = "off"
os.environ["OMP_NUM_THREADS"] = "10"
os.environ["OPENBLAS_NUM_THREADS"] = "10"
os.environ["MKL_NUM_THREADS"] = "10"
os.environ["VECLIB_MAXIMUM_THREADS"] = "10"
os.environ["NUMEXPR_NUM_THREADS"] = "10"
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')
import torch
from anndata import AnnData
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
from tqdm import tqdm
import gseapy as gp
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)
from collections import Counter, defaultdict
import scgpt as scg
from scgpt.tasks import GeneEmbedding
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.model import TransformerModel
from scgpt.preprocess import Preprocessor
from scgpt.utils import set_seed
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import cm
import csv
import pandas as pd
from pertTF import PerturbationTFModel
from config_gen import generate_config
from train_data_gen import produce_training_datasets
from train_function import train,wrapper_train,eval_testdata
import wandb, random
from sklearn.model_selection import KFold


# ---- basic logging ----
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s")
logger = logging.getLogger("cv_train_eval")

# ---- config defaults (you can also load from your existing config.json) ----
hyperparameter_defaults = dict(
    seed=42,
    dataset_name="pancreatic",
    do_train=True,
    load_model=None,
    GEPC=True,
    ecs_thres=0.7,
    dab_weight=0.0,
    this_weight=1.0,
    next_weight=0.0,
    n_rounds=1,
    next_cell_pred_type='identity',
    ecs_weight=1.0,
    cell_type_classifier=True,
    cell_type_classifier_weight=1.0,
    perturbation_classifier_weight=50.0,
    perturbation_input=False,
    CCE=False,
    mask_ratio=0.15,
    epochs=60,
    n_bins=51,
    lr=1e-3,
    batch_size=32,
    layer_size=32,
    nlayers=2,
    nhead=4,
    dropout=0.4,
    schedule_ratio=0.99,
    save_eval_interval=5,
    log_interval=60,
    fast_transformer=True,
    pre_norm=False,
    amp=True,
    do_sample_in_train=False,
    ADV=False,
    adv_weight=10000,
    adv_E_delay_epochs=2,
    adv_D_delay_epochs=2,
    lr_ADV=1e-3,
    DSBN=False,
    per_seq_batch_sample=False,
    use_batch_label=False,
    schedule_interval=1,
    explicit_zero_prob=True,
    n_hvg=10000,
    mask_value=-1,
    pad_value=-2,
    pad_token="<pad>",
    ps_weight=0.0,
)

# ---- generate / load config ----
config, run_session = generate_config(hyperparameter_defaults, wandb_mode="online")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# ---- setup output dir ----
dataset_name = config.dataset_name
save_dir = Path(f"/local/projects-t3/lilab/vmenon/Pert-TF-model/cv_{dataset_name}-{time.strftime('%b%d-%H-%M')}/")
save_dir.mkdir(parents=True, exist_ok=True)
logger.info(f"Outputs and checkpoints will be saved to {save_dir}")

# ---- load and preprocess data ----
adata0 = sc.read_h5ad("object_integrated_assay3_annotated_nounk_raw.cleaned.h5ad")
logger.info(f"Loaded AnnData: n_obs={adata0.n_obs}, n_vars={adata0.n_vars}")

# example cleaning similar to original
if 'sub.cluster' in adata0.obs.columns:
    valid = adata0.obs['sub.cluster'].notna()
    dropped = np.sum(~valid)
    adata0 = adata0[valid].copy()
    logger.info(f"Dropped {dropped} cells missing sub.cluster.")
adata0.obs['celltype'] = adata0.obs.get('sub.cluster', '')
adata0.layers['GPTin'] = adata0.X.copy()
adata0.obs['gene'] = (
    adata0.obs['gene']
    .str.replace('124_NANOGe_het','124_NANOGe-het', regex=False)
    .str.replace('123_NANOGe_het','123_NANOGe-het', regex=False)
)
genotypes = (
    adata0.obs['gene'].str.split('_').str[-1]
    .replace({'WT111':'WT','WT4':'WT','NGN3':'NEUROG3'})
    .fillna('WT')
)
mask = (genotypes != 'WT') | (np.random.rand(adata0.n_obs) < 0.01)
adata = adata0[mask, :].copy()
logger.info(f"Subsampled: retained {adata.n_obs} cells out of {adata0.n_obs}.")

# Preprocess once (will be reused for each fold split logic inside produce_training_datasets)
preprocessor = Preprocessor(
    use_key="GPTin",
    normalize_total=None,
    log1p=False,
    subset_hvg=False,
    hvg_flavor="seurat_v3",
    binning=config.n_bins,
    result_binned_key="X_binned",
)
preprocessor(adata, batch_key=None)
logger.info("Preprocessing complete.")

# ---- 5-fold cross validation ----
kf = KFold(n_splits=5, shuffle=True, random_state=config.seed)
obs_names = np.array(adata.obs_names)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(obs_names), start=1):
    logger.info(f"===== Fold {fold}/5 =====")
    save_dir_fold = save_dir / f"fold_{fold}"
    save_dir_fold.mkdir(parents=True, exist_ok=True)

    # assign split labels for produce_training_datasets to pick up (if it uses adata.obs['split'])
    adata.obs['split'] = 'unused'
    train_barcodes = obs_names[train_idx]
    val_barcodes = obs_names[val_idx]
    adata.obs.loc[train_barcodes, 'split'] = 'train'
    adata.obs.loc[val_barcodes, 'split'] = 'validation'

    # produce fold-specific data (only uses relevant obs internally)
    fold_data = produce_training_datasets(adata, config, next_cell_pred='identity')

    # save vocab
    vocab = fold_data['vocab']
    with open(save_dir_fold / 'vocab.json', 'w') as vf:
        json.dump(vocab.get_stoi(), vf, indent=2)
    torch.save(vocab, save_dir_fold / 'vocab.pt')

    ntokens = len(vocab)
    n_body_layers = 3  # as per your earlier code

    # instantiate model
    model = PerturbationTFModel(
        n_pert=fold_data['n_perturb'],
        nlayers_pert=n_body_layers,
        n_ps=1,
        ntoken=ntokens,
        d_model=config.layer_size,
        nhead=config.nhead,
        d_hid=config.layer_size,
        nlayers=config.nlayers,
        nlayers_cls=3,
        n_cls=fold_data['n_cls'],
        vocab=vocab,
        dropout=config.dropout,
        pad_token=config.pad_token,
        pad_value=config.pad_value,
        do_mvc=config.GEPC,
        do_dab=(config.dab_weight > 0),
        use_batch_labels=config.use_batch_label,
        num_batch_labels=fold_data['num_batch_types'],
        domain_spec_batchnorm=config.DSBN,
        n_input_bins=config.n_bins,
        ecs_threshold=config.ecs_thres,
        explicit_zero_prob=config.explicit_zero_prob,
        use_fast_transformer=config.fast_transformer,
        pre_norm=config.pre_norm,
    ).to(device)

    # patch if identity prediction
    if config.next_cell_pred_type == "identity":
        orig_encode = model.encode_batch_with_perturb
        def encode_batch_force_t0(
            src, values, src_key_padding_mask, batch_size,
            batch_labels=None, pert_labels=None, pert_labels_next=None,
            output_to_cpu=True, time_step=0, return_np=False
        ):
            return orig_encode(
                src, values, src_key_padding_mask, batch_size,
                batch_labels, pert_labels, pert_labels_next,
                output_to_cpu, 0, return_np
            )
        model.encode_batch_with_perturb = encode_batch_force_t0

    fold_run = wandb.init(
    project=run_session.project if hasattr(run_session, "project") else None,
    name=f"Complete_data_fold_{fold}",
    group="cross_validation",
    reinit=True,
    config=config.as_dict() if hasattr(config, "as_dict") else None,
)
    fold_run.config.update({"fold": fold})
    # train + validate
    train_result = wrapper_train(
        model, config, fold_data,
        eval_adata_dict={'validation': fold_data['adata_sorted']},
        save_dir=save_dir_fold,
        fold=fold,
        run=fold_run
    )
    best_model = train_result["model"]
    fold_results.append({"fold": fold,"best_model_epoch": train_result["best_model_epoch"],"best_val_loss": train_result["best_val_loss"],"save_path": str(save_dir_fold / "best_model.pt")})
    # save model state
    torch.save(best_model.state_dict(), save_dir_fold / "best_model.pt")

    
    # log artifact
    artifact = wandb.Artifact(f"best_model_fold{fold}", type="model")
    artifact.add_file(str(save_dir_fold / "best_model.pt"))
    fold_run.log_artifact(artifact)
    fold_run.finish()
    # cleanup
    del model, best_model, fold_data, vocab
    torch.cuda.empty_cache()

# ---- summary ----
df_summary = pd.DataFrame(fold_results)
summary_csv = save_dir / "fold_summary.csv"
df_summary.to_csv(summary_csv, index=False)
logger.info(f"Saved fold summary to {summary_csv}")



# finish wandb
run_session.finish()
wandb.finish()
logger.info("Done.")
