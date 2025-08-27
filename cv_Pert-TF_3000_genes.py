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
from torchtext._torchtext import Vocab as VocabPybind
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
from perttf.model.pertTF import PerturbationTFModel
from perttf.model.config_gen import generate_config
from perttf.model.train_data_gen import produce_training_datasets
from perttf.model.train_function import train, wrapper_train, eval_testdata
import wandb, random
from sklearn.model_selection import KFold


# ---- basic logging ----
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s")
logger = logging.getLogger("cv_train_eval")

# ---- config defaults ----
hyperparameter_defaults = dict(
    seed=42,
    dataset_name="pancreatic",
    do_train=True,
    load_model=None,
    GEPC=True,
    ecs_thres=0.7,
    dab_weight=0.0,
    this_weight=1.0,
    next_weight=10.0,
    n_rounds=1,
    next_cell_pred_type='identity',
    ecs_weight=1.0,
    cell_type_classifier=True,
    cell_type_classifier_weight=1.0,
    perturbation_classifier_weight=50.0,
    perturbation_input=False,
    CCE=False,
    mask_ratio=0.15,
    epochs=150,
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

    # === analysis knobs ===
    USE_HVG=True,          # True -> select HVGs; False -> keep all genes
    n_hvg=3000,            # number of HVGs when USE_HVG=True
    mask_value=-1,
    pad_value=-2,
    pad_token="<pad>",
    ps_weight=0.0,
    # === cell-level controls ===
    max_cells=None,        # e.g., 100_000 to cap; None to keep all
    max_cells_seed=1337,   # rng for max_cells sampling
    filter_missing_subcluster=True,  # drop cells with missing sub.cluster
    KEEP_WT_FRAC=1.0,      # 1.0 = keep all WT; 0.01 ~ keep 1% of WT
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

# ---- load data ----
adata0 = sc.read_h5ad("object_integrated_assay3_annotated_nounk_raw.cleaned.h5ad")
logger.info(f"Loaded AnnData: n_obs={adata0.n_obs}, n_vars={adata0.n_vars}")
_loaded_cells = int(adata0.n_obs)

def assert_cells(tag, ad, ge=None):
    if ge is None:
        ge = _loaded_cells
    if ad.n_obs < ge:
        raise RuntimeError(
            f"[{tag}] cell count fell from {ge} to {ad.n_obs}. "
            f"Unexpected downsampling — grep for 'random.sample', 'subsample', or 'Reduced AnnData'."
        )

# ---- optional: cap total cells (explicit + reproducible) ----
if config.max_cells is not None and adata0.n_obs > int(config.max_cells):
    rng = np.random.default_rng(int(config.max_cells_seed))
    idx = rng.choice(adata0.n_obs, size=int(config.max_cells), replace=False)
    adata0 = adata0[idx].copy()
    logger.info(f"[cell-cap] Reduced AnnData: n_obs={adata0.n_obs}, n_vars={adata0.n_vars}")
assert_cells("post-cap-check", adata0, ge=_loaded_cells if config.max_cells is None else min(_loaded_cells, int(config.max_cells)))

# ---- optional: drop NA sub.cluster ----
if config.filter_missing_subcluster and ('sub.cluster' in adata0.obs.columns):
    before = adata0.n_obs
    valid = adata0.obs['sub.cluster'].notna()
    dropped = int(np.sum(~valid))
    adata0 = adata0[valid].copy()
    logger.info(f"Dropped {dropped} cells missing sub.cluster (kept {adata0.n_obs}/{before}).")

# ---- celltype and GPTin layer ----
adata0.obs['celltype'] = adata0.obs.get('sub.cluster', '')
if 'GPTin' not in adata0.layers:
    adata0.layers['GPTin'] = adata0.X.copy()

# ---- tidy genotype labels ----
if 'gene' not in adata0.obs.columns:
    raise ValueError("adata0.obs['gene'] not found; cannot derive genotypes.")
adata0.obs['gene'] = (
    adata0.obs['gene']
      .astype(str)
      .str.replace('124_NANOGe_het','124_NANOGe-het', regex=False)
      .str.replace('123_NANOGe_het','123_NANOGe-het', regex=False)
)
genotypes = (
    adata0.obs['gene'].str.split('_').str[-1]
      .replace({'WT111':'WT','WT4':'WT','NGN3':'NEUROG3'})
      .fillna('WT')
)

# ---- REQUIRED: keep all non-WT + ~1% of WT (reproducible) ----
KEEP_WT_FRAC = float(getattr(config, "KEEP_WT_FRAC", 0.01))  # set 0.01 for ~1% WT
rng = np.random.default_rng(int(config.seed))

# mask = non-WT OR (WT and passes 1% draw)
mask = (genotypes != 'WT') | ((genotypes == 'WT') & (rng.random(adata0.n_obs) < KEEP_WT_FRAC))

adata = adata0[mask, :].copy()

# helpful logs
n_wt_total = int((genotypes == 'WT').sum())
n_wt_kept  = int(((genotypes == 'WT') & mask).sum())
kept_frac  = mask.mean()
logger.info(
    f"Genotype thinning: kept {adata.n_obs}/{adata0.n_obs} cells "
    f"(~{kept_frac:.2%}); WT kept {n_wt_kept}/{n_wt_total} "
    f"(~{(n_wt_kept / max(n_wt_total, 1)):.2%})."
)

assert adata.n_obs > 0, "No cells left after genotype thinning."

# ---- (Optional) HVG selection on GPTin -> create binned layer ----
USE_HVG = bool(config.USE_HVG)
n_hvg   = int(config.n_hvg)

if USE_HVG:
    ad_tmp = adata.copy()
    ad_tmp.X = ad_tmp.layers['GPTin'].copy()  # choose source for HVG calc
    sc.pp.normalize_total(ad_tmp, target_sum=1e4)
    sc.pp.log1p(ad_tmp)
    sc.pp.highly_variable_genes(ad_tmp, n_top_genes=n_hvg, flavor="seurat_v3", batch_key=None)

    # select by NAMES (avoids shape mismatches)
    hvg_genes = ad_tmp.var_names[ad_tmp.var['highly_variable']]
    adata = adata[:, hvg_genes].copy()  # <-- one slice; layers follow automatically

    # sanity check: layer and var are aligned
    assert adata.layers['GPTin'].shape[1] == adata.n_vars, \
        f"Layer/var mismatch: {adata.layers['GPTin'].shape[1]} vs {adata.n_vars}"
    logger.info(f"HVG selection: kept {adata.n_vars} genes (requested {n_hvg}).")

else:
    logger.info("HVG selection disabled (USE_HVG=False). Keeping all genes.")

# ---- binning (scGPT expects integer bins sometimes) ----
preprocessor = Preprocessor(
    use_key="GPTin",
    normalize_total=None,   # we already handled normalization/logging (only for HVG pick)
    log1p=False,            # keep GPTin as-is; we’re just binning it
    subset_hvg=False,       # we already subset if USE_HVG=True
    hvg_flavor="seurat_v3",
    binning=config.n_bins,
    result_binned_key="X_binned",
)
preprocessor(adata, batch_key=None)
logger.info("Preprocessing (binning) complete.")
assert 'X_binned' in adata.layers, "Binning did not produce X_binned."

# ---- 5-fold cross validation ----
kf = KFold(n_splits=5, shuffle=True, random_state=int(config.seed))
obs_names = np.array(adata.obs_names)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(obs_names), start=1):
    logger.info(f"===== Fold {fold}/5 =====")
    save_dir_fold = save_dir / f"fold_{fold}"
    save_dir_fold.mkdir(parents=True, exist_ok=True)

    # assign split labels for produce_training_datasets to pick up
    adata.obs['split'] = 'unused'
    train_barcodes = obs_names[train_idx]
    val_barcodes = obs_names[val_idx]
    adata.obs.loc[train_barcodes, 'split'] = 'train'
    adata.obs.loc[val_barcodes, 'split'] = 'validation'

    # produce fold-specific data
    fold_data = produce_training_datasets(adata, config, next_cell_pred='pert')

    # save vocab
    vocab = fold_data['vocab']
    with open(save_dir_fold / 'vocab.json', 'w') as vf:
        json.dump(vocab.get_stoi(), vf, indent=2)
    torch.save(vocab, save_dir_fold / 'vocab.pt')

    ntokens = len(vocab)
    n_body_layers = 3  # as per earlier code

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

    # force identity-time encoding if requested
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
        name=f"hvg{n_hvg if USE_HVG else 'all'}_10000_fold_{fold}",
        group="cross_validation",
        reinit=True,
        config=config.as_dict() if hasattr(config, "as_dict") else None,
    )
    fold_run.config.update({"fold": fold, "USE_HVG": USE_HVG, "n_hvg": n_hvg})

    # train + validate
    train_result = wrapper_train(
        model, config, fold_data,
        eval_adata_dict={'validation': fold_data['adata_sorted']},
        save_dir=save_dir_fold,
        fold=fold,
        run=fold_run
    )
    best_model = train_result["model"]
    fold_results.append({
        "fold": fold,
        "best_model_epoch": train_result["best_model_epoch"],
        "best_val_loss": train_result["best_val_loss"],
        "save_path": str(save_dir_fold / "best_model.pt")
    })
    # save model
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