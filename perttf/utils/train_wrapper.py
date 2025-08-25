import os, sys, time
from pathlib import Path
import torch
import wandb
import scgpt as scg
import scanpy as sc
from matplotlib import pyplot as plt
import copy
from typing import Dict, Mapping, Optional, Tuple, Any, Union, List, Tuple   

from perttf.utils.custom_tokenizer import tokenize_and_pad_batch, random_mask_value
from perttf.model.train_function import train, eval_testdata, evaluate, define_wandb_metrcis
from perttf.model.pertTF import PerturbationTFModel
from perttf.model.config_gen import generate_config
from perttf.model.pert_emb import load_pert_embedding_from_gears,load_pert_embedding_to_model
from perttf.utils.logger import create_logger
from perttf.utils.set_optimizer import create_optimizer_dict

def simplified_wrapper_train(
        model, 
        config, 
        train_loader,
        valid_loader,
        data_gen,
        logger = scg.logger,
        save_dir = None,
        device = None,
        eval_adata_dict: Dict = {},
        fold = ''
        ):
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #genes = adata.var.index.tolist()
    #vocab = Vocab(VocabPybind(genes + config.special_tokens, None))
    vocab = data_gen['vocab']
    gene_ids = data_gen['gene_ids']

    scaler = torch.cuda.amp.GradScaler(enabled=config.amp)
    DAB_separate_optim = True if config.dab_weight >0 else False


    num_batch_types = data_gen['num_batch_types']
    #vocab = data_gen['vocab']

    
    optimizer_dict = create_optimizer_dict(model, device, config, num_batch_types = num_batch_types)
    define_wandb_metrcis()
    wandb.watch(model, log='all')
    best_val_loss = float("inf")
    best_avg_bio = 0.0
    best_model = None
    for epoch in range(1, config.epochs + 1):
        epoch_start_time = time.time()
        
        # The loaders now handle all data prep internally and dynamically!
        if config.do_train:
            train(
                model,
                train_loader,
                config,
                vocab,
                optimizer_dict,
                epoch = epoch,
                logger = logger,
                device = device
            )
        val_loss, val_loss_next, val_mre, val_mre_next, val_dab, val_cls, val_pert, val_ps = evaluate(
            model,
            loader=valid_loader,
            config=config,
            vocab = vocab,
            epoch = epoch,
            device = device
        )
        elapsed = time.time() - epoch_start_time
        if logger is not None:
            logger.info("-" * 89)
            logger.info(
                f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
                f"valid loss/mse {val_loss:5.4f} | mre {val_mre:5.4f} | "
                f"valid loss/mse_next {val_loss_next:5.4f} | mre_next {val_mre_next:5.4f} | "
                f"valid dab {val_dab:5.4f} | valid cls {val_cls:5.4f} | valid pert {val_pert:5.4f} |"
                f"valid ps {val_ps:5.4f} |"
            )
            logger.info("-" * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            best_model_epoch = epoch
            if logger is not None:
                logger.info(f"Best model with score {best_val_loss:5.4f}")

        #if epoch % config.save_eval_interval == 0 or epoch == config.epochs:

        if epoch % 2 == 1:
            if logger is not None:
                logger.info(f"Saving model to {save_dir}")
            torch.save(best_model.state_dict(), save_dir / f"model_e{epoch}.pt")
            # change images of each epoch to subfolder
            save_dir2=save_dir / f'e{epoch}_imgs'
            save_dir2.mkdir(parents=True, exist_ok=True)
            # eval on testdata
            metrics_to_log={}
            for eval_dict_key, eval_adata in eval_adata_dict.items():
                results = eval_testdata(
                    #best_model,
                    model, # use current model
                    adata_t = eval_adata, #adata_t=data_gen['adata_sorted'], # if config.per_seq_batch_sample else adata,
                    gene_ids = gene_ids,
                    train_data_dict = data_gen,
                    config=config,
                    include_types=["cls"],
                    logger=logger,
                    epoch=epoch,
                    eval_key=eval_dict_key,
                )

                # metrics_to_log = {"test/" + k: v for k, v in results.items() if k != "adata"}
                #if "umap" in results:
                #    results["umap"].savefig(save_dir / f"embeddings_umap[cls]_e{best_model_epoch}.png", dpi=300,bbox_inches='tight')

                save_image_types=["batch_umap","celltype_umap","genotype_umap",
                    "genotype_umap2","pred_genotype",
                    "pred_celltype","genotype_next",
                    "next_umap_celltype","next_umap_genotype","next_umap_genotype_next"]
                for res_key, res_img_val in results.items():
                    if res_key in save_image_types:
                        res_img_val.savefig(save_dir2 / f"{eval_dict_key}_embeddings_{res_key}_e{epoch}.png", dpi=300,bbox_inches='tight')
                        metrics_to_log[f"test/{eval_dict_key}_{res_key}"] = wandb.Image(
                            str(save_dir2 / f"{eval_dict_key}_embeddings_{res_key}_e{epoch}.png"),
                            caption=f"{eval_dict_key}_{res_key} epoch {epoch}",
                        )
                # save the PS calculations
                if config.ps_weight > 0:
                    # plot the existing loness columns
                    adata_ret = results["adata"]
                    loness_columns = [x for x in adata_ret.obs if x.startswith('lonESS')]
                    for lon_c in loness_columns:
                        fig_lonc = sc.pl.umap(adata_ret,color=[lon_c],title=[f"loness {lon_c}  e{epoch}",],
                            frameon=False,return_fig=True, show=False,palette="tab20",)
                        plt.close()
                        # Replace '/' with '_' in ps_names
                        lon_c_rep=lon_c.replace('/', '_') 
                        fig_lonc.savefig(save_dir2 / f"{eval_dict_key}_loness_{lon_c_rep}_e{epoch}.png", dpi=300,bbox_inches='tight')
                    if ('ps_names' in data_gen) & ('ps_pred' in adata_ret.obsm) :
                        predicted_ps_names = data_gen['ps_names']
                        predicted_ps_score = adata_ret.obsm['ps_pred']
                        logger.info(f"predicted_ps_names: {predicted_ps_names}")
                        logger.info(f"predicted_ps_score: {predicted_ps_score.shape}")
                        for si_i in range(len(predicted_ps_names)):
                            lon_c = predicted_ps_names[si_i]
                            lon_c_rep=lon_c.replace('/', '_') 
                            adata_ret.obs[f'{lon_c_rep}_pred'] = predicted_ps_score[:,si_i]
                            fig_lonc_pred = sc.pl.umap(adata_ret,color=[f'{lon_c_rep}_pred'],title=[f"loness {lon_c_rep}_pred  e{epoch}",],
                                frameon=False,return_fig=True, show=False,palette="tab20",)
                            plt.close()
                            fig_lonc_pred.savefig(save_dir2 / f"{eval_dict_key}_loness_{lon_c_rep}_pred_e{epoch}.png", dpi=300,bbox_inches='tight')
                    results["adata"] = adata_ret
                if "adata" in results:
                    results["adata"].write_h5ad(save_dir / f'adata_last_validation_{eval_dict_key}.h5ad')

            #if "adata" in results_p:
            #    results_p["adata"].write_h5ad(save_dir / f'adata_last_validation.h5ad')

            metrics_to_log["test/best_model_epoch"] = best_model_epoch
            wandb.log(metrics_to_log)
            # wandb.log({"avg_bio": results.get("avg_bio", 0.0)})

        optimizer_dict['scheduler'].step()

        if DAB_separate_optim:
            optimizer_dict['scheduler_dab'].step()
        if config.ADV:
            optimizer_dict['scheduler_D'].step()
            optimizer_dict['scheduler_E'].step()
    
    # save the best model
    torch.save(best_model.state_dict(), save_dir / "best_model.pt")

    return best_model