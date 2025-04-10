
import torch
import os
from typing import Literal

def load_pert_embedding_from_gears(gears_path, adata, 
                                  intersect_type : Literal["common","gears"] = "common"):
    """
    load pretrained perturbation embeddings from GEARS model
    Args:
        gears_path: path to gears model
        adata:    scanpy object
        intersect_type:    choose the way to handle gears perturbed genes and adata.obs['genotype']. default intersect.
    Returns:
    """
    # load two required files
    gears_model_dict = torch.load(os.path.join(gears_path, 'model.pt'), map_location = torch.device('cpu'))
    pklfile=os.path.join(gears_path, 'pert_gene_list.pkl')
    import pickle
    with open(pklfile, 'rb') as f:
        pkl_data = pickle.load(f)
    #len(pkl_data['pert_gene_list'])
    gears_gene_list=pkl_data['pert_gene_list']

    # extract the perturbation column in adata
    a_genotype_list=adata.obs['genotype'].unique()
    # a_genotype_list

    if intersect_type == 'common':
        common_genotype_list = a_genotype_list[a_genotype_list.isin(gears_gene_list)]
    else:
        common_genotype_list = gears_gene_list

    print('common genes between GEARS embeddings and your adata genotypes: ' + ','.join(common_genotype_list))
    print('adata genotypes not in GEARS embeddings: ' + ','.join(a_genotype_list[~a_genotype_list.isin(gears_gene_list)]))


    import numpy as np
    common_genotype_indices = [gears_gene_list.index(genotype) for genotype in common_genotype_list if genotype in gears_gene_list]
    gears_model_subset = gears_model_dict['pert_emb.weight'][common_genotype_indices,:]
    genotype_index_gears = {genotype: index for index, genotype in enumerate(common_genotype_list)}
    #genotype_index_gears

    # add wild-type label
    genotype_index_gears['WT'] = len(genotype_index_gears)

    # Add a row of zeros at the end of weights so it represents "WT"
    gears_model_subset = np.vstack([gears_model_subset, np.zeros(gears_model_subset.shape[1])])
    # subset adata
    adata_subset = adata[adata.obs['genotype'].isin(genotype_index_gears.keys())]
    # construct return dict
    ret_dict = {'pert_embedding':gears_model_subset,
                'adata_subset':adata_subset,
                'genotype_index_gears':genotype_index_gears,}
    return ret_dict



def load_pert_embedding_to_model(o_model, model_weights):
    """
    load pretrained perturbation embeddings to model
    Args:
      omodel: model object
      model_weights: perturbation embeddings, must be a tensor or numpy array whose size is the same as o_model.pert_encoder.embedding.weight.shape
    Returns:
      model: model object with perturbation embeddings loaded
    """
    model_weights_tensor = torch.tensor(model_weights, 
                                             dtype=o_model.pert_encoder.embedding.weight.dtype, 
                                             device=o_model.pert_encoder.embedding.weight.device)
    if model_weights_tensor.shape != o_model.pert_encoder.embedding.weight.shape:
        raise ValueError(f"model_weights_tensor.shape {model_weights_tensor.shape} does not equal to o_model.pert_encoder.embedding.weight.shape {o_model.pert_encoder.embedding.weight.shape}")
    o_model.pert_encoder.embedding.weight.data = model_weights_tensor # torch.tensor(gears_model_subset,dtype=torch.double)
    return o_model


def generate_pert_embeddings(adata_target, adata_wt, candidate_genes,
                             model, gene_ids, cell_type_to_index, genotype_to_index, vocab,
                             config, device,
                             n_expands_per_epoch = 50,
                             n_epoch = 4, ):
    """
    Generate perturbation embeddings for a target dataset.

    Args:
      adata_target: AnnData object of the target dataset.
      adata_wt: AnnData object of the wildtype dataset.
      candidate_genes: List of candidate genes for perturbation.
    """
    # expand
    adata_bwmerge=sc.concat([adata_target]*n_expands_per_epoch + [adata_wt],axis=0)
    cell_emb_data_all = None
    perturb_info_all = None

    cs_matrix_res = np.zeros((adata_wt.shape[0], adata_target.shape[0] * n_expands_per_epoch, n_epoch))
    # loop over epochs
    for n_round in range(n_epoch):
        # assign genoytpe_next
        gt_next_1 = np.random.choice(list(candidate_genes), size = adata_target.shape[0] * n_expands_per_epoch)
        #adata_test_gw_wtmerge.obs.loc[adata_test_gw_wtmerge.obs['genotype']=='WT' ,'genotype_next'].value_counts()
        gt_next_2 = ['WT']*adata_wt.shape[0]
        gt_next = np.concatenate([gt_next_1,gt_next_2])
        adata_bwmerge.obs[ 'genotype_next'] = gt_next

        # feed into model
        model.to(device)
        eval_results_0 = eval_testdata(model, adata_bwmerge, gene_ids,
                                    train_data_dict={"cell_type_to_index":cell_type_to_index,
                                                      "genotype_to_index":genotype_to_index,
                                                      "vocab":vocab,},
                                    config = config,
                                    make_plots=False)
        #
        a_eva=eval_results_0['adata']
        cell_emb_data=a_eva.obsm['X_scGPT_next'] #[:10000,:]
        perturb_info=a_eva.obs[['genotype','genotype_next']]

        perturb_info['round']=n_round
        perturb_info['type']=['target']*adata_target.shape[0]*n_expands_per_epoch + ['wildtype']*adata_wt.shape[0]

        # calculate cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity

        # Calculate cosine similarity
        #adata_eva_copy_othergenes
        cell_emb_data_treat = cell_emb_data[perturb_info['type']=='target']

        cell_emb_data_ref = cell_emb_data[perturb_info['type']=='wildtype']
        # Calculate cosine similarity
        cosine_sim_matrix = cosine_similarity(cell_emb_data_ref, cell_emb_data_treat)
        cs_matrix_res[:,:,n_round] = cosine_sim_matrix
        if cell_emb_data_all is None:
            cell_emb_data_all = cell_emb_data
            perturb_info_all = perturb_info

        else:
            cell_emb_data_all = np.concatenate([cell_emb_data_all, cell_emb_data], axis=0)
            perturb_info_all = pd.concat([perturb_info_all, perturb_info], axis=0)


    return cell_emb_data_all, perturb_info_all, cs_matrix_res, a_eva
