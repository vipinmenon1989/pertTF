
import torch
import os

def load_pert_embedding_from_gears(gears_path, adata):
    """
    load pretrained perturbation embeddings from GEARS model
    Args:

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
    a_genotype_list

    common_genotype_list = a_genotype_list[a_genotype_list.isin(gears_gene_list)]

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