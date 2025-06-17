
import scanpy as sc
import numpy as np

def calculate_lonESS_score(adata, overall_fraction_dict=None,
                           recalculate_nn=False,
                           nn_name=None,
                           n_neighbors=30,
                           target_genotype = None,
                           n_pcs=20,
                           delta = 0.0001,
                           ):
  """
  calculate the lochNESS score for a single cell
  """
  n_cells = adata.n_obs
  # calculate the overall fraction
  if overall_fraction_dict is None:
    overall_fraction = adata.obs['genotype'].value_counts(normalize=True)
    # convert overall_fraction into a dictionary structure
    overall_fraction_dict = overall_fraction.to_dict()

  if recalculate_nn:
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs,key_added=nn_name)
  nn_key='distances' if nn_name is None else f'{nn_name}_distances'
  # iterate each cell

  lowess_vec=[]
  for cell_id in adata.obs_names:
    cell_index = adata.obs_names.get_loc(cell_id)
    if target_genotype is not None:
      cell_genotype=target_genotype
    else:
      cell_genotype=adata.obs.loc[cell_id,'genotype']
    neighboring_vec = adata.obsp[nn_key][cell_index, :]
    indices = neighboring_vec.nonzero()[1]
    neighboring_cells = adata.obs.index[indices]
    neighboring_genotypes = sum(adata.obs.loc[neighboring_cells, 'genotype'] == cell_genotype)
    lonESS_score = neighboring_genotypes / n_neighbors
    overall_score = overall_fraction_dict[cell_genotype]

    #loness_score_adjusted = lonESS_score - overall_score
    if overall_score ==0:
      overall_score = 0.0001
    lonESS_score_adjusted = lonESS_score / overall_score -1
    # generate a random noise and added to the score
    if delta > 0:
      noise = np.random.normal(0, delta)
      lonESS_score_adjusted += noise
    lowess_vec.append(lonESS_score_adjusted)
  return lowess_vec
