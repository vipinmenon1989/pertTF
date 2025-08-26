#!/bin/bash
#SBATCH --job-name=cv_pert_tf_10000
#SBATCH --account=ihc
#SBATCH --partition=ihc         # or whichever partition you use
#SBATCH --nodelist=ihc-h200-1
#SBATCH --time=60:00:00          # 30 days
#SBATCH --mem=350G
#SBATCH --output=//autofs/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/cvPertfscore_3000_gene.log
#SBATCH --error=/autofs/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/cvPertfscore_3000_gene.err
cd /autofs/projects-t3/lilab/vmenon/Pert-TF-model/pertTF/

source /local/projects-t3/lilab/vmenon/anaconda3/etc/profile.d/conda.sh
conda activate scgpt_menon

python cv_Pert-TF_3000_genes.py

