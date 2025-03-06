#!/bin/bash

# Set output and error files
#SBATCH -o /PATH/TO/HOME/pertTF/runs/%j/pertTF.log
#SBATCH -e /PATH/TO/HOME/pertTF/runs/%j/pertTF.err

# Set partition (can set to nano for environment testing or med-gpu and large-gpu for actual runs)
#SBATCH -p small-gpu
#SBATCH -N 1

# Set working directory to the scratch filesystem inside Pegasus
#SBATCH -D /scratch/ligrp/USER_SCRATCH_DIRECTORY/pertTF

# Job name 
#SBATCH -J pertTF
#SBATCH --export=None

# Set time
#SBATCH -t 1:30:00

# Replace with directory pertTF repository is cloned at
HOME_DIR=/PATH/TO/HOME/pertTF
RUN_DIR=$HOME_DIR/runs/${SLURM_JOB_ID}/
mkdir -p $RUN_DIR

# Set working directory 
WD=/scratch/ligrp/USER_SCRATCH_DIRECTORY/pertTF
DATA_DIR=$WD/data
DATA_PATH=$DATA_DIR/D18_diabetes_merged_reduced.h5ad

# Check if working directory exists
if [ -d "$WD" ]; then
    echo "Directory $WD exists. Aborting setup."
else
    # Set up working directory
    echo "Directory $WD does not exist. Initiating setup."
    mkdir -p $WD

    echo "created WD: $WD"
    cd $WD

    # Pull from the pertTF branch of my scGPT fork (I had to update one line of code in their codebase)
    git clone -b pertTF_branch https://github.com/zacheliason/scGPT.git $WD/scGPT
    mv $WD/scGPT/* $WD/
    rm -rf $WD/scGPT

    # Copy over the rest of the codebase
    cp -r $HOME_DIR/* $WD/

    # Download and set up UV (UV is a really fast Python package/project manager)
    export UV_ROOT=$WD/.uv
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH=$UV_ROOT/bin:$PATH

    # Create virtual environment
    uv python install 3.10
    uv venv $WD/.venv --python 3.10
    uv pip install -r $WD/demos/run_on_hpc/pyproject.toml
    source $WD/.venv/bin/activate

    # Load WandB key
    source $WD/demos/run_on_hpc/.env

    # Export env variables 
    export WANDB_API_KEY
    export WANDB_SILENT
    export WANDB__EXECUTABLE
    export KMP_WARNINGS

    # Download data
    # If we store the .h5ad files in a Google Drive folder, we can download them using gdown
    #mkdir -p $DATA_DIR
    #gdown --folder https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y -O $DATA_DIR/scGPT_human

    #echo "Download complete. Files saved to: $DATA_DIR"
fi

# Enter working directory
cd $WD

# Add python dependencies to python path
export PYTHONPATH=$(pwd)/perttf:$(pwd)/scgpt:$PYTHONPATH

echo "Running scGPT on data in $DATA_DIR and saving results to $RUN_DIR"
source $WD/.venv/bin/activate && python3 $WD/demos/run_on_hpc/perturb_celltype.py -d $DATA_PATH -o $RUN_DIR
