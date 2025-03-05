### scGPT Slurm Job Submission on HPC

This repository contains the necessary scripts to run scGPT on a High Performance Computing (HPC) cluster using Slurm. The scripts are designed to be used with the [scGPT](https://github.com/bowang-lab/scGPT) repository.

### Usage

1. Clone this repository to your HPC cluster account.
2. Modify the `.env_example` file to reflect your wandb API key and save it as `.env` in the same directory.
3. Modify the `perturb_celltype.py` file to reflect the correct paths and parameters for your HPC cluster.
4. Create a `data` directory. Move the `.h5ad` file you want to run scGPT on into the `data` directory.
5. Modify the `slurm_example.sh` file to reflect the correct paths and parameters for your HPC cluster. Make sure to update `$DATA_PATH` to reflect your .h5ad file as well. I renamed my `slurm_example.sh` script to `slurm.sh`.

You can then use the following command to submit a job to the HPC cluster:
```bash
sbatch slurm.sh
```

The `slurm.sh` file will check whether the required software is installed and will proceed to set up the working directory if it is not. This includes: 

- Cloning the scGPT repository
- Downloading the necessary data
- Setting up the Python environment (using UV)
    - (make sure `$HOME_DIRECTORY` reflects the directory path you clone this repository to)
