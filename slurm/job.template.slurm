#!/bin/bash
# Reference SBATCH layout used as a starting point by ``patch_sim.patch_job_script``.
# The actual file the BO controller writes per sequence is generated from this
# shape plus values from ``config.example.yaml`` (``slurm.*`` and
# ``slurm.module_block``).
#SBATCH --job-name=bo_<candidate>_<sequence>
#SBATCH --time=16:00:00
#SBATCH --mem=40GB
#SBATCH --output=slurm-%j-%N.out
#SBATCH --gres=gpu:1
#SBATCH --partition=<PARTITION>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mail-type=NONE

# TODO: load whichever modules / activate whichever env your cluster needs.
# Populate ``slurm.module_block`` (or ``resident_worker.module_block``) in
# the YAML config to emit additional lines here.

python equilibrium.py
python production.py
