#!/bin/bash
#SBATCH --job-name=latent_diffusion_nca
#SBATCH --output=slurm_output_%j.txt
#SBATCH --error=slurm_error_%j.txt
#SBATCH --mail-user=bben.cheng@mail.utoronto.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=h200x4-long
#SBATCH --gpus=4
#SBATCH --time=47:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --no-kill
#SBATCH --exclusive

set -eo pipefail

module purge
module load cuda12.8/toolkit/12.8.0

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ldm

export DATASET=/lustre/nvwulf/projects/YouGroup-nvwulf/wang159/data/celebahq256_imgs

cd /lustre/nvwulf/projects/YouGroup-nvwulf/wang159/latent-diffusion-nca

echo "Job started on $(hostname) at $(date)" >> slurm_output_$SLURM_JOB_ID.txt

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',   ' '\n' | wc -l)
python main.py --base configs/latent-diffusion/celebahq-ldm-vq-4.yaml -t --gpus 0,1,2,3

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Detected NUM_GPUS=$NUM_GPUS"
echo "Job finished at $(date)" >> slurm_output_$SLURM_JOB_ID.txt
