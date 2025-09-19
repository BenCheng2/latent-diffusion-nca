#!/bin/bash
#SBATCH --job-name=fid_eval_ldm
#SBATCH --output=slurm_output_%j.txt
#SBATCH --error=slurm_error_%j.txt
#SBATCH --mail-user=bben.cheng@mail.utoronto.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=h200x4-long
#SBATCH --gpus=4
#SBATCH --time=23:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --exclusive

set -eo pipefail

module purge
module load cuda12.8/toolkit/12.8.0

# conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ldm

CKPT=/lustre/nvwulf/projects/YouGroup-nvwulf/wang159/latent-diffusion-nca/logs/2025-09-17T16-47-07_celebahq-ldm-vq-4/checkpoints/epoch=000553.ckpt
LOGDIR=$(dirname $CKPT)/..
OUTDIR=/lustre/nvwulf/projects/YouGroup-nvwulf/wang159/latent-diffusion-nca/outputs/fid_eval_${SLURM_JOB_ID}
DATASET=/lustre/nvwulf/projects/YouGroup-nvwulf/wang159/data/celebahq/valid   # CelebA-HQ 真实数据路径

mkdir -p $OUTDIR

echo "Job started on $(hostname) at $(date)"

# ===== 保证 logdir 下有 config.yaml =====
if [ ! -f "$LOGDIR/config.yaml" ]; then
    echo "config.yaml not found in $LOGDIR, copying from project.yaml"
    cp $LOGDIR/configs/*-project.yaml $LOGDIR/config.yaml
fi

# ===== Step 1: 生成样本 =====
CUDA_VISIBLE_DEVICES=0 python scripts/sample_diffusion.py --resume $CKPT --n_samples 12500 --batch_size 128 --custom_steps 200 --eta 0.0 --logdir $OUTDIR/gpu0
CUDA_VISIBLE_DEVICES=1 python scripts/sample_diffusion.py --resume $CKPT --n_samples 12500 --batch_size 128 --custom_steps 200 --eta 0.0 --logdir $OUTDIR/gpu1
CUDA_VISIBLE_DEVICES=2 python scripts/sample_diffusion.py --resume $CKPT --n_samples 12500 --batch_size 128 --custom_steps 200 --eta 0.0 --logdir $OUTDIR/gpu2
CUDA_VISIBLE_DEVICES=3 python scripts/sample_diffusion.py --resume $CKPT --n_samples 12500 --batch_size 128 --custom_steps 200 --eta 0.0 --logdir $OUTDIR/gpu3

# ===== Step 2: 计算 FID =====
pip install --user pytorch-fid
pytorch-fid $DATASET $OUTDIR/*/img >> $OUTDIR/fid_result.txt

echo "FID result:"
cat $OUTDIR/fid_result.txt

echo "Job finished at $(date)"
