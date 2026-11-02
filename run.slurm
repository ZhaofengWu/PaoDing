#!/bin/zsh

#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH -t 365-0
#SBATCH --gres=gpu:1
#SBATCH --export=all
#SBATCH --output=/data/cl/user/zfw/slurm_output/sbatch-%j.out

echo "$@"
$@

