#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --mem-per-cpu=2G
#SBATCH --time=12:00:00
#SBATCH --job-name=lcld-tvae
#SBATCH --partition=short
#SBATCH --clusters=all

module load Anaconda3
source activate /data/PLACEHOLDER_PATH/PLACEHOLDER/conda_envs/c_gan
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/PLACEHOLDER_PATH/PLACEHOLDER/conda_envs/c_gan/lib

# Fixed parameters
use_case="lcld"
server="PLACEHOLDER_SERVER3"
wandbp="TVAE-dgm_${server}_${use_case}_hyperparam_search"
seed=9
eps=20

# Defaults for params that will change
default_bs=500
default_l2scale=0.00001
default_loss_factor=2

# python main_ctgan.py  ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=${default_optimiser} --generator_lr=${default_lr} --discriminator_lr=${default_lr} --batch_size=${default_bs} --generator_decay=${default_decay} --discriminator_decay=${default_decay} --pac=${default_pac}

batch_sizes="70 280 500"
for bs in $batch_sizes ;
do
  echo "Varying the weight decay"
  l2scales="0.000005 0.00001 0.0001 0.0002 0.0010" 
  for l2scale in $l2scales;
  do
   python main_tvae.py  ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps  --batch_size=${bs} --l2scale=${l2scale} --loss_factor=${default_loss_factor} 
  done


  echo "Varying the weight loss factor"
  loss_factors="1 2 3 4" 
  for loss_factor in $loss_factors ;
  do
  python main_tvae.py  ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps  --batch_size=${bs} --l2scale=${default_l2scale} --loss_factor=${loss_factor} 
  done
done


