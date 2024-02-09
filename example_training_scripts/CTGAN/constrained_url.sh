#!/bin/bash

# Fixed parameters
use_case="url"
wandbp="CTGAN-transform_only_feats_in_constr_${use_case}"
eps=150

# Defaults for params that will change
default_optimiser="adam"
default_lr=0.0002
default_bs=500
default_decay=0.000001
default_pac=1

seeds="2 5 7 9 21"
for seed in $seeds ;
do
  CUDA_VISIBLE_DEVICES=-1 python main_ctgan.py  ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=${default_optimiser} --generator_lr=${default_lr} --discriminator_lr=${default_lr} --batch_size=${default_bs} --generator_decay=${default_decay} --discriminator_decay=${default_decay} --pac=${default_pac}
#  CUDA_VISIBLE_DEVICES=-1 python main_ctgan.py  ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=${default_optimiser} --generator_lr=${default_lr} --discriminator_lr=${default_lr} --batch_size=${default_bs} --generator_decay=${default_decay} --discriminator_decay=${default_decay} --pac=${default_pac} --version="constrained" --label_ordering="corr"
#  CUDA_VISIBLE_DEVICES=-1 python main_ctgan.py  ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=${default_optimiser} --generator_lr=${default_lr} --discriminator_lr=${default_lr} --batch_size=${default_bs} --generator_decay=${default_decay} --discriminator_decay=${default_decay} --pac=${default_pac} --version="constrained" --label_ordering="kde"
  CUDA_VISIBLE_DEVICES=-1 python main_ctgan.py  ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=${default_optimiser} --generator_lr=${default_lr} --discriminator_lr=${default_lr} --batch_size=${default_bs} --generator_decay=${default_decay} --discriminator_decay=${default_decay} --pac=${default_pac} --version="constrained" --label_ordering="random"
done

