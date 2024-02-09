#!/bin/bash

# Fixed parameters
use_case="botnet"
server="PLACEHOLDER_SERVER4"
wandbp="CTGAN-dgm_${server}_${use_case}_hyerparam_search"
seed=9
eps=20

# Defaults for params that will change
default_optimiser="adam"
default_lr=0.0002
default_bs=500
default_decay=0.000001
default_pac=10

# python main_ctgan.py  ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=${default_optimiser} --generator_lr=${default_lr} --discriminator_lr=${default_lr} --batch_size=${default_bs} --generator_decay=${default_decay} --discriminator_decay=${default_decay} --pac=${default_pac}

batch_sizes="70 280 500"
for bs in $batch_sizes ;
do
  echo "Varying the learning rate for ${default_optimiser}"
  lrs="0.0001 0.0002 0.0010" # 0.0002 is the lr recommended by ctgan (which uses adam) for tabular data https://github.com/sdv-dev/CTGAN/blob/master/ctgan/synthesizers/ctgan.py#L153
  for lr in $lrs ;
  do
   python main_ctgan.py  ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=${default_optimiser} --generator_lr=${lr} --discriminator_lr=${lr} --batch_size=${bs} --generator_decay=${default_decay} --discriminator_decay=${default_decay} --pac=${default_pac}
  done

  optimiser="rmsprop"
  echo "Varying the learning rate for ${optimiser}"
  lrs="0.00005 0.00010 0.00020"
  for lr in $lrs ;
  do
   python main_ctgan.py  ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=${default_optimiser} --generator_lr=${lr} --discriminator_lr=${lr} --batch_size=${bs} --generator_decay=${default_decay} --discriminator_decay=${default_decay} --pac=${default_pac}
  done

  optimiser="sgd"
  echo "Varying the learning rate for ${optimiser}"
  lrs="0.0001 0.0010"
  for lr in $lrs ;
  do
   python main_ctgan.py  ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=${default_optimiser} --generator_lr=${lr} --discriminator_lr=${lr} --batch_size=${bs} --generator_decay=${default_decay} --discriminator_decay=${default_decay} --pac=${default_pac}
  done
done


echo "Varying pac for ${default_optimiser} with lr ${default_lr}"
pacs="1 5 15"
for pac in $pacs ;
do
 python main_ctgan.py  ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=${default_optimiser} --generator_lr=${default_lr} --discriminator_lr=${default_lr} --batch_size=${default_bs} --generator_decay=${default_decay} --discriminator_decay=${default_decay} --pac=${pac}
done
