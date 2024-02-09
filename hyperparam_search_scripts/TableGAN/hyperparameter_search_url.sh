#!/bin/bash

# Fixed parameters
use_case="url"
server="PLACEHOLDER_SERVER4"
wandbp="tableGAN_c-dgm_${server}_${use_case}_hyerparam_search"
seed=9
eps=300

# Defaults for params that will change
default_optimiser="adam"
default_lr=0.0002
default_bs=512
default_random_dim=random_dim

# python main_tableGAN.py ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=${default_optimiser} --lr=${default_lr} --batch_size=${default_bs} --random_dim=${default_random_dim}

batch_sizes="128 256 512"
for bs in $batch_sizes ;
do
  echo "Varying the learning rate for ${default_optimiser}"
  lrs="0.00005 0.00010 0.00020" # NOTE: 0.00100 leads to very shaky loss curves, tried for bs 64
  for lr in $lrs ;
  do
   python main_tableGAN.py ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=${default_optimiser} --lr=${lr} --batch_size=${bs} --random_dim=${default_random_dim}
  done

  optimiser="rmsprop"
  echo "Varying the learning rate for ${optimiser}"
  lrs="0.0001 0.0002 0.0010" # 0.0002 is the lr recommended by ctgan (which uses adam) for tabular data https://github.com/sdv-dev/CTGAN/blob/master/ctgan/synthesizers/ctgan.py#L153
  for lr in $lrs ;
  do
   python main_tableGAN.py ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=${optimiser} --lr=${lr} --batch_size=${bs} --random_dim=${default_random_dim}
#   # try also the recommended rep and pac settings from ctgan
#   rep=1
#   pac=10
#   weight_decay=0.000001
#   python main.py --wandb_project=$wandbp --seed=$seed --epochs=$eps --eval_epochs_interval=${eval_eps} --use_case=${use_case} --disc_repeats=${rep} --optimiser=${optimiser} --d_lr=${lr} --g_lr=${lr} --pac=${pac} --weight_decay=${weight_decay} --batch_size=${bs}
  done

  optimiser="sgd"
  echo "Varying the learning rate for ${optimiser}"
  lrs="0.0001 0.0010"
  for lr in $lrs ;
  do
   python main_tableGAN.py ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=${optimiser} --lr=${lr} --batch_size=${bs} --random_dim=${default_random_dim}
  done
done


echo "Varying the random_dim for ${default_optimiser} with lr ${default_lr}"
rds="50 200" # default was 100
for rd in $rds ;
do
 python main_tableGAN.py ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=${default_optimiser} --lr=${default_lr} --batch_size=${default_bs} --random_dim=${rd}
done

echo "Varying the num_channels for ${default_optimiser} with lr ${default_lr}"
ncs="32 128" # default was 100
for nc in $ncs ;
do
 python main_tableGAN.py ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=${default_optimiser} --lr=${default_lr} --batch_size=${default_bs} --num_channels=${nc}
done
