import warnings
from argparse import ArgumentParser, Namespace

import numpy as np
import pandas as pd
import torch

from data_processors.wgan.helpers import get_data, prepare_data_torch_scaling
from eval.eval import eval_synthetic_data
from synthetizers.WGAN.wgan import sample
from utils import set_seed
import logging

logger = logging.getLogger('MODEL_EVALUATION')
logger.setLevel(logging.INFO)

warnings.filterwarnings(action='ignore')
torch.set_printoptions(sci_mode=False)


def get_args():
    args = ArgumentParser()
    args.add_argument("use_case", type=str, help='use case', choices=['url', 'wids', 'lcld', 'botnet'])
    args.add_argument("path", type=str, help='path to the experiment dir')
    args.add_argument("--eval_epoch", default=None, type=int)
    args.add_argument("--seed", default=0, type=int)
    return args.parse_args()


if __name__ == '__main__':
    args = get_args()
    set_seed(args.seed)
    if args.eval_epoch is None:
        args.eval_epoch = ''
    else:
        args.eval_epoch = f'_{args.eval_epoch}'
    fh = logging.FileHandler(f'{args.path}/model_evaluation.log')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    print('Starting model evaluation..')

    # load model
    generator_model = torch.load(f'{args.path}/model{args.eval_epoch}.pt')
    dgm_model = Namespace(generator=generator_model)
    print(f'Loaded generator model: {args.path}/model{args.eval_epoch}.pt')

    # load data
    X_train, int_cols, bin_cols_idx = get_data(args.use_case, filepath=f"data/{args.use_case}/train_data.csv")
    X_test = pd.read_csv(f"data/{args.use_case}/test_data.csv")
    print(f'Loaded train and test data for: {args.use_case}')

    # scale train data
    X_train_scaled, scaler = prepare_data_torch_scaling(X_train, args.use_case, bin_cols_idx)
    print(f'Train data shape {X_train.shape}')
    print(f'Scaled train data shape {X_train_scaled.shape}')
    print(f'Test data shape {X_test.shape}')


    # sample train and test data
    for partition in ['train', 'test']:
        print(f'\nEvaluating for the {partition} partition')
        X_data = X_train if partition == 'train' else X_test
        sampled = sample(dgm_model, X_data.shape[0], X_train_scaled.shape[1])  # TODO: is the last arg correct even when partition is test?

        sampled = sampled.detach().numpy()
        sampled[:, int_cols] = np.trunc(sampled[:, int_cols])

        eval_synthetic_data(args.use_case, partition, X_data, sampled, X_data.columns.values.tolist(), log_wandb=False, logger=logger)

# PYTHONPATH=$(pwd) python scripts/model_evaluation.py url WGAN_out/url/unconstrained/400_70_10_10_5e-05_5e-05_constr-True_26-06-23--00-25-24/ --seed=21




