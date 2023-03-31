#!/usr/bin/env python
# encoding: utf-8
'''
@author: Xu Yan
@file: robust_test.py
@time: 2022/10/7 21:24
'''

import os
import yaml
import torch
import datetime
import importlib
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from pathlib import Path
from easydict import EasyDict
from argparse import ArgumentParser
from dataloader.corruption_dataset import SemanticKITTIC
from dataloader.dataset import get_model_class, get_collate_class
import warnings

warnings.filterwarnings("ignore")


def load_yaml(file_name):
    with open(file_name, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)
    return config


def parse_config():
    parser = ArgumentParser()
    # general
    parser.add_argument('--gpu', type=int, nargs='+', default=(0,), help='specify gpu devices')
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--config_path', default='config/semantickitti/2dpass-semantickitti.yaml')
    # testing
    parser.add_argument('--num_vote', type=int, default=12, help='number of voting in the test')
    parser.add_argument('--checkpoint', type=str, default=None, help='load checkpoint')
    # debug
    parser.add_argument('--save_prediction', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')

    args = parser.parse_args()
    config = load_yaml(args.config_path)
    config.update(vars(args))  # override the configuration using the value in args

    # voting test
    config['dataset_params']['val_data_loader']['batch_size'] = args.num_vote
    config['baseline_only'] = False
    config['submit_to_server'] = args.save_prediction
    config['test'] = True

    if args.num_vote > 1:
        config['dataset_params']['val_data_loader']['rotate_aug'] = True
        config['dataset_params']['val_data_loader']['transform_aug'] = True
    if args.debug:
        config['dataset_params']['val_data_loader']['batch_size'] = 2
        config['dataset_params']['val_data_loader']['num_workers'] = 0

    return EasyDict(config)


def build_loader(config, corruption):
    pc_dataset = SemanticKITTIC
    # dataset_type = point_dataset
    dataset_type = get_model_class(config['dataset_params']['dataset_type'])
    val_config = config['dataset_params']['val_data_loader']

    test_pt_dataset = pc_dataset(
        config,
        data_path=val_config['data_path'],
        corruption=corruption,
        num_vote=val_config["batch_size"]
    )
    test_dataset_loader = torch.utils.data.DataLoader(
        dataset=dataset_type(test_pt_dataset, config, val_config, num_vote=val_config["batch_size"]),
        batch_size=val_config["batch_size"],
        collate_fn=get_collate_class(config['dataset_params']['collate_type']),
        shuffle=False,
        num_workers=val_config["num_workers"]
    )
    return test_dataset_loader


if __name__ == '__main__':
    # parameters
    configs = parse_config()
    print(configs)

    # corruption dataset
    with open('config/corruption/semantickittic.yaml', 'r') as stream:
        corruption = yaml.safe_load(stream)
    print(corruption)

    save_path = os.path.join(Path(configs.checkpoint).parent,
                             'robust_test_' + str(datetime.datetime.now().strftime('%Y-%m-%d')))
    os.makedirs(save_path, exist_ok=True)

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, configs.gpu))
    num_gpu = len(configs.gpu)
    assert num_gpu == 1, 'multi-GPU testing is not available!'

    # reproducibility
    torch.manual_seed(configs.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(configs.seed)
    config_path = configs.config_path
    results_dict = {}

    model_file = importlib.import_module('network.' + configs['model_params']['model_architecture'])
    my_model = model_file.get_model(configs)

    pl.seed_everything(configs.seed)
    my_model = my_model.load_from_checkpoint(configs.checkpoint, config=configs)

    for idx, cor in enumerate(corruption['corruption_name']):
        print('[{}/{}] Start robust testing for {}...'.format(idx + 1, len(corruption['corruption_name']) + 1, cor))
        test_dataset_loader = build_loader(configs, cor)

        tester = pl.Trainer(
            gpus=[i for i in range(num_gpu)],
            accelerator='ddp',
            resume_from_checkpoint=configs.checkpoint
        )
        results = tester.test(my_model, test_dataset_loader)
        results_dict[cor] = [results[0]['val/mIoU'], results[0]['val/acc']]

    df = pd.DataFrame(results_dict)
    df.index = ['val/mIoU', 'val/acc']
    print(df.T)
    df.T.to_csv(os.path.join(save_path, 'summary.csv'))
