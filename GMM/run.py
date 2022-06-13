import os
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from extractor import SpecExtractor
from gmmer import GMMer

import utils


def main(args):
    # random seed
    if args.seed: utils.setup_seed(args.seed)
    # gmm
    # transform = utils.Wave2Spec(sr=args.sr)
    transform = utils.Wave2Mel(sr=args.sr)
    extractor = SpecExtractor(args=args, dim=1, transform=transform)
    gmmer = GMMer(args=args, extractor=extractor)
    # gmmer.test(train_dirs=sorted(args.train_dirs),
    #            valid_dirs=sorted(args.valid_dirs),
    #            save=args.save,
    #            use_search=True)
    # gmmer.eval(train_dirs=sorted(args.train_dirs),
    #            test_dirs=sorted(args.test_dirs),
    #            use_search=True)
    # for gmm_n in range(1, 5):
    #     gmmer.smote_test(train_dirs=sorted(args.train_dirs),
    #                      valid_dirs=sorted(args.valid_dirs),
    #                      save=args.save,
    #                      s_gmm_n=gmm_n,
    #                      t_gmm_n=1,
    #                      use_search=False)
    gmmer.search_gwrp(train_dirs=sorted(args.train_dirs),
                      valid_dirs=sorted(args.valid_dirs),
                      save=False,
                      step=1,
                      s_gmm_n=3,
                      t_gmm_n=1,
                      use_smote=False)
    gmmer.search_gwrp(train_dirs=sorted(args.train_dirs),
                      valid_dirs=sorted(args.valid_dirs),
                      save=False,
                      step=1,
                      s_gmm_n=4,
                      t_gmm_n=1,
                      use_smote=False)


def main_gwrp(args):
    # random seed
    if args.seed: utils.setup_seed(args.seed)
    # gmm
    # transform = utils.Wave2Spec(sr=args.sr)
    transform = utils.Wave2Mel(sr=args.sr)
    extractor = SpecExtractor(args=args, dim=1, transform=transform)
    gmmer = GMMer(args=args, extractor=extractor)
    gmmer.test(train_dirs=sorted(args.train_dirs),
               valid_dirs=sorted(args.valid_dirs),
               save=args.save,
               use_search=True)
    gmmer.eval(train_dirs=sorted(args.train_dirs),
               test_dirs=sorted(args.test_dirs),
               use_search=True)
    # for gmm_n in range(1, 8):
    #     gmmer.test(train_dirs=sorted(args.train_dirs),
    #                valid_dirs=sorted(args.valid_dirs),
    #                save=args.save,
    #                s_gmm_n=gmm_n,
    #                t_gmm_n=1,
    #                use_search=False)
    # gmmer.search_gwrp(train_dirs=sorted(args.train_dirs),
    #                   valid_dirs=sorted(args.valid_dirs),
    #                   save=False,
    #                   s_gmm_n=1,
    #                   t_gmm_n=1)


def run():
    # init config parameters
    params = utils.load_yaml(file_path='./config.yaml')
    parser = argparse.ArgumentParser(description=params['description'])
    for key, value in params.items():
        parser.add_argument(f'--{key}', default=value, type=type(value))
    args = parser.parse_args()
    args.test = False
    args.arcface = False
    args.save = True

    # utils.cal_statistic_data(args.train_dirs, sr=args.sr)
    for pool_type in ['mean', 'max', 'mean+max', 'min', 'gwrp', 'meanCatmax', 'meanCatgwrp', 'maxCatgwrp'][4:5]:
        args.pool_type = pool_type
        # args.version = f'GMM-logMel-{pool_type}-SMOTE(0.2)'
        # args.version = f'GMM-logMel-{pool_type}-Mix-ensemble'
        args.version = f'GMM-logMel-{pool_type}-Mix'
        # init logger and writer
        # time_str = time.strftime('%Y-%m-%d-%H', time.localtime(time.time()))
        # args.version = f'{time_str}-{args.version}'
        log_dir = f'runs/{args.version}'
        writer = SummaryWriter(log_dir=log_dir)
        logger = utils.get_logger(filename=os.path.join(log_dir, 'running.log'))
        # save config file
        # utils.save_yaml_file(file_path=os.path.join(log_dir, 'config.yaml'), data=vars(args))
        # run
        args.writer = writer
        args.logger = logger
        args.logger.info(args)
        print(args.version, args.test)
        main(args)
        # main_gwrp(args)

if __name__ == '__main__':
    run()
