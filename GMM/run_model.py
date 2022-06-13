import os
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from extractor import STgramMFNExtractor
from gmmer import GMMer

import utils



def main(args):
    # random seed
    if args.seed: utils.setup_seed(args.seed)
    # get info
    args.att2idx, args.idx2att = utils.map_attribuate(args.root_dir)
    args.file_att_2_idx, args.idx_2_file_att = utils.map_file_attribute(args.root_dir)
    args.num_classes, args.num_attributes = len(args.file_att_2_idx.keys()), len(args.att2idx.keys())
    args.logger.info(f'Num_classes: {args.num_classes}\tNum_attributes: {args.num_attributes}')
    # gmm
    transform = utils.Wave2Mel(sr=args.sr)
    extractor = STgramMFNExtractor(args=args, transform=transform)
    gmmer = GMMer(args=args, extractor=extractor)
    for gmm_n in [1, 2, 3, 4, 5, 6, 7, 8]:
        gmmer.test(train_dirs=sorted(args.train_dirs),
                   valid_dirs=sorted(args.valid_dirs),
                   save=args.save,
                   s_gmm_n=gmm_n,
                   t_gmm_n=1)

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
    args.pool_type = None
    # utils.cal_statistic_data(args.train_dirs, sr=args.sr)
    # model_version = '2022-05-24-12-STgram_MFN_Focalloss(alpha=norm(w))_addDomainAtt'
    model_version = '2022-06-02-14-STgram_MFN_ASLloss(alpha=norm(w),pos=2,neg=2,eps=0.1)_addDoaminAtt'
    args.version = f'GMM-{model_version}'
    args.model_path = f'../Classifier/runs/' \
                      f'{model_version}/' \
                      'model/best_checkpoint.pth.tar'
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


if __name__ == '__main__':
    run()
