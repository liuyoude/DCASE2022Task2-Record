import os
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from net import STgramMFN
from trainer import FTTester, Trainer
from dataset import ASDDataset

import utils


def main_compare(args):
    # random seed
    if args.seed: utils.setup_seed(args.seed)
    # set device
    cuda = torch.cuda.is_available()
    device_ids = args.device_ids
    args.dp = False
    if not cuda or device_ids is None:
        args.device = torch.device('cpu')
    else:
        args.device = torch.device(f'cuda:{int(device_ids[0])}')
        # if len(device_ids) > 1: args.dp = True
    # map
    args.att2idx, args.idx2att = utils.map_attribuate(args.root_dir)
    args.file_att_2_idx, args.idx_2_file_att = utils.map_file_attribute(args.root_dir)
    _, file_att_state = utils.get_file_attributes(args.root_dir)
    args.file_att_weights, args.samples_per_cls = utils.cal_file_att_weights(args.idx_2_file_att, file_att_state)
    args.samples_per_cls = None
    # args.file_att_weights, args.samples_per_cls = None, None
    args.num_classes, args.num_attributes = len(args.file_att_2_idx.keys()), len(args.att2idx.keys())
    args.logger.info(f'Num_classes: {args.num_classes}\tNum_attributes: {args.num_attributes}')
    # set model
    s_net = STgramMFN(num_classes=args.num_classes, num_attributes=args.num_attributes,
                      arcface=args.arcface, m=args.m, s=args.s, sub=args.sub)
    t_net = STgramMFN(num_classes=args.num_classes, num_attributes=args.num_attributes,
                      arcface=args.arcface, m=args.m, s=args.s, sub=args.sub)
    s_state_dict = torch.load(args.s_model_path, map_location=torch.device('cpu'))
    t_state_dict = torch.load(args.t_model_path, map_location=torch.device('cpu'))
    s_net.load_state_dict(s_state_dict['model'])
    t_net.load_state_dict(t_state_dict['model'])
    # if args.dp:
    #     net = nn.DataParallel(net, device_ids=args.device_ids)
    s_net = s_net.to(args.device)
    t_net = t_net.to(args.device)
    # tester
    tester = FTTester(args=args,
                      s_net=s_net,
                      t_net=t_net,
                      optimizer=None,
                      scheduler=None)
    tester.valid(valid_dirs=args.valid_dirs, save=args.save)


def main_mix(args):
    # random seed
    if args.seed: utils.setup_seed(args.seed)
    # set device
    cuda = torch.cuda.is_available()
    device_ids = args.device_ids
    args.dp = False
    if not cuda or device_ids is None:
        args.device = torch.device('cpu')
    else:
        args.device = torch.device(f'cuda:{int(device_ids[0])}')
        # if len(device_ids) > 1: args.dp = True
    # map
    args.att2idx, args.idx2att = utils.map_attribuate(args.root_dir)
    args.file_att_2_idx, args.idx_2_file_att = utils.map_file_attribute(args.root_dir)
    _, file_att_state = utils.get_file_attributes(args.root_dir)
    args.file_att_weights, args.samples_per_cls = utils.cal_file_att_weights(args.idx_2_file_att, file_att_state)
    args.samples_per_cls = None
    # args.file_att_weights, args.samples_per_cls = None, None
    args.num_classes, args.num_attributes = len(args.file_att_2_idx.keys()), len(args.att2idx.keys())
    args.logger.info(f'Num_classes: {args.num_classes}\tNum_attributes: {args.num_attributes}')
    # set model
    net = STgramMFN(num_classes=args.num_classes, num_attributes=args.num_attributes,
                    arcface=args.arcface, m=args.m, s=args.s, sub=args.sub)
    s_state_dict = torch.load(args.s_model_path, map_location=torch.device('cpu'))
    t_state_dict = torch.load(args.t_model_path, map_location=torch.device('cpu'))
    mix_state_dict = utils.mix_state_dict(s_state_dict['model'], t_state_dict['model'], alpha=args.alpha)
    net.load_state_dict(mix_state_dict)
    # if args.dp:
    #     net = nn.DataParallel(net, device_ids=args.device_ids)
    net = net.to(args.device)
    # tester
    tester = Trainer(args=args,
                     net=net,
                     optimizer=None,
                     scheduler=None)
    tester.valid(valid_dirs=args.valid_dirs, save=args.save)


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
    args.alpha = 0.2
    # utils.cal_statistic_data(args.train_dirs, sr=args.sr)
    args.model_version = '2022-05-01-16-STgram_MFN_FocalLoss(alpha=norm(w))'
    # args.s_model_path = f'./runs/' \
    #                     f'{args.model_version}-ft-source/' \
    #                     'model/best_checkpoint.pth.tar'
    args.s_model_path = f'../Classifier/runs/' \
                        f'{args.model_version}/' \
                        'model/best_checkpoint.pth.tar'
    args.t_model_path = f'./runs/' \
                        f'{args.model_version}-ft-target/' \
                        'model/best_checkpoint.pth.tar'

    # args.version = f'{args.model_version}-ft-Compare'
    args.version = f'{args.model_version}-ft-Mix-{args.alpha}'
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
    # main_compare(args)
    main_mix(args)


if __name__ == '__main__':
    run()
