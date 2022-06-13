import os
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from net import STgramMFN
from trainer import Trainer
from dataset import ASDDataset

import utils


def select_k(args, trainer):
    print('Select k for calculating anomaly score...')
    root_dir = os.path.join('./results', args.version)
    csv_path = os.path.join(root_dir, 'result-k.csv')
    csv_lines = []
    csv_lines.append(['k', 'auc_s', 'auc_t', 'pauc'])
    best_metric, best_k = 0, 0
    for k in range(1, 17):
        result_dir = os.path.join(root_dir, f'k={k}')
        metric = trainer.valid(valid_dirs=args.valid_dirs, result_dir=result_dir, k=k)
        auc_s, auc_t, pauc = metric['avg_auc_s'], metric['avg_auc_t'], metric['avg_pauc']
        sum_auc_pauc = auc_s + auc_t + pauc
        if sum_auc_pauc > best_metric:
            best_metric = sum_auc_pauc
            best_k = k
        csv_lines.append([k, f'{auc_s:.4f}', f'{auc_t:.4f}', f'{pauc:.4f}'])
        print([k, f'{auc_s:.4f}', f'{auc_t:.4f}', f'{pauc:.4f}'])
    csv_lines.append([])
    csv_lines.append(['Best k', best_k])
    print(['Best k', best_k])
    utils.save_csv(csv_path, csv_lines)


def main(args):
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
        if len(device_ids) > 1: args.dp = True
    # map
    args.att2idx, args.idx2att = utils.map_attribuate(args.root_dir)
    args.file_att_2_idx, args.idx_2_file_att = utils.map_file_attribute(args.root_dir)
    _, file_att_state = utils.get_file_attributes(args.root_dir)
    args.file_att_weights, args.samples_per_cls = utils.cal_file_att_weights(args.idx_2_file_att, file_att_state)
    args.samples_per_cls = None
    # args.file_att_weights, args.samples_per_cls = None, None
    args.num_classes, args.num_attributes = len(args.file_att_2_idx.keys()), len(args.att2idx.keys())
    args.logger.info(f'Num_classes: {args.num_classes}\tNum_attributes: {args.num_attributes}')
    # load data
    train_dataset = ASDDataset(args.train_dirs, args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers, drop_last=True)
    # set model
    net = STgramMFN(num_classes=args.num_classes, num_attributes=args.num_attributes,
                    arcface=args.arcface, m=args.m, s=args.s, sub=args.sub)
    state_dict = torch.load(args.model_path, map_location=torch.device('cpu'))
    load_epoch = state_dict['epoch']
    args.logger.info(f'load epoch: {load_epoch}')
    net.load_state_dict(state_dict['model'])
    if args.dp:
        net = nn.DataParallel(net, device_ids=args.device_ids)
    net = net.to(args.device)
    # optimizer & scheduler
    optimizer = torch.optim.Adam(net.parameters(), lr=float(args.lr))
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.start_scheduler_epoch,
    #                                                        eta_min=1e-5)
    scheduler = None
    # trainer
    trainer = Trainer(args=args,
                      net=net,
                      optimizer=optimizer,
                      scheduler=scheduler)
    # train model
    if not args.test:
        trainer.train(train_dataloader, valid_dirs=args.valid_dirs)
    # test model
    best_model_path = os.path.join(args.writer.log_dir, 'model', 'best_checkpoint.pth.tar')
    state_dict = torch.load(best_model_path)
    load_epoch = state_dict['epoch']
    args.logger.info(f'load epoch: {load_epoch}')
    if args.dp:
        trainer.net.module.load_state_dict(state_dict['model'])
    else:
        trainer.net.load_state_dict(state_dict['model'])
    trainer.valid(valid_dirs=args.valid_dirs, save=args.save)
    # trainer.valid(valid_dirs=args.valid_dirs, save=True, result_dir=f'./results/{args.version}/score=fileAtt_probs')
    # trainer.eval(test_dirs=args.test_dirs)
    # select_k(args, trainer)


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
    model_version = '2022-05-22-16-STgram_MFN_Focalloss(alpha=norm(w))_addSecAtt'
    args.model_path = f'../Classifier/runs/' \
                      f'{model_version}/' \
                      'model/best_checkpoint.pth.tar'
    for domain in ['target', 'source']:
        args.domain = domain
        if domain == 'source': args.batch_size = 154
        else: args.batch_size = 16
        args.version = f'{model_version}-ft-{domain}'
        log_dir = f'runs/{args.version}'
        if args.test:
            args.version = f'2022-05-23-20-STgram_MFN_Focalloss(alpha=norm(w))_addSecAttInTarget'
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
