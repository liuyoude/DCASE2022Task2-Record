import os
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformer.module import TransAE
from trainer import Trainer
from dataset import ASDDataset

import utils


def main(args):
    # utils.save_statistic_data(args.train_dirs, sr=args.sr)
    utils.setup_seed(args.seed)
    # set device
    cuda = torch.cuda.is_available()
    device_ids = args.device_ids
    args.dp = False
    if not cuda or device_ids is None:
        args.device = torch.device('cpu')
    else:
        args.device = torch.device(f'cuda:{device_ids[0]}')
        if len(device_ids) > 1: args.dp = True
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
    #                                                        T_max=args.epochs-args.start_scheduler_epoch,
    #                                                        eta_min=1e-4)
    # trainer
    csv_lines = []
    metrics = {'avg_auc_s': [], 'avg_auc_t': [], 'avg_pauc': [],
               'havg_auc_s': [], 'havg_auc_t': [], 'havg_pauc': []}
    idx = 0
    for train_dir, add_train_dir, valid_dir in zip(sorted(args.train_dirs), sorted(args.add_train_dirs),
                                                   sorted(args.valid_dirs)):
        idx += 1
        # if idx == 1 or idx == 2: continue
        # set model
        net = TransAE(max_seq_len=4,
                      frames=args.n_frames, n_mels=args.n_mels, n_fft=513,
                      lpe=False, cfp=True)
        if args.dp:
            net = nn.DataParallel(net, device_ids=args.device_ids)
        net = net.to(args.device)
        # optimizer & scheduler
        optimizer = torch.optim.Adam(net.parameters(), lr=float(args.lr))
        scheduler = None
        # set trainer
        trainer = Trainer(args=args,
                          net=net,
                          optimizer=optimizer,
                          scheduler=scheduler)
        args.machine = train_dir.split('/')[-2]
        # load data
        train_dataset = ASDDataset(args, [train_dir])
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                      shuffle=True, num_workers=args.num_workers, drop_last=True)
        # train model
        if not args.test:
            trainer.train(train_dataloader, valid_dir=valid_dir)
        # test model
        best_model_path = os.path.join(args.writer.log_dir, 'model', args.machine, 'best_checkpoint.pth.tar')
        if args.dp:
            trainer.net.module.load_state_dict(torch.load(best_model_path)['model'])
        else:
            trainer.net.load_state_dict(torch.load(best_model_path)['model'])
        metric, csv_lines = trainer.valid(valid_dir=valid_dir, save=True, csv_lines=csv_lines)
        for key in metrics.keys():
            metrics[key].append(metric[key])
        # trainer.eval(test_dirs=args.test_dirs)
    avg_auc_s, avg_auc_t, avg_pauc = np.mean(metrics['avg_auc_s']), np.mean(metrics['avg_auc_t']), np.mean(
        metrics['avg_pauc'])
    havg_auc_s, havg_auc_t, havg_pauc = np.mean(metrics['havg_auc_s']), np.mean(metrics['havg_auc_t']), np.mean(
        metrics['havg_pauc'])
    csv_lines.append(['(A)Total Average', f'{avg_auc_s:.4f}', f'{avg_auc_t:.4f}', f'{avg_pauc:.4f}'])
    csv_lines.append(['(H)Total Average', f'{havg_auc_s:.4f}', f'{havg_auc_t:.4f}', f'{havg_pauc:.4f}'])
    result_path = os.path.join('./results', f'{args.version}', 'result.csv')
    utils.save_csv(result_path, csv_lines)


def run():
    # init config parameters
    params = utils.load_yaml(file_path='./config.yaml')
    parser = argparse.ArgumentParser(description=params['description'])
    for key, value in params.items():
        parser.add_argument(f'--{key}', default=value, type=type(value))
    args = parser.parse_args()
    args.version = 'TransAE-noAdd'
    args.test = False
    # init logger and writer
    time_str = time.strftime('%Y-%m-%d-%H', time.localtime(time.time()))
    args.version = f'{time_str}-{args.version}'
    log_dir = f'runs/{args.version}'
    if args.test:
        args.version = f'2022-05-26-17-TransAE'
        log_dir = f'runs/{args.version}'
    # log_dir = 'runs/2022-03-16-15-ast_base384_Sgram_CELoss'
    writer = SummaryWriter(log_dir=log_dir)
    logger = utils.get_logger(filename=os.path.join(log_dir, 'running.log'))
    # save config file
    utils.save_yaml_file(file_path=os.path.join(log_dir, 'config.yaml'), data=vars(args))
    # run
    args.writer = writer
    args.logger = logger
    args.logger.info(args)
    print(args.version)
    main(args)


if __name__ == '__main__':
    run()