import os
import sys
import time

import scipy
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import librosa
import sklearn

from loss import ASDLoss
import utils


class Trainer:
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.net = kwargs['net']
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = self.args.writer
        self.logger = self.args.logger
        self.criterion = ASDLoss().to(self.args.device)
        self.wav2mel = utils.Wave2Mel(sr=self.args.sr, n_mels=self.args.n_mels)
        self.dims = self.args.n_frames * self.args.n_mels
        # self.num_patches, self.tdim, self.fdim = None, None, None
        # self.mean_std_dict = np.load('mean_std.npy', allow_pickle=True).item()
        self.csv_lines = []

    def train(self, train_loader, valid_dir):
        model_dir = os.path.join(self.writer.log_dir, 'model', self.args.machine)
        os.makedirs(model_dir, exist_ok=True)
        epochs = self.args.epochs
        valid_every_epochs = self.args.valid_every_epochs
        early_stop_epochs = self.args.early_stop_epochs
        num_steps = len(train_loader)
        self.sum_train_steps = 0
        self.sum_valid_steps = 0
        best_metric = 0
        no_better_epoch = 0
        for epoch in range(0, epochs + 1):
            # train
            sum_loss = 0
            self.net.train()
            train_bar = tqdm(train_loader, total=num_steps, desc=f'{self.args.machine}|Epoch-{epoch}')
            for x in train_bar:
                # forward
                b, n, f, m = x.shape
                x = x.reshape(b*n, f, m).float().to(self.args.device)
                center_idx = torch.zeros_like(x).bool()
                center_idx[:, 2, :] = True
                x_input = x[~center_idx].reshape(b*n, f-1, m)
                target = x[center_idx].reshape(b*n, 1, m)
                src_seq = torch.ones((x.size(0), 4), dtype=torch.int).to(self.args.device)
                src_len = (torch.ones((x.size(0), 1), dtype=torch.int) * 4).to(self.args.device)
                out, _ = self.net(src_seq, x_input, src_len, phase=None)
                loss = self.criterion(out, target)
                train_bar.set_postfix(loss=f'{loss.item():.3f}')
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # visualization
                self.writer.add_scalar(f'{self.args.machine}/train_loss', loss.item(), self.sum_train_steps)
                sum_loss += loss.item()
                self.sum_train_steps += 1
            avg_loss = sum_loss / num_steps
            if self.scheduler is not None and epoch >= self.args.start_scheduler_epoch:
                self.scheduler.step()
            self.logger.info(f'Epoch-{epoch}\tloss:{avg_loss:.5f}')
            # valid
            if epoch % valid_every_epochs == 0:
                metric, _ = self.valid(valid_dir, save=False)
                avg_auc_s, avg_auc_t, avg_pauc = metric['avg_auc_s'], metric['avg_auc_t'], metric['avg_pauc']
                self.writer.add_scalar(f'{self.args.machine}/auc_s', avg_auc_s, epoch)
                self.writer.add_scalar(f'{self.args.machine}/auc_t', avg_auc_t, epoch)
                self.writer.add_scalar(f'{self.args.machine}/pauc', avg_pauc, epoch)
                if avg_auc_s + avg_auc_t + avg_pauc >= best_metric:
                    no_better_epoch = 0
                    best_metric = avg_auc_s + avg_auc_t + avg_pauc
                    best_model_path = os.path.join(model_dir, 'best_checkpoint.pth.tar')
                    utils.save_model_state_dict(best_model_path, epoch=epoch,
                                                net=self.net.module if self.args.dp else self.net,
                                                optimizer=None)
                else:
                    # early stop
                    no_better_epoch += 1
                    if no_better_epoch > early_stop_epochs > 0: break
            # save last 10 epoch state dict
            # if epochs - epoch <= 10:
            #     model_path = os.path.join(model_dir, f'{epoch}_checkpoint.pth.tar')
            #     utils.save_model_state_dict(model_path, epoch=epoch,
            #                                 net=self.net.module if self.args.dp else self.net,
            #                                 optimizer=None)

    def valid(self, valid_dir, save=True, result_dir=None, csv_lines=[]):
        net = self.net.module if self.args.dp else self.net
        net.eval()

        metric = {}
        # csv_lines = []
        # sum_auc_s, sum_auc_t, sum_pauc, num, total_time = 0, 0, 0, 0, 0
        # h_sum_auc_s, h_sum_auc_t, h_sum_pauc = 0, 0, 0
        print('\n' + '=' * 20)
        result_dir = result_dir if result_dir else os.path.join('./results', self.args.version)
        os.makedirs(result_dir, exist_ok=True)
        target_dir = valid_dir
        start = time.perf_counter()
        machine_type = target_dir.split('/')[-2]
        # print(target_dir, target_dir.split('/'), machine_type)
        # get machine list
        machine_section_list = utils.get_machine_section_list(target_dir)
        csv_lines.append([machine_type])
        csv_lines.append(['section', 'AUC(Source)', 'AUC(Target)', 'pAUC'])
        performance = []
        for section_str in machine_section_list:
            csv_path = os.path.join(result_dir, f'anomaly_score_{machine_type}_{section_str}.csv')
            test_files, y_true, domain_list = utils.get_valid_file_list(target_dir, section_str)
            y_pred = [0. for _ in test_files]
            anomaly_score_list = []
            for file_idx, file_path in enumerate(test_files):
                x = self.transform(file_path)
                b, f, m = x.shape
                center_idx = torch.zeros_like(x).bool()
                center_idx[:, 2, :] = True
                x_input = x[~center_idx].reshape(b, f-1, m)
                target = x[center_idx].reshape(b, 1, m)
                src_seq = torch.ones((x.size(0), 4), dtype=torch.int).to(self.args.device)
                src_len = (torch.ones((x.size(0), 1), dtype=torch.int) * 4).to(self.args.device)
                with torch.no_grad():
                    out, _ = net(src_seq, x_input, src_len, phase=None)
                y_pred[file_idx] = self.criterion(out, target).item()
                anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])
            if save: utils.save_csv(csv_path, anomaly_score_list)
            # compute auc and pAuc
            auc_s, auc_t, p_auc = utils.cal_auc_pauc(y_true, y_pred, domain_list)
            performance.append([auc_s, auc_t, p_auc])
            csv_lines.append([section_str.split('_', 1)[1], auc_s, auc_t, p_auc])
        # calculate averages for AUCs and pAUCs
        amean_performance = np.mean(np.array(performance, dtype=float), axis=0)
        hmean_performance = scipy.stats.hmean(
            np.maximum(np.array(performance, dtype=float), sys.float_info.epsilon), axis=0)
        mean_auc_s, mean_auc_t, mean_p_auc = amean_performance[0], amean_performance[1], amean_performance[2]
        h_mean_auc_s, h_mean_auc_t, h_mean_p_auc = hmean_performance[0], hmean_performance[1], hmean_performance[2]
        # print(machine_type, 'AUC_clf:', mean_auc, 'pAUC_clf:', mean_p_auc)
        time_nedded = time.perf_counter() - start
        csv_lines.append(["arithmetic mean"] + list(amean_performance))
        csv_lines.append(["harmonic mean"] + list(hmean_performance))
        csv_lines.append([])
        self.logger.info(f'Test {machine_type}\tcost {time_nedded:.2f} sec\tavg_auc_s: {mean_auc_s:.3f}\t'
                         f'avg_auc_t: {mean_auc_t:.3f}\tavg_pauc: {mean_p_auc:.3f}')

        print(f'Test time: {time_nedded:.2f} sec')
        metric['avg_auc_s'], metric['avg_auc_t'], metric['avg_pauc'] = mean_auc_s, mean_auc_t, mean_p_auc
        metric['havg_auc_s'], metric['havg_auc_t'], metric['havg_pauc'] = h_mean_auc_s, h_mean_auc_t, h_mean_p_auc
        self.logger.info(f'avg_auc_s: {mean_auc_s:.3f}\tavg_auc_t: {mean_auc_t:.3f}\tavg_pauc: {mean_p_auc:.3f}')
        return metric, csv_lines

    # 需要重写
    def eval(self, test_dirs):
        self.net.eval()
        net = self.net.module if self.args.dp else self.net

        result_dir = os.path.join('./evaluator/teams', self.args.version)
        os.makedirs(result_dir, exist_ok=True)
        print('\n' + '=' * 20)
        for index, target_dir in enumerate(sorted(test_dirs)):
            machine_type = target_dir.split('/')[-2]
            # get machine list
            machine_id_list = utils.get_machine_id_list(target_dir)
            for id_str in machine_id_list:
                test_files = utils.get_eval_file_list(target_dir, id_str)
                csv_path = os.path.join(result_dir, f'anomaly_score_{machine_type}_{id_str}.csv')
                anomaly_score_list = []
                y_pred = [0. for _ in test_files]
                for file_idx, file_path in enumerate(test_files):
                    inputs, label = self.transform(file_path)
                    with torch.no_grad():
                        predict_ids, feature = net(*inputs, label)
                    probs = - torch.log_softmax(predict_ids, dim=1).mean(dim=0).squeeze().cpu().numpy()
                    y_pred[file_idx] = probs[label]
                    anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])
                    utils.save_csv(csv_path, anomaly_score_list)

    def transform(self, filename):
        # label, one_hot = utils.get_label('/'.join(filename.split('/')[-3:]), self.args.att2idx,
        #                                  self.args.file_att_2_idx)
        (x, _) = librosa.core.load(filename, sr=self.args.sr, mono=True)
        x = x[:self.args.sr * 10]  # (1, audio_length)
        x = torch.from_numpy(x)
        x_mel = self.wav2mel(x).squeeze()
        n_vectors = x_mel.shape[1] - self.args.n_frames + 1
        vectors = torch.zeros((n_vectors, self.dims))
        for t in range(self.args.n_frames):
            vectors[:, self.args.n_mels * t: self.args.n_mels * (t + 1)] = x_mel[:, t: t + n_vectors].T
        vectors = vectors[:: self.args.n_hop_frames, :]
        vectors = vectors.reshape(-1, self.args.n_mels, self.args.n_frames).transpose(2, 1)
        vectors = vectors.float().to(self.args.device)
        return vectors


if __name__ == '__main__':
    trainer = Trainer()
    data = torch.rand([10, 128, 313])
    print(trainer.data2patches(data).shape)
