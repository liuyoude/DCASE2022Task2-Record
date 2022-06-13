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
        self.criterion = ASDLoss(num_classes=self.args.num_classes,
                                 alpha=self.args.file_att_weights, samples_per_cls=self.args.samples_per_cls).to(self.args.device)
        self.wav2mel = utils.Wave2Mel(sr=self.args.sr)

    def train(self, train_loader, valid_dirs):
        model_dir = os.path.join(self.writer.log_dir, 'model')
        os.makedirs(model_dir, exist_ok=True)
        epochs = self.args.epochs
        valid_every_epochs = self.args.valid_every_epochs
        early_stop_epochs = self.args.early_stop_epochs
        num_steps = len(train_loader)
        self.sum_train_steps = 0
        self.sum_valid_steps = 0
        best_metric = 0
        no_better_epoch = 0
        for epoch in range(0, epochs+1):
            # train
            sum_loss = 0
            self.net.train()
            train_bar = tqdm(train_loader, total=num_steps, desc=f'Epoch-{epoch}')
            for (x_wavs, x_mels, labels, one_hots) in train_bar:
                # forward
                x_wavs, x_mels = x_wavs.float().to(self.args.device), x_mels.float().to(self.args.device)
                labels, one_hots = labels.long().to(self.args.device), one_hots.float().to(self.args.device)
                out_classes, out_atts, _ = self.net(x_wavs, x_mels, labels)
                loss, loss1, loss2 = self.criterion((out_classes, out_atts), (labels, one_hots))
                train_bar.set_postfix(loss=f'{loss.item()}')
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # visualization
                self.writer.add_scalar('loss/train_loss', loss.item(), self.sum_train_steps)
                self.writer.add_scalar('loss/ce_loss', loss1.item(), self.sum_train_steps)
                self.writer.add_scalar('loss/bce_loss', loss2.item(), self.sum_train_steps)
                sum_loss += loss.item()
                self.sum_train_steps += 1
            avg_loss = sum_loss / num_steps
            if self.scheduler is not None and epoch >= self.args.start_scheduler_epoch:
                self.scheduler.step()
            self.logger.info(f'Epoch-{epoch}\tloss:{avg_loss:.5f}')
            # valid
            if epoch % valid_every_epochs == 0:
                metric = self.valid(valid_dirs, save=False)
                avg_auc_s, avg_auc_t, avg_pauc = metric['avg_auc_s'], metric['avg_auc_t'], metric['avg_pauc']
                self.writer.add_scalar('metric/auc_s', avg_auc_s, epoch)
                self.writer.add_scalar('metric/auc_t', avg_auc_t, epoch)
                self.writer.add_scalar('metric/pauc', avg_pauc, epoch)
                sum_auc_pauc = avg_auc_s + avg_auc_t + avg_pauc
                if sum_auc_pauc >= best_metric:
                    no_better_epoch = 0
                    best_metric = sum_auc_pauc
                    best_model_path = os.path.join(model_dir, 'best_checkpoint.pth.tar')
                    utils.save_model_state_dict(best_model_path, epoch=epoch,
                                                net=self.net.module if self.args.dp else self.net,
                                                optimizer=None)
                else:
                    # early stop
                    no_better_epoch += 1
                    if no_better_epoch > early_stop_epochs > 0: break
            # save last 10 epoch state dict
            # if epoch > 150: break
            if epochs - epoch <= 20:
                model_path = os.path.join(model_dir, f'{epoch}_checkpoint.pth.tar')
                utils.save_model_state_dict(model_path, epoch=epoch,
                                            net=self.net.module if self.args.dp else self.net,
                                            optimizer=None)

    def valid(self, valid_dirs, save=True, result_dir=None, decision=False):
        net = self.net.module if self.args.dp else self.net
        net.eval()

        metric = {}
        csv_lines = []
        sum_auc_s, sum_auc_t, sum_pauc, num, total_time = 0, 0, 0, 0, 0
        h_sum_auc_s, h_sum_auc_t, h_sum_pauc = 0, 0, 0
        print('\n' + '=' * 20)
        result_dir = result_dir if result_dir else os.path.join('./results', self.args.version)
        os.makedirs(result_dir, exist_ok=True)
        for index, target_dir in enumerate(sorted(valid_dirs)):
            start = time.perf_counter()
            machine_type = target_dir.split('/')[-2]
            # print(target_dir, target_dir.split('/'), machine_type)
            # get machine list
            machine_section_list = utils.get_machine_section_list(target_dir)
            csv_lines.append([machine_type])
            csv_lines.append(['section', 'AUC(Source)', 'AUC(Target)', 'pAUC'])
            performance = []
            if decision:
                train_scores = []
                for section_str in machine_section_list:
                    train_dir = sorted(self.args.train_dirs)[index]
                    machine_section_file_atts, _ = utils.get_file_attributes(train_dir, machine=machine_type,
                                                                             section=section_str)
                    train_files = utils.get_filename_list(train_dir, pattern=f'{section_str}_*')
                    for train_file in train_files:
                        inputs, label, one_hot = self.transform(train_file)
                        with torch.no_grad():
                            out_classes, out_atts, _ = net(*inputs, label)
                        probs = torch.softmax(out_classes, dim=1).mean(dim=0).squeeze().cpu().numpy()
                        score = utils.cal_anomaly_score(probs, machine_section_file_atts,
                                                               self.args.file_att_2_idx)
                        train_scores.append(score)
                max_score = np.max(train_scores)
                min_score = np.min(train_scores)
                shape_hat, loc_hat, scale_hat = scipy.stats.gamma.fit(train_scores)
                decision_threshold = scipy.stats.gamma.ppf(q=self.args.decision_threshold, a=shape_hat, loc=loc_hat,
                                                           scale=scale_hat)
            for section_str in machine_section_list:
                csv_path = os.path.join(result_dir, f'anomaly_score_{machine_type}_{section_str}.csv')
                machine_section_file_atts, _ = utils.get_file_attributes(target_dir, machine=machine_type, section=section_str)
                test_files, y_true, domain_list = utils.get_valid_file_list(target_dir, section_str)
                y_pred = [0. for _ in test_files]
                anomaly_score_list = []
                if decision:
                    decision_path = os.path.join(result_dir, f'decision_result_{machine_type}_{section_str}.csv')
                    decision_result_list = [['decision threshold', decision_threshold, 'max', max_score, 'min', min_score]]
                for file_idx, file_path in enumerate(test_files):
                    inputs, label, one_hot = self.transform(file_path)
                    with torch.no_grad():
                        out_classes, out_atts, _ = net(*inputs, label)
                    probs = torch.softmax(out_classes, dim=1).mean(dim=0).squeeze().cpu().numpy()
                    y_pred[file_idx] = utils.cal_anomaly_score(probs, machine_section_file_atts,
                                                               self.args.file_att_2_idx)
                    # att_probs = torch.sigmoid(out_atts).mean(dim=0).squeeze().cpu().numpy()
                    # y_pred[file_idx] = utils.cal_anomaly_score_att(probs, att_probs, machine_section_file_atts,
                    #                                                self.args.file_att_2_idx, self.args.att2idx)
                    anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])
                    if decision:
                        if y_pred[file_idx] > decision_threshold:
                            decision_result_list.append([os.path.basename(file_path), y_pred[file_idx], 1])
                        else:
                            decision_result_list.append([os.path.basename(file_path), y_pred[file_idx], 0])
                if save: utils.save_csv(csv_path, anomaly_score_list)
                if decision: utils.save_csv(decision_path, decision_result_list)
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
            metric[machine_type] = {
                'avg_auc_s': mean_auc_s, 'avg_auc_t': mean_auc_t, 'avg_pauc': mean_p_auc,
            }
            # print(machine_type, 'AUC_clf:', mean_auc, 'pAUC_clf:', mean_p_auc)
            sum_auc_s += mean_auc_s
            sum_auc_t += mean_auc_t
            sum_pauc += mean_p_auc
            h_sum_auc_s += h_mean_auc_s
            h_sum_auc_t += h_mean_auc_t
            h_sum_pauc += h_mean_p_auc
            num += 1
            time_nedded = time.perf_counter() - start
            total_time += time_nedded
            csv_lines.append(["arithmetic mean"] + list(amean_performance))
            csv_lines.append(["harmonic mean"] + list(hmean_performance))
            csv_lines.append([])
            self.logger.info(f'Test {machine_type}\tcost {time_nedded:.2f} sec\tavg_auc_s: {mean_auc_s:.3f}\t'
                             f'avg_auc_t: {mean_auc_t:.3f}\tavg_pauc: {mean_p_auc:.3f}')
        print(f'Total test time: {total_time:.2f} sec')
        result_path = os.path.join(result_dir, 'result.csv')
        avg_auc_s, avg_auc_t, avg_pauc = sum_auc_s / num, sum_auc_t / num, sum_pauc / num
        h_avg_auc_s, h_avg_auc_t, h_avg_pauc = h_sum_auc_s / num, h_sum_auc_t / num, h_sum_pauc / num
        metric['avg_auc_s'], metric['avg_auc_t'], metric['avg_pauc'] = h_avg_auc_s, h_avg_auc_t, h_avg_pauc
        csv_lines.append(['(A)Total Average', f'{avg_auc_s:.4f}', f'{avg_auc_t:.4f}', f'{avg_pauc:.4f}'])
        csv_lines.append(['(H)Total Average', f'{h_avg_auc_s:.4f}', f'{h_avg_auc_t:.4f}', f'{h_avg_pauc:.4f}'])
        if save: utils.save_csv(result_path, csv_lines)
        self.logger.info(f'avg_auc_s: {avg_auc_s:.3f}\tavg_auc_t: {avg_auc_t:.3f}\tavg_pauc: {avg_pauc:.3f}')
        return metric

    def eval(self, train_dirs, test_dirs):
        net = self.net.module if self.args.dp else self.net
        net.eval()
        result_dir = os.path.join('./evaluator/teams', self.args.version)
        train_result_dir = os.path.join('./evaluator/trains', self.args.version)
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(train_result_dir, exist_ok=True)
        print('\n' + '=' * 20)
        for index, (target_dir, train_dir) in enumerate(zip(sorted(test_dirs), sorted(train_dirs[7:]))):
            machine_type = target_dir.split('/')[-2]
            # get machine list
            machine_section_list = utils.get_machine_section_list(target_dir)
            for section_str in machine_section_list:
                test_files = utils.get_eval_file_list(target_dir, section_str)
                train_files = utils.get_eval_file_list(train_dir, section_str)
                csv_path = os.path.join(result_dir, f'anomaly_score_{machine_type}_{section_str}.csv')
                train_csv_path = os.path.join(train_result_dir, f'anomaly_score_{machine_type}_{section_str}.csv')
                machine_section_file_atts, _ = utils.get_file_attributes(train_dir, machine=machine_type,
                                                                         section=section_str)
                print('create test anomaly score files...')
                anomaly_score_list = []
                y_pred = [0. for _ in test_files]
                for file_idx, file_path in enumerate(test_files):
                    inputs, label, one_hot = self.transform(file_path, label_needed=False)
                    with torch.no_grad():
                        out_classes, out_atts, _ = net(*inputs, label)
                    probs = torch.softmax(out_classes, dim=1).mean(dim=0).squeeze().cpu().numpy()
                    y_pred[file_idx] = utils.cal_anomaly_score(probs, machine_section_file_atts,
                                                               self.args.file_att_2_idx)
                    anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])
                    utils.save_csv(csv_path, anomaly_score_list)
                print('record train files score for computing threshold...')
                anomaly_score_list = []
                y_pred = [0. for _ in train_files]
                for file_idx, file_path in enumerate(train_files):
                    inputs, label, one_hot = self.transform(file_path)
                    with torch.no_grad():
                        out_classes, out_atts, _ = net(*inputs, label)
                    probs = torch.softmax(out_classes, dim=1).mean(dim=0).squeeze().cpu().numpy()
                    y_pred[file_idx] = utils.cal_anomaly_score(probs, machine_section_file_atts,
                                                               self.args.file_att_2_idx)
                    anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])
                    utils.save_csv(train_csv_path, anomaly_score_list)

    def transform(self, filename, label_needed=True):
        if label_needed:
            label, one_hot = utils.get_label('/'.join(filename.split('/')[-3:]), self.args.att2idx, self.args.file_att_2_idx)
        else:
            label, one_hot = None, None
        (x, _) = librosa.core.load(filename, sr=self.args.sr, mono=True)
        x = x[:self.args.sr * 10]  # (1, audio_length)
        x = torch.from_numpy(x)
        x_mel = self.wav2mel(x)
        x_mel = utils.normalize(x_mel, mean=self.args.mean, std=self.args.std)

        label = torch.from_numpy(np.array(label)).long().to(self.args.device) if label else None
        x = x.unsqueeze(0).float().to(self.args.device)
        x_mel = x_mel.unsqueeze(0).float().to(self.args.device)
        return (x, x_mel), label, one_hot


