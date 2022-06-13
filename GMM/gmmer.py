import logging
import os
import sys
import sklearn
import numpy as np
import time
import re
import joblib

import torch
import librosa
import scipy
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# import spafe.fbanks.gammatone_fbanks as gf
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
# from torch.utils.tensorboard import SummaryWriter
from visdom import Visdom
from tqdm import tqdm
import utils


# torch.manual_seed(666)


class GMMer(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.feature_extractor = kwargs['extractor']
        self.wav2mel = utils.Wave2Mel(sr=self.args.sr)
        self.logger = self.args.logger
        self.csv_lines = []

    def fit_GMM(self, data, n_components, means_init=None, precisions_init=None):
        gmm = GaussianMixture(n_components=n_components, covariance_type='full',
                              means_init=means_init, precisions_init=precisions_init,
                              tol=1e-6, reg_covar=1e-6, verbose=2)
        gmm.fit(data)
        return gmm

    def test(self, train_dirs, valid_dirs, save=True, s_gmm_n=2, t_gmm_n=2, use_search=False, use_smote=False):
        csv_lines = []
        sum_auc_s, sum_auc_t, sum_pauc, num, total_time = 0, 0, 0, 0, 0
        h_sum_auc_s, h_sum_auc_t, h_sum_pauc = 0, 0, 0
        result_dir = os.path.join('./results', self.args.version, f'GMM-{s_gmm_n}') if not use_search else \
            os.path.join('./results', self.args.version, f'GMM-Mix-research')
        os.makedirs(result_dir, exist_ok=True)
        print('\n' + '=' * 20)
        for index, (target_dir, train_dir) in enumerate(zip(sorted(valid_dirs), sorted(train_dirs[:7]))):
            time.sleep(1)
            start = time.perf_counter()
            machine_type = target_dir.split('/')[-2]
            decay = self.args.gwrp_decays[machine_type]
            self.feature_extractor.args.decay = decay
            s_gmm_n = s_gmm_n if not use_search else self.args.gmm_ns[machine_type]
            use_smote = use_smote if not use_search else self.args.smotes[machine_type]
            # result csv
            machine_section_list = utils.get_machine_section_list(target_dir)
            csv_lines.append([machine_type])
            csv_lines.append(['section', 'AUC(Source)', 'AUC(Target)', 'pAUC'])
            performance = []
            gmms, train_scores = [], []
            for section_str in machine_section_list:
                # train GMM
                self.logger.info(f'[{machine_type}|{section_str}] Fit GMM-{s_gmm_n,t_gmm_n}...')
                if self.args.pool_type == 'gwrp':
                    self.logger.info(f'Gwrp decay: {decay:.2f}')
                # train_files = utils.get_filename_list(train_dir, pattern=f'{section_str}_*')
                s_train_files = utils.get_filename_list(train_dir, pattern=f'{section_str}_source_*')
                t_train_files = utils.get_filename_list(train_dir, pattern=f'{section_str}_target_*')
                train_files = s_train_files + t_train_files
                self.logger.info(f'number of {section_str} files: {len(train_files)}')
                if not use_smote:
                    features = self.feature_extractor.extract(train_files)
                else:
                    features = self.feature_extractor.smote_extract(s_train_files, t_train_files)
                gmm = self.fit_GMM(features, n_components=s_gmm_n)
                gmms.append(gmm)
            #     for file in train_files:
            #         feature = self.feature_extractor.extract([file])
            #         score = - np.max(gmm._estimate_log_prob(feature))
            #         train_scores.append(score)
            # max_score = np.max(train_scores)
            # min_score = np.min(train_scores)
            # train_scores = (train_scores - min_score) / (max_score - min_score)
            # shape_hat, loc_hat, scale_hat = scipy.stats.gamma.fit(train_scores)
            # decision_threshold = scipy.stats.gamma.ppf(q=self.args.decision_threshold, a=shape_hat, loc=loc_hat,
            #                                            scale=scale_hat)
            for idx, section_str in enumerate(machine_section_list):
                gmm = gmms[idx]
                csv_path = os.path.join(result_dir, f'anomaly_score_{machine_type}_{section_str}.csv')
                decision_path = os.path.join(result_dir, f'decision_result_{machine_type}_{section_str}.csv')
                test_files, y_true, domain_list = utils.get_valid_file_list(target_dir, section_str)
                y_pred = [0. for _ in test_files]
                anomaly_score_list = []
                decision_result_list = []
                # decision_result_list.append(['decision threshold', decision_threshold, 'max', max_score, 'min', min_score])
                for file_idx, test_file in enumerate(test_files):
                    test_feature = self.feature_extractor.extract([test_file])
                    # y_pred[file_idx] = - min(np.max(s_gmm._estimate_log_prob(test_feature)),
                    #                          np.max(t_gmm._estimate_log_prob(test_feature)))
                    y_pred[file_idx] = - np.max(gmm._estimate_log_prob(test_feature))
                    # y_pred[file_idx] = (- np.max(gmm._estimate_log_prob(test_feature)) - min_score) / (max_score - min_score)
                    anomaly_score_list.append([os.path.basename(test_file), y_pred[file_idx]])
                    # if y_pred[file_idx] > decision_threshold:
                    #     decision_result_list.append([os.path.basename(test_file), y_pred[file_idx], 1])
                    # else:
                    #     decision_result_list.append([os.path.basename(test_file), y_pred[file_idx], 0])
                if save:
                    print(result_dir, csv_path)
                    utils.save_csv(csv_path, anomaly_score_list)
                    # utils.save_csv(decision_path, decision_result_list)
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
        csv_lines.append(['(A)Total Average', f'{avg_auc_s:.4f}', f'{avg_auc_t:.4f}', f'{avg_pauc:.4f}'])
        csv_lines.append(['(H)Total Average', f'{h_avg_auc_s:.4f}', f'{h_avg_auc_t:.4f}', f'{h_avg_pauc:.4f}'])
        if save: utils.save_csv(result_path, csv_lines)
        self.logger.info(f'avg_auc_s: {avg_auc_s:.3f}\tavg_auc_t: {avg_auc_t:.3f}\tavg_pauc: {avg_pauc:.3f}')


    def eval(self, train_dirs, test_dirs, gmm_n=2, use_search=False, use_smote=False):
        result_dir = os.path.join('./evaluator/teams', self.args.version)
        train_result_dir = os.path.join('./evaluator/trains', self.args.version)
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(train_result_dir, exist_ok=True)
        print('\n' + '=' * 20)
        for index, (target_dir, train_dir) in enumerate(zip(sorted(test_dirs), sorted(train_dirs[7:]))):
            machine_type = target_dir.split('/')[-2]
            decay = self.args.gwrp_decays[machine_type]
            self.feature_extractor.args.decay = decay
            gmm_n = gmm_n if not use_search else self.args.gmm_ns[machine_type]
            use_smote = use_smote if not use_search else self.args.smotes[machine_type]
            # get machine list
            machine_section_list = utils.get_machine_section_list(target_dir)
            for section_str in machine_section_list:
                # train GMM
                self.logger.info(f'[{machine_type}|{section_str}] Fit GMM-{gmm_n}...')
                if self.args.pool_type == 'gwrp':
                    self.logger.info(f'Gwrp decay: {decay:.2f}')
                s_train_files = utils.get_filename_list(train_dir, pattern=f'{section_str}_source_*')
                t_train_files = utils.get_filename_list(train_dir, pattern=f'{section_str}_target_*')
                train_files = s_train_files + t_train_files
                self.logger.info(f'number of {section_str} files: {len(train_files)}')
                if not use_smote:
                    features = self.feature_extractor.extract(train_files)
                else:
                    features = self.feature_extractor.smote_extract(s_train_files, t_train_files)
                gmm = self.fit_GMM(features, n_components=gmm_n)
                #
                test_files = utils.get_eval_file_list(target_dir, section_str)
                csv_path = os.path.join(result_dir, f'anomaly_score_{machine_type}_{section_str}.csv')
                train_csv_path = os.path.join(train_result_dir, f'anomaly_score_{machine_type}_{section_str}.csv')
                print('create test anomaly score files...')
                anomaly_score_list = []
                y_pred = [0. for _ in test_files]
                for file_idx, file_path in enumerate(test_files):
                    test_feature = self.feature_extractor.extract([file_path])
                    y_pred[file_idx] = - np.max(gmm._estimate_log_prob(test_feature))
                    anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])
                    utils.save_csv(csv_path, anomaly_score_list)
                print('record train files score for computing threshold...')
                anomaly_score_list = []
                y_pred = [0. for _ in train_files]
                for file_idx, file_path in enumerate(train_files):
                    test_feature = self.feature_extractor.extract([file_path])
                    y_pred[file_idx] = - np.max(gmm._estimate_log_prob(test_feature))
                    anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])
                    utils.save_csv(train_csv_path, anomaly_score_list)


    def search_gwrp(self, train_dirs, valid_dirs, save=False, step=1, s_gmm_n=2, t_gmm_n=2, use_smote=False):
        # set gwrp pool type
        best_gwrp_decays = {}
        best_metrics = {}
        best_sum_metrics = {}
        machine_list = ['ToyCar', 'ToyTrain', 'bearing', 'fan', 'gearbox', 'slider', 'valve']
        for machine in machine_list:
            best_metrics[machine] = {
                'avg_auc_s': 0,
                'avg_auc_t': 0,
                'avg_p_auc': 0,
            }
            best_sum_metrics[machine] = 0
            best_gwrp_decays[machine] = 0
        result_dir = os.path.join('./results', self.args.version)
        os.makedirs(result_dir, exist_ok=True)
        for decay in np.arange(0, 100+step, step):
            decay /= 100
            self.feature_extractor.args.decay = decay
            csv_lines = []
            sum_auc_s, sum_auc_t, sum_pauc, num, total_time = 0, 0, 0, 0, 0
            h_sum_auc_s, h_sum_auc_t, h_sum_pauc = 0, 0, 0
            print('\n' + '=' * 20)
            for index, (target_dir, train_dir) in enumerate(zip(sorted(valid_dirs), sorted(train_dirs))):
                time.sleep(1)
                start = time.perf_counter()
                machine_type = target_dir.split('/')[-2]
                # result csv
                machine_section_list = utils.get_machine_section_list(target_dir)
                csv_lines.append([machine_type])
                csv_lines.append(['section', 'AUC(Source)', 'AUC(Target)', 'pAUC'])
                performance = []
                for section_str in machine_section_list:
                    # train GMM
                    self.logger.info(f'[{machine_type}|{section_str}] Fit GMM-{s_gmm_n,t_gmm_n}...')
                    s_train_files = utils.get_filename_list(train_dir, pattern=f'{section_str}_source_*')
                    t_train_files = utils.get_filename_list(train_dir, pattern=f'{section_str}_target_*')
                    train_files = s_train_files + t_train_files
                    # train_files = utils.get_filename_list(train_dir, pattern=f'{section_str}_*')
                    self.logger.info(f'[decay={decay:.2f}] number of {section_str} files: {len(train_files)}')
                    # s_features, t_features, conv_I, conv = self.feature_extractor.gmm_extract(s_train_files, t_train_files)
                    # features = np.concatenate((s_features, t_features), axis=0)
                    # features = self.feature_extractor.extract(train_files)
                    if use_smote:
                        features = self.feature_extractor.smote_extract(s_train_files, t_train_files)
                    else:
                        features = self.feature_extractor.extract(train_files)
                    # s_conv_I = np.expand_dims(conv_I, axis=0).repeat(s_gmm_n, axis=0)
                    # t_conv_I = np.expand_dims(conv_I, axis=0).repeat(t_gmm_n, axis=0)
                    # s_gmm = self.fit_GMM(s_features, n_components=s_gmm_n, means_init=None, precisions_init=None)
                    # t_gmm = self.fit_GMM(t_features, n_components=t_gmm_n, means_init=None, precisions_init=None)
                    gmm = self.fit_GMM(features, n_components=s_gmm_n)
                    # t_gmm.covariances_ = np.expand_dims(conv, axis=0).repeat(t_gmm_n, axis=0)
                    # get test info
                    csv_path = os.path.join(result_dir, f'anomaly_score_{machine_type}_{section_str}.csv')
                    test_files, y_true, domain_list = utils.get_valid_file_list(target_dir, section_str)
                    y_pred = [0. for _ in test_files]
                    anomaly_score_list = []
                    for file_idx, test_file in enumerate(test_files):
                        test_feature = self.feature_extractor.extract([test_file])
                        # y_pred[file_idx] = - min(np.max(s_gmm._estimate_log_prob(test_feature)),
                        #                          np.max(t_gmm._estimate_log_prob(test_feature)))
                        y_pred[file_idx] = - np.max(gmm._estimate_log_prob(test_feature))
                        anomaly_score_list.append([os.path.basename(test_file), y_pred[file_idx]])
                        # if 'source' in test_file: y_pred[file_idx] = - np.max(s_gmm._estimate_log_prob(test_feature))
                        # if 'target' in test_file: y_pred[file_idx] = - np.max(t_gmm._estimate_log_prob(test_feature))
                    if save:
                        print(result_dir, csv_path)
                        utils.save_csv(csv_path, anomaly_score_list)
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
                self.logger.info(f'[decay={decay:.2f}] Test {machine_type}\tcost {time_nedded:.2f} sec\tavg_auc_s: {mean_auc_s:.3f}\t'
                                 f'avg_auc_t: {mean_auc_t:.3f}\tavg_pauc: {mean_p_auc:.3f}')
                sum_metrics = h_mean_auc_s + h_mean_auc_t + h_mean_p_auc
                if sum_metrics > best_sum_metrics[machine_type]:
                    best_sum_metrics[machine_type] = sum_metrics
                    best_gwrp_decays[machine_type] = decay
                    best_metrics[machine_type]['avg_auc_s'] = h_mean_auc_s
                    best_metrics[machine_type]['avg_auc_t'] = h_mean_auc_t
                    best_metrics[machine_type]['avg_p_auc'] = h_mean_p_auc
            print(f'Total test time: {total_time:.2f} sec')
            result_path = os.path.join(result_dir, 'result.csv')
            avg_auc_s, avg_auc_t, avg_pauc = sum_auc_s / num, sum_auc_t / num, sum_pauc / num
            h_avg_auc_s, h_avg_auc_t, h_avg_pauc = h_sum_auc_s / num, h_sum_auc_t / num, h_sum_pauc / num
            csv_lines.append(['(A)Total Average', f'{avg_auc_s:.4f}', f'{avg_auc_t:.4f}', f'{avg_pauc:.4f}'])
            csv_lines.append(['(H)Total Average', f'{h_avg_auc_s:.4f}', f'{h_avg_auc_t:.4f}', f'{h_avg_pauc:.4f}'])
            if save: utils.save_csv(result_path, csv_lines)
            self.logger.info(f'avg_auc_s: {avg_auc_s:.3f}\tavg_auc_t: {avg_auc_t:.3f}\tavg_pauc: {avg_pauc:.3f}')

        result_path = os.path.join(result_dir, f'result-gmm-{s_gmm_n}.csv')
        csv_lines = []
        sum_auc_s, sum_auc_t, sum_p_auc, num = 0, 0, 0, 0
        for machine in machine_list:
            csv_lines.append([machine, 'AUC(Source)', 'AUC(Target)', 'pAUC'])
            auc_s = best_metrics[machine]['avg_auc_s']
            auc_t = best_metrics[machine]['avg_auc_t']
            p_auc = best_metrics[machine]['avg_p_auc']
            decay = best_gwrp_decays[machine]
            csv_lines.append([f'decay={decay:.2f}', f'{auc_s:.4f}', f'{auc_t:.4f}', f'{p_auc:.4f}'])
            csv_lines.append([])
            sum_auc_s += auc_s
            sum_auc_t += auc_t
            sum_p_auc += p_auc
            num += 1
        avg_auc_s, avg_auc_t, avg_pauc = sum_auc_s / num, sum_auc_t / num, sum_p_auc / num
        csv_lines.append(['(H)Total Average', f'{avg_auc_s:.4f}', f'{avg_auc_t:.4f}', f'{avg_pauc:.4f}'])
        utils.save_csv(result_path, csv_lines)