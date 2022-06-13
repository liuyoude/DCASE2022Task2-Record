import numpy as np
import csv
import os
import re
import scipy
import time
import sys
from sklearn.preprocessing import scale
import utils

machines = ['ToyCar', 'ToyTrain', 'bearing', 'fan', 'gearbox', 'slider', 'valve']
valid_dirs = [
    '../../data/dcase2022dataset/dev_data/fan/test',
    '../../data/dcase2022dataset/dev_data/slider/test',
    '../../data/dcase2022dataset/dev_data/valve/test',
    '../../data/dcase2022dataset/dev_data/ToyCar/test',
    '../../data/dcase2022dataset/dev_data/ToyTrain/test',
    '../../data/dcase2022dataset/dev_data/bearing/test',
    '../../data/dcase2022dataset/dev_data/gearbox/test'
]
test_dirs = [
    '../../data/eval_dataset/fan/test',
    '../../data/eval_dataset/slider/test',
    '../../data/eval_dataset/valve/test',
    '../../data/eval_dataset/ToyCar/test',
    '../../data/eval_dataset/ToyTrain/test',
    '../../data/eval_dataset/bearing/test',
    '../../data/eval_dataset/gearbox/test'
]
decision = 0.9



def get_score_dict(result_dir):
    max_score_dict = {}
    score_dict = {}
    for machine in machines:
        score_dict[machine] = {}
        result_files = utils.get_filename_list(result_dir, ext='csv', pattern=f'anomaly_score_{machine}_*')
        max_score = float('-inf')
        for result_file in result_files:
            file_name = os.path.split(result_file)[-1]
            section_str = re.findall('section_[0-9][0-9]', file_name)[0]
            # section_str = '_'.join(file_name[:-4].split('_')[3:])
            y_pred = []
            with open(result_file, 'r') as f:
                csv_lines = csv.reader(f)
                for csv_line in csv_lines:
                    score = float(csv_line[1])
                    y_pred.append(score)
                    if abs(score) > max_score: max_score = abs(score)
                    # if score > max_score: max_score = score
            score_dict[machine][section_str] = y_pred
        max_score_dict[machine] = max_score
    return score_dict, max_score_dict


def get_label_dict(valid_dirs):
    label_dict = {}
    for index, target_dir in enumerate(sorted(valid_dirs)):
        machine = target_dir.split('/')[-2]
        label_dict[machine] = {}
        # get machine section list
        machine_section_list = utils.get_machine_section_list(target_dir)
        for section_str in machine_section_list:
            label_dict[machine][section_str] = {}
            test_files, y_true, domain_list = utils.get_valid_file_list(target_dir, section_str)
            label_dict[machine][section_str]['y_true'] = y_true
            label_dict[machine][section_str]['domain_list'] = domain_list
            label_dict[machine][section_str]['test_files'] = test_files
    return label_dict


def valid_ensemble(result_dirs, valid_dirs, save_dir, step=1, save=False):
    os.makedirs(save_dir, exist_ok=True)
    print('create weights list for ensemble...')
    if len(result_dirs) == 1:
        weights = [[100]]
    if len(result_dirs) == 2:
        weights = [[i, 100-i] for i in np.arange(0, 100+step, step)]
    elif len(result_dirs) == 3:
        weights = [[i, j, 100-i-j] for i in np.arange(0, 100+step, step)
                   for j in np.arange(0, 100-i+step, step)]
    elif len(result_dirs) == 4:
        weights = [[i, j, k, 100-i-j-k] for i in np.arange(0, 100+step, step)
                   for j in np.arange(0, 100-i+step, step)
                   for k in np.arange(0, 100-i-j+step, step)]
    print('get true lable dict...')
    label_dict = get_label_dict(valid_dirs)
    print('get score dict for weight and normalizing...')
    score_dict_list, max_score_dict_list = [], []
    for result_dir in result_dirs:
        score_dict, max_score_dict = get_score_dict(result_dir)
        score_dict_list.append(score_dict)
        max_score_dict_list.append(max_score_dict)
    print('start searching...')
    best_weights = {}
    best_metrics = {}
    h_best_metrics = {}
    best_sum_metrics = {}
    machine_list = ['ToyCar', 'ToyTrain', 'bearing', 'fan', 'gearbox', 'slider', 'valve']
    for machine in machine_list:
        best_metrics[machine] = {
            'avg_auc_s': 0,
            'avg_auc_t': 0,
            'avg_p_auc': 0,
        }
        h_best_metrics[machine] = {
            'avg_auc_s': 0,
            'avg_auc_t': 0,
            'avg_p_auc': 0,
        }
        best_sum_metrics[machine] = 0
        best_weights[machine] = 0
    for weight in weights:
        weight = [w / 100 for w in weight]
        csv_lines = []
        sum_auc_s, sum_auc_t, sum_pauc, num, total_time = 0, 0, 0, 0, 0
        h_sum_auc_s, h_sum_auc_t, h_sum_pauc = 0, 0, 0
        for index, target_dir in enumerate(sorted(valid_dirs)):
            machine = target_dir.split('/')[-2]
            if save:
                csv_lines.append([machine])
                csv_lines.append(['section', 'AUC(Source)', 'AUC(Target)', 'pAUC'])
            performance = []
            # get machine list
            machine_section_list = utils.get_machine_section_list(target_dir)
            for section_str in machine_section_list:
                csv_path = os.path.join(save_dir, f'anomaly_score_{machine}_{section_str}.csv')
                y_true = label_dict[machine][section_str]['y_true']
                domain_list = label_dict[machine][section_str]['domain_list']
                y_weight_pred, anomaly_score_list = [], []
                for idx in range(len(result_dirs)):
                    scores = score_dict_list[idx][machine][section_str]
                    max_score = max_score_dict_list[idx][machine]
                    scores = np.array(scores) / max_score
                    # scores = scale(np.array(scores))
                    if idx == 0:
                        y_weight_pred = weight[idx] * scores
                    else:
                        y_weight_pred += weight[idx] * scores
                y_weight_pred = list(y_weight_pred)
                if save:
                    for idx, test_file in enumerate(label_dict[machine][section_str]['test_files']):
                        anomaly_score_list.append([os.path.basename(test_file), y_weight_pred[idx]])
                # compute auc and pAuc
                auc_s, auc_t, p_auc = utils.cal_auc_pauc(y_true, y_weight_pred, domain_list)
                performance.append([auc_s, auc_t, p_auc])
                if save:
                    utils.save_csv(csv_path, anomaly_score_list)
                    csv_lines.append([section_str.split('_', 1)[1], auc_s, auc_t, p_auc])
            # calculate averages for AUCs and pAUCs
            amean_performance = np.mean(np.array(performance, dtype=float), axis=0)
            hmean_performance = scipy.stats.hmean(
                np.maximum(np.array(performance, dtype=float), sys.float_info.epsilon), axis=0)
            mean_auc_s, mean_auc_t, mean_p_auc = amean_performance[0], amean_performance[1], amean_performance[2]
            h_mean_auc_s, h_mean_auc_t, h_mean_p_auc = hmean_performance[0], hmean_performance[1], \
                                                       hmean_performance[2]
            sum_auc_s += mean_auc_s
            sum_auc_t += mean_auc_t
            sum_pauc += mean_p_auc
            h_sum_auc_s += h_mean_auc_s
            h_sum_auc_t += h_mean_auc_t
            h_sum_pauc += h_mean_p_auc
            num += 1
            if save:
                csv_lines.append(["arithmetic mean"] + list(amean_performance))
                # csv_lines.append(["harmonic mean"] + list(hmean_performance))
                csv_lines.append([])
            print(
                f'[weights={weight}] Test {machine}\tavg_auc_s: {mean_auc_s:.3f}\t'
                f'avg_auc_t: {mean_auc_t:.3f}\tavg_pauc: {mean_p_auc:.3f}')
            sum_metrics = h_mean_auc_s + h_mean_auc_t + h_mean_p_auc
            if sum_metrics > best_sum_metrics[machine]:
                best_sum_metrics[machine] = sum_metrics
                best_weights[machine] = weight
                best_metrics[machine]['avg_auc_s'] = mean_auc_s
                best_metrics[machine]['avg_auc_t'] = mean_auc_t
                best_metrics[machine]['avg_p_auc'] = mean_p_auc
                h_best_metrics[machine]['avg_auc_s'] = h_mean_auc_s
                h_best_metrics[machine]['avg_auc_t'] = h_mean_auc_t
                h_best_metrics[machine]['avg_p_auc'] = h_mean_p_auc
        result_path = os.path.join(save_dir, 'result.csv')
        avg_auc_s, avg_auc_t, avg_pauc = sum_auc_s / num, sum_auc_t / num, sum_pauc / num
        h_avg_auc_s, h_avg_auc_t, h_avg_pauc = h_sum_auc_s / num, h_sum_auc_t / num, h_sum_pauc / num
        csv_lines.append(['(A)Total Average', f'{avg_auc_s:.4f}', f'{avg_auc_t:.4f}', f'{avg_pauc:.4f}'])
        csv_lines.append(['(H)Total Average', f'{h_avg_auc_s:.4f}', f'{h_avg_auc_t:.4f}', f'{h_avg_pauc:.4f}'])
        if save: utils.save_csv(result_path, csv_lines)
        print(f'avg_auc_s: {avg_auc_s:.3f}\tavg_auc_t: {avg_auc_t:.3f}\tavg_pauc: {avg_pauc:.3f}')
    result_path = os.path.join(save_dir, f'result-ensemble-search.csv')
    csv_lines = []
    sum_auc_s, sum_auc_t, sum_p_auc, num = 0, 0, 0, 0
    h_sum_auc_s, h_sum_auc_t, h_sum_p_auc = 0, 0, 0
    for machine in machine_list:
        csv_lines.append([machine, 'AUC(Source)', 'AUC(Target)', 'pAUC'])
        auc_s = best_metrics[machine]['avg_auc_s']
        auc_t = best_metrics[machine]['avg_auc_t']
        p_auc = best_metrics[machine]['avg_p_auc']
        h_auc_s = h_best_metrics[machine]['avg_auc_s']
        h_auc_t = h_best_metrics[machine]['avg_auc_t']
        h_p_auc = h_best_metrics[machine]['avg_p_auc']
        weight = best_weights[machine]
        csv_lines.append([f'weight={weight}'])
        csv_lines.append([f'Arithmetic mean', f'{auc_s:.4f}', f'{auc_t:.4f}', f'{p_auc:.4f}'])
        csv_lines.append([f'Harmonic mean', f'{h_auc_s:.4f}', f'{h_auc_t:.4f}', f'{h_p_auc:.4f}'])
        csv_lines.append([])
        sum_auc_s += auc_s
        sum_auc_t += auc_t
        sum_p_auc += p_auc
        h_sum_auc_s += h_auc_s
        h_sum_auc_t += h_auc_t
        h_sum_p_auc += h_p_auc
        num += 1
    avg_auc_s, avg_auc_t, avg_pauc = sum_auc_s / num, sum_auc_t / num, sum_p_auc / num
    h_avg_auc_s, h_avg_auc_t, h_avg_pauc = h_sum_auc_s / num, h_sum_auc_t / num, h_sum_p_auc / num
    csv_lines.append(['(A)Total Average', f'{avg_auc_s:.4f}', f'{avg_auc_t:.4f}', f'{avg_pauc:.4f}'])
    csv_lines.append(['(H)Total Average', f'{h_avg_auc_s:.4f}', f'{h_avg_auc_t:.4f}', f'{h_avg_pauc:.4f}'])
    print(f'avg_auc_s: {avg_auc_s:.3f}\tavg_auc_t: {avg_auc_t:.3f}\tavg_pauc: {avg_pauc:.3f}')
    utils.save_csv(result_path, csv_lines)
    return best_weights



def test_ensemble(result_dirs, weights, valid_result_dirs, train_result_dirs, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    print('get score dict for weight and normalizing...')
    score_dict_list, train_score_dict_list, max_score_dict_list, train_max_score_dict_list = [], [], [], []
    for result_dir, train_result_dir, valid_result_dir in zip(result_dirs, train_result_dirs, valid_result_dirs):
        score_dict, _ = get_score_dict(result_dir)
        train_score_dict, train_max_score_dict = get_score_dict(train_result_dir)
        _, max_score_dict = get_score_dict(valid_result_dir)
        score_dict_list.append(score_dict)
        max_score_dict_list.append(max_score_dict)
        train_score_dict_list.append(train_score_dict)
        train_max_score_dict_list.append(max_score_dict)

    for index, target_dir in enumerate(sorted(test_dirs)):
        machine = target_dir.split('/')[-2]
        weight = weights[machine]
        # get machine list
        machine_section_list = utils.get_machine_section_list(target_dir)
        # get train score list for computing threshold
        train_weight_scores = []
        for section_str in machine_section_list:
            y_weight_pred = []
            for idx in range(len(result_dirs)):
                scores = train_score_dict_list[idx][machine][section_str]
                max_score = train_max_score_dict_list[idx][machine]
                scores = np.array(scores) / max_score
                # scores = scale(np.array(scores))
                if idx == 0:
                    y_weight_pred = weight[idx] * scores
                else:
                    y_weight_pred += weight[idx] * scores
            train_weight_scores += list(y_weight_pred)
        all_min_score = np.min(train_weight_scores)
        all_max_score = np.max(train_weight_scores)
        train_weight_scores = train_weight_scores - all_min_score
        shape_hat, loc_hat, scale_hat = scipy.stats.gamma.fit(train_weight_scores)
        decision_threshold = scipy.stats.gamma.ppf(q=decision, a=shape_hat, loc=loc_hat,
                                                   scale=scale_hat)
        # decision_threshold = np.mean(train_weight_scores)
        # create test csv files
        for section_str in machine_section_list:
            csv_path = os.path.join(save_dir, f'anomaly_score_{machine}_{section_str}.csv')
            decision_path = os.path.join(save_dir, f'decision_result_{machine}_{section_str}.csv')
            test_files = utils.get_eval_file_list(target_dir, section_str)
            y_weight_pred, anomaly_score_list, decision_result_list = [], [], []
            # decision_result_list.append(['decision threshold', decision_threshold, 'min', all_min_score, 'max', all_max_score])
            for idx in range(len(result_dirs)):
                scores = score_dict_list[idx][machine][section_str]
                max_score = max_score_dict_list[idx][machine]
                scores = np.array(scores) / max_score
                # scores = scale(np.array(scores))
                if idx == 0:
                    y_weight_pred = weight[idx] * scores
                else:
                    y_weight_pred += weight[idx] * scores
            y_weight_pred = y_weight_pred - all_min_score
            y_weight_pred = list(y_weight_pred)
            # save files
            for idx, test_file in enumerate(test_files):
                anomaly_score_list.append([os.path.basename(test_file), y_weight_pred[idx]])
                if y_weight_pred[idx] > decision_threshold:
                    decision_result_list.append([os.path.basename(test_file), 1])
                else:
                    decision_result_list.append([os.path.basename(test_file), 0])
            utils.save_csv(csv_path, anomaly_score_list)
            utils.save_csv(decision_path, decision_result_list)



if __name__ == '__main__':
    valid_results = [
        f'../GMM/results/GMM-logMel-gwrp-Mix-ensemble/GMM-Mix-research',
        f'../Classifier/results/2022-06-02-14-STgram_MFN_ASLloss(alpha=norm(w),pos=2,neg=2,eps=0.1)_addDoaminAtt',
        f'../Classifier/results/2022-05-01-16-STgram_MFN_FocalLoss(alpha=norm(w))',
        f'../Classifier/results/2022-05-01-16-STgram_MFN_FocalLoss(alpha=norm(w))-ft-target',
        f'../Classifier/results/2022-06-09-19-STgram_MFN_Focalloss_addSecAtt',
        f'../GMM/results/GMM-logMel-gwrp-SMOTE(0.2)/GMM-Mix-research',
    ]
    test_results = [
        f'../GMM/evaluator/teams/GMM-logMel-gwrp-Mix-ensemble',
        f'../Classifier/evaluator/teams/2022-06-02-14-STgram_MFN_ASLloss(alpha=norm(w),pos=2,neg=2,eps=0.1)_addDoaminAtt',
        f'',
        f'../Classifier/evaluator/teams/2022-05-01-16-STgram_MFN_FocalLoss(alpha=norm(w))-ft-target',
        f'../Classifier/evaluator/teams/2022-06-09-19-STgram_MFN_Focalloss_addSecAtt',
        f'../GMM/evaluator/teams/GMM-logMel-gwrp-SMOTE(0.2)',

    ]
    train_results = [
        f'../GMM/evaluator/trains/GMM-logMel-gwrp-Mix-ensemble',
        f'../Classifier/evaluator/trains/2022-06-02-14-STgram_MFN_ASLloss(alpha=norm(w),pos=2,neg=2,eps=0.1)_addDoaminAtt',
        f'',
        f'../Classifier/evaluator/trains/2022-05-01-16-STgram_MFN_FocalLoss(alpha=norm(w))-ft-target',
        f'../Classifier/evaluator/trains/2022-06-09-19-STgram_MFN_Focalloss_addSecAtt',
        f'../GMM/evaluator/trains/GMM-logMel-gwrp-SMOTE(0.2)',
    ]

    systems = [[0], [5], [0, 3], [0, 1, 3, 4], [5, 1, 3, 4]]
    for idxs in systems:
        save_dir = f'./result/ensemble({idxs})-final'
        result_dirs = [valid_results[idx] for idx in idxs]
        test_result_dirs = [test_results[idx] for idx in idxs]
        train_result_dirs = [train_results[idx] for idx in idxs]

        weights = valid_ensemble(result_dirs, valid_dirs, save_dir=save_dir, step=1, save=False)
        print(weights)
        save_dir = os.path.join(save_dir, 'submit')
        os.makedirs(save_dir, exist_ok=True)
        test_ensemble(test_result_dirs, weights, result_dirs, train_result_dirs, save_dir=save_dir)
