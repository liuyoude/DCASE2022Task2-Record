"""
functional functions
"""
import os
import random
import re
import itertools
import glob

import sklearn
import torch
import yaml
import csv
import logging
import torchaudio
import numpy as np
import librosa


def load_yaml(file_path='./config.yaml'):
    with open(file_path) as f:
        params = yaml.safe_load(f)
    return params


def save_yaml_file(file_path, data: dict):
    with open(file_path, "w") as f:
        yaml.safe_dump(data, f, encoding='utf-8', allow_unicode=True)


def save_csv(file_path, data: list):
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(data)


def save_model_state_dict(file_path, epoch=None, net=None, optimizer=None):
    import torch
    state_dict = {
        'epoch': epoch,
        'optimizer': optimizer.state_dict() if optimizer else None,
        'model': net.state_dict() if net else None,
    }
    torch.save(state_dict, file_path)


def get_logger(filename):
    logging.basicConfig(filename=filename, level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    return logger


def get_filename_list(dir_path, ext='wav'):
    """
    find all extention files under directory
    :param dir_path: directory path
    :param ext: extention name, like wav, png...
    :return: files path list
    """
    filename_list = []
    ext = ext if ext else '*'
    for root, dirs, files in os.walk(dir_path):
        file_path_pattern = os.path.join(root, f'*.{ext}')
        files = sorted(glob.glob(file_path_pattern))
        filename_list += files
    return filename_list


def get_machine_id_list(target_dir, ext='wav'):
    dir_path = os.path.abspath(f'{target_dir}/*.{ext}')
    files_path = sorted(glob.glob(dir_path))
    machine_id_list = sorted(list(set(
        itertools.chain.from_iterable([re.findall('id_[0-9][0-9]', ext_id) for ext_id in files_path])
    )))
    return machine_id_list


def get_label(filename, machine_type, factors):
    id_str = re.findall('id_[0-9][0-9]', filename)[0]
    if machine_type == 'ToyCar' or machine_type == 'ToyConveyor':
        id = int(id_str[-1]) - 1
    else:
        id = int(id_str[-1])
    label = int(factors[machine_type] * 7 + id)
    return label


# getting target dir file list and label list
def get_valid_file_list(target_dir,
                        section_name,
                        prefix_normal='normal',
                        prefix_anomaly='anomaly',
                        ext='wav'):
    normal_files_path = f'{target_dir}/{section_name}_*_{prefix_normal}_*.{ext}'
    normal_files = sorted(glob.glob(normal_files_path))
    normal_labels = np.zeros(len(normal_files))

    anomaly_files_path = f'{target_dir}/{section_name}_*_{prefix_anomaly}_*.{ext}'
    anomaly_files = sorted(glob.glob(anomaly_files_path))
    anomaly_labels = np.ones(len(anomaly_files))

    files = np.concatenate((normal_files, anomaly_files), axis=0)
    labels = np.concatenate((normal_labels, anomaly_labels), axis=0)

    domain_list = []
    for file in files: domain_list.append('source' if ('source' in file) else 'target')
    return files, labels, domain_list


def get_eval_file_list(target_dir, id_name, ext='wav'):
    files_path = f'{target_dir}/{id_name}*.{ext}'
    files = sorted(glob.glob(files_path))
    return files


def get_machine_section_list(target_dir, ext='wav'):
    dir_path = os.path.abspath(f'{target_dir}/*.{ext}')
    files_path = sorted(glob.glob(dir_path))
    machine_section_list = sorted(list(set(
        itertools.chain.from_iterable([re.findall('section_[0-9][0-9]', ext_section) for ext_section in files_path])
    )))
    return machine_section_list


def save_statistic_data(dirs: list, sr=16000, file_path='mean_std.npy'):
    if os.path.exists(file_path): return
    import torch
    print('Get mean and std of each machine type for training...')
    data_dict = {}
    wav2mel = Wave2Mel(sr=sr)
    for dir in dirs:
        mean, std, sum = 0, 0, 0
        machine_type = dir.split('/')[-2]
        filenames = get_filename_list(dir)
        for filename in filenames:
            x, _ = librosa.core.load(filename, sr=sr, mono=True)
            x_mel = wav2mel(torch.from_numpy(x))
            mean += torch.mean(x_mel)
            std += torch.std(x_mel)
            sum += 1
        mean /= sum
        std /= sum
        data_dict[machine_type] = [mean, std]
        print(f'-{machine_type}: mean:{mean:.3f}\tstd:{std:.3f}')
    np.save(file_path, data_dict)
    print('='*40)


def cal_auc_pauc(y_true, y_pred, domain_list, max_fpr=0.1):
    y_true_s = [y_true[idx] for idx in range(len(y_true)) if domain_list[idx] == "source" or y_true[idx] == 1]
    y_pred_s = [y_pred[idx] for idx in range(len(y_true)) if domain_list[idx] == "source" or y_true[idx] == 1]
    y_true_t = [y_true[idx] for idx in range(len(y_true)) if domain_list[idx] == "target" or y_true[idx] == 1]
    y_pred_t = [y_pred[idx] for idx in range(len(y_true)) if domain_list[idx] == "target" or y_true[idx] == 1]
    auc_s = sklearn.metrics.roc_auc_score(y_true_s, y_pred_s)
    auc_t = sklearn.metrics.roc_auc_score(y_true_t, y_pred_t)
    p_auc = sklearn.metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
    return auc_s, auc_t, p_auc




class Wave2Mel(object):
    def __init__(self, sr,
                 n_fft=1024,
                 n_mels=128,
                 win_length=1024,
                 hop_length=512,
                 power=2.0
                 ):
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sr,
                                                                  win_length=win_length,
                                                                  hop_length=hop_length,
                                                                  n_fft=n_fft,
                                                                  n_mels=n_mels,
                                                                  power=power)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='power')

    def __call__(self, x):
        # spec =  self.amplitude_to_db(self.mel_transform(x)).squeeze().transpose(-1,-2)
        return self.amplitude_to_db(self.mel_transform(x))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    print(get_filename_list('../../data/dataset/fan/train'))
