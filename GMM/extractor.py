import os.path
import torch
import librosa
import numpy as np
from collections import OrderedDict
import utils
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from collections import Counter

from net import STgramMFN


class SpecExtractor:
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.dim = kwargs['dim']
        self.transform = kwargs['transform']
        self.logger = self.args.logger
        self.pool_type = self.args.pool_type  # mean,max,mean+max,min,gwrp
        self.sm = SMOTE(sampling_strategy=0.2, random_state=self.args.seed, k_neighbors=3, n_jobs=32)
        # self.sm = BorderlineSMOTE(sampling_strategy=0.5, random_state=self.args.seed, k_neighbors=3, m_neighbors=6, n_jobs=32)
    def smote_extract(self, s_files, t_files):
        s_xs, t_xs = [], []
        for file in s_files:
            (x, _) = librosa.core.load(file, sr=self.args.sr, mono=True)
            x = x[:self.args.sr * 10]
            s_xs.append(x)
            # x = torch.from_numpy(x)
            # x_mel = self.transform(x)
            # s_xs.append(x_mel.reshape(-1).numpy())
        for file in t_files:
            (x, _) = librosa.core.load(file, sr=self.args.sr, mono=True)
            x = x[:self.args.sr * 10]
            t_xs.append(x)
            # x = torch.from_numpy(x)
            # x_mel = self.transform(x)
            # t_xs.append(x_mel.reshape(-1).numpy())
        s_y = [1 for _ in s_files]
        t_y = [0 for _ in t_files]
        y = s_y + t_y
        xs = s_xs + t_xs
        res_xs, res_y = self.sm.fit_resample(xs, y)
        print(Counter(y), Counter(res_y))
        features = []
        for x in res_xs:
            x = torch.from_numpy(np.array(x)).float()
            x_mel = self.transform(x)
            feature = self.get_feature(x_mel, dim=self.dim).reshape(1, -1)
            features.append(feature)
        features = torch.cat(features, dim=0)
        # conv = np.cov(features, rowvar=False)
        # conv_I = np.linalg.inv(conv)
        return features.numpy()

    def extract(self, files):
        """
        dim=0 : extract feature in frequency dimension
        dim=1 : extract feature in time dimension
        """
        if len(files) > 100:
            machine = files[0].split('/')[-3]
            section = files[0].split('/')[-1][:10]
            self.logger.info(f'[{machine}|{section}|sum={len(files)}] Extract {self.pool_type} features in time...')
        features = []
        for file in files:
            (x, _) = librosa.core.load(file, sr=self.args.sr, mono=True)
            x = x[:self.args.sr * 10]
            x = torch.from_numpy(x)
            x_mel = self.transform(x)
            feature = self.get_feature(x_mel, dim=self.dim).reshape(1, -1)
            features.append(feature)
        features = torch.cat(features, dim=0)
        return features.numpy()

    def get_feature(self, x_mel, dim=0):
        if self.pool_type == 'mean':
            feature = x_mel.mean(dim=dim)
        elif self.pool_type == 'max':
            feature, _ = x_mel.max(dim=dim)
        elif self.pool_type == 'mean+max':
            f_avg = x_mel.mean(dim=dim)
            f_max, _ = x_mel.max(dim=dim)
            feature = f_avg + f_max
        elif self.pool_type == 'meanCatmax':
            f_avg = x_mel.mean(dim=dim)
            f_max, _ = x_mel.max(dim=dim)
            feature = torch.cat((f_avg, f_max), dim=0)
        elif self.pool_type == 'gwrp':
            feature = utils.gwrp(x_mel.numpy(), decay=self.args.decay, dim=dim)
            feature = torch.from_numpy(feature)
        elif self.pool_type == 'meanCatgwrp':
            f_avg = x_mel.mean(dim=dim)
            f_gwrp = utils.gwrp(x_mel.numpy(), decay=self.args.decay, dim=dim)
            feature = torch.cat((f_avg, torch.from_numpy(f_gwrp)), dim=0)
            # feature = torch.from_numpy(feature)
        elif self.pool_type == 'maxCatgwrp':
            f_max, _= x_mel.max(dim=dim)
            f_gwrp = utils.gwrp(x_mel.numpy(), decay=self.args.decay, dim=dim)
            feature = torch.cat((f_max, torch.from_numpy(f_gwrp)), dim=0)
            # feature = torch.from_numpy(feature)
        elif self.pool_type == 'min':
            feature, _ = x_mel.min(dim=dim)
        else:
            raise ValueError('pool_type set error!"')
        return feature


class STgramMFNExtractor:
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.transform = kwargs['transform']
        self.model_path = self.args.model_path
        self.logger = self.args.logger
        self.net = self.load_model()

    def load_model(self):
        net = STgramMFN(num_classes=self.args.num_classes, num_attributes=self.args.num_attributes,
                        arcface=self.args.arcface, m=self.args.m, s=self.args.s, sub=self.args.sub)
        state_dict = torch.load(self.model_path, map_location=torch.device('cpu'))
        load_epoch = state_dict['epoch']
        self.logger.info(f'load epoch: {load_epoch}')
        net.load_state_dict(state_dict['model'])
        net.eval()
        return net

    def extract(self, files):
        """
        dim=0 : extract feature in frequency dimension
        dim=1 : extract feature in time dimension
        """
        if len(files) > 100:
            machine = files[0].split('/')[-3]
            section = files[0].split('/')[-1][:10]
            self.logger.info(f'[{machine}|{section}|sum={len(files)}] Extract STgramMFN features...')
        xs, x_mels = [], []
        for file in files:
            (x, _) = librosa.core.load(file, sr=self.args.sr, mono=True)
            x = x[:self.args.sr * 10]
            x = torch.from_numpy(x)
            x_mel = self.transform(x)
            xs.append(x.unsqueeze(0))
            x_mels.append(x_mel.unsqueeze(0))
        xs = torch.cat(xs, dim=0).float()
        x_mels = torch.cat(x_mels, dim=0).float()
        features = self.get_feature(xs, x_mels)
        return features.numpy()

    def get_feature(self, x, x_mel, label=None):
        with torch.no_grad():
            # _, _, features = self.net(x, x_mel, label)
            features = self.net.get_tgram(x.unsqueeze(1))
            features = features.mean(dim=2)
        return features.detach()
