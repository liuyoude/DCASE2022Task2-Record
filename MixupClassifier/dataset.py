import re
import torch
from torch.utils.data import Dataset
import torchaudio
import librosa
import random

import utils


class ASDDataset(Dataset):
    def __init__(self, dirs: list, args):
        self.source_filename_list = []
        self.target_filename_list = []
        self.sr = args.sr
        for dir in dirs:
            self.source_filename_list.extend(utils.get_filename_list(dir, pattern='*_source_*'))
            self.target_filename_list.extend(utils.get_filename_list(dir, pattern='*_target_*'))
        self.wav2mel = utils.Wave2Mel(sr=args.sr)
        self.mean = args.mean
        self.std = args.std
        self.att2idx = args.att2idx
        self.file_att_2_idx = args.file_att_2_idx

    def __getitem__(self, item):
        source_filename = self.source_filename_list[item]
        target_filename = random.choice(self.target_filename_list)
        return self.transform(source_filename, target_filename)

    def transform(self, source_filename, target_filename):
        s_label, s_one_hot = utils.get_label('/'.join(source_filename.split('/')[-3:]), self.att2idx, self.file_att_2_idx)
        (s_x, _) = librosa.core.load(source_filename, sr=self.sr, mono=True)
        s_x = s_x[:self.sr * 10]  # (1, audio_length)
        s_x_wav = torch.from_numpy(s_x)
        # s_x_mel = self.wav2mel(s_x_wav)
        # s_x_mel = utils.normalize(s_x_mel, mean=self.mean, std=self.std)
        # print(x.shape)
        t_label, t_one_hot = utils.get_label('/'.join(target_filename.split('/')[-3:]), self.att2idx,
                                             self.file_att_2_idx)
        (t_x, _) = librosa.core.load(target_filename, sr=self.sr, mono=True)
        t_x = t_x[:self.sr * 10]  # (1, audio_length)
        t_x_wav = torch.from_numpy(t_x)
        # t_x_mel = self.wav2mel(t_x_wav)
        # mixup
        alpha = random.random()
        x_wav = alpha * s_x_wav + (1 - alpha) * t_x_wav
        x_mel = self.wav2mel(x_wav)
        label = alpha * s_label + (1 - alpha) * t_label
        one_hot = alpha * s_one_hot + (1 - alpha) * t_one_hot
        return x_wav, x_mel, label, one_hot

    def __len__(self):
        return len(self.source_filename_list)