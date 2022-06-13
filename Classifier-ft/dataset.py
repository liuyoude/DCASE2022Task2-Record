import re
import torch
from torch.utils.data import Dataset
import torchaudio
import librosa

import utils


class ASDDataset(Dataset):
    def __init__(self, dirs: list, args):
        self.filename_list = []
        self.sr = args.sr
        pattern = f'*_{args.domain}_*'
        for dir in dirs:
            self.filename_list.extend(utils.get_filename_list(dir, pattern=pattern))
        self.wav2mel = utils.Wave2Mel(sr=args.sr)
        self.mean = args.mean
        self.std = args.std
        self.att2idx = args.att2idx
        self.file_att_2_idx = args.file_att_2_idx

    def __getitem__(self, item):
        filename = self.filename_list[item]
        return self.transform(filename)

    def transform(self, filename):
        label, one_hot = utils.get_label('/'.join(filename.split('/')[-3:]), self.att2idx, self.file_att_2_idx)
        (x, _) = librosa.core.load(filename, sr=self.sr, mono=True)
        x = x[:self.sr * 10]  # (1, audio_length)
        x_wav = torch.from_numpy(x)
        x_mel = self.wav2mel(x_wav)
        x_mel = utils.normalize(x_mel, mean=self.mean, std=self.std)
        # print(x.shape)
        return x_wav, x_mel, label, one_hot

    def __len__(self):
        return len(self.filename_list)