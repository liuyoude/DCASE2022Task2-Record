import re
import torch
import numpy as np
from torch.utils.data import Dataset
import torchaudio
import librosa

import utils


class ASDDataset(Dataset):
    def __init__(self, args, dirs: list):
        self.filename_list = []
        self.sr = args.sr
        for dir in dirs:
            self.filename_list.extend(utils.get_filename_list(dir))
        self.n_mels = args.n_mels
        self.n_frames = args.n_frames
        self.n_hop_frames = args.n_hop_frames
        self.dims = self.n_mels * self.n_frames
        self.wav2mel = utils.Wave2Mel(sr=self.sr, n_mels=self.n_mels)
        # self.mean_std_dict = np.load('mean_std.npy', allow_pickle=True).item()

    def __getitem__(self, item):
        filename = self.filename_list[item]
        return self.transform(filename)

    def transform(self, filename):
        # machine_type = filename.split('/')[-3]
        # mean, std = self.mean_std_dict[machine_type][0], self.mean_std_dict[machine_type][1]
        # label = utils.get_label(filename, machine_type, self.machine_factors)
        (x, _) = librosa.core.load(filename, sr=self.sr, mono=True)
        x = x[:self.sr * 10]  # (1, audio_length)
        x_wav = torch.from_numpy(x)
        x_mel = self.wav2mel(x_wav).squeeze()
        n_vectors = x_mel.shape[1] - self.n_frames + 1
        vectors = torch.zeros((n_vectors, self.dims))
        for t in range(self.n_frames):
            vectors[:, self.n_mels * t: self.n_mels * (t + 1)] = x_mel[:, t: t + n_vectors].T
        vectors = vectors[:: self.n_hop_frames, :]
        vectors = vectors.reshape(-1, self.n_mels, self.n_frames).transpose(2, 1)
        # x_mel = (x_mel - mean) / std
        # print(x.shape)
        return vectors

    def __len__(self):
        return len(self.filename_list)