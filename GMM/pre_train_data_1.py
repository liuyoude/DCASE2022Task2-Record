import joblib
import os
import re
import common as com
import torch
import librosa
import torchaudio
import tqdm
import numpy as np
import MFNmodel

param = com.yaml_load()
def save_pre_data_file(pre_data_dir):
    # database dirs
    dirs = com.select_dirs(param=param)  # ./dev_data
    pre_data_dir = param['pre_data_dir']  # ./database/pre_data
    os.makedirs(f'{pre_data_dir}', exist_ok=True)
    save_file_path = f'{pre_data_dir}/\\file_path.db'
    pre_data = {
        'path_src': [],
        'label_src': [],
        'path_tgt': [],
        'label_tgt': [],
    }
    for index, target_dir in enumerate(sorted(dirs)):   # fan /.../valve
        print('\n' + '='*20)
        print(f'[{index+1}/{len(dirs)}] {target_dir} preprocessing...')
        machine_type = os.path.split(target_dir)[1]
        section_names = com.get_section_names(target_dir, dir_name="./train")   # ['section_00', 'section_01', 'section_02']
        for section_name in section_names:
            files, labels = com.file_list_generator(target_dir = target_dir,
                                                    dir_name='./train',
                                                    section_name =section_name,
                                                    ext="wav")
            # for i in range(files.size):
            for file in files:
                id_str = re.findall('section_[0-9][0-9]', file)
                id = int(id_str[0][-1])
                label = int(param['ID_factor'][machine_type] * 6 + id)
                if 'source' in file:
                    pre_data['path_src'].append(file)
                    pre_data['label_src'].append(label)
                elif 'target' in file:
                    pre_data['path_tgt'].append(file)
                    pre_data['label_tgt'].append(label)

        with open(save_file_path, 'wb') as f:
            joblib.dump(pre_data, f)
        print(f'{machine_type}[{index+1}/{len(dirs)}] had saved')

def Extract_mean_covariance(pre_data_dir):
    dirs = com.select_dirs(param=param)

    for idx, target_dir in enumerate(dirs):
        section_id = com.get_section_names(target_dir=target_dir,  # ['section_00', 'section_01', 'section_02']
                                           dir_name="train")
        machine_type = os.path.split(target_dir)[1]
        print('==============================================================================================================')
        dit = {
            'covariance_I_list': [0 for _ in range(6)],
            'src_mean_list': [0 for _ in range(6)],
            'tgt_mean_list': [0 for _ in range(6)],
        }
        for section_name in section_id:
            files, labels = com.file_list_generator(target_dir=target_dir,
                                                    dir_name='train',
                                                    ext="wav",
                                                    section_name=section_name)
            label = int(section_name[-1])
            # label = int(param['ID_factor'][machine_type] * 6 + id)
            label = torch.from_numpy(np.array(label)).long().cuda()

            idx_src, idx_tgt = 0, 0
            for file_idx, file_path in enumerate(files):
                if 'source' in file_path: idx_src+=1
                elif 'target' in file_path: idx_tgt+=1
            data_train_srcmean = torch.zeros(idx_src, param['feature']['n_mels'])   # torch.Size([990, 128])
            data_train_tgtmean = torch.zeros(idx_tgt, param['feature']['n_mels'])   # torch.Size([10, 128])
            data_train_mean = torch.zeros(idx_src+idx_tgt, param['feature']['n_mels'])   # torch.Size([1000, 128])
            #  mean vectors
            idx_src, idx_tgt = 0, 0
            for file_idx, file_path in tqdm.tqdm(enumerate(files), total=len(files)):
                (x, _) = librosa.core.load(file_path, sr=param['feature']['sr'], mono=True)
                x_wav = x[None, None, :param['feature']['sr'] * 10]  # (1, audio_length)
                x_wav = torch.from_numpy(x_wav).cuda()
                x_wav = x_wav.float()
                x_mel = x[:param['feature']['sr'] * 10]  # (1, audio_length)
                x_mel = torch.from_numpy(x_mel)
                mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=param['feature']['sr'],
                                                                     win_length=param['feature']['win_length'],
                                                                     hop_length=param['feature']['hop_length'],
                                                                     n_fft=param['feature']['n_fft'],
                                                                     n_mels=param['feature']['n_mels'],
                                                                     power=param['feature']['power'])
                amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='power')
                melspec = amplitude_to_db(mel_transform(x_mel)).cuda()
                x_mel_mean = torch.mean(melspec, dim=1) # 128
                if 'source' in file_path:
                    data_train_srcmean[idx_src, :] = x_mel_mean   # (990, 128)
                    idx_src += 1
                elif 'target' in file_path:
                    data_train_tgtmean[idx_tgt, :] = x_mel_mean
                    idx_tgt += 1
                data_train_mean[file_idx, :] = x_mel_mean   # (1000, 128)
            src_mean = torch.mean(data_train_srcmean, dim=0)    # 128
            tgt_mean = torch.mean(data_train_tgtmean, dim=0)    # 128
            dit['src_mean_list'][label] = src_mean.cuda()
            dit['tgt_mean_list'][label] = tgt_mean.cuda()
            # data_train_mean = np.append(data_train_srcmean, data_train_tgtmean, axis=0)
            data_train_mean = data_train_mean.numpy()
            data_train_conv = np.cov(data_train_mean, rowvar=False)
            data_train_conv_I = np.linalg.inv(data_train_conv)
            data_train_conv_I= torch.FloatTensor(data_train_conv_I)
            dit['covariance_I_list'][label] = data_train_conv_I.cuda()
        save_file_path = f'{pre_data_dir}/\\mahanobis_path_{machine_type}.db'
        with open(save_file_path, 'wb') as f:
            joblib.dump(dit, f)
    return dit

if __name__ == '__main__':
    pre_data_dir = param['pre_data_dir']
    save_pre_data_file(pre_data_dir)
    # Extract_mean_covariance(pre_data_dir)




