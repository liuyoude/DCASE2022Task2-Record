import os
import torch
import numpy as np
import sklearn
import tqdm
import common as com
from sklearn import metrics
import torchaudio
import librosa
import joblib

param = com.yaml_load()
if __name__ == '__main__':
    os.makedirs(param['result_directory'], exist_ok=True)
    csv_lines = []
    performance_over_all = []
    file_name = param['pre_data_dir'] + f'/mahanobis_path.db'
    with open(file_name, 'rb') as f:
        dit = joblib.load(f)
    dirs = com.select_dirs(param=param)
    for idx, target_dir in enumerate(dirs):
        print('\n' + '='*20)
        print(f'[{idx+1}/{len(dirs)}] {target_dir} ')
        machine_type = os.path.split(target_dir)[1]
        if machine_type not in param['process_machines']:
            continue
        print("============== MODEL LOAD ==============")
        csv_lines.append([machine_type])
        csv_lines.append(["section", "AUC", "pAUC"])
        performance = []
        section_id = com.get_section_names(target_dir=target_dir, dir_name="test")   # ['section_00', 'section_01', 'section_02']
        for section_name in section_id:
            anomaly_score_csv = "{result}/{model_name}/anomaly_score_{machine_type}_{section_name}_{dir_name}.csv".format(
                result=param["result_directory"],
                model_name='MA',
                machine_type=machine_type,
                section_name=section_name,
                dir_name="test")
            anomaly_score_list = []
            test_files, y_true = com.file_list_generator(target_dir=target_dir,
                                                         dir_name="test",
                                                         section_name=section_name)
            label = int(section_name[-1])
            label = int(param['ID_factor'][machine_type] * 6 + label)
            label = torch.from_numpy(np.array(label)).long().cuda()
            domain_list = []
            y_pred = [0. for _ in test_files]
            for file_idx, file_path in enumerate(test_files):
                domain_list.append("source" if "source" in file_path else "target")
                (x, _) = librosa.core.load(file_path, sr=param['feature']['sr'], mono=True)
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

                melspec_mean = torch.mean(melspec, dim=1)
                if 'source' in file_path:
                    mahano_distance = torch.sqrt(torch.matmul(torch.matmul(melspec_mean - dit['src_mean_list'][label], dit['covariance_I_list'][label]), (melspec_mean - dit['src_mean_list'][label]).T))
                elif 'target' in file_path:
                    mahano_distance = torch.sqrt(torch.matmul(torch.matmul(melspec_mean - dit['tgt_mean_list'][label], dit['covariance_I_list'][label]), (melspec_mean - dit['tgt_mean_list'][label]).T))
                y_pred[file_idx] = mahano_distance.cpu().numpy()
                anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])
            com.save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_list)
            com.logger.info("anomaly score result ->  {}".format(anomaly_score_csv))

            # extract scores for calculation of AUC (source) and AUC (target)
            y_true_s_auc = [y_true[idx] for idx in range(len(y_true)) if domain_list[idx] == "source" or y_true[idx] == 1]
            y_pred_s_auc = [y_pred[idx] for idx in range(len(y_true)) if domain_list[idx] == "source" or y_true[idx] == 1]
            y_true_t_auc = [y_true[idx] for idx in range(len(y_true)) if domain_list[idx] == "target" or y_true[idx] == 1]
            y_pred_t_auc = [y_pred[idx] for idx in range(len(y_true)) if domain_list[idx] == "target" or y_true[idx] == 1]

            # compute auc and pAuc
            auc_s = metrics.roc_auc_score(y_true_s_auc, y_pred_s_auc)
            auc_t = metrics.roc_auc_score(y_true_t_auc, y_pred_t_auc)
            p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=param["max_fpr"])
            csv_lines.append([section_name.split("_", 1)[1], auc_s, auc_t, p_auc])
            performance.append([auc_s, auc_t, p_auc])
            print("\n============ END OF TEST FOR A MACHINE ID ============")
            # anomaly_score_csv = "{result}/{model_name}/anomaly_score_{machine_type}_{section_name}_{dir_name}.csv".format(
            #     result=param["result_directory"],
            #     model_name='MA',
            #     machine_type=machine_type,
            #     section_name=section_name,
            #     dir_name="test")
            # anomaly_score_list = []
            # # print("\n============== BEGIN TEST FOR A SECTION ==============")
            # # MA = []
            # # MA_1 = []
            # # MA_2 = []
            # # for train_section_name in section_id:
            # #     # source
            # #     files, labels = com.file_list_generator(target_dir=target_dir,
            # #                                             dir_name='train',
            # #                                             ext="wav",
            # #                                             section_name=train_section_name)
            # #     data_train_sormean = []
            # #     data_train_tarmean = []
            # #     #  mean vectors
            # #     for file_idx, file_path in tqdm.tqdm(enumerate(files), total=len(files)):
            # #         data = com.file_to_vectors(file_path,
            # #                                    n_mels=param['feature']['n_mels'],
            # #                                    n_frames=1,
            # #                                    n_fft=param['feature']['n_fft'],
            # #                                    hop_length=param['feature']['hop_length'],
            # #                                    power=param['feature']['power'])
            # #         data = torch.from_numpy(data).float()   # (313, 128)
            # #         data = data.numpy()
            # #         m = np.mean(data, axis=0)
            # #         data_train_sormean.append(m)
            # #     # target
            # #     target_files, target_labes = com.file_list_generator(target_dir=target_dir,
            # #                                                          dir_name='train_target',
            # #                                                          ext="wav",
            # #                                                          section_name=train_section_name)
            # #     #  mean vectors
            # #     for file_idx, file_path in tqdm.tqdm(enumerate(target_files), total=len(target_files)):
            # #         data = com.file_to_vectors(file_path,
            # #                                    n_mels=param['feature']['n_mels'],
            # #                                    n_frames=1,
            # #                                    n_fft=param['feature']['n_fft'],
            # #                                    hop_length=param['feature']['hop_length'],
            # #                                    power=param['feature']['power'])
            # #         data = torch.from_numpy(data).float()   # (313, 128)
            # #         data = data.numpy()
            # #         m = np.mean(data, axis=0)
            # #         data_train_tarmean.append(m)
            # #     sor_mean = np.mean(data_train_sormean, axis=0)
            # #     tar_mean = np.mean(data_train_tarmean, axis=0)
            # #     data_train_sormean = np.append(data_train_sormean, data_train_tarmean, axis=0)
            # #     data_train_conv = np.cov(data_train_sormean, rowvar=False)
            # #     data_train_conv_I = np.linalg.inv(data_train_conv)
            # #
            # #     for file_idx, file_path in tqdm.tqdm(enumerate(test_files), total=len(test_files)):
            # #         data = com.file_to_vectors(file_path,
            # #                                    n_mels=param['feature']['n_mels'],
            # #                                    n_frames=1,
            # #                                    n_fft=param['feature']['n_fft'],
            # #                                    hop_length=param['feature']['hop_length'],
            # #                                    power=param['feature']['power'])
            # #         data = torch.from_numpy(data).float()   # (313, 128)
            # #         data = data.numpy()
            # #         m = np.mean(data,axis=0)    # 128
            # #         if dir_name == "source_test":
            # #             delta = m - sor_mean  #   128  # 数据中的行 减 每列的平均值
            # #         elif dir_name == "target_test":
            # #             delta = m - tar_mean
            # #         left_term = np.dot(delta, data_train_conv_I) # 128
            # #         di = np.dot(left_term, delta.T) # 异常程度  (309, 309)
            # #         di = np.sqrt(di)
            # #         if train_section_name == 'section_00':
            # #             MA.append(di)
            # #         elif train_section_name == 'section_01':
            # #             MA_1.append(di)
            # #         elif train_section_name == 'section_02':
            # #             MA_2.append(di)
            # # for i in range(len(MA)):
            # #     if MA[i] > MA_1[i]:
            # #         MA[i] = MA_1[i]
            # #     elif MA[i] > MA_2[i]:
            # #         MA[i] = MA_2[i]
            # y_pred = [0. for _ in test_files]
            # for file_idx, file_path in tqdm.tqdm(enumerate(test_files), total=len(test_files)):
            #     y_pred[file_idx] = MA[file_idx]
            #     anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])
            # com.save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_list)
            # com.logger.info("anomaly score result ->  {}".format(anomaly_score_csv))
            #
            # auc = sklearn.metrics.roc_auc_score(y_true, y_pred)
            # p_auc = sklearn.metrics.roc_auc_score(y_true, y_pred, max_fpr=param['max_fpr'])
            # csv_lines.append([section_name.split('_', 1)[1], auc, p_auc])
            # performance.append([auc, p_auc])
            # com.logger.info("AUC : {}".format(auc))
            # com.logger.info("pAUC : {}".format(p_auc))
            # print("\n============ END OF TEST FOR A MACHINE ID ============")

        averaged_performance = np.mean(np.array(performance, dtype=float), axis=0)
        csv_lines.append(['Average'] + list(averaged_performance))
        csv_lines.append([])

    result_path = "{result}/MA/{file_name}".format(result=param["result_directory"], file_name=param["result_file"])
    com.logger.info("AUC and pAUC results -> {}".format(result_path))
    com.save_csv(save_file_path=result_path, save_data=csv_lines)
