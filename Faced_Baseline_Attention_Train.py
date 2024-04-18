# import packages
from cl_sster import cl_sster
import numpy as np
import os
import scipy.io as sio
from postprocessing_utils import calc_isc, calc_isc_train
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from postprocessing_utils import plot_dendrogram
import torch
import mne
# parameters
fs = 125
epochs_pretrain = 50
timeLen = 5
# Load video data
# datadir = r'D:\Data\Emotion\FACED\Processed_data_filter_epoch_0.50_40_manualRemove_ica'
# datadir = '/mnt/repo3/zqz/Emotion_EEG_Dataset/SEED_SJTU/SEED-V/AutoICA_Processed/SEED_V/Processed_data_filter_0.50_47.00_AutoICA_Cus_Threshold/data'
# datadir = './data/SEED_V/data'
datadir = './data/Faced/Processed_data_filter_epoch_0.50_40_manualRemove_ica'
print(datadir)
video_len = [81,63,73,78,69,90,56,60,105,45,60,81,35,44,38,43,55,69,73,129,77,75,34,37,67,63,54,77]
n_points = np.array(video_len).astype(int) * fs

n_vids = len(video_len)
data_paths = os.listdir(datadir)
data_paths.sort()
n_subs = 123
chn = 30
count = 0
data = np.zeros((n_subs, np.sum(n_points), chn))
for idx, path in enumerate(data_paths):
    if path[:3] == 'sub':
        print(f'Reading Sub {count}/{n_subs}')
        data[count,:,:] = sio.loadmat(os.path.join(datadir, path))['data_all_cleaned'].transpose()
        count += 1
print(data.shape)
from tqdm import tqdm
# Normalization without outliers
print('Normalizing')
for sub in tqdm(range(n_subs)):
    thr = 30 * np.median(abs(data[sub]))
    data[sub] = (data[sub] - np.mean(data[sub][data[sub] < thr])) / np.std(data[sub][data[sub] < thr])

# Train the cross-validation model

train_info = '_Faced_vid_'
epochs_pretrain = 50
timeLen = 5
from cl_sster import cl_sster
print('Initializing...')
gpu_index = 7
torch.cuda.set_device(gpu_index)
cl_model_ = cl_sster(n_folds=20, weight_decay=0.00015, epochs_pretrain=epochs_pretrain, timeLen=timeLen, fs=fs, data_type='video', model='attention', train_info = train_info, gpu_index=gpu_index)
cl_model = cl_model_
print('Loading Data')
cl_model.load_data(data, n_points) # fs: sampling rate
print('Data Loaded')
cl_model.train_cl_sster()