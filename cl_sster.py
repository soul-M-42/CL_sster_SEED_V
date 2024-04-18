import numpy as np
import os
import torch
from torch.utils.data import DataLoader
import argparse
import random
from io_utils import Dataset_raw, TrainSampler, ValSampler
from model import ConvNet_avgPool_share, ConvNet_avgPool_share_nopool, ConvNet_attention_simple
from train_simCLR import SimCLR
import pickle
from postprocessing_utils import calc_corr
import torch.nn.functional as F
import scipy
from torch.utils.data import DataLoader, Dataset


class FeatureDataset(Dataset):
    def __init__(self, features, labels, non_zero_indices):
        self.features = torch.from_numpy(features).float()  # 将特征数据转换为float32
        self.labels = torch.from_numpy(labels).long()       # 将标签数据转换为long
        self.non_zero_indices = non_zero_indices
        self.input_shape = None
        self.n_subs = self.labels.shape[0]
        self.n_samples = self.labels.shape[1]
        self.classes = 5

    def __len__(self):
        return self.n_subs * self.n_samples

    def __getitem__(self, idx):
        feature = self.features[idx//self.n_samples][idx%self.n_samples]
        label = self.labels[idx//self.n_samples][idx%self.n_samples]
        onehot = np.zeros(self.classes)
        onehot[label] = 1
        return feature, onehot



class cl_sster(object):
    def __init__(self, n_folds=10, timeLen=30, weight_decay=0.1, epochs_pretrain=50, timeFilterLen=30, avgPoolLen=15, device='cuda', gpu_index=0, randSeed=7, data_type='simulation', model='vanilla', fs=128, train_info = '', labels = None):
        self.n_folds = n_folds
        self.timeLen = timeLen
        self.weight_decay = weight_decay
        self.epochs_pretrain = epochs_pretrain
        self.timeFilterLen = timeFilterLen
        self.avgPoolLen = avgPoolLen
        self.device = device
        self.gpu_index = gpu_index
        self.randSeed = randSeed
        self.timeStep = int(timeLen / 2)
        self.learning_rate = 0.0007
        self.temperature = 0.07
        self.n_timeFilters = 16
        self.n_spatialFilters = 16
        self.activ = 'relu'
        self.fs = fs
        self.model = model
        self.labels = labels
        self.train_info = train_info

        
        
        random.seed(randSeed)
        np.random.seed(randSeed)
        torch.manual_seed(randSeed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        self.max_tol_pretrain = self.epochs_pretrain
        if self.device == 'cuda':
            print('is cuda available:', torch.cuda.is_available())
            torch.device('cuda')
            torch.cuda.set_device(self.gpu_index)

        self.stratified = ['initial', 'middle']
        
        self.save_dir = 'results/%s/%snfold%d_timeLen%d_wd%.6f_epochs%d_tfLen%d_avgPool%d_timeStep%.3f' % (
            data_type, self.train_info, self.n_folds, self.timeLen, self.weight_decay, self.epochs_pretrain, self.timeFilterLen, self.avgPoolLen, self.timeStep)
        print(self.train_info)
        if self.n_folds == 1:
            self.save_dir = self.save_dir + '_all'
        else:
            self.save_dir = self.save_dir + '_cv'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        print('The results will be saved to:', self.save_dir)

        
        
    def load_data(self, data, n_points):
        # data.shape: [n_subs, n_timepoints, n_channs]
        # fs: sampling rate
        self.data = data
        self.n_subs, self.n_channs = data.shape[0], data.shape[2]
        self.n_per = round(self.n_subs / self.n_folds)
        self.n_trials = len(n_points)

        n_points_remain = np.zeros(len(n_points), dtype=int)
        n_samples = np.zeros(len(n_points), dtype=int)
        for i in range(len(n_samples)):
            n_samples[i] = int((n_points[i] - self.timeLen * self.fs) / (self.timeStep * self.fs) + 1)
            n_points_remain[i] = int(n_points[i] - n_samples[i] * self.timeStep * self.fs)
        self.n_samples = n_samples
        self.n_points = n_points
        self.n_points_remain = n_points_remain

        
    def train_cl_sster(self):
        n_folds = self.n_folds
        # Initialize the results directory
        results = {}
        results['best_epoch'] = np.zeros(n_folds).astype(int)
        results['train_loss_history'] = np.zeros((n_folds, self.epochs_pretrain))
        results['val_loss_history'] = np.zeros((n_folds, self.epochs_pretrain))
        results['train_top1_history'] = np.zeros((n_folds, self.epochs_pretrain))
        results['val_top1_history'] = np.zeros((n_folds, self.epochs_pretrain))
        results['train_top5_history'] = np.zeros((n_folds, self.epochs_pretrain))
        results['val_top5_history'] = np.zeros((n_folds, self.epochs_pretrain))
        
        if n_folds == 1:
            val_fold = 0
            print('Train with all data')

            n_per = self.n_subs
            train_sub = np.arange(self.n_subs)
            val_sub = np.arange(10) # Use a few subjects as fake validation subjects to ensure the following function can be directly used
            data_train = self.data[train_sub].reshape(-1, self.data.shape[-1])
            data_val = self.data.reshape(-1, self.data.shape[-1])
            print('data_train.shape, data_val.shape', data_train.shape, data_val.shape)

            trainset = Dataset_raw(data_train, self.timeLen, self.timeStep, self.n_samples, self.n_points_remain, self.fs, len(train_sub))
            valset = Dataset_raw(data_val, self.timeLen, self.timeStep, self.n_samples, self.n_points_remain, self.fs, self.n_subs)

            train_sampler = TrainSampler(len(train_sub), 1, batch_size=self.n_trials, n_samples=self.n_samples, phase='train')
            val_sampler = ValSampler(self.n_subs, self.n_samples, train_sub, val_sub) 

            train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, pin_memory=True, num_workers=0)
            val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, pin_memory=True, num_workers=0)
            if self.model == 'vanilla':
                model = ConvNet_avgPool_share(self.n_timeFilters, self.timeFilterLen, self.n_spatialFilters, self.avgPoolLen, self.n_channs, self.stratified, self.activ, 'train').to(self.device)
            elif self.model == 'attention':
                model = ConvNet_attention_simple(n_timeFilters=4, \
                                            timeFilterLen0=30, \
                                            n_msFilters=2, \
                                            timeFilterLen=2, \
                                            avgPoolLen=15, \
                                            timeSmootherLen=3, \
                                            n_channs=self.n_channs, \
                                            stratified=['initial', 'middle1', 'middle2'],\
                                            multiFact=1, 
                                            activ='relu', 
                                            temp=1.0, 
                                            saveFea=False,
                                            phase='train').to(self.device)
            print(model)
            para_num = sum([p.data.nelement() for p in model.parameters()])
            print('Total number of parameters:', para_num)

            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.epochs_pretrain, gamma=0.8, last_epoch=-1, verbose=False) # No use
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

            if self.activ == 'logvar': # No use
                stratified_loss = 'minmax'
            else:
                stratified_loss = 'no'

            with torch.cuda.device(self.gpu_index):
                simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler,
                                log_dir=os.path.join(self.save_dir, str(val_fold)), stratified=stratified_loss,
                                device=self.device, temperature=self.temperature, epochs_pretrain=self.epochs_pretrain, max_tol_pretrain=self.max_tol_pretrain)
                model, best_epoch, train_top1_history, val_top1_history, train_top5_history, val_top5_history, train_loss_history, val_loss_history = simclr.train(
                    train_loader, val_loader, False)

            results['best_epoch'][val_fold] = best_epoch
            results['train_loss_history'][val_fold,:] = train_loss_history
            results['val_loss_history'][val_fold,:] = val_loss_history
            results['train_top1_history'][val_fold,:] = train_top1_history
            results['val_top1_history'][val_fold,:] = val_top1_history
            results['train_top5_history'][val_fold,:] = train_top5_history
            results['val_top5_history'][val_fold,:] = val_top5_history

            with open(os.path.join(self.save_dir, 'results.pkl'), 'wb') as f:
                pickle.dump(results, f)

            train_top1_best = results['train_top1_history'][np.arange(n_folds), results['best_epoch']]
            train_top5_best = results['train_top5_history'][np.arange(n_folds), results['best_epoch']]
            val_top1_best = results['val_top1_history'][np.arange(n_folds), results['best_epoch']]
            val_top5_best = results['val_top5_history'][np.arange(n_folds), results['best_epoch']]
            print('train top1 mean: %.3f, train top1 std: %.3f; val top1 mean: %.3f, val top1 std: %.3f' % (
                np.mean(train_top1_best), np.std(train_top1_best), np.mean(val_top1_best), np.std(val_top1_best)
            ))
            
        else:
            print('Conduct cross-validation')
            
            n_per = int(np.floor(self.n_subs / n_folds))
            for val_fold in range(n_folds):
                if val_fold < n_folds-1:
                    val_sub = np.arange(val_fold*n_per, (val_fold+1)*n_per)
                else:
                    val_sub = np.arange(val_fold*n_per, self.n_subs)
                print('val sub', val_sub)
                train_sub = list(set(np.arange(self.n_subs)) - set(val_sub))
                print('train sub', train_sub)
                data_train = self.data[train_sub].reshape(-1, self.data.shape[-1])
                data_val = self.data.reshape(-1, self.data.shape[-1])
                print('data_train.shape, data_val.shape', data_train.shape, data_val.shape)

                trainset = Dataset_raw(data_train, self.timeLen, self.timeStep, self.n_samples, self.n_points_remain, self.fs, len(train_sub))
                valset = Dataset_raw(data_val, self.timeLen, self.timeStep, self.n_samples, self.n_points_remain, self.fs, self.n_subs)

                train_sampler = TrainSampler(len(train_sub), 1, batch_size=self.n_trials, n_samples=self.n_samples, phase='train')
                val_sampler = ValSampler(self.n_subs, self.n_samples, train_sub, val_sub) 

                train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, pin_memory=True, num_workers=0)
                val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, pin_memory=True, num_workers=0)
                if self.model == 'vanilla':
                    model = ConvNet_avgPool_share(self.n_timeFilters, self.timeFilterLen, self.n_spatialFilters, self.avgPoolLen, self.n_channs, self.stratified, self.activ, 'train').to(self.device)
                elif self.model == 'attention':
                    model = ConvNet_attention_simple(n_timeFilters=8, \
                                                timeFilterLen0=30, \
                                                n_msFilters=2, \
                                                timeFilterLen=2, \
                                                avgPoolLen=15, \
                                                timeSmootherLen=3, \
                                                n_channs=self.n_channs, \
                                                stratified=['initial', 'middle1', 'middle2'],\
                                                multiFact=2, 
                                                activ='relu', 
                                                temp=1.0, 
                                                saveFea=False,
                                                phase='train').to(self.device)
                print(model)
                para_num = sum([p.data.nelement() for p in model.parameters()])
                print('Total number of parameters:', para_num)

                optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.epochs_pretrain, gamma=0.8, last_epoch=-1, verbose=False) # No use

                if self.activ == 'logvar': # No use
                    stratified_loss = 'minmax'
                else:
                    stratified_loss = 'no'

                with torch.cuda.device(self.gpu_index):
                    simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler,
                                    log_dir=os.path.join(self.save_dir, str(val_fold)), stratified=stratified_loss,
                                    device=self.device, temperature=self.temperature, epochs_pretrain=self.epochs_pretrain, max_tol_pretrain=self.max_tol_pretrain)
                    model, best_epoch, train_top1_history, val_top1_history, train_top5_history, val_top5_history, train_loss_history, val_loss_history = simclr.train(
                        train_loader, val_loader, False)

                results['best_epoch'][val_fold] = best_epoch
                results['train_loss_history'][val_fold,:] = train_loss_history
                results['val_loss_history'][val_fold,:] = val_loss_history
                results['train_top1_history'][val_fold,:] = train_top1_history
                results['val_top1_history'][val_fold,:] = val_top1_history
                results['train_top5_history'][val_fold,:] = train_top5_history
                results['val_top5_history'][val_fold,:] = val_top5_history

                with open(os.path.join(self.save_dir, 'results.pkl'), 'wb') as f:
                    pickle.dump(results, f)

            train_top1_best = results['train_top1_history'][np.arange(n_folds), results['best_epoch']]
            train_top5_best = results['train_top5_history'][np.arange(n_folds), results['best_epoch']]
            val_top1_best = results['val_top1_history'][np.arange(n_folds), results['best_epoch']]
            val_top5_best = results['val_top5_history'][np.arange(n_folds), results['best_epoch']]
            print('train top1 mean: %.3f, train top1 std: %.3f; val top1 mean: %.3f, val top1 std: %.3f' % (
                np.mean(train_top1_best), np.std(train_top1_best), np.mean(val_top1_best), np.std(val_top1_best)
            ))
            
    def get_hidden(self, fold, isNorm=False):
        n_points_cum_ori = np.concatenate((np.array([0]), np.cumsum(self.n_points))).astype(int)
        n_points_cum = np.concatenate((np.array([0]), np.cumsum((self.n_points-self.timeFilterLen+1)//self.avgPoolLen))).astype(int)

        n_latent_dims = self.n_timeFilters * self.n_spatialFilters

        state_dict = torch.load(os.path.join(self.save_dir, str(fold), 'checkpoint_%04d.pth.tar' % (self.epochs_pretrain-1)), map_location='cuda:0')
        print('checkpoint:', os.path.join(self.save_dir, str(fold), 'checkpoint_%04d.pth.tar' % (self.epochs_pretrain-1)))
        state_dict = state_dict['state_dict']
        if self.model == 'vanilla':
            model = ConvNet_avgPool_share(self.n_timeFilters, self.timeFilterLen, self.n_spatialFilters, self.avgPoolLen, self.n_channs, self.stratified, self.activ, 'infer').double()
        elif self.model == 'attention':
            model = ConvNet_attention_simple(n_timeFilters=8, \
                                                timeFilterLen0=30, \
                                                n_msFilters=2, \
                                                timeFilterLen=2, \
                                                avgPoolLen=15, \
                                                timeSmootherLen=3, \
                                                n_channs=self.n_channs, \
                                                stratified=['initial', 'middle1', 'middle2'],\
                                                multiFact=2, 
                                                activ='relu', 
                                                temp=1.0, 
                                                saveFea=False,
                                                phase='infer').double()
        model.load_state_dict(state_dict, strict=False)

        out = np.zeros((self.n_subs, n_latent_dims, n_points_cum[-1]))
        # sub_order = np.concatenate((train_sub, val_sub))
        count = 0
        with torch.cuda.device(self.gpu_index):
            model = model.to(self.device)
            for sub in np.arange(self.n_subs):
                # print(sub)
                for tr in range(self.n_trials):
                    data_now = self.data[sub,n_points_cum_ori[tr]:n_points_cum_ori[tr+1],:]
                    data_tensor = torch.from_numpy(data_now).to(self.device)
                    data_tensor.requires_grad = False
                    out_one = model(torch.unsqueeze(torch.unsqueeze(data_tensor.permute(1,0),0),0))
                    print(out_one.shape)
                    out_one = out_one.reshape(out_one.shape[0], n_latent_dims, -1)
                    if isNorm:
                        out_one = F.normalize(out_one, dim=1)
                    out_one = torch.squeeze(out_one)
                    out_one = out_one.detach().cpu().numpy().reshape(-1, out_one.shape[-1])
                    print(out_one.shape)
                    print(n_points_cum[tr], n_points_cum[tr+1])
                    out[count, :, n_points_cum[tr]:n_points_cum[tr]+out_one.shape[-1]] = out_one
                count += 1
        self.out = out
        self.n_points_cum = n_points_cum
        return out, n_points_cum
    
    def ref_feature(self, fold, n_seg, isNorm=False):
        self.n_seg = n_seg
        self.n_latent_dims = self.n_timeFilters * self.n_spatialFilters
        n_points_cum_ori = np.concatenate((np.array([0]), np.cumsum(self.n_points))).astype(int)
        n_points_cum = np.concatenate((np.array([0]), np.cumsum((self.n_points-self.timeFilterLen+1)//self.avgPoolLen))).astype(int)
        n_latent_dims = self.n_timeFilters * self.n_spatialFilters
        file_names = os.listdir(os.path.join(self.save_dir, str(fold)))
        best_model_names = [file_name for file_name in file_names if 'best' in file_name]
        if best_model_names:
            # 获取最新的一个模型文件名
            latest_best_model_name = sorted(best_model_names)[-1]

            # 加载模型
            state_dict = torch.load(os.path.join(self.save_dir, str(fold), latest_best_model_name), map_location='cuda:0')
            # 使用 state_dict 进行后续操作
        else:
            # 如果没有符合条件的模型文件
            print("No model with 'best' in its file name found.")
        # state_dict = torch.load(os.path.join(self.save_dir, str(fold), 'checkpoint_%04d.pth.tar' % (self.epochs_pretrain-1)), map_location='cuda:0')
        # print('checkpoint:', os.path.join(self.save_dir, str(fold), 'checkpoint_%04d.pth.tar' % (self.epochs_pretrain-1)))
        print('checkpoint:', os.path.join(self.save_dir, str(fold), latest_best_model_name))
        state_dict = state_dict['state_dict']
        if self.model == 'vanilla':
            model = ConvNet_avgPool_share(self.n_timeFilters, self.timeFilterLen, self.n_spatialFilters, self.avgPoolLen, self.n_channs, self.stratified, self.activ, 'infer').double()
        elif self.model == 'attention':
            model = ConvNet_attention_simple(n_timeFilters=8, \
                                                timeFilterLen0=30, \
                                                n_msFilters=2, \
                                                timeFilterLen=2, \
                                                avgPoolLen=15, \
                                                timeSmootherLen=3, \
                                                n_channs=self.n_channs, \
                                                stratified=['initial', 'middle1', 'middle2'],\
                                                multiFact=2, 
                                                activ='relu', 
                                                temp=1.0, 
                                                saveFea=False,
                                                phase='infer').double()
        model.load_state_dict(state_dict, strict=False)
        
        self.n_per = round(self.n_subs / self.n_folds)
        if fold < self.n_folds-1:
            val_sub = np.arange(self.n_per*fold, self.n_per*(fold+1))
        else:
            val_sub = np.arange(self.n_per*fold, self.n_subs)
        val_sub = [int(val) for val in val_sub]
        print('val', val_sub)

        train_sub = np.array(list(set(np.arange(self.n_subs)) - set(val_sub)))
        print('train', train_sub)

        # sub_order = np.concatenate((train_sub, val_sub))
        count = 0
        with torch.cuda.device(self.gpu_index):
            model = model.to(self.device)
            sample_points = []
            sample_labels = []
            for tr in range(self.n_trials):
                sample_i_points = list(range(n_points_cum_ori[tr], n_points_cum_ori[tr+1], self.timeStep * self.fs))
                sample_points = sample_points + sample_i_points
                sample_labels = sample_labels + len(sample_i_points) * [self.labels[tr]]
            
            sample_points = sample_points[:-5]
            sample_labels = sample_labels[:-5]

            self.feature_points = sample_points
            for sub in np.arange(self.n_subs):
                # print(sub)
                for sample_idx, sample_i_st in enumerate(sample_points):
                    sample_i_ed = sample_i_st + self.timeLen * self.fs
                    data_now = self.data[sub,sample_i_st:sample_i_ed,:]
                    if(data_now.shape[0] != sample_i_ed-sample_i_st):
                        # print('at the end.')
                        continue
                    data_tensor = torch.from_numpy(data_now).to(self.device)
                    data_tensor = torch.unsqueeze(torch.unsqueeze(data_tensor.permute(1,0),0),0)
                    data_tensor.requires_grad = False
                    segment_length = data_tensor.shape[-1] // self.n_seg  # 每段的长度
                    for seg in range(self.n_seg):
                        start_idx = seg * segment_length  # 当前段的起始索引
                        end_idx = (seg + 1) * segment_length  # 当前段的结束索引
                        if seg == self.n_seg - 1:  # 如果是最后一段，则结束索引为原始数据的末尾
                            end_idx = data_tensor.shape[-1]

                        segment_data = data_tensor[:, :, :, start_idx:end_idx]  # 当前段的数据
                        out_one = model(segment_data)
                        out_one = out_one.reshape(out_one.shape[0], n_latent_dims, -1)
                        if isNorm:
                            out_one = F.normalize(out_one, dim=1)
                        out_one = torch.squeeze(out_one)
                        out_one = out_one.detach().cpu().numpy().reshape(-1, out_one.shape[-1])
                        # print(out_one)
                        # detect zero channel
                        threshold = 0
                        variances = np.var(out_one, 1)
                        indices = variances > threshold
                        if not hasattr(self, 'non_zero_indices'):
                            self.non_zero_indices = {}
                            for fold in range(self.n_folds):
                                self.non_zero_indices[fold] = np.ones(self.n_latent_dims).astype(bool)
                        self.non_zero_indices[fold] = np.logical_and(self.non_zero_indices[fold], indices)
                        if not hasattr(self, 'feature_raw'):
                            self.feature_raw = np.zeros((self.n_folds, self.n_subs, len(sample_points), self.n_seg, out_one.shape[0], out_one.shape[1]))
                        if not hasattr(self, 'feature_label'):
                            self.feature_label = np.zeros((self.n_folds, self.n_subs, len(sample_points), 1))
                        self.feature_raw[fold, sub, sample_idx, seg] = out_one
                    self.feature_label[fold, sub, sample_idx] = sample_labels[sample_idx]
                    count += 1
        print(self.feature_raw)
        return 1

    def compute_de(self, fold, norm = 'global'):
        self.norm_method = norm
        # self.non_zero_indices[fold] = np.ones_like(self.non_zero_indices[fold]).astype(bool)
        print('Computing DE')
        if not hasattr(self, 'feature_nonzero'):
            self.feature_nonzero = {}
        self.feature_nonzero[fold] = self.feature_raw[fold][:, :, :, self.non_zero_indices[fold], :]
        if not hasattr(self, 'de'):
            self.de = {}
        # self.de[fold] = np.zeros((self.n_subs, len(self.feature_points), self.feature_nonzero[fold].shape[-2]))

        self.de[fold] = np.zeros((self.n_subs, len(self.feature_points), self.feature_nonzero[fold].shape[-2], self.n_seg))  # 存储de向量的数组

        for sub in range(self.n_subs):
            for sample in range(len(self.feature_points)):
                # self.de[fold][sub, sample] = 0.5*np.log(2*np.pi*np.exp(1)*(np.var(self.feature_nonzero[fold][sub, sample], 1)) + 1)

                sample_data = self.feature_nonzero[fold][sub][sample]  # 获取原始数据
                for seg in range(self.n_seg):

                    segment_data = sample_data[seg]  # 当前段的数据
                    # print(sample_data.shape)
                    # print(segment_data.shape)
                    self.de[fold][sub][sample, :, seg] = 0.5*np.log(2*np.pi*np.exp(1)*(np.var(segment_data, 1)) + 1)  # 计算de并存储结果
        print(self.de[fold].shape)
        # running_norm
        print('Running Norm')

        if not hasattr(self, 'de_norm'):
            self.de_norm = {}

        self.de_norm[fold] = np.zeros_like(self.de[fold])
        
        if fold < self.n_folds-1:
            val_sub = np.arange(self.n_per*fold, self.n_per*(fold+1))
        else:
            # val_sub = np.arange(n_per*fold, n_per*(fold+1)-1)
            val_sub = np.arange(self.n_per * fold, self.n_subs)
        val_sub = [int(val) for val in val_sub]
        train_sub = list(set(np.arange(self.n_subs)) - set(val_sub))
        if(self.norm_method == 'local'):
            # local_norm method
            for sub in range(self.n_subs):
                sub_mean = np.tile(np.mean(self.de[fold][sub], axis=2, keepdims=True), (1, self.de[fold][sub].shape[2]))
                sub_var = np.tile(np.var(self.de[fold][sub], axis=2, keepdims=True), (1, self.de[fold][sub].shape[2]))
                sub_mean = np.mean(sub_mean, axis=0)
                sub_var = np.mean(sub_var, axis=0)

                # print(sub_mean.shape, sub_var.shape)
                for sample in range(len(self.feature_points)):
                    de_one = self.de[fold][sub, sample]
                    de_one = (de_one - sub_mean) / np.sqrt(sub_var + 1e-5)


                    self.de_norm[fold][sub, sample] = de_one
        else:
            # old_norm method
            decay_rate = 0.990
            data_mean = np.mean(np.mean(self.de[fold][train_sub]))
            data_var = np.mean(np.var(self.de[fold][train_sub]))
            for sub in range(self.n_subs):
                running_sum = np.zeros(self.de[fold].shape[-1])
                running_square = np.zeros(self.de[fold].shape[-1])
                decay_factor = 1.
                for sample in range(len(self.feature_points)):
                    de_one = self.de[fold][sub, sample]
                    running_sum = running_sum + de_one
                    running_mean = running_sum / (sample+1)
                    # running_mean = counter / (counter+1) * running_mean + 1/(counter+1) * data_one
                    running_square = running_square + de_one**2
                    running_var = (running_square - 2 * running_mean * running_sum) / (sample+1) + running_mean**2

                    # print(decay_factor)
                    curr_mean = decay_factor*data_mean + (1-decay_factor)*running_mean
                    curr_var = decay_factor*data_var + (1-decay_factor)*running_var
                    decay_factor = decay_factor*decay_rate

                    # print(running_var[:3])
                    # if counter >= 2:
                    de_one = (de_one - curr_mean) / np.sqrt(curr_var + 1e-5)
                    self.de_norm[fold][sub, sample] = de_one
        
        # smooth_lds
        print('Smooth LDS')
        if not hasattr(self, 'de_smooth'):
            self.de_smooth = {}
        self.de_smooth[fold] = np.zeros_like(self.de_norm[fold])
        for sub in range(self.n_subs):
            for sample in range(len(self.feature_points)):
                # print(self.de_norm[fold][sub][sample].shape)
                self.de_smooth[fold][sub][sample] = self.LDS(self.de_norm[fold][sub][sample].transpose()).transpose()
        
        # if not hasattr(self, 'de_cus'):
        #     self.de_cus = {}
        # self.de_cus[fold] = np.zeros_like(self.de[fold])
        # for sub in range(self.n_subs):
        #     for row in range(np.count_nonzero(self.non_zero_indices[fold])):
        #         # print(self.de_norm[fold][sub].shape)
        #         self.de_cus[fold][sub, :, row] = np.interp(self.de[fold][sub, :, row], (self.de[fold][sub, :, row].min(), self.de[fold][sub, :, row].max()), (-1, 1))

        return


    def LDS(self, sequence):
        # print(sequence.shape) # (30, 256)

        # sequence_new = np.zeros_like(sequence) # (30, 256)
        ave = np.mean(sequence, axis=0)  # [256,]
        u0 = ave
        X = sequence.transpose((1, 0))  # [256, 30]

        V0 = 0.01
        A = 1
        T = 0.0001
        C = 1
        sigma = 1

        [m, n] = X.shape  # (1, 30)
        P = np.zeros((m, n))  # (1, 1, 30) dia
        u = np.zeros((m, n))  # (1, 30)
        V = np.zeros((m, n))  # (1, 1, 30) dia
        K = np.zeros((m, n))  # (1, 1, 30)

        K[:, 0] = (V0 * C / (C * V0 * C + sigma)) * np.ones((m,))
        u[:, 0] = u0 + K[:, 0] * (X[:, 0] - C * u0)
        V[:, 0] = (np.ones((m,)) - K[:, 0] * C) * V0

        for i in range(1, n):
            P[:, i - 1] = A * V[:, i - 1] * A + T
            K[:, i] = P[:, i - 1] * C / (C * P[:, i - 1] * C + sigma)
            u[:, i] = A * u[:, i - 1] + K[:, i] * (X[:, i] - C * A * u[:, i - 1])
            V[:, i] = (np.ones((m,)) - K[:, i] * C) * P[:, i - 1]

        X = u

        return X.transpose((1, 0))

    def get_feature_dataloader(self, fold, batch_size):
        self.n_per = round(self.n_subs / self.n_folds)    
        if fold < self.n_folds-1:
            val_sub = np.arange(self.n_per*fold, self.n_per*(fold+1))
        else:
            val_sub = np.arange(self.n_per*fold, self.n_subs)
        val_sub = [int(val) for val in val_sub]
        print('val', val_sub)

        train_sub = np.array(list(set(np.arange(self.n_subs)) - set(val_sub)))
        print('train', train_sub)

        train_features = self.de_smooth[fold][train_sub]
        val_features = self.de_smooth[fold][val_sub]
        train_labels = self.feature_label[fold, train_sub]
        val_labels = self.feature_label[fold, val_sub]
        train_dataset = FeatureDataset(train_features, train_labels, non_zero_indices = self.non_zero_indices)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = FeatureDataset(val_features, val_labels, non_zero_indices = self.non_zero_indices)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        return train_loader, val_loader

    def get_hidden_psd(self, fold, inds_sel, isNorm=False):
        n_points_cum_ori = np.concatenate((np.array([0]), np.cumsum(self.n_points))).astype(int)
        n_latent_dims = self.n_timeFilters * self.n_spatialFilters

        state_dict = torch.load(os.path.join(self.save_dir, str(fold), 'checkpoint_%04d.pth.tar' % (self.epochs_pretrain-1)), map_location='cuda:0')
        print('checkpoint:', os.path.join(self.save_dir, str(fold), 'checkpoint_%04d.pth.tar' % (self.epochs_pretrain-1)))
        state_dict = state_dict['state_dict']

        model = ConvNet_avgPool_share_nopool(self.n_timeFilters, self.timeFilterLen, self.n_spatialFilters, self.n_channs, self.stratified, 'infer').double()

        model.load_state_dict(state_dict, strict=False)

        psd = np.zeros((self.n_subs, len(inds_sel), self.n_trials))
        # sub_order = np.concatenate((train_sub, val_sub))
        count = 0
        with torch.cuda.device(self.gpu_index):
            model = model.to(self.device)
            for sub in np.arange(self.n_subs):
                # print(sub)
                for tr in range(self.n_trials):
                    data_now = self.data[sub,n_points_cum_ori[tr]:n_points_cum_ori[tr+1],:]
                    data_tensor = torch.from_numpy(data_now).to(self.device)
                    data_tensor.requires_grad = False

                    out_one = model(torch.unsqueeze(torch.unsqueeze(data_tensor.permute(1,0),0),0))
                    out_one = out_one.reshape(out_one.shape[0], n_latent_dims, -1)
                    if isNorm:
                        out_one = F.normalize(out_one, dim=1)                        
                    out_one = torch.squeeze(out_one)
                    out_one = out_one.detach().cpu().numpy().reshape(-1, out_one.shape[-1])
                    for i in range(len(inds_sel)):
                        psd[count,i,tr] = self.calc_psd(out_one[inds_sel[i]])
                count += 1
        return psd
    
    def calc_psd(self, signal):
        (f, S) = scipy.signal.periodogram(signal, self.fs, scaling='density')
        psd = np.mean(S[(f>=0.5) & (f<40)])
        return psd
    
    def get_hidden_nopool(self, fold, inds_sel, isNorm=False):
        n_points_cum_ori = np.concatenate((np.array([0]), np.cumsum(self.n_points))).astype(int)
        n_points_cum = np.concatenate((np.array([0]), np.cumsum(self.n_points-self.timeFilterLen+1))).astype(int)

        n_latent_dims = self.n_timeFilters * self.n_spatialFilters

        state_dict = torch.load(os.path.join(self.save_dir, str(fold), 'checkpoint_%04d.pth.tar' % (self.epochs_pretrain-1)), map_location='cuda:0')
        print('checkpoint:', os.path.join(self.save_dir, str(fold), 'checkpoint_%04d.pth.tar' % (self.epochs_pretrain-1)))
        state_dict = state_dict['state_dict']

        model = ConvNet_avgPool_share_nopool(self.n_timeFilters, self.timeFilterLen, self.n_spatialFilters, self.n_channs, self.stratified, 'infer').double()

        model.load_state_dict(state_dict, strict=False)

        out = np.zeros((self.n_subs, len(inds_sel), n_points_cum[-1]))
        # sub_order = np.concatenate((train_sub, val_sub))
        count = 0
        with torch.cuda.device(self.gpu_index):
            model = model.to(self.device)
            for sub in np.arange(self.n_subs):
                # print(sub)
                for tr in range(self.n_trials):
                    data_now = self.data[sub,n_points_cum_ori[tr]:n_points_cum_ori[tr+1],:]
                    data_tensor = torch.from_numpy(data_now).to(self.device)
                    data_tensor.requires_grad = False

                    out_one = model(torch.unsqueeze(torch.unsqueeze(data_tensor.permute(1,0),0),0))
                    out_one = out_one.reshape(out_one.shape[0], n_latent_dims, -1)[:,inds_sel,:]
                    if isNorm:
                        out_one = F.normalize(out_one, dim=1)
                    out_one = torch.squeeze(out_one)
                    out[count, :, n_points_cum[tr]:n_points_cum[tr+1]] = out_one.detach().cpu().numpy().reshape(-1, out_one.shape[-1])
                count += 1
        self.out = out
        self.n_points_cum = n_points_cum
        return out, n_points_cum
    
    def check_nonzero_dims(self):
        # Check if there are zero dimensions
        zero_dims = []
        for chn in range(256):
            for tr in range(self.n_trials):
                if np.any(np.sum(np.abs(self.out[:,chn,self.n_points_cum[tr]:self.n_points_cum[tr+1]]), axis=1) == 0):
                    zero_dims.append(chn)
                    break
                
        nonzero_dims = np.array(list(set(np.arange(256)) - set(np.unique(zero_dims))))
        print('length of nonzero dims', len(nonzero_dims))
        # sio.savemat(os.path.join(resultdir, 'nonzero_dims.mat'), {'nonzero_dims': nonzero_dims})
        self.nonzero_dims = nonzero_dims
        return nonzero_dims
        
    def calc_out_corr_dims(self):
        n_nonzero_dims = len(self.nonzero_dims)
        out_corr_dims = np.zeros((self.n_subs, n_nonzero_dims, n_nonzero_dims, self.n_trials))
        for tr in range(self.n_trials):
            for sub in range(self.n_subs):
                out_corr_dims[sub,:,:,tr] = np.corrcoef(self.out[sub, self.nonzero_dims, self.n_points_cum[tr]:self.n_points_cum[tr+1]])
        out_corr_dims_mean = np.mean(np.mean(out_corr_dims, axis=3), axis=0)
        return out_corr_dims_mean
    
    
    def get_correspond_dims(self, n_folds, out, calc_dims, isNorm=False, isPool=True):
        n_per = int(np.floor(self.n_subs / n_folds))
        n_latent_dims = self.n_timeFilters * self.n_spatialFilters
        correspondDims_fold = np.zeros((n_folds, len(calc_dims))).astype(int)
        corr_mean_fold = np.zeros((n_folds, len(calc_dims)))
        for val_fold in range(n_folds):
            if val_fold < n_folds-1:
                val_sub = np.arange(val_fold*n_per, (val_fold+1)*n_per)
            else:
                val_sub = np.arange(val_fold*n_per, self.n_subs)
            train_sub = list(set(np.arange(self.n_subs)) - set(val_sub))
            
            state_dict = torch.load(os.path.join(self.save_dir, str(val_fold), 'checkpoint_%04d.pth.tar' % (self.epochs_pretrain-1)), map_location='cuda:0')
            state_dict = state_dict['state_dict']
        
            model = ConvNet_avgPool_share(self.n_timeFilters, self.timeFilterLen, self.n_spatialFilters, self.avgPoolLen, self.n_channs, self.stratified, self.activ, 'infer').double()
            model.load_state_dict(state_dict, strict=False)
            if isPool:
                out2, n_points_cum = self.get_hidden(val_fold, isNorm=isNorm)
            else:
                out2, n_points_cum = self.get_hidden_nopool(val_fold, inds_sel=np.arange(n_latent_dims), isNorm=isNorm)
            
            corr_mat = np.zeros((len(train_sub), self.n_trials, len(calc_dims), self.n_timeFilters*self.n_spatialFilters))
            with torch.cuda.device(self.gpu_index):
                count = 0
                model = model.to(self.device)
                for sub in train_sub:
                    for tr in range(self.n_trials):
                        corr_mat[count, tr, :, :] = calc_corr(out[sub, calc_dims, n_points_cum[tr]:n_points_cum[tr+1]].transpose(), 
                                                        out2[sub, :, n_points_cum[tr]:n_points_cum[tr+1]].transpose())
                    count += 1
                        
            corr_mean = np.mean(np.mean(corr_mat, axis=0), axis=0) # We should compare np.mean and np.nanmean
            correspondDims_fold[val_fold, :] = np.nanargmax(np.abs(corr_mean), axis=1).astype(int)
            corr_mean_fold[val_fold, :] = corr_mean[np.arange(len(calc_dims)), correspondDims_fold[val_fold, :]]
        return correspondDims_fold, corr_mean_fold
        
    def get_correspond_dims_memEffi(self, n_folds, out_sel, isNorm=False, isPool=True):
        n_per = int(np.floor(self.n_subs / n_folds))
        n_latent_dims = self.n_timeFilters * self.n_spatialFilters
        n_sel = out_sel.shape[1]
        correspondDims_fold = np.zeros((n_folds, n_sel)).astype(int)
        corr_mean_fold = np.zeros((n_folds, n_sel))
        for val_fold in range(n_folds):
            if val_fold < n_folds-1:
                val_sub = np.arange(val_fold*n_per, (val_fold+1)*n_per)
            else:
                val_sub = np.arange(val_fold*n_per, self.n_subs)
            train_sub = list(set(np.arange(self.n_subs)) - set(val_sub))
            
            state_dict = torch.load(os.path.join(self.save_dir, str(val_fold), 'checkpoint_%04d.pth.tar' % (self.epochs_pretrain-1)), map_location='cuda:0')
            state_dict = state_dict['state_dict']
        
            model = ConvNet_avgPool_share(self.n_timeFilters, self.timeFilterLen, self.n_spatialFilters, self.avgPoolLen, self.n_channs, self.stratified, self.activ, 'infer').double()
            model.load_state_dict(state_dict, strict=False)
            if isPool:
                out2, n_points_cum = self.get_hidden(val_fold, isNorm=isNorm)
            else:
                out2, n_points_cum = self.get_hidden_nopool(val_fold, inds_sel=np.arange(n_latent_dims), isNorm=isNorm)
            
            corr_mat = np.zeros((len(train_sub), self.n_trials, n_sel, self.n_timeFilters*self.n_spatialFilters))
            with torch.cuda.device(self.gpu_index):
                count = 0
                model = model.to(self.device)
                for sub in train_sub:
                    for tr in range(self.n_trials):
                        corr_mat[count, tr, :, :] = calc_corr(out_sel[sub, :, n_points_cum[tr]:n_points_cum[tr+1]].transpose(), 
                                                        out2[sub, :, n_points_cum[tr]:n_points_cum[tr+1]].transpose())
                    count += 1
                        
            corr_mean = np.mean(np.mean(corr_mat, axis=0), axis=0) # We should compare np.mean and np.nanmean
            correspondDims_fold[val_fold, :] = np.nanargmax(np.abs(corr_mean), axis=1).astype(int)
            corr_mean_fold[val_fold, :] = corr_mean[np.arange(n_sel), correspondDims_fold[val_fold, :]]
        return correspondDims_fold, corr_mean_fold
                
        

        
        
                
   
                
