import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math

def stratified_layerNorm(out, n_samples):
    n_subs = int(out.shape[0] / n_samples)
    out_str = out.clone()
    for i in range(n_subs):
        out_oneSub = out[n_samples*i: n_samples*(i+1)]
        out_oneSub = out_oneSub.reshape(out_oneSub.shape[0], -1, out_oneSub.shape[-1]).permute(0,2,1)
        out_oneSub = out_oneSub.reshape(out_oneSub.shape[0]*out_oneSub.shape[1], -1)
        out_oneSub_str = out_oneSub.clone()
        # We don't care about the channels with very small activations
        # out_oneSub_str[:, out_oneSub.abs().sum(dim=0) > 1e-4] = (out_oneSub[:, out_oneSub.abs().sum(dim=0) > 1e-4] - out_oneSub[
        #     :, out_oneSub.abs().sum(dim=0) > 1e-4].mean(dim=0)) / (out_oneSub[:, out_oneSub.abs().sum(dim=0) > 1e-4].std(dim=0) + 1e-3)
        out_oneSub_str = (out_oneSub - out_oneSub.mean(dim=0)) / (out_oneSub.std(dim=0) + 1e-3)
        out_str[n_samples*i: n_samples*(i+1)] = out_oneSub_str.reshape(n_samples, -1, out_oneSub_str.shape[1]).permute(
            0,2,1).reshape(n_samples, out.shape[1], out.shape[2], -1)
    return out_str
    

class ConvNet_avgPool_share(nn.Module):
    def __init__(self, n_timeFilters, timeFilterLen, n_spatialFilters, avgPoolLen, n_channs, stratified, activ, phase):
        super(ConvNet_avgPool_share, self).__init__()
        # self.timeConv = nn.Conv2d(1, n_timeFilters, (1, timeFilterLen), padding=(0, (timeFilterLen-1)//2))
        self.spatialConv = nn.Conv2d(1, n_spatialFilters, (n_channs, 1))
        self.timeConv = nn.Conv2d(1, n_timeFilters, (1, timeFilterLen))
        self.avgpool = nn.AvgPool2d((1, avgPoolLen))
        self.stratified = stratified
        self.activ = activ
        assert phase in ['train', 'infer']
        self.phase = phase

    def forward(self, input):
        if 'initial' in self.stratified:
            if self.phase == 'train':
                input = stratified_layerNorm(input, int(input.shape[0]/2))
            elif self.phase == 'infer':
                input = stratified_layerNorm(input, int(input.shape[0]))
        out = self.spatialConv(input)
        out = out.permute(0,2,1,3)
        out = self.timeConv(out)
        
        if self.activ == 'square':
            out = self.avgpool(out ** 2)
        elif self.activ == 'logvar':
            out = torch.log(self.avgpool(out ** 2) + 1e-5)
        elif self.activ == 'relu':
            out = self.avgpool(F.relu(out))
        if 'middle' in self.stratified:
            if self.phase == 'train':
                out = stratified_layerNorm(out, int(out.shape[0]/2))
            elif self.phase == 'infer':
                out = stratified_layerNorm(out, int(out.shape[0]))
        out = out.reshape(out.shape[0], -1)
        return out
    
class ConvNet_avgPool_share_nopool(nn.Module):
    def __init__(self, n_timeFilters, timeFilterLen, n_spatialFilters, n_channs, stratified, phase):
        super(ConvNet_avgPool_share_nopool, self).__init__()
        # self.timeConv = nn.Conv2d(1, n_timeFilters, (1, timeFilterLen), padding=(0, (timeFilterLen-1)//2))
        self.spatialConv = nn.Conv2d(1, n_spatialFilters, (n_channs, 1))
        self.timeConv = nn.Conv2d(1, n_timeFilters, (1, timeFilterLen))
        self.stratified = stratified
        assert phase in ['train', 'infer']
        self.phase = phase

    def forward(self, input):
        if 'initial' in self.stratified:
            if self.phase == 'train':
                input = stratified_layerNorm(input, int(input.shape[0]/2))
            elif self.phase == 'infer':
                input = stratified_layerNorm(input, int(input.shape[0]))
        out = self.spatialConv(input)
        out = out.permute(0,2,1,3)
        out = self.timeConv(out)
        
        if 'middle' in self.stratified:
            if self.phase == 'train':
                out = stratified_layerNorm(out, int(out.shape[0]/2))
            elif self.phase == 'infer':
                out = stratified_layerNorm(out, int(out.shape[0]))
        out = out.reshape(out.shape[0], -1)
        return out

class ConvNet_attention_simple(nn.Module):
    def __init__(self, n_timeFilters, timeFilterLen0, n_msFilters, timeFilterLen, avgPoolLen, timeSmootherLen, n_channs, stratified, multiFact, activ, temp, saveFea, phase):
        super().__init__()
        self.timeConv = nn.Conv2d(1, n_timeFilters, (1, timeFilterLen0), padding=(0, (timeFilterLen0-1)//2))
        self.spatialConv = nn.Conv2d(n_timeFilters, n_timeFilters, (n_channs, 1))
        self.msConv1 = nn.Conv2d(n_timeFilters, n_timeFilters*n_msFilters, (n_channs, timeFilterLen), groups=n_timeFilters)
        self.msConv2 = nn.Conv2d(n_timeFilters, n_timeFilters*n_msFilters, (n_channs, timeFilterLen), dilation=(1,3), groups=n_timeFilters)
        self.msConv3 = nn.Conv2d(n_timeFilters, n_timeFilters*n_msFilters, (n_channs, timeFilterLen), dilation=(1,6), groups=n_timeFilters)
        self.msConv4 = nn.Conv2d(n_timeFilters, n_timeFilters*n_msFilters, (n_channs, timeFilterLen), dilation=(1,12), groups=n_timeFilters)

        n_msFilters_total = n_timeFilters * n_msFilters * 4

        # Attention
        seg_att = 15
        self.att_conv = nn.Conv2d(n_msFilters_total, n_msFilters_total, (1, seg_att), groups=n_msFilters_total)
        self.att_pool = nn.AvgPool2d((1, seg_att), stride=1)
        self.att_pointConv = nn.Conv2d(n_msFilters_total, n_msFilters_total, (1, 1))

        self.avgpool = nn.AvgPool2d((1, avgPoolLen))
        n_msFilters_total = n_timeFilters * n_msFilters * 4
        self.timeConv1 = nn.Conv2d(n_msFilters_total, n_msFilters_total * multiFact, (1, timeSmootherLen), groups=n_msFilters_total)
        self.timeConv2 = nn.Conv2d(n_msFilters_total * multiFact, n_msFilters_total * multiFact * multiFact, (1, timeSmootherLen), groups=n_msFilters_total * multiFact)
        self.stratified = stratified
        self.timeFilterLen = timeFilterLen
        self.saveFea = saveFea
        self.activ = activ
        self.temp = temp
        self.phase = phase


    def forward(self, input):
        # print(input.shape)
        if 'initial' in self.stratified:
            if(self.phase == 'train'):
                input = stratified_layerNorm(input, int(input.shape[0]/2))
            if(self.phase == 'infer'):
                input = stratified_layerNorm(input, int(input.shape[0]))
        # print(input)
        out = self.timeConv(input)
        # print(out)
        p = np.array([1,3,6,12]) * (self.timeFilterLen - 1)
        out1 = self.msConv1(F.pad(out, (int(p[0]//2), p[0]-int(p[0]//2)), "constant", 0))
        out2 = self.msConv2(F.pad(out, (int(p[1]//2), p[1]-int(p[1]//2)), "constant", 0))
        out3 = self.msConv3(F.pad(out, (int(p[2]//2), p[2]-int(p[2]//2)), "constant", 0))
        out4 = self.msConv4(F.pad(out, (int(p[3]//2), p[3]-int(p[3]//2)), "constant", 0))
        # print(out1.shape, out2.shape, out3.shape, out4.shape)
        out = torch.cat((out1, out2, out3, out4), 1) # (B, dims, 1, T)


        # Attention
        att_w = F.relu(self.att_conv(F.pad(out, (14, 0), "constant", 0)))
        att_w = self.att_pool(F.pad(att_w, (14, 0), "constant", 0)) # (B, dims, 1, T)
        att_w = self.att_pointConv(att_w)
        if self.activ == 'relu':
            att_w = F.relu(att_w)
        elif self.activ == 'softmax':
            att_w = F.softmax(att_w / self.temp, dim=1)
        out = att_w * F.relu(out)

        if self.saveFea:
            return out
        else:
            out = self.avgpool(out)
            if 'middle1' in self.stratified:
                if(self.phase == 'train'):
                    out = stratified_layerNorm(out, int(out.shape[0]/2))
                if(self.phase == 'infer'):
                    out = stratified_layerNorm(out, int(out.shape[0]))
            out = F.relu(self.timeConv1(out))
            out = F.relu(self.timeConv2(out))
            if 'middle2' in self.stratified:
                if(self.phase == 'train'):
                    out = stratified_layerNorm(out, int(out.shape[0]/2))
                if(self.phase == 'infer'):
                    out = stratified_layerNorm(out, int(out.shape[0]))
            out = out.reshape(out.shape[0], -1)
            return out