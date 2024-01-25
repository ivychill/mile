from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import numpy as np
from estimators import *
from utils_mile import *


class MILE(nn.Module):
    def __init__(self, varpercent=0.98):
        super(MILE, self).__init__()
        self.varpercent = varpercent

    def reduce_dim(self, x_sample):
        X = torch.flatten(x_sample, 1)
        pca = PCA(n_components=self.varpercent)
        pca.fit(X.numpy())
        x_pca = pca.transform(X.numpy())
        if x_pca.shape[-1] < 2:
            print(f'x_pca n_components < 2, set 2.')
            pca = PCA(n_components=2)
            pca.fit(X.numpy())
            x_pca = pca.transform(X.numpy())
        return x_pca
        
    def forward(self, x_sample, y_sample):
        x_pca = self.reduce_dim(x_sample)
        y_pca = self.reduce_dim(y_sample)
        # print(f'x_pca {x_pca.shape}, y_pca {y_pca.shape}')
        # MI_estimator = MI_LogDet_Estimator(Kmax=5, beta=1e-3, method='Kmeans')
        MI_estimator = MI_LogDet_Estimator(beta=1e-3, Kmax=5, Kmax_X=1, Kmax_Y=1, Kmax_XY=1, method='Kmeans')
        mi_est_bias, mi_est_unbias, mi_l, mi_u = MI_estimator.MILE_estimate(torch.from_numpy(x_pca), torch.from_numpy(y_pca))
        return mi_est_unbias


class MiBenchmark(nn.Module):
    def __init__(self):
        super(MiBenchmark, self).__init__()

    def name(self):
        return self.__class__.__name__
        
class CPC(MiBenchmark):
    def __init__(self):
        super(CPC, self).__init__()

    def forward(self, scores):
        return infonce_lower_bound(scores)
    
class MINE(MiBenchmark):
    def __init__(self):
        super(MINE, self).__init__()

    def forward(self, scores):
        return nwj_lower_bound(scores)

class SMILE(MiBenchmark):
    def __init__(self, clip=None):
        super(SMILE, self).__init__()
        self.clip = clip

    def name(self):
        estimator = f'SMILE (τ=∞)'
        if self.clip is not None: 
            estimator = f'SMILE (τ={self.clip})'
        return estimator

    def forward(self, scores):
        return smile_lower_bound(scores, clip=self.clip)

