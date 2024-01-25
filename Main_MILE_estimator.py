import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from utils_mile import *
from utils_smile import sample_correlated_gaussian,mi_to_rho,rho_to_mi

time_start = time.time()
def trans_func(x,func):
    if func=='self':
        return x
    elif func=='sin':
        return torch.sin(x)
    elif func=='cubic':
        return x ** 3

# generate data
seed = 1000
seed_torch(seed)

#data of the MI estimation
dim = 20
data_size = 4096
# generate the data
mi = 10.0
rho = mi_to_rho(dim, mi)
print(rho)
print('expected mi', rho_to_mi(dim, rho))
X, Y = sample_correlated_gaussian(rho, dim, batch_size=data_size, cubic=None)
# Y = trans_func(Y, 'cubic')

#use logDet MILE estimator
MI_estimator=MI_LogDet_Estimator(Kmax=1,beta=0.0,method='Kmeans')
mi_est,_,_,_=MI_estimator.MILE_estimate(X,Y)
print('single Gaussian MILE_logdet',mi_est)
print('\n')

MI_estimator = MI_LogDet_RobustEstimator(Kmax=10, beta=1e-3, method='GMM')
mi_est_unbias = MI_estimator.MILE_estimate(X, Y)
print('robust MILE_logdet_ubias (GMM)', mi_est_unbias)

MI_estimator = MI_LogDet_RobustEstimator(Kmax=5, beta=1e-3, method='GMM')
mi_est_unbias = MI_estimator.MILE_estimate(X, Y)
print('robust MILE_logdet_ubias (GMM)', mi_est_unbias)

MI_estimator=MI_LogDet_RobustEstimator(Kmax=5,beta=1e-3,method='Kmeans')
mi_est_unbias=MI_estimator.MILE_estimate(X,Y)
print('robust MILE_logdet_ubias (kmeans)',mi_est_unbias)

print('\n')
MI_estimator=MI_LogDet_Estimator(Kmax=5,beta=1e-3,method='GMM')
mi_est_bias, mi_est_unbias, mi_l, mi_u=MI_estimator.MILE_estimate(X,Y)
print('MILE_logdet_ubias (GMM)',mi_est_unbias)
print('MILE_logdet_bias (GMM)',mi_est_bias)
print('MILE_logdet_l_ubias  (GMM)',mi_l)
print('MILE_logdet_u_ubias  (GMM)',mi_u)

#
MI_estimator=MI_LogDet_Estimator(Kmax=5,beta=1e-3,method='Kmeans')
mi_est_bias, mi_est_unbias, mi_l, mi_u=MI_estimator.MILE_estimate(X,Y)
print('MILE_logdet_ubias (Kmeans)',mi_est_unbias)
print('MILE_logdet_bias (Kmeans)',mi_est_bias)
print('MILE_logdet_l_ubias  (Kmeans)',mi_l)
print('MILE_logdet_u_ubias  (Kmeans)',mi_u)
print('\n')
#
# #here we use pairs samples
MI_estimator=MI_LogDet_RobustEstimator(Kmax=5,beta=1e-3,method='Kmeans')
x_pos=torch.concat((X,Y),dim=1)
all_pairs=False
if all_pairs==True:
    x_tiled = torch.stack([X] * X.shape[0], dim=0)
    y_tiled = torch.stack([Y] * Y.shape[0], dim=1)
    x_neg = diag_remove(torch.cat((x_tiled, y_tiled), dim=2))
else:
    y_shuffle = np.random.permutation(Y.numpy())
    y_shuffle = torch.from_numpy(y_shuffle).type(torch.FloatTensor)
    x_neg = torch.cat((X, y_shuffle), dim=1)
mi_est_unbias=MI_estimator.MILE_estimate_pairs(x_pos,x_neg)
print('robust MILE_logdet_ubias (pairs input)',mi_est_unbias)
#
#use another method to format the pairs
# MI_estimator=MI_LogDet_RobustEstimator(Kmax=5,beta=1e-3,method='Kmeans')
# x_pos=torch.concat((X,Y),dim=1)
# all_pairs=True
# if all_pairs==True:
#     x_tiled = torch.stack([X] * X.shape[0], dim=0)
#     y_tiled = torch.stack([Y] * Y.shape[0], dim=1)
#     x_neg = diag_remove(torch.cat((x_tiled, y_tiled), dim=2))
# else:
#     y_shuffle = np.random.permutation(Y.numpy())
#     y_shuffle = torch.from_numpy(y_shuffle).type(torch.FloatTensor)
#     x_neg = torch.cat((X, y_shuffle), dim=1)
# mi_est_unbias=MI_estimator.MILE_estimate_pairs(x_pos,x_neg)
# print('robust MILE_logdet_ubias (pairs input)',mi_est_unbias)
