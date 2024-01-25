import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from utils_mile import *
from utils_smile import sample_correlated_gaussian,mi_to_rho,rho_to_mi
import copy
import gpytorch
import math

class myDataset(Dataset):  # Dataset class
    def __init__(self, X, Y):
        self.Data = X
        self.Label = Y

    def __getitem__(self, index):
        x = torch.from_numpy(self.Data[index]).float()
        y = torch.from_numpy(self.Label[index]).float()
        return x, y

    def __len__(self):
        return len(self.Data)

class Kernel_T(nn.Module):
    def __init__(self, in_dim, n_hidden, out_dim):
        super(Kernel_T, self).__init__()
        self.in_dim=in_dim//2
        self.fc1 = nn.Linear(self.in_dim, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, out_dim)
        self.loc1 = nn.Linear(in_dim,n_hidden)
        self.loc2 = nn.Linear(n_hidden, n_hidden)
        self.loc3 = nn.Linear(n_hidden,1)
        self.rho= nn.Parameter(torch.Tensor(1))
        self.covar_module = gpytorch.kernels.LinearKernel(ard_num_dims=self.in_dim, lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp))
    def forward(self, x):
        x1=x[:,0:self.in_dim]
        x2=x[:,self.in_dim:]
        x1 = F.relu(self.fc1(x1))
        x1 = F.relu(self.fc2(x1))
        x1 = F.relu(self.fc3(x1))
        x2 = F.relu(self.fc1(x2))
        x2 = F.relu(self.fc2(x2))
        x2 = F.relu(self.fc3(x2))
        a = F.relu(self.loc1(x))
        a = F.relu(self.loc2(a))
        a = self.loc3(a)
        T = a * torch.sum(torch.mul(x1,x2),dim=1).view(-1,1)
        # T = self.rho * (torch.matmul(x1, x2.T).diag()).view(-1, 1)
        # T = self.rho * (self.covar_module(x1, x2).evaluate().diag())
        return T

class Net(nn.Module):
    def __init__(self, in_dim, n_hidden, out_dim, clip_tau):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_dim, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, out_dim)
        self.tau=clip_tau
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x= torch.where(x > self.tau, self.tau, x)
        x=torch.where(x < -self.tau, -self.tau, x)
        return x

def trans_func(x,func):
    if func=='self':
        return x
    elif func=='sin':
        return torch.sin(x)
    elif func=='cubic':
        return x ** 3

def gen_x(mu, cov, nsamples):
    Mnorm = torch.distributions.multivariate_normal.MultivariateNormal(mu, cov)
    return Mnorm.rsample(sample_shape=torch.Size([nsamples]))

def gen_y(x, rho):
    nsamples = x.shape[0]
    dim = x.shape[1]
    mu = torch.zeros(dim)
    cov = torch.eye(dim, dim)
    Mnorm = torch.distributions.multivariate_normal.MultivariateNormal(mu, rho * cov)
    return trans_func(x) + Mnorm.rsample(sample_shape=torch.Size([nsamples]))


def Mnorm_sample(mu, cov, nsamples):
    Mnorm = torch.distributions.multivariate_normal.MultivariateNormal(mu, cov)
    return Mnorm.rsample(sample_shape=torch.Size([nsamples]))

# generate data
seed = 1000
seed_torch(seed)

# output variational type
# type 0,  ret = torch.mean(px) - torch.mean(torch.exp(qx-1.0)) #E[e(T-1)] (Nyugen et al. (NWJ))
# type 1, ret = torch.mean(px) - torch.log(torch.mean(torch.exp(qx)))  #(Donsker-Varadahn (DV))
# type 2,ay = torch.tensor([0.5])
#       ret = torch.mean(px) - (torch.mean(torch.exp(qx) / ay + torch.log(ay) - 1.0))
# type 3, infoNCE output

#sampling
# type 0, shuffle
# type 1, all pairs
# type 2, Nsample from all pairs
# type 3, resampling by GMM
# type 4, sample back
# type 5, sample back sysmetric

#surrogate type
f_divergence_type = 'KL_NWJ' #ok but biased
# f_divergence_type = 'KL_NWJ_refined' #ok but small than true
# f_divergence_type = 'KL_DV_refined' #ok==just InfoNCE
# f_divergence_type = 'KL_DV' #ok
# f_divergence_type = 'InfoNCE' #ok
# f_divergence_type = 'Jenson_Shannon'  # ok
# f_divergence_type = 'GAN_JS' #ok
# f_divergence_type = 'squared_Hellinger'  #ok


#data of the MI estimation
dim = 20
data_size = 4096
Nbatch = 128
Nsample = 128 *8
# generate the data
mi = 15.0
rho = mi_to_rho(dim, mi)
print('expected mi', rho_to_mi(dim, rho))

#for GMM fitting
Kmax = 5
from sklearn.mixture import GaussianMixture
GMM = GaussianMixture(n_components=Kmax)

# set the model
clip_tau = torch.tensor([50.0])  #method in smile
model = Net(in_dim= 2*dim, n_hidden=128, out_dim=1, clip_tau=clip_tau)  # simple network is good for our proposed model
# model = Kernel_T(in_dim= 2*dim, n_hidden=128, out_dim=dim)  # simple network is good for our proposed model
mi_estimator=MI_Estimator(network=model, Nbatch=Nbatch, Nsample=Nsample, f_divergence_type='GAN_JS',pair_type = 1, varitional_type=1, GMM_model=GMM)

# we can use movement method to smooth the estimation
n_epoch=1
plot_mi = []

LogDet_flag=True

for epoch in tqdm(range(n_epoch)):

    # test I([x1,x2];[y1,y2])
    # X, Y = sample_correlated_gaussian(rho, dim, batch_size=data_size,)
    # X2, Y2 = sample_correlated_gaussian(rho, dim, batch_size=data_size)
    # X=torch.cat((X,X2),dim=1)
    # Y=torch.cat((Y,Y2),dim=1)

    # test I([x,g(z)];[y,f(y)])
    X, Y = sample_correlated_gaussian(rho, dim, batch_size=data_size)
    fX = trans_func(X, 'sin')
    fY = trans_func(Y, 'sin')
    X = torch.cat((X, fX), dim=1)
    Y = torch.cat((Y, fY), dim=1)

    if LogDet_flag==False:
        # train the MI
        mi_train_est = mi_estimator.train(X, Y, epoch)
        # test the MI
        X_test, Y_test = mini_batch_load_xy(X, Y, batch_size=Nbatch * 2)
        mi_est = mi_estimator.eval(X_test, Y_test, pair_type=2, Nsample=Nsample).numpy()
    else:
        MI_estimator = MI_LogDet_Estimator(Kmax=5, beta=1e-3, method='Kmeans')
        _,mi_est,_,_ = MI_estimator.MILE_estimate(X, Y)
        mi_est = mi_est.numpy()

    plot_mi.append(mi_est)


plot_x = np.arange(len(plot_mi))
plot_y = np.array(plot_mi).reshape(-1, )
print('mi mean (test with cvvr)', np.mean(plot_y[len(plot_y) // 4 * 3:]))
print('mi std (test with cvvr)', np.std(plot_y[len(plot_y) // 4 * 3:]))
plt.figure(1)
plt.plot(plot_x, plot_y)
plt.show()

#use logDet MI estimator
MI_estimator=MI_LogDet_Estimator(Kmax=1,beta=0.0,method='Kmeans')
mi_est,_,_,_=MI_estimator.MILE_estimate(X,Y)
print('MILElogdet_singleGaussian',mi_est)

MI_estimator=MI_LogDet_Estimator(Kmax=5,beta=1e-3,method='Kmeans')
mi_est_bias, mi_est_unbias, mi_l, mi_u=MI_estimator.MILE_estimate(X,Y)
print('MILElogdet_bias',mi_est_bias)
print('MILElogdet_ubias',mi_est_unbias)
print('MILElogdet_l',mi_l)
print('MILElogdet_u',mi_u)

#
# use PCA first
varpercent = 0.98
from sklearn.decomposition import PCA
pca = PCA(n_components=varpercent)
pca.fit(X.numpy())
x_pca = pca.transform(X.numpy())

pca = PCA(n_components=varpercent)
pca.fit(Y.numpy())
y_pca = pca.transform(Y.numpy())

MI_estimator=MI_LogDet_Estimator(Kmax=5,beta=1e-3,method='Kmeans')
mi_est_bias, mi_est_unbias, mi_l, mi_u= MI_estimator.MILE_estimate(torch.from_numpy(x_pca), torch.from_numpy(y_pca))
print('pca MILElogdet_ubias',mi_est_unbias)
print('pca MILElogdet_bias',mi_est_bias)
print('pca MILElogdet_l',mi_l)
print('pca MILElogdet_u',mi_u)


MI_estimator=MI_LogDet_RobustEstimator(Kmax=5,beta=1e-3,method='Kmeans')
mi_est_unbias=MI_estimator.MILE_estimate(torch.from_numpy(x_pca), torch.from_numpy(y_pca))
print('pca robust MILElogdet_ubias',mi_est_unbias)