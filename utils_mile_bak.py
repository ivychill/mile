import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from datetime import datetime
import random
import os
import copy
import math
from torch.utils.data import DataLoader, Dataset


def seed_torch(seed=1000):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# draw a minibatch
def mini_batch_load_x(x_all, batch_size):
    Ndata = x_all.shape[0]
    indx = np.random.permutation(Ndata)
    x_minibatch = x_all[indx[:batch_size], :]
    return x_minibatch


def mini_batch_load_xy(x_all, y_all, batch_size):
    Ndata = x_all.shape[0]
    indx = np.random.permutation(Ndata)
    x_minibatch = x_all[indx[:batch_size], :]
    y_minibatch = y_all[indx[:batch_size]]
    return x_minibatch, y_minibatch


def GMM_fit(GMM, x_tensor):
    GMM.fit(X=x_tensor.numpy())
    w_em_x = torch.from_numpy(GMM.weights_).float()
    mu_em_x = torch.from_numpy(GMM.means_).float()
    cov_em_x = torch.from_numpy(GMM.covariances_).float()
    GMM_x = {'w': w_em_x, 'mu': mu_em_x, 'cov': cov_em_x}
    return GMM_x


def gumbel_sample(shape, eps=1e-20):
    U = torch.rand(shape)
    return -Variable(torch.log(-torch.log(U + eps) + eps))


def gumbel_softmax_sample(logits, temperature):
    y = F.log_softmax(logits, dim=-1) + gumbel_sample(logits.size())
    return F.softmax(y / temperature, dim=-1)


def MixMNorm_sample_gumble(w, mu, cov, Nsample, temperature=0.1):
    K = mu.shape[0]
    z = gumbel_softmax_sample(w.repeat(Nsample, 1), temperature)
    x_k = torch.zeros(Nsample, mu.shape[1], K)
    for j in range(K):
        Mnorm = torch.distributions.multivariate_normal.MultivariateNormal(mu[j, :], cov[j, :, :])
        xj = Mnorm.rsample(sample_shape=torch.Size([Nsample]))
        x_k[:, :, j] = xj
    z = z.unsqueeze(1).repeat(1, mu.shape[1], 1)
    x = torch.sum(z * x_k, dim=-1).squeeze()
    return x


def diag_remove(x_tensor):
    ns = x_tensor.shape[0]
    dim = x_tensor.shape[2]
    x_ = torch.reshape(x_tensor, [ns * ns, 1, dim]).squeeze()
    offdiag_term = np.reshape(np.eye(ns, ns), (ns * ns, 1))
    offdiag_indx = np.where(offdiag_term == 0)
    x = x_[offdiag_indx[0], :]
    return x


def remove_positive(xy_npairs, xy_npos, THR=1e-3):
    p = xy_npos.shape[1]
    for i in range(xy_npos.shape[0]):
        indx = np.where(torch.norm(xy_npairs.view(-1, p) - xy_npos[i], p=1, dim=1).numpy() < THR)
        indx = indx[0]
        for j in range(indx.shape[0]):
            xy_npairs = xy_npairs.view(-1, p)[torch.arange(xy_npairs.view(-1, p).shape[0]) != indx[j]].squeeze(dim=0)
    return xy_npairs


def sample_npairs(x_sample, y_sample, x_all, y_all, pair_type, Nsample, GMM_x, GMM_y):
    # type 0, shuffle
    # type 1, all pairs
    # type 2, Nsample from all pairs
    # type 3, resampling by GMM
    # type 4, sample back
    # type 5, sample back sysmetric
    # use generated or shuffled
    if pair_type == 0:
        y_shuffle = np.random.permutation(y_sample.numpy())
        y_shuffle = torch.from_numpy(y_shuffle).type(torch.FloatTensor)
        xy_npairs = torch.cat((x_sample, y_shuffle), dim=1)
    elif pair_type == 1:
        x_tiled = torch.stack([x_sample] * x_sample.shape[0], dim=0)
        y_tiled = torch.stack([y_sample] * y_sample.shape[0], dim=1)
        xy_npairs = diag_remove(torch.cat((x_tiled, y_tiled), dim=2))
    elif pair_type == 2:
        x_tiled = torch.stack([x_sample] * x_sample.shape[0], dim=0)
        y_tiled = torch.stack([y_sample] * y_sample.shape[0], dim=1)
        xy_npairs = diag_remove(torch.cat((x_tiled, y_tiled), dim=2))
        indx = np.random.permutation(xy_npairs.shape[0])
        xy_npairs = xy_npairs[indx[0:Nsample], :]
    elif pair_type == 3:
        x_resample = MixMNorm_sample_gumble(GMM_x['w'], GMM_x['mu'], GMM_x['cov'], Nsample, temperature=0.05)
        y_resample = MixMNorm_sample_gumble(GMM_y['w'], GMM_y['mu'], GMM_y['cov'], Nsample, temperature=0.05)
        xy_npairs = torch.cat((x_resample, y_resample), dim=1)
    elif pair_type == 4:
        x_resample = x_sample.repeat(Nsample // x_sample.shape[0], 1)
        y_resample = mini_batch_load_x(y_all, Nsample)
        xy_npairs = torch.cat((x_resample, y_resample), dim=1)
        xy_npairs = remove_positive(xy_npairs, torch.cat((x_sample, y_sample), dim=1))
    elif pair_type == 5:
        x_resample = x_sample.repeat((Nsample // 2) // x_sample.shape[0], 1)
        y_resample = mini_batch_load_x(y_all, Nsample // 2)
        xy_npairs1 = torch.cat((x_resample, y_resample), dim=1)
        y_resample = y_sample.repeat((Nsample // 2) // y_sample.shape[0], 1)
        x_resample = mini_batch_load_x(x_all, Nsample // 2)
        xy_npairs2 = torch.cat((x_resample, y_resample), dim=1)
        xy_npairs = torch.cat((xy_npairs1, xy_npairs2), dim=0)
        # xy_npairs = remove_positive(xy_npairs, torch.cat((x_sample, y_sample), dim=1))
    return xy_npairs[:, 0:x_sample.shape[1]], xy_npairs[:, x_sample.shape[1]:]


def surrogate_lower_bound(f_pxy, f_pxpy, f_divergence_type):
    # use proposed surrogate loss
    if f_divergence_type == 'KL_NWJ':
        # KL divergence
        g_pos = f_pxy
        g_neg = f_pxpy
        first_term = torch.mean(g_pos)
        second_term = -torch.mean(torch.exp(g_neg - 1.0))
    elif f_divergence_type == 'KL_NWJ_refined':
        # KL divergence
        g_pos = f_pxy
        g_neg = f_pxpy
        first_term = torch.mean(g_pos)
        second_term = 0
        for j in range(g_pos.shape[0]):
            second_term += torch.exp(g_neg + 0.1 * g_pos[j] - 1.0)
        second_term = -torch.sum(second_term) / g_neg.shape[0]
    elif f_divergence_type == 'InfoNCE':
        first_term = f_pxy.mean()
        second_term = 0.0
        for j in range(f_pxy.shape[0]):
            second_term += torch.log(torch.exp(f_pxy[j]) + torch.sum(torch.exp(f_pxpy)))
        second_term = -second_term / f_pxy.shape[0]
    elif f_divergence_type == 'reverse_KL':
        # reverse KL
        g_pos = -torch.exp(f_pxy)
        g_neg = -torch.exp(f_pxpy)
        first_term = torch.mean(g_pos)
        second_term = -torch.mean(-1.0 - torch.log(-g_neg))
    elif f_divergence_type == 'Jenson_Shannon':
        # js divergence
        g_pos = torch.log(torch.tensor([2.0])) - torch.log(1 + torch.exp(-f_pxy))
        g_neg = torch.log(torch.tensor([2.0])) - torch.log(1 + torch.exp(-f_pxpy))
        first_term = torch.mean(g_pos)
        second_term = -torch.mean(F.softplus(f_pxpy))
    elif f_divergence_type == 'GAN_JS':
        f_pos = f_pxy
        f_neg = f_pxpy
        first_term = -F.softplus(-f_pos).mean()
        second_term = -torch.mean(F.softplus(f_neg))
    elif f_divergence_type == 'PearsonChi2':
        # squared pearson chi2
        g_pos = f_pxy
        g_neg = f_pxpy
        first_term = torch.mean(g_pos)
        second_term = -torch.mean(g_neg ** 2 / 4 + g_neg)
    elif f_divergence_type == 'NeymanChi2':
        # neyman pearson chi2
        g_pos = torch.tensor([1.0]) - torch.exp(f_pxy)
        g_neg = torch.tensor([1.0]) - torch.exp(f_pxpy)
        first_term = torch.mean(g_pos)
        second_term = -torch.mean(torch.tensor([2.0]) - 2.0 * torch.sqrt(torch.tensor([1.0]) - g_neg))
    elif f_divergence_type == 'alpha_div':
        # alpha divergence
        alpha_d = torch.tensor([1.5])
        g_pos = f_pxy
        g_neg = f_pxpy
        first_term = torch.mean(g_pos)
        second_term = -torch.mean(1 / alpha_d * (g_neg * (alpha_d - 1) + 1) ** (alpha_d / (alpha_d - 1)) - 1 / alpha_d)
    elif f_divergence_type == 'squared_Hellinger':
        g_pos = torch.tensor([1.0]) - torch.exp(f_pxy)
        g_neg = torch.tensor([1.0]) - torch.exp(f_pxpy)
        first_term = torch.mean(g_pos)
        second_term = -torch.mean(g_neg / (1 - g_neg))
    return first_term + second_term


def V2LogDR(V, f_divergence_type):
    if (f_divergence_type == 'KL_NWJ') | (f_divergence_type == 'KL_NWJ_refined') | (f_divergence_type == 'InfoNCE'):
        logdr = V - 1
    elif f_divergence_type == 'reverse_KL':
        logdr = -V
    elif f_divergence_type == 'Jenson_Shannon':
        logdr = V
    elif f_divergence_type == 'GAN_JS':
        logdr = V
    elif f_divergence_type == 'PearsonChi2':
        logdr = torch.log(V / 2 + 1)
        # logdr = torch.log(torch.abs(V / 2 + 1))
    elif f_divergence_type == 'NeymanChi2':
        logdr = -V / 2
    elif f_divergence_type == 'squared_Hellinger':
        logdr = -2 * V
    elif f_divergence_type == 'alpha_div':
        alpha_d = torch.tensor([1.5])
        logdr = torch.log(V * (alpha_d - 1) + 1) / (alpha_d - 1)
    return logdr


def MI_variatonal_estimate(pred_xy, pred_x_y, varitional_type, ALPHA=1.0):
    if varitional_type == 0:
        ret = torch.mean(pred_xy) - ALPHA * torch.mean(
            torch.exp(pred_x_y - 1.0))  # E[e(T-1)] NWJ  (Nyugen et al. (NWJ))
    elif varitional_type == 1:
        ret = torch.mean(pred_xy) - ALPHA * torch.log(
            torch.mean(torch.exp(pred_x_y)))  # MINE / DV #(Donsker-Varadahn (DV))
    elif varitional_type == 2:
        ay = torch.tensor([.5])
        ret = torch.mean(pred_xy) - ALPHA * (torch.mean(torch.exp(pred_x_y) / ay + torch.log(ay) - 1.0))
    elif varitional_type == 3:
        first_term = pred_xy.mean()
        second_term = 0.0
        for j in range(pred_xy.shape[0]):
            second_term += torch.log(torch.exp(pred_xy[j]) + torch.sum(torch.exp(pred_x_y)))
        second_term = -second_term / pred_xy.shape[0]
        ret = first_term - second_term
    return ret


class myDataset(Dataset):  # Dataset class
    def __init__(self, X, Y):
        self.Data1 = X
        self.Data2 = Y

    def __getitem__(self, index):
        x = torch.from_numpy(self.Data1[index]).float()
        y = torch.from_numpy(self.Data2[index]).float()
        return x, y

    def __len__(self):
        return len(self.Data1)


class MI_Estimator(torch.nn.Module):
    def __init__(self, network, Nbatch, Nsample, f_divergence_type='GAN_JS', pair_type=0, varitional_type=1,
                 GMM_model=None):
        # output variational type
        # type 0,  ret = torch.mean(px) - torch.mean(torch.exp(qx-1.0)) #E[e(T-1)] (Nyugen et al. (NWJ))
        # type 1, ret = torch.mean(px) - torch.log(torch.mean(torch.exp(qx)))  #(Donsker-Varadahn (DV))
        # type 2,ay = torch.tensor([0.5])
        #       ret = torch.mean(px) - (torch.mean(torch.exp(qx) / ay + torch.log(ay) - 1.0))
        # type 3, infoNCE output

        # sampling
        # type 0, shuffle
        # type 1, all pairs
        # type 2, Nsample from all pairs
        # type 3, resampling by GMM
        # type 4, sample back
        # type 5, sample back sysmetric

        # surrogate f-divergence type
        # f_divergence_type = 'KL_NWJ' #ok
        # f_divergence_type = 'KL_NWJ_refined' #ok but small than true
        # f_divergence_type = 'InfoNCE' #ok
        # f_divergence_type = 'Jenson_Shannon'  # ok
        # f_divergence_type = 'GAN_JS' #ok
        # f_divergence_type = 'squared_Hellinger'  #ok

        super(MI_Estimator, self).__init__()

        self.f_divergence_type = f_divergence_type
        self.pair_type = pair_type
        self.varitional_type = varitional_type
        self.Nbatch = Nbatch
        self.Nsample = Nsample
        self.CVVR_flag = True
        self.CVVR_gamma = 0.9
        self.B = None

        self.model = network
        self.model_aux = network

        self.GMM = GMM_model

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[25, 50, 75, 100], gamma=0.5)

        self.MI_list = []

    def eval(self, x, y, pair_type=None, Nsample=None):
        x_sample = x
        y_sample = y

        if pair_type == None:
            pair_type = self.pair_type

        if Nsample == None:
            Nsample = self.Nsample

        if pair_type == 3:
            GMM_x = GMM_fit(self.GMM, x)
            GMM_y = GMM_fit(self.GMM, y)
        else:
            GMM_x = {}
            GMM_y = {}

        x_resample, y_resample = sample_npairs(x_sample, y_sample, x, y, pair_type, Nsample, GMM_x, GMM_y)

        xp_sample = torch.cat((x_sample, y_sample), dim=1)
        xn_sample = torch.cat((x_resample, y_resample), dim=1)

        self.model.eval()
        pred_xp = self.model(xp_sample)
        pred_xn = self.model(xn_sample)

        dr_pred_xp = V2LogDR(pred_xp, self.f_divergence_type)
        dr_pred_xn = V2LogDR(pred_xn, self.f_divergence_type)
        MI_direct_est = MI_variatonal_estimate(dr_pred_xp, dr_pred_xn, self.varitional_type).detach()

        if self.CVVR_flag == False:
            MI_est = MI_direct_est
        else:
            ##use cvvr
            pred_xy = self.model_aux(xp_sample)
            pred_x_y = self.model_aux(xn_sample)

            dr_pred_xy = V2LogDR(pred_xy, self.f_divergence_type)
            dr_pred_x_y = V2LogDR(pred_x_y, self.f_divergence_type)
            W = MI_variatonal_estimate(dr_pred_xy, dr_pred_x_y, self.varitional_type).detach()

            if self.B == None:
                MI_est = MI_direct_est
            else:
                MI_est = MI_direct_est - self.CVVR_gamma * (W - self.B)

        return MI_est

    def train(self, x, y, epoch):

        mydata = myDataset(x.numpy(), y.numpy())
        data_loader = DataLoader(mydata, batch_size=self.Nbatch, shuffle=True)
        if self.pair_type == 3:
            GMM_x = GMM_fit(self.GMM, x)
            GMM_y = GMM_fit(self.GMM, y)
        else:
            GMM_x = {}
            GMM_y = {}

        for i, traindata in enumerate(data_loader):
            x_sample, y_sample = traindata
            x_resample, y_resample = sample_npairs(x_sample, y_sample, x, y, self.pair_type, self.Nsample, GMM_x, GMM_y)

            pred_xp = self.model(torch.cat((x_sample, y_sample), dim=1))
            pred_xn = self.model(torch.cat((x_resample, y_resample), dim=1))

            dr_pred_xp = V2LogDR(pred_xp, self.f_divergence_type)
            dr_pred_xn = V2LogDR(pred_xn, self.f_divergence_type)
            MI_direct_est = MI_variatonal_estimate(dr_pred_xp, dr_pred_xn, self.varitional_type).detach()

            # use surrogate loss
            loss = -surrogate_lower_bound(pred_xp, pred_xn, self.f_divergence_type)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.MI_list.append(MI_direct_est.numpy())

        if (epoch >= 30) & (epoch % 5 == 0):
            self.model_aux = copy.deepcopy(self.model)
            self.B = torch.mean(torch.from_numpy(np.array(self.MI_list)[-5 * x.shape[0] // self.Nbatch:]).float())

        self.scheduler.step()

        return MI_direct_est


from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import math


def MNorm_log_pdf(x, mu, cov):
    Mnorm = torch.distributions.multivariate_normal.MultivariateNormal(mu, cov)
    # shape of log_pdf is [Ndata,1]
    return Mnorm.log_prob(x).view(-1, 1)


def soft_assignment(x, w, mu, cov):
    softmax = torch.nn.Softmax(dim=1)
    logpdf = torch.zeros((x.shape[0], w.shape[0]))
    for k in range(w.shape[0]):
        logpdf[:, k] = w[k].log() + MNorm_log_pdf(x, mu[k, :], cov[k, :, :]).squeeze()
    p = softmax(logpdf)  # p is the ownership/ soft assignment
    return p


class MI_LogDet_Estimator(torch.nn.Module):
    def __init__(self, beta, Kmax, method='GMM'):
        super(MI_LogDet_Estimator, self).__init__()

        self.beta = beta
        self.Kmax = Kmax
        self.method = method

    @staticmethod
    def covariance2entropy_singleGaussian(x):
        d = x.shape[1]
        Sigma = torch.cov(x.T)
        Ent = d / 2 * (torch.log(torch.tensor(2 * math.pi)) + 1) + 0.5 * torch.logdet(Sigma)
        return Ent

    @staticmethod
    def H_lower(w, mu, cov, beta):
        #here we do not use beta
        # beta=0.0
        H_l = torch.tensor([0.0])
        d = cov.shape[-1]
        K = w.shape[0]

        for i in range(K):
            pdf_mix = torch.tensor([0.0])
            B = MNorm_log_pdf(mu[i, :], mu[i, :], beta * torch.eye(d) + cov[i, :, :] + cov[i, :, :]).squeeze()
            for j in range(K):
                log_pdf = MNorm_log_pdf(mu[i, :], mu[j, :], beta * torch.eye(d) + cov[i, :, :] + cov[j, :, :]).squeeze()
                pdf_mix += w[j] * torch.exp(log_pdf-B)
            H_l += w[i] * (torch.log(pdf_mix)+B)
        return -H_l

    @staticmethod
    def H_upper(w, mu, cov, beta):
        d = cov.shape[-1]
        K = w.shape[0]
        H_u = torch.tensor([0.0])
        for j in range(K):
            H_u += w[j] * (
                    -torch.log(w[j]) + d / 2 * (torch.log(torch.tensor(2 * math.pi)) + 1) + 0.5 * torch.logdet(
                beta * torch.eye(d) + cov[j, :, :]))
        return H_u

    def covariance2entropy_estimator(self, x):
        # x tensor
        # w k dim
        # Sigma k * d* d
        d = x.shape[-1]
        Ns = x.shape[0]
        K = self.Kmax
        if K > 1:
            cluster = KMeans(n_clusters=K, n_init="auto", random_state=0).fit(x.detach().numpy())
            y_pred = cluster.predict(x.detach().numpy())
            w = torch.zeros(K)
            mu = torch.zeros((K, d))
            cov = torch.zeros((K, d, d))
            for j in range(K):
                indx = np.where(y_pred == j)
                w[j] = torch.from_numpy(np.array(np.sum(y_pred == j) / Ns)).float()
                mu[j, :] = torch.mean(x[indx[0], :], dim=0)
                cov[j, :, :] = torch.cov(x[indx[0], :].T)
            H_u = self.H_upper(w, mu, cov, self.beta)
            H_l = self.H_lower(w, mu, cov, self.beta)
            Ent = 0.5 * (self.H_lower(w, mu, cov, self.beta) + self.H_upper(w, mu, cov, self.beta))
        else:
            cov = torch.cov(x.T)
            Ent = d / 2 * (torch.log(torch.tensor(2 * math.pi)) + 1) + 0.5 * torch.logdet(
                self.beta * torch.eye(d) + cov)
            H_u = Ent
            H_l = Ent
        return Ent, H_l, H_u

    def covariance2entropy_estimator_GMM(self, x, cov_type='full'):
        # w k dim
        # Sigma k * d* d
        K = self.Kmax
        GMM = GaussianMixture(n_components=K, init_params='kmeans', covariance_type=cov_type)
        GMM.fit(X=x.detach().numpy())
        w = torch.from_numpy(GMM.weights_).float()
        mu = torch.from_numpy(GMM.means_).float()
        if cov_type == 'full':
            cov = torch.from_numpy(GMM.covariances_).float()
        else:
            cov_diag = torch.from_numpy(GMM.covariances_).float()
            cov = torch.zeros((K, x.shape[1], x.shape[1]))
            for k in range(K):
                cov[k, :, :] = torch.diag(cov_diag[k, :])
        d = mu.shape[1]
        if K > 1:
            H_u = self.H_upper(w, mu, cov, self.beta)
            H_l = self.H_lower(w, mu, cov, self.beta)
            Ent = 0.5 * (self.H_lower(w, mu, cov, self.beta) + self.H_upper(w, mu, cov, self.beta))
        else:
            Sigma = torch.cov(x.T)
            Ent = d / 2 * (torch.log(torch.tensor(2 * math.pi)) + 1) + 0.5 * torch.logdet(
                self.beta * torch.eye(d) + Sigma)
            H_u = Ent
            H_l = Ent
        return Ent, H_l, H_u

    def MILE_estimate(self, x, y):
        if self.method == 'Kmeans':
            x_n = torch.randn_like(x)
            mix_est, mix_est_l, mix_est_u = self.covariance2entropy_estimator(x_n)
            gauss_est = self.covariance2entropy_singleGaussian(x_n)
            bias_est_1x = gauss_est - mix_est
            # bias_est_1xl = gauss_est - mix_est_l
            # bias_est_1xu = gauss_est - mix_est_u

            y_n = torch.randn_like(y)
            mix_est, mix_est_l, mix_est_u = self.covariance2entropy_estimator(y_n)
            gauss_est = self.covariance2entropy_singleGaussian(y_n)
            bias_est_1y = gauss_est - mix_est

            x_n = torch.randn_like(torch.concatenate((x, y), dim=1))
            mix_est, mix_est_l, mix_est_u = self.covariance2entropy_estimator(x_n)
            gauss_est = self.covariance2entropy_singleGaussian(x_n)
            bias_est_2xy = gauss_est - mix_est
            # bias_est_2xyl = gauss_est - mix_est_l
            # bias_est_2xyu = gauss_est - mix_est_u

            Hx, Hx_l, Hx_u = self.covariance2entropy_estimator(x)
            Hy, Hy_l, Hy_u = self.covariance2entropy_estimator(y)
            Hxy, Hxy_l, Hxy_u = self.covariance2entropy_estimator(torch.concatenate((x, y), dim=1))

        elif self.method == 'GMM':

            x_n = torch.randn_like(x)
            mix_est, mix_est_l, mix_est_u = self.covariance2entropy_estimator_GMM(x_n)
            gauss_est = self.covariance2entropy_singleGaussian(x_n)
            bias_est_1x = gauss_est - mix_est
            # bias_est_1xl = gauss_est - mix_est_l
            # bias_est_1xu = gauss_est - mix_est_u

            y_n = torch.randn_like(y)
            mix_est, mix_est_l, mix_est_u = self.covariance2entropy_estimator_GMM(y_n)
            gauss_est = self.covariance2entropy_singleGaussian(y_n)
            bias_est_1y = gauss_est - mix_est

            x_n = torch.randn_like(torch.concatenate((x, y), dim=1))
            mix_est, mix_est_l, mix_est_u = self.covariance2entropy_estimator_GMM(x_n)
            gauss_est = self.covariance2entropy_singleGaussian(x_n)
            bias_est_2xy = gauss_est - mix_est
            # bias_est_2l = gauss_est - mix_est_l
            # bias_est_2u = gauss_est - mix_est_u

            Hx, Hx_l, Hx_u = self.covariance2entropy_estimator_GMM(x)
            Hy, Hy_l, Hy_u = self.covariance2entropy_estimator_GMM(y)
            Hxy, Hxy_l, Hxy_u = self.covariance2entropy_estimator_GMM(torch.concatenate((x, y), dim=1))

        # use H(x|y)=H(x,y)-H(x)
        # use MI(x,y)=H(x)+H(y)-H(x,y)
        MIx_y_bias = Hx + Hy - Hxy
        MIx_y_unbias = Hx + Hy - Hxy +  bias_est_1x+bias_est_1y - bias_est_2xy
        MIx_y_l = Hx_l + Hy_l - Hxy_l +  bias_est_1x+bias_est_1y - bias_est_2xy
        MIx_y_u = Hx_u + Hy_u - Hxy_u +  bias_est_1x+bias_est_1y - bias_est_2xy

        return MIx_y_bias, MIx_y_unbias, MIx_y_l, MIx_y_u


class MI_LogDet_RobustEstimator(torch.nn.Module):
    def __init__(self, beta, Kmax, method='GMM'):
        super(MI_LogDet_RobustEstimator, self).__init__()

        self.beta = beta
        self.Kmax = Kmax
        self.method = method

    @staticmethod
    def covariance2entropy_singleGaussian(x):
        d = x.shape[1]
        Sigma = torch.cov(x.T)
        Ent = d / 2 * (torch.log(torch.tensor(2 * math.pi)) + 1) + 0.5 * torch.logdet(Sigma)
        return Ent

    @staticmethod
    def H_upper(w, mu, cov, beta):
        d = cov.shape[-1]
        K = w.shape[0]
        H_u = torch.tensor([0.0])
        for j in range(K):
            H_u += w[j] * (
                    -torch.log(w[j]) + d / 2 * (torch.log(torch.tensor(2 * math.pi)) + 1) + 0.5 * torch.logdet(
                beta * torch.eye(d) + cov[j, :, :]))
        return H_u

    def covariance2entropy_estimator(self, x):
        # x tensor
        # w k dim
        # Sigma k * d* d
        d = x.shape[-1]
        Ns = x.shape[0]
        K = self.Kmax
        if K > 1:
            cluster = KMeans(n_clusters=K, n_init="auto", random_state=0).fit(x.detach().numpy())
            y_pred = cluster.predict(x.detach().numpy())
            w = torch.zeros(K)
            mu = torch.zeros((K, d))
            cov = torch.zeros((K, d, d))
            for j in range(K):
                indx = np.where(y_pred == j)
                w[j] = torch.from_numpy(np.array(np.sum(y_pred == j) / Ns)).float()
                mu[j, :] = torch.mean(x[indx[0], :], dim=0)
                cov[j, :, :] = torch.cov(x[indx[0], :].T)
            H_u = self.H_upper(w, mu, cov, self.beta)
            Ent = H_u
        else:
            cov = torch.cov(x.T)
            Ent = d / 2 * (torch.log(torch.tensor(2 * math.pi)) + 1) + 0.5 * torch.logdet(
                self.beta * torch.eye(d) + cov)
        return Ent

    def covariance2entropy_estimator_GMM(self, x, cov_type='full'):
        # w k dim
        # Sigma k * d* d
        K = self.Kmax
        GMM = GaussianMixture(n_components=K, init_params='kmeans', covariance_type=cov_type)
        GMM.fit(X=x.detach().numpy())
        w = torch.from_numpy(GMM.weights_).float()
        mu = torch.from_numpy(GMM.means_).float()
        if cov_type == 'full':
            cov = torch.from_numpy(GMM.covariances_).float()
        else:
            cov_diag = torch.from_numpy(GMM.covariances_).float()
            cov = torch.zeros((K, x.shape[1], x.shape[1]))
            for k in range(K):
                cov[k, :, :] = torch.diag(cov_diag[k, :])
        d = mu.shape[1]
        if K > 1:
            Ent = self.H_upper(w, mu, cov, self.beta)
        else:
            Sigma = torch.cov(x.T)
            Ent = d / 2 * (torch.log(torch.tensor(2 * math.pi)) + 1) + 0.5 * torch.logdet(
                self.beta * torch.eye(d) + Sigma)
        return Ent

    def MILE_estimate(self, x, y):
        if self.method == 'Kmeans':
            x_n = torch.randn_like(x)
            mix_est = self.covariance2entropy_estimator(x_n)
            bias_est_1x = self.covariance2entropy_singleGaussian(x_n) - mix_est

            y_n = torch.randn_like(y)
            mix_est = self.covariance2entropy_estimator(y_n)
            bias_est_1y = self.covariance2entropy_singleGaussian(y_n) - mix_est

            x_n = torch.randn_like(torch.concatenate((x, y), dim=1))
            mix_est = self.covariance2entropy_estimator(x_n)
            bias_est_2xy = self.covariance2entropy_singleGaussian(x_n) - mix_est

            Hx = self.covariance2entropy_estimator(x)
            Hy = self.covariance2entropy_estimator(y)
            Hxy = self.covariance2entropy_estimator(torch.concatenate((x, y), dim=1))

        elif self.method == 'GMM':
            x_n = torch.randn_like(x)
            mix_est = self.covariance2entropy_estimator_GMM(x_n)
            bias_est_1x = self.covariance2entropy_singleGaussian(x_n) - mix_est

            y_n = torch.randn_like(y)
            mix_est = self.covariance2entropy_estimator_GMM(y_n)
            bias_est_1y = self.covariance2entropy_singleGaussian(y_n) - mix_est

            x_n = torch.randn_like(torch.concatenate((x, y), dim=1))
            mix_est = self.covariance2entropy_estimator_GMM(x_n)
            bias_est_2xy = self.covariance2entropy_singleGaussian(x_n) - mix_est

            Hx = self.covariance2entropy_estimator_GMM(x)
            Hy = self.covariance2entropy_estimator_GMM(y)
            Hxy = self.covariance2entropy_estimator_GMM(torch.concatenate((x, y), dim=1))

        # use H(x|y)=H(x,y)-H(x)
        # use MI(x,y)=H(x)+H(y)-H(x,y)
        MIx_y_unbias = Hx + Hy - Hxy + bias_est_1x + bias_est_1y - bias_est_2xy

        return MIx_y_unbias

    def MILE_estimate_pairs(self, x_pos, x_neg):
        dim = x_pos.shape[1] // 2
        # x1_margin_sample=torch.concat((x_pos[:,0:dim],x_neg[:,0:dim]),dim=0)
        # x2_margin_sample=torch.concat((x_pos[:,dim:],x_neg[:,dim:]),dim=0)
        x1_margin_sample = x_neg[:, 0:dim]
        x2_margin_sample = x_neg[:, dim:]
        x1x2_joint_sample = x_pos

        if self.method == 'Kmeans':
            x_n = torch.randn_like(x1_margin_sample)
            mix_est = self.covariance2entropy_estimator(x_n)
            bias_est_1a = self.covariance2entropy_singleGaussian(x_n) - mix_est

            x_n = torch.randn_like(x2_margin_sample)
            mix_est = self.covariance2entropy_estimator(x_n)
            bias_est_1b = self.covariance2entropy_singleGaussian(x_n) - mix_est
            #
            x_n = torch.randn_like(x1x2_joint_sample)
            mix_est = self.covariance2entropy_estimator(x_n)
            bias_est_2 = self.covariance2entropy_singleGaussian(x_n) - mix_est

            Hx = self.covariance2entropy_estimator(x1_margin_sample)
            Hy = self.covariance2entropy_estimator(x2_margin_sample)
            Hxy = self.covariance2entropy_estimator(x1x2_joint_sample)

        elif self.method == 'GMM':

            x_n = torch.randn_like(x1_margin_sample)
            mix_est = self.covariance2entropy_estimator_GMM(x_n)
            bias_est_1a = self.covariance2entropy_singleGaussian(x_n) - mix_est

            x_n = torch.randn_like(x2_margin_sample)
            mix_est = self.covariance2entropy_estimator_GMM(x_n)
            bias_est_1b = self.covariance2entropy_singleGaussian(x_n) - mix_est
            #
            x_n = torch.randn_like(x1x2_joint_sample)
            mix_est = self.covariance2entropy_estimator_GMM(x_n)
            bias_est_2 = self.covariance2entropy_singleGaussian(x_n) - mix_est

            Hx = self.covariance2entropy_estimator_GMM(x1_margin_sample)
            Hy = self.covariance2entropy_estimator_GMM(x2_margin_sample)
            Hxy = self.covariance2entropy_estimator_GMM(x1x2_joint_sample)

        # use H(x|y)=H(x,y)-H(x)
        # use MI(x,y)=H(x)+H(y)-H(x,y)
        MIx_y_unbias = Hx + Hy - Hxy  + bias_est_1a+bias_est_1b-bias_est_2

        return MIx_y_unbias
