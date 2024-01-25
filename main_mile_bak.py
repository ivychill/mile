import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils_mile import *
from utils_smile import mi_schedule, mi_to_rho, rho_to_mi, sample_correlated_gaussian


def get_pair(X, Y, all_pairs=False):
    x_pos = torch.concat((X,Y), dim=1)
    if all_pairs == True:
        x_tiled = torch.stack([X] * X.shape[0], dim=0)
        y_tiled = torch.stack([Y] * Y.shape[0], dim=1)
        x_neg = diag_remove(torch.cat((x_tiled, y_tiled), dim=2))
    else:
        y_shuffle = np.random.permutation(Y.numpy())
        y_shuffle = torch.from_numpy(y_shuffle).type(torch.FloatTensor)
        x_neg = torch.cat((X, y_shuffle), dim=1)
    return x_pos, x_neg


def eval_step(rho, dim, cubic):
    # data_size = 1024
    data_size = 16384
    X, Y = sample_correlated_gaussian(rho, dim, batch_size=data_size, cubic=cubic)

    # time_start = time.time()
    
    MI_estimator = MI_LogDet_Estimator(Kmax=1, beta=0.0, method='Kmeans')
    single_gaussian, _, _, _ = MI_estimator.MILE_estimate(X,Y)
    
    # time_end = time.time()
    # print('single_gaussian cost ' + str(time_end - time_start) + 's')
    
    MI_estimator = MI_LogDet_Estimator(Kmax=5, beta=1e-3, method='Kmeans')
    kmeans_5_bias, kmeans_5_unbias, kmeans_5_lower, kmeans_5_upper = MI_estimator.MILE_estimate(X,Y)
    
    # time_end = time.time()
    # print('kmeans_5_bias, kmeans_5_unbias, kmeans_5_lower, kmeans_5_upper cost ' + str(time_end - time_start) + 's')
    
    # MI_estimator = MI_LogDet_RobustEstimator(Kmax=10, beta=1e-3, method='GMM')
    # robust_gmm_10_unbias = MI_estimator.MILE_estimate(X, Y)
    # MI_estimator = MI_LogDet_RobustEstimator(Kmax=5, beta=1e-3, method='GMM')
    # robust_gmm_5_unbias = MI_estimator.MILE_estimate(X, Y)

    MI_estimator = MI_LogDet_RobustEstimator(Kmax=1, beta=1e-3, method='Kmeans')
    robust_kmeans_5_unbias = MI_estimator.MILE_estimate(X,Y)
    
    # time_end = time.time()
    # print('robust_kmeans_5_unbias cost ' + str(time_end - time_start) + 's')
    
    MI_estimator = MI_LogDet_RobustEstimator(Kmax=1, beta=1e-3, method='Kmeans')
    x_pos, x_neg = get_pair(X, Y, all_pairs=False) 
    robust_kmeans_5_pair_unbias = MI_estimator.MILE_estimate_pairs(x_pos, x_neg)    
    x_pos, x_neg = get_pair(X, Y, all_pairs=True)
    robust_kmeans_5_matrix_unbias = MI_estimator.MILE_estimate_pairs(x_pos, x_neg)
    
    return single_gaussian.cpu().numpy(), kmeans_5_bias.cpu().numpy(), kmeans_5_unbias.cpu().numpy(), kmeans_5_lower.cpu().numpy(), kmeans_5_upper.cpu().numpy(), robust_kmeans_5_unbias.cpu().numpy(), robust_kmeans_5_pair_unbias.cpu().numpy(), robust_kmeans_5_matrix_unbias.cpu().numpy()
    # return single_gaussian.cpu().numpy(), kmeans_5_bias.cpu().numpy(), kmeans_5_unbias.cpu().numpy(), kmeans_5_lower.cpu().numpy(), kmeans_5_upper.cpu().numpy(), robust_gmm_10_unbias.cpu().numpy(), robust_gmm_5_unbias.cpu().numpy(), robust_kmeans_5_unbias.cpu().numpy(), robust_kmeans_5_pair_unbias.cpu().numpy(), robust_kmeans_5_matrix_unbias.cpu().numpy()


def plot(est_dict, iterations, file):
    mi_true = mi_schedule(iterations)
    ncols = 4
    nrows = 2
    # EMA_SPAN = 200
    fig, axs = plt.subplots(nrows, ncols, figsize=(8 * ncols, 8 * nrows))
    axs = np.ravel(axs)
    
    for i, key in enumerate(est_dict):
        plt.sca(axs[i])
        plt.title(key, fontsize=24)
        plt.plot(mi_true, color='k', label='True MI')
        mis = est_dict[key]
        p1 = plt.plot(mis, alpha=0.3)[0]
        plt.plot(mis, c=p1.get_color(), label='MI estimation')
        # mis_smooth = pd.Series(mis).ewm(span=EMA_SPAN).mean()
        # plt.plot(mis_smooth, c=p1.get_color(), label=key)
        plt.ylim(0, 11)
        plt.xlim(0, iterations)
        plt.ylabel('MI (nats)', fontsize=24)
        plt.xlabel('Steps', fontsize=24)
        plt.xticks(size = 24)
        plt.yticks(size = 24)
        plt.legend(loc="upper left", fontsize=24)
    
    plt.gcf().tight_layout()
    plt.savefig(file)
    plt.close()


if __name__ == "__main__":
    time_start = time.time()
    # seed = int(0)
    # seed_torch(seed)
    dim = 20
    # iterations = 10
    iterations = int(5e2)
    cubic = True
    fig_file = f'cubic_{iterations}.png'

    # Schedule of correlation over iterations
    mis = mi_schedule(iterations)
    rhos = mi_to_rho(dim, mis)

    single_gaussian_list = []
    kmeans_5_bias_list = []
    kmeans_5_unbias_list = []
    kmeans_5_lower_list = []
    kmeans_5_upper_list = []
    # robust_gmm_10_unbias_list = []
    # robust_gmm_5_unbias_list = []
    robust_kmeans_5_unbias_list = []
    robust_kmeans_5_pair_unbias_list = []
    robust_kmeans_5_matrix_unbias_list = []
    for index in range(iterations):
        single_gaussian, kmeans_5_bias, kmeans_5_unbias, kmeans_5_lower, kmeans_5_upper, robust_kmeans_5_unbias, robust_kmeans_5_pair_unbias, robust_kmeans_5_matrix_unbias = eval_step(rhos[index], dim, cubic=cubic)
        # single_gaussian, kmeans_5_bias, kmeans_5_unbias, kmeans_5_lower, kmeans_5_upper, robust_gmm_10_unbias, robust_gmm_5_unbias, robust_kmeans_5_unbias, robust_kmeans_5_pair_unbias, robust_kmeans_5_matrix_unbias = eval_step(rhos[index], dim, cubic=cubic)
        if index % (iterations/5/2) == 0:
                print(datetime.now())
                print(f'true MI {mis[index]}')
                print('single_gaussian', single_gaussian)
                print('kmeans_5_bias', kmeans_5_bias)
                print('kmeans_5_unbias', kmeans_5_unbias)
                print('kmeans_5_lower', kmeans_5_lower)
                print('kmeans_5_upper', kmeans_5_upper)
                # print('robust_gmm_10_unbias', robust_gmm_10_unbias)
                # print('robust_gmm_5_unbias', robust_gmm_5_unbias)
                print('robust_kmeans_5_unbias', robust_kmeans_5_unbias)
                print('robust_kmeans_5_pair_unbias', robust_kmeans_5_pair_unbias)
                print('robust_kmeans_5_matrix_unbias', robust_kmeans_5_matrix_unbias)

        single_gaussian_list.append(single_gaussian)
        kmeans_5_bias_list.append(kmeans_5_bias)
        kmeans_5_unbias_list.append(kmeans_5_unbias)
        kmeans_5_lower_list.append(kmeans_5_lower)
        kmeans_5_upper_list.append(kmeans_5_upper)
        # robust_gmm_10_unbias_list.append(robust_gmm_10_unbias)
        # robust_gmm_5_unbias_list.append(robust_gmm_5_unbias)
        robust_kmeans_5_unbias_list.append(robust_kmeans_5_unbias)
        robust_kmeans_5_pair_unbias_list.append(robust_kmeans_5_pair_unbias)
        robust_kmeans_5_matrix_unbias_list.append(robust_kmeans_5_matrix_unbias)
                                        
    est_dict = dict()
    est_dict[f'single_gaussian'] = single_gaussian_list
    est_dict[f'kmeans_5_bias'] = kmeans_5_bias_list
    est_dict[f'kmeans_5_unbias'] = kmeans_5_unbias_list
    est_dict[f'kmeans_5_lower'] = kmeans_5_lower_list
    est_dict[f'kmeans_5_upper'] = kmeans_5_upper_list
    # est_dict[f'robust_gmm_10_unbias'] = robust_gmm_10_unbias_list
    # est_dict[f'robust_gmm_5_unbias'] = robust_gmm_5_unbias_list
    est_dict[f'robust_kmeans_unbias'] = robust_kmeans_5_unbias_list
    est_dict[f'robust_kmeans_pair_unbias'] = robust_kmeans_5_pair_unbias_list
    est_dict[f'robust_kmeans_matrix_unbias'] = robust_kmeans_5_matrix_unbias_list
    
    plot(est_dict, iterations, fig_file)
    time_end = time.time()
    print('cost ' + str(time_end - time_start) + 's totally')