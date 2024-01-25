import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils_mile_ini import *
from utils_smile import mi_schedule, mi_to_rho, rho_to_mi, sample_correlated_gaussian


def eval_step(rho, dim):
    data_size = 4096
    X, Y = sample_correlated_gaussian(rho, dim, batch_size=data_size, cubic=True)
    #use logDet MI estimator
    MI_estimator = MI_LogDet_Estimator(Kmax=1, beta=0.0, method='Kmeans')
    mi_est,_,_,_ = MI_estimator.MILE_estimate(X,Y)
    MI_estimator = MI_LogDet_Estimator(Kmax=5, beta=1e-3, method='Kmeans')
    mi_est_bias, mi_est_unbias, mi_lower, mi_upper = MI_estimator.MILE_estimate(X,Y)
    MI_estimator = MI_LogDet_RobustEstimator(Kmax=5,beta=1e-3,method='Kmeans')
    mi_robust_est_unbias = MI_estimator.MILE_estimate(X,Y)
    
    return mi_est.cpu().numpy(), mi_est_bias.cpu().numpy(), mi_est_unbias.cpu().numpy(), mi_lower.cpu().numpy(), mi_upper.cpu().numpy(), mi_robust_est_unbias.cpu().numpy()


def plot(est_dict, iterations):
    mi_true = mi_schedule(iterations)
    ncols = 6
    nrows = 1
    # EMA_SPAN = 200
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axs = np.ravel(axs)
    
    for i, key in enumerate(est_dict):
        plt.sca(axs[i])
        plt.title(key, fontsize=18)
        mis = est_dict[key]
        p1 = plt.plot(mis, alpha=0.3)[0]
        plt.plot(mis, c=p1.get_color(), label=key)
        # mis_smooth = pd.Series(mis).ewm(span=EMA_SPAN).mean()
        # plt.plot(mis_smooth, c=p1.get_color(), label=key)
        plt.ylim(0, 11)
        plt.xlim(0, iterations)
        plt.plot(mi_true, color='k', label='True MI')
        if i == 0:
            plt.ylabel('MI (nats)')
            plt.xlabel('Steps')
            plt.legend()
    
    plt.gcf().tight_layout()
    # plt.savefig(f'gaussian.png')
    plt.savefig(f'cubic_20.png')
    plt.close()


if __name__ == "__main__":
    time_start = time.time()
    # seed = int(5e3)
    # seed_torch(seed)
    dim = 20
    iterations = 20

    # Schedule of correlation over iterations
    mis = mi_schedule(iterations)
    rhos = mi_to_rho(dim, mis)

    mi_est_list = []
    mi_est_bias_list = []
    mi_est_unbias_list = []
    mi_lower_list = []
    mi_upper_list = []
    mi_robust_est_unbias_list = []
    for index in range(iterations):
        mi_est, mi_est_bias, mi_est_unbias, mi_lower, mi_upper, mi_robust_est_unbias = eval_step(rhos[index], dim)
        if index % 1000 == 0:
                print(datetime.now())
                print('MILE_logdet',mi_est)
                print('MILE_logdet_ubias', mi_est_unbias)
                print('MILE_logdet_bias', mi_est_bias)
                print('MILE_logdet_lower', mi_lower)
                print('MILE_logdet_upper', mi_upper)
                print('robust MILE_logdet_ubias',mi_robust_est_unbias)
        mi_est_list.append(mi_est)
        mi_est_bias_list.append(mi_est_bias)
        mi_est_unbias_list.append(mi_est_unbias)
        mi_lower_list.append(mi_lower)
        mi_upper_list.append(mi_upper)
        mi_robust_est_unbias_list.append(mi_robust_est_unbias)
                                        
    est_dict = dict()
    est_dict[f'mi_est'] = mi_est_list
    est_dict[f'mi_est_bias'] = mi_est_bias_list
    est_dict[f'mi_est_unbias'] = mi_est_unbias_list
    est_dict[f'mi_lower'] = mi_lower_list
    est_dict[f'mi_upper'] = mi_upper_list
    est_dict[f'mi_robust_est_unbias'] = mi_robust_est_unbias_list
    plot(est_dict, iterations)
    time_end = time.time()
    print('cost ' + str(time_end - time_start) + 's totally')