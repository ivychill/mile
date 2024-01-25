import time
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils_mile import *
from utils_smile import mi_schedule, mi_to_rho, rho_to_mi, sample_correlated_gaussian
from log import get_logger, set_logger


def eval_step(rho, dim, y_transform='gaussian'):
    data_size = 16384
    X, Y = sample_correlated_gaussian(rho, dim, batch_size=data_size, y_transform=y_transform)
    if y_transform == 'cubic':
        MI_estimator=MI_LogDet_Estimator(beta=1e-3,Kmax=10,Kmax_X=2,Kmax_Y=10,Kmax_XY=10,method='GMM')
    else:
        # MI_estimator = MI_LogDet_Estimator(Kmax=5, beta=1e-3, method='Kmeans')
        MI_estimator = MI_LogDet_Estimator(beta=1e-3, Kmax=5, Kmax_X=1, Kmax_Y=1, Kmax_XY=1, method='Kmeans')
    kmeans_5_bias, kmeans_5_unbias, kmeans_5_lower, kmeans_5_upper = MI_estimator.MILE_estimate(X, Y)
    
    return kmeans_5_unbias.item(), kmeans_5_lower.item(), kmeans_5_upper.item()


def plot(mi_true, est_dict, iterations, file):
    plt.title('MILE')    
    # X = np.linspace(start=0, stop=iterations, num=iterations).reshape(-1, 1)
    X = np.arange(iterations)
    plt.ylim(0, 11)
    # plt.xlim(0, iterations)
    plt.xlabel('Steps')
    plt.ylabel('MI (nats)')
    plt.plot(mi_true, color='k', label='True MI')
    # plt.plot(X, est_dict['kmeans_5_unbias'], label="Mile estimation")
    # plt.fill_between(
    #     X.ravel(),
    #     est_dict['kmeans_5_lower'],
    #     est_dict['kmeans_5_upper'],
    #     alpha=0.5,
    #     label=r"Lower and Upper Bound",
    # )
    # plt.legend(loc="upper left")
    # plt.plot(X, 10*np.ones(x.shape[0]),'-.',color='red')
    plt.plot(X, np.squeeze(np.array(est_dict['kmeans_5_upper'])),'--',color='gray',marker='v')
    plt.plot(X, np.squeeze(np.array(est_dict['kmeans_5_unbias'])),'-',color='darkblue',marker='o')
    plt.plot(X, np.squeeze(np.array(est_dict['kmeans_5_lower'])),'--',color='gray',marker='^')
    plt.legend(['True MI','$MILE$'+' '+'$\widehat{I}^u$','$MILE$'+' '+'$\widehat{I}^m$','$MILE$'+' '+'$\widehat{I}^l$'])
    plt.savefig(file)
    plt.close()


if __name__ == "__main__":
    time_start = time.time()
    # seed = int(0)
    # seed_torch(seed)
    logger = get_logger()
    log_dir = Path('log')
    set_logger(logger, log_dir / "mile.log")
    dim = 20
    iterations = 10
    # iterations = int(5e2)

    # Schedule of correlation over iterations
    mi_seeds = [2,4,6,8,10]
    mis = mi_schedule(mi_seeds, iterations)
    rhos = mi_to_rho(dim, mis)

    plot_dir = Path(f'plot')
    plot_dir.mkdir(parents=True, exist_ok=True)

    y_transforms = ['gaussian', 'cubic', 'sin']
    for y_transform in y_transforms:        
        kmeans_5_unbias_list = []
        kmeans_5_lower_list = []
        kmeans_5_upper_list = []

        for index in range(iterations):
            kmeans_5_unbias, kmeans_5_lower, kmeans_5_upper = eval_step(rhos[index], dim, y_transform=y_transform)
            if index % (iterations/5/2) == 0:
                logger.debug(f'y_transform {y_transform}, true MI {mis[index]}')
                logger.debug(f'kmeans_5_unbias {kmeans_5_unbias}')
                logger.debug(f'kmeans_5_lower {kmeans_5_lower}')
                logger.debug(f'kmeans_5_upper {kmeans_5_upper}')

            kmeans_5_unbias_list.append(kmeans_5_unbias)
            kmeans_5_lower_list.append(kmeans_5_lower)
            kmeans_5_upper_list.append(kmeans_5_upper)

                                            
        est_dict = dict()
        est_dict[f'kmeans_5_unbias'] = kmeans_5_unbias_list
        est_dict[f'kmeans_5_lower'] = kmeans_5_lower_list
        est_dict[f'kmeans_5_upper'] = kmeans_5_upper_list
        
        file = plot_dir/f'mile_{y_transform}.png'
        plot(mis, est_dict, iterations, file)
        time_end = time.time()
        logger.debug(f'cost {time_end - time_start} s totally')