#!/usr/bin/env python
# coding: utf-8


from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from utils_smile import *
from utils_mile import *
from estimators import estimate_mutual_information
from log import get_logger, set_logger


def train_estimator(df, critic_params, data_params, mi_params, opt_params, **kwargs):
    """Main training loop that estimates time-varying MI."""
    # Ground truth rho is only used by conditional critic
    critic = CRITICS[mi_params.get('critic', 'separable')](
        rho=None, **critic_params).cuda()
    baseline = BASELINES[mi_params.get('baseline', 'constant')]()

    opt_crit = optim.Adam(critic.parameters(), lr=opt_params['learning_rate'])
    if isinstance(baseline, nn.Module):
        opt_base = optim.Adam(baseline.parameters(),
                              lr=opt_params['learning_rate'])
    else:
        opt_base = None

    def train_step(rho, data_params, mi_params):
        # Annoying special case:
        # For the true conditional, the critic depends on the true correlation rho,
        # so we rebuild the critic at each iteration.
        opt_crit.zero_grad()
        if isinstance(baseline, nn.Module):
            opt_base.zero_grad()

        if mi_params['critic'] == 'conditional':
            critic_ = CRITICS['conditional'](rho=rho).cuda()
        else:
            critic_ = critic

        x, y = sample_correlated_gaussian(
            dim=data_params['dim'], rho=rho, batch_size=data_params['batch_size'], y_transform=data_params['y_transform'])
        mi = estimate_mutual_information(
            mi_params['estimator'], x, y, critic_, baseline, mi_params.get('alpha_logit', None), **kwargs)        
        loss = -mi
        loss.backward()
        opt_crit.step()
        if isinstance(baseline, nn.Module):
            opt_base.step()

        return mi
    
    def test_step(df, true_mi, rho, data_params, mi_params, clip=None):
        if mi_params['critic'] == 'conditional':
                critic_ = CRITICS['conditional'](rho=rho).cuda()
        else:
            critic_ = critic
        
        mis = []
        # for index in range(2):  # commented
        for index in range(data_params['batch_size']):
            x, y = sample_correlated_gaussian(
                dim=data_params['dim'], rho=rho, batch_size=data_params['batch_size'], y_transform=data_params['y_transform'])
            mi = estimate_mutual_information(
                mi_params['estimator'], x, y, critic_, baseline, mi_params.get('alpha_logit', None), **kwargs)
            mis.append(mi.detach().cpu().numpy())
            
        mean = np.mean(mis)
        bias = mean - true_mi
        var = np.var(mis)
        mse = (np.square(np.subtract(mis, true_mi))).mean()
        
        y_transform = data_params['y_transform']
        key = mi_params['estimator']
        if clip is not None:
            key = f'{key}_{clip}'
        estimator = find_name(key)
        critic_name = mi_params['critic']
        logger.debug(f'y_transform {y_transform}, estimator {estimator}, critic {critic_name}, true_mi {true_mi}')
        logger.debug(f'bias:{bias:.4f}, var:{var:.4f}, mse:{mse:.4f}')

        df.at[('bias', estimator), true_mi] = round(bias, 4)
        df.at[('var', estimator), true_mi] = round(var, 4)
        df.at[('mse', estimator), true_mi] = round(mse, 4)
    
    # Schedule of correlation over iterations
    mis = mi_schedule(opt_params['mis'], opt_params['iterations'])
    rhos = mi_to_rho(data_params['dim'], mis)

    estimates = []
    for i in range(opt_params['iterations']):
        mi = train_step(rhos[i], data_params, mi_params)
        mi = mi.detach().cpu().numpy()
        estimates.append(mi)
        step_per_mi = opt_params['iterations']//len(opt_params['mis'])
        if mi_params['critic'] == 'concat' and i % step_per_mi == step_per_mi - 1:
            test_step(df, mis[i], rhos[i], data_params, mi_params, **kwargs)
    
    return np.array(estimates)


def mile_stat(df, mile_name, data_params, opt_params):
    def eval_step(rho, dim, y_transform='gaussian'):
        data_size = 16384
        X, Y = sample_correlated_gaussian(rho, dim, batch_size=data_size, y_transform=y_transform)
        # MI_estimator = MI_LogDet_Estimator(Kmax=5, beta=1e-3, method='Kmeans')
        if y_transform == 'cubic':
            MI_estimator = MI_LogDet_Estimator(beta=1e-3, Kmax=10, Kmax_X=2, Kmax_Y=10, Kmax_XY=10, method='GMM')
        else:
            MI_estimator = MI_LogDet_Estimator(beta=1e-3, Kmax=5, Kmax_X=1, Kmax_Y=1, Kmax_XY=1, method='Kmeans')
        kmeans_5_bias, kmeans_5_unbias, kmeans_5_lower, kmeans_5_upper = MI_estimator.MILE_estimate(X, Y)
        return kmeans_5_unbias.item(), kmeans_5_lower.item(), kmeans_5_upper.item()

    y_transform = data_params['y_transform']
    mis = mi_schedule(opt_params['mis'], opt_params['iterations_mile'])
    rhos = mi_to_rho(data_params['dim'], mis)

    kmeans_5_unbias_list = []
    kmeans_5_lower_list = []
    kmeans_5_upper_list = []

    for index in range(opt_params['iterations_mile']):
        kmeans_5_unbias, kmeans_5_lower, kmeans_5_upper = eval_step(rhos[index], data_params['dim'], y_transform=y_transform)
        step_per_mi = opt_params['iterations_mile']//len(opt_params['mis'])
        if index % step_per_mi == step_per_mi - 1:
            logger.debug(f'y_transform {y_transform}, true MI {mis[index]}')
            logger.debug(f'kmeans_5_unbias {kmeans_5_unbias}')
            logger.debug(f'kmeans_5_lower {kmeans_5_lower}')
            logger.debug(f'kmeans_5_upper {kmeans_5_upper}')

        kmeans_5_unbias_list.append(kmeans_5_unbias)
        kmeans_5_lower_list.append(kmeans_5_lower)
        kmeans_5_upper_list.append(kmeans_5_upper)
                                        
    mile_dict = dict()
    mile_dict[f'kmeans_5_unbias'] = kmeans_5_unbias_list
    mile_dict[f'kmeans_5_lower'] = kmeans_5_lower_list
    mile_dict[f'kmeans_5_upper'] = kmeans_5_upper_list
    
    for true_mi in opt_params['mis']:
        rho = mi_to_rho(data_params['dim'], true_mi)
        mis = []
        for index in range(data_params['batch_size']):
            kmeans_5_unbias, kmeans_5_lower, kmeans_5_upper = eval_step(rho, data_params['dim'], y_transform=y_transform)
            mis.append(kmeans_5_unbias)
            
        mean = np.mean(mis)
        bias = mean - true_mi
        var = np.var(mis)
        mse = (np.square(np.subtract(mis, true_mi))).mean()
        logger.debug(f'y_transform {y_transform}, estimator {mile_name}, true_mi {true_mi}')
        logger.debug(f'bias:{bias:.4f}, var:{var:.4f}, mse:{mse:.4f}')

        df.at[('bias', mile_name), true_mi] = round(bias, 4)
        df.at[('var', mile_name), true_mi] = round(var, 4)
        df.at[('mse', mile_name), true_mi] = round(mse, 4)
        
    return mile_dict


def find_name(name):
    if 'smile_' in name:
        clip = name.split('_')[-1]
        return f'SMILE (τ={clip})'
    else:
        return {
            'infonce': 'CPC',
            'js': 'JS',
            'nwj': 'NWJ',
            'flow': 'GM (Flow)',
            'smile': 'SMILE (τ=∞)'
        }[name]

def find_legend(label):
    return {'concat': 'Joint critic', 'separable': 'Separable critic'}[label]


# Plot 5 of the results, InfoNCE, NWJ, Smile 1.0, 5.0, infty
def plot(opt_params, mi_numpys, mile_dict, file):
    ncols = 5
    nrows = 1
    EMA_SPAN = 200
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axs = np.ravel(axs)
    mi_true = mi_schedule(opt_params['mis'], opt_params['iterations'])            
    for i, estimator in enumerate(['infonce', 'nwj']):
        key = f'{estimator}'
        plt.sca(axs[i])
        plt.title(find_name(key), fontsize=16)
        for net in ['concat', 'separable']:
            mis = mi_numpys[net][key]
            p1 = plt.plot(mis, alpha=0.3)[0]
            mis_smooth = pd.Series(mis).ewm(span=EMA_SPAN).mean()
            plt.plot(mis_smooth, c=p1.get_color(), label=find_legend(net))
        plt.ylim(0, 11)
        plt.xlim(0, opt_params['iterations'])
        plt.plot(mi_true, color='k', label='True MI')
        plt.ylabel('MI (nats)')
        plt.xlabel('Steps')
        plt.axhline(np.log(64), color='k', ls='--', label='log(bs)')
        plt.legend(loc="upper left")
        
    estimator = 'smile'
    for i, clip in enumerate([5.0, None]):
        if clip is None:
            key = estimator
        else:
            key = f'{estimator}_{clip}'

        plt.sca(axs[i+2])
        plt.title(find_name(key), fontsize=16)
        for net in ['concat', 'separable']:
            mis = mi_numpys[net][key]
            EMA_SPAN = 200
            p1 = plt.plot(mis, alpha=0.3)[0]
            mis_smooth = pd.Series(mis).ewm(span=EMA_SPAN).mean()
            plt.plot(mis_smooth, c=p1.get_color(), label=find_legend(net))
        plt.ylim(0, 11)
        plt.xlim(0, opt_params['iterations'])
        plt.plot(mi_true, color='k', label='True MI')
        plt.ylabel('MI (nats)')
        plt.xlabel('Steps')
        plt.axhline(np.log(64), color='k', ls='--', label='log(bs)')
        plt.legend(loc="upper left")
        
    # dedicated for mile
    mi_true = mi_schedule(opt_params['mis'], opt_params['iterations_mile'])
    plt.sca(axs[ncols-1])
    plt.title('MILE', fontsize=16)    
    X = np.arange(opt_params['iterations_mile'])
    plt.ylim(0, 11)
    plt.xlim(0, opt_params['iterations_mile'])
    plt.ylabel('MI (nats)')
    plt.xlabel('Steps')
    # plt.plot(X, mile_dict['kmeans_5_unbias'], label="Mile estimation")
    # plt.fill_between(
    #     X.ravel(),
    #     mile_dict['kmeans_5_lower'],
    #     mile_dict['kmeans_5_upper'],
    #     alpha=0.5,
    #     label=r"Lower and Upper Bound",
    # )
    plt.plot(X, np.squeeze(np.array(mile_dict['kmeans_5_upper'])),'--',color='gray',marker='v')
    plt.plot(X, np.squeeze(np.array(mile_dict['kmeans_5_unbias'])),'-',color='darkblue',marker='o')
    plt.plot(X, np.squeeze(np.array(mile_dict['kmeans_5_lower'])),'--',color='gray',marker='^')
    plt.plot(mi_true, color='k', label='True MI')
    plt.axhline(np.log(64), color='k', ls='--', label='log(bs)')
    plt.legend(['$MILE$'+' '+'$\widehat{I}^u$', '$MILE$'+' '+'$\widehat{I}^m$', '$MILE$'+' '+'$\widehat{I}^l$', 'True MI', 'log(bs)'])
    plt.gcf().tight_layout()
    plt.savefig(file)
    plt.close()


if __name__ == "__main__":
    logger = get_logger()
    log_dir = Path('log')
    set_logger(logger, log_dir / "smile.log")
    
    dim = 20
    mi_seeds = [2,4,6,8,10]
    # dim = 100
    # mi_seeds = [10,20,30,40,50]
    iterations = int(2e4)   # commented
    iterations_mile = 500   # commented
    
    CRITICS = {
        'separable': SeparableCritic,
        'concat': ConcatCritic,
    }

    BASELINES = {
        'constant': lambda: None,
        'unnormalized': lambda: mlp(dim=dim, hidden_dim=512, output_dim=1, layers=2, activation='relu').cuda(),
        'gaussian': lambda: log_prob_gaussian,
    }

    critic_params = {
        'dim': dim,
        'layers': 2,
        'embed_dim': 32,
        'hidden_dim': 256,
        'activation': 'relu',
    }

    opt_params = {
        'mis': mi_seeds,
        'iterations': iterations,
        'iterations_mile': iterations_mile,
        'learning_rate': 5e-4,
    }
    
    plot_dir = Path(f'plot')
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    y_transforms = ['gaussian', 'cubic', 'sin']
    estimators = ['infonce', 'nwj', 'smile']
    clips = [5.0]
    
    algorithms = []
    for estimator in estimators:
        algorithm = find_name(estimator)
        algorithms.append(algorithm)
    for clip in clips:
        key = f'smile_{clip}'
        algorithm = find_name(key)
        algorithms.append(algorithm)
    mile_name = 'MILE'
    algorithms.append(mile_name)
    # logger.debug(f'algorithms {algorithms}')    
    
    midx = pd.MultiIndex.from_product([['bias', 'var', 'mse'], algorithms])
    
    excel_dir = Path('doc')
    excel_dir.mkdir(exist_ok=True, parents=True)
    excel_path = excel_dir / f'smile_{dim}.xlsx'
    writer_mi = pd.ExcelWriter(excel_path, engine='xlsxwriter')
    
    for y_transform in y_transforms:
        data_params = {
            'dim': dim,
            'batch_size': 64,
            'y_transform': y_transform
        }
        
        mi_numpys = dict()
        df = pd.DataFrame([], index=midx, columns=mi_seeds)

        mile_dict = mile_stat(df, mile_name, data_params, opt_params)
        
        for critic_type in ['separable', 'concat']:
            mi_numpys[critic_type] = dict()
            for estimator in estimators:
                mi_params = dict(estimator=estimator, critic=critic_type, baseline='unnormalized')
                mis = train_estimator(df, critic_params, data_params, mi_params, opt_params)
                mi_numpys[critic_type][f'{estimator}'] = mis

            estimator = 'smile'
            for i, clip in enumerate(clips):
                mi_params = dict(estimator=estimator, critic=critic_type, baseline='unnormalized')
                mis = train_estimator(df, critic_params, data_params, mi_params, opt_params, clip=clip)
                mi_numpys[critic_type][f'{estimator}_{clip}'] = mis
        
        file = plot_dir/f'smile_{y_transform}_{dim}.png'
        plot(opt_params, mi_numpys, mile_dict, file)
        df.to_excel(writer_mi, sheet_name=data_params['y_transform'])

    writer_mi.close()