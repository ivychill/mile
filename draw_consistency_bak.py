import numpy as np
import pandas as pd
import torch
from pathlib import Path
import matplotlib.pyplot as plt


def load(sheet_name, n_col):
    file = 'doc/self_consistency_percentage.xlsx'
    # cpc_list = []
    # smile_inf_list = []
    # smile_5_list = []
    # mile_list = []
    # for index in range(1, n_col+1):
        # cpc = df_mile.at['CPC', index]
        # smile_inf = df_mile.at['SMILE (τ=∞)', index]
        # smile_5 = df_mile.at['SMILE (τ=5.0)', index]
        # mile = df_mile.at['MILE', index]

    mi_dict = dict()
    df_mile = pd.read_excel(file, sheet_name=sheet_name, header=None)
    # print(df_mile)
    for index, row in df_mile[1:n_col].iterrows():
        row_elements = row.tolist()
        if row_elements[0] == 'MINE':
            continue
        mi_dict[row_elements[0]] = row_elements[1:n_col+1]

    # print(mi_dict)
    #     cpc_list.append(cpc)
    #     smile_inf_list.append(smile_inf)
    #     smile_5_list.append(smile_5)
    #     mile_list.append(mile)
    # mi_dict = dict()
    # mi_dict[f'CPC'] = cpc_list
    # mi_dict[f'SMILE (τ=∞)'] = smile_inf_list
    # mi_dict[f'SMILE (τ=5.0)'] = smile_5_list
    # mi_dict[f'MILE'] = mile_list
    
    return mi_dict


def plot(mi_dict, plot_file, datasource, transform, n_col):
    xlabel = 'Rows used'
    if transform == 'Baseline':
        title = f'{datasource} (Y = X(rows))'
        ylabel = 'Percentage'
        ylim = 1
    elif transform == 'DataProcessing':
        title = f'{datasource} (Data processing)'
        ylabel = 'Ratio'
        ylim = 1
    elif transform == 'Additivity':
        title = f'{datasource} (Additivity)'
        ylabel = 'Ratio'
        ylim = 2
    else:
        print('error......')
        exit(1)
        
    plt.title(title, fontsize=16)    
    # X = np.arange(opt_params['iterations_mile'])
    if datasource == 'MNIST' and transform == 'Additivity':
        plt.ylim(-0.25, 2.25)
    # plt.ylim(0, ylim)
    # plt.xlim(0, n_col)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    for key in mi_dict:
        plt.plot(mi_dict[key], label=key)

    if transform == 'DataProcessing':
        plt.axhline(1, color='k', ls='--', label='Ideal')
    elif transform == 'Additivity':
        plt.axhline(2, color='k', ls='--', label='Ideal')
        
    plt.legend()
    plt.savefig(plot_file)
    plt.close()
    

if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)
    torch.set_printoptions(threshold=np.inf)
    plot_dir = Path(f'plot')
    plot_dir.mkdir(parents=True, exist_ok=True)
    transforms = ['Baseline', 'DataProcessing', 'Additivity']
    datasources = ['MNIST', 'CIFAR10']
    datasource = 'MNIST'
    n_col = 28
    for transform in transforms:
        sheet_name = f'{datasource}_{transform}'
        mi_dict = load(sheet_name, n_col)
        plot_file = plot_dir/f'{datasource}_{transform}.png'
        plot(mi_dict, plot_file, datasource, transform, n_col)
    
    datasource = 'CIFAR10'
    n_col = 32
    for transform in transforms:
        sheet_name = f'{datasource}_{transform}'
        mi_dict = load(sheet_name, n_col)
        plot_file = plot_dir/f'{datasource}_{transform}.png'
        plot(mi_dict, plot_file, datasource, transform, n_col)