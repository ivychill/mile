
import torch.optim as optim
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
from model import NetImage, NetImage2
from loss import *
from log import get_logger


class Baseline(object):
    def __init__(self, device, datasource):
        super(Baseline, self).__init__()
        self.logger = get_logger()
        self.device = device
        self.datasource = datasource
        self.range = range(self.datasource['image_size'])
        # self.range = [self.datasource['image_size']-1]   # commented
        self.range_mile = range(1, self.datasource['image_size'])
        # self.range_mile = [1, self.datasource['image_size']-1]     # commented

    def export(self, denominator):
        self.mi_x_y = dict()
        estimator_names = []
        estimator = MILE()
        estimator_name = estimator.__class__.__name__
        estimator_names.append(estimator_name)
        divergences = [CPC(), MINE(), SMILE(5), SMILE()]
        for divergence in divergences:
            divergence_name = divergence.name()
            estimator_names.append(divergence_name)
        midx = pd.MultiIndex.from_product([estimator_names, ['mean', 'ci_lower', 'ci_upper']])
        self.df_mi = pd.DataFrame([], index=midx, columns=self.range)
        self.df_percentage = pd.DataFrame([], index=midx, columns=self.range)
        self.mile(denominator)     # commented
        self.mi_benchmark(denominator)     # commented
        return self.mi_x_y, self.df_mi, self.df_percentage

    def mile(self, denominator):        
        image_size = self.datasource['image_size']
        estimator = MILE()
        estimator_name = estimator.__class__.__name__
        mis = [None] * image_size
        ci_lowers = [None] * image_size
        ci_uppers = [None] * image_size
        for t_row in self.range_mile:
            self.t_row = t_row + 1
            datasource_name = self.datasource['name']
            self.logger.debug(f'{datasource_name}, {self.__class__.__name__}, {estimator_name}, t_row {self.t_row}')
            Ts = []
            for i, (data, target) in enumerate(self.datasource['test_loader_mile']):
                x_sample, y_sample = self.transform(data)
                ret = estimator(x_sample, y_sample)
                Ts.append(ret.item())
            Ts = 1.0 * np.array(Ts)
            mean = np.mean(Ts)
            ci = np.std(Ts)
            # ci = 1.96 * np.std(Ts) / np.sqrt(len(Ts))   # 95% confidence
            ci_lower = mean - ci
            ci_upper = mean + ci
            mis[t_row] = mean
            ci_lowers[t_row] = ci_lower
            ci_uppers[t_row] = ci_upper
            
        self.set_denominator(estimator_name, mis)
        for index, mi in enumerate(mis):
            # self.df_mi.at[estimator_name, index+1] = mi
            self.df_mi.at[(estimator_name, 'mean'), index+1] = mi
        for index, ci_lower in enumerate(ci_lowers):
            self.df_mi.at[(estimator_name, 'ci_lower'), index+1] = ci_lower
        for index, ci_upper in enumerate(ci_uppers):
            self.df_mi.at[(estimator_name, 'ci_upper'), index+1] = ci_upper
            
        mean_percentages = self.get_percent(mis, denominator, estimator_name)
        ci_lower_percentages = self.get_percent(ci_lowers, denominator, estimator_name)
        ci_upper_percentages = self.get_percent(ci_uppers, denominator, estimator_name)
        for index, mean_percentage in enumerate(mean_percentages):
            # self.df_percentage.at[estimator_name, index+1] = percentage
            self.df_percentage.at[(estimator_name, 'mean'), index+1] = mean_percentage
        for index, ci_lower_percentage in enumerate(ci_lower_percentages):
            self.df_percentage.at[(estimator_name, 'ci_lower'), index+1] = ci_lower_percentage
        for index, ci_upper_percentage in enumerate(ci_upper_percentages):
            self.df_percentage.at[(estimator_name, 'ci_upper'), index+1] = ci_upper_percentage

    def mi_benchmark(self, denominator):
        # divergences = [SMILE(5)]    # commented
        divergences = [CPC(), MINE(), SMILE(5), SMILE()]
        for divergence in divergences:
            divergence_name = divergence.name()
            self.f_divergence = divergence
            image_size = self.datasource['image_size']
            mis = [None] * image_size
            ci_lowers = [None] * image_size
            ci_uppers = [None] * image_size
            mis = [None] * image_size
            for t_row in self.range:
                self.t_row = t_row + 1
                datasource_name = self.datasource['name']
                self.logger.debug(f'{datasource_name}, {self.__class__.__name__}, {divergence_name}, t_row {self.t_row}')
                self.model = self.get_model()
                # optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0005)
                self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
                # scheduler = lr_scheduler.StepLR(optimizer, 20, gamma=0.8)
                self.train()
                mean, ci_lower, ci_upper = self.test()  # whole test set
                mis[t_row] = mean
                ci_lowers[t_row] = ci_lower
                ci_uppers[t_row] = ci_upper
                
            self.set_denominator(divergence_name, mis)
            for index, mi in enumerate(mis):
                self.df_mi.at[(divergence_name, 'mean'), index+1] = mi
            for index, ci_lower in enumerate(ci_lowers):
                self.df_mi.at[(divergence_name, 'ci_lower'), index+1] = ci_lower
            for index, ci_upper in enumerate(ci_uppers):
                self.df_mi.at[(divergence_name, 'ci_upper'), index+1] = ci_upper
                
            mean_percentages = self.get_percent(mis, denominator, divergence_name)
            ci_lower_percentages = self.get_percent(ci_lowers, denominator, divergence_name)
            ci_upper_percentages = self.get_percent(ci_uppers, denominator, divergence_name)
            for index, mean_percentage in enumerate(mean_percentages):
                self.df_percentage.at[(divergence_name, 'mean'), index+1] = mean_percentage
            for index, ci_lower_percentage in enumerate(ci_lower_percentages):
                self.df_percentage.at[(divergence_name, 'ci_lower'), index+1] = ci_lower_percentage
            for index, ci_upper_percentage in enumerate(ci_upper_percentages):
                self.df_percentage.at[(divergence_name, 'ci_upper'), index+1] = ci_upper_percentage

    def get_model(self):
        return NetImage(image_size = self.datasource['image_size'], n_channels = self.datasource['n_channels']).to(self.device)

    def set_denominator(self, divergence_name, mis):
        self.mi_x_y[divergence_name] = mis
        self.logger.debug(f'{self.__class__.__name__} mi_x_y {np.array(self.mi_x_y)}')

    def get_percent(self, mis, denominator, divergence_name):
        percentage = [x/mis[-1] 
                      if (x is not None and 
                          x is not np.nan and 
                          mis[-1]!=0 and 
                          mis[-1] is not None and
                          mis[-1] is not np.nan)
                      else 0 
                      for x in mis]
        return percentage

    def train(self):
        self.model.train()
        Ts = []
        n_epoch = 2
        for epoch in range(n_epoch):
            # scheduler.step()
            for i, (data, target) in enumerate(self.datasource['train_loader']):
                ret = self.cal_loss(data)
                Ts.append(ret.item())
                loss = - ret  # maximize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # self.logger.debug(f'train Ts {np.array(Ts)}') # len 1876
        # self.plot_T(Ts)
    
    def test(self):
        self.model.eval()
        Ts = []
        for i, (data, target) in enumerate(self.datasource['test_loader']):
            ret = self.cal_loss(data)
            Ts.append(ret.item())

        Ts = 1.0 * np.array(Ts)
        
        mean = np.mean(Ts)
        ci = np.std(Ts)
        # ci = 1.96 * np.std(Ts) / np.sqrt(len(Ts))   # 95% confidence
        ci_lower = mean - ci
        ci_upper = mean + ci
        return mean, ci_lower, ci_upper
    
    # def mean_confidence_interval(self, data, confidence=0.90):
    #     # a = 1.0 * np.array(data)
    #     n = len(data)
    #     m, se = np.mean(data), scipy.stats.sem(data)
    #     h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    #     return m, m - h, m + h

    def cal_loss(self, data):
        data = data.to(self.device)
        x_sample, y_sample = self.transform(data)
        scores = self.model(x_sample, y_sample)
        ret = self.f_divergence(scores)
        return ret

    def transform(self, data):
        x_sample = data  # torch.Size([128, 1, 28, 28]) (B, C, H, W)
        y_sample = x_sample.clone()
        height = y_sample.size()[-2]
        y_sample[:, :, self.t_row:height, :] = 0
        return x_sample, y_sample

    def plot_T(self, Ts):
        plot_x = np.arange(len(Ts))
        plt.plot(plot_x, Ts)
        plt.show()


class DataProcessing(Baseline):
    def __init__(self, device, datasource):
        super(DataProcessing, self).__init__(device, datasource)
        self.t2_minus_t1 = 3
        self.range = range(self.t2_minus_t1, self.datasource['image_size'])
        # self.range = [self.datasource['image_size']-1]  # commented
        self.range_mile = range(self.t2_minus_t1, self.datasource['image_size'])
        # self.range_mile = [self.t2_minus_t1, self.datasource['image_size']-1]  # commented

    def get_model(self):
        return NetImage2(image_size = self.datasource['image_size'], n_channels = self.datasource['n_channels']).to(self.device)

    def set_denominator(self, divergence_name, mean):
        pass

    def get_percent(self, mis, denominator, divergence_name):
        mi_x_y = denominator[divergence_name]
        self.logger.debug(f'{divergence_name}, mi {mis}, mi_x_y {mi_x_y}')
        percentage = [x/y
                      if (x is not None and
                          x is not np.nan and
                          y != 0 and
                          y is not None and
                          y is not np.nan)
                      else 0
                      for (x,y) in zip(mis, mi_x_y)]
        return percentage

    def transform(self, data):
        x_sample = torch.cat((data, data), dim=1)
        # print(f'x_sample {x_sample.shape}')
        y_sample = x_sample.clone()
        channel = y_sample.size()[1]
        height = y_sample.size()[-2]
        width = y_sample.size()[-1]
        # y_sample[:, :, self.t_row:height, :width//2] = 0
        # y_sample[:, :, self.t_row-self.t2_minus_t1:height, width//2:] = 0
        y_sample[:, :channel//2, self.t_row:height, :] = 0
        y_sample[:, channel//2:, self.t_row-self.t2_minus_t1:height, :] = 0
        return x_sample, y_sample


class Additivity(Baseline):
    def __init__(self, device, datasource):
        super(Additivity, self).__init__(device, datasource)

    def get_model(self):
        return NetImage2(image_size = self.datasource['image_size'], n_channels = self.datasource['n_channels']).to(self.device)

    def set_denominator(self, divergence_name, mean):
        pass

    def get_percent(self, mis, denominator, divergence_name):
        mi_x_y = denominator[divergence_name]
        self.logger.debug(f'{divergence_name}, mi {mis}, mi_x_y {mi_x_y}')
        percentage = [x/y
                      if (x is not None and
                          x is not np.nan and
                          y != 0 and
                          y is not None and
                          y is not np.nan)
                      else 0
                      for (x,y) in zip(mis, mi_x_y)]
        return percentage

    def transform(self, data):
        pal = data[torch.randperm(data.size()[0])]
        x_sample = torch.cat((data, pal), dim=1)
        y_sample = x_sample.clone()
        height = y_sample.size()[-2]
        y_sample[:, :, self.t_row:height, :] = 0
        return x_sample, y_sample
