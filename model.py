
import torch
import torch.nn as nn
import torch.nn.functional as F


class NetImage(nn.Module):
    def __init__(self, image_size = 28, n_channels = 1):
        super(NetImage, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(n_channels * 2, 64, 5, stride=2, padding=2), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 5, stride=2, padding=2), nn.ReLU())
        self.fc1 = nn.Sequential(nn.Linear(128 * ((image_size//4)**2), 1024), nn.ReLU())
        self.fc2 = nn.Linear(1024, 1)

    def critic(self, xy_pairs):
        h = self.conv1(xy_pairs)
        h = self.conv2(h)
        h = torch.flatten(h, 1)
        h = self.fc1(h)
        h = self.fc2(h)
        return h
    
    def forward(self, x, y):
        # print(f'x {x.shape}, y {y.shape}')
        batch_size = x.size(0)
        # Tile all possible combinations of x and y
        x_tiled = torch.stack([x] * batch_size, dim=0)
        y_tiled = torch.stack([y] * batch_size, dim=1)
        # xy is [batch_size * batch_size, x_dim + y_dim]
        xy_matrix = torch.cat((x_tiled, y_tiled), dim=2)
        xy_pairs = torch.flatten(xy_matrix, end_dim=1)
        # print(f'xy_matrix {xy_matrix.shape}, xy_pairs {xy_pairs.shape}')
        # Compute scores for each x_i, y_j pair.
        scores = self.critic(xy_pairs)
        return torch.reshape(scores, [batch_size, batch_size]).t()


class NetImage2(NetImage):
    def __init__(self, image_size = 28, n_channels = 1):
        super(NetImage2, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(n_channels * 4, 64, 5, stride=2, padding=2), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 5, stride=2, padding=2), nn.ReLU())
        self.fc1 = nn.Sequential(nn.Linear(128 * ((image_size//4)**2), 1024), nn.ReLU())
        self.fc2 = nn.Linear(1024, 1)