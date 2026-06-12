import numpy as np
import torch
import torch.nn as nn
import sys, os
from collections import OrderedDict
import math
# from sklearn.metrics.pairwise import rbf_kernel
# from sklearn.metrics import pairwise
# from gpytorch.kernels import RBFKernel

def _linear_kernel(X, Y=None):
    """线性核 (Linear Kernel): K(X, Y) = X @ Y.T"""
    if Y is None:
        Y = X
    return torch.matmul(X, Y.T)

def _poly_kernel(X, Y=None, degree=3, c=1.0):
    """多项式核 (Polynomial Kernel): K(X, Y) = (X @ Y.T + c)^degree"""
    if Y is None:
        Y = X
    return (torch.matmul(X, Y.T) + c) ** degree

def _matern_kernel(X, Y=None, nu=1.5, gamma=None):
    """
    Matérn 核 (推荐 nu=1.5 也就是 Matern 3/2)
    非常适合带噪声的物理系统，比 RBF 更能容忍数据的“粗糙度”。
    """
    if Y is None:
        Y = X
    if gamma is None:
        gamma = 1.0 / X.shape[-1]
        
    # 安全计算欧氏距离 (加入 1e-12 防止距离为 0 时梯度 NaN)
    X_norm = torch.sum(X ** 2, axis=-1, keepdim=True)
    Y_norm = torch.sum(Y ** 2, axis=-1)
    dist_sq = torch.clamp(X_norm + Y_norm - 2 * torch.matmul(X, Y.T), min=1e-12)
    dist = torch.sqrt(dist_sq)
    
    length_scale = 1.0 / math.sqrt(gamma) if gamma > 0 else 1.0

    if nu == 1.5:  # Matern 3/2
        sqrt3_d = math.sqrt(3.0) * dist / length_scale
        return (1.0 + sqrt3_d) * torch.exp(-sqrt3_d)
    elif nu == 2.5: # Matern 5/2
        sqrt5_d = math.sqrt(5.0) * dist / length_scale
        return (1.0 + sqrt5_d + (5.0 / 3.0) * (dist ** 2) / (length_scale ** 2)) * torch.exp(-sqrt5_d)
    else:
        raise ValueError("Only nu=1.5 and 2.5 are supported.")

def _laplacian_kernel(X, Y=None, gamma=None):
    """
    Laplacian 核 (拉普拉斯核)
    基于 L1 或 L2 距离的绝对指数核，对异常值/极端噪声非常鲁棒。
    """
    if Y is None:
        Y = X
    if gamma is None:
        gamma = 1.0 / X.shape[-1]
        
    # 这里使用 L2 距离的变体
    X_norm = torch.sum(X ** 2, axis=-1, keepdim=True)
    Y_norm = torch.sum(Y ** 2, axis=-1)
    dist_sq = torch.clamp(X_norm + Y_norm - 2 * torch.matmul(X, Y.T), min=1e-12)
    dist = torch.sqrt(dist_sq)
    
    return torch.exp(-gamma * dist)

def _rbf_kernel(X, Y=None, gamma=None):

    if gamma is None:
        gamma = 1.0 / X.shape[1]  # Default gamma value

    if Y is None:
        Y = X

    X = X.float()
    Y = Y.float()

    # Compute pairwise Euclidean distances
    norm_X = torch.sum(X ** 2, dim=1, keepdim=True)
    norm_Y = torch.sum(Y ** 2, dim=1, keepdim=True)
    dist_matrix = norm_X - 2.0 * torch.matmul(X, Y.t()) + norm_Y.t()

    # Compute RBF kernel
    K = torch.exp(-gamma * dist_matrix)

    return K

def elup1(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x < 0.0, torch.exp(x), x + 1.0)
    
class SplitDiagGaussianDecoder(nn.Module):
    def __init__(self, lod: int, out_dim: int):
        super(SplitDiagGaussianDecoder, self).__init__()

        # 在 SplitDiagGaussianDecoder 中定义构建方法
        self._hidden_layers_mean, num_last_hidden_mean = self._build_hidden_layers_mean()
        self._hidden_layers_var, num_last_hidden_var = self._build_hidden_layers_var()

        self._out_layer_mean = nn.Linear(in_features=num_last_hidden_mean, out_features=out_dim)
        self._out_layer_var = nn.Linear(in_features=num_last_hidden_var, out_features=out_dim)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def _build_hidden_layers_mean(self):
        """
        构建用于均值解码器的隐藏层
        """
        return nn.ModuleList([
            nn.Linear(in_features=200, out_features=64),
            nn.Tanh(),
            nn.Linear(in_features=64, out_features=64),
            nn.Tanh(),
            nn.Linear(in_features=64, out_features=8)
        ]), 8

    def _build_hidden_layers_var(self):
        """
        构建用于方差解码器的隐藏层
        """
        return nn.ModuleList([
            nn.Linear(in_features=300, out_features=64),
            nn.Tanh(),
            nn.Linear(in_features=64, out_features=64),
            nn.Tanh(),
            nn.Linear(in_features=64, out_features=8)
        ]), 8

    def clean_state_dict_keys(self, state_dict):
        # """去掉 state_dict 中的前缀，使其适应 ModuleList 的键格式"""
        cleaned_state_dict = OrderedDict()
        for key, value in state_dict.items():
            if 'hidden_layers_mean' in key:
                new_key = key.replace('_module._hidden_layers_mean.', '')
                index_offset = 0  # 均值部分在列表中的偏移索引
            elif 'hidden_layers_var' in key:
                new_key = key.replace('_module._hidden_layers_var.', '')
                index_offset = len(self._hidden_layers_mean)  # 方差部分在列表中的偏移索引
            elif 'out_layer_mean' in key:
                new_key = key.replace('_module._out_layer_mean.', str(index_offset + len(self._hidden_layers_mean)) + '.')
                index_offset = len(self._hidden_layers_mean) + len(self._hidden_layers_var)
            elif 'out_layer_var' in key:
                new_key = key.replace('_module._out_layer_var.', str(index_offset + len(self._hidden_layers_mean) + len(self._hidden_layers_var)) + '.')
            else:
                continue
            cleaned_state_dict[new_key] = value

        return cleaned_state_dict

    def forward(self, latent_mean, latent_cov):
        # 使用均值解码器
        h_mean = latent_mean
        for layer in self._hidden_layers_mean:
            h_mean = layer(h_mean)
        mean = self._out_layer_mean(h_mean)

        # 使用方差解码器
        h_var = latent_cov
        for layer in self._hidden_layers_var:
            h_var = layer(h_var)
        log_var = self._out_layer_var(h_var)
        var = elup1(log_var)

        return mean, var

class MyLayerNorm2d(nn.Module):

    def __init__(self, channels):
        super(MyLayerNorm2d, self).__init__()
        self._scale = torch.nn.Parameter(torch.ones(1, channels, 1, 1))
        self._offset = torch.nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        normalized = (x - x.mean(dim=[-3, -2, -1], keepdim=True)) / x.std(dim=[-3, -2, -1], keepdim=True)
        return self._scale * normalized + self._offset

class Deep_Kernel(nn.Module):
    def __init__(self):
        super(Deep_Kernel, self).__init__()

        self.encoder_output_dim = 200
        # idk
        self.observation_dim = 108

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # self.pre_train_enc = torch.load(os.path.abspath("nas/rkn_share/experiments/quad_link/enc_params.pth"))
        # self.pre_train_dec = torch.load(os.path.abspath("nas/rkn_share/experiments/quad_link/dec_params.pth"))
        self.pre_train_enc = torch.load(os.path.abspath("nas/rkn_share/experiments/quad_link/encoder02.pth"))
        self.pre_train_dec = torch.load(os.path.abspath("nas/rkn_share/experiments/quad_link/decoder02.pth"))

        self.encoder = self._build_encoder(self.observation_dim, layer_norm=True)
        # self.encoder = self._build_encoder(self.observation_dim, layer_norm=False)
        # self.decoder = nn.ModuleList([
        #     *self._build_dec_hidden_layers_mean()[0],
        #     *self._build_dec_hidden_layers_var()[0]
        # ])
        self.decoder = SplitDiagGaussianDecoder(self.observation_dim, out_dim=8)
        self.mlp = self._build_mlp().to(self.device)
        self.encoder_loaded = False

        # 自定义的深度核层
    def _build_mlp(self):
        return nn.Sequential(
            nn.Linear(self.encoder_output_dim, 256),  # 假设 encoder 的输出大小是 90
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.observation_dim) 
        )

    # 编码器部分
    def _build_encoder(self, input_dim, layer_norm):
        layers = [
            nn.Conv2d(in_channels=1, out_channels=12, kernel_size=5, padding=2, stride=2)
        ]
        if layer_norm:
            layers.append(MyLayerNorm2d(channels=12))

        layers.extend([
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding=1, stride=2)
        ])

        if layer_norm:
            layers.append(MyLayerNorm2d(channels=12))

        layers.extend([
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=input_dim, out_features=self.encoder_output_dim),
            nn.ReLU()
        ])

        return nn.Sequential(*layers)
    # 构建解码器的均值部分
    # def _build_dec_hidden_layers_mean(self):
    #     return nn.ModuleList([
    #         nn.Linear(in_features=300, out_features=64),
    #         nn.Tanh(),
    #         nn.Linear(in_features=64, out_features=64),
    #         nn.Tanh(),
    #         nn.Linear(in_features=64, out_features=8)  # 添加从 64 到 8 的线性层
    #     ]), 8

    # # 构建解码器的方差部分
    # def _build_dec_hidden_layers_var(self):
    #     return nn.ModuleList([
    #         # nn.Linear(in_features=3 * self.observation_dim, out_features=64),
    #         nn.Linear(in_features=300, out_features=64),
    #         nn.Tanh(),
    #         nn.Linear(in_features=64, out_features=64),
    #         nn.Tanh(),
    #         nn.Linear(in_features=64, out_features=8)
    #     ]), 8

    def _freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def _freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False
    
    # def clean_state_dict_keys(self, state_dict):
        # """去掉 state_dict 中的前缀，使其适应 ModuleList 的键格式"""
        cleaned_state_dict = OrderedDict()
        for key, value in state_dict.items():
            if 'hidden_layers_mean' in key:
                new_key = key.replace('_module._hidden_layers_mean.', '')
                index_offset = 0  # 均值部分在列表中的偏移索引
            elif 'hidden_layers_var' in key:
                new_key = key.replace('_module._hidden_layers_var.', '')
                index_offset = len(self._build_dec_hidden_layers_mean()[0])  # 方差部分在列表中的偏移索引
            elif 'out_layer_mean' in key:
                new_key = key.replace('_module._out_layer_mean.', str(index_offset + len(self._build_dec_hidden_layers_mean()[0])) + '.')
                index_offset = len(self._build_dec_hidden_layers_mean()[0]) + len(self._build_dec_hidden_layers_var()[0])
            elif 'out_layer_var' in key:
                new_key = key.replace('_module._out_layer_var.', str(index_offset + len(self._build_dec_hidden_layers_mean()[0]) + len(self._build_dec_hidden_layers_var()[0])) + '.')
            else:
                continue
            cleaned_state_dict[new_key] = value

        return cleaned_state_dict

    def _load_model(self):
        self.encoder.load_state_dict(self.pre_train_enc, strict=False)
        cleaned_state_dict = self.decoder.clean_state_dict_keys(self.pre_train_dec)
        self.decoder.load_state_dict(cleaned_state_dict, strict=False)
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        self.encoder_loaded = True

    def forward(self, obs_1, obs_2=None):
        # 启用异常检测
        torch.autograd.set_detect_anomaly(True)

        if self.pre_train_enc and not self.encoder_loaded:
            self._load_model()

        if obs_2 is None:
            encoded_obs = self.encoder(obs_1)
            encoded_obs = encoded_obs.clone()
            # encoded_obs = self.mlp(encoded_obs)
            rbf_result = _rbf_kernel(encoded_obs)
        else:
            encoded_obs = None
            encoded_obs_1 = self.encoder(obs_1)
            encoded_obs_2 = self.encoder(obs_2)
            encoded_obs_1 = encoded_obs_1.clone()
            encoded_obs_2 = encoded_obs_2.clone()
            rbf_result = _rbf_kernel(encoded_obs_1, encoded_obs_2)

        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.decoder.parameters():
            param.requires_grad = False
        for param in self.mlp.parameters():
            param.requires_grad = True
        
        return rbf_result, encoded_obs

    def decode(self, latent_mean, latent_cov):
        """
        调用解码器的解码函数
        :param latent_mean: 输入的潜在均值
        :param latent_cov: 输入的潜在协方差
        :return: 解码后的均值和方差
        """
        if not self.encoder_loaded:
            self._load_model()

        # 使用解码器进行解码
        return self.decoder(latent_mean, latent_cov)


# def deep_kernel(observation):
#     global model
#     if 'model' not in globals():
#         model = Deep_Kernel()

#     # model.eval()
#     encoded_obs = model(observation)
#     rbf_result = _rbf_kernel(encoded_obs)
#     return rbf_result