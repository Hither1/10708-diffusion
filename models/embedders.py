import torch
from torch import nn


class Embedder(nn.Module):
    def __init__(self, inp_dim, emb_dim, expand_theta=False, layer_norm=False,):
        super().__init__()
        self.inp_dim = inp_dim
        self.emb_dim = emb_dim
        self.expand_theta = expand_theta
        self.layer_norm = layer_norm
        if self.expand_theta:
            embedder = [nn.Linear(self.inp_dim + 1, self.emb_dim)]
        else:
            embedder = [nn.Linear(self.inp_dim, self.emb_dim)]

        if self.layer_norm:
            embedder.append(nn.LayerNorm(self.emb_dim))
        else:
            embedder.append(nn.ReLU())

        embedder.append(nn.Linear(self.emb_dim, self.emb_dim))

        if self.layer_norm:
            embedder.append(nn.LayerNorm(self.emb_dim))
        else:
            embedder.append(nn.ReLU())
        self.embedder = nn.Sequential(*embedder)

    def forward(self, x):
        if self.expand_theta:
            pos = x[..., :2]
            theta = x[..., 2:3]
            features = x[..., 3:]
            inp = torch.cat([pos, features, torch.cos(theta), torch.sin(theta)], dim=-1)

        y = self.embedder(inp)
        return y

class Embedder_VAE(nn.Module):
    def __init__(self, inp_dim, emb_dim, expand_theta=False, layer_norm=False,):
        super().__init__()
        self.inp_dim = inp_dim
        self.emb_dim = emb_dim
        self.expand_theta = expand_theta
        self.layer_norm = layer_norm
        if self.expand_theta:
            embedder_mean = [nn.Linear(self.inp_dim + 1, self.emb_dim)]
            embedder_log_std = [nn.Linear(self.inp_dim + 1, self.emb_dim)]
        else:
            embedder_mean = [nn.Linear(self.inp_dim, self.emb_dim)]
            embedder_log_std = [nn.Linear(self.inp_dim, self.emb_dim)]
        if self.layer_norm:
            embedder_mean.append(nn.LayerNorm(self.emb_dim))
            embedder_log_std.append(nn.LayerNorm(self.emb_dim))
        else:
            embedder_mean.append(nn.ReLU())
            embedder_log_std.append(nn.ReLU())
        self.embedder_mean = nn.Sequential(*embedder_mean)
        self.embedder_log_std = nn.Sequential(*embedder_log_std)

    def forward(self, x):
        if self.expand_theta:
            pos = x[..., :2]
            theta = x[..., 2:3]
            features = x[..., 3:]
            inp = torch.cat([pos, features, torch.cos(theta), torch.sin(theta)], dim=-1)

        mean = self.embedder_mean(inp)
        std = self.embedder_log_std(inp)
        return mean, std