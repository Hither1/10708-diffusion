import torch
from torch import nn


class OutputModel(nn.Module):
    '''
    This class operates on the output of AutoBot-Ego's decoder representation. It produces the parameters of a
    bivariate Gaussian distribution.
    '''
    def __init__(
            self,
            emb_dim=64,
            dist_dim=2,
            min_std=0.01,
            num_hidden=2,
            layer_norm=True,
            dropout=0.0,
            out_mean=None,
            out_std=None,
            wa_std=False,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.dist_dim = dist_dim
        self.min_std = min_std
        self.num_hidden = num_hidden
        self.layer_norm = layer_norm
        if out_mean is None:
            self.out_mean = nn.Parameter(torch.zeros(self.dist_dim), requires_grad=False)
        else:
            self.out_mean = nn.Parameter(out_mean, requires_grad=False)

        if out_std is None:
            self.out_std = nn.Parameter(torch.ones(self.dist_dim), requires_grad=False)
        else:
            self.out_std = nn.Parameter(out_std, requires_grad=False)

        self.wa_std = wa_std

        if self.wa_std:
            self._std = nn.Parameter(torch.ones(self.out_std.shape), requires_grad=False)
        else:
            self.log_std = nn.Parameter(torch.zeros(self.out_std.shape), requires_grad=True)

        modules = []
        if layer_norm:
            modules.append(nn.LayerNorm(self.emb_dim))

        for _ in range(self.num_hidden):
            modules.append(nn.Linear(self.emb_dim, self.emb_dim))
            modules.append(nn.ReLU())
            if dropout > 0.0:
                modules.append(nn.Dropout(dropout))

        modules.append(nn.Linear(self.emb_dim, self.dist_dim))

        self.output_model = nn.Sequential(*modules)

    @property
    def std(self):
        if self.wa_std:
            return self._std
        else:
            #  return self.min_std + torch.exp(100. * self.log_std)
            return self.min_std + torch.exp(self.log_std)

    def update_std(self, std, tau):
        assert(self.wa_std)
        self._std.data = torch.clamp((1 - tau) * self._std.data + tau * (std / self.out_std), min=self.min_std)

    def forward(self, agent_decoder_state):
        start_shape = agent_decoder_state.shape[:-1]
        x = agent_decoder_state.reshape((-1, self.emb_dim))

        out = self.output_model(x).reshape((*start_shape, self.dist_dim))
        mean = out[..., :self.dist_dim]

        if self.out_std is not None:
            mean = mean * self.out_std
        if self.out_mean is not None:
            mean = mean + self.out_mean

        #  std = self.std.reshape((1, self.dist_dim)).repeat(x.shape[0], 1).reshape(*start_shape, self.dist_dim)
        std = torch.zeros_like(mean) + self.std
        if self.out_std is not None:
            std = std * self.out_std
        return torch.cat([mean, std], dim=-1)


class ResOutputModel(nn.Module):
    '''
    This class operates on the output of AutoBot-Ego's decoder representation. It produces the parameters of a
    bivariate Gaussian distribution.
    '''
    def __init__(
            self,
            emb_dim=64,
            dist_dim=2,
            min_std=0.01,
            num_hidden=2,
            layer_norm=True,
            dropout=0.0,
            out_mean=None,
            out_std=None,
            wa_std=False,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.dist_dim = dist_dim
        self.min_std = min_std
        self.num_hidden = num_hidden
        self.layer_norm = layer_norm
        if out_mean is None:
            self.out_mean = nn.Parameter(torch.zeros(self.dist_dim), requires_grad=False)
        else:
            self.out_mean = nn.Parameter(out_mean, requires_grad=False)

        if out_std is None:
            self.out_std = nn.Parameter(torch.ones(self.dist_dim), requires_grad=False)
        else:
            self.out_std = nn.Parameter(out_std, requires_grad=False)

        self.wa_std = wa_std

        if self.wa_std:
            self._std = nn.Parameter(torch.ones(self.out_std.shape), requires_grad=False)
        else:
            self.log_std = nn.Parameter(torch.zeros(self.out_std.shape), requires_grad=True)

        self.hidden_layers = []

        for _ in range(self.num_hidden):
            sub_modules = []
            if layer_norm:
                sub_modules.append(nn.LayerNorm(self.emb_dim))
            sub_modules.append(nn.Linear(self.emb_dim, self.emb_dim))
            sub_modules.append(nn.ReLU())
            if dropout > 0.0:
                sub_modules.append(nn.Dropout(dropout))
            sub_modules.append(nn.Linear(self.emb_dim, self.emb_dim))
            if dropout > 0.0:
                sub_modules.append(nn.Dropout(dropout))
            self.hidden_layers.append(nn.Sequential(*sub_modules))
        self.hidden_layers = nn.ModuleList(self.hidden_layers)

        # last layer
        sub_modules = []
        if layer_norm:
            sub_modules.append(nn.LayerNorm(self.emb_dim))
        sub_modules.append(nn.Linear(self.emb_dim, self.dist_dim))
        self.output_model = nn.Sequential(*sub_modules)

    @property
    def std(self):
        if self.wa_std:
            return self._std
        else:
            #  return self.min_std + torch.exp(100. * self.log_std)
            return self.min_std + torch.exp(self.log_std)

    def update_std(self, std, tau):
        assert(self.wa_std)
        self._std.data = torch.clamp((1 - tau) * self._std.data + tau * (std / self.out_std), min=self.min_std)

    def forward(self, agent_decoder_state): # torch.Size([1280, 1, 8, 512])
        start_shape = agent_decoder_state.shape[:-1]
        x = agent_decoder_state.reshape((-1, self.emb_dim))
        for i in range(self.num_hidden):
            x = x + self.hidden_layers[i](x)
        out = self.output_model(x).reshape((*start_shape, self.dist_dim))
        mean = out[..., :self.dist_dim]

        if self.out_std is not None:
            mean = mean * self.out_std
        if self.out_mean is not None:
            mean = mean + self.out_mean

        #  std = self.std.reshape((1, self.dist_dim)).repeat(x.shape[0], 1).reshape(*start_shape, self.dist_dim)
        std = torch.zeros_like(mean) + self.std
        if self.out_std is not None:
            std = std * self.out_std

        return torch.cat([mean, std], dim=-1)

    def get_std(self, mean): # torch.Size([1280, 1, 8, 512])
        #  std = self.std.reshape((1, self.dist_dim)).repeat(x.shape[0], 1).reshape(*start_shape, self.dist_dim)
        std = torch.zeros_like(mean) + self.std
        if self.out_std is not None:
            std = std * self.out_std
        return torch.cat([mean, std], dim=-1)
