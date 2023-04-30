import torch
from torch import nn


#  class InverseDynamicsModel(nn.Module):
    #  '''
    #  This class operates on the output of AutoBot-Ego's decoder representation. It produces the parameters of a
    #  bivariate Gaussian distribution.
    #  '''
    #  def __init__(
            #  self,
            #  inp_dim,
            #  emb_dim=64,
            #  dist_dim=2,
            #  min_std=0.01,
            #  num_hidden=1,
            #  dropout=0.0,
            #  norm='layer',
            #  tanh=False,
            #  action_mean=None,
            #  action_std=None,
            #  wa_std=False,
    #  ):
        #  super().__init__()
        #  self.inp_dim = inp_dim
        #  self.emb_dim = emb_dim
        #  self.dist_dim = dist_dim
        #  self.min_std = min_std
        #  self.num_hidden = num_hidden
        #  self.norm = norm
        #  self.tanh = tanh
        #  if action_mean is None:
            #  self.action_mean = nn.Parameter(torch.zeros(self.dist_dim), requires_grad=False)
        #  else:
            #  self.action_mean = nn.Parameter(action_mean, requires_grad=False)
        #  if action_std is None:
            #  self.action_std = nn.Parameter(torch.ones(self.dist_dim), requires_grad=False)
        #  else:
            #  self.action_std = nn.Parameter(action_std, requires_grad=False)

        #  self.wa_std = wa_std
        #  if self.wa_std:
            #  self._std = nn.Parameter(torch.ones(self.dist_dim,), requires_grad=False)
        #  else:
            #  self.log_std = nn.Parameter(torch.zeros(self.dist_dim,), requires_grad=True)

        #  self.encoder_layer = nn.Sequential(nn.Linear(self.inp_dim, self.emb_dim))

        #  modules = []
        #  if norm == 'layer':
            #  modules.append(nn.LayerNorm(self.emb_dim))
        #  elif norm == 'batch':
            #  modules.append(nn.BatchNorm1d(self.emb_dim))
        #  else:
            #  modules.append(nn.ReLU())

        #  for _ in range(self.num_hidden):
            #  modules.append(nn.Linear(self.emb_dim, self.emb_dim))
            #  modules.append(nn.ReLU())
            #  if norm == 'layer':
                #  modules.append(nn.LayerNorm(self.emb_dim))
            #  elif norm == 'batch':
                #  modules.append(nn.BatchNorm1d(self.emb_dim))
            #  if dropout > 0.0:
                #  modules.append(nn.Dropout(dropout))

        #  modules.append(nn.Linear(self.emb_dim, self.dist_dim))

        #  if self.tanh:
            #  modules.append(nn.Tanh())
        #  self.output_model = nn.Sequential(*modules)

    #  @property
    #  def std(self):
        #  if self.wa_std:
            #  return self._std
        #  else:
            #  #  return self.min_std + torch.exp(100. * self.log_std)
            #  return self.min_std + torch.exp(self.log_std)

    #  def update_std(self, std, tau):
        #  assert(self.wa_std)
        #  self._std.data = torch.clamp((1 - tau) * self._std.data + tau * (std / self.action_std), min=self.min_std)

    #  def forward(self, inp):
        #  start_shape = inp.shape[:-1]
        #  x = self.encoder_layer(inp.reshape((-1, self.inp_dim)))
        #  mean = self.output_model(x).reshape((*start_shape, self.dist_dim))

        #  if not self.tanh:
            #  if self.action_std is not None:
                #  mean = mean * self.action_std
            #  if self.action_mean is not None:
                #  mean = mean + self.action_mean

        #  std = self.std.reshape((1, self.dist_dim)).repeat(x.shape[0], 1).reshape(*start_shape, self.dist_dim)
        #  if self.action_std is not None:
            #  std = std * self.action_std
        #  return torch.cat([mean, std], dim=-1)


class InverseDynamicsModel(nn.Module):
    '''
    This class operates on the output of AutoBot-Ego's decoder representation. It produces the parameters of a
    bivariate Gaussian distribution.
    '''
    def __init__(
            self,
            inp_dim,
            emb_dim=64,
            dist_dim=2,
            min_std=0.01,
            num_hidden=1,
            dropout=0.0,
            norm='layer',
            tanh=False,
            action_mean=None,
            action_std=None,
            wa_std=False,
    ):
        super().__init__()
        self.inp_dim = inp_dim
        self.emb_dim = emb_dim
        self.dist_dim = dist_dim
        self.min_std = min_std
        self.num_hidden = num_hidden
        self.norm = norm
        self.tanh = tanh
        if action_mean is None:
            self.action_mean = nn.Parameter(torch.zeros(self.dist_dim), requires_grad=False)
        else:
            self.action_mean = nn.Parameter(action_mean, requires_grad=False)
        if action_std is None:
            self.action_std = nn.Parameter(torch.ones(self.dist_dim), requires_grad=False)
        else:
            self.action_std = nn.Parameter(action_std, requires_grad=False)

        self.wa_std = wa_std
        if self.wa_std:
            self._std = nn.Parameter(torch.ones(self.dist_dim,), requires_grad=False)
        else:
            self.log_std = nn.Parameter(torch.zeros(self.dist_dim,), requires_grad=True)

        self.encoder_layer = nn.Sequential(nn.Linear(self.inp_dim, self.emb_dim))

        self.hidden_layers = []

        for _ in range(self.num_hidden):
            sub_modules = []
            if norm == 'layer':
                sub_modules.append(nn.LayerNorm(self.emb_dim))
            elif norm == 'batch':
                sub_modules.append(nn.BatchNorm1d(self.emb_dim))
            sub_modules.append(nn.Linear(self.emb_dim, self.emb_dim))
            if dropout > 0.0:
                sub_modules.append(nn.Dropout(dropout))
            sub_modules.append(nn.ReLU())
            sub_modules.append(nn.Linear(self.emb_dim, self.emb_dim))
            if dropout > 0.0:
                sub_modules.append(nn.Dropout(dropout))
            self.hidden_layers.append(nn.Sequential(*sub_modules))
        self.hidden_layers = nn.ModuleList(self.hidden_layers)

        # last layer
        sub_modules = []
        if norm == 'layer':
            sub_modules.append(nn.LayerNorm(self.emb_dim))
        elif norm == 'batch':
            sub_modules.append(nn.BatchNorm1d(self.emb_dim))

        sub_modules.append(nn.Linear(self.emb_dim, self.dist_dim))
        if self.tanh:
            sub_modules.append(nn.Tanh())
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
        self._std.data = torch.clamp((1 - tau) * self._std.data + tau * (std / self.action_std), min=self.min_std)

    def forward(self, inp):
        start_shape = inp.shape[:-1]
        x = self.encoder_layer(inp.reshape((-1, self.inp_dim)))
        for i in range(self.num_hidden):
            x = x + self.hidden_layers[i](x)

        mean = self.output_model(x).reshape((*start_shape, self.dist_dim))

        if not self.tanh:
            if self.action_std is not None:
                mean = mean * self.action_std
            if self.action_mean is not None:
                mean = mean + self.action_mean

        std = self.std.reshape((1, self.dist_dim)).repeat(x.shape[0], 1).reshape(*start_shape, self.dist_dim)
        if self.action_std is not None:
            std = std * self.action_std
        return torch.cat([mean, std], dim=-1)
