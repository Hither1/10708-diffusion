import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt

from src.models.embedders import Embedder
from src.models.transformer_layers import TransformerEncoderLayer, TransformerDecoderLayer
from src.models.output_models import ResOutputModel, OutputModel
from src.models.diffusion.diffusion_utils import (
    AdaLayerNorm,
    vp_beta_schedule, 
    linear_beta_schedule, 
    SinusoidalPosEmb, 
    Whitener
)

class TimeSiren(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(TimeSiren, self).__init__() # just a fully connected NN with sin activations
        self.lin1 = nn.Linear(input_dim, emb_dim, bias=False)
        self.lin2 = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        x = torch.sin(self.lin1(x))
        x = self.lin2(x)
        return x


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(-1, num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def ddpm_schedules(beta1, beta2, T, schedule='vp'):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    # beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    # beta_t = (beta2 - beta1) * torch.arange(-1, T + 1, dtype=torch.float32) / T + beta1
    if schedule == 'linear':
        beta_t = (beta2 - beta1) * torch.arange(-1, T, dtype=torch.float32) / (T - 1) + beta1
    elif schedule == 'quadratic':
        beta_t = (beta2 - beta1) * torch.square(torch.arange(-1, T, dtype=torch.float32)) / torch.max(torch.square(torch.arange(-1, T, dtype=torch.float32))) + beta1
    elif schedule == 'cosine':
        beta_t = betas_for_alpha_bar(T,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
        beta_t = torch.tensor(beta_t, dtype=torch.float32)
    elif schedule == 'vp':
        beta_t = vp_beta_schedule(T)
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

    beta_t[0] = beta1  # modifying this so that beta_t[1] = beta1, and beta_t[n_T]=beta2, while beta[0] is never used
    # this is as described in Denoising Diffusion Probabilistic Models paper, section 4
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab
    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class Model_Cond_Diffusion(nn.Module):
    def __init__(self, nn_model, betas, n_T, y_dim, dropout, 
                emb_dim, 
                num_agents, 
                T=8,
                wp_dim=4,
                pca_params_path: str = 'pca_params.th',
                k: int = 10,

                guide_w=0.0):
        super(Model_Cond_Diffusion, self).__init__()
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.nn_model = nn_model
        self.n_T = n_T
        self.dropout = dropout
        self.y_dim = y_dim
        self.num_agents = num_agents
        self.T = T
        self.wp_dim = wp_dim
        self.guide_w = guide_w
        self.loss_mse = nn.MSELoss()
        
        self.pca_whitener = Whitener(pca_params_path)
        # rand = torch.tensor(np.random.randn(100, 32))
        # rand = self.pca_whitener.untransform_features(rand)
        # rand = rand.reshape(-1, 8, 4)
        # for i in range(len(rand)):
        #     plt.plot(rand[i, :, 0], rand[i, :, 1])
        # plt.savefig('./random.png')

    
    def _compute_loss(self, pred_wps, gt_wps, wps_mask):
        B = pred_wps.shape[0]
        num_agents = wps_mask.shape[-1]
        self.num_modes = 1

        shaped_pred_wps = pred_wps.reshape((B * self.num_modes, num_agents, self.T, self.wp_dim)).permute(0, 2, 1, 3)
        # shaped_pred_wps = pred_wps.reshape((B * self.num_modes, self.T, num_agents, self.wp_dim))
        shaped_gt_wps = gt_wps.unsqueeze(1).repeat_interleave(self.num_modes, 1).reshape((B * self.num_modes, self.T, num_agents, self.wp_dim))
        shaped_wps_mse_errors = F.mse_loss(shaped_pred_wps, shaped_gt_wps, reduction='none')

        if wps_mask is None:
            wps_mask = torch.ones(gt_wps.shape[:-1], dtype=bool, device=gt_wps.device)

        time_wps_mse_errors = shaped_wps_mse_errors.permute(0, 2, 1, 3).reshape((B * num_agents, self.T, self.wp_dim))
        w_mask = wps_mask.permute(0, 2, 1).reshape((B * num_agents, self.T))

        wps_mse_errors = (time_wps_mse_errors * w_mask.unsqueeze(-1)).sum(dim=1) / torch.clamp(w_mask.unsqueeze(-1).sum(dim=1), min=1.)

        wps_m = w_mask.any(dim=1)
        wps_mse_errors = wps_mse_errors[wps_m].mean(dim=0)

        time_wps_mse = (time_wps_mse_errors * w_mask.unsqueeze(-1)).sum(dim=0) / torch.clamp(w_mask.unsqueeze(-1).sum(dim=0), min=1.)

        wps_errors = torch.sqrt(wps_mse_errors)

        return wps_errors
        

    def loss_on_batch(self, gt_wps, traj_mask, agents_emb, agents_mask, map_emb, map_mask, route_emb, route_masks):
        _ts = torch.randint(1, self.n_T, (gt_wps.shape[0], 1)).to(agents_emb.device)

        # context_mask = torch.bernoulli(torch.zeros(agents_emb.shape[0]) + self.drop_prob).to(agents_emb.device)
        # randomly sample some noise, noise ~ N(0, 1)
        noise = torch.randn_like(gt_wps).to(agents_emb.device)
        
        y_t = self.sqrtab[_ts][..., None, None] * gt_wps + self.sqrtmab[_ts][..., None, None] * noise
        
        # use nn model to predict noise
        noise_pred_batch = self.nn_model(y_t, agents_mask[:, :, :self.num_agents], agents_emb, agents_mask, map_emb, map_mask, route_emb, route_masks, _ts) # / self.n_T

        reconstruction = (y_t - self.sqrtmab[_ts][..., None, None] * noise_pred_batch) / self.sqrtab[_ts][..., None, None]
        loss = self._compute_loss(noise, noise_pred_batch, traj_mask)

        return loss, y_t, reconstruction


    def sample(self, agents_emb, agents_masks, map_emb, map_masks, route_emb, route_masks, return_y_trace=True, extract_embedding=False):
        is_zero = False
        if self.guide_w > -1e-3 and self.guide_w < 1e-3:
            is_zero = True

        # how many noisy actions to begin with
        n_sample = agents_emb.shape[0]
        num_agents =  self.num_agents
        y_shape = (n_sample, 1, num_agents, self.y_dim)

        y_i = torch.randn(y_shape).to(agents_emb.device)
        y_i_store = []  # if want to trace how y_i evolved
        for i in range(self.n_T, 0, -1):
            t_is = torch.tensor([i ]).to(agents_emb.device) #/ self.n_T
            t_is = t_is.repeat(n_sample, 1)

            if not is_zero: # double batch
                y_i = y_i.repeat(2, 1)
                t_is = t_is.repeat(2, 1)

            z = torch.randn(y_shape).to(agents_emb.device) if i > 1 else 0
            traj_mask = torch.ones(agents_emb.shape[0], num_agents).to(agents_emb.device)
            eps = self.nn_model(y_i, traj_mask, agents_emb, agents_masks, map_emb, map_masks, route_emb, route_masks, t_is)
    
            y_i = self.oneover_sqrta[i] * (y_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z

            if return_y_trace and (i % (self.n_T // 5) == 0 or i == self.n_T or i < 4):
                y_i_store.append(y_i.detach().cpu())

        if return_y_trace:
            return y_i, y_i_store
        else:
            return y_i

    def sample_update(self, x_batch, betas, n_T, return_y_trace=False):
        original_nT = self.n_T

        # set new schedule
        self.n_T = n_T
        for k, v in ddpm_schedules(betas[0], betas[1], self.n_T).items():
            self.register_buffer(k, v.to(self.device))

        # also use this as a shortcut to avoid doubling batch when guide_w is zero
        is_zero = False
        if self.guide_w > -1e-3 and self.guide_w < 1e-3:
            is_zero = True

        # how many noisy actions to begin with
        n_sample = x_batch.shape[0]

        y_shape = (n_sample, self.y_dim)

        # sample initial noise, y_0 ~ N(0, 1),
        y_i = torch.randn(y_shape).to(self.device)

        if not is_zero:
            if len(x_batch.shape) > 2:
                # repeat x_batch twice, so can use guided diffusion
                x_batch = x_batch.repeat(2, 1, 1, 1)
            else:
                # repeat x_batch twice, so can use guided diffusion
                x_batch = x_batch.repeat(2, 1)
            # half of context will be zero
            context_mask = torch.zeros(x_batch.shape[0]).to(self.device)
            context_mask[n_sample:] = 1.0  # makes second half of batch context free
        else:
            context_mask = torch.zeros(x_batch.shape[0]).to(self.device)

        # run denoising chain
        y_i_store = []  # if want to trace how y_i evolved
        for i in range(self.n_T, 0, -1):
            t_is = torch.tensor([i / self.n_T]).to(self.device)
            t_is = t_is.repeat(n_sample, 1)

            if not is_zero:
                # double batch
                y_i = y_i.repeat(2, 1)
                t_is = t_is.repeat(2, 1)

            # I'm a bit confused why we are adding noise during denoising?
            z = torch.randn(y_shape).to(self.device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.nn_model(y_i, agents_emb, agents_mask, map_emb, map_mask, t_is)
            if not is_zero:
                eps1 = eps[:n_sample]
                eps2 = eps[n_sample:]
                eps = (1 + self.guide_w) * eps1 - self.guide_w * eps2
                y_i = y_i[:n_sample]
            y_i = self.oneover_sqrta[i] * (y_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z
            if return_y_trace and (i % 20 == 0 or i == self.n_T or i < 8):
                y_i_store.append(y_i.detach().cpu().numpy())

        # reset original schedule
        self.n_T = original_nT
        for k, v in ddpm_schedules(betas[0], betas[1], self.n_T).items():
            self.register_buffer(k, v.to(self.device))

        if return_y_trace:
            return y_i, y_i_store
        else:
            return y_i

    def sample_extra(self, x_batch, extra_steps=4, return_y_trace=False):
        # also use this as a shortcut to avoid doubling batch when guide_w is zero
        is_zero = False
        if self.guide_w > -1e-3 and self.guide_w < 1e-3:
            is_zero = True

        # how many noisy actions to begin with
        n_sample = x_batch.shape[0]

        y_shape = (n_sample, self.y_dim)

        # sample initial noise, y_0 ~ N(0, 1),
        y_i = torch.randn(y_shape).to(self.device)

        if not is_zero:
            if len(x_batch.shape) > 2:
                # repeat x_batch twice, so can use guided diffusion
                x_batch = x_batch.repeat(2, 1, 1, 1)
            else:
                # repeat x_batch twice, so can use guided diffusion
                x_batch = x_batch.repeat(2, 1)
            # half of context will be zero
            context_mask = torch.zeros(x_batch.shape[0]).to(self.device)
            context_mask[n_sample:] = 1.0  # makes second half of batch context free
        else:
            # context_mask = torch.zeros_like(x_batch[:,0]).to(self.device)
            context_mask = torch.zeros(x_batch.shape[0]).to(self.device)

        # run denoising chain
        y_i_store = []  # if want to trace how y_i evolved
        # for i_dummy in range(self.n_T, 0, -1):
        for i_dummy in range(self.n_T, -extra_steps, -1):
            i = max(i_dummy, 1)
            t_is = torch.tensor([i / self.n_T]).to(self.device)
            t_is = t_is.repeat(n_sample, 1)

            if not is_zero:
                # double batch
                y_i = y_i.repeat(2, 1)
                t_is = t_is.repeat(2, 1)

            z = torch.randn(y_shape).to(self.device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.nn_model(y_i, x_batch, t_is, context_mask)
            if not is_zero:
                eps1 = eps[:n_sample]
                eps2 = eps[n_sample:]
                eps = (1 + self.guide_w) * eps1 - self.guide_w * eps2
                y_i = y_i[:n_sample]
            y_i = self.oneover_sqrta[i] * (y_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z
            if return_y_trace and (i % 20 == 0 or i == self.n_T or i < 8):
                y_i_store.append(y_i.detach().cpu().numpy())

        if return_y_trace:
            return y_i, y_i_store
        else:
            return y_i


class Model_mlp(nn.Module):
    def __init__(self, n_T, y_dim, emb_dim, num_heads, 
                T=8,
                wp_dim=4,
                min_std=0.01,
                out_init_std=0.1,
                dt=0.5,
                f=1,
                activate='relu',
                num_enc_layers=4, 
                num_dec_layers=4, 
                dropout=0., 
                tx_hidden_factor=2, 
                n_layer=2,
                norm_first=True, 
                output_dim=None):
        super(Model_mlp, self).__init__()

        self.T = T
        self.wp_dim = wp_dim
        self.min_std = min_std
        self.out_init_std = out_init_std
        self.dt = dt
        self.f = f
        self.y_dim = y_dim
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        self.dropout = dropout
        self.tx_hidden_size = tx_hidden_factor * self.emb_dim
        self.norm_first = norm_first
        self.skip_temporal_attn_fn = True
        self.output_dim = y_dim  # sometimes overwrite, eg for discretised, mean/variance, mixture density models
        self.q_dec = None
        self.p_dec = None

        # self.t_embed_nn = TimeSiren(1, self.emb_dim)
        #TODO: does different way of encoding timestep affect?
        self.traj_embedder = Embedder(T*wp_dim, self.emb_dim, expand_theta=True, layer_norm=False)

        self.blocks = nn.Sequential(*[Block(
                emb_dim=emb_dim,
                n_head=num_heads,
                attn_pdrop=dropout,
                activate=activate,
                n_T=n_T,
        ) for n in range(n_layer)])

        self.output_model = OutputModel(
            emb_dim=self.emb_dim,
            dist_dim=self.wp_dim,
            min_std=self.min_std,
            layer_norm=self.norm_first,
            dropout=self.dropout,
            out_std=self.out_init_std * self.dt * self.f * torch.ones((self.T, self.wp_dim)),
        )
        self.create_decoder()

        self.final = nn.Linear(self.emb_dim, self.output_dim)

    def map_dec_fn(self, out_emb, out_masks, map_emb, map_masks, layer, route_emb=None, route_masks=None):
        '''
        :param out_emb: (B, T, M, d)
        :param out_masks: (B, T, M)
        :param map_emb: (B, P, d)
        :param map_masks: (B, P)
        :param route_emb: (B, R, d)
        :param route_masks: (B, R)
        :return: (B, T, M, d)
        '''
        # TODO is this the right way to do the time stuff
        B, T, M, d = out_emb.shape
        out_emb = out_emb.transpose(1, 2).reshape((B * M, T, d))
        out_masks = out_masks.transpose(1, 2).reshape((B * M, T))
        # TODO make sure this is right
        out_masks = torch.where(out_masks.all(dim=-1, keepdims=True), torch.zeros_like(out_masks), out_masks)
        map_emb = map_emb.repeat_interleave(M, 0)
        map_masks = map_masks.repeat_interleave(M, 0)
        map_masks = torch.where(map_masks.all(dim=-1, keepdims=True), torch.zeros_like(map_masks), map_masks)
        if route_emb is not None and route_masks is not None:
            R = route_emb.shape[1]
            route_ego_masks = torch.ones((B * M, R), device=map_masks.device, dtype=bool)
            route_ego_masks[::M] = route_masks

            route_ego_emb = torch.zeros((B * M, R, d), device=map_emb.device, dtype=map_emb.dtype)
            route_ego_emb[::M] = route_emb

            map_emb = torch.cat([map_emb, route_ego_emb], dim=1)
            map_masks = torch.cat([map_masks, route_ego_masks], dim=1)
        
        map_cross_atten_emb = layer(
            out_emb,
            map_emb,
            tgt_key_padding_mask=out_masks,
            memory_key_padding_mask=map_masks).reshape((B, M, T, d)).transpose(1, 2)
        return map_cross_atten_emb

    def dec_fn(self, out_emb, out_masks, agents_emb, agents_masks, layer):
        '''
        :param out_emb: (B, T, M, d)
        :param out_masks: (B, T, M)
        :param agents_emb: (B, H, N, d)
        :param agents_masks: (B, H, N)
        :return: (B, T, d)
        '''
        B, T, M, d = out_emb.shape
        _, H, N, _ = agents_emb.shape

        out_emb = out_emb.transpose(1, 2).reshape((B * M, T, d))
        out_masks = out_masks.transpose(1, 2).reshape((B * M, T))
        # TODO make sure this is right
        out_masks = torch.where(out_masks.all(dim=-1, keepdims=True), torch.zeros_like(out_masks), out_masks)
        agents_emb = agents_emb.reshape((B, H * N, d)).repeat_interleave(M, 0)
        agents_masks = agents_masks.reshape((B, H * N)).repeat_interleave(M, 0)
        agents_masks = torch.where(agents_masks.all(dim=-1, keepdims=True), torch.zeros_like(agents_masks), agents_masks)
        out_cross_atten_emb = layer(
            out_emb,
            agents_emb,
            tgt_key_padding_mask=out_masks,
            memory_key_padding_mask=agents_masks).reshape((B, M, T, d)).transpose(1, 2)
        return out_cross_atten_emb

    def create_decoder(self):
        if self.q_dec:
            self.Q = nn.Parameter(torch.normal(
                torch.zeros((1, self.num_modes, 1, 1, self.emb_dim)),
                torch.ones((1, self.num_modes, 1, 1, self.emb_dim))),
                requires_grad=True)

            self.dec_time_pe = TimeEncoding(self.emb_dim, dropout=self.dropout, max_len=self.T)

        # TODO norm memory
        self.tx_decoder = []
        for _ in range(self.num_dec_layers):
            self.tx_decoder.append(
                TransformerDecoderLayer(
                    d_model=self.emb_dim,
                    nhead=self.num_heads,
                    dropout=self.dropout,
                    dim_feedforward=self.tx_hidden_size,
                    norm_first=self.norm_first,
                    batch_first=True,
                ))
        self.tx_decoder = nn.ModuleList(self.tx_decoder)

        self.map_dec_layers = []
        for _ in range(self.num_dec_layers):
            map_decoder_layer = TransformerDecoderLayer(
                d_model=self.emb_dim,
                nhead=self.num_heads,
                dropout=self.dropout,
                dim_feedforward=self.tx_hidden_size,
                norm_first=self.norm_first,
                batch_first=True,
            )
            self.map_dec_layers.append(map_decoder_layer)
        self.map_dec_layers= nn.ModuleList(self.map_dec_layers)

        if self.p_dec:
            self.P = nn.Parameter(torch.normal(
                torch.zeros((1, self.num_modes, 1, self.emb_dim)),
                torch.ones((1, self.num_modes, 1, self.emb_dim))),
                requires_grad=True)

            self.prob_tx_decoder = []
            for _ in range(self.num_dec_layers):
                self.prob_tx_decoder.append(
                    TransformerDecoderLayer(
                        d_model=self.emb_dim,
                        nhead=self.num_heads,
                        dropout=self.dropout,
                        dim_feedforward=self.tx_hidden_size,
                        norm_first=self.norm_first,
                        batch_first=True,
                    ))
            self.prob_tx_decoder = nn.ModuleList(self.prob_tx_decoder)

            self.prob_map_dec_layers = []
            for _ in range(self.num_dec_layers):
                map_decoder_layer = TransformerDecoderLayer(
                    d_model=self.emb_dim,
                    nhead=self.num_heads,
                    dropout=self.dropout,
                    dim_feedforward=self.tx_hidden_size,
                    norm_first=self.norm_first,
                    batch_first=True,
                )
                self.prob_map_dec_layers.append(map_decoder_layer)
            self.prob_map_dec_layers= nn.ModuleList(self.prob_map_dec_layers)

    def get_decoding(self, traj_emb, traj_mask, agents_emb, agents_masks, map_emb, map_masks, route_emb, route_masks, modes=None):
        num_modes = 1
        B = traj_emb.shape[0]
        num_agents = traj_mask.shape[-1]

        out_masks = traj_mask.reshape((B, 1, -1, num_agents))
        out_masks = out_masks[:, :, -1:, :].repeat(1, num_modes, self.T, 1)

        if self.q_dec:
            out_seq = agents_emb.reshape((B, 1, -1, num_agents, self.emb_dim))

            out_seq = out_seq[:, :, -1:, :, :].repeat(1, num_modes, self.T, 1, 1)
            out_seq = out_seq + self.Q.repeat(B, 1, 1, num_agents, 1)
            out_seq = self.dec_time_pe(out_seq.transpose(2, 3).reshape((-1, self.T, self.emb_dim))).reshape((B, self.num_modes, num_agents, self.T, self.emb_dim)).transpose(2, 3)
        else:
            # TODO other way to encode each individual agent
            out_seq = traj_emb.reshape((B, 1, -1, num_agents, self.emb_dim))
            out_seq = out_seq[:, :, -1:, :, :].repeat(1, num_modes, self.T, 1, 1)

        # TODO should we fold modes into batch or time
        out_seq = out_seq.reshape((B, num_modes * self.T, num_agents, -1))
        out_masks = out_masks.reshape((B, num_modes * self.T, num_agents))

        # TODO handle time
        #  time_masks = self.generate_decoder_mask(seq_len=self.T, device=ego_in.device)
        # TODO fix these functions
        for d in range(self.num_dec_layers):
            out_seq = self.map_dec_fn(
                out_seq,
                out_masks.clone(),
                map_emb,
                map_masks.clone(),
                route_emb=route_emb,
                route_masks=route_masks.clone(),
                layer=self.map_dec_layers[d],
            )
            out_seq = self.dec_fn(
                out_seq,
                out_masks.clone(),
                agents_emb,
                agents_masks.clone(),
                layer=self.tx_decoder[d])

        out_seq = out_seq.view((B, num_modes, self.T, num_agents, -1))
        return out_seq

    def get_output(self, out_seq):
        B = out_seq.shape[0]
        num_modes = out_seq.shape[1]
        num_agents = out_seq.shape[-2]
        #  outputs = self.output_model(out_seq).reshape((B, num_modes, self.T, self.num_agents, -1))
        outputs = self.output_model(out_seq.permute(0, 3, 1, 2, 4).reshape((B * num_agents, num_modes, self.T, self.emb_dim)))
        outputs = outputs.reshape((B, num_agents, num_modes, self.T, -1)).permute(0, 2, 3, 1, 4)
        return outputs

    def forward(self, y, traj_mask, agents_emb, agents_masks, map_emb, map_masks, route_emb, route_masks, t):
        # but we feed in batch_size, height, width, channels
        # mask out context embedding, x_e, if context_mask == 1
        # x_e = x_e * (-1 * (1 - context_mask))
        # roughly following this: https://jalammar.github.io/illustrated-transformer/
        traj_emb = self.traj_embedder(y)
        for block_idx in range(len(self.blocks)):   
            traj_emb = self.blocks[block_idx](traj_emb, traj_mask, agents_emb, agents_masks, map_emb, map_masks, route_emb, route_masks, t)
        
        # out = self.get_decoding(traj_emb, traj_mask, agents_emb, agents_masks, map_emb, map_masks, route_emb, route_masks)
        # out = self.get_output(out)
        # out = out[..., :4].permute(0, 1, 3, 2, 4).flatten(start_dim=3) 
        out = self.final(traj_emb)
 
        return out

class Block(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self,
                 class_type='adalayernorm',
                 H=1,
                 emb_dim=256,
                 n_head=16,
                 seq_len=256,
                 attn_pdrop=0.1,
                 num_enc_layers=2,
                 tx_hidden_factor=2,
                 activate='relu',
                 norm_first=True,
                 if_upsample=False, # only need for dalle_conv attention
                 n_T=100,
                 ):
        super().__init__()
        self.if_upsample = if_upsample
        self.num_enc_layers = num_enc_layers
        self.tx_hidden_size = tx_hidden_factor * emb_dim
        self.emb_dim = emb_dim
        self.norm_first = norm_first
        self.num_heads = n_head
        self.dropout = attn_pdrop
        self.H = H
        self.skip_temporal_attn_fn = self.H <= 1
        
        self.ln1 = AdaLayerNorm(emb_dim, diffusion_step=n_T, emb_type='adalayernorm')
        self.create_agent_encoder()
        self.ln2 = nn.LayerNorm(emb_dim)
        self.ln1_1 = AdaLayerNorm(emb_dim, diffusion_step=n_T, emb_type='adalayernorm')

    def create_agent_encoder(self):
        self.social_attn_layers = []
        for _ in range(self.num_enc_layers):
            tx_encoder_layer = TransformerEncoderLayer(
                d_model=self.emb_dim,
                nhead=self.num_heads,
                dropout=self.dropout,
                dim_feedforward=self.tx_hidden_size,
                norm_first=self.norm_first,
                batch_first=True,
            )
            self.social_attn_layers.append(tx_encoder_layer)
        self.social_attn_layers = nn.ModuleList(self.social_attn_layers)

        if not self.skip_temporal_attn_fn:
            self.time_encoder = TimeEncoding(self.emb_dim, dropout=self.dropout, max_len=self.H)

            self.temporal_attn_layers = []
            for _ in range(self.num_enc_layers):
                tx_encoder_layer = TransformerEncoderLayer(
                    d_model=self.emb_dim,
                    nhead=self.num_heads,
                    dropout=self.dropout,
                    dim_feedforward=self.tx_hidden_size,
                    norm_first=self.norm_first,
                    batch_first=True,
                )
                self.temporal_attn_layers.append(tx_encoder_layer)
            self.temporal_attn_layers = nn.ModuleList(self.temporal_attn_layers)

        self.agent_cross_layers = []
        for _ in range(self.num_enc_layers):
            agent_cross_layer = TransformerDecoderLayer(
                d_model=self.emb_dim,
                nhead=self.num_heads,
                dropout=self.dropout,
                dim_feedforward=self.tx_hidden_size,
                norm_first=self.norm_first,
                batch_first=True,
            )
            self.agent_cross_layers.append(agent_cross_layer)
        self.agent_cross_layers= nn.ModuleList(self.agent_cross_layers)

        self.map_cross_layers = []
        for _ in range(self.num_enc_layers):
            map_cross_layer = TransformerDecoderLayer(
                d_model=self.emb_dim,
                nhead=self.num_heads,
                dropout=self.dropout,
                dim_feedforward=self.tx_hidden_size,
                norm_first=self.norm_first,
                batch_first=True,
            )
            self.map_cross_layers.append(map_cross_layer)
        self.map_cross_layers= nn.ModuleList(self.map_cross_layers)

    def social_attn_fn(self, traj_emb, traj_masks, layer):
        '''
        :param agents_emb: (B, H, N, d)
        :param agent_masks: (B, H, N)
        :return: (B, H, N, d)
        '''
        B, H, N, d = traj_emb.shape
        traj_emb = traj_emb.reshape((B * H, N, d))
        traj_masks = traj_masks.reshape((B * H, N))
        traj_masks = torch.where(traj_masks.all(dim=-1, keepdims=True), torch.zeros_like(traj_masks), traj_masks)
        soc_emb = layer(
            traj_emb,
            src_key_padding_mask=traj_masks)
        soc_emb = soc_emb.reshape((B, H, N, d))
        if soc_emb.isnan().any():
            import pdb; pdb.set_trace()
        return soc_emb

    def agent_cross_fn(self, traj_emb, traj_masks, agents_emb, agents_masks, layer):
        '''
        :param traj_emb: (B, H, N, d)
        :param agents_masks: (B, H, N)
        :param map_emb: (B, P, d)
        :param map_masks: (B, P)
        :return: (B, H, N, d)
        '''
        B, H, N, d = agents_emb.shape
        _, _, N_traj, d = traj_emb.shape

        traj_emb = traj_emb.repeat(1, H, 1, 1).reshape((B * H, -1, d))
        traj_masks = traj_masks.repeat(1, H, 1).reshape((B * H, -1))
        traj_masks = torch.where(traj_masks.all(dim=-1, keepdims=True), torch.zeros_like(traj_masks), traj_masks)
        agents_emb = agents_emb.reshape((B * H, N, d))
        agents_masks = agents_masks.reshape((B * H, N))
        agents_masks = torch.where(agents_masks.all(dim=-1, keepdims=True), torch.zeros_like(agents_masks), agents_masks)
   
        cross_atten_emb = layer(
            traj_emb, # torch.squeeze(traj_emb),
            agents_emb,
            tgt_key_padding_mask=traj_masks,
            memory_key_padding_mask=agents_masks).reshape((B, H, N_traj, d))
        if cross_atten_emb.isnan().any():
            import pdb; pdb.set_trace()
        return cross_atten_emb
      

    def map_cross_fn(self, traj_emb, traj_masks, map_emb, map_masks, route_emb, route_masks, layer):
        '''
        :param agents_emb: (B, H, N, d)
        :param agents_masks: (B, H, N)
        :param map_emb: (B, P, d)
        :param map_masks: (B, P)
        :return: (B, H, N, d)
        '''
        # TODO is this the right way to do the time stuff
        B, H, N, d = traj_emb.shape
        traj_emb = traj_emb.transpose(1, 2).reshape((B * N, H, d))
        traj_masks = traj_masks.unsqueeze(1).transpose(1, 2).reshape((B * N, H))
        traj_masks = torch.where(traj_masks.all(dim=-1, keepdims=True), torch.zeros_like(traj_masks), traj_masks)
        map_emb = map_emb.unsqueeze(1).repeat(N, H, 1, 1).reshape((B * N * H, -1, d))
        map_masks = map_masks.unsqueeze(1).repeat(N, H, 1).reshape((B * N * H, -1))
        map_masks = torch.where(map_masks.all(dim=-1, keepdims=True), torch.zeros_like(map_masks), map_masks)

        if route_emb is not None and route_masks is not None:
            R = route_emb.shape[1]
            route_ego_masks = torch.ones((B * N, R), device=map_masks.device, dtype=bool)
            route_ego_masks[::N] = route_masks

            route_ego_emb = torch.zeros((B * N, R, d), device=map_emb.device, dtype=map_emb.dtype)
            route_ego_emb[::N] = route_emb

            map_emb = torch.cat([map_emb, route_ego_emb], dim=1)
            map_masks = torch.cat([map_masks, route_ego_masks], dim=1)

        map_cross_atten_emb = layer(
            traj_emb,
            map_emb,
            tgt_key_padding_mask=traj_masks,
            memory_key_padding_mask=map_masks).reshape((B, N, H, d)).transpose(1, 2)
        if map_cross_atten_emb.isnan().any():
            import pdb; pdb.set_trace()
        return map_cross_atten_emb


    def _get_attn_image(self, attn_name, locs, idx, base_color='white', mask=None, im_shape=(50, 50)):

        black = viz_utils.get_color('black')
        color = viz_utils.get_color(base_color)

        if 'dec' in attn_name:
            attn = self.attn_weights[attn_name].view((locs.shape[0], self.attn_weights[attn_name].shape[0] // locs.shape[0], self.num_modes * self.T, -1))[idx, 0].mean(dim=0).detach().cpu().numpy()
        else:
            attn = self.attn_weights[attn_name][idx, 0].detach().cpu().numpy()


        attn = attn / attn.max()
        attn_colors = attn[..., None] * (color - black) + black

        try:
            if mask is None:
                return viz_utils.get_image(locs[idx], attn_colors, im_shape=im_shape)
            else:
                return viz_utils.get_image(locs[idx][mask[idx]], attn_colors[mask[idx]], im_shape=im_shape)
        except:
            import pdb; pdb.set_trace()


    def forward(self, traj_emb, traj_mask, agents_emb, agents_masks, map_emb, map_masks, route_emb,
            route_masks, t):    
        traj_emb = self.ln1(traj_emb, t) #.reshape((B * num_agents, 1, self.T))

        for i in range(self.num_enc_layers):
            if not self.skip_temporal_attn_fn:
                agents_emb = self.temporal_attn_fn(agents_emb, agents_masks.clone(), layer=self.temporal_attn_layers[i])

            traj_emb = self.social_attn_fn(traj_emb, traj_mask.clone(), layer=self.social_attn_layers[i]) # FullAttention
            
        # CrossAttention
        traj_emb = self.ln1_1(traj_emb, t)
        traj_emb = self.agent_cross_fn(
            traj_emb,
            traj_mask.clone(),
            agents_emb,
            agents_masks.clone(),
            layer=self.agent_cross_layers[i],
        )

        traj_emb = self.map_cross_fn(
            traj_emb,
            traj_mask.clone(),
            map_emb,
            map_masks.clone(),
            route_emb,
            route_masks.clone(),
            layer=self.map_cross_layers[i],
        )

        traj_emb = self.ln1_1(traj_emb, t)

        return traj_emb