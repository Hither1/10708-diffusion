import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt

from src.models.embedders import Embedder
from src.models.map_models import MapEncoder
from src.models.transformer_layers import TransformerEncoderLayer, TransformerDecoderLayer
from src.models.diffusion.transformer_utils import AdaLayerNorm
from src.models.output_models import ResOutputModel, OutputModel
from src.models.diffusion.diffusion_utils import (
    vp_beta_schedule, 
    linear_beta_schedule, 
    SinusoidalPosEmb, 
    Whitener
)

class Model_Cond_Diffusion(nn.Module):
    def __init__(self, nn_model, betas, n_T, y_dim, emb_dim, dropout, num_agents, 
                n_layer=1,
                sigma_data: float = 0.5,
                sigma_min: float = 2e-3,
                sigma_max: float = 1e1,
                sigma_churn: float = 10,
                rho: float = 7,
                sigma_noise: float = 1,

                p_mean: float = -1.2,
                p_std: float = 1.2,

                pca_params_path: str = 'pca_params.th',
                k: int = 32,

                guide_w=0.0):
        super(Model_Cond_Diffusion, self).__init__()

        self.nn_model = nn_model
        self.n_T = n_T
        self.dropout = dropout
        self.loss_mse = nn.MSELoss()
        self.y_dim = y_dim
        self.wp_dim = 4
        self.num_agents = num_agents
        self.guide_w = guide_w
        self.feature_dim = emb_dim
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_churn = sigma_churn
        self.sigma_noise = sigma_noise
        self.rho = rho

        self.p_mean = p_mean
        self.p_std = p_std
        self.u_min, self.u_max = torch.log(torch.tensor(0.02)), torch.log(torch.tensor(100))

        self.k = k
        self.num_modes = 1
        self.predictions_per_sample = 32

        # Noise schedule
        self.sigma = lambda t: t

        # Compute discretized timesteps
        sigmas = (
            self.sigma_max ** (1 / self.rho) + \
            (torch.arange(self.n_T+1) / (self.n_T-1)) * \
            (self.sigma_min**(1 / self.rho) - self.sigma_max**(1 / self.rho)) \
        )**self.rho
        sigmas[-1] = 0
        self.register_buffer('ts', sigmas)

        # Noise level encoder
        self.sigma_encoder = nn.Sequential(
            SinusoidalPosEmb(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim * 2),
            nn.ReLU(),
            nn.Linear(self.feature_dim * 2, self.feature_dim)
        )

        self.trajectory_encoder = Embedder(self.k, emb_dim, expand_theta=True, layer_norm=False)
        self.whitening = False
        if self.whitening:
            self.whitener = Whitener(pca_params_path)
            # self.pca_whitener = PCAWhitener(self.k, pca_params_path)


    def loss_on_batch(self, gt_trajectory, traj_mask, obs):
        self.nn_model.training = True
        B, T, N, H, D = gt_trajectory.shape
        if self.whitening:
            gt_trajectory = self.whitener.transform_features(gt_trajectory.flatten(start_dim=3)).reshape(B, T, N, H, D)

        gt_trajectory = gt_trajectory.flatten(start_dim=-2)

        sigma = (torch.randn(gt_trajectory.shape[0], device=gt_trajectory.device) * self.p_std + self.p_mean).exp()[:, None, None, None]
        # elif: 
        # ts = torch.randint(low=0, high=self.n_T+1, size=(gt_trajectory.shape[0], ), device=gt_trajectory.device)
        # sigma = torch.stack([self.ts[t] for t in ts])[:, None, None, None]
        # else: 
        #     sigma = (torch.rand(gt_trajectory.shape[0], device=gt_trajectory.device) * (self.u_max - self.u_min) + self.u_min).exp()[:, None, None, None]
        noisy_trajectory = torch.normal(gt_trajectory, sigma)

        # Predict denoised trajectory
        predictions, logits = self.denoise_with_preconditioning(
                noisy_trajectory,
                traj_mask,
                sigma,
                obs
        )

        if self.whitening:
            features = predictions[..., :self.wp_dim].flatten(start_dim=-2)
            predictions[..., :self.wp_dim] = self.whitener.untransform_features(features).reshape(B, self.num_modes, N, H, self.wp_dim)
            # std = predictions[..., self.wp_dim:].flatten(start_dim=-2)
            # predictions[..., self.wp_dim:] = self.whitener.untransform_std(std).reshape(B, self.num_modes, N, H, self.wp_dim)
            noisy_trajectory = self.whitener.untransform_features(noisy_trajectory)

        loss_weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data)**2
        return logits, noisy_trajectory, predictions

    def denoise_with_preconditioning(self, trajectory, trajectory_masks, sigma, obs):
        """
        Denoise ego_trajectory with noise level sigma
        Equivalent to evaluating D_theta in this paper: https://arxiv.org/pdf/2206.00364
        Returns denoised trajectory, not the residual noise
        """
        c_in = 1 / torch.sqrt(sigma**2 + self.sigma_data**2)
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / torch.sqrt(sigma**2 + self.sigma_data**2)
        # c_noise = .25 * torch.log(sigma)
        c_noise = torch.log(sigma)
        out, logits = self.denoise( 
                trajectory, 
                trajectory_masks,
                c_noise,
                obs
        )
        
        # out, logits = self.denoise( 
        #         c_in * trajectory, 
        #         trajectory_masks,
        #         c_noise,
        #         obs
        # )
        # B, H, N, _ = trajectory.shape
        # print(c_skip.shape, trajectory.shape, out.shape) # torch.Size([128, 1, 1, 1]) torch.Size([128, 1, 10, 32]) torch.Size([128, 1, 10, 8, 8])
        # out = (c_skip * trajectory).reshape((B, H, N, self.T, self.wp_dim)) + c_out * out
        return out, logits

    def denoise(self, trajectories, trajectory_masks, sigma, obs):
        """
        Denoise ego_trajectory with noise level sigma (no preconditioning)
        Equivalent to evaluating F_theta in this paper: https://arxiv.org/pdf/2206.00364
        """
        # Diffusion
        trajectory_features = self.trajectory_encoder(trajectories)
        if trajectory_masks is None:
            trajectory_masks = torch.zeros(trajectory_features.shape[:-1], dtype=bool, device=trajectory_features.device)

        # Add diffusion noise encodings
        sigma_embeddings = self.sigma_encoder(sigma).squeeze(1)

        trajectory_features = trajectory_features + sigma_embeddings
        
        out, logits, features = self.nn_model(trajectory_features, trajectory_masks, obs) # torch.Size([32, 1, 8, 10, 8])

        return out.permute(0, 1, 3, 2, 4)[:, :, :self.num_agents], logits # torch.Size([32, 1, 10, 8, 8])

    def sample(self, obs, return_y_trace=True, extract_embedding=False):
        self.nn_model.training = False
        agents_features = obs['vehicle_features']
        batch_size = agents_features.shape[0]
        num_agents = min(self.num_agents, agents_features.shape[2]) # agents_features.shape[2]
        y_shape = (batch_size, 1, num_agents, 32)

        ego_gt_trajectory = torch.randn(y_shape).to(agents_features.device)
        ego_gt_trajectory = ego_gt_trajectory.repeat_interleave(self.predictions_per_sample, 0)
        # traj_mask = None
        traj_mask = ~obs['vehicle_masks'][:, :, :self.num_agents]
        traj_mask = traj_mask.repeat_interleave(self.predictions_per_sample, 0)

        # Sampling / inference
        ego_trajectory = torch.randn(ego_gt_trajectory.shape, device=agents_features.device) * self.sigma(self.ts[0])
        log_prob = Normal(0, 1).log_prob(ego_trajectory).sum(dim=(-1, -2, -3))

        y_i_store = []  
        # Heun's 2nd order method (algorithm 1): https://arxiv.org/pdf/2206.00364
        for i in range(self.n_T):
            t = self.ts[i]
            t_next = self.ts[i+1]
            sigma = self.sigma(t)[None, None, None, None].repeat(batch_size * self.predictions_per_sample, 1, 1, 1)
            sigma_next = self.sigma(t_next)[None, None, None, None].repeat(batch_size * self.predictions_per_sample, 1, 1, 1)

            # Increase noise temporarily.
            # gamma = min(self.sigma_churn / self.n_T, np.sqrt(2) - 1) if self.sigma_min <= self.sigma(t) <= self.sigma_max else 0
            # t_hat = torch.as_tensor(sigma + gamma * sigma, device=agents_features.device)
            # ego_trajectory = ego_trajectory + (t_hat ** 2 - sigma ** 2).sqrt() * self.sigma_noise * torch.randn(ego_trajectory.shape, device=agents_features.device)
            
            denoised, logits = self.denoise_with_preconditioning(
                ego_trajectory.clone(), 
                traj_mask, # None
                sigma,
                obs
            )
            d_i = (1 / sigma) * (ego_trajectory - denoised[..., :self.wp_dim].flatten(start_dim=-2))
            ego_trajectory_next = ego_trajectory + (t_next - t) * d_i
            
                
            if (sigma_next != 0).all(): # Apply second order correction
                denoised_next, logits = self.denoise_with_preconditioning(
                        ego_trajectory_next, 
                        traj_mask,
                        sigma_next,
                        obs
                )
                d_ip = (1 / sigma_next) * (ego_trajectory_next - denoised_next[..., :self.wp_dim].flatten(start_dim=-2))
                ego_trajectory_next = ego_trajectory + (t_next - t) * 0.5 * (d_i + d_ip)

            ego_trajectory = ego_trajectory_next

            eps = torch.randn_like(ego_trajectory)
            with torch.enable_grad():
                f = lambda x: (-1 / sigma) * self.denoise_with_preconditioning(x, traj_mask, sigma, obs)[0][..., :self.wp_dim].flatten(start_dim=-2)
                ego_trajectory.requires_grad_(True)
                f_eps = torch.sum(f(ego_trajectory) * eps)
                grad_f_eps = torch.autograd.grad(f_eps, ego_trajectory)[0]
            ego_trajectory.requires_grad_(False)
            out = torch.sum(grad_f_eps * eps, dim=tuple(range(1, len(ego_trajectory.shape))))
        
            log_prob = log_prob - (t_next - t) * out

            if return_y_trace and (i % (self.n_T // 5) == 0 or i == self.n_T or (self.n_T - i) < 3):
                if self.whitening:
                    y_i_store.append(self.whitener.untransform_features(ego_trajectory).detach().reshape(batch_size, self.predictions_per_sample, num_agents, self.y_dim))
                else:
                    y_i_store.append(ego_trajectory.detach().reshape(batch_size, self.predictions_per_sample, num_agents, self.y_dim))

        ego_trajectory = ego_trajectory.reshape(batch_size, self.predictions_per_sample, num_agents, self.y_dim)
        log_prob = log_prob.reshape(batch_size, self.predictions_per_sample)
        log_prob = log_prob - log_prob.max(dim=1, keepdim=True).values
        prob = (log_prob / 10).exp() # smoothing factor
        prob = prob / prob.sum(dim=1, keepdim=True)
        best_trajectory = ego_trajectory[range(batch_size), prob.argmax(dim=1)]

        if self.whitening:
            best_trajectory = self.whitener.untransform_features(best_trajectory)
        if return_y_trace:
            return best_trajectory, y_i_store
        else:
            return best_trajectory

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

            z = torch.randn(y_shape).to(self.device) if i > 1 else 0

            # split predictions and compute weighting
            eps, _, _ = self.nn_model(y_i, agents_emb, agents_mask, map_emb, map_mask, t_is)
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
            eps, _, _ = self.nn_model(y_i, x_batch, t_is, context_mask)
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
    def __init__(self, n_T, y_dim, emb_dim, 
                num_heads, 
                num_modes,
                num_map_pts,
                agent_dim,
                map_dim,
                light_dim,
                stop_dim,
                route_dim,
                num_map_enc_layers,
                carla_maps_dict,
                num_route_pts,
                T=8,
                wp_dim=4,
                min_std=0.01,
                out_init_std=0.1,
                dt=0.5,
                f=1,
                activate='relu',
                num_enc_layers=2, 
                num_dec_layers=2, 
                dropout=0., 
                tx_hidden_factor=2, 
                norm_first=True, 
                output_dim=None):
        super(Model_mlp, self).__init__()

        self.T = T
        self.wp_dim = wp_dim
        self.min_std = min_std
        self.out_init_std = out_init_std
        self.dt = dt
        self.f = f
        self.num_map_pts = num_map_pts
        self.agent_dim = agent_dim
        self.map_dim = map_dim
        self.light_dim = light_dim
        self.stop_dim = stop_dim
        self.route_dim = route_dim
        self.num_map_enc_layers = num_map_enc_layers
        self.carla_maps_dict = carla_maps_dict
        self.num_route_pts = num_route_pts
        self.y_dim = y_dim
        self.emb_dim = emb_dim
        self.feature_dim = emb_dim
        self.num_heads = num_heads
        self.num_modes = num_modes
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        self.dropout = dropout
        self.tx_hidden_size = tx_hidden_factor * self.emb_dim
        self.norm_first = norm_first
        self.skip_temporal_attn_fn = True
        self.k = 32  # sometimes overwrite, eg for discretised, mean/variance, mixture density models
        self.q_dec = None
        if self.num_modes > 1:
            self.p_dec = True
        else:
            self.p_dec = False

        self.create_agent_embedder()
        self.create_map_encoder()
        self.create_route_encoder()
        self.create_agent_encoder()
        self.create_decoder()
        self.create_output_model()
        self.create_prob_model()

        # Decoder layers
        self.final = nn.Linear(self.emb_dim, self.k)

    def create_agent_embedder(self):
        self.agents_embedder = Embedder(self.agent_dim, self.emb_dim, expand_theta=True, layer_norm=False)
        #  if self.q_dec:
            #  self.q_decoding_agents_embedder = Embedder(self.agent_dim, self.emb_dim, expand_theta=True, layer_norm=True)
        #  if self.p_dec:
            #  self.p_decoding_agents_embedder = Embedder(self.agent_dim, self.emb_dim, expand_theta=True, layer_norm=True)

    def create_map_encoder(self):
        self.map_encoder = MapEncoder(
            self.map_dim,
            self.light_dim,
            self.stop_dim,
            self.emb_dim,
            self.num_map_enc_layers,
            num_heads=self.num_heads,
            tx_hidden_size=self.tx_hidden_size,
            norm_first=self.norm_first,
            dropout=self.dropout,
        )

    def create_route_encoder(self):
        self.route_encoder = Embedder(self.route_dim, self.emb_dim, expand_theta=True, layer_norm=True)

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

        # TODO have map specific parameters for these things
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

        self.agent_emb_ln = nn.LayerNorm(self.emb_dim)


    def create_output_model(self):
        self.output_model = ResOutputModel(
            emb_dim=self.emb_dim,
            dist_dim=self.wp_dim,
            min_std=self.min_std,
            layer_norm=self.norm_first,
            dropout=self.dropout,
            #  out_mean=self.output_mean[self.H:self.H+self.T, None] if self.output_mean is not None else None,
            #  out_std=self.output_std[self.H:self.H+self.T, None] if self.output_std is not None else None,
            # TODO best way to do this
            out_std=self.out_init_std * self.dt * self.f * torch.ones((self.T, self.wp_dim)),
            #  out_std=self.out_init_std * torch.ones((self.T, self.wp_dim)),
            #  wa_std=True,
        )

        # self.inverse_dynamics_model = InverseDynamicsModel(
        #     inp_dim=1 + self.wp_dim * self.inv_T, # TODO
        #     emb_dim=self.emb_dim,
        #     dist_dim=self.act_dim,
        #     min_std=self.min_std,
        #     num_hidden=self.inv_layers,
        #     norm='layer',
        #     dropout=self.dropout,
        #     #  action_mean=self.action_mean[self.H-1] if self.action_mean is not None else None,
        #     #  action_std=self.action_std[self.H-1] if self.action_std is not None else None,
        #     #  action_std=self.act_init_std * torch.ones((self.act_dim,)),
        #     #  tanh=True,
        #     #  wa_std=True,
        # )


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


    def map_cross_fn(self, agents_emb, agents_masks, map_emb, map_masks, layer):
        '''
        :param agents_emb: (B, H, N, d)
        :param agents_masks: (B, H, N)
        :param map_emb: (B, P, d)
        :param map_masks: (B, P)
        :return: (B, H, N, d)
        '''
        B, H, N, d = agents_emb.shape
        agents_emb = agents_emb.reshape((B * H, N, d))
        agents_masks = agents_masks.reshape((B * H, N))
        agents_masks = torch.where(agents_masks.all(dim=-1, keepdims=True), torch.zeros_like(agents_masks), agents_masks)
        map_emb = map_emb.unsqueeze(1).repeat(1, H, 1, 1).reshape((B * H, -1, d))
        map_masks = map_masks.unsqueeze(1).repeat(1, H, 1).reshape((B * H, -1))
        map_masks = torch.where(map_masks.all(dim=-1, keepdims=True), torch.zeros_like(map_masks), map_masks)
        map_cross_atten_emb = layer(
            agents_emb,
            map_emb,
            tgt_key_padding_mask=agents_masks,
            memory_key_padding_mask=map_masks).reshape((B, H, N, d))
        if map_cross_atten_emb.isnan().any():
            import pdb; pdb.set_trace()
        return map_cross_atten_emb

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


    def social_attn_fn(self, agents_emb, agent_masks, layer):
        '''
        :param agents_emb: (B, H, N, d)
        :param agent_masks: (B, H, N)
        :return: (B, H, N, d)
        '''
        B, H, N, d = agents_emb.shape
        agents_emb = agents_emb.reshape((B * H, N, d))
        agent_masks = agent_masks.reshape((B * H, N))
        agent_masks = torch.where(agent_masks.all(dim=-1, keepdims=True), torch.zeros_like(agent_masks), agent_masks)
        agents_soc_emb = layer(
            agents_emb,
            src_key_padding_mask=agent_masks)
        agents_soc_emb = agents_soc_emb.reshape((B, H, N, d))
        if agents_soc_emb.isnan().any():
            import pdb; pdb.set_trace()
        return agents_soc_emb


    def create_prob_model(self):
        modules = []
        if self.norm_first:
            modules.append(nn.LayerNorm(self.emb_dim))

        # TODO generalize
        for _ in range(2):
            modules.append(nn.Linear(self.emb_dim, self.emb_dim))
            modules.append(nn.ReLU())
            if self.dropout > 0.0:
                modules.append(nn.Dropout(self.dropout))

        modules.append(nn.Linear(self.emb_dim, 1))
        self.prob_model = nn.Sequential(*modules)


    def get_encoding(self, traj_emb, traj_mask, obs):
        features = self.process_observations(obs)
        agents_features, agents_masks, light_features, light_masks, stop_features, stop_masks, walker_features, walker_masks, route_features, route_masks, map_features, map_masks = features
        agents_emb = self.agents_embedder(agents_features)

        if not self.training:
            predictions_per_sample = 32
            agents_emb = agents_emb.repeat_interleave(predictions_per_sample, 0)
            agents_masks = agents_masks.repeat_interleave(predictions_per_sample, 0)

        _zeros = torch.zeros_like(agents_emb)
        _zeros[:, :, :traj_emb.shape[2]] += traj_emb
        traj_emb = _zeros
        agents_emb =  agents_emb + traj_emb

        map_emb, map_masks = self.map_encoder(
            map_features,
            map_masks,
            light_features=light_features,
            light_masks=light_masks,
            stop_features=stop_features,
            stop_masks=stop_masks,
        )
        route_emb = self.route_encoder(route_features)

        if not self.training:
            map_emb = map_emb.repeat_interleave(predictions_per_sample, 0)
            map_masks = map_masks.repeat_interleave(predictions_per_sample, 0)
            route_emb = route_emb.repeat_interleave(predictions_per_sample, 0)
            route_masks = route_masks.repeat_interleave(predictions_per_sample, 0)

        # Process through AutoBot's encoder
        for i in range(self.num_enc_layers):
            if not self.skip_temporal_attn_fn:
                agents_emb = self.temporal_attn_fn(agents_emb, agents_masks.clone(), layer=self.temporal_attn_layers[i])
            agents_emb = self.map_cross_fn(
                agents_emb,
                agents_masks.clone(),
                map_emb,
                map_masks.clone(),
                layer=self.map_cross_layers[i],
            )
            agents_emb = self.social_attn_fn(agents_emb, agents_masks.clone(), layer=self.social_attn_layers[i])

        agents_emb = self.agent_emb_ln(agents_emb)

        
        return agents_emb, agents_masks, map_emb, map_masks, route_emb, route_masks, features

    def get_decoding(self, agents_emb, agents_masks, map_emb, map_masks, route_emb, route_masks, features, modes=None):
        if modes is None:
            num_modes = self.num_modes
        else:
            num_modes = 1
        B = agents_emb.shape[0]
        num_agents = agents_masks.shape[-1]

        out_masks = agents_masks.reshape((B, 1, -1, num_agents))
        out_masks = out_masks[:, :, -1:, :].repeat(1, num_modes, self.T, 1)

        if self.q_dec:
            out_seq = agents_emb.reshape((B, 1, -1, num_agents, self.emb_dim))

            out_seq = out_seq[:, :, -1:, :, :].repeat(1, num_modes, self.T, 1, 1)
            out_seq = out_seq + self.Q.repeat(B, 1, 1, num_agents, 1)
            out_seq = self.dec_time_pe(out_seq.transpose(2, 3).reshape((-1, self.T, self.emb_dim))).reshape((B, self.num_modes, num_agents, self.T, self.emb_dim)).transpose(2, 3)
        else:
            out_seq = agents_emb.reshape((B, 1, -1, num_agents, self.emb_dim))
            out_seq = out_seq[:, :, -1:, :, :].repeat(1, num_modes, self.T, 1, 1)

        out_seq = out_seq.reshape((B, num_modes * self.T, num_agents, -1))
        out_masks = out_masks.reshape((B, num_modes * self.T, num_agents))
        for d in range(self.num_dec_layers):
            # TODO masking
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

    def get_prob_decoding(self, agents_emb, agents_masks, map_emb, map_masks, route_emb, route_masks, features):
        B = agents_emb.shape[0]
        num_agents = agents_masks.shape[-1]
        prob_seq = agents_emb.reshape((B, 1, -1, num_agents, self.emb_dim))  # TODO maybe detach

        prob_seq = prob_seq[:, :, -1, :, :].repeat(1, self.num_modes, 1, 1)
        prob_masks = agents_masks.reshape((B, 1, -1, num_agents))
        prob_masks = prob_masks[:, :, -1, :].repeat(1, self.num_modes, 1)

        prob_seq = prob_seq + self.P.repeat(B, 1, num_agents, 1)

        prob_seq = prob_seq.view((B, self.num_modes, num_agents, -1))
        prob_masks = prob_masks.view((B, self.num_modes, num_agents))

        for d in range(self.num_dec_layers):
            prob_seq = self.map_dec_fn(
                prob_seq,
                prob_masks.clone(),
                map_emb,
                map_masks.clone(),
                layer=self.prob_map_dec_layers[d],
                route_emb=route_emb,
                route_masks=route_masks.clone(),
            )
            prob_seq = self.dec_fn(
                prob_seq,
                prob_masks.clone(),
                agents_emb,
                agents_masks.clone(),
                layer=self.prob_tx_decoder[d])

        prob_seq = prob_seq.view((B, self.num_modes, num_agents, -1))

        return prob_seq

    def get_output(self, out_seq):
        B = out_seq.shape[0]
        num_modes = out_seq.shape[1]
        num_agents = out_seq.shape[3]
        outputs = self.output_model(out_seq.permute(0, 3, 1, 2, 4).reshape((B * num_agents, num_modes, self.T, self.emb_dim)))
        outputs = outputs.reshape((B, num_agents, num_modes, self.T, -1)).permute(0, 2, 3, 1, 4)
        return outputs

    def get_prob_output(self, prob_seq):
        B = prob_seq.shape[0]
        num_modes = prob_seq.shape[1]
        num_agents = prob_seq.shape[2]
        logits = self.prob_model(prob_seq).reshape((B, num_modes, num_agents))
        return logits
    

    def forward(self, traj_emb, traj_mask, obs, modes=None):
        # but we feed in batch_size, height, width, channels
        # mask out context embedding, x_e, if context_mask == 1
        # x_e = x_e * (-1 * (1 - context_mask))
        # roughly following this: https://jalammar.github.io/illustrated-transformer/
        
        agents_emb, agents_masks, map_emb, map_masks, route_emb, route_masks, features = self.get_encoding(traj_emb, traj_mask, obs)
        out_seq = self.get_decoding(agents_emb, agents_masks, map_emb, map_masks, route_emb, route_masks, features, modes=modes)
        outputs = self.get_output(out_seq)

        if self.p_dec:
            prob_seq = self.get_prob_decoding(agents_emb, agents_masks, map_emb, map_masks, route_emb, route_masks, features)
            logits = self.get_prob_output(prob_seq)
        else:
            logits = None

        return outputs, logits, features


    def _get_map_features(self, obs):
        refs = obs['ref']
        towns = obs['town']
        B = len(towns)
        map_features = torch.zeros((B, self.num_map_pts, self.map_dim), dtype=torch.float32, device=refs.device)
        map_masks = torch.ones((B, self.num_map_pts), dtype=bool, device=refs.device)

        for town in np.unique(towns):
            idxs = np.where(town == towns)[0]
            if isinstance(town, bytes):
                town_map_features, town_map_masks = self.carla_maps_dict[town.decode('utf-8')].get_model_features(refs[idxs])
            else:
                town_map_features, town_map_masks = self.carla_maps_dict[town].get_model_features(refs[idxs])
            map_features[idxs] = town_map_features
            map_masks[idxs] = town_map_masks
        return map_features, map_masks

    
    def process_observations(self, obs):
        '''
        :param observations: (B, T, N+2, A+1) where N+2 is [ego, other_agents, env]
        :return: a tensor of only the agent dynamic states, active_agent masks and env masks.
        '''
        agents_features = obs['vehicle_features']
        agents_masks = ~obs['vehicle_masks']

        map_features, map_masks = self._get_map_features(obs)
        map_masks = ~map_masks

        # TODO should we use historical context for lights
        light_features = obs['light_features'][:, -1]
        light_masks = ~obs['light_masks'][:, -1]

        stop_features = obs['stop_features']
        stop_masks = ~obs['stop_masks']

        walker_features = obs['walker_features']
        walker_masks = ~obs['walker_masks']

        route_features = obs['route_features']
        route_masks = ~obs['route_masks']

        route_masks = route_masks | (route_features[..., :2].norm(dim=-1) > 1.)
        route_masks = route_masks | (torch.cumsum(~route_masks, dim=-1) > self.num_route_pts)

        return agents_features, agents_masks, light_features, light_masks, stop_features, stop_masks, walker_features, walker_masks, route_features, route_masks, map_features, map_masks


class Mlp_motion_diffuser(nn.Module):
    def __init__(self, n_T, y_dim, emb_dim, 
                num_heads, 
                num_modes,
                num_map_pts,
                agent_dim,
                map_dim,
                light_dim,
                stop_dim,
                route_dim,
                num_map_enc_layers,
                carla_maps_dict,
                num_route_pts,
                T=8,
                wp_dim=4,
                min_std=0.01,
                out_init_std=0.1,
                dt=0.5,
                f=1,
                activate='relu',
                num_enc_layers=2, 
                num_dec_layers=2, 
                dropout=0., 
                tx_hidden_factor=2, 
                n_layer=1,
                norm_first=True, 
                output_dim=None):
        super(Mlp_motion_diffuser, self).__init__()

        self.T = T
        self.wp_dim = wp_dim
        self.min_std = min_std
        self.out_init_std = out_init_std
        self.dt = dt
        self.f = f
        self.num_map_pts = num_map_pts
        self.agent_dim = agent_dim
        self.map_dim = map_dim
        self.light_dim = light_dim
        self.stop_dim = stop_dim
        self.route_dim = route_dim
        self.num_map_enc_layers = num_map_enc_layers
        self.carla_maps_dict = carla_maps_dict
        self.num_route_pts = num_route_pts
        self.y_dim = y_dim
        self.emb_dim = emb_dim
        self.feature_dim = emb_dim
        self.num_heads = num_heads
        self.num_modes = num_modes
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        self.dropout = dropout
        self.tx_hidden_size = tx_hidden_factor * self.emb_dim
        self.norm_first = norm_first
        self.skip_temporal_attn_fn = True
        self.k = 32  # sometimes overwrite, eg for discretised, mean/variance, mixture density models
        self.q_dec = None
        self.p_dec = None

        self.create_agent_embedder()
        self.create_map_encoder()
        self.create_route_encoder()
        self.agent_emb_ln = nn.LayerNorm(self.emb_dim)

        self.blocks = nn.Sequential(*[Block(
                emb_dim=emb_dim,
                n_head=num_heads,
                attn_pdrop=dropout,
                activate=activate,
        ) for n in range(n_layer)])

        self.output_model = ResOutputModel(
            emb_dim=self.emb_dim,
            dist_dim=self.wp_dim,
            min_std=self.min_std,
            layer_norm=self.norm_first,
            dropout=self.dropout,
            out_std=self.out_init_std * self.dt * self.f * torch.ones((self.T, self.wp_dim)),
        )

        self.final = nn.Linear(self.emb_dim, self.k) # Decoder layers

    def create_agent_embedder(self):
        self.agents_embedder = Embedder(self.agent_dim, self.emb_dim, expand_theta=True, layer_norm=False)
        #  if self.q_dec:
            #  self.q_decoding_agents_embedder = Embedder(self.agent_dim, self.emb_dim, expand_theta=True, layer_norm=True)
        #  if self.p_dec:
            #  self.p_decoding_agents_embedder = Embedder(self.agent_dim, self.emb_dim, expand_theta=True, layer_norm=True)

    def create_map_encoder(self):
        self.map_encoder = MapEncoder(
            self.map_dim,
            self.light_dim,
            self.stop_dim,
            self.emb_dim,
            self.num_map_enc_layers,
            num_heads=self.num_heads,
            tx_hidden_size=self.tx_hidden_size,
            norm_first=self.norm_first,
            dropout=self.dropout,
        )

    def create_route_encoder(self):
        self.route_encoder = Embedder(self.route_dim, self.emb_dim, expand_theta=True, layer_norm=True)


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


    def process_observations(self, obs):
        '''
        :param observations: (B, T, N+2, A+1) where N+2 is [ego, other_agents, env]
        :return: a tensor of only the agent dynamic states, active_agent masks and env masks.
        '''
        agents_features = obs['vehicle_features']
        agents_masks = ~obs['vehicle_masks']

        map_features, map_masks = self._get_map_features(obs)
        map_masks = ~map_masks

        light_features = obs['light_features'][:, -1]
        light_masks = ~obs['light_masks'][:, -1]

        stop_features = obs['stop_features']
        stop_masks = ~obs['stop_masks']

        walker_features = obs['walker_features']
        walker_masks = ~obs['walker_masks']

        route_features = obs['route_features']
        route_masks = ~obs['route_masks']

        route_masks = route_masks | (route_features[..., :2].norm(dim=-1) > 1.)
        route_masks = route_masks | (torch.cumsum(~route_masks, dim=-1) > self.num_route_pts)

        return agents_features, agents_masks, light_features, light_masks, stop_features, stop_masks, walker_features, walker_masks, route_features, route_masks, map_features, map_masks


    def _get_map_features(self, obs):
        refs = obs['ref']
        towns = obs['town']
        B = len(towns)
        map_features = torch.zeros((B, self.num_map_pts, self.map_dim), dtype=torch.float32, device=refs.device)
        map_masks = torch.ones((B, self.num_map_pts), dtype=bool, device=refs.device)

        for town in np.unique(towns):
            idxs = np.where(town == towns)[0]
            if isinstance(town, bytes):
                town_map_features, town_map_masks = self.carla_maps_dict[town.decode('utf-8')].get_model_features(refs[idxs])
            else:
                town_map_features, town_map_masks = self.carla_maps_dict[town].get_model_features(refs[idxs])
            map_features[idxs] = town_map_features
            map_masks[idxs] = town_map_masks
        return map_features, map_masks


    def forward(self, traj_emb, traj_mask, obs, modes=None):
        # roughly following this: https://jalammar.github.io/illustrated-transformer/
        B, H, N, _ = traj_emb.shape
        features = self.process_observations(obs)
        agents_features, agents_masks, light_features, light_masks, stop_features, stop_masks, walker_features, walker_masks, route_features, route_masks, map_features, map_masks = features                                                                                                                                                                                                                                                                                                                                            
        agents_emb = self.agents_embedder(agents_features)
        
        map_emb, map_masks = self.map_encoder(
            map_features,
            map_masks,
            light_features=light_features,
            light_masks=light_masks,
            stop_features=stop_features,
            stop_masks=stop_masks,
        )

        route_emb = self.route_encoder(route_features)
        route_mask_out = (~route_masks).any(dim=0)
        route_emb = route_emb[:, route_mask_out]
        route_masks = route_masks[:, route_mask_out]

        if not self.training:
            predictions_per_sample = 32
            agents_emb = agents_emb.repeat_interleave(predictions_per_sample, 0)
            agents_masks = agents_masks.repeat_interleave(predictions_per_sample, 0)
            map_emb = map_emb.repeat_interleave(predictions_per_sample, 0)
            map_masks = map_masks.repeat_interleave(predictions_per_sample, 0)
            route_emb = route_emb.repeat_interleave(predictions_per_sample, 0)
            route_masks = route_masks.repeat_interleave(predictions_per_sample, 0)

        agents_emb = self.agent_emb_ln(agents_emb)

        for block_idx in range(len(self.blocks)):   
            traj_emb = self.blocks[block_idx](traj_emb, traj_mask, agents_emb, agents_masks, map_emb, map_masks, route_emb, route_masks)
        
        # out = self.get_decoding(traj_emb, traj_mask, agents_emb, agents_masks, map_emb, map_masks, route_emb, route_masks)
        # out = self.get_output(out)
        # out = out[..., :4].permute(0, 1, 3, 2, 4).flatten(start_dim=3) 
        out = self.final(traj_emb)
        out = self.output_model.get_std(out.reshape((B, H, N, self.T, self.wp_dim)))
 
        return out.permute(0, 1, 3, 2, 4), None, features

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
        
        self.create_agent_encoder()

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
        _, _, M, _ = traj_emb.shape

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
            memory_key_padding_mask=agents_masks).reshape((B, H, M, d))
        if cross_atten_emb.isnan().any():
            import pdb; pdb.set_trace()
        return cross_atten_emb
      

    def map_cross_fn(self, traj_emb, traj_masks, map_emb, map_masks, route_emb, route_masks, layer):
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


    def forward(self, traj_emb, traj_mask, agents_emb, agents_masks, map_emb, map_masks, route_emb,
            route_masks):    
        for i in range(self.num_enc_layers):
            if not self.skip_temporal_attn_fn:
                agents_emb = self.temporal_attn_fn(agents_emb, agents_masks.clone(), layer=self.temporal_attn_layers[i])

            traj_emb = self.social_attn_fn(traj_emb, traj_mask.clone(), layer=self.social_attn_layers[i]) # FullAttention

        # CrossAttention
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

        return traj_emb