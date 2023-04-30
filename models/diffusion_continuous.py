import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt

from src.models.utils import get_cosine_schedule_with_warmup, PIDController
from src.models.inverse_dynamics_models import InverseDynamicsModel
from src.envs.carla.features.scenarios import transform_points
from src.models import viz_utils

from src.models.transformer_layers import TransformerEncoderLayer, TransformerDecoderLayer
from src.models.utils import nll_pytorch_dist, get_cosine_schedule_with_warmup, TimeEncoding
from src.models.embedders import Embedder
from src.models.map_models import MapEncoder
from src.envs.carla.features.scenarios import TOWNS
from src.envs.carla.features.carla_map_features import CarlaMapFeatures
# from src.models.diffusion.continuous_sde import Model_Cond_Diffusion, Model_mlp
# from src.models.diffusion.continuous_marginal import Model_Cond_Diffusion, Model_mlp
from src.models.diffusion.continuous_new import Model_Cond_Diffusion, Model_mlp, Mlp_motion_diffuser

from PIL import Image, ImageDraw



def fill_nearest_neighbor(array, mask):
    # Create a copy of the input array to modify
    filled_array = array.clone()

    filled_array[:, :, 0, :] = torch.where(~mask[:, :, 0, None], filled_array[:, :, 1, :], filled_array[:, :, 0, :])
    filled_array[:, :, 0, :] = torch.where(~mask[:, :, 1, None], torch.zeros_like(filled_array[:, :, 1, :]), filled_array[:, :, 0, :])

    for t in range(1, filled_array.shape[2]):
        # Set the current time step to the previous time step where mask is False
        filled_array[:, :, t, :] = torch.where(~mask[:, :, t, None], filled_array[:, :, t-1, :], filled_array[:, :, t, :])

    return torch.clamp(filled_array, max=2.0)


class WPImitationModel(pl.LightningModule):
    def __init__(
            self,
            *args,
            inv_T=1,
            inv_layers=1,
            wp_dim=4,
            dt=0.05,
            out_init_std=0.05,
            act_init_std=0.1,
            action_mean=None,
            action_std=None,
            std_update_tau=1.e-3,
            wp_pred_type='frame',
            agent_dim=7,
            act_dim=2,
            map_dim=4,
            light_dim=4,
            stop_dim=3,
            walker_dim=6,
            route_dim=4,
            num_map_pts=100,
            num_route_pts=20,
            num_modes=1,
            predictions_per_sample=32,
            T=1,
            H=1,
            f=1,
            emb_dim=128,
            num_enc_layers=1,
            num_dec_layers=1,
            num_map_enc_layers=1,
            num_heads=16,
            tx_hidden_factor=2,
            dropout=0.0,
            lr=1e-4,
            warmup_steps=1.e3,
            min_std=0.05,
            x_weight=20.,
            soft_targets=True,
            wd=1.e-2,
            output_mean=None,
            output_std=None,
            norm_first=False,
            use_scheduler=True,
            use_wandb=False,
            visualize=False,
            im_shape=(512, 512),
            carla_maps_path='maps/',
            max_token_distance=30.,
            max_z_distance=7.5,
            n_T=100,
            sigma_max=8e1,
            n_layer=1,
            p_mean=-1.2,
            p_std=1.2,
            **kwargs,
    ):
        self.inv_T = inv_T
        self.inv_layers = inv_layers
        self.wp_dim = wp_dim
        self.dt = dt
        self.out_init_std = out_init_std
        self.act_init_std = act_init_std
        self.action_mean = action_mean
        self.action_std = action_std
        self.std_update_tau = std_update_tau
        self.wp_pred_type = wp_pred_type
        self.num_agents = 3

        self.turn_controller = PIDController(K_P=0.9, K_I=0.75, K_D=0.3, n=20)
        self.speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=20)

        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.agent_dim = agent_dim
        self.act_dim = act_dim
        self.map_dim = map_dim
        self.light_dim = light_dim
        self.stop_dim = stop_dim
        self.walker_dim = walker_dim
        self.route_dim = route_dim
        self.num_map_pts = num_map_pts
        self.num_route_pts = num_route_pts
        self.num_modes = num_modes
        self.predictions_per_sample = predictions_per_sample
        self.T = T
        self.H = H # history
        self.f = f # frequency
        self.emb_dim = emb_dim
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        if (self.num_modes == 1) and (self.T == 1):
            self.q_dec = False
        else:
            self.q_dec = True
        if self.num_modes > 1:
            self.p_dec = True
        else:
            self.p_dec = False
        self.num_map_enc_layers = num_map_enc_layers
        self.num_heads = num_heads
        self.tx_hidden_size = tx_hidden_factor * self.emb_dim
        self.x_weight = x_weight
        self.soft_targets = soft_targets
        self.dropout = dropout
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.wd = wd
        self.norm_first = norm_first
        self.skip_temporal_attn_fn = self.H <= 1
        self.use_scheduler = use_scheduler

        self.use_wandb = use_wandb
        self.visualize = visualize
        self.im_shape = im_shape

        self.min_std = min_std
        self.output_mean = output_mean
        self.output_std = output_std

        self.n_T = n_T
        self.max_token_distance = max_token_distance
        self.carla_maps_dict = {}
        for town in TOWNS:
            self.carla_maps_dict[town] = CarlaMapFeatures(town, map_data_path=carla_maps_path, torch_device=self.device, max_token_distance=max_token_distance, max_z_distance=max_z_distance, max_map_pts=self.num_map_pts)

        self._dynamic_feature_names = ['vehicle_features', 'vehicle_masks', 'light_features', 'light_masks', 'walker_features', 'walker_masks']

        self.create_agent_embedder()
        self.create_map_encoder()
        self.create_route_encoder()
        self.agent_emb_ln = nn.LayerNorm(self.emb_dim)
        self.create_prob_model() # Mlp_motion_diffuser
        nn_model = Model_mlp(n_T=n_T, y_dim=self.T * self.wp_dim, emb_dim=self.emb_dim, num_heads=self.num_heads, 
                                num_modes=num_modes,
                                num_map_pts=num_map_pts, 
                                agent_dim=agent_dim,
                                map_dim=map_dim, 
                                light_dim=light_dim,
                                stop_dim=stop_dim,
                                route_dim=route_dim,
                                num_map_enc_layers=num_map_enc_layers,
                                carla_maps_dict=self.carla_maps_dict, 
                                num_route_pts=num_route_pts)
        self.diffusion = Model_Cond_Diffusion(nn_model, betas=(1e-4, 0.02), 
                                                n_T=n_T, 
                                                p_mean=p_mean,
                                                p_std=p_std,
                                                y_dim=self.T * self.wp_dim, 
                                                emb_dim=self.emb_dim, 
                                                num_agents=self.num_agents, 
                                                dropout=self.dropout,
                                                sigma_max=sigma_max,
                                                n_layer=n_layer
                                                )

        self.apply(self._init_weights)
        self.n_training_step = 0
        
        if self.visualize:
            self.attn_weights = {}
            for name, module in self.named_modules():
                if name[-4:] == 'attn':
                    module.name = name
                    def hook(module, input, output):
                        self.attn_weights[module.name] = output[1].detach()
                    module.register_forward_hook(hook)


    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.elementwise_affine == True:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def create_agent_embedder(self):
        self.agents_embedder = Embedder(self.agent_dim, self.emb_dim, expand_theta=True, layer_norm=False)

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

    def create_output_model(self):
        self.inverse_dynamics_model = InverseDynamicsModel(
            inp_dim=1 + self.wp_dim * self.inv_T, # TODO
            emb_dim=self.emb_dim,
            dist_dim=self.act_dim,
            min_std=self.min_std,
            num_hidden=self.inv_layers,
            norm='layer',
            dropout=self.dropout,
        )

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


    def get_prob_decoding(self, agents_emb, agents_masks, map_emb, map_masks, route_emb, route_masks, features):
        B = agents_emb.shape[0]
        num_agents = agents_masks.shape[-1]
        prob_seq = agents_emb.reshape((B, 1, -1, num_agents, self.emb_dim))

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

    def get_prob_output(self, prob_seq):
        B = prob_seq.shape[0]
        num_modes = prob_seq.shape[1]
        num_agents = prob_seq.shape[2]
        logits = self.prob_model(prob_seq).reshape((B, num_modes, num_agents))
        return logits

    def get_pred_next_wps(self, obs, outputs):
        if self.wp_pred_type == 'frame':
            output_thetas = torch.cumsum(outputs[..., 2], dim=2)
            output_speeds = torch.cumsum(outputs[..., 3:4], dim=2)
            output_others = outputs[..., 4:self.wp_dim]

            thetas = output_thetas - outputs[..., 2]

            rot_matrix = torch.stack([
                torch.stack([ torch.cos(thetas), torch.sin(thetas)], dim=-1),
                torch.stack([-torch.sin(thetas), torch.cos(thetas)], dim=-1)
            ], dim=-2)

            output_pos_diffs = torch.matmul(outputs[..., None, :2], rot_matrix).squeeze(-2)
            output_pos = torch.cumsum(output_pos_diffs, dim=2)

            outputs_mean = torch.cat([output_pos, output_thetas.unsqueeze(-1), output_speeds, output_others], dim=-1)

            # cumulative std
            outputs_std = torch.sqrt(torch.cumsum(outputs[..., self.wp_dim:] ** 2, dim=2))
            pred_next_wps = torch.cat([outputs_mean, outputs_std], dim=-1)

        elif self.wp_pred_type == 'bicycle':
            B = outputs.shape[0]
            num_agents = outputs.shape[3]

            d_thetas = outputs[..., 2]
            d_speeds = outputs[..., 3]

            # TODO dt already incorporated in out_init_std
            delta_total_thetas = torch.cumsum(d_thetas, dim=2)
            delta_total_speeds = torch.cumsum(d_speeds, dim=2)
            thetas = obs['vehicle_features'][:, -1, :, 2].reshape((B, 1, 1, num_agents)) + delta_total_thetas
            speeds = obs['vehicle_features'][:, -1, :, 3].reshape((B, 1, 1, num_agents)) + delta_total_speeds

            eff_thetas = thetas - 0.5 * d_thetas
            eff_speeds = speeds - 0.5 * d_speeds

            dxs = eff_speeds * torch.cos(eff_thetas) * self.dt * self.f
            xs = torch.cumsum(dxs, dim=2)
            dys = eff_speeds * torch.sin(eff_thetas) * self.dt * self.f
            ys = torch.cumsum(dys, dim=2)

            outputs_mean = torch.stack([xs, ys, thetas, speeds], dim=-1)

            outputs_std = torch.sqrt(torch.cumsum(outputs[..., self.wp_dim:] ** 2, dim=2))
            pred_next_wps = torch.cat([outputs_mean, outputs_std], dim=-1)
        else:
            raise NotImplementedError
        return pred_next_wps
    
    def _get_map_features(self, obs):
        refs = obs['ref']
        towns = obs['town']

        B = len(towns)

        map_features = torch.zeros((B, self.num_map_pts, self.map_dim), dtype=torch.float32, device=self.device)
        map_masks = torch.ones((B, self.num_map_pts), dtype=bool, device=self.device)

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

    # TODO this needs to be implemented
    def temporal_attn_fn(self, agents_emb, agent_masks, layer):
        '''
        :param agents_emb: (B, H, N, d)
        :param agent_masks: (B, H, N)
        :return: (B, H, N, d)
        '''
        B, H, N, d = agents_emb.shape
        agents_emb = agents_emb.transpose(1, 2).reshape((B * N, H, d))
        agent_masks = agent_masks.transpose(1, 2).reshape((B * N, H))
        agent_masks = torch.where(agent_masks.all(dim=-1, keepdims=True), torch.zeros_like(agent_masks), agent_masks)
        # TODO start_t should handle eval when not enough context
        agents_temp_emb = layer(
            self.time_encoder(agents_emb, start_t=self.H-H),
            src_key_padding_mask=agent_masks)
        agents_temp_emb = agents_temp_emb.reshape((B, N, H, d)).transpose(1, 2)
        if agents_temp_emb.isnan().any():
            import pdb; pdb.set_trace()
        return agents_temp_emb


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


    def _inverse_action(self, obs, next_ego_wps):
        ego_speeds = obs['vehicle_features'][:, self.H - 1, 0, 3:4]
        B = ego_speeds.shape[0]
        inps = torch.cat([ego_speeds, next_ego_wps[:, :self.inv_T].reshape((B, self.wp_dim * self.inv_T))], dim=-1)
        # TODO smarter version of this
        #  if self.training:
            #  if self.output_std is not None:
                #  inps = inps + torch.randn_like(inps) * 0.01 * self.output_std.flatten()[self.wp_dim-1:self.wp_dim*(1+self.inv_T)].to(device=inps.device)
            #  else:
                #  inps = inps + torch.randn_like(inps) * 1.e-4
        acts = self.inverse_dynamics_model(inps)

        return acts

    @torch.inference_mode()
    def get_action(self, full_obs, return_features=False):
        obs = {}
        for k in full_obs:
            if k in self._dynamic_feature_names:
                obs[k] = full_obs[k][:, -self.H:]
            else:
                obs[k] = full_obs[k]
        pred_wps, pred_actions, logits, features = self.forward(obs)
        self.prev_preds = [pred_wps, pred_actions, logits]

        actions = pred_actions[:, :self.act_dim]
        if return_features:
            return actions, features
        else:
            return actions
        
    def _compute_regression_loss(self, pred, gt):
        #  regression_loss = nll_pytorch_dist(pred, gt, dist='normal')
        regression_loss = nll_pytorch_dist(pred, gt, dist='laplace')
        mse = F.mse_loss(pred[..., :gt.shape[-1]], gt, reduction='none')
        return regression_loss, mse

    def _compute_mse_loss(self, pred, gt):
        #  regression_loss = nll_pytorch_dist(pred, gt, dist='normal')
        mse = F.mse_loss(pred[..., :gt.shape[-1]], gt, reduction='none')
        return mse
    
    def _compute_modes_loss(self, logits, mse_error, regression_loss):
        #  target_labels = mse_error.argmin(dim=1)
        target_labels = regression_loss.argmin(dim=1)

        fit_loss = regression_loss[torch.arange(target_labels.shape[0], device=target_labels.device), target_labels]

        if self.soft_targets:
            # TODO what are the right targets
            targets = torch.softmax(-regression_loss, dim=1)
            #  targets = torch.softmax(-mse_error, dim=1)
            logit_loss = F.cross_entropy(logits, targets.detach(), reduction='none')
        else:
            logit_loss = F.cross_entropy(logits, target_labels.detach(), reduction='none')
        pred_labels = logits.argmax(dim=1)

        loss = fit_loss + self.x_weight * logit_loss
        return loss, fit_loss, logit_loss, pred_labels, target_labels
    

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.
        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.
        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}


    def _compute_loss(self, pred_wps, logits, gt_wps, wps_mask):
        B = pred_wps.shape[0]
        num_agents = wps_mask.shape[-1]
        shaped_pred_wps = pred_wps.reshape((B * self.num_modes, self.T, num_agents, 2 * self.wp_dim))
        shaped_gt_wps = gt_wps.unsqueeze(1).repeat_interleave(self.num_modes, 1).reshape((B * self.num_modes, self.T, num_agents, self.wp_dim))
        shaped_wps_regression_loss, shaped_wps_mse_errors = self._compute_regression_loss(shaped_pred_wps, shaped_gt_wps)
        
        #if diffusion_loss_weight is not None:
          #  shaped_wps_mse_errors = diffusion_loss_weight * shaped_wps_mse_errors
        if wps_mask is None:
            wps_mask = torch.ones(gt_wps.shape[:-1], dtype=bool, device=gt_wps.device)

        if self.num_modes > 1:
            time_wps_regression_loss = shaped_wps_regression_loss.reshape((B, self.num_modes, self.T, num_agents)).permute(0, 3, 1, 2).reshape((B * num_agents, self.num_modes, self.T))
            time_wps_mse_errors = shaped_wps_mse_errors.reshape((B, self.num_modes, self.T, num_agents, self.wp_dim)).permute(0, 3, 1, 2, 4).reshape((B * num_agents, self.num_modes, self.T, self.wp_dim))
            w_mask = wps_mask.unsqueeze(1).permute(0, 3, 1, 2).reshape((B * num_agents, 1, self.T))
            wps_regression_loss = (time_wps_regression_loss * w_mask).sum(dim=2) / torch.clamp(w_mask.sum(dim=2), min=1.)
            wps_mse_errors = (time_wps_mse_errors * w_mask.unsqueeze(-1)).sum(dim=2) / torch.clamp(w_mask.unsqueeze(-1).sum(dim=2), min=1.)

            logits = logits.transpose(1, 2).reshape((B * num_agents, self.num_modes))

            wps_mse_error = wps_mse_errors.sum(dim=-1)

            wps_m = w_mask.squeeze(1).any(dim=1)

            unmasked_wps_loss, _, unmasked_logit_loss, labels, target_labels = self._compute_modes_loss(logits, wps_mse_error, wps_regression_loss)
            wps_loss = unmasked_wps_loss[wps_m].mean()
            logit_loss = unmasked_logit_loss[wps_m].mean()

            wps_mse_errors = wps_mse_errors[torch.arange(labels.shape[0], device=labels.device), labels][wps_m].mean(dim=0)
            time_wps_min_mse_errors = time_wps_mse_errors[torch.arange(target_labels.shape[0], device=target_labels.device), target_labels]
            time_wps_mse = (time_wps_min_mse_errors * w_mask.reshape((B * num_agents, self.T, 1))).sum(dim=0) / torch.clamp(w_mask.reshape((B * num_agents, self.T, 1)).sum(dim=0), min=1.)
        else:
            time_wps_regression_loss = shaped_wps_regression_loss.permute(0, 2, 1).reshape((B * num_agents, self.T))
            time_wps_mse_errors = shaped_wps_mse_errors.permute(0, 2, 1, 3).reshape((B * num_agents, self.T, self.wp_dim))
            w_mask = wps_mask.permute(0, 2, 1).reshape((B * num_agents, self.T))

            wps_regression_loss = (time_wps_regression_loss * w_mask).sum(dim=1) / torch.clamp(w_mask.sum(dim=1), min=1.)
            wps_mse_errors = (time_wps_mse_errors * w_mask.unsqueeze(-1)).sum(dim=1) / torch.clamp(w_mask.unsqueeze(-1).sum(dim=1), min=1.)

            wps_m = w_mask.any(dim=1)
            wps_loss = wps_regression_loss[wps_m].mean()
            wps_mse_errors = wps_mse_errors[wps_m].mean(dim=0)

            time_wps_mse = (time_wps_mse_errors * w_mask.unsqueeze(-1)).sum(dim=0) / torch.clamp(w_mask.unsqueeze(-1).sum(dim=0), min=1.)

            logit_loss = 0.

        wps_errors = torch.sqrt(wps_mse_errors)
        time_wps_errors = torch.sqrt(time_wps_mse)
        return wps_loss, logit_loss, wps_errors, time_wps_errors

    def _select_mse(self, pred_wps, logits, gt_wps, wps_mask):
        B = pred_wps.shape[0] # 4, 32, 8, 10, 4
        num_agents = wps_mask.shape[-1]
        shaped_pred_wps = pred_wps.reshape((B * self.predictions_per_sample, self.T, num_agents, self.wp_dim))
        shaped_gt_wps = gt_wps.unsqueeze(1).repeat_interleave(self.predictions_per_sample, 1).reshape((B * self.predictions_per_sample, self.T, num_agents, self.wp_dim))
        shaped_wps_mse_errors = self._compute_mse_loss(shaped_pred_wps, shaped_gt_wps)

        time_wps_mse_errors = shaped_wps_mse_errors.reshape((B, self.predictions_per_sample, self.T, num_agents, self.wp_dim)).permute(0, 3, 1, 2, 4).reshape((B * num_agents, self.predictions_per_sample, self.T, self.wp_dim))
        w_mask = wps_mask.unsqueeze(1).permute(0, 3, 1, 2).reshape((B * num_agents, 1, self.T))
        wps_mse_errors = (time_wps_mse_errors * w_mask.unsqueeze(-1)).sum(dim=2) / torch.clamp(w_mask.unsqueeze(-1).sum(dim=2), min=1.)

        wps_mse_error = wps_mse_errors.sum(dim=-1)

        best_pred = pred_wps.permute(0, 3, 1, 2, 4).reshape((B * num_agents, self.predictions_per_sample, self.T, self.wp_dim))
        indices = torch.argmin(wps_mse_error, dim=1)
        best_pred = [x[i] for x, i in zip(best_pred, indices)]
        best_pred = torch.stack(best_pred)
        # best_pred = torch.gather(best_pred, 1, torch.argmin(wps_mse_error, dim=1)[:, None, None, None].expand(B * num_agents, 1, self.T, self.wp_dim))

        return best_pred.reshape((B, num_agents, 1, self.T, self.wp_dim)).permute(0, 2, 3, 1, 4)

    def _compute_loss_val(self, pred_wps, logits, gt_wps, wps_mask):
        B = pred_wps.shape[0]
        num_agents = wps_mask.shape[-1]
        shaped_pred_wps = pred_wps.reshape((B * self.num_modes, self.T, num_agents, self.wp_dim))
        shaped_gt_wps = gt_wps.unsqueeze(1).repeat_interleave(self.num_modes, 1).reshape((B * self.num_modes, self.T, num_agents, self.wp_dim))
        shaped_wps_mse_errors = self._compute_mse_loss(shaped_pred_wps, shaped_gt_wps)
        
        #if diffusion_loss_weight is not None:
          #  shaped_wps_mse_errors = diffusion_loss_weight * shaped_wps_mse_errors

        if wps_mask is None:
            wps_mask = torch.ones(gt_wps.shape[:-1], dtype=bool, device=gt_wps.device)

        if self.num_modes > 1:
            time_wps_regression_loss = shaped_wps_regression_loss.reshape((B, self.num_modes, self.T, num_agents)).permute(0, 3, 1, 2).reshape((B * num_agents, self.num_modes, self.T))
            time_wps_mse_errors = shaped_wps_mse_errors.reshape((B, self.num_modes, self.T, num_agents, self.wp_dim)).permute(0, 3, 1, 2, 4).reshape((B * num_agents, self.num_modes, self.T, self.wp_dim))
            w_mask = wps_mask.unsqueeze(1).permute(0, 3, 1, 2).reshape((B * num_agents, 1, self.T))
            wps_regression_loss = (time_wps_regression_loss * w_mask).sum(dim=2) / torch.clamp(w_mask.sum(dim=2), min=1.)
            wps_mse_errors = (time_wps_mse_errors * w_mask.unsqueeze(-1)).sum(dim=2) / torch.clamp(w_mask.unsqueeze(-1).sum(dim=2), min=1.)

            logits = logits.transpose(1, 2).reshape((B * num_agents, self.num_modes))

            wps_mse_error = wps_mse_errors.sum(dim=-1)

            wps_m = w_mask.squeeze(1).any(dim=1)

            unmasked_wps_loss, _, unmasked_logit_loss, labels, target_labels = self._compute_modes_loss(logits, wps_mse_error, wps_regression_loss)
            wps_loss = unmasked_wps_loss[wps_m].mean()
            logit_loss = unmasked_logit_loss[wps_m].mean()

            wps_mse_errors = wps_mse_errors[torch.arange(labels.shape[0], device=labels.device), labels][wps_m].mean(dim=0)
            time_wps_min_mse_errors = time_wps_mse_errors[torch.arange(target_labels.shape[0], device=target_labels.device), target_labels]
            time_wps_mse = (time_wps_min_mse_errors * w_mask.reshape((B * num_agents, self.T, 1))).sum(dim=0) / torch.clamp(w_mask.reshape((B * num_agents, self.T, 1)).sum(dim=0), min=1.)
        else:
            time_wps_mse_errors = shaped_wps_mse_errors.permute(0, 2, 1, 3).reshape((B * num_agents, self.T, self.wp_dim))
            w_mask = wps_mask.permute(0, 2, 1).reshape((B * num_agents, self.T))

            wps_mse_errors = (time_wps_mse_errors * w_mask.unsqueeze(-1)).sum(dim=1) / torch.clamp(w_mask.unsqueeze(-1).sum(dim=1), min=1.)

            wps_m = w_mask.any(dim=1)
            wps_mse_errors = wps_mse_errors[wps_m].mean(dim=0)

            time_wps_mse = (time_wps_mse_errors * w_mask.unsqueeze(-1)).sum(dim=0) / torch.clamp(w_mask.unsqueeze(-1).sum(dim=0), min=1.)

            logit_loss = 0.

        wps_errors = torch.sqrt(wps_mse_errors)
        time_wps_errors = torch.sqrt(time_wps_mse)
        return logit_loss, wps_errors, time_wps_errors

    def _compute_action_loss(self, pred_actions, gt_actions, wps_mask, actions_mask):
        actions_mask = actions_mask & wps_mask[:, 0, 0]
        act_regression_loss, act_mse_errors = self._compute_regression_loss(pred_actions, gt_actions)
        act_loss = act_regression_loss[actions_mask].mean()
        act_mse_errors = act_mse_errors[actions_mask].mean(dim=0)
        act_errors = torch.sqrt(act_mse_errors)

        return act_loss, act_errors

    def run_step(self, batch, prefix, optimizer_idx=None):
        full_obs = batch.get_obs(ref_t=self.H-1, f=self.f, device=self.device)
        obs = {}
        for k in full_obs:
            if k in self._dynamic_feature_names:
                obs[k] = full_obs[k][:, :self.H]
            else:
                obs[k] = full_obs[k]
        
        gt_wps, wps_mask = batch.get_traj(ref_t=self.H-1, f=self.f, num_features=self.wp_dim)
        gt_wps = gt_wps[:, self.H:self.H+self.T, :, :self.wp_dim].to(self.device) 
        wps_mask = wps_mask[:, self.H:self.H+self.T, :].to(self.device)

        batch_size = gt_wps.shape[0]
        veh_masks = obs['vehicle_masks'].any(dim=1).any(dim=0)
        obs['vehicle_features'] = obs['vehicle_features'][:, :, veh_masks] # [:, :, :self.num_agents]
        obs['vehicle_masks'] = obs['vehicle_masks'][:, :, veh_masks] # [:, :, :self.num_agents]
        gt_wps = gt_wps[:, :, veh_masks][:, :, :self.num_agents]
        wps_mask = wps_mask[:, :, veh_masks][:, :, :self.num_agents]

        if prefix == 'val':
            diffusion_pred, diffusion_trace = self.diffusion.sample(obs)
            diffusion_loss_weight = None
            self.traces = []
            for trace in diffusion_trace: # torch.Size([4, 32, N, 32])
                trace = trace.reshape((batch_size, self.predictions_per_sample, -1, self.T, self.wp_dim))[:, :, :self.num_agents].permute(0, 1, 3, 2, 4)
                trace = self.get_pred_next_wps(obs, trace)
                self.traces.append(trace)
            logits = None
            # diffusion_pred = diffusion_pred.reshape((batch_size, -1, 1, self.T, self.wp_dim)).permute(0, 3, 1, 2, 4).to(self.device)
            # pred_wps = self.get_pred_next_wps(obs, diffusion_pred)
            diffusion_pred = self._select_mse(diffusion_trace[-1].reshape((batch_size, self.predictions_per_sample, -1, self.T, self.wp_dim)).permute(0, 1, 3, 2, 4), 
                                            logits, gt_wps, wps_mask).to(self.device)
            pred_wps = self.get_pred_next_wps(obs, diffusion_pred)
            logit_loss, wps_errors, time_wps_errors = self._compute_loss_val(pred_wps, logits, gt_wps, wps_mask)
            wps_loss = torch.tensor(0)

        elif (optimizer_idx is None) or (optimizer_idx == 0):
            wps = gt_wps.permute(0, 2, 1, 3)
            wps = fill_nearest_neighbor(wps, wps_mask.permute(0, 2, 1))
    
            logits, corrupted, diffusion_pred = self.diffusion.loss_on_batch(wps.unsqueeze(1),  # gt_wps, 
                                                                            ~obs['vehicle_masks'][:, :, :self.num_agents], #~wps_mask.any(dim=1), 
                                                                            obs)
            # for i in range(len(diffusion_loss_weight)):
               # self.log(f'{prefix}/diffusion_loss_weight[{i}]', diffusion_loss_weight[i].detach(), batch_size=batch_size)
            
            corrupted = corrupted.reshape((batch_size, -1, 1, self.T, self.wp_dim)).permute(0, 3, 1, 2, 4).to(self.device)
            self.pred_wps_corrupted = self.get_pred_next_wps(obs, corrupted)
            diffusion_loss_weight = None

            diffusion_pred = diffusion_pred.reshape((batch_size, self.num_modes, -1, self.T, self.wp_dim * 2))[:, :self.num_agents].permute(0, 1, 3, 2, 4).to(self.device)
            pred_wps = self.get_pred_next_wps(obs, diffusion_pred)
            wps_loss, logit_loss, wps_errors, time_wps_errors = self._compute_loss(pred_wps, logits, gt_wps, wps_mask)

        features = self.process_observations(obs)
        wps_error = wps_errors.sum(dim=-1)

        if self.use_wandb:
            self.log(f'{prefix}/wps_loss', wps_loss.detach(), batch_size=batch_size)
            self.log(f'{prefix}/wps_error', wps_error.detach(), batch_size=batch_size)
            # self.log(f'{prefix}/unw_wps_error', unw_wps_error.detach(), batch_size=batch_size)

            for i in range(len(wps_errors)):
                self.log(f'{prefix}/wps_error[{i}]', wps_errors[i].detach(), batch_size=batch_size)
                # if self.num_modes > 1:
                #     self.log(f'{prefix}/logit_loss', logit_loss.detach(), batch_size=batch_size)
        
        self.prev_preds = [pred_wps, gt_wps, None]

        if optimizer_idx is None:
            return wps_loss, pred_wps, features
        elif optimizer_idx == 0:
            return wps_loss, pred_wps, features
        else:
            raise NotImplementedError

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        self.n_training_step += 1
        loss, diffusion_pred, features = self.run_step(batch, prefix='train', optimizer_idx=optimizer_idx)
        if self.n_training_step % 100 == 0:
            if self.visualize:
                viz = self.get_visualizations_train(features)
            if self.use_wandb and self.visualize:
                images = viz.transpose(0, 2, 1, 3, 4).reshape((viz.shape[0] * viz.shape[2], viz.shape[1] * viz.shape[3], viz.shape[-1]))
                images = Image.fromarray(images.astype(np.uint8))
                I1 = ImageDraw.Draw(images)
                for row in range(diffusion_pred.shape[0]):
                    for i, T in enumerate(['Forward', 'Pred', 'Pred. ego', 'GT']):
                        I1.text((int(1.5 * self.im_shape[0]) + i * self.im_shape[0], 10 + row * self.im_shape[0]), T, fill=(128, 128, 128))
                self.logger.log_image(key="train/viz", images=[images])
        #  if self.use_scheduler:
            #  if self.use_wandb:
                #  self.log('train/lr', self.scheduler.get_last_lr()[0])

        return loss

    def validation_step(self, batch, batch_idx):
        wp_loss, pred_actions, features = self.run_step(batch, prefix='val')
        B = pred_actions.shape[0]
        if batch_idx == 0:
            if self.visualize:
                viz = self.get_visualizations(features)
            if self.use_wandb and self.visualize:
                images = viz.transpose(0, 2, 1, 3, 4).reshape((viz.shape[0] * viz.shape[2], viz.shape[1] * viz.shape[3], viz.shape[-1]))
                images = Image.fromarray(images.astype(np.uint8))
                I1 = ImageDraw.Draw(images)
                for row in range(B):
                    labels = ['T='+str(i) for i in range(self.n_T) if (i % (self.n_T // 5) == 0 or i == self.n_T or (self.n_T - i) < 3)]
                    labels.extend(['Pred', 'Pred. ego', 'GT'])
                    for i, T in enumerate(labels):
                        I1.text((int(1.5 * self.im_shape[0]) + i * self.im_shape[0], 10 + row * self.im_shape[0]), T, fill=(128, 128, 128))
           
                self.logger.log_image(key="val/viz", images=[images])

        return wp_loss, pred_actions
        

    def configure_optimizers(self):
        all_params = set(self.parameters())
        # act_params = set(self.inverse_dynamics_model.parameters())
        # all_params = all_params - act_params
        wd_params = set()
        for m in self.modules():
            # TODO see if you need to add any other layers to this list
            if isinstance(m, nn.Linear):
                wd_params.add(m.weight)
                #  TODO should we remove biases
                #  wd_params.add(m.bias)
            #  else:
                #  if hasattr(m, 'bias'):
                    #  wd_params.add(m.bias)
        no_wd_params = all_params - wd_params
        main_optimizer = torch.optim.AdamW(
            [{'params': list(no_wd_params), 'weight_decay': 0}, {'params': list(wd_params), 'weight_decay': self.wd}],
            lr=self.lr)
        main_scheduler = get_cosine_schedule_with_warmup(main_optimizer, self.warmup_steps, self.trainer.estimated_stepping_batches)
        main_d = {"optimizer": main_optimizer, "lr_scheduler": {"scheduler": main_scheduler, "interval": "step"}}
        # TODO wd or lr schedule for inverse dynamics model
        # act_optimizer = torch.optim.AdamW(list(act_params), lr=self.lr)
        # act_scheduler = get_cosine_schedule_with_warmup(act_optimizer, self.warmup_steps, self.trainer.estimated_stepping_batches)
        # act_d = {"optimizer": act_optimizer, "lr_scheduler": {"scheduler": act_scheduler, "interval": "step"}}
        d = [main_d] #, act_d]
        return d

    # TODO
    @torch.inference_mode()
    def get_pid_action(self, full_obs, return_features=False):
        obs = {}
        for k in full_obs:
            if k in self._dynamic_feature_names:
                obs[k] = full_obs[k][:, -self.H:]
            else:
                obs[k] = full_obs[k]
        pred_wps, logits, features = self.forward(obs, return_act=False)

        if logits is None:
            ego_waypoints = pred_wps[0, 0, :, 0, :]
        else:
            ego_logits = logits[0, :, 0]
            ego_labels = ego_logits.argmax()

            ego_wp_diffs = pred_wps[0, :, :, 0, :]
            ego_waypoints = ego_wp_diffs[ego_labels]
        ego_speed = obs['vehicle_features'][0, -1, 0, 3]

        actions = self.control_pid(ego_waypoints, ego_speed)

        self.prev_preds = [pred_wps, actions, logits]
        if return_features:
            return actions, features
        else:
            return actions

    def control_pid(self, waypoints, curr_speed):
        wps = waypoints[:, :2].data.cpu().numpy() * self.max_token_distance
        speed = curr_speed.data.cpu().numpy() * self.max_token_distance

        #  desired_speed = 2. * np.linalg.norm(wps[1] - wps[0], axis=-1)
        #  desired_speed = np.linalg.norm(wps[1] + wps[0], axis=-1)
        #  desired_speed = np.linalg.norm(wps[1], axis=-1)
        #  desired_speed = 2. * np.linalg.norm(wps[0], axis=-1)
        #  desired_speed = 4. * np.linalg.norm(wps[0], axis=-1) - speed
        desired_speed = speed + waypoints[0, 3].data.cpu().numpy() * self.max_token_distance
        #  desired_speed = speed + waypoints[:2, 3].sum().data.cpu().numpy() * self.max_token_distance

        brake = desired_speed < 0.4 or ((speed / desired_speed) > 1.1)
        #  brake = desired_speed < 0.3 or ((speed / desired_speed) > 1.05)

        delta = np.clip(desired_speed - speed, 0.0, 0.25)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.75)
        if brake:
            gas = -1.0
        else:
            gas = throttle

        #  aim = (wps[1] + wps[0]) * 0.5
        #  aim = 0.5 * wps[1]
        #  aim = wps[0]
        aim = 0.5 * wps[0]

        angle = np.arctan2(aim[1], aim[0]) * 2. / np.pi
        if brake:
            angle = 0.0
        steer = self.turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)

        return steer, gas

    # TODO might be better in visualizing or just everywhere to have num agents before time
    @torch.inference_mode()
    def get_visualizations_train(self, features=None, preds=None, obs=None):
        if preds is None:
            preds = self.prev_preds
        scene_image = self.get_visualizations_inner(features=features, obs=obs)

        num_samples = scene_image.shape[0]
        pred_image = np.zeros((num_samples, 4, self.im_shape[0], self.im_shape[1], 3), dtype=np.uint8)

        if features is not None:
            agents_features, agents_masks, light_features, light_masks, stop_features, stop_masks, walker_features, walker_masks, route_features, route_masks, map_features, map_masks = features
        else:
            assert(obs is not None)
            agents_features, agents_masks, light_features, light_masks, stop_features, stop_masks, walker_features, walker_masks, route_features, route_masks, map_features, map_masks = self.process_observations(obs)

        pred_wps = preds[0][..., :2].detach().cpu()
        gt_wps = preds[1][..., :2].detach().cpu()
        corrupted = self.pred_wps_corrupted[..., :2].detach().cpu()
        B = pred_wps.shape[0]
        num_agents = gt_wps.shape[-2]
        
        vehicles = agents_features.detach().cpu()[:, -1][:, :num_agents]
        vehicles_masks = ~agents_masks.detach().cpu()[:, -1][:, :num_agents]

        conversion = np.array([-1., 1.])

        if preds[2] is None:
            labels = torch.zeros((vehicles.shape[0], vehicles.shape[1]), dtype=torch.int64, device=vehicles.device)
            order = torch.arange(self.num_modes, device=vehicles.device)[None, :, None].repeat(vehicles.shape[0], 1, vehicles.shape[1])
        else:
            logits = preds[2].detach().cpu()
            labels = logits.argmax(dim=1)
            order = logits.argsort(dim=1)

        transformed_wps = transform_points(
            pred_wps.reshape((B, self.num_modes * self.T, num_agents, 2)).transpose(1, 2),
            vehicles[..., :3],
            invert=True).transpose(1, 2).reshape((B, self.num_modes, self.T, num_agents, 2))

        transformed_gt_wps = transform_points(
            gt_wps.reshape((B, self.T, num_agents, 2)).transpose(1, 2),
            vehicles[..., :3],
            invert=True).transpose(1, 2).reshape((B, 1, self.T, num_agents, 2))

        transformed_corrupted = transform_points(
            corrupted.reshape((B, self.T, num_agents, 2)).transpose(1, 2),
            vehicles[..., :3],
            invert=True).transpose(1, 2).reshape((B, 1, self.T, num_agents, 2))

        shaped_order = order.unsqueeze(2).repeat_interleave(self.T, 2).permute(0, 2, 3, 1).reshape((-1, self.num_modes))
        shaped_transformed_wps = transformed_wps.permute(0, 2, 3, 1, 4).reshape((-1, self.num_modes, 2))
        ordered_shaped_transformed_wps = shaped_transformed_wps[torch.arange(shaped_transformed_wps.shape[0], device=shaped_transformed_wps.device).unsqueeze(1), shaped_order]
        ordered_transformed_wps = ordered_shaped_transformed_wps.reshape((B, self.T, num_agents, self.num_modes, 2)).permute(0, 2, 1, 3, 4)

        shaped_transformed_gt_wps = transformed_gt_wps.permute(0, 2, 3, 1, 4).reshape((-1, 1, 2))
        # ordered_shaped_transformed_gt_wps = shaped_transformed_gt_wps[torch.arange(shaped_transformed_gt_wps.shape[0], device=shaped_transformed_gt_wps.device).unsqueeze(1), shaped_order]
        ordered_transformed_gt_wps = shaped_transformed_gt_wps.reshape((B, self.T, num_agents, 1, 2)).permute(0, 2, 1, 3, 4)

        shaped_transformed_corrupted = transformed_corrupted.permute(0, 2, 3, 1, 4).reshape((-1, 1, 2))
        # ordered_shaped_transformed_corrupted = shaped_transformed_corrupted[torch.arange(shaped_transformed_corrupted.shape[0], device=shaped_transformed_corrupted.device).unsqueeze(1), shaped_order]
        ordered_transformed_corrupted = shaped_transformed_corrupted.reshape((B, self.T, num_agents, 1, 2)).permute(0, 2, 1, 3, 4)

        for row_idx in range(num_samples):
            masked_wps = ordered_transformed_wps[row_idx][vehicles_masks[row_idx]].numpy()[:, :, 0] # .reshape((-1, self.num_modes, 2))
            wp_locs = np.clip((0.5 * scene_image.shape[-2] * (conversion * masked_wps + 1.)).astype(np.int32), 0, scene_image.shape[-2] - 1)
            mode_colors = viz_utils.get_mode_colors(self.num_modes)[None, :, :3].repeat(wp_locs.shape[0], 0)   
            pred_image[row_idx, 1] = viz_utils.get_image_agent(wp_locs, mode_colors.reshape((-1, 3)), im_shape=self.im_shape)

            masked_gt_wps = ordered_transformed_gt_wps[row_idx][vehicles_masks[row_idx]].numpy() # .reshape((-1, self.num_modes, 2))
            gt_wp_locs = np.clip((0.5 * scene_image.shape[-2] * (conversion * masked_gt_wps + 1.)).astype(np.int32), 0, scene_image.shape[-2] - 1)
            pred_image[row_idx, 3] = viz_utils.get_image_agent(gt_wp_locs, mode_colors.reshape((-1, 3)), im_shape=self.im_shape)

            ego_masked_wps = ordered_transformed_wps[row_idx][0, :, 0].numpy() # .reshape((-1, self.num_modes, 2))
            ego_wp_locs = np.clip((0.5 * scene_image.shape[-2] * (conversion * ego_masked_wps + 1.)).astype(np.int32), 0, scene_image.shape[-2] - 1)
            ego_mode_colors = viz_utils.get_mode_colors(self.num_modes)[None, :, :3].repeat(ego_wp_locs.shape[0], 0)
            pred_image[row_idx, 2] = viz_utils.get_image_agent(ego_wp_locs[None], ego_mode_colors.reshape((-1, 3)), im_shape=self.im_shape)

            masked_corrupted = ordered_transformed_corrupted[row_idx][vehicles_masks[row_idx]].numpy() # .reshape((-1, self.num_modes, 2))
            corrupted_locs = np.clip((0.5 * scene_image.shape[-2] * (conversion * masked_corrupted + 1.)).astype(np.int32), 0, scene_image.shape[-2] - 1)
            pred_image[row_idx, 0] = viz_utils.get_image_agent(corrupted_locs, mode_colors.reshape((-1, 3)), im_shape=self.im_shape)

        compact_scene_image = np.stack([
            scene_image[:, 0],
            # scene_image[:, 1:1+self.num_enc_layers+self.num_dec_layers].mean(axis=1),
            # scene_image[:, 1+self.num_enc_layers+self.num_dec_layers:1+2*self.num_enc_layers+2*self.num_dec_layers].mean(axis=1),
        ], axis=1)
        return np.concatenate([compact_scene_image, pred_image], axis=1)

    @torch.inference_mode()
    def get_visualizations(self, features=None, preds=None, obs=None):
        if preds is None:
            preds = self.prev_preds
        scene_image = self.get_visualizations_inner(features=features, obs=obs)

        num_samples = scene_image.shape[0]

        if features is not None:
            agents_features, agents_masks, light_features, light_masks, stop_features, stop_masks, walker_features, walker_masks, route_features, route_masks, map_features, map_masks = features
        else:
            assert(obs is not None)
            agents_features, agents_masks, light_features, light_masks, stop_features, stop_masks, walker_features, walker_masks, route_features, route_masks, map_features, map_masks = self.process_observations(obs)

        pred_wps = preds[0][..., :2].detach().cpu()
        gt_wps = preds[1][..., :2].detach().cpu()
        B = pred_wps.shape[0]
        num_agents = gt_wps.shape[-2]
        traces = [t[..., :2].cpu() for t in self.traces]

        vehicles = agents_features.detach().cpu()[:, -1][:, :num_agents]
        vehicles_masks = ~agents_masks.detach().cpu()[:, -1][:, :num_agents]
        conversion = np.array([-1., 1.])
            
        pred_image = np.zeros((num_samples, len(self.traces) + 3, self.im_shape[0], self.im_shape[1], 3), dtype=np.uint8)
 
    
        if preds[2] is None:
            labels = torch.zeros((vehicles.shape[0], vehicles.shape[1]), dtype=torch.int64, device=vehicles.device)
            order = torch.arange(self.num_modes, device=vehicles.device)[None, :, None].repeat(vehicles.shape[0], 1, vehicles.shape[1])
        else:
            logits = preds[2].detach().cpu()
            labels = logits.argmax(dim=1)
            order = logits.argsort(dim=1)

        transformed_wps = transform_points(
            pred_wps.reshape((B, self.num_modes * self.T, num_agents, 2)).transpose(1, 2),
            vehicles[..., :3],
            invert=True).transpose(1, 2).reshape((B, self.num_modes, self.T, num_agents, 2))

        transformed_gt_wps = transform_points(
            gt_wps.reshape((B, self.num_modes * self.T, num_agents, 2)).transpose(1, 2),
            vehicles[..., :3],
            invert=True).transpose(1, 2).reshape((B, self.num_modes, self.T, num_agents, 2))
        shaped_order = order.unsqueeze(2).repeat_interleave(self.T, 2).permute(0, 2, 3, 1).reshape((-1, self.num_modes))
        shaped_transformed_wps = transformed_wps.permute(0, 2, 3, 1, 4).reshape((-1, self.num_modes, 2))
        ordered_shaped_transformed_wps = shaped_transformed_wps[torch.arange(shaped_transformed_wps.shape[0], device=shaped_transformed_wps.device).unsqueeze(1), shaped_order]
        ordered_transformed_wps = ordered_shaped_transformed_wps.reshape((B, self.T, num_agents, self.num_modes, 2)).permute(0, 2, 1, 3, 4)

        shaped_transformed_gt_wps = transformed_gt_wps.permute(0, 2, 3, 1, 4).reshape((-1, self.num_modes, 2))
        ordered_shaped_transformed_gt_wps = shaped_transformed_gt_wps[torch.arange(shaped_transformed_gt_wps.shape[0], device=shaped_transformed_gt_wps.device).unsqueeze(1), shaped_order]
        ordered_transformed_gt_wps = ordered_shaped_transformed_gt_wps.reshape((B, self.T, num_agents, self.num_modes, 2)).permute(0, 2, 1, 3, 4)

        ordered_transformed_traces = []
        traceorder = torch.arange(self.predictions_per_sample, device=vehicles.device)[None, :, None].repeat(vehicles.shape[0], 1, vehicles.shape[1])
        shaped_trace_order = traceorder.unsqueeze(2).repeat_interleave(self.T, 2).permute(0, 2, 3, 1).reshape((-1, self.predictions_per_sample))
        for trace in traces:
            transformed_trace = transform_points(
            trace.reshape((B, self.predictions_per_sample * self.T, -1, 2)).transpose(1, 2),
                vehicles[..., :3],
                invert=True).transpose(1, 2).reshape((B, self.predictions_per_sample, self.T, -1, 2))

            shaped_transformed_trace = transformed_trace.permute(0, 2, 3, 1, 4).reshape((-1, self.predictions_per_sample, 2))
           
            ordered_shaped_transformed_trace = shaped_transformed_trace[torch.arange(shaped_transformed_trace.shape[0], device=shaped_transformed_trace.device).unsqueeze(1), shaped_trace_order]
            ordered_transformed_trace = ordered_shaped_transformed_trace.reshape((B, self.T, num_agents, self.predictions_per_sample, 2)).permute(0, 2, 1, 3, 4)

            ordered_transformed_traces.append(ordered_transformed_trace)

        for row_idx in range(num_samples):
            masked_wps = ordered_transformed_wps[row_idx][vehicles_masks[row_idx]].numpy() # .reshape((-1, self.num_modes, 2))
            wp_locs = np.clip((0.5 * scene_image.shape[-2] * (conversion * masked_wps + 1.)).astype(np.int32), 0, scene_image.shape[-2] - 1)
            mode_colors = viz_utils.get_mode_colors(self.num_modes)[None, :, :3].repeat(wp_locs.shape[0], 0)
            # pred_image[row_idx, 1] = viz_utils.get_image_agent(wp_locs, mode_colors.reshape((-1, 3)), im_shape=self.im_shape)
            pred_image[row_idx, len(self.traces) + 0] = viz_utils.get_image_agent(wp_locs, mode_colors.reshape((-1, 3)), im_shape=self.im_shape)

            masked_gt_wps = ordered_transformed_gt_wps[row_idx][vehicles_masks[row_idx]].numpy() # .reshape((-1, self.num_modes, 2))
            gt_wp_locs = np.clip((0.5 * scene_image.shape[-2] * (conversion * masked_gt_wps + 1.)).astype(np.int32), 0, scene_image.shape[-2] - 1)
            # pred_image[row_idx, 3] = viz_utils.get_image_agent(gt_wp_locs, mode_colors.reshape((-1, 3)), im_shape=self.im_shape)
            pred_image[row_idx, len(self.traces) + 2] = viz_utils.get_image_agent(gt_wp_locs, mode_colors.reshape((-1, 3)), im_shape=self.im_shape)

            ego_masked_wps = ordered_transformed_wps[row_idx][0].numpy() # .reshape((-1, self.num_modes, 2))
            ego_wp_locs = np.clip((0.5 * scene_image.shape[-2] * (conversion * ego_masked_wps + 1.)).astype(np.int32), 0, scene_image.shape[-2] - 1)
            ego_mode_colors = viz_utils.get_mode_colors(self.num_modes)[None, :, :3].repeat(ego_wp_locs.shape[0], 0)
            # pred_image[row_idx, 2] = viz_utils.get_image_agent(ego_wp_locs[None], ego_mode_colors.reshape((-1, 3)), im_shape=self.im_shape)
            pred_image[row_idx, len(self.traces) + 1] = viz_utils.get_image_agent(ego_wp_locs[None], ego_mode_colors.reshape((-1, 3)), im_shape=self.im_shape)

            for i, trace in enumerate(ordered_transformed_traces):
                prediction_colors = viz_utils.get_mode_colors(self.predictions_per_sample)[None, :, :3].repeat(wp_locs.shape[0], 0)
                masked_trace = trace[row_idx][vehicles_masks[row_idx]].permute(0, 2, 1, 3).numpy() # .reshape((-1, self.predictions_per_sample, 2))
                trace_locs = np.clip((0.5 * scene_image.shape[-2] * (conversion * masked_trace + 1.)).astype(np.int32), 0, scene_image.shape[-2] - 1)
        
                pred_image[row_idx, i] = viz_utils.get_image_mode(trace_locs, prediction_colors, im_shape=self.im_shape)


        compact_scene_image = np.stack([
            scene_image[:, 0],
            # scene_image[:, 1:1+self.num_enc_layers+self.num_dec_layers].mean(axis=1),
            # scene_image[:, 1+self.num_enc_layers+self.num_dec_layers:1+2*self.num_enc_layers+2*self.num_dec_layers].mean(axis=1),
        ], axis=1)
        return np.concatenate([compact_scene_image, pred_image], axis=1)
    
    @torch.inference_mode()
    def get_visualizations_inner(self, features=None, obs=None):
        if features is not None:
            agents_features, agents_masks, light_features, light_masks, stop_features, stop_masks, walker_features, walker_masks, route_features, route_masks, map_features, map_masks = features
        else:
            assert(obs is not None)
            agents_features, agents_masks, light_features, light_masks, stop_features, stop_masks, walker_features, walker_masks, route_features, route_masks, map_features, map_masks = self.process_observations(obs)

        vehicles = agents_features.detach().cpu().numpy()[:, -1]
        vehicles_masks = ~agents_masks.detach().cpu().numpy()[:, -1]
        lights = light_features.detach().cpu().numpy()[:, :, :2]
        light_masks = ~light_masks.detach().cpu().numpy()
        stops = stop_features.detach().cpu().numpy()[:, :, :2]
        stop_masks = ~stop_masks.detach().cpu().numpy()
        routes = route_features.detach().cpu().numpy()[:, :, :2]
        route_masks = ~route_masks.detach().cpu().numpy()
        roads = map_features.detach().cpu().numpy()[:, :, :2]
        road_masks = ~map_masks.detach().cpu().numpy()
        maps = np.concatenate([roads, lights, stops, routes], axis=1)
        map_masks = np.concatenate([road_masks, light_masks, stop_masks, route_masks], axis=1)
        #  route_masks = obs['route_masks'].detach().cpu().numpy()
        #  maps = np.concatenate([roads, lights, stops, routes], axis=1)
        #  map_masks = np.concatenate([road_masks, light_masks, stop_masks, route_masks], axis=1)
        num_samples = vehicles.shape[0]
        num_figs = 1 # + 2 * self.num_enc_layers + 2 * self.num_dec_layers

        scene_image = np.zeros((num_samples, num_figs, self.im_shape[0], self.im_shape[1], 3), dtype=np.uint8)
        conversion = np.array([-1., 1.])
        veh_im_locs = np.clip((0.5 * scene_image.shape[-2] * (conversion * vehicles[..., :2] + 1.)).astype(np.int32), 0, scene_image.shape[-2] - 1)
        map_im_locs = np.clip((0.5 * scene_image.shape[-2] * (conversion * maps[..., :2] + 1.)).astype(np.int32), 0, scene_image.shape[-2] - 1)

        # TODO make this even more efficient by doing it over all row_idx at once
        for row_idx in range(num_samples):
            # scene graph
            scene_image[row_idx, 0] = self._get_obs_image(
                map_im_locs,
                veh_im_locs,
                row_idx,
                map_masks=map_masks,
                veh_masks=vehicles_masks,
                im_shape=self.im_shape)

            # for i in range(self.num_enc_layers):
            #     # map atten
            #     scene_image[row_idx, 1 + i] = self._get_attn_image(
            #         f"map_cross_layers.{i}.multihead_attn",
            #         map_im_locs[:, :-routes.shape[1]],
            #         row_idx,
            #         base_color='red',
            #         mask=map_masks[:, :-routes.shape[1]],
            #         im_shape=self.im_shape)

            #     # agent atten
            #     scene_image[row_idx, 1 + self.num_enc_layers + self.num_dec_layers + i] = self._get_attn_image(
            #         f"social_attn_layers.{i}.self_attn",
            #         veh_im_locs,
            #         row_idx,
            #         base_color='green',
            #         mask=vehicles_masks,
            #         im_shape=self.im_shape)

            # for i in range(self.num_dec_layers):
            #     # map atten decoder
            #     scene_image[row_idx, 1 + self.num_enc_layers + i] = self._get_attn_image(
            #         f"map_dec_layers.{i}.multihead_attn",
            #         map_im_locs,
            #         row_idx,
            #         base_color='red',
            #         mask=map_masks,
            #         im_shape=self.im_shape)

            #     # agent atten decoder
            #     scene_image[row_idx, 1 + 2 * self.num_enc_layers + self.num_dec_layers + i] = self._get_attn_image(
            #         f"tx_decoder.{i}.multihead_attn",
            #         veh_im_locs,
            #         row_idx,
            #         base_color='green',
            #         mask=vehicles_masks,
            #         im_shape=self.im_shape)

        return scene_image
    
    def _get_obs_image(
            self,
            map_locs,
            veh_locs,
            idx,
            map_color='red',
            veh_color='green',
            map_masks=None,
            veh_masks=None,
            im_shape=(50, 50),
    ):
        if map_masks is None:
            map_loc = map_locs[idx]
        else:
            map_loc = map_locs[idx][map_masks[idx]]
        if veh_masks is None:
            veh_loc = veh_locs[idx]
        else:
            veh_loc = veh_locs[idx][veh_masks[idx]]

        map_c = viz_utils.get_color(map_color)
        veh_c = viz_utils.get_color(veh_color)

        scene_image = viz_utils.get_image(
            np.concatenate([map_loc, veh_loc], axis=0),
            np.concatenate([map_c[None].repeat(map_loc.shape[0], 0), veh_c[None].repeat(veh_loc.shape[0], 0)], axis=0),
            im_shape=im_shape)
        return scene_image
    
    @torch.inference_mode()
    def get_logits_attention(self, full_obs):
        obs = {}
        for k in full_obs:
            if k in self._dynamic_feature_names:
                obs[k] = full_obs[k][:, -self.H:]
            else:
                obs[k] = full_obs[k]
        agents_emb, agents_masks, map_emb, map_masks, route_emb, route_masks, features = self.get_encoding(obs)
        prob_seq = self.get_prob_decoding(agents_emb, agents_masks, map_emb, map_masks, route_emb, route_masks, features)
        logits = self.get_prob_output(prob_seq)

        B = logits.shape[0]
        num_agents = agents_masks.shape[-1]

        attns = []
        for i in range(self.num_dec_layers):
            attn_name = f"prob_tx_decoder.{i}.multihead_attn"
            attn = self.attn_weights[attn_name].view((B, num_agents, self.num_modes, -1))[0, 0].mean(dim=0).detach()

            #  attn = attn / attn.max()
            attns.append(attn)

        return logits, attns
    
    @torch.inference_mode()
    def get_wandb_info(self, features=None, preds=None, obs=None):
        return {}