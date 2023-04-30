import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np


from src.models.transformer_layers import TransformerEncoderLayer, TransformerDecoderLayer
#  from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer
from src.models.utils import nll_pytorch_dist, get_cosine_schedule_with_warmup, TimeEncoding
from src.models.embedders import Embedder
from src.models.map_models import MapEncoder

from src.models.output_models import ResOutputModel
from src.models import viz_utils
from src.envs.carla.features.scenarios import TOWNS
from src.envs.carla.features.carla_map_features import CarlaMapFeatures
from src.models.diffusion.diffusion_transformer import DiffusionTransformer


class ImitationModel(pl.LightningModule):
    '''
    AutoBot-Ego Class.
    '''
    def __init__(
            self,
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
            im_shape=(128, 128),
            # TODO map kwargs
            carla_maps_path='maps/',
            max_token_distance=30.,
            max_z_distance=7.5,
            **kwargs,
        ):
        super().__init__()
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

        self.max_token_distance = max_token_distance
        self.carla_maps_dict = {}
        for town in TOWNS:
            self.carla_maps_dict[town] = CarlaMapFeatures(town, map_data_path=carla_maps_path, torch_device=self.device, max_token_distance=max_token_distance, max_z_distance=max_z_distance, max_map_pts=self.num_map_pts)

        self._dynamic_feature_names = ['vehicle_features', 'vehicle_masks', 'light_features', 'light_masks', 'walker_features', 'walker_masks']

        self.create_agent_embedder()
        self.create_map_encoder()
        self.create_route_encoder()
        self.agent_emb_ln = nn.LayerNorm(self.emb_dim)
        self.create_prob_model()
        self.diffusion = DiffusionTransformer(carla_maps_path=carla_maps_path, T=T)

        self.apply(self._init_weights)
        self.diffusion.quantize = self.diffusion.quantize.load_from_checkpoint('./pretrain/vqvae.ckpt', strict=False)
        
        if self.visualize:
            self.attn_weights = {}
            for name, module in self.named_modules():
                if name[-4:] == 'attn':
                    module.name = name
                    def hook(module, input, output):
                        self.attn_weights[module.name] = output[1].detach()
                    module.register_forward_hook(hook)

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
        #  self.route_encoder = nn.Sequential(
            #  nn.Linear(self.route_dim, self.emb_dim),
            #  nn.LayerNorm(self.emb_dim) if self.norm_first else nn.ReLU())
        #  self.route_encoder = Embedder(self.route_dim, self.emb_dim, expand_theta=True, layer_norm=False)
        self.route_encoder = Embedder(self.route_dim, self.emb_dim, expand_theta=True, layer_norm=True)


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

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.elementwise_affine == True:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

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

    # TODO handle time correctly
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

    def get_encoding(self, obs):
        #  agents_features, agents_masks, map_features, map_masks, light_features, light_masks, stop_features, stop_masks, walker_features, walker_masks, route_features, route_masks = self.process_observations(obs)
        features = self.process_observations(obs)
        agents_features, agents_masks, light_features, light_masks, stop_features, stop_masks, walker_features, walker_masks, route_features, route_masks, map_features, map_masks = features
        agents_emb = self.agents_embedder(agents_features)
        agents_emb = self.agent_emb_ln(agents_emb)

        map_emb, map_masks = self.map_encoder(
            map_features,
            map_masks,
            light_features=light_features,
            light_masks=light_masks,
            stop_features=stop_features,
            stop_masks=stop_masks,
        )

        route_emb = self.route_encoder(route_features)

        return agents_emb, agents_masks, map_emb, map_masks, route_emb, route_masks, features


    def forward(self, obs, gt_wps, wps_mask, modes=None):
        agents_emb, agents_masks, map_emb, map_masks, route_emb, route_masks, features = self.get_encoding(obs)

        diffusion_out = self.diffusion(gt_wps, wps_mask, agents_emb, agents_masks, map_emb, map_masks, route_emb, route_masks)

        if self.p_dec:
            prob_seq = self.get_prob_decoding(agents_emb, agents_masks, map_emb, map_masks, route_emb, route_masks, features)
            logits = self.get_prob_output(prob_seq)
        else:
            logits = None

        return diffusion_out['loss'], diffusion_out['pred_index'], logits, features

    @torch.inference_mode()
    def sample(self, obs, modes=None):
        agents_emb, agents_masks, map_emb, map_masks, route_emb, route_masks, features = self.get_encoding(obs)
        diffusion_out = self.diffusion.sample_fast(agents_emb, agents_masks, map_emb, map_masks, route_emb, route_masks)

        if self.p_dec:
            prob_seq = self.get_prob_decoding(agents_emb, agents_masks, map_emb, map_masks, route_emb, route_masks, features)
            logits = self.get_prob_output(prob_seq)
        else:
            logits = None

        return torch.tensor(0), diffusion_out['content_token'], logits, features

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
    def get_action(self, full_obs, return_features=False):
        obs = {}
        for k in full_obs:
            if k in self._dynamic_feature_names:
                obs[k] = full_obs[k][:, -self.H:]
            else:
                obs[k] = full_obs[k]
        pred, logits, features = self.forward(obs)
        if logits is None:
            modes = pred[:, 0:1, 0, :self.act_dim]
        else:
            modes = logits.argmax(dim=-1)
            actions = pred[torch.arange(modes.shape[0], device=modes.device), modes, 0, :self.act_dim]
        if return_features:
            return actions, features
        else:
            return actions


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

    def _compute_regression_loss(self, pred, gt):
        #  regression_loss = nll_pytorch_dist(pred, gt, dist='normal')
        regression_loss = nll_pytorch_dist(pred, gt, dist='laplace')
        mse = F.mse_loss(pred[..., :gt.shape[-1]], gt, reduction='none')
        return regression_loss, mse

    def _compute_loss(self, pred_actions, gt_actions, logits, actions_mask=None):
        B = pred_actions.shape[0]
        shaped_pred_actions = pred_actions.reshape((B * self.num_modes, self.T, 2 * self.act_dim))
        shaped_gt_actions = gt_actions.unsqueeze(1).repeat_interleave(self.num_modes, 1).reshape((B * self.num_modes, self.T, self.act_dim))
        shaped_regression_loss, shaped_mse_errors = self._compute_regression_loss(shaped_pred_actions, shaped_gt_actions)

        if actions_mask is None:
            actions_mask = torch.ones(gt_actions.shape[:-1], dtype=bool, device=gt_actions.device)

        if self.num_modes > 1:
            time_regression_loss = shaped_regression_loss.reshape((B, self.num_modes, self.T))
            time_mse_errors = shaped_mse_errors.reshape((B, self.num_modes, self.T, self.act_dim))

            a_mask = actions_mask.unsqueeze(1)
            regression_loss = (time_regression_loss * a_mask).sum(dim=2) / a_mask.sum(dim=2)
            mse_errors = (time_mse_errors * a_mask.unsqueeze(-1)).sum(dim=2) / a_mask.unsqueeze(-1).sum(dim=2)

            mse_error = mse_errors.sum(dim=-1)

            actions_m = actions_mask.any(dim=-1)

            unmasked_loss, unmasked_fit_loss, unmasked_logit_loss, labels, _ = self._compute_modes_loss(logits, mse_error, regression_loss)
            loss = unmasked_loss[actions_m].mean()
            #  fit_loss = unmasked_fit_loss[actions_m].mean()
            logit_loss = unmasked_logit_loss[actions_m].mean()

            regression_loss = regression_loss[torch.arange(labels.shape[0], device=labels.device), labels][actions_m].mean()
            mse_errors = mse_errors[torch.arange(labels.shape[0], device=labels.device), labels][actions_m].mean(dim=0)
        else:
            regression_loss = shaped_regression_loss[actions_mask].mean()
            mse_errors = shaped_mse_errors[actions_mask].mean(dim=0)

            loss = regression_loss
            logit_loss = 0.

        return loss, logit_loss, mse_errors
    

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
    def get_visualizations(self, features=None, obs=None):
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

        # TODO what is the right attentions to look at for roads
        # TODO we might want to do atten within map and with map

        #  num_samples = min(vehicles.shape[0], 100)
        num_samples = vehicles.shape[0]
        num_figs = 1 + 2 * self.num_enc_layers + 2 * self.num_dec_layers

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

    @torch.inference_mode()
    def get_wandb_info(self, features=None, preds=None, obs=None):
        return {}