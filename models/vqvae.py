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
#  from src.models.output_models import OutputModel
from src.models.output_models import ResOutputModel
from src.models import viz_utils
from src.envs.carla.features.scenarios import TOWNS, transform_points
from src.envs.carla.features.carla_map_features import CarlaMapFeatures
# import seaborn as sns


class VQVAE(pl.LightningModule):
    def __init__(
            self,
            agent_dim=7,
            act_dim=2,
            map_dim=4,
            light_dim=4,
            stop_dim=3,
            walker_dim=6,
            wp_dim=4,
            dt=0.05,
            out_init_std=0.05,
            num_agents=20,
            num_map_pts=100,
            num_route_pts=20,
            num_modes=1,
            T=1,
            H=1,
            f=1,
            num_embeddings=32,
            emb_dim=256,
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
            commitment_cost=2, 
            decay=0.0,
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
        self.wp_dim = wp_dim
        self.dt = dt
        self.out_init_std = out_init_std
        self.num_agents = num_agents
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
            # TODO device
            self.carla_maps_dict[town] = CarlaMapFeatures(town, map_data_path=carla_maps_path, torch_device=self.device, max_token_distance=max_token_distance, max_z_distance=max_z_distance, max_map_pts=self.num_map_pts)

        self._dynamic_feature_names = ['vehicle_features', 'vehicle_masks', 'light_features', 'light_masks', 'walker_features', 'walker_masks']
        
        self.traj_embedders = []
        for _ in range(self.num_agents):
            embedder = Embedder(self.T * self.wp_dim, self.emb_dim, expand_theta=True, layer_norm=False)
            self.traj_embedders.append(embedder)
        self.traj_embedders = nn.ModuleList(self.traj_embedders)

        self.agent_emb_ln = nn.LayerNorm(self.emb_dim)
        self.create_decoder()
        self.create_output_model()
        self.create_prob_model()

        if decay > 0.0:
            self.vq_vae = VectorQuantizerEMA(num_agents, num_embeddings, emb_dim, 
                                              commitment_cost, decay)
        else:
            self.vq_vae = VectorQuantizer(num_agents, num_embeddings, emb_dim,
                                           commitment_cost)

        self.apply(self._init_weights)

        if self.visualize:
            self.attn_weights = {}
            for name, module in self.named_modules():
                if name[-4:] == 'attn':
                    module.name = name
                    def hook(module, input, output):
                        self.attn_weights[module.name] = output[1].detach()
                    module.register_forward_hook(hook)
        

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

    def create_output_model(self):
        # TODO add LN to outputModel
        # TODO add dropout
        self.output_model = ResOutputModel(
            emb_dim=self.emb_dim,
            dist_dim=self.wp_dim,
            min_std=self.min_std,
            layer_norm=self.norm_first,
            dropout=self.dropout,
            out_std=self.out_init_std * self.dt * self.f * torch.ones((self.T, self.wp_dim)),
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

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
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

    def get_encoding(self, gt_wps, wps_mask):
        gt_wps = gt_wps.permute(0, 2, 3, 1).unsqueeze(1).flatten(start_dim=3)

        traj_emb = torch.cat([self.traj_embedders[i](gt_wps[:, :, i]).unsqueeze(2) for i in range(self.num_agents)], dim=2)
        traj_emb = self.agent_emb_ln(traj_emb)

        return traj_emb

    def get_decoding(self, agents_emb, agents_masks, modes=None):
        if modes is None:
            num_modes = self.num_modes
        else:
            num_modes = 1
        B = agents_emb.shape[0]

        out_masks = agents_masks.reshape((B, 1, -1, self.num_agents))
        out_masks = out_masks[:, :, -1:, :].repeat(1, num_modes, self.T, 1)

        if self.q_dec:
            out_seq = agents_emb.reshape((B, 1, -1, self.num_agents, self.emb_dim))

            out_seq = out_seq[:, :, -1:, :, :].repeat(1, num_modes, self.T, 1, 1)
            out_seq = out_seq + self.Q.repeat(B, 1, 1, self.num_agents, 1)
            out_seq = self.dec_time_pe(out_seq.transpose(2, 3).reshape((-1, self.T, self.emb_dim))).reshape((B, self.num_modes, self.num_agents, self.T, self.emb_dim)).transpose(2, 3)
        else:
            out_seq = agents_emb.reshape((B, 1, -1, self.num_agents, self.emb_dim))
            out_seq = out_seq[:, :, -1:, :, :].repeat(1, num_modes, self.T, 1, 1)

        out_seq = out_seq.reshape((B, num_modes * self.T, self.num_agents, -1))
        out_masks = out_masks.reshape((B, num_modes * self.T, self.num_agents))

        for d in range(self.num_dec_layers):
            out_seq = self.dec_fn(
                out_seq,
                out_masks.clone(),
                agents_emb,
                agents_masks.clone(),
                layer=self.tx_decoder[d])

        out_seq = out_seq.view((B, num_modes, self.T, self.num_agents, -1))
        return out_seq

    def get_prob_decoding(self, agents_emb, agents_masks):
        B = agents_emb.shape[0]

        # TODO maybe detach
        prob_seq = agents_emb.reshape((B, 1, -1, self.num_agents, self.emb_dim))
        prob_seq = prob_seq[:, :, -1, 0:1, :].repeat(1, self.num_modes, 1, 1)

        prob_masks = agents_masks.reshape((B, 1, -1, self.num_agents))
        prob_masks = prob_masks[:, :, -1, 0:1].repeat(1, self.num_modes, 1)

        prob_seq = prob_seq + self.P.repeat(B, 1, 1, 1)

        prob_seq = prob_seq.view((B, self.num_modes, 1, -1))
        prob_masks = prob_masks.view((B, self.num_modes, 1))

        for d in range(self.num_dec_layers):
            prob_seq = self.dec_fn(
                prob_seq,
                prob_masks.clone(),
                agents_emb,
                agents_masks.clone(),
                layer=self.prob_tx_decoder[d])

        prob_seq = prob_seq.view((B, self.num_modes, 1, -1))
        return prob_seq

    def get_output(self, out_seq):
        B = out_seq.shape[0]
        num_modes = out_seq.shape[1]
        #  outputs = self.output_model(out_seq).reshape((B, num_modes, self.T, self.num_agents, -1))
        outputs = self.output_model(out_seq.permute(0, 3, 1, 2, 4).reshape((B * self.num_agents, num_modes, self.T, self.emb_dim)))
        outputs = outputs.reshape((B, self.num_agents, num_modes, self.T, -1)).permute(0, 2, 3, 1, 4)
        return outputs

    def get_prob_output(self, prob_seq):
        B = prob_seq.shape[0]
        num_modes = prob_seq.shape[1]
        logits = self.prob_model(prob_seq).reshape((B, num_modes))
        return logits

    def get_pred_next_wps(self, obs, outputs):
        output_thetas = torch.cumsum(outputs[..., 2], dim=2)
        output_others = torch.cumsum(outputs[..., 3:self.wp_dim], dim=2)

        thetas = output_thetas - outputs[..., 2]

        rot_matrix = torch.stack([
            torch.stack([ torch.cos(thetas), torch.sin(thetas)], dim=-1),
            torch.stack([-torch.sin(thetas), torch.cos(thetas)], dim=-1)
        ], dim=-2)

        output_pos_diffs = torch.matmul(outputs[..., None, :2], rot_matrix).squeeze(-2)
        output_pos = torch.cumsum(output_pos_diffs, dim=2)

        outputs_mean = torch.cat([output_pos, output_thetas.unsqueeze(-1), output_others], dim=-1)

        outputs_std = torch.sqrt(torch.cumsum(outputs[..., self.wp_dim:] ** 2, dim=2))
        pred_next_wps = torch.cat([outputs_mean, outputs_std], dim=-1)
        return pred_next_wps

    def forward(self, gt_wps, wps_mask, prefix, modes=None, return_wp=True):
        traj_emb = self.get_encoding(gt_wps, wps_mask[:, 0])
        q_loss, e_loss, traj_emb, perplexity, encodings = self.vq_vae(traj_emb)
        out_seq = self.get_decoding(traj_emb, wps_mask[:, 0], modes=modes)
        outputs = self.get_output(out_seq)

        if self.p_dec:
            prob_seq = self.get_prob_decoding(traj_emb, wps_mask[:, 0])
            logits = self.get_prob_output(prob_seq)
        else:
            logits = None

        return outputs, logits, q_loss, e_loss, perplexity, encodings

    @torch.inference_mode()
    def get_discretized(self, gt_wps, wps_mask, modes=None, return_wp=True):
        traj_emb = self.get_encoding(gt_wps, wps_mask)
    
        q_loss, e_loss, traj_emb, perplexity, encodings = self.vq_vae(traj_emb)

        return traj_emb, encodings

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

        attns = []
        for i in range(self.num_dec_layers):
            attn_name = f"prob_tx_decoder.{i}.multihead_attn"
            attn = self.attn_weights[attn_name].view((B, self.num_agents, self.num_modes, -1))[0, 0].mean(dim=0).detach()

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

    @torch.inference_mode()
    def get_decoded_wps(self, indices):
        B = indices.size(0)
        traj_emb = []
        for i in range(indices.size(1)):
            ind = indices[:, i].unsqueeze(1)
            encodings = torch.zeros(ind.shape[0], self.vq_vae._num_embeddings, device=ind.device)
            encodings.scatter_(1, ind, 1)
        
            # Quantize and unflatten
            quantized = torch.matmul(encodings, self.vq_vae._embeddings[i].weight).view(B, self.H, 1, -1)
            traj_emb.append(quantized)

        traj_emb = torch.cat(traj_emb, 2)
        traj_masks = torch.ones(traj_emb.shape[0], self.num_agents).to(traj_emb.device)
        out_seq = self.get_decoding(traj_emb, traj_masks)
        outputs = self.get_output(out_seq)

        return outputs

    @torch.inference_mode()
    def get_quantized_emb(self, indices):
        B = indices.size(0)
        traj_emb = []
        for i in range(indices.size(1)):
            ind = indices[:, i].unsqueeze(1)
            encodings = torch.zeros(ind.shape[0], self.vq_vae._num_embeddings, device=ind.device)
            encodings.scatter_(1, ind, 1)
        
            # Quantize and unflatten
            quantized = torch.matmul(encodings, self.vq_vae._embeddings[i].weight).view(B, self.H, 1, -1)
            traj_emb.append(quantized)

        traj_emb = torch.cat(traj_emb, 2)

        return traj_emb

    def run_step(self, batch, prefix, optimizer_idx=None):
        full_obs = batch.get_obs(ref_t=self.H-1, f=self.f, device=self.device)
        obs = {}
        for k in full_obs:
            if k in self._dynamic_feature_names:
                obs[k] = full_obs[k][:, :self.H]
            else:
                obs[k] = full_obs[k]

        gt_wps, wps_mask = batch.get_traj(ref_t=self.H-1, f=self.f, num_features=self.wp_dim)
        # print('gt', gt_wps[..., 2])

        gt_wps = gt_wps[:, :self.T, :self.num_agents, :self.wp_dim].to(self.device)
        wps_mask = wps_mask[:, :self.T, :self.num_agents].to(self.device)

        features = self.process_observations(obs)

        if (optimizer_idx is None) or (optimizer_idx == 0):
            outputs, logits, q_loss, e_loss, perplexity, encoding_indices = self.forward(gt_wps, wps_mask, prefix)
            # if prefix == 'val':
            #     print(encoding_indices)
            pred_next_wps = self.get_pred_next_wps(obs, outputs)
            # print('pred m', pred_next_wps[..., 2])
            # print('pred v', pred_next_wps[..., 6])

            wps_loss, logit_loss, wps_errors, time_wps_errors = self._compute_loss(pred_next_wps, logits, gt_wps, wps_mask)
            wps_error = wps_errors.sum(dim=-1)

            if self.use_wandb:
                batch_size = gt_wps.shape[0]
                self.log(f'{prefix}/wps_loss', wps_loss.detach(), batch_size=batch_size)
                self.log(f'{prefix}/wps_error', wps_error.detach(), batch_size=batch_size)
                self.log(f'{prefix}/q_loss', q_loss.detach(), batch_size=batch_size)
                self.log(f'{prefix}/e_loss', e_loss.detach(), batch_size=batch_size)
                self.log(f'{prefix}/perplexity', perplexity.detach(), batch_size=batch_size)

                for i in range(len(wps_errors)):
                    self.log(f'{prefix}/wps_error[{i}]', wps_errors[i].detach(), batch_size=batch_size)

                if self.num_modes > 1:
                    self.log(f'{prefix}/logit_loss', logit_loss.detach(), batch_size=batch_size)


        if optimizer_idx is None:
            self.prev_preds = [pred_next_wps, gt_wps, logits]
            return wps_loss + 10 * (q_loss + e_loss), pred_next_wps, features
        else:
            raise NotImplementedError

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        loss, *_ = self.run_step(batch, prefix='train', optimizer_idx=optimizer_idx)

        # if self.use_scheduler:
        #     if self.use_wandb:
        #         self.log('train/lr', self.scheduler.get_last_lr()[0])

        #  if self.use_wandb:
            #  for i in range(self.act_dim):
                #  self.log(f'train/std[{i}]', self.output_model.std[i])

        return loss

    def validation_step(self, batch, batch_idx):
        wp_loss, pred_actions, features = self.run_step(batch, prefix='val')

        if batch_idx == 0:
            if self.visualize:
                viz = self.get_visualizations(features)
            if self.use_wandb and self.visualize:
                images = viz.transpose(0, 2, 1, 3, 4).reshape((viz.shape[0] * viz.shape[2], viz.shape[1] * viz.shape[3], viz.shape[-1]))
                self.logger.log_image(key="val/viz", images=[images])
                # self.logger.log_image(key="val/codebook", images=sns.heatmap())

        return wp_loss, pred_actions

    def _compute_modes_loss(self, logits, mse_error, regression_loss):
        target_labels = mse_error.argmin(dim=1)

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
        regression_loss = nll_pytorch_dist(pred, gt, dist='laplace')
        mse = F.mse_loss(pred[..., :gt.shape[-1]], gt, reduction='none')
        return regression_loss, mse

    def _compute_loss(self, pred_wps, logits, gt_wps, wps_mask):
        B = pred_wps.shape[0]
        shaped_pred_wps = pred_wps.reshape((B * self.num_modes, self.T, self.num_agents, 2 * self.wp_dim))
        shaped_gt_wps = gt_wps.unsqueeze(1).repeat_interleave(self.num_modes, 1).reshape((B * self.num_modes, self.T, self.num_agents, self.wp_dim))
        shaped_wps_regression_loss, shaped_wps_mse_errors = self._compute_regression_loss(shaped_pred_wps, shaped_gt_wps)

        if wps_mask is None:
            wps_mask = torch.ones(gt_wps.shape[:-1], dtype=bool, device=gt_wps.device)

        if self.num_modes > 1:
            time_wps_regression_loss = shaped_wps_regression_loss.reshape((B, self.num_modes, self.T, self.num_agents)).permute(0, 3, 1, 2).reshape((B * self.num_agents, self.num_modes, self.T))
            time_wps_mse_errors = shaped_wps_mse_errors.reshape((B, self.num_modes, self.T, self.num_agents, self.wp_dim)).permute(0, 3, 1, 2, 4).reshape((B * self.num_agents, self.num_modes, self.T, self.wp_dim))

            w_mask = wps_mask.unsqueeze(1).permute(0, 3, 1, 2).reshape((B * self.num_agents, 1, self.T))
            wps_regression_loss = (time_wps_regression_loss * w_mask).sum(dim=2) / torch.clamp(w_mask.sum(dim=2), min=1.)
            wps_mse_errors = (time_wps_mse_errors * w_mask.unsqueeze(-1)).sum(dim=2) / torch.clamp(w_mask.unsqueeze(-1).sum(dim=2), min=1.)
            
            logits = logits.transpose(1, 2).reshape((B * self.num_agents, self.num_modes))

            wps_mse_error = wps_mse_errors.sum(dim=-1)

            wps_m = w_mask.squeeze(1).any(dim=1)

            unmasked_wps_loss, _, unmasked_logit_loss, labels, target_labels = self._compute_modes_loss(logits, wps_mse_error, wps_regression_loss)
            wps_loss = unmasked_wps_loss[wps_m].mean()
            #  wps_loss = unmasked_wps_loss[wps_m].sum() / w_mask.sum()
            logit_loss = unmasked_logit_loss[wps_m].mean()

            wps_mse_errors = wps_mse_errors[torch.arange(labels.shape[0], device=labels.device), labels][wps_m].mean(dim=0)
            #  wps_mse_errors = wps_mse_errors[torch.arange(labels.shape[0], device=labels.device), labels][wps_m].sum(dim=0) / w_mask.sum()
            time_wps_min_mse_errors = time_wps_mse_errors[torch.arange(target_labels.shape[0], device=target_labels.device), target_labels]
            time_wps_mse = (time_wps_min_mse_errors * w_mask.reshape((B * self.num_agents, self.T, 1))).sum(dim=0) / torch.clamp(w_mask.reshape((B * self.num_agents, self.T, 1)).sum(dim=0), min=1.)

        else:
            time_wps_regression_loss = shaped_wps_regression_loss.permute(0, 2, 1).reshape((B * self.num_agents, self.T))
            time_wps_mse_errors = shaped_wps_mse_errors.permute(0, 2, 1, 3).reshape((B * self.num_agents, self.T, self.wp_dim))
            w_mask = wps_mask.permute(0, 2, 1).reshape((B * self.num_agents, self.T))

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

    def _compute_action_loss(self, pred_actions, gt_actions, wps_mask, actions_mask):
        # TODO is this right because we are using wp to generate actions
        actions_mask = actions_mask & wps_mask[:, 0, 0]
        act_regression_loss, act_mse_errors = self._compute_regression_loss(pred_actions, gt_actions)
        act_loss = act_regression_loss[actions_mask].mean()
        act_mse_errors = act_mse_errors[actions_mask].mean(dim=0)
        act_errors = torch.sqrt(act_mse_errors)

        return act_loss, act_errors

    def configure_optimizers(self):
        all_params = set(self.parameters())
       
        all_params = all_params 
        wd_params = set()
        for m in self.modules():
            # TODO see if you need to add any other layers to this list
            if isinstance(m, nn.Linear):
                wd_params.add(m.weight)
                #  TODO should we remove biases
                #  wd_params.add(m.bias)
        no_wd_params = all_params - wd_params
        main_optimizer = torch.optim.AdamW(
            [{'params': list(no_wd_params), 'weight_decay': 0}, {'params': list(wd_params), 'weight_decay': self.wd}],
            lr=self.lr)
        main_scheduler = get_cosine_schedule_with_warmup(main_optimizer, self.warmup_steps, self.trainer.estimated_stepping_batches)
        main_d = {"optimizer": main_optimizer, "lr_scheduler": {"scheduler": main_scheduler, "interval": "step"}}
      
        #  act_d = {"optimizer": act_optimizer, "lr_scheduler": torch.optim.lr_scheduler.ExponentialLR(act_optimizer, gamma=0.98)}
    
        d = [main_d]
        return d

    def _get_attn_image(self, attn_name, locs, idx, base_color='white', mask=None, im_shape=(50, 50)):

        black = viz_utils.get_color('black')
        color = viz_utils.get_color(base_color)

        if 'dec' in attn_name:
            attn = self.attn_weights[attn_name].view((locs.shape[0], self.num_agents, self.num_modes * self.T, -1))[idx, 0].mean(dim=0).detach().cpu().numpy()
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
    def get_visualizations_attn(self, features=None, obs=None):
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

            for i in range(self.num_dec_layers):

                # agent atten decoder
                scene_image[row_idx, 1 + 2 * self.num_enc_layers + self.num_dec_layers + i] = self._get_attn_image(
                    f"tx_decoder.{i}.multihead_attn",
                    veh_im_locs,
                    row_idx,
                    base_color='green',
                    mask=vehicles_masks,
                    im_shape=self.im_shape)

        return scene_image

    @torch.inference_mode()
    def get_visualizations(self, features=None, preds=None, obs=None):
        if preds is None:
            preds = self.prev_preds
        scene_image = self.get_visualizations_attn(features=features, obs=obs)

        num_samples = scene_image.shape[0]
        pred_image = np.zeros((num_samples, 3, self.im_shape[0], self.im_shape[1], 3), dtype=np.uint8)

        # TODO make sure this is right in all cases
        #  agents_features, agents_masks, light_features, light_masks, stop_features, stop_masks, walker_features, walker_masks, route_features, route_masks, map_features, map_masks = features
        if features is not None:
            agents_features, agents_masks, light_features, light_masks, stop_features, stop_masks, walker_features, walker_masks, route_features, route_masks, map_features, map_masks = features
        else:
            assert(obs is not None)
            agents_features, agents_masks, light_features, light_masks, stop_features, stop_masks, walker_features, walker_masks, route_features, route_masks, map_features, map_masks = self.process_observations(obs)

        vehicles = agents_features.detach().cpu()[:, -1]
        vehicles_masks = ~agents_masks.detach().cpu()[:, -1]

        conversion = np.array([-1., 1.])

        if preds[2] is None:
            labels = torch.zeros((vehicles.shape[0], vehicles.shape[1]), dtype=torch.int64, device=vehicles.device)
            order = torch.arange(self.num_modes, device=vehicles.device)[None, :, None].repeat(vehicles.shape[0], 1, vehicles.shape[1])
        else:
            logits = preds[2].detach().cpu()
            labels = logits.argmax(dim=1)
            order = logits.argsort(dim=1)

        pred_wps = preds[0][..., :2].detach().cpu()
        gt_wps = preds[1][..., :2].detach().cpu()

        B = pred_wps.shape[0]

        transformed_wps = transform_points(
            pred_wps.reshape((B, self.num_modes * self.T, self.num_agents, 2)).transpose(1, 2),
            vehicles[..., :3],
            invert=True).transpose(1, 2).reshape((B, self.num_modes, self.T, self.num_agents, 2))

        transformed_gt_wps = transform_points(
            gt_wps.reshape((B, self.num_modes * self.T, self.num_agents, 2)).transpose(1, 2),
            vehicles[..., :3],
            invert=True).transpose(1, 2).reshape((B, self.num_modes, self.T, self.num_agents, 2))

        shaped_order = order.unsqueeze(2).repeat_interleave(self.T, 2).permute(0, 2, 3, 1).reshape((-1, self.num_modes))
        shaped_transformed_wps = transformed_wps.permute(0, 2, 3, 1, 4).reshape((-1, self.num_modes, 2))
        ordered_shaped_transformed_wps = shaped_transformed_wps[torch.arange(shaped_transformed_wps.shape[0], device=shaped_transformed_wps.device).unsqueeze(1), shaped_order]
        ordered_transformed_wps = ordered_shaped_transformed_wps.reshape((B, self.T, self.num_agents, self.num_modes, 2))

        shaped_transformed_gt_wps = transformed_gt_wps.permute(0, 2, 3, 1, 4).reshape((-1, self.num_modes, 2))
        ordered_shaped_transformed_gt_wps = shaped_transformed_gt_wps[torch.arange(shaped_transformed_gt_wps.shape[0], device=shaped_transformed_gt_wps.device).unsqueeze(1), shaped_order]
        ordered_transformed_gt_wps = ordered_shaped_transformed_gt_wps.reshape((B, self.T, self.num_agents, self.num_modes, 2))

        # TODO confusingly the indexing swaps the axis so we have to swap back to keep things consistent
        for row_idx in range(num_samples):
            #  masked_wps = picked_wps[row_idx][:, vehicles_masks[row_idx]].numpy()
            #  wp_locs = np.clip((0.5 * scene_image.shape[-2] * (conversion * masked_wps + 1.)).astype(np.int32), 0, scene_image.shape[-2] - 1)

            # TODO adjust color based on time
            #  pred_image[row_idx, 0] = viz_utils.get_image(wp_locs.reshape((-1, 2)), viz_utils.get_color('green'), im_shape=self.im_shape)
            masked_wps = ordered_transformed_wps[row_idx][:, vehicles_masks[row_idx]].numpy().reshape((-1, self.num_modes, 2))
            wp_locs = np.clip((0.5 * scene_image.shape[-2] * (conversion * masked_wps + 1.)).astype(np.int32), 0, scene_image.shape[-2] - 1)
            mode_colors = viz_utils.get_mode_colors(self.num_modes)[None, :, :3].repeat(wp_locs.shape[0], 0)
            pred_image[row_idx, 0] = viz_utils.get_image(wp_locs.reshape((-1, 2)), mode_colors.reshape((-1, 3)), im_shape=self.im_shape)
            
            masked_gt_wps = ordered_transformed_gt_wps[row_idx][:, vehicles_masks[row_idx]].numpy().reshape((-1, self.num_modes, 2))
            gt_wp_locs = np.clip((0.5 * scene_image.shape[-2] * (conversion * masked_gt_wps + 1.)).astype(np.int32), 0, scene_image.shape[-2] - 1)
            pred_image[row_idx, 2] = viz_utils.get_image(gt_wp_locs.reshape((-1, 2)), mode_colors.reshape((-1, 3)), im_shape=self.im_shape)

            #  pred_image[row_idx, 1] = viz_utils.get_image(ego_wp_locs.reshape((-1, 2)), mode_colors.reshape((-1, 3)), im_shape=self.im_shape)

            ego_masked_wps = ordered_transformed_wps[row_idx][:, 0].numpy().reshape((-1, self.num_modes, 2))
            ego_wp_locs = np.clip((0.5 * scene_image.shape[-2] * (conversion * ego_masked_wps + 1.)).astype(np.int32), 0, scene_image.shape[-2] - 1)
            ego_mode_colors = viz_utils.get_mode_colors(self.num_modes)[None, :, :3].repeat(ego_wp_locs.shape[0], 0)
            pred_image[row_idx, 1] = viz_utils.get_image(ego_wp_locs.reshape((-1, 2)), ego_mode_colors.reshape((-1, 3)), im_shape=self.im_shape)

        compact_scene_image = np.stack([
            scene_image[:, 0],
            scene_image[:, 1:1+self.num_enc_layers+self.num_dec_layers].mean(axis=1),
            scene_image[:, 1+self.num_enc_layers+self.num_dec_layers:1+2*self.num_enc_layers+2*self.num_dec_layers].mean(axis=1),
        ], axis=1)
        return np.concatenate([compact_scene_image, pred_image], axis=1)
        #  return np.concatenate([scene_image, pred_image], axis=1)
        #  return pred_image
        #  return scene_image

    @torch.inference_mode()
    def get_wandb_info(self, features=None, preds=None, obs=None):
        return {}


class VectorQuantizer(nn.Module):
    def __init__(self, num_agents, num_embeddings, embedding_dim, commitment_cost, device=None):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_agents = num_agents
        self._num_embeddings = num_embeddings
        
        self._embeddings = []
        for _ in range(self._num_agents):
            emb = nn.Embedding(self._num_embeddings, self._embedding_dim)
            emb.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
            self._embeddings.append(emb)
        self._embeddings = nn.ModuleList(self._embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        q_loss, e_loss = 0, 0
        quantized_inputs = []
        all_encoding_indices = []
        perplexity = 0
        for i in range(self._num_agents):
            inp = inputs[:, :, i].permute(0, 2, 1).contiguous()
            input_shape = inp.shape

            # Flatten input
            flat_input = inp.view(-1, self._embedding_dim)
        
            # Calculate distances
            distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embeddings[i].weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embeddings[i].weight.t()))

            # Encoding
            encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1) 

            encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
            encodings.scatter_(1, encoding_indices, 1)
            all_encoding_indices.append(encoding_indices)
        
            # Quantize and unflatten
            quantized = torch.matmul(encodings, self._embeddings[i].weight).view(input_shape)

            e_latent_loss = F.mse_loss(quantized.detach(), inp)
            q_latent_loss = F.mse_loss(quantized, inp.detach())

            q_loss += q_latent_loss
            e_loss += self._commitment_cost * e_latent_loss
        
            quantized = inp + (quantized - inp).detach()
            avg_probs = torch.mean(encodings, dim=0)
            
            perplexity += torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
            quantized_inputs.append(quantized.permute(0, 2, 1).clone())
        
        quantized = torch.stack(quantized_inputs, dim=2)
        encoding_indices = torch.stack(all_encoding_indices, dim=1)
        return q_loss / self._num_agents, e_loss / self._num_agents, quantized.contiguous(), perplexity / self._num_agents, encoding_indices


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._num_agents = num_agents
        
        self._embeddings = []
        for _ in range(self._num_agents):
            emb = nn.Embedding(self._num_embeddings, self._embedding_dim)
            emb.weight.data.normal_() # .uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
            self._embeddings.append(emb)
        self._embeddings = nn.ModuleList(self._embeddings)
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon


    def forward(self, inputs):
        q_loss, e_loss = 0, 0
        quantized_inputs = []
        all_encoding_indices = []
        perplexity = 0
        for i in range(self._num_agents):
            inp = inputs[:, :, i].permute(0, 2, 1).contiguous()
            input_shape = inp.shape
        
            # Flatten input
            flat_input = inp.view(-1, self._embedding_dim)
        
            # Calculate distances
            distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
            # Encoding
            encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
            encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
            encodings.scatter_(1, encoding_indices, 1)
        
            quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
            if self.training:
                self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
                n = torch.sum(self._ema_cluster_size.data)
                self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n)
            
                dw = torch.matmul(encodings.t(), flat_input)
                self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
                self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
            e_latent_loss = F.mse_loss(quantized.detach(), inp)
            loss = self._commitment_cost * e_latent_loss
        
            # Straight Through Estimator
            quantized = inputs + (quantized - inp).detach()
            avg_probs = torch.mean(encodings, dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        quantized = torch.stack(quantized_inputs, dim=2)
        encoding_indices = torch.stack(all_encoding_indices, dim=1)
        
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity / self._num_agents, encodings