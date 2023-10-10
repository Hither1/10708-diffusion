from dataclasses import dataclass
from typing import Dict, List, Tuple, cast
import os
import math

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.nn import functional as F
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
import cv2

from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.features.generic_agents import GenericAgents
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.features.vector_set_map import VectorSetMap
from nuplan.planning.training.modeling.models.urban_driver_open_loop_model import (
    UrbanDriverOpenLoopModel,
    convert_predictions_to_trajectory,
    UrbanDriverOpenLoopModelParams
)
from nuplan.planning.training.modeling.models.diffusion_utils import (
    SinusoidalPosEmb,
    Whitener,
    DummyWhitener,
    rand_log_logistic,
    DiffAndWhitener,
    # BoundsWhitener,
    Standardizer
)
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.lightning_module_wrapper import LightningModuleWrapper
from nuplan.planning.script.builders.model_builder import build_torch_module_wrapper
from nuplan.planning.training.modeling.models.encoder_decoder_layers import (
    ParallelAttentionLayer
)
from nuplan.planning.training.modeling.models.positional_embeddings import VolumetricPositionEncoding2D
from nuplan.planning.training.modeling.models.torch_transformer_layers import TransformerEncoderLayer, TransformerEncoder
from nuplan.planning.training.callbacks.utils.visualization_utils import get_generic_raster_from_vector_map


class UrbanDriverDiffusionModel(UrbanDriverOpenLoopModel):
    def __init__(
        self,

        model_params,
        feature_params,
        target_params,

        T: int = 32,
        sigma_data: float = 0.5,
        sigma_min: float = 2e-3,
        sigma_max: float = 8e1,
        rho: float = 7,

        p_mean: float = -1.2,
        p_std: float = 1.2,
        absolute_params_path: str = '',
        relative_params_path: str = '',

        predictions_per_sample: int = 4,

        num_encoder_layers: int = 2,
        num_trajectory_decoder_layers: int = 2,
        num_global_decoder_layers: int = 2,

        use_loss_weight: bool = True,
        use_weight_init: bool = True,

        unconditional: bool = False,    # ignores scene tokens
        use_single_trajectory_token: bool = False,  # decodes entire trajectory from single token, otherwise uses H tokens
        use_deltas: bool = False,       # predict displacements instead of absolute coordinates
        use_relative: bool = False,     # use relative temporal attention over trajectory features, removes absolute temporal embeddings for traj
        load_checkpoint_path: str = '',

        max_dist: float = 50,           # used to normalize ALL tokens (incl. trajectory)
        use_noise_token: bool = True,   # concat "noise level" token when self-attending, otherwise add to all tokens
        noise_scheduler: str = 'beso',  # denotes commonly used noise schedule parameters (edm or beso)

        easy_validation: bool = False,  # instead of starting from pure noise, start with not that much noise at inference
        standardize_input: bool = True, # standardize input so that everything is [-1,1]
        whiten_trajectory: bool = True, # whiten trajectory so its unit gaussian
        use_positional_encodings: bool = True,
    ):
        super().__init__(model_params, feature_params, target_params)

        self.feature_dim = model_params.global_embedding_size
        self.output_dim = target_params.future_trajectory_sampling.num_poses * Trajectory.state_size()

        self.T = T
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

        self.p_mean = p_mean
        self.p_std = p_std

        self.H = target_params.future_trajectory_sampling.num_poses
        
        self.unconditional = unconditional
        self.use_single_trajectory_token = use_single_trajectory_token
        self.use_deltas = use_deltas
        self.use_relative = use_relative
        self.predictions_per_sample = predictions_per_sample

        self.num_encoder_layers = num_encoder_layers
        self.num_trajectory_decoder_layers = num_trajectory_decoder_layers
        self.num_global_decoder_layers = num_global_decoder_layers

        self.max_dist = max_dist
        # self.max_token_dist = max_token_dist
        self.use_noise_token = use_noise_token
        self.noise_scheduler = noise_scheduler

        self.easy_validation = easy_validation
        self.standardize_input = standardize_input
        self.whiten_trajectory = whiten_trajectory
        self.use_positional_encodings = use_positional_encodings

        self.standardizer = Standardizer(max_dist=max_dist)
        params_path = relative_params_path if use_deltas else absolute_params_path
        whitener_cls = Whitener if whiten_trajectory else DummyWhitener
        self.whitener = whitener_cls(params_path, use_deltas)

        if not easy_validation:
            if noise_scheduler == 'edm':
                self.set_sampling_steps(
                    T=32, 
                    sigma_max=8e1, 
                    sigma_min=2e-3, 
                    rho=7, 
                    strategy='edm',
                    init=True
                )
            elif noise_scheduler == 'beso':
                self.set_sampling_steps(
                    T=32,
                    sigma_max=1.0,
                    sigma_min=.001,
                    rho=5,
                    strategy='edm',
                    init=True
                )
            elif noise_scheduler == 'beso2':
                self.set_sampling_steps(
                    T=32,
                    sigma_max=1.0,
                    sigma_min=.0001,
                    rho=5,
                    strategy='edm',
                    init=True
                )
            else:
                raise NotImplementedError
        else:
            # for easy validation only
            self.set_sampling_steps(
                T=10,
                sigma_max=0.02,
                sigma_min=.0001,
                rho=5,
                strategy='edm',
                init=True
            )

        # Noise level encoder
        if use_noise_token:
            self.sigma_encoder = nn.Linear(1, self.feature_dim)
        else:
            self.sigma_encoder = nn.Sequential(
                SinusoidalPosEmb(self.feature_dim),
                nn.Linear(self.feature_dim, self.feature_dim * 2),
                nn.ReLU(),
                nn.Linear(self.feature_dim * 2, self.feature_dim)
            )

        # Diffusion model components
        del self.global_head # don't need this

        self.encoder_layers = torch.nn.ModuleList([
            ParallelAttentionLayer(
                d_model=self.feature_dim, 
                self_attention1=True, self_attention2=False,
                cross_attention1=False, cross_attention2=False,
                rotary_pe=use_positional_encodings
            )
        for _ in range(num_encoder_layers)])

        self.trajectory_dim = self.H * 3 if use_single_trajectory_token else 3
        self.trajectory_encoder = nn.Linear(self.trajectory_dim, self.feature_dim)
        # if not use_single_trajectory_token and not use_relative:
        self.trajectory_time_embeddings = nn.Embedding(self.H, self.feature_dim)

        self.extended_type_embedding = nn.Embedding(2, self.feature_dim) # trajectory, noise token

        # Decoder layers
        self.decoder_layers = torch.nn.ModuleList([
            ParallelAttentionLayer(
                d_model=self.feature_dim, 
                self_attention1=True, self_attention2=False,
                cross_attention1=True, cross_attention2=False,
                rotary_pe=use_positional_encodings
            )
        for _ in range(num_trajectory_decoder_layers)])

        self.global_attention_layers = torch.nn.ModuleList([
            ParallelAttentionLayer(
                d_model=self.feature_dim, 
                self_attention1=True, self_attention2=False,
                cross_attention1=False, cross_attention2=False,
                rotary_pe=use_positional_encodings
            )
        for _ in range(num_global_decoder_layers)])

        self.decoder_mlp = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.trajectory_dim)
        )

        self.rel_pos_enc = VolumetricPositionEncoding2D(self.feature_dim)

        self.use_loss_weight = use_loss_weight

        # Weight initialization
        if use_weight_init:
            self.apply(self._init_weights)

        # Load weights
        if load_checkpoint_path:
            try:
                base_path = '/'.join(load_checkpoint_path.split('/')[:-2])
                config_path = os.path.join(base_path, 'code/hydra/config.yaml')
                model_config = OmegaConf.load(config_path)
                torch_module_wrapper = build_torch_module_wrapper(model_config.model)

                model = LightningModuleWrapper.load_from_checkpoint(
                    load_checkpoint_path, model=torch_module_wrapper
                ).model

                self.load_state_dict(model.state_dict(), strict=True)
                print('Loaded model weights in constructor')
            except Exception as E:
                print('Failed to load model weights in constructor -- this is fine for evaluation')

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def sigma(self, t):
        return t

    def set_sampling_steps(self, T=32, sigma_max=8e1, sigma_min=2e-3, rho=7, strategy='edm', freeze_t=0, freeze_steps=0, init=False):
        self.T = T
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.rho = rho

        # Compute discretized timesteps
        if strategy == 'edm':
            sigmas = (
                self.sigma_max ** (1 / self.rho) + \
                (torch.arange(T+1) / (T-1)) * \
                (self.sigma_min**(1 / self.rho) - self.sigma_max**(1 / self.rho)) \
            )**self.rho
            sigmas[-1] = 0
        elif strategy == 'linear':
            sigmas = torch.linspace(sigma_max, sigma_min, T)
            sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
        elif strategy == 'exponential':
            sigmas = torch.linspace(math.log(sigma_max), math.log(sigma_min), T).exp()
            sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
        else:
            raise NotImplementedError

        if freeze_steps > 0:
            freeze_sigma = sigmas[freeze_t:freeze_t+1].repeat(freeze_steps)
            sigmas = torch.cat([
                sigmas[:freeze_t], freeze_sigma, sigmas[freeze_t:]
            ])
            self.T += freeze_steps

        if init:
            self.register_buffer('ts', sigmas)
        else:
            self.ts = sigmas.to(self.ts.device)

    def forward(self, features: FeaturesType) -> TargetsType:
        use_guidance = 'guidance_target' in features

        # Recover features
        vector_set_map_data = cast(VectorSetMap, features["vector_set_map"])
        ego_agent_features = cast(GenericAgents, features["generic_agents"])
        batch_size = ego_agent_features.batch_size

        scene_features, scene_feature_masks = self.encode_scene_features(ego_agent_features, vector_set_map_data)

        # Only use for denoising
        if 'trajectory' in features:
            ego_gt_trajectory = features['trajectory'].data.clone()
            ego_gt_trajectory = self.standardizer.transform_features(ego_gt_trajectory)

        # Visualize the noisy samples
        # self.visualize_noisy_samples(features, ego_gt_trajectory)

        # Condition dropout
        if self.unconditional:
            scene_features = 0 * scene_features
            scene_feature_masks = False * scene_feature_masks

        if self.training:
            if not self.easy_validation:
                if self.noise_scheduler == 'edm':
                    sigma = (torch.randn(batch_size, device=ego_gt_trajectory.device) * self.p_std + self.p_mean).exp()[:,None]
                elif self.noise_scheduler in ('beso', 'beso2'):
                    sigma = rand_log_logistic(
                        (batch_size,1), 
                        loc=math.log(self.sigma_data), scale=0.5, 
                        min_value=self.sigma_min, max_value=self.sigma_max, 
                        device=ego_gt_trajectory.device
                    )
            else:
                # import pdb; pdb.set_trace()
                # log_uniform = torch.distributions.uniform.Uniform(np.log(1e-4), np.log(1e-1))
                # sigma = log_uniform.sample((batch_size,1)).exp().to(ego_gt_trajectory.device)
                sigma = self.ts.clone()[:-1][np.random.randint(0,self.ts.shape[0]-1,(batch_size,1))]
            
            # Add noise in whitened space
            ego_gt_trajectory_whitened = self.whitener.transform_features(ego_gt_trajectory)
            ego_noisy_trajectory = torch.normal(ego_gt_trajectory_whitened, sigma)
            # ego_noisy_trajectory = self.whitener.untransform_features(ego_noisy_trajectory_whitened)

            predictions = self.denoise_with_preconditioning(
                ego_noisy_trajectory,
                sigma,
                scene_features,
                scene_feature_masks
            )
            
            predictions = self.whitener.untransform_features(predictions)
            predictions = self.standardizer.untransform_features(predictions)
            output = {"trajectory": Trajectory(data=convert_predictions_to_trajectory(predictions))}
            if self.use_loss_weight:
                loss_weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data)**2
                loss_weight = loss_weight.clamp(0,1000) # TODO: lol
                output.update({'loss_weight': loss_weight})
            return output
        else:
            # Multiple predictions per sample
            predictions_per_sample = self.predictions_per_sample
            scene_features = (
                scene_features[0].repeat_interleave(predictions_per_sample,0),
                scene_features[1].repeat_interleave(predictions_per_sample,0),
                scene_features[2].repeat_interleave(predictions_per_sample,0)
            )
            # scene_features = scene_features.repeat_interleave(predictions_per_sample,0)
            scene_feature_masks = scene_feature_masks.repeat_interleave(predictions_per_sample,0)

            # # Stochastic sampling params
            # S_churn = 80 if 'S_churn' not in features else features['S_churn']
            # S_noise = 1.0 # 1.003

            w = 0.0 if 'w' not in features else features['w']
    
            # Sampling / inference
            trajectory_feature_size = self.H * 3
            if not self.easy_validation:
                ego_trajectory = torch.randn(batch_size * predictions_per_sample, trajectory_feature_size, device=scene_feature_masks.device)
                log_prob = Normal(0,1).log_prob(ego_trajectory).sum(dim=-1)
                ego_trajectory = ego_trajectory * self.sigma(self.ts[0])
            else:
                # for easy val, just noise the trajectory a bit
                ego_gt_trajectory = ego_gt_trajectory.repeat_interleave(predictions_per_sample,0)
                ego_gt_trajectory_whitened = self.whitener.transform_features(ego_gt_trajectory)
                ego_trajectory = torch.normal(ego_gt_trajectory_whitened, self.ts[0])
                log_prob = Normal(0,self.ts[0]).log_prob(ego_trajectory).sum(dim=-1)

            if use_guidance:
                target = features['guidance_target']
                target = torch.as_tensor(target, device=ego_trajectory.device, dtype=ego_trajectory.dtype)

            intermediate_trajectories = []
            # all_weights = []

            # Heun's 2nd order method (algorithm 1): https://arxiv.org/pdf/2206.00364
            for i in range(self.T):
                t = self.ts[i]
                t_next = self.ts[i+1]
                sigma = self.sigma(t)[None,None].repeat(batch_size*predictions_per_sample,1)
                sigma_next = self.sigma(t_next)[None,None].repeat(batch_size*predictions_per_sample,1)

                # Stochastic sampler only -- churning 
                # eps = torch.randn(batch_size * predictions_per_sample, trajectory_feature_size, device=scene_feature_masks.device) * (S_noise**2)
                # gamma = min(S_churn/self.T, np.sqrt(2)-1)
                # t_hat = t + gamma * t
                # sigma = sigma * (gamma+1)
                # ego_trajectory = ego_trajectory + torch.sqrt(t_hat**2 - t**2) * eps
                t_hat = t

                denoised, weights = self.denoise_with_preconditioning(
                    ego_trajectory,
                    sigma,
                    scene_features,
                    scene_feature_masks,
                    return_weights=True
                )
                # all_weights.append(weights)
                data_score_fn = (denoised - ego_trajectory) / (sigma**2)
                dxdt = -sigma * data_score_fn
                if use_guidance and features['guidance_mode'] in ('ours', 'motiondiffuser'):
                    smoothing = features['guidance_mode'] == 'ours'
                    guidance_score_fn = self.compute_guidance_score(ego_trajectory, target, sigma, smoothing)
                    guidance_score_fn = (guidance_score_fn * sigma).clamp(-1,1) / sigma
                    guidance_weight = features['guidance_weight'] # if i > 16 else 0.0
                    dxdt += sigma * guidance_score_fn * guidance_weight # features['guidance_weight'] # * guidance_weight
                ego_trajectory_next = ego_trajectory + (t_next - t_hat) * dxdt
                
                # Apply second order correction
                if (sigma_next != 0).all():
                    denoised_next, weights = self.denoise_with_preconditioning(
                        ego_trajectory_next, 
                        sigma_next,
                        scene_features,
                        scene_feature_masks,
                        return_weights=True
                    )
                    # all_weights.append(weights)
                    data_score_fn_next = (denoised_next - ego_trajectory_next) / (sigma_next**2)
                    dxdt_next = -sigma_next * data_score_fn_next
                    if use_guidance and features['guidance_mode'] in ('ours', 'motiondiffuser'):
                        smoothing = features['guidance_mode'] == 'ours'
                        guidance_score_fn_next = self.compute_guidance_score(ego_trajectory_next, target, sigma_next, smoothing)
                        guidance_score_fn_next = (guidance_score_fn_next * sigma_next).clamp(-1,1) / sigma_next
                        dxdt_next += sigma_next * guidance_score_fn_next * features['guidance_weight']
                    ego_trajectory_next = ego_trajectory + (t_next - t_hat) * 0.5 * (dxdt + dxdt_next)
                
                ego_trajectory = ego_trajectory_next

                if use_guidance:
                    if features['guidance_mode'] == 'inpaint':
                        ego_trajectory = self.enforce_target_condition(ego_trajectory, target)
                    if features['guidance_mode'] == 'ctg':
                        ego_trajectory = ego_trajectory - self.compute_guidance_score(ego_trajectory, target, sigma) * .1

                # Same for log probability
                eps = torch.randn_like(ego_trajectory)
                with torch.enable_grad():
                    f = lambda x: (-1 / sigma) * (self.denoise_with_preconditioning(
                        x, 
                        sigma,
                        scene_features,
                        scene_feature_masks,
                        return_weights=False
                        ) - x)
                    ego_trajectory.requires_grad_(True)
                    f_eps = torch.sum(f(ego_trajectory) * eps)
                    grad_f_eps = torch.autograd.grad(f_eps, ego_trajectory)[0]
                ego_trajectory.requires_grad_(False)
                out = torch.sum(grad_f_eps * eps, dim=tuple(range(1, len(ego_trajectory.shape))))
                log_prob = log_prob - (t_next - t) * out

                # Visualize intermediary trajectories
                c_in = 1 / torch.sqrt(sigma**2 + self.sigma_data**2)
                temp_ego_trajectory = denoised * c_in
                temp_ego_trajectory = self.whitener.untransform_features(temp_ego_trajectory)
                temp_ego_trajectory = self.standardizer.untransform_features(temp_ego_trajectory)
                temp_ego_trajectory = temp_ego_trajectory.reshape(batch_size, predictions_per_sample, self.output_dim)
                intermediate_trajectories.append(temp_ego_trajectory)

            ego_trajectory = self.whitener.untransform_features(ego_trajectory)
            ego_trajectory = self.standardizer.untransform_features(ego_trajectory)
            ego_trajectory = ego_trajectory.reshape(batch_size, predictions_per_sample, self.output_dim)
            log_prob = log_prob.reshape(batch_size, predictions_per_sample)
            log_prob = log_prob - log_prob.max(dim=1, keepdim=True).values
            prob = log_prob.exp()
            # prob = (log_prob / 10).exp() # smoothing factor
            prob = prob / prob.sum(dim=1, keepdim=True)
            best_trajectory = ego_trajectory[range(batch_size), prob.argmax(dim=1)]

            return {
                "trajectory": Trajectory(data=convert_predictions_to_trajectory(best_trajectory)),
                "multimodal_trajectories": ego_trajectory,
                "probabilities": prob,
                "intermediate": intermediate_trajectories,
                # "attention_weights": torch.stack(all_weights, dim=0) # .mean(dim=0)
            }

    def encode_scene_features(self, ego_agent_features, vector_set_map_data):
        batch_size = ego_agent_features.batch_size

        # Extract features across batch
        agent_features, agent_avails, agent_positions = self.extract_agent_features(ego_agent_features, batch_size, return_positions=True)
        map_features, map_avails, map_positions = self.extract_map_features(vector_set_map_data, batch_size, return_positions=True)

        if self.standardize_input:
            # Normalize features
            agent_features[...,:2] /= self.max_dist    # x,y
            agent_features[...,3:5] /= self.max_dist   # vx,vy
            agent_features[...,5:7] /= self.max_dist   # ax,ay
            map_features[...,:2] /= self.max_dist      # x,y

            agent_positions /= self.max_dist
            map_positions /= self.max_dist        

            # Ignore distant features
            # The cutoff distance is less than the distance used for normalization
            cutoff_ratio = 1.0
            agent_avails = agent_avails * (agent_features[...,:2].norm(dim=-1) <= cutoff_ratio)
            map_avails = map_avails * (map_features[...,:2].norm(dim=-1) <= cutoff_ratio)
            agent_features[~agent_avails] = 0
            # agent_positions[~agent_avails] = 0
            map_features[~map_avails] = 0
            # map_positions[~map_avails] = 0
        
        features = torch.cat([agent_features, map_features], dim=1)
        avails = torch.cat([agent_avails, map_avails], dim=1)
        positions = torch.cat([agent_positions, map_positions], dim=1)

        # embed inputs
        feature_embedding = self.feature_embedding(features)

        # calculate positional embedding, then transform [num_points, 1, feature_dim] -> [1, 1, num_points, feature_dim]
        pos_embedding = None # self.positional_embedding(features).unsqueeze(0).transpose(1, 2)

        # invalid mask
        invalid_mask = ~avails
        invalid_polys = invalid_mask.all(-1)

        # local subgraph
        embeddings = self.local_subgraph(feature_embedding, invalid_mask, pos_embedding)
        if hasattr(self, "global_from_local"):
            embeddings = self.global_from_local(embeddings)
        embeddings = F.normalize(embeddings, dim=-1) * (self._model_params.global_embedding_size**0.5)
        # embeddings = embeddings.transpose(0, 1)

        type_embedding = self.type_embedding(
            batch_size,
            self._feature_params.max_agents,
            self._feature_params.agent_features,
            self._feature_params.map_features,
            self._feature_params.max_elements,
            device=features.device,
        ) # .transpose(0, 1)

        # disable certain elements on demand
        if self._feature_params.disable_agents:
            invalid_polys[
                :, 1 : (1 + self._feature_params.max_agents * len(self._feature_params.agent_features))
            ] = 1  # agents won't create attention

        if self._feature_params.disable_map:
            invalid_polys[
                :, (1 + self._feature_params.max_agents * len(self._feature_params.agent_features)) :
            ] = 1  # map features won't create attention

        invalid_polys[:, 0] = 0  # make ego always available in global graph

        # # global attention layers (transformer)
        # outputs, attns = self.global_head(embeddings, type_embedding, invalid_polys)

        # features = self.scene_encoder_layers(embeddings + type_embedding, src_key_padding_mask=invalid_polys)

        pos_enc = self.rel_pos_enc(positions)
        for layer in self.encoder_layers:
            if self.use_positional_encodings:
                embeddings, _ = layer(embeddings, invalid_polys, None, None, seq1_pos=pos_enc, seq1_sem_pos=type_embedding)
            else:
                embeddings, _ = layer(embeddings, invalid_polys, None, None, seq1_sem_pos=type_embedding)

        return (embeddings, type_embedding, pos_enc), invalid_polys
        # return embeddings + type_embedding, invalid_polys

    def denoise_with_cfg(self, ego_trajectory, sigma, scene_features, scene_feature_masks, w=0):
        """
        Denoise using classifier-free guidance
        """
        # masked_scene_features = scene_features * 0.
        # masked_scene_feature_masks = scene_feature_masks * False
        # cond_denoised = self.denoise_with_preconditioning(ego_trajectory, sigma, scene_features, scene_feature_masks)
        # uncond_denoised = self.denoise_with_preconditioning(ego_trajectory, sigma, masked_scene_features, masked_scene_feature_masks)
        # return ((1+w) * cond_denoised) - (w * uncond_denoised)
        # TODO: fix this later
        raise NotImplementedError

    def denoise_with_preconditioning(self, ego_trajectory, sigma, scene_features, scene_feature_masks, return_weights=False):
        """
        Denoise ego_trajectory with noise level sigma
        Equivalent to evaluating D_theta in this paper: https://arxiv.org/pdf/2206.00364
        Returns denoised trajectory, not the residual noise
        """
        c_in = 1 / torch.sqrt(sigma**2 + self.sigma_data**2)
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / torch.sqrt(sigma**2 + self.sigma_data**2)
        c_noise = .25 * torch.log(sigma)
        denoise_out, all_weights = self.denoise( 
            c_in * ego_trajectory, 
            c_noise,
            scene_features,
            scene_feature_masks
        )
        out = c_skip * ego_trajectory + c_out * denoise_out

        if return_weights:
            return out, all_weights
        else:
            return out

    def denoise(self, ego_trajectory, sigma, scene_features, scene_feature_masks):
        """
        Denoise ego_trajectory with noise level sigma (no preconditioning)
        Equivalent to evaluating F_theta in this paper: https://arxiv.org/pdf/2206.00364
        """
        batch_size = ego_trajectory.shape[0]
        scene_features, type_embedding, scene_pos_enc = scene_features
        
        # We have absolute trajectory (ego_trajectory) 
        # and whitened displacements (ego_trajectory_whitened)
        ego_trajectory_abs = self.whitener.untransform_features(ego_trajectory)

        # Trajectory features
        ego_trajectory = ego_trajectory.reshape(ego_trajectory.shape[0],self.H,3)
        trajectory_features = self.trajectory_encoder(ego_trajectory)

        trajectory_time_embedding = self.trajectory_time_embeddings(torch.arange(self.H, device=ego_trajectory.device))[None].repeat(batch_size,1,1)
        trajectory_type_embedding = self.extended_type_embedding(
            torch.as_tensor([0], device=ego_trajectory.device)
        )[None].repeat(batch_size,self.H,1)
        trajectory_masks = torch.zeros(trajectory_features.shape[:-1], dtype=bool, device=trajectory_features.device)
        trajectory_pos_enc = self.rel_pos_enc(ego_trajectory_abs.reshape(batch_size,16,3)[...,:2])

        # Sigma encoding
        sigma_embeddings = self.sigma_encoder(sigma)
        sigma_embeddings = sigma_embeddings.reshape(batch_size,1,self.feature_dim)
        sigma_masks = torch.zeros(batch_size, 1, dtype=torch.bool, device=sigma.device)
        sigma_type_embedding = self.extended_type_embedding(
            torch.as_tensor([1], device=ego_trajectory.device)
        )[None].repeat(batch_size,1,1)
        sigma_pos_enc = self.rel_pos_enc(torch.zeros(batch_size,1,2,device=sigma.device))

        for layer in self.decoder_layers:
            if self.use_positional_encodings:
                trajectory_features, scene_features = layer(
                    trajectory_features, trajectory_masks, 
                    scene_features, scene_feature_masks,
                    seq1_pos=trajectory_pos_enc, seq2_pos=scene_pos_enc,
                    seq1_sem_pos=trajectory_time_embedding,
                )
            else:
                trajectory_features, scene_features = layer(
                    trajectory_features, trajectory_masks, 
                    scene_features, scene_feature_masks,
                    seq1_sem_pos=trajectory_time_embedding,
                )

        all_features = torch.cat([scene_features, sigma_embeddings, trajectory_features], dim=1)
        all_masks = torch.cat([scene_feature_masks, sigma_masks, trajectory_masks], dim=1)
        all_type_embedding = torch.cat([type_embedding, sigma_type_embedding, trajectory_type_embedding], dim=1)
        all_pos_enc = torch.cat([scene_pos_enc, sigma_pos_enc, trajectory_pos_enc], dim=1)

        for layer in self.global_attention_layers:
            if self.use_positional_encodings:
                all_features, _ = layer(
                    all_features, all_masks, None, None,
                    seq1_pos=all_pos_enc, seq1_sem_pos=all_type_embedding
                )
            else:
                all_features, _ = layer(
                    all_features, all_masks, None, None,
                    seq1_sem_pos=all_type_embedding
                )

        trajectory_features = all_features[:,-self.H:]
        out = self.decoder_mlp(trajectory_features).reshape(trajectory_features.shape[0],-1)

        return out, None # , all_weights

    def enforce_target_condition(self, trajectory, target):
        padded_target = torch.zeros_like(trajectory)
        padded_target[:,45:47] = target
        new_target = self.whitener.transform_features(padded_target)
        new_trajectory = trajectory.clone()
        new_trajectory[:,45:47] = new_target[:,45:47]
        return new_trajectory

    def compute_guidance_score(self, trajectory, target, sigma, smoothing=True):
        with torch.enable_grad():
            trajectory = trajectory.clone()
            trajectory.requires_grad_(True)
            trajectory_ = self.standardizer.untransform_features(trajectory)
            # trajectory_ = trajectory
            trajectory_ = trajectory_.reshape(trajectory_.shape[0],16,3)[...,:,:2]
            # trajectory_ = trajectory_.reshape(trajectory_.shape[0],16,3)[...,:,:2]
            # loss = trajectory_.abs().sum()
            # print(loss)

            # padded_target = torch.zeros_like(trajectory)
            # padded_target[:,45:47] = target
            # new_target = self.whitener.transform_features(padded_target)
            # print(f'old: {target} new: {new_target[0,:2]}')
            # target = new_target[0,:2]
            
            loss_unreduced = F.mse_loss(trajectory_[None].repeat(target.shape[0],1,1,1), target[:, None, None].repeat(1, trajectory.shape[0],1,1), reduction='none')
            loss_unreduced = loss_unreduced.sum(dim=-1)
            # loss_unreduced = F.mse_loss(trajectory_, target[None, None].repeat(trajectory.shape[0],1,1), reduction='none')
            # loss_unreduced = loss_unreduced.sum(dim=-1)
            if smoothing:
                # loss = (loss_unreduced * torch.logspace(-2,0,16,device=loss_unreduced.device)[None]).sum(-1).mean(0)
                loss = (loss_unreduced * torch.logspace(-2,0,16,device=loss_unreduced.device)[None]).sum(-1).mean(1).min()
            else:
                # loss = loss_unreduced.min(dim=1).values.mean(0)
                loss = loss_unreduced[:,-1].mean(0)
            # loss = torch.nn.functional.softmin(loss_unreduced, dim=1).sum(-1).mean(0)
            # print(f'loss: {loss}')

            # dists = point_to_line_distance(trajectory_, target[0], target[-1])
            # loss = dists.sum() #  + trajectory_.norm()
            # print(dists)

            # dists = (trajectory_[:,None] - target[None])
            # min_dists = torch.linalg.vector_norm(dists, norm_ord, dim=-1).min(dim=1).values
            # print(min_dists)
            # loss = min_dists.sum()
            
            # dists = point_to_polyline_distance(trajectory_, target)
            # print(dists)
            # loss = dists.sum()
            grad = torch.autograd.grad(loss, trajectory)[0]
            # trajectory.requires_grad_(False)
        
        return grad

    def visualize_noisy_samples(self, features, ego_gt_trajectory):
        self.set_sampling_steps(T=32, sigma_max=1, sigma_min=2e-3, rho=7, strategy='edm', freeze_t=0, freeze_steps=0)
        sigmas = torch.as_tensor([0.01, 0.1, 1.0, 10.0, 80.0], device=features['trajectory'].data.device)
        frames = []
        for sigma in sigmas:
            ego_gt_trajectory = features['trajectory'].data.clone()
            ego_gt_trajectory = self.standardizer.transform_features(ego_gt_trajectory)
            ego_gt_trajectory_whitened = self.whitener.transform_features(ego_gt_trajectory)
            sigma = sigma.unsqueeze(0)[:,None]

            sigma_frames = []
            for _ in range(10):
                ego_noisy_trajectory_whitened = torch.normal(ego_gt_trajectory_whitened, sigma)
                c_in = 1 / torch.sqrt(sigma**2 + self.sigma_data**2)
                ego_noisy_trajectory_whitened = ego_noisy_trajectory_whitened * c_in
                ego_noisy_trajectory = self.whitener.untransform_features(ego_noisy_trajectory_whitened)

                ego_noisy_trajectory = self.standardizer.untransform_features(ego_noisy_trajectory)
                ego_noisy_trajectory = ego_noisy_trajectory.reshape(-1,16,3)

                frame = get_generic_raster_from_vector_map(
                    features['vector_set_map'].to_device('cpu'),
                    features['generic_agents'].to_device('cpu'),
                    trajectories=ego_noisy_trajectory.cpu().numpy(),
                    pixel_size=0.2,
                    radius=200
                )
                frame = cv2.putText(frame, f'sigma={round(sigma.item(),2)}', (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                sigma_frames.append(frame)

            sigma_frame = np.stack(sigma_frames, axis=0)
            frames.append(sigma_frame)

        # Also visualize white noise as reference
        white_frames = []
        for _ in range(10):
            ego_noisy_trajectory = torch.normal(torch.zeros_like(ego_gt_trajectory), torch.ones_like(sigma))
            ego_noisy_trajectory = self.whitener.untransform_features(ego_noisy_trajectory)
            ego_noisy_trajectory = self.standardizer.untransform_features(ego_noisy_trajectory)
            ego_noisy_trajectory = ego_noisy_trajectory.reshape(-1,16,3)

            frame = get_generic_raster_from_vector_map(
                features['vector_set_map'].to_device('cpu'),
                features['generic_agents'].to_device('cpu'),
                trajectories=ego_noisy_trajectory.cpu().numpy(),
                pixel_size=0.2,
                radius=200
            )
            frame = cv2.putText(frame, 'white noise', (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            white_frames.append(frame)

        white_frames = np.stack(white_frames, axis=0)
        frames.append(white_frames)

        frames = np.stack(frames, axis=0)
        frames = np.concatenate(frames, axis=2)

        from PIL import Image
        frames = [Image.fromarray(frame) for frame in frames]
        fname_dir = '/zfsauton2/home/brianyan/nuplan-devkit/nuplan/planning/simulation/planner/ml_planner/viz/'
        fname = f'{fname_dir}noisy.gif'
        frames[0].save(fname, save_all=True, append_images=frames[1:], duration=int(10 * 0.5), loop=0)

        import pdb; pdb.set_trace()
        raise

    def cem_test(self, features: FeaturesType, scoring_fn) -> TargetsType:
        # Recover features
        vector_set_map_data = cast(VectorSetMap, features["vector_set_map"])
        ego_agent_features = cast(GenericAgents, features["generic_agents"])
        batch_size = ego_agent_features.batch_size

        assert batch_size == 1, 'Have not implemented batching for CEM yet'

        scene_features, scene_feature_masks = self.encode_scene_features(ego_agent_features, vector_set_map_data)

        # Multiple predictions per sample
        predictions_per_sample = self.predictions_per_sample
        scene_features = (
            scene_features[0].repeat_interleave(predictions_per_sample,0),
            scene_features[1].repeat_interleave(predictions_per_sample,0),
            scene_features[2].repeat_interleave(predictions_per_sample,0)
        )
        # scene_features = scene_features.repeat_interleave(predictions_per_sample,0)
        scene_feature_masks = scene_feature_masks.repeat_interleave(predictions_per_sample,0)

        # Sampling / inference
        trajectory_feature_size = self.H * 3
        ego_trajectory = torch.randn(batch_size * predictions_per_sample, trajectory_feature_size, device=scene_feature_masks.device)
        log_prob = Normal(0,1).log_prob(ego_trajectory).sum(dim=-1)
        ego_trajectory = ego_trajectory * self.sigma(self.ts[0])

        intermediate_trajectories = []
        # all_weights = []

        # Stochastic sampler params
        S_churn = 80
        S_noise = 1.003
        S_tmin = .01
        S_tmax = 1.0

        # CEM params
        filter_ratio = 1.0
        num_to_discard = int(filter_ratio * predictions_per_sample)

        # Heun's 2nd order method (algorithm 1): https://arxiv.org/pdf/2206.00364
        for i in range(self.T):
            t = self.ts[i]
            t_next = self.ts[i+1]
            sigma = self.sigma(t)[None,None].repeat(batch_size*predictions_per_sample,1)
            sigma_next = self.sigma(t_next)[None,None].repeat(batch_size*predictions_per_sample,1)

            # Stochastic sampler only -- churning
            use_churn = ((sigma > S_tmin) * (sigma < S_tmax)).all()
            eps = torch.randn(batch_size * predictions_per_sample, trajectory_feature_size, device=scene_feature_masks.device) * (S_noise**2)
            gamma = min(S_churn/self.T, np.sqrt(2)-1) if use_churn else 0.0
            t_hat = t + gamma * t
            sigma = sigma * (gamma+1)
            ego_trajectory = ego_trajectory + torch.sqrt(t_hat**2 - t**2) * eps
            # t_hat = t

            denoised, weights = self.denoise_with_preconditioning(
                ego_trajectory,
                sigma,
                scene_features,
                scene_feature_masks,
                return_weights=True
            )

            if use_churn:
                # Compute scores
                scores = scoring_fn(denoised)
                scores[scores < 0] = 0
                scores_denom = scores.sum()

                if not np.isclose(scores_denom, 0.0):
                    scores_norm = scores / scores_denom
                    
                    # Resample low-scoring samples
                    discard_idx = np.argsort(scores)[:num_to_discard]
                    new_idx = np.random.choice(range(scores.shape[0]), p=scores_norm, size=num_to_discard)

                    denoised[discard_idx] = denoised[new_idx]
                    ego_trajectory[discard_idx] = ego_trajectory[new_idx]
                    log_prob[discard_idx] = log_prob[new_idx]

            # all_weights.append(weights)
            data_score_fn = (denoised - ego_trajectory) / (sigma**2)
            dxdt = -sigma * data_score_fn
            ego_trajectory_next = ego_trajectory + (t_next - t_hat) * dxdt
            
            # Apply second order correction
            if (sigma_next != 0).all():
                denoised_next, weights = self.denoise_with_preconditioning(
                    ego_trajectory_next, 
                    sigma_next,
                    scene_features,
                    scene_feature_masks,
                    return_weights=True
                )
                # all_weights.append(weights)
                data_score_fn_next = (denoised_next - ego_trajectory_next) / (sigma_next**2)
                dxdt_next = -sigma_next * data_score_fn_next
                ego_trajectory_next = ego_trajectory + (t_next - t_hat) * 0.5 * (dxdt + dxdt_next)
            
            ego_trajectory = ego_trajectory_next

            # Same for log probability
            eps = torch.randn_like(ego_trajectory)
            with torch.enable_grad():
                f = lambda x: (-1 / sigma) * (self.denoise_with_preconditioning(
                    x, 
                    sigma,
                    scene_features,
                    scene_feature_masks,
                    return_weights=False
                    ) - x)
                ego_trajectory.requires_grad_(True)
                f_eps = torch.sum(f(ego_trajectory) * eps)
                grad_f_eps = torch.autograd.grad(f_eps, ego_trajectory)[0]
            ego_trajectory.requires_grad_(False)
            out = torch.sum(grad_f_eps * eps, dim=tuple(range(1, len(ego_trajectory.shape))))
            log_prob = log_prob - (t_next - t) * out

            # Visualize intermediary trajectories
            c_in = 1 / torch.sqrt(sigma**2 + self.sigma_data**2)
            temp_ego_trajectory = denoised * c_in
            temp_ego_trajectory = self.whitener.untransform_features(temp_ego_trajectory)
            temp_ego_trajectory = self.standardizer.untransform_features(temp_ego_trajectory)
            temp_ego_trajectory = temp_ego_trajectory.reshape(batch_size, predictions_per_sample, self.output_dim)
            intermediate_trajectories.append(temp_ego_trajectory)

        ego_trajectory = self.whitener.untransform_features(ego_trajectory)
        ego_trajectory = self.standardizer.untransform_features(ego_trajectory)
        ego_trajectory = ego_trajectory.reshape(batch_size, predictions_per_sample, self.output_dim)
        log_prob = log_prob.reshape(batch_size, predictions_per_sample)
        log_prob = log_prob - log_prob.max(dim=1, keepdim=True).values
        prob = log_prob.exp()
        # prob = (log_prob / 10).exp() # smoothing factor
        prob = prob / prob.sum(dim=1, keepdim=True)
        best_trajectory = ego_trajectory[range(batch_size), prob.argmax(dim=1)]

        # plot_denoising(intermediate_trajectories, features)

        return {
            "trajectory": Trajectory(data=convert_predictions_to_trajectory(best_trajectory)),
            "multimodal_trajectories": ego_trajectory,
            "probabilities": prob,
            "intermediate": intermediate_trajectories,
            # "attention_weights": torch.stack(all_weights, dim=0) # .mean(dim=0)
        }


def plot_denoising(intermediate_trajectories, features):
    from PIL import Image

    colors = np.array([
        np.linspace(0,0,16),
        np.linspace(0,1,16),
        np.linspace(1,0,16)
    ]).T

    frames = []
    for i, intermediate_trajs in enumerate(intermediate_trajectories):
        trajs = intermediate_trajs.squeeze(0).reshape(-1,16,3).cpu().numpy()

        frame = get_generic_raster_from_vector_map(
            features['vector_set_map'].to_device('cpu'),
            features['generic_agents'].to_device('cpu'),
            trajectories=trajs,
            pixel_size=0.1,
            radius=60
        )

        frames.append(frame)
        # plt.close()

    frames = [Image.fromarray(frame) for frame in frames]
    fname_dir = '/zfsauton2/home/brianyan/nuplan-devkit/nuplan/planning/simulation/planner/ml_planner/viz/'
    fname = f'{fname_dir}denoise.gif'
    frames[0].save(fname, save_all=True, append_images=frames[1:], duration=int(32 * 0.25), loop=0)

    raise