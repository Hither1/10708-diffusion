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
from einops import rearrange

from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.features.generic_agents import GenericAgents
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.features.agents_trajectory import AgentTrajectory
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
from nuplan.planning.training.callbacks.utils.visualization_utils import (
    get_generic_raster_from_vector_map,
    get_raster_from_vector_map_with_new_agents,
)
from nuplan.planning.training.preprocessing.features.agent_history import AgentHistory
from nuplan.planning.training.preprocessing.features.vector_set_map import VectorSetMap
from nuplan.planning.training.preprocessing.feature_builders.agent_history_feature_builder import (
    AgentHistoryFeatureBuilder,
)
from nuplan.planning.training.preprocessing.feature_builders.vector_set_map_feature_builder import (
    VectorSetMapFeatureBuilder,
)
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (
    EgoTrajectoryTargetBuilder,
)
from nuplan.planning.training.preprocessing.target_builders.agent_trajectory_target_builder import (
    AgentTrajectoryTargetBuilder
)


class UrbanDriverDiffusionMAStagedModel(UrbanDriverOpenLoopModel):
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

        max_agents: int = 30,

        early_map_attention: bool = True,
        use_coarse_to_fine_attention: bool = True,

        remove_absolute_pos: bool = True
    ):
        super().__init__(model_params, feature_params, target_params)

        self.feature_builders = [
            AgentHistoryFeatureBuilder(
                    trajectory_sampling=feature_params.past_trajectory_sampling,
                    max_agents=max_agents
                ),
            VectorSetMapFeatureBuilder(
                map_features=feature_params.map_features,
                max_elements=feature_params.max_elements,
                max_points=feature_params.max_points,
                radius=feature_params.vector_set_map_feature_radius,
                interpolation_method=feature_params.interpolation_method
            )
        ]
        self.target_builders = [
            EgoTrajectoryTargetBuilder(target_params.future_trajectory_sampling),
            AgentTrajectoryTargetBuilder(
                trajectory_sampling=feature_params.past_trajectory_sampling,
                future_trajectory_sampling=target_params.future_trajectory_sampling,
                max_agents=max_agents
            )
        ]

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

        self.early_map_attention = early_map_attention
        self.use_coarse_to_fine_attention = use_coarse_to_fine_attention

        self.remove_absolute_pos = remove_absolute_pos
        self.subgoal_idx = 4

        self.standardizer = Standardizer(max_dist=max_dist)
        params_path = relative_params_path if use_deltas else absolute_params_path
        whitener_cls = Whitener if whiten_trajectory else DummyWhitener
        self.whitener = whitener_cls(params_path, use_deltas, subgoal_idx=self.subgoal_idx)

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

        if early_map_attention:
            self.map_layers = torch.nn.ModuleList([
                ParallelAttentionLayer(
                    d_model=self.feature_dim, 
                    self_attention1=True, self_attention2=False,
                    cross_attention1=False, cross_attention2=False,
                    rotary_pe=use_positional_encodings
                )
            for _ in range(2)])

        self.encoder_layers = torch.nn.ModuleList([
            ParallelAttentionLayer(
                d_model=self.feature_dim, 
                self_attention1=True, self_attention2=False,
                cross_attention1=False, cross_attention2=False,
                rotary_pe=use_positional_encodings
            )
        for _ in range(num_encoder_layers)])

        self.trajectory_dim = self.H * 3 if use_single_trajectory_token else 3
        # self.trajectory_encoder = nn.Linear(self.trajectory_dim, self.feature_dim)
        # self.trajectory_time_embeddings = nn.Embedding(self.H, self.feature_dim)

        self.extended_type_embedding = nn.Embedding(3, self.feature_dim) # subgoal, noise token, trajectory

        # Subgoal decoder layers
        self.subgoal_encoder = nn.Linear(3, self.feature_dim)
        self.subgoal_decoder_attention_layers = torch.nn.ModuleList([
            ParallelAttentionLayer(
                d_model=self.feature_dim, 
                self_attention1=False, self_attention2=False,
                cross_attention1=True, cross_attention2=False,
                rotary_pe=use_positional_encodings
            )
        for _ in range(num_trajectory_decoder_layers)])

        self.subgoal_decoder_mlp = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, 3)
        )

        # Trajectory decoder layers
        # self.trajectory_decoder_subgoal_attention_layers = torch.nn.ModuleList([
        #     ParallelAttentionLayer(
        #         d_model=self.feature_dim, 
        #         self_attention1=True, self_attention2=False,
        #         cross_attention1=False, cross_attention2=False,
        #         rotary_pe=use_positional_encodings
        #     )
        # for _ in range(2)])

        self.trajectory_decoder_scene_attention_layers = torch.nn.ModuleList([
            ParallelAttentionLayer(
                d_model=self.feature_dim, 
                self_attention1=False, self_attention2=False,
                cross_attention1=True, cross_attention2=False,
                rotary_pe=use_positional_encodings
            )
        for _ in range(2)])

        self.trajectory_decoder_mlp = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.H * 3)
        )

        # Multi-agent prediction head
        self.ma_decoder_mlp = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.H * 3)
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
        ego_agent_features = cast(AgentHistory, features["agent_history"])
        batch_size = ego_agent_features.batch_size

        scene_features, scene_feature_masks = self.encode_scene_features(ego_agent_features, vector_set_map_data)

        # Multi-agent auxiliary task
        multiagent_predictions = self.ma_decoder_mlp(scene_features[0][:,1:31])
        multiagent_predictions = rearrange(multiagent_predictions, 'b a (t d) -> b t a d', t=16, d=3)
        # multiagent_predictions = self.standardizer.untransform_features(multiagent_predictions)
        multiagent_predictions = AgentTrajectory(
            data=multiagent_predictions,
            mask=torch.ones_like(multiagent_predictions[...,0]).bool()
        )

        # Only use for denoising
        if 'trajectory' in features:
            ego_gt_subgoal = features['trajectory'].data.clone()[:,self.subgoal_idx]
            ego_gt_subgoal = self.standardizer.transform_features(ego_gt_subgoal)

        # Visualize the noisy samples
        self.visualize_noisy_samples(features, ego_gt_subgoal)

        # Condition dropout
        if self.unconditional:
            scene_features = 0 * scene_features
            scene_feature_masks = False * scene_feature_masks

        if self.training:
            if not self.easy_validation:
                if self.noise_scheduler == 'edm':
                    sigma = (torch.randn(batch_size, device=ego_gt_subgoal.device) * self.p_std + self.p_mean).exp()[:,None]
                elif self.noise_scheduler in ('beso', 'beso2'):
                    sigma = rand_log_logistic(
                        (batch_size,1), 
                        loc=math.log(self.sigma_data), scale=0.5, 
                        min_value=self.sigma_min, max_value=self.sigma_max, 
                        device=ego_gt_subgoal.device
                    )
            else:
                sigma = self.ts.clone()[:-1][np.random.randint(0,self.ts.shape[0]-1,(batch_size,1))]
            
            # Add noise in whitened space
            ego_gt_subgoal_whitened = self.whitener.transform_features(ego_gt_subgoal)
            ego_noisy_subgoal = torch.normal(ego_gt_subgoal_whitened, sigma)

            pred_subgoal = self.denoise_with_preconditioning(
                ego_noisy_subgoal,
                sigma,
                scene_features,
                scene_feature_masks
            )
            pred_subgoal = self.whitener.untransform_features(pred_subgoal)
            pred_trajectory = self.decode_trajectory(scene_features, scene_feature_masks, ego_gt_subgoal)
            pred_trajectory = self.standardizer.untransform_features(pred_trajectory)

            output = {
                'subgoal': pred_subgoal,
                'trajectory': Trajectory(data=convert_predictions_to_trajectory(pred_trajectory)),
                'agent_trajectories': multiagent_predictions
            }
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
                scene_features[2].repeat_interleave(predictions_per_sample,0),
                scene_features[3].repeat_interleave(predictions_per_sample,0),
            )
            # scene_features = scene_features.repeat_interleave(predictions_per_sample,0)
            scene_feature_masks = scene_feature_masks.repeat_interleave(predictions_per_sample,0)

            # # Stochastic sampling params
            # S_churn = 80 if 'S_churn' not in features else features['S_churn']
            # S_noise = 1.0 # 1.003

            w = 0.0 if 'w' not in features else features['w']
    
            # Sampling / inference
            trajectory_feature_size = 3
            if not self.easy_validation:
                ego_subgoal = torch.randn(batch_size * predictions_per_sample, trajectory_feature_size, device=scene_feature_masks.device)
                log_prob = Normal(0,1).log_prob(ego_subgoal).sum(dim=-1)
                ego_subgoal = ego_subgoal * self.sigma(self.ts[0])
            else:
                # for easy val, just noise the trajectory a bit
                ego_gt_subgoal = ego_gt_subgoal.repeat_interleave(predictions_per_sample,0)
                ego_gt_subgoal_whitened = self.whitener.transform_features(ego_gt_subgoal)
                ego_subgoal = torch.normal(ego_gt_subgoal_whitened, self.ts[0])
                log_prob = Normal(0,self.ts[0]).log_prob(ego_subgoal).sum(dim=-1)

            # if use_guidance:
            #     target = features['guidance_target']
            #     target = torch.as_tensor(target, device=ego_subgoal.device, dtype=ego_subgoal.dtype)

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
                    ego_subgoal,
                    sigma,
                    scene_features,
                    scene_feature_masks,
                    return_weights=True
                )
                # all_weights.append(weights)
                data_score_fn = (denoised - ego_subgoal) / (sigma**2)
                dxdt = -sigma * data_score_fn
                if use_guidance and features['guidance_mode'] in ('ours', 'motiondiffuser'):
                    smoothing = features['guidance_mode'] == 'ours'
                    guidance_score_fn = self.compute_guidance_score(ego_subgoal, target, sigma, smoothing)
                    guidance_score_fn = (guidance_score_fn * sigma).clamp(-1,1) / sigma
                    guidance_weight = features['guidance_weight'] # if i > 16 else 0.0
                    dxdt += sigma * guidance_score_fn * guidance_weight # features['guidance_weight'] # * guidance_weight
                ego_subgoal_next = ego_subgoal + (t_next - t_hat) * dxdt
                
                # Apply second order correction
                if (sigma_next != 0).all():
                    denoised_next, weights = self.denoise_with_preconditioning(
                        ego_subgoal_next, 
                        sigma_next,
                        scene_features,
                        scene_feature_masks,
                        return_weights=True
                    )
                    # all_weights.append(weights)
                    data_score_fn_next = (denoised_next - ego_subgoal_next) / (sigma_next**2)
                    dxdt_next = -sigma_next * data_score_fn_next
                    if use_guidance and features['guidance_mode'] in ('ours', 'motiondiffuser'):
                        smoothing = features['guidance_mode'] == 'ours'
                        guidance_score_fn_next = self.compute_guidance_score(ego_subgoal_next, target, sigma_next, smoothing)
                        guidance_score_fn_next = (guidance_score_fn_next * sigma_next).clamp(-1,1) / sigma_next
                        dxdt_next += sigma_next * guidance_score_fn_next * features['guidance_weight']
                    ego_subgoal_next = ego_subgoal + (t_next - t_hat) * 0.5 * (dxdt + dxdt_next)
                
                ego_subgoal = ego_subgoal_next

                # Same for log probability
                eps = torch.randn_like(ego_subgoal)
                with torch.enable_grad():
                    f = lambda x: (-1 / sigma) * (self.denoise_with_preconditioning(
                        x, 
                        sigma,
                        scene_features,
                        scene_feature_masks,
                        return_weights=False
                        ) - x)
                    ego_subgoal.requires_grad_(True)
                    f_eps = torch.sum(f(ego_subgoal) * eps)
                    grad_f_eps = torch.autograd.grad(f_eps, ego_subgoal)[0]
                ego_subgoal.requires_grad_(False)
                out = torch.sum(grad_f_eps * eps, dim=tuple(range(1, len(ego_subgoal.shape))))
                log_prob = log_prob - (t_next - t) * out

                # # Visualize intermediary trajectories
                # c_in = 1 / torch.sqrt(sigma**2 + self.sigma_data**2)
                # temp_ego_trajectory = denoised * c_in
                # temp_ego_trajectory = self.whitener.untransform_features(temp_ego_trajectory)
                # temp_ego_trajectory = self.standardizer.untransform_features(temp_ego_trajectory)
                # temp_ego_trajectory = temp_ego_trajectory.reshape(batch_size, predictions_per_sample, self.output_dim)
                # intermediate_trajectories.append(temp_ego_trajectory)

            pred_trajectory = self.decode_trajectory(scene_features, scene_feature_masks, ego_subgoal)
            pred_trajectory = self.standardizer.untransform_features(pred_trajectory)
            pred_trajectory = pred_trajectory.reshape(batch_size, predictions_per_sample, self.H * 3)

            ego_subgoal = self.whitener.untransform_features(ego_subgoal)
            ego_subgoal = self.standardizer.untransform_features(ego_subgoal)
            ego_subgoal = ego_subgoal.reshape(batch_size, predictions_per_sample, 3)

            log_prob = log_prob.reshape(batch_size, predictions_per_sample)
            log_prob = log_prob - log_prob.max(dim=1, keepdim=True).values
            prob = log_prob.exp()
            # prob = (log_prob / 10).exp() # smoothing factor
            prob = prob / prob.sum(dim=1, keepdim=True)
            best_trajectory = pred_trajectory[range(batch_size), prob.argmax(dim=1)]
            best_ego_subgoal = ego_subgoal[range(batch_size), prob.argmax(dim=1)]

            return {
                "trajectory": Trajectory(data=convert_predictions_to_trajectory(best_trajectory)),
                "multimodal_trajectories": pred_trajectory,
                "probabilities": prob,
                "intermediate": intermediate_trajectories,
                'agent_trajectories': multiagent_predictions,
                'multimodal_subgoals': ego_subgoal,
                'subgoal': best_ego_subgoal,
            }
        
    def extract_agent_features(self, agent_history):
        ego_features = agent_history.ego
        agent_features = agent_history.data
        agent_masks = agent_history.mask

        B, T, A, D = agent_features.shape
        device = agent_features.device

        ego_agent_features = torch.cat([ego_features[:,:,None], agent_features], dim=2)
        ego_agent_masks = torch.cat([torch.ones_like(agent_masks[:,:,:1]), agent_masks], dim=2)

        # All of this is just to match the format of the old agent features
        # B x 5 x (A+1) x 7 -> B x (A+1) x 20 (5?) x 8
        ego_agent_features = ego_agent_features.permute(0,2,1,3)
        ego_agent_masks = ego_agent_masks.permute(0,2,1)

        ego_agent_positions = ego_agent_features[:,:,-1,:2]

        ego_agent_features = torch.cat([ego_agent_features, torch.zeros(B, A+1, 15, D, dtype=torch.bool, device=device)], dim=2)
        ego_agent_masks = torch.cat([ego_agent_masks, torch.zeros(B, A+1, 15, dtype=torch.bool, device=device)], dim=2)

        ego_agent_features = torch.cat([ego_agent_features, torch.zeros_like(ego_agent_features[...,:1])], dim=-1)

        return ego_agent_features, ego_agent_masks, ego_agent_positions

    def encode_scene_features(self, ego_agent_features, vector_set_map_data):
        batch_size = ego_agent_features.batch_size

        # Extract features across batch
        agent_features, agent_avails, agent_positions = self.extract_agent_features(ego_agent_features)
        map_features, map_avails, map_positions = self.extract_map_features(vector_set_map_data, batch_size, return_positions=True)

        if self.remove_absolute_pos:
            agent_features[...,:2] = 0.
            map_features[...,:2] = 0.
        
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
        local_embeddings = self.local_subgraph(feature_embedding, invalid_mask, pos_embedding)
        if hasattr(self, "global_from_local"):
            local_embeddings = self.global_from_local(local_embeddings)
        local_embeddings = F.normalize(local_embeddings, dim=-1) * (self._model_params.global_embedding_size**0.5)
        # embeddings = embeddings.transpose(0, 1)

        if self.early_map_attention:
            local_map_features = local_embeddings[:,31:]
            map_pos_enc = self.rel_pos_enc(positions[:,31:])

            for layer in self.map_layers:
                local_map_features, _ = layer(local_map_features, invalid_polys[:,31:].clone(), None, None, seq1_pos=map_pos_enc)

            embeddings = torch.cat([local_embeddings[:,:31], local_map_features], dim=1)
        else:
            embeddings = local_embeddings

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

        return (embeddings, type_embedding, pos_enc, positions), invalid_polys
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

    def denoise(self, ego_subgoal, sigma, scene_features, scene_feature_masks):
        """
        Denoise ego_trajectory with noise level sigma (no preconditioning)
        Equivalent to evaluating F_theta in this paper: https://arxiv.org/pdf/2206.00364
        """
        batch_size = ego_subgoal.shape[0]
        scene_features, scene_type_embedding, scene_pos_enc, scene_pos = scene_features
        
        # We have absolute trajectory (ego_trajectory) 
        # and whitened displacements (ego_trajectory_whitened)
        ego_subgoal_abs = self.whitener.untransform_features(ego_subgoal)

        # Trajectory features
        ego_subgoal = ego_subgoal.reshape(ego_subgoal.shape[0],3)
        subgoal_features = self.subgoal_encoder(ego_subgoal)[:,None] # (B,1,D)
        subgoal_type_embedding = self.extended_type_embedding(
            torch.as_tensor([0], device=ego_subgoal.device))[None].repeat(batch_size,1,1)
        subgoal_masks = torch.zeros(subgoal_features.shape[:-1], dtype=bool, device=ego_subgoal.device)
        subgoal_pos = ego_subgoal_abs.reshape(batch_size,1,3)[...,:2]
        subgoal_pos_enc = self.rel_pos_enc(subgoal_pos)

        # trajectory_time_embedding = self.trajectory_time_embeddings(torch.arange(self.H, device=ego_trajectory.device))[None].repeat(batch_size,1,1)
        # trajectory_type_embedding = self.extended_type_embedding(
        #     torch.as_tensor([0], device=ego_trajectory.device)
        # )[None].repeat(batch_size,self.H,1)
        # trajectory_masks = torch.zeros(trajectory_features.shape[:-1], dtype=bool, device=trajectory_features.device)
        # trajectory_pos = ego_trajectory_abs.reshape(batch_size,16,3)[...,:2]
        # trajectory_pos_enc = self.rel_pos_enc(trajectory_pos)

        # Sigma encoding
        sigma_embeddings = self.sigma_encoder(sigma)
        sigma_embeddings = sigma_embeddings.reshape(batch_size,1,self.feature_dim)
        sigma_masks = torch.zeros(batch_size, 1, dtype=torch.bool, device=sigma.device)
        sigma_type_embedding = self.extended_type_embedding(
            torch.as_tensor([1], device=ego_subgoal.device)
        )[None].repeat(batch_size,1,1)
        sigma_pos = torch.zeros(batch_size,1,2,device=sigma.device)
        sigma_pos_enc = self.rel_pos_enc(sigma_pos)

        # Concat noise features
        scene_features = torch.cat([scene_features, sigma_embeddings], dim=1)
        scene_feature_masks = torch.cat([scene_feature_masks, sigma_masks], dim=1)
        scene_pos = torch.cat([scene_pos, sigma_pos], dim=1)
        scene_pos_enc = torch.cat([scene_pos_enc, sigma_pos_enc], dim=1)
        scene_type_embedding = torch.cat([scene_type_embedding, sigma_type_embedding], dim=1)

        if self.use_positional_encodings:
            seq1_pos, seq2_pos = subgoal_pos_enc, scene_pos_enc
        else:
            seq1_pos, seq2_pos = None, None

        if self.use_coarse_to_fine_attention:
            # generate attention mask
            attn_mask = self.generate_coarse_to_fine_masks(sigma, subgoal_pos, scene_pos)
        else:
            attn_mask = None

        for layer in self.subgoal_decoder_attention_layers:
            subgoal_features, scene_features = layer(
                subgoal_features, subgoal_masks,
                scene_features, scene_feature_masks,
                seq1_pos=seq1_pos, seq2_pos=seq2_pos,
                seq1_sem_pos=subgoal_type_embedding, seq2_sem_pos=scene_type_embedding,
                attn_mask_12=attn_mask
            )

        # subgoal_features = all_features[:,-1]
        out = self.subgoal_decoder_mlp(subgoal_features.squeeze(1))

        return out, None # , all_weights
    
    def generate_coarse_to_fine_masks(self, sigmas, positions1, positions2):
        """
        Generates attention masks for coarse-to-fine attention
        This is batched -- generates distinct mask per element of batch
        True = cannot attend

        sigmas: (B,)
        positions1 and positions2: (B,S1 / S2,2), xy coordinates
        """
        # Get distance threshold from noise level (sigmas)
        # TODO: idk what i'm doing
        sigmas = torch.exp(sigmas / 0.25)
        threshold = 5. * (torch.e ** (8. * sigmas)) # (B,)

        # Construct masks
        pairwise_dists = torch.norm(positions1[:,:,None] - positions2[:,None], dim=-1) # (B,S1,S2)
        masks = pairwise_dists > threshold[:,None]

        # there are 8 attention heads in ba sing se
        masks = masks.repeat(8,1,1)

        return masks

    def decode_trajectory(self, scene_features, scene_feature_masks, ego_subgoal):
        """
        Use deterministic decoder to predict trajectory from subgoal
        """
        scene_features, scene_type_embedding, scene_pos_enc, scene_pos = scene_features
        batch_size = scene_features.shape[0]

        ego_subgoal_abs = self.whitener.untransform_features(ego_subgoal)

        # Subgoal features
        ego_subgoal = ego_subgoal.reshape(ego_subgoal.shape[0],3)
        subgoal_features = self.subgoal_encoder(ego_subgoal)[:,None] # (B,1,D)
        subgoal_type_embedding = self.extended_type_embedding(
            torch.as_tensor([0], device=ego_subgoal.device))[None].repeat(batch_size,1,1)
        subgoal_masks = torch.zeros(subgoal_features.shape[:-1], dtype=bool, device=ego_subgoal.device)
        subgoal_pos = ego_subgoal_abs.reshape(batch_size,1,3)[...,:2]
        subgoal_pos_enc = self.rel_pos_enc(subgoal_pos)

        # Cross-attend to scene
        for layer in self.trajectory_decoder_scene_attention_layers:
            subgoal_features, _ = layer(
                subgoal_features, subgoal_masks,
                scene_features, scene_feature_masks,
                seq1_pos=subgoal_pos_enc, seq2_pos=scene_pos_enc,
                seq1_sem_pos=subgoal_type_embedding, seq2_sem_pos=scene_type_embedding,
            )

        pred_trajectory = self.trajectory_decoder_mlp(subgoal_features)

        return pred_trajectory
        
    def visualize_noisy_samples(self, features, ego_gt_subgoal):
        self.set_sampling_steps(T=32, sigma_max=1, sigma_min=2e-3, rho=7, strategy='edm', freeze_t=0, freeze_steps=0)
        sigmas = torch.as_tensor([0.01, 0.1, 1.0, 10.0, 80.0], device=features['trajectory'].data.device)
        frames = []
        for sigma in sigmas:
            ego_gt_subgoal = features['trajectory'].data.clone()[:,4]
            ego_gt_subgoal = self.standardizer.transform_features(ego_gt_subgoal)
            ego_gt_subgoal_whitened = self.whitener.transform_features(ego_gt_subgoal)
            sigma = sigma.unsqueeze(0)[:,None]

            sigma_frames = []
            for _ in range(10):
                ego_noisy_subgoal_whitened = torch.normal(ego_gt_subgoal_whitened, sigma)
                c_in = 1 / torch.sqrt(sigma**2 + self.sigma_data**2)
                ego_noisy_subgoal_whitened = ego_noisy_subgoal_whitened * c_in
                ego_noisy_subgoal = self.whitener.untransform_features(ego_noisy_subgoal_whitened)

                ego_noisy_subgoal = self.standardizer.untransform_features(ego_noisy_subgoal)
                ego_noisy_subgoal = ego_noisy_subgoal.reshape(-1,3)

                frame = get_raster_from_vector_map_with_new_agents(
                    features['vector_set_map'].to_device('cpu'),
                    features['agent_history'].to_device('cpu'),
                    goal=ego_noisy_subgoal.cpu().numpy(),
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
            ego_noisy_subgoal = torch.normal(torch.zeros_like(ego_gt_subgoal), torch.ones_like(sigma))
            ego_noisy_subgoal = self.whitener.untransform_features(ego_noisy_subgoal)
            ego_noisy_subgoal = self.standardizer.untransform_features(ego_noisy_subgoal)
            ego_noisy_subgoal = ego_noisy_subgoal.reshape(-1,16,3)

            frame = get_raster_from_vector_map_with_new_agents(
                features['vector_set_map'].to_device('cpu'),
                features['agent_history'].to_device('cpu'),
                goal=ego_noisy_subgoal.cpu().numpy(),
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