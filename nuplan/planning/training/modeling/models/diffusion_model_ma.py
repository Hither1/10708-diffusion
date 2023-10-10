"""
Significant parts of the diffusion model code are from here: 
https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL/blob/master/agents/diffusion.py
and https://arxiv.org/pdf/2206.00364
"""

from typing import List, Optional, cast
import math

import torch
from torch import nn
from torch.distributions.normal import Normal
from torch.func import jacrev, vmap
import numpy as np
from omegaconf import OmegaConf

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.models.lanegcn_utils import (
    Actor2ActorAttention,
    Actor2LaneAttention,
    Lane2ActorAttention,
    LaneNet,
    LinearWithGroupNorm,
)
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.feature_builders.agents_feature_builder import AgentsFeatureBuilder
from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import (
    LaneOnRouteStatusData,
    LaneSegmentTrafficLightData,
)
from nuplan.planning.training.preprocessing.feature_builders.vector_map_feature_builder import VectorMapFeatureBuilder
from nuplan.planning.training.preprocessing.features.agents import Agents
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.features.trajectories import Trajectories
from nuplan.planning.training.preprocessing.features.vector_map import VectorMap
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (
    EgoTrajectoryTargetBuilder,
)
from nuplan.planning.training.preprocessing.feature_builders.ego_trajectory_feature_builder import EgoTrajectoryFeatureBuilder
from nuplan.planning.training.preprocessing.target_builders.agent_trajectory_target_builder import AgentTrajectoryTargetBuilder
from nuplan.planning.training.modeling.models.diffusion_utils import (
    vp_beta_schedule, 
    linear_beta_schedule, 
    SinusoidalPosEmb, 
    PCAWhitener,
    Whitener
)
from nuplan.planning.training.modeling.models.transformer_model import TransformerModel
from nuplan.planning.script.builders.model_builder import build_torch_module_wrapper
from nuplan.planning.training.modeling.lightning_module_wrapper import LightningModuleWrapper


def convert_predictions_to_trajectory(predictions: torch.Tensor) -> torch.Tensor:
    """
    Convert predictions tensor to Trajectory.data shape
    :param predictions: tensor from network
    :return: data suitable for Trajectory
    """
    num_batches = predictions.shape[0]
    return predictions.view(num_batches, -1, Trajectory.state_size())


class DiffusionModelMA(TorchModuleWrapper):
    """
    Vector-based model that uses a series of MLPs to encode ego and agent signals, a lane graph to encode vector-map
    elements and a fusion network to capture lane & agent intra/inter-interactions through attention layers.
    Dynamic map elements such as traffic light status and ego route information are also encoded in the fusion network.

    Implementation of the original LaneGCN paper ("Learning Lane Graph Representations for Motion Forecasting").
    """

    def __init__(
        self,
        feature_dim: int,
        vector_map_feature_radius: int,
        vector_map_connection_scales: Optional[List[int]],

        past_trajectory_sampling: TrajectorySampling,
        future_trajectory_sampling: TrajectorySampling,

        T: int = 64,
        sigma_data: float = 1.0,
        sigma_min: float = 2e-3,
        sigma_max: float = 8e1,
        rho: float = 7,

        p_mean: float = -1.2,
        p_std: float = 1.2,

        use_pca: bool = True,
        params_path: str = '',
        pca_params_path: str = '',
        k: int = 10,

        predictions_per_sample: int = 4
    ):
        """
        :param feature_dim: hidden layer dimension
        :param vector_map_feature_radius: The query radius scope relative to the current ego-pose.
        :param vector_map_connection_scales: The hops of lane neighbors to extract, default 1 hop
        :param past_trajectory_sampling: Sampling parameters for past trajectory
        :param future_trajectory_sampling: Sampling parameters for future trajectory
        """
        super().__init__(
            feature_builders=[
                VectorMapFeatureBuilder(
                    radius=vector_map_feature_radius,
                    connection_scales=vector_map_connection_scales,
                ),
                AgentsFeatureBuilder(trajectory_sampling=past_trajectory_sampling)
            ],
            target_builders=[
                EgoTrajectoryTargetBuilder(future_trajectory_sampling=future_trajectory_sampling),
                AgentTrajectoryTargetBuilder(
                    trajectory_sampling=past_trajectory_sampling, 
                    future_trajectory_sampling=future_trajectory_sampling
                )
            ],
            future_trajectory_sampling=future_trajectory_sampling,
        )

        self.past_trajectory_sampling = past_trajectory_sampling
        self.future_trajectory_sampling = future_trajectory_sampling
        self.H = future_trajectory_sampling.num_poses
        self.k = k
        self.feature_dim = feature_dim

        # +1 on input dim for both agents and ego to include both history and current steps
        self.ego_input_dim = (past_trajectory_sampling.num_poses + 1) * Agents.ego_state_dim()
        self.agent_input_dim = (past_trajectory_sampling.num_poses + 1) * Agents.agents_states_dim()
        self.output_dim = future_trajectory_sampling.num_poses * Trajectory.state_size()

        # Encoder layers
        self.ego_encoder = nn.Linear(self.ego_input_dim, self.feature_dim)
        self.agent_encoder = nn.Linear(self.agent_input_dim, self.feature_dim)
        self.lane_encoder = nn.Linear(8, self.feature_dim)
        
        self.trajectory_encoder = nn.Linear(self.k if use_pca else 3, self.feature_dim)
        if not use_pca:
            self.trajectory_time_embeddings = nn.Embedding(self.H, self.feature_dim)

        # Type embeddings
        self.segment_embeddings = nn.Embedding(4, self.feature_dim) # ego, vehicle, lane, trajectory

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=8,
            batch_first=True
        )
        self.transformer_encoder_layers = nn.TransformerEncoder(
            encoder_layer,
            num_layers=8,
        )
        # decoder_layer = nn.TransformerDecoderLayer(
        #     d_model=self.feature_dim,
        #     nhead=8,
        #     batch_first=True
        # )
        # self.transformer_decoder_layers = nn.TransformerDecoder(
        #     decoder_layer,
        #     num_layers=8
        # )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.k if use_pca else 3)
        )

        # Diffusion parameters / noise schedule
        self.T = T
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

        self.p_mean = p_mean
        self.p_std = p_std

        # Compute discretized timesteps
        sigmas = (
            self.sigma_max ** (1 / self.rho) + \
            (torch.arange(self.T+1) / (self.T-1)) * \
            (self.sigma_min**(1 / self.rho) - self.sigma_max**(1 / self.rho)) \
        )**self.rho
        sigmas[-1] = 0
        self.register_buffer('ts', sigmas) # TODO: implement other schedules, this assumes linear

        self.predictions_per_sample = predictions_per_sample

        # Noise level encoder
        self.sigma_encoder = nn.Sequential(
            SinusoidalPosEmb(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim * 2),
            nn.ReLU(),
            nn.Linear(self.feature_dim * 2, self.feature_dim)
        )

        # Whitening
        self.use_pca = use_pca
        if use_pca:
            self.whitener = PCAWhitener(k, pca_params_path)
        else:
            self.whitener = Whitener(params_path)

        # Weight initialization
        self.apply(self._init_weights)

        # Load transformer weights
        # TODO: make this configurable
        config_path = '/zfsauton/datasets/ArgoRL/brianyan/nuplan_exp/transformer/2023.04.04.03.11.49/code/hydra/config.yaml'
        checkpoint_path = '/zfsauton/datasets/ArgoRL/brianyan/nuplan_exp/transformer/2023.04.04.03.11.49/best_model/epoch=65-step=132329.ckpt'
        model_config = OmegaConf.load(config_path)
        torch_module_wrapper = build_torch_module_wrapper(model_config.model)
        model = LightningModuleWrapper.load_from_checkpoint(
            checkpoint_path, model=torch_module_wrapper
        ).model

        self.ego_encoder.load_state_dict(model.ego_encoder.state_dict())
        self.agent_encoder.load_state_dict(model.agent_encoder.state_dict())
        self.lane_encoder.load_state_dict(model.lane_encoder.state_dict())
        self.transformer_encoder_layers.load_state_dict(model.transformer_encoder_layers.state_dict())

    def sigma(self, t):
        return t

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Predict
        :param features: input features containing
                        {
                            "vector_map": VectorMap,
                            "agents": Agents,
                        }
        :return: targets: predictions from network
                        {
                            "trajectory": Trajectory,
                        }
        """
        # Recover features
        vector_map_data = cast(VectorMap, features["vector_map"])
        ego_agent_features = cast(Agents, features["agents"])
        batch_size = ego_agent_features.batch_size

        # agent_trajectories = features['agent_trajectories'].data[0].cpu().numpy().reshape(-1,3)
        # agent_histories = features['agents'].agents[0].cpu().numpy().reshape(-1,8)
        # import matplotlib.pyplot as plt
        # fig = plt.figure(figsize=(10,10))
        # plt.scatter(agent_trajectories[:,0], agent_trajectories[:,1], color='blue')
        # plt.scatter(agent_histories[:,0], agent_histories[:,1], color='red')
        # plt.savefig('/zfsauton2/home/brianyan/nuplan-devkit/test.png')

        # Compute scene features (once only)
        scene_features, scene_feature_masks = self.encode_scene_features(ego_agent_features, vector_map_data)

        if self.training:
            # Preprocess GT trajectories
            assert 'trajectory' in features
            assert 'agent_trajectories' in features

            ego_gt_trajectory = features['trajectory'].data.clone()
            ego_gt_trajectory = self.whitener.transform_features(ego_gt_trajectory)

            agent_gt_trajectory = [
                features['agent_trajectories'].data[idx] \
                .permute(1,0,2).reshape(-1,self.H*3) \
                for idx in range(batch_size)]
            agent_gt_trajectory, agent_gt_trajectory_masks = pad_and_concatenate(agent_gt_trajectory)

            # Sample noisy trajectories
            sigma = (torch.randn(batch_size, device=ego_gt_trajectory.device) * self.p_std + self.p_mean).exp()[:,None]
            ego_noisy_trajectory = torch.normal(ego_gt_trajectory, sigma)
            agent_noisy_trajectory = torch.normal(agent_noisy_trajectory, sigma)

            # Predict denoised trajectory
            predictions = self.denoise_with_preconditioning(
                ego_noisy_trajectory,
                sigma,
                scene_features,
                scene_feature_masks
            )
            predictions = self.whitener.untransform_features(predictions)
            loss_weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data)**2
            loss_weight = loss_weight.clamp(0,1000) # TODO: lol
            return {
                "trajectory": Trajectory(data=convert_predictions_to_trajectory(predictions)),
                "loss_weight": loss_weight
            }
        else:
            # Multiple predictions per sample
            predictions_per_sample = self.predictions_per_sample
            scene_features = scene_features.repeat_interleave(predictions_per_sample,0)
            scene_feature_masks = scene_feature_masks.repeat_interleave(predictions_per_sample,0)

            # Sampling / inference
            trajectory_feature_size = self.k if self.use_pca else self.H * 3
            ego_trajectory = torch.randn(batch_size * predictions_per_sample, trajectory_feature_size, device=scene_features.device)
            log_prob = Normal(0,1).log_prob(ego_trajectory).sum(dim=-1)
            ego_trajectory = ego_trajectory * self.sigma(self.ts[0])

            # Heun's 2nd order method (algorithm 1): https://arxiv.org/pdf/2206.00364
            for i in range(self.T):
                t = self.ts[i]
                t_next = self.ts[i+1]
                sigma = self.sigma(t)[None,None].repeat(batch_size*predictions_per_sample,1)
                sigma_next = self.sigma(t_next)[None,None].repeat(batch_size*predictions_per_sample,1)

                denoised = self.denoise_with_preconditioning(
                    ego_trajectory,
                    sigma,
                    scene_features,
                    scene_feature_masks
                )
                d_i = (1 / sigma) * (ego_trajectory - denoised)
                ego_trajectory_next = ego_trajectory + (t_next - t) * d_i
                
                # Apply second order correction
                if (sigma_next != 0).all():
                    denoised_next = self.denoise_with_preconditioning(
                        ego_trajectory_next, 
                        sigma_next,
                        scene_features,
                        scene_feature_masks
                    )
                    d_ip = (1 / sigma_next) * (ego_trajectory_next - denoised_next)
                    ego_trajectory_next = ego_trajectory + (t_next - t) * 0.5 * (d_i + d_ip)
                
                ego_trajectory = ego_trajectory_next

                # Same for log probability
                eps = torch.randn_like(ego_trajectory)
                with torch.enable_grad():
                    f = lambda x: (-1 / sigma) * (self.denoise_with_preconditioning(x,sigma,scene_features,scene_feature_masks) - x)
                    ego_trajectory.requires_grad_(True)
                    f_eps = torch.sum(f(ego_trajectory) * eps)
                    grad_f_eps = torch.autograd.grad(f_eps, ego_trajectory)[0]
                ego_trajectory.requires_grad_(False)
                out = torch.sum(grad_f_eps * eps, dim=tuple(range(1, len(ego_trajectory.shape))))
                log_prob = log_prob - (t_next - t) * out

            ego_trajectory = self.whitener.untransform_features(ego_trajectory)
            ego_trajectory = ego_trajectory.reshape(batch_size, predictions_per_sample, self.output_dim)
            log_prob = log_prob.reshape(batch_size, predictions_per_sample)
            log_prob = log_prob - log_prob.max(dim=1, keepdim=True).values
            prob = log_prob.exp()
            prob = prob / prob.sum(dim=1, keepdim=True)
            best_trajectory = ego_trajectory[range(batch_size), prob.argmax(dim=1)]

            return {
                "trajectory": Trajectory(data=convert_predictions_to_trajectory(best_trajectory)),
                "multimodal_trajectories": ego_trajectory,
                "probabilities": prob
            }

    def encode_scene_features(self, ego_agent_features, vector_map_data):
        # Extract batches
        batch_size = ego_agent_features.batch_size

        # Ego features
        ego_features = torch.stack(ego_agent_features.ego, dim=0)
        ego_features = self.ego_encoder(ego_features.reshape(batch_size,1,-1))
        ego_masks = torch.ones(batch_size, 1, device=ego_features.device, dtype=bool)

        # Agent features
        agent_features = [ego_agent_features.get_flatten_agents_features_in_sample(idx) for idx in range(batch_size)]
        agent_features, agent_masks = pad_and_concatenate(agent_features)
        agent_features = self.agent_encoder(agent_features)

        # Lane features
        lane_centers = [vector_map_data.coords[i].mean(dim=1) for i in range(batch_size)]
        lane_lights = [vector_map_data.traffic_light_data[i] for i in range(batch_size)]
        lane_route = [vector_map_data.on_route_status[i] for i in range(batch_size)]

        lane_centers, lane_masks = pad_and_concatenate(lane_centers)
        lane_lights, _ = pad_and_concatenate(lane_lights)
        lane_route, _ = pad_and_concatenate(lane_route)
        lane_features = torch.cat([lane_centers, lane_lights, lane_route], dim=2)
        lane_features, lane_masks = lane_features[:,::20], lane_masks[:,::20]   # Subsample lane points
        lane_features = self.lane_encoder(lane_features)

        # Add segment encodings
        ego_features += self.segment_embeddings(torch.as_tensor([0], device=ego_features.device))[None]
        agent_features += self.segment_embeddings(torch.as_tensor([1], device=ego_features.device))[None]
        lane_features += self.segment_embeddings(torch.as_tensor([2], device=ego_features.device))[None]

        # Concat all features
        all_features = torch.cat([ego_features, agent_features, lane_features], dim=1)
        all_masks = torch.cat([ego_masks, agent_masks, lane_masks], dim=1)

        # # Transformers !!
        # all_features = self.transformer_encoder_layers(all_features, src_key_padding_mask=~all_masks)
        return all_features, all_masks
        
    def denoise_with_preconditioning(self, ego_trajectory, sigma, scene_features, scene_feature_masks):
        """
        Denoise ego_trajectory with noise level sigma
        Equivalent to evaluating D_theta in this paper: https://arxiv.org/pdf/2206.00364
        Returns denoised trajectory, not the residual noise
        """
        c_in = 1 / torch.sqrt(sigma**2 + self.sigma_data**2)
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / torch.sqrt(sigma**2 + self.sigma_data**2)
        c_noise = .25 * torch.log(sigma)
        out = \
            c_skip * ego_trajectory + \
            c_out * self.denoise( 
                c_in * ego_trajectory, 
                c_noise,
                scene_features,
                scene_feature_masks
        )
        return out

    def denoise(self, ego_trajectory, sigma, scene_features, scene_feature_masks):
        """
        Denoise ego_trajectory with noise level sigma (no preconditioning)
        Equivalent to evaluating F_theta in this paper: https://arxiv.org/pdf/2206.00364
        """
        # Trajectory features
        if not self.use_pca:
            ego_trajectory = ego_trajectory.reshape(ego_trajectory.shape[0],self.H,3)
            trajectory_features = self.trajectory_encoder(ego_trajectory)
            trajectory_features += self.trajectory_time_embeddings(torch.arange(self.H, device=ego_trajectory.device))
        else:
            trajectory_features = self.trajectory_encoder(ego_trajectory)[:,None]
        trajectory_masks = torch.ones(trajectory_features.shape[:-1], dtype=bool, device=trajectory_features.device)

        # Add segment encodings
        trajectory_features += self.segment_embeddings(torch.as_tensor([3], device=scene_features.device))[None]

        # Concat all features
        all_features = torch.cat([scene_features, trajectory_features], dim=1)
        all_masks = torch.cat([scene_feature_masks, trajectory_masks], dim=1)

        # Add diffusion noise encodings
        sigma_embeddings = self.sigma_encoder(sigma)
        all_features += sigma_embeddings

        # Transformers !!
        all_features = self.transformer_encoder_layers(all_features, src_key_padding_mask=~all_masks)

        if self.use_pca:
            trajectory_features = all_features[:,-1:]
        else:
            trajectory_features = all_features[:,-self.H:]
    
        out = self.decoder(trajectory_features).reshape(trajectory_features.shape[0],-1)
        return out


def pad_and_concatenate(features):
    """
    Takes list of K features, each shape (N_i, d)
    Concatenates features with padding; final output shape is (K, max_i N_i, d)
    """
    batch_size = len(features)
    device = features[0].device
    num_features = max([feature.shape[0] for feature in features])
    masks = []
    for i in range(batch_size):
        N_i, d = features[i].shape
        padding = torch.zeros(num_features - N_i, d, device=device)
        features[i] = torch.cat([features[i], padding], dim=0)
        mask = torch.zeros(num_features, device=device, dtype=bool)
        mask[:N_i] = True
        masks.append(mask)
        
    features = torch.stack(features, dim=0)
    masks = torch.stack(masks, dim=0)
    return features, masks
