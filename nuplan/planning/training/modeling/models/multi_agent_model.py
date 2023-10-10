"""
Copyright 2022 Motional

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple, cast
import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import OmegaConf
from einops import rearrange

from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.lightning_module_wrapper import LightningModuleWrapper
from nuplan.planning.script.builders.model_builder import build_torch_module_wrapper
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.feature_builders.agent_history_feature_builder import (
    AgentHistoryFeatureBuilder,
)
from nuplan.planning.training.preprocessing.feature_builders.vector_set_map_feature_builder import (
    VectorSetMapFeatureBuilder,
)
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.features.agents_trajectory import AgentTrajectory
from nuplan.planning.training.preprocessing.features.agent_history import AgentHistory
from nuplan.planning.training.preprocessing.features.vector_set_map import VectorSetMap
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (
    EgoTrajectoryTargetBuilder,
)
from nuplan.planning.training.preprocessing.target_builders.agent_trajectory_target_builder import (
    AgentTrajectoryTargetBuilder
)
from nuplan.planning.training.modeling.models.encoders import (
    AgentEncoder, 
    MapEncoder,
    CrossModalEncoder,
    TrajectoryEncoder
)
from nuplan.planning.training.modeling.models.encoder_decoder_layers import ParallelAttentionLayer
from .positional_embeddings import VolumetricPositionEncoding as VolPE


def convert_deltas_to_trajectory(predictions: torch.Tensor) -> torch.Tensor:
    # TODO: implement this
    return predictions


def convert_predictions_to_trajectory(predictions: torch.Tensor) -> torch.Tensor:
    """
    Convert predictions tensor to Trajectory.data shape
    :param predictions: tensor from network
    :return: data suitable for Trajectory
    """
    num_batches = predictions.shape[0]
    return predictions.view(num_batches, -1, Trajectory.state_size())


def convert_relative_to_absolute_trajectory(trajectories: torch.Tensor, agent_history: AgentHistory):
    B, T, A = trajectories.shape[:3]
    ref = agent_history.data[:,-1,:,:3]
    ref = ref[None].repeat(T,1,1,1)
    ref = rearrange(ref, 't b a d -> (b a t) d')
    trajectories = rearrange(trajectories, 'b t a d -> (b a t) d')
    
    ref_xy, ref_heading = ref[...,:2], ref[...,2]
    R = torch.stack([
        torch.stack([torch.cos(ref_heading), -torch.sin(ref_heading)], dim=-1),
        torch.stack([torch.sin(ref_heading),  torch.cos(ref_heading)], dim=-1)
    ], dim=-2)

    # Rotate + translate to agent frame
    rotated_xy = torch.bmm(R, trajectories[...,:2,None]).squeeze(-1)
    transformed_xy = rotated_xy + ref[:,:2]
    transformed_heading = ref_heading + trajectories[...,2]

    transformed_trajectories = torch.cat([transformed_xy, transformed_heading[...,None]], dim=-1)
    transformed_trajectories = rearrange(transformed_trajectories, '(b a t) d -> b t a d', b=B, a=A, t=T)

    return transformed_trajectories


class MultiAgentModel(TorchModuleWrapper):
    def __init__(
        self,
        # Agent params
        max_agents: int = 30,
        # Map params
        map_features: List[str] = None,
        max_map_elements: Dict[str, int] = None,
        max_map_points: Dict[str, int] = None,
        vector_set_map_feature_radius: int = 35,
        interpolation_method: str = 'linear',
        total_max_points: int = 30,
        # Model params
        embedding_dim: int = 128,
        rotary_pe: bool = True,
        # Trajectory params
        past_trajectory_sampling: TrajectorySampling = None,
        future_trajectory_sampling: TrajectorySampling = None,
    ):
        super().__init__(
            feature_builders=[
                AgentHistoryFeatureBuilder(
                    trajectory_sampling=past_trajectory_sampling,
                    max_agents=max_agents
                ),
                VectorSetMapFeatureBuilder(
                    map_features=map_features,
                    max_elements=max_map_elements,
                    max_points=max_map_points,
                    radius=vector_set_map_feature_radius,
                    interpolation_method=interpolation_method
                )
            ],
            target_builders=[
                EgoTrajectoryTargetBuilder(future_trajectory_sampling),
                AgentTrajectoryTargetBuilder(
                    trajectory_sampling=past_trajectory_sampling,
                    future_trajectory_sampling=future_trajectory_sampling,
                    max_agents=max_agents
                )
            ],
            future_trajectory_sampling=future_trajectory_sampling,
        )

        # Encoder
        self.agent_encoder = AgentEncoder(
            embedding_dim=embedding_dim,
            history_length=past_trajectory_sampling.num_poses+1,
            num_transformer_layers=2,
            dropout=0.1,
            rotary_pe=rotary_pe
        )
        self.map_encoder = MapEncoder(
            embedding_dim=embedding_dim,
            map_features=map_features,
            total_max_points=total_max_points,
            num_transformer_layers=2,
            dropout=0.1,
            rotary_pe=rotary_pe
        )
        self.cross_modal_encoder = CrossModalEncoder(
            embedding_dim=embedding_dim,
            num_transformer_layers=2,
            dropout=0.1,
            rotary_pe=rotary_pe
        )

        # # Decoder
        # self.decoder = DiffusionDecoder(
        #     embedding_dim=embedding_dim,
        #     trajectory_shape=(future_trajectory_sampling.num_poses, Trajectory.state_size()),
        # )

    def forward(self, features):
        # agent_features, agent_masks, agent_positions = self.agent_encoder(features['agent_history'])
        # map_features, map_masks, map_positions = self.map_encoder(features['vector_set_map'])
        # scene_features, scene_masks, scene_positions = self.cross_modal_encoder(
        #     agent_features, agent_masks, agent_positions,
        #     map_features,   map_masks,   map_positions
        # )

        # out = self.decoder(
        #     scene_features, 
        #     scene_masks,
        #     features['agent_history']
        # )
        raise NotImplementedError


class MultiAgentDeterministicModel(MultiAgentModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        embedding_dim = kwargs['embedding_dim']

        self.output_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 16 * 3)
        )

    def forward(self, features):
        batch_size = features['agent_history'].batch_size
        num_agents = features['agent_history'].data.shape[2]+1

        agent_features, agent_masks, agent_positions = self.agent_encoder(features['agent_history'])
        map_features, map_masks, map_positions = self.map_encoder(features['vector_set_map'])
        scene_features, scene_masks, scene_positions = self.cross_modal_encoder(
            agent_features, agent_masks, agent_positions,
            map_features,   map_masks,   map_positions
        )

        output_features = scene_features[:,:num_agents]
        predictions = self.output_mlp(output_features)

        # Split ego / agent predictions
        ego_trajectory = predictions[:,:1].reshape(batch_size, 16, 3)
        agent_trajectory = predictions[:,1:num_agents].reshape(
            batch_size, num_agents-1, 16, 3
        ).permute(0,2,1,3)

        # Convert to agent frame
        agent_trajectory = convert_deltas_to_trajectory(agent_trajectory)
        agent_trajectory = convert_relative_to_absolute_trajectory(agent_trajectory, features['agent_history'])
        
        return {
            'trajectory': Trajectory(data=ego_trajectory),
            'agent_trajectories': AgentTrajectory(
                data=agent_trajectory,
                mask=torch.ones_like(agent_trajectory[...,0]).bool()
            )
        }


class MultiAgentDiffusionModel(MultiAgentModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        embedding_dim = kwargs['embedding_dim']
        self.rotary_pe = kwargs['rotary_pe']

        self.trajectory_encoder = TrajectoryEncoder(
            embedding_dim=embedding_dim,
            num_transformer_layers=1
        )

        self.trajectory_to_scene_attention_layers = nn.ModuleList([
            ParallelAttentionLayer(
                d_model=embedding_dim,
                self_attention1=True,
                self_attention2=False,
                cross_attention1=True,
                cross_attention2=False,
                rotary_pe=self.rotary_pe
            ) for _ in range(2)
        ])

        self.output_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 3)
        )

    def forward(self, features):
        batch_size = features['agent_history'].batch_size
        num_agents = features['agent_history'].data.shape[2]+1

        agent_features, agent_masks, agent_positions = self.agent_encoder(features['agent_history'])
        map_features, map_masks, map_positions = self.map_encoder(features['vector_set_map'])
        scene_features, scene_masks, scene_positions = self.cross_modal_encoder(
            agent_features, agent_masks, agent_positions,
            map_features,   map_masks,   map_positions
        )

        # Encode noisy trajectory
        # trajectory_positions = features['agent_trajectories'].data
        # trajectory_positions = self.pos_emb(trajectory_positions)
        # trajectory_mask = features['agent_trajectories'].mask
        trajectory_features, trajectory_mask, trajectory_positions = \
            self.trajectory_encoder(features['trajectory'], features['agent_trajectories']) # (B, 16, A, 384)

        # Trajectory features cross-attend to scene and self-attend (scene features aren't updated)
        # Done in parallel for all agents (can't see noisy futures of other agents)
        trajectory_features = rearrange(trajectory_features, 'b t a d -> (b a) t d')
        trajectory_mask = rearrange(trajectory_mask, 'b t a -> (b a) t')
        trajectory_positions = rearrange(trajectory_positions, 'b t a d f -> (b a) t d f')

        scene_features = scene_features.repeat_interleave(num_agents,dim=0)
        scene_masks = scene_masks.repeat_interleave(num_agents,dim=0)
        scene_positions = scene_positions.repeat_interleave(num_agents,dim=0)

        for layer in self.trajectory_to_scene_attention_layers:
            if self.rotary_pe:
                trajectory_features, _ = layer(
                    trajectory_features, ~trajectory_mask,
                    scene_features, ~scene_masks,
                    seq1_pos=trajectory_positions, seq2_pos=scene_positions
                )
            else:
                trajectory_features, _ = layer(
                    trajectory_features, ~trajectory_mask,
                    scene_features, ~scene_masks
                )

        predictions = self.output_mlp(trajectory_features)
        predictions = rearrange(predictions, '(b a) t d -> b a t d', b=batch_size, a=num_agents)

        # Split ego / agent predictions
        ego_trajectory = predictions[:,:1].reshape(batch_size, 16, 3)
        agent_trajectory = predictions[:,1:num_agents].reshape(
            batch_size, num_agents-1, 16, 3
        ).permute(0,2,1,3)

        # Convert to agent frame
        agent_trajectory = convert_deltas_to_trajectory(agent_trajectory)
        agent_trajectory = convert_relative_to_absolute_trajectory(agent_trajectory, features['agent_history'])
        
        return {
            'trajectory': Trajectory(data=ego_trajectory),
            'agent_trajectories': AgentTrajectory(
                data=agent_trajectory,
                mask=torch.ones_like(agent_trajectory[...,0]).bool()
            )
        }
