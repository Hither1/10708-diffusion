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
import numpy as np

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.models.urban_driver_open_loop_model_utils import (
    LocalSubGraph,
    MultiheadAttentionGlobalHead,
    SinusoidalPositionalEmbedding,
    TypeEmbedding,
    pad_avails,
    pad_polylines,
)
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.feature_builders.generic_agents_feature_builder import (
    GenericAgentsFeatureBuilder,
)
from nuplan.planning.training.preprocessing.feature_builders.vector_set_map_feature_builder import (
    VectorSetMapFeatureBuilder,
)
from nuplan.planning.training.preprocessing.features.generic_agents import GenericAgents
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.features.vector_set_map import VectorSetMap
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (
    EgoTrajectoryTargetBuilder,
)
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.lightning_module_wrapper import LightningModuleWrapper
from nuplan.planning.script.builders.model_builder import build_torch_module_wrapper

from nuplan.planning.training.modeling.models.urban_driver_open_loop_model import UrbanDriverOpenLoopModel, convert_predictions_to_trajectory


class BehaviorTransformer(UrbanDriverOpenLoopModel):
    def __init__(
        self,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.kmeans = KMeans()
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=256, nhead=8, batch_first=False), 
            num_layers=2
        )
        self.bin_embeddings = nn.Embedding(32, 256)
        self.bin_classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.offset_mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 48)
        )

        del self.global_head
        

    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Predict
        :param features: input features containing
                        {
                            "vector_set_map": VectorSetMap,
                            "generic_agents": GenericAgents,
                        }
        :return: targets: predictions from network
                        {
                            "trajectory": Trajectory,
                        }
        """
        # Recover features
        vector_set_map_data = cast(VectorSetMap, features["vector_set_map"])
        ego_agent_features = cast(GenericAgents, features["generic_agents"])
        batch_size = ego_agent_features.batch_size

        # Extract features across batch
        agent_features, agent_avails = self.extract_agent_features(ego_agent_features, batch_size)
        map_features, map_avails = self.extract_map_features(vector_set_map_data, batch_size)
        all_features = torch.cat([agent_features, map_features], dim=1)
        avails = torch.cat([agent_avails, map_avails], dim=1)

        # embed inputs
        feature_embedding = self.feature_embedding(all_features)

        # calculate positional embedding, then transform [num_points, 1, feature_dim] -> [1, 1, num_points, feature_dim]
        pos_embedding = self.positional_embedding(all_features).unsqueeze(0).transpose(1, 2)

        # invalid mask
        invalid_mask = ~avails
        invalid_polys = invalid_mask.all(-1)

        # local subgraph
        embeddings = self.local_subgraph(feature_embedding, invalid_mask, pos_embedding)
        if hasattr(self, "global_from_local"):
            embeddings = self.global_from_local(embeddings)
        embeddings = F.normalize(embeddings, dim=-1) * (self._model_params.global_embedding_size**0.5)
        embeddings = embeddings.transpose(0, 1)

        type_embedding = self.type_embedding(
            batch_size,
            self._feature_params.max_agents,
            self._feature_params.agent_features,
            self._feature_params.map_features,
            self._feature_params.max_elements,
            device=all_features.device,
        ).transpose(0, 1)

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

        # global attention layers (transformer)
        # outputs, attns = self.global_head(embeddings, type_embedding, invalid_polys)
        query_features = embeddings[[0]] + self.bin_embeddings(torch.arange(32, device=embeddings.device))[:,None]
        query_features = self.transformer_decoder(
            query_features, embeddings + type_embedding, memory_key_padding_mask=invalid_polys
        )
        offsets = self.offset_mlp(query_features).permute(1,0,2)
        bin_logits = self.bin_classifier(query_features).squeeze(dim=-1).permute(1,0)
        bin_idxs = bin_logits.argmax(dim=1)

        outputs = self.kmeans.decode(bin_idxs, offsets)

        return {
            "trajectory": Trajectory(data=convert_predictions_to_trajectory(outputs)),
            "offsets": offsets,
            "logits": bin_logits,
            "kmeans": self.kmeans
        }


class KMeans(nn.Module):
    def __init__(self):
        super().__init__()
        centers = np.load('/zfsauton/datasets/ArgoRL/brianyan/cluster_centers.npy') # TODO: magic string
        centers = torch.from_numpy(centers).float()
        self.register_buffer('centers', centers)

    def encode(self, trajectory):
        """
        Returns one-hot action bin and continuous action offset
        """
        dists = (self.centers[None] - trajectory[:,None]).norm(dim=-1)
        center_idxs = torch.argmin(dists, dim=1)
        return center_idxs, trajectory - self.centers[center_idxs]

    def decode(self, center_idx, offset):
        """
        Returns continuous trajectory given center_idx and continuous offset
        """
        batch_size = center_idx.shape[0]
        return self.centers[center_idx] + offset[range(batch_size),center_idx]
