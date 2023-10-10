from typing import List, Optional, cast

import torch
from torch import nn

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
from nuplan.planning.training.preprocessing.feature_builders.vector_set_map_feature_builder import VectorSetMapFeatureBuilder
from nuplan.planning.training.preprocessing.features.agents import Agents
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.features.vector_map import VectorMap
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (
    EgoTrajectoryTargetBuilder,
)


def convert_predictions_to_trajectory(predictions: torch.Tensor) -> torch.Tensor:
    """
    Convert predictions tensor to Trajectory.data shape
    :param predictions: tensor from network
    :return: data suitable for Trajectory
    """
    num_batches = predictions.shape[0]
    return predictions.view(num_batches, -1, Trajectory.state_size())


class TransformerModel(TorchModuleWrapper):
    """
    Vector-based model that uses a series of MLPs to encode ego and agent signals, a lane graph to encode vector-map
    elements and a fusion network to capture lane & agent intra/inter-interactions through attention layers.
    Dynamic map elements such as traffic light status and ego route information are also encoded in the fusion network.

    Implementation of the original LaneGCN paper ("Learning Lane Graph Representations for Motion Forecasting").
    """

    def __init__(
        self,
        map_net_scales: int,
        num_res_blocks: int,
        num_attention_layers: int,
        a2a_dist_threshold: float,
        l2a_dist_threshold: float,
        num_output_features: int,
        feature_dim: int,
        vector_map_feature_radius: int,
        vector_map_connection_scales: Optional[List[int]],
        past_trajectory_sampling: TrajectorySampling,
        future_trajectory_sampling: TrajectorySampling,
    ):
        """
        :param map_net_scales: Number of scales to extend the predecessor and successor lane nodes.
        :param num_res_blocks: Number of residual blocks for the GCN (LaneGCN uses 4).
        :param num_attention_layers: Number of times to repeatedly apply the attention layer.
        :param a2a_dist_threshold: [m] distance threshold for aggregating actor-to-actor nodes
        :param l2a_dist_threshold: [m] distance threshold for aggregating map-to-actor nodes
        :param num_output_features: number of target features
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
            target_builders=[EgoTrajectoryTargetBuilder(future_trajectory_sampling=future_trajectory_sampling)],
            future_trajectory_sampling=future_trajectory_sampling,
        )

        # LaneGCN components
        self.feature_dim = feature_dim

        # +1 on input dim for both agents and ego to include both history and current steps
        self.ego_input_dim = (past_trajectory_sampling.num_poses + 1) * Agents.ego_state_dim()
        self.agent_input_dim = (past_trajectory_sampling.num_poses + 1) * Agents.agents_states_dim()

        # Encoder layers
        self.ego_encoder = nn.Linear(self.ego_input_dim, self.feature_dim)
        self.agent_encoder = nn.Linear(self.agent_input_dim, self.feature_dim)
        self.lane_encoder = nn.Linear(8, self.feature_dim)

        # Type embeddings
        self.segment_embeddings = nn.Embedding(3, self.feature_dim)

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

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, num_output_features),
        )

        # Weight initialization
        self.apply(self._init_weights)

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
        ego_past_trajectory = ego_agent_features.ego  # batch_size x num_frames x 3

        # Extract batches
        batch_size = ego_agent_features.batch_size

        # Ego features
        ego_features = torch.stack(ego_past_trajectory, dim=0)
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

        # Transformers !!
        all_features = torch.cat([ego_features, agent_features, lane_features], dim=1)
        all_masks = torch.cat([ego_masks, agent_masks, lane_masks], dim=1)
        all_features = self.transformer_encoder_layers(all_features, src_key_padding_mask=~all_masks)

        # Decoder
        predictions = self.decoder(all_features[:,:1])
        return {"trajectory": Trajectory(data=convert_predictions_to_trajectory(predictions))}

        # # Viz
        # data = lane_centers[10].cpu().numpy()[::10]
        # import matplotlib.pyplot as plt
        # plt.scatter(data[:,0], data[:,1], s=2)
        # plt.xlim(-50,50)
        # plt.ylim(-50,50)
        # plt.savefig('/zfsauton2/home/brianyan/nuplan-devkit/test.png')

        # import pdb; pdb.set_trace()
        # raise


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
        padding = torch.zeros(num_features - features[i].shape[0], features[i].shape[1], device=device)
        features[i] = torch.cat([features[i], padding], dim=0)
        mask = torch.zeros(num_features, device=device, dtype=bool)
        mask[:features[i].shape[0]] = True
        masks.append(mask)
        
    features = torch.stack(features, dim=0)
    masks = torch.stack(masks, dim=0)
    return features, masks