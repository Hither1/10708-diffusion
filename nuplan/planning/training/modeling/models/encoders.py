from typing import Dict, List, Tuple, cast
import math

import torch
import torch.nn as nn
from einops import rearrange

from nuplan.planning.training.preprocessing.features.agent_history import AgentHistory
from nuplan.planning.training.preprocessing.features.vector_set_map import VectorSetMap
from nuplan.planning.training.modeling.models.urban_driver_open_loop_model_utils import (
    LocalSubGraph,
    pad_polylines,
    pad_avails
)
from nuplan.planning.training.modeling.models.encoder_decoder_layers import (
    SelfAttentionLayer, 
    ParallelAttentionLayer
)
from .positional_embeddings import VolumetricPositionEncoding as VolPE


class AgentEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        history_length: int,
        num_transformer_layers: int,
        dropout: float = 0.0,
        n_heads: int = 8,
        rotary_pe: bool = True
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.history_length = history_length
        self.rotary_pe = rotary_pe

        self.pos_emb = VolPE(embedding_dim)

        # Modules
        self.feature_emb = nn.Linear(7, embedding_dim)
        self.temporal_encoding = nn.Embedding(history_length, embedding_dim)
        self.spatial_transformer_layers = nn.ModuleList([
            SelfAttentionLayer(embedding_dim, dropout, n_heads, rotary_pe=rotary_pe) for _ in range(num_transformer_layers)])
        self.temporal_transformer_layers = nn.ModuleList([
            SelfAttentionLayer(embedding_dim, dropout, n_heads, rotary_pe=rotary_pe) for _ in range(num_transformer_layers)])

    def forward(self, agent_history: AgentHistory):
        B, A, T = agent_history.batch_size, agent_history.data.shape[2]+1, self.history_length

        # Combine ego and agent features
        agent_features = torch.cat([
            agent_history.ego[:,:,None], 
            agent_history.data
        ], dim=2)
        agent_masks = torch.cat([
            torch.ones(B, T, 1, dtype=torch.bool, device=agent_history.mask.device), 
            agent_history.mask
        ], dim=2)

        agent_features[~agent_masks] = 0
        agent_positions = self.pos_emb(agent_features[...,:3])
        agent_features = self.feature_emb(agent_features)

        # Temporal attention (across time)
        agent_features = rearrange(agent_features, 'b t a d -> (b a) t d')
        agent_masks = rearrange(agent_masks, 'b t a -> (b a) t')
        agent_positions = rearrange(agent_positions, 'b t a f d -> (b a) t f d')
        # Add temporal encodings
        agent_features = agent_features + self.temporal_encoding(
            torch.arange(self.history_length, device=agent_features.device))[None]
        agent_exists = agent_masks.any(dim=1) # need to ignore rows with no agents to prevent nans
        for layer in self.temporal_transformer_layers:
            if self.rotary_pe:
                agent_features[agent_exists] = layer(agent_features[agent_exists], ~agent_masks[agent_exists], agent_positions[agent_exists])
            else:
                agent_features[agent_exists] = layer(agent_features[agent_exists], ~agent_masks[agent_exists])
        
        # Spatial attention (across agents)
        agent_features = rearrange(agent_features, '(b a) t d -> (b t) a d', b=B, a=A)
        agent_masks = rearrange(agent_masks, '(b a) t -> (b t) a', b=B, a=A)
        agent_positions = rearrange(agent_positions, '(b a) t f d -> (b t) a f d', b=B, a=A)
        for layer in self.spatial_transformer_layers:
            if self.rotary_pe:
                agent_features = layer(agent_features, ~agent_masks, agent_positions)
            else:
                agent_features = layer(agent_features, ~agent_masks)

        agent_features = rearrange(agent_features, '(b t) a d -> b t a d', b=B, t=T)
        agent_masks = rearrange(agent_masks, '(b t) a -> b t a', b=B, a=A)
        agent_positions = rearrange(agent_positions, '(b t) a f d -> b t a f d', b=B, t=T)

        # Reduce features
        agent_features = agent_features[:,-1]
        agent_masks = agent_masks[:,-1]
        agent_positions = agent_positions[:,-1]
        
        return agent_features, agent_masks, agent_positions


class MapEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        map_features: List[str],
        total_max_points: int,
        num_transformer_layers: int,
        dropout: float = 0.0,
        n_heads: int = 8,
        rotary_pe: bool = True,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.map_features = map_features
        self.total_max_points = total_max_points
        self.rotary_pe = rotary_pe

        self.feature_dim = 8

        # Modules
        self.feature_emb = nn.Linear(self.feature_dim, embedding_dim)
        # self.pointnet = LocalSubGraph(num_layers=3, dim_in=embedding_dim)
        self.interlane_transformer_layers = nn.ModuleList([
            SelfAttentionLayer(embedding_dim, dropout, n_heads, rotary_pe=rotary_pe) for _ in range(num_transformer_layers)])
        self.intralane_transformer_layers = nn.ModuleList([
            SelfAttentionLayer(embedding_dim, dropout, n_heads, rotary_pe=rotary_pe) for _ in range(num_transformer_layers)])
        self.lane_query_features = nn.Embedding(1, embedding_dim)
        
        self.pos_emb = VolPE(embedding_dim)

    def preprocess_features(self, vector_set_map_data: VectorSetMap) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract map features into format expected by network and build accompanying availability matrix.
        :param vector_set_map_data: VectorSetMap features to be extracted
        :return:
            map_features: <torch.FloatTensor: batch_size, num_elements (polylines) (max_lanes),
                num_points_per_element, feature_dimension>. Stacked map features.
            map_avails: <torch.BoolTensor: batch_size, num_elements (polylines) (max_lanes),
                num_points_per_element>. Bool specifying whether feature is available or zero padded.
        """
        map_features = []  # List[<torch.FloatTensor: max_map_features, total_max_points, feature_dim>: batch_size]
        map_avails = []  # List[<torch.BoolTensor: max_map_features, total_max_points>: batch_size]

        # features have different size across batch so we use per sample feature extraction
        for sample_idx in range(vector_set_map_data.batch_size):

            sample_map_features = []
            sample_map_avails = []

            for feature_name in self.map_features:
                coords = vector_set_map_data.coords[feature_name][sample_idx]
                tl_data = (
                    vector_set_map_data.traffic_light_data[feature_name][sample_idx]
                    if feature_name in vector_set_map_data.traffic_light_data
                    else None
                )
                avails = vector_set_map_data.availabilities[feature_name][sample_idx]

                # add traffic light data if exists for feature
                if tl_data is not None:
                    coords = torch.cat((coords, tl_data), dim=2)

                # maintain fixed number of points per map element (polyline)
                coords = coords[:, : self.total_max_points, ...]
                avails = avails[:, : self.total_max_points]

                if coords.shape[1] < self.total_max_points:
                    coords = pad_polylines(coords, self.total_max_points, dim=1)
                    avails = pad_avails(avails, self.total_max_points, dim=1)

                # maintain fixed number of features per point
                coords = coords[..., : self.feature_dim]
                if coords.shape[2] < self.feature_dim:
                    coords = pad_polylines(coords, self.feature_dim, dim=2)

                sample_map_features.append(coords)
                sample_map_avails.append(avails)

            map_features.append(torch.cat(sample_map_features))
            map_avails.append(torch.cat(sample_map_avails))

        map_features = torch.stack(map_features)
        map_avails = torch.stack(map_avails)

        return map_features, map_avails

    def forward(self, vector_set_map: VectorSetMap):
        map_features, map_mask = self.preprocess_features(vector_set_map)
        map_positions = map_features[...,:2]
        # since we're doing 3D positional embeddings (third dimension is just heading, not z), we need to add fake dimension
        map_positions = torch.cat([map_positions, torch.zeros_like(map_positions[...,:1])], dim=-1)
        map_features = self.feature_emb(map_features)

        # Concatenate learned query features for each lane segment (attention pooling)
        query_features = self.lane_query_features(torch.as_tensor([0], device=map_features.device))
        query_features = query_features[None,None].repeat(map_features.shape[0], map_features.shape[1], 1, 1)
        map_features = torch.cat([map_features, query_features], dim=2)
        query_mask = map_mask.any(dim=2, keepdim=True)
        map_mask = torch.cat([map_mask, query_mask], dim=2)

        # We'll just average the positions for these query features
        query_positions = map_positions.sum(dim=2) / map_mask.sum(dim=2).unsqueeze(-1)
        query_positions[map_mask.sum(dim=2) == 0] = 0
        map_positions = torch.cat([map_positions, query_positions[:,:,None]], dim=2)
        
        map_pos_emb = self.pos_emb(map_positions)

        B, T, A, _ = map_features.shape # TODO: rename these

        # map_features = self.pointnet(map_features, ~map_mask)

        # Intra-lane attention (within lane segments)
        map_features = rearrange(map_features, 'b t a d -> (b t) a d')
        map_mask = rearrange(map_mask, 'b t a -> (b t) a')
        map_pos_emb = rearrange(map_pos_emb, 'b t a f d -> (b t) a f d')

        segment_exists = map_mask.any(dim=1) # need to ignore segments with no points to prevent nans
        for layer in self.intralane_transformer_layers:
            if self.rotary_pe:
                map_features[segment_exists] = layer(map_features[segment_exists], ~map_mask[segment_exists], map_pos_emb[segment_exists])
            else:
                map_features[segment_exists] = layer(map_features[segment_exists], ~map_mask[segment_exists])

        # Inter-lane attention (between lane segments)
        map_features = rearrange(map_features, '(b t) a d -> b t a d', b=B, t=T)
        map_mask = rearrange(map_mask, '(b t) a -> b t a', b=B, t=T)
        map_pos_emb = rearrange(map_pos_emb, '(b t) a f d -> b t a f d', b=B, t=T)

        segment_features = map_features[:,:,-1]
        segment_mask = map_mask[:,:,-1]
        segment_pos_emb = map_pos_emb[:,:,-1]

        for layer in self.interlane_transformer_layers:
            if self.rotary_pe:
                segment_features = layer(segment_features, ~segment_mask, segment_pos_emb)
            else:
                segment_features = layer(segment_features, ~segment_mask)

        return segment_features, segment_mask, segment_pos_emb


class CrossModalEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_transformer_layers: int,
        dropout: float = 0.0,
        n_heads: int = 8,
        rotary_pe: bool = True
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.rotary_pe = rotary_pe

        # Modules
        self.type_encoding = nn.Embedding(3, embedding_dim)
        self.transformer_layers = nn.ModuleList([
            SelfAttentionLayer(embedding_dim, dropout, n_heads, rotary_pe=rotary_pe) for _ in range(num_transformer_layers)])

    def forward(
        self,
        agent_features: torch.Tensor,   # (B, A, D)
        agent_masks: torch.Tensor,      # (B, A)
        agent_positions,
        map_features: torch.Tensor,     # (B, M, D)
        map_masks: torch.Tensor,        # (B, M)
        map_positions
    ):
        B, A, M = agent_features.shape[0], agent_features.shape[1], map_features.shape[1]

        # Add type encodings
        agent_types = torch.cat([
            torch.zeros(B, 1, dtype=torch.long, device=agent_features.device),
            torch.ones(B, A-1, dtype=torch.long, device=agent_features.device)
        ], dim=1)
        agent_features = agent_features + self.type_encoding(
            torch.cat([
                torch.zeros(B, 1, dtype=torch.long, device=agent_features.device),
                torch.ones(B, A-1, dtype=torch.long, device=agent_features.device)
            ], dim=1)
        )
        map_features = map_features + self.type_encoding(
            torch.as_tensor([[2]], dtype=torch.long, device=agent_features.device).repeat(B,M)
        )

        features = torch.cat([agent_features, map_features], dim=1)
        masks = torch.cat([agent_masks, map_masks], dim=1)
        positions = torch.cat([agent_positions, map_positions], dim=1)

        # features = self.transformer_layers(features, src_key_padding_mask=~masks)
        for layer in self.transformer_layers:
            if self.rotary_pe:
                features = layer(features, ~masks, positions)
            else:
                features = layer(features, ~masks)

        return features, masks, positions


class TrajectoryEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_transformer_layers: int,
        dropout: float = 0.0,
        n_heads: int = 8,
        rotary_pe: bool = True
    ):
        super().__init__()

        self.rotary_pe = rotary_pe

        self.pos_emb = VolPE(embedding_dim)

        self.temporal_embedding = nn.Embedding(16, embedding_dim)

        self.transformer_layers = nn.ModuleList([
            SelfAttentionLayer(embedding_dim, dropout, n_heads, rotary_pe=rotary_pe)
            for _ in range(num_transformer_layers)])

    def forward(self, ego_trajectory, agent_trajectories):
        # Combine ego and other agent trajectories and masks
        trajectory = torch.cat([
            ego_trajectory.data[:,:,None],
            agent_trajectories.data
        ], dim=2)
        trajectory_mask = torch.cat([
            torch.zeros_like(agent_trajectories.mask)[:,:,:1],
            agent_trajectories.mask
        ], dim=2)
        
        trajectory_positions = self.pos_emb(trajectory[...,:3])

        B, T, A, _ = trajectory.shape
        
        # Features (non-positional ones) are just the ordering
        trajectory_features = self.temporal_embedding(torch.arange(16, device=trajectory.device))
        trajectory_features = trajectory_features[None,:,None].repeat(B,1,A,1)

        trajectory_features = rearrange(trajectory_features, 'b t a d -> (b a) t d')
        trajectory_mask = rearrange(trajectory_mask, 'b t a -> (b a) t')
        trajectory_positions = rearrange(trajectory_positions, 'b t a d f -> (b a) t d f')

        agent_exists = trajectory_mask.any(dim=1)

        for layer in self.transformer_layers:
            if self.rotary_pe:
                trajectory_features[agent_exists] = layer(
                    trajectory_features[agent_exists], 
                    ~trajectory_mask[agent_exists], 
                    trajectory_positions[agent_exists]
                )
            else:
                trajectory_features[agent_exists] = layer(
                    trajectory_features[agent_exists],
                    ~trajectory_mask[agent_exists]
                )

        trajectory_features = rearrange(trajectory_features, '(b a) t d -> b t a d', b=B, a=A)
        trajectory_mask = rearrange(trajectory_mask, '(b a) t -> b t a', b=B, a=A)
        trajectory_positions = rearrange(trajectory_positions, '(b a) t d f -> b t a d f', b=B, a=A)

        return trajectory_features, trajectory_mask, trajectory_positions


def apply_rotary_embeddings(features, positions):
    """
    2D Rotary positional embeddings as described in:
    https://arxiv.org/pdf/2111.12591.pdf, https://arxiv.org/pdf/2104.09864.pdf

    Code basically copied from: 
    https://github.com/rabbityl/lepard/blob/main/models/position_encoding.py

    Instead of adding positional embeddings, we multiply rotation matrices
    features: (batch_size, num_tokens, feature_dim)
    positions: (batch_size, num_tokens, 2)
    """
    batch_size, num_tokens, feature_dim = features.shape

    # Compute rotary embeddings
    x, y = positions[...,0:1], positions[...,1:2]
    div_term = torch.exp( torch.arange(0, feature_dim // 2, 2, dtype=torch.float, device=positions.device) * \
        (-math.log(10000.0) / (feature_dim // 2)))
    div_term = div_term.view(1,1,-1)

    sinx = torch.sin(x * div_term)
    cosx = torch.cos(x * div_term)
    siny = torch.sin(y * div_term)
    cosy = torch.cos(y * div_term)

    sinx, cosx, siny, cosy = map( lambda  feat:torch.stack([feat, feat], dim=-1).view(batch_size, num_tokens, -1), \
        [ sinx, cosx, siny, cosy] )
    
    sin_pos = torch.cat([sinx, siny], dim=-1)
    cos_pos = torch.cat([cosx, cosy], dim=-1)

    # Apply embeddings to features
    features2 = torch.stack([-features[..., 1::2], features[..., ::2]], dim=-1).reshape_as(features).contiguous()
    features = features * cos_pos + features2 * sin_pos

    return features
