import torch
from torch import nn


from src.models.transformer_layers import TransformerEncoderLayer
from src.models.embedders import Embedder


class MapEncoder(nn.Module):
    def __init__(
            self,
            map_dim,
            light_dim,
            stop_dim,
            emb_dim,
            num_enc_layers,
            num_heads=16,
            tx_hidden_size=512,
            norm_first=False,
            dropout=0.0,
    ):
        super().__init__()
        self.map_dim = map_dim
        self.light_dim = light_dim
        self.stop_dim = stop_dim
        self.emb_dim = emb_dim
        self.num_enc_layers = num_enc_layers
        self.num_heads = num_heads
        self.tx_hidden_size = tx_hidden_size
        self.norm_first = norm_first
        self.dropout = dropout

        #  self.map_embedder = nn.Sequential(
            #  nn.Linear(self.map_dim, self.emb_dim),
            #  nn.ReLU())
        #  self.light_embedder = nn.Sequential(
            #  nn.Linear(self.light_dim, self.emb_dim),
            #  nn.ReLU())
        #  self.stop_embedder = nn.Sequential(
            #  nn.Linear(self.stop_dim, self.emb_dim),
            #  nn.ReLU())
        self.map_embedder = Embedder(self.map_dim, self.emb_dim, expand_theta=True, layer_norm=False)
        self.light_embedder = Embedder(self.light_dim, self.emb_dim, expand_theta=True, layer_norm=False)
        self.stop_embedder = Embedder(self.stop_dim, self.emb_dim, expand_theta=True, layer_norm=False)

        self.map_enc_layers = []
        for _ in range(self.num_enc_layers):
            map_encoder_layer = TransformerEncoderLayer(
                d_model=self.emb_dim,
                nhead=self.num_heads,
                dropout=self.dropout,
                dim_feedforward=self.tx_hidden_size,
                norm_first=self.norm_first,
                batch_first=True
            )
            self.map_enc_layers.append(map_encoder_layer)
        self.map_enc_layers = nn.ModuleList(self.map_enc_layers)
        if self.norm_first:
            self.map_enc_ln = nn.LayerNorm(self.emb_dim)

    def map_enc_fn(self, map_emb, map_masks, layer):
        '''
        :param map_emb: (B, P, H)
        :param map_masks: (B, P)
        :return: (B, P, H)
        '''
        map_masks = torch.where(map_masks.all(dim=-1, keepdims=True), torch.zeros_like(map_masks), map_masks)
        map_self_attn_emb = layer(
            map_emb,
            src_key_padding_mask=map_masks,
        )
        return map_self_attn_emb

    def forward(self, map_features, map_masks, light_features=None, light_masks=None, stop_features=None, stop_masks=None):
        map_emb = self.map_embedder(map_features)

        if light_features is not None:
            light_emb = self.light_embedder(light_features)
            map_emb = torch.cat([map_emb, light_emb], dim=1)
            map_masks = torch.cat([map_masks, light_masks], dim=1)

        if stop_features is not None:
            stop_emb = self.stop_embedder(stop_features)
            map_emb = torch.cat([map_emb, stop_emb], dim=1)
            map_masks = torch.cat([map_masks, stop_masks], dim=1)

        for i in range(self.num_enc_layers):
            map_emb = self.map_enc_fn(map_emb, map_masks.clone(), layer=self.map_enc_layers[i])
        if self.norm_first:
            map_emb = self.map_enc_ln(map_emb)
        return map_emb, map_masks
