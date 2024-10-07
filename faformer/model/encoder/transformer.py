import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, einsum
from torch_geometric.utils import to_dense_batch

from faformer.model.encoder.fa import FrameAveraging
from faformer.model.encoder.nn_utils import MLPWrapper


class FAFFN(FrameAveraging):
    def __init__(
        self, 
        d_model, 
        proj_drop,
        activation,
        mlp_ratio=4.0,
    ):
        super(FAFFN, self).__init__()

        self.W_frame = MLPWrapper(
            3, d_model, d_model, activation, nn.LayerNorm, True, proj_drop
        )
        self.ffn = MLPWrapper(
            d_model * 2, int(d_model * mlp_ratio), d_model, activation, nn.LayerNorm, True, proj_drop
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, token_embs, geo_feats, batch_idx):     
        token_embs = self.ln(token_embs)

        batch_token_embs, _ = to_dense_batch(token_embs, batch=batch_idx)
        batch_geo_feats, coord_mask = to_dense_batch(geo_feats, batch=batch_idx)  # [B, N, 3]

        B, N = batch_token_embs.size(0), batch_token_embs.size(1)
        
        geo_frame_feats, F_ops, center = self.create_frame(batch_geo_feats, coord_mask)  # [B*8, N, 3]
        geo_frame_feats = self.W_frame(geo_frame_feats).view(B, 8, N, -1).mean(dim=1)  # [B, N, H]

        h_token_embs = torch.cat([batch_token_embs, geo_frame_feats], dim=-1)  # [B, N, d_model]
        return self.ffn(h_token_embs)[coord_mask]


class EdgeModule(FrameAveraging):
    def __init__(
        self, 
        d_model, 
        d_edge_model,
        proj_drop=0.,
        activation='gelu',
    ):
        super(EdgeModule, self).__init__()

        self.coord_mlp = MLPWrapper(
            3 + 1, d_edge_model, d_edge_model, activation, nn.LayerNorm, True, proj_drop
        )
        self.edge_mlp = MLPWrapper(
            d_model * 2 + d_edge_model, d_model, d_model, activation, nn.LayerNorm, True, proj_drop
        )

        self.att_mlp = nn.Sequential(
            nn.Linear(d_model, 1), 
            nn.Sigmoid()
        )

    def forward(self, token_embs, geo_feats, neighbor_indices, neighbor_masks):
        N, N_neighbors = token_embs.size(0), neighbor_indices.size(-1)

        radial_coords = geo_feats.unsqueeze(dim=1) - geo_feats[neighbor_indices]  # [N, N_neighbors, 3]
        radial_coord_norm = torch.sum(radial_coords ** 2, dim=-1).unsqueeze(-1)  # [N, N_neighbors, 1]
        
        """local frame features"""
        frame_feats, _, _ = self.create_frame(radial_coords, neighbor_masks)  # [N*8, N_neighbors, 3]
        frame_feats = frame_feats.view(N, 8, N_neighbors, -1)  # [N, 8, N_neighbors, d_model]

        radial_coord_norm = radial_coord_norm.unsqueeze(dim=1).expand(N, 8, N_neighbors, -1)
        frame_feats = self.coord_mlp(torch.cat([frame_feats, radial_coord_norm], dim=-1)).mean(dim=1)  # [N, N_neighbors, d_model]

        pair_embs = torch.cat([token_embs.unsqueeze(dim=1).expand(N, N_neighbors, -1), token_embs[neighbor_indices]], dim=-1)  # [N, N_neighbors, d_model*2]
        pair_embs = self.edge_mlp(torch.cat([pair_embs, frame_feats], dim=-1))
        return pair_embs * self.att_mlp(pair_embs)


class MLPAttnEdgeAggregation(FrameAveraging):
    def __init__(
        self, 
        d_model,
        d_edge_model,
        n_heads, 
        proj_drop=0., 
        attn_drop=0.,
        activation='gelu',
    ):
        super(MLPAttnEdgeAggregation, self).__init__()

        self.d_head, self.d_edge_head, self.n_heads = d_model // n_heads, d_edge_model // n_heads, n_heads

        self.layernorm_qkv = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 3),
        )
        self.layernorm_qkv_edge = nn.Sequential(
            nn.LayerNorm(d_edge_model),
            nn.Linear(d_edge_model, d_edge_model * 2),
        )

        self.mlp_attn = nn.Linear(self.d_head, 1, bias=False)
        self.edge_attn = nn.Linear(self.d_edge_head, 1, bias=False)
        self.W_output = MLPWrapper(
            d_model + d_edge_model, d_model, d_model, activation, nn.LayerNorm, True, proj_drop
        )
        self.W_gate = nn.Linear(d_model, 1)
        self.attn_dropout = nn.Dropout(attn_drop)

        if n_heads > 1:
            self.W_frame_agg = nn.Sequential(nn.Linear(n_heads, 1), nn.SiLU())

        self._init_params()

    def _init_params(self):
        nn.init.constant_(self.W_gate.weight, 0.)
        nn.init.constant_(self.W_gate.bias, 1.)

    def forward(self, token_embs, geo_feats, edge_feats, neighbor_indices, batch_idx, neighbor_masks):
        # token_embs: [N, -1], geo_feats: [N, 3], edge_feats: [N, N_neighbor, -1]
        # neighbor_indices: [N, N_neighbor], neighbor_masks: [N, N_neighbor]
        residual, n_tokens = token_embs, token_embs.size(0)
        d_head, n_heads, device = self.d_head, self.n_heads, token_embs.device

        """qkv transformation"""
        q_s, k_s, v_s = self.layernorm_qkv(token_embs).chunk(3, dim=-1)
        q_s, k_s, v_s = map(lambda x: rearrange(x, 'n (h d) -> n h d', h=n_heads), (q_s, k_s, v_s))
        q_edge_s, v_edge_s = self.layernorm_qkv_edge(edge_feats).chunk(2, dim=-1)
        q_edge_s, v_edge_s = map(lambda x: rearrange(x, 'n m (h d) -> n m h d', h=n_heads), (q_edge_s, v_edge_s))
        gate_s = self.W_gate(token_embs).sigmoid()  # for residue connection of geometric features

        """attention map"""
        message = (q_s.unsqueeze(dim=1) + k_s[neighbor_indices]).view(n_tokens, -1, n_heads, d_head)
        attn_map = self.mlp_attn(message).squeeze(dim=-1)
        attn_map = attn_map + self.edge_attn(q_edge_s).squeeze(dim=-1)  # [N, N_neighbor, n_heads]
        attn_map.masked_fill_(~neighbor_masks.unsqueeze(dim=-1), -1e9)
        attn_map = self.attn_dropout(
            nn.Softmax(dim=-1)(attn_map.transpose(1, 2))
        )  # [N, n_heads, N_neighbor]

        """context aggregation"""
        v_s_neighs = v_s[neighbor_indices].view(n_tokens, -1, n_heads, d_head)  # [N, N_neighbor, n_heads, D]
        scalar_context = einsum(attn_map, v_s_neighs, 'n h m, n m h d -> n h d').view(n_tokens, -1)  # [N, n_heads*D]
        edge_context = einsum(attn_map, v_edge_s, 'n h m, n m h d -> n h d').view(n_tokens, -1)  # [N, n_heads*D]
        scalar_output = self.W_output(torch.cat([scalar_context, edge_context], dim=-1)) + residual

        if n_heads == 1:
            geo_context = einsum(attn_map, geo_feats[neighbor_indices], 'n h m, n m d -> n h d').squeeze(dim=1)
        else:
            # use FA to aggregate the geometric features in different heads to make sure equivariance
            batch_geo_feats, coord_mask = to_dense_batch(geo_feats, batch=batch_idx)  # [batch_B, batch_N, 3]
            batch_B, batch_N = batch_geo_feats.size(0), batch_geo_feats.size(1)
            geo_frame_feats, F_ops, center = self.create_frame(batch_geo_feats, coord_mask)  # [batch_B*8, batch_N, 3]
            
            # flattern the batch dimension for aligning with attention map and neighbor indices
            geo_frame_feats = geo_frame_feats.view(batch_B, 8, batch_N, 3).transpose(2, 1).reshape(batch_B, batch_N, 8*3)  # [batch_B, batch_N, 8*3]
            geo_frame_feats_flattern = geo_frame_feats[coord_mask].view(n_tokens, 8, 3).transpose(1, 0)  # [8, N, 3]

            geo_frame_feats_flattern = geo_frame_feats_flattern.unsqueeze(dim=2).expand(-1, -1, n_heads, -1).reshape(8*n_tokens, n_heads*3)  # [8*N, n_heads*3]
            neighbor_indices_expand = neighbor_indices.unsqueeze(dim=0).expand(8, -1, -1).reshape(8*n_tokens, -1)  # [8*N, N_neighbor]
            geo_neighbors = geo_frame_feats_flattern[neighbor_indices_expand].view(8, n_tokens, -1, n_heads, 3)  # [8, N, N_neighbor, n_heads, 3]
            attn_map_expand = attn_map.unsqueeze(dim=0).expand(8, -1, -1, -1).transpose(3, 2)  # [8, N, n_heads, N_neighbor]

            geo_context = einsum(attn_map_expand, geo_neighbors, 'f n m h, f n m h d -> f n h d')  # [8, N, n_heads, 3]
            geo_context = self.W_frame_agg(geo_context.transpose(3, 2)).squeeze(-1).transpose(1, 0).reshape(n_tokens, -1)  # [N, 8*3]

            batch_geo_context, _ = to_dense_batch(geo_context, batch=batch_idx)  # [batch_B, batch_N, 8*3]
            batch_geo_context = batch_geo_context.view(batch_B, batch_N, 8, 3).transpose(2, 1)  # [batch_B, 8, batch_N, 3]
            geo_context = self.invert_frame(batch_geo_context, coord_mask, F_ops, center)  # [batch_B, batch_N, 3]
            geo_context = geo_context[coord_mask]  # [N, 3]

        geo_output = geo_context * gate_s + geo_feats * (1 - gate_s)  # [N, 3]
        return scalar_output, geo_output


class FAFormerEncoderLayer(nn.Module):
    def __init__(
        self, 
        d_model,
        d_edge_model,
        n_heads,
        proj_drop=0.,
        attn_drop=0.,
        activation='gelu',
    ):
        super(FAFormerEncoderLayer, self).__init__()

        self.self_attn = MLPAttnEdgeAggregation(d_model, d_edge_model, n_heads, proj_drop, attn_drop, activation)
        self.ffn = FAFFN(d_model, proj_drop, activation)
        self.edge_module = EdgeModule(d_model, d_edge_model, proj_drop, activation)

    def forward(self, token_embs, geo_feats, edge_feats, neighbor_indices, batch_idx, neighbor_masks):
        # neighbor_indices: [B, N, N_neighbor], neighbor_masks: [B, N, N_neighbor]
        token_embs, geo_feats = self.self_attn(token_embs, geo_feats, edge_feats, neighbor_indices, batch_idx, neighbor_masks)
        
        edge_feats = edge_feats + self.edge_module(token_embs, geo_feats, neighbor_indices, neighbor_masks)

        token_embs = token_embs + self.ffn(token_embs, geo_feats, batch_idx)
        return token_embs, geo_feats, edge_feats


class FAFormer(nn.Module):
    def __init__(self, config):
        super(FAFormer, self).__init__()

        self.input_transform = nn.Linear(config.d_input, config.d_model)
        self.edge_module = EdgeModule(config.d_model, config.d_edge_model, config.proj_drop, config.activation)
        self.layers = nn.ModuleList([FAFormerEncoderLayer(
            config.d_model, config.d_edge_model, config.n_heads, config.proj_drop, config.attn_drop, config.activation) for _ in range(config.n_layers)]
        )
        self.pos_emb = nn.Embedding(config.n_pos, config.d_model)
        self.dropout_module = nn.Dropout(config.proj_drop)

        self.n_neighbors = config.n_neighbors
        self.valid_radius = config.valid_radius
        self.embedding_grad_frac = config.embedding_grad_frac

    def _build_graph(self, coords, batch_idx, n_neighbors, valid_radius):
        exclude_self_mask = torch.eye(coords.shape[0], dtype=torch.bool, device=coords.device)  # 1: diagonal elements
        batch_mask = batch_idx.unsqueeze(0) == batch_idx.unsqueeze(1)  # [N, N], True if the token is in the same batch

        # calculate relative distance
        rel_pos = rearrange(coords, 'n d -> n 1 d') - rearrange(coords, 'n d -> 1 n d')
        rel_dist = rel_pos.norm(dim = -1).detach()  # [N, N]
        rel_dist.masked_fill_(exclude_self_mask | ~batch_mask, 1e9)

        dist_values, nearest_indices = rel_dist.topk(n_neighbors, dim = -1, largest = False)
        neighbor_mask = dist_values <= valid_radius  # [N, N_neighbors], True if distance is within valid radius
        return nearest_indices, neighbor_mask

    def forward(self, features, coords, pad_mask=None):
        B, N, device = coords.shape[0], coords.shape[1], coords.device
        
        if pad_mask is None:
            pad_mask = features.sum(dim=-1) == 0  # [B, N], True if the token is padding

        """flattern the batch dimension for acceleration"""
        batch_idx = torch.arange(B, device=coords.device).unsqueeze(-1).repeat(1, N)[~pad_mask]
        features = features[~pad_mask]  # [-1, 3]
        coords = coords[~pad_mask]  # [-1, 3]

        token_embs = self.input_transform(features)  # [-1, d_model]

        # for learnable positional embedding
        pos_tokens = torch.arange(N).unsqueeze(0).repeat(B, 1).to(device)
        token_embs = token_embs + self.pos_emb(pos_tokens[~pad_mask])  # [-1, d_model]
        token_embs = self.dropout_module(token_embs)

        token_embs = self.embedding_grad_frac * token_embs + (1 - self.embedding_grad_frac) * token_embs.detach()

        """generate graph"""
        nearest_indices, neighbor_mask = self._build_graph(coords, batch_idx, int(min(self.n_neighbors, N)), self.valid_radius)

        edge_feats = self.edge_module(token_embs, coords, nearest_indices, neighbor_mask)
        for i, layer in enumerate(self.layers):
            token_embs, coords, edge_feats = layer(token_embs, coords, edge_feats, nearest_indices, batch_idx, neighbor_mask)

        token_embs, _ = to_dense_batch(token_embs, batch=batch_idx, max_num_nodes=N)
        coords, _ = to_dense_batch(coords, batch=batch_idx, max_num_nodes=N)

        return token_embs, coords


if __name__ == "__main__":

    from faformer.model.encoder.config import FAFormerConfig

    model = FAFormer(
        FAFormerConfig(
            d_input=10,
            n_layers=2,
            n_neighbors=2, 
            n_heads=1,
            d_model=4,
            d_edge_model=4,
            norm="layer",
            activation="swiglu",
        )
    )
    features = torch.randn(2, 5, 10)
    coords = torch.rand(2, 5, 3)
    features, coords = model(features, coords)
    print(features.shape)
    print(coords.shape)
