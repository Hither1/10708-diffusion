import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from einops import rearrange
# from image_synthesis.distributed.distributed import is_primary, get_rank

from inspect import isfunction
from torch.cuda.amp import autocast
from torch.utils.checkpoint import checkpoint
from src.models.transformer_layers import TransformerEncoderLayer, TransformerDecoderLayer

"""
Noise schedules
"""
def vp_beta_schedule(timesteps, b_min=1e-4, b_max=3.):
    t = torch.arange(0, timesteps + 1)
    T = timesteps
    alpha = torch.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
    betas = 1 - alpha
    return betas


def linear_beta_schedule(timesteps, b_min, b_max):
    betas = torch.arange(1, timesteps+1) / timesteps
    betas = torch.clip(betas, b_min, b_max)
    return betas


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    

class PCAWhitener(nn.Module):
    def __init__(self, k, pca_params_path):
        super().__init__()

        self.k = k
        params = torch.load(pca_params_path)
        self.register_buffer('mean', params['mean'].float())
        self.register_buffer('components', params['components'].float()[:k])
        self.register_buffer('explained_variance', params['explained_variance'].float()[:k])

    def transform_features(self, features):
        """
        Transform raw trajectory features (N,16*3) to low-dimensional subspace features (N,k)
        """
        features = features.reshape(-1, 32)
        features = (features - self.mean) @ self.components.T
        features = self.explained_variance**(-.5) * features
        return features

    def untransform_features(self, features):
        """
        Transform low-dimensional subspace features (N,k) to raw trajectory features (N,16*3)
        """
        features = self.explained_variance**(.5) * features
        features = (features @ self.components) + self.mean
        return features

class Whitener(nn.Module):
    def __init__(self, params_path):
        super().__init__()

        params = torch.load(params_path)
        self.register_buffer('mean', torch.as_tensor(params['mean']).float())
        self.register_buffer('std', torch.as_tensor(params['std']).float())

    def transform_features(self, features):
        """
        Transform raw trajectory features (N,16*3) to low-dimensional subspace features (N,k)
        """
        features = features.reshape(-1, 32)
        features = (features - self.mean) / self.std
        return features

    def untransform_features(self, features):
        """
        Transform low-dimensional subspace features (N,k) to raw trajectory features (N,16*3)
        """
        features = (features * self.std) + self.mean
        return features

    def untransform_std(self, features):
        """
        Transform low-dimensional subspace std (N,k) to raw trajectory features (N,16*3)
        """
        features = (features * self.std)
        return features
        
# def heun_sampler(denoise_fn, sigma_fn, N):
#     """
#     Implementation of the pseudocode here: https://arxiv.org/pdf/2206.00364
#     Does not support scaling schedules
#     """
#     ts = None
#     x_curr = torch.normal()



class GELU2(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x * F.sigmoid(1.702 * x)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, num_steps, dim, rescale_steps=4000):
        super().__init__()
        self.dim = dim
        self.num_steps = float(num_steps)
        self.rescale_steps = float(rescale_steps)

    def forward(self, x):
        x = x / self.num_steps * self.rescale_steps
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class AdaLayerNorm(nn.Module):
    def __init__(self, n_embd, diffusion_step, emb_type="adalayernorm_abs"):
        super().__init__()
        if "abs" in emb_type:
            self.emb = SinusoidalPosEmb(diffusion_step, n_embd)
        else:
            self.emb = nn.Embedding(diffusion_step + 1, n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd*2)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)
        self.diff_step = diffusion_step

    def forward(self, x, timestep):
        # if timestep[0] >= self.diff_step:
        #     _emb = self.emb.weight.mean(dim=0, keepdim=True).repeat(len(timestep), 1)
        #     emb = self.linear(self.silu(_emb)).unsqueeze(1).unsqueeze(1)

        # else:
        emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)

        scale, shift = torch.chunk(emb, 2, dim=-1)
        x = self.layernorm(x) * (1 + scale) + shift

        return x

class AdaInsNorm(nn.Module):
    def __init__(self, n_embd, diffusion_step, emb_type="adainsnorm_abs"):
        super().__init__()
        if "abs" in emb_type:
            self.emb = SinusoidalPosEmb(diffusion_step, n_embd)
        else:
            self.emb = nn.Embedding(diffusion_step, n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd*2)
        self.instancenorm = nn.InstanceNorm1d(n_embd)

    def forward(self, x, timestep):
        emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.instancenorm(x.transpose(-1, -2)).transpose(-1,-2) * (1 + scale) + shift
        return x




class Text2ImageTransformer(nn.Module):
    def __init__(
        self,
        n_layer=14,
        n_embd=256,
        n_head=16,
        attn_pdrop=0,
        resid_pdrop=0,
        num_embed=32,
        block_activate='GELU',
        attn_type='selfcross',
        condition_dim=512,
        diffusion_step=1000,
        timestep_type='adalayernorm',
        mlp_type='fc',
        checkpoint=False,
    ):
        super().__init__()

        self.use_checkpoint = checkpoint

        # transformer
        all_attn_type = [attn_type] * n_layer

        self.blocks = nn.Sequential(*[Block(
                emb_dim=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                activate=block_activate,
                attn_type=all_attn_type[n],
                condition_dim = condition_dim,
                diffusion_step = diffusion_step,
                timestep_type = timestep_type,
        ) for n in range(n_layer)])

        # final prediction head
        out_cls = num_embed - 1 # self.content_emb.num_embed-1
        self.to_logits = nn.Sequential(
            nn.LayerNorm(n_embd),
            nn.Linear(n_embd, out_cls),
        )


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.elementwise_affine == True:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def parameters(self, recurse=True, name=None):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # return super().parameters(recurse=True)
        if name is None or name == 'none':
            return super().parameters(recurse=recurse)
        else:
            # separate out all parameters to those that will and won't experience regularizing weight decay
            print("GPTLikeTransformer: get parameters by the overwrite method!")
            decay = set()
            no_decay = set()
            whitelist_weight_modules = (torch.nn.Linear, )
            blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
            for mn, m in self.named_modules():
                for pn, p in m.named_parameters():
                    fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                    if pn.endswith('bias'):
                        # all biases will not be decayed
                        no_decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                        # weights of whitelist modules will be weight decayed
                        decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                        # weights of blacklist modules will NOT be weight decayed
                        no_decay.add(fpn)
            # special case the position embedding parameter as not decayed
            module_name = ['condition_emb', 'content_emb']
            pos_emb_name = ['pos_emb', 'width_emb', 'height_emb', 'pad_emb', 'token_type_emb']
            for mn in module_name:
                if hasattr(self, mn) and getattr(self, mn) is not None:
                    for pn in pos_emb_name:
                        if hasattr(getattr(self, mn), pn):
                            if isinstance(getattr(getattr(self, mn), pn), torch.nn.Parameter):
                                no_decay.add('{}.{}'.format(mn, pn))

            # validate that we considered every parameter
            param_dict = {pn: p for pn, p in self.transformer.named_parameters()}# if p.requires_grad} 
            inter_params = decay & no_decay
            union_params = decay | no_decay
            assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
            assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                        % (str(param_dict.keys() - union_params), )

            # create the pytorch optimizer object
            optim_groups = [
                {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
                {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
            ]
            return optim_groups

    def forward(
            self, 
            traj_emb, 
            wps_mask,
            agents_emb,
            agents_masks,
            map_emb,
            map_masks,
            t):

        for block_idx in range(len(self.blocks)):   
            if self.use_checkpoint == False:
                traj_emb = self.blocks[block_idx](traj_emb, wps_mask, agents_emb, agents_masks, map_emb, map_masks, t.cuda()) # B x (Ld+Lt) x D, B x (Ld+Lt) x (Ld+Lt)
            else:
                traj_emb = checkpoint(self.blocks[block_idx], traj_emb, wps_mask, agents_emb, agents_masks, map_emb, map_masks, t.cuda())
        logits = self.to_logits(traj_emb) # B x (Ld+Lt) x n
        out = rearrange(torch.squeeze(logits), 'b l c -> b c l')
        return out
