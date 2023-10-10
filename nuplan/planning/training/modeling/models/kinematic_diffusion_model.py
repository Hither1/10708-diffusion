from typing import cast
import os
import math

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image

from diffusers import DDIMScheduler, DDPMScheduler

from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.modeling.models.urban_driver_open_loop_model import (
    convert_predictions_to_trajectory,
)
from nuplan.planning.training.modeling.models.diffusion_utils import (
    Whitener,
    DummyWhitener,
    # BoundsWhitener,
    Standardizer,
    SinusoidalPosEmb,
    VerletStandardizer
)
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.models.encoder_decoder_layers import (
    ParallelAttentionLayer
)
from nuplan.planning.training.preprocessing.features.agent_history import AgentHistory
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
# from nuplan.planning.training.callbacks.utils.visualization_utils import get_raster_from_vector_map_with_new_agents
from nuplan.planning.training.modeling.models.positional_embeddings import VolumetricPositionEncoding2D, RotaryPositionEncoding
from nuplan.planning.training.callbacks.utils.visualization_utils import (
    visualize
)


class KinematicDiffusionModel(TorchModuleWrapper):
    def __init__(
        self,

        feature_dim,
        past_trajectory_sampling,
        future_trajectory_sampling,

        map_params,

        T: int = 32,

        absolute_params_path: str = '',
        relative_params_path: str = '',

        predictions_per_sample: int = 16,

        max_dist: float = 50,           # used to normalize ALL tokens (incl. trajectory)
        easy_validation: bool = False,  # instead of starting from pure noise, start with not that much noise at inference,
    ):
        super().__init__(
            feature_builders=[
                AgentHistoryFeatureBuilder(
                    trajectory_sampling=past_trajectory_sampling,
                    max_agents=10
                ),
                VectorSetMapFeatureBuilder(
                    map_features=map_params['map_features'],
                    max_elements=map_params['max_elements'],
                    max_points=map_params['max_points'],
                    radius=map_params['vector_set_map_feature_radius'],
                    interpolation_method=map_params['interpolation_method']
                )
            ],
            target_builders=[
                EgoTrajectoryTargetBuilder(future_trajectory_sampling),
                AgentTrajectoryTargetBuilder(
                    trajectory_sampling=past_trajectory_sampling,
                    future_trajectory_sampling=future_trajectory_sampling,
                    max_agents=10
                )
            ],
            future_trajectory_sampling=future_trajectory_sampling
        )

        self.feature_dim = feature_dim

        self.T = T

        self.H = 16
        self.output_dim = self.H * 3
        
        self.predictions_per_sample = predictions_per_sample

        self.max_dist = max_dist

        self.easy_validation = easy_validation

        self.standardizer = VerletStandardizer() # Standardizer(max_dist=max_dist)

        # DIFFUSER
        self.scheduler = DDIMScheduler(
            num_train_timesteps=self.T,
            beta_schedule='scaled_linear',
            prediction_type='epsilon',
        )
        self.scheduler.set_timesteps(self.T)

        self.history_encoder = nn.Sequential(
            nn.Linear(7, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )

        self.sigma_encoder = nn.Sequential(
            SinusoidalPosEmb(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )
        self.sigma_proj_layer = nn.Linear(self.feature_dim * 2, self.feature_dim)

        self.trajectory_encoder = nn.Linear(3, self.feature_dim)
        self.trajectory_time_embeddings = RotaryPositionEncoding(self.feature_dim)
        # self.trajectory_time_embeddings = nn.Embedding(self.H+5, self.feature_dim)
        self.type_embedding = nn.Embedding(3, self.feature_dim) # trajectory, noise token

        self.global_attention_layers = torch.nn.ModuleList([
            ParallelAttentionLayer(
                d_model=self.feature_dim, 
                self_attention1=True, self_attention2=False,
                cross_attention1=False, cross_attention2=False,
                rotary_pe=True
            )
        for _ in range(8)])

        self.decoder_mlp = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, 3)
        )

        self.all_trajs = []

        self.apply(self._init_weights)

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

    def forward(self, features: FeaturesType) -> TargetsType:
        # Recover features
        ego_agent_features = cast(AgentHistory, features["agent_history"])
        batch_size = ego_agent_features.batch_size

        state_features = self.encode_scene_features(ego_agent_features)

        # Only use for denoising
        if 'trajectory' in features:
            ego_gt_trajectory = features['trajectory'].data.clone()
            ego_gt_trajectory = self.standardizer.transform_features(ego_agent_features, ego_gt_trajectory)

        if self.training:
            noise = torch.randn(ego_gt_trajectory.shape, device=ego_gt_trajectory.device)
            timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (ego_gt_trajectory.shape[0],), 
                                        device=ego_gt_trajectory.device).long()

            ego_noisy_trajectory = self.scheduler.add_noise(ego_gt_trajectory, noise, timesteps)

            pred_noise = self.denoise(ego_noisy_trajectory, timesteps, state_features)

            output = {
                # TODO: fix trajectory, right now we only care about epsilon at training and this is a dummy value
                "trajectory": Trajectory(data=convert_predictions_to_trajectory(pred_noise)),
                "epsilon": pred_noise,
                "gt_epsilon": noise
            }
            return output
        else:
            # Multiple predictions per sample
            state_features = (
                state_features[0].repeat_interleave(self.predictions_per_sample,0),
                state_features[1].repeat_interleave(self.predictions_per_sample,0)
            )
            ego_agent_features = ego_agent_features.repeat_interleave(self.predictions_per_sample,0)

            # Sampling / inference
            if 'warm_start' in features:
                ego_trajectory = features['warm_start'].clone().to(state_features[0].device).squeeze(0)
                ego_trajectory = self.standardizer.transform_features(ego_agent_features, ego_trajectory)
                timesteps = self.scheduler.timesteps[features['warm_start_steps']:]
                noise = torch.randn(ego_trajectory.shape, device=ego_trajectory.device)
                ego_trajectory = self.scheduler.add_noise(ego_trajectory, noise, timesteps[0])
            else:
                ego_trajectory = torch.randn((batch_size * self.predictions_per_sample, self.H * 3), device=state_features[0].device)
                timesteps = self.scheduler.timesteps
            
            intermediate_trajectories = []

            for t in tqdm(timesteps):
                residual = torch.zeros_like(ego_trajectory)

                # If constraints exist, then compute epsilons and add them
                if 'constraints' in features:
                    with torch.enable_grad():
                        for _ in range(8):
                            ego_trajectory.requires_grad_(True)
                            ego_trajectory_unstd = self.standardizer.untransform_features(ego_agent_features, ego_trajectory)
                            constraint_scores = compute_constraint_scores(features['constraints'], ego_trajectory_unstd)

                            grad = torch.autograd.grad([constraint_scores.mean()], [ego_trajectory])[0]
                            posterior_variance = self.scheduler._get_variance(t,t-1)
                            model_std = torch.exp(0.5 * posterior_variance)
                            grad = model_std * grad

                            ego_trajectory = ego_trajectory.detach()
                            ego_trajectory = ego_trajectory - 0.01 * grad

                with torch.no_grad():
                    residual += self.denoise(ego_trajectory, t.to(ego_trajectory.device), state_features)

                out = self.scheduler.step(residual, t, ego_trajectory)
                ego_trajectory = out.prev_sample
                ego_trajectory_clean = out.pred_original_sample

                # Intermediate trajectories
                int_ego_trajectory = self.standardizer.untransform_features(ego_agent_features, ego_trajectory_clean)
                intermediate_trajectories.append(int_ego_trajectory)

            ego_trajectory = self.standardizer.untransform_features(ego_agent_features, ego_trajectory)

            if 'constraints' in features:
                # If we have constraints, we can take the trajectory with the lowest constraint cost
                scores = compute_constraint_scores(features['constraints'], ego_trajectory)
                ego_trajectory = ego_trajectory.reshape(batch_size, self.predictions_per_sample, self.output_dim)
                scores = scores.reshape(batch_size, self.predictions_per_sample)
                best_trajectory = ego_trajectory[range(batch_size), scores.argmin(dim=1)]

                constraint_residual = compute_constraint_residual(features['constraints'], ego_trajectory.reshape(-1,self.output_dim))
                constraint_residual = constraint_residual[None][range(batch_size), scores.argmin(dim=1)]
            else:
                # Otherwise, who cares TODO later
                ego_trajectory = ego_trajectory.reshape(batch_size, self.predictions_per_sample, self.output_dim)
                best_trajectory = ego_trajectory[:,0]

            return {
                "trajectory": Trajectory(data=convert_predictions_to_trajectory(best_trajectory)),
                "multimodal_trajectories": ego_trajectory,
                "grad": convert_predictions_to_trajectory(constraint_residual) if 'constraints' in features else None,
                "score": scores,
                "intermediate": intermediate_trajectories
            }
        
    def encode_scene_features(self, ego_agent_features):
        ego_features = ego_agent_features.ego
        ego_features = self.history_encoder(ego_features) # Bx5x7 -> Bx5xD
        
        # ego_time_embedding = self.trajectory_time_embeddings(torch.arange(5, device=ego_features.device))[None].repeat(ego_features.shape[0],1,1)
        # ego_features += ego_time_embedding

        ego_type_embedding = self.type_embedding(torch.as_tensor([[0]], device=ego_features.device))
        ego_type_embedding = ego_type_embedding.repeat(ego_features.shape[0],5,1)

        return ego_features, ego_type_embedding

    def denoise(self, ego_trajectory, sigma, state_features):
        """
        Denoise ego_trajectory with noise level sigma (no preconditioning)
        Equivalent to evaluating F_theta in this paper: https://arxiv.org/pdf/2206.00364
        """
        batch_size = ego_trajectory.shape[0]

        state_features, state_type_embedding = state_features
        
        # Trajectory features
        ego_trajectory = ego_trajectory.reshape(ego_trajectory.shape[0],self.H,3)
        trajectory_features = self.trajectory_encoder(ego_trajectory)

        # trajectory_time_embedding = self.trajectory_time_embeddings(torch.arange(5, self.H+5, device=ego_trajectory.device))[None].repeat(batch_size,1,1)
        # trajectory_features += trajectory_time_embedding
        trajectory_type_embedding = self.type_embedding(
            torch.as_tensor([1], device=ego_trajectory.device)
        )[None].repeat(batch_size,self.H,1)

        # Concatenate all features
        all_features = torch.cat([state_features, trajectory_features], dim=1)
        all_type_embedding = torch.cat([state_type_embedding, trajectory_type_embedding], dim=1)

        # Sigma encoding
        sigma = sigma.reshape(-1,1)
        if sigma.numel() == 1:
            sigma = sigma.repeat(batch_size,1)
        sigma = sigma.float() / self.T
        sigma_embeddings = self.sigma_encoder(sigma)
        sigma_embeddings = sigma_embeddings.reshape(batch_size,1,self.feature_dim)

        # Concatenate sigma features and project back to original feature_dim
        sigma_embeddings = sigma_embeddings.repeat(1,all_features.shape[1],1)
        all_features = torch.cat([all_features, sigma_embeddings], dim=2)
        all_features = self.sigma_proj_layer(all_features)

        # Generate attention mask
        seq_len = all_features.shape[1]
        indices = torch.arange(seq_len, device=all_features.device)
        dists = (indices[None] - indices[:,None]).abs()
        attn_mask = dists > 1       # TODO: magic number

        # Generate relative temporal embeddings
        temporal_embedding = self.trajectory_time_embeddings(indices[None].repeat(batch_size,1))

        # Global self-attentions
        for layer in self.global_attention_layers:            
            all_features, _ = layer(
                all_features, None, None, None,
                seq1_pos=temporal_embedding,
                seq1_sem_pos=all_type_embedding,
                attn_mask_11=attn_mask
            )

        trajectory_features = all_features[:,-self.H:]
        out = self.decoder_mlp(trajectory_features).reshape(trajectory_features.shape[0],-1)

        return out # , all_weights

    def cem_test(self, features: FeaturesType, warm_start = None) -> TargetsType:
        # Recover features
        ego_agent_features = cast(AgentHistory, features["agent_history"])
        batch_size = ego_agent_features.batch_size * self.predictions_per_sample

        state_features = self.encode_scene_features(ego_agent_features)

        # Only use for denoising
        if 'trajectory' in features:
            ego_gt_trajectory = features['trajectory'].data.clone()
            ego_gt_trajectory = self.standardizer.transform_features(ego_agent_features, ego_gt_trajectory)

        # Multiple predictions per sample
        state_features = (
            state_features[0].repeat_interleave(self.predictions_per_sample,0),
            state_features[1].repeat_interleave(self.predictions_per_sample,0)
        )
        ego_agent_features = ego_agent_features.repeat_interleave(self.predictions_per_sample,0)

        # Sampling / inference
        # if 'warm_start' in features:
        #     ego_trajectory = features['warm_start'].clone().to(state_features[0].device).squeeze(0)
        #     ego_trajectory = self.standardizer.transform_features(ego_agent_features, ego_trajectory)
        #     timesteps = self.scheduler.timesteps[features['warm_start_steps']:]
        #     noise = torch.randn(ego_trajectory.shape, device=ego_trajectory.device)
        #     ego_trajectory = self.scheduler.add_noise(ego_trajectory, noise, timesteps[0])
        # else:

        def rollout(
            ego_trajectory, 
            ego_agent_features, 
            initial_rollout=True, 
            deterministic=True, 
            n_trunc_steps=5, 
            noise_scale=1.0, 
            ablate_diffusion=False
        ):
            if initial_rollout:
                timesteps = self.scheduler.timesteps
            else:
                timesteps = self.scheduler.timesteps[-n_trunc_steps:]

            if ablate_diffusion and not initial_rollout:
                timesteps = []

            saved_latent = ego_trajectory.clone()
            
            temp = []

            for t in timesteps:
                if t == self.scheduler.timesteps[-n_trunc_steps]:
                    saved_latent = ego_trajectory.clone()

                residual = torch.zeros_like(ego_trajectory)

                with torch.no_grad():
                    residual += self.denoise(ego_trajectory, t.to(ego_trajectory.device), state_features)

                if deterministic:
                    eta = 0.0
                else:
                    prev_alpha = self.scheduler.alphas[t-1]
                    alpha = self.scheduler.alphas[t]
                    eta = noise_scale * torch.sqrt((1 - prev_alpha) / (1 - alpha)) * \
                          torch.sqrt((1 - alpha) / prev_alpha)

                out = self.scheduler.step(residual, t, ego_trajectory, eta=eta)
                ego_trajectory = out.prev_sample

                temp_ego_trajectory = self.standardizer.untransform_features(ego_agent_features, ego_trajectory)
                temp.append(temp_ego_trajectory)

            ego_trajectory = self.standardizer.untransform_features(ego_agent_features, ego_trajectory)
            scores = compute_constraint_scores(features['constraints'], ego_trajectory)

            return ego_trajectory, scores, saved_latent, temp

        def renoise(ego_trajectory, ego_agent_features, t):
            ego_trajectory = self.standardizer.transform_features(ego_agent_features, ego_trajectory)
            noise = torch.randn(ego_trajectory.shape, device=ego_trajectory.device)
            ego_trajectory = self.scheduler.add_noise(ego_trajectory, noise, self.scheduler.timesteps[-t])
            return ego_trajectory

        # CEM params
        cem_iters = 5
        n_trunc_steps = 5
        elite_ratio = 0.1 # 1 / batch_size
        num_elites = int(batch_size * elite_ratio)
        noise_scale = 1.0
        mixture_ratio = 0.0 # if features['counter'] != 22 else 1.0
        ablate_diffusion = False

        trajectory_shape = (batch_size, self.H * 3)
        device = state_features[0].device

        use_warm_start = warm_start is not None
        num_keep = int(mixture_ratio * batch_size)
        if (not use_warm_start) or (num_keep == 0):
            noisy_trajectory = (torch.randn(trajectory_shape, device=device))
        else:
            print('num_keep != 0, something is up !!!!')
            prev_trajectory, scores = warm_start
            best_k = torch.argsort(scores)[:num_keep]
            noisy_trajectory = renoise(prev_trajectory, ego_agent_features, n_trunc_steps)
            noisy_trajectory = noisy_trajectory[best_k]
            noisy_trajectory = torch.cat([noisy_trajectory,
                torch.randn((batch_size - num_keep, trajectory_shape[1]), device=device)
            ], dim=0)

        # # VIZ INTIIALS
        # if use_warm_start:
        #     frame1 = visualize(
        #         features['vector_set_map'].to_device('cpu'),
        #         features['agent_history'].to_device('cpu'),
                
        #         trajs=prev_trajectory.reshape(-1,16,3),
        #         trajs_color='blue',

        #         pixel_size=0.1,
        #         radius=60
        #     )
        #     frame2 = visualize(
        #         features['vector_set_map'].to_device('cpu'),
        #         features['agent_history'].to_device('cpu'),
                
        #         trajs=self.standardizer.untransform_features(ego_agent_features, noisy_trajectory).reshape(-1,16,3),
        #         trajs_color='blue',

        #         pixel_size=0.1,
        #         radius=60
        #     )
        #     vizframe = np.concatenate([frame1, frame2], axis=0)
        #     cv2.imwrite(f"/zfsauton2/home/brianyan/tuplan_garage/viz/viz_{features['counter']}.png", vizframe)

        frames = []

        for i in range(cem_iters):
            # Rollout + scoring
            ego_trajectory, scores, _, denoising_samples = rollout(
                noisy_trajectory, ego_agent_features,
                initial_rollout=(i == 0),
                deterministic=False,
                n_trunc_steps=n_trunc_steps,
                noise_scale=noise_scale,
                ablate_diffusion=ablate_diffusion
            )

            # if features['counter'] == 23:
            #     # import pdb; pdb.set_trace()
            #     temp_frames = []
            #     for traj in denoising_samples:
            #         temp_frame = visualize(
            #             features['vector_set_map'].to_device('cpu'),
            #             features['agent_history'].to_device('cpu'),
                        
            #             trajs=traj.reshape(-1,16,3),
            #             trajs_color='blue',

            #             pixel_size=0.1,
            #             radius=60
            #         )
            #         temp_frames.append(temp_frame)
            #     temp_frames = [Image.fromarray(frame) for frame in temp_frames]
            #     fname = '/zfsauton2/home/brianyan/tuplan_garage/viz/denoising_old.gif'
            #     temp_frames[0].save(fname, save_all=True, append_images=temp_frames[1:], duration=10, loop=0)
            #     # raise

            # # VIZ
            # frame = visualize(
            #     features['vector_set_map'].to_device('cpu'),
            #     features['agent_history'].to_device('cpu'),
                
            #     trajs=ego_trajectory.reshape(-1,16,3),
            #     trajs_c=scores.detach().cpu().numpy(),
            #     trajs_cmap='winter_r',

            #     pixel_size=0.1,
            #     radius=60
            # )
            # frames.append(frame)
            
            if i < cem_iters - 1:
                # Select elites
                sorted_k = torch.argsort(scores)
                best_k, worst_k = sorted_k[:num_elites], sorted_k[num_elites:]

                # Resample using elites
                new_k = best_k[torch.randint(0,num_elites,(batch_size-num_elites,))]
                ego_trajectory[worst_k] = ego_trajectory[new_k]

                # if features['counter'] == 23:
                #     # Visualize renoising
                #     noisy_trajectory_viz1 = self.standardizer.untransform_features(ego_agent_features, noisy_trajectory)
                #     noisy_trajectory_viz2 = renoise(ego_trajectory, ego_agent_features, n_trunc_steps)
                #     noisy_trajectory_viz2 = self.standardizer.untransform_features(ego_agent_features, noisy_trajectory_viz2)
                #     renoise_frames = []
                #     for i in range(128):
                #         frame0 = visualize(
                #             features['vector_set_map'].to_device('cpu'),
                #             features['agent_history'].to_device('cpu'),
                            
                #             trajs=noisy_trajectory_viz1[i].detach().cpu().numpy().reshape(-1,16,3),
                #             trajs_color='blue',

                #             pixel_size=0.2,
                #             radius=120
                #         )
                #         frame1 = visualize(
                #             features['vector_set_map'].to_device('cpu'),
                #             features['agent_history'].to_device('cpu'),
                            
                #             trajs=ego_trajectory[i].detach().cpu().numpy().reshape(-1,16,3),
                #             trajs_color='blue',

                #             pixel_size=0.2,
                #             radius=120
                #         )
                #         frame2 = visualize(
                #             features['vector_set_map'].to_device('cpu'),
                #             features['agent_history'].to_device('cpu'),
                            
                #             trajs=noisy_trajectory_viz2[i].detach().cpu().numpy().reshape(-1,16,3),
                #             trajs_color='blue',

                #             pixel_size=0.2,
                #             radius=120
                #         )
                #         renoise_frame = np.concatenate([frame0, frame1, frame2], axis=0)
                #         renoise_frames.append(renoise_frame)
                #     renoise_frames = np.concatenate(renoise_frames, axis=1)
                #     cv2.imwrite('/zfsauton2/home/brianyan/tuplan_garage/viz/renoise.png', renoise_frames)
                #     import pdb; pdb.set_trace()
                
                noisy_trajectory = renoise(ego_trajectory, ego_agent_features, n_trunc_steps)

        # # MORE VIZ
        # if features['counter'] == 23:
        #     print(scores)
        #     frames = [Image.fromarray(frame) for frame in frames]
        #     fname = '/zfsauton2/home/brianyan/tuplan_garage/viz/cem_old.gif'
        #     frames[0].save(fname, save_all=True, append_images=frames[1:], duration=10, loop=0)
        #     print(f'SAVING GIF TO {fname}')
        #     raise

        best_trajectory = ego_trajectory[scores.argmin()]
        best_trajectory = best_trajectory.reshape(-1,16,3)
        
        return {
            "trajectory": Trajectory(data=convert_predictions_to_trajectory(best_trajectory)),
            "multimodal_trajectories": ego_trajectory,
            "score": scores
        }


def compute_constraint_scores(constraints, trajectory):
    total_cost = torch.zeros(trajectory.shape[0], device=trajectory.device)
    for constraint in constraints:
        total_cost += constraint(trajectory)
    return total_cost


def compute_constraint_residual(constraints, trajectory):
    """
    Compute the gradient of the sum of all the constraints w.r.t trajectory
    """
    with torch.enable_grad():
        trajectory.requires_grad_(True)
        total_cost = compute_constraint_scores(constraints, trajectory)
        total_cost.mean().backward()
        grad = trajectory.grad
        trajectory.requires_grad_(False)
    return grad
