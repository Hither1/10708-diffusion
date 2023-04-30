import torch
#  import numpy as np


from src.models.wp_imitation_model import WPImitationModel
from src.envs.carla.features.scenarios import transform_points
#  from src.models import viz_utils


# TODO this is just open loop plan
class WPPlanModel(WPImitationModel):
    # TODO planning args
    def __init__(
            self,
            *args,
            plan_steps=1,
            plan_horizon=1,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._plan_steps = plan_steps
        self._plan_horizon = plan_horizon

    @torch.no_grad()
    #  def get_action(self, full_obs, long_route, return_features=False):
    def get_action(self, full_obs, return_features=False):
        obs = {}
        for k in full_obs:
            if k in self._dynamic_feature_names:
                obs[k] = full_obs[k][:, -self.H:]
            else:
                obs[k] = full_obs[k]
        actions = self.mle_plan(obs)
        return actions

    # TODO all other routes need to be done with respect to all modes for ego vehicle
    def get_features(self, obs, wp_diffs, return_features=False):
        new_vehicle_features = []
        history_vehicle_features = obs['vehicle_features'][..., :self.wp_dim].repeat(self.num_modes, 1, 1, 1)
        for t in range(self.T):
            history_vehicle_features = torch.cat([
                history_vehicle_features[:, 1:],
                obs['vehicle_features'][..., :self.wp_dim] + wp_diffs[:, :, t].view((self.num_modes, self.H, self.num_agents, self.wp_dim))
            ], dim=1)
            new_vehicle_features.append(history_vehicle_features)
        new_vehicle_features = torch.stack(new_vehicle_features, dim=1).view((self.num_modes, self.T, self.H, self.num_agents, self.wp_dim))
        # TODO for now assume same z
        new_refs = obs['ref'].repeat(self.num_modes, self.T, 1).clone()
        new_refs[:, :, [0, 1, 3]] = new_refs[:, :, [0, 1, 3]] + self.max_token_distance * wp_diffs[:, :, :, 0, :3].view((self.num_modes, self.T, 3))

        new_obs = {}
        new_obs['town'] = obs['town'].repeat(self.num_modes)
        new_obs['ref'] = new_refs
        ego_ref = new_vehicle_features[:, :, -1, 0, :3]

        new_obs['ego_vehicle_features'] = new_vehicle_features[:, :, :, 0].clone()
        new_obs['ego_vehicle_features'][..., :3] = transform_points(
            new_obs['ego_vehicle_features'][..., :3].reshape((self.num_modes, self.T, -1, 3)),
            ego_ref).reshape((self.num_modes, self.T, self.H, 3))
        new_obs['ego_vehicle_masks'] = obs['vehicle_masks'][:, None, -1:, 0].clone().repeat(self.num_modes, self.T, self.H) & (new_obs['ego_vehicle_features'][..., :2].abs() <= 1.).all(dim=-1)

        # TODO other vehicle features
        # num_modes, T, num_modes, H, num_agents-1, wp_dim
        new_obs['other_vehicle_features'] = new_vehicle_features[None, :, :, :, 1:].repeat(self.num_modes, 1, 1, 1, 1, 1).transpose(1, 2)
        new_obs['other_vehicle_features'][..., :3] = transform_points(
            new_obs['other_vehicle_features'][..., :3].reshape((self.num_modes, self.T, -1, 3)),
            ego_ref).reshape((self.num_modes, self.T, self.num_modes, self.H, self.num_agents - 1, 3))
        new_obs['other_vehicle_masks'] = obs['vehicle_masks'][:, None, None, -1:, 1:].clone().repeat(self.num_modes, self.T, self.num_modes, self.H, 1) & (obs['vehicle_masks'][:, None, None, -1:, :1].clone()) & (new_obs['other_vehicle_features'][..., :2].abs() <= 1.).all(dim=-1)

        # TODO walker
        new_obs['walker_features'] = obs['walker_features'][:, None, -1:].clone().repeat(self.num_modes, self.T, self.H, 1, 1)
        new_obs['walker_features'][..., :2] = transform_points(
            new_obs['walker_features'][..., :2].reshape((self.num_modes, self.T, -1, 2)),
            ego_ref).reshape((self.num_modes, self.T, self.H, -1, 2))
        new_obs['walker_masks'] = obs['walker_masks'][:, None, -1:].clone().repeat(self.num_modes, self.T, self.H, 1) & (new_obs['walker_features'][..., :2].abs() <= 1.).all(dim=-1)

        # TODO assuming lights stay the same
        new_obs['light_features'] = obs['light_features'][:, None, -1:].clone().repeat(self.num_modes, self.T, self.H, 1, 1)
        new_obs['light_features'][..., :2] = transform_points(
            new_obs['light_features'][..., :2].reshape((self.num_modes, self.T, -1, 2)),
            ego_ref).reshape((self.num_modes, self.T, self.H, -1, 2))
        new_obs['light_masks'] = obs['light_masks'][:, None, -1:].clone().repeat(self.num_modes, self.T, self.H, 1) & (new_obs['light_features'][..., :2].abs() <= 1.).all(dim=-1)

        new_obs['stop_features'] = obs['stop_features'][:, None].clone().repeat(self.num_modes, self.T, 1, 1)
        new_obs['stop_features'] = transform_points(
            new_obs['stop_features'],
            ego_ref)
        new_obs['stop_masks'] = obs['stop_masks'][:, None].clone().repeat(self.num_modes, self.T, 1) & (new_obs['stop_features'][..., :2].abs() <= 1.).all(dim=-1)

        # TODO maybe find way to mask out route points you have already passed
        new_obs['route_features'] = obs['route_features'][:, None].clone().repeat(self.num_modes, self.T, 1, 1)
        new_obs['route_features'] = transform_points(
            new_obs['route_features'],
            ego_ref)
        new_obs['route_masks'] = obs['route_masks'][:, None].clone().repeat(self.num_modes, self.T, 1) & (new_obs['route_features'][..., :2].abs() <= 1.).all(dim=-1)

        for k in new_obs:
            if 'features' in k:
                mask_k = k[:-8] + 'masks'
                new_obs[k][~new_obs[mask_k]] = 0.

        return new_obs

    def get_reward(self, features, obs):
        speed = features['ego_vehicle_features'][:, :, -1, 3]
        speed_limit = obs['vehicle_features'][:, -1, 0, 6]
        speed_pen = -1 * (((speed - speed_limit) / speed_limit) ** 2)

        dist = 0.25
        # TODO time
        vehicle_dists = torch.norm(features['other_vehicle_features'][:, :, :, -1, :, :2], dim=-1)
        all_veh_pen = torch.where(
            features['other_vehicle_masks'][:, :, :, -1] & (vehicle_dists < dist),
            -1 * (((vehicle_dists - dist) / dist) ** 2),
            torch.zeros_like(vehicle_dists))

        min_veh_pen = all_veh_pen.min(dim=2)[0]
        veh_pen = min_veh_pen.sum(dim=-1)

        # TODO figure out which route pt to compare to and get distance from traj
        route_pts = features['route_features'][..., :2]
        route_dists = torch.where(
            features['route_masks'],
            route_pts.norm(dim=-1),
            torch.inf)
        min_route_dist = 2. / self.max_token_distance
        close_mask = route_dists <= min_route_dist
        last_close_ind = close_mask.cumsum(dim=-1).argmax(dim=-1)

        closest_ind = route_dists.argmin(dim=-1)

        close_ind = torch.where(
            #  last_close_ind == 0,
            close_mask.sum(dim=-1) == 0,
            closest_ind,
            last_close_ind)

        # TODO make sure this is reasonable even if you outrun all your points
        close_pt = route_pts[torch.arange(self.num_modes, device=self.device).unsqueeze(1), torch.arange(self.T, device=self.device).unsqueeze(0), close_ind]

        next_ind = torch.clamp(close_ind + 1, max=close_pt.shape[2])
        prev_ind = close_ind - 1
        #  other_ind = torch.where(
            #  (next_ind != close_ind) & features['route_masks'][torch.arange(self.num_modes).unsqueeze(1), torch.arange(self.T).unsqueeze(0), next_ind],
            #  next_ind,
            #  prev_ind)
        #  other_ind = torch.where(
            #  close_ind == 0,
            #  next_ind,
            #  prev_ind)
        other_ind = prev_ind
        other_ind = torch.where(
            features['route_masks'][torch.arange(self.num_modes).unsqueeze(1), torch.arange(self.T).unsqueeze(0), other_ind],
            other_ind,
            -1)
        other_pt = torch.where(
            (other_ind == -1).unsqueeze(-1),
            torch.zeros_like(close_pt),
            route_pts[torch.arange(self.num_modes, device=self.device).unsqueeze(1), torch.arange(self.T, device=self.device).unsqueeze(0), other_ind])

        a = other_pt - close_pt
        b = close_pt
        traj_dist = (a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]) / (torch.norm(a, dim=-1) + 1.e-6)

        # TODO use lane width
        #  route_diff = torch.abs(close_pt[..., 1])
        route_diff = torch.abs(traj_dist)
        #  if (route_diff > (3.5 / self.max_token_distance)).any():
            #  import pdb; pdb.set_trace()
        #  route_diff = torch.where(
            #  route_diff < (3.5 / self.max_token_distance),
            #  torch.zeros_like(route_diff),
            #  route_diff)
        route_pen = -route_diff
        #  if (route_pen < 0.).any():
            #  import pdb; pdb.set_trace()

        speed_coeff = 0.1
        dist_coeff = 1.
        route_coeff = 1.
        reward = speed_coeff * speed_pen + dist_coeff * veh_pen + route_coeff * route_pen
        return reward

    # TODO
    def mle_plan(self, obs):
        pred, logits, _ = self.forward(obs, return_act=False)
        wp_diffs = pred[..., :self.wp_dim]
        features = self.get_features(obs, wp_diffs)
        rewards = self.get_reward(features, obs)
        # TODO weighted sum
        values = rewards.reshape((self.num_modes, self.T)).sum(dim=-1)
        print(values)

        ego_labels = values.argmax(dim=-1)
        #  ego_labels = logits[0, :, 0].argmax(dim=-1)
        #  import pdb; pdb.set_trace()
        ego_pred_next_wps = wp_diffs[..., 0, :]
        next_ego_wps = ego_pred_next_wps[torch.arange(ego_pred_next_wps.shape[0], device=ego_labels.device), ego_labels]
        pred = self._inverse_action(obs, next_ego_wps)
        actions = pred[..., :self.act_dim]
        return actions
