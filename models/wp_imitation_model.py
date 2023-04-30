import torch
from torch import nn
import numpy as np

from src.models.utils import get_cosine_schedule_with_warmup, PIDController
from src.models.imitation_vq_diffusion import ImitationModel
from src.models.output_models import ResOutputModel
from src.models.inverse_dynamics_models import InverseDynamicsModel
from src.envs.carla.features.scenarios import transform_points
from src.models import viz_utils


class WPImitationModel(ImitationModel):
    def __init__(
            self,
            *args,
            inv_T=1,
            inv_layers=1,
            wp_dim=4,
            dt=0.05,
            out_init_std=0.05,
            act_init_std=0.1,
            action_mean=None,
            action_std=None,
            std_update_tau=1.e-3,
            wp_pred_type='frame',
            **kwargs,
    ):
        self.inv_T = inv_T
        self.inv_layers = inv_layers
        self.wp_dim = wp_dim
        self.dt = dt
        self.out_init_std = out_init_std
        self.act_init_std = act_init_std
        self.action_mean = action_mean
        self.action_std = action_std
        self.std_update_tau = std_update_tau
        self.wp_pred_type = wp_pred_type

        self.turn_controller = PIDController(K_P=0.9, K_I=0.75, K_D=0.3, n=20)
        self.speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=20)

        super().__init__(*args, **kwargs)

    def create_output_model(self):
        self.output_model = ResOutputModel(
            emb_dim=self.emb_dim,
            dist_dim=self.wp_dim,
            min_std=self.min_std,
            layer_norm=self.norm_first,
            dropout=self.dropout,
            #  out_mean=self.output_mean[self.H:self.H+self.T, None] if self.output_mean is not None else None,
            #  out_std=self.output_std[self.H:self.H+self.T, None] if self.output_std is not None else None,
            # TODO best way to do this
            out_std=self.out_init_std * self.dt * self.f * torch.ones((self.T, self.wp_dim)),
            #  out_std=self.out_init_std * torch.ones((self.T, self.wp_dim)),
            #  wa_std=True,
        )

        self.inverse_dynamics_model = InverseDynamicsModel(
            inp_dim=1 + self.wp_dim * self.inv_T, # TODO
            emb_dim=self.emb_dim,
            dist_dim=self.act_dim,
            min_std=self.min_std,
            num_hidden=self.inv_layers,
            norm='layer',
            dropout=self.dropout,
            #  action_mean=self.action_mean[self.H-1] if self.action_mean is not None else None,
            #  action_std=self.action_std[self.H-1] if self.action_std is not None else None,
            #  action_std=self.act_init_std * torch.ones((self.act_dim,)),
            #  tanh=True,
            #  wa_std=True,
        )

    def get_prob_decoding(self, agents_emb, agents_masks, map_emb, map_masks, route_emb, route_masks, features):
        B = agents_emb.shape[0]
        num_agents = agents_masks.shape[-1]

        # TODO maybe detach
        prob_seq = agents_emb.reshape((B, 1, -1, num_agents, self.emb_dim))

        prob_seq = prob_seq[:, :, -1, :, :].repeat(1, self.num_modes, 1, 1)
        prob_masks = agents_masks.reshape((B, 1, -1, num_agents))
        prob_masks = prob_masks[:, :, -1, :].repeat(1, self.num_modes, 1)

        prob_seq = prob_seq + self.P.repeat(B, 1, num_agents, 1)

        prob_seq = prob_seq.view((B, self.num_modes, num_agents, -1))
        prob_masks = prob_masks.view((B, self.num_modes, num_agents))

        for d in range(self.num_dec_layers):
            prob_seq = self.map_dec_fn(
                prob_seq,
                prob_masks.clone(),
                map_emb,
                map_masks.clone(),
                layer=self.prob_map_dec_layers[d],
                route_emb=route_emb,
                route_masks=route_masks.clone(),
            )
            prob_seq = self.dec_fn(
                prob_seq,
                prob_masks.clone(),
                agents_emb,
                agents_masks.clone(),
                layer=self.prob_tx_decoder[d])

        prob_seq = prob_seq.view((B, self.num_modes, num_agents, -1))

        return prob_seq

    def get_prob_output(self, prob_seq):
        B = prob_seq.shape[0]
        num_modes = prob_seq.shape[1]
        num_agents = prob_seq.shape[2]
        logits = self.prob_model(prob_seq).reshape((B, num_modes, num_agents))
        return logits

    def get_pred_next_wps(self, obs, outputs):
        if self.wp_pred_type == 'frame':
            output_thetas = torch.cumsum(outputs[..., 2], dim=2)
            output_speeds = torch.cumsum(outputs[..., 3:4], dim=2)
            output_others = outputs[..., 4:self.wp_dim]

            thetas = output_thetas - outputs[..., 2]

            rot_matrix = torch.stack([
                torch.stack([ torch.cos(thetas), torch.sin(thetas)], dim=-1),
                torch.stack([-torch.sin(thetas), torch.cos(thetas)], dim=-1)
            ], dim=-2)

            output_pos_diffs = torch.matmul(outputs[..., None, :2], rot_matrix).squeeze(-2)
            output_pos = torch.cumsum(output_pos_diffs, dim=2)

            outputs_mean = torch.cat([output_pos, output_thetas.unsqueeze(-1), output_speeds, output_others], dim=-1)

            # cumulative std
            outputs_std = torch.sqrt(torch.cumsum(outputs[..., self.wp_dim:] ** 2, dim=2))
            pred_next_wps = torch.cat([outputs_mean, outputs_std], dim=-1)
        elif self.wp_pred_type == 'bicycle':
            B = outputs.shape[0]
            num_agents = outputs.shape[3]

            d_thetas = outputs[..., 2]
            d_speeds = outputs[..., 3]

            # TODO dt already incorporated in out_init_std
            delta_total_thetas = torch.cumsum(d_thetas, dim=2)
            delta_total_speeds = torch.cumsum(d_speeds, dim=2)
            thetas = obs['vehicle_features'][:, -1, :, 2].reshape((B, 1, 1, num_agents)) + delta_total_thetas
            speeds = obs['vehicle_features'][:, -1, :, 3].reshape((B, 1, 1, num_agents)) + delta_total_speeds

            eff_thetas = thetas - 0.5 * d_thetas
            eff_speeds = speeds - 0.5 * d_speeds

            dxs = eff_speeds * torch.cos(eff_thetas) * self.dt * self.f
            xs = torch.cumsum(dxs, dim=2)
            dys = eff_speeds * torch.sin(eff_thetas) * self.dt * self.f
            ys = torch.cumsum(dys, dim=2)

            outputs_mean = torch.stack([xs, ys, thetas, speeds], dim=-1)

            outputs_std = torch.sqrt(torch.cumsum(outputs[..., self.wp_dim:] ** 2, dim=2))
            pred_next_wps = torch.cat([outputs_mean, outputs_std], dim=-1)
        else:
            raise NotImplementedError
        return pred_next_wps


    def forward(self, obs, gt_wps, wps_mask, modes=None, return_wp=True, return_act=True):
        if return_wp:
            diffusion_kl_loss, pred_index, logits, features = super().forward(obs, gt_wps, wps_mask, modes=modes)

            # pred_next_wps = self.get_pred_next_wps(obs, outputs)

        if not return_act:
            return diffusion_kl_loss, pred_index, logits, features # pred_next_wps
        else:
            # TODO modes and time
            if gt_next_wps is None:
                ego_pred_next_wps = pred_next_wps[..., 0, :self.wp_dim]
                if logits is None:
                    next_ego_wps = ego_pred_next_wps[:, 0:1]
                else:
                    B = logits.shape[0]
                    ego_labels = logits[:, :, 0].argmax(dim=1)
                    next_ego_wps = ego_pred_next_wps[torch.arange(B, device=ego_labels.device), ego_labels]
            else:
                next_ego_wps = gt_next_wps[:, :, 0]

            acts = self._inverse_action(obs, next_ego_wps)

            # TODO
            if return_wp:
                return pred_next_wps, acts, logits, features
            else:
                return acts

    def _inverse_action(self, obs, next_ego_wps):
        ego_speeds = obs['vehicle_features'][:, self.H - 1, 0, 3:4]
        B = ego_speeds.shape[0]
        inps = torch.cat([ego_speeds, next_ego_wps[:, :self.inv_T].reshape((B, self.wp_dim * self.inv_T))], dim=-1)
        # TODO smarter version of this
        #  if self.training:
            #  if self.output_std is not None:
                #  inps = inps + torch.randn_like(inps) * 0.01 * self.output_std.flatten()[self.wp_dim-1:self.wp_dim*(1+self.inv_T)].to(device=inps.device)
            #  else:
                #  inps = inps + torch.randn_like(inps) * 1.e-4
        acts = self.inverse_dynamics_model(inps)

        return acts

    @torch.inference_mode()
    def get_action(self, full_obs, return_features=False):
        obs = {}
        for k in full_obs:
            if k in self._dynamic_feature_names:
                obs[k] = full_obs[k][:, -self.H:]
            else:
                obs[k] = full_obs[k]
        pred_wps, pred_actions, logits, features = self.forward(obs)
        self.prev_preds = [pred_wps, pred_actions, logits]

        actions = pred_actions[:, :self.act_dim]
        if return_features:
            return actions, features
        else:
            return actions

    def _compute_loss(self, pred_wps, logits, gt_wps, wps_mask):
        B = pred_wps.shape[0]
        num_agents = wps_mask.shape[-1]
        shaped_pred_wps = pred_wps.reshape((B * self.num_modes, self.T, num_agents, 2 * self.wp_dim))
        shaped_gt_wps = gt_wps.unsqueeze(1).repeat_interleave(self.num_modes, 1).reshape((B * self.num_modes, self.T, num_agents, self.wp_dim))
        shaped_wps_regression_loss, shaped_wps_mse_errors = self._compute_regression_loss(shaped_pred_wps, shaped_gt_wps)

        if wps_mask is None:
            wps_mask = torch.ones(gt_wps.shape[:-1], dtype=bool, device=gt_wps.device)

        if self.num_modes > 1:
            time_wps_regression_loss = shaped_wps_regression_loss.reshape((B, self.num_modes, self.T, num_agents)).permute(0, 3, 1, 2).reshape((B * num_agents, self.num_modes, self.T))
            time_wps_mse_errors = shaped_wps_mse_errors.reshape((B, self.num_modes, self.T, num_agents, self.wp_dim)).permute(0, 3, 1, 2, 4).reshape((B * num_agents, self.num_modes, self.T, self.wp_dim))

            w_mask = wps_mask.unsqueeze(1).permute(0, 3, 1, 2).reshape((B * num_agents, 1, self.T))
            # TODO should a mean be done here
            wps_regression_loss = (time_wps_regression_loss * w_mask).sum(dim=2) / torch.clamp(w_mask.sum(dim=2), min=1.)
            wps_mse_errors = (time_wps_mse_errors * w_mask.unsqueeze(-1)).sum(dim=2) / torch.clamp(w_mask.unsqueeze(-1).sum(dim=2), min=1.)

            logits = logits.transpose(1, 2).reshape((B * num_agents, self.num_modes))

            wps_mse_error = wps_mse_errors.sum(dim=-1)

            wps_m = w_mask.squeeze(1).any(dim=1)

            unmasked_wps_loss, _, unmasked_logit_loss, labels, target_labels = self._compute_modes_loss(logits, wps_mse_error, wps_regression_loss)
            wps_loss = unmasked_wps_loss[wps_m].mean()
            logit_loss = unmasked_logit_loss[wps_m].mean()

            wps_mse_errors = wps_mse_errors[torch.arange(labels.shape[0], device=labels.device), labels][wps_m].mean(dim=0)
            time_wps_min_mse_errors = time_wps_mse_errors[torch.arange(target_labels.shape[0], device=target_labels.device), target_labels]
            time_wps_mse = (time_wps_min_mse_errors * w_mask.reshape((B * num_agents, self.T, 1))).sum(dim=0) / torch.clamp(w_mask.reshape((B * num_agents, self.T, 1)).sum(dim=0), min=1.)
        else:

            time_wps_regression_loss = shaped_wps_regression_loss.permute(0, 2, 1).reshape((B * num_agents, self.T))
            time_wps_mse_errors = shaped_wps_mse_errors.permute(0, 2, 1, 3).reshape((B * num_agents, self.T, self.wp_dim))
            w_mask = wps_mask.permute(0, 2, 1).reshape((B * num_agents, self.T))

            wps_regression_loss = (time_wps_regression_loss * w_mask).sum(dim=1) / torch.clamp(w_mask.sum(dim=1), min=1.)
            wps_mse_errors = (time_wps_mse_errors * w_mask.unsqueeze(-1)).sum(dim=1) / torch.clamp(w_mask.unsqueeze(-1).sum(dim=1), min=1.)

            wps_m = w_mask.any(dim=1)
            wps_loss = wps_regression_loss[wps_m].mean()
            wps_mse_errors = wps_mse_errors[wps_m].mean(dim=0)

            time_wps_mse = (time_wps_mse_errors * w_mask.unsqueeze(-1)).sum(dim=0) / torch.clamp(w_mask.unsqueeze(-1).sum(dim=0), min=1.)

            logit_loss = 0.

        wps_errors = torch.sqrt(wps_mse_errors)
        time_wps_errors = torch.sqrt(time_wps_mse)
        return wps_loss, logit_loss, wps_errors, time_wps_errors

    def _compute_action_loss(self, pred_actions, gt_actions, wps_mask, actions_mask):
        # TODO is this right because we are using wp to generate actions
        actions_mask = actions_mask & wps_mask[:, 0, 0]
        act_regression_loss, act_mse_errors = self._compute_regression_loss(pred_actions, gt_actions)
        act_loss = act_regression_loss[actions_mask].mean()
        act_mse_errors = act_mse_errors[actions_mask].mean(dim=0)
        act_errors = torch.sqrt(act_mse_errors)

        return act_loss, act_errors

    def run_step(self, batch, prefix, optimizer_idx=None):
        full_obs = batch.get_obs(ref_t=self.H-1, f=self.f, device=self.device)
        obs = {}
        for k in full_obs:
            if k in self._dynamic_feature_names:
                obs[k] = full_obs[k][:, :self.H]
            else:
                obs[k] = full_obs[k]

        gt_wps, wps_mask = batch.get_traj(ref_t=self.H-1, f=self.f, num_features=self.wp_dim)
        gt_wps = gt_wps[:, self.H:self.H+self.T, :, :self.wp_dim].to(self.device)
        wps_mask = wps_mask[:, self.H:self.H+self.T].to(self.device)

        if prefix == 'val':
            diffusion_kl_loss, diffusion_pred_index, logits, features = super().sample(obs)

        elif (optimizer_idx is None) or (optimizer_idx == 0):
            #  pred_wps, pred_actions, logits, features = self.forward(obs, gt_wps)
            diffusion_kl_loss, diffusion_pred_index, logits, features = self.forward(obs, gt_wps, wps_mask, return_act=False)

        outputs = self.diffusion.quantize.get_decoded_wps(diffusion_pred_index)
        pred_wps = self.get_pred_next_wps(obs, outputs)

        wps_loss, logit_loss, wps_errors, time_wps_errors = self._compute_loss(pred_wps, logits, gt_wps, wps_mask)
        wps_error = wps_errors.sum(dim=-1)
            #  self.output_model.update_std(time_wps_errors, tau=self.std_update_tau)

        if self.use_wandb:
            batch_size = gt_wps.shape[0]
            self.log(f'{prefix}/wps_loss', wps_loss.detach(), batch_size=batch_size)
            self.log(f'{prefix}/wps_error', wps_error.detach(), batch_size=batch_size)
            self.log(f'{prefix}/diffusion_kl_loss', diffusion_kl_loss.detach(), batch_size=batch_size)

                # for i in range(len(wps_errors)):
                #     self.log(f'{prefix}/wps_error[{i}]', wps_errors[i].detach(), batch_size=batch_size)

                # if self.num_modes > 1:
                #     self.log(f'{prefix}/logit_loss', logit_loss.detach(), batch_size=batch_size)

        if optimizer_idx is None:
            self.prev_preds = [pred_wps, gt_wps, logits]
            return diffusion_kl_loss, pred_wps, features
        elif optimizer_idx == 0:
            return diffusion_kl_loss, pred_wps, features
        else:
            raise NotImplementedError

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        loss, *_ = self.run_step(batch, prefix='train', optimizer_idx=optimizer_idx)

        #  if self.use_scheduler:
            #  if self.use_wandb:
                #  self.log('train/lr', self.scheduler.get_last_lr()[0])

        return loss

    def validation_step(self, batch, batch_idx):
        wp_loss, pred_actions, features = self.run_step(batch, prefix='val')

        if batch_idx == 0:
            if self.visualize:
                viz = self.get_visualizations(features)
            if self.use_wandb and self.visualize:
                images = viz.transpose(0, 2, 1, 3, 4).reshape((viz.shape[0] * viz.shape[2], viz.shape[1] * viz.shape[3], viz.shape[-1]))
                self.logger.log_image(key="val/viz", images=[images])

        return wp_loss, pred_actions

    def configure_optimizers(self):
        all_params = set(self.parameters())
        # all_params = all_params - act_params
        wd_params = set()
        for m in self.modules():
            # TODO see if you need to add any other layers to this list
            if isinstance(m, nn.Linear):
                wd_params.add(m.weight)
                #  TODO should we remove biases
                #  wd_params.add(m.bias)
            #  else:
                #  if hasattr(m, 'bias'):
                    #  wd_params.add(m.bias)
        no_wd_params = all_params - wd_params
        main_optimizer = torch.optim.AdamW(
            [{'params': list(no_wd_params), 'weight_decay': 0}, {'params': list(wd_params), 'weight_decay': self.wd}],
            lr=self.lr)
        main_scheduler = get_cosine_schedule_with_warmup(main_optimizer, self.warmup_steps, self.trainer.estimated_stepping_batches)
        main_d = {"optimizer": main_optimizer, "lr_scheduler": {"scheduler": main_scheduler, "interval": "step"}}
        # TODO wd or lr schedule for inverse dynamics model
        # act_optimizer = torch.optim.AdamW(list(act_params), lr=self.lr)
        # #  act_d = {"optimizer": act_optimizer, "lr_scheduler": torch.optim.lr_scheduler.ExponentialLR(act_optimizer, gamma=0.98)}
        # act_scheduler = get_cosine_schedule_with_warmup(act_optimizer, self.warmup_steps, self.trainer.estimated_stepping_batches)
        # act_d = {"optimizer": act_optimizer, "lr_scheduler": {"scheduler": act_scheduler, "interval": "step"}}
        d = [main_d]#act_d
        return d

    # TODO
    @torch.inference_mode()
    def get_pid_action(self, full_obs, return_features=False):
        obs = {}
        for k in full_obs:
            if k in self._dynamic_feature_names:
                obs[k] = full_obs[k][:, -self.H:]
            else:
                obs[k] = full_obs[k]
        pred_wps, logits, features = self.forward(obs, return_act=False)

        if logits is None:
            ego_waypoints = pred_wps[0, 0, :, 0, :]
        else:
            ego_logits = logits[0, :, 0]
            ego_labels = ego_logits.argmax()

            ego_wp_diffs = pred_wps[0, :, :, 0, :]
            ego_waypoints = ego_wp_diffs[ego_labels]
        ego_speed = obs['vehicle_features'][0, -1, 0, 3]

        actions = self.control_pid(ego_waypoints, ego_speed)

        self.prev_preds = [pred_wps, actions, logits]
        if return_features:
            return actions, features
        else:
            return actions

    def control_pid(self, waypoints, curr_speed):
        wps = waypoints[:, :2].data.cpu().numpy() * self.max_token_distance
        speed = curr_speed.data.cpu().numpy() * self.max_token_distance

        #  desired_speed = 2. * np.linalg.norm(wps[1] - wps[0], axis=-1)
        #  desired_speed = np.linalg.norm(wps[1] + wps[0], axis=-1)
        #  desired_speed = np.linalg.norm(wps[1], axis=-1)
        #  desired_speed = 2. * np.linalg.norm(wps[0], axis=-1)
        #  desired_speed = 4. * np.linalg.norm(wps[0], axis=-1) - speed
        desired_speed = speed + waypoints[0, 3].data.cpu().numpy() * self.max_token_distance
        #  desired_speed = speed + waypoints[:2, 3].sum().data.cpu().numpy() * self.max_token_distance

        brake = desired_speed < 0.4 or ((speed / desired_speed) > 1.1)
        #  brake = desired_speed < 0.3 or ((speed / desired_speed) > 1.05)

        delta = np.clip(desired_speed - speed, 0.0, 0.25)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.75)
        if brake:
            gas = -1.0
        else:
            gas = throttle

        #  aim = (wps[1] + wps[0]) * 0.5
        #  aim = 0.5 * wps[1]
        #  aim = wps[0]
        aim = 0.5 * wps[0]

        angle = np.arctan2(aim[1], aim[0]) * 2. / np.pi
        if brake:
            angle = 0.0
        steer = self.turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)

        return steer, gas

    # TODO might be better in visualizing or just everywhere to have num agents before time
    @torch.inference_mode()
    def get_visualizations(self, features=None, preds=None, obs=None):
        if preds is None:
            preds = self.prev_preds
        scene_image = super().get_visualizations(features=features, obs=obs)

        num_samples = scene_image.shape[0]
        pred_image = np.zeros((num_samples, 3, self.im_shape[0], self.im_shape[1], 3), dtype=np.uint8)

        # TODO make sure this is right in all cases
        #  agents_features, agents_masks, light_features, light_masks, stop_features, stop_masks, walker_features, walker_masks, route_features, route_masks, map_features, map_masks = features
        if features is not None:
            agents_features, agents_masks, light_features, light_masks, stop_features, stop_masks, walker_features, walker_masks, route_features, route_masks, map_features, map_masks = features
        else:
            assert(obs is not None)
            agents_features, agents_masks, light_features, light_masks, stop_features, stop_masks, walker_features, walker_masks, route_features, route_masks, map_features, map_masks = self.process_observations(obs)

        vehicles = agents_features.detach().cpu()[:, -1]
        vehicles_masks = ~agents_masks.detach().cpu()[:, -1]

        conversion = np.array([-1., 1.])

        if preds[2] is None:
            labels = torch.zeros((vehicles.shape[0], vehicles.shape[1]), dtype=torch.int64, device=vehicles.device)
            order = torch.arange(self.num_modes, device=vehicles.device)[None, :, None].repeat(vehicles.shape[0], 1, vehicles.shape[1])
        else:
            logits = preds[2].detach().cpu()
            labels = logits.argmax(dim=1)
            order = logits.argsort(dim=1)
            # TODO correct way to do this
            #  mode_colors = viz_utils.get_mode_colors(self.num_modes)[order][..., :3]

        pred_wps = preds[0][..., :2].detach().cpu()
        gt_wps = preds[1][..., :2].detach().cpu()
        B = pred_wps.shape[0]
        num_agents = agents_masks.shape[-1]

        transformed_wps = transform_points(
            pred_wps.reshape((B, self.num_modes * self.T, num_agents, 2)).transpose(1, 2),
            vehicles[..., :3],
            invert=True).transpose(1, 2).reshape((B, self.num_modes, self.T, num_agents, 2))

        transformed_gt_wps = transform_points(
            gt_wps.reshape((B, self.num_modes * self.T, num_agents, 2)).transpose(1, 2),
            vehicles[..., :3],
            invert=True).transpose(1, 2).reshape((B, self.num_modes, self.T, num_agents, 2))

        shaped_order = order.unsqueeze(2).repeat_interleave(self.T, 2).permute(0, 2, 3, 1).reshape((-1, self.num_modes))
        shaped_transformed_wps = transformed_wps.permute(0, 2, 3, 1, 4).reshape((-1, self.num_modes, 2))
        ordered_shaped_transformed_wps = shaped_transformed_wps[torch.arange(shaped_transformed_wps.shape[0], device=shaped_transformed_wps.device).unsqueeze(1), shaped_order]
        ordered_transformed_wps = ordered_shaped_transformed_wps.reshape((B, self.T, num_agents, self.num_modes, 2))

        shaped_transformed_gt_wps = transformed_gt_wps.permute(0, 2, 3, 1, 4).reshape((-1, self.num_modes, 2))
        ordered_shaped_transformed_gt_wps = shaped_transformed_gt_wps[torch.arange(shaped_transformed_gt_wps.shape[0], device=shaped_transformed_gt_wps.device).unsqueeze(1), shaped_order]
        ordered_transformed_gt_wps = ordered_shaped_transformed_gt_wps.reshape((B, self.T, num_agents, self.num_modes, 2))

        for row_idx in range(num_samples):
            # TODO adjust color based on time
            masked_wps = ordered_transformed_wps[row_idx][:, vehicles_masks[row_idx]].numpy().reshape((-1, self.num_modes, 2))
            wp_locs = np.clip((0.5 * scene_image.shape[-2] * (conversion * masked_wps + 1.)).astype(np.int32), 0, scene_image.shape[-2] - 1)
            mode_colors = viz_utils.get_mode_colors(self.num_modes)[None, :, :3].repeat(wp_locs.shape[0], 0)
            pred_image[row_idx, 0] = viz_utils.get_image(wp_locs.reshape((-1, 2)), mode_colors.reshape((-1, 3)), im_shape=self.im_shape)

            masked_gt_wps = ordered_transformed_gt_wps[row_idx][:, vehicles_masks[row_idx]].numpy().reshape((-1, self.num_modes, 2))
            gt_wp_locs = np.clip((0.5 * scene_image.shape[-2] * (conversion * masked_gt_wps + 1.)).astype(np.int32), 0, scene_image.shape[-2] - 1)
            pred_image[row_idx, 2] = viz_utils.get_image(gt_wp_locs.reshape((-1, 2)), mode_colors.reshape((-1, 3)), im_shape=self.im_shape)

            ego_masked_wps = ordered_transformed_wps[row_idx][:, 0].numpy().reshape((-1, self.num_modes, 2))
            ego_wp_locs = np.clip((0.5 * scene_image.shape[-2] * (conversion * ego_masked_wps + 1.)).astype(np.int32), 0, scene_image.shape[-2] - 1)
            ego_mode_colors = viz_utils.get_mode_colors(self.num_modes)[None, :, :3].repeat(ego_wp_locs.shape[0], 0)
            pred_image[row_idx, 1] = viz_utils.get_image(ego_wp_locs.reshape((-1, 2)), ego_mode_colors.reshape((-1, 3)), im_shape=self.im_shape)

        compact_scene_image = np.stack([
            scene_image[:, 0],
            scene_image[:, 1:1+self.num_enc_layers+self.num_dec_layers].mean(axis=1),
            scene_image[:, 1+self.num_enc_layers+self.num_dec_layers:1+2*self.num_enc_layers+2*self.num_dec_layers].mean(axis=1),
        ], axis=1)
        return np.concatenate([compact_scene_image, pred_image], axis=1)