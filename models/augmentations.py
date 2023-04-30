from copy import deepcopy
import torch

"""
Non-behaviorally-equivalent augmentations aren't applied here -- that's handled in the dataset proper
Can still use them, but we don't want to induce invariation to them (e.g. if we perturb the ego, that's a different state now)

Behaviorally-Equivalent Augmentations
1. Window crop: crop sub-region of scene graph (if we rescale back, then that's random shift)
2. Gaussian jitter: add random Gaussian noise to all entity positions
3. Rotation: randomly rotate entire graph around ego
The rest should probably not be done when joint training with RL...
4. Random shift: window crop, but then rescale entire graph back
5. Token dropout: randomly dropout tokens (this is not really behaviorally equivalent...)

Pretraining tasks
1. Instance discrimination:
    For each x, augment twice to get x_1, x_2.
    Task is to match x_1 to x_2 (negative samples from other minibatch pairs)
    Follows SimCLR / CURL
2. Masked reconstruction / in-painting:
    Mask out a bunch of tokens, or like a patch of the graph
    Then reconstruct it
"""


class SceneTokenAugmentation:
    def __init__(self,
        window_crop_offset=.3,
        gaussian_jitter=.01,
        random_rotation=0.0
    ):
        self.window_crop_offset = window_crop_offset
        self.gaussian_jitter = gaussian_jitter
        self.random_rotation = random_rotation

    def __call__(self, scene_graph):
        with torch.no_grad():
            scene_graph = deepcopy(scene_graph)
            batch_size = scene_graph['vehicle_features'].shape[0]

            # Sample augmentation params
            bounding_box = torch.stack([-torch.ones(2), torch.ones(2)])[None].repeat(batch_size,1,1)
            bounding_box += torch.zeros((batch_size,2)).uniform_(-self.window_crop_offset, self.window_crop_offset)[:,None]
            bounding_box = bounding_box.to(scene_graph['vehicle_features'].device)

            vehicle_features, vehicle_masks = scene_graph['vehicle_features'], scene_graph['vehicle_masks']
            vehicle_features[vehicle_masks] += torch.normal(0, self.gaussian_jitter, vehicle_features[vehicle_masks].shape, device=vehicle_features.device)
            vehicle_features[:,:,:2] = vehicle_features[:,:,:2].clamp(bounding_box[:,None,0], bounding_box[:,None,1])

            dynamic_features, dynamic_masks = scene_graph['dynamic_features'], scene_graph['dynamic_masks']
            dynamic_features[dynamic_masks] += torch.normal(0, self.gaussian_jitter, dynamic_features[dynamic_masks].shape, device=dynamic_features.device)
            dynamic_features[:,:,:2] = dynamic_features[:,:,:2].clamp(bounding_box[:,None,0], bounding_box[:,None,1])

            stop_features, stop_masks = scene_graph['stop_features'], scene_graph['stop_mask']
            stop_features[stop_masks] += torch.normal(0, self.gaussian_jitter, stop_features[stop_masks].shape, device=stop_features.device)
            stop_features = stop_features.clamp(bounding_box[:,None,0], bounding_box[:,None,1])

            # TODO: image augmentation

            scene_graph['vehicle_features'], scene_graph['vehicle_masks'] = vehicle_features, vehicle_masks
            scene_graph['dynamic_features'], scene_graph['dynamic_masks'] = dynamic_features, dynamic_masks
            scene_graph['stop_features'], scene_graph['stop_mask'] = stop_features, stop_masks

        return scene_graph
