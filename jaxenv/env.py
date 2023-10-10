import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces
from typing import Tuple, Optional, NamedTuple
import chex
from flax import struct

import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from PIL import Image

# from bicycle import EgoState, propagate_state
from bicycle_v2 import EgoState, compute_dynamics
from collisions import check_collisions


@struct.dataclass
class EnvParams:
    max_steps: int = 40

    # Bicycle params
    dt: float = 0.1
    wheel_base: float = 3.089
    min_vel: float = 0.0
    max_vel: float = 5.0
    min_yaw_rate: float = -jnp.pi / 4
    max_yaw_rate: float = jnp.pi / 4
    accel_coeff: float = 1.0
    steer_coeff: float = 0.2

    # Ego parameters
    width: float = 2.3
    length: float = 5.2


# From the log
@struct.dataclass
class ScenarioData:
    initial_ego_state: EgoState
    ego_trajectory: chex.Array  # (3,)
    agent_states: chex.Array    # (A x 8) [x,y,heading,xvel,yvel,xbbox,ybbox]
    map_states: chex.Array      # (N x 8)     [x,y,(on_route,2),(light_status,4)]


@struct.dataclass
class EnvState:
    ego_state: EgoState
    timestep: int


class Observation:
    @classmethod
    def pack_tensor(cls, ego_features, agent_features, map_features, timestep):
        return jnp.concatenate([
            ego_features.flatten(),
            agent_features.flatten(),
            map_features.flatten(),
            jnp.array([timestep], 'float32')
        ])

    # @classmethod
    # def unpack_tensor(cls, tensor):
    #     leading_shape = tensor.shape[:-1]
    #     return (
    #         tensor[...,cls.cidx0:cls.cidx1].reshape(*leading_shape,1,cls.ego_feature_dim),
    #         tensor[...,cls.cidx1:cls.cidx2].reshape(*leading_shape,cls.num_agents,cls.agent_feature_dim),
    #         tensor[...,cls.cidx2:cls.cidx3].reshape(*leading_shape,cls.num_map_points,cls.map_feature_dim),
    #         tensor[...,cls.cidx3:cls.cidx4].reshape(*leading_shape,1,1)
    #     )

    @classmethod
    @property
    def ego_feature_dim(cls):
        return 5

    @classmethod
    @property
    def num_agents(cls):
        return 5

    @classmethod
    @property
    def agent_feature_dim(cls):
        return 6+1

    @classmethod
    @property
    def num_map_points(cls):
        return 200

    @classmethod
    @property
    def map_feature_dim(cls):
        return 8+1

    @classmethod
    @property
    def cidx0(cls):
        return 0

    @classmethod
    @property
    def cidx1(cls):
        return cls.cidx0 + cls.ego_feature_dim

    @classmethod
    @property
    def cidx2(cls):
        return cls.cidx1 + cls.num_agents * cls.agent_feature_dim

    @classmethod
    @property
    def cidx3(cls):
        return cls.cidx2 + cls.num_map_points * cls.map_feature_dim

    @classmethod
    @property
    def cidx4(cls):
        return cls.cidx3 + 1


# class EnvObs:
#     ego_feature_dim: int = 5
#     num_agents: int = 10
#     agent_feature_dim: int = 6+1
#     num_map_points: int = 200
#     map_feature_dim: int = 8+1

#     @classmethod
#     def pack_observation_tensor(cls, ego_features, agent_features, map_features, timestep):
#         return jnp.concatenate([
#             ego_features.flatten(),
#             agent_features.flatten(),
#             map_features.flatten(),
#             jnp.array([timestep], 'float32')
#         ])

#     @classmethod
#     def unpack_observation_tensor(cls, tensor):
#         return (
#             tensor[:cls.ego_feature_dim].reshape(1,cls.ego_feature_dim),
#             tensor[cls.ego_feature_dim:cls.ego_feature_dim+(cls.num_agents*cls.agent_feature_dim)].reshape(cls.num_agents,cls.agent_feature_dim),
#             tensor[cls.ego_feature_dim+(cls.num_agents*cls.agent_feature_dim):cls.ego_feature_dim+(cls.num_agents*cls.agent_feature_dim)+(cls.num_map_points*cls.map_feature_dim)].reshape(cls.num_map_points,cls.map_feature_dim),
#             tensor[-1]
#         )


def transform_to_relative_coords(global_coords, ego_coords):
    """
    Transforms global coordinates to relative coordianates in ego frame
    global_coords: (*, 2 or 3)
    ego_coords: (3,)
    """
    global_coords_shape = global_coords.shape
    global_coords = global_coords.reshape(-1,global_coords_shape[-1])
    includes_heading = global_coords_shape[-1] == 3

    global_coords_xy = global_coords[:,:2]
    global_coords_heading = global_coords[:,2:3]
    global_coords_xy = global_coords_xy - ego_coords[:2]

    # Rotate to ego heading
    heading = ego_coords[2]
    rot_matrix = jnp.array([[jnp.cos(heading), -jnp.sin(heading)], [jnp.sin(heading), jnp.cos(heading)]])
    global_coords_xy = global_coords_xy @ rot_matrix
    if includes_heading:
        global_coords_heading = global_coords_heading - heading

    global_coords = jnp.concatenate([global_coords_xy, global_coords_heading, global_coords[:,3:]], axis=1)
    global_coords = global_coords.reshape(global_coords_shape)
    return global_coords


def transform_relative_to_pixel(coords, radius=50.0, size=200, pixel_size=0.5):
    coords_shape = coords.shape
    coords = coords.reshape(-1, coords_shape[-1])

    index_points = (radius + coords) / pixel_size

    new_coords = index_points.reshape(coords_shape)
    return new_coords


class NuPlan(environment.Environment):
    def __init__(self, scenario_data):
        super().__init__()

        self.scenario_data = scenario_data

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Performs step transitions in the environment."""
        timestep = state.timestep + 1
        
        ego_state = state.ego_state
        for _ in range(5):
            ego_state = compute_dynamics(ego_state, action, params)
        
        next_state = EnvState(
            ego_state=ego_state,
            timestep=timestep
        )
        reward = self.calculate_reward(next_state, params)
        done = self.is_terminal(next_state, params)

        next_obs = self.get_obs(next_state)
        return next_obs, next_state, reward, done, {}

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""

        state = EnvState(
            ego_state=self.scenario_data.initial_ego_state,
            timestep=0
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> chex.Array:

        """Applies observation function to state."""
        initial_ego_state = self.scenario_data.initial_ego_state
        origin = jnp.array([
            initial_ego_state.x,
            initial_ego_state.y,
            initial_ego_state.yaw
        ])

        # Ego feature
        pose = transform_to_relative_coords(jnp.array([[
            state.ego_state.x,
            state.ego_state.y,
            state.ego_state.yaw,
        ]]), origin)[0]
        pose = jnp.concatenate([
            pose[:2] / 50,
            pose[2:]
        ])
        speed = jnp.sqrt(state.ego_state.vel_x**2 + state.ego_state.vel_y**2)
        ego_features = jnp.concatenate([
            pose,
            jnp.array([
                speed,
                state.ego_state.yaw_rate
            ])
        ])

        # Agent features
        agent_states = self.scenario_data.agent_states[state.timestep]
        agent_poses = transform_to_relative_coords(agent_states[:,:3], origin)
        agent_poses = jnp.concatenate([
            agent_poses[:,:2] / 50,
            agent_poses[:,2:]
        ], axis=1)
        agent_speeds = jnp.sqrt(agent_states[:,3:4]**2 + agent_states[:,4:5]**2)
        agent_features = jnp.concatenate([
            agent_poses,
            agent_speeds,
            agent_states[:,5:7]
        ], axis=1)

        # Map features
        map_states = self.scenario_data.map_states
        map_poses = transform_to_relative_coords(map_states[:,:2], origin)
        map_features = jnp.concatenate([map_poses, map_states[:,4:10]], axis=1)

        # Padding / discarding / masks
        agent_dists = jnp.linalg.norm(agent_poses[...,:2], axis=-1)
        sorted_agent_idxs = jnp.argsort(agent_dists)
        agent_features = agent_features[sorted_agent_idxs]
        agent_features = agent_features[:Observation.num_agents]
        agent_masks = jnp.concatenate([
            jnp.ones(agent_features.shape[0], dtype=bool),
            jnp.zeros(Observation.num_agents-agent_features.shape[0], dtype=bool)
        ])[:,None]
        agent_features = jnp.concatenate([
            agent_features,
            jnp.zeros((Observation.num_agents-agent_features.shape[0],agent_features.shape[1]))
        ], axis=0)
        agent_features = jnp.concatenate([agent_features, agent_masks], axis=-1)

        map_dists = jnp.linalg.norm(map_poses[...,:2], axis=-1)
        sorted_map_idxs = jnp.argsort(map_dists)
        map_features = map_features[sorted_map_idxs]
        map_features = map_features[:Observation.num_map_points]
        map_masks = jnp.concatenate([
            jnp.ones(map_features.shape[0], dtype=bool), 
            jnp.zeros(Observation.num_map_points-map_features.shape[0], dtype=bool)
        ])[:,None]
        map_features = jnp.concatenate([
            map_features,
            jnp.zeros((Observation.num_map_points-map_features.shape[0],map_features.shape[1]))
        ], axis=0)
        map_features = jnp.concatenate([map_features, map_masks], axis=-1)

        obs = Observation.pack_tensor(
            ego_features,
            agent_features,
            map_features,
            state.timestep
        )
        return obs

    def calculate_reward(self, state: EnvState, params: EnvParams) -> float:
        # Imitation reward
        current = jnp.array([state.ego_state.x, state.ego_state.y])
        actual = self.scenario_data.ego_trajectory[state.timestep][:2]
        dist = jnp.linalg.norm(current - actual, ord=1)

        # imitation_reward = -dist
        imitation_reward = 0

        # Collision reward
        agent_states = self.scenario_data.agent_states[state.timestep]
        agent_poses = jnp.concatenate([
            jnp.array([[state.ego_state.x, state.ego_state.y, state.ego_state.yaw]]),
            agent_states[...,:3]
        ], axis=0)
        agent_lengths = jnp.concatenate([
            jnp.array([[params.length, params.width]]) / 2,
            agent_states[...,5:7] / 2
        ], axis=0)

        is_collisions = check_collisions(agent_poses, agent_lengths)
        
        collision_reward = is_collisions.any() * -100

        reward = imitation_reward + collision_reward
        return reward

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        return state.timestep >= params.max_steps

    @property
    def name(self) -> str:
        """Environment name."""
        return "NuPlan"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 2

    def action_space(
        self, params: Optional[EnvParams] = None
    ) -> spaces.Box:
        """Action space of the environment."""
        return spaces.Box(low=-1, high=1, shape=(2,))

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(low=-jnp.inf, high=jnp.inf, shape=(6,))

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        raise NotImplementedError

    def render(self, state, env_params, radius=50, pixel_size=0.1):
        size = int(2 * radius / pixel_size)

        # Features to visualize (convert to numpy to use cv2)
        ego_state = np.array([state.ego_state.x, state.ego_state.y, state.ego_state.yaw])
        # ego_state = np.asarray(self.scenario_data.ego_trajectory[state.timestep])
        agent_states = np.asarray(self.scenario_data.agent_states[state.timestep])
        map_states = np.asarray(self.scenario_data.map_states[:,:4]).reshape(-1,2,2)

        # Render components
        map_frame = self.render_map(ego_state, map_states, size, radius, pixel_size)
        agent_frame = self.render_agents(ego_state, agent_states, size, radius, pixel_size)
        ego_frame = self.render_ego(ego_state, size, radius, pixel_size, env_params)

        # Compose visualization
        frame = np.full((size, size, 3), (0,0,0), dtype=np.uint8)
        frame[map_frame.nonzero()] = (255,255,255)
        frame[agent_frame.nonzero()] = (255,0,55)
        frame[ego_frame.nonzero()] = (55,0,255)

        return frame

    def render_map(self, ego_state, map_states, size, radius, pixel_size):
        frame = np.zeros((size, size), dtype=np.uint8)

        # Transform map lines -> relative ego frame -> pixel coordinates
        map_states = np.stack([
            transform_to_relative_coords(map_states[:,0], ego_state),
            transform_to_relative_coords(map_states[:,1], ego_state)
        ], axis=1)
        # map_states = np.clip(map_states, -radius, radius)
        map_states = transform_relative_to_pixel(map_states, radius, size, pixel_size)
        map_states = map_states.astype(int)

        # Draw lines
        cv2.polylines(
            frame,
            map_states,
            isClosed=False,
            color=1,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        return frame

    def render_agents(self, ego_state, agent_states, size, radius, pixel_size):
        frame = np.zeros((size, size), dtype=np.uint8)

        agent_pose = agent_states[...,:3]
        agent_lengths = agent_states[...,5:7] / 2
        agent_boxes = np.stack([
            np.concatenate([agent_lengths[...,1:2], agent_lengths[...,0:1]], axis=-1),
            np.concatenate([-agent_lengths[...,1:2], agent_lengths[...,0:1]], axis=-1),
            np.concatenate([-agent_lengths[...,1:2], -agent_lengths[...,0:1]], axis=-1),
            np.concatenate([agent_lengths[...,1:2], -agent_lengths[...,0:1]], axis=-1),
        ], axis=-2)
        agent_boxes = np.concatenate([agent_boxes, np.ones(agent_boxes.shape[:-1])[...,None]], axis=-1)

        # Transform boxes -> agent frame -> relative ego frame -> pixel coordinates
        agent_transform = Rotation.from_euler('z', agent_states[:,2], degrees=False).as_matrix().astype(np.float32)
        agent_transform[:,:2,2] = agent_states[:,:2]
        agent_boxes = (agent_transform @ agent_boxes.transpose(0,2,1)).transpose(0,2,1)[...,:2]
        agent_boxes = transform_to_relative_coords(agent_boxes, ego_state)
        agent_boxes = np.clip(agent_boxes, -radius, radius)
        agent_boxes = transform_relative_to_pixel(agent_boxes, radius, size, pixel_size)
        agent_boxes = np.asarray(agent_boxes).astype(int)

        # Draw boxes
        for box in agent_boxes:
            cv2.fillPoly(frame, box[None], color=1, lineType=cv2.LINE_AA)

        return frame

    def render_ego(self, ego_state, size, radius, pixel_size, env_params):
        frame = np.zeros((size, size), dtype=np.uint8)

        length = env_params.length
        width = env_params.width

        agent_box = np.array([
            [length, width],
            [-length, width],
            [-length, -width],
            [length, -width]
        ]) / 2
        agent_box = transform_relative_to_pixel(agent_box, radius, size, pixel_size)
        agent_box = agent_box.astype(int)

        # Draw box
        cv2.fillPoly(frame, agent_box[None], color=1, lineType=cv2.LINE_AA)

        return frame


def make_env(scenario_path='/scenario_data.npz'):
    # Initializing environment
    scenario_data = jnp.load('scenario_data.npz', allow_pickle=True)
    scenario_data = ScenarioData(
        initial_ego_state=EgoState(**scenario_data['initial_ego_state'][()]),
        ego_trajectory=jnp.asarray(scenario_data['ego_trajectory']),
        agent_states=jnp.asarray(scenario_data['agent']),
        map_states=jnp.asarray(scenario_data['map'])
    )

    env = NuPlan(scenario_data)
    env_params = env.default_params

    return env, env_params


def main():
    # Initializing environment
    env, env_params = make_env()

    def rollout(rng):
        rng, key_reset, key_act, key_step = jax.random.split(rng, 4)

        obs, state = env.reset(key_reset)
        done = False

        rewards = []
        obses = [obs]
        dones = []
        infos = []
        states = [state]

        for _ in range(20):
            action = env.action_space(env_params).sample(key_act)
            obs, state, reward, done, info = env.step(key_step, state, action, env_params)

            rewards.append(reward)
            obses.append(obs)
            dones.append(done)
            infos.append(info)
            states.append(state)

        return obses, rewards, dones, states

    rng = jax.random.PRNGKey(5)
    num_rollouts = 10
    rngs = jax.random.split(rng, num_rollouts)

    # rollout_jit = jax.jit(jax.vmap(rollout))
    # outs = jax.block_until_ready(rollout_jit(rngs))
    # outs = jax.block_until_ready(jax.vmap(rollout)(rngs))

    for i, rng in enumerate(rngs):
        out = jax.block_until_ready(rollout(rng))

        # flattened_obs = out[0][0]
        # env_obs = Observation.unpack_tensor(flattened_obs)
        # # env_obs = EnvObs.from_tensor(flattened_obs)

        # import pdb; pdb.set_trace()

        # rewards = jnp.array(out[1]).sum()
        # print(rewards)

        frames = []
        for state in out[3]:
            frame = env.render(state, env_params)
            frames.append(frame)
        frames = [Image.fromarray(frame) for frame in frames]
        frames[0].save(f'videos/test{i}.gif', save_all=True, append_images=frames[1:], duration=100, loop=0)

    print('Done!')


if __name__ == '__main__':
    main()
