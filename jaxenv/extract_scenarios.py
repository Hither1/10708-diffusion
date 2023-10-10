import os
from pathlib import Path
import tempfile

import numpy as np
import torch
import hydra

from nuplan.planning.script.run_simulation import get_scenarios
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import (
    build_ego_features_from_tensor,
    compute_yaw_rate_from_state_tensors,
    convert_absolute_quantities_to_relative,
    filter_agents_tensor,
    pack_agents_tensor,
    pad_agent_states,
    sampled_past_ego_states_to_tensor,
    sampled_past_timestamps_to_tensor,
    sampled_tracked_objects_to_tensor_list,
)
from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import (
    LaneOnRouteStatusData,
    LaneSegmentConnections,
    LaneSegmentCoords,
    LaneSegmentGroupings,
    LaneSegmentTrafficLightData,
    get_neighbor_vector_map,
    get_on_route_status,
    get_traffic_light_encoding,
)
from nuplan.common.actor_state.state_representation import Point2D, StateSE2


def load_scenarios():
    # Location of path with all simulation configs
    CONFIG_PATH = '../nuplan/planning/script/config/simulation'
    CONFIG_NAME = 'default_simulation'

    # Select the planner and simulation challenge
    PLANNER = 'simple_planner'  # [simple_planner, ml_planner]
    CHALLENGE = 'open_loop_boxes'  # [open_loop_boxes, closed_loop_nonreactive_agents, closed_loop_reactive_agents]
    DATASET_PARAMS = [
        'scenario_builder=nuplan_mini',  # use nuplan mini database
        'scenario_filter=all_scenarios',  # initially select all scenarios in the database
        'scenario_filter.scenario_types=[near_multiple_vehicles, on_pickup_dropoff, starting_unprotected_cross_turn, high_magnitude_jerk]',  # select scenario types
        'scenario_filter.num_scenarios_per_type=10',  # use 10 scenarios per scenario type
    ]

    # Name of the experiment
    EXPERIMENT = 'simulation_simple_experiment'

    # Initialize configuration management system
    hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
    hydra.initialize(config_path=CONFIG_PATH)

    # Compose the configuration
    cfg = hydra.compose(config_name=CONFIG_NAME, overrides=[
        f'experiment_name={EXPERIMENT}',
        f'planner={PLANNER}',
        f'+simulation={CHALLENGE}',
        *DATASET_PARAMS,
        # 'worker=sequential'
    ])

    scenarios = get_scenarios(cfg)
    return scenarios


def get_map_states(scenario, ego_state, radius):
    ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
    (
        lane_seg_coords,
        lane_seg_conns,
        lane_seg_groupings,
        lane_seg_lane_ids,
        lane_seg_roadblock_ids,
    ) = get_neighbor_vector_map(scenario.map_api, ego_coords, radius)

    # compute route following status
    on_route_status = get_on_route_status(scenario.get_route_roadblock_ids(), lane_seg_roadblock_ids)

    # get traffic light status
    traffic_light_data = scenario.get_traffic_light_status_at_iteration(0)
    traffic_light_data = get_traffic_light_encoding(lane_seg_lane_ids, traffic_light_data)

    lane_segment_coords: torch.tensor = torch.tensor(lane_seg_coords.to_vector(), dtype=torch.float64)
    # lane_segment_conns: torch.tensor = torch.tensor(lane_seg_conns.to_vector(), dtype=torch.int64)
    on_route_status: torch.tensor = torch.tensor(on_route_status.to_vector(), dtype=torch.float32)
    traffic_light_array: torch.tensor = torch.tensor(traffic_light_data.to_vector(), dtype=torch.float32)
    # lane_segment_groupings: List[torch.tensor] = []
    # for lane_grouping in lane_seg_groupings.to_vector():
    #     lane_segment_groupings.append(torch.tensor(lane_grouping, dtype=torch.int64))
    map_states = torch.cat([
        lane_segment_coords.reshape(lane_segment_coords.shape[0],2*2), 
        on_route_status, 
        traffic_light_array
    ], dim=1)

    return map_states


def get_scenario_features(scenario, iteration=0, num_samples=40, time_horizon=20, radius=50):
    """
    Extract relevant scenario features and return as arrays
    Features consist of:
        1) Objects with history: cars
        2) Map objects: road points
    TODO: expand feature set
    """
    # Ego features
    ego_track = scenario.get_ego_future_trajectory(
        iteration=iteration, num_samples=num_samples, time_horizon=time_horizon
    )
    ego_tracks = list(ego_track)
    ego_trajectory = [track.rear_axle.serialize() for track in ego_tracks]
    ego_trajectory = np.array(ego_trajectory)

    ego_state = ego_tracks[0]
    initial_ego_state = {
        'x': ego_state.rear_axle.x,
        'y': ego_state.rear_axle.y,
        'yaw': ego_state.rear_axle.heading,
        'vel_x': ego_state.dynamic_car_state.rear_axle_velocity_2d.x,
        'vel_y': ego_state.dynamic_car_state.rear_axle_velocity_2d.y,
        'yaw_rate': ego_state.dynamic_car_state.angular_velocity
    }

    # Agent features
    tracks = scenario.get_future_tracked_objects(iteration=iteration, num_samples=num_samples, time_horizon=time_horizon)
    tracks = [track.tracked_objects for track in tracks]
    tensor_list = sampled_tracked_objects_to_tensor_list(tracks)
    tensor_list = filter_agents_tensor(tensor_list, reverse=True)
    padded_agent_states = pad_agent_states(tensor_list, reverse=True)
    agent_states = torch.stack(padded_agent_states)

    # Reorder agent features
    agent_states = agent_states[:,:,1:] # ignore track_id
    agent_states = agent_states[:,:, [5,6,2,0,1,3,4]] # change ordering

    # Map features
    map_states = [get_map_states(scenario, ego_tracks[i], radius) for i in range(0,len(ego_tracks),5)]
    map_states = torch.cat(map_states, dim=0).unique(dim=0)

    # Center everything on ego (otherwise we lose precision later)
    origin = torch.as_tensor([initial_ego_state['x'], initial_ego_state['y']])
    agent_states[:,:,:2] -= origin
    map_states[:,:2] -= origin
    map_states[:,2:4] -= origin
    ego_trajectory[:,:2] -= origin.numpy()
    initial_ego_state['x'] = 0.0
    initial_ego_state['y'] = 0.0
    
    assert agent_states.shape[0] == num_samples, 'Invalid number of agent states fetched'

    return {
        'ego_trajectory': ego_trajectory,
        'agent': agent_states.numpy(),
        'map': map_states.numpy(),
        'initial_ego_state': initial_ego_state
    }

scenarios = load_scenarios()
scenario_idx = 14
features = get_scenario_features(scenarios[scenario_idx])
# print(features['ego'].shape, features['agent'].shape, features['map'].shape)
np.savez('/zfsauton2/home/brianyan/nuplan-devkit/jaxenv/scenario_data', **features)

# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(20,20))
# plt.scatter(features['map'][:,0], features['map'][:,1], c='black')
# plt.scatter(features['agent'][0,:,0], features['agent'][0,:,1], c='green')
# plt.scatter(features['ego'][:,0], features['ego'][:,1], c='blue')
# plt.savefig('/zfsauton2/home/brianyan/nuplan-devkit/test.png')

# print('Done')