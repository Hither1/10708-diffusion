from scipy.spatial.transform import Rotation
import cv2
import numpy as np


def agents_raster(state):
    size = 512
    radius = 50

    frame = np.zeros((size, size), dtype=np.uint8)

    idx = 0

    center = np.array([state.ego_state.position_x[idx], state.ego_state.position_y[idx], state.ego_state.heading[idx]])

    agent_states = self.scenario_data.agent_states[state.timestep[idx],:,:3]
    agent_lengths = self.scenario_data.agent_states[state.timestep[idx],:,5:7] / 2
    agent_boxes = jnp.stack([
        jnp.concatenate([agent_lengths[...,1:2], agent_lengths[...,0:1]], axis=-1),
        jnp.concatenate([-agent_lengths[...,1:2], agent_lengths[...,0:1]], axis=-1),
        jnp.concatenate([-agent_lengths[...,1:2], -agent_lengths[...,0:1]], axis=-1),
        jnp.concatenate([agent_lengths[...,1:2], -agent_lengths[...,0:1]], axis=-1),
    ], axis=-2)
    agent_boxes = np.concatenate([agent_boxes, np.ones(agent_boxes.shape[:-1])[...,None]], axis=-1)

    agent_transform = Rotation.from_euler('z', agent_states[:,2], degrees=False).as_matrix().astype(jnp.float32)
    agent_transform[:,:2,2] = agent_states[:,:2]
    agent_boxes = (agent_transform @ agent_boxes.transpose(0,2,1)).transpose(0,2,1)[...,:2]

    agent_boxes = transform_to_relative_coords(agent_boxes, center)

    map_states = np.concatenate([
        jnp.clip(
            transform_to_relative_coords(self.scenario_data.map_states[:,:2], center),
            -radius, radius
        ),
        self.scenario_data.map_states[:,2:]
    ], axis=-1)

    map_pixel_states = transform_relative_to_pixel(map_states[:,:2])
    import pdb; pdb.set_trace()
    for box in map_pixel_states:
        cv2.fillPoly(frame, box[None], color='purple', lineType=cv2.LINE_AA)
    
    cv2.imwrite('test.png', frame)

    import pdb; pdb.set_trace()
    raise
    
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10,10))
    plt.scatter(map_states[:,0], map_states[:,1], color='black')
    for agent_idx in range(agent_boxes.shape[0]):
        plt.scatter(agent_boxes[agent_idx,:,0], agent_boxes[agent_idx,:,1], color='green')

    plt.savefig('test.png')

    import pdb; pdb.set_trace()
    raise