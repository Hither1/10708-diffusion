import jax
import jax.numpy as jnp
# import numpy as jnp


def do_bboxes_intersect(other_bboxes, ego_bbox):
    # bboxes B, 4, 2
    # ego_bbox 4, 2
    B = other_bboxes.shape[0]

    other_normals = []
    ego_normals = []
    for i in range(4):
        for j in range(i, 4):
            other_p1 = other_bboxes[:, i]
            other_p2 = other_bboxes[:, j]
            other_normals.append(jnp.stack([other_p2[..., 1] - other_p1[..., 1], other_p1[..., 0] - other_p2[..., 0]], axis=-1))

            ego_p1 = ego_bbox[i]
            ego_p2 = ego_bbox[j]
            ego_normals.append(jnp.stack([ego_p2[..., 1] - ego_p1[..., 1], ego_p1[..., 0] - ego_p2[..., 0]], axis=-1))

    # other_normals B, 10, 2
    other_normals = jnp.stack(other_normals, axis=1)
    # ego_normals 10, 2
    ego_normals = jnp.stack(ego_normals, axis=0)

    # a_other_projects (B, 10, 4)
    a_other_projects = jnp.sum(other_normals.reshape((B, 10, 1, 2)) * other_bboxes.reshape((B, 1, 4, 2)), axis=-1)
    # b_other_projects (B, -1, 4)
    b_other_projects = jnp.sum(other_normals.reshape((B, 10, 1, 2)) * ego_bbox.reshape((1, 1, 4, 2)), axis=-1)
    other_separates = (a_other_projects.max(axis=-1) < b_other_projects.min(axis=-1)) | (b_other_projects.max(axis=-1) < b_other_projects.min(axis=-1))

    # a_ego_projects (1, 10, 4)
    a_ego_projects = jnp.sum(ego_bbox.reshape((1, 1, 4, 2)) * ego_normals.reshape((1, 10, 1, 2)), axis=-1)

    # b_ego_projects (B, 10, 4)
    b_ego_projects = jnp.sum(other_bboxes.reshape((B, 1, 4, 2)) * ego_normals.reshape((1, 10, 1, 2)), axis=-1)

    ego_separates = (a_ego_projects.max(axis=-1) < b_ego_projects.min(axis=-1)) | (b_ego_projects.max(axis=-1) < b_ego_projects.min(axis=-1))

    separates = other_separates.any(axis=-1) | ego_separates.any(axis=-1)

    return ~separates


# def check_collisions(P2, extent2, extent1):
    """
    P1, P2: (..., 1x3), (..., Nx3) object poses
    Returns (..., N) collision mask (True if collision)
    """

def check_collisions(poses, extents):
    bboxes = jnp.stack([
        jnp.stack([-extents[..., 0], -extents[..., 1]], axis=-1),
        jnp.stack([extents[..., 0], -extents[..., 1]], axis=-1),
        jnp.stack([extents[..., 0], extents[..., 1]], axis=-1),
        jnp.stack([-extents[..., 0], extents[..., 1]], axis=-1),
    ], axis=-2)

    theta = poses[..., 2]
    # TODO figure out what is the right rotation matrix
    rot_matrix = jnp.stack([
        jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1),
        jnp.stack([-jnp.sin(theta), jnp.cos(theta)], axis=-1)
    ], axis=-2)
    bboxes = (bboxes @ rot_matrix)
    bboxes += poses[..., None, :2]

    ego_bbox = bboxes[0]
    agent_bboxes = bboxes[1:]

    # bboxes_shape = agent_bboxes.shape[:3]
    is_collision = do_bboxes_intersect(agent_bboxes.reshape((-1, 4, 2)), ego_bbox)
    # is_collision = shaped_is_collision.reshape(bboxes_shape)

    # return is_collision
    return is_collision


if __name__ == '__main__':
    poses = jnp.array([
        [0.0, 0.0, 0.0],
        [-1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0]
    ])
    extents = jnp.array([
        [0.1, 0.1],
        [10.0, 0.1],
        [10.0, 0.1]
    ])
    print(jax.block_until_ready(check_collisions(poses, extents)))
