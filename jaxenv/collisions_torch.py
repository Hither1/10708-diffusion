import torch


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
            other_normals.append(torch.stack([other_p2[..., 1] - other_p1[..., 1], other_p1[..., 0] - other_p2[..., 0]], dim=-1))

            ego_p1 = ego_bbox[i]
            ego_p2 = ego_bbox[j]
            ego_normals.append(torch.stack([ego_p2[..., 1] - ego_p1[..., 1], ego_p1[..., 0] - ego_p2[..., 0]], dim=-1))

    # other_normals B, 10, 2
    other_normals = torch.stack(other_normals, dim=1)
    # ego_normals 10, 2
    ego_normals = torch.stack(ego_normals, dim=1).permute(1,0)

    # a_other_projects (B, 10, 4)
    a_other_projects = torch.sum(other_normals.reshape((B, 10, 1, 2)) * other_bboxes.reshape((B, 1, 4, 2)), dim=-1)
    # b_other_projects (B, -1, 4)
    b_other_projects = torch.sum(other_normals.reshape((B, 10, 1, 2)) * ego_bbox.reshape((1, 1, 4, 2)), dim=-1)
    other_separates = (a_other_projects.max(dim=-1)[0] < b_other_projects.min(dim=-1)[0]) | (b_other_projects.max(dim=-1)[0] < b_other_projects.min(dim=-1)[0])

    # a_ego_projects (1, 10, 4)
    a_ego_projects = torch.sum(ego_bbox.reshape((1, 1, 4, 2)) * ego_normals.reshape((1, 10, 1, 2)), dim=-1)

    # b_ego_projects (B, 10, 4)
    b_ego_projects = torch.sum(other_bboxes.reshape((B, 1, 4, 2)) * ego_normals.reshape((1, 10, 1, 2)), dim=-1)

    ego_separates = (a_ego_projects.max(dim=-1)[0] < b_ego_projects.min(dim=-1)[0]) | (b_ego_projects.max(dim=-1)[0] < b_ego_projects.min(dim=-1)[0])

    separates = other_separates.any(dim=-1) | ego_separates.any(dim=-1)

    return ~separates


# def check_collisions(P2, extent2, extent1):
    """
    P1, P2: (..., 1x3), (..., Nx3) object poses
    Returns (..., N) collision mask (True if collision)
    """

def check_collisions(poses, extents):
    bboxes = torch.stack([
        torch.stack([-extents[..., 0], -extents[..., 1]], dim=-1),
        torch.stack([extents[..., 0], -extents[..., 1]], dim=-1),
        torch.stack([extents[..., 0], extents[..., 1]], dim=-1),
        torch.stack([-extents[..., 0], extents[..., 1]], dim=-1),
    ], dim=-2)

    theta = poses[..., 2]
    # TODO figure out what is the right rotation matrix
    rot_matrix = torch.stack([
        torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1),
        torch.stack([-torch.sin(theta), torch.cos(theta)], dim=-1)
    ], dim=-2)
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
    poses = torch.as_tensor([
        [0.0, 0.0, 0.0],
        [-1.0, -1.0, .75],
        [1.0, 1.0, 0.0]
    ])
    extents = torch.as_tensor([
        [0.1, 0.1],
        [10.0, 0.1],
        [10.0, 0.1]
    ])
    print(check_collisions(poses, extents))