#  import io
import math
#  import carla
import numpy as np
from collections import deque
#  import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Laplace, Normal
from torch.optim.lr_scheduler import LambdaLR
import shapely


def wrap_angle(ang, min_ang=-np.pi, max_ang=np.pi):
    diff = max_ang - min_ang
    return min_ang + torch.remainder(ang, diff)


# TODO fold into time encoding
class TimeEncoding(nn.Module):
    '''
    Standard positional encoding.
    '''
    def __init__(self, d_model, dropout=0.1, max_len=20):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, start_t=0):
        '''
        :param x: must be (B, H, d)
        :return:
        '''
        x = x + self.pe[:, start_t:start_t+x.shape[1], :]
        return self.dropout(x)


def time_encoding(x):
    """
    Computes time encoding for x using positional_encoding_1d
    Assumes x.shape == (B, T, A, *)
    """
    B, T, A, _ = x.shape
    times = torch.linspace(-1,1,T,device=x.device)
    encoding = positional_encoding_1d(times, i=64, d_model=128)
    encoding = encoding[None,:,None].repeat(B,1,A,1)
    return encoding


def positional_encoding_1d(x, i=32, d_model=64):
    encoding_sin = torch.sin(x[..., None] / 10000**(torch.arange(0,2*i,2,device=x.device)/d_model))
    encoding_cos = torch.cos(x[..., None] / 10000**(torch.arange(0,2*i,2,device=x.device)/d_model))
    encoding = torch.cat([encoding_sin, encoding_cos], dim=-1)
    return encoding


def gaussian_encoding_1d(x, d_model=128):
    sig = 0.1
    #  dic = torch.arange(-d_model/2, d_model/2, dtype=torch.float, device=x.device) / d_model
    dic = torch.linspace(-1., 1., d_model, device=x.device)
    rbfemb = (-0.5 * (x[..., None] - dic) **2 / (sig ** 2)).exp()
    rbfemb = rbfemb/ (rbfemb.norm(dim=1).max())
    return rbfemb


def positional_encoding_2d(pos, d_model=128):
    """
    Computes positional encoding following original Transformer paper
    """
    pos_shape = pos.shape
    assert pos_shape[-1] == 2, 'Invalid positional_encoding_2d input shape (expected 2)'

    pos = pos.reshape(-1,2)
    x_encoding = positional_encoding_1d(pos[:,0], i=d_model/4, d_model=d_model/2)
    y_encoding = positional_encoding_1d(pos[:,1], i=d_model/4, d_model=d_model/2)
    encoding = torch.cat([x_encoding, y_encoding], dim=1)
    return encoding.reshape(pos_shape[:-1] + (d_model,))


def weight_init(module):
    '''
    Copy-paste from: https://github.com/roggirg/AutoBots/blob/master/models/context_encoders.py
    '''
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.xavier_normal_(module.weight.data, gain=np.sqrt(2))
        nn.init.constant_(module.bias.data, 0)
    return module


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=256, n_layers=1):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_size)]
        for _ in range(n_layers-1):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, output_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        input_shape = x.shape
        x = x.reshape(-1,x.shape[-1])
        x = self.layers(x)
        x = x.reshape(input_shape[:-1] + (x.shape[-1],))
        return x


class ImageEncoder(nn.Module):
    """
    This is basically copied from drqv2: https://github.com/facebookresearch/drqv2/blob/main/drqv2.py
    """
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(weight_init)

    def forward(self, obs):
        obs = obs - 0.5
        h = self.convnet(obs)
        return h


class StaticEncoder(nn.Module):
    def __init__(self, embedding_size, image_shape=(3,84,84)):
        super().__init__()

        self.conv = ImageEncoder(image_shape)
        self.trunk = nn.Sequential(
            nn.Linear(32, embedding_size),
            nn.LayerNorm(embedding_size),
            nn.Tanh()
        )

        # the grid centers aren't the limits, so the actual extent is a little smaller than 2*radius x 2*radius...
        row = torch.linspace(-7/8, 7/8, 8)
        positions = torch.meshgrid(row, row)
        positions = torch.stack(positions, dim=-1).reshape(64,2)
        self.register_buffer('position_encoding', positional_encoding_2d(positions, d_model=embedding_size))

        self.apply(weight_init)

    def forward(self, x):
        x = self.conv(x)
        x = F.adaptive_avg_pool2d(x,(8,8))
        x = x.reshape(x.shape[0], x.shape[1], 64).permute(0,2,1)
        x = self.trunk(x)

        x = x + self.position_encoding[None].repeat(x.shape[0],1,1).detach()
        return x


def soft_update_params(net, target_net, tau=.05):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


"""
Copy-pasted from https://github.com/roggirg/AutoBots/blob/master/utils/train_helpers.py
"""

def get_laplace_dist(pred):
    d = pred.shape[-1] // 2
    return Laplace(pred[..., :d], pred[..., d:2*d])


def get_normal_dist(pred):
    d = pred.shape[-1] // 2
    return Normal(pred[..., :d], pred[..., d:2*d])


def nll_pytorch_dist(pred, data, dist='laplace'):
    if dist == 'laplace':
        biv_lapl_dist = get_laplace_dist(pred)
    elif dist == 'normal':
        biv_lapl_dist = get_normal_dist(pred)
    else:
        raise NotImplementedError

    loss = -biv_lapl_dist.log_prob(data).sum(dim=-1)
    # print('0', loss[..., 0].sum())
    # print('1', loss[..., 1].sum())
    # print('2', loss[..., 2].sum())
    # print('3', loss[..., 3].sum())
    return -biv_lapl_dist.log_prob(data).sum(dim=-1)


#  def fig_to_numpy(fig, dpi=60):
    #  buf = io.BytesIO()
    #  fig.savefig(buf, format="png", dpi=dpi)
    #  buf.seek(0)
    #  img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    #  buf.close()
    #  img = cv2.imdecode(img_arr, 1)
    #  return img

def fig_to_numpy(fig, dpi=60):
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


"""
Copy-pasted from HuggingFace: https://github.com/huggingface/transformers/blob/v4.19.4/src/transformers/optimization.py#L75
"""
def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def dist_to_line(point, point1_on_line, point2_on_line):
    a_vec = point2_on_line - point1_on_line
    b_vec = point - point1_on_line
    return abs(np.cross(a_vec, b_vec) / np.linalg.norm(a_vec))


class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._window = deque([0 for _ in range(n)], maxlen=n)
        self._max = 0.0
        self._min = 0.0

    def step(self, error):
        self._window.append(error)
        self._max = max(self._max, abs(error))
        self._min = -abs(self._max)

        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = self._window[-1] - self._window[-2]
        else:
            integral = 0.0
            derivative = 0.0

        return self._K_P * error + self._K_I * integral + self._K_D * derivative


#  def check_collisions(P2, extent2, extent1):
    #  """
    #  P1, P2: (..., 1x3), (..., Nx3) object poses
    #  Returns (..., N) collision mask (True if collision)
    #  """
    #  bboxes = torch.stack([
        #  torch.stack([-extent2[..., 0], -extent2[..., 1]], dim=-1),
        #  torch.stack([extent2[..., 0], -extent2[..., 1]], dim=-1),
        #  torch.stack([extent2[..., 0], extent2[..., 1]], dim=-1),
        #  torch.stack([-extent2[..., 0], extent2[..., 1]], dim=-1),
    #  ], dim=-2)

    #  theta = P2[..., 2]
    #  # TODO figure out what is the right rotation matrix
    #  rot_matrix = torch.stack([
        #  torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1),
        #  torch.stack([-torch.sin(theta), torch.cos(theta)], dim=-1)
    #  ], dim=-2)
    #  bboxes = (bboxes @ rot_matrix)
    #  bboxes += P2[..., None, :2]

    #  above_x = bboxes[..., 0] >= -extent1[..., None, None, 0]
    #  above_y = bboxes[..., 1] >= -extent1[..., None, None, 1]
    #  below_x = bboxes[..., 0] <= extent1[..., None, None, 0]
    #  below_y = bboxes[..., 1] <= extent1[..., None, None, 1]
    #  is_x_collision = above_x.any(dim=-1) & below_x.any(dim=-1)
    #  is_y_collision = above_y.any(dim=-1) & below_y.any(dim=-1)
    #  is_collision = is_x_collision & is_y_collision

    #  return is_collision


def check_center_collisions(P2, extent2):
    point = -P2[..., None, :2]
    theta = P2[..., 2]
    # TODO figure out what is the right rotation matrix
    #  rot_matrix = torch.stack([
        #  torch.stack([ torch.cos(theta), torch.sin(theta)], dim=-1),
        #  torch.stack([-torch.sin(theta), torch.cos(theta)], dim=-1)
    #  ], dim=-2)
    rot_matrix = torch.stack([
        torch.stack([torch.cos(theta), -torch.sin(theta)], dim=-1),
        torch.stack([torch.sin(theta),  torch.cos(theta)], dim=-1)
    ], dim=-2)
    point = torch.squeeze(point @ rot_matrix, dim=-2)

    above_x = point[..., 0] >= -extent2[..., 0]
    above_y = point[..., 1] >= -extent2[..., 1]
    below_x = point[..., 0] <= extent2[..., 0]
    below_y = point[..., 1] <= extent2[..., 1]
    #  is_x_collision = above_x.any(dim=-1) & below_x.any(dim=-1)
    #  is_y_collision = above_y.any(dim=-1) & below_y.any(dim=-1)
    #  is_collision = is_x_collision & is_y_collision
    is_x_collision = above_x & below_x
    is_y_collision = above_y & below_y
    is_collision = is_x_collision & is_y_collision
    return is_collision


#  def check_collisions(P2, extent2, extent1):
    #  """
    #  P1, P2: (..., 1x3), (..., Nx3) object poses
    #  Returns (..., N) collision mask (True if collision)
    #  """
    #  bboxes = torch.stack([
        #  torch.stack([-extent2[..., 0], -extent2[..., 1]], dim=-1),
        #  torch.stack([extent2[..., 0], -extent2[..., 1]], dim=-1),
        #  torch.stack([extent2[..., 0], extent2[..., 1]], dim=-1),
        #  torch.stack([-extent2[..., 0], extent2[..., 1]], dim=-1),
    #  ], dim=-2)

    #  theta = P2[..., 2]
    #  # TODO figure out what is the right rotation matrix
    #  rot_matrix = torch.stack([
        #  torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1),
        #  torch.stack([-torch.sin(theta), torch.cos(theta)], dim=-1)
    #  ], dim=-2)
    #  bboxes = (bboxes @ rot_matrix)
    #  bboxes += P2[..., None, :2]

    #  above_x = bboxes[..., 0] >= -extent1[..., None, None, 0]
    #  above_y = bboxes[..., 1] >= -extent1[..., None, None, 1]
    #  below_x = bboxes[..., 0] <= extent1[..., None, None, 0]
    #  below_y = bboxes[..., 1] <= extent1[..., None, None, 1]
    #  is_x_collision = above_x.any(dim=-1) & below_x.any(dim=-1)
    #  is_y_collision = above_y.any(dim=-1) & below_y.any(dim=-1)
    #  is_collision = is_x_collision & is_y_collision

    #  if is_collision.any():
        #  ego_bboxes = torch.stack([
            #  torch.stack([-extent1[..., 0], -extent1[..., 1]], dim=-1),
            #  torch.stack([extent1[..., 0], -extent1[..., 1]], dim=-1),
            #  torch.stack([extent1[..., 0], extent1[..., 1]], dim=-1),
            #  torch.stack([-extent1[..., 0], extent1[..., 1]], dim=-1),
        #  ], dim=-2)
        #  # TODO for now assuming all the same ego vehicle
        #  ego_bboxes = ego_bboxes[0, 0].cpu().numpy()
        #  ego_poly = shapely.Polygon(ego_bboxes)
        #  for i, j, k in zip(*torch.where(is_collision)):
            #  rel_bbox = bboxes[i, j, k].cpu().numpy()
            #  rel_poly = shapely.Polygon(rel_bbox)
            #  is_collision[i, j, k] = rel_poly.intersects(ego_poly)
            #  if not is_collision[i, j, k]:
                #  print('changed coll')

    #  return is_collision


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
    ego_normals = torch.stack(ego_normals, dim=1)

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


def check_collisions(P2, extent2, extent1):
    """
    P1, P2: (..., 1x3), (..., Nx3) object poses
    Returns (..., N) collision mask (True if collision)
    """
    bboxes = torch.stack([
        torch.stack([-extent2[..., 0], -extent2[..., 1]], dim=-1),
        torch.stack([extent2[..., 0], -extent2[..., 1]], dim=-1),
        torch.stack([extent2[..., 0], extent2[..., 1]], dim=-1),
        torch.stack([-extent2[..., 0], extent2[..., 1]], dim=-1),
    ], dim=-2)

    theta = P2[..., 2]
    # TODO figure out what is the right rotation matrix
    rot_matrix = torch.stack([
        torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1),
        torch.stack([-torch.sin(theta), torch.cos(theta)], dim=-1)
    ], dim=-2)
    bboxes = (bboxes @ rot_matrix)
    bboxes += P2[..., None, :2]

    ego_bboxes = torch.stack([
        torch.stack([-extent1[..., 0], -extent1[..., 1]], dim=-1),
        torch.stack([extent1[..., 0], -extent1[..., 1]], dim=-1),
        torch.stack([extent1[..., 0], extent1[..., 1]], dim=-1),
        torch.stack([-extent1[..., 0], extent1[..., 1]], dim=-1),
    ], dim=-2)
    # TODO for now assuming all the same ego vehicle
    ego_bbox = ego_bboxes[0, 0]

    bboxes_shape = bboxes.shape[:3]
    shaped_is_collision = do_bboxes_intersect(bboxes.reshape((-1, 4, 2)), ego_bbox)
    is_collision = shaped_is_collision.reshape(bboxes_shape)

    return is_collision
