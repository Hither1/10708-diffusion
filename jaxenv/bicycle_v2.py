import jax
import jax.numpy as jnp
from typing import Tuple, Optional
import chex
from flax import struct


@struct.dataclass
class EgoState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    vel_x: float = 0.0
    vel_y: float = 0.0
    yaw_rate: float = 0.0


def compute_dynamics(ego_state, action, env_params):
    """
    Below is copied from nuPlan codebase:

    Class to forward simulate a dynamical system
    for 1 step, given an initial condition and
    an input.

    The model is a Kinematic bicycle model
    based on first order Euler discretization.
    Reference point is rear axle of vehicle.
    State is (x, y, yaw, vel_x, vel_y, yaw_rate).
    Input is (acceleration, steering_angle).

    Note: Forward Euler means that the inputs
    at time 0 will affect x,y,yaw at time 2.

    Adapted from https://arxiv.org/abs/1908.00219 (Eq.ns 6 in
    the paper have slightly different kinematics)
    """
    accel = action[0] * env_params.accel_coeff
    steer = action[1] * env_params.steer_coeff

    vel_init = jnp.sqrt(ego_state.vel_x**2 + ego_state.vel_y**2)
    vel = vel_init + accel * env_params.dt
    vel = jnp.clip(
        vel, 
        env_params.min_vel, 
        env_params.max_vel
    )

    yaw_rate = jnp.clip(
        vel_init * jnp.tan(steer) / env_params.wheel_base,
        env_params.min_yaw_rate,
        env_params.max_yaw_rate
    )
    yaw = ego_state.yaw + ego_state.yaw_rate * env_params.dt

    vel_x = vel * jnp.cos(ego_state.yaw)
    vel_y = vel * jnp.sin(ego_state.yaw)

    x = ego_state.x + ego_state.vel_x * env_params.dt
    y = ego_state.y + ego_state.vel_y * env_params.dt

    return EgoState(
        x=x,
        y=y,
        yaw=yaw,
        vel_x=vel_x,
        vel_y=vel_y,
        yaw_rate=yaw_rate
    )
