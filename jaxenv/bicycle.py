import jax
import jax.numpy as jnp
from typing import Tuple, Optional
import chex
from flax import struct


@struct.dataclass
class EgoState:
    position_x: float = 0.0
    position_y: float = 0.0
    heading: float = 0.0
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    acceleration_x: float = 0.0
    acceleration_y: float = 0.0
    angular_velocity: float = 0.0
    angular_acceleration: float = 0.0
    tire_steering_angle: float = 0.0
    tire_steering_rate: float = 0.0


def update_commands(ego_state, accel_cmd, steer_cmd, env_params):
    accel = ego_state.acceleration_x
    steer = ego_state.tire_steering_angle

    ideal_accel = accel_cmd # (accel_cmd + 0.75) * 0.3
    ideal_steer = steer + steer_cmd * env_params.dt * 0.1

    updated_accel = env_params.dt / (env_params.dt + env_params.accel_time_constant) * (ideal_accel - accel) + accel
    updated_steer = (
        env_params.dt / (env_params.dt + env_params.steer_angle_time_constant) * (ideal_steer - steer)
        + steer
    )
    updated_steer_rate = (updated_steer - steer) / env_params.dt

    return EgoState(
        position_x=ego_state.position_x,
        position_y=ego_state.position_y,
        heading=ego_state.heading,
        velocity_x=ego_state.velocity_x,
        velocity_y=ego_state.velocity_y,
        acceleration_x=updated_accel,
        acceleration_y=0.0,
        tire_steering_angle=steer,
        tire_steering_rate=updated_steer_rate
    )


def get_state_dot(ego_state, env_params):
    longitudinal_speed = ego_state.velocity_x
    x_dot = longitudinal_speed * jnp.cos(ego_state.heading)
    y_dot = longitudinal_speed * jnp.sin(ego_state.heading)
    yaw_dot = longitudinal_speed * jnp.tan(ego_state.tire_steering_angle) / env_params.wheel_base

    return EgoState(
        position_x=x_dot,
        position_y=y_dot,
        heading=yaw_dot,
        velocity_x=ego_state.acceleration_x,
        velocity_y=ego_state.acceleration_y,
        tire_steering_angle=ego_state.tire_steering_rate
    )


def forward_integrate(init, delta, dt):
    return init + delta * dt


def principal_value(angle, min_=-jnp.pi):
    lhs = (angle - min_) % (2 * jnp.pi) + min_
    return lhs


def propagate_state(ego_state, accel_cmd, steer_cmd, env_params):
    propagating_state = update_commands(ego_state, accel_cmd, steer_cmd, env_params)

    # Compute state derivatives
    state_dot = get_state_dot(propagating_state, env_params)

    # Integrate position and heading
    dx = state_dot.position_x * env_params.dt
    next_x = ego_state.position_x + dx
    # next_x = forward_integrate(ego_state.position_x, state_dot.position_x, env_params.dt)
    dy = state_dot.position_y * env_params.dt
    next_y = ego_state.position_y + dy
    # next_y = forward_integrate(ego_state.position_y, state_dot.position_y, env_params.dt)
    next_heading = forward_integrate(
        ego_state.heading, state_dot.heading, env_params.dt
    )
    # Wrap angle between [-pi, pi]
    next_heading = principal_value(next_heading)

    # Compute rear axle velocity in car frame
    # next_point_velocity_x = jnp.clip(
    #     forward_integrate(ego_state.velocity_x, state_dot.velocity_x, env_params.dt),
    #     0.0,
    #     5.0
    # )
    next_point_velocity_x = 3.0
    next_point_velocity_y = 0.0  # Lateral velocity is always zero in kinematic bicycle model

    # Integrate steering angle and clip to bounds
    next_point_tire_steering_angle = jnp.clip(
        forward_integrate(ego_state.tire_steering_angle, state_dot.tire_steering_angle, env_params.dt),
        -env_params.max_steering_angle,
        env_params.max_steering_angle,
    )

    # Compute angular velocity
    next_point_angular_velocity = (
        next_point_velocity_x * jnp.tan(next_point_tire_steering_angle) / env_params.wheel_base
    )

    angular_accel = (next_point_angular_velocity - ego_state.angular_velocity) / env_params.dt

    # jax.debug.breakpoint()
    # jax.debug.print("{} {}", dx, dy)

    return EgoState(
        position_x=next_x,
        position_y=next_y,
        heading=next_heading,
        velocity_x=next_point_velocity_x,
        velocity_y=next_point_velocity_y,
        acceleration_x=state_dot.velocity_x,
        acceleration_y=state_dot.velocity_y,
        angular_velocity=next_point_angular_velocity,
        angular_acceleration=angular_accel,
        tire_steering_angle=next_point_tire_steering_angle,
        tire_steering_rate=state_dot.tire_steering_angle
    )
