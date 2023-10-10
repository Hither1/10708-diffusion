from typing import Dict, List, cast

import torch

from nuplan.planning.training.modeling.objectives.abstract_objective import AbstractObjective
from nuplan.planning.training.modeling.objectives.scenario_weight_utils import extract_scenario_type_weight
from nuplan.planning.training.modeling.types import FeaturesType, ScenarioListType, TargetsType
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.features.agents_trajectory import AgentTrajectory


class TrajectoryWeightDecayImitationObjectiveMA(AbstractObjective):
    """
    Objective that drives the model to imitate the signals from expert behaviors/trajectories.
    When comparing model's predictions to expert's trajectory, it assigns more weight to earlier timestamps than later ones.
    Formula: mean(loss * exp(-index / poses))
    """

    def __init__(self, scenario_type_loss_weighting: Dict[str, float], weight: float = 1.0):
        """
        Initializes the class

        :param name: name of the objective
        :param weight: weight contribution to the overall loss
        """
        self._name = 'trajectory_weight_decay_imitation_objective_ma'
        self._weight = weight
        self._decay = 1.0

        self._fn_xy = torch.nn.modules.loss.L1Loss(reduction='none')
        self._fn_heading = torch.nn.modules.loss.L1Loss(reduction='none')
        self._scenario_type_loss_weighting = scenario_type_loss_weighting

    def name(self) -> str:
        """
        Name of the objective
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["agent_trajectories"]

    def compute(self, predictions: FeaturesType, targets: TargetsType, scenarios: ScenarioListType) -> torch.Tensor:
        """
        Computes the objective's loss given the ground truth targets and the model's predictions
        and weights it based on a fixed weight factor.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: loss scalar tensor
        """
        predicted_trajectory = cast(AgentTrajectory, predictions["agent_trajectories"])
        targets_trajectory = cast(AgentTrajectory, targets["agent_trajectories"])
        loss_weights = extract_scenario_type_weight(
            scenarios, self._scenario_type_loss_weighting, device=predicted_trajectory.data.device
        )

        # Add exponential decay of loss such that later error induce less penalty
        planner_output_steps = 16 # predicted_trajectory.xy.shape[1]
        decay_weight = torch.ones_like(predicted_trajectory.data[...,:2])
        decay_value = torch.exp(-torch.Tensor(range(planner_output_steps)) / (planner_output_steps * self._decay))
        decay_weight = decay_value[None,:,None].to(predicted_trajectory.data.device)

        broadcast_shape_xy = tuple([-1] + [1 for _ in range(predicted_trajectory.data.dim() - 1)])
        broadcasted_loss_weights_xy = loss_weights.view(broadcast_shape_xy)
        broadcast_shape_heading = tuple([-1] + [1 for _ in range(predicted_trajectory.data.dim() - 1)])
        broadcasted_loss_weights_heading = loss_weights.view(broadcast_shape_heading)

        weighted_xy_loss = self._fn_xy(predicted_trajectory.data[...,:2], targets_trajectory.data[...,:2]) * broadcasted_loss_weights_xy
        weighted_heading_loss = (
            self._fn_heading(predicted_trajectory.data[...,2], targets_trajectory.data[...,2])
            * broadcasted_loss_weights_heading.squeeze(-1)
        )

        # Handling masks
        weighted_xy_loss = weighted_xy_loss.mean(-1)
        weighted_xy_loss *= targets_trajectory.mask
        weighted_heading_loss *= targets_trajectory.mask

        # # Assert that broadcasting was done correctly
        # assert weighted_xy_loss.size() == predicted_trajectory.xy.size()
        # assert weighted_heading_loss.size() == predicted_trajectory.heading.size()

        return self._weight * (
            torch.mean(weighted_xy_loss * decay_weight) + torch.mean(weighted_heading_loss * decay_weight)
        )
