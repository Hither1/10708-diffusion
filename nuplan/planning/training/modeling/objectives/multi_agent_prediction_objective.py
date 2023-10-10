from typing import Dict, List, cast

import torch

from nuplan.planning.training.modeling.objectives.abstract_objective import AbstractObjective
from nuplan.planning.training.modeling.objectives.scenario_weight_utils import extract_scenario_type_weight
from nuplan.planning.training.modeling.types import FeaturesType, ScenarioListType, TargetsType
from nuplan.planning.training.preprocessing.features.agents_trajectory import AgentTrajectory


class MultiAgentPredictionObjective(AbstractObjective):

    def __init__(self, scenario_type_loss_weighting: Dict[str, float], weight: float = 1.0):
        self._name = 'multi_agent_prediction_objective'
        self._weight = weight
        self._fn_xy = torch.nn.modules.loss.MSELoss(reduction='none')
        self._fn_heading = torch.nn.modules.loss.L1Loss(reduction='none')

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

        pred = predicted_trajectory.data
        target = targets_trajectory.data
        masks = targets_trajectory.mask

        # Remove nans -- should only correspond to missing agents
        pred[~masks] = 0

        pred_xy, target_xy = pred[...,:2], target[...,:2]
        pred_heading, target_heading = pred[...,2], target[...,2]

        loss_xy = self._fn_xy(pred_xy, target_xy) * masks[...,None]
        loss_heading = self._fn_heading(pred_heading, target_heading) * masks

        # loss = loss_xy.mean(dim=3) + loss_heading
        # loss = loss.sum(dim=2).sum(dim=1) / masks.sum(dim=2).sum(dim=1)
        # loss = loss.mean(dim=0)
        
        loss = pred_xy.mean()

        # loss = (loss_xy.mean(dim=3) + loss_heading).sum(dim=2).mean(dim=1).mean(dim=0)
        loss = loss * self._weight

        print(loss)
        if torch.isnan(loss):
            import pdb; pdb.set_trace()

        return loss
