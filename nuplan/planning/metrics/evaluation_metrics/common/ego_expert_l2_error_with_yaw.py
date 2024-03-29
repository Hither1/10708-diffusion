from typing import List

from nuplan.planning.metrics.evaluation_metrics.base.metric_base import MetricBase
from nuplan.planning.metrics.metric_result import MetricStatistics, MetricStatisticsType, TimeSeries
from nuplan.planning.metrics.utils.expert_comparisons import compute_traj_errors
from nuplan.planning.metrics.utils.state_extractors import extract_ego_center_with_heading, extract_ego_time_point
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory


class EgoExpertL2ErrorWithYawStatistics(MetricBase):
    """Ego pose and heading L2 error metric w.r.t expert."""

    def __init__(self, name: str, category: str, discount_factor: float, heading_diff_weight: float = 2.5) -> None:
        """
        Initializes the EgoExpertL2ErrorWithYawStatistics class
        :param name: Metric name
        :param category: Metric category
        :param discount_factor: Displacement at step i is dicounted by discount_factor^i
        :heading_diff_weight: The weight of heading differences.
        """
        super().__init__(name=name, category=category)
        self._discount_factor = discount_factor
        self._heading_diff_weight = heading_diff_weight

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Returns the estimated metric
        :param history: History from a simulation engine
        :param scenario: Scenario running this metric
        :return the estimated metric.
        """
        ego_states = history.extract_ego_state
        expert_states = scenario.get_expert_ego_trajectory()

        ego_traj = extract_ego_center_with_heading(ego_states)
        expert_traj = extract_ego_center_with_heading(expert_states)

        error = compute_traj_errors(
            ego_traj=ego_traj,
            expert_traj=expert_traj,
            discount_factor=self._discount_factor,
            heading_diff_weight=self._heading_diff_weight,
        )

        ego_timestamps = extract_ego_time_point(ego_states)

        statistics_type_list = [MetricStatisticsType.MAX, MetricStatisticsType.MEAN, MetricStatisticsType.P90]

        time_series = TimeSeries(unit='None', time_stamps=list(ego_timestamps), values=list(error))

        metric_statistics = self._compute_time_series_statistic(
            time_series=time_series, statistics_type_list=statistics_type_list
        )

        results: List[MetricStatistics] = self._construct_metric_results(
            metric_statistics=metric_statistics, scenario=scenario, time_series=time_series
        )
        return results, None
