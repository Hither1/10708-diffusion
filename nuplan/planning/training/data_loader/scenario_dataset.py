import logging
from typing import List, Optional, Tuple

import torch.utils.data

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.data_augmentation.abstract_data_augmentation import AbstractAugmentor
from nuplan.planning.training.modeling.types import FeaturesType, ScenarioListType, TargetsType
from nuplan.planning.training.preprocessing.feature_preprocessor import FeaturePreprocessor

from diskcache import Cache
import multiprocessing as mp

logger = logging.getLogger(__name__)


class ScenarioDataset(torch.utils.data.Dataset):
    """
    Dataset responsible for consuming scenarios and producing pairs of model inputs/outputs.
    """

    def __init__(
        self,
        scenarios: List[AbstractScenario],
        feature_preprocessor: FeaturePreprocessor,
        augmentors: Optional[List[AbstractAugmentor]] = None,
        cache_dir = None,
        load_single_sample = False
    ) -> None:
        """
        Initializes the scenario dataset.
        :param scenarios: List of scenarios to use as dataset examples.
        :param feature_preprocessor: Feature and targets builder that converts samples to model features.
        :param augmentors: Augmentor object for providing data augmentation to data samples.
        """
        super().__init__()

        if len(scenarios) == 0:
            logger.warning('The dataset has no samples')

        self._scenarios = scenarios
        self._feature_preprocessor = feature_preprocessor
        self._augmentors = augmentors

        self._load_single_sample = load_single_sample

        # if cache_dir is not None:
        #     # self._cache = Cache(
        #     #     directory=cache_dir, 
        #     #     size_limit=int(300 * 1024 ** 3)
        #     # )
        #     # self._cache = mp.Manager().dict()

        #     self._dummy_cache = None

    def __getitem__(self, idx: int) -> Tuple[FeaturesType, TargetsType, ScenarioListType]:
        """
        Retrieves the dataset examples corresponding to the input index
        :param idx: input index
        :return: model features and targets
        """
        # idx = idx % 100
        # scenario = self._scenarios[idx]

        # if idx in self._cache:
        #     features, targets = self._cache[idx]
        # else:
        #     features, targets, _ = self._feature_preprocessor.compute_features(scenario)
        #     self._cache[idx] = features, targets

        # idx = 0
        # scenario = self._scenarios[idx]

        # if self._dummy_cache is None:
        #     features, targets, _ = self._feature_preprocessor.compute_features(scenario)
        #     self._dummy_cache = features, targets
        # else:
        #     features, targets = self._dummy_cache

        if self._load_single_sample:
            idx = 0

        scenario = self._scenarios[idx]
        features, targets, _ = self._feature_preprocessor.compute_features(scenario)

        if self._augmentors is not None:
            for augmentor in self._augmentors:
                augmentor.validate(features, targets)
                features, targets = augmentor.augment(features, targets, scenario)

        features = {key: value.to_feature_tensor() for key, value in features.items()}
        targets = {key: value.to_feature_tensor() for key, value in targets.items()}
        scenarios = [scenario]

        return features, targets, scenarios

    def __len__(self) -> int:
        """
        Returns the size of the dataset (number of samples)

        :return: size of dataset
        """
        return len(self._scenarios)
