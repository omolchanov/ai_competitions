from abc import ABC, abstractmethod

import pandas as pd


class BaseClassifier(ABC):

    @abstractmethod
    def fit_classifier(self, features_train: pd.DataFrame, target_train: pd.Series) -> None:
        pass

    def evaluate_classifier(self, features_test: pd.DataFrame, target_test: pd.Series) -> None:
        pass

    def find_best_parameters(self, params: dict, features: pd.DataFrame, target: pd.Series) -> None:
        pass
