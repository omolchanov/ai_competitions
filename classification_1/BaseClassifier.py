from abc import ABC, abstractmethod
import pandas as pd


class BaseClassifier(ABC):

    @abstractmethod
    def fit_classifier(self, features_train: pd.DataFrame, target_train: pd.Series) -> None:
        """
        Abstract method for a logic responsible for fitting/training classifier as well as
        building the neural network model
        :param features_train: a Dataframe containing samples of training data
        :param target_train: a Series containing target variables of training data
        """
        pass

    @abstractmethod
    def evaluate_classifier(self, features_test: pd.DataFrame, target_test: pd.Series) -> None:
        """
        Abstract method intended for wire-framing a logic for evaluating the classifiers
        :param features_test: a Dataframe containing samples of test data
        :param target_test: a Series containing target variables of test data
        """
        pass

    @abstractmethod
    def find_best_parameters(self, params: dict, features: pd.DataFrame, target: pd.Series) -> None:
        """
        Abstract method acts as skeleton for a logic responsible for finding
        the best parameters for a particular classifier
        :param params: dictionary of parameters for spanning amongst them
        :param features: a Dataframe containing samples
        :param target:  a Series containing target variables
        """
        pass
