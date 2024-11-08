from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

import pandas as pd

from BaseClassifier import BaseClassifier


class SkLearnClassifier(BaseClassifier):
    def __init__(self, clf):
        self.clf = clf

    def fit_classifier(self, features_train: pd.DataFrame, target_train: pd.Series) -> None:
        """
        Implements a logic responsible for fitting/training a SkLearn classifier
        :param features_train: a Dataframe containing samples of training data
        :param target_train: a Series containing target variables of training data
        """
        self.clf.fit(features_train, target_train)

    def evaluate_classifier(self, features_test: pd.DataFrame, target_test: pd.Series) -> None:
        """
        Evaluates a SkLearn classifier
        :param features_test: a Dataframe containing samples of test data
        :param target_test: a Series containing target variables of test data
        """
        y_pred = self.clf.predict(features_test)

        print('\n', self.clf)
        print('Accuracy score: %.3f' % accuracy_score(y_pred, target_test))
        print(classification_report(y_pred, target_test))

    def find_best_parameters(self, params: dict, features: pd.DataFrame, target: pd.Series) -> None:
        """
        Finds the best hypermeters for a Sklearn classifier with the grid search
        :param params: dictionary of parameters for spanning amongst them
        :param features: a Dataframe containing samples
        :param target:  a Series containing target variables
        """
        print('\n', self.clf)

        grid = GridSearchCV(self.clf, params, cv=5, n_jobs=1, verbose=True)
        grid.fit(features, target)

        print('Best score: ', grid.best_score_)
        print('Best parameters: ', grid.best_params_, '\n')
