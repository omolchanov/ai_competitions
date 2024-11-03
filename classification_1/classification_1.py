"""
The main script service for the Kaggle simple classification competition
https://www.kaggle.com/competitions/classification-task-simple-2022
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sklearn.base

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from Dataframe import Dataframe

from SkLearnClassifier import SkLearnClassifier
from NNClassifier import NNClassifier


def analyze_data_service() -> None:
    """
    Recalls functions for analysing and
    plotting the distribution of the target variable
    """

    Dataframe().analyze_distribution()
    Dataframe().plot_data_distribution()


def classify_service(clf: sklearn.base.BaseEstimator | NNClassifier) -> None:
    """
    Identifies the classifier's object and runs evaluation logic basing on its type
    :param clf: Sklearn or Keras NN classifier
    """

    if isinstance(clf, DecisionTreeClassifier):
        X_train, X_test, y_train, y_test = Dataframe().get_df()

        clf_obj = SkLearnClassifier(clf)
        clf_obj.fit_classifier(X_train, y_train)
        clf_obj.evaluate_classifier(X_test, y_test)

    elif isinstance(clf, LogisticRegression):
        _, _, y_train, y_test = Dataframe().get_df()
        X_train_scaled, X_test_scaled = Dataframe().get_scaled_df()

        clf_obj = SkLearnClassifier(clf)
        clf_obj.fit_classifier(X_train_scaled, y_train)
        clf_obj.evaluate_classifier(X_test_scaled, y_test)

    elif isinstance(clf, NNClassifier):
        X, y = Dataframe().get_nn_df()

        clf.fit_classifier(X, y)
        clf.evaluate_classifier(X, y)


def find_best_parameters_service(clf: sklearn.base.BaseEstimator) -> None:
    """
    Identifies the classifier's object and runs logic that finds the
    hyperparameters for a particular classifier
    :param clf: Sklearn classifier
    """

    X_train, X_test, y_train, y_test = Dataframe().get_df()
    X_train_scaled, X_test_scaled = Dataframe().get_scaled_df()

    if isinstance(clf, DecisionTreeClassifier):
        skl_clf = SkLearnClassifier(clf)
        params = {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth': range(1, 15),
            'max_features': range(4, 15)
        }

        skl_clf.find_best_parameters(params, X_train, y_train)

    elif isinstance(clf, LogisticRegression):
        skl_clf = SkLearnClassifier(clf)
        params = {
            'penalty': ['l1', 'l2', 'elasticnet'],
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
        }

        skl_clf.find_best_parameters(params, X_train_scaled, y_train)


if __name__ == '__main__':

    # Analyze distribution of the target variable
    analyze_data_service()

    # Evaluate Decision tree and Logistic Regression classifiers
    tree_clf = DecisionTreeClassifier(
        criterion='log_loss',
        max_depth=14,
        max_features=8
    )

    lg_clf = LogisticRegression(
        C=1,
        penalty='l2',
        solver='lbfgs',
        max_iter=100000,
        class_weight={0: 0.3, 1: 0.27, 2: 0.23, 3: 0.2},
        multi_class='multinomial'
    )

    # Evaluate Keras neural network classifier
    nn_clf = NNClassifier()

    # Perform classification with Decision tree and Logistic Regression classifiers
    classify_service(tree_clf)
    classify_service(lg_clf)
    classify_service(nn_clf)

    # Finding the best parameters for Decision tree and Logistic Regression classifiers
    find_best_parameters_service(tree_clf)
    find_best_parameters_service(lg_clf)
