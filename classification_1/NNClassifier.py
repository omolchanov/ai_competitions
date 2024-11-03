import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

from BaseClassifier import BaseClassifier


class NNClassifier(BaseClassifier):

    def __init__(self):
        self.model = None

    def fit_classifier(self, features_train: pd.DataFrame, target_train: pd.Series) -> None:
        """
        Implements a logic responsible for fitting/training a Keras NN classifier
        :param features_train: a Dataframe containing samples of training data
        :param target_train: a Series containing target variables of training data
        """

        self.model = Sequential([
            Dense(32, activation='tanh'),
            Dense(64, activation='tanh'),
            Dense(128, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        self.model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
        self.model.fit(features_train, target_train, epochs=200, batch_size=10, verbose=True)

    def evaluate_classifier(self, features_test: pd.DataFrame, target_test: pd.Series) -> None:
        """
        Evaluates a Keras NN classifier
        :param features_test: a Dataframe containing samples of test data
        :param target_test: a Series containing target variables of test data
        :param features_test:
        :param target_test:
        :return:
        """
        loss, accuracy = self.model.evaluate(features_test, target_test)
        print('Loss: %.2f | Accuracy: %.2f' % (loss, accuracy * 100))

    def find_best_parameters(self, params: dict, features: pd.DataFrame, target: pd.Series) -> None:
        pass


