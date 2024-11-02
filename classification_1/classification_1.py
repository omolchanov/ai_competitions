from abc import ABC, abstractmethod

import sklearn.base
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import classification_report, accuracy_score

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# Pandas configuration
pd.set_option("display.precision", 2)
pd.set_option("expand_frame_repr", False)
pd.set_option('display.max_rows', None)

# Plots configuration
sns.set_theme(style='darkgrid')

# Import datasets
df = pd.read_csv('data/Train.csv')
df_test = pd.read_csv('data/Test.csv')

# Setting independent and target variables
X = df.iloc[:, 1:-1]
y = df['class'].astype(int)

# Split the dataframe onto train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

# Scale/normalize the data
X_train_scaled = MinMaxScaler().fit_transform(X_train)
X_test_scaled = MinMaxScaler().fit_transform(X_test)


class DataAnalyzer:

    @staticmethod
    def analyze_data():
        print(df.info())
        print('\n The number of classes: ', df.value_counts('class', normalize=True))
        print('\n Distribution of classes: ', df.groupby('class').size())

    @staticmethod
    def plot_data_distribution():
        plt.figure(figsize=(20, 10))

        ax = sns.histplot(data=df, x=y)
        ax.set(
            title='Classes Histogram',
            xlabel='Classes',
            ylabel='Frequency'
        )

        plt.show()


class BaseClassifier(ABC):

    @abstractmethod
    def fit_classifier(self, features_train: pd.DataFrame) -> None:
        pass

    def evaluate_classifier(self, features_test: pd.DataFrame) -> None:
        pass

    def find_best_parameters(self, params: dict, features: pd.DataFrame) -> None:
        pass


class SkLearnClassifier(BaseClassifier):
    def __init__(self, clf):
        self.clf = clf

    def fit_classifier(self, features_train) -> None:
        self.clf.fit(features_train, y_train)

    def evaluate_classifier(self, features_test) -> None:
        y_pred = self.clf.predict(features_test)

        print('\n', self.clf)
        print('Accuracy score: %.3f' % accuracy_score(y_pred, y_test))
        print(classification_report(y_pred, y_test))

    def find_best_parameters(self, params: dict, features: pd.DataFrame) -> None:
        print('\n', self.clf)

        grid = GridSearchCV(self.clf, params, cv=5, n_jobs=1, verbose=True)
        grid.fit(features, y_train)

        print('Best score: ', grid.best_score_)
        print('Best parameters: ', grid.best_params_, '\n')


def classify_service(
        clf: sklearn.base.BaseEstimator,
        features_train: pd.DataFrame,
        target_test: pd.DataFrame) -> None:

    clf_obj = SkLearnClassifier(clf)
    clf_obj.fit_classifier(features_train)
    clf_obj.evaluate_classifier(target_test)


def find_best_parameters_service(clf: sklearn.base.BaseEstimator) -> None:

    if isinstance(clf, DecisionTreeClassifier):
        skl_clf = SkLearnClassifier(clf)
        params = {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth': range(1, 15),
            'max_features': range(4, 15)
        }

        skl_clf.find_best_parameters(params, X_train)

    elif isinstance(clf, LogisticRegression):
        skl_clf = SkLearnClassifier(clf)

        params = {
            'penalty': ['l1', 'l2', 'elasticnet'],
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
        }

        skl_clf.find_best_parameters(params, X_train_scaled)


if __name__ == '__main__':

    # Analyze distribution of the target variable
    DataAnalyzer.analyze_data()
    DataAnalyzer.plot_data_distribution()

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

    # Perform classification with Decision tree and Logistic Regression classifiers
    classify_service(tree_clf, X_train, X_test)
    classify_service(lg_clf, X_train_scaled, X_test_scaled)

    # Finding the best parameters for Decision tree and Logistic Regression classifiers
    find_best_parameters_service(tree_clf)
    find_best_parameters_service(lg_clf)
