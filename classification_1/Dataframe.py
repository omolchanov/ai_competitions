from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# Pandas configuration
pd.set_option("display.precision", 2)
pd.set_option("expand_frame_repr", False)
pd.set_option('display.max_rows', None)


class Dataframe:
    """
    The Dataframe class is intended for analysis of data in the dataframe
    and provides basic ETL functionality
    """

    TRAIN_DATAFRAME_PATH = 'data/Train.csv'
    TEST_DATAFRAME_PATH = 'data/Test.csv'

    def _load(self) -> pd.DataFrame:
        """
        Lodas the training dataframe for CSV file
        :return: dataframe
        """

        df = pd.read_csv(self.TRAIN_DATAFRAME_PATH)
        return df

    @staticmethod
    def _remove_outliers(df, upper_coef, lower_coef):
        """
        Removes outliers from the dataframe basing on upper and lower coefficients
        applied to the values' inter-quartiles. Plots the whiskey plots showing
        the distribution of the values
        :param df: dataframe
        :param upper_coef: the coef for dropping the samples with values over the Q3
        :param lower_coef: the coef for dropping the samples with values under the Q1
        :return: dataframe with removed outliers
        """

        def draw_whiskey_plot(title):

            fig, ax = plt.subplots(1, df.shape[1], sharex=False)

            for i, c in enumerate(df.columns):
                sns.boxplot(y=df[c], ax=ax[i])

            plt.subplots_adjust(wspace=1.75)
            plt.suptitle(title)

            plt.show()

        print('\n Removing outliers')
        print('\n BEFORE')
        print(df.info())

        draw_whiskey_plot('Before removing outliers')

        for c in df.columns[:-1]:
            q1 = np.percentile(df[c], 25)
            q3 = np.percentile(df[c], 75)

            iqr = q3 - q1

            upper = np.where(df[c] >= (q3 + upper_coef * iqr))
            lower = np.where(df[c] <= (q3 - lower_coef * iqr))

            df.drop(upper[0], inplace=True)
            df.drop(lower[0], inplace=True)

            df.reset_index(drop=True, inplace=True)

        print('\n AFTER')
        print(df.info())

        draw_whiskey_plot('After removing outliers')

        return df

    @staticmethod
    def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes the duplicated samples from the dataframe
        :param df: dataframe
        :return: dataframe without duplicate
        """
        print('\n Removing duplicates')
        print('\n BEFORE')
        print(df.info())

        df.drop_duplicates(keep=False)

        print('\n AFTER')
        print(df.info())

        return df

    def get_df(self, test_size=0.2) -> [pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Sets the samples matrix and target variable
        Splits the dataframe
        :param test_size: the part of the dataframe assigned to the test set
        :return: train/test sets of samples and target variables
        """
        df = self._load()

        X = df.iloc[:, 1:-1]
        y = df['class'].astype('int')

        return train_test_split(X, y, test_size=test_size, random_state=17)

    def get_scaled_df(self) -> [pd.DataFrame, pd.DataFrame]:
        """
        Scales the train and test sets with MinMaxScaler
        :return: scaled train and test sets
        """
        X_train, X_test, _, _ = self.get_df()

        X_train_scaled = MinMaxScaler().fit_transform(X_train)
        X_test_scaled = MinMaxScaler().fit_transform(X_test)

        return X_train_scaled, X_test_scaled

    def get_nn_df(self) -> [pd.DataFrame, pd.Series]:
        """
        Prepares a Dataframe with samples and Series
        for training the Keras NN
        :return: train set of samples and target variables
        """
        df = self._load()
        df = df.drop(['Unnamed: 0'], axis=1)

        df = Dataframe._remove_duplicates(df)
        df = Dataframe._remove_outliers(df, 1, 1.45)

        X = df.iloc[:, 1:-1]
        y = df['class'].astype(int)

        return X, y

    def analyze_distribution(self) -> None:
        """
        Provides a values count for each class in the dataframe
        Provides a percentage distribution of each class in the dataframe
        """
        df = self._load()
        print(df.info())
        print('\n The number of classes: ', df.value_counts('class', normalize=True))
        print('\n Distribution of classes: ', df.groupby('class').size())

    def plot_data_distribution(self) -> None:
        """
        Builds a histogram for distribution of classes
        """
        df = self._load()
        y = df['class'].astype(int)

        plt.figure(figsize=(20, 10))

        ax = sns.histplot(data=df, x=y)
        ax.set(
            title='Classes Histogram',
            xlabel='Classes',
            ylabel='Frequency'
        )

        plt.show()
