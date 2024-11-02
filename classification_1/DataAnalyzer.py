import matplotlib.pyplot as plt
import seaborn as sns

# Plots configuration
sns.set_theme(style='darkgrid')


class DataAnalyzer:

    @staticmethod
    def analyze_distribution(df):
        print(df.info())
        print('\n The number of classes: ', df.value_counts('class', normalize=True))
        print('\n Distribution of classes: ', df.groupby('class').size())

    @staticmethod
    def plot_data_distribution(df, y):
        plt.figure(figsize=(20, 10))

        ax = sns.histplot(data=df, x=y)
        ax.set(
            title='Classes Histogram',
            xlabel='Classes',
            ylabel='Frequency'
        )

        plt.show()
