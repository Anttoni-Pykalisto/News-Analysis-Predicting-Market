import pandas as pd
import numpy as np
from sklearn import preprocessing


def data_separation(start_train, end_train, start_test, end_test, dataframe):
    training_df = dataframe[(dataframe.index > start_train) & (dataframe.index < end_train)]
    training_matrix = training_df.values
    testing_df = dataframe[(dataframe.index > start_test) & (dataframe.index < end_test)]
    testing_matrix = testing_df.values
    return training_df, testing_df, training_matrix, testing_matrix


def data_normalisation(dataframe):
    min_max_scalar = preprocessing.MinMaxScaler(feature_range=(0, 1))
    return min_max_scalar.fit_transform(dataframe.values), dataframe


def normalised_training(training_matrix, scaled_values):
    training_size = len(training_matrix[:, 0])
    return scaled_values[0:training_size, :]


def normalised_testing(testing_matrix, scaled_values):
    return scaled_values[len(testing_matrix[:, 0]):len(testing_matrix[:, 0]) + len(testing_matrix[:, 0]), :]


def create_samples(normalised_data, window, feature):
    samples = []
    for i in range(len(normalised_data[:, 0]) - (window + 1)):
        samples.append(normalised_data[i:i + (window + 1)])

    samples = np.array(samples)
    x_samples = samples[:, :-1]
    x_samples = np.reshape(x_samples, (x_samples.shape[0], x_samples.shape[1], feature))
    y_samples = samples[:, -1][:, -1]
    return x_samples, y_samples


class MergeDatasets:
    def __init__(self, news_dataframe, stock_dataframe):
        self.news = news_dataframe
        self.stock = stock_dataframe

    def merge(self):
        return pd.merge(self.news, self.stock, how='inner', left_index=True, right_index=True)

