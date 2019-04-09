# Adding Components
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from Data_Access import get_apple_news, get_apple_stock, get_general_news, get_DJIA_index
from Sentiment_Model import LSTM
import Preprocess
from Preprocess import MergeDatasets
from Model import build_model

import numpy as np

# Selecting variant for analysis
bool_apple = True
bool_news = False

# Parameters for Sentiment Analysis Model
class_num = 2
batch_size = 24
sequence_length = 250
word_vector_dimension = 300
lstm_units = 64


# Extract Word Vectors
word_vectors = np.load('Word Vectors/wordVectors.npy').tolist()

# Extract Data From News Source

if bool_news & bool_apple:
    news = get_apple_news()
elif bool_news:
    news = get_general_news()

# Extract Data From Stock Market
if bool_apple:
    ticker = get_apple_stock()
else:
    ticker = get_DJIA_index()

# Structured as alternative branch when news is included
if bool_news:
    # Create Object of Sentiment Analysis Model
    sentiment = LSTM(class_num, batch_size, sequence_length,
                     word_vector_dimension, lstm_units, word_vectors, training=False)

    # Preprocess News Data
    news_sentiment = sentiment.evaluate(news, class_num, batch_size, sequence_length,
                                        word_vector_dimension, lstm_units, word_vectors)

    # Join Stock And News Data Into New Data Source
    merge = MergeDatasets(news_dataframe=news_sentiment, stock_dataframe=ticker)
    ticker = merge.merge()

# Preprocess Batches For Data Here
training_df, testing_df, training_matrix, testing_matrix = Preprocess.data_separation('2010-01-01', '2017-01-01',
                                                                                      '2017-01-01', '2018-01-01', ticker)
normalised_data = Preprocess.data_normalisation(ticker)
normalised_training_data = Preprocess.normalised_training(training_matrix, normalised_data)
normalised_testing_data = Preprocess.normalised_testing(testing_matrix, normalised_data)

# Parameters for Time Series Model
feature_length = 10
window_length = 20
batch_size = 256

# Create Training samples
x_train, y_train = Preprocess.create_samples(normalised_training_data, 20, 10)

# Building Time-series model
model = build_model([batch_size, window_length, feature_length])

# Back-test output
x_test, y_test = Preprocess.create_samples(normalised_testing_data, 20, 10)

import datetime  # For datetime objects

# Import the backtrader platform
import backtrader as bt
from Trading_Strategy import SentimentStrategy


if __name__ == '__main__':
    cerebro = bt.Cerebro()
    cerebro.addstrategy(SentimentStrategy)

    datapath1 = 'Data/AAPL_2018_2019.csv'
    datapath2 = 'Data/DJIA_2018_2019.csv'

    # Create a Data Feed
    data = bt.feeds.YahooFinanceCSVData(
        dataname=datapath1,
        fromdate=datetime.datetime(2018, 1, 1),
        todate=datetime.datetime(2019, 1, 1),
        reverse=False)


    # Create a Data Feed
    data2 = bt.feeds.YahooFinanceCSVData(
        dataname=datapath2,
        fromdate=datetime.datetime(2018, 1, 1),
        todate=datetime.datetime(2019, 1, 1),
        reverse=False)

    cerebro.adddata(data)
    cerebro.adddata(data)
    cerebro.broker.setcash(100000.0)

    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.run()

    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.plot()
