import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
import time, math

# Extracting and Processing data
aapl = pd.read_csv('Data/EOD-AAPL.csv')
aapl = aapl.reindex(index=aapl.index[::-1])
aapl.index = aapl.index[::-1]
aapl.index = aapl.Date
aapl['Open'] = aapl['Adj_Open']
aapl['High'] = aapl['Adj_High']
aapl['Low'] = aapl['Adj_Low']
aapl['Close'] = aapl['Adj_Close']
aapl.drop(['Volume'], 1, inplace=True)
aapl.drop(['Dividend'], 1, inplace=True)
aapl.drop(['Date'], 1, inplace=True)
aapl.drop(['Split'], 1, inplace=True)
aapl.drop(['Adj_Open'], 1, inplace=True)
aapl.drop(['Adj_High'], 1, inplace=True)
aapl.drop(['Adj_Low'], 1, inplace=True)
aapl.drop(['Adj_Close'], 1, inplace=True)
aapl.drop(['Adj_Volume'], 1, inplace=True)

# Separating Data for Training and Testing
training_df = aapl[aapl.index < '2016-01-01']
training_matrix = training_df.values
testing_df = aapl[(aapl.index > '2016-01-01') & (aapl.index < '2019-01-01')]
testing_matrix = testing_df.values

# Scaling Data from 0 to 1
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
scaled_values = min_max_scaler.fit_transform(aapl.values)

# Creating scaled data for training and testing
training_scaled = scaled_values[0:len(training_matrix[:, 0]), :]
testing_scaled = scaled_values[len(training_matrix[:, 0]):len(training_matrix[:, 0]) + len(testing_matrix[:, 0]), :]

feature_number = 4
window_number = 10
dropout = 0.3

# Creating Samples
samples = []
for i in range(len(training_scaled[:, 0]) - (window_number+1)):
    samples.append(training_scaled[i:i + (window_number+1)])

# Checking that all samples were lined up
print(len(samples))
samples = np.array(samples)
print(samples.shape)

# Gets all values but from last column
x_train = samples[:, :-1]
print(x_train.shape)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], feature_number))
print(x_train.shape)

# Gets all last column values
y_train = samples[:, -1][:, -1]
print(y_train.shape)

print(x_train[0], y_train[0])

model = Sequential()

model.add(LSTM(256, input_shape=(window_number, feature_number), return_sequences=True))
model.add(Dropout(dropout))

model.add(LSTM(256, input_shape=(window_number, feature_number), return_sequences=False))
model.add(Dropout(dropout))

model.add(Dense(32, kernel_initializer="uniform", activation='relu'))
model.add(Dense(1, kernel_initializer="uniform", activation='relu'))

start = time.time()
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
print("Compilation Time : ", time.time() - start)

model.fit(x=x_train, y=y_train, batch_size=512, epochs=100, validation_split=0.1, verbose=1)

# Creating Testing Samples
samples = []
for i in range(len(testing_scaled[:, 0]) - (window_number+1)):
    samples.append(testing_scaled[i:i + (window_number+1)])

# Checking that all samples were lined up
print(len(samples))
samples = np.array(samples)
print(samples.shape)

# Gets all values but from last column
x_test = samples[:, :-1]
# print(x_train.shape)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], feature_number))
print(x_train.shape)

# Gets all last column values
y_test = samples[:, -1][:, -1]
print(y_test.shape)

print(x_test[0], y_test[0])

train_score = model.evaluate(x_train, y_train, verbose=0)
print('Train Score: %.5f MSE (%.2f RMSE)' % (train_score[0], math.sqrt(train_score[0])))

test_Score = model.evaluate(x_test, y_test, verbose=0)
print('Test Score: %.5f MSE (%.2f RMSE)' % (test_Score[0], math.sqrt(test_Score[0])))

p = model.predict(x_test)
print(p.shape)

def denormalize(original, normalized_value):
    original = original['Close'].values.reshape(-1, 1)
    normalized_value = normalized_value.reshape(-1, 1)

    # return df.shape, p.shape
    min_max_scaler = preprocessing.MinMaxScaler()
    a = min_max_scaler.fit_transform(original)
    new = min_max_scaler.inverse_transform(normalized_value)
    return new


reg_p = denormalize(aapl, p)
reg_y = denormalize(aapl, y_test)

plt.plot(reg_p, color='red', label='Prediction')
plt.plot(reg_y, color='blue', label='Actual')
plt.legend(loc='best')
plt.show()
