import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Retrieving data
data = pd.read_csv('Data/Combined_News_DJIA.csv')

# Creating test and training data
train_data = data[data['Date'] < '2015-01-01']
test_data = data[data['Date'] > '2014-12-31']

train_headlines = []
for row in range(0, len(train_data.index)):
    train_headlines.append(' '.join(str(x) for x in train_data.iloc[row, 2:27]))

# Using Bag of Words model for Logistic Regression
bow_vectorizer = CountVectorizer()
basic_train = bow_vectorizer.fit_transform(train_headlines)

# Adding Logistic Regression model
bow_model = LogisticRegression()
bow_model = bow_model.fit(basic_train, train_data["Label"])

# Creating testing matrix
test_headlines = []
for row in range(0, len(test_data.index)):
    test_headlines.append(' '.join(str(x) for x in test_data.iloc[row, 2:27]))
basic_test = bow_vectorizer.transform(test_headlines)

# Making prediction based on Logistic Regression model
bag_of_words_predictions = bow_model.predict(basic_test)
result_bow = pd.crosstab(test_data["Label"], bag_of_words_predictions, rownames=["Actual"], colnames=["Predicted"])

# Using n-gram model for Logistic Regression
ngram_vectorizer = CountVectorizer(ngram_range=(2, 3))
ngram_train = ngram_vectorizer.fit_transform(train_headlines)

ngram_model = LogisticRegression()
ngram_model = ngram_model.fit(ngram_train, train_data["Label"])

test_headlines = []
for row in range(0, len(test_data.index)):
    test_headlines.append(' '.join(str(x) for x in test_data.iloc[row, 2:27]))
ngram_test = ngram_vectorizer.transform(test_headlines)
ngram_predictions = ngram_model.predict(ngram_test)

result_ngram = pd.crosstab(test_data["Label"], ngram_predictions, rownames=["Actual"], colnames=["Predicted"])
