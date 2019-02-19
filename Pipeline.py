data =
input = data.retrieve()
preprocessing = Preprocessing()
preprocessing.set_input(input)
(preprocessed_data, original_dataframe) = preprocessing.preprocess()
vectorization =
vectorization.set_input(preprocessed_data, original_dataframe)
output = vectorization.fit()