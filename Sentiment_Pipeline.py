from Sentiment_Data_Access import Data
from Sentiment_Preprocessing import Preprocess, get_training_batch_queue, get_testing_batch_queue
from Sentiment_Model import LSTM

# Setting up general parameters
sequence_length = 250
word_vector_dimension = 300
file_number = 25000

# LSTM Model Hyperparameters
batch_size = 24
lstm_units = 64
class_num = 2
iterations = 10000

# Retrieving Word Vectors and Training Files
sent_data = Data()
id_matrix = sent_data.get_id_matrix()
word_list = sent_data.get_word_list()
word_vectors = sent_data.get_word_vectors()
positive_files, negative_files = sent_data.get_training_files()

# Setting up preprocessing functionality
pre_process = Preprocess()

# Creating new ID matrix only necessary if not present in Word Vectors folder
if len(id_matrix) == 0:
    pre_process.create_id_matrix(file_number, sequence_length, word_list, positive_files, negative_files)
    id_matrix = sent_data.retry_id_matrix_download()

# Structuring model
model = LSTM(class_num, batch_size, sequence_length, word_vector_dimension, lstm_units, word_vectors)

print("model created")

# Training model
batch_queue, label_queue = get_training_batch_queue(iterations, batch_size, sequence_length, id_matrix)
model.fit(iterations, batch_queue, label_queue)
test_batch_queue, test_label_queue = get_testing_batch_queue(iterations, batch_size, sequence_length, id_matrix)
model.test(iterations, test_batch_queue, test_label_queue)
