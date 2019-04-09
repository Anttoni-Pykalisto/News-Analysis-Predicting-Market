import numpy as np
import re
from random import randint


# Class for Preprocessing
class Preprocess:
    def __init__(self):
        # Regex to ensure tokens are alphanumeric
        self.make_alphanum = re.compile("[^A-Za-z0-9 ]+")

    # Cleaning Sentences
    def clean_sentences(self, string):
        string = string.lower().replace("<br />", " ")
        return re.sub(self.make_alphanum, "", string.lower())

    # Creating ID matrix
    def create_id_matrix(self, file_number, sequence_length, word_list, positive_files, negative_files):
        # Initialise shape of ID matrix
        id_matrix = np.zeros((file_number, sequence_length), dtype='int32')

        # Inserting content from positive files to ID matrix
        file_count = 0
        for pf in positive_files:
            with open(pf, "r", encoding='UTF-8') as f:
                index = 0
                line = f.readline()
                split = self.clean_sentences(line).split()
                for word in split:
                    try:
                        id_matrix[file_count][index] = word_list.index(word)
                    except ValueError:
                        id_matrix[file_count][index] = 399999   # Case if word is not part of lexicon
                    index = index + 1
                    if index >= sequence_length:
                        break
                file_count = file_count + 1

        # Inserting content from negative files to ID matrix
        for nf in negative_files:
            with open(nf, "r", encoding='UTF-8') as f:
                index = 0
                line = f.readline()
                split = self.clean_sentences(line).split()
                for word in split:
                    try:
                        id_matrix[file_count][index] = word_list.index(word)
                    except ValueError:
                        id_matrix[file_count][index] = 399999  # Case if word is not part of lexicon
                    index = index + 1
                    if index >= sequence_length:
                        break
                file_count = file_count + 1

        # Saving ID Matrix
        np.save('Word Vectors/idMatrix', id_matrix)
        print("New ID Matrix Created")


# Creating Training batches
def get_training_batch(batch_size, sequence_length, id_matrix):
    labels = []
    batch = np.zeros([batch_size, sequence_length])
    for i in range(batch_size):
        if i % 2 == 0:
            num = randint(1, 11499)
            labels.append([1, 0])
        else:
            num = randint(13499, 24999)
            labels.append([0, 1])
        batch[i] = id_matrix[num-1:num]
    return batch, labels


# Creating training batch queue
def get_training_batch_queue(iterations, batch_size, sequence_length, id_matrix):
    batch_queue = []
    label_queue = []
    for i in range(iterations):
        batch, label = get_training_batch(batch_size, sequence_length, id_matrix)
        batch_queue.append(batch)
        label_queue.append(label)
    return batch_queue, label_queue


# Creating Testing batches
def get_testing_batch(batch_size, sequence_length, id_matrix):
    labels = []
    batch = np.zeros([batch_size, sequence_length])
    for i in range(batch_size):
        num = randint(11499, 13499)
        if num <= 12499:
            labels.append([1, 0])
        else:
            labels.append([0, 1])
        batch[i] = id_matrix[num-1:num]
    return batch, labels


# Creating testing batch queue
def get_testing_batch_queue(iterations, batch_size, sequence_length, id_matrix):
    batch_queue = []
    label_queue = []
    for i in range(iterations):
        batch, label = get_testing_batch(batch_size, sequence_length, id_matrix)
        batch_queue.append(batch)
        label_queue.append(label)
    return batch_queue, label_queue
