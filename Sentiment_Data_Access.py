import numpy as np
from os import listdir
from os.path import isfile, join


# Class for Data Access
class Data:
    def __init__(self):
        # Setting up word vectors
        self.word_list = np.load('Word Vectors/wordList.npy')
        self.word_vectors = np.load('Word Vectors/wordVectors.npy')
        try:
            self.id_matrix = np.load('Word Vectors/idMatrix.npy')
        except IOError:
            self.id_matrix = []

        self.word_list = self.word_list.tolist()
        self.word_list = [word.decode('UTF-8') for word in self.word_list]

        # Checking that both numpy files have loaded properly
        print(len(self.word_list))
        print(self.word_vectors.shape)

        # Loading training set
        self.positive_files = ['Data/Sentiment Training/Positive Reviews/'
                               + f for f in listdir('Data/Sentiment Training/Positive Reviews/')
                               if isfile(join('Data/Sentiment Training/Positive Reviews/', f))]
        self.negative_files = ['Data/Sentiment Training/Negative Reviews/'
                               + f for f in listdir('Data/Sentiment Training/Negative Reviews/')
                               if isfile(join('Data/Sentiment Training/Negative Reviews/', f))]

        # Checking that both training sets have loaded properly
        print(len(self.positive_files))
        print(len(self.negative_files))

    def get_word_list(self):
        return self.word_list

    def get_word_vectors(self):
        return self.word_vectors

    def get_id_matrix(self):
        return self.id_matrix

    def get_training_files(self):
        return self.positive_files, self.negative_files

    def retry_id_matrix_download(self):
        self.id_matrix = np.load('Word Vectors/idMatrix.npy')
        return self.id_matrix
