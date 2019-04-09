import tensorflow as tf
import numpy as np
import datetime
import re
from nltk.corpus import stopwords


class LSTM:
    def __init__(self, class_num, batch_size, sequence_length,
                 word_vector_dimension, lstm_units, word_vectors,  training=True):
        # tf.reset_default_graph()
        if training:
            with tf.Session() as sess:
                # Creating placeholder label and input
                self.labels = tf.placeholder(tf.float32, [batch_size, class_num])
                self.input_data = tf.placeholder(tf.int32, [batch_size, sequence_length])

                # Getting word vectors
                data = tf.Variable(tf.zeros([batch_size, sequence_length, word_vector_dimension]), dtype=tf.float32)
                data = tf.nn.embedding_lookup(word_vectors, self.input_data)

                # Feeding both the LSTM cell and the 3-D tensor full of input data into a function called tf.nn.dynamic_rnn
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_units)
                lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=0.75)
                value, _ = tf.nn.dynamic_rnn(lstm_cell, data, dtype=tf.float32)

                # The first output of the dynamic RNN function can be thought of as the last hidden state vector
                weight = tf.Variable(tf.truncated_normal([lstm_units, class_num]))
                bias = tf.Variable(tf.constant(0.0, shape=[class_num]))
                value = tf.transpose(value, [1, 0, 2])
                last = tf.gather(value, int(value.get_shape()[0]) - 1)
                prediction = (tf.matmul(last, weight) + bias)

                # Correcting prediction and accuracy metrics to track how the network is doing
                corrected_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.labels, 1))
                self.accuracy = tf.reduce_mean(tf.cast(corrected_prediction, tf.float32))

                # Defining a standard cross entropy loss with a softmax layer put on top of the final prediction values
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=self.labels))
                self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

                self.sess = sess

    # Training Model
    def fit(self, iterations, batch_queue, label_queue):
        # Ensuring it is the same session as the one initiated in the constructor
        with self.sess as sess:
            # Using tensorboard to visualize the loss and accuracy values
            tf.summary.scalar('Loss', self.loss)
            tf.summary.scalar('Accuracy', self.accuracy)
            merged = tf.summary.merge_all()
            log = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
            writer = tf.summary.FileWriter(log, sess.graph)

            sess = tf.InteractiveSession()
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())

            print("entering loop")

            for i in range(iterations):
                # Next Batch of reviews
                next_batch = batch_queue[i]
                next_labels = label_queue[i]
                sess.run(self.optimizer, {self.input_data: next_batch, self.labels: next_labels})

                # Write summary to Tensorboard
                if i % 50 == 0:
                    summary = sess.run(merged, {self.input_data: next_batch, self.labels: next_labels})
                    writer.add_summary(summary, i)

                # Save the network every 10,000 training iterations
                if i % 10000 == 0 and i != 0:
                    save_path = saver.save(sess, "Sentiment Training/pretrained_lstm.ckpt", global_step=i)
                    print("saved to %s" % save_path)
            writer.close()

        self.sess = sess

    # Testing trained model
    def test(self, iterations, batch_queue, label_queue):
        with self.sess as sess:
            for i in range(iterations):
                next_batch = batch_queue[i]
                next_labels = label_queue[i]
                print("Accuracy for this batch:", (sess.run(self.accuracy, {self.input_data: next_batch, self.labels: next_labels})) * 100)
        self.sess = sess

    # Load saved model
    def load_saved_model(self):
        # tf.reset_default_graph()
        with tf.InteractiveSession() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint('models'))
            return sess

    # Evaluates sentiment of input text
    def evaluate(self, text, class_num, batch_size, sequence_length, word_vector_dimension, lstm_units, word_vectors):
        print('inside function')
        with tf.Session() as sess:
            # Creating placeholder input
            input_data = tf.placeholder(tf.int32, [batch_size, sequence_length])

            # Getting word vectors
            data = tf.Variable(tf.zeros([batch_size, sequence_length, word_vector_dimension]), dtype=tf.float32)
            data = tf.nn.embedding_lookup(word_vectors, input_data)

            # Feeding both the LSTM cell and the 3-D tensor full of input data into a function called tf.nn.dynamic_rnn
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_units)
            lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=0.75)
            value, _ = tf.nn.dynamic_rnn(lstm_cell, data, dtype=tf.float32)

            # The first output of the dynamic RNN function can be thought of as the last hidden state vector
            weight = tf.Variable(tf.truncated_normal([lstm_units, class_num]))
            bias = tf.Variable(tf.constant(0.0, shape=[class_num]))
            value = tf.transpose(value, [1, 0, 2])
            last = tf.gather(value, int(value.get_shape()[0]) - 1)
            prediction = (tf.matmul(last, weight) + bias)

            # Load saved model weightings
            sess = tf.InteractiveSession()
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint('models'))

            # Converts text into indexes of words
            input_text = self.get_text_matrix(text)

            # Creates prediction of positive and negative sentiment based on input
            sentiment = sess.run(prediction, {input_data: input_text}).eval()

            # Returning positive sentiment - negative sentiment
            return sentiment[0] - sentiment[1]

    # Cleans input from non-alphanumeric characters and stopwords
    def clean_input(self, string):
        string = string.lower().replace("\n", " ")
        new_input = ""
        for s in string.split(" "):
            if s not in stopwords:
                new_input += (s + " ")
        return re.sub(re.compile("[^A-Za-z0-9]"), "", new_input)

    # Converts text input into an indexed list using ID matrix
    def get_text_matrix(self, text):
        clean_text = self.clean_input(text)
        split_text = clean_text.split()
        text_matrix = np.zeros([len(split_text)], dtype='int32')
        for indexCounter, word in enumerate(split_text):
            try:
                # Add ID of respective word to matrix
                text_matrix[indexCounter] = self.wordsList.index(word)
            except ValueError:
                # ID for unknown words
                text_matrix[0, indexCounter] = 399999
        return text_matrix
