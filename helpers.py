import numpy as np
import collections
import os
import tensorflow as tf
from itertools import groupby

all_words = []

def batch(inputs, max_sequence_length=None):
    """
    Args:
        inputs:
            list of sentences (integer lists)
        max_sequence_length:
            integer specifying how large should `max_time` dimension be.
            If None, maximum sequence length would be used
    
    Outputs:
        inputs_time_major:
            input sentences transformed into time-major matrix 
            (shape [max_time, batch_size]) padded with 0s
        sequence_lengths:
            batch-sized list of integers specifying amount of active 
            time steps in each input sequence
    """
    
    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)
    
    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)
    
    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32) # == PAD
    
    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i, j] = element

    # [batch_size, max_time] -> [max_time, batch_size]
    inputs_time_major = inputs_batch_major.swapaxes(0, 1)

    return inputs_time_major, sequence_lengths

def _read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        w = f.read().decode("utf-8").replace("\n", " <eos> ").split(" ")
    return w

def _build_vocab(filename):
    data = _read_words(filename)
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    global all_words
    all_words = words
    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id


def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]+[word_to_id["<eos>"]]


def load_y_labels(data_path="data/"):

    train_path = os.path.join(data_path, "sample_train1.txt")
    valid_path = os.path.join(data_path, "sample_val1.txt")
    test_path = os.path.join(data_path, "sample_test1.txt")

    word_to_id = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    train_data_batch_ = [list(group) for k, group in groupby(train_data, lambda x: x == word_to_id['<eos>']) if not k]
    valid_data_batch_ = [list(group) for k, group in groupby(train_data, lambda x: x == word_to_id['<eos>']) if not k]
    test_data_batch_ = [list(group) for k, group in groupby(train_data, lambda x: x == word_to_id['<eos>']) if not k] 
    return train_data_batch_, valid_data_batch_, test_data_batch_, vocabulary
    
def read_and_load_data(batch, batch_size):
    l = len(batch)
    for ndx in range(0, l, batch_size):
        yield batch[ndx:min(ndx + batch_size, l)]    
    
def random_sequences(length_from, length_to,
                     vocab_lower, vocab_upper,
                     batch_size):
    """ Generates batches of random integer sequences,
        sequence length in [length_from, length_to],
        vocabulary in [vocab_lower, vocab_upper]
    """
    if length_from > length_to:
            raise ValueError('length_from > length_to')

    def random_length():
        if length_from == length_to:
            return length_from
        return np.random.randint(length_from, length_to + 1)
    
    while True:
        yield [
            np.random.randint(low=vocab_lower,
                              high=vocab_upper,
                              size=random_length()).tolist()
            for _ in range(batch_size)
        ]