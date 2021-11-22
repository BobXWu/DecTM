import os
import scipy.sparse
from utils import data_utils


class TextData(object):
    def __init__(self, data_dir):
        self.read_data(data_dir)

    def read_data(self, data_dir):
        self.train_texts = data_utils.read_text(os.path.join(data_dir, 'train_texts.txt'))
        self.test_texts = data_utils.read_text(os.path.join(data_dir, 'test_texts.txt'))
        self.vocab = data_utils.read_text(os.path.join(data_dir, 'vocab.txt'))

        self.train_bow_matrix = scipy.sparse.load_npz(os.path.join(data_dir, 'train_bow_matrix.npz')).toarray()
        self.test_bow_matrix = scipy.sparse.load_npz(os.path.join(data_dir, 'test_bow_matrix.npz')).toarray()

        self.train_labels = data_utils.read_text(os.path.join(data_dir, 'train_labels.txt'))
        self.test_labels = data_utils.read_text(os.path.join(data_dir, 'test_labels.txt'))
