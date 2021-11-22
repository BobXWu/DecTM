import os
import importlib
import numpy as np
import scipy.sparse
from Runner import Runner
from utils import Data
from utils import data_utils
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--en1_units', type=int, default=100)
    parser.add_argument('--en2_units', type=int, default=100)
    parser.add_argument('--num_topic', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.002)
    parser.add_argument('--num_epoch', type=int, default=500)
    parser.add_argument('--keep_prob', type=float, default=1.0)
    parser.add_argument('--num_top_word', type=int, default=15)
    parser.add_argument('--test_index', type=int, default=0)

    args = parser.parse_args()

    return args


def print_topic_words(beta, vocab, num_top_word):
    topic_str_list = []
    for i, topic_dist in enumerate(beta):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(num_top_word + 1):-1]
        topic_str = ' '.join(topic_words)
        topic_str_list.append(topic_str)
        print('Topic {}: {}'.format(i + 1, topic_str))
    return topic_str_list


def main():
    args = parse_args()
    dataset_name = os.path.basename(args.data_dir)
    data = Data.TextData(args.data_dir)
    args.vocab_size = data.train_bow_matrix.shape[1]

    runner = Runner(args)
    beta = runner.train(data.train_bow_matrix)

    ##### save output #####
    output_prefix = '{}/{}_K{}_{}th'.format(args.output_dir, dataset_name, args.num_topic, args.test_index)

    data_utils.make_dir(os.path.dirname(output_prefix))

    topic_output_path = '{}_T{}'.format(output_prefix, args.num_top_word)
    topic_str_list = print_topic_words(beta, data.vocab, num_top_word=args.num_top_word)
    data_utils.save_text(topic_str_list, topic_output_path)

    train_theta = runner.test(data.train_bow_matrix)
    test_theta = runner.test(data.test_bow_matrix)
    scipy.sparse.save_npz('{}_train_theta.npz'.format(output_prefix), scipy.sparse.csr_matrix(train_theta))
    scipy.sparse.save_npz('{}_test_theta.npz'.format(output_prefix), scipy.sparse.csr_matrix(test_theta))

    scipy.sparse.save_npz('{}_beta.npz'.format(output_prefix), scipy.sparse.csr_matrix(beta))


if __name__ == "__main__":
    main()
