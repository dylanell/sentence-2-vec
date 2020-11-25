"""
Script to train sentence2vec model.
"""

import yaml

from util.pytorch_utils import build_processed_qa_dataloaders
from model.sentence_2_vec_triplet import Sentence2VecTriplet


def main():
    # parse configuration file
    with open('config.yaml', 'r') as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)

    # path to processed qa data file
    data_file = '{}qa_pairs_processed.csv'.format(config['dataset_directory'])

    # build data iterators and vocabulary object
    train_iter, val_iter, vocab = build_processed_qa_dataloaders(
        data_file, batch_size=config['batch_size'])

    # add to config data
    config['vocab'] = vocab

    # initialize model
    if config['model_type'] == 'triplet':
        model = Sentence2VecTriplet(config)
    else:
        print('[ERROR]: unknown model type \'{}\''.format(
            config['model_type']))
        exit()

    # train model
    model.train_epochs(train_iter)

    # save learned sentence vectors for training and validation splits
    model.generate_sentence_embeddings(train_iter, 'train_vecs')
    model.generate_sentence_embeddings(val_iter, 'val_vecs')


if __name__ == '__main__':
    main()
