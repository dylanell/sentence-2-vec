"""
Script to train sentence2vec model.
"""

import yaml
import torch

from util.pytorch_utils import build_processed_qa_dataloaders
from model.sentence_2_vec_triplet import Sentence2VecTriplet


def main():
    # parse configuration file
    with open('config.yaml', 'r') as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)

    # path to processed qa data file
    data_file = '{}qa_pairs_processed.csv'.format(config['dataset_directory'])

    # build data iterators and vocabulary object
    train_iter, val_iter, vocab, wordvecs = build_processed_qa_dataloaders(
        data_file, batch_size=config['batch_size'],
        wordvec_file=config['wordvec_file'],
        wordvec_dim=config['wordvec_dim'],
        cache_dir=config['output_directory'])

    # save vocab
    vocab_file = '{}{}_vocab.txt'.format(
        config['output_directory'], config['model_name'])
    with open(vocab_file, 'w') as fp:
        for word, index in dict(vocab.stoi).items():
            fp.write('{},{}\n'.format(word, index))

    # store wordvecs in config for model
    config['wordvecs'] = wordvecs

    # initialize model
    model = Sentence2VecTriplet(config)

    # train model if we have trainable parameters
    if len(list(model.parameters())) > 0:
        model.train_epochs(train_iter)

    # write trained word vectors to file
    torch.save(model, '{}{}_model.pt'.format(
        config['output_directory'], config['model_name']))


if __name__ == '__main__':
    main()
