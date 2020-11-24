"""
Script to train sentence2vec model.
"""

import yaml

from util.pytorch_utils import build_processed_qa_dataloaders


def main():
    # parse configuration file
    with open('config.yaml', 'r') as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)

    # path to processed qa data file
    data_file = '{}qa_pairs_processed.csv'.format(config['dataset_directory'])

    train_iter, val_iter = build_processed_qa_dataloaders(data_file)
    

if __name__ == '__main__':
    main()
