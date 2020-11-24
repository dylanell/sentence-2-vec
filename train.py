"""
Script to train sentence2vec model.
"""

import yaml


def main():
    # parse configuration file
    with open('config.yaml', 'r') as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)
        

if __name__ == '__main__':
    main()
