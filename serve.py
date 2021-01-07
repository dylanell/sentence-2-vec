"""
Script to launch a trained S2V model serving API.
"""

import yaml
import pandas as pd
import numpy as np
import torch

from model.sentence_2_vec_triplet import Sentence2VecTriplet
from util.text_utils import process_text


def main():
    # parse configuration file
    with open('config.yaml', 'r') as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)

    # artifact file paths
    model_file = '{}{}_model.pt'.format(
        config['output_directory'], config['model_name'])
    vocab_file = '{}{}_vocab.txt'.format(
        config['output_directory'], config['model_name'])
    clusters_file = '{}{}_clusters.csv'.format(
        config['output_directory'], config['model_name'])

    # load vocab to dictionary
    with open(vocab_file, 'r') as fp:
        vocab_stoi = {
            line.strip('\n').split(' ')[0]: int(line.strip('\n').split(' ')[1])
            for line in fp}

    # add vocab info to config
    config['vocab_len'] = len(vocab_stoi)

    # load clusters to dataframe
    clusters_df = pd.read_csv(clusters_file)

    # create numpy array of average cluster vectors
    avg_vecs = np.stack(clusters_df['avg_vector'].map(
        lambda x: np.fromstring(x[1:-1], sep=' ')).tolist())

    # load model
    model = Sentence2VecTriplet(config)
    model.load_state_dict(
        torch.load(model_file, map_location=torch.device('cpu')))

    #query = 'does a diabetic seizure cause you to become physically ill?'
    #query = 'What\'s the normal blood sugar'
    #query = 'Can a Diabetic eat sweet corn?'
    #query = 'What causes diabetes?'
    #query = 'What happens to you if you have diabetes?'
    #query = 'how to decrease blood glucose levels'
    #query = 'Info on diabetes cook books?'
    #query = 'How do i measure my blood sugar?'
    #query = 'Can I have alcohol if I have diabetes?'
    #query = 'How is insulin produced in the body?'
    query = 'What is diabetic shock syndrome?'

    # process query text to tokens
    tokens = process_text(query)

    # convert tokens to indices with vocab
    token_idx = torch.tensor([
        vocab_stoi.get(tok, vocab_stoi['<unk>']) for tok in tokens],
        dtype=torch.long).unsqueeze(1)

    # compute query vector using trained sentence2vec model
    query_vec = model(token_idx).detach().cpu().numpy()

    # compute closest avg cluster vector to this query vector
    dists = np.linalg.norm(avg_vecs - query_vec, ord=1, axis=1)
    row = clusters_df.iloc[np.argmin(dists), :][['label', 'summary']]
    print('[INPUT]: \'{}\'\n[LABEL]: \'{}\''.format(query, row['summary']))


if __name__ == '__main__':
    main()
