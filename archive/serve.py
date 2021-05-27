"""
Script to launch a trained S2V model serving API using mlserving.
"""

import yaml
import pandas as pd
import numpy as np
import torch
from mlserving import ServingApp
from mlserving.predictors import RESTPredictor

from model.sentence_2_vec_triplet import Sentence2VecTriplet
from util.text_utils import process_text


class SentenceTopicPredictor(RESTPredictor):
    def __init__(self):
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
            self.vocab_stoi = {
                line.strip('\n').split(' ')[0]:\
                int(line.strip('\n').split(' ')[1])for line in fp}

        # add vocab info to config
        config['vocab_len'] = len(self.vocab_stoi)

        # load clusters to dataframe
        clusters_df = pd.read_csv(clusters_file)

        # get list of cluster labels
        self.cluster_summaries = clusters_df['summary'].tolist()

        # create numpy array of average cluster vectors
        self.cluster_vecs = np.stack(clusters_df['avg_vector'].map(
            lambda x: np.fromstring(x[1:-1], sep=' ')).tolist())

        # load model
        self.model = Sentence2VecTriplet(config)
        self.model.load_state_dict(
            torch.load(model_file, map_location=torch.device('cpu')))

        # clean
        del clusters_df

    def pre_process(self, input_data, req):
        # check if query is a string
        assert isinstance(input_data['query'], str), "input must be string"

        # process query text to tokens
        tokens = process_text(input_data['query'])

        # convert tokens to indices with vocab
        token_idxs = torch.tensor([
            self.vocab_stoi.get(tok, self.vocab_stoi['<unk>'])
            for tok in tokens], dtype=torch.long).unsqueeze(1)

        return token_idxs

    def predict(self, token_idxs, req):
        # compute Sentence2Vec query vector
        query_vec = self.model(token_idxs).detach().cpu().numpy()

        # compute closest avg cluster vector to this query vector
        dists = np.linalg.norm(self.cluster_vecs - query_vec, ord=1, axis=1)
        query_label = self.cluster_summaries[np.argmin(dists)]

        return query_label

    def post_process(self, query_label, req):
        return {'query_topic': query_label}


app = ServingApp()

app.add_inference_handler('/api/v1/predict', SentenceTopicPredictor())


if __name__ == '__main__':
    app.run()
