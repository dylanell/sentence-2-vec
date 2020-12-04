"""
Sentence2Vec model using triplet margin loss implemented as a torch.nn.Module.
"""

# TODO: build_vocab is nondeterministic, therefore when trying to load a
#  pretrained model there can be a vocab size mismatch or indices will be
#  different than during training. FIX: Figure out if vocab can be generated
#  deterministically or provide manually constructed vocab to Field object.
#  QUICK FIX: For now, save trained word embeddings and sentence embeddings
#  for training and validation sets to use offline.

# TODO: Writing sentence vectors and other data to a dataframe is too memory
#  intensive. Instead, append data to csv/text files on disk.

import time
import torch
from torch.utils.tensorboard import SummaryWriter

import util.pytorch_utils as pu


# torch activation functions
activations = {
    'relu': torch.nn.ReLU(),
    'leaky_relu': torch.nn.LeakyReLU(),
    'tanh': torch.nn.Tanh(),
    'sigmoid': torch.nn.Sigmoid(),
    'identity': torch.nn.Identity()
}

# distance metric callables
metrics = {
    'l2': lambda v, k: torch.norm(v - k, dim=1, p=2.0),
    'l1': lambda v, k: torch.norm(v - k, dim=1, p=1.0),
    'cosine': lambda v, k: 1.0 - torch.nn.functional.cosine_similarity(
        v, k, dim=1),
    'hyperbolic_l2': lambda v, k: pu.hyperbolic_distance(v, k, p=2),
    'hyperbolic_l1': lambda v, k: pu.hyperbolic_distance(v, k, p=1),
    'sub_sample_10_l2': lambda v, k: pu.sub_sampled_distance(v, k, n=10, p=2),
    'sub_sample_20_l2': lambda v, k: pu.sub_sampled_distance(v, k, n=20, p=2),
    'sub_sample_10_l1': lambda v, k: pu.sub_sampled_distance(v, k, n=10, p=1),
    'sub_sample_20_l1': lambda v, k: pu.sub_sampled_distance(v, k, n=20, p=1)
}


class Sentence2VecTriplet(torch.nn.Module):
    def __init__(self, config):
        super(Sentence2VecTriplet, self).__init__()
        # convert index sentences to word embedding matrices
        self.embedding_layer = torch.nn.Embedding(
            len(config['vocab']), config['embed_dimensionality'])

        # compute self attention on word embeddings with transformer encoder
        self.transformer_layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(
                d_model=config['embed_dimensionality'], nhead=8,
                activation='relu', dropout=0.0)
            for n in range(config['number_transformers'])])

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        print('[INFO]: using {} device'.format(self.device))

        # if model file provided, load pretrained params
        if config['model_file']:
            self.model.load_state_dict(
                torch.load(config['model_file'], map_location=self.device))

        # send model to found device
        self.to(self.device)

        self.config = config

    def forward(self, x):
        # compute embeddings from indexed inputs
        z = self.embedding_layer(x)

        # feed embeddings to multiple ocnsecutive transformers
        for transformer in self.transformer_layers:
            z = activations[self.config['transformer_activation']](
                transformer(z))

        # sum word vectors along sentence length dimension
        z = torch.sum(z, dim=0)

        # apply output process function
        if self.config['output_process'] == 'normalize':
            # normalize outputs (all lie on unit ball)
            z = torch.nn.functional.normalize(z, dim=1)
            # constrain outputs to all lie within unit ball
        elif self.config['output_process'] == 'open_unit_ball_constrain':
            # norm outputs
            z_norms = torch.norm(z, p=2.0, dim=1)

            # constrain any outputs outside unit ball to just within
            z[z_norms > 1] = z[z_norms > 1] / (
                (1+1e-3) * z_norms[z_norms > 1]).unsqueeze(-1)

            # constrain all outputs inside unit ball relative to largest norm
            #z = z / ((1+1e-3) * torch.max(torch.norm(z, p=2.0, dim=1)))

        return z

    def train_epochs(self, train_iter):
        # initialize optimizer
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'])

        # initialize tensorboard writer
        writer = SummaryWriter('{}runs/{}/'.format(
            self.config['output_directory'], self.config['model_name']))

        print('[INFO]: training...')

        for e in range(self.config['number_epochs']):
            # get epoch start time
            epoch_start = time.time()

            # reset loss accumulator
            epoch_loss = 0.

            for i, data in enumerate(train_iter):
                # parse batch
                question_batch = data.question.to(self.device)
                answer_batch = data.answer.to(self.device)

                # get batch size dynamically to account for end of epoch
                batch_size = question_batch.shape[1]

                # feed question and answer through model
                enc_question_batch = self.forward(question_batch)
                enc_answer_batch = self.forward(answer_batch)

                # question, answer pairs correspond to (anchor, positive)
                # triplet batches
                anchor_batch = enc_question_batch
                pos_batch = enc_answer_batch

                # construct 'negative' triplet labels by randomly sampling
                # incorrect answer
                # NOTE: right now, this actually leaves one case where the pos
                # and neg samples are identical
                neg_batch = enc_answer_batch[
                    torch.randperm(batch_size, dtype=torch.long)]

                if self.config['loss'] == 'margin':
                    # triplet loss using margin from Pytorch
                    loss = \
                        torch.nn.functional.triplet_margin_with_distance_loss(
                            anchor_batch, pos_batch, neg_batch,
                            distance_function=metrics[
                                self.config['distance_metric']],
                            margin=self.config['margin'])
                elif self.config['loss'] == 'softmax':
                    # distance between anchor and positive batch
                    d_pos = metrics[self.config['distance_metric']](
                        anchor_batch, pos_batch)

                    # distance between anchor and negative batch
                    d_neg = metrics[self.config['distance_metric']](
                        anchor_batch, neg_batch)

                    # softmax on (d_pos, d_neg)
                    out = torch.nn.functional.softmax(
                        torch.cat([d_pos.unsqueeze(1), d_neg.unsqueeze(1)],
                        dim=1), dim=1)

                    # triplet loss from softmax
                    loss = torch.mean(
                        torch.abs(out[:, 0]) + torch.abs(1 - out[:, 1]))

                epoch_loss += loss.item()

                # update parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # compute epoch time
            epoch_time = time.time() - epoch_start

            # save model
            torch.save(self.state_dict(), '{}{}.pt'.format(
                self.config['output_directory'], self.config['model_name']))

            # report epoch metrics
            avg_epoch_loss = epoch_loss / i

            # add metrics to tensorboard
            writer.add_scalar('Loss/Train', avg_epoch_loss, e + 1)

            # print epoch metrics
            template = '[INFO]: Epoch {}, Epoch Time {:.2f}s, ' \
                       'Train Loss: {:.2f}'
            print(template.format(e + 1, epoch_time, avg_epoch_loss))

    def generate_sentence_embeddings(self, data_iter, filename):
        # open file objects in append mode
        question_tok_fp = open(
            '{}{}_{}_question_tok.txt'.format(
                self.config['output_directory'], self.config['model_name'],
                filename), 'w+')
        answer_tok_fp = open(
            '{}{}_{}_answer_tok.txt'.format(
                self.config['output_directory'], self.config['model_name'],
                filename), 'w+')
        question_vec_fp = open(
            '{}{}_{}_question_vec.txt'.format(
                self.config['output_directory'], self.config['model_name'],
                filename), 'w+')
        answer_vec_fp = open(
            '{}{}_{}_answer_vec.txt'.format(
                self.config['output_directory'], self.config['model_name'],
                filename), 'w+')

        print('[INFO]: writing \'{}\' sentence embeddings...'
              .format(filename))

        for i, data in enumerate(data_iter):
            # parse batch
            question_idx = data.question.to(self.device)
            answer_idx = data.answer.to(self.device)

            # feed question and answer through model
            question_vec = self.forward(question_idx)
            answer_vec = self.forward(answer_idx)

            # convert question indices to list of word tokens
            question_tok = [
                ' '.join([self.config['vocab'].itos[idx] for idx in s
                          if idx != 1]) for s in question_idx.T.tolist()]

            # convert answer indices to list of word tokens
            answer_tok = [
                ' '.join([self.config['vocab'].itos[idx] for idx in s
                          if idx != 1]) for s in answer_idx.T.tolist()]

            # write question token data for this batch
            for tok in question_tok:
                question_tok_fp.write('{}\n'.format(tok))

            # write answer token data for this batch
            for tok in answer_tok:
                answer_tok_fp.write('{}\n'.format(tok))

            # write question vec data
            for vec in question_vec:
                line = ','.join([str(n) for n in vec.tolist()])
                question_vec_fp.write('{}\n'.format(line))

            # write answer vec data
            for vec in answer_vec:
                line = ','.join([str(n) for n in vec.tolist()])
                answer_vec_fp.write('{}\n'.format(line))

        # close file objects
        question_tok_fp.close()
        answer_tok_fp.close()
        question_vec_fp.close()
        answer_vec_fp.close()
