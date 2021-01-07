"""
Sentence2Vec model using triplet margin loss implemented as a torch.nn.Module.
"""

import time
import torch
from torch.utils.tensorboard import SummaryWriter


class Sentence2VecTriplet(torch.nn.Module):
    def __init__(self, config):
        super(Sentence2VecTriplet, self).__init__()
        # try to get a gpu otherwise use cpu
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        print('[INFO]: using {} device'.format(self.device))

        # construct embedding layer
        self.embedding = torch.nn.Embedding(
            config['vocab_len'], config['wordvec_dim'])

        # compute self attention on word embeddings with transformer encoder
        self.transformer_layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(
                d_model=config['wordvec_dim'],
                nhead=config['number_attention_heads'],
                activation='relu', dropout=0.0)
            for n in range(config['number_transformers'])])

        # linear layer to produce output dimension
        self.linear_layer = torch.nn.Linear(
            config['wordvec_dim'], config['output_dim'])

        # initialize transformer activation function
        if config['activation'] == 'relu':
            self.act_fn = torch.nn.ReLU()
        elif config['activation'] == 'leaky_relu':
            self.act_fn = torch.nn.LeakyReLU()
        elif config['activation'] == 'tanh':
            self.act_fn = torch.nn.Tanh()
        elif config['activation'] == 'sigmoid':
            self.act_fn = torch.nn.Sigmoid()
        elif config['activation'] == 'identity':
            self.act_fn = torch.nn.Identity()
        else:
            print('[INFO]: unsupported activation \'{}\''.format(
                config['activation']))
            exit()

        # initialize distance function
        if config['distance_metric'] == 'l2':
            self.distance_metric_fn = torch.nn.PairwiseDistance(p=2.0)
        elif config['distance_metric'] == 'l1':
            self.distance_metric_fn = torch.nn.PairwiseDistance(p=1.0)
        elif config['distance_metric'] == 'cosine':
            self.distance_metric_fn = CosineDistance()
        else:
            print('[INFO]: unsupported metric \'{}\''.format(
                config['distance_metric']))
            exit()

        # send model to found device
        self.to(self.device)

        self.config = config

    def forward(self, x):
        # compute embeddings from indexed inputs
        z = self.embedding(x)

        # feed embeddings to multiple ocnsecutive transformers
        for transformer in self.transformer_layers:
            z = self.act_fn(transformer(z))

        # sum word vectors along sentence length dimension
        z = torch.sum(z, dim=0)

        # feed pooled transformer outputs to final linear layer with activation
        z = self.act_fn(self.linear_layer(z))

        # anormalize outputs
        z = torch.nn.functional.normalize(z, dim=1)

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
                    # loss from: FaceNet: A Unified Embedding for Face
                    # Recognition and Clustering

                    # 'Margin Ranking Loss'

                    # triplet loss using margin from Pytorch
                    loss = \
                        torch.nn.functional.triplet_margin_with_distance_loss(
                            anchor_batch, pos_batch, neg_batch,
                            distance_function=self.distance_metric_fn,
                            margin=self.config['margin'])
                elif self.config['loss'] == 'ratio':
                    # loss from: 'Learning Thematic Similarity Metric Using
                    # Triplet Networks' and 'DEEP METRIC LEARNING USING TRIPLET
                    # NETWORK'

                    # 'Ratio Loss'

                    # distance between anchor and positive batch
                    d_pos = self.distance_metric_fn(anchor_batch, pos_batch)

                    # distance between anchor and negative batch
                    d_neg = self.distance_metric_fn(anchor_batch, neg_batch)

                    # softmax on (d_pos, d_neg)
                    out = torch.nn.functional.softmax(
                        torch.cat([d_pos.unsqueeze(1), d_neg.unsqueeze(1)],
                        dim=1), dim=1)

                    # ratio loss
                    loss = torch.mean(
                        torch.abs(out[:, 0]) + torch.abs(1 - out[:, 1]))
                else:
                    print('[INFO]: unsupported loss\'{}\''.format(
                        self.config['loss']))
                    exit()

                epoch_loss += loss.item()

                # update parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # compute epoch time
            epoch_time = time.time() - epoch_start

            # report epoch metrics
            avg_epoch_loss = epoch_loss / i

            # add metrics to tensorboard
            writer.add_scalar('Loss/Train', avg_epoch_loss, e + 1)

            # print epoch metrics
            template = '[INFO]: Epoch {}, Epoch Time {:.2f}s, ' \
                       'Train Loss: {:.4f}'
            print(template.format(e + 1, epoch_time, avg_epoch_loss))

            # save checkpoint
            torch.save(self.state_dict(), '{}{}_model.pt'.format(
                self.config['output_directory'], self.config['model_name']))


class CosineDistance(torch.nn.Module):
    """
    Computes batch-wise cosine distance between two batches of row vectors.
    :param x1: batch of row vectors.
    :param x2: batch of row vectors.
    """
    def __init__(self):
        super(CosineDistance, self).__init__()

    def forward(self, x1, x2):
        return 1.0 - torch.nn.functional.cosine_similarity(x1, x2, dim=1)
