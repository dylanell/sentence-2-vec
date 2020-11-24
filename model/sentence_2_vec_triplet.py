"""
Sentence2Vec model implemented as a torch.nn.Module.
"""

import time
import torch
from torch.utils.tensorboard import SummaryWriter

# torch activation functions
activation = {
    'relu': torch.nn.ReLU(),
    'leaky_relu': torch.nn.LeakyReLU(),
    'tanh': torch.nn.Tanh(),
    'sigmoid': torch.nn.Sigmoid()
}


class Sentence2VecTriplet(torch.nn.Module):
    def __init__(self, config):
        super(Sentence2VecTriplet, self).__init__()
        # activation functions
        self.out_act = activation[config['output_activation']]

        # convert index sentences to word embedding matrices
        self.embedding_layer = torch.nn.Embedding(
            len(config['vocab']), config['embed_dimensionality'])

        # compute self attention on word embeddings with transformer encoder
        self.transformer_layer = torch.nn.TransformerEncoderLayer(
            d_model=config['embed_dimensionality'], nhead=8)

        # n-gram CNN processing 3 words at a time
        self.textcnn_3 = torch.nn.Conv2d(
            in_channels=1, out_channels=config['conv_output_channels'],
            kernel_size=(3, config['embed_dimensionality']))

        # n-gram processing 4 words at a time
        self.textcnn_4 = torch.nn.Conv2d(
            in_channels=1, out_channels=config['conv_output_channels'],
            kernel_size=(4, config['embed_dimensionality']))

        # n-gram CNN processing 45words at a time
        self.textcnn_5 = torch.nn.Conv2d(
            in_channels=1, out_channels=config['conv_output_channels'],
            kernel_size=(5, config['embed_dimensionality']))

        # linear layer to control output representation dimensionality
        self.linear_layer = torch.nn.Linear(3 * config['conv_output_channels'],
                                            config['output_dimensionality'])

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        print('[INFO]: using {} device'.format(self.device))
        self.to(self.device)

        self.config = config

    def forward(self, x):
        # compute embeddings from indexed inputs
        z1 = self.embedding_layer(x)

        # feed embeddings to transformer encoder
        z2 = self.transformer_layer(z1)

        # bring batch dim to front and add dummy channel dim of 1
        z3 = torch.unsqueeze(torch.transpose(z2, 1, 0), 1)

        # feed to text cnn with relu activation
        z4_3 = torch.relu(self.textcnn_3(z3))
        z4_4 = torch.relu(self.textcnn_4(z3))
        z4_5 = torch.relu(self.textcnn_5(z3))

        # average pooling over time dim and reshape
        z4_3_pool = torch.squeeze(torch.mean(z4_3, dim=2))
        z4_4_pool = torch.squeeze(torch.mean(z4_4, dim=2))
        z4_5_pool = torch.squeeze(torch.mean(z4_5, dim=2))

        # concatenate to vectorize channel outputs for each n-gram convolution
        z4 = torch.cat([z4_3_pool, z4_4_pool, z4_5_pool], dim=1)

        # transform to output dimensionality
        z5 = self.linear_layer(z4)

        # add final activation
        if self.out_act is not None:
            y = self.out_act(z5)
        else:
            y = z5

        return y

    def train_epochs(self, train_iter):
        # define loss function
        loss_fn = torch.nn.TripletMarginLoss()

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

                # compute triplet loss
                loss = loss_fn(anchor_batch, pos_batch, neg_batch)

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
            print('[INFO]: epoch: {}, loss: {}'.format(e + 1, avg_epoch_loss))

            # add metrics to tensorboard
            writer.add_scalar('Loss/Train', avg_epoch_loss, e+1)

            # print epoch metrics
            template = '[INFO]: Epoch {}, Epoch Time {:.2f}s, ' \
                       'Train Loss: {:.2f}'
            print(template.format(e+1, epoch_time, avg_epoch_loss))
