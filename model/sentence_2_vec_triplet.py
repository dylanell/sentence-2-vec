"""
Sentence2Vec model implemented as a torch.nn.Module.
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
        if config['output_activation'] is not None:
            self.out_act = activation[config['output_activation']]
        else:
            self.out_act = torch.nn.Identity()

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

        # if model file provided, load pretrained params
        if config['model_file']:
            self.model.load_state_dict(
                torch.load(config['model_file'], map_location=self.device))

        # send model to found device
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
        z5 = self.out_act(self.linear_layer(z4))

        # normalize outputs
        y = torch.nn.functional.normalize(z5, dim=1)

        return y

    def train_epochs(self, train_iter):
        # define loss function
        loss_fn = torch.nn.TripletMarginLoss(
            margin=self.config['margin'], p=self.config['p_norm'])

        # initialize optimizer
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'])

        # initialize tensorboard writer
        writer = SummaryWriter('{}runs/{}/'.format(
            self.config['output_directory'], self.config['model_name']))

        # push output to [0, 1]
        triplet_labels = torch.tensor([[0, 1]], requires_grad=False).repeat(
            self.config['batch_size'], 1).to(self.device)

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

                # triplet loss using margin from Pytorch
                loss = loss_fn(anchor_batch, pos_batch, neg_batch)

                # triplet loss from original paper (with variable p-norm)
                #d_pos = torch.nn.functional.pairwise_distance(
                #    anchor_batch, pos_batch, p=self.config['p_norm'])
                #d_neg = torch.nn.functional.pairwise_distance(
                #    anchor_batch, neg_batch, p=self.config['p_norm'])
                #d_pos_neg = torch.cat(
                #    [d_pos.unsqueeze(1), d_neg.unsqueeze(1)], dim=1)
                #out = torch.nn.functional.softmax(d_pos_neg, dim=1)
                #loss = torch.mean(torch.sqrt(torch.sum(
                #    torch.pow(out - triplet_labels[:d_pos.shape[0]], 2))))

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
                self.config['output_directory'],
                self.config['model_name'], filename), 'w+')
        answer_tok_fp = open(
            '{}{}_{}_answer_tok.txt'.format(
                self.config['output_directory'],
                self.config['model_name'], filename), 'w+')
        question_vec_fp = open(
            '{}{}_{}_question_vec.txt'.format(
                self.config['output_directory'],
                self.config['model_name'], filename), 'w+')
        answer_vec_fp = open(
            '{}{}_{}_answer_vec.txt'.format(
                self.config['output_directory'],
                self.config['model_name'], filename), 'w+')

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
