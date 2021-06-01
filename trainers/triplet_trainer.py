"""
Trainer class for the Sentence2Vec task.
"""


import time
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn.functional import triplet_margin_with_distance_loss
from torch.nn import PairwiseDistance

from models.cosine_distance import CosineDistance


class TripletTrainer():
    def __init__(
            self, encoder: Module, dataloader: DataLoader):
        """ Initialize a Triplet Net trainer class to train an encoder model on triplets from a dataloader.

        Args:
            encoder:
            dataloader:

        """

        self._encoder = encoder
        self._dataloader = dataloader


    def train(
            self, num_epochs: int=1, lr: float=0.0001, 
            weight_decay=0.01):
        """ Train num_epochs through all batches from the dataloader.
        
        Args:
            num_epochs:
            lr:
            weight_decay:

        """

        # set encoder model to train mode
        self._encoder.train()

        # initialize optimizer
        optimizer = Adam(
            self._encoder.parameters(), lr=lr,
            weight_decay=weight_decay)
        
        print('[INFO]: training...')

        # train through epochs
        for e in range(num_epochs):

            epoch_start = time.time()
            epoch_loss = 0.

            for i, batch in enumerate(self._dataloader):

                anc_batch = batch['anc_idxs']
                pos_batch = batch['pos_idxs']
                neg_batch = batch['neg_idxs']

                anc_enc = self._encoder(anc_batch)
                pos_enc = self._encoder(pos_batch)
                neg_enc = self._encoder(neg_batch)

                loss = triplet_margin_with_distance_loss(
                    anc_enc, pos_enc, neg_enc, 
                    distance_function=CosineDistance(), margin=0.2)

                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_time = time.time() - epoch_start
            avg_epoch_loss = epoch_loss / i

            self._encoder.save('artifacts/models')

            print(f'[INFO]: epoch: {e+1}, loss: {avg_epoch_loss:.4f}, '\
                f'time: {epoch_time:.2f}s')

        print('[INFO]: done')