"""
Pytorch utility functions.
"""

import torch
import torchtext
import random


def build_processed_qa_dataloaders(
        data_file, split=0.7, batch_size=32, random_seed=42,
        wordvec_file=None, wordvec_dim=300, cache_dir='/tmp/'):
    """
    Builds a training and validation dataloader for QA pairs pre-processed
    text data ready to be tokenized by whitespace.
    :param data_file: path to qa datafile ([question col, answer col] format)
    :param split: ratio of training to validation data (default=0.7)
    :param batch_size: size of batch returned by dataloaders (default=32)
    :return: training dataloader, validation dataloader
    """

    # seed rng
    random.seed(random_seed)

    # create a field for processing texts
    texts_field = torchtext.data.Field()

    # construct field objects for each column of examples
    fields = [('question', texts_field), ('answer', texts_field)]

    # construct dataset object
    dataset = torchtext.data.TabularDataset(
        path=data_file,
        format='CSV',
        fields=fields,
        skip_header=True
    )

    # split dataset into training and validation splits
    train_ds, val_ds = dataset.split(split_ratio=split)
    #train_ds = dataset

    # build vocabulary for the TEXTS field using only words from the training
    # split (used for both questions and answers)
    # NOTE: this is REQUIRED before creating an iterator since an iterator uses
    # the vocab object to create vectors of word indices
    if wordvec_file is not None:
        # build vocab from pretrained word vectors
        texts_field.build_vocab(
            train_ds,
            vectors=torchtext.vocab.Vectors(wordvec_file, cache_dir),
            vectors_cache=cache_dir)
    else:
        # build custom vocab from training data
        texts_field.build_vocab(train_ds)

    # construct training dataset iterator
    train_iter = torchtext.data.Iterator(
        train_ds,
        batch_size=batch_size,
        shuffle=True
    )

    # construct validation dataset iterator
    val_iter = torchtext.data.Iterator(
        val_ds,
        batch_size=batch_size,
        shuffle=False
    )

    # get vocab
    vocab = texts_field.vocab

    # add wordvecs to config data
    if wordvec_file is not None:
        # wordvecs loaded from wordvec file
        wordvecs = texts_field.vocab.vectors
    else:
        # wordvecs constructed from train_ds vocab and normal distribution
        wordvecs = torch.normal(
            mean=0, std=1,
            size=(len(vocab), wordvec_dim))

    return train_iter, val_iter, vocab, wordvecs

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
