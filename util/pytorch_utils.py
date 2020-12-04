"""
Pytorch utility functions.
"""

import torch
import torchtext


def build_processed_qa_dataloaders(data_file, split=0.7, batch_size=32):
    """
    Builds a training and validation dataloader for QA pairs pre-processed
    text data ready to be tokenized by whitespace.
    :param data_file: path to qa datafile ([question col, answer col] format)
    :param split: ratio of training to validation data (default=0.7)
    :param batch_size: size of batch returned by dataloaders (default=32)
    :return: training dataloader, validation dataloader
    """

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

    # build vocabulary for the TEXTS field using only words from the training
    # split (used for both questions and answers)
    # NOTE: this is REQUIRED before creating an iterator since an iterator uses
    # the vocab object to create vectors of word indices
    texts_field.build_vocab(train_ds)
    vocab = texts_field.vocab

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

    return train_iter, val_iter, vocab


def hyperbolic_distance(v, k, p=2.0):
    """
    Computes row-wise hyperbolic distance between 2d batch arrays v and k. Row vectors of v and k must be constrained to the open unit ball (lie just
    within) and v and k must be the same size.
    :param v: batch array or row vectors
    :param k: batch array of row vectors
    :param euclidean_p: Lp norm value for computing euclidean distance
    """

    # compute euclidean distance terms
    sq_euc_dists = torch.norm(v - k, dim=1, p=p)**2
    sq_v_norms = torch.norm(v, dim=1, p=p)**2
    sq_k_norms = torch.norm(k, dim=1, p=p)**2

    # compute argumebt for acosh function
    acosh_arg = 1 + (2 * sq_euc_dists / ((1 - sq_v_norms) * (1 - sq_k_norms)))

    # compute hyperbolic distance
    hyper_dist = torch.acosh(acosh_arg)

    return hyper_dist


def sub_sampled_distance(v, k, n=10, p=2.0):
    """
    Computes row-wise sub-sampled euclidean distance between 2d batch arrays v
    and k. The same indices are uniformly sampled and selected from both v and
    k.
    :param v: batch array or row vectors
    :param k: batch array of row vectors
    :param n: number of indices to sample from both v and k
    :param p: Lp norm value for computing euclidean distance
    """

    # sample a set of random indices
    idxs = torch.randperm(v.shape[1])[:n]

    # compute euclidean distance between sub-sampled v and k
    sub_dist = torch.norm(v[:, idxs] - k[:, idxs], dim=1, p=p)

    return sub_dist


def open_unit_ball_constrain(z):
    # norm outputs
    z_norms = torch.norm(z, p=2.0, dim=1)

    # constrain any outputs outside unit ball to just within
    #z[z_norms > 1] = z[z_norms > 1] / (
    #    (1+1e-3) * z_norms[z_norms > 1]).unsqueeze(-1)

    # constrain all outputs inside unit ball relative to largest norm
    z = z / ((1+1e-3) * torch.max(torch.norm(z, p=2.0, dim=1)))

    return z
