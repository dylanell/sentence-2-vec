"""
Pytorch Dataset class to interface with a SQLite DB with added DataLoader.
"""


import sqlite3
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import gensim
from typing import List


class QATripletDataset(Dataset):
    def __init__(self, database_file: str, table: str, vocab_file: str):
        """Initialize a Pytorch Dataset to pull batches from a sqlite database.

        Args:
            database_file: Path to sqlite database file.
            table: Name of sqlite database table.
            vovab_file: Path to gensim text dictionary file.

        """

        self._db = sqlite3.connect(database_file)
        self._cur = self._db.cursor()
        self._table = table
        self._vocab = gensim.corpora.Dictionary.load(vocab_file)
        self._pad_idx = self._vocab.doc2idx(['<PAD>'])[0]
        self._len = len(self)

    def __len__(self):
        query = f'SELECT COUNT(*) FROM {self._table}'
        self._cur.execute(query)

        return self._cur.fetchone()[0]

    def __getitem__(self, idx: int):
        query = f"""SELECT proc_question, proc_answer FROM {self._table} 
            WHERE id = {idx}"""
        self._cur.execute(query)
        row = self._cur.fetchone()

        anc_str = row[0]
        pos_str = row[1]

        other_idx = torch.randint(low=0, high=self._len, size=(1,)).item()
        while other_idx == idx:
            other_idx = torch.randint(low=0, high=self._len, size=(1,)).item()

        query = f"""SELECT proc_answer FROM {self._table} 
            WHERE id = {other_idx}"""
        self._cur.execute(query)
        row = self._cur.fetchone()

        neg_str = row[0]

        anc_idxs = torch.LongTensor(self._vocab.doc2idx(
            anc_str.split(), unknown_word_index=self._pad_idx))
        pos_idxs = torch.LongTensor(self._vocab.doc2idx(
            pos_str.split(), unknown_word_index=self._pad_idx))
        neg_idxs = torch.LongTensor(self._vocab.doc2idx(
            neg_str.split(), unknown_word_index=self._pad_idx))

        return {
            'anc_str': anc_str, 
            'pos_str': pos_str,
            'neg_str': neg_str,
            'anc_idxs': anc_idxs,
            'pos_idxs': pos_idxs,
            'neg_idxs': neg_idxs
        }

    def _collate_fn(self, batch: List[dict]):
        anc_str_list = [x['anc_str'] for x in batch]
        pos_str_list = [x['pos_str'] for x in batch]
        neg_str_list = [x['neg_str'] for x in batch]
        anc_idxs_list = [x['anc_idxs'] for x in batch]
        pos_idxs_list = [x['pos_idxs'] for x in batch]
        neg_idxs_list = [x['neg_idxs'] for x in batch]

        return {
            'anc_str': anc_str_list, 
            'pos_str': pos_str_list,
            'neg_str': neg_str_list,
            'anc_idxs': anc_idxs_list,
            'pos_idxs': pos_idxs_list,
            'neg_idxs': neg_idxs_list
        }
    
    def build_dataloader(self, batch_size: int=64, shuffle: bool=True):
        return DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, 
            collate_fn=self._collate_fn)

    def get_vocab(self):
        return self._vocab

    def close(self):
        self._cur.close()
        self._db.close()

