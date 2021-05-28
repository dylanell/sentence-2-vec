"""
Pytorch Dataset class to interface with a SQLite DB with added DataLoader.
"""

import sqlite3
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import gensim
from typing import List


class SQLiteTextDataset(Dataset):
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

    def __len__(self):
        query = f'SELECT COUNT(*) FROM {self._table}'
        self._cur.execute(query)

        return self._cur.fetchone()[0]

    def __getitem__(self, idx: int):
        query = f"""SELECT proc_question, proc_answer FROM {self._table} 
            WHERE id = {idx}"""
        self._cur.execute(query)

        row = self._cur.fetchone()

        question_str = row[0]
        answer_str = row[1]

        question_idxs = torch.tensor(self._vocab.doc2idx(
            question_str.split(), unknown_word_index=self._pad_idx))
        answer_idxs = torch.tensor(self._vocab.doc2idx(
            answer_str.split(), unknown_word_index=self._pad_idx))

        return {
            'question_str': question_str, 
            'answer_str': answer_str,
            'question_idxs': question_idxs,
            'answer_idxs': answer_idxs
        }

    def _collate_fn(self, batch: List[dict]):
        question_str_list = [x['question_str'] for x in batch]
        answer_str_list = [x['answer_str'] for x in batch]
        question_idxs_list = [x['question_idxs'] for x in batch]
        answer_idxs_list = [x['answer_idxs'] for x in batch]

        return {
            'question_str': question_str_list, 
            'answer_str': answer_str_list,
            'question_idxs': question_idxs_list,
            'answer_idxs': answer_idxs_list
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

