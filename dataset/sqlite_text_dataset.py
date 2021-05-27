"""
Pytorch Dataset class to interface with a SQLite DB with added DataLoader.
"""

import sqlite3
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import gensim
from typing import List


class SQLiteTextDataset(Dataset):
    def __init__(self, database_file: str, table: str, vocab_file: str):
        """Initialize a Pytorch Dataset to pull batches from a sqlite database.

        Args:
            db_path: path to .sqlite database file.

        """

        self._db = sqlite3.connect(database_file)
        self._cur = self._db.cursor()
        self._table = table
        self._vocab = gensim.corpora.Dictionary.load(vocab_file)

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

        question_idxs = self._vocab.doc2idx(question_str.split())
        answer_idxs = self._vocab.doc2idx(answer_str.split())

        return {
            'question_str': question_str, 
            'answer_str': answer_str,
            'question_idxs': question_idxs,
            'answer_idxs': answer_idxs
        }

    def _collate_fn(self, batch: List[dict]):
        return {
            'question_str': [x['question_str'] for x in batch], 
            'answer_str': [x['answer_str'] for x in batch],
            'question_idxs': [x['question_idxs'] for x in batch],
            'answer_idxs': [x['answer_idxs'] for x in batch]
        }
    
    def build_dataloader(self, batch_size: int=64, shuffle: bool=True):
        return DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, 
            collate_fn=self._collate_fn)

    def close(self):
        self._cur.close()
        self._db.close()

