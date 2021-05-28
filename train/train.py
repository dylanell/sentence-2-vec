"""
Main model training script.
"""


import sys
sys.path.append('..')

import yaml 

from dataset.sqlite_text_dataset import SQLiteTextDataset

from model.max_over_time_cnn import MaxOverTimeCNN


def main():
    with open('train_conf.yml', 'r') as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)
    
    dataset = SQLiteTextDataset(
        config['database_file'], config['database_table'], 
        config['vocab_file'])

    dataloader = dataset.build_dataloader(batch_size=64, shuffle=True)

    model = MaxOverTimeCNN(
        len(dataset.get_vocab()), 64, 32, acceleration=True)

    batch = next(iter(dataloader))

    question_idxs_batch = batch['question_idxs']

    out = model(batch['question_idxs'])

    print(out.shape)

    dataset.close()


if __name__ == '__main__':
    main()