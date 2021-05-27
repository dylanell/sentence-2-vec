"""
Main model training script.
"""

import yaml
from torch.utils.data import DataLoader 

from dataset.sqlite_text_dataset import SQLiteTextDataset


def main():
    with open('.train_conf.yml', 'r') as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)
    
    dataset = SQLiteTextDataset(
        config['database_file'], config['database_table'], 
        config['vocab_file'])

    dataloader = dataset.build_dataloader(batch_size=64, shuffle=True)

    batch = next(iter(dataloader))

    dataset.close()


if __name__ == '__main__':
    main()