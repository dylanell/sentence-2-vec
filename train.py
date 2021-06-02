"""
Main model training script.
"""


import yaml 

from datasets.qa_triplet_dataset import QATripletDataset
from models.max_over_time_cnn import MaxOverTimeCNN
from trainers.triplet_trainer import TripletTrainer


def main():
    with open('config/train_cfg.yml', 'r') as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)
    
    dataset = QATripletDataset(
        config['database_file'], config['database_table'], 
        config['vocab_file'])

    dataloader = dataset.build_dataloader(
        batch_size=config['batch_size'], shuffle=True)

    encoder = MaxOverTimeCNN(
        len(dataset.get_vocab()), config['wordvec_dim'], config['sentvec_dim'], 
        acceleration=True)

    print(f'[INFO]: encoder architecture:\n{encoder}')

    trainer = TripletTrainer(encoder, dataloader)

    trainer.train(config['num_epochs'])

    encoder.save('artifacts/models')

    dataset.close()


if __name__ == '__main__':
    main()