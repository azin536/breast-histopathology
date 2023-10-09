import click

import numpy as np

from omegaconf import OmegaConf

from src.data_pipeline import DataGenerator
from src.model_building import CancerNet
from src.preparation import DataPreparator
from src.training import Trainer


@click.command()
@click.option('--config_path', type=str, default="config.yaml")
def main(config_path: str):
    config = OmegaConf.load(config_path)
    batch_size = config.data_pipeline.batch_size
    preprarator = DataPreparator(config)
    train_paths, val_paths = preprarator.get_train_val_paths()
    train_labels = [np.float64(path.split('/')[-1][-5: -4]) for path in train_paths]
    train_steps = int(np.ceil(len(train_paths) / batch_size))
    val_steps = int(np.ceil(len(val_paths) / batch_size))
    train_seq = DataGenerator(train_paths, train_labels,
                              train_steps,
                              batch_size, (50, 50))
    val_labels = [np.float64(path.split('/')[-1][-5: -4]) for path in val_paths]
    val_seq = DataGenerator(val_paths, val_labels,
                            val_steps,
                            batch_size, (50, 50))
    n_iter_val = len(val_paths) // batch_size
    model = CancerNet(config)
    compiled_model = model.get_compiled_model()
    trainer = Trainer(config)
    trainer.run(compiled_model, train_seq, train_steps, val_seq, val_steps)


if __name__ == '__main__':
    main()
