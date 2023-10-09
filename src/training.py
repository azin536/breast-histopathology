import omegaconf

import tensorflow.keras as tfk
import tensorflow.keras.callbacks as tfkc

from typing import List

from .data_pipeline import DataGenerator


class Trainer:
    def __init__(self, config: omegaconf.dictconfig.DictConfig) -> None:
        self.config = config

    def run(self, model: tfk.Model, train_seq: DataGenerator, n_iter_train: int,
            val_seq: DataGenerator, n_iter_val: int) -> None:
        """Trains the built model using train and val generator.

        Args:
            model (tfk.Model): built and compiled model
            train_seq (DataGenerator): train generator
            n_iter_train (int): number of steps per epoch for train data
            val_seq (DataGenerator): val generator
            n_iter_val (int): number of steps per epoch for val data
        """
        callbacks = self._get_callbacks()
        history = model.fit(train_seq, steps_per_epoch=n_iter_train,
                            validation_data=val_seq, validation_steps=n_iter_val,
                            epochs=self.config.training.num_epochs,
                            callbacks=callbacks)

    def _get_callbacks(self) -> List:
        """Gets model and tensorboard checkpoints.

        Returns:
            List: list of tf keras callbacks.
        """
        callbacks = list()
        to_track = self.config.training.track_metric
        checkpoint_path = "run" + "/sm-{epoch}"
        checkpoint_path = checkpoint_path + "-{" + to_track + ":4.5f}"
        mc = tfkc.ModelCheckpoint(file_path=checkpoint_path, save_weights_only=False)
        tb = tfkc.TensorBoard(log_dir='run/' + 'tensorboard')
        callbacks.append(mc)
        callbacks.append(tb)
        return callbacks
