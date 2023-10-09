import omegaconf

import tensorflow.keras as tfk

tfkl = tfk.layers
tfkm = tfk.models
K = tfk.backend


class CancerNet:
    def __init__(self, config: omegaconf.dictconfig.DictConfig) -> None:
        self.config = config
        self.input_shape = config.model_building.input_shape
        self.classes = config.model_building.num_classes
    
    def _build_model(self) -> tfk.Model:
        """Builds the arch of the model.

        Returns:
            tfk.Model: the built model.
        """
        model = tfkm.Sequential()
        channel_dim = -1
        
        model.add(tfkl.SeparableConv2D(32, (3, 3), padding='same',
                                       input_shape=self.input_shape))
        model.add(tfkl.Activation('relu'))
        model.add(tfkl.BatchNormalization(axis=channel_dim))
        model.add(tfkl.MaxPooling2D(pool_size=(2, 2)))
        model.add(tfkl.Dropout(0.25))
                  
        model.add(tfkl.SeparableConv2D(64, (3, 3), padding='same'))
        model.add(tfkl.Activation('relu'))
        model.add(tfkl.BatchNormalization(axis=channel_dim))
        model.add(tfkl.SeparableConv2D(64, (3, 3), padding='same'))
        model.add(tfkl.Activation('relu'))
        model.add(tfkl.BatchNormalization(axis=channel_dim))
        model.add(tfkl.MaxPooling2D(pool_size=(2, 2)))
        model.add(tfkl.Dropout(0.25))
                  
        model.add(tfkl.SeparableConv2D(128, (3, 3), padding='same'))
        model.add(tfkl.Activation('relu'))
        model.add(tfkl.BatchNormalization(axis=channel_dim))
        model.add(tfkl.SeparableConv2D(128, (3, 3), padding='same'))
        model.add(tfkl.Activation('relu'))
        model.add(tfkl.BatchNormalization(axis=channel_dim))
        model.add(tfkl.SeparableConv2D(128, (3, 3), padding='same'))
        model.add(tfkl.Activation('relu'))
        model.add(tfkl.BatchNormalization(axis=channel_dim))
        model.add(tfkl.MaxPooling2D(pool_size=(2, 2)))
        model.add(tfkl.Dropout(0.25))
                  
        model.add(tfkl.Flatten())
        model.add(tfkl.Dense(256))
        model.add(tfkl.Activation('relu'))
        model.add(tfkl.BatchNormalization())
        model.add(tfkl.Dropout(0.5))
                  
        model.add(tfkl.Dense(self.classes))
        model.add(tfkl.Activation('sigmoid'))    
        return model
    
    def get_compiled_model(self):
        model = self._build_model()
        init_lr = self.config.model_building.init_lr
        num_epochs = self.config.model_building.num_epochs
        optimization = tfk.optimizers.legacy.Adagrad(init_lr,
                                                     decay=init_lr / num_epochs)
        model.compile(loss=K.binary_crossentropy, optimizer=optimization,
                      metrics=['accuracy'])
        return model