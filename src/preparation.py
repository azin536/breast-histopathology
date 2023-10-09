import random
import omegaconf
import os

from tqdm import tqdm
from typing import List, Union


class DataPreparator:
    def __init__(self, config: omegaconf.dictconfig.DictConfig) -> None:
        self.config = config

    def _get_original_paths(self) -> List:
        """Gets each png file path.

        Returns:
            List: list of each images of dataset paths.
        """
        random.seed(7)
        original_paths = list()
        for id_ in tqdm(os.listdir(self.config.data_preparation.dataset_path)):
            if 'IDC' not in id_:
                id_dir = str(self.config.data_preparation.dataset_path) + '/' + id_
                for label in os.listdir(id_dir):
                    label_dir = id_dir + '/' + label
                    for image in os.listdir(label_dir):
                        if not os.path.isdir(image):
                            original_paths.append(label_dir + '/' + image)
        random.shuffle(original_paths)
        return original_paths

    def get_train_val_paths(self) -> Union[List, List, List]:
        """Splits the original paths to train, val.

        Returns:
            Union[List, List, List]: list of train and val paths.
        """
        
        original_paths = self._get_original_paths()
        index = int(len(original_paths) * self.config.data_pipeline.train_split)
        train_path = original_paths[: index]
        index = int(len(train_path) * self.config.data_pipeline.val_split)
        val_paths = train_path[: index]
        train_paths = train_path[index: ]
        return train_paths, val_paths

    def get_test_paths(self) -> Union[List, List, List]:
        """Splits the original paths to test.

        Returns:
            Union[List, List, List]: list of test paths.
        """
        original_paths = self._get_original_paths()
        index = int(len(original_paths) * self.config.data_pipeline.train_split)
        test_paths = original_paths[index: ]
        return test_paths
