import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow.keras as tfk

from omegaconf import OmegaConf
from sklearn.metrics import classification_report, confusion_matrix
from typing import List, Union

from src.data_pipeline import DataGenerator
from src.preparation import DataPreparator


def _get_best_model() -> str:
    """Gets the best saved model.

    Returns:
        str: the best model with lowest track metric
    """
    checkpoints = os.listdir('run/checkpoints')
    min_checkpoint = checkpoints[0].split('-')[-1]
    min_ind = 0
    for i, checkpoint in enumerate(checkpoints):
        if checkpoint.split['-'][-1] < min_checkpoint:
            min_ind = i
            min_checkpoint = checkpoint.split['-'][-1]
    return "run/checkpoints" + checkpoints[min_ind]


def _get_classification_report(trues: List, predictions: np.array, threshold: float):
    """Gets classification report

    Args:
        trues (List): labels
        predictions (np.array): preds
        threshold (float): threshold
    """
    preds_th = [int(pred > threshold) for pred in predictions]
    print(classification_report(trues, preds_th))


def _calculate_metrics(trues: List, predictions: np.array, threshold: float):
    """Calculates confustion matrix, accuracy, specificity, and sensitivity.

    Args:
        trues (List): labels
        predictions (np.array): preds
        threshold (float): threshold
    """
    preds_th = [int(pred > threshold) for pred in predictions]
    cm = confusion_matrix(trues, preds_th)
    total = sum(sum(cm))
    accuracy = (cm[0, 0] + cm[1, 1]) / total
    specificity = cm[1, 1] / (cm[1, 1] + cm[0, 1])
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[1, 0])
    dataframe = pd.DataFrame(cm, index=['non', 'cancer'], columns=['non', 'cancer'])
    sns.heatmap(dataframe, annot=True, cbar=None, cmap='Pastel1', fmt='g')
    plt.xlabel('pred')
    plt.ylabel('true')
    plt.show()
    print("accuracy =", accuracy)
    print("specificity =", specificity)
    print("sensitivity =", sensitivity)


def main(config_path):
    config = OmegaConf.load(config_path)
    threshold = config.evaluation.threshold
    preparator = DataPreparator(config)
    test_paths = preparator.get_test_paths()
    test_labels = [np.float64(path.split('/')[-1][-5: -4]) for path in test_paths]
    test_seq = DataGenerator(config, test_paths, test_labels)
    model_path = _get_best_model()
    model = tfk.models.load_model(model_path)
    preds = model.predict(test_seq)
    _get_classification_report(test_labels, preds, threshold)
    _calculate_metrics(test_labels, preds, threshold)


if __name__ == '__main__':
    main()
