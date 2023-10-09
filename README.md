# breast-histopathology

## Data preparation
First download the [dataset](https://www.kaggle.com/paultimothymooney/breast-histopathology-images/).

## Train
Following command will extract paths from dataset, create data generator, and fit the model you built in `src/model_building.py`:
```
python train.py
```

## Evaluate
For evaluation run:
```
python evaluate.py
```
This script will print following metrics:

- confusion matrix
- classification report
- accuracy
- specificity
- sensitivity

## Tensorboard
In the process of training, history of training would get save in `run/tensorboard`. To see the history saved you must run the bash `run_tensorboard.sh`:
```
sh run_tensorboard.sh
```

[Reference](https://data-flair.training/blogs/project-in-python-breast-cancer-classification/)