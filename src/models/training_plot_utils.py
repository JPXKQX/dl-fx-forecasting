import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import logging

from typing import NoReturn, Union
from tensorflow.python.keras.callbacks import History
from pathlib import Path
from sklearn import metrics

mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


log = logging.getLogger("Plotter Model training")


def save_r2_time_struc(r2time, outfile: Union[str, Path]) -> NoReturn:
    log.info(f"Plotting R2 with time structure to {outfile}")
    plt.figure(figsize=(12, 9))
    r2time.plot(legend=False)
    plt.ylabel("R-Squared with time structure")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig(outfile)


def plot_roc(name: str, labels, predictions, **kwargs):
    """ Plots the Receiver Operating Characteristic (ROC) curve, which is the curve of 
    true positive rate vs. false positive rate at different classification thresholds.
    It shows the range of performance the model can reach just by tuning the output
    threshold.

    Args:
        name (str): name of the model to show in the legend.
        labels: true labels
        predictions: values predicted by the model
    """
    fp, tp, _ = metrics.roc_curve(labels, predictions)

    plt.plot(100 * fp, 100 * tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5, 20])
    plt.ylim([80, 100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')


def plot_metrics(
    history: History, 
    filename: Union[str, Path],
    skip_val: bool = False,
    problem: str = 'regression'
) -> NoReturn:
    """ Produce plots of your model's accuracy and loss on the training and validation 
    set. These are useful to check for overfitting.

    Args:
        history (History): model fitting history.
        filename (str): filename at which save model trainig performance plots.
        problem (str, optional): type of problem. Choices are: classification, and 
        regression (default).

    Raise:
        Exception: when problem is not one of the choices.
    """
    if problem == 'regression':
        metrics = ['loss', 'mae', 'mse', 'msle']
    elif problem == 'classification':
        metrics = ['loss', 'prc', 'precision', 'recall']
    else:
        raise Exception(f"Problem type \'{problem}\' is not valid. Please select "
                        f"specify one of the following: classification or regression.")
        
    metrics = ['loss', 'prc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n + 1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        if skip_val:
            plt.plot(history.epoch, history.history['val_' + metric],
                     color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8, 1])
        else:
            plt.ylim([0, 1])

    plt.tight_layout()
    plt.legend()
    plt.savefig(filename)


def plot_cm(labels, predictions, p: float = 0.5):
    """ Uses a confusion matrix to summarize the actual vs. predicted labels, where the
    X axis is the predicted label and the Y axis is the actual label.

    Args:
        labels: true labels
        predictions: values predicted by the model
        p (float, optional): Classification threshold. Defaults to 0.5.
    """
    cm = metrics.confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    log.info('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
    log.info('Legitimate Transactions Incorrectly Detected (False Positives): ', 
             cm[0][1])
    log.info('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
    log.info('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
    log.info('Total Fraudulent Transactions: ', np.sum(cm[1]))


def plot_prc(name, labels, predictions, **kwargs):
    """ Area under the interpolated precision-recall curve, obtained by plotting 
    (recall, precision) points for different values of the classification threshold. 
    Depending on how it's calculated, PR AUC may be equivalent to the average precision 
    of the model.

    Args:
        name (str): name of the model to show in the legend.
        labels: true labels
        predictions: values predicted by the model
    """
    precision, recall, _ = metrics.precision_recall_curve(labels, predictions)

    plt.plot(precision, recall, label=name, linewidth=2, **kwargs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')