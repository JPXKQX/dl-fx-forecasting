import os
import json
import logging
import pandas as pd

from tensorflow.keras import callbacks
from typing import Union, NoReturn, Dict, Callable
from pathlib import Path
from ray import tune


log = logging.getLogger("Ray tune")



class TuneReporter(callbacks.Callback):
    """Tune Callback for Keras."""
    def __init__(self, reporter = None, logs: Dict = None):
        self.iteration = 0
        logs = logs or {}
        super(TuneReporter, self).__init__()

    def on_epoch_end(self, epoch, logs: Dict = None) -> NoReturn:
        logs = logs or {}
        self.iteration += 1
        if "acc" in logs:
            tune.report(
                keras_info=logs, val_loss=logs['val_loss'], mean_accuracy=logs["acc"]
            )
        else:
            tune.report(
                keras_info=logs, 
                val_loss=logs['val_loss'], 
                mean_accuracy=logs.get("accuracy")
            )


def new_callback(model_path: str = None) -> callbacks.Callback:
    """ Create a callback depending on the arguments specified.
    
    Args:
        model_path (str): path at which to sabe model checkpoints. Defaults to None, 
        which means that it correspond to the final run with best configuration.

    Returns:
        [tf.keras.callback.Callback]: the desired callback.
    """
    if model_path:
        log.info("Creating model checkpoint callback.")
        os.makedirs(model_path, exist_ok=True)
        return callbacks.ModelCheckpoint(
            filepath='train_model_epoch{epoch:03d}.h5', 
            monitor='val_loss', 
            save_best_only=True, 
            verbose=1
        )

    # Ray callback reporting the val_loss after each epoch (by specifying freq='epoch') 
    log.info("Creating tune reporter callback")
    return TuneReporter(freq="epoch")


def get_training_ray_method(
    model, 
    X: pd.DataFrame, 
    y: pd.DataFrame,
    snapshot_dir: Union[str, Path],
    final_run: bool = False
) -> Callable:
    def func(config):
        # Include checkpoint_dir argument.
        mo = model(**config)
        callback = new_callback(None if final_run else snapshot_dir)
        history = mo.fit(X, y, callback)

        # Get last metric features
        metrics = {}
        for k, v in history.history.items():
            metrics[k] = v[-1]

        tune.report(**metrics) 
    return func
