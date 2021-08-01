import tensorflow.keras.metrics as tfmetrics


CLASSIFICATION_METRICS = [
      tfmetrics.TruePositives(name='tp'),
      tfmetrics.FalsePositives(name='fp'),
      tfmetrics.TrueNegatives(name='tn'),
      tfmetrics.FalseNegatives(name='fn'), 
      tfmetrics.Precision(name='precision'),
      tfmetrics.Recall(name='recall'),
      tfmetrics.AUC(name='auc'),
      tfmetrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

REGRESSION_METRICS = [
    tfmetrics.MeanAbsoluteError(name='mae'),
    tfmetrics.MeanSquaredError(name='mse'),
    tfmetrics.MeanSquaredLogarithmicError(name='msle')
]
