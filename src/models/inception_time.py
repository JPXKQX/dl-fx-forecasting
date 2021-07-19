import os
import logging 
import pandas as pd
import numpy as np

from src.models import model_constants, training_plot_utils
from src.data.constants import ROOT_DIR
from sklearn.utils.class_weight import compute_class_weight
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, NoReturn, Union
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Concatenate, Add, \
    Activation, Input, GlobalAveragePooling1D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback


logger = logging.getLogger("InceptionTime")


@dataclass
class InceptionTime:
    output_dims: int = 1
    depth: int = 6
    n_filters: int = 32
    batch_size: int = 64
    n_epochs: int = 200
    inception_kernels: List[int] = field(default_factory=lambda: [10, 20, 40])
    bottleneck_size: int = 32
    verbose: int = 2
    optimizer: str = 'adam'
    loss: str = 'mse'
    problem: str = 'regression'
    metrics: List[str] = field(default_factory=lambda: ['mae'])
    output_predictions: Path = Path(ROOT_DIR )

    def __post_init__(self) -> NoReturn:
        if self.problem not in ['regression', 'classification']:
            raise Exception(f"Problem {self.problem} is not supported yet. Please, "
                f"select one of the following: regression (default) or classifation.")
        self.output_predictions = self.output_predictions / "data" / "predictions" / \
            f"InceptionTime{self.problem.capitalize()}"
        os.makedirs(self.output_predictions, exist_ok=True)

    def __str__(self):
        return f"inceptionTime_{self.depth}depth_{self.n_filters}filters_" \
               f"{'-'.join(map(str, self.inception_kernels))}kernels"

    def _set_callbacks(
        self, 
        patience_reduce_lr: int = 4,
        patience_fit: int = 20,
        include_validation: bool = True
    ) -> NoReturn:
        logger.info("Two callbacks have been added to the model fitting: "
                    "ModelCheckpoint and ReduceLROnPlateau.")
        reduce_lr = ReduceLROnPlateau(
            monitor='loss', factor=0.5, patience=patience_reduce_lr, min_lr=1e-5
        )
        early_stopping = EarlyStopping(
            monitor='val_loss' if include_validation else 'loss', patience=patience_fit
        )
        self.callbacks = [early_stopping, reduce_lr]

    def _get_metrics(self):
        if self.problem == 'classification':
            return model_constants.CLASSIFICATION_METRICS
        elif self.problem == 'regression':
            return model_constants.REGRESSION_METRICS
        else:
            raise Exception(f"Problem type \'{self.problem}\' is not valid. Please "
                            f"select specify one of the following: classification or "
                            f"regression.")

    def _inception_module(self, input_tensor, stride=1, activation='linear'):
        if int(input_tensor.shape[-2]) > 1:
            input_inception = Conv1D(
                filters=self.bottleneck_size, kernel_size=1, padding='same', 
                activation=activation, use_bias=False
            )(input_tensor)
        else:
            input_inception = input_tensor

        # As presented in original paper InceptionTime: Finding AlexNet for Time Series 
        # Classification. https://arxiv.org/pdf/1909.04939.pdf
        conv_list = []
        for kernel_size in self.inception_kernels:
            conv_list.append(
                Conv1D(
                    filters=self.n_filters, kernel_size=kernel_size, 
                    strides=stride, padding='same', activation=activation,
                    use_bias=False
                )(input_inception)
            )

        max_pool_1 = MaxPool1D(
            pool_size=3, strides=stride, padding='same'
        )(input_tensor)

        conv_6 = Conv1D(
            filters=self.n_filters, kernel_size=1, padding='same', 
            activation=activation, use_bias=False
        )(max_pool_1)

        conv_list.append(conv_6)

        x = Concatenate(axis=-1)(conv_list)
        x = BatchNormalization()(x)
        x = Activation(activation='relu')(x)
        return x

    def _shortcut_layer(self, input_tensor, out_inception):
        shortcut_y = Conv1D(
            filters=int(out_inception.shape[-1]), kernel_size=1, padding='same', 
            use_bias=False
        )(input_tensor)
        shortcut_y = BatchNormalization()(shortcut_y)

        x = Add()([shortcut_y, out_inception])
        x = Activation('relu')(x)
        return x

    def build_model(self, input_shape: tuple) -> Model:
        logger.debug(f'Input data has shaper {input_shape}')
        input_layer = Input(input_shape)

        x = input_layer
        input_res = input_layer

        for d in range(self.depth):

            x = self._inception_module(x)

            if d % 3 == 2:
                input_res = x = self._shortcut_layer(input_res, x)

        gap_layer = GlobalAveragePooling1D()(x)
        
        output_layer = Dense(100, activation='relu')(gap_layer)
        f_act = 'linear' if self.problem == 'regression' else 'softmax'
        output_layer = Dense(self.output_dims, activation=f_act)(output_layer)

        self.model = Model(inputs=input_layer, outputs=output_layer)

        logger.info(self.model.summary())
        self.model.compile(
            loss=self.loss, 
            optimizer=self.optimizer, 
            metrics=self._get_metrics()
        )
        return self.model

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame, 
        more_callbacks: Union[Callback, List[Callback]] = [],
        validation_split: float = 0.2
    ) -> NoReturn:
        features, labels = self.reshape_data(X, y)

        # Update output dim and build model
        self.output_dims = labels.shape[1]
        self.build_model(features.shape[1:])
        self._set_callbacks(include_validation=bool(validation_split > 0))
        if isinstance(more_callbacks, list):
            self.callbacks.extend(more_callbacks)
        else:
            self.callbacks.append(more_callbacks)
        
        # Set weights for unbalanced datasets
        if self.problem == 'classification':
            classes = np.sort(y.iloc[:, 0].unique())
            weights = compute_class_weight('balanced', classes, y.values.ravel())
            class_weights = dict(enumerate(weights))
            logger.info(f"Weights for each class have been used: {class_weights}")
        else:
            class_weights = None

        history = self.model.fit(
            features, labels, validation_split=validation_split, verbose=self.verbose, 
            callbacks=[self.callbacks], epochs=self.n_epochs, class_weight=class_weights
        )

        # Save fig with results
        training_plot_utils.plot_metrics(
            history, "/tmp/history_{str(self)}.png", validation_split > 0, self.problem
        )
        return history

    def evaluate(self, features, labels):
        features, labels = self.reshape_data(features, labels)
        baseline_results = self.model.evaluate(features, labels)
        for name, value in zip(self.model.metrics_names, baseline_results):
            print(name, ': ', value, '\n')

    def predict(
        self, 
        X: pd.DataFrame,
        filename: Union[Path, str] = None
    ) -> pd.DataFrame:
        y_hat = self.model.predict(self.reshape_features(X))
        if self.problem == 'classification':
            y_hat = pd.DataFrame(
                y_hat, 
                index=X.index, 
                columns=[-1., 0., 1.] if self.output_dims == 3 else [0, 1]
            )
        else:
            y_hat = pd.DataFrame(y_hat, index=X.index)
        if filename is not None:
            if isinstance(filename, str):
                y_hat.to_csv(self.output_predictions / f"{filename}.csv")
            elif isinstance(filename, bool):
                y_hat.to_csv(self.output_predictions / f"{str(self)}.csv")
            else:
                raise Exception(f"Filename argument {filename} not recognized.")
        return y_hat

    def reshape_data(self, X: pd.DataFrame, y:pd.DataFrame) -> tuple[pd.DataFrame]:
        features = self.reshape_features(X)

        logger.info("The input data is contains only temporal features (air quality"
                    " variables).")
        if self.problem == 'classification':
            y = pd.get_dummies(y.iloc[:, 0], prefix='increment')

        return features, y.values

    def reshape_features(self, features: pd.DataFrame) -> pd.DataFrame:
        # Process temporal feaures. Including scaling ignoring timestep.
        n_time_steps = len(set(map(lambda x: x.split("_")[-1], features.columns)))
        n_vars = len(features.columns) // n_time_steps
        features_values = features.values.reshape((-1, n_vars, n_time_steps))
        
        return features_values

    def get_params(self, deep=True):
        return {
            "n_filters": self.n_filters, "bottleneck_size": self.bottleneck_size,
            "optimizer": self.optimizer, "loss": self.loss, "depth": self.depth,
            "batch_size": self.batch_size, "n_epochs": self.n_epochs}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        self.__post_init__()
        return self

    def save(self, filename: str, path: str):
        self.model.save(f"{path}/{filename}.h5")
        
    def load(self, filename: str):
        self.model = load_model(filename)
        print(self.model.summary())