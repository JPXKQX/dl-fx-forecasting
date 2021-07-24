import tensorflow as tf
from dataclasses import dataclass, field
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from typing import List, Union, NoReturn


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


@dataclass
class MultiLayerPerceptron:
    n_neurons: List[int] = field(default_factory=lambda: [64])
    f_act: Union[str, List[str]] = 'relu'
    optimizer: str = 'adam'
    loss: str = 'mse'
    problem: str = 'regression'
    validation_split: float = 0.2
    metrics: List[str] = field(default_factory=lambda: ['mae'])

    def __post_init__(self) -> NoReturn:
        if isinstance(self.f_act, str):
            self.f_act = [self.f_act] * len(self.n_neurons)

    def compile(self, input_shape: int, output_shape: int) -> NoReturn:
        layers = []
        for i, neurons in enumerate(self.n_neurons):
            if i == 0:
                layers.append(Dense(neurons, activation=self.f_act[i], 
                                    input_shape=(input_shape, )))
            else:
                layers.append(Dense(neurons, activation=self.f_act[i]))
        
        layers.append(Dense(output_shape))

        self.model = Sequential(layers, name="MLP")
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

    def fit(self,
        X,
        y,
        epochs: int = 200,
        validation_split: float = None,
        patience: int = 10
    ):
        n_features = X.shape[1] if len(X.shape) > 1 else 1
        n_labels = y.shape[1] if len(y.shape) > 1 else 1
        self.compile(n_features, n_labels)
        
        # Create callbacks
        if validation_split is None: validation_split = self.validation_split
        reduce_lr = ReduceLROnPlateau(
            monitor='loss', factor=0.5, patience=int(patience // 5), min_lr=1e-5
        )
        early_stopping = EarlyStopping(
            monitor='val_loss' if validation_split > 0 else 'loss', patience=patience
        )        
        return self.model.fit(
            X, y, epochs=epochs, verbose=2, validation_split=validation_split, 
            callbacks=[reduce_lr, early_stopping]
        )

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self, deep=True):
        return {
            "n_neurons": self.n_neurons, "f_act": self.f_act,
            "optimizer": self.optimizer, "loss": self.loss}

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
