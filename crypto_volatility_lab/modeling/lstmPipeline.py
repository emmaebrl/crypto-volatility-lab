import random
import numpy as np
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from typing import Tuple
import tensorflow as tf
import os
from crypto_volatility_lab.modeling.modelPipelineBase import ModelPipelineBase

os.environ["TF_DETERMINISTIC_OPS"] = "1"


class LSTMPipeline(ModelPipelineBase):
    def __init__(
        self,
        lstm_units: Tuple[int, int] = (48, 16),
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate

        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)

    def create_model(self, input_shape: Tuple[int, ...]) -> Sequential:
        model = Sequential(
            [
                Input(shape=input_shape),
                LSTM(
                    self.lstm_units[0],
                    activation="relu",
                    return_sequences=True,
                ),
                Dropout(self.dropout_rate),
                LSTM(self.lstm_units[1], activation="relu"),
                Dropout(self.dropout_rate),
                Dense(self.forecast_horizon),
            ]
        )
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss="mse")
        return model
