from typing import Tuple
import numpy as np
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
import tensorflow as tf
import random
import os
from crypto_volatility_lab.modeling.modelPipelineBase import ModelPipelineBase

os.environ["TF_DETERMINISTIC_OPS"] = "1"


class LSTMGRUPipeline(ModelPipelineBase):
    def __init__(
        self,
        gru_unit: int = 16,
        lstm_unit: int = 48,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gru_unit = gru_unit
        self.lstm_unit = lstm_unit
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
                    self.lstm_unit,
                    activation="relu",
                    return_sequences=True,
                ),
                Dropout(self.dropout_rate),
                GRU(self.gru_unit, activation="relu"),
                Dropout(self.dropout_rate),
                Dense(self.forecast_horizon),
            ]
        )
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss="mse")
        return model
