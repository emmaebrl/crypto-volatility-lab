from typing import Tuple
import numpy as np
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Input, Flatten  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
import tensorflow as tf
import random
import os
from crypto_volatility_lab.modeling.modelPipelineBase import ModelPipelineBase

os.environ["TF_DETERMINISTIC_OPS"] = "1"


class TCNNPipeline(ModelPipelineBase):
    def __init__(
        self,
        num_filters: int = 64,
        kernel_size: int = 3,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate

        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)

    def create_model(self, input_shape: Tuple[int, ...]) -> Sequential:
        model = Sequential(
            [
                Input(shape=input_shape),
                Conv1D(
                    filters=self.num_filters,
                    kernel_size=self.kernel_size,
                    activation="relu",
                    padding="same",
                ),
                Dropout(self.dropout_rate),
                Conv1D(
                    filters=self.num_filters,
                    kernel_size=self.kernel_size,
                    activation="relu",
                    padding="same",
                ),
                Dropout(self.dropout_rate),
                Flatten(),
                Dense(self.forecast_horizon),
            ]
        )
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss="mse")
        return model
