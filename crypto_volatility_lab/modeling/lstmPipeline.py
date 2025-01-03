import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Reshape # type: ignore
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed
from tqdm import tqdm

class LSTMPipeline:
    def __init__(self, window_size=25, forecast_horizon=5, in_sample_size=100, 
                 out_sample_size=5, lstm_units=50, n_jobs=1, epochs=1, batch_size=32):
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.in_sample_size = in_sample_size
        self.out_sample_size = out_sample_size
        self.lstm_units = lstm_units
        self.n_jobs = n_jobs
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler()

    def create_lagged_features(self, data):
        X, y = [], []
        for t in range(self.window_size, len(data) - self.forecast_horizon):
            X.append(data[t - self.window_size:t])
            y.append(data[t:t + self.forecast_horizon])
        return np.array(X), np.array(y)

    def rolling_window_indices(self, data_length):
        indices = []
        for start in range(data_length - self.in_sample_size - self.out_sample_size):
            train_start, train_end = start, start + self.in_sample_size
            test_start, test_end = train_end, train_end + self.out_sample_size
            indices.append((train_start, train_end, test_start, test_end))
        return indices

    def create_lstm_model(self, input_shape, forecast_horizon, num_features):
        model = Sequential([
            LSTM(self.lstm_units, activation='relu', input_shape=input_shape),
            Dense(forecast_horizon * num_features),
            Reshape((forecast_horizon, num_features))
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def process_window(self, data_scaled, X, y, indices):
        train_start, train_end, test_start, test_end = indices
        X_train, y_train = X[train_start:train_end], y[train_start:train_end]
        X_test, y_test = X[test_start:test_end], y[test_start:test_end]

        model = self.create_lstm_model(X_train.shape[1:], self.forecast_horizon, data_scaled.shape[1])
        model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        test_loss = model.evaluate(X_test, y_test, verbose=0)

        return train_start, test_start, test_loss

    def run(self, data):
        data_scaled = self.scaler.fit_transform(data)
        X, y = self.create_lagged_features(data_scaled)
        indices_list = self.rolling_window_indices(len(X))

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.process_window)(data_scaled, X, y, indices) for indices in tqdm(indices_list, desc="Training progress")
        )

        return results