import pandas as pd
import matplotlib.pyplot as plt
import os  # or from pathlib import Path
from keras.callbacks import CSVLogger


import tensorflow as tf
from tensorflow.python.client import device_lib

import os


import numpy as np

MODEL_PATH = "model.keras"
BATCH_SIZE = 16
LOOK_BACK = 100
FORECAST_HORIZON = 100

seed = 101
tf.keras.utils.set_random_seed(seed)


X_test = np.arange(0, 2000, 0.5)
y_test = 2 * np.sin(X_test) + 0.8 * np.random.rand(X_test.shape[0])


df = pd.DataFrame({"x": X_test, "TARGET_y": y_test})
#### DEF FUNCTIONS ####


def prepare_time_series_data(
    df,
    target_columns,
    lookback=12,
    forecast_horizon=1,
):
    target_columns = list(target_columns)  # convert if it's an Index
    non_target_cols = [
        col for col in df.columns if col not in (target_columns + ["datetime"])
    ]

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])

        # Encode cyclical features
        df["month_sin"] = np.sin(2 * np.pi * df["datetime"].dt.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["datetime"].dt.month / 12)
        df["day_sin"] = np.sin(2 * np.pi * df["datetime"].dt.dayofyear / 365)
        df["day_cos"] = np.cos(2 * np.pi * df["datetime"].dt.dayofyear / 365)
        df = df.drop(columns=["datetime"])

    # lagged_features = []

    # for col in non_target_cols:
    #     lag_dict = {
    #         f"{col}_lag1": df[col].shift(1),
    #         f"{col}_lag7": df[col].shift(7),
    #         f"{col}_lag30": df[col].shift(30),
    #         f"{col}__mean_1": df[col].shift(1).rolling(1).mean(),
    #         f"{col}__mean_30": df[col].shift(1).rolling(30).mean()
    #     }
    #     lagged_features.append(pd.DataFrame(lag_dict))

    # Concatenate all lag features at once
    #  df_lags = pd.concat(lagged_features, axis=1)
    #  df = pd.concat([df, df_lags], axis=1)

    df = df.dropna()  # Drop rows with NaNs from lagging

    return df[non_target_cols].values, df[target_columns].values


def manual_train_test_split(X, y, train_size=0.7, val_size=0.1):
    total_len = len(X)

    train_end = int(total_len * train_size)
    val_end = train_end + int(total_len * val_size)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    return X_train, X_val, X_test, y_train, y_val, y_test


def make_time_series_datasets(
    X_train, y_train, X_val, y_val, X_test, y_test, look_back, forecast, batch_size
):
    def create_dataset(X_data, y_data, shuffle=False, repeat=False):
        # Ensure input and output are tensors
        X_data = tf.convert_to_tensor(X_data, dtype=tf.float32)
        y_data = tf.convert_to_tensor(y_data, dtype=tf.float32)

        # Create windowed dataset
        dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=tf.concat([X_data, y_data], axis=-1),  # combine to keep alignment
            targets=None,
            sequence_length=look_back + forecast,
            sequence_stride=1,
            shuffle=shuffle,
            batch_size=1,
        )

        def split_input_target(sequence):
            input_seq = sequence[:, :look_back, : X_data.shape[-1]]
            target_seq = sequence[:, look_back:, X_data.shape[-1] :]
            return input_seq, target_seq

        dataset = dataset.map(split_input_target)

        return dataset

    train_dataset = create_dataset(X_train, y_train, shuffle=True, repeat=True)
    val_dataset = create_dataset(X_val, y_val)
    test_dataset = create_dataset(X_test, y_test)

    return train_dataset, val_dataset, test_dataset


def build_and_compile_model(num_features, out_steps):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(LOOK_BACK, num_features)),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(out_steps * len(targets)),
            tf.keras.layers.Reshape((out_steps, len(targets))),
        ]
    )

    model.compile(loss="mae", optimizer=tf.keras.optimizers.Adam(), metrics=["mape"])
    return model


def build_and_compile_mlp_model(input_dim, output_dim):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(128),
            tf.keras.layers.Activation("tanh"),
            tf.keras.layers.Dense(
                64,
            ),
            tf.keras.layers.Activation("tanh"),
            tf.keras.layers.Dense(output_dim),
        ]
    )

    model.compile(optimizer="adam", loss="mae", metrics=["mape"])
    model.summary()
    return model


#### START ####

targets = df.columns[df.columns.map(lambda col: col.startswith("TARGET"))]

X, y = prepare_time_series_data(
    df, target_columns=targets, lookback=670, forecast_horizon=FORECAST_HORIZON
)


X = np.asarray(X).astype(np.float32)
y = np.asarray(y).astype(np.float32)


X_train, X_val, X_test, y_train, y_val, y_test = manual_train_test_split(
    X,
    y,
)


train_dataset, val_dataset, test_dataset = make_time_series_datasets(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    LOOK_BACK,
    FORECAST_HORIZON,
    BATCH_SIZE,
)


def dataset_to_numpy(dataset):
    x_list, y_list = [], []
    for x_batch, y_batch in dataset:
        x_list.append(x_batch.numpy())
        y_list.append(y_batch.numpy())

    X = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y


print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val:", X_val.shape, "y_val:", y_val.shape)
print("X_test:", X_test.shape, "y_test:", y_test.shape)


# Extract all data
X_train, y_train = dataset_to_numpy(train_dataset)
X_val, y_val = dataset_to_numpy(val_dataset)
X_test, y_test = dataset_to_numpy(test_dataset)

y_train = y_train.reshape(y_train.shape[0], y_train.shape[1] * y_train.shape[2])
y_val = y_val.reshape(y_val.shape[0], y_val.shape[1] * y_val.shape[2])
y_test = y_test.reshape(y_test.shape[0], y_test.shape[1] * y_test.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1] * X_val.shape[2])
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])

# print shapes
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val:", X_val.shape, "y_val:", y_val.shape)
print("X_test:", X_test.shape, "y_test:", y_test.shape)


csv_logger = CSVLogger("training.log", separator=",", append=False)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=100000, restore_best_weights=True
)
model = build_and_compile_model(out_steps=FORECAST_HORIZON, num_features=len(targets))
input_dim = X_train.shape[1]
output_dim = y_train.shape[1]

model = build_and_compile_mlp_model(input_dim, output_dim)
history = model.fit(
    train_dataset,
    epochs=20,
    callbacks=[csv_logger, early_stopping],
    validation_data=val_dataset,
).history


# Helper function to plot input sequence, true future, and predicted future
def multi_step_output_plot(true_future, prediction):
    plt.figure(figsize=(18, 6))
    print(np.shape(true_future))

    plt.plot(true_future, label="True Future")
    plt.plot(prediction, label="Predicted Future")

    plt.legend(loc="upper left")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.title("Multi-Step Time Series Forecasting")
    plt.grid(True)
    plt.show()


# Take one batch from validation dataset
for x_batch, y_batch in val_dataset.take(1):
    input_seq = x_batch  # First sequence in batch
    true_future = y_batch[0]  # Corresponding true future values
    prediction = model.predict(input_seq)[0]  # Shape: (output steps,)
    print(y_batch[0])
    print(prediction)

    multi_step_output_plot(np.squeeze(true_future), np.squeeze(prediction))
