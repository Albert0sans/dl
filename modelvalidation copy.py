import matplotlib.pyplot as plt
from keras.callbacks import CSVLogger
import tensorflow as tf
import numpy as np


BATCH_SIZE = 32
LOOK_BACK = 120
FORECAST_HORIZON = 120

seed = 101
tf.keras.utils.set_random_seed(seed)


# Time variable
X = np.arange(0, 2000, 0.1).reshape(-1, 1)

# Complex time series with trend, seasonality, and noise
y = np.sin(X)

#### DEF FUNCTIONS ####


def manual_train_test_split(X, y, train_size=0.7, val_size=0.1):
    total_len = len(X)

    train_end = int(total_len * train_size)
    val_end = train_end + int(total_len * val_size)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    train_mean = X_train.mean()
    train_std = X_train.std()
    train_mean_y = y_train.mean()
    train_std_y = y_train.std()
    X_train = (X_train - train_mean) / train_std
    X_val = (X_val - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std

    y_train = (y_train - train_mean_y) / train_std_y
    y_val = (y_val - train_mean_y) / train_std_y
    y_test = (y_test - train_mean_y) / train_std_y

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
            shuffle=True,
            batch_size=batch_size,
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


def build_and_compile_model(num_features, out_steps, targets):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
            #   tf.keras.layers.Input(shape=(LOOK_BACK, num_features)),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(out_steps * targets),
            tf.keras.layers.Reshape([out_steps, targets]),
        ]
    )

    model.compile(
        loss="mse",
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=["mse", "mape"],
    )
    return model


#### START ####


X_train, X_val, X_test, y_train, y_val, y_test = manual_train_test_split(X, y)

print(np.shape(X_train))
print(np.shape(X_val))
print(np.shape(X_test))
print(np.shape(y_train))
print(np.shape(y_val))
print(np.shape(y_test))


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

csv_logger = CSVLogger("training.log", separator=",", append=False)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=100000, restore_best_weights=True
)
model = build_and_compile_model(
    num_features=X_train.shape[-1], out_steps=FORECAST_HORIZON, targets=1
)

history = model.fit(
    train_dataset,
    epochs=4,
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


print("HERE")

# Take one batch from validation dataset
for x_batch, y_batch in test_dataset.take(5):
    input_seq = x_batch  # First sequence in batch
    true_future = y_batch  # Corresponding true future values
    prediction = model.predict(input_seq)[0][0]  # Shape: (output steps,)

    multi_step_output_plot(np.squeeze(true_future), np.squeeze(prediction))
