import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

# Set random seed
seed = 101
keras.utils.set_random_seed(seed)

# Generate the time series data
X_train = np.arange(0, 100, 0.5)
y_train = np.sin(X_train)

X_test = np.arange(100, 200, 0.5)
y_test = np.sin(X_test)

n_features = 1

train_series = y_train.reshape((len(y_train), n_features))
test_series = y_test.reshape((len(y_test), n_features))

# Define a function to create dataset from the time series data
def create_dataset(data, look_back=20, future_steps=40):
    X, y = [], []
    for i in range(len(data) - look_back - future_steps):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back:i + look_back + future_steps])
    return np.array(X), np.array(y)

look_back = 20
future_steps = 40  # Number of steps to predict

# Create training and testing datasets using the custom function
X_train_data, y_train_data = create_dataset(train_series, look_back, future_steps)
X_test_data, y_test_data = create_dataset(test_series, look_back, future_steps)

# Create tf.data.Dataset objects
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_data, y_train_data))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test_data, y_test_data))

# Batch and shuffle the datasets
batch_size = 1
train_dataset = train_dataset.shuffle(buffer_size=100).batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

# Define and compile the model
n_neurons = 2

model = keras.Sequential([
    keras.layers.Input(shape=(look_back, n_features)),
    keras.layers.LSTM(2),
    keras.layers.Dense(future_steps),  # Predict the next 'future_steps' values
])

model.compile(
    loss='mae',
    optimizer=keras.optimizers.Adam(),
    metrics=['mape']
)

# Train the model
model.fit(train_dataset, epochs=50)

# Predict using the model
# Get the last sequence of test data and predict the next 40 values directly
last_sequence = X_test_data[-1].reshape((1, look_back, n_features))
predictions = model.predict(last_sequence)

# Create the plot
x = np.arange(110, 200, 0.5)
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
ax.plot(X_train, y_train, lw=2, label='train data')
ax.plot(X_test, y_test, lw=3, c='y', label='test data')

# Plot the predictions directly after the test data
ax.plot(x[:40], predictions.flatten(), lw=3, c='r', linestyle=':', label='predictions')

ax.legend(loc="lower left")
plt.show()
