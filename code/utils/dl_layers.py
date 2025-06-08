from utils.sklearn3dwrapper import sklearn3dWrapper
import tensorflow as tf
#import ydf
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
#Input shape is: (look_back,forecast)
import numpy as np
from utils.layers.transformer import transformer_timeseries

tf.config.set_visible_devices([], 'GPU')

import tensorflow as tf
from keras.saving import register_keras_serializable

@register_keras_serializable()
class ZeroBaseline(tf.keras.Model):
    def __init__(self, out_steps, out_width, **kwargs):
        super().__init__(**kwargs)
        self.out_steps = out_steps
        self.out_width = out_width

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        return tf.fill([batch_size, self.out_steps, self.out_width], 0.0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "out_steps": self.out_steps,
            "out_width": self.out_width,
        })
        return config


def gbt(OUT_STEPS,OUT_WIDTH):
    return ydf.GradientBoostedTreesLearner(label="targets",task=ydf.Task.REGRESSION)


def rf(OUT_STEPS,OUT_WIDTH):
    # Create the full pipeline
    

    # Wrap your RandomForestRegressor
    return sklearn3dWrapper(
        RandomForestRegressor( n_estimators=100, n_jobs=-1,max_depth=6),
        target_shape=(OUT_STEPS,OUT_WIDTH) 
    
    )
    
    
def transformer(INPUT_WIDTH,in_features,OUT_STEPS,out_features):
    return transformer_timeseries(
 
    input_shape= (INPUT_WIDTH,in_features),
    head_size=32,
    num_heads=4,
    ff_dim=2,
    num_transformer_blocks=4,
    mlp_units=[32],
    mlp_dropout=0.2,
    dropout=0.25,

    output_size=[OUT_STEPS,out_features]
)
    
def hgb(OUT_STEPS,OUT_WIDTH):
    # Create the full pipeline
    

    # Wrap your RandomForestRegressor
    return sklearn3dWrapper(
        HistGradientBoostingRegressor( max_iter=100,),
        target_shape=(OUT_STEPS,OUT_WIDTH) 
    
    )
def extrarf(OUT_STEPS,OUT_WIDTH):
    # Create the full pipeline
    

    # Wrap your RandomForestRegressor
    return sklearn3dWrapper(
        ExtraTreesRegressor( n_estimators=100, max_depth=6,n_jobs=-1),
        target_shape=(OUT_STEPS,OUT_WIDTH) 
    
    )


def auto_encoder(OUT_STEPS, INPUT_WIDTH, in_features, out_features):
    # Encoder model
    encoder = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(INPUT_WIDTH, in_features)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
    ], name="encoder")

    # Decoder model
    decoder = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(OUT_STEPS * out_features, activation='linear'),
        tf.keras.layers.Reshape((OUT_STEPS, out_features)),
    ], name="decoder")

    # Connect encoder and decoder
    inputs = tf.keras.Input(shape=(INPUT_WIDTH, in_features))
    encoded = encoder(inputs)
    decoded = decoder(encoded)

    # Final autoencoder model
    model = tf.keras.Model(inputs=inputs, outputs=decoded, name="autoencoder")

    return model


def rnn_model(OUT_STEPS,INPUT_WIDTH, in_features,out_features):
    return tf.keras.Sequential(
        name="rnn_model",
        layers=[
            tf.keras.layers.Input(shape=(INPUT_WIDTH, in_features)),
            tf.keras.layers.LSTM(32, activation="relu", return_sequences=True),
            tf.keras.layers.LSTM(16, activation="relu"),
            tf.keras.layers.Dense(
                OUT_STEPS * out_features,
            ),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([OUT_STEPS, out_features]),
        ],
    )
def cnnlstm1(OUT_STEPS, INPUT_WIDTH, in_features, out_features):
    """
    An improved CNN-LSTM model for time series forecasting.

    Args:
        OUT_STEPS (int): The number of future time steps to predict.
        INPUT_WIDTH (int): The number of past time steps used as input.
        in_features (int): The number of input features per time step.
        out_features (int): The number of output features per time step.

    Returns:
        tf.keras.Sequential: The compiled Keras model.
    """
    model = tf.keras.Sequential(
        name="cnn_lstm_improved",
        layers=[
            tf.keras.layers.Input(shape=(INPUT_WIDTH, in_features)),

            # --- Improved Convolutional Block ---
            # Increase filters for more feature learning capacity
            tf.keras.layers.Conv1D(filters=32, kernel_size=5, padding="causal", activation="relu"),
            # Batch Normalization for stable training and faster convergence
            tf.keras.layers.BatchNormalization(),
            # Dropout for regularization to prevent overfitting
            tf.keras.layers.Dropout(0.3),

           

            # --- Improved LSTM Layers ---
            # Increase LSTM units for greater capacity
            # Use return_sequences=True to pass output to the next LSTM layer
            # Added recurrent_dropout for regularization within the LSTM's recurrent connections
            tf.keras.layers.LSTM(32, activation="relu", return_sequences=True, recurrent_dropout=0.2),
            tf.keras.layers.Dropout(0.4), # Dropout after LSTM output

            # Final LSTM layer, no return_sequences as it feeds into a Dense layer
            tf.keras.layers.LSTM(16, activation="relu", recurrent_dropout=0.2),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(32,),

            # Output Dense Layer
            tf.keras.layers.Dense(OUT_STEPS * out_features),

            # Reshape to desired output dimensions [batch, out_steps, features]
            tf.keras.layers.Reshape([OUT_STEPS, out_features]),
        ],
    )
    return model
def cnnlstm2(OUT_STEPS, INPUT_WIDTH, in_features, out_features):
    """
    An improved CNN-LSTM model for time series forecasting.

    Args:
        OUT_STEPS (int): The number of future time steps to predict.
        INPUT_WIDTH (int): The number of past time steps used as input.
        in_features (int): The number of input features per time step.
        out_features (int): The number of output features per time step.

    Returns:
        tf.keras.Sequential: The compiled Keras model.
    """
    model = tf.keras.Sequential(
        name="cnn_lstm_improved",
        layers=[
            tf.keras.layers.Input(shape=(INPUT_WIDTH, in_features)),

            # --- Improved Convolutional Block ---
            # Increase filters for more feature learning capacity
            tf.keras.layers.Conv1D(filters=128, kernel_size=5, padding="causal", activation="relu"),
            # Batch Normalization for stable training and faster convergence
            tf.keras.layers.BatchNormalization(),
            # Dropout for regularization to prevent overfitting
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding="causal", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),

            # A final Conv1D with fewer filters before LSTM
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding="causal", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),

            # --- Improved LSTM Layers ---
            # Increase LSTM units for greater capacity
            # Use return_sequences=True to pass output to the next LSTM layer
            # Added recurrent_dropout for regularization within the LSTM's recurrent connections
            tf.keras.layers.LSTM(64, activation="relu", return_sequences=True, recurrent_dropout=0.2),
            tf.keras.layers.Dropout(0.4), # Dropout after LSTM output

            # Final LSTM layer, no return_sequences as it feeds into a Dense layer
            tf.keras.layers.LSTM(32, activation="relu", recurrent_dropout=0.2),
            tf.keras.layers.Dropout(0.4),

            # Output Dense Layer
            tf.keras.layers.Dense(OUT_STEPS * out_features),

            # Reshape to desired output dimensions [batch, out_steps, features]
            tf.keras.layers.Reshape([OUT_STEPS, out_features]),
        ],
    )
    return model
def cnnlstm3(OUT_STEPS, INPUT_WIDTH, in_features, out_features):
    """
    An improved CNN-LSTM model for time series forecasting.

    Args:
        OUT_STEPS (int): The number of future time steps to predict.
        INPUT_WIDTH (int): The number of past time steps used as input.
        in_features (int): The number of input features per time step.
        out_features (int): The number of output features per time step.

    Returns:
        tf.keras.Sequential: The compiled Keras model.
    """
    model = tf.keras.Sequential(
        name="cnn_lstm_improved",
        layers=[
            tf.keras.layers.Input(shape=(INPUT_WIDTH, in_features)),

            # --- Improved Convolutional Block ---
            # Increase filters for more feature learning capacity
            tf.keras.layers.Conv1D(filters=128, kernel_size=5, padding="causal", activation="relu"),
            # Batch Normalization for stable training and faster convergence
            tf.keras.layers.BatchNormalization(),
            # Dropout for regularization to prevent overfitting
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding="causal", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),

            # A final Conv1D with fewer filters before LSTM
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding="causal", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),

            # --- Improved LSTM Layers ---
            # Increase LSTM units for greater capacity
            # Use return_sequences=True to pass output to the next LSTM layer
            # Added recurrent_dropout for regularization within the LSTM's recurrent connections
            tf.keras.layers.LSTM(64, activation="relu", return_sequences=True, recurrent_dropout=0.2),
            tf.keras.layers.Dropout(0.4), # Dropout after LSTM output

            # Final LSTM layer, no return_sequences as it feeds into a Dense layer
            tf.keras.layers.LSTM(32, activation="relu", recurrent_dropout=0.2),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            # Output Dense Layer
            tf.keras.layers.Dense(OUT_STEPS * out_features),

            # Reshape to desired output dimensions [batch, out_steps, features]
            tf.keras.layers.Reshape([OUT_STEPS, out_features]),
        ],
    )
    return model
def rnn_model_gru(OUT_STEPS,INPUT_WIDTH, in_features,out_features):
    return tf.keras.Sequential(
        name="rnn_model_gru",
        layers=[
            tf.keras.layers.Input(shape=(INPUT_WIDTH, in_features)),
            tf.keras.layers.GRU(32, activation="relu", return_sequences=True),
            tf.keras.layers.GRU(16, activation="relu"),
            tf.keras.layers.Dense(
                OUT_STEPS * out_features,
            ),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([OUT_STEPS, out_features]),
        ],
    )


def multi_dense_model(OUT_STEPS,INPUT_WIDTH, in_features,out_features):
   # input shape is: [batch, time, features]
    return tf.keras.Sequential(
        name="multi_dense_model",
        layers=[
            tf.keras.layers.Input(shape=(INPUT_WIDTH, in_features)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            # Shape => [batch, out_steps*features]
            tf.keras.layers.Dense(
                OUT_STEPS * out_features,
            ),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([OUT_STEPS, out_features]),
        ],
    )


def cnn_layer(OUT_STEPS,INPUT_WIDTH, in_features,out_features):
    return tf.keras.Sequential(
        name="cnn_layer",
        layers=[
           tf.keras.layers.Input(shape=(INPUT_WIDTH,in_features)),
            # Conv1D expects input shape [batch, time, features]
             #   tf.keras.layers.Lambda(lambda x: tf.print("Input shape:", tf.shape(x)) or x),

            tf.keras.layers.Conv1D(
                filters=64, kernel_size=3, padding="causal", activation="relu"
            ),
            tf.keras.layers.Conv1D(
                filters=64, kernel_size=3, padding="causal", activation="relu"
            ),
            tf.keras.layers.Conv1D(
                filters=64, kernel_size=3, padding="causal", activation="relu"
            ),
            # Optional: Global average pooling or flattening
            tf.keras.layers.GlobalAveragePooling1D(),  # Shape => [batch, filters]
            tf.keras.layers.Dense(OUT_STEPS * out_features),
            # Reshape to desired output shape [batch, OUT_STEPS, num_features]
            tf.keras.layers.Reshape([OUT_STEPS, out_features]),
        ],
    )
from keras.saving import register_keras_serializable

@register_keras_serializable()
class AutoregressiveWrapperLSTM(tf.keras.Model):
    def __init__(self, OUT_STEPS, num_features, autoregression=True,**kwargs):
        super().__init__()
        self.out_steps = OUT_STEPS
        self.num_features = num_features
        self.autoregression = autoregression
        self.lstm_cell = tf.keras.layers.LSTMCell(32)
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(num_features)
        # CNN Layer
        #self.layer = input_layer
    def warmup(self, inputs):
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        x, *state = self.lstm_rnn(inputs)

        # predictions.shape => (batch, features)
        prediction = self.dense(x)
        return prediction, state
    @tf.function
    def call(self, inputs, training=None):

        if self.autoregression:
                # Use a TensorArray to capture dynamically unrolled outputs.
                predictions = []
                # Initialize the LSTM state.
                prediction, state = self.warmup(inputs)

                # Insert the first prediction.
                predictions.append(prediction)
                x = prediction
                # Run the rest of the prediction steps.
                for n in tf.range(1, self.out_steps):
                    # Use the last LSTM output (not dense output) as input
                    x, state = self.lstm_cell(x, states=state, training=training)
                    prediction = self.dense(x)
                    predictions.append(prediction)

                # predictions.shape => (time, batch, features)
                predictions = tf.stack(predictions)
                # predictions.shape => (batch, time, features)
                predictions = tf.transpose(predictions, [1, 0, 2])
                return predictions


        
        else:
            # If autoregression is off, just predict the entire sequence at once
            return self.layer(inputs, training=training)
    def get_config(self):
        config = super().get_config()
        config.update({
            "OUT_STEPS": self.out_steps,
            "num_features": self.num_features,
            "autoregression": self.autoregression,
          #  "input_layer": tf.keras.layers.serialize(self.layer),
        })
        return config

    @classmethod
    def from_config(cls, config):
        
        return cls( **config)
    
    
    
def generator(batch_size,OUT_STEPS, out_features):
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(OUT_STEPS*1, activation='relu'),
        tf.keras.layers.Reshape((OUT_STEPS, 1)),
    ])

def discriminator(INPUT_WIDTH, in_features):

    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(INPUT_WIDTH, in_features)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),

        tf.keras.layers.Dense(1, activation='relu'),
    ])    

    
@register_keras_serializable()
class GenerativeAdversialEncoderWrapper(tf.keras.Model):
    def __init__(self, OUT_STEPS,generator,discriminator, num_features, **kwargs):
        super().__init__()
        self.out_steps = OUT_STEPS
        self.num_features = num_features
        
        self.d_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.g_optimizer = tf.keras.optimizers.Adam(1e-4)
        
        self.generator = generator
        self.discriminator=discriminator
    def compile(self, optimizer, loss, **kwargs):
        super().compile(run_eagerly=True,loss=loss) 
        self.loss_fn = loss
        
    def call(self, inputs, training=None):
        return self.generator(inputs, training=training)

    def train_step(self, data):
        # Handle optional sample_weight
        if len(data) == 3:
            inputs, real_data, sample_weight = data
        else:
            inputs, real_data = data
            sample_weight = None

        batch_size = tf.shape(real_data)[0]
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_data = self.generator(inputs, training=True)
            real_output = self.discriminator(real_data, training=True)
            fake_output = self.discriminator(generated_data, training=True)

            gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
            
            disc_loss_real = self.loss_fn(tf.ones_like(real_output), real_output)
            disc_loss_fake = self.loss_fn(tf.zeros_like(fake_output), fake_output)

            # Apply sample weights if provided
            if sample_weight is not None:
                sample_weight = tf.cast(sample_weight, dtype=disc_loss_real.dtype)
                disc_loss_real *= sample_weight
                disc_loss_fake *= sample_weight
                # Average over batch
                disc_loss_real = tf.reduce_mean(disc_loss_real)
                disc_loss_fake = tf.reduce_mean(disc_loss_fake)

            disc_loss = disc_loss_real + disc_loss_fake

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return {
            "gen_loss": gen_loss,
            "disc_loss": disc_loss
        }



    def get_config(self):
        config = super().get_config()
        config.update({
            "OUT_STEPS": self.out_steps,
            "num_features": self.num_features,
            "generator": tf.keras.layers.serialize(self.generator),
            "discriminator": tf.keras.layers.serialize(self.discriminator),
        })
        return config

    @classmethod
    def from_config(cls, config):
        generator = tf.keras.layers.deserialize(config.pop("generator"))
        discriminator = tf.keras.layers.deserialize(config.pop("discriminator"))
        return cls(generator=generator, discriminator=discriminator, **config)
    def get_compile_config(self):
        return {
            "optimizer": tf.keras.optimizers.serialize(self.g_optimizer),
            "loss": tf.keras.losses.serialize(self.loss_fn),
           
        }

    def compile_from_config(self,compile_config):
        self.optimizer = tf.keras.optimizers.deserialize(compile_config["optimizer"])
        self.loss_fn = tf.keras.utils.deserialize_keras_object(compile_config["loss"])
     #   metrics = tf.keras.utils.deserialize_keras_object(compile_config["metric"])

        self.compile(optimizer=self.optimizer, loss=self.loss_fn,)# metrics=metrics)
        
        


class AutoregressiveWrapper:
    def __init__(self, model, out_steps, num_features, autoregression=True):
        """
        Parameters:
        - model: any object with `fit` and `predict` methods
        - out_steps: number of future steps to autoregressively predict
        - num_features: number of output features per step
        - autoregression: whether to apply autoregressive training/prediction
        """
        self.model = model
        self.out_steps = out_steps
        self.num_features = num_features
        self.autoregression = autoregression

    def fit(self, X, y, *args, **kwargs):
        """
        Fit model using autoregressive-style data preparation.
        X: [batch, time, features]
        y: [batch, time, features] — target sequence
        """
        if not self.autoregression:
            return self.model.fit(X, y, *args, **kwargs)

        X_ar = []
        y_ar = []

        for i in range(out_steps):  # iterate over batch
            input_seq = X[i]
            target_seq = y[i]

            # Generate multiple (input, target) pairs by sliding over time
            for t in range(len(input_seq) - self.out_steps):
                X_ar.append(input_seq[t : t + self.out_steps])
                y_ar.append(target_seq[t + 1 : t + 1 + self.out_steps])

        X_ar = np.array(X_ar)
        y_ar = np.array(y_ar)

        return self.model.fit(X_ar, y_ar, *args, **kwargs)

    def predict(self, input_seq, *args, **kwargs):
        if not self.autoregression:
            return self.model.predict(input_seq, *args, **kwargs)

        predictions = []
        current_input = input_seq

        for step in range(self.out_steps):
            pred = self.model.predict(current_input, *args, **kwargs)

            if pred.ndim == 2:
                pred = pred[:, np.newaxis, :]

            predictions.append(pred)

            current_input = np.concatenate([current_input[:, 1:], pred], axis=1)

        return np.concatenate(predictions, axis=1)

    def __getattr__(self, name):
        return getattr(self.model, name)

