import pickle
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
from pickle import dump,load
import os
from scipy.stats import ttest_ind

from tensorflow.keras import Model as KerasModel
from sklearn.base import BaseEstimator
tf.keras.config.enable_unsafe_deserialization()
import numpy as np
import vectorbt as vbt
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))



class MultiClassModel:
    def __init__(
        self,
        model,
        model_name,
        target_indices,
        retrain=False,
        epochs=10,
        batch_size=32,

    ):
        self.retrain = retrain
        self.model = model
        self.model_name = model_name
        self.target_indices = target_indices
        self.epochs = epochs
        self.batch_size=batch_size
        
        # Assertions for target_indices and generators
        assert isinstance(self.target_indices, list) and len(self.target_indices) < 1000 # Adjusted assertion as X_train is not available
       


        self.sample_weights = None 


        self.model_path = f"./models/{model_name}.keras"
                        

        
    def evaluate(self,test_dataset):
        
                # Keras prediction on a dataset directly
        eval_ds,_=get_data_set(test_dataset)
        eval_ds=eval_ds.batch(self.batch_size)
        values=self.model.evaluate(eval_ds)
              
        metrics_dict = dict(zip(self.model.metrics_names, values))
        return metrics_dict
          
    def fit(self,train_generator_fn,val_generator_fn, **kwargs):
        assert (train_generator_fn is not None)  and (val_generator_fn is not None)
        
        train_generator,train_samples=get_data_set(train_generator_fn)
        
        
        val_generator,val_samples=get_data_set(val_generator_fn)

        
        train_generator_ds=train_generator.batch(32).repeat()
        val_generator_ds=val_generator.batch(32).repeat()

       
        self.train_steps = train_samples // self.batch_size
        self.val_steps = val_samples // self.batch_size
        model = None
        history = None
        history_path = os.path.splitext(self.model_path)[0] + "_history.pkl"

        # Try loading model and history if not retraining
        if not self.retrain:
            try:
                model = load_keras_model(model_path=self.model_path)
                if os.path.exists(history_path):
                    with open(history_path, "rb") as f:
                        history = pickle.load(f)
                    print("Model and history loaded.")
                else:
                    print("Model loaded, but no history found.")
            except Exception as e:
                print(f"Could not load model: {e}")

        if model is not None:
            self.model = model
            return history

        # Compile model
        self.model.compile(
            loss="mse",
            optimizer=tf.optimizers.Adam(learning_rate=1e-5, clipnorm=1.0),
            metrics=[tf.metrics.MeanAbsoluteError()],
        )

        # Define callbacks
       # early_stopping = tf.keras.callbacks.EarlyStopping(
       #     monitor="val_loss",
       #     patience=6,
       #     restore_best_weights=True
       # )

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.model_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=0
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=3,
            min_lr=1e-5
        )

        # Fit model
        history_obj = self.model.fit(
            train_generator_ds,
            epochs=self.epochs,
            validation_data=val_generator_ds,
            callbacks=[
                       #arly_stopping, 
                       checkpoint,
                       reduce_lr],
            verbose=1,
            validation_steps=self.val_steps,
            steps_per_epoch=self.train_steps,
            **kwargs
        )
        
        print(history_obj)
        # Save history
        
        with open(history_path, "wb") as f:
            pickle.dump(history_obj, f)

        return history_obj

   

    def predict(self,data):
        
        
        data_generator,_=get_data_set(data)
        data_generator=data_generator.batch(self.batch_size)
        pred = self.model.predict(data_generator,verbose=0)
        if(pred.shape[-1] > len(self.target_indices)):
            try:
                return pred[:,:,self.target_indices]
            except:
                pass
        return pred



tf.keras.config.enable_unsafe_deserialization()

def load_keras_model(model_path):
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        new_model = tf.keras.models.load_model(model_path, custom_objects={'stock_return_loss': stock_return_loss})
        return new_model
    return False
    
def save_pickle_model(model,model_path):
    with open(model_path,"wb+") as f:
        dump(model,f,protocol=5)

def load_pickle_model(model_path):
    if os.path.exists(model_path):
        with open(model_path,"rb") as f:
                return load(f)
    return False

def simplebacktest(y_true,y_pred)-> tuple[float, float, float, float, float]:
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    entries = y_pred > 0.0
    exits = y_pred < -0.0
    
    pf = vbt.Portfolio.from_signals(y_true, entries, exits, init_cash=100,fees=0.001,sl_stop=0.05,tp_stop=0.1)
    stats = pf.stats(silence_warnings=True)
    win_rate = stats["Win Rate [%]"]
    avg_losing = stats["Avg Losing Trade [%]"]
    avg_winning = stats["Avg Winning Trade [%]"]
    benchmark_return = stats["Benchmark Return [%]"]
    total_return = stats["Total Return [%]"]
    return total_return - benchmark_return

def differentiable_sum_binary_sequences(binary_array, steepness=100.0):
  binary_array = tf.cast(binary_array, dtype=tf.float32)
  shifted_array_next = tf.roll(binary_array, shift=-1, axis=0)

  is_one = tf.sigmoid(steepness * (binary_array - 0.5))
  is_zero = tf.sigmoid(steepness * (0.5 - binary_array))

  approx_10_sequence = is_one[:-1] * tf.sigmoid(steepness * (0.5 - shifted_array_next[:-1]))
  approx_01_sequence = is_zero[:-1] * tf.sigmoid(steepness * (shifted_array_next[:-1] - 0.5))

  total_approx_sum = tf.reduce_sum(approx_10_sequence) + tf.reduce_sum(approx_01_sequence)
  return total_approx_sum

def profit_loss_differentiable(y_true, y_pred):
    fees = tf.constant(0.001, dtype=tf.float32) 
    batch_size = tf.shape(y_pred)[0]
    num_features = tf.shape(y_pred)[2]

    # This part might need adjustment depending on how your y_pred is structured for time series
    # and how you want to handle the "initial trade signal".
    # The original code's `shifted_y_pred` seems to be trying to bring values from the future.
    # If `y_pred` is (batch, sequence_length, features), shifting `y_pred` itself might be more appropriate.
    # For a general solution, let's assume `y_pred` is aligned with `y_true`.
    
    # If the intention was to shift y_pred for entries/exits logic,
    # consider how `initial_trade_signal` is used or if `tf.roll` on `y_pred` is sufficient.
    # For now, I'll simplify `shifted_y_pred` to just `y_pred` for clarity in this context.
    # If your model predicts at time `t` for `t+1`, then `y_pred` might need to be shifted to align with `y_true` (actual returns at `t+1`).

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32) # Using original y_pred for signals

    entries = tf.sigmoid(10000000000 * (y_pred)) # Approximating binary entry decision
    ops = differentiable_sum_binary_sequences(tf.squeeze(entries, axis=-1)) # Squeeze if entries has a feature dimension of 1
    fees_incurred = tf.math.log(1.0 - fees) * ops

    # Realized returns should align with actual price movements when a position is open
    # This simplified version assumes `entries` directly gates `y_true`.
    realized_returns = entries * y_true
    
    total_profit = (tf.exp(tf.reduce_sum(realized_returns) + fees_incurred) - 1) * 100.0
    
    # Calculate a benchmark return for comparison if needed
    total_return_factor = tf.exp(tf.reduce_sum(y_true))
    total_return = (total_return_factor - 1) * 100
    
    # Return negative of profit/benchmark to minimize (maximize profit)
    # Handle cases where total_return might be zero to avoid division by zero
    return -tf.reduce_mean(total_profit / (total_return + tf.keras.backend.epsilon()))

def profit_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    entries = tf.where(y_pred > 0, 1.0, tf.where(y_pred < 0, -1.0, 0.0))
    exits = tf.where(y_pred > 0, 0.0, tf.where(y_pred < 0, -1.0, 0.0))

    transitions = tf.cast(entries - exits, tf.float32)
    in_market = tf.clip_by_value(tf.cumsum(transitions), 0.0, 1.0)
    realized_returns = in_market * y_true
    total_profit = (tf.exp(tf.reduce_sum(realized_returns)) - 1.0) * 100.0
    return -total_profit

from tensorflow.keras import backend as K

from keras import saving
@saving.register_keras_serializable()
def stock_return_loss(y_true, y_pred):
      # Asegúrate de que y_true y y_pred sean tensores de TensorFlow
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Calcula el signo de cada valor
    sign_true = tf.sign(y_true)
    sign_pred = tf.sign(y_pred)

    # Si y_true es 0, podemos definir el signo de y_pred como "correcto"
    # si también es 0, o dejar que el modelo aprenda a predecir 0.
    # Para simplificar, asumiremos que los valores reales no son exactamente 0
    # o que 0 se considera un "signo" neutro que no penaliza si la predicción es 0.

    # Penalización si los signos son diferentes
    # Usamos tf.not_equal para obtener un booleano (True si son diferentes, False si son iguales)
    # y luego tf.cast para convertirlo a float (1.0 si son diferentes, 0.0 si son iguales).
    # Multiplicamos por un factor de penalización si los signos no coinciden.
    sign_mismatch_penalty = tf.cast(tf.not_equal(sign_true, sign_pred), tf.float32) * 10.0 # Ajusta el peso según la importancia

    # También puedes añadir una pequeña penalización por la magnitud si lo deseas,
    # pero con un peso mucho menor que la penalización por el signo.
    # Por ejemplo, un MSE/MAE muy pequeño para cuando los signos coinciden.
    magnitude_error = tf.square(y_true - y_pred) # O tf.abs(y_true - y_pred)
    
    # Combinar las penalizaciones
    # Solo aplica la penalización de magnitud si los signos son los mismos.
    # Si los signos son diferentes, la penalización principal es por el signo.
    
    # Crear una máscara donde los signos son iguales
    signs_are_equal = tf.cast(tf.equal(sign_true, sign_pred), tf.float32)
    
    # Penalización total: gran penalización si los signos no coinciden,
    # y una penalización de magnitud (pequeña) si los signos coinciden.
    loss = (sign_mismatch_penalty) + (magnitude_error * signs_are_equal * 0.1) # 0.1 es un peso pequeño para la magnitud

    return tf.reduce_mean(loss)
        
        
def get_data_set(generator_fn)->tuple[tf.data.Dataset,int]:
    data_tmp=generator_fn()
    sample_X_train, sample_y_train = next(iter(data_tmp))
    

    ds= tf.data.Dataset.from_generator(
                    generator_fn,
                    output_signature=(
                        tf.TensorSpec(shape=(sample_X_train.shape), dtype=tf.float32),
                        tf.TensorSpec(shape=(sample_y_train.shape), dtype=tf.float32)
                    )
                )
    total=sum(1 for _ in data_tmp)+1

    return ds,total
    