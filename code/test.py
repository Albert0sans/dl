import tensorflow as tf
from keras import backend as K
import vectorbt as vbt 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



import time

def differentiable_sum_binary_sequences(binary_array, steepness=100.0):
  """
  Approximates the total count of '1,0' and '0,1' sequences in a 1D TensorFlow
  array using differentiable operations.

  Args:
    binary_array: A 1D TensorFlow tensor. Values should ideally be close to 0 or 1.
    steepness: A float that controls how sharply the sigmoid approximates a step.
               Higher values mean a closer, but potentially less stable, approximation.

  Returns:
    A scalar TensorFlow tensor (float32) representing the approximate total count.
  """
  binary_array = tf.cast(binary_array, dtype=tf.float32)

  if tf.shape(binary_array)[0] < 2:
    return tf.constant(0.0, dtype=tf.float32)

  shifted_array_next = tf.roll(binary_array, shift=-1, axis=0)

  # Approximate "is 1" (value is near 1)
  is_one = tf.sigmoid(steepness * (binary_array - 0.5))
  # Approximate "is 0" (value is near 0)
  is_zero = tf.sigmoid(steepness * (0.5 - binary_array))

  # Approximate "1,0" sequence: (current is 1) AND (next is 0)
  # Logical AND is approximated by multiplication of sigmoid outputs
  approx_10_sequence = is_one[:-1] * tf.sigmoid(steepness * (0.5 - shifted_array_next[:-1]))

  # Approximate "0,1" sequence: (current is 0) AND (next is 1)
  approx_01_sequence = is_zero[:-1] * tf.sigmoid(steepness * (shifted_array_next[:-1] - 0.5))

  # Sum the approximate counts
  total_approx_sum = tf.reduce_sum(approx_10_sequence) + tf.reduce_sum(approx_01_sequence)

  return total_approx_sum

def profit_loss_differentiable(y_true, y_pred):
    """
    TensorFlow-only implementation of a proxy profit loss.
    Entry: 1 if y_pred > 0 (long), -1 if y_pred < 0 (short), else 0.
    Computes cumulative profit from trading based on binary signals.
    """
    fees = tf.constant(0.001, dtype=tf.float32) 

    # Apply the formula: ln(1 + X)
    # tf.math.log is the natural logarithm function in TensorFlow
    
    print(fees)
    
    shifted_y_pred = tf.concat([[-1.0], y_pred[:-1]], axis=0)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(shifted_y_pred, tf.float32)
    
    entries = tf.sigmoid(10000000000 * (y_pred))
    ops=differentiable_sum_binary_sequences(entries)
    print(f"ops {ops}")
    fees = tf.math.log(1.0 - fees)*(ops)


    realized_returns = entries * y_true

   
    total_profit = (tf.exp(tf.reduce_sum(realized_returns)+fees)-1 ) * 100.0
    total_return_factor = tf.exp(tf.reduce_sum(y_true[1:]))

    total_return = (total_return_factor - 1) * 100
    return total_profit,total_return

def simplebacktest(y_true_log_returns, y_pred_log_returns) -> tuple[float, float,]:
    # Convert log returns to price series
    fees=0.001
    prices = 100*np.exp(np.cumsum(y_true_log_returns))
    


    # Entry and exit signals based on predicted returns
    entries = y_pred_log_returns > 0.0
    exits = y_pred_log_returns < 0.0  # <- negative returns
    
    # Vectorbt Portfolio
    pf = vbt.Portfolio.from_signals(prices, entries, exits, init_cash=100,direction='longonly',fees=fees)
    
   
    # Vectorbt stats
    stats = pf.stats(silence_warnings=True)

    return stats["Total Return [%]"], stats["Benchmark Return [%]"]
    
    






def profit_loss_numpy(y_true, y_pred)->tuple[float,float]:
   
    y_pred[0] = -1  # Or any fill value you want for the first position
    y_pred[1:] = y_pred[:-1]
    entries = y_pred > 0.0
    exits = y_pred < 0.0  # <- negative returns
    realized_returns = entries * y_true

    
    total_profit = (np.exp(np.sum(realized_returns)) -1) * 100.0
    total_return_factor = np.exp(np.sum(y_true[1:]))

    total_return = (total_return_factor - 1) * 100
    return total_profit ,total_return

    
    

if __name__ == '__main__':
    
 
    for i in range(0,100):
        # 1. Generate dummy data for actual log returns (y_true)
        # Let's simulate 100 periods of log returns
        num_periods = 10000
        # Simulate real prices
        actual_log_returns = np.random.randn(num_periods) * 0.05  # Small random log returns
    
        predicted_signals = np.where(actual_log_returns > 0, 1,-1)
        

        # Convert to TensorFlow tensors for the loss function
        tf_y_true = tf.constant(actual_log_returns, dtype=tf.float32)
        tf_y_pred = tf.constant(predicted_signals, dtype=tf.float32)

        # 3. Calculate the differentiable_profit_loss
        # We need to run this in eager mode, as it's not part of a compiled graph here.
        with tf.GradientTape() as tape: # Using GradientTape to ensure eager execution context
            tape.watch([tf_y_true, tf_y_pred]) # Not strictly needed for loss calculation, but good practice
            total_keras,bench_keras = profit_loss_differentiable(tf_y_true, tf_y_pred)


        total,bench=simplebacktest(actual_log_returns,predicted_signals)
        print(f"total {total} bench {bench}, res={total/bench}") 
        print(f"total_keras {total_keras.numpy()} bench_keras {bench_keras.numpy()} res={total_keras.numpy()/bench_keras.numpy()}")
        time.sleep(1)
        assert np.allclose(total_keras.numpy(), total), f"Mismatch: {total_keras.numpy()} vs {total}"
        assert np.allclose(bench_keras.numpy(), bench), f"Mismatch: {bench_keras.numpy()} vs {bench}"

    