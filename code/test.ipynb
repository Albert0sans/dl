import tensorflow as tf
import time
import matplotlib.pyplot as plt

cpu_times = []
gpu_times = []
sizes = [1, 10, 100, 500, 1000, 2000, 3000, 4000, 5000, 8000, 10000,50000]

# Check if a GPU is available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    gpu_available = True
    print("GPU is available, proceeding with GPU computations.")
else:
    gpu_available = False
    print("No GPU found, skipping GPU computations.")
    gpu_times = [0.0] * len(sizes) # Initialize with zeros for plotting

for size in sizes:
    # CPU computation
    tf.keras.backend.clear_session()
    start_cpu = time.time()
    with tf.device('CPU:0'):
        v1_cpu = tf.Variable(tf.random.normal((size, size)))
        v2_cpu = tf.Variable(tf.random.normal((size, size)))
        op_cpu = tf.matmul(v1_cpu, v2_cpu)
        # No explicit session needed in TF 2.x for eager execution
        result_cpu = op_cpu.numpy() # Execute the operation and get the NumPy array

    cpu_times.append(time.time() - start_cpu)
    print(f'CPU time for size {size}: {time.time() - start_cpu:.4f} sec')

    # GPU computation (if available)
    if gpu_available:
        tf.keras.backend.clear_session()
        start_gpu = time.time()
        with tf.device('GPU:0'):
            v1_gpu = tf.Variable(tf.random.normal((size, size)))
            v2_gpu = tf.Variable(tf.random.normal((size, size)))
            op_gpu = tf.matmul(v1_gpu, v2_gpu)
            # No explicit session needed in TF 2.x for eager execution
            result_gpu = op_gpu.numpy() # Execute the operation and get the NumPy array

        gpu_times.append(time.time() - start_gpu)
        print(f'GPU time for size {size}: {time.time() - start_gpu:.4f} sec')

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(sizes, cpu_times, label='CPU')
if gpu_available:
    ax.plot(sizes, gpu_times, label='GPU')
plt.xlabel('MATRIX SIZE')
plt.ylabel('TIME (sec)')
plt.legend()
plt.title('CPU vs GPU Matrix Multiplication Time')
plt.grid(True)
plt.show()