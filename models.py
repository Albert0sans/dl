    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(
                128, activation=tf.keras.layers.LeakyReLU(alpha=0.01)
            ),
            tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
            tf.keras.layers.Dense(output_dim),
        ]