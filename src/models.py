"""Keras architectures for spatial and sequential learning."""

from tensorflow import keras


def build_cnn():
    model = keras.Sequential(
        [
            keras.layers.Input((32, 32, 1)),
            keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            keras.layers.MaxPooling2D(2),
            keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            keras.layers.MaxPooling2D(2),
            keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.25),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_gru(lookback=36):
    model = keras.Sequential(
        [
            keras.layers.Input((lookback, 4)),
            keras.layers.GRU(32, activation="tanh", recurrent_activation="sigmoid"),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(1e-3, clipnorm=1.0), loss="mse")
    return model
