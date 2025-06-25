"""
Arquitectura CNN-LSTM parametrizable + utilidades de compilación y callbacks.
"""

import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv1D, ReLU, BatchNormalization,
                                     AveragePooling1D, GlobalAveragePooling1D,
                                     Dense, TimeDistributed, LSTM)
from tensorflow.keras.models import Model
from modelos.metrics import F1Score

# ------------------------------------------------------------------
#  builder flexible
# ------------------------------------------------------------------
def build_cnn_lstm_flexible(seq_len: int,
                            input_shape: tuple[int, int],
                            *,
                            filters_init: int = 8,
                            kernel_size: int = 100,
                            lstm_units: int = 64,
                            dense_units: int = 50) -> Model:
    """
    Parameters
    ----------
    seq_len      : nº de ventanas consecutivas (L)
    input_shape  : (time_steps, n_chan)
    filters_init : filtros primer bloque Conv1D
    kernel_size  : tamaño kernel Conv1D
    lstm_units   : unidades de la capa LSTM
    dense_units  : tamaño del embedding intermedio
    """
    # --- Encoder CNN ---
    inp_cnn = Input(shape=input_shape)
    x = inp_cnn
    filters = filters_init
    for _ in range(3):
        x = Conv1D(filters, kernel_size=kernel_size, padding="same")(x)
        x = ReLU()(x)
        x = BatchNormalization()(x)
        x = AveragePooling1D(2)(x)
        filters *= 2
    x = GlobalAveragePooling1D()(x)
    x = Dense(dense_units, activation="relu")(x)
    encoder = Model(inp_cnn, x, name="Encoder")

    # --- Secuencia ---
    seq_in = Input(shape=(seq_len, *input_shape))      # (L, time, chan)
    y = TimeDistributed(encoder)(seq_in)               # (L, dense_units)
    y = LSTM(lstm_units)(y)
    out = Dense(5, activation="softmax", dtype="float32")(y)

    return Model(seq_in, out, name="CNN_LSTM")


# ------------------------------------------------------------------
#  compilación & callbacks
# ------------------------------------------------------------------
def compile_model(model: Model,
                  *,
                  optimizer_name: str = "adam",
                  lr: float = 1e-3) -> Model:
    if optimizer_name.lower() == "sgd":
        opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    else:  # adam por defecto
        opt = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(optimizer=opt,
                  loss="categorical_crossentropy",
                  metrics=["accuracy", F1Score()])
    return model


def get_callbacks():
    """EarlyStopping + ReduceLROnPlateau."""
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                          patience=15,
                                          verbose=1,
                                          restore_best_weights=True)
    rlrop = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                 patience=7,
                                                 factor=0.5,
                                                 min_lr=1e-6,
                                                 verbose=1)
    return [es, rlrop]
