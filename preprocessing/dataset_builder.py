"""
Generador tf.data.Dataset que ensambla secuencias de L ventanas
sin cargar todos los pacientes a memoria.
"""

from __future__ import annotations
import os
import numpy as np
import tensorflow as tf


# --------------------------------------------------------------
# utilidades básicas
# --------------------------------------------------------------
def _cargar_npz(path_npz):
    d = np.load(path_npz, allow_pickle=True)
    return d["X"], d["y"], d["identificador"].item()


def _split_pacientes(path_folder: str, seed: int = 42):
    archivos = sorted([os.path.join(path_folder, f)
                       for f in os.listdir(path_folder) if f.endswith(".npz")])
    rng = np.random.default_rng(seed)
    rng.shuffle(archivos)

    n = len(archivos)
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)

    return archivos[:n_train], archivos[n_train:n_train+n_val], archivos[n_train+n_val:]


def _gen_sequences(list_npz: list[str], L: int):
    for npz in list_npz:
        X, y, _ = _cargar_npz(npz)
        if len(X) <= L:
            continue
        for i in range(L, len(X)):
            yield X[i-L:i].astype("float32"), y[i].astype("float32")


# --------------------------------------------------------------
# API principal
# --------------------------------------------------------------
def crear_tf_dataset(path_npz_folder: str,
                     L: int = 5,
                     batch_size: int = 32,
                     shuffle_buffer: int = 1000,
                     repeat: bool = True,
                     seed: int = 42):
    """
    Devuelve (train_ds, val_ds, test_ds).
    - Si repeat=True, train_ds.repeat() para epoch infinito.
    - Cada item: (L, time, chan) , (5,)
    """
    train_ids, val_ids, test_ids = _split_pacientes(path_npz_folder, seed)

    # ejemplo para shapes dinámicas
    ejemplo_X, _, _ = _cargar_npz(train_ids[0])
    time_steps, n_chan = ejemplo_X.shape[1], ejemplo_X.shape[2]
    out_seq_shape = (L, time_steps, n_chan)
    out_lbl_shape = (5,)

    def _make_ds(file_list, shuffle=False, rep=False):
        gen = lambda: _gen_sequences(file_list, L)
        ds = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(out_seq_shape, tf.float32),
                tf.TensorSpec(out_lbl_shape, tf.float32)
            )
        )
        if shuffle:
            ds = ds.shuffle(shuffle_buffer, seed=seed)
        if rep:
            ds = ds.repeat()
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds = _make_ds(train_ids, shuffle=True, rep=repeat)
    val_ds   = _make_ds(val_ids,   shuffle=False, rep=False)
    test_ds  = _make_ds(test_ids,  shuffle=False, rep=False)
    return train_ds, val_ds, test_ds
