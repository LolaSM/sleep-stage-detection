"""
Creación de archivos .npz a partir de los EDF de señal + scoring.
Cada .npz contendrá todas las ventanas del paciente + etiquetas one-hot.
"""

import os
import mne
import numpy as np
from tensorflow.keras.utils import to_categorical

# -------------------------------------------
# MAPEADO GLOBAL FIJO DE ETIQUETAS (0..4)
# -------------------------------------------
_GLOBAL_LABEL_MAPPING = {
    "W":  0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "R":  4,
}


# ------------------------------------------------------------------
#  utilidades de carga / preprocesado
# ------------------------------------------------------------------
def _cargar_datos(nombre_archivo: str, path: str) -> mne.io.Raw:
    """Lee señal y anotaciones de un paciente y devuelve un Raw de MNE."""
    archivo_senal   = os.path.join(path, f"{nombre_archivo}.edf")
    archivo_scoring = os.path.join(path, f"{nombre_archivo}_sleepscoring.edf")

    raw = mne.io.read_raw_edf(archivo_senal, preload=True, verbose=False)
    raw.set_annotations(mne.read_annotations(archivo_scoring))
    return raw


def _filtrar_resamplear(raw: mne.io.Raw,
                        fmin: float = .5,
                        fmax: float = 40.,
                        sfreq_nuevo: int = 100) -> mne.io.Raw:
    """Filtro pasa-banda + remuestreo in-place."""
    raw.filter(fmin, fmax, fir_design="firwin", verbose=False)
    raw.resample(sfreq_nuevo, verbose=False)
    return raw


# ------------------------------------------------------------------
#  función principal por paciente
# ------------------------------------------------------------------
def procesar_paciente(nombre_archivo: str,
                      path: str,
                      fs: int = 100,
                      dur_epoch: int = 30,
                      canales: list[str] | None = None):
    """
    Devuelve
    -------
    X : (n_ventanas, dur_epoch*fs, n_canales)   float32
    y : (n_ventanas, 5)                         float32  (one-hot)
    fs: int  (frecuencia de muestreo aplicada)
    """
    raw = _filtrar_resamplear(_cargar_datos(nombre_archivo, path), sfreq_nuevo=fs)

    # --- anotaciones de sueño ---
    sleep_annots = [
        (a["onset"], a["duration"], a["description"].split()[-1])
        for a in raw.annotations
        if a["description"].startswith("Sleep stage ")
           and a["description"].split()[-1] in _GLOBAL_LABEL_MAPPING
    ]
    if not sleep_annots:
        print(f"{nombre_archivo}: sin anotaciones Sleep stage -> omitido.")
        return None, None, None

    # --- canales ---
    if canales is None:
        canales = raw.info["ch_names"]
    datos = raw.copy().pick_channels(canales).get_data()   # (n_canales, n_muestras)
    n_chan, n_muestras = datos.shape
    muestras_por_epoca = dur_epoch * fs

    # --- ventana + etiquetas ---
    X, y_idx = [], []
    for onset, _, token in sleep_annots:
        ini = int(onset * fs)
        fin = ini + muestras_por_epoca
        if fin > n_muestras:
            continue
        ventana = datos[:, ini:fin]
        if ventana.shape[1] != muestras_por_epoca:
            continue
        X.append(ventana.T.astype("float32"))              # (time, chan)
        y_idx.append(_GLOBAL_LABEL_MAPPING[token])

    if not X:
        print(f"{nombre_archivo}: no se generaron ventanas válidas.")
        return None, None, None

    X = np.stack(X, axis=0)
    y = to_categorical(np.array(y_idx, dtype=np.int32), 5).astype("float32")
    print(f"{nombre_archivo}: {len(X)} ventanas generadas ({n_chan} canales).")
    return X, y, fs


# ------------------------------------------------------------------
#  serialización .npz por paciente
# ------------------------------------------------------------------
def _guardar_npz(identificador: str,
                 X: np.ndarray,
                 y: np.ndarray,
                 fs: int,
                 path_dest: str):
    os.makedirs(path_dest, exist_ok=True)
    np.savez(os.path.join(path_dest, f"{identificador}.npz"),
             identificador=identificador, X=X, y=y, fs=fs)


def procesamiento_completo(path_origen: str,
                           path_destino: str,
                           canales: list[str] | None = None):
    """
    Recorre todos los *_sleepscoring.edf de la carpeta `path_origen`
    y genera un .npz por paciente en `path_destino`.
    """
    archivos = [f for f in os.listdir(path_origen) if f.endswith("_sleepscoring.edf")]
    pacientes = sorted({f.split("_")[0] for f in archivos})

    for p in pacientes:
        print(f"\nProcesando paciente: {p}")
        X, y, fs = procesar_paciente(p, path_origen, canales=canales)
        if X is not None:
            _guardar_npz(p, X, y, fs, path_destino)
