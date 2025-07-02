"""
Script único:   python3 train_cnn_lstm_model.py
1) Asegura .npz (procesado)            2) Grid-Search 20 ep + ES(5)
3) Entrena modelo final con hp óptimos 4) Evalúa en test
"""

from __future__ import annotations
import os, itertools, json, pickle, time

# Configuración para usar múltiples núcleos de CPU
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())
os.environ["TF_INTRA_OP_PARALLELISM_THREADS"] = str(os.cpu_count())
os.environ["TF_INTER_OP_PARALLELISM_THREADS"] = str(max(1, os.cpu_count() // 2))

# Configuración para resolver problema de libdevice missing
import subprocess

# Intentar encontrar el directorio de CUDA automáticamente
def find_cuda_dir():
    try:
        # Buscar en las rutas del sistema directamente para libdevice.10.bc
        for search_path in [
            "/usr/local/cuda", 
            "/usr/local/cuda-*",
            "/usr/cuda",
            "/usr/cuda-*",
            "/opt/cuda", 
            "/opt/cuda-*"
        ]:
            # Expandir comodines si los hay
            import glob
            for cuda_path in glob.glob(search_path):
                # Verificar si existe el archivo libdevice específicamente
                libdevice_path = os.path.join(cuda_path, "nvvm", "libdevice")
                if os.path.exists(os.path.join(libdevice_path, "libdevice.10.bc")):
                    return cuda_path
                
        # Intentar encontrar nvcc para determinar el directorio de CUDA
        try:
            nvcc_path = subprocess.check_output(["which", "nvcc"]).decode().strip()
            if nvcc_path:
                # Normalmente nvcc está en /usr/local/cuda/bin/nvcc
                cuda_path = os.path.dirname(os.path.dirname(nvcc_path))
                # Verificar si existe el directorio de libdevice
                if os.path.exists(os.path.join(cuda_path, "nvvm", "libdevice")):
                    return cuda_path
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
    except Exception as e:
        print(f"Error al buscar CUDA: {e}")
    
    return None

# Configurar XLA_FLAGS para encontrar libdevice
cuda_dir = find_cuda_dir()
libdevice_found = False

if cuda_dir and os.path.exists(os.path.join(cuda_dir, "nvvm", "libdevice")):
    os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={cuda_dir}"
    print(f" CUDA directorio configurado: {cuda_dir}")
    libdevice_found = True
else:
    # Buscar en rutas adicionales donde podría estar libdevice
    print(" No se pudo encontrar el directorio de CUDA con libdevice")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping

# --- utilidades propias ---
from modelos.cnn_lstm_model import (build_cnn_lstm_flexible,
                                    compile_model,
                                    get_callbacks)
from preprocessing.dataset_builder import crear_tf_dataset
from preprocessing.procesado_pacientes import procesamiento_completo
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score


# --------------------------------------------------------------
#  GPU SET-UP
# --------------------------------------------------------------
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    # Configurar todas las GPUs disponibles
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    try:
        tf.config.experimental.set_memory_allocator("cuda_malloc_async")
    except (AttributeError, ValueError):
        pass
    
    # Configurar precision basado en si encontramos libdevice
    from tensorflow.keras import mixed_precision
    if libdevice_found:
        mixed_precision.set_global_policy("mixed_float16")
        precision_mode = "mixed_float16 ON"
    else:
        # Si no se encuentra libdevice, usar float32 para evitar errores de JIT
        mixed_precision.set_global_policy("float32")
        precision_mode = "float32 (libdevice no encontrado)"
    
    # Estrategia para múltiples GPUs (independiente de la precisión)
    strategy = tf.distribute.MirroredStrategy()
    print(f"  GPUs disponibles: {len(gpus)} — {precision_mode}")
    print(f"  Estrategia de distribución: {strategy.__class__.__name__} con {strategy.num_replicas_in_sync} réplicas")
else:
    strategy = tf.distribute.get_strategy()  # Estrategia por defecto (para CPU)
    print("  GPU no detectada: se usará CPU")

# --------------------------------------------------------------
#  CONFIGURACIÓN
# --------------------------------------------------------------
RECORDINGS_PATH = "recordings"
DATASET_PATH    = "pacientes"

SEQ_LENGTH   = 10 
BATCH_SIZE   = 64
EPOCHS_FINAL = 100 #Cambio de 50 a 100
MODEL_PATH   = "modelos/CNN_LSTM_100_epochs.keras"
#-------EXPERIMENTO I-------
CANALES_INTERES = ["ECG"]
#-------EXPERIMENTO II-------
#CANALES_INTERES = ["EEG C4-M1", "ECG"] 
#-------EXPERIMENTO III-------
#CANALES_INTERES = ["EOG E1-M2", "EOG E2-M2", "ECG"] 
#-------EXPERIMENTO IV-------
#CANALES_INTERES = ["EOG E1-M2", "EOG E2-M2", "EEG C4-M1", "EMG chin"] 

# --------------------------------------------------------------
#  1. Preprocesar pacientes si faltan .npz
# --------------------------------------------------------------
if not any(f.endswith(".npz") for f in os.listdir(DATASET_PATH)):
    print("No se encontraron .npz  ->  lanzando procesamiento completo…")
    procesamiento_completo(RECORDINGS_PATH, DATASET_PATH, canales=CANALES_INTERES)
else:
    print(f"{len([f for f in os.listdir(DATASET_PATH) if f.endswith('.npz')])} archivos .npz detectados.")

# --------------------------------------------------------------
#  2. Dataset para Grid-Search (con repeat para compatibilidad con distribución)
# --------------------------------------------------------------
# Crear datasets con repeat=True para distribución 
train_ds_gs, val_ds_gs, _ = crear_tf_dataset(DATASET_PATH,
                                             L=SEQ_LENGTH,
                                             batch_size=BATCH_SIZE,
                                             shuffle_buffer=1000,
                                             repeat=True)

# Determinar el número de steps por época para limitar el entrenamiento
train_npz_files = [f for f in os.listdir(DATASET_PATH) if f.endswith('.npz')]
train_size = len(train_npz_files) * 800  # Estimación aproximada del total de ventanas
steps_per_epoch = train_size // (BATCH_SIZE * strategy.num_replicas_in_sync)  # Ajustar por número de réplicas

input_shape_window = val_ds_gs.element_spec[0].shape[2:]   # (time, chan)

# --------------------------------------------------------------
#  3. Grid-Search con checkpoint para continuar si se interrumpe
# --------------------------------------------------------------
# Grid search simplificado con menos combinaciones
# Fijamos algunos parámetros basados en resultados anteriores
param_grid = {
    "filters_init": [8],         # Fijamos en 8 basado en ejecuciones anteriores
    "kernel_size" : [100],       # Fijamos en 100 basado en ejecuciones anteriores
    "lstm_units"  : [64, 100],    
    "dense_units" : [50, 100],   
    "dropout_rate": [0, 0.2],    
    "optimizer"   : ["adam"],    
    "lr"          : [5e-4, 1e-4], 
}

def combinations(d):
    keys, vals = zip(*d.items())
    for prod in itertools.product(*vals):
        yield dict(zip(keys, prod))

# Archivo para guardar resultados parciales
GRID_RESULTS_FILE = "grid_search_results.json"

# Comprobar si hay resultados previos y preguntar si continuar
results = []
start_idx = 1
continuar_grid_search = True

if os.path.exists(GRID_RESULTS_FILE):
    try:
        with open(GRID_RESULTS_FILE, 'r') as f:
            results = json.load(f)
        if results:
            # Encontrar el último índice procesado basado en la cantidad de resultados guardados
            start_idx = len(results) + 1
            print(f"\n Se encontraron {len(results)} configuraciones ya evaluadas")
            print("\nResultados disponibles:")
            for idx, res in enumerate(results, 1):
                print(f"{idx:02d}. {res}")
    except Exception as e:
        print(f"Error al cargar resultados previos: {e}")
        results = []

# Generar todas las combinaciones de hiperparámetros antes de empezar
all_combinations = list(combinations(param_grid))
total_combinations = len(all_combinations)

print(f"\n GRID SEARCH (máx 20 ep / ES=5) ")
print(f"\nTotal: {total_combinations} configuraciones. Comenzando desde la #{start_idx}")

for i, hp in enumerate(all_combinations[start_idx-1:], start_idx):
    tf.keras.backend.clear_session()
    print(f"\n  {i:02d}/{total_combinations}. {hp}")
    with strategy.scope():
        model = build_cnn_lstm_flexible(seq_len=SEQ_LENGTH,
                                      input_shape=input_shape_window,
                                      filters_init=hp["filters_init"],
                                      kernel_size=hp["kernel_size"],
                                      lstm_units=hp["lstm_units"],
                                      dense_units=hp["dense_units"],
                                      dropout_rate=hp["dropout_rate"])
        model = compile_model(model, optimizer_name=hp["optimizer"], lr=hp["lr"])

    # Con datasets repetidos, limitamos los pasos por época
    history = model.fit(
        train_ds_gs,
        validation_data=val_ds_gs,
        epochs=20,
        steps_per_epoch=steps_per_epoch,
        validation_steps=steps_per_epoch // 5,  # Aproximadamente 20% para validación
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True,
                                 monitor="val_loss")],
        verbose=1,  # Cambiado a 1 para ver progreso
    )
    best_val = min(history.history["val_loss"])
    
    # Guardar esta configuración en los resultados
    hp_result = {**hp, "val_loss": best_val}
    results.append(hp_result)
    
    # Guardar resultados parciales en disco después de cada configuración
    print(f"\n Guardando resultados parciales ({len(results)}/{total_combinations})")
    try:
        with open(GRID_RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"Error al guardar resultados: {e}")
    
    # Imprimir el mejor resultado hasta ahora
    best_so_far = sorted(results, key=lambda d: d["val_loss"])[0]
    print(f"\n Mejor configuración hasta ahora: {best_so_far}")

# Obtener la mejor configuración
if results:
    best_hp = sorted(results, key=lambda d: d["val_loss"])[0]
    print("\n Mejor configuración final:", best_hp)
else:
    print("\n No hay resultados de grid search disponibles. No se puede continuar.")
    exit(1)

# --------------------------------------------------------------
#  4. Dataset definitivo (con repeat=True para distribución)
# --------------------------------------------------------------

# Preguntar si se desea saltar directamente al entrenamiento final
SKIP_TO_FINAL_FILE = ".skip_to_final"

# Comprobar si existe el archivo para saltar directo al entrenamiento final
skip_to_final = os.path.exists(SKIP_TO_FINAL_FILE)
if skip_to_final:
    print("\n Saltando directamente al entrenamiento final...")
    os.remove(SKIP_TO_FINAL_FILE)  # Eliminar el archivo para futuras ejecuciones

# Crear datasets para entrenamiento final
train_ds, val_ds, test_ds = crear_tf_dataset(DATASET_PATH,
                                             L=SEQ_LENGTH,
                                             batch_size=BATCH_SIZE,
                                             shuffle_buffer=1000,
                                             repeat=True)

# Calcular steps para el entrenamiento final
final_steps_per_epoch = steps_per_epoch  # Ya calculado anteriormente

# --------------------------------------------------------------
#  5. Entrenamiento final
# --------------------------------------------------------------
with strategy.scope():
    model = build_cnn_lstm_flexible(seq_len=SEQ_LENGTH,
                                  input_shape=input_shape_window,
                                  filters_init=best_hp["filters_init"],
                                  kernel_size=best_hp["kernel_size"],
                                  lstm_units=best_hp["lstm_units"],
                                  dense_units=best_hp["dense_units"],
                                  dropout_rate=best_hp["dropout_rate"])
    model = compile_model(model,
                        optimizer_name=best_hp["optimizer"],
                        lr=best_hp["lr"])

# Definir pesos para compensar el desbalance de clases (valores menos extremos)
class_weights = {
    0: 1.0,     # W
    1: 2.5,     # N1 - peso alto pero menos extremo 
    2: 0.8,     # N2 - menos reducción para la clase mayoritaria 
    3: 1.3,     # N3 - ligeramente reducido
    4: 1.2      # R - normalizado
}

print("\n\u2728 Usando pesos de clase para manejar el desbalance:\n", class_weights)

# Con datasets repetidos, limitamos los pasos por época en el entrenamiento final
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINAL,
    steps_per_epoch=final_steps_per_epoch,
    validation_steps=final_steps_per_epoch // 5,  # Mismo ratio que en grid search
    callbacks=get_callbacks() + [CSVLogger("training_log.csv", append=False)],
    class_weight=class_weights,  # Añadir pesos de clase
    verbose=2,
)

model.save(MODEL_PATH)
print(f"\nModelo guardado en → {MODEL_PATH}")

# --------------------------------------------------------------
#  6. Gráficas de entrenamiento
# --------------------------------------------------------------
# Crear directorio para gráficos si no existe
graficos_dir = "graficos"
os.makedirs(graficos_dir, exist_ok=True)

# Guardar gráficos de entrenamiento
plt.figure(figsize=(14,5))
plt.subplot(1,2,1); plt.plot(history.history["loss"]); plt.plot(history.history["val_loss"]); plt.title("Loss"); plt.grid(True); plt.legend(["train","val"])
plt.subplot(1,2,2); plt.plot(history.history["accuracy"]); plt.plot(history.history["val_accuracy"]); plt.title("Accuracy"); plt.grid(True); plt.legend(["train","val"])
plt.tight_layout()
# Guardar en archivo en lugar de mostrar
grafico_entrenamiento = os.path.join(graficos_dir, "entrenamiento_metricas.png")
plt.savefig(grafico_entrenamiento, dpi=300)
plt.close()
print(f"\nGráfico de métricas de entrenamiento guardado en → {grafico_entrenamiento}")

# --------------------------------------------------------------
#  7. Evaluación en test
# --------------------------------------------------------------
y_true, y_pred = [], []
for xb, yb in test_ds:
    pr = model.predict(xb, verbose=0)
    y_pred.append(pr); y_true.append(yb.numpy())

y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred)
true_lbl = y_true.argmax(1); pred_lbl = y_pred.argmax(1)

print("\nClassification Report:\n",
      classification_report(true_lbl, pred_lbl, target_names=["W","N1","N2","N3","R"]))

cm = confusion_matrix(true_lbl, pred_lbl, normalize="true")
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, cmap="Blues",
            xticklabels=["W","N1","N2","N3","R"],
            yticklabels=["W","N1","N2","N3","R"])
plt.title("Confusion matrix (normalizada)")
plt.xlabel("Predicted"); plt.ylabel("True")
# Guardar en archivo en lugar de mostrar
grafico_confusion = os.path.join(graficos_dir, "matriz_confusion.png")
plt.savefig(grafico_confusion, dpi=300)
plt.close()
print(f"\nMatriz de confusión guardada en → {grafico_confusion}")

print("Cohen κ :", cohen_kappa_score(true_lbl, pred_lbl))
