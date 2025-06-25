# Sleep Stage Classification with CNN-LSTM
Este proyecto implementa un modelo de aprendizaje profundo basado en una arquitectura **CNN-LSTM** para la detección automática de fases del sueño (W, N1, N2, N3, REM) siguiedno la clasificación AASM. Se utilizan señales fisiológicas multicanal extraídas en formato '.edf' de registros de polisomnografía (PSG), incluyendo entre otras:

- ECG
- EEG C4-M1
- EOG E1-M1
- EOG E2-M2
- EMG chin
  
## Objetivo

Desarrollar y evaluar un modelo profundo que permita realizar la clasificación de las fases del sueño con distintos grados de complejidad de entrada, desde una única señal (ECG) hasta un montaje multicanal mínimo (EEG, EOG, EMG), valorando la viabilidad de sistemas menos invasivos y más portables.

## Estructura del proyecto

sleep-stage-detection
- recordings/ # Archivos .edf originales (no se suben al repo)
- pacientes/ # Archivos procesados .npz (no se suben al repo)
- modelos/ 
  - cnn_lstm_model.py # Definición del modelo CNN-LSTM y callbacks
  - modelo.keras # Modelos guardados
- preprocessing/ # Procesamiento de datos y anotaciones
  - procesado_pacientes.py # Lectura EDF, filtrado, generación de .npz
  - dataset_builder.py # Creación de tf.data.Dataset con secuencias L
- train_cnn_lstm_model.py # Script principal (pipeline completo)
- graficos/ 
  - entrenamiento_metricas.png # Curvas de pérdida y precisión
  - matriz_confusion.png # Matriz de confusión final
- .gitignore
- README.md

## Requisitos

- Python 3.9+
- TensorFlow 2.x
- Keras
- Pandas
- NumPy
- Matplotlib
- Seaborn
- MNE
- Scikit-learn

## Ejecución del proyecto

python train_cnn_lstm_model.py

El script ejecuta:

1. Procesado de señales desde archivos .edf

2. Filtrado, resampleo y ventanado (30 s)

3. División en train/val/test por sujeto (80/10/10)

4. Grid Search de hiperparámetros

5. Entrenamiento con callbacks (EarlyStopping, ReduceLROnPlateau)

6. Evaluación y visualización de resultados

## Experimentos realizados

Se han comparado 4 configuraciones de entrada:

1. ECG

2. ECG + EEG (C4-M1)

3. ECG + EOG (E1-M2, E2-M2)

4. ECG + EEG + EOG + EMG (montaje mínimo AASM)

Métricas evaluadas: accuracy, precision, recall, F1-score, índice de Kappa de Cohen.

## Arquitectura del modelo

El modelo está compuesto por dos bloques principales:

### CNN (Feature extractor): Extrae un vector de x características de cada ventana de las señales.
Conv1D espera entradas de la forma (L,3000,n_canales):
- 3 bloques operacionales, cada uno con:
  - `Conv1D`: kernel `(1x100)`, padding `'same'`, filtros: `8 → 16 → 32`
  - `ReLU`
  - `BatchNormalization`
  - `AveragePooling1D`: pool size `2`
- `GlobalAveragePooling1D(2)`
- `Dense(dense_units, activation=relu)`: vector de características de salida

### LSTM (Sequence modeling): Aprende la dinámica temporal a partir de secuencias de 10 vectores de x características
- Entrada: secuencia de vectores de x características (`L` longitudes --10)
- `LSTM(64 o 100)` unidades
- `Dense(5, activation='softmax')`: salida con 5 clases para clasificar en fases W, N1, N2, N3, R

