# Trabajo práctico integrador: CNN y GRU con Keras

Integrantes: Facundo Tomás Hernández y Facundo Thomas Ramirez.

Implementación reproducible de los dos módulos de la consigna en `consigna/TP_REDES_NEURONALES.pdf`. Los dígitos vienen incluidos en scikit-learn; los lotes visuales y la serie temporal se generan localmente con semillas fijas. No se descargan datos externos. Los lotes son simulados y la serie es sintética, por lo que los resultados no prueban generalización a escritores nuevos ni a procesos temporales reales.

## Entregables

- `informe/Informe_tecnico.pdf`: informe técnico de cuatro páginas, con diseño, justificación matemática, análisis de sesgo y varianza, resultados y límites.
- `src/`, `train.py`, `predict.py` y `tests/`: generación, partición, entrenamiento, evaluación, inferencia y comprobaciones de las fronteras de datos y de la arquitectura. El código Python está redactado en inglés.
- `artifacts/cnn.keras` y `artifacts/gru.keras`: modelos completos con arquitectura y pesos; también se entregan `*.weights.h5`.
- `artifacts/gru_scalers.npz`: escaladores ajustados solo con datos de entrenamiento, necesarios para interpretar la salida de la GRU.
- `artifacts/*_metrics.json`: métricas de todos los experimentos. Gráficos: `cnn_fold_accuracy.png`, `cnn_learning_curves.png` y `gru_learning_curve.png`.
- `consigna/TP_REDES_NEURONALES.pdf`: consigna original.

## Ejecución

Se recomienda Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python predict.py
```

En Windows, activar con `.venv\Scripts\activate`. Para repetir el entrenamiento y regenerar el informe:

```bash
python train.py
python make_report.py
```

El entrenamiento sobrescribe los artefactos. En CPU puede tardar varios minutos.

## Diseño y resultados

La CNN clasifica 1797 imágenes de dígitos manuscritos (10 clases) de `sklearn.datasets.load_digits`. Las imágenes originales de 8 × 8 se interpolan a 32 × 32 y se asignan una sola vez a 12 lotes de adquisición simulados, cada uno con brillo, contraste y desenfoque propios. Los lotes 0–7 se usan para desarrollo, con cuatro divisiones GroupKFold; los 8–11 se reservan para prueba. La red usa tres convoluciones 3 × 3 con padding `same` (la última con stride 2), dos max pooling 2 × 2 y global average pooling. Con un máximo de 150 épocas y early stopping de paciencia 12, la exactitud media de validación fue 0,9751 ± 0,0050 y la de prueba 0,9763 (F1 macro 0,9764). Como comparación, con 30 épocas y paciencia 5 la validación queda en 0,9586, por debajo de una regresión logística (0,9602): es sesgo por entrenamiento incompleto.

La GRU pronostica un paso adelante en una serie multivariable sintética de 1600 instantes. La división cronológica es entrenamiento `[0,960)`, validación `[960,1280)` y prueba `[1280,1600)`. Cada ventana de 36 pasos queda completamente dentro de su tramo y los escaladores se ajustan solo con entrenamiento. Frente a LSTM y RNN simple entrenadas con el mismo protocolo, la GRU obtuvo el menor MAE de validación (0,0891). En prueba obtuvo MAE 0,0960 y RMSE 0,1201, frente a MAE 0,1139 del pronóstico de persistencia y un piso de ruido de 0,0917.

Las métricas detalladas, incluidos folds, curvas, matriz de confusión, comparación de celdas y evolución temporal del error, están en `artifacts/`. El conjunto de prueba se evalúa después de seleccionar los modelos y se mantiene separado de la validación.
