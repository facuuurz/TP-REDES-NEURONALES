# Trabajo práctico integrador: CNN y GRU con Keras

Implementación reproducible de los dos módulos de la consigna en `consigna/TP_REDES_NEURONALES.pdf`. Los dígitos vienen incluidos en scikit-learn; los lotes visuales y la serie temporal se generan localmente con semillas fijas. No se descargan datos externos. Los lotes son simulados y la serie es sintética, por lo que los resultados no prueban generalización a escritores nuevos ni a procesos temporales reales.

## Entregables

- `informe/Informe_tecnico.pdf`: informe técnico de tres páginas, con diseño, resultados y límites.
- `src/`, `train.py`, `predict.py` y `tests/`: generación, partición, entrenamiento, evaluación, inferencia y comprobaciones de las fronteras de datos. El código Python está redactado en inglés.
- `artifacts/cnn.keras` y `artifacts/gru.keras`: modelos completos con arquitectura y pesos; también se entregan `*.weights.h5`.
- `artifacts/gru_scalers.npz`: escaladores ajustados solo con datos de entrenamiento, necesarios para interpretar la salida de la GRU.
- `artifacts/*_metrics.json`, `artifacts/cnn_fold_accuracy.png` y `artifacts/gru_learning_curve.png`: métricas y gráficos observados.
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

La CNN clasifica 1797 imágenes de dígitos manuscritos (10 clases) de `sklearn.datasets.load_digits`. Las imágenes originales de 8 × 8 se interpolan a 32 × 32 y se asignan una sola vez a 12 lotes de adquisición simulados, cada uno con brillo, contraste y desenfoque propios. Los lotes 0–7 se usan para desarrollo, con cuatro divisiones GroupKFold; los 8–11 se reservan para prueba. La exactitud media de validación fue 0,9313 ± 0,0180 y la exactitud de prueba fue 0,9593 (F1 macro 0,9595).

La GRU pronostica un paso adelante en una serie multivariable sintética de 1600 instantes. La división cronológica es entrenamiento `[0,960)`, validación `[960,1280)` y prueba `[1280,1600)`. Cada ventana de 36 pasos queda completamente dentro de su tramo y los escaladores se ajustan solo con entrenamiento. En prueba obtuvo MAE 0,0960 y RMSE 0,1201, frente a MAE 0,1139 y RMSE 0,1428 del pronóstico de persistencia.

Las métricas detalladas, incluidos folds, matriz de confusión y evolución temporal del error, están en `artifacts/`. El conjunto de prueba se evalúa después de seleccionar los modelos y se mantiene separado de la validación.
