# Trabajo práctico integrador: CNN y GRU con Keras

Implementación reproducible de los dos módulos de la consigna. Los dígitos vienen incluidos en scikit-learn; los lotes visuales y la serie temporal se generan localmente con semillas fijas. No se descarga información externa. Las conclusiones sobre nuevos tipos de adquisición y procesos temporales reales son limitadas.

## Entregables

- `informe/Informe_tecnico.pdf`: metodología, justificación matemática, resultados y límites (máximo cinco páginas).
- `src/` y `train.py`: generación, partición, entrenamiento y evaluación de ambos modelos.
- `artifacts/cnn.keras` y `artifacts/gru.keras`: modelos completos, incluidos sus pesos; también se incluyen `*.weights.h5`.
- `artifacts/gru_scalers.npz`: medias y desviaciones necesarias para inferencia con la GRU.
- `artifacts/*_metrics.json`, `artifacts/*_learning_curve.png`: métricas y curvas observadas.
- `predict.py`: carga los modelos y ejecuta ejemplos de inferencia.

## Ejecución

Se recomienda Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
python predict.py
python -m unittest discover -s tests -v
python make_report.py
```

En Windows, activar con `.venv\Scripts\activate`. La ejecución en CPU puede tardar varios minutos. `train.py` sobrescribe los artefactos con un nuevo entrenamiento.

## Diseño y evaluación

La CNN clasifica 1797 imágenes reales de dígitos manuscritos (10 clases) incluidas en `sklearn.datasets.load_digits`. Las imágenes originales de 8 × 8 se interpolan a 32 × 32 y se asignan una sola vez a 12 lotes de adquisición simulados, cada uno con brillo, contraste, desenfoque y ruido propios. Los lotes 0–7 se destinan al desarrollo, con cuatro particiones `GroupKFold`; los 8–11 se reservan para prueba. Tras la validación, se entrena un modelo final con todos los lotes de desarrollo durante la mediana de las épocas óptimas. No hay imágenes repetidas entre grupos. No se conocen identificadores de escritor, por lo que esta prueba no demuestra generalización a autores nuevos.

La GRU pronostica un paso adelante en una serie multivariable de 1600 instantes. Las variables son el valor anterior, componentes estacionales senoidales y un control autorregresivo. La división cronológica es entrenamiento `[0,960)`, validación `[960,1280)` y prueba `[1280,1600)`. Cada ventana de 36 pasos queda completamente dentro de su tramo. Los escaladores se ajustan solo con entrenamiento. La comparación básica es la persistencia del último valor observado.

En ambas tareas, el conjunto de prueba se evalúa una sola vez después de fijar el entrenamiento. Los generadores están disponibles en `src/data.py`, por lo que los pesos exportados se pueden volver a evaluar sin descargar datos.
