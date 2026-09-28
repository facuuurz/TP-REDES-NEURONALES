"""Create the assignment report from measured training artifacts."""

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import BaseDocTemplate, Frame, Image, KeepTogether, PageTemplate, Paragraph, Spacer, Table, TableStyle


AUTHORS = ["Facundo Tomás Hernández", "Facundo Thomas Ramirez"]
TEACHERS = "BioIng. Alejandro Hadad, AS. Joaquín López y Mg. Santiago Díaz"
DELIVERY = "30 de septiembre de 2026"

ROOT = Path("artifacts")
DESTINATION = Path("informe/Informe_tecnico.pdf")
cnn = json.loads((ROOT / "cnn_metrics.json").read_text())
gru = json.loads((ROOT / "gru_metrics.json").read_text())
DESTINATION.parent.mkdir(exist_ok=True)

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleCustom", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=16, leading=20, textColor=colors.HexColor("#142c44"), spaceAfter=6))
styles.add(ParagraphStyle(name="SectionCustom", parent=styles["Heading2"], fontSize=10.5, leading=13, textColor=colors.HexColor("#183e5f"), spaceBefore=8, spaceAfter=4))
styles.add(ParagraphStyle(name="BodyCustom", parent=styles["BodyText"], fontSize=8.9, leading=12.4, spaceAfter=5, alignment=4))
styles.add(ParagraphStyle(name="CaptionCustom", parent=styles["BodyText"], fontSize=7.8, leading=10, textColor=colors.HexColor("#516575"), spaceAfter=6))
styles.add(ParagraphStyle(name="InfoCustom", parent=styles["BodyText"], fontSize=8.6, leading=11.5, textColor=colors.HexColor("#34495e")))
styles.add(ParagraphStyle(name="SmallCustom", parent=styles["BodyText"], fontSize=8, leading=10.5, spaceAfter=4))


def p(text, style="BodyCustom"):
    return Paragraph(text, styles[style])


def n(value, digits=4):
    return f"{value:.{digits}f}".replace(".", ",")


def table(rows, widths):
    result = Table(rows, colWidths=widths, hAlign="LEFT", repeatRows=1)
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9f0f5")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#183e5f")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.8),
        ("LEADING", (0, 0), (-1, -1), 9.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#9bb2c3")),
    ]))
    return result


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#b5c5d0"))
    canvas.line(1.8 * cm, 1.3 * cm, 19.8 * cm, 1.3 * cm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#607786"))
    canvas.drawString(1.8 * cm, 0.95 * cm, "Trabajo Práctico Integrador | Redes Neuronales | " + ", ".join(AUTHORS))
    canvas.drawRightString(19.8 * cm, 0.95 * cm, f"Página {doc.page}")
    canvas.restoreState()


folds = cnn["group_kfold"]
final_cv, short_cv, stride1_cv = folds["final"], folds["truncated_budget"], folds["stride_1"]
logistic = cnn["logistic_regression_reference"]
short_best = [f["best_epoch"] for f in short_cv["folds"]]
final_best = [f["best_epoch"] for f in final_cv["folds"]]
errors = sum(sum(row) for row in cnn["confusion_matrix"]) - sum(cnn["confusion_matrix"][i][i] for i in range(10))
confusions = sorted(
    ((cnn["confusion_matrix"][i][j], i, j) for i in range(10) for j in range(10) if i != j),
    reverse=True,
)[:2]
cells = gru["cell_comparison"]

kernels = {"conv2d": (3, 1), "max_pooling2d": (2, 2), "conv2d_1": (3, 1), "max_pooling2d_1": (2, 2), "conv2d_2": (3, 2)}
labels = {
    "conv2d": "Conv2D 32 filtros 3x3, S=1, same",
    "max_pooling2d": "MaxPooling 2x2, S=2",
    "conv2d_1": "Conv2D 64 filtros 3x3, S=1, same",
    "max_pooling2d_1": "MaxPooling 2x2, S=2",
    "conv2d_2": "Conv2D 64 filtros 3x3, S=2, same",
    "global_average_pooling2d": "Global average pooling",
    "dense": "Densa 64, ReLU",
    "dropout": "Dropout 0,25",
    "dense_1": "Densa 10, softmax",
}
architecture_rows = [["Capa", "Salida", "Parámetros", "Campo receptivo"]]
receptive, jump = 1, 1
for layer in cnn["architecture"]:
    name = layer["layer"]
    if name in kernels:
        kernel, stride = kernels[name]
        receptive, jump = receptive + (kernel - 1) * jump, jump * stride
        field = f"{receptive}x{receptive}"
    else:
        field = "imagen completa" if name == "global_average_pooling2d" else "-"
    architecture_rows.append([labels[name], " x ".join(str(v) for v in layer["output_shape"]), f"{layer['parameters']:,}".replace(",", "."), field])

story = [
    p("CNN y GRU con Keras: diseño, evaluación y análisis de sesgo y varianza", "TitleCustom"),
    p("<b>Trabajo Práctico Integrador: Redes Neuronales</b> | Inteligencia Artificial, Licenciatura en Sistemas de Información, FCyT-UADER", "InfoCustom"),
    p(f"<b>Integrantes:</b> {', '.join(AUTHORS)} | <b>Docentes:</b> {TEACHERS} | <b>Entrega:</b> {DELIVERY}", "InfoCustom"),
    Spacer(1, 0.25 * cm),
    p("1. Objetivo y protocolo", "SectionCustom"),
    p("Se implementaron dos tareas independientes: clasificación espacial de dígitos con una red convolucional (CNN) y pronóstico de un paso sobre una serie multivariada con una unidad recurrente con compuertas (GRU). Se fijaron semillas y todas las decisiones (presupuesto de épocas, stride, tipo de celda) se tomaron solo con datos de desarrollo o validación. Cada conjunto de prueba se evaluó una única vez, después de fijar los pesos. Se exportaron los modelos completos y sus pesos."),
    p("2. Módulo CNN: datos y partición", "SectionCustom"),
    p(f"Se utilizaron 1797 imágenes de dígitos manuscritos de scikit-learn (diez clases). Cada imagen de 8 x 8 se interpoló a 32 x 32 y se asignó una sola vez a uno de doce lotes de adquisición simulados, equilibrados por clase, con brillo, contraste y desenfoque propios, más ruido reproducible por muestra. Los lotes 0 a 7 forman desarrollo ({cnn['samples']['development']} imágenes) y los lotes 8 a 11 se reservan para prueba ({cnn['samples']['test']} imágenes). Dentro de desarrollo, GroupKFold de cuatro particiones deja dos lotes completos para validación en cada iteración: ninguna imagen ni lote cruza las fronteras. Como cada lote contiene las diez clases en proporciones similares, sus condiciones de captura no están asociadas a la etiqueta; la validación por grupos mide entonces la robustez frente a condiciones de adquisición no vistas. El conjunto no incluye identificadores de autor, por lo que no demuestra generalización a escritores nuevos."),
    p("3. Arquitectura CNN y justificación", "SectionCustom"),
    p("Cada activación de una convolución discreta 2D es y[i, j] = suma<sub>m,n,c</sub> w[m, n, c] · x[S·i + m, S·j + n, c] + b: depende solo de una vecindad de K x K píxeles (localidad espacial) y usa los mismos pesos en todas las posiciones. Por eso una capa tiene (K·K·C<sub>entrada</sub> + 1)·C<sub>salida</sub> parámetros: la segunda convolución usa (3·3·32 + 1)·64 = 18.496, mientras que una capa densa entre los mismos mapas (16·16·32 entradas y 16·16·64 salidas) requeriría más de 134 millones. El tamaño de salida es O = floor((H + 2P - K) / S) + 1. Con K = 3 y padding 'same' (P = 1), S = 1 conserva el tamaño (O = H) y S = 2 lo reduce a la mitad; el max pooling (K = 2, S = 2, P = 0) también divide por dos. La tabla muestra la secuencia de tamaños 32, 16, 8 y 4, y el campo receptivo, que crece según r<sub>l</sub> = r<sub>l-1</sub> + (K - 1)·j<sub>l-1</sub>, con j<sub>l</sub> = j<sub>l-1</sub>·S: cada unidad de la última convolución observa 18 x 18 píxeles, suficiente para cubrir trazos completos del dígito."),
    table(architecture_rows, [6.6 * cm, 3.2 * cm, 2.6 * cm, 3.6 * cm]),
    Spacer(1, 0.15 * cm),
    p(f"Total: {cnn['total_parameters']:,} parámetros entrenables.".replace(",", "."), "CaptionCustom"),
    p(f"La convolución con S = 1 es equivariante a traslaciones: desplazar la entrada desplaza el mapa de salida. El pooling y el stride 2 conservan esa propiedad solo para desplazamientos múltiplos del paso y agregan invariancia local a corrimientos pequeños; de ahí que la equivarianza sea parcial. La última reducción usa una convolución con stride 2 en lugar de un tercer pooling: el submuestreo se aprende y la capa evalúa 16 posiciones en lugar de 64, con un cuarto de las multiplicaciones. Con el mismo protocolo, la variante con S = 1 obtuvo {n(stride1_cv['validation_accuracy_mean'])} ± {n(stride1_cv['validation_accuracy_std'])} de exactitud de validación y la elegida {n(final_cv['validation_accuracy_mean'])} ± {n(final_cv['validation_accuracy_std'])}: igual media y menor dispersión entre lotes."),
    p("ReLU tiene derivada 1 para entradas positivas, por lo que no satura y no atenúa el gradiente al atravesar capas. La salida softmax con entropía cruzada da un gradiente respecto de cada logit igual a p<sub>k</sub> - y<sub>k</sub>, acotado entre -1 y 1. Global average pooling reemplaza el aplanado: la capa densa recibe 64 valores en lugar de 1.024. Dropout 0,25 reduce la coadaptación de unidades. Se usa Adam con tasa 0,001, lotes de 32 imágenes (unas 28 actualizaciones por época), recorte de la norma del gradiente a 1,0 y early stopping sobre la pérdida de validación con paciencia 12 y un máximo de 150 épocas, restituyendo los mejores pesos."),
    p("4. CNN: análisis de sesgo y varianza", "SectionCustom"),
    p("Se compararon tres configuraciones con las mismas particiones GroupKFold y una referencia lineal (regresión logística sobre los píxeles). La brecha es la diferencia media entre exactitud de entrenamiento y de validación."),
    table([
        ["Configuración", "Mejores épocas", "Entrenamiento", "Validación", "Brecha"],
        ["Presupuesto corto: 30 épocas, paciencia 5", ", ".join(map(str, short_best)), n(short_cv["training_accuracy_mean"]), f"{n(short_cv['validation_accuracy_mean'])} ± {n(short_cv['validation_accuracy_std'])}", n(short_cv["generalization_gap_mean"])],
        ["Elegida: 150 épocas, paciencia 12, S = 2", ", ".join(map(str, final_best)), n(final_cv["training_accuracy_mean"]), f"{n(final_cv['validation_accuracy_mean'])} ± {n(final_cv['validation_accuracy_std'])}", n(final_cv["generalization_gap_mean"])],
        ["Variante: 150 épocas, paciencia 12, S = 1", ", ".join(str(f["best_epoch"]) for f in stride1_cv["folds"]), n(stride1_cv["training_accuracy_mean"]), f"{n(stride1_cv['validation_accuracy_mean'])} ± {n(stride1_cv['validation_accuracy_std'])}", n(stride1_cv["generalization_gap_mean"])],
        ["Referencia: regresión logística", "-", "-", n(logistic["validation_accuracy_mean"]), "-"],
    ], [6.4 * cm, 2.8 * cm, 2.4 * cm, 3.0 * cm, 1.6 * cm]),
    Spacer(1, 0.2 * cm),
    p(f"<b>Sesgo.</b> Con el presupuesto corto, los cuatro folds alcanzan el tope de 30 épocas con la mejor época en {min(short_best)}-{max(short_best)}: la pérdida de validación seguía bajando cuando se cortó. La exactitud de validación ({n(short_cv['validation_accuracy_mean'])}) queda por debajo de la referencia lineal ({n(logistic['validation_accuracy_mean'])}), aunque la CNN tiene mucha más capacidad. Ese resultado indica sesgo por entrenamiento incompleto, no por falta de capacidad. Con el presupuesto elegido, las mejores épocas van de {min(final_best)} a {max(final_best)}, la exactitud de entrenamiento llega a {n(final_cv['training_accuracy_mean'])} y la de validación sube a {n(final_cv['validation_accuracy_mean'])}, por encima de la referencia lineal."),
    p(f"<b>Varianza.</b> La brecha media de la configuración elegida es {n(final_cv['generalization_gap_mean'])}: el modelo ajusta casi por completo el entrenamiento y pierde {n(100 * final_cv['generalization_gap_mean'], 1)} puntos porcentuales en lotes no vistos. En las curvas, la pérdida de entrenamiento sigue bajando mientras la de validación alcanza un mínimo (línea punteada) y luego oscila o sube: es el inicio del sobreajuste, que early stopping corta restituyendo los pesos del mínimo. La desviación entre folds ({n(final_cv['validation_accuracy_std'])}) indica baja sensibilidad a qué lotes se usan para validar."),
    KeepTogether([
        Image(str(ROOT / "cnn_fold_accuracy.png"), width=13.0 * cm, height=4.62 * cm),
        p("Exactitud de entrenamiento y validación por fold, con lotes excluyentes, para el presupuesto corto y el elegido.", "CaptionCustom"),
    ]),
    KeepTogether([
        Image(str(ROOT / "cnn_learning_curves.png"), width=17.5 * cm, height=4.9 * cm),
        p("Pérdida de entropía cruzada por época en cada fold de la configuración elegida. La línea punteada marca la época restituida por early stopping.", "CaptionCustom"),
    ]),
    p(f"<b>Modelo final y prueba.</b> Se reentrenó con todo el desarrollo durante {cnn['selected_epochs']} épocas (mediana de las mejores épocas) y se evaluó una vez sobre los lotes 8 a 11: exactitud {n(cnn['test_accuracy'])} y F1 macro {n(cnn['test_macro_f1'])}, con {errors} errores sobre {cnn['samples']['test']} imágenes. La exactitud de prueba coincide con la media de validación cruzada, por lo que los lotes reservados no revelan degradación. Las confusiones más frecuentes son {confusions[0][1]} clasificado como {confusions[0][2]} ({confusions[0][0]} casos) y {confusions[1][1]} clasificado como {confusions[1][2]} ({confusions[1][0]} casos); la matriz completa está en cnn_metrics.json."),
    p("5. Módulo GRU: serie y corte temporal", "SectionCustom"),
    p("Se generó una serie sintética de 1600 instantes con un componente autorregresivo, seno y coseno estacionales (período 48), un control correlacionado y ruido gaussiano de desviación 0,11. El objetivo en t depende de su valor en t-1, de la estación en t y del control en t-1. Cada entrada contiene los 36 pasos anteriores de las cuatro variables y el objetivo es el valor siguiente. Los tramos son entrenamiento [0, 960), validación [960, 1280) y prueba [1280, 1600). Todas las ventanas se construyen dentro de su propio tramo: se descartan los primeros 36 objetivos de cada frontera para que ninguna ventana de validación o prueba use historia de un tramo anterior. La media y la desviación de las entradas y del objetivo se ajustan solo con entrenamiento y se guardan en gru_scalers.npz."),
    p("6. Arquitectura GRU y estabilidad del gradiente", "SectionCustom"),
    p("En BPTT el gradiente hacia un estado pasado es dL/dh<sub>k</sub> = dL/dh<sub>T</sub> · J<sub>T</sub> · J<sub>T-1</sub> ··· J<sub>k+1</sub>, con J<sub>t</sub> = dh<sub>t</sub>/dh<sub>t-1</sub>. En una RNN simple, h<sub>t</sub> = tanh(W x<sub>t</sub> + U h<sub>t-1</sub> + b) y J<sub>t</sub> = diag(1 - h<sub>t</sub><super>2</super>) · U. Como la derivada de tanh no supera 1, la norma del producto está acotada por ||U||<super>T-k</super>: decae geométricamente si ||U|| &lt; 1 (desaparición) y puede crecer sin límite si es mayor (explosión)."),
    p("La GRU calcula una compuerta de actualización z<sub>t</sub> = sigmoid(W<sub>z</sub> x<sub>t</sub> + U<sub>z</sub> h<sub>t-1</sub>), una de reinicio r<sub>t</sub> y un candidato g<sub>t</sub> = tanh(W x<sub>t</sub> + U (r<sub>t</sub> · h<sub>t-1</sub>)), y combina h<sub>t</sub> = z<sub>t</sub> · h<sub>t-1</sub> + (1 - z<sub>t</sub>) · g<sub>t</sub> (productos elemento a elemento). Entonces J<sub>t</sub> = diag(z<sub>t</sub>) + (términos que pasan por g<sub>t</sub> y z<sub>t</sub>): el primer término es una vía aditiva que no se multiplica por U ni por la derivada de tanh. Cuando la red aprende z<sub>t</sub> cercano a 1, ese factor se aproxima a la identidad y el gradiente atraviesa muchos pasos sin atenuarse. Esto mitiga la desaparición, aunque no la elimina. Contra la explosión, el optimizador recorta la norma del gradiente de cada tensor de pesos a 1,0 (g se reemplaza por g · min(1, 1/||g||)). La tanh mantiene el estado acotado en (-1, 1) y las compuertas sigmoid quedan en (0, 1)."),
    p("Para evaluar la elección se entrenaron las tres celdas con 32 unidades, la misma cabeza (densa de 16 unidades ReLU y salida lineal), pérdida MSE, Adam con tasa 0,001, lotes de 32, orden cronológico conservado (shuffle=False) y early stopping con paciencia 7 sobre un máximo de 55 épocas. La selección se hizo por MAE de validación; la columna de prueba se informa después."),
    table(
        [["Celda", "Parámetros", "Mejor época", "MAE validación", "MAE prueba", "RMSE prueba"]]
        + [[label, f"{cells[key]['parameters']:,}".replace(",", "."), str(cells[key]["best_epoch"]), n(cells[key]["validation_mae"]), n(cells[key]["test_mae"]), n(cells[key]["test_rmse"])]
           for key, label in [("gru", "GRU (elegida)"), ("lstm", "LSTM"), ("simple_rnn", "RNN simple")]],
        [3.6 * cm, 2.4 * cm, 2.4 * cm, 3.0 * cm, 2.6 * cm, 2.6 * cm],
    ),
    Spacer(1, 0.2 * cm),
    p(f"La GRU obtiene el menor error de validación ({n(cells['gru']['validation_mae'])}) con {100 * (1 - cells['gru']['parameters'] / cells['lstm']['parameters']):.0f}% menos parámetros que la LSTM. La RNN simple queda muy cerca porque la dinámica de esta serie es de corto alcance (el objetivo depende del paso anterior): aquí la propagación del gradiente a lo largo de muchos pasos no es el factor limitante. La ventaja de las compuertas aparece cuando las dependencias abarcan muchos pasos; la GRU ofrece esa protección con menos parámetros que la LSTM. La ventana de 36 pasos cubre tres cuartos del ciclo estacional. La salida lineal no impone un rango artificial a la regresión y MSE penaliza más los errores grandes, en línea con el RMSE informado."),
    p("7. Resultados observados: GRU", "SectionCustom"),
    p("Como la serie es sintética, se conoce la parte predecible del proceso: aplicar la ecuación verdadera deja solo el ruido, cuyo error fija un piso que ningún modelo puede superar en promedio. La persistencia (repetir el último valor observado) sirve como referencia ingenua."),
    table([
        ["Modelo", "MAE prueba", "RMSE prueba"],
        ["GRU", n(gru["test_mae"]), n(gru["test_rmse"])],
        ["Persistencia", n(gru["persistence_test_mae"]), n(gru["persistence_test_rmse"])],
        ["Piso de ruido (ecuación verdadera)", n(gru["noise_floor_test_mae"]), n(gru["noise_floor_test_rmse"])],
    ], [7.0 * cm, 3.5 * cm, 3.5 * cm]),
    Spacer(1, 0.2 * cm),
    p(f"<b>Sesgo.</b> La GRU reduce el MAE de la persistencia en {n(100 * (1 - gru['test_mae'] / gru['persistence_test_mae']), 1)}% y queda solo {n(100 * (gru['test_mae'] / gru['noise_floor_test_mae'] - 1), 1)}% por encima del piso de ruido: casi todo el error remanente es irreducible, por lo que el sesgo es bajo. <b>Varianza.</b> El MAE es {n(gru['training_mae'])} en entrenamiento, {n(gru['validation_mae'])} en validación y {n(gru['test_mae'])} en prueba; las brechas son pequeñas frente al error medio y en la curva ambas pérdidas convergen sin divergir. <b>Estabilidad predictiva.</b> El MAE por cuartos cronológicos de la prueba es {', '.join(n(v) for v in gru['test_mae_by_chronological_quarter'])}: no crece a medida que la prueba se aleja del entrenamiento."),
    KeepTogether([
        Image(str(ROOT / "gru_learning_curve.png"), width=10.5 * cm, height=5.74 * cm),
        p(f"Pérdida MSE por época en entrenamiento y validación cronológica de la GRU (mejor época: {gru['best_epoch']}).", "CaptionCustom"),
    ]),
    p("8. Limitaciones", "SectionCustom"),
    p("La CNN se evalúa sobre lotes simulados, no sobre personas o dispositivos nuevos. La serie de la GRU es un proceso sintético estacionario de dependencia corta, útil para verificar el protocolo pero no extrapolable a un dominio aplicado. No se reclaman conclusiones para otras poblaciones. Los resultados se reproducen con el código; distintas CPU o versiones de bibliotecas pueden introducir diferencias numéricas pequeñas."),
    p("9. Archivos, reproducción y referencias", "SectionCustom"),
    p("pip install -r requirements.txt; python train.py; python predict.py; python -m unittest discover -s tests -v; python make_report.py. cnn.keras y gru.keras contienen arquitectura y pesos; *.weights.h5 son los pesos; gru_scalers.npz se necesita para interpretar la salida de la GRU; cnn_metrics.json y gru_metrics.json guardan todos los resultados. Referencias: documentación de Keras (keras.io/api) y scikit-learn (scikit-learn.org/stable); Goodfellow, Bengio y Courville, Deep Learning, MIT Press, 2016, capítulos 9 y 10; Cho et al., Learning Phrase Representations using RNN Encoder-Decoder, EMNLP 2014.", "SmallCustom"),
]

doc = BaseDocTemplate(str(DESTINATION), pagesize=(21.59 * cm, 27.94 * cm), leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.5 * cm, bottomMargin=1.7 * cm, title="Informe técnico: CNN y GRU con Keras", author=", ".join(AUTHORS))
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, leftPadding=0, bottomPadding=0, rightPadding=0, topPadding=0)
doc.addPageTemplates(PageTemplate(id="report", frames=frame, onPage=footer))
doc.build(story)
print(DESTINATION)
