"""Create the assignment report from measured training artifacts."""

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
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
styles.add(ParagraphStyle(name="SectionCustom", parent=styles["Heading2"], fontSize=10.5, leading=13, textColor=colors.HexColor("#183e5f"), spaceBefore=9, spaceAfter=4))
styles.add(ParagraphStyle(name="BodyCustom", parent=styles["BodyText"], fontSize=9, leading=12.6, spaceAfter=4))
styles.add(ParagraphStyle(name="BulletCustom", parent=styles["BodyText"], fontSize=9, leading=12.6, leftIndent=11, bulletIndent=2, spaceAfter=2.5))
styles.add(ParagraphStyle(name="FormulaCustom", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.4, leading=13, alignment=TA_CENTER, textColor=colors.HexColor("#142c44"), spaceBefore=3, spaceAfter=5))
styles.add(ParagraphStyle(name="CaptionCustom", parent=styles["BodyText"], fontSize=7.8, leading=10, textColor=colors.HexColor("#516575"), spaceAfter=6))
styles.add(ParagraphStyle(name="InfoCustom", parent=styles["BodyText"], fontSize=8.6, leading=11.5, textColor=colors.HexColor("#34495e")))
styles.add(ParagraphStyle(name="SmallCustom", parent=styles["BodyText"], fontSize=8, leading=10.5, spaceAfter=3))


def p(text, style="BodyCustom"):
    return Paragraph(text, styles[style])


def b(text):
    return Paragraph(text, styles["BulletCustom"], bulletText="•")


def f(text):
    return Paragraph(text, styles["FormulaCustom"])


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
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
        ("TOPPADDING", (0, 0), (-1, -1), 3.2),
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
short_best = [fold["best_epoch"] for fold in short_cv["folds"]]
final_best = [fold["best_epoch"] for fold in final_cv["folds"]]
errors = sum(sum(row) for row in cnn["confusion_matrix"]) - sum(cnn["confusion_matrix"][i][i] for i in range(10))
confusions = sorted(
    ((cnn["confusion_matrix"][i][j], i, j) for i in range(10) for j in range(10) if i != j),
    reverse=True,
)[:2]
cells = gru["cell_comparison"]

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
kernels = {"conv2d": (3, 1), "max_pooling2d": (2, 2), "conv2d_1": (3, 1), "max_pooling2d_1": (2, 2), "conv2d_2": (3, 2)}
architecture_rows = [["Capa", "Salida", "Parámetros", "Campo receptivo"]]
receptive, jump = 1, 1
for layer in cnn["architecture"]:
    name = layer["layer"]
    field = "-"
    if name in kernels:
        kernel, stride = kernels[name]
        receptive, jump = receptive + (kernel - 1) * jump, jump * stride
        field = f"{receptive} x {receptive}"
    architecture_rows.append([labels[name], " x ".join(str(v) for v in layer["output_shape"]), f"{layer['parameters']:,}".replace(",", "."), field])


def config_row(label, result):
    return [
        label,
        ", ".join(str(fold["best_epoch"]) for fold in result["folds"]),
        n(result["training_accuracy_mean"]),
        f"{n(result['validation_accuracy_mean'])} ± {n(result['validation_accuracy_std'])}",
        n(result["generalization_gap_mean"]),
    ]


story = [
    p("CNN y GRU con Keras: diseño, evaluación y análisis de sesgo y varianza", "TitleCustom"),
    p("<b>Trabajo Práctico Integrador: Redes Neuronales</b> | Inteligencia Artificial, Licenciatura en Sistemas de Información, FCyT-UADER", "InfoCustom"),
    p(f"<b>Integrantes:</b> {', '.join(AUTHORS)} | <b>Docentes:</b> {TEACHERS} | <b>Entrega:</b> {DELIVERY}", "InfoCustom"),
    Spacer(1, 0.2 * cm),

    p("1. Objetivo y protocolo", "SectionCustom"),
    p("Se implementaron dos modelos independientes: una CNN que clasifica dígitos manuscritos y una GRU que pronostica un paso de una serie multivariada. Todas las decisiones (épocas, stride, tipo de celda) se tomaron con datos de desarrollo o validación; cada conjunto de prueba se evaluó una sola vez, con los pesos ya fijados. Se fijaron semillas y se exportaron los modelos completos y sus pesos."),

    p("2. Módulo CNN: datos y partición", "SectionCustom"),
    b(f"<b>Datos:</b> 1797 dígitos de scikit-learn (10 clases), interpolados de 8 x 8 a 32 x 32 y repartidos en 12 lotes de adquisición simulados, cada uno con brillo, contraste y desenfoque propios."),
    b(f"<b>Partición:</b> lotes 0 a 7 para desarrollo ({cnn['samples']['development']} imágenes) y 8 a 11 para prueba ({cnn['samples']['test']}). Dentro de desarrollo, GroupKFold de 4 particiones: cada fold valida con dos lotes completos que el modelo no vio. Ninguna imagen ni lote cruza las fronteras."),
    b("<b>Qué mide:</b> cada lote tiene las diez clases en proporciones similares, así que la validación por grupos mide robustez ante condiciones de captura no vistas. Como no hay identificadores de autor, no prueba generalización a escritores nuevos."),

    p("3. Arquitectura CNN y justificación", "SectionCustom"),
    p("Tres convoluciones 3 x 3 con padding 'same', dos max pooling y global average pooling. El tamaño de cada mapa sale de:"),
    f("O = floor( (H + 2P - K) / S ) + 1"),
    p("Con K = 3 y P = 1, S = 1 conserva el tamaño y S = 2 lo divide por dos; el pooling (K = 2, S = 2, P = 0) también lo divide por dos. Así se obtiene la secuencia 32, 16, 8 y 4. Como la convolución comparte los mismos pesos en todas las posiciones, cada capa tiene:"),
    f("parámetros = (K · K · C<sub>entrada</sub> + 1) · C<sub>salida</sub>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;por ejemplo  (3 · 3 · 32 + 1) · 64 = 18.496"),
    p("Esa cantidad no depende del tamaño de la imagen: una capa densa entre esos mismos mapas (16 x 16 x 32 entradas y 16 x 16 x 64 salidas) necesitaría más de 134 millones de pesos. El campo receptivo, es decir cuántos píxeles de la imagen original ve cada unidad, crece capa a capa con:"),
    f("r<sub>l</sub> = r<sub>l-1</sub> + (K - 1) · j<sub>l-1</sub>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;con  j<sub>l</sub> = j<sub>l-1</sub> · S  (distancia entre unidades vecinas)"),
    table(architecture_rows, [6.8 * cm, 3.2 * cm, 2.6 * cm, 3.2 * cm]),
    p(f"Total: {cnn['total_parameters']:,} parámetros entrenables. Cada unidad de la última convolución ve {receptive} x {receptive} píxeles, suficiente para abarcar el trazo del dígito.".replace(",", "."), "CaptionCustom"),
    b(f"<b>Stride 2 en la última convolución:</b> reemplaza un tercer pooling por un submuestreo aprendido y evalúa 16 posiciones en lugar de 64. Rinde igual que S = 1 ({n(final_cv['validation_accuracy_mean'])} contra {n(stride1_cv['validation_accuracy_mean'])} en validación) con menos dispersión entre lotes ({n(final_cv['validation_accuracy_std'])} contra {n(stride1_cv['validation_accuracy_std'])})."),
    b("<b>Pooling y equivarianza parcial:</b> la convolución con S = 1 es equivariante a traslaciones (desplazar la entrada desplaza la salida). El pooling y el stride 2 la conservan solo para desplazamientos múltiplos del paso y agregan invariancia local a corrimientos pequeños."),
    b("<b>ReLU:</b> su derivada es 1 para entradas positivas, así que no satura ni atenúa el gradiente entre capas."),
    b("<b>Softmax con entropía cruzada:</b> el gradiente respecto de cada logit es p<sub>k</sub> - y<sub>k</sub>, acotado entre -1 y 1."),
    b("<b>Global average pooling:</b> la capa densa recibe 64 valores en lugar de 1.024, con muchos menos parámetros que aplanar."),
    b("<b>Hiperparámetros:</b> dropout 0,25; Adam con tasa 0,001; lotes de 32; recorte de la norma del gradiente a 1,0; early stopping sobre la pérdida de validación con paciencia 12 y máximo de 150 épocas, restituyendo los mejores pesos."),

    p("4. CNN: análisis de sesgo y varianza", "SectionCustom"),
    p("Aquí sesgo se usa en sentido estadístico (error sistemático por subajuste), no como sesgo inductivo de la arquitectura. Se compararon tres configuraciones con las mismas particiones y una referencia lineal (regresión logística sobre los píxeles). La brecha es entrenamiento menos validación."),
    table([
        ["Configuración", "Mejores épocas", "Entrenamiento", "Validación", "Brecha"],
        config_row("Presupuesto corto: 30 épocas, paciencia 5", short_cv),
        config_row("Elegida: 150 épocas, paciencia 12, S = 2", final_cv),
        config_row("Variante: 150 épocas, paciencia 12, S = 1", stride1_cv),
        ["Referencia: regresión logística", "-", "-", n(logistic["validation_accuracy_mean"]), "-"],
    ], [6.4 * cm, 2.8 * cm, 2.4 * cm, 3.0 * cm, 1.6 * cm]),
    Spacer(1, 0.2 * cm),
    b(f"<b>Sesgo:</b> con el presupuesto corto los cuatro folds llegan al tope de 30 épocas (mejores épocas {min(short_best)} a {max(short_best)}) y la validación ({n(short_cv['validation_accuracy_mean'])}) queda por debajo de la regresión logística ({n(logistic['validation_accuracy_mean'])}). Es subajuste por entrenamiento incompleto, no por falta de capacidad. Con la configuración elegida la validación sube a {n(final_cv['validation_accuracy_mean'])}."),
    b(f"<b>Varianza:</b> la brecha es de {n(100 * final_cv['generalization_gap_mean'], 1)} puntos porcentuales. En las curvas, la pérdida de validación llega a un mínimo (línea punteada) y luego oscila o sube: es el inicio del sobreajuste, que early stopping corta. La dispersión entre folds ({n(final_cv['validation_accuracy_std'])}) es baja."),
    b(f"<b>Prueba:</b> el modelo final se reentrenó con todo el desarrollo durante {cnn['selected_epochs']} épocas (mediana de las mejores). Sobre los lotes 8 a 11 obtuvo exactitud {n(cnn['test_accuracy'])} y F1 macro {n(cnn['test_macro_f1'])} ({errors} errores en {cnn['samples']['test']} imágenes), en línea con la validación cruzada. Las confusiones más frecuentes son {confusions[0][1]} tomado como {confusions[0][2]} y {confusions[1][1]} tomado como {confusions[1][2]}."),
    KeepTogether([
        Image(str(ROOT / "cnn_fold_accuracy.png"), width=10.5 * cm, height=3.73 * cm),
        p("Exactitud de entrenamiento y validación por fold, con presupuesto corto y con el elegido.", "CaptionCustom"),
    ]),
    KeepTogether([
        Image(str(ROOT / "cnn_learning_curves.png"), width=15.5 * cm, height=4.34 * cm),
        p("Pérdida por época en cada fold de la configuración elegida. La línea punteada marca la época restituida por early stopping.", "CaptionCustom"),
    ]),

    p("5. Módulo GRU: serie y corte temporal", "SectionCustom"),
    b("<b>Serie:</b> 1600 instantes sintéticos con cuatro variables: el objetivo (autorregresivo), seno y coseno de período 48 y un control correlacionado, más ruido de desviación 0,11. El objetivo en t depende de su valor en t-1, de la estación y del control en t-1."),
    b("<b>Entrada y salida:</b> los 36 pasos anteriores de las cuatro variables predicen el valor siguiente del objetivo."),
    b("<b>Corte cronológico:</b> entrenamiento [0, 960), validación [960, 1280) y prueba [1280, 1600). Cada ventana queda completa dentro de su tramo y la normalización se ajusta solo con entrenamiento (gru_scalers.npz), así el futuro no contamina el pasado."),

    p("6. GRU: estabilidad del gradiente y elección de la celda", "SectionCustom"),
    p("En BPTT, el gradiente hacia un estado pasado es un producto de jacobianos:"),
    f("dL/dh<sub>k</sub> = dL/dh<sub>T</sub> · J<sub>T</sub> · J<sub>T-1</sub> ··· J<sub>k+1</sub>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;con  J<sub>t</sub> = dh<sub>t</sub> / dh<sub>t-1</sub>"),
    b("<b>RNN simple:</b> h<sub>t</sub> = tanh(W x<sub>t</sub> + U h<sub>t-1</sub>), entonces J<sub>t</sub> = diag(1 - h<sub>t</sub><super>2</super>) · U y el producto queda acotado por ||U||<super>T-k</super>. Si ||U|| &lt; 1 el gradiente desaparece; si es mayor, puede explotar."),
    b("<b>GRU:</b> con una compuerta de actualización z<sub>t</sub> = sigmoid(W<sub>z</sub> x<sub>t</sub> + U<sub>z</sub> h<sub>t-1</sub>) y un candidato g<sub>t</sub>, el estado se combina como:"),
    f("h<sub>t</sub> = z<sub>t</sub> · h<sub>t-1</sub> + (1 - z<sub>t</sub>) · g<sub>t</sub>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;por lo tanto  J<sub>t</sub> = diag(z<sub>t</sub>) + (términos por g<sub>t</sub> y z<sub>t</sub>)"),
    p("El término diag(z<sub>t</sub>) es una vía aditiva que no se multiplica por U ni por la derivada de tanh: si la red aprende z<sub>t</sub> cercano a 1, el gradiente atraviesa muchos pasos sin atenuarse. Mitiga la desaparición, aunque no la elimina. Contra la explosión se recorta la norma del gradiente a 1,0."),
    p("Para evaluar la elección, las tres celdas se entrenaron con el mismo protocolo y se eligió por MAE de validación:"),
    table(
        [["Celda", "Parámetros", "Mejor época", "MAE validación", "MAE prueba", "RMSE prueba"]]
        + [[label, f"{cells[key]['parameters']:,}".replace(",", "."), str(cells[key]["best_epoch"]), n(cells[key]["validation_mae"]), n(cells[key]["test_mae"]), n(cells[key]["test_rmse"])]
           for key, label in [("gru", "GRU (elegida)"), ("lstm", "LSTM"), ("simple_rnn", "RNN simple")]],
        [3.6 * cm, 2.4 * cm, 2.4 * cm, 3.0 * cm, 2.6 * cm, 2.6 * cm],
    ),
    Spacer(1, 0.2 * cm),
    b(f"<b>Resultado:</b> la GRU tiene el menor error de validación con {100 * (1 - cells['gru']['parameters'] / cells['lstm']['parameters']):.0f}% menos parámetros que la LSTM. La RNN simple queda cerca porque esta serie tiene dependencia corta: aquí el gradiente no necesita recorrer muchos pasos. La ventaja de las compuertas aparece con dependencias largas, y la GRU la ofrece con menos parámetros."),
    b("<b>Hiperparámetros:</b> 32 unidades (tanh en el estado, sigmoid en las compuertas), densa de 16 unidades ReLU y salida lineal, porque es una regresión sin rango fijo. Pérdida MSE, Adam con tasa 0,001, lotes de 32, orden cronológico conservado y early stopping con paciencia 7 (máximo 55 épocas). La ventana de 36 pasos cubre tres cuartos del ciclo estacional."),

    p("7. GRU: resultados y sesgo y varianza", "SectionCustom"),
    p("Como la serie es sintética se conoce su parte predecible: la ecuación verdadera solo falla por el ruido, lo que fija un piso de error. La persistencia (repetir el último valor) es la referencia ingenua."),
    table([
        ["Modelo", "MAE prueba", "RMSE prueba"],
        ["GRU", n(gru["test_mae"]), n(gru["test_rmse"])],
        ["Persistencia", n(gru["persistence_test_mae"]), n(gru["persistence_test_rmse"])],
        ["Piso de ruido (ecuación verdadera)", n(gru["noise_floor_test_mae"]), n(gru["noise_floor_test_rmse"])],
    ], [7.0 * cm, 3.5 * cm, 3.5 * cm]),
    Spacer(1, 0.2 * cm),
    b(f"<b>Sesgo:</b> la GRU mejora {n(100 * (1 - gru['test_mae'] / gru['persistence_test_mae']), 1)}% a la persistencia y queda solo {n(100 * (gru['test_mae'] / gru['noise_floor_test_mae'] - 1), 1)}% por encima del piso: casi todo el error restante es ruido irreducible."),
    b(f"<b>Varianza:</b> MAE de {n(gru['training_mae'])} en entrenamiento, {n(gru['validation_mae'])} en validación y {n(gru['test_mae'])} en prueba. Las brechas son pequeñas y las curvas convergen sin divergir."),
    b(f"<b>Estabilidad predictiva:</b> el MAE por cuartos cronológicos de la prueba es {', '.join(n(v) for v in gru['test_mae_by_chronological_quarter'])}; no crece al alejarse del entrenamiento."),
    KeepTogether([
        Image(str(ROOT / "gru_learning_curve.png"), width=9.5 * cm, height=5.2 * cm),
        p(f"Pérdida MSE por época de la GRU en entrenamiento y validación (mejor época: {gru['best_epoch']}).", "CaptionCustom"),
    ]),

    p("8. Limitaciones y reproducción", "SectionCustom"),
    p("Los lotes son simulados y la serie es sintética y de dependencia corta: sirven para verificar el protocolo, pero no se extrapolan a datos reales. Para reproducir: pip install -r requirements.txt, python train.py, python predict.py, python -m unittest discover -s tests y python make_report.py. Los modelos están en artifacts/ (cnn.keras, gru.keras y sus .weights.h5) junto con todas las métricas."),
    p("Referencias: documentación de Keras (keras.io/api) y scikit-learn (scikit-learn.org/stable); Goodfellow, Bengio y Courville, Deep Learning, MIT Press, 2016, capítulos 9 y 10.", "SmallCustom"),
]

doc = BaseDocTemplate(str(DESTINATION), pagesize=(21.59 * cm, 27.94 * cm), leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.5 * cm, bottomMargin=1.7 * cm, title="Informe técnico: CNN y GRU con Keras", author=", ".join(AUTHORS))
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, leftPadding=0, bottomPadding=0, rightPadding=0, topPadding=0)
doc.addPageTemplates(PageTemplate(id="report", frames=frame, onPage=footer))
doc.build(story)
print(DESTINATION)
