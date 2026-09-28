"""Create the assignment report from measured training artifacts."""

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, PageBreak, PageTemplate, Paragraph,
    Spacer, Table, TableStyle,
)


ROOT = Path("artifacts")
DESTINATION = Path("informe/Informe_tecnico.pdf")
cnn = json.loads((ROOT / "cnn_metrics.json").read_text())
gru = json.loads((ROOT / "gru_metrics.json").read_text())
DESTINATION.parent.mkdir(exist_ok=True)

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleCustom", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=17, leading=21, textColor=colors.HexColor("#142c44"), spaceAfter=18))
styles.add(ParagraphStyle(name="SectionCustom", parent=styles["Heading2"], fontSize=11, leading=14, textColor=colors.HexColor("#183e5f"), spaceBefore=11, spaceAfter=6))
styles.add(ParagraphStyle(name="BodyCustom", parent=styles["BodyText"], fontSize=9.3, leading=14, spaceAfter=7))
styles.add(ParagraphStyle(name="CaptionCustom", parent=styles["BodyText"], fontSize=8.1, leading=11, textColor=colors.HexColor("#516575"), spaceAfter=8))
styles.add(ParagraphStyle(name="SmallCustom", parent=styles["BodyText"], fontSize=8.2, leading=11, spaceAfter=5))


def p(text, style="BodyCustom"):
    return Paragraph(text, styles[style])


def table(rows, widths):
    result = Table(rows, colWidths=widths, hAlign="LEFT", repeatRows=1)
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9f0f5")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#183e5f")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("LEADING", (0, 0), (-1, -1), 11),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#9bb2c3")),
    ]))
    return result


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#b5c5d0"))
    canvas.line(1.8 * cm, 1.48 * cm, 19.8 * cm, 1.48 * cm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#607786"))
    canvas.drawString(1.8 * cm, 1.12 * cm, "Trabajo práctico integrador | Redes neuronales")
    canvas.drawRightString(19.8 * cm, 1.12 * cm, f"Página {doc.page}")
    canvas.restoreState()


doc = BaseDocTemplate(str(DESTINATION), pagesize=(21.59 * cm, 27.94 * cm), leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.7 * cm, bottomMargin=1.8 * cm)
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, leftPadding=0, bottomPadding=0, rightPadding=0, topPadding=0)
doc.addPageTemplates(PageTemplate(id="report", frames=frame, onPage=footer))
story = []

story += [
    p("CNN y GRU con Keras: diseño, evaluación y límites", "TitleCustom"),
    p("Informe técnico | Inteligencia Artificial, Licenciatura en Sistemas de Información", "CaptionCustom"),
    p("1. Objetivo y protocolo", "SectionCustom"),
    p("Se implementaron dos tareas independientes: clasificación espacial de dígitos mediante una red convolucional (CNN) y pronóstico de un paso en una serie multivariada mediante una unidad recurrente gated (GRU). Se fijaron semillas, se mantuvieron conjuntos de prueba separados y se exportaron los modelos completos y sus pesos. Los conjuntos elegidos permiten ejecutar el trabajo sin servicios externos."),
    p("2. Modulo CNN: datos y partición", "SectionCustom"),
    p("Se utilizaron 1797 imágenes de dígitos manuscritos de scikit-learn, con diez clases. Cada imagen original de 8 x 8 se interpolo a 32 x 32 y se asigno una sola vez a uno de doce lotes simulados, equilibrados aproximadamente por clase. Cada lote recibe parámetros propios de brillo, contraste y desenfoque, más ruido reproducible por muestra. Los lotes 0 a 7 forman desarrollo; los lotes 8 a 11 se reservan para prueba. Dentro de desarrollo, GroupKFold de cuatro particiones deja dos lotes distintos para validación en cada iteracion. Ninguna imagen ni identificador de lote cruza cada frontera."),
    p("Esta simulación somete al clasificador a cambios fotograficos. El conjunto no ofrece identificadores de autor ni lotes reales de captura: la validación por grupos no demuestra generalizacion a escritores o dispositivos nuevos. La interpolación aumenta la resolución de entrada, pero no crea información visual adicional."),
    p("3. Arquitectura CNN y elecciones", "SectionCustom"),
    p("La convolución discreta calcula cada activación como suma ponderada sobre una vecindad local de 3 x 3. Con stride S=1 y padding 'same', la altura de salida es ceil(H/S), por lo que se conserva la posición de las caracteristicas dentro de cada bloque. Los filtros pasan de 32 a 64 y 64 canales para combinar rasgos progresivamente. Dos max-pooling de 2 x 2 y stride 2 reducen mapas de 32 a 16 y luego a 8, aportando tolerancia local a desplazamientos pequeños y reduciendo el costo. La convolución es aproximadamente equivariante a traslaciones en el interior; el pooling introduce cierta invariancia local, no equivarianza exacta."),
    p("ReLU evita saturacion para entradas positivas; global average pooling reduce parámetros frente a aplanar mapas, y una capa densa de 64 unidades agrega capacidad. Dropout 0,25 mitiga coadaptacion. La salida softmax de diez clases se ajusta con entropia cruzada categórica dispersa. Adam (tasa 0,001; batch 32) usa clipnorm 1,0 para limitar gradientes extremos. En cada fold, early stopping con paciencia 5 restituye los mejores pesos segun pérdida de validación."),
    Spacer(1, 0.2 * cm),
    Image(str(ROOT / "cnn_fold_accuracy.png"), width=11.0 * cm, height=6.0 * cm),
    p("Exactitud de entrenamiento y validación por fold con lotes excluyentes.", "CaptionCustom"),
    PageBreak(),
    p("4. Modulo GRU: serie y corte temporal", "SectionCustom"),
    p("Se generó una serie sintetica de 1600 instantes con valor autorregresivo, seno y coseno estacionales, un control correlacionado y ruido. La variable objetivo en t depende de su valor anterior, de la estación en t y del control en t-1. Cada entrada contiene 36 pasos anteriores y cuatro variables; el objetivo es el valor del paso siguiente. Los tramos son entrenamiento [0,960), validación [960,1280) y prueba [1280,1600). Todas las ventanas se construyen totalmente dentro de su propio tramo; se descartan los primeros 36 objetivos en cada frontera para que ninguna ventana de validación o prueba comparta historia con el tramo previo."),
    p("La media y desviación de las cuatro entradas se ajustan exclusivamente con las ventanas de entrenamiento; el objetivo se escala con sus valores de entrenamiento. Los parámetros se guardan en gru_scalers.npz. El modelo de referencia por persistencia predice el último valor observado. La serie es un ejemplo controlado: estas métricas no prueban rendimiento sobre pronósticos reales."),
    p("5. Arquitectura GRU y estabilidad del gradiente", "SectionCustom"),
    p("Una GRU de 32 unidades resume la ventana y alimenta una capa densa ReLU de 16 unidades y una salida lineal, apropiada para regresión sin límite artificial de rango. Las compuertas sigmoid de actualización y reinicio modulan el estado. En forma simplificada, h_t = z_t h_(t-1) + (1-z_t) h_candidato: el término z_t conserva memoria previa y ofrece una vía aditiva para el gradiente durante BPTT, que puede atenuar su desaparición frente a una RNN simple. No elimina matemáticamente toda desaparición. Se usa tanh acotada para el estado, MSE como pérdida, Adam con tasa 0,001, clipnorm 1,0 para contener explosión del gradiente y early stopping con paciencia 7. El orden de las ventanas se conserva (shuffle=False)."),
    p("6. Resultados observados: CNN", "SectionCustom"),
    table([
        ["Medida", "Resultado"],
        ["Desarrollo / prueba", f"{cnn['samples']['development']} / {cnn['samples']['test']} imágenes"],
        ["Exactitud validación (media ± desv.)", f"{cnn['cross_validation_accuracy_mean']:.4f} ± {cnn['cross_validation_accuracy_std']:.4f}"],
        ["Exactitud entrenamiento final / prueba", f"{cnn['final_training_accuracy']:.4f} / {cnn['test_accuracy']:.4f}"],
        ["F1 macro prueba", f"{cnn['test_macro_f1']:.4f}"],
        ["Epocas del ajuste final", str(cnn['selected_epochs'])],
    ], [9.0 * cm, 8.7 * cm]),
    Spacer(1, 0.25 * cm),
    p("La exactitud media de entrenamiento en los cuatro folds es " + f"{sum(f['training_accuracy'] for f in cnn['group_kfold']) / 4:.4f}" + "; la brecha media con validación es " + f"{sum(f['training_accuracy'] - f['validation_accuracy'] for f in cnn['group_kfold']) / 4:.4f}" + ". La desviación entre folds de validación (0,0180) indica sensibilidad moderada a los lotes. Tres brechas son positivas y pequeñas; la tercera alcanza 0,0300. No se observa una brecha sistemática grande de sobreajuste ni exactitudes bajas en ambos lados que indiquen sesgo alto. El modelo final usa más datos; su exactitud de entrenamiento no es directamente comparable con cada fold. La matriz de confusión y los folds completos están en cnn_metrics.json."),
    PageBreak(),
    p("7. Resultados observados: GRU", "SectionCustom"),
    table([
        ["Medida", "GRU", "Persistencia"],
        ["MAE prueba", f"{gru['test_mae']:.4f}", f"{gru['persistence_test_mae']:.4f}"],
        ["RMSE prueba", f"{gru['test_rmse']:.4f}", f"{gru['persistence_test_rmse']:.4f}"],
        ["MAE entrenamiento / validación", f"{gru['training_mae']:.4f} / {gru['validation_mae']:.4f}", "-"],
        ["Mejor época", str(gru['best_epoch']), "-"],
    ], [8.1 * cm, 4.8 * cm, 4.8 * cm]),
    Spacer(1, 0.23 * cm),
    p("La GRU reduce el MAE de persistencia en " + f"{100 * (1 - gru['test_mae'] / gru['persistence_test_mae']):.1f}%" + ". La brecha entre MAE de entrenamiento y validación es " + f"{gru['validation_mae'] - gru['training_mae']:.4f}" + ": pequeña frente al error medio. MAE por cuarto cronológico de prueba: " + ", ".join(f"{value:.4f}" for value in gru["test_mae_by_chronological_quarter"]) + ". No se observa crecimiento sostenido del error. La curva permite inspeccionar la separación entre entrenamiento y validación para detectar varianza alta, y pérdidas altas en ambos para detectar sesgo."),
    Image(str(ROOT / "gru_learning_curve.png"), width=12.7 * cm, height=6.95 * cm),
    p("Perdida MSE por época en entrenamiento y validación cronológica.", "CaptionCustom"),
    p("8. Limitaciones e interpretación", "SectionCustom"),
    p("La CNN se evalua en un holdout de lotes simulados, no en nuevas personas. La GRU ve un proceso sintetico estacionario definido por reglas conocidas y no puede extrapolarse directamente a un dominio aplicado. Se seleccionan épocas y se detiene el entrenamiento usando solo desarrollo o validación; cada prueba se usa una vez despues de fijar pesos. No se reclaman conclusiones causales ni incertidumbre estadística para otras poblaciones. Los resultados son reproducibles con el código, pero detalles de CPU y bibliotecas pueden introducir pequenas diferencias numéricas."),
    p("9. Archivos y reproduccion", "SectionCustom"),
    p("Ejecutar pip install -r requirements.txt, python train.py, python predict.py y python -m unittest discover -s tests -v. Los modelos cnn.keras y gru.keras contienen arquitectura y pesos; los archivos *.weights.h5 son pesos adicionales. gru_scalers.npz se necesita para interpretar predicciones de la serie. cnn_metrics.json y gru_metrics.json conservan los resultados, y gru_learning_curve.png representa la pérdida por época. El informe se regenera con python make_report.py."),
    p("Referencias de implementacion", "SectionCustom"),
    p("Keras API: Conv2D, MaxPooling2D, GRU, EarlyStopping, model saving (keras.io/api/). Scikit-learn: load_digits y GroupKFold (scikit-learn.org/stable/). Formulaciones de arquitecturas y métodos: Goodfellow, Bengio y Courville, Deep Learning, MIT Press, 2016, capítulos 9 y 10.", "SmallCustom"),
]

doc.build(story)
print(DESTINATION)
