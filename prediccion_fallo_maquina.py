# =============================================================
#  PREDICCIÓN DE FALLO DE MÁQUINA — Gradient Boosting (XGBoost)
#  Caso: Planta Industrial — ¿La máquina fallará en las próximas horas?
#  Algoritmo: GradientBoostingClassifier (mismo concepto que XGBoost)
# =============================================================

# ── PASO 0: Configurar Sistema de Logging ────────────────────
import logging
import os
from datetime import datetime

# Crear carpeta outputs si no existe
os.makedirs("./outputs", exist_ok=True)

# Configurar logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"./outputs/ejecucion_{timestamp}.log"

# Crear logger
logger = logging.getLogger("ModeloPrediccion")
logger.setLevel(logging.DEBUG)

# Handler para archivo
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)

# Handler para consola
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Formato de logs
formato = logging.Formatter(
    '%(asctime)s - [%(levelname)-8s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(formato)
console_handler.setFormatter(formato)

# Agregar handlers al logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info("=" * 70)
logger.info("  INICIANDO PREDICCIÓN DE FALLO DE MÁQUINA — GRADIENT BOOSTING")
logger.info("=" * 70)
logger.info(f"📁 Archivo de log: {log_file}")

# ── PASO 1: Importar las herramientas necesarias ──────────────
logger.info("\n[PASO 1] Importando librerías...")
import numpy as np                          # Matemáticas y arrays
import pandas as pd                         # Manejo de datos en tablas
import matplotlib.pyplot as plt             # Gráficos
import matplotlib.gridspec as gridspec      # Layout de gráficos
import json

from sklearn.ensemble import GradientBoostingClassifier  # Nuestro modelo
from sklearn.model_selection import train_test_split     # Dividir datos
from sklearn.metrics import (                            # Métricas de evaluación
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

logger.info("✅ Todas las librerías importadas correctamente")
logger.debug(f"   NumPy v{np.__version__}")
logger.debug(f"   Pandas v{pd.__version__}")
logger.debug(f"   Scikit-learn: OK")

# ── PASO 2: Crear datos simulados de sensores industriales ───
logger.info("\n[PASO 2] Generando datos simulados de sensores...")

np.random.seed(42)   # Fijar semilla para resultados reproducibles
N = 500              # 500 registros de sensores históricos

logger.debug("   Creando dataset de operación NORMAL (400 registros)...")
# Generamos lecturas de sensores para operación NORMAL (400 registros)
normal = pd.DataFrame({
    "temperatura":     np.random.normal(70, 5, 400),    # ~70°C normal
    "vibracion":       np.random.normal(4.0, 0.5, 400), # ~4 Hz normal
    "presion_aceite":  np.random.normal(5.0, 0.3, 400), # ~5 bar normal
    "rpm":             np.random.normal(1500, 50, 400),  # ~1500 RPM
    "fallo":           0                                 # Sin fallo
})

logger.debug("   Creando dataset de situación de FALLO (100 registros)...")
# Generamos lecturas de sensores para situación de FALLO (100 registros)
fallo = pd.DataFrame({
    "temperatura":     np.random.normal(95, 8, 100),    # Temperatura alta
    "vibracion":       np.random.normal(8.5, 1.0, 100), # Vibración alta
    "presion_aceite":  np.random.normal(3.2, 0.5, 100), # Presión baja
    "rpm":             np.random.normal(1800, 100, 100), # RPM elevadas
    "fallo":           1                                 # Con fallo
})

logger.debug("   Mezclando datos aleatoriamente...")
# Unimos ambos datasets y mezclamos aleatoriamente
datos = pd.concat([normal, fallo], ignore_index=True)
datos = datos.sample(frac=1, random_state=42).reset_index(drop=True)

logger.info(f"✅ Dataset generado: {len(datos)} registros")
logger.info(f"   • Operación normal : {(datos.fallo == 0).sum()} registros ({(datos.fallo == 0).sum()/len(datos)*100:.1f}%)")
logger.info(f"   • Con fallo        : {(datos.fallo == 1).sum()} registros ({(datos.fallo == 1).sum()/len(datos)*100:.1f}%)")

# ── PASO 3: Separar Variables ────────────────────────────────
logger.info("\n[PASO 3] Separando variables de entrada y objetivo...")

X = datos[["temperatura", "vibracion", "presion_aceite", "rpm"]]
y = datos["fallo"]

logger.info(f"✅ Variables de entrada (X):")
for col in X.columns:
    logger.debug(f"   • {col}: min={X[col].min():.2f}, max={X[col].max():.2f}, media={X[col].mean():.2f}")
logger.info(f"✅ Variable objetivo (y): 'fallo' → 0=Normal | 1=Fallo")

# ── PASO 4: Dividir en Entrenamiento y Prueba ────────────────
logger.info("\n[PASO 4] Dividiendo datos en entrenamiento (80%) y prueba (20%)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% para prueba
    random_state=42,     # Resultado reproducible
    stratify=y           # Mantener proporción de clases
)

logger.info(f"✅ División completada:")
logger.info(f"   • Train (entrena) : {len(X_train)} registros (80%)")
logger.info(f"   • Test  (evalúa)  : {len(X_test)} registros (20%)")
logger.debug(f"   • Clases en train: {(y_train == 0).sum()} normal, {(y_train == 1).sum()} fallo")
logger.debug(f"   • Clases en test : {(y_test == 0).sum()} normal, {(y_test == 1).sum()} fallo")

# ── PASO 5: Crear y Entrenar el Modelo ───────────────────────
logger.info("\n[PASO 5] Creando y entrenando modelo Gradient Boosting...")

modelo = GradientBoostingClassifier(
    n_estimators=100,    # 100 árboles secuenciales
    learning_rate=0.1,   # Velocidad de aprendizaje
    max_depth=4,         # Profundidad máxima de cada árbol
    random_state=42
)

logger.debug("   Parámetros del modelo:")
logger.debug(f"   • n_estimators: 100")
logger.debug(f"   • learning_rate: 0.1")
logger.debug(f"   • max_depth: 4")
logger.debug("   Iniciando entrenamiento...")

modelo.fit(X_train, y_train)
logger.info("✅ Modelo entrenado exitosamente")

# ── PASO 6: Evaluar el Modelo ────────────────────────────────
logger.info("\n[PASO 6] Evaluando el modelo en datos de prueba...")

y_pred = modelo.predict(X_test)        # Predicciones del modelo
y_prob = modelo.predict_proba(X_test)  # Probabilidad de cada clase

# Calcular métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

logger.info(f"✅ Métricas del modelo:")
logger.info(f"   • Exactitud (Accuracy)  : {accuracy*100:.2f}%")
logger.info(f"   • Precisión (Precision) : {precision*100:.2f}%")
logger.info(f"   • Sensibilidad (Recall) : {recall*100:.2f}%")
logger.info(f"   • F1-Score              : {f1:.4f}")

# Reporte de clasificación
report_text = classification_report(
    y_test, y_pred,
    target_names=["Normal (0)", "Fallo (1)"],
    digits=2,
    output_dict=False
)
logger.debug("\n📊 REPORTE DETALLADO DE CLASIFICACIÓN:")
logger.debug("\n" + str(report_text))

# ── PASO 7: Caso de Uso Real — Nueva Lectura de Sensores ─────
logger.info("\n[PASO 7] Realizando predicción con nueva lectura de sensores...")

nueva_lectura = pd.DataFrame({
    "temperatura":    [92],   # Temperatura alta → señal de alerta
    "vibracion":      [7.8],  # Vibración elevada → preocupante
    "presion_aceite": [3.5],  # Presión baja → señal de alerta
    "rpm":            [1750]  # RPM elevadas
})

logger.info(f"📊 Datos de entrada:")
logger.info(f"   • Temperatura    : {nueva_lectura['temperatura'].values[0]}°C  (normal: ~70°C)")
logger.info(f"   • Vibración      : {nueva_lectura['vibracion'].values[0]} Hz  (normal: ~4 Hz)")
logger.info(f"   • Presión aceite : {nueva_lectura['presion_aceite'].values[0]} bar (normal: ~5 bar)")
logger.info(f"   • RPM            : {nueva_lectura['rpm'].values[0]}     (normal: ~1500)")

prediccion = modelo.predict(nueva_lectura)[0]
probabilidad = modelo.predict_proba(nueva_lectura)[0]

logger.info(f"\n🔮 Resultado de la predicción:")
logger.info(f"   • Estado predicho         : {'❌ FALLO DETECTADO' if prediccion == 1 else '✅ OPERACIÓN NORMAL'}")
logger.info(f"   • Probabilidad de fallo   : {probabilidad[1]*100:.2f}%")
logger.info(f"   • Probabilidad de normalidad : {probabilidad[0]*100:.2f}%")

if probabilidad[1] >= 0.8:
    accion = "⚠️  ACCIÓN: Programar mantenimiento URGENTE"
    logger.warning(accion)
elif probabilidad[1] >= 0.5:
    accion = "⚠️  ACCIÓN: Monitorear cada 30 minutos"
    logger.warning(accion)
else:
    accion = "✅  ACCIÓN: Sin intervención necesaria"
    logger.info(accion)

# ── PASO 8: Importancia de Variables ─────────────────────────
logger.info("\n[PASO 8] Calculando importancia de variables...")

importancias = pd.Series(
    modelo.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

logger.info(f"✅ Importancia de variables (sensores más críticos):")
for i, (var, imp) in enumerate(importancias.items(), 1):
    barra = "█" * int(imp * 40)
    logger.info(f"   {i}. {var:<18} {barra} {imp*100:.2f}%")

# ── PASO 9: Guardar Resultados en JSON ───────────────────────
logger.info("\n[PASO 9] Guardando resultados en archivo JSON...")

resultados = {
    "timestamp": datetime.now().isoformat(),
    "modelo": {
        "tipo": "GradientBoostingClassifier",
        "parametros": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 4
        }
    },
    "metricas_globales": {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    },
    "datos": {
        "total_registros": len(datos),
        "registros_normales": int((datos.fallo == 0).sum()),
        "registros_fallo": int((datos.fallo == 1).sum()),
        "split_train": len(X_train),
        "split_test": len(X_test)
    },
    "prediccion_nueva_lectura": {
        "entrada": {
            "temperatura": float(nueva_lectura['temperatura'].values[0]),
            "vibracion": float(nueva_lectura['vibracion'].values[0]),
            "presion_aceite": float(nueva_lectura['presion_aceite'].values[0]),
            "rpm": float(nueva_lectura['rpm'].values[0])
        },
        "prediccion": int(prediccion),
        "probabilidad_normal": float(probabilidad[0]),
        "probabilidad_fallo": float(probabilidad[1]),
        "accion_recomendada": accion
    },
    "importancia_variables": {
        str(var): float(imp) for var, imp in importancias.items()
    },
    "matriz_confusion": {
        "TP": int(confusion_matrix(y_test, y_pred)[1, 1]),
        "TN": int(confusion_matrix(y_test, y_pred)[0, 0]),
        "FP": int(confusion_matrix(y_test, y_pred)[0, 1]),
        "FN": int(confusion_matrix(y_test, y_pred)[1, 0])
    }
}

json_file = f"./outputs/resultados_{timestamp}.json"
with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(resultados, f, indent=4, ensure_ascii=False)

logger.info(f"✅ Resultados guardados en: {json_file}")

# ── PASO 10: Generar Gráficos ─────────────────────────────────
logger.info("\n[PASO 10] Generando gráficos...")

fig = plt.figure(figsize=(14, 10))
fig.suptitle(
    "Predicción de Fallo de Máquina — Gradient Boosting (XGBoost)",
    fontsize=14, fontweight="bold", color="#1A2744", y=0.98
)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# ── Gráfico 1: Distribución de datos por clase ───────────────
logger.debug("   Generando Gráfico 1: Distribución de datos...")
ax1 = fig.add_subplot(gs[0, 0])
colores = {"Normal": "#1A5C3A", "Fallo": "#8B3A2A"}
for label, color, subset in [
    ("Normal", "#1A5C3A", datos[datos.fallo == 0]),
    ("Fallo",  "#8B3A2A", datos[datos.fallo == 1])
]:
    ax1.scatter(
        subset["temperatura"], subset["vibracion"],
        c=color, label=label, alpha=0.5, s=20, edgecolors="none"
    )
ax1.scatter(
    nueva_lectura["temperatura"], nueva_lectura["vibracion"],
    c="#C9A84C", s=200, marker="*", zorder=5, label="Nueva lectura"
)
ax1.set_xlabel("Temperatura (°C)", fontsize=9)
ax1.set_ylabel("Vibración (Hz)", fontsize=9)
ax1.set_title("Temperatura vs Vibración por clase", fontsize=10, fontweight="bold")
ax1.legend(fontsize=8)
ax1.set_facecolor("#F0F2F8")
ax1.grid(True, alpha=0.3)

# ── Gráfico 2: Importancia de Variables ──────────────────────
logger.debug("   Generando Gráfico 2: Importancia de variables...")
ax2 = fig.add_subplot(gs[0, 1])
colores_vars = ["#C9A84C", "#1A2744", "#5B8DB8", "#4DB8A0"]
bars = ax2.barh(
    importancias.index[::-1],
    np.array(importancias.values[::-1]),
    color=colores_vars[::-1], edgecolor="white", height=0.6
)
for bar, val in zip(bars, importancias.values[::-1]):
    ax2.text(
        bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
        f"{val*100:.1f}%", va="center", fontsize=9, color="#1A2744", fontweight="bold"
    )
ax2.set_xlabel("Importancia relativa", fontsize=9)
ax2.set_title("¿Qué sensor predice mejor el fallo?", fontsize=10, fontweight="bold")
ax2.set_facecolor("#F0F2F8")
ax2.grid(True, alpha=0.3, axis="x")
ax2.set_xlim(0, importancias.max() * 1.25)

# ── Gráfico 3: Matriz de Confusión ───────────────────────────
logger.debug("   Generando Gráfico 3: Matriz de confusión...")
ax3 = fig.add_subplot(gs[1, 0])
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Fallo"]
)
disp.plot(ax=ax3, colorbar=False, cmap="Blues")
ax3.set_title("Matriz de Confusión\n(¿qué predijo bien el modelo?)", fontsize=10, fontweight="bold")
ax3.set_xlabel("Predicción", fontsize=9)
ax3.set_ylabel("Real", fontsize=9)

# ── Gráfico 4: Probabilidad de Fallo por Registro ────────────
logger.debug("   Generando Gráfico 4: Probabilidad de fallo...")
ax4 = fig.add_subplot(gs[1, 1])
probs_fallo = modelo.predict_proba(X_test)[:, 1]
colores_puntos = ["#8B3A2A" if p >= 0.5 else "#1A5C3A" for p in probs_fallo]
ax4.scatter(
    range(len(probs_fallo)), probs_fallo,
    c=colores_puntos, alpha=0.7, s=18, edgecolors="none"
)
ax4.axhline(0.5, color="#C9A84C", linewidth=1.5, linestyle="--", label="Umbral 50%")
ax4.axhline(0.8, color="#8B3A2A", linewidth=1.2, linestyle=":", label="Umbral crítico 80%")
ax4.set_xlabel("Registro de prueba", fontsize=9)
ax4.set_ylabel("Probabilidad de fallo", fontsize=9)
ax4.set_title("Probabilidad predicha por registro\n(rojo = predijo fallo)", fontsize=10, fontweight="bold")
ax4.legend(fontsize=8)
ax4.set_facecolor("#F0F2F8")
ax4.grid(True, alpha=0.3)
ax4.set_ylim(-0.05, 1.05)

plot_file = f"./outputs/resultado_modelo_{timestamp}.png"
plt.savefig(
    plot_file,
    dpi=150, bbox_inches="tight",
    facecolor="white"
)
logger.info(f"✅ Gráfico guardado: {plot_file}")

plt.close()

# ── RESUMEN FINAL ─────────────────────────────────────────────
logger.info("\n" + "=" * 70)
logger.info("  📊 RESUMEN DE LA EJECUCIÓN")
logger.info("=" * 70)
logger.info(f"Archivos generados:")
logger.info(f"  ✅ Log de ejecución    : {log_file}")
logger.info(f"  ✅ Resultados JSON     : {json_file}")
logger.info(f"  ✅ Gráficos            : {plot_file}")
logger.info(f"\nMétricas finales:")
logger.info(f"  • Accuracy : {accuracy*100:.2f}%")
logger.info(f"  • Precision: {precision*100:.2f}%")
logger.info(f"  • Recall   : {recall*100:.2f}%")
logger.info(f"  • F1-Score : {f1:.4f}")
logger.info("\n" + "=" * 70)
logger.info("  ✅ Script finalizado exitosamente")
logger.info("=" * 70)