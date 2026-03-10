# 🏭 Predicción de Fallo de Máquina — Gradient Boosting (XGBoost)

> **Machine Learning aplicado a mantenimiento predictivo industrial**  
> Modelo que predice si una máquina fallará basándose en lecturas de sensores en tiempo real.

---

## 📌 Descripción

Este proyecto implementa un modelo de **Gradient Boosting** (equivalente a XGBoost) para predecir fallos en maquinaria industrial antes de que ocurran, usando datos de 4 sensores clave:

| Sensor | Descripción | Valor normal |
|---|---|---|
| `temperatura` | Temperatura del motor | ~70 °C |
| `vibracion` | Vibración del eje | ~4.0 Hz |
| `presion_aceite` | Presión del sistema de lubricación | ~5.0 bar |
| `rpm` | Revoluciones por minuto | ~1500 RPM |

**Variable objetivo:** `fallo` → `0` = Operación normal · `1` = Fallo detectado

---

## 🎯 Objetivo

Anticipar fallos de maquinaria con suficiente tiempo para programar mantenimiento, reduciendo:
- Paradas no programadas
- Costos de reparación de emergencia
- Pérdidas de producción

---

## 🗂️ Estructura del Proyecto

```
ml-fallo-maquina-xgboost/
│
├── prediccion_fallo_maquina.py   # Script principal con el modelo
├── resultado_modelo.png          # Gráficos generados por el modelo
└── README.md                     # Este archivo
```

---

## ⚙️ Requisitos

```bash
pip install scikit-learn pandas numpy matplotlib
```

> Para usar XGBoost real en producción:
> ```bash
> pip install xgboost
> ```
> Y reemplazar en el script:
> ```python
> # Línea original
> from sklearn.ensemble import GradientBoostingClassifier
> modelo = GradientBoostingClassifier(...)
>
> # Reemplazo con XGBoost
> from xgboost import XGBClassifier
> modelo = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4)
> ```

---

## 🚀 Cómo ejecutar

```bash
python prediccion_fallo_maquina.py
```

### Salida esperada en consola

```
=======================================================
  PREDICCIÓN DE FALLO DE MÁQUINA — GRADIENT BOOSTING
=======================================================

📦 Dataset: 500 registros de sensores
   ✅ Operación normal : 400 registros
   ❌ Con fallo        : 100 registros

✂️  División de datos:
   Train (entrena) : 400 registros (80%)
   Test  (evalúa)  : 100  registros (20%)

🔧 CASO REAL — Nueva lectura de sensores:
   Temperatura    : 92°C   (normal: ~70°C)
   Vibración      : 7.8 Hz (normal: ~4 Hz)
   Presión aceite : 3.5 bar (normal: ~5 bar)
   RPM            : 1750    (normal: ~1500)

   🔮 Predicción del modelo    : ❌ FALLO DETECTADO
   📊 Probabilidad de fallo    : 100.0%
   ⚠️  ACCIÓN: Programar mantenimiento URGENTE
```

---

## 📊 Gráficos Generados

El script genera automáticamente 4 visualizaciones:

| Gráfico | Descripción |
|---|---|
| **Temperatura vs Vibración** | Separación visual entre clase Normal y Fallo |
| **Importancia de Variables** | Qué sensor aportó más al modelo |
| **Matriz de Confusión** | Aciertos y errores del modelo por clase |
| **Probabilidad por registro** | Nivel de confianza del modelo en cada predicción |

![Resultado del modelo](resultado_modelo.png)

---

## 🧠 Cómo funciona el algoritmo

```
Datos históricos (sensores)
        ↓
   Árbol 1  →  predice con error E1
        ↓
   Árbol 2  →  corrige E1, predice con error E2
        ↓
   Árbol 3  →  corrige E2 ...
        ↓
   ... (100 árboles)
        ↓
   Predicción final = votación ponderada de los 100 árboles
```

**Gradient Boosting** = construir árboles de decisión **secuencialmente**, donde cada árbol aprende de los errores del anterior. Es el mismo principio matemático de XGBoost.

---

## 📈 Métricas del Modelo

| Métrica | Clase Normal | Clase Fallo |
|---|---|---|
| **Precision** | 1.00 | 1.00 |
| **Recall** | 1.00 | 1.00 |
| **F1-Score** | 1.00 | 1.00 |

> **Nota sobre Recall:** en contexto industrial, el Recall de la clase Fallo es la métrica más crítica. Un Falso Negativo (fallo no detectado) puede costar 10x más que una alarma falsa.

---

## 🏭 Casos de Uso Reales Similares

| Industria | Aplicación | Resultado reportado |
|---|---|---|
| Minería de cobre | Predicción de fallo en molinos SAG | -75% paradas no programadas |
| Industria naval | Fallo de motor en cruceros | 10 días de anticipación |
| Planta cementera | Monitoreo de 14 sensores | -34% paradas en 3 meses |
| Refinería | Calidad de producto en tiempo real | 4h de ahorro por lote |

---

## 🔄 Próximos Pasos (roadmap)

- [ ] Conectar a datos reales de sensores (CSV / API SCADA)
- [ ] Agregar Feature Engineering (ventanas temporales, desviaciones)
- [ ] Implementar reentrenamiento automático (MLOps)
- [ ] Desplegar como API REST con FastAPI
- [ ] Agregar monitoreo de Data Drift

---

## 📚 Conceptos Clave

- **Overfitting:** cuando el modelo memoriza en vez de aprender — evitado con `max_depth=4`
- **Train/Test split:** 80% entrena, 20% evalúa — nunca evaluar con datos de entrenamiento
- **Feature Importance:** la vibración resultó ser la variable más predictiva del fallo
- **Umbral de decisión:** probabilidad ≥ 80% → mantenimiento urgente · ≥ 50% → monitorear

---

## 👤 Autor

Desarrollado como demostración práctica de Machine Learning aplicado a mantenimiento predictivo industrial.

---

## 📄 Licencia

MIT — libre para uso académico y comercial.