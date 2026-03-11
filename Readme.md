# 🏭 Predicción de Fallo de Máquina — Gradient Boosting (XGBoost)

> **Machine Learning aplicado a mantenimiento predictivo industrial**  
> Modelo que predice si una máquina fallará basándose en lecturas de sensores en tiempo real.

---

## 📌 Descripción

Este proyecto implementa un modelo de **Gradient Boosting** (equivalente a XGBoost) para predecir fallos en maquinaria industrial antes de que ocurran, usando datos de **7 sensores** clave:

| Sensor | Descripción | Valor normal | Impacto en fallo |
|---|---|---|---|
| `temperatura` | Temperatura del motor | ~75 °C | Media |
| `vibracion` | Vibración del eje | ~5.2 Hz | ⭐⭐⭐ Máximo |
| `presion_aceite` | Presión del sistema de lubricación | ~4.8 bar | Media |
| `rpm` | Revoluciones por minuto | ~1520 RPM | Bajo |
| `humedad` | Humedad ambiental | ~45% | Bajo |
| `velocidad_aire` | Velocidad de aire refrigeración | ~8.5 m/s | Bajo |
| `carga` | Carga de operación | ~65% | Bajo |

**Variable objetivo:** `fallo` → `0` = Operación normal · `1` = Fallo detectado

---

## 🎯 Objetivo

Anticipar fallos de maquinaria con suficiente tiempo para programar mantenimiento preventivo, reduciendo:
- ❌ Paradas no programadas → ✅ Paradas controladas
- ❌ Costos de reparación de emergencia (5-10x más)
- ❌ Pérdidas de producción
- ✅ Vida útil de equipos

---

## 📊 Resultados Logrados

### Métricas Finales del Modelo
```
✅ Accuracy (Exactitud)     : 88.75%  ← Bien clasificados
✅ Precision (Precisión)    : 88.89%  ← Cuando dice fallo, acierta 89%
⚠️  Recall (Sensibilidad)   : 80.00%  ← Detecta 80% de fallos reales
📈 F1-Score                 : 0.8421  ← Balance Precisión-Recall
```

### Matriz de Confusión
```
                Predicción
              Normal  Fallo
Real Normal     94      6    ← 6 falsas alarmas (aceptable)
Real Fallo      12     48    ← 12 fallos no detectados (crítico)
```

---

## 🗂️ Estructura del Proyecto

```
prediccion-fallo-industrial-gradient-boosting/
│
├── prediccion_fallo_maquina.py              # Script principal
├── Readme.md                                 # Documentación general
├── LECCIONES_APRENDIDAS.md                  # 📌 Guía de mejores prácticas
├── requiremts.txt                           # Dependencias
└── outputs/                                  # Resultados generados
    ├── ejecucion_*.log                      # Logs de ejecución
    ├── resultados_*.json                    # Métricas en JSON
    └── resultado_modelo_*.png               # Gráficos
```

---

## ⚙️ Requisitos

```bash
# Crear entorno virtual (recomendado)
python3 -m venv venv
source venv/bin/activate  # En macOS/Linux

# Instalar dependencias
pip install -r requiremts.txt
```

### Contenido de `requiremts.txt`
```
scikit-learn>=1.8.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.5.0
```

> **Para producción (XGBoost real):**
> ```bash
> pip install xgboost>=2.0.0
> ```

---

## 🚀 Cómo ejecutar

```bash
cd prediccion-fallo-industrial-gradient-boosting
python3 prediccion_fallo_maquina.py
```

### Salida esperada en consola

```
======================================================================
  INICIANDO PREDICCIÓN DE FALLO DE MÁQUINA — GRADIENT BOOSTING
======================================================================

[PASO 2] Generando datos simulados de sensores...
✅ Dataset generado: 800 registros
   • Operación normal : 500 registros (62.5%)
   • Con fallo        : 300 registros (37.5%)

[PASO 3] Separando variables de entrada y objetivo...
✅ Variables de entrada (X): 7 características

[PASO 4] Dividiendo datos en entrenamiento (80%) y prueba (20%)...
✅ División completada:
   • Train (entrena) : 640 registros (80%)
   • Test  (evalúa)  : 160 registros (20%)

[PASO 5] Creando y entrenando modelo Gradient Boosting...
✅ Modelo entrenado exitosamente

[PASO 6] Evaluando el modelo en datos de prueba...
✅ Métricas del modelo:
   • Exactitud (Accuracy)  : 88.75%
   • Precisión (Precision) : 88.89%
   • Sensibilidad (Recall) : 80.00%
   • F1-Score              : 0.8421

[PASO 7] Realizando predicción con nueva lectura de sensores...
📊 Datos de entrada:
   • Temperatura    : 92°C
   • Vibración      : 7.8 Hz
   • Presión aceite : 3.5 bar
   • RPM            : 1750
   • Humedad        : 55%
   • Velocidad aire : 10.5 m/s
   • Carga          : 70%

🔮 Resultado de la predicción:
   • Estado predicho         : ❌ FALLO DETECTADO
   • Probabilidad de fallo   : 87.43%
   • Probabilidad de normalidad : 12.57%

⚠️  ACCIÓN: Programar mantenimiento URGENTE
```

---

## 📊 Gráficos Generados

El script genera automáticamente **4 visualizaciones** en `outputs/resultado_modelo_*.png`:

| Gráfico | Descripción | Interpretación |
|---|---|---|
| **Temperatura vs Vibración** | Scatter plot de dos sensores principales | Muestra separación entre clases |
| **Importancia de Variables** | Bar chart horizontal | Vibración > Presión > Temperatura |
| **Matriz de Confusión** | Heatmap 2x2 | TP, TN, FP, FN |
| **Probabilidad por registro** | Scatter plot temporal | Distribución de confianza |

---

## 🧠 Algoritmo: Gradient Boosting Explicado

```
PASO 1: Árbol 1
   ├─ Aprende patrones iniciales
   └─ Predice con error E1

PASO 2: Árbol 2
   ├─ Recibe residuos (errores) de Árbol 1
   ├─ Se especializa en corregir E1
   └─ Predice con error E2

PASO 3: Árbol 3 → Árbol 4 → ... → Árbol 50
   └─ Cada árbol corrige los errores anteriores

PREDICCIÓN FINAL:
   Resultado = Árbol1 + α×Árbol2 + α×Árbol3 + ... + α×Árbol50
   (donde α es learning_rate = 0.15)
```

### Ventajas sobre Decision Trees
- ✅ Mejor generalización (menos overfitting)
- ✅ Captura relaciones complejas entre sensores
- ✅ Proporciona importancia de variables
- ✅ Robusto ante datos ruidosos

---

## 🔍 Parámetros del Modelo (Configuración Final)

```python
GradientBoostingClassifier(
    n_estimators=50,        # Número de árboles secuenciales
    learning_rate=0.15,     # Velocidad de aprendizaje (contracción)
    max_depth=2,            # Profundidad máxima de cada árbol
    min_samples_split=10,   # Mínimo de muestras para dividir nodo
    min_samples_leaf=5,     # Mínimo de muestras por hoja
    subsample=0.8,          # Fracción de datos por árbol
    random_state=42         # Reproducibilidad
)
```

**¿Por qué estos valores?** → Ver **LECCIONES_APRENDIDAS.md**

---

## ⚠️ Problemas Comunes Evitados

### ❌ Problema 1: Overfitting (100% accuracy perfecto)
```
SÍNTOMA: Accuracy=100%, Precision=100%, Recall=100%
CAUSA:   max_depth muy alto, separación de clases obvia
SOLUCIÓN: max_depth=2, subsample=0.8, min_samples_split=10
```

### ❌ Problema 2: Desbalance de clases
```
SÍNTOMA: Modelo predice siempre "normal"
CAUSA:   Muchos más registros normales que fallos
SOLUCIÓN: stratify=y en train_test_split, aumentar datos de fallo
```

### ❌ Problema 3: Datos con separación artificial
```
SÍNTOMA: Temperatura normal=70°C, con fallo=95°C (demasiado obvio)
CAUSA:   Datos simulados no realistas
SOLUCIÓN: Aumentar varianza, agregar más sensores con ruido
```

---

## 📈 Mejoras Implementadas (Iteraciones)

| Iteración | Cambio | Accuracy | Impacto |
|-----------|--------|----------|--------|
| **v1** | Datos con 4 sensores, separación clara | 100.00% | ❌ Overfitting |
| **v2** | +3 sensores (7 total), mayor ruido | 88.75% | ✅ Realista |
| **v3** | max_depth=2, subsample=0.8 | 88.75% | ✅ Estable |
| **v4** | min_samples_split=10, min_samples_leaf=5 | 88.75% | ✅ Robusto |

---

## 🎯 Interpretación de Resultados

### Matriz de Confusión Analizada
```
Verdaderos Positivos (TP) = 48
  ✅ Máquina fallará → Modelo predijo fallo CORRECTO

Verdaderos Negativos (TN) = 94
  ✅ Máquina normal → Modelo predijo normal CORRECTO

Falsos Positivos (FP) = 6
  ⚠️ Máquina normal → Modelo predijo fallo (alarma falsa)
  IMPACTO: Costo de mantenimiento innecesario (~$500)

Falsos Negativos (FN) = 12
  ❌ Máquina fallará → Modelo predijo normal (NO DETECTÓ)
  IMPACTO: Parada de emergencia (~$50,000)
```

### Métricas Explicadas
- **Precision = TP/(TP+FP) = 88.89%** → De 100 alertas, 89 son verdaderos fallos
- **Recall = TP/(TP+FN) = 80%** → De 60 fallos reales, detectamos 48 (80%)
- **F1-Score = 2×(P×R)/(P+R) = 0.8421** → Balance entre precisión y recall

---

## 🏭 Casos de Uso en Producción

```python
# Lectura en tiempo real de sensores (ejemplo pseudocódigo)
while True:
    datos_sensor = leer_scada()  # Lee de PLC/SCADA
    probabilidad = modelo.predict_proba(datos_sensor)
    
    if probabilidad[1] >= 0.8:
        enviar_alerta_urgente()      # Parar máquina
        programar_mantenimiento()    # Notificar técnicos
    
    elif probabilidad[1] >= 0.5:
        enviar_alerta_moderada()     # Monitorear cada 30 min
    
    tiempo.sleep(300)  # Verificar cada 5 minutos
```

---

## 📚 Archivos de Salida Generados

Cada ejecución crea en `outputs/`:

1. **ejecucion_YYYYMMDD_HHMMSS.log** (7 KB)
   - Log completo con todos los detalles de entrenamiento
   - Útil para debugging y auditoría

2. **resultados_YYYYMMDD_HHMMSS.json** (1.3 KB)
   - Métricas en formato JSON (para APIs)
   - Predicción de ejemplo
   - Importancia de variables

3. **resultado_modelo_YYYYMMDD_HHMMSS.png** (195 KB)
   - 4 gráficos en una sola imagen
   - Calidad 150 DPI para presentaciones

---

## 🔗 Próximos Pasos (Roadmap)

### Fase 1: Mejoras Inmediatas ✅
- [x] Corregir overfitting (max_depth=2)
- [x] Agregar más características
- [x] Mejorar balance de clases
- [x] Documentar lecciones aprendidas

### Fase 2: Integración con Datos Reales (1-2 semanas)
- [ ] Conectar a base de datos histórica
- [ ] Feature Engineering (ventanas temporales, derivadas)
- [ ] Validación cruzada K-Fold
- [ ] Comparar con otros modelos (Random Forest, SVM)

### Fase 3: Despliegue en Producción (1 mes)
- [ ] API REST con FastAPI
- [ ] Modelo serializado (.pkl, .h5)
- [ ] Monitoreo de Data Drift
- [ ] Dashboard de predicciones en tiempo real

### Fase 4: MLOps & Automatización (2 meses)
- [ ] Reentrenamiento automático semanal
- [ ] Versionado de modelos (MLflow)
- [ ] A/B testing de mejoras
- [ ] Alertas ante degradación de métricas

---

## 📚 Referencias y Conceptos

### Definiciones Clave
- **Overfitting:** Modelo memoriza en vez de aprender generalizaciones
- **Underfitting:** Modelo demasiado simple, no captura patrones
- **Train/Test Split:** 80% entrena, 20% evalúa (nunca mezclar)
- **Stratification:** Mantener proporción de clases en split
- **Feature Importance:** Contribución relativa de cada característica
- **Threshold:** Punto de corte para decisión (default 0.5, ajustable)

### Algoritmos Relacionados
- **Random Forest:** Árboles paralelos (más robusto, menos interpretable)
- **XGBoost:** Gradient Boosting optimizado (production-ready, más rápido)
- **LightGBM:** Alternativa a XGBoost (menor consumo de memoria)
- **SVM:** Máquinas de soporte vectorial (bueno para datos balanceados)

### Métricas en Contexto Industrial
| Métrica | Fórmula | Importancia |
|---------|---------|------------|
| Accuracy | (TP+TN)/(Total) | General |
| Precision | TP/(TP+FP) | ⭐⭐⭐ Reducir falsas alarmas |
| Recall | TP/(TP+FN) | ⭐⭐⭐⭐ Detectar fallos reales |
| F1-Score | 2×P×R/(P+R) | Balance |

---

## 👤 Notas de Implementación

### Para Productivización
1. **Serializar el modelo:**
   ```python
   import pickle
   with open('modelo_fallo.pkl', 'wb') as f:
       pickle.dump(modelo, f)
   ```

2. **Crear API REST:**
   ```python
   from fastapi import FastAPI
   app = FastAPI()
   
   @app.post("/prediccion/")
   def predecir(temperatura: float, vibracion: float, ...):
       prediccion = modelo.predict([[...]])
       return {"fallo": int(prediccion), "probabilidad": float(proba)}
   ```

3. **Monitoreo en producción:**
   - Registrar todas las predicciones
   - Comparar predicción vs realidad (cuando se confirme)
   - Detectar data drift (cambio en distribución de sensores)
   - Reentrenar cada 3 meses

---

## 📄 Licencia

MIT — Libre para uso académico y comercial

## 📞 Soporte

Ver archivo: **LECCIONES_APRENDIDAS.md** para guía de troubleshooting