# Prompt para agente Windows con GPU

Usa este prompt tal cual o ajústalo mínimamente si cambias rutas.

---

Actúa como un ML Engineer senior especializado en series temporales multiserie, forecasting retail y entrenamiento reproducible en Windows con GPU NVIDIA.

Trabaja sobre este repositorio local:

- repo root: `RetoEstudia`

Archivos de entrada ya preparados:

- dataset original parseado: `backend/data/dataset.csv`
- dataset limpio por receta base: `backend/data/training/demand_history_clean.csv`
- dataset train-ready: `backend/data/training/demand_training_features.csv`
- resumen EDA: `backend/data/training/eda_summary.json`
- reporte EDA legible: `backend/data/training/eda_report.md`
- registry actual de modelos clásicos: `backend/data/training/model_registry.json`

Objetivo:

1. Auditar la calidad del dataset limpio y confirmar que no haya leakage temporal.
2. Construir un pipeline de entrenamiento reproducible para forecasting de demanda semanal por SKU.
3. Comparar un baseline clásico ya existente contra al menos un modelo tabular potente y, si la relación señal/datos lo justifica, un modelo deep learning con GPU.
4. Dejar artefactos versionables, métricas comparables y un script de inferencia.

Restricciones y criterios:

- Frecuencia temporal: semanal.
- Series: 20 SKU.
- Longitud histórica: 520 semanas por SKU.
- Horizonte objetivo: 4, 8 y 12 semanas.
- Métrica principal: `MAPE`.
- Métricas secundarias: `MAE`, `RMSE`, `sMAPE`.
- Split temporal obligatorio:
  - train: desde el inicio hasta `t - 24`
  - validation: siguientes 12 semanas
  - test: últimas 12 semanas
- No mezclar información futura en features de lag, rolling o encoding.
- Mantener reproducibilidad con seeds fijas.
- Registrar configuración, métricas y rutas de artefactos.

Qué debes hacer:

## 1. Auditoría inicial

- Leer `backend/data/training/eda_summary.json` y `backend/data/training/demand_training_features.csv`.
- Verificar:
  - cantidad de filas esperada
  - 20 SKU
  - continuidad semanal por SKU
  - porcentaje imputado
  - columnas con valores extremos residuales
- Generar una nota corta de hallazgos y riesgos.

## 2. Ingeniería de variables

Crear un dataset de features orientado a forecasting multiserie con:

- identificadores:
  - `sku`
  - `producto`
- tiempo:
  - `fecha`
  - `year`
  - `month`
  - `iso_week`
  - `time_index`
- target:
  - `cleaned_value`
- flags:
  - `is_imputed`
  - `was_anomaly`
- lags:
  - `lag_1`, `lag_2`, `lag_4`, `lag_8`, `lag_12`, `lag_26`, `lag_52`
- rolling windows sobre `cleaned_value.shift(1)`:
  - mean/std/min/max para ventanas `4`, `8`, `12`, `26`, `52`
- estacionalidad:
  - seno/coseno de semana del año
  - seno/coseno de mes
- cambios:
  - delta vs `lag_1`
  - delta porcentual vs `lag_4` y `lag_52`

Todos los features deben construirse sin fuga temporal.

## 3. Baselines

Reproducir y validar baselines:

- seasonal naive
- Holt-Winters
- seasonal regression

Tomar como referencia `backend/data/training/model_registry.json`.

## 4. Modelo tabular principal

Entrenar un modelo tabular fuerte usando GPU si está disponible. Prioridad:

1. CatBoost con soporte GPU
2. XGBoost GPU
3. LightGBM GPU si el entorno lo soporta bien

Configurar búsqueda razonable de hiperparámetros, sin sobreoptimizar:

- profundidad
- learning rate
- regularización
- número de árboles
- subsample / colsample

Entrenar para horizontes:

- estrategia recursive
  o
- direct multi-horizon si el diseño queda más limpio

Comparar claramente contra los baselines.

## 5. Modelo deep learning opcional pero recomendado si el tiempo da

Si el entorno tiene CUDA bien configurado, probar uno de estos:

- NHITS
- N-BEATSx
- Temporal Fusion Transformer

Usar PyTorch Lightning o framework equivalente.

No entrenes un modelo deep si observas que:

- el dataset es demasiado pequeño para justificarlo
- el costo computacional supera el beneficio frente al modelo tabular

Si decides no entrenarlo, documenta por qué.

## 6. Entregables obligatorios

Crear dentro del repo una carpeta tipo:

- `backend/gpu_training/`

Y dejar:

- script reproducible de entrenamiento
- script de inferencia
- archivo de configuración
- `metrics_summary.json`
- `experiment_report.md`
- modelo serializado
- tabla CSV con predicciones de test y forecast a 12 semanas por SKU

## 7. Formato del reporte

Quiero que respondas con:

1. resumen ejecutivo
2. decisiones de limpieza y features
3. tabla comparativa de modelos
4. mejor modelo final y por qué
5. riesgos y límites
6. rutas exactas de archivos generados

## 8. Criterio de calidad

No quiero una demo superficial. Quiero un pipeline serio, reproducible y defendible técnicamente.

Si encuentras inconsistencias en el dataset limpio, arréglalas primero y documenta el cambio.

---

Si debes elegir una sola ruta por relación costo/beneficio, prioriza:

- CatBoost GPU con features temporales bien hechas

y compáralo contra los baselines clásicos ya presentes en el repo.
