# GPU Training Report

## 1. Resumen ejecutivo

Se entrenó CatBoost GPU multiserie con validación temporal estricta. En la imputación `linear`, el mejor baseline global fue `holt_winters` con MAPE `4.09`, mientras `catboost_gpu_global` logró MAPE `4.83`. Aun así, el modelo GPU ganó en `10` combinaciones SKU/imputación y queda integrado como candidato opcional.

## 2. Decisiones de limpieza y features

- Se construyó un modelo CatBoost GPU multiserie con `sku` y `producto` como categóricas.
- Las features incluyen lags `1/2/4/8/12/26/52`, ventanas rolling `4/8/12/26/52`, estacionalidad cíclica y cambios porcentuales.
- Se respetó un split temporal estricto: train hasta `t-24`, validación siguiente `12` semanas, test últimas `12` semanas.
- Se evaluaron las imputaciones `linear`, `forward_fill` y `seasonal_median` para no sesgar el modelo GPU a una sola receta.

## 3. Tabla comparativa de modelos

### linear

- `seasonal_naive`: MAPE `6.74` | MAE `69.55` | RMSE `98.97` | sMAPE `7.01`
- `holt_winters`: MAPE `4.09` | MAE `39.6` | RMSE `53.5` | sMAPE `4.1`
- `seasonal_regression`: MAPE `4.28` | MAE `40.99` | RMSE `56.75` | sMAPE `4.23`
- `catboost_gpu_global`: MAPE `4.83` | MAE `47.5` | RMSE `67.2` | sMAPE `4.81`

### forward_fill

- `seasonal_naive`: MAPE `7.15` | MAE `73.96` | RMSE `108.22` | sMAPE `7.42`
- `holt_winters`: MAPE `4.51` | MAE `42.6` | RMSE `56.97` | sMAPE `4.51`
- `seasonal_regression`: MAPE `4.47` | MAE `42.54` | RMSE `57.62` | sMAPE `4.42`
- `catboost_gpu_global`: MAPE `5.23` | MAE `52.65` | RMSE `76.56` | sMAPE `5.22`

### seasonal_median

- `seasonal_naive`: MAPE `10.87` | MAE `109.36` | RMSE `163.09` | sMAPE `11.37`
- `holt_winters`: MAPE `8.48` | MAE `79.17` | RMSE `113.51` | sMAPE `8.27`
- `seasonal_regression`: MAPE `8.54` | MAE `78.38` | RMSE `113.15` | sMAPE `8.22`
- `catboost_gpu_global`: MAPE `9.14` | MAE `88.48` | RMSE `121.55` | sMAPE `8.97`

## 4. Mejor modelo final y por qué

Mantener `linear` como receta principal y promover el modelo GPU solo cuando mejore el MAPE del mejor baseline clásico en ese SKU. El backend queda preparado para mezclar ambos mundos sin forzar una sustitución global.

## 5. Riesgos y límites

- CatBoost GPU no domina globalmente a Holt-Winters o regresión estacional en este dataset.
- Las bandas del modelo GPU usan desviación estándar de residuales de validación por horizonte, no cuantiles dedicados.
- El modelo deep se omite por relación costo/beneficio: solo `20` series y `520` semanas por SKU.

## 6. Rutas exactas de archivos generados

- `config`: `backend/gpu_training/artifacts/config.json`
- `metrics_summary`: `backend/gpu_training/artifacts/metrics_summary.json`
- `experiment_report`: `backend/gpu_training/artifacts/experiment_report.md`
- `model_registry`: `backend/gpu_training/artifacts/model_registry.json`
- `predictions`: `backend/gpu_training/artifacts/predictions.csv`
- `models_dir`: `backend/gpu_training/artifacts/models`
