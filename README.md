# Predictor de Demanda

Entrega del reto "Predictor de demanda con datos sucios e incompletos".

## Qué incluye

- `backend/`
  - parser reproducible de `dataset.pdf` a CSV
  - pipeline de limpieza, imputación y forecast
  - API FastAPI para modo interactivo
  - generación de artefactos JSON para frontend estático
- `frontend/`
  - dashboard HTML/CSS/JS sin build step
  - compatible con GitHub Pages
  - soporte para `demo estático` y `live API`

## Estructura

```text
backend/
  app/
  data/
  scripts/
frontend/
  data/
  index.html
  app.js
  styles.css
```

## Dataset

El dataset original viene en `dataset.pdf`. El script:

```bash
python3 backend/scripts/parse_pdf_dataset.py
```

genera:

```text
backend/data/dataset.csv
```

El parser reconstruye filas partidas y resuelve casos donde el PDF concatena producto y valor.

## Backend

Crear entorno e instalar dependencias:

```bash
python3 -m venv backend/.venv
backend/.venv/bin/pip install -r backend/requirements.txt
```

Generar artefactos:

```bash
backend/.venv/bin/python backend/scripts/build_artifacts.py
```

Ejecutar pipeline formal de limpieza + EDA + entrenamiento:

```bash
backend/.venv/bin/python backend/scripts/run_training_pipeline.py
```

Salidas principales:

- `backend/data/training/demand_history_clean.csv`
- `backend/data/training/demand_training_features.csv`
- `backend/data/training/eda_summary.json`
- `backend/data/training/eda_report.md`
- `backend/data/training/model_registry.json`

Levantar API:

```bash
backend/.venv/bin/python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Endpoints:

- `GET /api/health`
- `GET /api/meta`
- `GET /api/sku/{sku}?imputation=seasonal_median&horizon=12`

## Frontend

Servir `frontend/` por HTTP:

```bash
python3 -m http.server 4173 --directory frontend
```

Abrir:

```text
http://127.0.0.1:4173
```

## Modos de uso

### 1. Demo público en GitHub Pages

- El frontend usa `frontend/data/catalog.json` y `frontend/data/series/*.json`.
- No necesita backend.

### 2. Live API con backend local

Abrir:

```text
http://127.0.0.1:4173/?apiBaseUrl=http://127.0.0.1:8000
```

Si haces port forwarding desde VS Code, reemplaza esa URL por la URL forwarded.

## Por qué no Colab como backend principal

Sí se puede improvisar un backend en Colab con túneles, pero no es una base estable para una entrega:

- la sesión expira
- la URL cambia
- depende de mantener el notebook activo
- complica CORS y disponibilidad para una demo pública

Por eso la solución principal deja:

- frontend público en GitHub Pages
- backend local con port forwarding cuando necesites interacción en vivo

## VS Code

Se incluye `.vscode/launch.json` con:

- `Backend API`
- `Frontend Static Server`
- `Frontend + Backend`

## Modelado

Imputaciones:

- `seasonal_median`
- `linear`
- `forward_fill`

Modelos comparados:

- `seasonal_naive`
- `holt_winters`
- `seasonal_regression`

Selección:

- mejor modelo por `MAPE` sobre holdout

Estado actual:

- el backend sí entrena modelos reales por SKU
- deja registro de la mejor receta por SKU en `backend/data/training/model_registry.json`
- el frontend usa esos resultados precomputados para la demo pública

## Validaciones hechas

- PDF a CSV: `10400` filas, `520` semanas, `20` SKU
- backend: endpoints probados localmente
- frontend: probado en modo estático y en modo `live API`
