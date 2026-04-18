# Reporte técnico: limpieza, imputación y preparación del predictor de demanda

## 1. Propósito del trabajo

El objetivo del proyecto fue transformar un historial semanal de ventas con datos incompletos y anómalos en una base confiable para pronosticar demanda. La entrega debía cumplir dos metas al mismo tiempo:

1. producir una visualización pública en GitHub Pages;
2. dejar un backend reproducible que limpiara los datos, ajustara modelos y generara pronósticos.

Para resolver eso se construyó un pipeline completo que:

- extrajo el dataset desde `dataset.pdf`;
- clasificó los problemas de calidad del dato;
- limpió e imputó la serie por SKU;
- comparó varios modelos de forecast;
- publicó resultados precomputados para el frontend;
- dejó además salidas formales de EDA y entrenamiento.

---

## 2. Insumos y contexto

Se trabajó con dos PDFs:

- `3. Reto; Predictor de demanda.pdf`: documento del reto
- `dataset.pdf`: dataset fuente

El dataset final extraído contiene:

- `10400` filas
- `520` semanas
- `20` SKU
- rango temporal: `2014-01-06` a `2023-12-18`

Esto implica que el problema no era una única serie, sino un conjunto de `20` series semanales paralelas, una por SKU.

---

## 3. Qué hice exactamente

### 3.1 Extracción del PDF a CSV

El archivo `dataset.pdf` no venía en un formato CSV limpio. Primero se implementó un parser reproducible en:

- `backend/scripts/parse_pdf_dataset.py`

Ese parser hace lo siguiente:

1. extrae texto con `pdftotext -raw`;
2. reconstruye filas lógicas cuando el PDF parte una fila en dos líneas;
3. identifica las columnas base:
   - `fecha`
   - `sku`
   - `producto`
   - `unidades_vendidas`
   - `nota`
4. valida que cada semana tenga exactamente `20` registros;
5. exporta el resultado a:
   - `backend/data/dataset.csv`

### 3.2 Corrección del error estructural más importante

Durante la extracción apareció un problema concreto en el SKU de agua:

- `Agua_500ml_x24`

En varias filas el PDF pegaba el texto del producto con el número, por ejemplo:

- `Agua_500ml_x241435.0`

Si se parseaba de forma ingenua, el producto quedaba truncado y el valor numérico se inflaba artificialmente. Eso generaba cifras absurdas de demanda.

La corrección que implementé fue de dos pasadas:

1. construir un catálogo de nombres de producto por `SKU`;
2. usar ese catálogo para separar correctamente producto y valor cuando el PDF los concatenaba.

Esta decisión fue importante porque sin ella el pipeline habría quedado técnicamente funcional pero analíticamente incorrecto.

---

## 4. Cómo clasifiqué la información

La clasificación de la información se hizo en dos niveles:

### 4.1 Clasificación estructural

Cada registro quedó organizado con estas columnas:

- `fecha`: semana del registro
- `sku`: identificador del producto
- `producto`: nombre del producto
- `unidades_vendidas`: valor original
- `nota`: etiqueta de calidad original

### 4.2 Clasificación por calidad del dato

La columna `nota` fue la base de la clasificación de anomalías. A partir del dataset real, las etiquetas observadas fueron:

- `ok`
- `missing`
- `missing_bloque`
- `negativo`
- `outlier_alto`
- `outlier_bajo`
- `duplicado`

Conteo final por tipo:

- `ok`: `8492`
- `missing`: `601`
- `missing_bloque`: `470`
- `negativo`: `128`
- `outlier_alto`: `392`
- `outlier_bajo`: `233`
- `duplicado`: `84`

Con eso, el porcentaje global de datos inválidos o problemáticos fue:

- `18.35%`

### 4.3 Interpretación de cada categoría

Las traté así:

- `ok`: dato confiable
- `missing`: dato faltante aislado
- `missing_bloque`: bloque consecutivo de semanas faltantes
- `negativo`: valor inconsistente para demanda
- `outlier_alto`: pico anómalo no representativo
- `outlier_bajo`: valor anormalmente bajo
- `duplicado`: observación inflada por duplicación

Para limpieza, todas salvo `ok` se trataron como observaciones no confiables.

---

## 5. Cómo limpié el dataset

La limpieza se hizo por SKU, no mezclando productos entre sí. Esa decisión es clave porque cada producto tiene nivel, tendencia y estacionalidad distintos.

### 5.1 Regla de invalidez

Se definió como inválido todo registro cuya `nota` fuera alguna de estas:

- `duplicado`
- `missing`
- `missing_bloque`
- `negativo`
- `outlier_alto`
- `outlier_bajo`

Técnicamente, eso produjo dos columnas auxiliares:

- `is_invalid`
- `base_clean_value`

La lógica fue:

- si `nota == ok`, el valor se conserva;
- si no, el valor base para modelar pasa a `NaN`.

### 5.2 Razón de esta decisión

Elegí esta estrategia porque el reto no pedía preservar devoluciones ni reconciliar contabilidad; pedía construir un forecast de demanda útil. En ese contexto:

- un valor negativo no representa demanda real;
- un duplicado no representa ventas reales;
- un outlier extremo puede sesgar severamente el ajuste;
- un missing no debe quedar sin tratar.

Por eso la decisión más segura fue convertir esos casos en valores ausentes controlados y después imputarlos.

---

## 6. EDA realizado y qué aprendimos del dataset

El pipeline formal de EDA quedó en:

- `backend/app/training_pipeline.py`

y produce:

- `backend/data/training/eda_summary.json`
- `backend/data/training/eda_report.md`

### 6.1 Hallazgos globales

- dataset multiserie con `20` SKU
- frecuencia semanal consistente
- `18.35%` de datos inválidos
- peor semana por tasa de invalidez: `2019-10-21`
- peor tasa semanal inválida: `50.0%`

### 6.2 SKU con mayor fricción de limpieza

Los SKU con más problemas de calidad fueron:

- `SKU-016` `Ketchup_397g`: `23.46%` inválido
- `SKU-012` `Shampoo_400ml`: `21.35%` inválido
- `SKU-017` `Galletas_200g`: `20.58%` inválido
- `SKU-014` `Detergente_1kg`: `20.38%` inválido
- `SKU-005` `Harina_1kg`: `20.19%` inválido

### 6.3 Por qué esta EDA fue importante

No se hizo una EDA decorativa. Sirvió para tres decisiones concretas:

1. confirmar que el problema debía resolverse por SKU;
2. confirmar que había rachas de faltantes y no solo huecos aislados;
3. justificar que la imputación tenía que respetar estructura temporal y estacionalidad.

---

## 7. Técnicas de imputación que implementé

Se implementaron tres técnicas y el sistema compara resultados por método:

- `seasonal_median`
- `linear`
- `forward_fill`

Estas tres quedaron expuestas tanto en backend como en frontend.

### 7.1 Interpolación lineal

#### Qué hace

Rellena un valor faltante interpolando entre el valor válido anterior y el siguiente.

#### Cuándo funciona bien

- huecos cortos
- series relativamente suaves
- cambios graduales

#### Ventaja

Es simple y razonable cuando la serie no tiene saltos bruscos.

#### Limitación

Puede suavizar en exceso si el patrón real es estacional o si hay bloques largos perdidos.

### 7.2 Forward fill

#### Qué hace

Propaga hacia adelante el último valor válido observado.

#### Cuándo funciona bien

- procesos relativamente estables
- faltantes cortos
- series donde el último valor es una aproximación razonable del siguiente

#### Ventaja

Es robusto y muy simple.

#### Limitación

Introduce persistencia artificial y puede sesgar si la serie tiene tendencia o estacionalidad fuerte.

### 7.3 Mediana estacional

#### Qué hace

Para cada semana faltante, busca el comportamiento histórico de la misma semana del año y usa la mediana estacional. Luego, si hace falta, completa bordes con interpolación.

#### Por qué la elegí como método base

El dataset es semanal y cubre `10` años. Eso significa que sí existe historia suficiente para aprovechar repetición estacional. Para demanda de productos de consumo, este enfoque es muy razonable porque:

- respeta patrones semanales/anuales;
- es más robusto que la media frente a outliers;
- se comporta mejor que el forward fill en bloques largos;
- aprovecha mejor la estructura temporal del problema.

#### Por qué mediana y no media

Elegí mediana porque el dataset venía contaminado con valores extremos. La media se mueve demasiado por outliers; la mediana es más robusta.

---

## 8. Por qué tomé esas decisiones de imputación

No quise dejar una sola imputación fija “por fe”. Implementé varias porque el reto pedía comparar alternativas y porque, técnicamente, distintas series se benefician de estrategias distintas.

La lógica fue:

- `linear` como baseline suave;
- `forward_fill` como baseline conservador;
- `seasonal_median` como método más alineado con demanda semanal estacional.

Esto permite responder una pregunta importante del reto: no solo “cómo limpiar”, sino “qué efecto tiene el método de limpieza sobre el forecast”.

De hecho, al materializar el entrenamiento en `model_registry.json`, quedó explícito que el mejor método puede variar por SKU. Por ejemplo:

- para `SKU-019`, la mejor receta resultó ser `forward_fill + seasonal_regression`, con `MAPE 3.06%`

Eso confirma que la mejor decisión no siempre es universal; depende de la forma de la serie.

---

## 9. Cómo preparé la data para entrenamiento

Además del dataset limpio, se construyó un dataset train-ready con features explícitas:

- `raw_value`
- `base_clean_value`
- `cleaned_value`
- `is_imputed`
- `was_anomaly`
- `iso_week`
- `month`
- `year`
- `time_index`

Archivos:

- `backend/data/training/demand_history_clean.csv`
- `backend/data/training/demand_training_features.csv`

### Por qué agregué estas columnas

Porque el entrenamiento serio no debe depender solo de la serie final. También conviene conservar trazabilidad:

- qué valor era original;
- cuál fue invalidado;
- cuál fue imputado;
- dónde hubo anomalía;
- en qué posición temporal está cada observación.

Eso facilita:

- auditoría;
- entrenamiento posterior con modelos más complejos;
- análisis de sesgo por imputación;
- experimentación con features temporales y flags de calidad.

---

## 10. Entrenamiento del modelo actual

El backend actual sí entrena modelos. No es una maqueta.

Modelos implementados:

- `seasonal_naive`
- `holt_winters`
- `seasonal_regression`

### 10.1 Cómo se entrena

Para cada SKU:

1. se toma la serie limpia;
2. se separa un `holdout` temporal de `12` semanas;
3. se ajusta cada modelo con la parte de entrenamiento;
4. se predice sobre el holdout;
5. se comparan `MAE` y `MAPE`;
6. se selecciona el mejor modelo;
7. se vuelve a ajustar sobre toda la serie limpia para generar forecast.

### 10.2 Por qué esta estrategia es correcta

El forecast debe validarse temporalmente, no con splits aleatorios. Por eso usé holdout al final de la serie. Esa decisión evita leakage y se parece más al caso real de negocio:

- entrenar con pasado;
- evaluar en futuro cercano;
- proyectar semanas siguientes.

---

## 11. Qué resultados deja el pipeline

El pipeline deja tres niveles de salida:

### 11.1 Salida analítica

- EDA del dataset
- dataset limpio
- dataset train-ready

### 11.2 Salida de entrenamiento

- mejor receta por SKU
- comparación de modelos por imputación
- métricas por serie

Archivo central:

- `backend/data/training/model_registry.json`

### 11.3 Salida para consumo del frontend

- `backend/data/precomputed.json`
- `frontend/data/catalog.json`
- `frontend/data/series/*.json`

Esto permite que GitHub Pages muestre el resultado sin depender del backend en tiempo real.

---

## 12. Por qué no usé Colab como backend principal

Aunque era posible improvisar una API desde Colab, decidí no basar la solución en eso por razones técnicas:

- sesión efímera;
- URL no estable;
- dependencia de mantener el notebook activo;
- mala reproducibilidad para evaluación;
- mayor fragilidad para una demo pública.

Por eso la arquitectura final fue:

- frontend público estático;
- backend local reproducible;
- opción de port forwarding cuando se quiera modo live.

---

## 13. Resumen de decisiones y justificación

### Decisión 1: parsear el PDF con script reproducible

Porque el dataset fuente no estaba listo para modelar y una conversión manual no era defendible.

### Decisión 2: clasificar anomalías usando `nota`

Porque el propio dataset ya traía semántica de calidad del dato y aprovecharla era mejor que intentar redetectar todo desde cero.

### Decisión 3: tratar todas las anomalías como inválidas para forecast

Porque el objetivo era demanda útil, no reconstrucción contable.

### Decisión 4: limpiar por SKU

Porque cada producto tiene patrón temporal distinto.

### Decisión 5: implementar varias imputaciones

Porque no existe una única estrategia óptima para todas las series.

### Decisión 6: usar mediana estacional como método base

Porque el dataset es largo, semanal y contaminado con outliers.

### Decisión 7: validar con holdout temporal

Porque en forecasting no se debe mezclar futuro y pasado aleatoriamente.

### Decisión 8: dejar artefactos de entrenamiento persistidos

Porque una solución seria debe dejar datasets limpios, EDA, métricas y recetas reutilizables.

---

## 14. Limitaciones actuales

La solución actual es sólida para el reto, pero deja abiertas mejoras:

- entrenar modelos GPU más potentes con features avanzadas;
- evaluar estrategias direct multi-horizon;
- incorporar variables exógenas si existieran;
- guardar modelos serializados por SKU;
- automatizar backtesting expandido.

Para eso ya quedó preparado el prompt en:

- `docs/windows-gpu-agent-prompt.md`

---

## 15. Conclusión

El trabajo realizado no se limitó a “hacer una gráfica”. Se construyó una tubería completa de datos y forecast:

- extracción reproducible del PDF;
- clasificación explícita de anomalías;
- limpieza robusta por SKU;
- comparación de técnicas de imputación;
- entrenamiento real de modelos clásicos;
- selección por métricas temporales;
- publicación de resultados en frontend estático;
- preparación de datasets y artefactos para una segunda fase de entrenamiento más fuerte en GPU.

La decisión central fue tratar el problema como un caso real de datos sucios en retail: primero asegurar integridad, luego imputar con criterio temporal, y solo después modelar.

---

## 16. Respuestas a la sección 4 del PDF original

Esta sección responde de forma directa las preguntas del apartado **4. Preguntas del reporte** del documento `3. Reto; Predictor de demanda.pdf`.

### 16.1 ¿Qué tipo de anomalía fue la más frecuente en tu dataset y qué decisión tomaste para tratarla? ¿Por qué ese método y no otro?

La anomalía individual más frecuente fue:

- `missing`: `601` registros

Si se toma la ausencia de dato como una sola familia, entonces:

- `missing + missing_bloque = 1071` registros

Eso muestra que el problema dominante del dataset no fueron los negativos ni los outliers, sino la falta de información.

La decisión que tomé fue:

1. convertir todos esos registros no confiables en `NaN` dentro de `base_clean_value`;
2. imputarlos por SKU;
3. usar `seasonal_median` como método base para la solución visible del reto.

Escogí `seasonal_median` porque la serie es semanal y cubre `10` años, así que sí existe suficiente historia para mirar el comportamiento de la misma semana del año. Ese método fue preferible a una media simple o a un forward fill puro por tres razones:

- conserva mejor la estacionalidad;
- es más robusto frente a outliers;
- se comporta mejor cuando hay bloques consecutivos faltantes.

No elegí media simple porque el dataset venía contaminado con valores extremos. Tampoco dejé solo forward fill porque puede “aplanar” artificialmente la serie cuando el hueco es largo. Por eso la mediana estacional fue la mejor decisión metodológica para el contexto del reto.

### 16.2 ¿Qué modelo elegiste y cómo interpretas el valor de MAPE que obtuviste? ¿Es un resultado aceptable para el contexto de inventario?

La solución no fuerza un único modelo global; compara tres alternativas por SKU:

- `seasonal_naive`
- `holt_winters`
- `seasonal_regression`

Luego selecciona el mejor por `MAPE` usando holdout temporal de `12` semanas.

En la configuración base del dashboard, usando `seasonal_median`, el modelo que más veces resultó ganador fue:

- `seasonal_regression`: `12` de `20` SKU

El `MAPE` promedio del mejor modelo por SKU en esa configuración fue:

- `7.82%`

Yo interpreto ese resultado como bueno para planeación de inventario de corto plazo. Un `MAPE` cercano al `8%` significa, en términos prácticos, que el error porcentual medio está en un rango manejable para decisiones de reposición de `8` a `12` semanas, sobre todo considerando que el punto de partida era un dataset con `18.35%` de registros conflictivos.

¿Es aceptable? Sí, con una precisión importante: es aceptable para apoyo táctico de inventario, no como promesa de exactitud perfecta. Para compras, un error de ese orden suele ser utilizable si se combina con stock de seguridad y lectura del intervalo de confianza. En otras palabras, el forecast sirve para planear, pero no debe interpretarse como una cifra exacta e infalible.

### 16.3 ¿Cómo cambia el intervalo de confianza del forecast al aumentar el horizonte de predicción (de 4 a 12 semanas)? ¿Qué implica esto para el equipo de compras?

Al aumentar el horizonte, el intervalo de confianza se ensancha. Eso era esperable y también quedó cuantificado en los artefactos del proyecto.

Tomando la configuración base del frontend:

- ancho promedio del intervalo al horizonte `4`: `507.96`
- ancho promedio del intervalo al horizonte `12`: `879.79`

Eso representa un aumento aproximado de:

- `73.2%`

La interpretación es directa: mientras más lejos está la semana proyectada, mayor es la incertidumbre. El forecast a `4` semanas sirve para decisiones más firmes de reabastecimiento; el forecast a `12` semanas debe leerse como una banda de planeación más amplia.

Para el equipo de compras esto implica tres cosas:

1. las decisiones cercanas pueden tomarse con mayor confianza;
2. las decisiones lejanas deben manejarse con mayor prudencia y margen operativo;
3. no se debe comprar usando solo el valor central del forecast, sino revisando también el intervalo inferior y superior.

En términos de negocio, el mensaje es que el modelo no solo entrega una cifra, sino también una medida explícita del riesgo asociado a planear con mayor anticipación.
