# Consolidación de la unidad 9 — Machine Learning

## Contexto

La carpeta `ESTADISTICA/9.Machine Learning/` acumuló 7 notebooks heterogéneos de
distinta procedencia (dos cátedras externas, un TP de alumno, material de Kaggle) que
se solapan entre sí y no forman una secuencia dictable. El estado real:

- **Dos no corren**: `cluster-jer-rquico-y-kmeans.ipynb` lee
  `../input/base-censo-comercios/...` y `modelos-xgboost-y-rf.ipynb` lee
  `../input/banco-propension/...`; ninguno de los dos archivos existe en el repo.
- **Dos requieren `xgboost`**, que no está instalado.
- **El TP** (`political identification prediction using ML.ipynb`) depende de
  `google.colab.files.upload()` y de tres Excel preprocesados con dummies
  inconsistentes entre sí.
- **Contenido duplicado**: k-means aparece en 3 notebooks, XGBoost en 3, árboles en 2.
- **Sin cierre aplicado**: el de a priori termina en un scatter de lift y nunca muestra
  para qué serviría; el de k-means termina en la curva de silueta.
- `9.clase.ipynb` (que era solo un placeholder "Proximamente ...") figura como borrado
  sin commitear.
- Ninguno de los 7 notebooks tiene código en R (verificado con grep de `%%R`,
  `library(`, `<-`, `%>%`, `ggplot`, `rpy2`). **El único código en R del corpus es
  `02_modelos.Rmd`** del estudio `SAIMO JOVEN/ej_retail`, que se traduce a Python.
  Sí hay **prosa copiada literal de la documentación de scipy en inglés** en el de
  clustering (los listados `single(y)` / `complete(y)` / `ward(y)` y el bloque
  `metric: str or function`) que hay que reescribir en español.

**Resultado buscado**: 6 notebooks dictables, uno introductorio y uno por familia de
método, con rigor teórico completo, sin redundancia, cada uno cerrando en una aplicación
concreta y con visualizaciones de proceso y de resultado.

## Decisiones tomadas

| Tema | Decisión |
| --- | --- |
| Nombres | descriptivos (`9.0.intro-ml.ipynb`, …); los 7 fuente se archivan en `_fuente/` |
| Orden | siguiendo `MLtypes.png` de izquierda a derecha: supervisado → no supervisado |
| Regularización | notebook nuevo, **primero** del bloque supervisado (ver abajo) |
| Logística | método propio, después de regularización, reutilizando L1/L2 |
| Datos árboles | **dos variantes**: encuesta 134 y mesas CABA 2025; después doy mi opinión |
| Datos clustering | solo QOG |
| Datos regularización | **QOG, no los de retail** (ver abajo) |
| Reglas de asociación | votaciones nominales de diputados, simuladas y guardadas como CSV |
| Paquetes | `python -m pip install xgboost shap` en el Anaconda existente |
| Datos encuesta | a `dataset/encuesta134/`, fuente única `134NODUMMY.xlsx`, dummies en notebook |
| Outputs | ejecutados y guardados, como `4.clase.ipynb` y `7.clase.ipynb` |

### Por qué la regularización va primera en el bloque supervisado

Es la continuación directa de la unidad 5 (regresión lineal múltiple): el mismo modelo
OLS más un término de penalización. Y ubicarla antes de la logística **elimina
redundancia**: L1/L2 se explica una sola vez sobre el caso más simple —target continuo—
y el notebook de logística después solo la reutiliza (`penalty='l1'/'l2'/'elasticnet'`)
en lugar de reintroducirla desde cero.

### Por qué la logística es notebook propio

Comparte con los árboles el ser clasificación supervisada, pero la maquinaria
explicativa es otra: los árboles se enseñan por partición del espacio, impureza, poda y
ensambles; el logit por función de enlace, verosimilitud y coeficientes. Meterlo dentro
del notebook de árboles rompería la columna vertebral de ese notebook. **No pisa a la
unidad 6**: allá el logit se usa para estimar e interpretar efectos, acá para clasificar
casos nuevos — mismo modelo, otra pregunta y otro protocolo de evaluación. El notebook
abre explicitando esa diferencia.

### Por qué QOG y no los datos de retail para la regularización

El Rmd `02_modelos.Rmd` aporta su **arquitectura narrativa y su prosa** —que son
excelentes y se traducen casi literalmente—, pero no sus datos. Razones:

1. El propio Rmd concede que "con 54 observaciones y 7 predictores la regularización
   tiene margen acotado para mejorar la predicción". QOG permite mostrar el caso donde
   el margen es enorme.
2. La colinealidad de los indicadores de gobernanza es un problema **real y conocido**
   de la política comparada, así que la lección aterriza sustantivamente y no solo
   mecánicamente.
3. No agrega datos nuevos al repo ni abre la cuestión de procedencia de un estudio de
   cliente anonimizado.

Números ya verificados sobre `dataset/QOG/qog_bas_cs_jan26.dta`:

**Acto 1** — `ti_cpi` (percepción de corrupción) contra 10 indicadores institucionales y
socioeconómicos, n=152: R²=0.826, R²ajustado=0.814, **solo 4 de 10 coeficientes
significativos al 5%**, VIF máximo 40.9 (`vdem_polyarchy`, que correlaciona 0.978 con
`vdem_libdem`). Es exactamente la firma que describe el Rmd para el caso retail
(R²=0.77, ningún coeficiente significativo, VIF 31).

**Acto 2** — el mismo target contra los 197 indicadores numéricos con ≥60% de cobertura,
n=178, o sea p/n = 1.11:

| Modelo | R² validación cruzada (5-fold) | Variables |
| --- | --- | --- |
| OLS | +0.207 ± 0.800 | 197 |
| Ridge | +0.879 ± 0.020 | 197 |
| Lasso | +0.948 ± 0.021 | **19** |
| Elastic Net | +0.948 ± 0.021 | 19 |

OLS se desintegra —el desvío entre folds es cuatro veces la media— y Lasso se queda con
19 predictores sustantivamente sensatos (`vdem_corr`, `fh_rol`, `wjp_civ_just`,
`wdi_gdpcapcon2015`, …).

**Advertencia de leakage que se enseña explícitamente**: se excluyen las familias
`wbgi_*`, `icrg_*` y el resto de `ti_*` porque el Control of Corruption del Banco
Mundial se construye **usando el CPI como insumo** — `wbgi_cce` correlaciona 0.992 con
el target. Un predictor casi perfecto que es en realidad el mismo dato con otro nombre
es el caso de leakage más difícil de ver y el más frecuente en datos secundarios.

## Estructura final

```
ESTADISTICA/9.Machine Learning/
├── 9.0.intro-ml.ipynb              marco conceptual
├── 9.1.regularizacion.ipynb        superv. · regresión: OLS → Ridge → Lasso → ElasticNet  (QOG)
├── 9.2.logistica.ipynb             superv. · clasificación paramétrica                    (encuesta 134)
├── 9.3.arboles.ipynb               superv. · árbol → RF → XGBoost + comparación final     (encuesta 134)
├── 9.3b.arboles-mesas.ipynb        misma teoría, variante de datos                        (mesas CABA 2025)
├── 9.4.clustering.ipynb            no superv. · jerárquico + k-means                      (QOG)
├── 9.5.reglas-asociacion.ipynb     no superv. · a priori                                  (votaciones sim.)
├── _fuente/                        los 7 notebooks originales + su data/ + 02_modelos.Rmd
└── media/                          MLtypes.png, ml_cheatsheet.jpeg, … + nuevas
```

`9.3` y `9.3b` son la misma secuencia teórica sobre datos distintos: se construyen ambos,
se dictan sobre uno, y al entregar doy mi recomendación de cuál conservar.

`9.1` y `9.4` comparten el dataset QOG pero con conjuntos de variables y preguntas
distintas (predecir corrupción vs. tipología de regímenes); se cruzan referencias entre
ambos donde tocan el mismo problema (colinealidad / reducción de dimensionalidad) en vez
de repetirlo.

## Movimiento de datos

- `ESTADISTICA/9.Machine Learning/data/*.xlsx` → `dataset/encuesta134/` (los 4 archivos).
- `dataset/` está en `.gitignore`, así que solo `134NODUMMY.xlsx` se agrega con
  `git add -f`. `D134.xlsx` y `kmeans134.xlsx` quedan sin trackear porque pasan a ser
  derivables: **los notebooks leen únicamente `134NODUMMY.xlsx`** (la versión etiquetada)
  y construyen las dummies con pandas en el propio notebook. Esto elimina la divergencia
  entre los tres Excel y hace el preprocesamiento visible y auditable.
- Rutas de lectura: `../../dataset/encuesta134/134NODUMMY.xlsx`, siguiendo la convención
  que ya usa `4.clase.ipynb` para QOG.
- Nuevo: `dataset/votaciones_sim/votaciones_hcdn_sim.csv` (generado, `git add -f`).
- Se copia `02_modelos.Rmd` a `_fuente/` como referencia de la traducción.

## Correcciones metodológicas que se aplican (sin señalarlas aparte)

Detectadas al revisar el TP y `modelos-xgboost-y-rf.ipynb`; se incorporan directamente:

1. **Baseline obligatorio.** El TP reporta 40-45% de accuracy como "muy alta en ciencias
   sociales" cuando la clase mayoritaria ya da ~35-40%. Todo notebook supervisado incluye
   `DummyClassifier`/`DummyRegressor` y reporta la ganancia sobre él.
2. **Un split único no es evidencia.** Las comparaciones 30/70 vs 40/60 del TP son ruido.
   Todo se evalúa con `RepeatedStratifiedKFold` y se reporta media ± desvío.
3. **Sin leakage.** Imputación, escalado, dummies y selección de variables van dentro de
   un `Pipeline`/`ColumnTransformer` evaluado en CV. El TP imputaba la moda y corría RFE
   sobre la muestra completa; `modelos-xgboost-y-rf.ipynb` escalaba antes de partir.
4. **Métricas adecuadas a 7 clases desbalanceadas**: macro-F1, balanced accuracy, matriz
   de confusión normalizada y reporte por clase que hace visible que las clases chicas no
   se predicen nunca. Accuracy deja de ser la métrica protagonista.
5. **Fuera el pseudo-Wald.** El TP calcula `(coef / desvío de los coefs entre clases)²`
   con p-valor `1-exp(-w)`, que no es un test de Wald. Se reemplaza por
   `statsmodels.MNLogit` con errores estándar y odds ratios con IC correctos.
6. **Estandarizar antes de clusterizar.** El k-means del TP mezcla dummies 0/1 con
   ordinales −2..2 sin escalar, así que la distancia queda dominada por las de mayor
   rango. Además `iloc[:, :-1]` descartaba una variable real, no la de cluster.
7. **VIF con intercepto**, y el path de regularización como el instrumento correcto para
   colinealidad en contexto predictivo.
8. **Bugs de código**: `GradientBoost` predecía con `adb_clf`; `base_estimator=` ya no
   existe en sklearn 1.5 (es `estimator=`); `sns.distplot` y `groupby().mean()` sobre
   columnas no numéricas están rotos en pandas 2.2; `sample(frac=0.01, replace=True)`
   muestreaba con reposición; `groupby("x")['a','b']` es error en pandas 2.x;
   `multi_class="multinomial"` está deprecado.
9. **Semillas fijas** en todo, y `random_state` explícito en cada estimador.

## Contenido por notebook

Convenciones comunes: encabezados `##` por lámina (como `4.clase.ipynb`), LaTeX con
`$$`, `sns.set_theme(style="whitegrid")` + `plt.rcParams["figure.figsize"]` en la celda de
librerías, imágenes locales con `<figure style="text-align:center; margin:1em 0;">` +
`<img src="media/...">`, imágenes didácticas externas embebidas por URL. Objetivo de
extensión: 60-80 celdas por notebook, ~35% código.

### 9.0 — Introducción teórica

- Predicción vs. explicación: qué separa a ML de las unidades 0-8, y cuándo al analista
  de opinión pública le importa cada cosa.
- `media/MLtypes.png` como esqueleto, explicitando que el diagrama **omite dos ramas del
  no supervisado que sí vemos** (reducción de dimensionalidad y reglas de asociación);
  se lo extiende con un diagrama propio.
- Supervisado (clasificación vs. regresión), no supervisado (clustering, reglas de
  asociación, reducción de dimensionalidad), por refuerzo (agente/entorno/recompensa, y
  por qué no se dicta en esta materia).
- Vocabulario con figura generada para cada concepto: train/validation/test, sobreajuste,
  sesgo-varianza (curva de error train vs. test según complejidad), k-fold (esquema de
  folds), baseline, y el mapa de métricas con la advertencia de desbalanceo.
- Fugas de información con los casos concretos: escalar antes de partir, imputar con la
  muestra completa, seleccionar variables mirando el test, y **el predictor que es el
  target con otro nombre** (el caso `wbgi_cce`/`ti_cpi` que aparece en 9.1).
- Reutilizar `media/ml_cheatsheet.jpeg` y `media/forecasting_models.jpeg` donde encajen.
- **Cierre aplicado**: árbol de decisión de método — ¿tengo etiqueta? ¿es categórica?
  ¿cuántos casos? ¿me importa interpretar o predecir? → qué método y en qué notebook está.
  Diagrama + una función Python que recorre el árbol con las respuestas del usuario.

### 9.1 — Regularización: Ridge, Lasso y Elastic Net (QOG)

Traducción a Python de `02_modelos.Rmd` (`glmnet` → `sklearn.linear_model`, `car::vif` →
`statsmodels`, `ggplot2` → matplotlib/seaborn), con su arquitectura narrativa y datos de
QOG. Target: `ti_cpi`.

- **Acto 1 — el problema.** Matriz de correlaciones en heatmap; identificación de los
  bloques de variables que se mueven juntas; OLS con los 10 predictores y la firma de la
  colinealidad (R²=0.826 con 4/10 coeficientes significativos); VIF con su definición
  $1/(1-R_j^2)$ y su interpretación como factor de inflación de la varianza; diagnóstico
  de residuos (los 4 paneles clásicos, generados con matplotlib); la respuesta manual
  —sacar la variable más colineal— con test F anidado, y la **advertencia sobre eliminar
  variables por p-valor de forma iterativa**, que invalida los p-valores restantes.
- **Acto 2 — la respuesta automática.** La función objetivo penalizada; Ridge (L2) y por
  qué reparte el peso entre predictores correlacionados en vez de concentrarlo; Lasso
  (L1) y por qué el pico del valor absoluto en cero produce ceros exactos, con la
  geometría de la región factible (politopo con vértices vs. esfera) dibujada; Elastic
  Net como combinación con `l1_ratio`; el compromiso sesgo-varianza y por qué se acepta
  un estimador sesgado; **estandarizar no es opcional** cuando la penalización opera
  sobre magnitudes.
- Elección de $\lambda$ por validación cruzada: curva de error con bandas ± 1 desvío,
  `lambda.min` vs. `lambda.1se` (el criterio conservador, que en sklearn hay que
  construir a mano).
- **Trayectoria de coeficientes** vs. $\lambda$ para Ridge y Lasso lado a lado: la
  visualización de proceso central del notebook. En Lasso las líneas tocan cero y se
  quedan; en Ridge se acercan sin llegar.
- Tabla de coeficientes OLS / OLS reducido / Ridge / Lasso / Elastic Net lado a lado, y
  la distinción entre R² de entrenamiento (donde OLS gana por construcción, porque es
  exactamente lo que minimiza) y R² de validación cruzada, que es el número comparable.
- **Acto 3 — p ≈ n.** Los 197 indicadores con ≥60% de cobertura contra n=178: OLS colapsa
  (+0.207 ± 0.800) y Lasso llega a +0.948 con 19 variables. Es donde la regularización
  deja de ser un ajuste fino y pasa a ser la única opción viable. Discusión del leakage
  de las familias `wbgi_*`/`icrg_*`.
- **Cierre aplicado**: análisis de residuos como herramienta sustantiva, calcado del
  "sucursales fuera de la predicción" del Rmd. Gráfico observado vs. predicho con los
  países etiquetados y los 8 de mayor desvío destacados: **qué países son mucho menos (o
  mucho más) corruptos de lo que su estructura institucional y económica predice**. Esos
  son los casos desviados que ameritan estudio en profundidad. Producto: tabla ordenada
  por residuo con el país, el valor observado, el predicho y la brecha.

### 9.2 — Regresión logística como clasificador (encuesta 134)

- Puente explícito con la unidad 6, y con la regularización ya vista en 9.1.
- Teoría: probabilidad → odds → log-odds; función logística (figura); máxima verosimilitud
  vs. MCO; frontera de decisión lineal (figura con 2 features); umbral de decisión y su
  desplazamiento; binaria → multinomial (softmax) y one-vs-rest.
- Regularización L1/L2 **reutilizada**, no reintroducida: se aplica el path de
  coeficientes vs. C aprendido en 9.1, que reemplaza al RFE artesanal del TP.
- Práctica: `ColumnTransformer` (imputación + escalado + dummies) dentro de un `Pipeline`,
  evaluado con `RepeatedStratifiedKFold`. Se conserva `HIJOS_COMBINADA` y la exclusión
  justificada de `NH_EDAD`/`TH_EDAD` por el error documentado de configuración del
  cuestionario.
- Baseline, métricas honestas, matriz de confusión normalizada.
- Interpretación: odds ratios con IC vía `statsmodels.MNLogit`, en heatmap por clase
  (rescata la visualización del TP, ahora bien fundada).
- Re-especificación honesta: agrupar las clases con pocos casos y mostrar cuánto cambia la
  métrica, dejando claro que cambia la pregunta de investigación y no solo el modelo.
- **Cierre aplicado**: scoring de casos nuevos. (a) probabilidad de cada etiqueta para 3
  perfiles de votante construidos a mano; (b) ranking de encuestados persuadibles
  (probabilidad máxima baja / segunda clase cercana) con curva de ganancia acumulada, para
  mostrar cuánto se gana priorizando a quién hablarle.

### 9.3 / 9.3b — Árboles, Random Forest y XGBoost

Teoría común a las dos variantes:

- Partición recursiva (figura: espacio partido al lado del árbol); Gini vs. entropía
  (curvas de impureza); ganancia de información; `max_depth`/`min_samples_leaf`; **poda
  por complejidad-costo** con `cost_complexity_pruning_path` — el hueco de sobreajuste del
  notebook fuente, que entrenaba un árbol sin podar.
- `validation_curve` sobre `max_depth`: la ilustración canónica de sesgo-varianza.
- Inestabilidad: el mismo árbol reajustado en 6 bootstraps da 6 árboles distintos → motiva
  bagging.
- Random Forest: bagging + submuestreo de variables por corte; error out-of-bag;
  importancia por impureza **vs.** por permutación, mostrando que la primera infla las
  variables de alta cardinalidad.
- Boosting: intuición secuencial de residuos (figura de las 3 primeras iteraciones);
  AdaBoost → gradient boosting → XGBoost; `learning_rate`/`max_depth`/`subsample` y
  regularización —conectando con 9.1, porque XGBoost penaliza L1 y L2 igual que Lasso y
  Ridge—; **early stopping** con conjunto de validación y curva de aprendizaje por ronda.
- Un árbol del ensamble dibujado, `PartialDependenceDisplay` de las variables top, y SHAP
  para explicar una predicción individual.
- **Tabla comparativa final** bajo la misma CV: nulo / logit regularizado / árbol podado /
  RF / XGBoost con accuracy, macro-F1, ROC-AUC (OvR) y tiempo. Hereda lo mejor de
  `modelos-xgboost-y-rf.ipynb` (modelo nulo, tabla de métricas, curvas ROC, discusión de
  desbalanceo) pero con CV en lugar de un split único.

Cierres aplicados, distintos por variante:

- **9.3 (encuesta 134)**: comparación honesta contra el baseline + curva de aprendizaje
  vs. tamaño de muestra, para responder cuántos casos habría que relevar para que el
  ensamble sirva. Producto: recomendación de tamaño de muestra para la próxima ola.
- **9.3b (mesas CABA 2025)**: `dataset/elecciones_caba_2025/mesas25.csv` (7.268 mesas),
  predecir el partido ganador desde barrio, comuna, padrón y participación — **sin usar
  columnas de votos como predictores**, que sería leakage directo. Cierre: mesas mal
  clasificadas mapeadas sobre `dataset/barrios_caba/` como territorios de comportamiento
  atípico, más un ranking de circuitos a priorizar.

### 9.4 — Clustering jerárquico y k-means (QOG)

Datos: `../../dataset/QOG/qog_bas_cs_jan26.dta`, 194 países. Subconjunto de indicadores
institucionales distinto del de 9.1, con `cname` como etiqueta legible del dendrograma.

- Distancias: euclídea, Manhattan, Minkowski, correlación; por qué estandarizar, con el
  mismo dataset clusterizado con y sin escalar.
- Jerárquico aglomerativo vs. divisivo; los criterios de linkage
  (single/complete/average/centroid/Ward) **reescritos en español** — hoy son prosa copiada
  de scipy en inglés — conservando las imágenes de `saedsayad.com` que ya usa el fuente, y
  agregando una comparación generada de los 4 dendrogramas sobre los mismos datos.
- Dendrograma de los 194 países con nombres legibles y el corte marcado; `fcluster` /
  `cut_tree`; coeficiente cofenético para elegir linkage.
- k-means: algoritmo de Lloyd paso a paso (4 paneles con la evolución de los centroides),
  `k-means++`, `n_init`, sensibilidad a la inicialización; limitaciones (clusters
  esféricos, outliers, escala) con un contraejemplo generado.
- Elección de k: codo (matplotlib puro, sin `yellowbrick`), silueta **con diagrama por
  cluster** y no solo el promedio, Calinski-Harabasz, y el criterio sustantivo — que en
  ciencia política suele mandar más que la métrica.
- Perfilado: heatmap de medias estandarizadas por cluster, boxplots de las variables
  discriminantes, **biplot de componentes principales** con las flechas de las variables y
  los ejes rotulados con su variable dominante (técnica tomada del Rmd, incluido el
  truco de fijar el signo de cada componente), y validación externa contra `fh_status` y
  `ht_region` con tabla de contingencia y ARI.
- Subclusterización del grupo más grande (la técnica que el TP aplicaba al peronismo).
- MANOVA rescatada del fuente, con la advertencia de que testear sobre los clusters que
  uno mismo construyó es circular.
- **Cierre aplicado**: selección de casos para un diseño comparado. ¿Cuáles son los pares
  reales de Argentina según la estructura de los datos y no según la intuición regional?
  Producto: tabla de los k vecinos más cercanos con su distancia, y qué preguntas de
  investigación habilita cada elección de par.

### 9.5 — Reglas de asociación (votaciones de diputados)

- El problema: dado un conjunto de transacciones, ¿qué ítems co-ocurren más de lo
  esperable por azar? Diferencia con correlación y con clasificación.
- **Métricas con fórmula e interpretación** — el hueco grande del notebook fuente, que
  nunca define nada: soporte, confianza, lift, leverage, conviction. Cada una calculada a
  mano sobre un ejemplo de 3 ítems antes de llamar a la librería, y la explicación de por
  qué el lift es el que separa asociación real de frecuencia base.
- El algoritmo a priori: propiedad de clausura descendente (si un itemset es infrecuente,
  ningún superconjunto puede serlo) con diagrama del retículo podado; costo combinatorio.
- Datos: generador reproducible en el notebook (~257 diputados × ~15 proyectos, ~6 bloques
  con disciplina partidaria + desviaciones, semilla fija) volcado a
  `dataset/votaciones_sim/votaciones_hcdn_sim.csv`. Supuestos del generador documentados.
- Formato transaccional: de la matriz diputado × proyecto al one-hot, con ítems
  `favor_X` / `contra_X` para que las reglas capten oposición y no solo apoyo.
- `mlxtend`: itemsets frecuentes, generación de reglas, filtrado por antecedente y por
  consecuente (rescatado del fuente), ordenamiento por lift. Verificar la firma de
  `association_rules` en mlxtend 0.23.4, que agregó el parámetro `num_itemsets`.
- Visualizaciones: scatter soporte-confianza-lift y two-key plot (del fuente), matriz
  antecedente × consecuente en heatmap, y **grafo de reglas con `networkx`** donde los
  nodos son votaciones y el peso de la arista es el lift — es lo que hace legible el
  resultado.
- **Cierre aplicado, tres productos concretos**: (1) mapa de bloques *de facto* —
  comunidades del grafo de reglas comparadas contra los bloques formales, para ver dónde
  se rompe la disciplina; (2) lista de diputados pivote — los que violan las reglas de
  alta confianza de su propio bloque, o sea a quién hay que negociar; (3) predicción de
  voto para un proyecto nuevo a partir del patrón parcial de un diputado, evaluada contra
  la regla ingenua "vota como su bloque".

## Archivos a tocar

**Nuevos**: los 7 notebooks listados arriba; `_fuente/` con los 7 originales, su `data/` y
`02_modelos.Rmd`; `dataset/encuesta134/`; `dataset/votaciones_sim/votaciones_hcdn_sim.csv`;
imágenes nuevas en `media/`.

**Modificados**: [README.md](README.md) — fila de la unidad 9 en la tabla del programa
(hoy dice solo "Aprendizaje supervisado, entrenamiento y validación, clasificación"),
árbol de directorios, y tabla de datasets con `encuesta134/` y `votaciones_sim/` y la
unidad EST 9 agregada a las filas de `QOG/` y `elecciones_caba_2025/`.

**Eliminados**: `ESTADISTICA/9.Machine Learning/data/` (movida), y se confirma el borrado
ya pendiente de `9.clase.ipynb` (era un placeholder).

## Verificación

1. `python -m pip install xgboost shap` en el Anaconda de
   `C:/Users/mmill/anaconda3`, y confirmar `import xgboost, shap` desde ese intérprete.
2. Ejecutar cada notebook end-to-end con
   `jupyter nbconvert --to notebook --execute --inplace` y confirmar 0 errores y outputs
   guardados. Los notebooks del repo guardan outputs, así que esto además valida el
   formato de entrega.
3. Grep de rutas rotas: ningún notebook final debe contener `../input/`, `data/`,
   `files.upload` ni `read_excel('D134` sin prefijo `../../dataset/`.
4. Grep de R (`%%R`, `library(`, `<-`, `%>%`, `ggplot`, `rpy2`) en los notebooks finales
   → 0 coincidencias.
5. Checklist manual por notebook: ≥1 visualización de proceso, ≥1 de resultado/aplicación,
   cierre aplicado que produce un objeto concreto (lista, ranking, tabla, mapa) y no solo
   una métrica.
6. Confirmar que `media/MLtypes.png` se referencia y renderiza en `9.0`, y que las
   imágenes por URL cargan.
7. Confirmar ausencia de redundancia: que ningún método se explique teóricamente en dos
   notebooks (L1/L2 solo en 9.1; los específicos solo retoman brevemente el marco de 9.0).
8. Reproducir los números del acto 2 de 9.1 en el notebook ejecutado y confirmar que
   coinciden con los verificados (OLS +0.207 ± 0.800, Lasso +0.948 ± 0.021, 19 variables).
9. `git status` + `git add -f` de `dataset/encuesta134/134NODUMMY.xlsx` y
   `dataset/votaciones_sim/votaciones_hcdn_sim.csv`, y verificar que quedaron trackeados.
10. Al terminar, comparar `9.3` vs `9.3b` y entregar mi recomendación de cuál conservar,
    con el argumento didáctico.
