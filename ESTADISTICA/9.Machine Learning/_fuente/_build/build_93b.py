# -*- coding: utf-8 -*-
"""Construye 9.3b.arboles-mesas.ipynb  (variante mesas CABA 2025)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nbb import md, code, write_nb, requisitos, REPO

OUT = os.path.join(REPO, "ESTADISTICA", "9.Machine Learning", "9.3b.arboles-mesas.ipynb")

C = []
A = C.append

A(md(r"""
# Árboles, Random Forest y XGBoost sobre datos electorales

**Unidad 9 · Machine Learning · Notebook 3b de 6 · variante de datos**

Este notebook recorre los mismos métodos que `9.3` —árbol de decisión, Random Forest,
boosting— sobre un problema completamente distinto: **7.233 mesas de la elección legislativa
porteña de 2025**.

Se puede leer solo. Pero leído junto a `9.3` responde una pregunta que ningún notebook
individual puede responder: **¿de qué depende que convenga usar un ensamble?**
"""))

A(md(r"""
## Por qué otra variante

En `9.3` los ensambles **no** le ganaron al modelo lineal. Random Forest empató con la
regresión logística (0.788 contra 0.791) y XGBoost quedó por detrás. El diagnóstico fue el
tamaño de la muestra: con 132 casos no hay datos para sostener un modelo flexible.

Esa conclusión es correcta pero incompleta, porque deja sin responder si los ensambles sirven
para algo. Acá cambiamos una sola cosa —pasamos de 132 casos a 7.233— manteniendo el tipo de
problema (clasificación política) y los métodos. Vamos a ver:

- la progresión **árbol → bosque → boosting**, con una sorpresa en el medio: Random Forest con
  los valores por defecto queda **por debajo** de un árbol único bien calibrado;
- la **trampa de las variables de alta cardinalidad**, con un caso real de 1.122 niveles;
- cómo se chequea si las observaciones son **independientes**, y qué significa que el chequeo
  dé un resultado inesperado;
- y un cierre aplicado que se puede **mapear**.

La teoría de los métodos está desarrollada en `9.3`: impureza, ganancia de información,
bagging, decorrelación, gradiente. Acá la usamos y la mostramos funcionando, sin volver a
derivarla.
"""))

A(md("## Requisitos"))

A(requisitos(extras=["xgboost", "geopandas"]))

A(md("## Librerías"))

A(code(r"""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import (StratifiedGroupKFold, StratifiedKFold,
                                     cross_val_predict, cross_validate, learning_curve,
                                     train_test_split, validation_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from xgboost import XGBClassifier

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)
pd.set_option("display.width", 220)

SEED = 42
rng = np.random.default_rng(SEED)
PALETA = ["#0073C2", "#EFC000", "#CD534C", "#868686", "#1E2749"]
"""))

# ------------------------------------------------------------------ datos
A(md(r"""
## Los datos: mesa por mesa

`mesas25.csv` tiene el resultado de la elección de legisladores de la Ciudad de Buenos Aires
de 2025, **desagregado por mesa de votación**. Una fila por mesa, con los votos de cada una de
las 17 listas más los votos en blanco, nulos, recurridos y el padrón.

Es el nivel de desagregación más fino disponible públicamente en datos electorales argentinos,
y por eso es tan útil: 7.268 unidades de análisis con comportamiento electoral medido sin
error de muestreo.
"""))

A(code(r"""
mesas = pd.read_csv("../../dataset/elecciones_caba_2025/mesas25.csv")
print(f"mesas: {mesas.shape[0]}   columnas: {mesas.shape[1]}")

NO_PARTIDOS = ["circuito", "comuna_x", "mesa", "escuela", "barrio", "comuna_y",
               "electores_totales", "cant_votantes", "voto_en_blanco",
               "votos_de_identidad_impugnada", "votos_nulos", "votos_recurridos",
               "no_voto", "comuna"]
PARTIDOS = [c for c in mesas.columns if c not in NO_PARTIDOS]

total = mesas[PARTIDOS].sum().sort_values(ascending=False)
pd.DataFrame({"votos": total, "%": (total / total.sum() * 100).round(2)}).head(8)
"""))

A(md(r"""
### La variable a explicar

Definimos la etiqueta como **qué lista ganó en cada mesa**. Es una decisión de diseño: podría
ser el porcentaje de una lista (regresión), la participación, el voto en blanco. Elegimos el
ganador porque es la pregunta con la que trabaja un analista de campaña.
"""))

A(code(r"""
mesas["ganador"] = mesas[PARTIDOS].idxmax(axis=1)
conteo = mesas["ganador"].value_counts()
pd.DataFrame({"mesas ganadas": conteo,
              "%": (conteo / len(mesas) * 100).round(2)})
"""))

A(md(r"""
La elección fue una **carrera de dos**: *La Libertad Avanza* ganó en 3.636 mesas y *Es Ahora
Buenos Aires* en 3.597. Un tercer espacio, *Buenos Aires Primero*, ganó en 35.

Nos quedamos con las dos primeras y descartamos esas 35 mesas. Con eso el problema queda
**binario y casi perfectamente balanceado**, lo que tiene una ventaja pedagógica que conviene
señalar: **el baseline es 50.3%**, así que por una vez la exactitud es una métrica honesta y
todo lo que supere 0.50 es señal real. En `9.2` y `9.3`, con clases desbalanceadas, había que
mirar el $F_1$ macro para no engañarse.
"""))

A(code(r"""
DOS = ["la_libertad_avanza", "es_ahora_buenos_aires"]
datos = mesas[mesas["ganador"].isin(DOS)].copy().reset_index(drop=True)

datos["participacion"] = datos["cant_votantes"] / datos["electores_totales"]
datos["margen"] = ((datos["la_libertad_avanza"] - datos["es_ahora_buenos_aires"])
                   / datos["cant_votantes"] * 100)

print(f"mesas conservadas: {len(datos)} de {len(mesas)}  "
      f"(se descartan {len(mesas) - len(datos)} donde ganó un tercer espacio)")
print(f"\nbaseline (clase mayoritaria): "
      f"{datos['ganador'].value_counts(normalize=True).iloc[0]:.4f}")
print(f"\nparticipación: media {datos['participacion'].mean():.3f}, "
      f"rango [{datos['participacion'].min():.3f}, {datos['participacion'].max():.3f}]")
print(f"padrón por mesa: media {datos['electores_totales'].mean():.0f}, "
      f"rango [{datos['electores_totales'].min()}, {datos['electores_totales'].max()}]")
"""))

A(md(r"""
### Reconstruir los nombres de los barrios

La columna `barrio` es un **código numérico** de 1 a 48. Para poder leer los resultados hay
que recuperar los nombres, que están en el GeoJSON de barrios de la Ciudad.

No hay tabla de correspondencia, así que hay que inferir el criterio de codificación. La pista
está en que los códigos bajos corresponden todos a la comuna 1: el orden no es alfabético
global sino **por comuna y después alfabético dentro de cada comuna**.
"""))

A(code(r"""
with open("../../dataset/barrios_caba/caba_barrios.geojson", encoding="utf-8") as f:
    geojson_barrios = json.load(f)

geo = pd.DataFrame([{"nombre_barrio": f["properties"]["BARRIO"],
                     "comuna_geo": int(f["properties"]["COMUNA"])}
                    for f in geojson_barrios["features"]])

# hipótesis: los códigos se asignan ordenando por (comuna, nombre)
geo = geo.sort_values(["comuna_geo", "nombre_barrio"]).reset_index(drop=True)
geo["barrio"] = np.arange(1, len(geo) + 1)

# validación: la comuna del GeoJSON debe coincidir con la de las mesas para cada código
comuna_por_codigo = datos.groupby("barrio")["comuna_y"].agg(lambda s: s.mode().iloc[0])
control = geo.set_index("barrio").join(comuna_por_codigo.rename("comuna_mesas"))
coinciden = (control["comuna_geo"] == control["comuna_mesas"]).sum()

print(f"códigos cuya comuna coincide entre el GeoJSON y las mesas: {coinciden} de {len(control)}")
if coinciden == len(control):
    print("La hipótesis de codificación se valida en los 48 barrios.")
    datos = datos.merge(geo[["barrio", "nombre_barrio"]], on="barrio", how="left")
control.head(10)
"""))

A(md(r"""
La validación es completa: los 48 códigos asignan la comuna correcta. Es un buen ejemplo de
una tarea cotidiana y poco glamorosa del trabajo con datos secundarios —reconstruir una
codificación no documentada— y de cómo **verificarla contra una variable independiente** en
lugar de asumir que la hipótesis es correcta.
"""))

A(code(r"""
resumen_barrio = (datos.groupby("nombre_barrio")
                  .agg(mesas=("mesa", "size"),
                       participacion=("participacion", "mean"),
                       margen_lla=("margen", "mean"),
                       gana_lla=("ganador", lambda s: (s == "la_libertad_avanza").mean()))
                  .sort_values("margen_lla"))

fig, ax = plt.subplots(figsize=(9, 11))
colores = [PALETA[2] if v < 0 else PALETA[0] for v in resumen_barrio["margen_lla"]]
ax.barh(resumen_barrio.index, resumen_barrio["margen_lla"], color=colores)
ax.axvline(0, color="black", lw=1.2)
ax.set_xlabel("Margen promedio LLA − Es Ahora BA (puntos porcentuales)")
ax.set_title("El voto tiene una estructura territorial fuerte\n"
             "azul: gana LLA · rojo: gana Es Ahora Buenos Aires", fontsize=11.5)
ax.tick_params(axis="y", labelsize=8)
plt.tight_layout()
plt.show()
"""))

A(md(r"""
El gradiente es muy marcado: de unos $-20$ puntos en La Boca y Villa Soldati a más de $+20$ en
Puerto Madero y Palermo. Es el eje norte-sur que estructura históricamente el voto porteño, y
es la razón por la que el barrio va a ser el predictor dominante.
"""))

# ------------------------------------------------------------------ leakage
A(md(r"""
## Qué no puede entrar al modelo

Antes de elegir predictores, la pregunta de `9.0`: **¿qué información está disponible en el
momento de predecir?**

El dataset tiene 17 columnas con los votos de cada lista. Usarlas para predecir quién ganó
sería absurdo: **el ganador se calcula a partir de ellas**. Un modelo con esos predictores
tendría exactitud 1.0 y valor cero. Es el caso más obvio de fuga de información, pero el
mecanismo es idéntico al de `wbgi_cce` en `9.1`, donde no era obvio para nada.

Quedan cuatro predictores legítimos, todos conocidos **antes** de contar los votos:

| Predictor | Tipo | Qué capta |
|---|---|---|
| `nombre_barrio` | categórica, 48 niveles | ubicación: composición social del territorio |
| `comuna_y` | categórica, 15 niveles | agregación mayor del barrio |
| `electores_totales` | cuantitativa | tamaño del padrón de la mesa |
| `participacion` | cuantitativa | proporción del padrón que votó |

Notar que `participacion` **sí** es legítima aunque se conozca después de la votación: es un
dato del acto electoral, no del recuento por lista, y en un uso real de proyección se conoce
antes de terminar el escrutinio. Que sea legítima o no depende de para qué se use el modelo, y
eso hay que decidirlo explícitamente.
"""))

A(code(r"""
NUMERICAS = ["electores_totales", "participacion"]
CATEGORICAS = ["nombre_barrio", "comuna_y"]

X = datos[NUMERICAS + CATEGORICAS].copy()
X[NUMERICAS] = X[NUMERICAS].astype(float)
y = datos["ganador"]
y_bin = (y == "la_libertad_avanza").astype(int)

preprocesador = ColumnTransformer([
    ("num", StandardScaler(), NUMERICAS),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAS),
])

print(f"matriz de diseño: {preprocesador.fit_transform(X).shape}")
print(f"n = {len(X)}   ->  p/n = {preprocesador.fit_transform(X).shape[1] / len(X):.3f}")
print("\nRelación p/n opuesta a la de 9.1 y 9.3: acá sobran observaciones.")
"""))

A(code(r"""
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
METRICAS = ["accuracy", "f1_macro", "roc_auc"]

def evaluar(estimador, X, y, nombre, cv=CV, **kw):
    r = cross_validate(estimador, X, y, cv=cv, scoring=METRICAS, n_jobs=-1, **kw)
    return {"modelo": nombre,
            "exactitud": r["test_accuracy"].mean(),
            "exac_sd": r["test_accuracy"].std(),
            "f1_macro": r["test_f1_macro"].mean(),
            "roc_auc": r["test_roc_auc"].mean()}

def con_pre(modelo):
    return Pipeline([("pre", preprocesador), ("clf", modelo)])
"""))

# ------------------------------------------------------------------ árbol
A(md(r"""
## Un árbol, y esta vez se lee bien

Con nombres de barrio en lugar de códigos, un árbol de profundidad 3 produce reglas
directamente legibles.
"""))

A(code(r"""
arbol3 = con_pre(DecisionTreeClassifier(max_depth=3, random_state=SEED))
arbol3.fit(X, y_bin)

nombres = [n.split("__", 1)[1] for n in arbol3[:-1].get_feature_names_out()]

fig, ax = plt.subplots(figsize=(19, 8))
plot_tree(arbol3[-1], feature_names=nombres,
          class_names=["Es Ahora BA", "LLA"], filled=True, rounded=True,
          fontsize=8, ax=ax)
ax.set_title("Árbol de profundidad 3 · ¿qué lista gana la mesa?", fontsize=13)
plt.tight_layout()
plt.show()
"""))

A(code(r"""
print(export_text(arbol3[-1], feature_names=nombres,
                  class_names=["Es Ahora BA", "LLA"], max_depth=3,
                  spacing=2, decimals=2))
"""))

A(md(r"""
Las reglas dicen algo sustantivo. El árbol usa la **participación** como primer corte —donde
votó más gente gana LLA— y después discrimina por barrio. Esa es una regularidad conocida del
comportamiento electoral porteño: la participación es más alta en los barrios de mayor nivel
socioeconómico, y esos barrios votan distinto.

El árbol encontró esa interacción territorio–participación sin que nadie la declarara. En un
modelo lineal habría que especificarla a mano.
"""))

A(code(r"""
# La frontera de decisión sobre las dos variables cuantitativas
sub = datos.sample(2200, random_state=SEED)
arbol_2d = DecisionTreeClassifier(max_depth=4, random_state=SEED).fit(
    datos[["participacion", "electores_totales"]], y_bin)

g1, g2 = np.meshgrid(np.linspace(datos["participacion"].min(), datos["participacion"].max(), 400),
                     np.linspace(datos["electores_totales"].min(),
                                 datos["electores_totales"].max(), 400))
Z = arbol_2d.predict(np.c_[g1.ravel(), g2.ravel()]).reshape(g1.shape)

fig, ax = plt.subplots(figsize=(9.5, 6))
ax.contourf(g1, g2, Z, alpha=0.18, colors=[PALETA[2], PALETA[0]], levels=[-0.5, 0.5, 1.5])
ax.contour(g1, g2, Z, levels=[0.5], colors="black", linewidths=1.6)
for etiqueta, color, nombre in [(0, PALETA[2], "Es Ahora BA"), (1, PALETA[0], "LLA")]:
    m = (sub["ganador"] == "la_libertad_avanza").astype(int) == etiqueta
    ax.scatter(sub.loc[m, "participacion"], sub.loc[m, "electores_totales"],
               s=10, color=color, alpha=0.45, label=nombre)
ax.set_xlabel("Participación en la mesa")
ax.set_ylabel("Padrón de la mesa")
ax.set_title("Frontera escalonada de un árbol sobre las dos variables cuantitativas\n"
             "(muestra de 2.200 mesas para que el gráfico se lea)", fontsize=11)
ax.legend(fontsize=9, markerscale=2)
plt.tight_layout()
plt.show()
"""))

A(md(r"""
Con 7.233 puntos se ve bien la naturaleza escalonada de la frontera: bloques rectangulares,
sin diagonales. La participación hace casi todo el trabajo; el tamaño del padrón, poco.
"""))

A(md(r"""
## Poda: ahora la curva de validación tiene un máximo claro

En `9.3` la curva de validación era plana y ruidosa porque con 132 casos no había suficiente
información para distinguir profundidades. Con 7.233 el panorama cambia.
"""))

A(code(r"""
profundidades = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, None]
etiquetas_prof = [str(p) if p is not None else "sin límite" for p in profundidades]

tr, te = validation_curve(
    con_pre(DecisionTreeClassifier(random_state=SEED)), X, y_bin,
    param_name="clf__max_depth", param_range=profundidades,
    cv=CV, scoring="accuracy", n_jobs=-1)

posiciones = np.arange(len(profundidades))
fig, ax = plt.subplots(figsize=(10, 5.2))
ax.plot(posiciones, tr.mean(axis=1), "o-", lw=2.2, color=PALETA[0], label="entrenamiento")
ax.fill_between(posiciones, tr.mean(axis=1) - tr.std(axis=1),
                tr.mean(axis=1) + tr.std(axis=1), alpha=0.15, color=PALETA[0])
ax.plot(posiciones, te.mean(axis=1), "s-", lw=2.2, color=PALETA[2], label="validación cruzada")
ax.fill_between(posiciones, te.mean(axis=1) - te.std(axis=1),
                te.mean(axis=1) + te.std(axis=1), alpha=0.15, color=PALETA[2])
mejor_i = int(np.argmax(te.mean(axis=1)))
ax.axvline(mejor_i, color=PALETA[1], ls="--", lw=1.8,
           label=f"óptimo: profundidad {etiquetas_prof[mejor_i]}")
ax.axhline(0.5027, color=PALETA[3], ls=":", lw=1.8, label="baseline (0.503)")
ax.set_xticks(posiciones); ax.set_xticklabels(etiquetas_prof)
ax.set_xlabel("Profundidad máxima")
ax.set_ylabel("Exactitud")
ax.set_title("Con 7.233 casos el compromiso sesgo-varianza se ve con nitidez", fontsize=11.5)
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()

print(f"profundidad óptima: {etiquetas_prof[mejor_i]}   "
      f"exactitud = {te.mean(axis=1)[mejor_i]:.4f}")
print(f"árbol sin límite  : entrenamiento = {tr.mean(axis=1)[-1]:.4f}   "
      f"validación = {te.mean(axis=1)[-1]:.4f}")
"""))

A(md(r"""
Esta es la curva del libro de texto. El entrenamiento sube monótonamente hacia 1.0; la
validación sube, alcanza un máximo y baja. La brecha entre las dos curvas **es** la varianza
del modelo, y se abre exactamente donde empieza el sobreajuste.

Comparar con la figura equivalente de `9.3` es instructivo: el mismo gráfico, el mismo código,
y con 132 casos no se distingue nada.
"""))

# ------------------------------------------------------------------ RF
A(md(r"""
## Random Forest
"""))

A(code(r"""
rf = con_pre(RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1))

# error out-of-bag, gratis
rf_oob = RandomForestClassifier(n_estimators=300, oob_score=True,
                                random_state=SEED, n_jobs=-1)
rf_oob.fit(preprocesador.fit_transform(X), y_bin)
print(f"exactitud out-of-bag        : {rf_oob.oob_score_:.4f}")

s = cross_validate(rf, X, y_bin, cv=CV, scoring=["accuracy"], n_jobs=-1)
print(f"exactitud en validación cruz.: {s['test_accuracy'].mean():.4f} "
      f"± {s['test_accuracy'].std():.4f}")
"""))

A(code(r"""
# Importancia: impureza vs permutación
X_tr, X_te, y_tr, y_te = train_test_split(X, y_bin, test_size=0.3,
                                          random_state=SEED, stratify=y_bin)
rf.fit(X_tr, y_tr)

cols = [n.split("__", 1)[1] for n in rf[:-1].get_feature_names_out()]
imp_gini = pd.Series(rf[-1].feature_importances_, index=cols)

# agrupamos las dummies de una misma variable original
def agrupar(serie):
    grupos = {}
    for k, v in serie.items():
        if k in NUMERICAS:
            base = k
        elif k.startswith("comuna_y"):
            base = "comuna (15 niveles)"
        else:
            base = "barrio (48 niveles)"
        grupos[base] = grupos.get(base, 0) + v
    return pd.Series(grupos).sort_values(ascending=False)

perm = permutation_importance(rf, X_te, y_te, n_repeats=15, random_state=SEED,
                              scoring="accuracy", n_jobs=-1)
imp_perm = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4))
g = agrupar(imp_gini).sort_values()
axes[0].barh(g.index, g.values, color=PALETA[3])
axes[0].set_xlabel("Reducción de impureza (dummies agrupadas)")
axes[0].set_title("Importancia por impureza", fontsize=11)

pp = imp_perm.sort_values()
axes[1].barh(pp.index, pp.values, xerr=pd.Series(perm.importances_std,
                                                 index=X.columns)[pp.index],
             capsize=3, color=PALETA[0])
axes[1].set_xlabel("Caída de exactitud al permutar")
axes[1].set_title("Importancia por permutación", fontsize=11)
plt.tight_layout()
plt.show()
"""))

A(md(r"""
Los dos métodos coinciden en el orden: **participación primero, barrio segundo**, y el padrón
casi no aporta. Cuando impureza y permutación coinciden, se puede confiar; el problema surge
cuando difieren, y para eso hay que provocarlo.
"""))

# ------------------------------------------------------------------ cardinalidad
A(md(r"""
## La trampa de la alta cardinalidad, con un caso real

El dataset trae una variable que en `9.3` tuvimos que fabricar con ruido artificial:
**`escuela`**, el establecimiento donde funciona la mesa. Tiene **1.122 niveles** distintos
para 7.233 observaciones, o sea unas 6 mesas por escuela.

Es exactamente el perfil de variable que la importancia por impureza sobrevalora. Y a la vez
es plausible que aporte información real: la escuela es una unidad geográfica más fina que el
barrio.

Las dos cosas pueden ser verdad a la vez, y hay que distinguirlas.
"""))

A(code(r"""
print(f"escuelas distintas: {datos['escuela'].nunique()}")
print(f"mesas por escuela : media {datos.groupby('escuela').size().mean():.1f}, "
      f"máximo {datos.groupby('escuela').size().max()}")

CAT_ESC = CATEGORICAS + ["escuela"]
X_esc = datos[NUMERICAS + CAT_ESC].copy()
X_esc[NUMERICAS] = X_esc[NUMERICAS].astype(float)

pre_esc = ColumnTransformer([
    ("num", StandardScaler(), NUMERICAS),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_ESC),
])
print(f"\nmatriz de diseño con escuela: {pre_esc.fit_transform(X_esc).shape}")
"""))

A(code(r"""
rf_esc = Pipeline([("pre", pre_esc),
                   ("clf", RandomForestClassifier(n_estimators=300, random_state=SEED,
                                                  n_jobs=-1))])
Xe_tr, Xe_te, ye_tr, ye_te = train_test_split(X_esc, y_bin, test_size=0.3,
                                              random_state=SEED, stratify=y_bin)
rf_esc.fit(Xe_tr, ye_tr)

cols_esc = [n.split("__", 1)[1] for n in rf_esc[:-1].get_feature_names_out()]
imp_esc = pd.Series(rf_esc[-1].feature_importances_, index=cols_esc)

def agrupar_esc(serie):
    grupos = {}
    for k, v in serie.items():
        if k in NUMERICAS:
            base = k
        elif k.startswith("comuna_y"):
            base = "comuna (15 niveles)"
        elif k.startswith("escuela"):
            base = "escuela (1122 niveles)"
        else:
            base = "barrio (48 niveles)"
        grupos[base] = grupos.get(base, 0) + v
    return pd.Series(grupos).sort_values(ascending=False)

perm_esc = permutation_importance(rf_esc, Xe_te, ye_te, n_repeats=15,
                                  random_state=SEED, scoring="accuracy", n_jobs=-1)
imp_perm_esc = pd.Series(perm_esc.importances_mean, index=X_esc.columns)

comparacion_imp = pd.DataFrame({
    "por impureza": agrupar_esc(imp_esc),
    "por permutación": imp_perm_esc.rename(index={
        "nombre_barrio": "barrio (48 niveles)",
        "comuna_y": "comuna (15 niveles)",
        "escuela": "escuela (1122 niveles)"}),
}).round(4)
comparacion_imp
"""))

A(code(r"""
fig, ax = plt.subplots(figsize=(10, 4.6))
t = comparacion_imp.copy()
t = t / t.sum()                       # a proporciones, para poder compararlas
t = t.sort_values("por impureza")
pos = np.arange(len(t))
ancho = 0.38
ax.barh(pos - ancho/2, t["por impureza"], ancho, color=PALETA[3], label="por impureza")
ax.barh(pos + ancho/2, t["por permutación"], ancho, color=PALETA[0],
        label="por permutación")
ax.set_yticks(pos); ax.set_yticklabels(t.index)
ax.set_xlabel("Importancia relativa (cada método normalizado a 1)")
ax.set_title("La escuela pesa mucho más por impureza que por permutación", fontsize=11.5)
ax.legend(fontsize=9.5)
plt.tight_layout()
plt.show()
"""))

A(md(r"""
La diferencia es la esperada: la importancia por impureza le asigna a `escuela` una porción
del total mucho mayor que la permutación. El mecanismo es el de `9.3`: 1.122 niveles ofrecen
muchísimos cortes candidatos, y algunos reducen la impureza por azar.

Pero la permutación **no** la manda exactamente a cero. ¿Entonces aporta algo, o no? Para
responder eso hace falta medir la exactitud con y sin ella — y antes de eso, revisar un supuesto
que la escuela pone en cuestión.
"""))

A(md(r"""
## ¿Son independientes las observaciones?

Antes de responder cuánto aporta la escuela hay que revisar un supuesto que la validación
cruzada da por sentado: que las observaciones son **independientes**.

Y no lo son de manera obvia: **seis mesas de la misma escuela comparten el mismo barrio, el
mismo edificio y aproximadamente el mismo electorado**. Si algunas mesas de una escuela caen
en entrenamiento y otras en validación, el modelo puede estar "reconociendo" la escuela en
lugar de generalizando — el cuarto tipo de fuga de información de `9.0`.

Se mide primero, se corrige después.
"""))

A(code(r"""
# ¿Qué tan homogéneas son las mesas de una misma escuela?
por_escuela = datos.groupby("escuela")["ganador"].agg(
    mesas="size", prop_lla=lambda s: (s == "la_libertad_avanza").mean())
con_varias = por_escuela[por_escuela["mesas"] >= 3]

unanimes = ((con_varias["prop_lla"] == 0) | (con_varias["prop_lla"] == 1)).mean()
dominante = np.maximum(con_varias["prop_lla"], 1 - con_varias["prop_lla"]).mean()

print(f"escuelas con al menos 3 mesas: {len(con_varias)}")
print(f"  unánimes (todas sus mesas con la misma ganadora): {unanimes:.1%}")
print(f"  proporción media de la lista dominante dentro de la escuela: {dominante:.3f}")

fig, ax = plt.subplots(figsize=(8.5, 4.4))
ax.hist(con_varias["prop_lla"], bins=21, color=PALETA[0], alpha=0.85, edgecolor="white")
ax.set_xlabel("Proporción de mesas de la escuela ganadas por LLA")
ax.set_ylabel("Cantidad de escuelas")
ax.set_title("Si las mesas de una escuela fueran idénticas, todo estaría en 0 y en 1",
             fontsize=11)
plt.tight_layout()
plt.show()
"""))

A(md(r"""
Las mesas de una misma escuela **se parecen pero no son iguales**: la lista dominante se lleva
en promedio el 83% de las mesas de su escuela, y solo el 40% de las escuelas son unánimes. El
histograma tiene masa en los extremos pero también en el medio.

Eso significa que hay dependencia, pero que la etiqueta **no es determinística dentro de la
escuela**. Para saber si eso produce fuga hay que comparar dos esquemas de validación:
"""))

A(code(r"""
# StratifiedGroupKFold: los mismos pliegues estratificados, pero sin partir escuelas
CV_GRUPO = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

comparacion_cv = []
for nombre_X, XX, pre in [("sin escuela", X, preprocesador),
                          ("con escuela", X_esc, pre_esc)]:
    modelo = Pipeline([("pre", pre),
                       ("clf", RandomForestClassifier(n_estimators=300,
                                                      random_state=SEED, n_jobs=-1))])
    s1 = cross_validate(modelo, XX, y_bin, cv=CV, scoring=["accuracy"], n_jobs=-1)
    s2 = cross_validate(modelo, XX, y_bin, cv=CV_GRUPO, scoring=["accuracy"],
                        groups=datos["escuela"], n_jobs=-1)
    comparacion_cv.append({
        "predictores": nombre_X,
        "StratifiedKFold": s1["test_accuracy"].mean(),
        "sd_normal": s1["test_accuracy"].std(),
        "StratifiedGroupKFold": s2["test_accuracy"].mean(),
        "sd_grupo": s2["test_accuracy"].std(),
        "diferencia": s1["test_accuracy"].mean() - s2["test_accuracy"].mean(),
    })

pd.DataFrame(comparacion_cv).round(4)
"""))

A(md(r"""
### El resultado, que no es el que uno esperaría

La fuga por agrupamiento tiene una firma reconocible: el esquema que **parte** los grupos
debería dar una exactitud **más alta** que el que los respeta, porque el primero deja que el
modelo reconozca grupos ya vistos. O sea, esperaríamos `StratifiedKFold` > `StratifiedGroupKFold`.

**Se observa exactamente lo contrario.** Con `escuela` entre los predictores, el esquema que
respeta los grupos da 0.764 y el que los parte da 0.729: el que "debería" estar inflado es el
más bajo. Sin `escuela` pasa lo mismo, con una diferencia menor.

Entonces la conclusión firme es: **no hay evidencia de que nuestras estimaciones estén
infladas por fuga por agrupamiento.** Eso es lo que había que verificar, y quedó verificado.

Sobre por qué la diferencia va en la otra dirección conviene ser prudente, porque los dos
esquemas **no responden la misma pregunta** y por eso no son estrictamente comparables:

- `StratifiedKFold` estima *qué tan bien predigo una mesa nueva de una escuela que conozco*.
- `StratifiedGroupKFold` estima *qué tan bien predigo en una escuela entera que nunca vi*.

En el segundo escenario las 1.122 columnas de `escuela` valen cero en el pliegue de validación
—son categorías no vistas— así que el modelo se apoya en barrio y participación. Que eso rinda
**mejor** sugiere que los efectos por escuela estimados sobre 4 o 5 mesas son sobre todo ruido:
memorizarlos no ayuda a predecir mesas nuevas ni siquiera en la misma escuela.

Eso responde la pregunta que quedó abierta sobre la alta cardinalidad: **`escuela` no aporta
señal que valga 1.122 columnas.** La exactitud con y sin ella es indistinguible bajo el
esquema aleatorio (0.7287 contra 0.7290). Lo que la importancia por impureza señalaba como la
segunda variable del modelo, medido correctamente, no aporta nada.

La lección general no es "el agrupamiento no importa" sino **"hay que chequearlo"**. En otros
diseños —varias respuestas del mismo encuestado, el mismo país medido en años distintos, mesas
de un padrón que se repite entre elecciones— la diferencia entre los dos esquemas es grande, va
en la dirección del optimismo, y usar el esquema equivocado invalida el resultado.
"""))

# ------------------------------------------------------------------ boosting
A(md(r"""
## Boosting
"""))

A(code(r"""
Xb_tr, Xb_val, yb_tr, yb_val = train_test_split(X, y_bin, test_size=0.25,
                                                random_state=SEED, stratify=y_bin)
pre_fit = preprocesador.fit(Xb_tr)
Mtr, Mval = pre_fit.transform(Xb_tr), pre_fit.transform(Xb_val)

xgb = XGBClassifier(n_estimators=2000, max_depth=5, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                    eval_metric="logloss", early_stopping_rounds=60,
                    random_state=SEED, n_jobs=-1)
xgb.fit(Mtr, yb_tr, eval_set=[(Mtr, yb_tr), (Mval, yb_val)], verbose=False)

hist = xgb.evals_result()
fig, ax = plt.subplots(figsize=(9.5, 5))
ax.plot(hist["validation_0"]["logloss"], lw=2, color=PALETA[0], label="entrenamiento")
ax.plot(hist["validation_1"]["logloss"], lw=2, color=PALETA[2], label="validación")
ax.axvline(xgb.best_iteration, color=PALETA[1], ls="--", lw=1.8,
           label=f"mejor iteración = {xgb.best_iteration}")
ax.set_xlabel("Ronda de boosting"); ax.set_ylabel("Log loss")
ax.set_title("Con 7.233 casos el ensamble puede crecer mucho antes de sobreajustar",
             fontsize=11.5)
ax.legend(fontsize=9.5)
plt.tight_layout()
plt.show()

print(f"rondas solicitadas: 2000   |   usadas: {xgb.best_iteration + 1}")
"""))

A(md(r"""
Contraste directo con `9.3`: allá el *early stopping* cortaba en pocas decenas de rondas
porque los 132 casos se agotaban enseguida. Acá el ensamble sostiene cientos de rondas antes
de que la validación empiece a subir. **Esa es la diferencia que hace el tamaño de la
muestra**, y es la razón por la que el boosting tiene fama de ser el mejor método para datos
tabulares: la fama se construyó con datasets grandes.
"""))

A(code(r"""
# Efecto de la tasa de aprendizaje
fig, ax = plt.subplots(figsize=(9.5, 5))
for eta, color in [(0.3, PALETA[2]), (0.1, PALETA[1]), (0.03, PALETA[0])]:
    m = XGBClassifier(n_estimators=800, max_depth=5, learning_rate=eta,
                      subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                      eval_metric="logloss", random_state=SEED, n_jobs=-1)
    m.fit(Mtr, yb_tr, eval_set=[(Mval, yb_val)], verbose=False)
    curva = m.evals_result()["validation_0"]["logloss"]
    ax.plot(curva, lw=2, color=color, label=f"learning_rate = {eta}")
    k = int(np.argmin(curva))
    ax.scatter([k], [curva[k]], s=80, color=color, zorder=5)

ax.set_xlabel("Ronda de boosting"); ax.set_ylabel("Log loss de validación")
ax.set_title("Tasas bajas llegan más abajo, pero necesitan más árboles\n"
             "(los puntos marcan el mínimo de cada curva)", fontsize=11)
ax.legend(fontsize=9.5)
plt.tight_layout()
plt.show()
"""))

# ------------------------------------------------------------------ comparación
A(md(r"""
## La comparación
"""))

A(code(r"""
modelos = [
    ("BASELINE (clase mayoritaria)", con_pre(DummyClassifier(strategy="most_frequent"))),
    ("Logística regularizada", con_pre(LogisticRegression(C=1.0, max_iter=5000,
                                                          random_state=SEED))),
    ("Árbol prof. 3", con_pre(DecisionTreeClassifier(max_depth=3, random_state=SEED))),
    (f"Árbol prof. óptima ({etiquetas_prof[mejor_i]})",
     con_pre(DecisionTreeClassifier(max_depth=profundidades[mejor_i], random_state=SEED))),
    ("Árbol sin podar", con_pre(DecisionTreeClassifier(random_state=SEED))),
    ("Random Forest, por defecto", con_pre(RandomForestClassifier(
        n_estimators=300, random_state=SEED, n_jobs=-1))),
    ("Random Forest, hojas ≥ 5", con_pre(RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, random_state=SEED, n_jobs=-1))),
    ("XGBoost (500, lr=0.05)", con_pre(XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=1.0, eval_metric="logloss",
        random_state=SEED, n_jobs=-1))),
]

POR_NOMBRE = dict(modelos)      # para referirnos a cada modelo sin depender del orden

tabla = pd.DataFrame([evaluar(est, X, y_bin, nombre) for nombre, est in modelos])
tabla = tabla.sort_values("exactitud", ascending=False).reset_index(drop=True)
tabla.round(4)
"""))

A(code(r"""
fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))

t = tabla.sort_values("exactitud")
colores = []
for m in t["modelo"]:
    if m.startswith("BASELINE"):
        colores.append(PALETA[3])
    elif "Logística" in m:
        colores.append(PALETA[1])
    elif "Árbol" in m:
        colores.append(PALETA[2])
    else:
        colores.append(PALETA[0])
axes[0].barh(t["modelo"], t["exactitud"], xerr=t["exac_sd"], capsize=4,
             color=colores, alpha=0.9)
axes[0].axvline(0.5027, color=PALETA[3], ls="--", lw=1.8)
axes[0].set_xlim(0.45, 0.85)
axes[0].set_xlabel("Exactitud en validación cruzada")
axes[0].set_title("Amarillo: lineal · Rojo: árbol · Azul: ensambles", fontsize=11)
for i, (v, s) in enumerate(zip(t["exactitud"], t["exac_sd"])):
    axes[0].text(v + s + 0.004, i, f"{v:.3f}", va="center", fontsize=9)

for nombre, est in [("Árbol prof. 3", POR_NOMBRE["Árbol prof. 3"]),
                    ("Random Forest, hojas ≥ 5", POR_NOMBRE["Random Forest, hojas ≥ 5"]),
                    ("XGBoost", POR_NOMBRE["XGBoost (500, lr=0.05)"])]:
    est.fit(X_tr, y_tr)
    RocCurveDisplay.from_estimator(est, X_te, y_te, ax=axes[1], name=nombre)
axes[1].plot([0, 1], [0, 1], ls="--", color=PALETA[3], lw=1.4)
axes[1].set_title("Curvas ROC sobre un conjunto apartado", fontsize=11)

plt.tight_layout()
plt.show()
"""))

A(md(r"""
### Cómo se lee la tabla

Con 7.233 casos aparece la progresión que en `9.3` no se veía, pero con dos matices que vale
mirar de frente:

| Modelo | Exactitud | Sobre el baseline |
|---|---|---|
| Baseline | 0.503 | — |
| Árbol de profundidad 3 | 0.666 | +16 puntos |
| Árbol sin podar | 0.716 | +21 puntos |
| **Random Forest, por defecto** | **0.729** | +23 puntos |
| Árbol con profundidad óptima (15) | 0.749 | +25 puntos |
| Regresión logística | 0.757 | +25 puntos |
| **Random Forest con hojas ≥ 5** | **0.772** | +27 puntos |
| **XGBoost** | **0.775** | **+27 puntos** |

**Primer matiz: Random Forest con los valores por defecto queda por debajo de un árbol único
bien calibrado.** 0.729 contra 0.749. Eso contradice la intuición de que un ensamble siempre
mejora, y tiene una explicación precisa.

En `9.3` dijimos que `n_estimators` no sobreajusta, y es verdad. Pero de ahí se sigue con
frecuencia una conclusión falsa: que Random Forest no necesita calibración. **La cantidad de
árboles no sobreajusta; la profundidad de cada árbol sí.** Y el valor por defecto de
scikit-learn es `min_samples_leaf=1`, o sea árboles crecidos hasta hojas de una sola
observación. Con solo cuatro predictores el submuestreo de variables no alcanza a decorrelacionar
lo suficiente, y el promedio de árboles sobreajustados sigue sobreajustado.

Poniendo `min_samples_leaf=5` —una restricción mínima— el bosque salta de **0.729 a 0.772** y
pasa a empatar con XGBoost. Cuatro puntos de exactitud por un solo hiperparámetro.

**Segundo matiz: la regresión logística se defiende bien.** Con 0.757 le gana a un árbol único
y al bosque sin calibrar, y queda a menos de dos puntos de los mejores ensambles. La
comparación honesta no es "los ensambles ganan por lejos" sino "los ensambles bien calibrados
ganan por unos dos puntos". Sigue siendo una ventaja real —dos puntos sobre 7.233 mesas son
unas 140 mesas mejor clasificadas— y sigue siendo mucho menos de lo que la fama del boosting
sugiere.

El AUC-ROC ordena igual que la exactitud, lo que confirma que el resultado no depende del
umbral de 0.5.
"""))

A(code(r"""
# La matriz de confusión del mejor modelo, fuera de muestra
mejor_modelo = POR_NOMBRE["XGBoost (500, lr=0.05)"]
y_pred = cross_val_predict(mejor_modelo, X, y_bin, cv=CV, n_jobs=-1)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
for ax, norm, titulo in [(axes[0], None, "Recuentos"),
                         (axes[1], "true", "Normalizada por fila")]:
    cm = confusion_matrix(y_bin, y_pred, normalize=norm)
    sns.heatmap(cm, annot=True, fmt="d" if norm is None else ".3f", cmap="Blues",
                xticklabels=["Es Ahora BA", "LLA"],
                yticklabels=["Es Ahora BA", "LLA"], cbar=False, ax=ax)
    ax.set_xlabel("Predicho"); ax.set_ylabel("Observado"); ax.set_title(titulo, fontsize=11)
fig.suptitle("XGBoost, predicciones fuera de muestra", fontsize=12.5)
plt.tight_layout()
plt.show()

print(f"exactitud fuera de muestra: {accuracy_score(y_bin, y_pred):.4f}")
"""))

A(md(r"""
La matriz es casi simétrica: el modelo se equivoca de manera parecida en las dos direcciones.
Con clases balanceadas y sin costos asimétricos declarados, eso es lo deseable.
"""))

A(code(r"""
# Curva de aprendizaje: acá sí se aplana
fig, ax = plt.subplots(figsize=(9.5, 5.2))
for nombre, est, color in [
        ("XGBoost", POR_NOMBRE["XGBoost (500, lr=0.05)"], PALETA[0]),
        ("Random Forest (hojas ≥ 5)", POR_NOMBRE["Random Forest, hojas ≥ 5"], PALETA[4]),
        ("Árbol prof. 3", POR_NOMBRE["Árbol prof. 3"], PALETA[2]),
        ("Logística", POR_NOMBRE["Logística regularizada"], PALETA[1])]:
    tam, _, te_s = learning_curve(est, X, y_bin,
                                  train_sizes=np.linspace(0.05, 1.0, 10),
                                  cv=StratifiedKFold(4, shuffle=True, random_state=SEED),
                                  scoring="accuracy", n_jobs=-1)
    ax.plot(tam, te_s.mean(axis=1), "o-", lw=2.1, color=color, label=nombre)
    ax.fill_between(tam, te_s.mean(axis=1) - te_s.std(axis=1),
                    te_s.mean(axis=1) + te_s.std(axis=1), alpha=0.12, color=color)

ax.axhline(0.5027, color=PALETA[3], ls=":", lw=1.8, label="baseline")
ax.axvline(105, color="gray", ls="--", lw=1.4)
ax.annotate("tamaño de la muestra\nde 9.3 (105 casos)", (105, 0.56),
            xytext=(30, -6), textcoords="offset points", fontsize=8.5, color="gray")
ax.set_xlabel("Casos de entrenamiento")
ax.set_ylabel("Exactitud en validación")
ax.set_title("Curvas de aprendizaje: el boosting necesita datos para despegar", fontsize=11.5)
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()
"""))

A(md(r"""
Este gráfico es la respuesta a la pregunta con la que abrió el notebook.

En la zona izquierda —el rango donde vive `9.3`, marcado con la línea vertical— las curvas
están mezcladas y la logística compite de igual a igual, o incluso adelante. A medida que crece
la muestra las curvas se separan: el árbol único se aplana enseguida, la logística se aplana
después, y las dos flexibles siguen subiendo. Al final el boosting y el bosque calibrado quedan
por encima de la lineal, con una brecha chica pero estable.

**La respuesta a "¿conviene un ensamble?" es: depende de cuántos casos haya.** No es una
propiedad del método sino de la relación entre la complejidad del modelo y la información
disponible, que es exactamente el compromiso sesgo-varianza de `9.0` visto desde otro ángulo.
"""))

# ------------------------------------------------------------------ cierre aplicado
A(md(r"""
## Cierre aplicado: el mapa de lo que el modelo no explica

El modelo predice el ganador de una mesa a partir de barrio, padrón y participación. Acierta
en unas tres de cada cuatro mesas. La pregunta aplicada es qué hacer con eso, y tiene dos
respuestas.

**La primera: dónde se define la elección.** Las mesas donde el modelo asigna una probabilidad
cercana a 0.5 son las competitivas — no las que el modelo predice mal, sino las que
genuinamente están en disputa según su estructura territorial. Ahí es donde un voto adicional
cambia el resultado.

**La segunda: dónde falla la lógica territorial.** Cuando el modelo predice con confianza y se
equivoca, ese territorio se comporta distinto de lo que su composición sugiere. Para un
analista, esas son las zonas que hay que ir a mirar de cerca.
"""))

A(code(r"""
proba = cross_val_predict(mejor_modelo, X, y_bin, cv=CV, method="predict_proba",
                          n_jobs=-1)[:, 1]

resultados = datos[["mesa", "escuela", "circuito", "nombre_barrio", "comuna_y",
                    "participacion", "electores_totales", "ganador", "margen"]].copy()
resultados["prob_lla"] = proba
resultados["gano_lla"] = y_bin.values
resultados["predicho_lla"] = (proba > 0.5).astype(int)
resultados["acierto"] = (resultados["predicho_lla"] == resultados["gano_lla"]).astype(int)
resultados["confianza"] = np.abs(proba - 0.5) * 2
resultados["competitiva"] = resultados["confianza"] < 0.2

print(f"exactitud global: {resultados['acierto'].mean():.4f}")
print(f"mesas competitivas (probabilidad entre 0.4 y 0.6): "
      f"{resultados['competitiva'].sum()} ({resultados['competitiva'].mean():.1%})")
print(f"\nexactitud en las competitivas: "
      f"{resultados.loc[resultados['competitiva'], 'acierto'].mean():.3f}")
print(f"exactitud en las de alta confianza (>0.8): "
      f"{resultados.loc[resultados['confianza'] > 0.8, 'acierto'].mean():.3f}")
"""))

A(md(r"""
La probabilidad predicha está **calibrada** en un sentido útil: donde el modelo dice estar
seguro acierta mucho más que donde dice dudar. Eso valida usar la confianza como medida de
competitividad, y no solo como medida de la performance del modelo.
"""))

A(code(r"""
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))

axes[0].hist(proba, bins=45, color=PALETA[0], alpha=0.85, edgecolor="white")
axes[0].axvline(0.5, color="black", ls="--", lw=1.6)
axes[0].axvspan(0.4, 0.6, color=PALETA[1], alpha=0.25, label="zona competitiva")
axes[0].set_xlabel("Probabilidad predicha de que gane LLA")
axes[0].set_ylabel("Cantidad de mesas")
axes[0].set_title("Distribución de las probabilidades predichas", fontsize=11)
axes[0].legend(fontsize=9)

bins = np.linspace(0, 1, 11)
resultados["bin_prob"] = pd.cut(resultados["prob_lla"], bins, include_lowest=True)
cal = resultados.groupby("bin_prob", observed=True).agg(
    predicho=("prob_lla", "mean"), observado=("gano_lla", "mean"), n=("mesa", "size"))
axes[1].plot([0, 1], [0, 1], ls="--", color=PALETA[3], lw=1.6,
             label="calibración perfecta")
axes[1].plot(cal["predicho"], cal["observado"], "o-", lw=2.2, color=PALETA[0],
             ms=8, label="observado")
for _, f in cal.iterrows():
    axes[1].annotate(f"{int(f['n'])}", (f["predicho"], f["observado"]),
                     xytext=(4, -11), textcoords="offset points", fontsize=7.5,
                     color="gray")
axes[1].set_xlabel("Probabilidad predicha (promedio del tramo)")
axes[1].set_ylabel("Proporción real de mesas ganadas por LLA")
axes[1].set_title("Curva de calibración\n(los números son la cantidad de mesas)", fontsize=11)
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.show()
"""))

A(code(r"""
# Errores por barrio: ¿dónde falla la lógica territorial?
por_barrio = (resultados.groupby("nombre_barrio")
              .agg(mesas=("mesa", "size"),
                   tasa_acierto=("acierto", "mean"),
                   competitivas=("competitiva", "mean"),
                   margen_medio=("margen", "mean"),
                   participacion=("participacion", "mean"))
              .sort_values("tasa_acierto"))

print("Los 8 barrios donde el modelo falla más:")
print(por_barrio.head(8).round(3).to_string())
print("\nLos 5 barrios donde el modelo acierta más:")
print(por_barrio.tail(5).round(3).to_string())
"""))

A(code(r"""
# El mapa
try:
    import geopandas as gpd

    barrios_gdf = gpd.read_file("../../dataset/barrios_caba/caba_barrios.geojson")
    barrios_gdf["nombre_barrio"] = barrios_gdf["BARRIO"]
    mapa = barrios_gdf.merge(por_barrio.reset_index(), on="nombre_barrio", how="left")

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 8))

    mapa.plot(column="margen_medio", cmap="RdBu", legend=True, ax=axes[0],
              edgecolor="white", linewidth=0.6,
              legend_kwds={"label": "Margen LLA − Es Ahora BA (pp)", "shrink": 0.6})
    axes[0].set_title("Lo que el modelo explica bien:\nla estructura territorial del voto",
                      fontsize=11.5)
    axes[0].axis("off")

    mapa.plot(column="tasa_acierto", cmap="viridis", legend=True, ax=axes[1],
              edgecolor="white", linewidth=0.6,
              legend_kwds={"label": "Tasa de acierto del modelo", "shrink": 0.6})
    for _, f in mapa.nsmallest(5, "tasa_acierto").iterrows():
        c = f.geometry.centroid
        axes[1].annotate(f["nombre_barrio"], (c.x, c.y), fontsize=7.5,
                         ha="center", color="white", fontweight="bold")
    axes[1].set_title("Lo que no explica:\nbarrios donde el modelo se equivoca más",
                      fontsize=11.5)
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
except ImportError:
    print("geopandas no está instalado en este entorno; el mapa es opcional.")
    print("Para instalarlo:  pip install geopandas")
"""))

A(code(r"""
# El producto: ranking de circuitos por competitividad
por_circuito = (resultados.groupby("circuito")
                .agg(mesas=("mesa", "size"),
                     electores=("electores_totales", "sum"),
                     prob_media_lla=("prob_lla", "mean"),
                     mesas_competitivas=("competitiva", "sum"),
                     margen_medio=("margen", "mean"),
                     barrio=("nombre_barrio", lambda s: s.mode().iloc[0]))
                .assign(pct_competitivas=lambda t: t["mesas_competitivas"] / t["mesas"]))

prioritarios = por_circuito.nlargest(12, "mesas_competitivas")
print("Circuitos a priorizar: los que concentran más mesas en disputa\n")
prioritarios[["barrio", "mesas", "electores", "mesas_competitivas",
              "pct_competitivas", "margen_medio"]].round(3)
"""))

A(code(r"""
fig, ax = plt.subplots(figsize=(10, 6))
p = prioritarios.sort_values("mesas_competitivas")
etiquetas = [f"circ. {int(i)} · {r['barrio'][:16]}" for i, r in p.iterrows()]
ax.barh(etiquetas, p["mesas_competitivas"], color=PALETA[1], alpha=0.9,
        label="mesas competitivas")
ax.barh(etiquetas, p["mesas"], color=PALETA[3], alpha=0.30, zorder=0,
        label="total de mesas del circuito")
for i, (_, r) in enumerate(p.iterrows()):
    ax.text(r["mesas"] + 1, i, f"{r['electores']:,.0f} electores", va="center",
            fontsize=8, color="#555")
ax.set_xlabel("Cantidad de mesas")
ax.set_title("Los 12 circuitos con más mesas en disputa\n"
             "donde un voto adicional cambia más resultados", fontsize=11.5)
ax.legend(fontsize=9, loc="lower right")
plt.tight_layout()
plt.show()
"""))

A(md(r"""
### El producto

Tres salidas concretas, todas usables por alguien que tenga que decidir dónde poner recursos:

1. **El ranking de circuitos en disputa.** No es el ranking de circuitos donde la elección
   estuvo pareja —eso se lee del resultado— sino de aquellos donde **la estructura territorial
   no determina el resultado**. Son los lugares donde la campaña, la organización y la
   fiscalización tienen margen para mover algo, y vienen con la cantidad de electores asociada
   para poder ponderar el esfuerzo.

2. **El mapa de los barrios donde falla el modelo.** Un barrio con tasa de acierto baja es un
   barrio internamente heterogéneo o con un comportamiento que su composición no anticipa. Son
   candidatos para trabajo cualitativo: entender qué pasa ahí que no pasa en barrios
   parecidos.

3. **Una medida de competitividad calibrada por mesa.** La curva de calibración muestra que la
   probabilidad predicha se corresponde con la frecuencia observada, así que sirve como
   insumo cuantitativo y no solo como ordenamiento.

Y una limitación que hay que decir: el modelo describe la elección de 2025, y su capacidad de
predecir la próxima depende de que la estructura territorial del voto se mantenga. Las
elecciones porteñas de la última década sugieren que se mantiene bastante, pero eso es un
supuesto sustantivo, no un resultado de este notebook.
"""))

A(md(r"""
## Síntesis

1. Con 7.233 mesas y un baseline de 0.503, todos los modelos le ganan al baseline por entre 16
   y 27 puntos, y el mejor es **XGBoost con 0.775**.

2. La curva de validación sobre `max_depth` tiene un **máximo claro** y la brecha entre
   entrenamiento y validación muestra la varianza. Con 132 casos, en `9.3`, el mismo gráfico no
   mostraba nada.

3. **`n_estimators` no sobreajusta, pero la profundidad de los árboles sí.** Random Forest con
   los valores por defecto de scikit-learn (`min_samples_leaf=1`) rinde **0.729**, por debajo de
   un árbol único bien calibrado. Poniendo `min_samples_leaf=5` sube a **0.772**. Que un método
   sea robusto no lo vuelve libre de hiperparámetros.

4. Los ensambles bien calibrados le ganan al modelo lineal, pero **por unos dos puntos**
   (0.775 y 0.772 contra 0.757), no por goleada. La ventaja es real y consistente —el efecto de
   la participación depende del barrio, o sea que hay interacción— y bastante menor que lo que
   la fama del boosting sugiere.

5. `escuela`, con **1.122 niveles**, es un caso real de la trampa de la alta cardinalidad: la
   importancia por impureza la señala como la segunda variable del modelo y, medida
   correctamente, **no aporta nada** (0.7287 con ella contra 0.7290 sin ella).

6. **La dependencia entre observaciones se chequea, no se supone.** Las mesas de una escuela
   son 83% homogéneas. La firma de la fuga por agrupamiento sería `StratifiedKFold` >
   `StratifiedGroupKFold`, y observamos lo contrario: no hay optimismo por agrupamiento acá. En
   otros diseños sí lo hay y usar el esquema equivocado invalida el resultado.

7. La curva de aprendizaje conjunta responde la pregunta del notebook: **por debajo de unos mil
   casos los métodos empatan; por encima, los flexibles se despegan.** "¿Conviene un ensamble?"
   no es una propiedad del método sino de la relación entre complejidad e información.

8. El cierre no es una métrica: es un **ranking de circuitos en disputa**, un **mapa de dónde
   falla la lógica territorial** y una **medida calibrada de competitividad por mesa** —donde el
   modelo dice estar seguro acierta el 96%, y donde dice dudar acierta el 57%.

## Comparación con 9.3

| | `9.3` · encuesta | `9.3b` · mesas |
|---|---|---|
| Casos | 132 | 7.233 |
| Predictores | 56 columnas | 62 columnas |
| Baseline | 0.545 | 0.503 |
| Árbol único, mejor versión | 0.671 | 0.749 |
| Random Forest | 0.788 | 0.772 |
| XGBoost | 0.765 | **0.775** |
| Modelo lineal | **0.791** | 0.757 |
| Ventaja del mejor ensamble sobre el lineal | **−0.003 (empatan)** | **+0.018** |
| ¿Sobreajusta el árbol sin podar? | sí, y no se nota | sí, y se ve en la curva |

Los dos notebooks usan los mismos métodos y casi el mismo código. Lo único que cambia de fondo
es la cantidad de datos, y el veredicto se invierte: con 132 casos el modelo lineal empata con
el mejor ensamble; con 7.233 los ensambles se ponen adelante.

**Esa es la lección que ninguno de los dos notebooks puede dar solo.** Y notar la magnitud:
incluso con 7.233 casos la ventaja del boosting sobre una regresión logística es de menos de
dos puntos. La elección del método casi nunca es lo que más mueve un resultado; los datos y las
variables sí.
"""))

write_nb(C, OUT)
