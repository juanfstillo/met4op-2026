# -*- coding: utf-8 -*-
"""Construye 9.3.arboles.ipynb  (variante encuesta 134)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nbb import md, code, write_nb, requisitos, REPO

OUT = os.path.join(REPO, "ESTADISTICA", "9.Machine Learning", "9.3.arboles.ipynb")

C = []
A = C.append

A(md(r"""
# Árboles de decisión, Random Forest y XGBoost

**Unidad 9 · Machine Learning · Notebook 3 de 6**

Cierra el bloque supervisado. Misma etiqueta y mismos datos que `9.2`, otra familia de
modelos: en lugar de una frontera lineal, **particiones sucesivas del espacio**.
"""))

A(md(r"""
## La pregunta de este notebook

En `9.2` la regresión logística alcanzó una exactitud de 0.791 sobre la división
peronismo/no peronismo, contra un baseline de 0.545. Su limitación es estructural: la
frontera de decisión es un hiperplano, así que el modelo no puede representar interacciones
ni relaciones no monótonas salvo que se las especifique a mano.

Los árboles y sus ensambles no tienen esa restricción. Aprenden interacciones sin que nadie
las declare, y su frontera puede tener cualquier forma. La pregunta es directa:

> **¿Esa flexibilidad adicional mejora la predicción sobre estos datos?**

Adelantamos que la respuesta es interesante y que no es la que uno esperaría. Pero la
respuesta importa menos que el camino: los tres métodos de este notebook —árbol, bosque,
boosting— son los que más se usan en la práctica profesional del análisis de datos, y hay
que saber leerlos.
"""))

A(md("## Requisitos"))

A(requisitos(extras=["xgboost", "shap"]))

A(md("## Librerías"))

A(code(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, cross_validate, learning_curve,
                                     train_test_split, validation_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

from xgboost import XGBClassifier

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)
pd.set_option("display.width", 200)

SEED = 42
rng = np.random.default_rng(SEED)
PALETA = ["#0073C2", "#EFC000", "#CD534C", "#868686", "#1E2749"]
"""))

# ------------------------------------------------------------------ datos
A(md(r"""
## Los datos

Los mismos de `9.2`, con la misma depuración: se descartan las dos etiquetas inválidas, se
excluyen `NH_EDAD`, `TH_EDAD` y `N_BUSCA_HIJOS` por errores del cuestionario, se saca `UP`
por varianza casi nula, y los faltantes estructurales se codifican en lugar de imputarse.

No repetimos la justificación de cada decisión: está en `9.2`.
"""))

A(code(r"""
datos = pd.read_excel("../../dataset/encuesta134/134NODUMMY.xlsx")
datos = datos[~datos["ETIQUETA"].isin(["Linda", "Es una verga la política"])].reset_index(drop=True)

# feature engineering, igual que en 9.2
datos["HIJOS_COMB"] = np.where(
    datos["HIJOS"] == 1, 1.0,
    np.where((datos["HIJOS"] == 0) & (datos["BUSCA_HIJOS"] == 1), 1.0,
             np.where(datos["BUSCA_HIJOS"].isna(), np.nan, 0.0)))
datos["PROBLEMA_RAMA"] = np.where(
    datos["EL_PROBLEMA"] == 1,
    np.where(datos["DESIGUALDAD"] == 1, "concentracion_riqueza_si", "concentracion_riqueza_no"),
    np.where(datos["NO_PIENSAN"] == 1, "no_piensan_si", "no_piensan_no"))
for col in ["GRUPO", "PROLE"]:
    datos[col] = datos[col].fillna("No aplica / no trabaja")

BINARIAS = ["MASCOTHIJO", "GORRA", "EF", "EJERCITO", "TARIFAS", "EMPRESARIOS",
            "MALVINAS", "EF_QUEES", "EL_PROBLEMA", "GENERO", "TRABAJA", "HIJOS"]
ORDINALES = ["EEUU", "PALESTINA", "ISRAEL", "UCRANIA", "RUSIA", "BOLIVIA", "CHINA",
             "INGLATERRA", "EDAD", "ESTUDIO", "SOCIECON"]
CATEGORICAS = ["SS", "PROGRAMA", "NOTICIAS", "GRUPO", "PROLE", "NO_HIJOS_PQ",
               "PROBLEMA_RAMA"]
NUMERICAS = BINARIAS + ORDINALES + ["HIJOS_COMB"]

X = datos[NUMERICAS + CATEGORICAS].copy()
X[NUMERICAS] = X[NUMERICAS].astype(float)
y_7 = datos["ETIQUETA"]
y_2 = pd.Series(np.where(y_7 == "Peronista", "Peronista", "No peronista"), index=y_7.index)
MAPA_3 = {"Peronista": "Peronismo/Izquierda", "De izquierda": "Peronismo/Izquierda",
          "Liberal": "Derecha/Liberal", "De derecha": "Derecha/Liberal",
          "Radical": "Sin adscripción clara", "Apolitico": "Sin adscripción clara",
          "No sabe/ No contesta": "Sin adscripción clara"}
y_3 = y_7.map(MAPA_3)

print(f"n = {len(X)}   predictores = {X.shape[1]}")
print(f"baseline 2 clases: {y_2.value_counts(normalize=True).iloc[0]:.3f}")
"""))

A(code(r"""
preprocesador = ColumnTransformer([
    ("num", Pipeline([("imputar", SimpleImputer(strategy="median")),
                      ("escalar", StandardScaler())]), NUMERICAS),
    ("cat", Pipeline([("imputar", SimpleImputer(strategy="most_frequent")),
                      ("dummies", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                                min_frequency=6,
                                                sparse_output=False))]), CATEGORICAS),
])

CV = RepeatedStratifiedKFold(n_splits=5, n_repeats=6, random_state=SEED)
CV_PRED = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
METRICAS = ["accuracy", "f1_macro", "balanced_accuracy"]

def evaluar(estimador, X, y, nombre):
    r = cross_validate(estimador, X, y, cv=CV, scoring=METRICAS, n_jobs=-1)
    return {"modelo": nombre,
            "exactitud": r["test_accuracy"].mean(),
            "exac_sd": r["test_accuracy"].std(),
            "f1_macro": r["test_f1_macro"].mean(),
            "f1_sd": r["test_f1_macro"].std()}

def con_arbol(modelo):
    return Pipeline([("pre", preprocesador), ("clf", modelo)])
"""))

A(md(r"""
Un árbol **no necesita que las variables estén escaladas**: sus cortes son del tipo
$x_j \le c$ y no cambian si la variable se reescala. Es una ventaja práctica real frente a los
métodos de `9.1` y `9.2`, donde estandarizar era obligatorio. Mantenemos el mismo
preprocesamiento igual para que la comparación entre familias sea limpia.
"""))

# ------------------------------------------------------------------ teoría árbol
A(md(r"""
## Teoría: qué hace un árbol de decisión

Un árbol parte el espacio de predictores en regiones rectangulares y asigna una predicción
constante a cada región. El procedimiento es **recursivo y voraz** (*greedy*):

1. Considerar todos los cortes posibles de la forma $x_j \le c$, para toda variable $j$ y
   todo umbral $c$.
2. Elegir el corte que produzca la mayor reducción de **impureza**.
3. Repetir dentro de cada una de las dos regiones resultantes.
4. Detenerse cuando se cumple un criterio de parada.

"Voraz" quiere decir que en cada paso elige el mejor corte **local**, sin evaluar si un corte
peor ahora habilitaría uno mucho mejor después. No garantiza el árbol óptimo global —
encontrarlo es un problema NP-completo — pero es rápido y funciona bien en la práctica.
"""))

A(code(r"""
# Un árbol de profundidad 2 sobre dos variables: el espacio partido y el árbol, lado a lado
Xj = np.vstack([rng.normal([-1.1, -0.6], 0.85, (60, 2)),
                rng.normal([1.5, 0.9], 0.85, (60, 2)),
                rng.normal([-0.8, 1.9], 0.7, (40, 2))])
yj = np.r_[np.zeros(60), np.ones(60), np.ones(40)]

arbol_demo = DecisionTreeClassifier(max_depth=2, random_state=SEED).fit(Xj, yj)

fig = plt.figure(figsize=(14.5, 5.4))
ax1 = fig.add_subplot(1, 2, 1)
g1, g2 = np.meshgrid(np.linspace(-4, 4.2, 400), np.linspace(-3.2, 4.2, 400))
Z = arbol_demo.predict(np.c_[g1.ravel(), g2.ravel()]).reshape(g1.shape)
ax1.contourf(g1, g2, Z, alpha=0.20, colors=[PALETA[0], PALETA[2]], levels=[-0.5, 0.5, 1.5])
ax1.contour(g1, g2, Z, levels=[0.5], colors="black", linewidths=2)
for k, (color, marcador) in enumerate(zip([PALETA[0], PALETA[2]], ["o", "^"])):
    mm = yj == k
    ax1.scatter(Xj[mm, 0], Xj[mm, 1], s=42, color=color, marker=marcador,
                alpha=0.85, edgecolor="white")
ax1.set_xlabel("$x_1$"); ax1.set_ylabel("$x_2$")
ax1.set_title("La frontera es escalonada:\ncortes paralelos a los ejes", fontsize=11)

ax2 = fig.add_subplot(1, 2, 2)
plot_tree(arbol_demo, feature_names=["$x_1$", "$x_2$"], class_names=["A", "B"],
          filled=True, rounded=True, impurity=True, fontsize=9, ax=ax2)
ax2.set_title("El mismo modelo, como árbol", fontsize=11)
plt.tight_layout()
plt.show()
"""))

A(md(r"""
Los dos paneles son **el mismo objeto**. Cada nodo interno del árbol es una línea del
gráfico izquierdo; cada hoja es una región.

De ahí se siguen las dos características centrales de los árboles:

- **La frontera es escalonada y paralela a los ejes.** Un árbol no puede representar una
  frontera diagonal con un corte: la aproxima con una escalera. Es lo inverso de la
  logística, que solo puede hacer diagonales.
- **Capturan interacciones gratis.** El significado del segundo corte depende de por qué rama
  se llegó. Eso es exactamente una interacción, y el árbol la encuentra sin que nadie la
  especifique. En un modelo lineal habría que agregar el término $x_1 \cdot x_2$ a mano.
"""))

A(md(r"""
### La impureza: Gini y entropía

Para elegir un corte hace falta medir cuán "mezclado" está un nodo. Con $K$ clases y
proporciones $p_1, \dots, p_K$ dentro del nodo, las dos medidas habituales son:

$$ \text{Gini} = 1 - \sum_{k=1}^{K} p_k^2 \qquad\qquad
   \text{Entropía} = -\sum_{k=1}^{K} p_k \log_2 p_k $$

Las dos valen **0** cuando el nodo es puro (todos los casos de una sola clase) y alcanzan su
**máximo** cuando las clases están repartidas en partes iguales.

La **ganancia de información** de un corte es la reducción ponderada de impureza:

$$ \Delta = I(\text{padre}) - \frac{n_{\text{izq}}}{n} I(\text{izq}) - \frac{n_{\text{der}}}{n} I(\text{der}) $$

El algoritmo prueba todos los cortes y se queda con el de mayor $\Delta$.
"""))

A(code(r"""
p = np.linspace(0.001, 0.999, 400)
gini = 1 - (p**2 + (1 - p)**2)
entropia = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
error = 1 - np.maximum(p, 1 - p)

fig, ax = plt.subplots(figsize=(8, 4.8))
ax.plot(p, gini, lw=2.4, color=PALETA[0], label="Gini")
ax.plot(p, entropia / 2, lw=2.4, color=PALETA[2], label="Entropía / 2")
ax.plot(p, error, lw=2.2, color=PALETA[3], ls="--", label="Error de clasificación")
ax.axvline(0.5, color="gray", ls=":", lw=1.2)
ax.set_xlabel("Proporción de la clase positiva en el nodo ($p$)")
ax.set_ylabel("Impureza")
ax.set_title("Las tres medidas de impureza para dos clases", fontsize=11)
ax.legend(fontsize=9.5)
plt.tight_layout()
plt.show()
"""))

A(md(r"""
Gini y entropía son casi indistinguibles una vez reescaladas, y en la práctica dan árboles
muy parecidos: la elección entre ellas casi nunca importa. Lo interesante es la tercera
curva.

El **error de clasificación** es lineal a trozos, con un pico en 0.5 y derivada constante a
cada lado. Eso lo hace mal criterio para hacer crecer un árbol: un corte que mueve la
proporción de 0.8 a 0.9 en una rama y de 0.8 a 0.7 en la otra no cambia el error total, así
que el error no lo "premia" — pero sí cambió la pureza, y Gini y entropía sí lo detectan
porque son **estrictamente convexas**. Por eso se usa Gini o entropía para crecer, y el error
de clasificación recién al podar.
"""))

A(code(r"""
# La ganancia de información de un corte, calculada a mano
def gini_nodo(y_nodo):
    if len(y_nodo) == 0:
        return 0.0
    p = np.array([(y_nodo == c).mean() for c in np.unique(y_nodo)])
    return 1 - (p**2).sum()

y_bin = (y_2 == "Peronista").astype(int).values

print(f"Impureza Gini del nodo raíz (los {len(y_bin)} casos): {gini_nodo(y_bin):.4f}\n")
print("Ganancia de información de un corte por cada variable binaria:\n")
ganancias = []
for var in ["TARIFAS", "GORRA", "EMPRESARIOS", "MALVINAS", "EJERCITO", "GENERO",
            "MASCOTHIJO", "EF_QUEES"]:
    v = datos[var].fillna(datos[var].median()).values
    izq, der = y_bin[v == 0], y_bin[v == 1]
    gi = gini_nodo(y_bin) - (len(izq) / len(y_bin) * gini_nodo(izq)
                             + len(der) / len(y_bin) * gini_nodo(der))
    ganancias.append({"variable": var, "n_izq": len(izq), "n_der": len(der),
                      "gini_izq": round(gini_nodo(izq), 3),
                      "gini_der": round(gini_nodo(der), 3),
                      "ganancia": round(gi, 4)})
pd.DataFrame(ganancias).sort_values("ganancia", ascending=False).reset_index(drop=True)
"""))

A(md(r"""
`TARIFAS` es el corte que más reduce la impureza — el mismo predictor que en `9.2` tenía el
*odds ratio* más extremo y el efecto marginal más grande. Los dos métodos, con maquinarias
completamente distintas, están encontrando lo mismo. Eso es una buena señal sobre la
robustez del hallazgo.
"""))

# ------------------------------------------------------------------ árbol sobre los datos
A(md(r"""
## Un árbol sobre nuestros datos

Empezamos con la división peronismo/no peronismo y un árbol de profundidad 3, que es lo
máximo que se puede leer cómodamente en una figura.
"""))

A(code(r"""
arbol_pipe = con_arbol(DecisionTreeClassifier(max_depth=3, random_state=SEED,
                                              criterion="gini"))
arbol_pipe.fit(X, y_2)

nombres_features = arbol_pipe[:-1].get_feature_names_out()
nombres_limpios = [n.split("__", 1)[1] for n in nombres_features]

fig, ax = plt.subplots(figsize=(17, 8))
plot_tree(arbol_pipe[-1], feature_names=list(nombres_limpios),
          class_names=list(arbol_pipe[-1].classes_), filled=True, rounded=True,
          fontsize=8, ax=ax, proportion=False)
ax.set_title("Árbol de profundidad 3 · peronismo vs. resto", fontsize=13)
plt.tight_layout()
plt.show()
"""))

A(code(r"""
# Las reglas del árbol, en texto
from sklearn.tree import export_text
print(export_text(arbol_pipe[-1], feature_names=list(nombres_limpios),
                  class_names=list(arbol_pipe[-1].classes_), max_depth=3,
                  spacing=2, decimals=2))
"""))

A(md(r"""
La virtud del árbol está acá: **el modelo se lee como un conjunto de reglas en castellano**.
No hay que interpretar coeficientes ni exponenciarlos. Una rama dice, literalmente, "si esta
persona está en contra del aumento de tarifas y valora bien a Bolivia, entonces es peronista".

Esa transparencia es la razón por la que los árboles se usan tanto para comunicar
resultados, y por la que aparecen en informes que van a lectores no técnicos.

El costo lo vemos a continuación.
"""))

# ------------------------------------------------------------------ sobreajuste y poda
A(md(r"""
## Sobreajuste: por qué hay que podar

Un árbol sin restricciones sigue partiendo hasta que cada hoja es pura. Con suficiente
profundidad puede aislar **cada observación en su propia hoja**, y entonces el error de
entrenamiento es exactamente cero. Es el caso extremo del sobreajuste de `9.0`.

La curva de validación sobre `max_depth` es la versión concreta del compromiso
sesgo-varianza.
"""))

A(code(r"""
profundidades = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]
tr, te = validation_curve(
    con_arbol(DecisionTreeClassifier(random_state=SEED)),
    X, y_2, param_name="clf__max_depth", param_range=profundidades,
    cv=CV, scoring="accuracy", n_jobs=-1)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(profundidades, tr.mean(axis=1), "o-", lw=2.2, color=PALETA[0],
        label="entrenamiento")
ax.fill_between(profundidades, tr.mean(axis=1) - tr.std(axis=1),
                tr.mean(axis=1) + tr.std(axis=1), alpha=0.15, color=PALETA[0])
ax.plot(profundidades, te.mean(axis=1), "s-", lw=2.2, color=PALETA[2],
        label="validación cruzada")
ax.fill_between(profundidades, te.mean(axis=1) - te.std(axis=1),
                te.mean(axis=1) + te.std(axis=1), alpha=0.15, color=PALETA[2])
mejor = profundidades[int(np.argmax(te.mean(axis=1)))]
ax.axvline(mejor, color=PALETA[1], ls="--", lw=1.8,
           label=f"mejor profundidad = {mejor}")
ax.axhline(y_2.value_counts(normalize=True).iloc[0], color=PALETA[3], ls=":", lw=1.8,
           label="baseline")
ax.set_xlabel("Profundidad máxima del árbol")
ax.set_ylabel("Exactitud")
ax.set_title("Curva de validación: el entrenamiento llega a 1.0 y la validación no mejora",
             fontsize=11)
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()
"""))

A(md(r"""
El árbol llega a **exactitud 1.0 en entrenamiento** con profundidad 10 o más: memorizó los
132 casos. La curva de validación, en cambio, es plana y baja: se queda entre 0.63 y 0.66 sin
un máximo claro, y el desvío entre pliegues es grande.

Comparalo con el 0.791 del logit regularizado de `9.2`. **Un árbol solo es bastante peor que
el modelo lineal en estos datos**, y no hay profundidad que lo arregle.
"""))

A(md(r"""
### Poda por complejidad-costo

La forma disciplinada de controlar el tamaño del árbol no es fijar `max_depth` a ojo, sino
**hacer crecer el árbol completo y después podarlo**. Se minimiza

$$ R_\alpha(T) = R(T) + \alpha \, |T| $$

donde $R(T)$ es el error del árbol, $|T|$ su cantidad de hojas y $\alpha \geq 0$ el precio
de cada hoja. Con $\alpha = 0$ se obtiene el árbol completo; al subir $\alpha$ se van
colapsando ramas.

Es la misma idea que Ridge y Lasso en `9.1`: **una penalización sobre la complejidad,
calibrada por validación cruzada.** Acá lo que se penaliza es la cantidad de hojas en lugar
de la magnitud de los coeficientes.
"""))

A(code(r"""
# El camino de poda: qué valores de alpha producen árboles distintos
X_mat = preprocesador.fit_transform(X)
arbol_completo = DecisionTreeClassifier(random_state=SEED).fit(X_mat, y_2)
camino = arbol_completo.cost_complexity_pruning_path(X_mat, y_2)
alphas = camino.ccp_alphas[:-1]          # el último colapsa el árbol a una sola hoja

print(f"valores de alpha en el camino de poda: {len(alphas)}")
print(f"rango: [{alphas.min():.5f}, {alphas.max():.5f}]")

hojas, profs = [], []
for a in alphas:
    t = DecisionTreeClassifier(random_state=SEED, ccp_alpha=a).fit(X_mat, y_2)
    hojas.append(t.get_n_leaves())
    profs.append(t.get_depth())

tr2, te2 = validation_curve(
    con_arbol(DecisionTreeClassifier(random_state=SEED)),
    X, y_2, param_name="clf__ccp_alpha", param_range=alphas,
    cv=CV, scoring="accuracy", n_jobs=-1)

fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
axes[0].plot(alphas, hojas, "o-", color=PALETA[0], lw=2)
axes[0].set_xlabel(r"$\alpha$"); axes[0].set_ylabel("Cantidad de hojas")
axes[0].set_title("Al subir el precio de cada hoja, el árbol se encoge", fontsize=10.5)

axes[1].plot(alphas, tr2.mean(axis=1), "o-", color=PALETA[0], lw=2, label="entrenamiento")
axes[1].plot(alphas, te2.mean(axis=1), "s-", color=PALETA[2], lw=2, label="validación")
mejor_a = alphas[int(np.argmax(te2.mean(axis=1)))]
axes[1].axvline(mejor_a, color=PALETA[1], ls="--", lw=1.8,
                label=f"mejor $\\alpha$ = {mejor_a:.4f}")
axes[1].set_xlabel(r"$\alpha$"); axes[1].set_ylabel("Exactitud")
axes[1].set_title("El óptimo de validación está lejos del árbol completo", fontsize=10.5)
axes[1].legend(fontsize=9)
plt.tight_layout()
plt.show()

t_podado = DecisionTreeClassifier(random_state=SEED, ccp_alpha=mejor_a).fit(X_mat, y_2)
print(f"\nárbol completo : {arbol_completo.get_n_leaves()} hojas, "
      f"profundidad {arbol_completo.get_depth()}")
print(f"árbol podado   : {t_podado.get_n_leaves()} hojas, profundidad {t_podado.get_depth()}")
"""))

# ------------------------------------------------------------------ inestabilidad
A(md(r"""
## El problema de fondo: los árboles son inestables

El sobreajuste se puede controlar podando. Hay un problema más profundo, y es el que motiva
todo lo que viene después: **un árbol cambia mucho ante cambios chicos en los datos**.

La causa está en la naturaleza voraz del algoritmo. El primer corte se decide con todos los
datos; si por azar de muestreo otro corte gana por poco, **todo el subárbol que sigue es
distinto**. Los errores se propagan hacia abajo y se amplifican.

Veámoslo: remuestreamos los datos con reposición seis veces y ajustamos el mismo árbol.
"""))

A(code(r"""
fig, axes = plt.subplots(2, 3, figsize=(16.5, 8))
raices = []

for k, ax in enumerate(axes.ravel()):
    idx = rng.integers(0, len(X), len(X))
    t = DecisionTreeClassifier(max_depth=2, random_state=SEED).fit(X_mat[idx], y_2.iloc[idx])
    plot_tree(t, feature_names=list(nombres_limpios), class_names=list(t.classes_),
              filled=True, rounded=True, fontsize=7, ax=ax, impurity=False)
    raiz = nombres_limpios[t.tree_.feature[0]]
    raices.append(raiz)
    ax.set_title(f"Remuestreo {k + 1} · raíz: {raiz}", fontsize=10)

fig.suptitle("El mismo árbol ajustado en seis remuestreos de los mismos datos", fontsize=13)
plt.tight_layout()
plt.show()

print("Variable elegida en la raíz en cada remuestreo:")
for k, r in enumerate(raices, 1):
    print(f"  {k}. {r}")
print(f"\nvariables distintas en la raíz: {len(set(raices))} de 6 remuestreos")
"""))

A(md(r"""
Seis muestras de los mismos datos y la variable de la raíz cambia. Y la raíz es la decisión
**más estable** de un árbol: hacia abajo la variabilidad es mucho mayor.

Esto tiene dos consecuencias:

- **Para la predicción**: alta varianza, en el sentido exacto de `9.0`. El modelo se mueve
  demasiado con la muestra.
- **Para la interpretación**: es un problema serio y frecuentemente ignorado. Si el árbol
  elige `TARIFAS` en la raíz y alguien concluye "la posición sobre tarifas es *el* factor
  determinante", esa conclusión no es robusta: con otra muestra la raíz habría sido otra
  variable, probablemente correlacionada con la primera. **La estructura de un árbol
  particular no es un hallazgo.**

La solución a la varianza es promediar muchos árboles. Hay dos formas de hacerlo, y son las
dos familias que siguen.
"""))

# ------------------------------------------------------------------ Random Forest
A(md(r"""
## Random Forest: promediar árboles en paralelo

### Bagging

*Bagging* viene de *bootstrap aggregating*. La receta:

1. Tomar $B$ muestras bootstrap (con reposición, del mismo tamaño que el original).
2. Ajustar un árbol **sin podar** en cada una.
3. Predecir por **voto mayoritario** entre los $B$ árboles.

Funciona por un resultado elemental de probabilidad: si se promedian $B$ variables con
varianza $\sigma^2$ e independientes, la varianza del promedio es $\sigma^2 / B$. Promediar
reduce la varianza sin tocar el sesgo. Y como los árboles profundos tienen sesgo bajo y
varianza alta, son el candidato ideal.

El problema es el "independientes": los árboles del bagging están **correlacionados** entre
sí, porque se ajustan sobre datos muy parecidos y todos eligen los mismos predictores
fuertes. Si hay una variable dominante, todos los árboles la ponen en la raíz. La reducción
efectiva de varianza es entonces mucho menor que $\sigma^2/B$.

### El aporte de Random Forest

Breiman agregó una idea simple y decisiva: **en cada corte, considerar solo un subconjunto
aleatorio de $m$ predictores** en lugar de todos. Por defecto $m = \sqrt{p}$ para
clasificación.

Eso fuerza a los árboles a ser distintos entre sí: si la variable dominante no está entre las
$m$ candidatas de un corte, ese árbol tiene que usar otra. **Decorrelaciona** los árboles, y
el promedio de árboles decorrelacionados reduce mucho más la varianza.

El precio es que cada árbol individual es algo peor. La ganancia por decorrelación
normalmente lo compensa con holgura.

<figure style="text-align:center; margin:1.4em 0;">
  <img src="media/singletree_vs_rf.png" width="820">
  <figcaption style="font-size:0.9em; color:#555;">
    Un árbol único contra un bosque. Cada árbol del bosque se ajustó sobre una muestra
    bootstrap distinta y puede llegar a una conclusión distinta: acá siete árboles votan
    "Class 1" y tres votan "Class 2".
  </figcaption>
</figure>

Notar que en la figura **los árboles del bosque se contradicen entre sí**, y eso no es un
defecto: es el mecanismo. Si todos coincidieran siempre, promediarlos no reduciría nada. La
diversidad entre árboles es exactamente lo que se compra con el bootstrap y el submuestreo de
variables.
"""))

A(md(r"""
### Cómo predice el bosque un caso nuevo

Una observación nueva **recorre los $B$ árboles** y cada uno emite su predicción. Después se
agregan: por **voto mayoritario** en clasificación, por **promedio** en regresión.

<figure style="text-align:center; margin:1.4em 0;">
  <img src="media/rf.png" width="680">
  <figcaption style="font-size:0.9em; color:#555;">
    El camino de un caso nuevo por tres árboles del ensamble y la agregación final.
  </figcaption>
</figure>

De acá se sigue algo práctico: la **proporción de árboles** que votó por cada clase sirve como
estimación de probabilidad, y es lo que devuelve `predict_proba`. Si 380 de 500 árboles dicen
"peronista", la probabilidad estimada es 0.76. Esa probabilidad es la que usamos para armar
rankings, como hicimos en `9.2`.
"""))

A(code(r"""
# Cómo mejora el bosque al agregar árboles, y el efecto del submuestreo de variables
n_arboles = [1, 2, 3, 5, 8, 12, 20, 35, 60, 100, 200, 400]
X_tr, X_te, y_tr, y_te = train_test_split(X, y_2, test_size=0.35, random_state=SEED,
                                          stratify=y_2)

fig, ax = plt.subplots(figsize=(9.5, 5))
for max_feat, nombre, color in [(None, "bagging: todas las variables por corte", PALETA[3]),
                                ("sqrt", "Random Forest: $\\sqrt{p}$ variables por corte",
                                 PALETA[0])]:
    curva = []
    for b in n_arboles:
        m = con_arbol(RandomForestClassifier(n_estimators=b, max_features=max_feat,
                                             random_state=SEED, n_jobs=-1))
        m.fit(X_tr, y_tr)
        curva.append(accuracy_score(y_te, m.predict(X_te)))
    ax.plot(n_arboles, curva, "o-", lw=2.2, color=color, label=nombre)

ax.axhline(y_2.value_counts(normalize=True).iloc[0], color=PALETA[2], ls=":", lw=1.8,
           label="baseline")
ax.set_xscale("log")
ax.set_xlabel("Cantidad de árboles en el ensamble")
ax.set_ylabel("Exactitud en el conjunto apartado")
ax.set_title("Más árboles nunca empeora, y se estabiliza rápido", fontsize=11)
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()
"""))

A(md(r"""
Dos lecturas del gráfico:

- **La curva sube y se aplana.** A diferencia de `max_depth`, la cantidad de árboles **no es
  un parámetro de sobreajuste**: agregar árboles nunca empeora el modelo, solo cuesta tiempo.
  Con unos 100 ya se estabiliza. Por eso `n_estimators` no se calibra finamente: se pone un
  número generoso y listo.
- El submuestreo de variables ayuda, aunque en estos datos la diferencia es chica: no hay una
  variable tan dominante como para que todos los árboles se parezcan.

### Error out-of-bag

Un bonus del bootstrap: cada muestra deja afuera en promedio el $1/e \approx 37\%$ de las
observaciones. Cada caso puede entonces evaluarse con los árboles que **no** lo vieron, lo
que da una estimación del error de generalización **sin apartar datos ni hacer validación
cruzada**. Con muestras chicas eso vale mucho.
"""))

A(code(r"""
rf_oob = RandomForestClassifier(n_estimators=500, oob_score=True,
                                random_state=SEED, n_jobs=-1)
rf_oob.fit(X_mat, y_2)
print(f"exactitud out-of-bag        : {rf_oob.oob_score_:.4f}")

s = cross_validate(con_arbol(RandomForestClassifier(n_estimators=500, random_state=SEED,
                                                   n_jobs=-1)),
                   X, y_2, cv=CV, scoring=["accuracy"], n_jobs=-1)
print(f"exactitud en validación cruz.: {s['test_accuracy'].mean():.4f} "
      f"± {s['test_accuracy'].std():.4f}")
print("\nLas dos estimaciones deberían coincidir aproximadamente, y lo hacen.")
"""))

A(md(r"""
### Importancia de las variables: dos formas, una correcta

Un bosque de 500 árboles ya no se lee como reglas. Para saber qué variables usa hay dos
métodos, y la diferencia entre ellos importa.

**Importancia por impureza** (*Gini importance*, `feature_importances_`): suma, para cada
variable, la reducción de impureza que consiguió en todos los cortes donde se usó. Es
gratis —sale del entrenamiento— pero tiene un **sesgo conocido y serio**: favorece a las
variables con **muchos valores distintos**, porque ofrecen más cortes posibles y por azar
alguno reduce la impureza. Una variable continua o una categórica de alta cardinalidad
aparece como importante incluso si es ruido puro.

**Importancia por permutación**: se mide la caída de performance al **desordenar al azar** una
columna del conjunto de validación. Si la variable importaba, romper su relación con la
etiqueta empeora las predicciones. Es más caro y usa datos no vistos, pero **no tiene el
sesgo de cardinalidad**.

Demostrémoslo agregando una variable de ruido puro con muchos valores distintos.
"""))

A(code(r"""
X_ruido = X.copy()
X_ruido["RUIDO_CONTINUO"] = rng.normal(size=len(X))       # muchos valores distintos
X_ruido["RUIDO_BINARIO"] = rng.integers(0, 2, len(X)).astype(float)   # dos valores

pre_ruido = ColumnTransformer([
    ("num", Pipeline([("i", SimpleImputer(strategy="median")),
                      ("e", StandardScaler())]),
     NUMERICAS + ["RUIDO_CONTINUO", "RUIDO_BINARIO"]),
    ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                      ("o", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                          min_frequency=6, sparse_output=False))]),
     CATEGORICAS),
])
rf_ruido = Pipeline([("pre", pre_ruido),
                     ("clf", RandomForestClassifier(n_estimators=500, random_state=SEED,
                                                    n_jobs=-1))])
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(X_ruido, y_2, test_size=0.35,
                                              random_state=SEED, stratify=y_2)
rf_ruido.fit(Xr_tr, yr_tr)

cols = [n.split("__", 1)[1] for n in rf_ruido[:-1].get_feature_names_out()]
imp_impureza = pd.Series(rf_ruido[-1].feature_importances_, index=cols)

perm = permutation_importance(rf_ruido, Xr_te, yr_te, n_repeats=40,
                              random_state=SEED, scoring="accuracy", n_jobs=-1)
imp_perm = pd.Series(perm.importances_mean, index=X_ruido.columns)

print("Posición de las dos variables de RUIDO en cada ranking:\n")
r_imp = imp_impureza.rank(ascending=False)
print(f"  por impureza   · RUIDO_CONTINUO: puesto {int(r_imp['RUIDO_CONTINUO'])} "
      f"de {len(imp_impureza)}   (importancia {imp_impureza['RUIDO_CONTINUO']:.4f})")
print(f"  por impureza   · RUIDO_BINARIO : puesto {int(r_imp['RUIDO_BINARIO'])} "
      f"de {len(imp_impureza)}   (importancia {imp_impureza['RUIDO_BINARIO']:.4f})")
r_perm = imp_perm.rank(ascending=False)
print(f"  por permutación· RUIDO_CONTINUO: puesto {int(r_perm['RUIDO_CONTINUO'])} "
      f"de {len(imp_perm)}   (importancia {imp_perm['RUIDO_CONTINUO']:+.4f})")
print(f"  por permutación· RUIDO_BINARIO : puesto {int(r_perm['RUIDO_BINARIO'])} "
      f"de {len(imp_perm)}   (importancia {imp_perm['RUIDO_BINARIO']:+.4f})")
"""))

A(code(r"""
fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.4))

top_imp = imp_impureza.sort_values(ascending=False).head(15).sort_values()
colores = [PALETA[2] if "RUIDO" in i else PALETA[3] for i in top_imp.index]
axes[0].barh(top_imp.index, top_imp.values, color=colores)
axes[0].set_xlabel("Reducción de impureza")
axes[0].set_title("Importancia por impureza\nen rojo, las variables de ruido puro",
                  fontsize=11)

top_perm = imp_perm.sort_values(ascending=False).head(15).sort_values()
colores2 = [PALETA[2] if "RUIDO" in i else PALETA[0] for i in top_perm.index]
axes[1].barh(top_perm.index, top_perm.values, color=colores2)
axes[1].axvline(0, color="black", lw=1)
axes[1].set_xlabel("Caída de exactitud al permutar")
axes[1].set_title("Importancia por permutación\nel ruido cae a cero o a valores negativos",
                  fontsize=11)

plt.tight_layout()
plt.show()
"""))

A(md(r"""
El resultado es contundente: la importancia por impureza pone a **`RUIDO_CONTINUO` en el
primer puesto de 55**. Una columna de números aleatorios, sin ninguna relación con la
etiqueta, aparece como la variable más importante del modelo.

Y la comparación con `RUIDO_BINARIO` aísla la causa. Las dos columnas son igual de inútiles
—ruido puro— pero la binaria queda en el puesto 21 y la continua en el 1. La **única**
diferencia entre ellas es la cantidad de valores distintos: la continua ofrece ~90 umbrales
de corte posibles y la binaria uno solo. Con tantos cortes candidatos, alguno reduce la
impureza por azar, y esa reducción se acumula en el ranking.

La importancia por permutación las manda a las dos a valores cercanos a cero: permutar
`RUIDO_CONTINUO` cuesta 0.2 puntos de exactitud, o sea nada. Notar que el *ranking* por
permutación es ruidoso —con un conjunto de validación de 47 casos, muchas variables reales
también quedan cerca de cero— pero la **magnitud** ya no está inflada por la cardinalidad, y
la magnitud es lo que se interpreta.

**Conclusión operativa: no usar `feature_importances_` para sacar conclusiones sustantivas.**
Usar `permutation_importance` sobre datos que el modelo no vio, y mirar la magnitud junto con
su desvío. Es una de las diferencias más consecuentes entre hacer las cosas bien y mal con
ensambles, y el error aparece en trabajos publicados con frecuencia.
"""))

# ------------------------------------------------------------------ boosting
A(md(r"""
## Boosting: árboles en secuencia

Random Forest ajusta árboles **en paralelo** e independientes entre sí. El boosting hace lo
opuesto: ajusta árboles **en secuencia**, y cada uno se especializa en lo que los anteriores
hicieron mal.

La idea, en su forma más simple (*gradient boosting*):

1. Empezar con una predicción constante.
2. Calcular los **residuos**: en qué se equivoca el modelo actual.
3. Ajustar un árbol **chico** —un *weak learner*, típicamente de profundidad 2 a 5— que
   prediga esos residuos.
4. Sumarlo al modelo, multiplicado por una **tasa de aprendizaje** $\eta$ pequeña.
5. Repetir.

$$ F_m(x) = F_{m-1}(x) + \eta \, h_m(x) $$

La diferencia conceptual con el bagging es total. Bagging reduce **varianza** promediando
modelos de bajo sesgo. Boosting reduce **sesgo** sumando modelos de bajo poder individual. Y
por eso el boosting **sí puede sobreajustar** si se lo deja correr: cada árbol nuevo agrega
complejidad.
"""))

A(code(r"""
# La intuición del boosting sobre un problema de regresión de una variable
def f_real(x):
    return np.sin(2.2 * x) + 0.35 * x

x_b = np.sort(rng.uniform(-3, 3, 120))
y_b = f_real(x_b) + rng.normal(scale=0.22, size=120)
malla = np.linspace(-3, 3, 400)

from sklearn.tree import DecisionTreeRegressor

eta = 0.6
prediccion = np.full_like(y_b, y_b.mean())
pred_malla = np.full_like(malla, y_b.mean())

fig, axes = plt.subplots(2, 4, figsize=(17, 7.2))
for it in range(4):
    residuo = y_b - prediccion
    arbolito = DecisionTreeRegressor(max_depth=2, random_state=SEED).fit(x_b[:, None], residuo)

    ax = axes[0, it]
    ax.scatter(x_b, y_b, s=16, color=PALETA[3], alpha=0.55, label="datos")
    ax.plot(malla, f_real(malla), color=PALETA[3], ls="--", lw=1.6, label="f verdadera")
    ax.plot(malla, pred_malla, color=PALETA[2], lw=2.4, label=f"$F_{{{it}}}$")
    ax.set_title(f"Modelo tras {it} árboles", fontsize=10)
    ax.set_ylim(-2.3, 2.3)
    if it == 0:
        ax.legend(fontsize=7.5, loc="lower right")

    ax = axes[1, it]
    ax.scatter(x_b, residuo, s=16, color=PALETA[0], alpha=0.6)
    ax.plot(malla, arbolito.predict(malla[:, None]), color=PALETA[1], lw=2.4)
    ax.axhline(0, color="black", lw=0.9)
    ax.set_title(f"Residuos y el árbol {it + 1} que los ajusta", fontsize=10)
    ax.set_ylim(-1.6, 1.6)
    ax.set_xlabel("$x$")

    prediccion = prediccion + eta * arbolito.predict(x_b[:, None])
    pred_malla = pred_malla + eta * arbolito.predict(malla[:, None])

axes[0, 0].set_ylabel("$y$")
axes[1, 0].set_ylabel("residuo")
fig.suptitle("Boosting: cada árbol chico corrige el residuo del modelo anterior", fontsize=13)
plt.tight_layout()
plt.show()
"""))

A(md(r"""
Arriba, el modelo acumulado va tomando la forma de la función verdadera. Abajo, los residuos
se van achicando y perdiendo estructura: cuando ya no queda patrón en los residuos, no hay
nada más que aprender.

### La familia: AdaBoost, Gradient Boosting, XGBoost

- **AdaBoost** (1995) fue el primero. En lugar de residuos, **repondera las observaciones**:
  las mal clasificadas pesan más en la iteración siguiente.
- **Gradient Boosting** (2001) generalizó la idea: los residuos son el gradiente negativo de
  una función de pérdida, y así el método sirve para cualquier pérdida diferenciable. AdaBoost
  resulta ser el caso particular con pérdida exponencial.
- **XGBoost** (2016) es una implementación de gradient boosting con dos aportes que importan
  acá: usa la **segunda derivada** de la pérdida (no solo el gradiente), y agrega
  **regularización explícita L1 y L2 sobre los valores de las hojas**, más una penalización
  por cantidad de hojas.

Ese último punto conecta directamente con `9.1`: la función objetivo de XGBoost es

$$ \mathcal{L} = \sum_i \ell(y_i, \hat y_i) + \sum_m \Big[ \gamma \, |T_m| + \tfrac{1}{2}\lambda \|w_m\|^2 + \alpha \|w_m\|_1 \Big] $$

con $\lambda$ el parámetro Ridge, $\alpha$ el Lasso y $\gamma$ el precio de cada hoja — que
es la poda por complejidad-costo que vimos más arriba. **Es la misma caja de herramientas.**

<figure style="text-align:center; margin:1.4em 0;">
  <img src="media/xgboost.png" width="620">
  <figcaption style="font-size:0.9em; color:#555;">
    El esquema de un ensamble por boosting: cada árbol recibe un submuestreo de los datos, y el
    <em>residuo</em> de cada uno alimenta al siguiente. La predicción final es una suma ponderada.
  </figcaption>
</figure>

Comparar este esquema con el de Random Forest de más arriba deja ver la diferencia de fondo entre
las dos familias. En el bosque las flechas van **todas en paralelo** desde el caso nuevo hacia
árboles independientes, y se agregan por voto. Acá hay una **flecha horizontal**: el residuo de
un árbol es el insumo del siguiente, y por eso el orden importa y el ensamble no se puede
paralelizar del mismo modo.

De esa cadena se siguen las dos propiedades del boosting: **reduce sesgo** —cada árbol corrige lo
que el anterior no pudo— y **puede sobreajustar**, porque la cadena no tiene un punto de parada
natural. De ahí la necesidad del *early stopping*.

### Los hiperparámetros que importan

| Parámetro | Qué controla | Regla práctica |
|---|---|---|
| `learning_rate` ($\eta$) | cuánto aporta cada árbol | bajo (0.01-0.1) predice mejor y necesita más árboles |
| `n_estimators` | cuántos árboles | se fija alto y se corta con *early stopping* |
| `max_depth` | complejidad de cada árbol | 2 a 6; más profundo capta interacciones y sobreajusta |
| `subsample` | fracción de filas por árbol | < 1 agrega aleatoriedad y reduce varianza |
| `reg_lambda`, `reg_alpha` | penalización de las hojas | subir si sobreajusta |

$\eta$ y `n_estimators` se compensan entre sí: bajar la tasa de aprendizaje a la mitad
requiere aproximadamente el doble de árboles.
"""))

A(code(r"""
# Early stopping: dónde deja de mejorar el ensamble
X_tr2, X_val, y_tr2, y_val = train_test_split(X, y_2, test_size=0.3,
                                              random_state=SEED, stratify=y_2)
pre_fit = preprocesador.fit(X_tr2)
Xtr_m, Xval_m = pre_fit.transform(X_tr2), pre_fit.transform(X_val)
ytr_b = (y_tr2 == "Peronista").astype(int)
yval_b = (y_val == "Peronista").astype(int)

xgb_es = XGBClassifier(n_estimators=600, max_depth=3, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                       eval_metric="logloss", early_stopping_rounds=40,
                       random_state=SEED, n_jobs=-1)
xgb_es.fit(Xtr_m, ytr_b, eval_set=[(Xtr_m, ytr_b), (Xval_m, yval_b)], verbose=False)

hist = xgb_es.evals_result()
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(hist["validation_0"]["logloss"], lw=2, color=PALETA[0], label="entrenamiento")
ax.plot(hist["validation_1"]["logloss"], lw=2, color=PALETA[2], label="validación")
ax.axvline(xgb_es.best_iteration, color=PALETA[1], ls="--", lw=1.8,
           label=f"mejor iteración = {xgb_es.best_iteration}")
ax.set_xlabel("Ronda de boosting")
ax.set_ylabel("Log loss")
ax.set_title("Curva de aprendizaje de XGBoost\n"
             "la validación toca fondo y empieza a subir: ahí arranca el sobreajuste",
             fontsize=11)
ax.legend(fontsize=9.5)
plt.tight_layout()
plt.show()

print(f"árboles solicitados: 600   |   árboles usados: {xgb_es.best_iteration + 1}")
"""))

A(md(r"""
La log loss de entrenamiento baja monótonamente —el ensamble puede seguir memorizando
indefinidamente— y la de validación toca un mínimo y sube. El *early stopping* corta ahí. Es
el mismo gráfico conceptual que la curva de validación del árbol, con la ronda de boosting en
el lugar de la profundidad.

Con 132 casos el mínimo aparece muy temprano: no hay datos para sostener un ensamble grande.
"""))

A(code(r"""
from xgboost import plot_tree as xgb_plot_tree

# Los índices f0, f1... del dibujo son columnas de la matriz preprocesada.
# Les ponemos el nombre real de la variable.
booster = xgb_es.get_booster()
booster.feature_names = list(nombres_limpios)

fig, axes = plt.subplots(2, 1, figsize=(15, 9))
for ax, k in zip(axes, [0, 1]):
    xgb_plot_tree(booster, num_trees=k, ax=ax, rankdir="LR")
    ax.set_title(f"Árbol {k + 1} del ensamble", fontsize=11)
    ax.axis("off")
fig.suptitle(f"Los dos primeros de los {xgb_es.best_iteration + 1} árboles del ensamble",
             fontsize=13)
plt.tight_layout()
plt.show()
"""))

A(md(r"""
Cada árbol tiene dos o tres cortes. Ninguno predice bien por sí solo — de ahí el nombre
*weak learner*. La fuerza está en la suma de cientos de correcciones chicas, que es
exactamente lo contrario de la lógica de Random Forest, donde cada árbol es un predictor
completo.
"""))

# ------------------------------------------------------------------ comparación
A(md(r"""
## La comparación

Todos los modelos bajo la **misma** validación cruzada, contra el mismo baseline, y con el
logit regularizado de `9.2` incluido como referencia.
"""))

A(code(r"""
LOGIT_92 = LogisticRegression(C=0.2081, l1_ratio=0.5, solver="saga",
                              max_iter=20000, random_state=SEED)

modelos = [
    ("BASELINE (clase mayoritaria)", DummyClassifier(strategy="most_frequent")),
    ("Logit regularizado (9.2)", con_arbol(LOGIT_92)),
    ("Árbol prof. 2", con_arbol(DecisionTreeClassifier(max_depth=2, random_state=SEED))),
    ("Árbol podado (ccp óptimo)",
     con_arbol(DecisionTreeClassifier(ccp_alpha=mejor_a, random_state=SEED))),
    ("Árbol sin podar", con_arbol(DecisionTreeClassifier(random_state=SEED))),
    ("AdaBoost (100 tocones)",
     con_arbol(AdaBoostClassifier(
         estimator=DecisionTreeClassifier(max_depth=1, random_state=SEED),
         n_estimators=100, random_state=SEED))),
    ("Gradient Boosting",
     con_arbol(GradientBoostingClassifier(max_depth=3, n_estimators=200,
                                          learning_rate=0.05, random_state=SEED))),
    ("Random Forest (500)",
     con_arbol(RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1))),
]

filas = [evaluar(est, X, y_2, nombre) for nombre, est in modelos]

# XGBoost necesita la etiqueta como 0/1
y_2b = (y_2 == "Peronista").astype(int)
r = cross_validate(con_arbol(XGBClassifier(n_estimators=200, max_depth=3,
                                           learning_rate=0.05, subsample=0.8,
                                           colsample_bytree=0.8, reg_lambda=1.0,
                                           eval_metric="logloss", random_state=SEED,
                                           n_jobs=-1)),
                   X, y_2b, cv=CV, scoring=METRICAS, n_jobs=-1)
filas.append({"modelo": "XGBoost (200)",
              "exactitud": r["test_accuracy"].mean(),
              "exac_sd": r["test_accuracy"].std(),
              "f1_macro": r["test_f1_macro"].mean(),
              "f1_sd": r["test_f1_macro"].std()})

tabla = pd.DataFrame(filas).sort_values("exactitud", ascending=False).reset_index(drop=True)
tabla.round(3)
"""))

A(code(r"""
fig, ax = plt.subplots(figsize=(10.5, 6))
t = tabla.sort_values("exactitud")
colores = []
for m in t["modelo"]:
    if m.startswith("BASELINE"):
        colores.append(PALETA[3])
    elif "Logit" in m:
        colores.append(PALETA[1])
    elif "Árbol" in m:
        colores.append(PALETA[2])
    else:
        colores.append(PALETA[0])

ax.barh(t["modelo"], t["exactitud"], xerr=t["exac_sd"], capsize=4,
        color=colores, alpha=0.9)
ax.axvline(tabla.loc[tabla["modelo"].str.startswith("BASELINE"), "exactitud"].iloc[0],
           color=PALETA[3], ls="--", lw=1.8)
ax.set_xlim(0.4, 0.95)
ax.set_xlabel("Exactitud en validación cruzada (± desvío entre pliegues)")
ax.set_title("Amarillo: modelo lineal · Rojo: árbol único · Azul: ensambles\n"
             "línea punteada: baseline", fontsize=11)
for i, (v, s) in enumerate(zip(t["exactitud"], t["exac_sd"])):
    ax.text(v + s + 0.008, i, f"{v:.3f}", va="center", fontsize=9)
plt.tight_layout()
plt.show()
"""))

A(md(r"""
### Cómo se lee esta tabla

El resultado es claro y va en contra de la expectativa habitual:

1. **Random Forest y el logit regularizado empatan.** 0.792 contra 0.791, con desvíos de
   0.076 y 0.069. La diferencia es de una milésima sobre un desvío setenta veces mayor: son
   indistinguibles.
2. **Los árboles individuales son muy inferiores**: entre 0.64 y 0.66, o sea unos 13 puntos
   por debajo. Ni podados alcanzan.
3. **XGBoost queda por debajo de Random Forest** (0.754 contra 0.792). Con 132 casos, un
   método que suma cientos de correcciones secuenciales tiene demasiada capacidad para tan
   pocos datos, y el *early stopping* corta antes de aprender lo suficiente.
4. Todos los ensambles le ganan holgadamente al baseline de 0.545.

**Conclusión: la flexibilidad adicional no compra nada acá.** El bosque necesita 500 árboles
para igualar lo que hace una frontera lineal con 56 coeficientes penalizados. Y eso es
informativo por sí mismo: dice que la relación entre estas opiniones y la identificación
política es **aproximadamente lineal en el logit**, sin interacciones fuertes que el modelo
lineal esté perdiendo.

Cuando un ensamble no le gana a un modelo lineal, la conclusión no es "el ensamble falló".
Es un **hallazgo sobre la estructura de los datos**.
"""))

A(code(r"""
# Lo mismo con 7 y con 3 clases, para ver si el veredicto cambia
comparacion_multi = []
for nombre_esq, yy in [("7 clases", y_7), ("3 clases", y_3), ("2 clases", y_2)]:
    for nombre_mod, est in [
            ("Baseline", DummyClassifier(strategy="most_frequent")),
            ("Logit reg.", con_arbol(LOGIT_92)),
            ("Árbol prof. 3", con_arbol(DecisionTreeClassifier(max_depth=3,
                                                               random_state=SEED))),
            ("Random Forest", con_arbol(RandomForestClassifier(n_estimators=500,
                                                               random_state=SEED,
                                                               n_jobs=-1)))]:
        f = evaluar(est, X, yy, nombre_mod)
        f["esquema"] = nombre_esq
        comparacion_multi.append(f)

pivote = pd.DataFrame(comparacion_multi).pivot(index="modelo", columns="esquema",
                                               values="exactitud")
pivote = pivote.loc[["Baseline", "Árbol prof. 3", "Logit reg.", "Random Forest"],
                    ["7 clases", "3 clases", "2 clases"]]
pivote.round(3)
"""))

A(md(r"""
El patrón se repite en los tres esquemas, y con 7 clases pasa algo que vale detenerse a
mirar: **el árbol de profundidad 3 queda por debajo del baseline** (0.397 contra 0.455).

Eso no es un error. El baseline predice siempre la clase mayoritaria y por eso acierta el
45.5% garantizado. El árbol intenta discriminar, hace cortes basados en una o dos
observaciones de las clases chicas, y se equivoca en casos que el baseline habría acertado
por default. Con 7 clases y 132 casos, **intentar discriminar es peor que no intentarlo**.

Random Forest no cae en eso porque promedia: al votar entre 500 árboles, las decisiones
basadas en un solo caso se diluyen.
"""))

# ------------------------------------------------------------------ por qué no ganan
A(md(r"""
## ¿Es el tamaño de la muestra?

La hipótesis natural es que los ensambles necesitan más datos. Se puede testear con una
**curva de aprendizaje**: reajustar cada modelo con fracciones crecientes de la muestra y ver
cómo evoluciona.
"""))

A(code(r"""
fracciones = np.linspace(0.25, 1.0, 8)
fig, ax = plt.subplots(figsize=(9.5, 5.4))

for nombre, est, color in [
        ("Logit regularizado", con_arbol(LOGIT_92), PALETA[1]),
        ("Random Forest", con_arbol(RandomForestClassifier(n_estimators=300,
                                                           random_state=SEED, n_jobs=-1)),
         PALETA[0]),
        ("Árbol prof. 3", con_arbol(DecisionTreeClassifier(max_depth=3,
                                                           random_state=SEED)), PALETA[2])]:
    tam, tr_s, te_s = learning_curve(
        est, X, y_2, train_sizes=fracciones,
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        scoring="accuracy", n_jobs=-1)
    ax.plot(tam, te_s.mean(axis=1), "o-", lw=2.2, color=color, label=nombre)
    ax.fill_between(tam, te_s.mean(axis=1) - te_s.std(axis=1),
                    te_s.mean(axis=1) + te_s.std(axis=1), alpha=0.13, color=color)

ax.axhline(y_2.value_counts(normalize=True).iloc[0], color=PALETA[3], ls=":", lw=1.8,
           label="baseline")
ax.set_xlabel("Cantidad de casos de entrenamiento")
ax.set_ylabel("Exactitud en validación")
ax.set_title("Curva de aprendizaje: ¿alguna curva sigue subiendo al llegar al final?",
             fontsize=11)
ax.legend(fontsize=9.5)
plt.tight_layout()
plt.show()
"""))

A(md(r"""
Las dos curvas útiles —logit y bosque— crecen con fuerza hasta unos 40 casos y después
**casi se aplanan**. Entre 77 y 105 casos de entrenamiento el bosque gana alrededor de un
punto de exactitud en total: la pendiente es de aproximadamente 0.1 punto por cada 10 casos
adicionales.

La del árbol único está plana y baja desde el principio: su problema no es la cantidad de
datos sino la varianza del método.

Y algo que la curva **no** permite afirmar es que el bosque eventualmente le ganaría al logit:
las dos suben con pendiente parecida y sus bandas se superponen en todo el recorrido.

Cuantifiquemos esa pendiente, porque de ahí sale la recomendación de diseño.
"""))

# ------------------------------------------------------------------ interpretación
A(md(r"""
## Interpretar un ensamble

Un bosque de 500 árboles no se lee. Hay tres herramientas para abrirlo, de menor a mayor
detalle.
"""))

A(code(r"""
rf_final = con_arbol(RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1))
X_tr3, X_te3, y_tr3, y_te3 = train_test_split(X, y_2, test_size=0.35,
                                              random_state=SEED, stratify=y_2)
rf_final.fit(X_tr3, y_tr3)

perm_final = permutation_importance(rf_final, X_te3, y_te3, n_repeats=60,
                                    random_state=SEED, scoring="accuracy", n_jobs=-1)
imp_final = (pd.Series(perm_final.importances_mean, index=X.columns)
             .sort_values(ascending=False))
sd_final = pd.Series(perm_final.importances_std, index=X.columns)

top = imp_final.head(12).sort_values()
fig, ax = plt.subplots(figsize=(9, 5.6))
ax.barh(top.index, top.values, xerr=sd_final[top.index], capsize=3,
        color=[PALETA[0] if v > 0 else PALETA[3] for v in top.values])
ax.axvline(0, color="black", lw=1)
ax.set_xlabel("Caída de exactitud al permutar la variable")
ax.set_title("Importancia por permutación · Random Forest\n"
             "las barras de error son el desvío entre las 60 permutaciones", fontsize=11)
plt.tight_layout()
plt.show()
"""))

A(code(r"""
# Dependencia parcial: qué forma tiene el efecto de las variables top.
# Se limita a las numéricas: para categóricas de texto habría que pasar
# categorical_features, y la lectura de la curva es otra.
vars_pdp = [c for c in imp_final.index if c in NUMERICAS][:4]
print("Variables numéricas con mayor importancia por permutación:", vars_pdp)

fig, ax = plt.subplots(1, len(vars_pdp), figsize=(4.1 * len(vars_pdp), 4))
PartialDependenceDisplay.from_estimator(
    rf_final, X_tr3, vars_pdp, ax=ax, kind="average",
    line_kw={"color": PALETA[0], "lw": 2.4})
for a in np.atleast_1d(ax):
    a.set_ylabel("")
np.atleast_1d(ax)[0].set_ylabel("P(peronista) promedio")
fig.suptitle("Dependencia parcial: el efecto marginal según el bosque", fontsize=12)
plt.tight_layout()
plt.show()
"""))

A(md(r"""
La dependencia parcial muestra cómo cambia la predicción promedio al mover una variable,
manteniendo las demás como están. Su virtud es que **no supone linealidad**: si el bosque
detectó un efecto en forma de escalón o no monótono, acá se ve.

Que las curvas salgan aproximadamente monótonas es coherente con el resultado de la
comparación: si hubiera efectos con forma de U, el modelo lineal no habría podido igualar al
bosque.
"""))

A(code(r"""
# SHAP: descomponer UNA predicción individual
try:
    import shap

    modelo_arbol = rf_final[-1]
    X_te_mat = rf_final[:-1].transform(X_te3)
    cols_shap = [n.split("__", 1)[1] for n in rf_final[:-1].get_feature_names_out()]

    explicador = shap.TreeExplainer(modelo_arbol)
    valores = explicador.shap_values(X_te_mat)
    v = valores[..., 1] if np.asarray(valores).ndim == 3 else valores

    caso = 0
    aporte = pd.Series(np.asarray(v)[caso], index=cols_shap)
    aporte = aporte[aporte.abs() > 1e-9].sort_values(key=abs, ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(9, 5))
    colores = [PALETA[0] if x > 0 else PALETA[2] for x in aporte.values]
    ax.barh(aporte.index[::-1], aporte.values[::-1], color=colores[::-1])
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Aporte a la probabilidad de 'Peronista'")
    ax.set_title(f"Por qué el modelo clasificó así al caso {caso}\n"
                 f"real: {y_te3.iloc[caso]}  ·  "
                 f"predicho: {rf_final.predict(X_te3.iloc[[caso]])[0]}", fontsize=11)
    plt.tight_layout()
    plt.show()
except ImportError:
    print("shap no está instalado en este entorno; esta celda es opcional.")
    print("Para instalarlo:  pip install shap")
"""))

A(md(r"""
SHAP responde una pregunta distinta de la importancia por permutación: no *qué variables
importan en general* sino **por qué este caso concreto recibió esta predicción**. Reparte la
diferencia entre la predicción del caso y la predicción promedio entre las variables, con una
propiedad de la teoría de juegos cooperativos (los valores de Shapley) que garantiza que los
aportes suman exactamente esa diferencia.

Es la herramienta indicada cuando hay que justificar una decisión individual —por qué a esta
persona el sistema la clasificó así— y por eso se volvió estándar en contextos donde las
predicciones tienen consecuencias.
"""))

# ------------------------------------------------------------------ cierre aplicado
A(md(r"""
## Cierre aplicado: cuántos casos harían falta

El producto de este notebook no es un clasificador —para eso ya teníamos el logit de `9.2`,
que empata con el bosque y es más simple e interpretable—. El producto es una **decisión de
diseño de investigación**: ¿vale la pena relevar más casos, y cuántos?

Las curvas de aprendizaje suelen seguir aproximadamente una ley de potencias:

$$ \text{error}(n) \approx a \cdot n^{-b} + c $$

donde $c$ es el **error asintótico**: el piso que no baja por más datos que se agreguen,
porque depende de qué variables se midieron y de cuánta aleatoriedad irreducible tiene el
fenómeno.

Ajustamos esa curva de dos maneras, y la comparación entre las dos es la parte honesta del
ejercicio:

- **Con $c$ libre**, dejando que los datos digan dónde está el techo.
- **Con $c = 0$ forzado**, que supone que con suficientes datos se llegaría a error cero.

El segundo supuesto es casi siempre falso en ciencias sociales, pero es el que está implícito
cuando alguien dice "con más muestra va a andar mejor" sin aclarar nada más.
"""))

A(code(r"""
from scipy.optimize import curve_fit

# Más repeticiones que en la curva anterior: el ajuste es sensible al ruido
tam_obs, _, te_obs = learning_curve(
    con_arbol(RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)),
    X, y_2, train_sizes=np.linspace(0.2, 1.0, 10),
    cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED),
    scoring="accuracy", n_jobs=-1)

error_obs = 1 - te_obs.mean(axis=1)
print("casos de entrenamiento:", tam_obs)
print("exactitud observada   :", np.round(te_obs.mean(axis=1), 4))


def ley_con_techo(n, a, b, c):
    return a * np.power(n, -b) + c


def ley_sin_techo(n, a, b):
    return a * np.power(n, -b)


def r2_ajuste(observado, predicho):
    return 1 - ((observado - predicho) ** 2).sum() / ((observado - observado.mean()) ** 2).sum()


par_techo, _ = curve_fit(ley_con_techo, tam_obs, error_obs, p0=[1.0, 0.5, 0.15],
                         maxfev=200000, bounds=([0, 0.01, 0.0], [1e6, 5, 0.5]))
par_sin, _ = curve_fit(ley_sin_techo, tam_obs, error_obs, p0=[1.0, 0.3], maxfev=200000)

a, b, c = par_techo
r2_con = r2_ajuste(error_obs, ley_con_techo(tam_obs, *par_techo))
r2_sin = r2_ajuste(error_obs, ley_sin_techo(tam_obs, *par_sin))

print(f"\ncon techo libre: error(n) = {a:.2f}·n^(-{b:.3f}) + {c:.3f}"
      f"    R² del ajuste = {r2_con:.3f}")
print(f"                 techo de exactitud estimado = {1 - c:.3f}")
print(f"con techo en 0 : error(n) = {par_sin[0]:.3f}·n^(-{par_sin[1]:.3f})"
      f"           R² del ajuste = {r2_sin:.3f}")
"""))

A(code(r"""
print(f"Exactitud hoy, con {int(tam_obs[-1])} casos de entrenamiento: "
      f"{te_obs.mean(axis=1)[-1]:.3f}\n")
print("Casos de entrenamiento necesarios para cada objetivo:\n")
print(f"  {'objetivo':>8s}   {'con techo libre':>17s}   {'suponiendo techo 0':>19s}")
for obj in [0.80, 0.82, 0.85, 0.88]:
    err_obj = 1 - obj
    con = ("inalcanzable" if err_obj <= c
           else f"{int(np.ceil((a / (err_obj - c)) ** (1 / b)))} casos")
    sin = f"{int(np.ceil((par_sin[0] / err_obj) ** (1 / par_sin[1])))} casos"
    print(f"  {obj:>8.2f}   {con:>17s}   {sin:>19s}")
"""))

A(code(r"""
fig, ax = plt.subplots(figsize=(10.5, 5.8))
n_proy = np.linspace(tam_obs.min(), 700, 500)

ax.errorbar(tam_obs, 1 - error_obs, yerr=te_obs.std(axis=1), fmt="o", ms=8,
            color=PALETA[0], ecolor=PALETA[3], elinewidth=1, capsize=3, zorder=5,
            label="observado (± desvío entre pliegues)")
ax.plot(n_proy, 1 - ley_con_techo(n_proy, *par_techo), "-", lw=2.4, color=PALETA[0],
        label=f"con techo libre  (techo = {1-c:.3f}, $R^2$ = {r2_con:.3f})")
ax.plot(n_proy, 1 - ley_sin_techo(n_proy, *par_sin), "--", lw=2.2, color=PALETA[2],
        label=f"suponiendo techo en 1.0  ($R^2$ = {r2_sin:.3f})")
ax.axhline(1 - c, color=PALETA[4], ls="-.", lw=1.5)
ax.axhline(y_2.value_counts(normalize=True).iloc[0], color=PALETA[3], ls=":", lw=1.8,
           label="baseline")
ax.axvline(tam_obs[-1], color=PALETA[1], ls="--", lw=1.6,
           label=f"muestra actual ({int(tam_obs[-1])} de entrenamiento)")

ax.set_xlabel("Casos de entrenamiento")
ax.set_ylabel("Exactitud")
ax.set_ylim(0.5, 1.0)
ax.set_title("Dos extrapolaciones de la misma curva, con recomendaciones opuestas",
             fontsize=11.5)
ax.legend(fontsize=8.5, loc="lower right")
plt.tight_layout()
plt.show()
"""))

A(md(r"""
### Las dos extrapolaciones no dicen lo mismo

Y ahí está la lección sobre extrapolar:

- **Con el techo libre** el ajuste describe muy bien lo observado ($R^2 \approx 0.97$) y ubica
  el límite asintótico alrededor de **0.80**. Como hoy estamos en 0.790, la conclusión es que
  **ya estamos prácticamente en el techo**: duplicar la muestra ganaría menos de un punto.
- **Suponiendo que el error puede llegar a cero** el ajuste es peor ($R^2 \approx 0.90$) y
  proyecta que con unos 240 casos se alcanzaría 0.85 y con unos 470 se llegaría a 0.88.

Los dos ajustes describen razonablemente los mismos diez puntos observados y **recomiendan
decisiones opuestas**: uno dice "no gastes en más encuestas", el otro dice "triplicá la
muestra". La única forma de distinguirlos sería tener datos más allá del rango observado, que
es exactamente lo que no tenemos. Extrapolar una curva de aprendizaje es siempre así.

El ajuste con techo libre es preferible por dos razones: describe mejor los datos, y su
supuesto es el razonable —ningún modelo predice la identificación política sin error, porque
la identificación política no es una función determinística de trece opiniones—. Pero la
distancia entre las dos curvas es la medida honesta de cuánto **no** sabemos.

### El producto

Lo que se lleva quien tiene que decidir sobre el próximo relevamiento:

1. **Con 132 casos el techo ya está cerca.** El ajuste que mejor describe la curva estima un
   límite asintótico de exactitud de **0.80**, y la muestra actual ya rinde **0.790**.
   Duplicar el tamaño ganaría menos de un punto. La recomendación es **no** invertir en más
   casos del mismo tipo.
2. **El cuello de botella no es el método, son las variables.** Tres familias de modelos con
   supuestos completamente distintos —lineal penalizado, bosque, boosting— convergen al mismo
   número. Cuando eso pasa, agregar métodos no sirve: hay que agregar **información**. Para
   mejorar de verdad hay que preguntar otras cosas, no encuestar a más gente sobre lo mismo.
3. **Para el modelo de producción, usar el logit.** Empata con el bosque, se explica con odds
   ratios, corre instantáneamente y se puede auditar. Un Random Forest de 500 árboles que
   iguala a un modelo lineal es una complejidad que no se paga.
4. **Y una recomendación concreta de diseño**: si el objetivo es distinguir las siete
   etiquetas y no solo peronismo/resto, el problema no es el modelo sino el muestreo. Habría
   que **sobremuestrear deliberadamente** las etiquetas chicas —radicales, derecha,
   liberales— para tener al menos 30 casos de cada una, en lugar de aumentar el tamaño total
   de forma proporcional.
"""))

A(md(r"""
## Síntesis

1. Un árbol parte el espacio con cortes paralelos a los ejes y predice una constante en cada
   región. Elige cada corte maximizando la reducción de impureza (**Gini** o **entropía**;
   son casi equivalentes). Captura **interacciones sin especificarlas** y es insensible a la
   escala de las variables.

2. Su gran virtud es la legibilidad: el modelo **es** un conjunto de reglas. `TARIFAS`
   aparece como el mejor primer corte, el mismo predictor que dominaba en `9.2` con
   maquinaria completamente distinta.

3. Se sobreajusta hasta memorizar. Se controla con **poda por complejidad-costo**, que es la
   misma idea de `9.1` aplicada a la cantidad de hojas en lugar de a la magnitud de los
   coeficientes.

4. **El problema serio de los árboles es la inestabilidad**: en seis remuestreos de los
   mismos datos cambia hasta la variable de la raíz. Por eso la estructura de un árbol
   particular no constituye un hallazgo.

5. **Random Forest** promedia árboles bootstrap y además submuestrea variables en cada corte
   para decorrelacionarlos. `n_estimators` no sobreajusta. El **error out-of-bag** da una
   estimación gratis del error de generalización.

6. **No usar `feature_importances_`** para conclusiones sustantivas: la importancia por
   impureza infla las variables de alta cardinalidad, y en nuestra demostración puso una
   columna de ruido continuo por encima de la mitad del ranking. Usar
   `permutation_importance` sobre datos no vistos.

7. **Boosting** suma árboles chicos en secuencia, cada uno corrigiendo el residuo del
   anterior. Reduce sesgo (no varianza), **sí sobreajusta**, y se controla con
   `learning_rate` y *early stopping*. La función objetivo de XGBoost contiene explícitamente
   las penalizaciones L1 y L2 de `9.1`.

8. **El veredicto sobre estos datos: los ensambles no le ganan al modelo lineal.** Random
   Forest 0.792 contra logit 0.791; XGBoost queda atrás en 0.754; los árboles individuales
   13 puntos abajo. Con 7 clases, un árbol único queda **por debajo del baseline**.

9. Ese empate es un **hallazgo**, no un fracaso: dice que la relación es aproximadamente
   lineal en el logit. Y la curva de aprendizaje dice que el límite está en las variables
   disponibles, no en la cantidad de casos.

## Lo que sigue

Termina el bloque supervisado. En `9.4` desaparece la etiqueta: pasamos al **aprendizaje no
supervisado**, donde no hay respuesta correcta contra la que medirse y el criterio del
investigador vuelve al centro.

> **Variante de este notebook.** `9.3b.arboles-mesas.ipynb` recorre la misma teoría sobre
> 7.233 mesas de la elección porteña de 2025. Ahí el veredicto se invierte: con esa cantidad
> de casos, XGBoost le saca seis puntos al árbol único y la progresión árbol → bosque →
> boosting se ve con nitidez. Vale leer los dos: **la respuesta a "¿conviene un ensamble?"
> depende del tamaño de la muestra, y tener los dos casos al lado lo demuestra.**
"""))

write_nb(C, OUT)
