<p align="center">
  <img src="image.jpeg" alt="Logo MET4OP" width="200"/>
</p>

<h1 align="center">Metodología de Análisis en Opinión Pública</h1>

<p align="center">
  Materia electiva · Ciencia Política · Universidad de Buenos Aires
</p>

---

## Descripción

Repositorio oficial de la materia **Metodología de Análisis en Opinión Pública (MET4OP)**. El cursado apunta a que los estudiantes incorporen herramientas estadísticas y computacionales para resolver problemas concretos del análisis de la opinión pública: desde la manipulación y visualización de datos hasta la corrección de sesgos en encuestas y el perfilado electoral.

La materia combina un fuerte contenido teórico-estadístico con práctica orientada a la implementación en **Python**, articulando los contenidos propios de la orientación de opinión pública con las demandas empíricas del analista y/o investigador.

El dictado se organiza en **dos módulos paralelos**: `PROGRAMACION` (9 unidades) y `ESTADISTICA` (10 unidades). Cada unidad vive en su propia carpeta, con el notebook de clase, la carpeta `media/` de imágenes y una guía `Ejercicios.docx`.

---

## Organización del repositorio

```
met4op-2026/
├── ESTADISTICA/
│   ├── 0.Estadistica descriptiva/            0.clase.ipynb
│   ├── 1.Probabilidad/                       1.clase.ipynb · 1bis.Combinatoria.ipynb
│   ├── 2.Distribucíon Normal -IC/            2.clase.ipynb
│   ├── 3.Test de hipotesis/                  3.clase.ipynb
│   ├── 4.Regresión lineal simple/            4.clase.ipynb
│   ├── 5.Regresión lineal multiple/          5.clase.ipynb
│   ├── 6.Regresión logistica/                6.clase.ipynb
│   ├── 7.Introduccíon al Muestreo y Ponderacíon/   7.clase.ipynb
│   ├── 8.Causalidad/                         8.clase.ipynb
│   └── 9.Machine Learning/                   9.0.intro-ml.ipynb · 9.1.regularizacion.ipynb
│                                             9.2.logistica.ipynb · 9.3.arboles.ipynb
│                                             9.3b.arboles-encuesta.ipynb · 9.4.clustering.ipynb
│                                             9.5.reglas-asociacion.ipynb
├── PROGRAMACION/
│   ├── 0.Entornos_github/                    0_clase_git.ipynb · 1_clase_entornos.ipynb
│   ├── 1.Variables, tipos de datos/          01_clase_datatypes.ipynb
│   ├── 2.Estructuras de control/             02_clase_estructuras.ipynb
│   ├── 3.Numpy -Pandas/                      clase_03.ipynb
│   ├── 4.Funciones/                          04_clase_funciones.ipynb
│   ├── 5.Geopandas - GIS/                    05_clase_geopandas.ipynb
│   ├── 6.Bases de datos/                     06_clase_bases-de-datos.ipynb
│   ├── 7.Visualizacíon/                      07_Clase_Visualizacion.ipynb · 07bis_prompting.ipynb
│   └── 8.POO/                                08_clase_POO.ipynb
└── dataset/
    ├── QOG/
    ├── barrios_caba/
    ├── censo2010/
    ├── circuitos-electorales/
    ├── elecciones_2019/
    ├── elecciones_caba_2025/
    ├── encuesta134/
    ├── votaciones_sim/
    ├── extras/
    └── otros/
```

---

## Programa

### Módulo `PROGRAMACION`

| # | Unidad                           | Contenidos                                                                                                       |
| - | -------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| 0 | Entornos y GitHub                | Git: `init`, `commit`, ramas, remotos, `.gitignore` · entornos virtuales                                         |
| 1 | Variables y tipos de datos       | Referencias de objetos, números, strings y f-strings, listas, tuplas, diccionarios, operadores                   |
| 2 | Estructuras de control           | Condicionales (`if`/`elif`/`else`), bucles `for` y `while`                                                       |
| 3 | NumPy y Pandas                   | Series y DataFrames, importación, ordenamiento, `groupby`, `having`, `crosstab`, `unstack`, joins                |
| 4 | Funciones                        | Definición, recursividad, orden superior, `lambda`, `map`/`filter`, `apply`, `groupby`, `pipe`                   |
| 5 | GeoPandas y GIS                  | Formatos vectoriales, CRS y Web Mercator, mapas temáticos, `dissolve`, problema MAUP                             |
| 6 | Bases de datos                   | SQL (DDL, CRUD, JOINs, subconsultas), modelado, ORM con SQLAlchemy, NoSQL, transacciones ACID, backup            |
| 7 | Visualización                    | Matplotlib y Seaborn: `relplot`, `catplot`, `boxplot`, `histplot`, `kdeplot`, grillas · *bis*: prompting con LLM |
| 8 | Programación orientada a objetos | Clases, métodos de instancia, herencia, polimorfismo, `isinstance`, integración con Pandas                       |

### Módulo `ESTADISTICA`

| # | Unidad                                 | Contenidos                                                                                                                                                                                   |
| - | -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0 | Estadística descriptiva                | Unidad estadística y variables, distribuciones de frecuencias, medidas de resumen, de posición y de variabilidad, análisis bivariado                                                         |
| 1 | Probabilidad                           | Ley de los grandes números, teoría de conjuntos, definición de Laplace, probabilidad condicional, probabilidad total, Bayes · *bis*: combinatoria                                            |
| 2 | Distribución normal e IC               | Descriptiva vs. inferencial, parámetros y estimadores, distribución muestral, estandarización z y t, α, nivel de confianza e intervalos                                                      |
| 3 | Test de hipótesis                      | Formulación de H₀/H₁, pruebas de una y dos colas, errores tipo I y II, t para una media, t para medias independientes (pooled y Welch) y apareadas, no paramétricos (Mann-Whitney, Wilcoxon) |
| 4 | Regresión lineal simple                | Asociación y correlación, MCO/OLS, lectura e inferencia del modelo, variable explicativa binaria, supuestos                                                                                  |
| 5 | Regresión lineal múltiple              | Variables omitidas, ecuación general, matriz de correlación, de la simple a la múltiple, supuestos                                                                                           |
| 6 | Regresión logística                    | Modelo logit, odds y odds ratios, interpretación de coeficientes                                                                                                                             |
| 7 | Introducción al muestreo y ponderación | Muestreo probabilístico, diseños polietápicos (UPE), raking / *iterative proportional fitting*, librería `balance`                                                                           |
| 8 | Causalidad                             | Inferencia causal, contrafácticos, identificación                                                                                                                                            |
| 9 | Machine Learning                       | Supervisado / no supervisado / por refuerzo, sobreajuste y sesgo-varianza, validación cruzada, baseline, métricas y fugas de información · regularización (Ridge, Lasso, Elastic Net) · logística como clasificador · árbol de decisión, Random Forest y XGBoost · clustering jerárquico, k-means y PCA · reglas de asociación (a priori) |

---

## Datasets

| Carpeta                  | Contenido                                                                             | Se usa en             |
| ------------------------ | ------------------------------------------------------------------------------------- | --------------------- |
| `QOG/`                   | Quality of Government (corte transversal) y V-Dem 2026                                | EST 0, EST 4, EST 9   |
| `censo2010/`             | CPV 2010: persona, hogar, vivienda, radio, fracción, depto, provincia · labels y docs | PROG 3, PROG 5, EST 7 |
| `circuitos-electorales/` | Shapefile de circuitos electorales                                                    | PROG 5                |
| `barrios_caba/`          | GeoJSON de barrios de CABA                                                            | PROG 5, PROG 7, EST 9 |
| `elecciones_2019/`       | Shapefile de CABA, resultados en CSV y tabla `rosetta`                                | PROG 4, PROG 7        |
| `elecciones_caba_2025/`  | Resultados CABA 2025 por circuito y por mesa                                          | PROG 7, EST 9         |
| `encuesta134/`           | Encuesta propia de identificación política, 134 casos · versión etiquetada y dummies  | EST 9                 |
| `votaciones_sim/`        | Votaciones nominales simuladas de la Cámara de Diputados (257 × 15) · datos ficticios | EST 9                 |
| `extras/`, `otros/`      | Datos auxiliares de ejercicios                                                        | varias                |

Algunas unidades traen además su propia carpeta `data/` con el archivo puntual del ejemplo (por ejemplo `Salary_Data.csv` en EST 4 y `ejemplo_ponderacion.csv` en EST 7).

> **Sobre `votaciones_sim/`**: los diputados y las votaciones son simulados con estructura
> realista (siete bloques, dos dimensiones latentes, disciplina partidaria diferencial) y semilla
> fija. El generador está documentado en `9.5.reglas-asociacion.ipynb`. Ninguna conclusión sobre
> esos datos dice nada sobre la política argentina real.

---

## Período de Cursada

- **Inicio:** Segundo cuatrimestre 2026
- **Carga horaria:** 96 horas-reloj · 2 clases semanales de 3 horas cada una

## Stack

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Arrays-013243?logo=numpy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
![GeoPandas](https://img.shields.io/badge/GeoPandas-GIS-1E8449?logo=qgis&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Seaborn-11557c?logo=python&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-stats-8CAAE6?logo=scipy&logoColor=white)
![statsmodels](https://img.shields.io/badge/statsmodels-OLS%20%2F%20GLM-4B8BBE)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-ORM-D71F00?logo=sqlalchemy&logoColor=white)
![balance](https://img.shields.io/badge/balance-raking-6E5494)

---

## Equipo Docente

| Nombre                   | Cargo                      |
| ------------------------ | -------------------------- |
| Tomás Alberto Olego      | Profesor Titular           |
| Juan Stillo              | Jefe de Trabajos Prácticos |
| Manuel Miller            | Ayudante                   |
| Valentina González Sixto | Ayudante                   |

---

## Institución

Facultad de Ciencias Sociales · Universidad de Buenos Aires

---

<p align="center"><sub>MET4OP · UBA · 2026</sub></p>
