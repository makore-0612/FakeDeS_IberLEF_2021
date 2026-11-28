# Ensamble: XGBoost y stacking (sistema aparte)

Notebook: `ensamble_stacking.ipynb` (≈ 5 min con 2 núcleos). Fecha: 26 de noviembre de 2026. Paso 4 de `notas/siguientes_pasos.md` (sin BETO). Teoría: Alegre Ventura (2026), *Métodos de ensamble*, láminas 4, 14–24. El test se cargó **una sola vez**, al final. Paquete nuevo: `xgboost==3.4.1` (agregado a `environment.yml`).

## Qué se hizo
- **XGBoost** con `GridSearchCV` sobre los 5 folds agrupados (`PredefinedSplit` con `fold_grupo`): 3 representaciones (estilo, palabras, palabras + estilo) × profundidad {2, 4} × learning rate {0.05, 0.1} × árboles {200, 500} = 24 combinaciones. Regla de un SE con preferencia por la simplicidad.
- **Stacking**: bases LR texto (lineal), XGBoost (boosting) y RF estilo (bagging); predicciones out-of-fold; meta-modelo LR con `GridSearchCV` de `C` en {0.01, 0.1, 1, 10, 100}. Referencia: soft voting.
- **CV anidada** para el stacking: en cada fold externo, CV interna de los modelos base, grid search de `C` y entrenamiento del meta-modelo sin ver el fold.

## Resultados (F1 de fake)
| Sistema | CV (anidada para el stacking) | Test | IC95 test |
|---|---|---|---|
| LR texto (principal) | 0.730 | 0.676 | [0.629, 0.722] |
| LR texto, umbral 0.34 | 0.744 | **0.735** | [0.696, 0.771] |
| **Stacking** | 0.684 | 0.696 | [0.652, 0.741] |
| Soft voting | 0.698 | 0.645 | [0.595, 0.693] |
| XGBoost (palabras, prof. 2, lr 0.05, 200 árboles) | 0.677 | 0.642 | [0.592, 0.691] |
| RF estilo | 0.608 | 0.505 | [0.446, 0.560] |

## Hallazgos
- **El ensamble no mitiga los errores.** En la CV anidada, el stacking pierde contra la LR sola en 3 de 5 folds (−0.046 ± 0.139).
- **Bases parecidas y desiguales:** LR–XGBoost Spearman 0.80; 125 noticias (13 %) las fallan los tres; los tres comparten el sesgo de longitud (XGBoost y RF detectan el 3 % de las falsas más largas del test).
- **Pesos del meta-modelo:** LR 68 %, XGBoost 22 %, RF negativo; inestables entre folds (C de 0.01 a 100; el peso de XGBoost cambia de signo).
- **Test:** stacking − principal = +0.020, IC95 pareado [−0.009, +0.049], McNemar p = 0.665. Stacking − umbral 0.34 = −0.039 [−0.075, −0.004].
- **El stacking es casi la LR con otro umbral:** Spearman 0.984 con la LR. La LR sola, con el umbral que marca las mismas 237 falsas (0.444), saca 0.711, más que el stacking (0.696).

## Para el reporte
Es un resultado negativo bien fundamentado: ilustra las condiciones de las láminas 4 y 21 (bases precisas y con errores distintos) y la regla de la lámina 24 (agregar complejidad solo si mejora la validación). Para un ensamble útil haría falta una base que lea la noticia de otra forma (BETO, paso 3).

## Limitaciones
- Hiperparámetros de XGBoost elegidos sin anidar (igual que los de la LR y el bosque).
- XGBoost sin n-gramas de caracteres, por tiempo de cómputo.
- Sin ajuste de umbral del meta-modelo.

## Salidas
`resultados/xgb_grid.csv`, `resultados/stacking_cv.csv`, `resultados/stacking_test.csv`, `resultados/predicciones_test_stacking.csv`, `resultados/seleccion_stacking.json`, figura `6_1_stacking_f1_test.png`.
