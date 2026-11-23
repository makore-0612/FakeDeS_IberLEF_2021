# Modelado: modelos lineales (CRISP-DM, fase 4)

Notebook: `modelos_lineales.ipynb` (≈ 18 min de ejecución en 4 núcleos). Fecha: 22 de noviembre de 2026. **No se cargó el test.**

## Qué se hizo
- Baselines: clase mayoritaria, aleatoria estratificada y réplicas aproximadas de los baselines oficiales (BoW + SVM; 3-gramas de caracteres + SVM).
- Etapa 1: 4 representaciones TF-IDF × 4 estructuras (solo titular, solo texto, concatenado, bloques) × 3 modelos (LR, SVM lineal, Complement NB).
- Etapa 1b: texto con y sin nombres de medio.
- Etapa 2: regularización × `min_df` × minúsculas, con ampliación automática de la orilla (102 + 40 combinaciones).
- Etapa 3: selección χ² y bloque de rasgos de estilo.
- Diagnósticos: folds aleatorios frente a agrupados; *leave-one-topic-out*.
- Todo se guardó en `resultados/cv_lineales.csv` (1,094 filas) con el formato común.

## Resultados (F1 de fake, CV agrupada por fuente, media ± sd)
| Sistema | F1 fake | Nota |
|---|---|---|
| Clase mayoritaria | 0.377 ± 0.344 | Depende de la clase mayoritaria de cada fold |
| Aleatoria estratificada | 0.495 ± 0.019 | |
| Réplica 3-gramas + SVM | 0.679 ± 0.070 | Oficial en test: .7063 |
| Réplica BoW + SVM | 0.707 ± 0.051 | Oficial en test: .7217 |
| **LR, palabras (1,2) + caracteres (2,5), solo texto, min_df 20, C = 1 (elegido)** | **0.730 ± 0.095** | 18,252 atributos |
| SVM, misma representación, min_df 20, C = 0.1 (mejor media) | 0.733 ± 0.093 | |
| SVM + χ² k = 1,000 (regla original) | 0.702 ± 0.120 | |

## Hallazgos
- El texto solo es la mejor estructura (LR 0.707 de media); el titular solo da 0.46 y los bloques separados empeoran (0.626).
- Borrar los nombres de medio cuesta ≤ 0.003.
- Los folds aleatorios inflan la F1 entre 0.09 y 0.11: es lo que se aprende del estilo de medios ya vistos.
- Validar con un tema nuevo no derrumba el modelo (0.71 a 0.91), pero ese diagnóstico no agrupa por fuente.
- Ni χ² ni el bloque de estilo mejoran la media; el estilo reduce la sd de 0.093 a 0.049.

## Decisión que el equipo debe validar
⚠️ **Se revisó la regla de selección después de ver la CV (antes del test).** La regla original (un error estándar; menos atributos) elegía SVM + χ² con 0.702, por debajo de la réplica BoW. La regla revisada exige superar a la réplica BoW, prefiere procesos sin χ² ni estilo, luego LR y luego menos atributos. Las configuraciones de ambas reglas y la de mejor media quedan en `seleccion_lineal.json`; el test se reportará para todas, con la elegida como principal.

## Limitaciones
- La ventaja de la elegida sobre la réplica BoW en CV (0.023) es menor que la sd: no es una mejora demostrada.
- El ganador de SVM queda en la orilla de `min_df` (20), pero la curva es plana (diferencias < 0.01).
- Las réplicas no lematizan: no son idénticas a los baselines oficiales.
- No se ajustó el umbral de decisión.
- joblib imprime avisos `KeyError ... joblib_memmapping_folder` al limpiar temporales en Windows; no afectan los resultados.
