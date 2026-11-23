# Comprensión del negocio (CRISP-DM, fase 1)

Notebook: `analisis_exploratorio.ipynb`, Secciones 1 y 2. Fecha: 22 de noviembre de 2026.

## Qué se hizo
- Objetivos del negocio, evaluación de la situación (recursos, requisitos, supuestos, restricciones, riesgos, costos), objetivos de minería y plan, siguiendo las tareas de CRISP-DM de la sesión 1 del curso.
- Métrica oficial implementada en `utils_fakedes.py` (`f1_fake`, `metricas`) y probada con un ejemplo de juguete calculado a mano: 6 de 6 pruebas ✅.
- Estado del arte: overview oficial y 9 artículos de CEUR-WS Vol. 2943 (detalle en `notas/estado_del_arte.md`).

## Decisiones
| Decisión | Alternativas | Por qué |
|---|---|---|
| Métrica principal: F1 de `Fake`; macro-F1 como complementaria | Macro-F1 como principal (conclusión del EDA anterior) | Es la métrica del ranking oficial ("We used the F1 over the fake class to rank participants", overview p. 228) y la de la propuesta |
| No comparar macro-F1 con la tabla oficial | Usar la columna `Fmacro` del overview | Esa columna es idéntica a la F1 de fake en los 24 sistemas: no es una macro-F1 |
| Meta: superar BoW + SVM (.7217); acercarse a BETO (.7321) | Meta = ganador (.7666) | El ganador usó muestras de test de forma transductiva y HAHA (2.º) usó fuente y tema; nosotros solo usamos titular y texto |
| Solo `Headline` y `Text` como entrada | Usar fuente y tema como HAHA | La fuente es un atajo (V de Cramér = 0.996 en train) que no generaliza; HAHA cae de .7548 a .6975 sin metadatos |

## Limitaciones
- No hay acuerdo entre anotadores publicado: la exactitud de las etiquetas no es evaluable.
- La propuesta dice que el conjunto de evaluación solo trae id, titular y texto; `test.xlsx` sí trae etiquetas. Por eso solo `evaluacion.ipynb` lo carga.
