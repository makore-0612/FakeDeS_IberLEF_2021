# Ajuste del umbral de decisión (sistema aparte)

Notebook: `umbral.ipynb` (≈ 1.5 min). Fecha: 26 de noviembre de 2026. Paso 1 de `notas/siguientes_pasos.md`. El test se cargó **una sola vez**, al final; como ya se había visto en `evaluacion.ipynb`, el resultado es un **sistema aparte** y el principal no cambia.

## Qué se hizo
- Probabilidades de *fake* fuera de fold del principal (CV agrupada por fuente). Con el umbral 0.5 se reproducen las 5 F1 por fold guardadas (media 0.7298).
- Rejilla de umbrales de 0.20 a 0.80 de 0.01 en 0.01. Regla fijada antes: mayor F1 de fake media entre folds; en empate, el más cercano a 0.5.
- CV anidada: el umbral se elige con CV interna sobre 4 folds y se mide en el quinto.
- Aplicación única en el test con los mismos 1,000 remuestreos bootstrap de `evaluacion.ipynb`.

## Resultados (F1 de fake)
| Sistema | CV agrupada | CV anidada | Test | IC95 test |
|---|---|---|---|---|
| LR texto (principal, umbral 0.5) | 0.730 | 0.730 | 0.676 | [0.629, 0.722] |
| **LR texto, umbral 0.34** | 0.756 | 0.744 | **0.735** | [0.696, 0.771] |
| Oficial BoW + SVM / BETO | — | — | .7217 / .7321 | — |

Umbral 0.34 en el test: precisión de fake 0.684 (antes 0.839), exhaustividad 0.794 (antes 0.566), exactitud 0.713 (antes 0.729), macro-F1 0.711 (antes 0.722).

## Hallazgos
- **Mejora clara en F1 de fake frente al principal:** +0.058, IC95 pareado [+0.012, +0.103]; gana en 993 de 1,000 remuestreos.
- **No acierta más noticias:** McNemar p = 0.498 (65 contra 74). Detecta 65 falsas más y marca 74 verdaderas más como falsas.
- **Frente al oficial:** 0.735 > .7217 y > .7321 en estimación puntual, pero .7217 está dentro del IC (lo supera en 744 de 1,000 remuestreos). No demostrado.
- **Estable:** las CV internas eligen 0.36–0.43; en el test toda la zona 0.30–0.45 da entre 0.705 y 0.739.
- **La CV anidada esperaba poco** (+0.014 ± 0.047; mejora en 3 de 5 folds). La ganancia en el test es mayor porque el test está corrido hacia *True* (solo el 56.6 % de las falsas pasa de 0.5, contra 69.8 % en CV) y bajar el umbral compensa ese corrimiento.
- **Falsas largas:** Q4 pasa de 24.2 % a 63.6 % detectadas. El costo cae en las verdaderas cortas: Q1 baja de 69.7 % a 39.4 % acertadas.

## Decisiones que el equipo debe validar
1. Reportar el umbral 0.34 como sistema aparte junto al principal (propuesta), o solo como análisis de sensibilidad.
2. Qué umbral usa la API: `modelos/config_umbral.json` guarda el 0.34.

## Limitaciones
- La decisión de probar el umbral vino de ver la baja exhaustividad en el test.
- El test está balanceado; con menos falsas en la práctica, el umbral bajo daría más falsas alarmas.
- Probabilidades sin calibrar: el umbral se eligió con modelos de 4/5 de los datos y se aplica al modelo entrenado con todo.

## Salidas
`resultados/umbral_cv.csv`, `resultados/umbral_anidado.csv`, `resultados/umbral_test.csv`, `resultados/predicciones_test_umbral.csv`, `modelos/config_umbral.json`, figuras `2_1_umbral_curva_cv.png` y `4_1_umbral_matrices_confusion.png`.
