# Modelado: árboles sobre rasgos de estilo (CRISP-DM, fase 4)

Notebook: `modelo_arboles.ipynb` (≈ 5 min). Fecha: 22 de noviembre de 2026. No se cargó el test.

## Resultados (F1 de fake, CV agrupada)
| Sistema | F1 fake |
|---|---|
| LR sobre los 10 rasgos escalados (mejor C = 0.01) | 0.672 ± 0.094 |
| Random Forest, mejor media (sin límite de profundidad, hojas de 1) | 0.626 ± 0.125 |
| **Random Forest elegido (profundidad 3, hojas de 20, `sqrt`)** | **0.608 ± 0.168** |
| Modelo de texto elegido (referencia) | 0.730 ± 0.095 |

## Hallazgos
- La no linealidad no ayuda: el bosque queda por debajo de la LR sobre los mismos rasgos. Las 24 combinaciones están dentro de un SE (0.594 a 0.626).
- Rasgos importantes (permutación): números citados (0.062) y longitud (0.031, muy variable entre folds). Los signos de exclamación e interrogación no importan.
- Con folds aleatorios el bosque sube +0.142: usa los rasgos para reconocer medios.
- Frente al modelo de texto: pierde en 4 de 5 folds (−0.104 ± 0.191). Hay 86 noticias que acierta solo el bosque, 173 solo el texto y 147 ninguno: algo de complementariedad.

## Decisiones
- Regla de un SE con preferencia por la simplicidad (menor profundidad y hojas más grandes). El ganador por media está en la orilla, pero la curva es plana; no se amplió.
- El bosque se evalúa en el test solo como sistema secundario.

## Limitaciones
- La primera ejecución completa falló de forma intermitente (proceso paralelo de joblib); la segunda corrió sin errores con el mismo código.
