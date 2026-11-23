# Evaluación en el test (CRISP-DM, fase 5)

Notebook: `evaluacion.ipynb` (único que carga el test). Fecha: 22 de noviembre de 2026. Se evaluó **una sola vez** cada sistema fijado antes de abrir el test.

## Resultados (F1 de fake en el test; misma partición y métrica que la tabla oficial)
| Sistema | CV | Test | IC95 (bootstrap, 1,000) |
|---|---|---|---|
| **LR texto (principal)** | 0.730 | **0.676** | [0.629, 0.722] |
| SVM texto (mejor media en CV) | 0.733 | 0.675 | [0.625, 0.721] |
| Réplica BoW + SVM (nuestra) | 0.707 | 0.660 | [0.608, 0.707] |
| Réplica 3-gramas + SVM (nuestra) | 0.679 | 0.618 | [0.563, 0.665] |
| SVM + χ² (regla original) | 0.702 | 0.611 | [0.554, 0.660] |
| Random Forest (estilo) | 0.608 | 0.505 | [0.446, 0.560] |
| Oficial BoW + SVM / BETO / 1.er lugar | — | .7217 / .7321 / .7666 | — |

Principal: precisión de fake 0.839, exhaustividad 0.566, macro-F1 0.722, exactitud 0.729.

## Hallazgos
- **No se cumple el criterio de éxito** (superar .7217). Nuestra réplica BoW también queda .06 por debajo de la oficial: el preprocesamiento oficial (lematización, sin números ni puntuación) parece importar.
- Principal frente a réplica BoW: +0.016 (IC95 [−0.028, +0.060]); McNemar p = 0.163 (0.327 corregido). Sin diferencia demostrada.
- Brecha CV → test del principal: −0.054.
- Errores: las falsas largas casi no se detectan (Q4: 24 %; Q1: 78 %). Las de redes o verificadores se detectan mejor (68 % contra 49 %, Fisher p corregido = 0.002). COVID-19 queda igual que los temas conocidos.
- Interpretación: el modelo separa registro periodístico (verbos de atribución, cifras, fechas) de registro informal (conectores, discurso); LIME coincide con los coeficientes (Spearman 0.83, R² local mediano 0.755).

## Salidas
`resultados/test_final.csv`, `resultados/predicciones_test.csv`, `modelos/modelo_principal.joblib`, `modelos/config_principal.json`, figuras `3_1_f1_test_vs_oficiales.png` y `5_1_matriz_confusion.png`.
