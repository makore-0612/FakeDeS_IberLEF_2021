# Resumen para la reunión del equipo (22 de noviembre de 2026)

## Estado por fase de CRISP-DM
| Fase | Estado | Notebook |
|---|---|---|
| Comprensión del negocio | ✅ Objetivos, riesgos, métrica (F1 de fake) y estado del arte | `analisis_exploratorio.ipynb` §1–2 |
| Comprensión de los datos | ✅ EDA completado (casi duplicados, pares, medio en el texto, pruebas estadísticas, calidad por dimensiones) | `analisis_exploratorio.ipynb` §3–19 |
| Preparación | ✅ 961 noticias de entrenamiento, test aparte, folds por fuente | `preparacion_datos.ipynb` |
| Modelado | ✅ Lineales (1,094 filas de CV) y árboles sobre estilo | `modelos_lineales.ipynb`, `modelo_arboles.ipynb` |
| Evaluación | ✅ Test evaluado una vez | `evaluacion.ipynb` |
| Despliegue | ⏳ Modelo serializado; API pendiente | `modelos/` |

## Resultados frente a los baselines oficiales (F1 de fake en el test)
| Sistema | Test | IC95 |
|---|---|---|
| **Nuestro principal (LR, palabras + caracteres, solo texto)** | **0.676** | [0.629, 0.722] |
| Nuestra réplica BoW + SVM | 0.660 | [0.608, 0.707] |
| Oficial BoW + SVM | 0.7217 | — |
| Oficial BETO | 0.7321 | — |
| 1.er lugar | 0.7666 | — |

**No superamos el baseline oficial.** Nuestra réplica del baseline también queda .06 abajo, así que una parte se debe al preprocesamiento. El principal tiene alta precisión (0.84) pero baja exhaustividad (0.57): falla sobre todo en falsas largas escritas como nota de prensa.

**Hallazgo de los números (verificación posterior, sin volver a evaluar).** La anonimización quedó homogénea: ninguna partición conserva números sueltos y los URLs del test pasaron a `*URL*`. Solo los 12 teléfonos de la 1.0 (`*PHONE*`) no tienen equivalente en el test, porque ahí quedaron como `*NUMBER*`. El problema es de contenido: en train las verdaderas citan casi 3 veces más números que las falsas (mediana de 1.28 contra 0.45 marcas por 100 palabras; 87 % contra 64 % de textos con cifras), pero en el test las falsas citan el doble (0.90; 82.5 %) y la brecha casi desaparece (verdaderas 1.18; 90.6 %). El modelo depende de eso: "number", "en number", "de number" y "number de" están entre los 6 atributos más fuertes hacia *True*. Los atributos del marcador aportan una mediana del 4 % de cada decisión, y sin ellos cambiarían 26 de 961 predicciones en train. En el Random Forest, `est_numeros` es el rasgo más importante. **Hipótesis:** "citar cifras = verdadera" empuja a las falsas del test (cifras de COVID) hacia *True* y contribuye a la baja exhaustividad. Se probaría en CV una variante sin el marcador, reportada como sistema aparte.

## Decisiones que el equipo debe validar
1. **Regla de selección revisada después de la CV y antes del test** (validada el 22-nov; documentada en `modelos_lineales.ipynb` §8.2).
2. **Borrar los nombres de medio** en lugar de marcarlos con `<MEDIO>` (no cuesta desempeño: ≤ 0.003).
3. **Eliminar de train 4 casi duplicados de noticias del test** (los participantes oficiales no lo hicieron).
4. **Usar solo el texto, sin el titular** (el titular empeoró la CV).
5. Si se prueba en CV una variante sin el marcador `*NUMBER*` (hallazgo de los números).
6. Qué pasos siguientes hacer antes del 2 de diciembre (ver abajo).

## Lo que falta para el 2 de diciembre
- Umbral de decisión elegido en CV (sistema aparte), API de demostración (preguntar antes de crear cuentas), `environment.yml` y `reproducibilidad.xlsx`, reporte final y PDF de estudio.
- Opcional, si hay Colab: BETO con cabeza + cola y ensamble. Detalle y tiempos en `notas/siguientes_pasos.md`.
