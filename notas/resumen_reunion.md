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

## Actualización del 26 de noviembre: umbral de decisión
Umbral elegido en CV agrupada: **0.34** (`umbral.ipynb`, `notas/umbral.md`). En el test, como sistema aparte: F1 de fake **0.735** (IC95 [0.696, 0.771]) contra 0.676 del principal (+0.058, IC95 pareado [+0.012, +0.103]). Supera a .7217 en estimación puntual, sin significancia. No acierta más noticias (exactitud 0.713, McNemar p = 0.498): cambia falsas no detectadas por falsas alarmas. La CV anidada solo esperaba +0.014.

## Actualización del 26 de noviembre: ensamble (XGBoost + stacking)
Bases: LR texto, XGBoost (grid search en CV agrupada; elegido: palabras, profundidad 2, 200 árboles) y RF estilo; meta-modelo LR con grid search de `C` sobre predicciones out-of-fold (`ensamble_stacking.ipynb`, `notas/ensamble_stacking.md`). **No mejora a la LR sola:** CV anidada 0.684 contra 0.730; test 0.696 contra 0.676 (+0.020, IC95 pareado [−0.009, +0.049]) y por debajo del umbral 0.34 (0.735). Las bases comparten el sesgo de longitud y el stacking se comporta casi como la LR con otro umbral (Spearman 0.984). Es un resultado negativo útil para el reporte.

## Actualización del 27 de noviembre: BETO y stacking con BETO
BETO (`dccuchile/bert-base-spanish-wwm-uncased`, cabeza + cola, 4 épocas, 3 semillas) se entrenó localmente en la GPU de la laptop (≈ 75 min). Detalle en `beto_resultados.ipynb`, `ensamble_beto.ipynb` y `notas/ensamble_beto.md`.

**BETO solo:** 0.656 en CV y 0.693 en el test.
- **Hallazgo:** en el fold 1 todas las falsas son de *El Ruinaversal* (sátira), BETO las lee como verdaderas y sus semillas difieren 0.46 de F1.
- Se revisó y se siguió con la configuración fija.

**Ahora el ensamble sí complementa a la LR.** BETO se equivoca en otras noticias (Spearman con la LR 0.674, contra 0.801 de XGBoost) y el meta-modelo le da un peso estable.
- **En CV:** el stacking LR + BETO con meta-modelo LR (sistema 3) es el mejor, 0.746 contra 0.730. No es concluyente: el esquema B infla ≈ +0.047.
- **En el test:** los 5 stackings con BETO le ganan al principal, de +0.059 a +0.080 con IC pareado que excluye el 0.
  - **Empatan con el umbral 0.34.**
  - La mejora no es solo un cambio de umbral: la LR con el umbral equivalente saca 0.706 a 0.709, y los stackings 0.736 a 0.757.

**Meta-modelos más complejos:**
- **MLP:** no le gana a la LR; la regla de un SE elige la red más chica.
- **FT-Transformer (exploratorio):** tampoco gana en CV.

**Si se elige un sistema de aquí, la CV indica el 3:** 0.736 en el test, IC [0.690, 0.778].

**Decisiones para el equipo** (detalle en `notas/ensamble_beto.md`):
- reportar el 3 y el umbral 0.34 como empatados;
- mencionar la sátira como limitación de BETO.

## Actualización del 27 de noviembre (tarde): exploraciones con más bases y meta-modelos
Todo esto está en las Secciones 10 a 13 de `ensamble_beto.ipynb` y en `notas/ensamble_beto.md`. Es **exploratorio**: se propuso después de ver el test.

**Bases nuevas:**
- Boostings por vista (caracteres, estilo, palabras + estilo): no le suman al stacking LR + BETO.
- SVM y kNN sobre *embeddings* de BETO.
- Naive Bayes.
- Dos transformers más:
  - **BERTIN**, la mejor base en CV con 0.810 (detecta la sátira de *El Ruinaversal*);
  - **XLM-RoBERTa**, débil con 0.639.

**Prueba de contaminación de BERTIN:**
- Su preentrenamiento (mC4) sí incluye páginas de los sitios del corpus, pero no las noticias concretas, y no completa mejor la sátira que el test.
- Hay contaminación de estilo, no memorización.
- En el test, BERTIN baja a 0.726, así que su CV es optimista.

**Meta-modelos:**
- El umbral anidado no ayuda al sistema 3.
- Los pesos no negativos sobre todas las bases (E3/E4) dan 0.81 en CV y hasta 0.755 en el test (E4). Esto no se distingue del sistema 3 (0.736) ni del umbral 0.34 (0.735).

**Auditoría de fuga: sin fuga hacia el test.** La decisión no cambia: si se elige un sistema de aquí, es el 3.

**Finalistas:** sistema 3 (final por protocolo; test 0.736), E3 (0.750) y E4 (0.755, mejor exploratorio); empatan en el test. Detalle: `finalistas.ipynb`.

**Resumen de todo el proyecto en un solo notebook:** `resumen_proyecto.ipynb` (lee los resultados guardados; corre en segundos, también en Colab).

## Lo que falta para el 2 de diciembre
- ~~Umbral de decisión elegido en CV (sistema aparte)~~ ✅. ~~`environment.yml` y `reproducibilidad.xlsx`~~ ✅ (ya estaban en el repo). Falta: API de demostración (preguntar antes de crear cuentas), reporte final y PDF de estudio.
- ~~Opcional, si hay Colab: BETO con cabeza + cola y ensamble~~ ✅ (27-nov, en la GPU local; `notas/ensamble_beto.md`).
