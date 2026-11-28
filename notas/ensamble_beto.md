# Ensamble con BETO: stacking con meta-modelos LR y MLP (sistema aparte)

Notebooks: `beto_resultados.ipynb` y `ensamble_beto.ipynb` (≈ 75 min; el álgebra lineal corre en un solo hilo para que la MLP sea reproducible). Entrenamiento: `beto_entrenar.py` (≈ 75 min de GPU para BETO, 68 para BERTIN y 70 para XLM-RoBERTa). Fecha: 26–27 de noviembre de 2026. Pasos 3 y 4 de `notas/siguientes_pasos.md`. Teoría: Alegre Ventura (2026), *Métodos de ensamble*, láminas 4 y 20–24. El test se usó solo para reportar. Paquetes nuevos en `requirements_beto.txt` (`torch==2.11.0` con CUDA 12.8, `transformers==5.24.1`, `accelerate==1.16.0`, `sentencepiece==0.2.3`), no en `environment.yml`. Las Secciones 10 a 13 del notebook son **exploratorias**: se propusieron después de ver el test (ver al final).

## Qué se hizo
- **BETO** (`dccuchile/bert-base-spanish-wwm-uncased`) con ajuste fino:
  - Configuración fija: truncado cabeza + cola (128 + 382 *tokens*), AdamW 2e-5, *weight decay* 0.01, 10 % de calentamiento, lote efectivo 16, 4 épocas, pesos de clase balanceados, fp16 y *gradient checkpointing*.
  - Entrenamientos: 5 folds agrupados × 3 semillas (42, 43, 44) para las predicciones *out-of-fold*, más 3 con las 961 noticias para el test. Se promedian las probabilidades de las 3 semillas.
  - Sin *grid search*: cada combinación costaría 15 entrenamientos.
- **Siete sistemas fijados antes de ver resultados:**
  1. BETO solo.
  2. Soft voting LR + BETO.
  3. Stacking LR + BETO con meta-modelo LR.
  4. Las 4 bases (LR, XGBoost, RF y BETO) con meta-modelo LR.
  5. Las 4 bases con meta-modelo MLP.
  6. Las 4 bases + longitud con meta-modelo LR.
  7. Las 4 bases + longitud con meta-modelo MLP.
- **Meta-modelo LR:** `GridSearchCV` de `C` ∈ {0.01, 0.1, 1, 10, 100} con `PredefinedSplit(fold_grupo)`.
- **Meta-modelo MLP:** *grid* de 4 arquitecturas × 5 `alpha`, `lbfgs`, 5 semillas promediadas y regla de un SE.
- **Estimación con esquema B:** para cada fold externo, el meta-modelo (incluido su *grid*) se entrena solo con el *out-of-fold* de los otros 4. Se aplicó también al stacking viejo, para comparar.
- **Exploratorio:** FT-Transformer como meta-modelo, en PyTorch puro y en CPU. *Grid* de 8 combinaciones, 5 semillas.
- **Dónde corre cada cosa:** BETO en la GPU (RTX 3050 Ti de 4 GB; usó 2.09 GB como máximo). LR, RF, MLP, XGBoost y el FT-Transformer en CPU: scikit-learn no usa GPU, los meta-modelos son diminutos y XGBoost se dejó igual que en `ensamble_stacking.ipynb`. Cuando la CPU estaba ocupada, BETO pasó de ≈ 3.3 a ≈ 4.3 min por entrenamiento, sin cambiar sus resultados.

## Resultados (F1 de fake)
| Sistema | CV (esquema B para los meta-modelos) | Test | IC95 test |
|---|---|---|---|
| LR texto (principal) | 0.730 | 0.676 | [0.629, 0.722] |
| LR texto, umbral 0.34 (referencia) | 0.744 (anidada) | 0.735 | [0.696, 0.771] |
| 1. BETO solo | 0.656 | 0.693 | [0.643, 0.736] |
| 2. Soft voting LR + BETO | 0.682 | 0.708 | [0.660, 0.752] |
| **3. Stacking LR (LR + BETO)** | **0.746** | 0.736 | [0.690, 0.778] |
| 4. Stacking LR (4 bases) | 0.730 | 0.738 | [0.693, 0.778] |
| 5. Stacking MLP (4 bases) | 0.734 | 0.745 | [0.700, 0.783] |
| 6. Stacking LR (4 bases + longitud) | 0.708 | **0.757** | [0.712, 0.796] |
| 7. Stacking MLP (4 bases + longitud) | 0.723 | 0.736 | [0.692, 0.776] |
| Stacking viejo recalculado (LR + XGBoost + RF) | 0.731 | 0.703 | [0.657, 0.748] |
| Stacking viejo reportado (referencia) | 0.684 (anidada) | 0.696 | [0.652, 0.741] |
| FT-Transformer (exploratorio) | 0.714 | 0.755 | [0.713, 0.795] |
| XGBoost (Linux) | 0.685 | 0.637 | [0.586, 0.686] |
| RF estilo | 0.608 | 0.505 | [0.446, 0.560] |

Tabla completa (todas las métricas en CV y test): `resultados/beto_resumen_sistemas.csv`.

## Hallazgos
- **BETO se equivoca en otras noticias que la LR.** Spearman BETO–LR = 0.674, contra 0.801 de XGBoost–LR. El oráculo sube de 0.872 (3 bases) a 0.921 (4 bases). Hay 109 noticias que falla la LR y acierta BETO, y 126 al revés. Ahora sí se cumple la condición de la lámina 4.
- **El fold 1 es todo sátira.** Sus 93 falsas son todas de *El Ruinaversal*.
  - Como la CV agrupa por fuente, BETO nunca ve sátira al entrenar en ese fold y la lee como verdadera: F1 de 0.303, 0.189 y 0.648 según la semilla, una diferencia de 0.4596, contra 0.0742 o menos en los demás folds.
  - No es un error: la pérdida baja igual que en los otros folds.
  - Se revisó con Miguel (26-nov) y se siguió con la configuración fija.
  - BETO solo en CV queda en 0.656: gana a la LR en 3 de 5 folds y pierde fuerte en el de la sátira.
- **En CV, el stacking LR + BETO (sistema 3) es el mejor:** 0.746 ± 0.055, y gana en 4 de 5 folds; todos los stackings con BETO pierden solo en el fold de la sátira.
  - La diferencia contra la LR (+0.0165 ± 0.1132) no es concluyente.
  - El esquema B infla: el stacking viejo da 0.731 con esquema B contra 0.684 con CV anidada (+0.047).
- **Pesos del sistema 3:** LR 5.128 y BETO 2.581, positivos y estables en los 5 folds. Con más entradas (sistemas 4 y 6) los pesos vuelven a cambiar de signo entre folds.
- **En el test, los 5 stackings con BETO le ganan al principal** con IC pareado que excluye el 0 (+0.059 a +0.080).
  - McNemar solo es significativa para el sistema 6 (p = 0.006).
  - **Empatan con el umbral 0.34** (+0.001 a +0.022, IC con 0).
  - Ganan al stacking viejo reportado (+0.040 a +0.061).
- **Ya no es solo mover el umbral:** la LR con el umbral equivalente saca 0.706 a 0.709 en el test, y los stackings con BETO 0.736 a 0.757. El stacking viejo sigue siendo un umbral disfrazado: 0.703 contra 0.711 de la LR con su umbral.
- **Falsas largas:**
  - En el test, BETO detecta el 42.4 % del cuartil más largo contra 24.2 % de la LR, y los sistemas 3 y 5 el 54.5 % y el 57.6 %.
  - En CV pasaba lo contrario: BETO 16.1 % contra 48.4 % de la LR.
  - Por eso la MLP del sistema 7 aprendió a desconfiar de BETO en noticias largas (figura de dependencia), y en el test eso le costó (30.3 %).
- **La red neuronal no le gana a la LR como meta-modelo.**
  - La regla de un SE elige la red más chica (4 neuronas, `alpha` = 1), casi lineal.
  - En CV, el sistema 5 (0.734) empata con el 4 (0.730), y el mejor meta-modelo es una LR (sistema 3).
- **El FT-Transformer (exploratorio) tampoco gana.**
  - En CV saca 0.714, abajo de la LR (0.730) y de la MLP 7 (0.723).
  - En el test empata con el sistema 6 (0.755 contra 0.757).
  - Tiene 713 parámetros para ≈ 770 filas por fold y costó 26 min contra segundos de la LR.

## Para el reporte
Resultado positivo moderado: con una base que lee la noticia de otra forma, el stacking **sí** complementa a la LR. Lo hace por la razón que dice la teoría, errores distintos, y no solo por mover el umbral. La ganancia es clara en el test contra el principal (+0.06 a +0.08), pero no contra el umbral 0.34. La CV agrupada por fuente destapa un punto ciego de BETO: la sátira.

## Decisiones para el equipo
1. **Si se elige un sistema de aquí, que sea el 3 (stacking LR + BETO con meta-modelo LR)**, porque es el mejor en CV: test 0.736 [0.690, 0.778]. El 6 es mejor en el test (0.757), pero fue el peor stacking en CV, así que elegirlo sería decidir con el test.
2. Reportar el umbral 0.34 y el stacking 3 como **empatados** en el test.
3. Mencionar el hallazgo de la sátira (*El Ruinaversal*) como limitación de BETO y como argumento a favor de la CV agrupada por fuente.
4. ~~Probar en CV el umbral sobre el stacking 3, con otra CV anidada~~ ✅ hecho (E1): no ayuda (0.7463 → 0.7472).
5. Mencionar BERTIN como hallazgo con su advertencia de contaminación de estilo: es la mejor base en CV, pero su ventaja no se sostiene en el test.

## Limitaciones
- Esquema B en lugar de CV anidada completa: la fuga residual infla la CV, ≈ +0.047 en el stacking viejo.
- BETO sin *grid search*, con 3 semillas; muy inestable en el fold de la sátira.
- XGBoost de Linux (0.6855) en lugar del de Windows (0.6773); el stacking viejo recalculado coincide en el 96.5 % con el reportado.
- El orden entre sistemas cambia entre CV y test; con 572 noticias no se separan stackings que difieran ±0.02.
- **Incidentes de ejecución:**
  - Durante el entrenamiento de BETO se presionó el botón de encendido y la laptop se apagó con 13 de 18 piezas listas. El script se relanzó y saltó las piezas existentes; cada pieza fija su semilla, así que el resultado no cambia.
  - La primera corrida de `ensamble_beto.ipynb` se detuvo a la mitad para apagar la laptop, después de cargar el test. Se volvió a correr completa sin cambiar el código. Ninguna decisión se tomó viendo el test.
- No se revisó si el test tiene sátira, para no usar el test con fines de decisión.
- Las exploraciones (Secciones 10 a 13) se propusieron después de ver el test, y algunas cifras del test salieron en pantalla al depurarlas con copias de prueba. Por eso son exploratorias.
- BERTIN tiene contaminación de estilo (conocía los sitios del corpus por su preentrenamiento): su CV no es comparable con la de BETO y la LR.
- XLM-RoBERTa se entrenó con la tabla de *embeddings* de palabras congelada para caber en 4 GB.
- La MLP (sistemas 5 y 7) variaba ≈ 0.002 de F1 en CV entre corridas por el álgebra lineal multihilo. Se resolvió fijando el álgebra lineal (BLAS) a un hilo en el notebook: ahora da siempre lo mismo.
- MarIA (`PlanTL-GOB-ES/roberta-base-bne`) ya no tenía pesos públicos el 27-nov; se usaron BERTIN y XLM-RoBERTa en su lugar.

## Exploraciones (Secciones 10 a 13 de `ensamble_beto.ipynb`, 27-nov)
Propuestas por el equipo **después** de ver los resultados del test de la Sección 7. Todas están etiquetadas como exploratorias y ninguna es candidata a sistema final. Lo nuevo de las Secciones 11 a 13 se aplicó al test en una sola pasada (Sección 13.1).

| Exploración | CV | Test | IC95 test | Comentario |
|---|---|---|---|---|
| FT-Transformer como meta-modelo (Sec. 10) | 0.714 | 0.755 | [0.713, 0.795] | No gana en CV; empata con el sistema 6 en el test |
| Stacking LR + BETO + 3 boostings (Sec. 11, fuera de la regla) | 0.733 | 0.7335 | [0.687, 0.774] | No pasó la regla (no superó al sistema 3 en CV); test solo descriptivo |
| SVM-RBF sobre *embeddings* de BETO (Sec. 12) | 0.672 | 0.6747 | [0.628, 0.723] | `C` = 100 |
| kNN sobre *embeddings* (Sec. 12) | 0.559 | 0.4861 | [0.423, 0.539] | k = 31 y k = 61 empatan; la base más débil |
| Naive Bayes, palabras (Sec. 12) | 0.673 | 0.6166 | [0.562, 0.664] | Spearman 0.867 con la LR; el más inestable entre folds |
| BERTIN (Sec. 12) | **0.810** | 0.7258 | [0.684, 0.767] | La mejor base en CV; la caída más grande en el test |
| XLM-RoBERTa (Sec. 12) | 0.639 | 0.6708 | [0.616, 0.718] | Entrenado con la tabla de *embeddings* congelada (4 GB) |
| E1. Sistema 3 + umbral anidado (Sec. 13) | 0.747 | 0.7440 | [0.702, 0.781] | El umbral anidado casi no cambia nada (0.7463 → 0.7472) |
| E2. Stacking LR (LR + 3 transformers) | 0.798 | 0.7397 | [0.696, 0.779] | |
| E3. Stacking con pesos no negativos, todas las bases | 0.812 | 0.7496 | [0.708, 0.789] | Se queda con 6 bases; BERTIN pesa más |
| E4. E3 + umbral anidado | **0.813** | **0.7550** | [0.715, 0.794] | El mejor exploratorio en el test; no se distingue del sistema 3 ni del umbral 0.34 |

**Hallazgos de las exploraciones:**
- **Boostings (Sec. 11):** cambiar la semilla no los diversifica (Spearman 0.976 a 0.978 entre semillas de XGBoost; AdaBoost da modelos idénticos). Cambiar la vista sí (los ganadores por vista tienen Spearman 0.725 a 0.764 entre sí), pero todos quedan abajo de la LR. Ganadores: HistGradientBoosting con caracteres (0.6921), XGBoost con estilo (0.6369) y HistGradientBoosting con palabras + estilo (0.6791).
- **Bases nuevas (Sec. 12):** con las cinco, el oráculo sube de 0.871 (LR + BETO) a 0.943. BERTIN acierta sola 50 noticias que fallan la LR y BETO, y detecta el 98.9 % de las falsas de *El Ruinaversal* (BETO, 19.4 %).
- **Contaminación (Sec. 12.4):**
  - Los **sitios** del corpus sí están en la muestra de mC4 con la que se preentrenó BERTIN: páginas de *El Ruinaversal*, *El Dizque*, *Hay Noticia*, El Universal, El País y Forbes.
  - **Ninguna noticia concreta del corpus** apareció con su URL exacta en la parte indexada, y el control positivo sí funcionó.
  - Al adivinar *tokens* tapados, BERTIN no completa mejor la sátira que el test (diferencia en diferencias −1.20 puntos, IC95 [−3.3, +0.9]); sí completa mejor las verdaderas (+3.35, IC95 [+2.0, +4.7]).
  - Conclusión: **contaminación de estilo, no memorización de noticias**. La caída de BERTIN de 0.810 (CV) a 0.7258 (test) indica que su CV es optimista.
- **Meta-modelos (Sec. 13):** el umbral anidado no ayuda al sistema 3. Los pesos no negativos evitan los signos cambiantes y seleccionan bases. En CV, todo lo que lleva a BERTIN sube +0.05 a +0.07 sobre el sistema 3, pero en el test solo +0.004 a +0.019, sin significancia. Ninguna McNemar es significativa.
- **Auditoría de fuga (Tabla 14.3):** ninguna fuente aparece en dos folds; no hay textos duplicados entre folds ni entre corpus y test; ningún `fit` ni ninguna elección usa el test. Quedan tres advertencias conocidas: fuga residual del esquema B, hiperparámetros de las bases sin anidar y exploraciones diseñadas después de ver el test.

**Decisión que no cambia:** el sistema que se elegiría con la CV de los sistemas fijados sigue siendo el **3**. E4 es el mejor exploratorio en el test (0.7550), pero se diseñó después de ver el test y no se distingue del sistema 3 ni del umbral 0.34.

## Los tres finalistas (`finalistas.ipynb`, 27-nov)
Por decisión del equipo se documentan a fondo tres finalistas: el **sistema 3** (elegido por el protocolo), **E3** y **E4** (los mejores en CV, diseñados después de ver el test). `finalistas.ipynb` trae métricas en CV y test, matrices de confusión, pérdidas de entrenamiento, búsqueda de `C`, pesos, curva del umbral, SHAP exacto por base y LIME por palabra (`finalistas_lime.py`, en GPU).

| | Sistema 3 | E3 | E4 |
|---|---|---|---|
| CV | 0.7463 | 0.8116 | 0.813 |
| Test | 0.7357 [0.690, 0.778] | 0.7496 [0.708, 0.789] | 0.7550 [0.715, 0.794] |
| Exactitud (test) | 0.7500 | 0.7430 | 0.7413 |
| Base que más pesa (SHAP) | BETO 53.4 %, LR 46.6 % | BERTIN 50.0 % | igual que E3 |

- En el test no se distinguen: E4 − sistema 3 = +0.019 (IC [−0.019, +0.056], McNemar p = 0.691).
- Los tres deciden sobre todo por el registro (coloquial → falsa; crónica con cifras → verdadera), según LIME y el SHAP de la LR.
- **Recomendación:** el sistema 3 como sistema final; E4 como el mejor resultado exploratorio, con su advertencia.

## Verificaciones (sección 6 de la tarea)
| Verificación | Estado |
|---|---|
| `folds.csv` regenerado idéntico al del zip | ✅ (md5 `32bea9ad…`) |
| LR texto out-of-fold: 0.7848, 0.8541, 0.6345, 0.6375, 0.7381 | ✅ |
| XGBoost media 0.6773 / RF estilo 0.6083 | ⚠️ XGBoost 0.6855 en Linux (aceptado 26-nov) / ✅ RF |
| Principal en el test 100 % igual a `predicciones_test.csv` | ✅ |
| `beto_oof.csv` 961 filas, `beto_test.csv` 572, ids coinciden, sin NaN | ✅ |
| `beto_entrenar.py` solo lee `id_unico` y `texto_sin_medio` del test | ✅ |
| BETO en CV ≥ 0.60 / semillas ≤ 0.10 por fold | ✅ 0.6559 / ⚠️ 0.4596 en el fold de la sátira (revisado y aceptado) |
| Cifras del texto contra las salidas (script) | ✅ 219 cifras en `ensamble_beto.ipynb`, 15 en `beto_resultados.ipynb` y 139 en esta nota, todas con respaldo (`_generadores/revisar_cifras.py`, fuera del repo) |
| Notebooks corren de principio a fin con `nbconvert --execute` | ✅ (`ensamble_beto.ipynb` en 75 min; `beto_resultados.ipynb` en segundos; `finalistas.ipynb` en ≈ 1 min) |

## Salidas
`beto_entrenar.py` (ahora acepta `--modelo`, `--prefijo` y `--congelar_embeddings`), `beto_embeddings.py`, `finalistas_lime.py`, `finalistas.ipynb`, `resumen_proyecto.ipynb`, `contaminacion_mlm.py`, `contaminacion_busqueda.py`, `requirements_beto.txt`, `beto_resultados.ipynb`, `ensamble_beto.ipynb`.

Resultados:
- de BETO: `resultados/beto_oof.csv`, `beto_test.csv`, `beto_log.csv`, `beto_parciales/` (18 piezas) y `beto_corrida.log`;
- de BERTIN y XLM-RoBERTa: `resultados/bertin_*` y `resultados/xlmr_*` (oof, test, log, parciales y corrida);
- *embeddings* y contaminación: `resultados/beto_embeddings.npz`, `contaminacion_mlm.csv`, `contaminacion_busqueda.csv`, `contaminacion_busqueda_noticias.csv` y `contaminacion_busqueda_control.csv`;
- del ensamble: `resultados/beto_stacking_cv.csv`, `beto_stacking_test.csv`, `beto_meta_grid.csv`, `beto_resumen_sistemas.csv`, `predicciones_test_beto.csv`, `seleccion_beto.json`, `beto_exploratorio_ft.csv`, `beto_exploratorio_boosting.csv` y `beto_exploratorio_extra.csv`.

Figuras: `2_1_beto_curvas_perdida`, `2_2_beto_matriz_cv`, `5_1_beto_mlp_grid`, `5_2_beto_dependencia`, `6_1_beto_matrices_cv`, `7_1_beto_matrices_test`, `7_2_beto_f1_test`, `10_1_beto_ft_perdida`, `10_2_beto_ft_matrices`, `11_1_beto_boosting_matrices_cv`, `12_1_beto_contaminacion`, `13_1_beto_umbral_sistema3` y `13_2_beto_exploratorios_matrices_test`.
