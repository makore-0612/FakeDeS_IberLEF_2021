# Estado del arte: FakeDeS @ IberLEF 2021 (Fase 0.5)

Fecha de revisión: 22 de noviembre de 2026.

## Fuentes consultadas

| Fuente | Estado |
|---|---|
| Actas IberLEF 2021, CEUR-WS Vol. 2943 (`ceur-ws.org/Vol-2943/`), sección FakeDeS, pp. 599–667 | ✅ Índice consultado; 9 *system papers* abiertos y leídos |
| Overview de FakeDeS 2021 (Gómez-Adorno et al., *Procesamiento del Lenguaje Natural* 67, pp. 223–231) | ✅ Leído completo (PDF de `core.ac.uk/download/479277994.pdf`, descargado por el equipo; copia local `479277994.pdf`) |
| Posadas-Durán et al. (2019), artículo del corpus | ❌ Sin verificar (no se buscó todavía) |
| Overview de MEX-A3T 2020 | ❌ Sin verificar |
| Kiesel et al. (2019), SemEval-2019 Task 4 (`aclanthology.org/S19-2145`) | ⚠️ Solo ficha bibliográfica (ACL Anthology); no se leyó el artículo |

## Lo que dice el overview [10]

- **Métrica:** "We used the F1 score over the fake class as the main evaluation measure" (p. 227) y "We used the F1 over the fake class to rank participants" (p. 228). Lo confirman YETI y CiTIUS en sus artículos.
- ⚠️ **Inconsistencia en la tabla 4 del overview:** la columna `Fmacro` es idéntica a la columna `Fake` para los 24 sistemas (p. ej. GDUFS_DM: Fake .7666, True .7649, Fmacro .7666; el promedio real sería .7658). Por eso HAHA y zk15120170770 llaman "F1-macro" a su F1 de fake. **No usamos esa columna; si reportamos macro-F1, la calculamos nosotros y no la comparamos con la tabla.**
- **Baselines oficiales** (p. 227), los tres con preprocesamiento clásico (minúsculas, tokenización; quitar puntuación y números; lematización):
  - BoW + SVM lineal, todo el vocabulario, **conservando stopwords**: F1 fake .7217, F1 true .7359, exactitud .729.
  - 3-gramas de caracteres + SVM lineal, todo el vocabulario: F1 fake .7063, F1 true .6883, exactitud .6976.
  - BETO uncased (`dccuchile/bert-base-spanish-wwm-uncased`), Adam lr 2e-5, 10 épocas, 512 tokens: F1 fake .7321, F1 true .7432, exactitud .7378.
- **Test (tabla 2):** 572 noticias, 286 / 286. COVID-19 237 (41.4 %), Sociedad 195, Política 107, Internacional 14, Ciencia 13, Ambiental 4, Deporte 2. Las 90 publicaciones de redes (todas falsas) se reparten en COVID-19 59, Política 17, Sociedad 10, Ciencia 3 y Ambiental 1.
- **Complementariedad (tabla 5):** la exactitud máxima posible (MPA) es .9965 con los 21 equipos, .9528 con transformers y .9580 con BoW / n-gramas / estilometría. Los dos enfoques fallan en noticias distintas.
- **Por tema (tabla 6):** COVID-19 no queda peor que Política o Sociedad (promedios .66, .68 y .68). Deporte tiene solo 2 noticias, así que su F1 no es interpretable. Solo 2 noticias verdaderas (Política y Sociedad) fueron mal clasificadas por todos los sistemas, y ambas son desmentidos publicados por medios ("No, no ha habido…", "La falsa noticia de…").
- 3.º y 4.º lugar (Chats, SINAI) no enviaron artículo.

## Tabla de sistemas

| Sistema | Lugar | Preprocesamiento | Representación | Modelo | Trucos | F1 fake (test) | ¿Usa información que nosotros no tendremos? |
|---|---|---|---|---|---|---|---|
| GDUF_DM [1] | 1 | Titular + texto concatenados | BETO cased (y SpanBERTa): embeddings de los primeros 512 y los últimos 512 tokens | BERT + "Sample Memory" (atención sobre embeddings de muestras) | Cabeza + cola; acumulación de gradiente; *early stopping* en dev; tasas de aprendizaje por capa | .7666 (SpanBERTa: .6907) | **Sí:** la memoria se inicializa con muestras de train, dev **y test** (transductivo) |
| HAHA [2] | 2 | Concatena `Topic`, `Source`, `Headline`, `Text` y `Link` (el texto también lista `Category`, lo que sería fuga; en el test no había etiquetas, así que parece una errata del artículo); del enlace conserva solo el dominio; quita URLs y puntuación | TF-IDF + `LabelEncoder` de `Topic` y `Source` | *Stacking*: LR, SGD, PassiveAggressive, Ridge, LinearSVC → LightGBM | CV 5-fold para el *stacking* | .7548 (con metadatos); **.6975 solo texto y sin stopwords** | **Sí:** fuente, dominio y tema. Los propios autores escriben que "it is harmful to treat all news from a link as fake news or real news" |
| CiTIUS [3] | 6 | — | Sentence-BERT (media de vectores) | Similitud de oraciones + heurísticas lingüísticas | Train + dev para el envío final | .701 (BERT *fine-tuning*: dev .7655 → **test .4719**) | No |
| zk15120170770 [4] | 7 | — | Modelo preentrenado | Fine-tuning | — | .7053 | No determinado |
| ForceNLP [5] | 8 | Umbral de frecuencia 3 (`CountVectorizer`) | n-gramas de caracteres (3–5), palabras, skip-gramas, POS, stopwords y puntuación | LR, SVM, Multinomial NB | — | .6925 (LR); MNB: dev .7576 → test .4928 | No |
| TSIA [6] | 10 | — | BETO y XLM-R Large (+ CNN) | Fine-tuning | Voto duro de 3 corridas | .6860 | No |
| LCAD-UFES [7] | 5 | POS + NER con spaCy | Secuencias POS+NER | GB, RF, SVM, WiSARD | — | GB .7114 (máx.), .7083 (media) | No |
| YETI [8] | — | — | ALBERT | Fine-tuning | — | .6316 | No |
| Bribones [9] | — | — | Bolsa de palabras | Random Forest, k-NN | — | No extraído | No |
| Baseline BETO [10] | — | Minúsculas | BETO uncased, 512 tokens | Fine-tuning | — | .7321 | No |
| Baseline BoW + SVM [10] | — | Minúsculas, conserva stopwords | Bolsa de palabras | SVM lineal | — | .7217 | No |
| Baseline 3-gramas de caracteres + SVM [10] | — | Minúsculas, sin puntuación ni números | 3-gramas de caracteres | SVM lineal | — | .7063 | No |

## Síntesis

**Lo que se repite en varios equipos:**
1. **Dev es optimista y no predice el test.** CiTIUS: BERT pasa de .7655 en dev a .4719 en test. ForceNLP: MNB pasa de .7576 a .4928, y el orden entre modelos se invierte. HAHA reporta ≈ .90 en dev contra .75 en test. GDUF_DM: .8507 en dev contra .7666 en test. Esto respalda no usar `development` como único conjunto de validación (comparte el 90.5 % de sus fuentes con train, según el EDA).
2. **Los metadatos de medio inflan el resultado.** La única comparación disponible es de HAHA: .7548 con metadatos frente a .6975 sin ellos. ⚠️ Cambian dos cosas a la vez (metadatos y stopwords), así que la diferencia de .057 no se atribuye solo a los metadatos.
3. **Los métodos clásicos bien ajustados están cerca de los transformers.** El baseline de caracteres (.7062) supera a la mitad de los sistemas con transformer que leímos (TSIA .6860, YETI .6316 y el BERT de CiTIUS .4719).
4. **Textos largos:** GDUF_DM midió que el 41 % de las noticias de train + dev supera los 512 tokens de BETO; por eso usó cabeza + cola.

**Lo que solo le funcionó a un equipo:** la Sample Memory (GDUF_DM, transductiva) y el *stacking* con LightGBM (HAHA, con metadatos).

**Diferencias con nuestra configuración:** todos los equipos eligieron con `development` o con splits aleatorios; ninguno agrupó por fuente. Sus puntajes de dev no son comparables con nuestro CV agrupado. Los puntajes de test sí son comparables: es la misma partición oficial (`test.xlsx`, 572 noticias) y la misma métrica (F1 de fake).

## Ideas para probar (hipótesis, no recetas)

| # | Idea | Origen | Fase donde se prueba |
|---|---|---|---|
| 1 | TF-IDF de caracteres (`char_wb` 2–5 o 3–5) como representación principal, en bloque aparte del de palabras | Baseline oficial, ForceNLP | Fase 6 (`modelos_lineales.ipynb`) |
| 2 | Conservar las stopwords; no quitarlas sin probarlo | HAHA (quitarlas bajó el desempeño en dev) | Fase 6 |
| 3 | Titular y texto en bloques separados frente a concatenados | GDUF_DM concatena; prompt, riesgo 7 | Fase 6 |
| 4 | *Stacking* de lineales sin metadatos | HAHA | Fase 7 (si sobra tiempo) |
| 5 | Truncado cabeza + cola para BETO | GDUF_DM; Sun et al. (2019) | Solo planeado (`notas/siguientes_pasos.md`) |
| 6 | Medir la brecha entre CV agrupado y folds aleatorios como diagnóstico del atajo por medio | CiTIUS, ForceNLP (brecha dev–test); Kiesel et al. (2019) | Fase 5 y Fase 8 |

## Resultados publicados como referencia externa

| Referencia | F1 fake (test) | Fuente | Comparable |
|---|---|---|---|
| 1.er lugar (GDUF_DM) | .7666 | [1], tabla 2 | Sí (misma partición y métrica); usa test de forma transductiva |
| 2.º lugar (HAHA) | .7548 | [2], tabla 2; [1] | Sí, pero usa metadatos |
| BETO (baseline oficial) | .7321 | [10], tabla 4 | Sí |
| BoW + SVM (baseline oficial) | .7217 | [10], tabla 4 | Sí |
| 3-gramas de caracteres + SVM (baseline oficial) | .7063 | [10], tabla 4 (ForceNLP [5] lo cita como .7062) | Sí |
| Oráculo (MPA, 21 equipos) | exactitud .9965 | [10], tabla 5 | Cota teórica, no alcanzable |

## Referencias

1. Huang, X., Xiong, J. y Jiang, S. (2021). GDUF_DM at FakeDeS 2021: Spanish Fake News Detection with BERT and Sample Memory. En *Proceedings of IberLEF 2021*, CEUR-WS Vol. 2943, pp. 621–629. https://ceur-ws.org/Vol-2943/fakedes_paper4.pdf
2. Li, K. (2021). HAHA at FakeDeS 2021: A Fake News Detection Method Based on TF-IDF and Ensemble Machine Learning. CEUR-WS Vol. 2943, pp. 630–638. https://ceur-ws.org/Vol-2943/fakedes_paper5.pdf
3. Gamallo, P. (2021). CiTIUS at FakeDeS 2021: A Hybrid Strategy for Fake News Detection. CEUR-WS Vol. 2943, pp. 609–614. https://ceur-ws.org/Vol-2943/fakedes_paper2.pdf
4. Zhao, K., Zhou, S. y Li, W. (2021). zk15120170770 at FakeDeS 2021: Fake news detection based on Pre-training Model. CEUR-WS Vol. 2943, pp. 655–660. https://ceur-ws.org/Vol-2943/fakedes_paper8.pdf
5. Reyes-Magaña, J. y Argota Vega, L. E. (2021). ForceNLP at FakeDeS 2021: Analysis of Text Features Applied to Fake News Detection in Spanish. CEUR-WS Vol. 2943, pp. 639–645. https://ceur-ws.org/Vol-2943/fakedes_paper6.pdf
6. Guan, Z. (2021). TSIA team at FakeDeS 2021: Fake News Detection in Spanish Using Multi-Model Ensemble Learning. CEUR-WS Vol. 2943, pp. 661–667. https://ceur-ws.org/Vol-2943/fakedes_paper9.pdf
7. Spalenza, M. A., Lusquino-Filho, L., Franca, F. M. G., Lima, P. M. V. y de Oliveira, E. (2021). LCAD - UFES at FakeDeS 2021: Fake News Detection Using Named Entity Recognition and Part-of-Speech Sequences. CEUR-WS Vol. 2943, pp. 646–654. https://ceur-ws.org/Vol-2943/fakedes_paper7.pdf
8. Luo, H. (2021). YETI at FakeDeS 2021: Fake News Detection in Spanish with ALBERT. CEUR-WS Vol. 2943, pp. 615–620. https://ceur-ws.org/Vol-2943/fakedes_paper3.pdf
9. Lomas Barrie, V., Perez, N., Lara, V. M. y Neme, A. (2021). Bribones tras la esmeralda perdida@FakeDeS 2021: Fake news detection based on random forests, k-nearest neighbors, and n-grams for a Spanish corpora. CEUR-WS Vol. 2943, pp. 599–608. https://ceur-ws.org/Vol-2943/fakedes_paper1.pdf
10. Gómez-Adorno, H., Posadas-Durán, J. P., Bel Enguix, G. y Porto Capetillo, C. (2021). Overview of FakeDeS at IberLEF 2021: Fake News Detection in Spanish Shared Task. *Procesamiento del Lenguaje Natural*, 67, 223–231. DOI 10.26342/2021-67-19. ISSN 1135-5948.
11. Kiesel, J., Mestre, M., Shukla, R., Vincent, E., Adineh, P., Corney, D., Stein, B. y Potthast, M. (2019). SemEval-2019 Task 4: Hyperpartisan News Detection. En *Proceedings of SemEval-2019*. https://aclanthology.org/S19-2145/ (DOI 10.18653/v1/S19-2145). Solo ficha bibliográfica consultada.
