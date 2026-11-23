# Preparación de los datos (CRISP-DM, fase 3)

Notebook: `preparacion_datos.ipynb`. Fecha: 22 de noviembre de 2026. Los `.xlsx` de `data/` no se modificaron.

## Salidas
| Archivo | Contenido |
|---|---|
| `data/processed/corpus.parquet` | 961 noticias de train + development preparadas, con `fold_grupo` y `fold_aleatorio` |
| `data/processed/test.parquet` | 572 noticias de test preparadas (**solo** para `evaluacion.ipynb`) |
| `data/processed/lexico_medios.csv` | 235 candidatos a nombre de medio; 222 incluidos |
| `resultados/folds.csv` | Fold de cada noticia (agrupado por fuente y aleatorio), semilla 42 |
| `resultados/preparacion_bitacora.csv` | Bitácora de calidad |
| `utils_fakedes.py` | `preparar_noticia` y `rasgos_estilo`: la misma preparación para el modelado, la evaluación y la API |

## Tamaño antes y después
| | Antes | Después |
|---|---|---|
| train | 676 (338 F / 338 T) | 672 (335 / 337) |
| development | 295 (142 / 153) | 289 (139 / 150) |
| test | 572 (286 / 286) | 572 (sin cambios) |
| Pares casi duplicados entre particiones | 9 | 0 |

## Decisiones
| Decisión | Alternativas descartadas | Por qué |
|---|---|---|
| Eliminar las copias de train de 4 noticias del test y las 6 copias de development | Conservar con bandera | En CV agrupada, dos copias en folds distintos inflan el puntaje; las copias del test son fuga directa |
| Llevar el test al formato de la 1.0 (`*NUMBER*`, `*URL*`, sin titular repetido) | Quitar las marcas de la 1.0 | Los números originales de la 1.0 no existen |
| Titular vacío → cadena vacía | "sin titular" | La cadena vacía no aporta términos a TF-IDF; una frase nueva introduce palabras nunca vistas |
| **Borrar** los nombres de medio en lugar de marcarlos con `<MEDIO>` | Marcador `<MEDIO>` (lo sugería el prompt) | El marcador conserva el atajo "menciona un medio" (21.6 % contra 8.2 % en la 1.0), y el tokenizador lo convierte en la palabra "medio" |
| Léxico solo de train + development; excluir nombres que aparecen en > 2 % de noticias de otras fuentes | Usar las fuentes del test; sin filtro | Usar el test sería fuga; sin filtro se borrarían palabras como "tiempo" o "proceso" |
| No usar rasgos del titular | Incluir la longitud del titular | 72 titulares vacíos en test; diferencia no significativa en test |
| Folds `StratifiedGroupKFold` por fuente, semilla 42 | Otra semilla; partir fuentes grandes | El desbalance del fold 0 (58.0 % de falsas) es estructural: 8.7 puntos con las 30 semillas probadas |

## Limitaciones (para revisar con el equipo)
- *el país* y *el mundo* no se borran (son expresiones comunes); son medios grandes de verdaderas.
- El fold 0 tiene más falsas que los demás; la F1 por fold varía en parte por eso.
- El 80 % de los 122 pares verdadera–falsa candidatos queda repartido entre folds (igual que por azar).
- 28 textos de la 1.0 repiten el titular con variaciones y se dejaron sin recortar.
