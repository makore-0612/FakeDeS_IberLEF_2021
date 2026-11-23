# Comprensión de los datos (CRISP-DM, fase 2)

Notebook: `analisis_exploratorio.ipynb`, Secciones 3 a 16. Fecha: 22 de noviembre de 2026. No se modificó ningún archivo de `data/`.

## Qué se agregó al EDA existente
Diccionario de datos con roles (3.4), verificación de archivos (3.5), reglas de validez (5.6), V de Cramér (8.4), Mann-Whitney con tamaño de efecto y Bonferroni (9.3, 10.3), atípicos de longitud (9.4), casi duplicados y pares (12), atípicos del texto (13), el medio dentro del texto (14) y el reporte por dimensiones de calidad (16). Se corrigió la errata "CRISM-DM" y se reorganizó el notebook con secciones numeradas.

## Números clave
| Hallazgo | Cifra | Sección |
|---|---|---|
| Titulares faltantes, todos en falsas de test | 72 de 286 falsas | 5.2 |
| Enlaces "Perma \| …" en lugar de URL, todos en falsas de test | 40 | 5.6 |
| Reglas de validez cumplidas | 7 de 8 | 5.6 |
| V de Cramér fuente–clase (train / dev / test) | 0.996 / 1.000 / 0.956 | 8.4 |
| Longitud: falsas más cortas, r (train / dev / test) | −0.37 / −0.30 / −0.44 | 9.3 |
| Rasgos de estilo significativos tras Bonferroni | 4 de 8; mayúsculas invierten el signo | 10.3 |
| Vocabulario del test ausente en la 1.0 | 45.1 % | 11.2 |
| Casi duplicados (coseno ≥ 0.8) | 11 pares: 4 train→test, 5 train→dev, 1 dentro de dev, 1 dentro de test con clase distinta | 12.1 |
| Pares verdadera–falsa reconstruidos (umbral 0.3, v1.0) | 124 (25.8 % de las falsas), 78.2 % con el mismo tema contra 18.4 % al azar | 12.3 |
| Grupo más grande si se agrupa por fuente / por fuente + par | 13.9 % / 86.1 % | 12.4 |
| Falsas de test con fuente verificador o red social | 117 de 286 | 13.2 |
| Marca del medio en el texto, v1.0 (verdaderas / falsas) | 18.9 % / 7.7 %; en la 2.0, 10.8 % / 14.3 % | 14 |

## Decisiones (se aplican en `preparacion_datos.ipynb`)
| Decisión | Alternativas descartadas | Por qué |
|---|---|---|
| Folds agrupados solo por fuente normalizada | Fuente + par; sin agrupar | Fuente + par une el 86.1 % en un grupo; sin agrupar, el modelo mide el reconocimiento del medio |
| Eliminar las 4 copias de train de noticias del test | Conservarlas, como hicieron los participantes | Fuga directa hacia la evaluación final |
| Conservar textos largos y cortos | Eliminar atípicos por IQR | Los atípicos largos son casi todos verdaderas legítimas (51 de 55 en la 1.0) |
| Enmascarar el medio (`<MEDIO>`) y comparar con y sin enmascarar | No hacer nada | La señal cambia de dirección entre versiones |
| No usar `development` como única validación | Validar en development, como proponía el EDA anterior | Comparte el 90.5 % de sus fuentes con train |

## Correcciones al EDA anterior
- "Más exclamaciones e interrogaciones en las falsas de ambas versiones": tras Mann-Whitney y Bonferroni, las exclamaciones solo distinguen en la 2.0 y las interrogaciones en ninguna.
- "Textos de test presentes en la versión 1.0: 0": era cierto para coincidencias exactas, pero hay 4 casi duplicados.
- Vocabulario nuevo del test: 45.1 % (antes 44.5 %), por la versión de `stopwordsiso`.

## Limitaciones
- La identificación de publicaciones de redes es aproximada (por fuente).
- La detección de idioma es heurística (palabras vacías) y no distingue variantes del español.
- Para correr el notebook fuera de Jupyter hay que activar el entorno (`conda activate fakedes`); sin activación, matplotlib falla al cargar sus DLL en Windows. El kernel `Python (fakedes)` ya incluye el PATH.
