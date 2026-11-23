# Siguientes pasos (para decidir con el equipo)

Fecha: 22 de noviembre de 2026. Entrega: 3 de diciembre. Todo lo que se pruebe a partir de aquí se elige en CV agrupada y se reporta en el test como **sistema aparte** (el test ya se vio).

| # | Paso | Qué haría falta | Tiempo | GPU / Colab | Paquetes nuevos | Riesgos |
|---|---|---|---|---|---|---|
| 1 | **Ajustar el umbral de decisión** del principal para subir la exhaustividad (hoy 0.566) | Elegir el umbral sobre las probabilidades de CV que maximiza la F1 de fake por fold; aplicarlo una vez en el test | 1–2 h | No | Ninguno | Ganancia pequeña si el problema es de registro y no de umbral |
| 2 | **Replicar el preprocesamiento oficial** (lematizar, quitar números y puntuación) | Lematizador en español; repetir la etapa 1 en CV | 3–4 h | No | `spacy` + `es_core_news_sm` (≈ 15 MB) | Quitar puntuación borra señal de estilo; probar ambas versiones |
| 3 | **BETO con truncado cabeza + cola** (como GDUFS_DM) | Notebook `modelo_transformer.ipynb` en Colab con los mismos folds (`resultados/folds.csv`); 5 folds × 3–4 épocas | 1 día (≈ 2–3 h de GPU) | **Sí** (Colab T4 basta) | `transformers`, `torch`, `datasets` (en Colab) | Con 961 noticias es inestable entre semillas; repetir con 3 semillas. BETO oficial: .7321 |
| 4 | **Ensamble** texto + estilo (+ BETO) | Promediar probabilidades o *stacking* con pesos elegidos en CV (predicciones fuera de fold) | 3–4 h (sin BETO) | Solo si incluye BETO | Ninguno | El oráculo texto + estilo llega al 84.7 % en CV, pero un ensamble real gana mucho menos |
| 4b | **Variante sin el marcador `*NUMBER*`** (o sin cifras en ambas versiones) | Quitar el marcador del texto y `est_numeros` de los rasgos; repetir en CV la configuración elegida | 1 h | No | Ninguno | En train las cifras son señal real (verdaderas citan 3 veces más); quitarlas puede bajar la CV aunque mejore el test |
| 5 | **API de demostración** | FastAPI local o Gradio en Hugging Face Spaces; cargar `modelos/modelo_principal.joblib` y aplicar `uf.preparar_noticia` con el léxico de `data/processed/lexico_medios.csv` | 3–4 h | No | `fastapi` + `uvicorn` o `gradio` | Publicar requiere crear cuenta (preguntar antes); sin datos ni claves en el código; aclarar en la interfaz que es un clasificador de estilo, no un verificador |
| 6 | Exportar el entorno y la hoja `resultados/reproducibilidad.xlsx` (Fase 10) | `conda env export --no-builds > environment.yml`; hojas de configuración, hiperparámetros, resultados por fold, resumen y faltantes | 2 h | No | Ninguno | — |
| 7 | PDF de estudio y reporte final | A partir de las notas y los notebooks | 1 día | No | Ninguno | — |

**Recomendación de orden:** 1 → 5 → 6 → 2 → 3 → 4 → 7. El umbral y la API son baratos y cubren entregables de la propuesta; BETO es el único paso con posibilidad realista de superar .7217, pero depende de Colab.
