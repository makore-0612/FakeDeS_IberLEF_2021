'''
Prueba de contaminación de los transformers preentrenados (BETO, BERTIN y XLM-RoBERTa) para FakeDeS 2021

Idea: si un modelo vio un texto durante su preentrenamiento, adivina sus palabras tapadas mejor de lo normal.
A cada modelo preentrenado (sin ajuste fino) se le tapa el 15 % de los tokens de cada noticia (mismo truncado
cabeza + cola que en el entrenamiento) y se mide la pérdida y el acierto top-1 en esos tokens.
Después, en ensamble_beto.ipynb, se comparan grupos de noticias (El Ruinaversal, otras falsas, verdaderas y test).

No usa etiquetas del test: del test solo lee id_unico y texto_sin_medio. Escribe resultados/contaminacion_mlm.csv.

Uso (con el entorno fakedes activado, desde la carpeta del repo):
    python contaminacion_mlm.py
'''
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

MODELOS = {'BETO': 'dccuchile/bert-base-spanish-wwm-uncased',
           'BERTIN': 'bertin-project/bertin-roberta-base-spanish',
           'XLM-RoBERTa': 'FacebookAI/xlm-roberta-base'}
CABEZA = 128
COLA = 382
PROPORCION = 0.15
LOTES = {'BETO': 8, 'BERTIN': 8, 'XLM-RoBERTa': 1}   # XLM-R: 250 mil tokens de vocabulario, sus predicciones no caben en 4 GB con lotes de 8
SEMILLA = 42
RUTA_PROCESADOS = os.path.join('data', 'processed')
RUTA_SALIDA = os.path.join('resultados', 'contaminacion_mlm.csv')


def tokenizar_cabeza_cola(texto, tokenizador):
    '''
    Función que convierte un texto en ids con truncado cabeza + cola (igual que beto_entrenar.py)

    inputs:
    --------
        texto (str): Texto preparado de la noticia
        tokenizador (transformers tokenizer): Tokenizador del modelo

    outputs:
    --------
        list: Ids con los tokens especiales de inicio y fin
    '''
    ids = tokenizador.encode(texto, add_special_tokens = False)
    if len(ids) > CABEZA + COLA:
        ids = ids[:CABEZA] + ids[-COLA:]
    return [tokenizador.cls_token_id] + ids + [tokenizador.sep_token_id]


def posiciones_a_tapar(n_tokens, numero):
    '''
    Función que elige al azar (con semilla fija por noticia) el 15 % de las posiciones que no son especiales

    inputs:
    --------
        n_tokens (int): Largo de la secuencia con tokens especiales
        numero (int): Número de la noticia (para la semilla)

    outputs:
    --------
        numpy.ndarray: Posiciones a tapar
    '''
    rng = np.random.default_rng(SEMILLA + numero)
    candidatas = np.arange(1, n_tokens - 1)
    cuantas = max(1, int(round(PROPORCION * len(candidatas))))
    return np.sort(rng.choice(candidatas, size = cuantas, replace = False))


def medir(textos, tokenizador, modelo, dispositivo, lote):
    '''
    Función que tapa el 15 % de los tokens de cada texto y mide pérdida y acierto top-1 del modelo en ellos

    inputs:
    --------
        textos (list): Textos preparados
        tokenizador (transformers tokenizer): Tokenizador del modelo
        modelo (transformers model): Modelo de lenguaje enmascarado preentrenado
        dispositivo (torch.device): GPU
        lote (int): Noticias por paso

    outputs:
    --------
        tuple: (pérdida media por texto, acierto top-1 por texto, tokens tapados por texto)
    '''
    perdidas, aciertos, tapados = [], [], []
    for inicio in range(0, len(textos), lote):
        entradas, mascaras, objetivos = [], [], []
        for numero in range(inicio, min(inicio + lote, len(textos))):
            ids = tokenizar_cabeza_cola(textos[numero], tokenizador)
            posiciones = posiciones_a_tapar(len(ids), numero)
            objetivo = [-100] * len(ids)
            for p in posiciones:
                objetivo[p] = ids[p]
                ids[p] = tokenizador.mask_token_id
            entradas.append(ids)
            objetivos.append(objetivo)
        largo = 0
        for ids in entradas:
            largo = max(largo, len(ids))
        for i in range(len(entradas)):
            relleno = largo - len(entradas[i])
            mascaras.append([1] * len(entradas[i]) + [0] * relleno)
            entradas[i] = entradas[i] + [tokenizador.pad_token_id] * relleno
            objetivos[i] = objetivos[i] + [-100] * relleno
        entradas = torch.tensor(entradas).to(dispositivo)
        mascaras = torch.tensor(mascaras).to(dispositivo)
        objetivos = torch.tensor(objetivos).to(dispositivo)
        with torch.no_grad():
            with torch.autocast(device_type = 'cuda', dtype = torch.float16):
                logits = modelo(input_ids = entradas, attention_mask = mascaras).logits.float()
        perdida_token = torch.nn.functional.cross_entropy(logits.transpose(1, 2), objetivos, ignore_index = -100, reduction = 'none')
        tapado = objetivos != -100
        acierto_token = (logits.argmax(dim = -1) == objetivos) & tapado
        for i in range(len(entradas)):
            n = int(tapado[i].sum())
            perdidas.append(float(perdida_token[i][tapado[i]].mean()))
            aciertos.append(float(acierto_token[i].sum()) / n)
            tapados.append(n)
    return perdidas, aciertos, tapados


def main():
    '''
    Función principal: mide los tres modelos en el corpus y el test y guarda una fila por noticia y modelo

    inputs:
    --------
        Ninguno

    outputs:
    --------
        None
    '''
    if not torch.cuda.is_available():
        sys.exit('No hay GPU con CUDA disponible')
    dispositivo = torch.device('cuda')
    corpus = pd.read_parquet(os.path.join(RUTA_PROCESADOS, 'corpus.parquet'), columns = ['id_unico', 'texto_sin_medio'])
    test = pd.read_parquet(os.path.join(RUTA_PROCESADOS, 'test.parquet'),
                           columns = ['id_unico', 'texto_sin_medio'])
    partes = []
    for nombre, ruta in MODELOS.items():
        inicio = time.time()
        tokenizador = AutoTokenizer.from_pretrained(ruta)
        modelo = AutoModelForMaskedLM.from_pretrained(ruta).to(dispositivo)
        modelo.eval()
        for conjunto, datos in [('corpus', corpus), ('test', test)]:
            perdidas, aciertos, tapados = medir(datos['texto_sin_medio'].tolist(), tokenizador, modelo, dispositivo, LOTES[nombre])
            partes.append(pd.DataFrame({'id_unico': datos['id_unico'].values, 'conjunto': conjunto, 'modelo': nombre,
                                        'tokens_tapados': tapados, 'perdida': perdidas, 'acierto_top1': aciertos}))
        print(f'{nombre}: listo en {(time.time() - inicio) / 60:.1f} min', flush = True)
        del modelo
        torch.cuda.empty_cache()
    pd.concat(partes, ignore_index = True).to_csv(RUTA_SALIDA, index = False, encoding = 'utf-8')
    print(f'Listo: {RUTA_SALIDA}', flush = True)


if __name__ == '__main__':
    main()
