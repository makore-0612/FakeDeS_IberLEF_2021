'''
Embeddings de BETO sin ajustar (dccuchile/bert-base-spanish-wwm-uncased) para FakeDeS 2021

Para cada noticia calcula el promedio de los vectores de la última capa de BETO preentrenado (sin ajuste fino),
con el mismo truncado cabeza + cola de beto_entrenar.py. No usa etiquetas: es una representación del texto, como el TF-IDF.
Del test solo lee id_unico y texto_sin_medio. Escribe resultados/beto_embeddings.npz.

Uso (con el entorno fakedes activado, desde la carpeta del repo):
    python beto_embeddings.py
'''
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

MODELO = 'dccuchile/bert-base-spanish-wwm-uncased'
CABEZA = 128
COLA = 382
LOTE = 16
RUTA_PROCESADOS = os.path.join('data', 'processed')
RUTA_SALIDA = os.path.join('resultados', 'beto_embeddings.npz')


def tokenizar_cabeza_cola(texto, tokenizador):
    '''
    Función que convierte un texto en ids con truncado cabeza + cola (igual que beto_entrenar.py)

    inputs:
    --------
        texto (str): Texto preparado de la noticia
        tokenizador (transformers tokenizer): Tokenizador de BETO

    outputs:
    --------
        list: Ids con [CLS] al inicio y [SEP] al final
    '''
    ids = tokenizador.encode(texto, add_special_tokens = False)
    if len(ids) > CABEZA + COLA:
        ids = ids[:CABEZA] + ids[-COLA:]
    return [tokenizador.cls_token_id] + ids + [tokenizador.sep_token_id]


def embeddings(textos, tokenizador, modelo, dispositivo):
    '''
    Función que calcula el promedio de los vectores de la última capa de cada texto

    inputs:
    --------
        textos (list): Textos preparados
        tokenizador (transformers tokenizer): Tokenizador de BETO
        modelo (transformers model): BETO preentrenado
        dispositivo (torch.device): GPU o CPU

    outputs:
    --------
        numpy.ndarray: Matriz (textos × 768)
    '''
    todos = []
    for inicio in range(0, len(textos), LOTE):
        lote = []
        for texto in textos[inicio:inicio + LOTE]:
            lote.append(tokenizar_cabeza_cola(texto, tokenizador))
        largo = 0
        for ids in lote:
            largo = max(largo, len(ids))
        entradas, mascaras = [], []
        for ids in lote:
            relleno = largo - len(ids)
            entradas.append(ids + [tokenizador.pad_token_id] * relleno)
            mascaras.append([1] * len(ids) + [0] * relleno)
        entradas = torch.tensor(entradas).to(dispositivo)
        mascaras = torch.tensor(mascaras).to(dispositivo)
        with torch.no_grad():
            with torch.autocast(device_type = 'cuda', dtype = torch.float16):
                salida = modelo(input_ids = entradas, attention_mask = mascaras).last_hidden_state.float()
        suma = (salida * mascaras.unsqueeze(-1)).sum(dim = 1)
        todos.append((suma / mascaras.sum(dim = 1, keepdim = True)).cpu().numpy())
    return np.concatenate(todos)


def main():
    '''
    Función principal: calcula y guarda los embeddings del corpus y del test

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
    tokenizador = AutoTokenizer.from_pretrained(MODELO)
    modelo = AutoModel.from_pretrained(MODELO).to(dispositivo)
    modelo.eval()
    inicio = time.time()
    X_corpus = embeddings(corpus['texto_sin_medio'].tolist(), tokenizador, modelo, dispositivo)
    X_test = embeddings(test['texto_sin_medio'].tolist(), tokenizador, modelo, dispositivo)
    np.savez_compressed(RUTA_SALIDA, ids_corpus = np.array(corpus['id_unico'].tolist(), dtype = str), X_corpus = X_corpus,
                        ids_test = np.array(test['id_unico'].tolist(), dtype = str), X_test = X_test)
    print(f'Embeddings: corpus {X_corpus.shape}, test {X_test.shape}, en {(time.time() - inicio) / 60:.1f} min -> {RUTA_SALIDA}')


if __name__ == '__main__':
    main()
