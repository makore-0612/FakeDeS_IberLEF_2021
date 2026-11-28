'''
Entrenamiento de BETO (dccuchile/bert-base-spanish-wwm-uncased) para FakeDeS 2021

Produce las probabilidades de 'Fake' out-of-fold (5 folds agrupados por fuente × 3 semillas) y las del test
(3 semillas entrenadas con las 961 noticias). Cada predicción se guarda apenas termina en resultados/beto_parciales/;
si el script se interrumpe, al relanzarlo salta lo que ya existe. No guarda los pesos del modelo.

Uso (con el entorno fakedes activado, desde la carpeta del repo):
    python beto_entrenar.py --prueba     # 1 época del fold 0 con la semilla 42: mide tiempo y memoria
    python beto_entrenar.py              # los 18 entrenamientos (15 out-of-fold + 3 del test)

Con --modelo y --prefijo se entrena otro transformer con exactamente el mismo protocolo, por ejemplo MarIA:
    python beto_entrenar.py --modelo PlanTL-GOB-ES/roberta-base-bne --prefijo maria
Sin esos argumentos se usa BETO y los archivos se llaman beto_*. Para modelos con vocabulario muy grande
(XLM-RoBERTa, 250 mil tokens) se agrega --congelar_embeddings: la tabla de embeddings de palabras no se ajusta
y todo lo demás sí; sin eso el optimizador no cabe en 4 GB de GPU.
'''
import argparse
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

# ---------------------------------------------------------------- Configuración (fija, sin grid search)
MODELO = 'dccuchile/bert-base-spanish-wwm-uncased'
LONGITUD = 512                     # tokens totales, incluidos [CLS] y [SEP]
CABEZA = 128                       # primeros tokens del texto
COLA = 382                         # últimos tokens del texto (128 + 382 + 2 especiales = 512)
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP = 0.1                       # fracción de pasos con calentamiento lineal
EPOCAS = 4
LOTE = 2                           # noticias por paso (4 GB de VRAM)
ACUMULACION = 8                    # lote efectivo = LOTE × ACUMULACION = 16
LOTE_PREDICCION = 16
SEMILLAS = [42, 43, 44]
FOLDS = [0, 1, 2, 3, 4]

RUTA_PROCESADOS = os.path.join('data', 'processed')
RUTA_RESULTADOS = 'resultados'
RUTA_PARCIALES = os.path.join(RUTA_RESULTADOS, 'beto_parciales')
RUTA_LOG = os.path.join(RUTA_RESULTADOS, 'beto_log.csv')
PREFIJO = 'beto'
CONGELAR_EMBEDDINGS = False      # solo se activa con --congelar_embeddings (XLM-RoBERTa)


def fijar_semilla(semilla):
    '''
    Función que fija la semilla de Python, NumPy y PyTorch

    inputs:
    --------
        semilla (int): Semilla del entrenamiento

    outputs:
    --------
        None
    '''
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.cuda.manual_seed_all(semilla)


def cargar_datos():
    '''
    Función que lee solo las columnas necesarias: el corpus con etiquetas y el test sin etiquetas

    inputs:
    --------
        Ninguno

    outputs:
    --------
        tuple: (corpus con id_unico, texto_sin_medio, y, fold_grupo; test con id_unico y texto_sin_medio)
    '''
    corpus = pd.read_parquet(os.path.join(RUTA_PROCESADOS, 'corpus.parquet'),
                             columns = ['id_unico', 'texto_sin_medio', 'y', 'fold_grupo'])
    test = pd.read_parquet(os.path.join(RUTA_PROCESADOS, 'test.parquet'),
                           columns = ['id_unico', 'texto_sin_medio'])
    return corpus, test


def tokenizar_cabeza_cola(texto, tokenizador):
    '''
    Función que convierte un texto en ids de BETO con truncado cabeza + cola (Sun et al., 2019)

    inputs:
    --------
        texto (str): Texto preparado de la noticia
        tokenizador (transformers tokenizer): Tokenizador de BETO

    outputs:
    --------
        list: Ids con [CLS] al inicio y [SEP] al final, de largo máximo LONGITUD
    '''
    ids = tokenizador.encode(texto, add_special_tokens = False)
    if len(ids) > CABEZA + COLA:
        ids = ids[:CABEZA] + ids[-COLA:]
    return [tokenizador.cls_token_id] + ids + [tokenizador.sep_token_id]


def tokenizar_todos(textos, tokenizador):
    '''
    Función que tokeniza una lista de textos con truncado cabeza + cola

    inputs:
    --------
        textos (list): Textos preparados
        tokenizador (transformers tokenizer): Tokenizador de BETO

    outputs:
    --------
        list: Una lista de ids por texto
    '''
    todos = []
    for texto in textos:
        todos.append(tokenizar_cabeza_cola(texto, tokenizador))
    return todos


def armar_lote(ejemplos, id_relleno):
    '''
    Función que rellena un lote de ids al largo del más largo y arma la máscara de atención

    inputs:
    --------
        ejemplos (list): Tuplas (ids, etiqueta); la etiqueta puede ser -1 si no se conoce
        id_relleno (int): Id del token de relleno

    outputs:
    --------
        dict: 'input_ids', 'attention_mask' y 'labels' como tensores
    '''
    largo = 0
    for ids, _ in ejemplos:
        largo = max(largo, len(ids))
    entradas, mascaras, etiquetas = [], [], []
    for ids, etiqueta in ejemplos:
        relleno = largo - len(ids)
        entradas.append(ids + [id_relleno] * relleno)
        mascaras.append([1] * len(ids) + [0] * relleno)
        etiquetas.append(etiqueta)
    return {'input_ids': torch.tensor(entradas), 'attention_mask': torch.tensor(mascaras),
            'labels': torch.tensor(etiquetas)}


def pesos_clase(y):
    '''
    Función que calcula pesos balanceados por clase, como class_weight='balanced' de scikit-learn

    inputs:
    --------
        y (numpy.ndarray): Etiquetas de entrenamiento (1 = Fake, 0 = True)

    outputs:
    --------
        torch.Tensor: Peso de la clase 0 y de la clase 1
    '''
    n = len(y)
    n_true = int((y == 0).sum())
    n_fake = int((y == 1).sum())
    return torch.tensor([n / (2 * n_true), n / (2 * n_fake)], dtype = torch.float)


def entrenar(ids_entrenamiento, y_entrenamiento, semilla, epocas, tokenizador, dispositivo):
    '''
    Función que ajusta BETO con AdamW, calentamiento lineal, fp16 y acumulación de gradiente

    inputs:
    --------
        ids_entrenamiento (list): Ids tokenizados de cada noticia
        y_entrenamiento (numpy.ndarray): Etiquetas (1 = Fake, 0 = True)
        semilla (int): Semilla del entrenamiento
        epocas (int): Número de épocas
        tokenizador (transformers tokenizer): Tokenizador de BETO
        dispositivo (torch.device): GPU o CPU

    outputs:
    --------
        tuple: (modelo entrenado, lista con la pérdida media de cada época)
    '''
    fijar_semilla(semilla)
    modelo = AutoModelForSequenceClassification.from_pretrained(MODELO, num_labels = 2)
    if CONGELAR_EMBEDDINGS:
        for parametro in modelo.get_input_embeddings().parameters():
            parametro.requires_grad = False          # la tabla de palabras no se ajusta (para caber en 4 GB)
    modelo.gradient_checkpointing_enable()
    modelo.to(dispositivo)

    ejemplos = list(zip(ids_entrenamiento, [int(v) for v in y_entrenamiento]))
    generador = torch.Generator()
    generador.manual_seed(semilla)
    cargador = DataLoader(ejemplos, batch_size = LOTE, shuffle = True, generator = generador, num_workers = 0,
                          collate_fn = lambda lote: armar_lote(lote, tokenizador.pad_token_id))

    sin_decaimiento = ['bias', 'LayerNorm.weight']
    grupo_con, grupo_sin = [], []
    for nombre, parametro in modelo.named_parameters():
        if not parametro.requires_grad:
            continue
        if any(s in nombre for s in sin_decaimiento):
            grupo_sin.append(parametro)
        else:
            grupo_con.append(parametro)
    optimizador = torch.optim.AdamW([{'params': grupo_con, 'weight_decay': WEIGHT_DECAY},
                                     {'params': grupo_sin, 'weight_decay': 0.0}], lr = LR)
    pasos_totales = int(np.ceil(len(cargador) / ACUMULACION)) * epocas
    programador = get_linear_schedule_with_warmup(optimizador, num_warmup_steps = int(WARMUP * pasos_totales),
                                                  num_training_steps = pasos_totales)
    escalador = torch.amp.GradScaler('cuda')
    perdida_fn = torch.nn.CrossEntropyLoss(weight = pesos_clase(y_entrenamiento).to(dispositivo))

    perdidas = []
    modelo.train()
    for epoca in range(epocas):
        suma, cuenta = 0.0, 0
        optimizador.zero_grad()
        for i, lote in enumerate(cargador):
            lote = {k: v.to(dispositivo) for k, v in lote.items()}
            with torch.autocast(device_type = 'cuda', dtype = torch.float16):
                salida = modelo(input_ids = lote['input_ids'], attention_mask = lote['attention_mask'])
                perdida = perdida_fn(salida.logits.float(), lote['labels'])
            escalador.scale(perdida / ACUMULACION).backward()
            suma += perdida.item() * len(lote['labels'])
            cuenta += len(lote['labels'])
            if (i + 1) % ACUMULACION == 0 or (i + 1) == len(cargador):
                escalador.unscale_(optimizador)
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
                escalador.step(optimizador)
                escalador.update()
                programador.step()
                optimizador.zero_grad()
        perdidas.append(suma / cuenta)
        print(f'    época {epoca + 1}: pérdida {perdidas[-1]:.4f}', flush = True)
    return modelo, perdidas


def predecir(modelo, ids, tokenizador, dispositivo):
    '''
    Función que calcula la probabilidad de 'Fake' de cada noticia sin guardar activaciones

    inputs:
    --------
        modelo (transformers model): BETO entrenado
        ids (list): Ids tokenizados de cada noticia
        tokenizador (transformers tokenizer): Tokenizador de BETO
        dispositivo (torch.device): GPU o CPU

    outputs:
    --------
        numpy.ndarray: Probabilidad de 'Fake' (clase 1) de cada noticia, en el mismo orden
    '''
    modelo.eval()
    ejemplos = list(zip(ids, [-1] * len(ids)))
    cargador = DataLoader(ejemplos, batch_size = LOTE_PREDICCION, shuffle = False, num_workers = 0,
                          collate_fn = lambda lote: armar_lote(lote, tokenizador.pad_token_id))
    probabilidades = []
    with torch.no_grad():
        for lote in cargador:
            with torch.autocast(device_type = 'cuda', dtype = torch.float16):
                salida = modelo(input_ids = lote['input_ids'].to(dispositivo),
                                attention_mask = lote['attention_mask'].to(dispositivo))
            prob = torch.softmax(salida.logits.float(), dim = 1)[:, 1]
            probabilidades.append(prob.cpu().numpy())
    return np.concatenate(probabilidades)


def anotar_log(fila):
    '''
    Función que agrega un renglón a la bitácora de entrenamientos apenas termina uno

    inputs:
    --------
        fila (dict): fold, semilla, minutos, pérdidas por época y memoria máxima

    outputs:
    --------
        None
    '''
    existe = os.path.exists(RUTA_LOG)
    pd.DataFrame([fila]).to_csv(RUTA_LOG, mode = 'a', header = not existe, index = False, encoding = 'utf-8')


def correr_pieza(nombre, ids_ent, y_ent, ids_pred, id_pred, semilla, fold, tokenizador, dispositivo):
    '''
    Función que entrena una pieza (fold o test con una semilla), predice y guarda su CSV parcial y su renglón de bitácora

    inputs:
    --------
        nombre (str): Nombre del archivo parcial sin extensión
        ids_ent (list): Ids de las noticias de entrenamiento
        y_ent (numpy.ndarray): Etiquetas de entrenamiento
        ids_pred (list): Ids de las noticias a predecir
        id_pred (numpy.ndarray): id_unico de las noticias a predecir
        semilla (int): Semilla
        fold (int o str): Fold que se predice ('test' para el test)
        tokenizador (transformers tokenizer): Tokenizador de BETO
        dispositivo (torch.device): GPU

    outputs:
    --------
        None
    '''
    ruta = os.path.join(RUTA_PARCIALES, f'{nombre}.csv')
    if os.path.exists(ruta):
        print(f'{nombre}: ya existe, se salta', flush = True)
        return
    print(f'{nombre}: entrenando con {len(ids_ent)} noticias', flush = True)
    torch.cuda.reset_peak_memory_stats()
    inicio = time.time()
    modelo, perdidas = entrenar(ids_ent, y_ent, semilla, EPOCAS, tokenizador, dispositivo)
    prob = predecir(modelo, ids_pred, tokenizador, dispositivo)
    minutos = (time.time() - inicio) / 60
    memoria = torch.cuda.max_memory_allocated() / 1024 ** 3
    pd.DataFrame({'id_unico': id_pred, 'prob': prob}).to_csv(ruta, index = False, encoding = 'utf-8')
    fila = {'fold': fold, 'semilla': semilla, 'minutos': round(minutos, 2), 'perdida_final': round(perdidas[-1], 5),
            'memoria_max_gb': round(memoria, 3)}
    for e in range(EPOCAS):
        fila[f'perdida_epoca_{e + 1}'] = round(perdidas[e], 5)
    anotar_log(fila)
    print(f'{nombre}: listo en {minutos:.1f} min; memoria máxima {memoria:.2f} GB', flush = True)
    del modelo
    torch.cuda.empty_cache()


def armar_salidas(corpus, test):
    '''
    Función que junta las piezas parciales en <prefijo>_oof.csv y <prefijo>_test.csv (promedio de las 3 semillas)

    inputs:
    --------
        corpus (pandas.DataFrame): Corpus con id_unico y fold_grupo
        test (pandas.DataFrame): Test con id_unico

    outputs:
    --------
        bool: True si estaban todas las piezas y se escribieron las salidas
    '''
    oof = corpus[['id_unico', 'fold_grupo']].copy()
    for semilla in SEMILLAS:
        partes = []
        for k in FOLDS:
            ruta = os.path.join(RUTA_PARCIALES, f'oof_fold{k}_semilla{semilla}.csv')
            if not os.path.exists(ruta):
                return False
            partes.append(pd.read_csv(ruta))
        prob = pd.concat(partes).set_index('id_unico')['prob']
        oof[f'prob_s{semilla}'] = oof['id_unico'].map(prob).values
    oof[f'prob_{PREFIJO}'] = oof[[f'prob_s{s}' for s in SEMILLAS]].mean(axis = 1)

    salida_test = test[['id_unico']].copy()
    for semilla in SEMILLAS:
        ruta = os.path.join(RUTA_PARCIALES, f'test_semilla{semilla}.csv')
        if not os.path.exists(ruta):
            return False
        prob = pd.read_csv(ruta).set_index('id_unico')['prob']
        salida_test[f'prob_s{semilla}'] = salida_test['id_unico'].map(prob).values
    salida_test[f'prob_{PREFIJO}'] = salida_test[[f'prob_s{s}' for s in SEMILLAS]].mean(axis = 1)

    oof.to_csv(os.path.join(RUTA_RESULTADOS, f'{PREFIJO}_oof.csv'), index = False, encoding = 'utf-8')
    salida_test.to_csv(os.path.join(RUTA_RESULTADOS, f'{PREFIJO}_test.csv'), index = False, encoding = 'utf-8')
    return True


def prueba_de_tiempo(corpus, tokenizador, dispositivo):
    '''
    Función que entrena una sola época del fold 0 con la semilla 42 y mide minutos y memoria máxima

    inputs:
    --------
        corpus (pandas.DataFrame): Corpus con id_unico, texto_sin_medio, y y fold_grupo
        tokenizador (transformers tokenizer): Tokenizador de BETO
        dispositivo (torch.device): GPU

    outputs:
    --------
        None
    '''
    dentro = corpus[corpus['fold_grupo'] != 0]
    afuera = corpus[corpus['fold_grupo'] == 0]
    ids_dentro = tokenizar_todos(dentro['texto_sin_medio'].tolist(), tokenizador)
    ids_afuera = tokenizar_todos(afuera['texto_sin_medio'].tolist(), tokenizador)
    torch.cuda.reset_peak_memory_stats()
    inicio = time.time()
    modelo, perdidas = entrenar(ids_dentro, dentro['y'].values, 42, 1, tokenizador, dispositivo)
    minutos_epoca = (time.time() - inicio) / 60
    inicio = time.time()
    predecir(modelo, ids_afuera, tokenizador, dispositivo)
    minutos_pred = (time.time() - inicio) / 60
    memoria = torch.cuda.max_memory_allocated() / 1024 ** 3
    por_entrenamiento = EPOCAS * minutos_epoca + minutos_pred
    print(f'Prueba: 1 época con {len(dentro)} noticias en {minutos_epoca:.1f} min; predicción de {len(afuera)} en {minutos_pred:.2f} min')
    print(f'Memoria máxima de GPU: {memoria:.2f} GB')
    print(f'Estimación: {por_entrenamiento:.1f} min por entrenamiento de {EPOCAS} épocas; '
          f'total de 18 entrenamientos ≈ {18 * por_entrenamiento / 60:.1f} h '
          f'(los 3 del test usan 961 noticias en vez de ≈ 770, así que tardan un poco más)')


def main():
    '''
    Función principal: prueba de tiempo o corrida completa con reanudación

    inputs:
    --------
        Ninguno (lee --prueba, --modelo y --prefijo de la línea de comandos)

    outputs:
    --------
        None
    '''
    global MODELO, RUTA_PARCIALES, RUTA_LOG, PREFIJO, CONGELAR_EMBEDDINGS
    argumentos = argparse.ArgumentParser()
    argumentos.add_argument('--prueba', action = 'store_true', help = '1 época del fold 0 con semilla 42')
    argumentos.add_argument('--modelo', default = MODELO, help = 'Modelo de Hugging Face (por omisión, BETO)')
    argumentos.add_argument('--prefijo', default = 'beto', help = 'Prefijo de los archivos de salida (por omisión, beto)')
    argumentos.add_argument('--congelar_embeddings', action = 'store_true',
                            help = 'No ajustar la tabla de embeddings de palabras (para modelos con vocabulario grande)')
    leidos = argumentos.parse_args()
    CONGELAR_EMBEDDINGS = leidos.congelar_embeddings
    prueba = leidos.prueba
    MODELO = leidos.modelo
    PREFIJO = leidos.prefijo
    RUTA_PARCIALES = os.path.join(RUTA_RESULTADOS, f'{PREFIJO}_parciales')
    RUTA_LOG = os.path.join(RUTA_RESULTADOS, f'{PREFIJO}_log.csv')
    print(f'Modelo: {MODELO}; archivos: {PREFIJO}_*; embeddings de palabras congelados: {CONGELAR_EMBEDDINGS}', flush = True)

    if not torch.cuda.is_available():
        sys.exit('No hay GPU con CUDA disponible; ver el plan B (Colab) en TAREA_STACKING_BETO.md')
    dispositivo = torch.device('cuda')
    print(f'GPU: {torch.cuda.get_device_name(0)}; torch {torch.__version__}', flush = True)

    corpus, test = cargar_datos()
    tokenizador = AutoTokenizer.from_pretrained(MODELO)
    if prueba:
        prueba_de_tiempo(corpus, tokenizador, dispositivo)
        return

    os.makedirs(RUTA_PARCIALES, exist_ok = True)
    ids_corpus = tokenizar_todos(corpus['texto_sin_medio'].tolist(), tokenizador)
    ids_test = tokenizar_todos(test['texto_sin_medio'].tolist(), tokenizador)
    y = corpus['y'].values
    folds = corpus['fold_grupo'].values

    for semilla in SEMILLAS:
        for k in FOLDS:
            dentro = np.where(folds != k)[0]
            afuera = np.where(folds == k)[0]
            ids_dentro = [ids_corpus[i] for i in dentro]
            ids_afuera = [ids_corpus[i] for i in afuera]
            correr_pieza(f'oof_fold{k}_semilla{semilla}', ids_dentro, y[dentro], ids_afuera,
                         corpus['id_unico'].values[afuera], semilla, k, tokenizador, dispositivo)
    for semilla in SEMILLAS:
        correr_pieza(f'test_semilla{semilla}', ids_corpus, y, ids_test, test['id_unico'].values,
                     semilla, 'test', tokenizador, dispositivo)

    if armar_salidas(corpus, test):
        print(f'Listo: resultados/{PREFIJO}_oof.csv y resultados/{PREFIJO}_test.csv', flush = True)


if __name__ == '__main__':
    main()
