'''
LIME de los tres finalistas (sistema 3, E3 y E4) sobre noticias del test, de punta a punta (texto -> bases -> meta-modelo)

LIME explica una predicción quitando palabras de la noticia al azar y viendo cómo cambia la probabilidad final.
Para eso hay que pasar cada texto perturbado por todas las bases con peso y luego por el meta-modelo. Como el proyecto
no guarda pesos de modelos, este script reentrena las bases con las 961 noticias (igual que para el test en
ensamble_beto.ipynb), predice los textos perturbados y descarta los modelos. No guarda pesos.

Noticias explicadas (regla fija, sobre el test): una falsa detectada por los tres finalistas, una falsa larga detectada
por los tres, una falsa que E4 rescata y el sistema 3 no, una falsa que ninguno detecta, una verdadera que los tres
marcan como falsa y una verdadera que los tres aciertan. E3 y E4 tienen la misma probabilidad (E4 solo cambia el
umbral), así que comparten la explicación.

Lee resultados/finalistas_entradas.csv y resultados/finalistas_meta.json (de ensamble_beto.ipynb).
Escribe resultados/finalistas_lime.csv, finalistas_lime_resumen.csv y finalistas_lime_verificacion.csv.

Uso (con el entorno fakedes activado, desde la carpeta del repo; necesita GPU):
    python finalistas_lime.py
'''
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from lime.lime_text import LimeTextExplainer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC
from transformers import AutoModel, AutoTokenizer
from xgboost import XGBClassifier

import utils_fakedes as uf
import beto_entrenar as be
import beto_embeddings as emb

R = uf.RUTA_RESULTADOS
N_MUESTRAS = 1000                 # textos perturbados por noticia (LIME usa 5000 por omisión; 1000 por costo de GPU)
N_PALABRAS = 12                   # palabras que se reportan por explicación
SEMILLA = uf.SEMILLA
TRANSFORMERS = {'BETO': 'dccuchile/bert-base-spanish-wwm-uncased', 'BERTIN': 'bertin-project/bertin-roberta-base-spanish'}


def a_denso(X):
    '''
    Función que convierte una matriz dispersa en densa

    inputs:
    --------
        X (scipy.sparse matrix o numpy.ndarray): Matriz

    outputs:
    --------
        numpy.ndarray: Matriz densa
    '''
    if hasattr(X, 'toarray'):
        return X.toarray()
    return np.asarray(X)


def elegir_noticias(entradas):
    '''
    Función que elige las 6 noticias del test a explicar con una regla fija

    inputs:
    --------
        entradas (pandas.DataFrame): Filas del test de finalistas_entradas.csv

    outputs:
    --------
        pandas.DataFrame: id_unico y caso de cada noticia elegida
    '''
    t = entradas.copy()
    t['det 3'] = t['prob 3'] >= 0.5
    t['det E3'] = t['prob E3'] >= 0.5
    t['det E4'] = t['pred E4'] == 'Fake'
    falsa = t['Category'] == 'Fake'
    larga = t['est_log_palabras'] >= t['est_log_palabras'].quantile(0.75)
    todos = t['det 3'] & t['det E3'] & t['det E4']
    ninguno = ~t['det 3'] & ~t['det E3'] & ~t['det E4']
    casos = [('falsa detectada por los tres', t[falsa & todos & ~larga].sort_values('prob E3', ascending = False)),
             ('falsa larga detectada por los tres', t[falsa & todos & larga].sort_values('prob E3', ascending = False)),
             ('falsa que E4 detecta y el sistema 3 no', t[falsa & t['det E4'] & ~t['det 3']].sort_values('prob E3', ascending = False)),
             ('falsa que ninguno detecta', t[falsa & ninguno].sort_values('prob E3')),
             ('verdadera que los tres marcan como falsa', t[~falsa & todos].sort_values('prob E3', ascending = False)),
             ('verdadera que los tres aciertan', t[~falsa & ninguno].sort_values('prob E3'))]
    filas = []
    for caso, candidatas in casos:
        if len(candidatas) > 0:
            filas.append({'id_unico': candidatas.iloc[0]['id_unico'], 'caso': caso})
    return pd.DataFrame(filas)


def recolectar_perturbaciones(textos):
    '''
    Función que genera con LIME los textos perturbados de cada noticia (primera pasada, sin modelo)

    inputs:
    --------
        textos (list): Texto de cada noticia a explicar

    outputs:
    --------
        list: Una lista de textos perturbados por noticia (el primero es el original)
    '''
    todas = []
    for texto in textos:
        guardados = []

        def recolectar(lote):
            guardados.extend(lote)
            return np.tile([0.5, 0.5], (len(lote), 1))

        LimeTextExplainer(class_names = ['True', 'Fake'], random_state = SEMILLA).explain_instance(
            texto, recolectar, num_features = N_PALABRAS, num_samples = N_MUESTRAS)
        todas.append(list(guardados))
    return todas


def predecir_transformer(nombre, corpus, textos, dispositivo):
    '''
    Función que ajusta un transformer con las 961 noticias (3 semillas) y promedia su probabilidad en los textos dados

    inputs:
    --------
        nombre (str): 'BETO' o 'BERTIN'
        corpus (pandas.DataFrame): Noticias de entrenamiento con texto_sin_medio y y
        textos (list): Textos a predecir
        dispositivo (torch.device): GPU

    outputs:
    --------
        tuple: (probabilidad promedio, lista de pérdidas por época de cada semilla)
    '''
    be.MODELO = TRANSFORMERS[nombre]
    be.CONGELAR_EMBEDDINGS = False
    tokenizador = AutoTokenizer.from_pretrained(be.MODELO)
    ids_corpus = be.tokenizar_todos(corpus['texto_sin_medio'].tolist(), tokenizador)
    ids_textos = be.tokenizar_todos(textos, tokenizador)
    probs = []
    perdidas = []
    for semilla in be.SEMILLAS:
        inicio = time.time()
        modelo, perdida = be.entrenar(ids_corpus, corpus['y'].values, semilla, be.EPOCAS, tokenizador, dispositivo)
        probs.append(be.predecir(modelo, ids_textos, tokenizador, dispositivo))
        perdidas.append(perdida)
        print(f'{nombre}, semilla {semilla}: {(time.time() - inicio) / 60:.1f} min', flush = True)
        del modelo
        torch.cuda.empty_cache()
    return np.mean(probs, axis = 0), perdidas


def embeddings_beto(textos, dispositivo):
    '''
    Función que calcula los embeddings de BETO sin ajustar (igual que beto_embeddings.py)

    inputs:
    --------
        textos (list): Textos
        dispositivo (torch.device): GPU

    outputs:
    --------
        numpy.ndarray: Matriz textos × 768
    '''
    tokenizador = AutoTokenizer.from_pretrained(emb.MODELO)
    modelo = AutoModel.from_pretrained(emb.MODELO).to(dispositivo)
    modelo.eval()
    resultado = emb.embeddings(textos, tokenizador, modelo, dispositivo)
    del modelo
    torch.cuda.empty_cache()
    return resultado


def main():
    '''
    Función principal: elige noticias, genera perturbaciones, predice con todas las bases, explica y guarda

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
    with open(os.path.join(R, 'finalistas_meta.json'), encoding = 'utf-8') as f:
        meta = json.load(f)
    with open(os.path.join(R, 'seleccion_lineal.json'), encoding = 'utf-8') as f:
        config_lr = json.load(f)['elegida']
    entradas = pd.read_csv(os.path.join(R, 'finalistas_entradas.csv'))
    entradas_test = entradas[entradas['conjunto'] == 'test'].reset_index(drop = True)
    corpus = pd.read_parquet(os.path.join(uf.RUTA_PROCESADOS, 'corpus.parquet'))
    test = pd.read_parquet(os.path.join(uf.RUTA_PROCESADOS, 'test.parquet'), columns = ['id_unico', 'texto_sin_medio'])

    elegidas = elegir_noticias(entradas_test)
    textos = test.set_index('id_unico').loc[elegidas['id_unico'], 'texto_sin_medio'].tolist()
    print(elegidas.to_string(index = False), flush = True)
    perturbados = recolectar_perturbaciones(textos)
    todos = []
    for lista in perturbados:
        todos.extend(lista)
    print(f'{len(todos)} textos perturbados en total', flush = True)
    tabla = pd.DataFrame({'texto_sin_medio': todos})

    bases = meta['bases_E3']
    pesos_e3 = np.array(meta['E3']['pesos'])
    con_peso = []
    for base, peso in zip(bases, pesos_e3):
        if peso > 1e-9 or base in meta['sistema_3']['entradas']:
            con_peso.append(base)
    print('Bases que se recalculan (las demás tienen peso 0 en E3):', con_peso, flush = True)

    probs = {}
    if 'LR texto' in con_peso:
        modelo = uf.construir_pipeline(config_lr).fit(corpus, corpus['Category'])
        probs['LR texto'] = modelo.predict_proba(tabla)[:, list(modelo.classes_).index('Fake')]
    if 'XGBoost' in con_peso:
        cx = meta['config_xgb']
        modelo = Pipeline([('representacion', uf.crear_representacion('palabras (1,2)', 'solo texto', sin_medio = True, min_df = 20, minusculas = True)),
                           ('clasificador', XGBClassifier(max_depth = cx['max_depth'], learning_rate = cx['learning_rate'], n_estimators = cx['n_estimators'],
                                                          subsample = 0.8, colsample_bytree = 0.5, tree_method = 'hist', n_jobs = 2, random_state = SEMILLA))])
        probs['XGBoost'] = modelo.fit(corpus, corpus['y']).predict_proba(tabla)[:, 1]
    columna_hgb = 'HistGradientBoosting (caracteres)'
    if columna_hgb in con_peso:
        modelo = Pipeline([('representacion', uf.crear_representacion('caracteres (2,5)', 'solo texto', sin_medio = True, min_df = 20, minusculas = True)),
                           ('denso', FunctionTransformer(a_denso, accept_sparse = True)),
                           ('clasificador', HistGradientBoostingClassifier(max_depth = 3, learning_rate = 0.05, max_iter = 200, early_stopping = False, random_state = SEMILLA))])
        probs[columna_hgb] = modelo.fit(corpus, corpus['y']).predict_proba(tabla)[:, 1]
    if 'Naive Bayes (palabras)' in con_peso:
        alpha = meta['eleccion_embeddings_y_nb']['Naive Bayes']
        modelo = Pipeline([('representacion', uf.crear_representacion('palabras (1,2)', 'solo texto', sin_medio = True, min_df = 20, minusculas = True)),
                           ('nb', ComplementNB(alpha = alpha))])
        probs['Naive Bayes (palabras)'] = modelo.fit(corpus, corpus['y']).predict_proba(tabla)[:, 1]
    if 'SVM-RBF (embeddings)' in con_peso:
        guardados = np.load(os.path.join(R, 'beto_embeddings.npz'))
        emb_corpus = pd.DataFrame(guardados['X_corpus'], index = guardados['ids_corpus']).loc[corpus['id_unico']].values
        c = meta['eleccion_embeddings_y_nb']['SVM-RBF']
        modelo = Pipeline([('escala', StandardScaler()), ('svm', CalibratedClassifierCV(SVC(kernel = 'rbf', C = c, gamma = 'scale'), ensemble = False))])
        modelo.fit(emb_corpus, corpus['y'].values)
        probs['SVM-RBF (embeddings)'] = modelo.predict_proba(embeddings_beto(todos, dispositivo))[:, 1]
    perdidas = {}
    for nombre in TRANSFORMERS:
        if nombre in con_peso:
            probs[nombre], perdidas[nombre] = predecir_transformer(nombre, corpus, todos, dispositivo)

    # Probabilidad final de cada finalista para cada texto perturbado
    def matriz(columnas, inicio, fin):
        filas = []
        for k in range(inicio, fin):
            fila = []
            for base in columnas:
                if base in probs:
                    fila.append(probs[base][k])
                else:
                    fila.append(np.nan)
            filas.append(fila)
        return np.array(filas)

    resumen = []
    explicaciones = []
    verificacion = []
    inicio = 0
    for i in range(len(elegidas)):
        fin = inicio + len(perturbados[i])
        id_unico = elegidas.iloc[i]['id_unico']
        guardada = entradas_test.set_index('id_unico').loc[id_unico]
        X3 = matriz(meta['sistema_3']['entradas'], inicio, fin)
        z3 = X3 @ np.array(meta['sistema_3']['pesos']) + meta['sistema_3']['intercepto']
        X_e3 = matriz(bases, inicio, fin)
        for j, base in enumerate(bases):
            if base not in probs:
                X_e3[:, j] = guardada[base]          # peso 0: su valor no cambia la predicción
        z_e3 = X_e3 @ pesos_e3 + meta['E3']['intercepto']
        p3 = 1 / (1 + np.exp(-z3))
        p_e3 = 1 / (1 + np.exp(-z_e3))
        for base in probs:
            verificacion.append({'id_unico': id_unico, 'base': base, 'prob guardada (ensamble_beto)': float(guardada[base]),
                                 'prob recalculada': float(probs[base][inicio]), 'diferencia': float(probs[base][inicio] - guardada[base])})
        for sistema, prob in [('3. Stacking LR (LR + BETO)', p3), ('E3 / E4 (pesos no negativos)', p_e3)]:
            def devolver(lote, prob = prob, esperados = perturbados[i]):
                assert list(lote) == esperados, 'LIME generó perturbaciones distintas a las de la primera pasada'
                return np.column_stack([1 - prob, prob])
            exp = LimeTextExplainer(class_names = ['True', 'Fake'], random_state = SEMILLA).explain_instance(
                textos[i], devolver, num_features = N_PALABRAS, num_samples = N_MUESTRAS)
            resumen.append({'id_unico': id_unico, 'caso': elegidas.iloc[i]['caso'], 'sistema': sistema,
                            'prob de fake (texto completo)': float(prob[0]), 'R² local de LIME': float(exp.score),
                            'palabras en la noticia': len(textos[i].split())})
            for rango, (palabra, peso) in enumerate(exp.as_list(label = 1)):
                explicaciones.append({'id_unico': id_unico, 'caso': elegidas.iloc[i]['caso'], 'sistema': sistema,
                                      'rango': rango + 1, 'palabra': palabra, 'peso hacia Fake': peso})
        inicio = fin

    pd.DataFrame(explicaciones).to_csv(os.path.join(R, 'finalistas_lime.csv'), index = False, encoding = 'utf-8')
    pd.DataFrame(resumen).to_csv(os.path.join(R, 'finalistas_lime_resumen.csv'), index = False, encoding = 'utf-8')
    pd.DataFrame(verificacion).to_csv(os.path.join(R, 'finalistas_lime_verificacion.csv'), index = False, encoding = 'utf-8')
    filas = []
    for nombre, lista in perdidas.items():
        for semilla, perdida in zip(be.SEMILLAS, lista):
            fila = {'modelo': nombre, 'semilla': semilla}
            for e, valor in enumerate(perdida):
                fila[f'perdida_epoca_{e + 1}'] = valor
            filas.append(fila)
    pd.DataFrame(filas).to_csv(os.path.join(R, 'finalistas_lime_entrenamiento.csv'), index = False, encoding = 'utf-8')
    print('Listo: resultados/finalistas_lime*.csv', flush = True)


if __name__ == '__main__':
    main()
