'''
Funciones compartidas por los notebooks del proyecto FakeDeS (IberLEF 2021)
'''
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Constantes del proyecto (las mismas en todos los notebooks)
SEMILLA = 42
CLASE_POSITIVA = 'Fake'
COLORES_CLASE = {'True': 'C0', 'Fake': 'red'}
COLORES_PARTICION = {'train': 'C0', 'development': 'orange', 'test': 'red'}
TINTA = '#2b2b2b'
TINTA_SUAVE = '#6b6b6b'

RUTA_DATOS = 'data'
ARCHIVOS = {'train': 'train.xlsx', 'development': 'development.xlsx', 'test': 'test.xlsx'}
RUTA_FIGURAS = 'figuras'
RUTA_RESULTADOS = 'resultados'

TEMAS_ES = {'Covid-19': 'COVID-19', 'Sociedad': 'Society', 'Política': 'Politics', 'Internacional': 'International',
            'Ciencia': 'Science', 'Ambiental': 'Environment', 'Deporte': 'Sport'}


def configurar_graficas():
    '''
    Función que fija el estilo común de las figuras del proyecto

    inputs:
    --------
        Ninguno

    outputs:
    --------
        None
    '''
    plt.rcParams.update({
        'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': '#e6e6e6', 'grid.linewidth': 0.8, 'axes.axisbelow': True,
        'font.size': 11, 'axes.titlesize': 13, 'axes.titlelocation': 'left', 'axes.labelsize': 11,
        'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11, 'legend.frameon': False,
        'text.color': TINTA, 'axes.labelcolor': TINTA, 'axes.edgecolor': TINTA_SUAVE,
        'xtick.color': TINTA, 'ytick.color': TINTA,
    })


def crear_carpetas(*carpetas):
    '''
    Función que crea las carpetas de salida si no existen

    inputs:
    --------
        carpetas (str): Rutas de las carpetas

    outputs:
    --------
        None
    '''
    for carpeta in carpetas:
        os.makedirs(carpeta, exist_ok = True)


def guardar(fig, nombre):
    '''
    Función que guarda una figura en la carpeta de figuras como PNG

    inputs:
    --------
        fig (matplotlib.figure.Figure): Figura a guardar
        nombre (str): Nombre del archivo sin extensión ('N_M_descripcion')

    outputs:
    --------
        str: Ruta del archivo guardado
    '''
    crear_carpetas(RUTA_FIGURAS)
    ruta = os.path.join(RUTA_FIGURAS, f'{nombre}.png')
    fig.savefig(ruta, dpi = 150, bbox_inches = 'tight')
    return ruta


def guardar_csv(df, nombre, index = False):
    '''
    Función que guarda una tabla en la carpeta de resultados

    inputs:
    --------
        df (pandas.DataFrame): Tabla a guardar
        nombre (str): Nombre del archivo sin extensión
        index (bool): Si se guarda el índice

    outputs:
    --------
        str: Ruta del archivo guardado
    '''
    crear_carpetas(RUTA_RESULTADOS)
    ruta = os.path.join(RUTA_RESULTADOS, f'{nombre}.csv')
    df.to_csv(ruta, index = index, encoding = 'utf-8')
    return ruta


def cargar_corpus(ruta_datos = RUTA_DATOS):
    '''
    Función que lee los tres archivos y los une en un esquema común (columnas, clase y temas de la versión 1.0)

    inputs:
    --------
        ruta_datos (str): Carpeta con los .xlsx originales

    outputs:
    --------
        pandas.DataFrame: Corpus con las columnas originales más 'particion' y 'version'
    '''
    crudos = {p: pd.read_excel(os.path.join(ruta_datos, a)) for p, a in ARCHIVOS.items()}
    test = crudos['test'].copy()
    test.columns = crudos['train'].columns
    test['Category'] = test['Category'].map({True: 'True', False: 'Fake'})
    test['Topic'] = test['Topic'].map(TEMAS_ES)
    corpus = pd.concat([crudos['train'].assign(particion = 'train'),
                        crudos['development'].assign(particion = 'development'),
                        test.assign(particion = 'test')], ignore_index = True)
    corpus['version'] = np.where(corpus['particion'] == 'test', '2.0', '1.0')
    return corpus.astype({c: 'object' for c in ['Category', 'Topic', 'Source', 'Headline', 'Text', 'Link']})


def normalizar_fuente(serie):
    '''
    Función que unifica los nombres de las fuentes (minúsculas y sin espacios en los extremos)

    inputs:
    --------
        serie (pandas.Series): Columna 'Source'

    outputs:
    --------
        pandas.Series: Fuente normalizada; los faltantes quedan como 'sin fuente'
    '''
    return serie.fillna('sin fuente').astype(str).str.strip().str.lower()


def f1_fake(y_real, y_pred):
    '''
    Función que calcula la métrica oficial de FakeDeS 2021: F1 de la clase 'Fake'

    inputs:
    --------
        y_real (array-like): Clases reales ('True' / 'Fake')
        y_pred (array-like): Clases predichas ('True' / 'Fake')

    outputs:
    --------
        float: F1 de la clase 'Fake' (0 si no hay verdaderos positivos)
    '''
    return f1_score(y_real, y_pred, pos_label = CLASE_POSITIVA, average = 'binary', zero_division = 0)


def metricas(y_real, y_pred):
    '''
    Función que calcula la métrica oficial y las complementarias del proyecto

    inputs:
    --------
        y_real (array-like): Clases reales ('True' / 'Fake')
        y_pred (array-like): Clases predichas ('True' / 'Fake')

    outputs:
    --------
        dict: F1_fake, F1_true, F1_macro, accuracy, precision_fake y recall_fake
    '''
    return {
        'F1_fake': f1_fake(y_real, y_pred),
        'F1_true': f1_score(y_real, y_pred, pos_label = 'True', average = 'binary', zero_division = 0),
        'F1_macro': f1_score(y_real, y_pred, average = 'macro', zero_division = 0),
        'accuracy': accuracy_score(y_real, y_pred),
        'precision_fake': precision_score(y_real, y_pred, pos_label = CLASE_POSITIVA, zero_division = 0),
        'recall_fake': recall_score(y_real, y_pred, pos_label = CLASE_POSITIVA, zero_division = 0),
    }


# ---------------------------------------------------------------- Preparación del texto
RUTA_PROCESADOS = os.path.join('data', 'processed')
MARCA_MEDIO = '<MEDIO>'                  # solo si se quiere marcar en vez de borrar
PATRON_URL = re.compile(r'https?://\S+|www\.\S+')
# Número suelto (con separadores . o ,); no toca dígitos pegados a letras (CR7, 5G) ni sufijos tipo COVID-19
PATRON_NUMERO = re.compile(r'(?<!\w)(?<![A-Za-zÁÉÍÓÚáéíóúÑñ]-)\d+(?:[.,]\d+)*(?!\w)')


def normalizar_espacios(texto):
    '''
    Función que convierte espacios duros y de ancho cero en espacios normales y colapsa los repetidos

    inputs:
    --------
        texto (str): Texto a normalizar

    outputs:
    --------
        str: Texto con un solo espacio entre palabras y saltos de línea conservados
    '''
    texto = re.sub('[   ]', ' ', texto)
    texto = re.sub('[​‌‍﻿]', '', texto)
    texto = re.sub(r'[ \t]+', ' ', texto)
    return re.sub(r' *\n *', '\n', texto).strip()


def quitar_titular_inicial(titular, texto):
    '''
    Función que elimina el titular cuando el texto empieza repitiéndolo (formato de la versión 1.0)

    inputs:
    --------
        titular (str): Titular de la noticia
        texto (str): Texto de la noticia

    outputs:
    --------
        str: Texto sin el titular inicial (sin cambios si no empieza con él)
    '''
    t = titular.strip()
    if t and texto.lower().startswith(t.lower()):
        return texto[len(t):].lstrip(' .:-\n')
    return texto


def anonimizar_como_v1(texto):
    '''
    Función que aplica al texto la anonimización de la versión 1.0 (URLs y números sueltos)

    inputs:
    --------
        texto (str): Texto a anonimizar

    outputs:
    --------
        str: Texto con '*URL*' y '*NUMBER*' en lugar de URLs y números sueltos
    '''
    texto = PATRON_URL.sub('*URL*', texto)
    return PATRON_NUMERO.sub('*NUMBER*', texto)


def patron_medios(lexico):
    '''
    Función que compila una expresión regular con los nombres de medios a enmascarar

    inputs:
    --------
        lexico (list): Nombres de medios en minúsculas

    outputs:
    --------
        re.Pattern o None: Patrón que reconoce los nombres como palabras completas (None si el léxico está vacío)
    '''
    if not lexico:
        return None
    alternativas = '|'.join(re.escape(n) for n in sorted(lexico, key = len, reverse = True))
    return re.compile(r'(?<!\w)(?:' + alternativas + r')(?!\w)', flags = re.IGNORECASE)


def enmascarar_medios(texto, patron, reemplazo = ''):
    '''
    Función que borra (o reemplaza) los nombres de medios del léxico

    inputs:
    --------
        texto (str): Texto a enmascarar
        patron (re.Pattern o None): Resultado de patron_medios
        reemplazo (str): Texto que sustituye cada nombre; vacío para borrarlo

    outputs:
    --------
        str: Texto sin los nombres de medios
    '''
    if patron is None:
        return texto
    return re.sub(r'[ 	]{2,}', ' ', patron.sub(reemplazo, texto)).strip()


def preparar_noticia(titular, texto, patron = None):
    '''
    Función que aplica la preparación completa a una noticia nueva (la misma que se usa en el corpus)

    inputs:
    --------
        titular (str o None): Titular; vacío o None si no existe
        texto (str): Texto de la noticia
        patron (re.Pattern o None): Patrón de medios cuyos nombres se borran; None para no borrar

    outputs:
    --------
        tuple: (titular preparado, texto preparado)
    '''
    titular = normalizar_espacios(titular if isinstance(titular, str) else '')
    texto = quitar_titular_inicial(titular, normalizar_espacios(texto))
    titular, texto = anonimizar_como_v1(titular), anonimizar_como_v1(texto)
    return enmascarar_medios(titular, patron), enmascarar_medios(texto, patron)


# ---------------------------------------------------------------- Rasgos de estilo
RASGOS_ESTILO = ['est_log_palabras', 'est_palabras_por_oracion', 'est_long_media_palabra', 'est_prop_mayusculas',
                 'est_prop_palabras_mayusculas', 'est_exclamaciones', 'est_interrogaciones', 'est_comillas',
                 'est_numeros', 'est_prop_vacias']


def rasgos_estilo(texto, vacias):
    '''
    Función que calcula los rasgos de estilo densos de un texto (sin usar el titular)

    inputs:
    --------
        texto (str): Texto preparado de la noticia
        vacias (set): Palabras vacías del español en minúsculas

    outputs:
    --------
        dict: Un valor por cada nombre de RASGOS_ESTILO
    '''
    limpio = texto.replace('*NUMBER*', ' ').replace('*URL*', ' ').replace(MARCA_MEDIO, ' ')
    palabras = re.findall(r'[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+', limpio)
    n_pal, n_car = max(len(palabras), 1), max(len(limpio), 1)
    letras = re.findall(r'[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]', limpio)
    oraciones = max(len([o for o in re.split(r'[.!?¡¿]+\s', limpio) if o.strip()]), 1)
    return {
        'est_log_palabras': np.log1p(len(palabras)),
        'est_palabras_por_oracion': len(palabras) / oraciones,
        'est_long_media_palabra': np.mean([len(p) for p in palabras]) if palabras else 0.0,
        'est_prop_mayusculas': sum(l.isupper() for l in letras) / max(len(letras), 1),
        'est_prop_palabras_mayusculas': sum(len(p) >= 3 and p.isupper() for p in palabras) / n_pal,
        'est_exclamaciones': 1000 * len(re.findall(r'[!¡]', limpio)) / n_car,
        'est_interrogaciones': 1000 * len(re.findall(r'[?¿]', limpio)) / n_car,
        'est_comillas': 1000 * len(re.findall(r'["“”«»]', limpio)) / n_car,
        'est_numeros': 100 * texto.count('*NUMBER*') / n_pal,
        'est_prop_vacias': sum(p.lower() in vacias for p in palabras) / n_pal,
    }


# ---------------------------------------------------------------- Modelado
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC

REPRESENTACIONES = ['palabras (1,1)', 'palabras (1,2)', 'caracteres (2,5)', 'palabras (1,2) + caracteres (2,5)']
ESTRUCTURAS = ['solo titular', 'solo texto', 'concatenado', 'bloques']


def unir_columnas(X):
    '''
    Función que concatena titular y texto en una sola cadena por noticia (para la estructura 'concatenado')

    inputs:
    --------
        X (pandas.DataFrame): Dos columnas de texto (titular, texto)

    outputs:
    --------
        pandas.Series: Titular y texto separados por un salto de línea
    '''
    return X.iloc[:, 0].fillna('') + '\n' + X.iloc[:, 1].fillna('')


def crear_vectorizador(representacion, min_df = 2, minusculas = True):
    '''
    Función que crea el vectorizador TF-IDF de una representación

    inputs:
    --------
        representacion (str): Uno de REPRESENTACIONES
        min_df (int): Número mínimo de documentos en que debe aparecer un término
        minusculas (bool): Si se pasa el texto a minúsculas

    outputs:
    --------
        sklearn transformer: TfidfVectorizer o unión de dos
    '''
    palabras = lambda n: TfidfVectorizer(ngram_range = (1, n), min_df = min_df, lowercase = minusculas, sublinear_tf = True)
    caracteres = lambda: TfidfVectorizer(analyzer = 'char_wb', ngram_range = (2, 5), min_df = min_df, lowercase = minusculas, sublinear_tf = True)
    opciones = {'palabras (1,1)': lambda: palabras(1), 'palabras (1,2)': lambda: palabras(2), 'caracteres (2,5)': caracteres,
                'palabras (1,2) + caracteres (2,5)': lambda: FeatureUnion([('palabras', palabras(2)), ('caracteres', caracteres())])}
    return opciones[representacion]()


def crear_representacion(representacion, estructura, sin_medio = True, min_df = 2, minusculas = True, estilo = False):
    '''
    Función que crea el transformador que convierte una noticia preparada en su matriz de atributos

    inputs:
    --------
        representacion (str): Uno de REPRESENTACIONES
        estructura (str): Uno de ESTRUCTURAS
        sin_medio (bool): Si se usan las columnas con los nombres de medio borrados
        min_df (int): Número mínimo de documentos por término
        minusculas (bool): Si se pasa el texto a minúsculas
        estilo (bool): Si se agrega el bloque de rasgos de estilo escalados

    outputs:
    --------
        sklearn.compose.ColumnTransformer: Transformador que recibe el DataFrame preparado
    '''
    sufijo = '_sin_medio' if sin_medio else ''
    tit, txt = f'titular{sufijo}', f'texto{sufijo}'
    vec = lambda: crear_vectorizador(representacion, min_df, minusculas)
    bloques = {'solo titular': [('titular', vec(), tit)],
               'solo texto': [('texto', vec(), txt)],
               'concatenado': [('todo', Pipeline([('unir', FunctionTransformer(unir_columnas)), ('vec', vec())]), [tit, txt])],
               'bloques': [('titular', vec(), tit), ('texto', vec(), txt)]}[estructura]
    if estilo:
        bloques = bloques + [('estilo', StandardScaler(), RASGOS_ESTILO)]
    return ColumnTransformer(bloques)


def crear_modelo(nombre, valor):
    '''
    Función que crea un clasificador lineal con su hiperparámetro de regularización

    inputs:
    --------
        nombre (str): 'LR' (regresión logística), 'SVM' (SVM lineal) o 'CNB' (Complement NB)
        valor (float): C para LR y SVM; alpha para CNB

    outputs:
    --------
        sklearn estimator: Clasificador sin entrenar
    '''
    if nombre == 'LR':
        return LogisticRegression(C = valor, class_weight = 'balanced', solver = 'liblinear', max_iter = 5000)
    if nombre == 'SVM':
        return LinearSVC(C = valor, class_weight = 'balanced', random_state = SEMILLA, max_iter = 20000)
    return ComplementNB(alpha = valor)


def construir_pipeline(config):
    '''
    Función que construye el Pipeline completo (representación + clasificador) a partir de una configuración

    inputs:
    --------
        config (dict): Llaves 'representacion', 'estructura', 'sin_medio', 'min_df', 'minusculas', 'modelo', 'valor' y, opcionales, 'estilo' y 'k_chi2'

    outputs:
    --------
        sklearn.pipeline.Pipeline: Pipeline sin entrenar
    '''
    pasos = [('representacion', crear_representacion(config['representacion'], config['estructura'], config['sin_medio'],
                                                      config['min_df'], config['minusculas'], config.get('estilo', False)))]
    if config.get('k_chi2'):
        pasos.append(('seleccion', SelectKBest(chi2, k = config['k_chi2'])))
    pasos.append(('clasificador', crear_modelo(config['modelo'], config['valor'])))
    return Pipeline(pasos)


def construir_replica(nombre):
    '''
    Función que construye la réplica aproximada de un baseline oficial de FakeDeS 2021 (sin lematización)

    inputs:
    --------
        nombre (str): 'BoW + SVM' (bolsa de palabras) o '3-gramas de caracteres + SVM'

    outputs:
    --------
        sklearn.pipeline.Pipeline: Pipeline sin entrenar que recibe las columnas 'titular' y 'texto'
    '''
    vectorizador = CountVectorizer() if nombre == 'BoW + SVM' else CountVectorizer(analyzer = 'char', ngram_range = (3, 3))
    representacion = ColumnTransformer([('todo', Pipeline([('unir', FunctionTransformer(unir_columnas)), ('vec', vectorizador)]), ['titular', 'texto'])])
    return Pipeline([('representacion', representacion), ('clasificador', LinearSVC(random_state = SEMILLA, max_iter = 20000))])


def evaluar_por_folds(df, columna_fold, funcion):
    '''
    Función que recorre los folds y mide las predicciones de una o varias configuraciones

    inputs:
    --------
        df (pandas.DataFrame): Corpus con 'Category' y la columna de folds
        columna_fold (str): 'fold_grupo', 'fold_aleatorio' u otra columna de partición
        funcion (callable): Recibe (entrenamiento, validacion) y devuelve {nombre: predicciones de validacion}

    outputs:
    --------
        pandas.DataFrame: Una fila por fold y configuración con las métricas de uf.metricas
    '''
    filas = []
    for k in sorted(df[columna_fold].unique()):
        entrenamiento, validacion = df[df[columna_fold] != k], df[df[columna_fold] == k]
        for nombre, prediccion in funcion(entrenamiento, validacion).items():
            filas.append({'config': nombre, 'fold': k, 'n': len(validacion), **metricas(validacion['Category'], prediccion)})
    return pd.DataFrame(filas)


def resumir_cv(cv, por = 'config'):
    '''
    Función que resume los resultados por fold en media, desviación estándar y error estándar

    inputs:
    --------
        cv (pandas.DataFrame): Resultado de evaluar_por_folds
        por (str o list): Columnas que identifican la configuración

    outputs:
    --------
        pandas.DataFrame: Media y sd de F1_fake, F1_macro y accuracy, y el error estándar de F1_fake, ordenado de mayor a menor F1_fake
    '''
    resumen = cv.groupby(por)[['F1_fake', 'F1_macro', 'accuracy']].agg(['mean', 'std'])
    resumen.columns = [f'{m}_{e}' for m, e in resumen.columns]
    resumen['F1_fake_se'] = resumen['F1_fake_std'] / np.sqrt(cv.groupby(por)['fold'].nunique())
    return resumen.sort_values('F1_fake_mean', ascending = False)


def construir_modelo_estilo(config):
    '''
    Función que construye un modelo sobre los rasgos de estilo densos (Random Forest o regresión logística escalada)

    inputs:
    --------
        config (dict): 'modelo' ('RF' o 'LR') y 'parametros' (hiperparámetros del clasificador)

    outputs:
    --------
        sklearn.pipeline.Pipeline: Pipeline sin entrenar que recibe el DataFrame preparado
    '''
    from sklearn.ensemble import RandomForestClassifier
    seleccion = ColumnTransformer([('estilo', 'passthrough', RASGOS_ESTILO)])
    if config['modelo'] == 'RF':
        clasificador = RandomForestClassifier(n_estimators = 500, class_weight = 'balanced', random_state = SEMILLA, n_jobs = -1, **config['parametros'])
        return Pipeline([('rasgos', seleccion), ('clasificador', clasificador)])
    return Pipeline([('rasgos', seleccion), ('escala', StandardScaler()),
                     ('clasificador', LogisticRegression(class_weight = 'balanced', max_iter = 5000, **config['parametros']))])
