'''
Búsqueda directa de las fuentes del corpus en la muestra de mC4 con la que se preentrenó BERTIN

Consulta el buscador público de Hugging Face (datasets-server, sin cuenta ni token) en
bertin-project/mc4-es-sampled (configuración gaussian, partición train) con el nombre del dominio web de las 5 fuentes
de falsas y las 5 de verdaderas con más noticias en el corpus (por ejemplo, 'elruinaversal'). El buscador no busca
frases exactas, así que se cuentan, entre los primeros 100 resultados, las páginas cuya URL es de ese dominio.
Además, noticia por noticia: para cada noticia de esas fuentes (3 de falsas y 2 de verdaderas como control) busca las
palabras de su URL y revisa si esa misma URL aparece en mC4 (contaminación de la noticia, no solo del sitio).
Solo se envían nombres de medios públicos y las palabras de URLs públicas.
Control positivo: las páginas de ejemplo que encontró la búsqueda por dominio (que sí están en mC4) se buscan con el
mismo método noticia por noticia; si el método funciona, deben salir 'encontrada'.
Escribe resultados/contaminacion_busqueda.csv, resultados/contaminacion_busqueda_noticias.csv y
resultados/contaminacion_busqueda_control.csv.

Uso (con el entorno fakedes activado, desde la carpeta del repo):
    python contaminacion_busqueda.py
'''
import json
import os
import time
import urllib.parse
import urllib.request

import pandas as pd

URL = 'https://datasets-server.huggingface.co/search'
DATASET = 'bertin-project/mc4-es-sampled'
CONFIGURACION = 'gaussian'
RUTA_SALIDA = os.path.join('resultados', 'contaminacion_busqueda.csv')
RUTA_NOTICIAS = os.path.join('resultados', 'contaminacion_busqueda_noticias.csv')
RUTA_CONTROL = os.path.join('resultados', 'contaminacion_busqueda_control.csv')
FUENTES_NOTICIAS = ['el ruinaversal', 'el dizque', 'hay noticia', 'forbes', 'el universal']


def dominio_de(url):
    '''
    Función que extrae el dominio de una URL sin 'www.'

    inputs:
    --------
        url (str): URL completa

    outputs:
    --------
        str: Dominio, por ejemplo 'elpais.com'
    '''
    resto = url.split('://', 1)[-1].split('/', 1)[0].lower()
    if resto.startswith('www.'):
        resto = resto[4:]
    return resto


def es_del_dominio(url, dominio):
    '''
    Función que revisa si una URL es de un dominio (incluye subdominios como archivo.eluniversal.com.mx)

    inputs:
    --------
        url (str): URL completa
        dominio (str): Dominio, por ejemplo 'eluniversal.com.mx'

    outputs:
    --------
        bool: True si la URL es de ese dominio
    '''
    host = dominio_de(url)
    return host == dominio or host.endswith('.' + dominio)


def consultar(texto, largo, intentos = 10):
    '''
    Función que consulta el buscador de Hugging Face con reintentos

    inputs:
    --------
        texto (str): Texto a buscar
        largo (int): Número de resultados a pedir
        intentos (int): Reintentos si el índice no responde

    outputs:
    --------
        dict: Respuesta del buscador o {'error': ...}
    '''
    parametros = urllib.parse.urlencode({'dataset': DATASET, 'config': CONFIGURACION, 'split': 'train',
                                         'query': texto, 'offset': 0, 'length': largo})
    datos = {'error': 'sin intentos'}
    for intento in range(intentos):
        try:
            with urllib.request.urlopen(f'{URL}?{parametros}', timeout = 120) as respuesta:
                return json.load(respuesta)
        except Exception as error:
            datos = {'error': str(error)}
        time.sleep(45)
    return datos


def buscar(texto, dominio, intentos = 10):
    '''
    Función que busca un texto en la muestra de mC4 y cuenta cuántos de los primeros 100 resultados son del dominio

    inputs:
    --------
        texto (str): Texto a buscar (nombre del dominio sin terminación)
        dominio (str): Dominio web de la fuente, por ejemplo 'elruinaversal.com'
        intentos (int): Reintentos si el índice todavía se está cargando

    outputs:
    --------
        dict: resultados totales, páginas del dominio entre los primeros 100, si el índice es parcial y un ejemplo
    '''
    datos = consultar(texto, 100, intentos)
    if 'error' in datos:
        return {'resultados de la búsqueda': None, 'páginas del dominio (primeros 100)': None, 'índice parcial': None,
                'ejemplo del dominio': f'sin respuesta: {datos["error"][:80]}'}
    del_dominio = []
    for fila in datos.get('rows', []):
        url = fila['row'].get('url', '')
        if es_del_dominio(url, dominio):
            del_dominio.append(url)
    return {'resultados de la búsqueda': datos.get('num_rows_total'), 'páginas del dominio (primeros 100)': len(del_dominio),
            'índice parcial': datos.get('partial'), 'ejemplo del dominio': del_dominio[0] if del_dominio else ''}


def palabras_de_url(url):
    '''
    Función que saca las palabras de la última parte de una URL (el 'slug' de la noticia)

    inputs:
    --------
        url (str): URL de la noticia

    outputs:
    --------
        str: Palabras separadas por espacios (sin números)
    '''
    partes = url.rstrip('/').split('/')
    slug = partes[-1]
    if slug.isdigit() or len(slug) < 8:
        slug = partes[-2] if len(partes) > 1 else slug
    palabras = []
    for palabra in slug.replace('.html', '').replace('_', '-').split('-'):
        if palabra.isalpha() and len(palabra) > 2:
            palabras.append(palabra)
    return ' '.join(palabras[:10])


def buscar_noticia(url):
    '''
    Función que busca las palabras de la URL de una noticia y revisa si la misma noticia está en mC4

    inputs:
    --------
        url (str): URL de la noticia en el corpus

    outputs:
    --------
        str: 'encontrada', 'no encontrada' o 'sin respuesta'
    '''
    consulta = palabras_de_url(url)
    if not consulta:
        return 'sin palabras en la URL'
    datos = consultar(consulta, 10)
    if 'error' in datos:
        return 'sin respuesta'
    ruta = url.split('://', 1)[-1].split('/', 1)[-1].rstrip('/').lower()
    for fila in datos.get('rows', []):
        otra = fila['row'].get('url', '').split('://', 1)[-1].split('/', 1)[-1].rstrip('/').lower()
        if es_del_dominio(fila['row'].get('url', ''), dominio_de(url).split('.', 1)[-1] if dominio_de(url).count('.') > 2 else dominio_de(url)) and otra == ruta:
            return 'encontrada'
    return 'no encontrada'


def main():
    '''
    Función principal: elige las fuentes, busca cada una y guarda la tabla

    inputs:
    --------
        Ninguno

    outputs:
    --------
        None
    '''
    corpus = pd.read_parquet(os.path.join('data', 'processed', 'corpus.parquet'), columns = ['id_unico', 'Category', 'Link'])
    folds = pd.read_csv(os.path.join('resultados', 'folds.csv'))
    folds = folds[folds['particion'] != 'test'].set_index('id_unico')
    corpus['fuente'] = corpus['id_unico'].map(folds['fuente'])
    corpus['dominio'] = corpus['Link'].astype(str).str.extract(r'https?://(?:www\.)?([^/]+)')[0]
    conteo = corpus.groupby(['fuente', 'Category']).agg(noticias = ('id_unico', 'size'),
                                                       dominio = ('dominio', lambda s: s.value_counts().index[0])).reset_index()
    elegidas = pd.concat([conteo[conteo['Category'] == 'Fake'].nlargest(5, 'noticias'),
                          conteo[conteo['Category'] == 'True'].nlargest(5, 'noticias')])
    filas = []
    for fuente, clase, n, dominio in zip(elegidas['fuente'], elegidas['Category'], elegidas['noticias'], elegidas['dominio']):
        resultado = buscar(dominio.split('.')[0], dominio)
        filas.append({'fuente': fuente, 'dominio': dominio, 'clase en el corpus': clase, 'noticias en el corpus': n, **resultado})
        print(filas[-1], flush = True)
    pd.DataFrame(filas).to_csv(RUTA_SALIDA, index = False, encoding = 'utf-8')
    print(f'Listo: {RUTA_SALIDA}', flush = True)

    noticias = []
    for fuente in FUENTES_NOTICIAS:
        de_fuente = corpus[corpus['fuente'] == fuente]
        for id_unico, clase, url in zip(de_fuente['id_unico'], de_fuente['Category'], de_fuente['Link'].astype(str)):
            noticias.append({'id_unico': id_unico, 'fuente': fuente, 'clase en el corpus': clase, 'url': url,
                             'en mC4 (BERTIN)': buscar_noticia(url)})
        resumen = pd.DataFrame(noticias)
        resumen = resumen[resumen['fuente'] == fuente]['en mC4 (BERTIN)'].value_counts().to_dict()
        print(fuente, resumen, flush = True)
    pd.DataFrame(noticias).to_csv(RUTA_NOTICIAS, index = False, encoding = 'utf-8')
    print(f'Listo: {RUTA_NOTICIAS}', flush = True)

    control = []
    for fila in filas:
        ejemplo = str(fila['ejemplo del dominio'])
        if ejemplo.startswith('http') and fila['fuente'] in FUENTES_NOTICIAS:
            control.append({'fuente': fila['fuente'], 'url (página que sí está en mC4)': ejemplo,
                            'resultado del método noticia por noticia': buscar_noticia(ejemplo)})
    pd.DataFrame(control).to_csv(RUTA_CONTROL, index = False, encoding = 'utf-8')
    print(pd.DataFrame(control), flush = True)
    print(f'Listo: {RUTA_CONTROL}')


if __name__ == '__main__':
    main()
