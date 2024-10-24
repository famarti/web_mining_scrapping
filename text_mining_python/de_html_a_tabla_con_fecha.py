# -*- coding: utf-8 -*-
"""
Este script transforma un grupo de páginas HTML en un dataset para entrenar.
Genera archivos .joblib para vectores, fechas y categorías.
"""

from sklearn.feature_extraction.text import CountVectorizer
from bs4 import BeautifulSoup
import re
import os
import joblib
from typing import Pattern, Optional, List, Tuple
from custom_tokenizers import tokenizador
from datetime import datetime

STOPWORDS_FILE = "text_mining_python\\stopwords_es.txt"
DIR_BASE_CATEGORIAS = "pagina_12_noticias"
VECTORS_FILE = "vectores_con_fecha.joblib"
TARGETS_FILE = "targets_con_fecha.joblib"
DATES_FILE = "fechas.joblib"
FEATURE_NAMES_FILE = "features_con_fecha.joblib"

MARCADOR_COMIENZO_INTERESANTE = '<div class="article-main-content article-text">'
MARCADOR_FIN_INTERESANTE = '<div class="share-mobile hide-on-desktop">'

# Expresión regular para extraer la parte del HTML que nos interesa
extractor_de_parte_de_html_que_interesa = re.compile(
    re.escape(MARCADOR_COMIENZO_INTERESANTE) + "(.*?)" + re.escape(MARCADOR_FIN_INTERESANTE), re.DOTALL
)

# Parámetros del vectorizador
MIN_DF = 3
MAX_DF = 0.8
MIN_NGRAMS = 1
MAX_NGRAMS = 2


def extraer_fecha(html_doc: str) -> Optional[str]:
    """Extrae la fecha de publicación de un artículo HTML."""
    soup = BeautifulSoup(html_doc, 'html.parser')
    time_tag = soup.find('time')

    if time_tag and time_tag.has_attr('datetime'):
        fecha_iso = time_tag['datetime']
        fecha_formateada = datetime.fromisoformat(fecha_iso).strftime('%Y-%m-%d %H:%M')
        return fecha_formateada
    else:
        print("No se pudo encontrar la fecha en el HTML")
        return None


def pasar_html_a_texto(html_doc: str) -> Optional[str]:
    """Convierte la parte extraída del HTML a texto plano."""
    soup = BeautifulSoup(html_doc, 'html.parser')
    article_content = soup.find('div', class_='article-main-content article-text')

    if article_content:
        texto = article_content.get_text(separator=" ", strip=True)
        fecha = extraer_fecha(html_doc)
        if fecha:
            print(f"Fecha extraída: {fecha}")
        else:
            print("No se pudo extraer la fecha.")
        
        return texto, fecha
    else:
        print("No se pudo encontrar la parte del HTML de interés")
        return None, None


def leer_archivo(path: str) -> str:
    return open(path, "rt", encoding="utf-8").read()


def htmls_y_target_y_fecha(dir_de_1_categoria: str) -> Tuple[List[str], List[str], List[str]]:
    """Lee archivos HTML y extrae el texto, la categoría y la fecha."""
    htmls = []
    fechas = []
    for archivo_html in os.listdir(dir_de_1_categoria):
        path_completo_html = os.path.join(dir_de_1_categoria, archivo_html)
        if os.path.isfile(path_completo_html):
            texto, fecha = pasar_html_a_texto(leer_archivo(path_completo_html))
            if texto is not None and fecha is not None:
                htmls.append(texto)
                fechas.append(fecha)
            else:
                print(f"No se pudo extraer texto o fecha de {path_completo_html}")
    
    target_class = [dir_de_1_categoria] * len(htmls)
    return htmls, target_class, fechas


if __name__ == "__main__":

    todos_los_htmls = []
    todos_los_targets = []
    todas_las_fechas = []

    un_dir_por_categoria = [subdir for subdir in os.listdir(DIR_BASE_CATEGORIAS) if os.path.isdir(os.path.join(DIR_BASE_CATEGORIAS, subdir))]

    for dir_por_categoria in un_dir_por_categoria:
        htmls, targets, fechas = htmls_y_target_y_fecha(os.path.join(DIR_BASE_CATEGORIAS, dir_por_categoria))
        todos_los_htmls.extend(htmls)
        todos_los_targets.extend(targets)
        todas_las_fechas.extend(fechas)

    mi_lista_stopwords = leer_archivo(STOPWORDS_FILE).splitlines()
    mi_tokenizer = tokenizador()
    vectorizer = CountVectorizer(stop_words=mi_lista_stopwords, tokenizer=mi_tokenizer,
                                 lowercase=True, strip_accents='unicode', decode_error='ignore',
                                 ngram_range=(MIN_NGRAMS, MAX_NGRAMS), min_df=MIN_DF, max_df=MAX_DF)

    todos_los_vectores = vectorizer.fit_transform(todos_los_htmls)

    joblib.dump(todos_los_vectores, VECTORS_FILE)
    joblib.dump(todos_los_targets, TARGETS_FILE)
    joblib.dump(todas_las_fechas, DATES_FILE)

    print(f"Dataset generado: {VECTORS_FILE}, {TARGETS_FILE}, {DATES_FILE}")
