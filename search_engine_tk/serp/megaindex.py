import urllib

import grab

from pandas.io import json
from pyyaxml.search import SearchResultItem


def get_serp(query, region=213, num_res=10):
    """

    Parameters
    ----------
    query: str,
        поисковый запрос запрос
    region: int,
        номер региона по нумерации яндекса
    num_res: int,
        глубина ТОПа
     Returns
    -------
    : list of pyyaxml.search.SearchResultItem
        список, упорядоченный по позиции в выдаче список объектов типа SerpItem длиной top,
        соответствующий ТОП{top} поисковой выдачи
    """
    params = {'request': query,
              'user': 'timonof@gmail.com',
              'password': 'En90fB',
              'lr': region,
              'how_title': 1,
              }
    url = 'http://api.megaindex.ru/scan_yandex_position' + '?' + urllib.parse.urlencode(params)

    g = grab.Grab()
    g.setup(url=url)
    g.request()
    jsonText = g.doc.unicode_body()

    data = json.loads(jsonText)
    if data['status'] != 0:
        raise Exception("Something wrong")
    res = data['data']
    serp = ['' for n in range(num_res)]
    for result in res:
        if result['position'] <= num_res:
            result_url = 'http://www.' if result['www'] else 'http://'
            result_url += result['domain'] + '/' + result['path']
            serp[result['position']-1] = SearchResultItem(result_url, result['title'], result['text'])
    return serp


def megaindex_wordstat3(queries):
    """
    Возвращяет значения wordstat по точному соответствию
    Parameters
    ----------
    query: str,
        поисковый запрос запрос

    Returns
    -------
    : list of int
    """

    res = []
    for query in queries:
        params = {'user': 'timonof@gmail.com',
                  'password': 'En90fB',
                  'request': query
                  }
        url = 'http://api.megaindex.ru/scan_wordstat3?' + urllib.parse.urlencode(params)
        g = grab.Grab()
        g.setup(url=url)
        try:
            g.request()
        except grab.error.GrabTimeoutError:
            continue
        jsonText = g.doc.unicode_body()
        data = json.loads(jsonText)
        try:
            res.append(int(data['data']))
        except:
            res.append(0)

    return res
