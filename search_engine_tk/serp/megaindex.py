from pandas.io import json

__author__ = 'lvova'
import grab
import urllib
from pyyaxml.search import SearchResultItem


def megaindex_position(query, region=213, num_res=10):
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
        except grab.error.GrabTimeoutError as e:
            continue
        jsonText = g.doc.unicode_body()
        data = json.loads(jsonText)
        res.append(data['data'])
    return res


