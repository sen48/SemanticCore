"""
Поиск яндекс XML
"""
from pyyaxml.search import YaSearch


API_USER = 'kinetica-agency'  # 'kin-wm'
API_KEY = '03.220366537:4eff804afa959d40397b6ba743cf8de6'  # '03.96758214:0fcd1eb66ab02f7636a75dd69d894f5f'


def get_serp(query, region, num_res=100):
    """
    Возвращает список объектов SearchResultItem соответствующий первым num_res результатам выдачи Яндекса
    по запросу query в регионе region
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
    """
    return _site_list(query, region, num_res)


def _site_list(query, region, num_res, page=0):
    y = YaSearch(API_USER, API_KEY)
    results = y.search(query, site=None, page=page, region=region, num_res=num_res)
    serp = []
    for result in results.items:
        serp.append(result)
    return serp


def corresponding_page(query, region, site):
    """
    Возвращает SearchResultItem соответствующий странице сайта site наиболее релевантной запросу query в регионе region
    Parameters
    ----------
    site: str,
        адрес сайта без http:// и www
    query: str,
        поисковый запрос запрос
    region: int,
        номер региона по нумерации яндекса
    Returns
    -------
    : pyyaxml.search.SearchResultItem
    """
    y = YaSearch(API_USER, API_KEY)
    results = y.search(query, site=site, page=0, region=region)
    if len(results.items) > 0:
        return results.items[0]
    else:
        return None


