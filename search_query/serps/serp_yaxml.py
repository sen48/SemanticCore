
# -*- coding: utf-8 -*-
from pyyaxml.search import YaSearch


API_USER = 'kinetica-agency'  # 'kin-wm'
API_KEY = '03.220366537:4eff804afa959d40397b6ba743cf8de6'  # '03.96758214:0fcd1eb66ab02f7636a75dd69d894f5f'


class YaException(Exception):
    pass


def get_serp(query, region, num_res=100):
    return _site_list(query, region, num_res)


def _site_list(query, region, num_res, page=0):

    y = YaSearch(API_USER, API_KEY)
    results = y.search(query, site=None, page=page, region=region, num_res=num_res)
    serp = []
    for result in results.items:
        serp.append(result)
    return serp


def corresponding_page(query, region, site):
    y = YaSearch(API_USER, API_KEY)
    results = y.search(query, site=site, page=0, region=region)
    try:
        return results.items[0]
    except IndexError:
        raise YaException('Ничего не найдено')


