"""
Тут все нужно переделывать
"""


import urllib
from urllib.parse import urlparse
import mysql.connector
from mysql.connector import errorcode
from search_engine_tk.serp.serp_yaxml import corresponding_page, YaException, get_serp
from search_engine_tk.serp_item import SerpItem


class ReadSerpException(Exception):
    pass


class WriteSerpException(Exception):
    pass


data_base_config = {
    'user': 'root',
    'password': '135',
    'host': 'localhost',
    'database': 'serps',
    'raise_on_warnings': True,
}



def _read_from_db(get_str, data=None):
    res = []
    try:
        con = mysql.connector.connect(**data_base_config)
        cur = con.cursor()
        cur.execute(get_str, data)
        res = list(cur)
        cur.close()
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)
    else:
        con.close()
    return res


def read_url_position(hostname, query, region):
    """
    Возвращяет позицию релевантной страницы и соответствующий ей объект класса SerpItem
    :param hostname: str, адрес сайта без http:// и www
    :param query: str, запрос
    :param region: int, номер региона
    :return: пара состоящая из номера позиции и объекта класса SerpItem,
             соответствующего странице сайта в поисковой выдаче. Если сайт не присутствует в ТОП100,
             то номер позиции принимается 200, и выполняется поис наиболее релевантной страницы сайта по
             запросу со словом site. Если такая не найдена, то вместо url, title, snippet в конструкторе SerpItem
             используются пустые строки.
    """
    get_pos_url_id = '''SELECT
                            t2.url_id, t2.url, pos, title, snippet
                        FROM
                            serp_items as t1
                                INNER JOIN
                            urls as t2
                                ON
                                    t1.url_id = t2.url_id
                        WHERE t1.query_id =
                        (SELECT query_id FROM queries WHERE key_words = %(query)s  AND region = %(region)s )
                        AND t2.domain_id = (SELECT id FROM domains WHERE domain = %(hostname)s )
                        ORDER BY pos ASC'''
    data = {
        'hostname': hostname,
        'query': query,
        'region': region,
    }
    rows = _read_from_db(get_pos_url_id, data)
    if len(rows) == 0:
        try:
            result = corresponding_page(query, region, hostname)
            return 0, SerpItem(200, result.url, result.title, result.snippet)
        except YaException:
            return 0, SerpItem(200, '', '', '')
    else:
        return rows[0][2], SerpItem(rows[0][0], rows[0][1], rows[0][3], rows[0][4])


def read_serp(query, region, num_res):
    """
    Возвращает выдачу яндекса по запросу


    :param query: str, запрос
    :param region: int, номер региона
    :param num_res: int, глубина ТОПа
    :return: списокб упорядоченный по позиции в выдаче кортеж объектов типа SerpItem длиной top,
        соответствующий ТОП{top} поисковой выдачи
    """

    get_serp = '''SELECT
                        urls.url_id,  serp_items.pos, urls.url, title, serp_items.snippet
                  FROM
                                (serp_items
                            INNER JOIN
                                queries
                            ON
                                serp_items.query_id = queries.query_id)
                        INNER JOIN
                            urls
                        ON
                            urls.url_id = serp_items.url_id
                  WHERE key_words = %(query)s AND region = %(region)s'''

    data_serp = {
        'query': query,
        'region': region,
    }
    rows = _read_from_db(get_serp, data_serp)
    # Если в БД нет нужного зароса (пары (запрос, регион)) записываем его в базу и считываем
    if len(rows) < num_res:
        _write_serp_db(query, region)
        rows = _read_from_db(get_serp, data_serp)
        if len(rows) < num_res:
            raise ReadSerpException("Can't read serp")

    # из rows делаем массив, нужного формата ( см ":return:")
    res = [None for i in range(num_res)]
    for row in rows:
        if row[1] - 1 < num_res:
            res[row[1] - 1] = SerpItem(row[0], row[2], row[3], row[4])
    if any([not isinstance(v, SerpItem) for v in res]):
        _write_serp_db(query, region)
        rows = _read_from_db(get_serp, data_serp)
        for row in rows:
            if row[1] - 1 < num_res:
                res[row[1] - 1] = SerpItem(row[0], row[2], row[3], row[4])
    if any([not isinstance(v, SerpItem) for v in res]):
        raise ReadSerpException("Can't read serp")
    return res


def _cut_pref(url):
    if url.startswith('http://'):
        url = url[7:]
    if url.startswith('https://'):
        url = url[8:]
    if url.startswith('www.'):
        url = url[4:]
    return url


def _write_db(list_of_sql_queries):
    con = mysql.connector.connect(**data_base_config)
    cur = con.cursor()
    for (add_query, data) in list_of_sql_queries:
        cur.execute(add_query, data)
    con.commit()
    cur.close()
    con.close()


def _write_serp_db(query, region, num_res=100):
    s = get_serp(query, region, num_res)
    if len(s) != num_res:
        raise WriteSerpException('WriteSerpException')
    for i, result in enumerate(s):
        url = unquote(result.url)
        q_list = []
        hostname = _cut_pref(urlparse(url).hostname)
        add_query = 'INSERT IGNORE INTO queries(key_words, region) VALUES (%(query)s, %(region)s)'
        data_query = {
            'query': query,
            'region': region,
        }
        q_list.append((add_query, data_query))
        add_domain = 'INSERT IGNORE INTO domains(domain) VALUES (%(hostname)s)'
        data_domain = {
            'hostname': hostname,
        }
        q_list.append((add_domain, data_domain))
        add_url = 'INSERT IGNORE INTO urls(url, title, domain_id) VALUES (%(url)s, %(title)s, ' \
                  '(SELECT id FROM domains WHERE domain = %(hostname)s))'
        data_url = {
            'url': url,
            'title': result.title,
            'hostname': hostname,
        }
        q_list.append((add_url, data_url))
        add_serp = 'INSERT INTO serp_items(query_id, url_id, pos, snippet) ' \
                   'VALUES  (' \
                   '(SELECT query_id FROM queries WHERE key_words = %(query)s AND region = %(region)s), ' \
                   '(SELECT url_id FROM urls WHERE url = %(url)s), ' \
                   '%(pos)s, ' \
                   '%(snippet)s) ' \
                   'ON DUPLICATE KEY UPDATE ' \
                   'query_id = (SELECT query_id FROM queries WHERE key_words = %(query)s AND region = %(region)s), ' \
                   'url_id = (SELECT url_id FROM urls WHERE url = %(url)s), ' \
                   'pos = %(pos)s, ' \
                   'snippet = %(snippet)s'
        data_serp = {
            'url': url,
            'query': query,
            'region': region,
            'pos': i + 1,
            'snippet': result.snippet,
        }
        q_list.append((add_serp, data_serp))
        _write_db(q_list)

"""
def get_all_queries():
    get_pos_url_id = '''SELECT
                            key_words, region
                        FROM
                            queries'''
    return _read_from_db(get_pos_url_id)
"""

def unquote(url):
    try:
        new_url = urllib.parse.unquote(url, encoding='utf-8', errors='strict')
    except UnicodeDecodeError:
        try:
            new_url = urllib.parse.unquote(url, encoding='cp1251', errors='strict')
        except UnicodeDecodeError:
            new_url = urllib.parse.unquote(url, encoding='koi8-r', errors='strict')
    return new_url

if __name__ == '__main__':

    queries = []
    """[
               'каски защитные купить',
                ]"""
    for q in queries:
        print(q)
        _write_serp_db(q, 213, 10)

