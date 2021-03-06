"""
Модуль предназначен для получения данных поисковой выдачи Яндекса.
Чтобы не тратить лимиты Яндекс XML на одни и те же запросы, данные поисковой выдачи записываются в базу данных MySQL
(см. serps.mwb)
Требует доработки. Например, не проверяется цедосность данных полученных от Яндекс XML. Иногда просто зависает.
"""

import urllib

import mysql.connector

from datetime import date, timedelta
from search_engine_tk.serp.yaxml import corresponding_page, get_serp
from search_engine_tk.serp_item import SerpItem


SERP_LIFETIME = 14  # days

DB_CONFIG = {
    'user': 'root',
    'password': '135',
    'host': 'localhost',
    'database': 'serps',
    'raise_on_warnings': True,
}


class ReadSerpException(Exception):
    pass


class WriteSerpException(Exception):
    pass


def read_serp(query, region, num_res):
    """
    Возвращает выдачу яндекса по запросу. Сначала подходящие записи ищутся в БД, если их там нет, или меньше, чем
    num_res, или записи устарели, то полечаем нвые через Яндеки XML и записываем в базу.

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
    : list of SerpItem objects
        список, упорядоченный по позиции в выдаче список объектов типа SerpItem длиной top,
        соответствующий ТОП{top} поисковой выдачи
    """

    select_serp = '''SELECT
                        urls.url_id,  serp_items.pos, urls.url, title, serp_items.snippet, queries.record_date
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
    rows = _read_from_db(select_serp, data_serp)

    # Если в БД нет нужного зароса (пары (запрос, регион)) или снято меньше позиций, чем нужно,
    # или записям больше SERP_LIFETIME дней, записываем в базу и считываем.

    if len(rows) < num_res or (len(rows) > 0 and date.today() - rows[0][5] > timedelta(days=SERP_LIFETIME)):
        _write_serp_db(query, region)
        rows = _read_from_db(select_serp, data_serp)
        if len(rows) < num_res:
            raise ReadSerpException("Can't read serp")

    # из rows делаем массив, нужного формата ( см ":return:")
    res = [None] * num_res
    for row in rows:
        if row[1] - 1 < num_res:
            res[row[1] - 1] = SerpItem(row[0], row[2], row[3], row[4])
    if any([not isinstance(v, SerpItem) for v in res]):
        _write_serp_db(query, region)
        rows = _read_from_db(select_serp, data_serp)
        for row in rows:
            if row[1] - 1 < num_res:
                res[row[1] - 1] = SerpItem(row[0], row[2], row[3], row[4])
    if any([not isinstance(v, SerpItem) for v in res]):
        raise ReadSerpException("Can't read serp")
    return res


def read_url_position(hostname, query, region):
    """
    Возвращяет позицию релевантной страницы и соответствующий ей объект класса SerpItem.
    Parameters
    ----------
    hostname: str,
        адрес сайта без http:// и www
    query: str,
        поисковый запрос запрос
    region: int,
        номер региона по нумерации яндекса
    Returns
    -------
    rows[0][2]: int
        номера позиции
    : SerpItem
         соответствует странице сайта в поисковой выдаче. Если сайт не присутствует в ТОП100,
         то номер позиции принимается 200, и выполняется поис наиболее релевантной страницы сайта по
         запросу со словом site. Если такая не найдена, то вместо url, title, snippet в конструкторе SerpItem
         используются пустые строки.
    """
    # TODO: сделать проверку наличия актуальных данных в базе
    read_serp(query, region, 100)  # вместо проверки

    select_pos_url_id = '''SELECT
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
    rows = _read_from_db(select_pos_url_id, data)
    if len(rows) == 0:
        result = corresponding_page(query, region, hostname)
        if result:
            return 0, SerpItem(200, result.url, result.title, result.snippet)
        else:
            return 0, SerpItem(200, '', '', '')
    else:
        return rows[0][2], SerpItem(rows[0][0], rows[0][1], rows[0][3], rows[0][4])


def _cut_pref(url):
    if url.startswith('http://'):
        url = url[7:]
    if url.startswith('https://'):
        url = url[8:]
    if url.startswith('www.'):
        url = url[4:]
    return url


def _read_from_db(get_str, data=None):
    res = []
    try:
        con = mysql.connector.connect(**DB_CONFIG)
        cur = con.cursor()
        cur.execute(get_str, data)
        res = list(cur)
        cur.close()
    except mysql.connector.Error as err:
        if err.errno == mysql.connector.errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
        elif err.errno == mysql.connector.errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)
    else:
        con.close()
    return res


def _write_db(list_of_sql_queries):
    con = mysql.connector.connect(**DB_CONFIG)
    cur = con.cursor()
    for (add_query, data) in list_of_sql_queries:
        cur.execute(add_query, data)
    con.commit()
    cur.close()
    con.close()


def _write_serp_db(query, region, num_res=100):
    del_old_serp_items = 'DELETE FROM serp_items WHERE ' \
                         'query_id = (SELECT query_id FROM queries WHERE key_words = %(query)s AND region = %(region)s)'
    add_query = 'INSERT INTO queries(key_words, region, record_date) VALUES (%(query)s, %(region)s, %(date)s)' \
                'ON DUPLICATE KEY UPDATE record_date = %(date)s'
    data_query = {
        'query': query,
        'region': region,
        'date': str(date.today()),
    }
    _write_db([(add_query, data_query), (del_old_serp_items, data_query)])
    s = get_serp(query, region, num_res)
    if len(s) != num_res:
        raise WriteSerpException('WriteSerpException')
    for i, result in enumerate(s):
        url = unquote(result.url)
        q_list = []
        hostname = _cut_pref(urllib.parse.urlparse(url).hostname)

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

    queries = ['часы']

    for q in queries:
        _write_serp_db(q, 213, 10)
        read_serp(q, 213, 10)

