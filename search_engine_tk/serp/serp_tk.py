from urllib.request import urlopen
import pycurl
from io import BytesIO
import urllib
import urllib.parse
import math
import os.path
import random
import time
import tkinter
from bs4 import BeautifulSoup
import chardet
from grab import Grab
import lxml.html

from ips import IPS2, IPS1

CAPTCHA = ''
SE = 'rambler'


LIMIT_TRIES_CHOSE_PROXY = len(IPS2)
# Взято с сайта: http://fineproxy.org/#ixzz3QlZ31sQ9
AGENTS = ['Mozilla/5.0 (X11; Ubuntu; Linux i686; rv:13.0) Gecko/20100101 Firefox/13.0',
          'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:8.0) Gecko/20100101 Firefox/8.0',
          'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.3) Gecko/2008092417 Firefox/3.0.3',
          'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
          'Chrome/37.0.2049.0 Safari/537.36',
          'User-Agent: Mozilla/5.0 (Windows NT 5.1; rv:11.0) Gecko/20100101 Firefox/11.0',
          'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) '
          'Chrome/40.0.2214.94 Safari/537.36']


class HttpCodeException(Exception):
    pass


class ProxyException(Exception):
    pass


class SearchEngineException(Exception):
    pass


class ScriptException(Exception):
    pass


class Proxy:
    def __init__(self, num=None):
        self.userpwd = 'pervuchin:artem'
        if num and 0 <= num <= len(IPS2) - 1:
            self.prx = IPS2[num]
        else:
            self.prx = IPS2[random.randint(0, len(IPS2) - 1)]
        prpt = self.prx.split(':')
        self.ip = prpt[0]
        self.port = int(prpt[1])


def get_page(url, proxy=None, agent_num=0):
    c = pycurl.Curl()
    c.setopt(c.URL, url)

    buffer = BytesIO()
    c.setopt(c.WRITEDATA, buffer)
    c.get_body = buffer.getvalue

    head_buffer = BytesIO()
    c.setopt(pycurl.HEADERFUNCTION, head_buffer.write)
    c.get_head = head_buffer.getvalue

    if proxy:
        c.setopt(pycurl.PROXY, proxy.ip)
        c.setopt(pycurl.PROXYPORT, proxy.port)
        c.setopt(pycurl.PROXYTYPE, pycurl.PROXYTYPE_HTTP)
        c.setopt(pycurl.PROXYUSERPWD, proxy.userpwd)
        c.setopt(pycurl.COOKIEJAR, "cookies{}.txt".format(proxy.ip))

    c.setopt(pycurl.FOLLOWLOCATION, 1)
    c.setopt(pycurl.MAXREDIRS, 5)
    c.setopt(pycurl.CONNECTTIMEOUT, 60)
    c.setopt(pycurl.TIMEOUT, 120)
    c.setopt(pycurl.NOSIGNAL, 1)
    if 0 < agent_num < len(AGENTS):
        ag = AGENTS[agent_num]
    else:
        ag = AGENTS[0]
    c.setopt(pycurl.USERAGENT, ag)
    http_header = [
        'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language: ru-ru,ru;q=0.8,en-us;q=0.5,en;q=0.3',
        'Accept-Charset:utf-8;q=0.7,*;q=0.5',
        'Connection: keep-alive',
    ]
    c.setopt(pycurl.HTTPHEADER, http_header)
    time.sleep(1)

    try:
        c.perform()
    except pycurl.error:
        raise ScriptException("Script failure, please check log messages.")

    if c.getinfo(pycurl.HTTP_CODE) != 200:
        raise HttpCodeException('HTTP code is {0}'.format(c.getinfo(pycurl.HTTP_CODE)))

    return c


def is_valid_site(site):
    return len(site) > 0


def test_agents(url='http://example.com/'):
    for i in range(0, len(IPS2)):
        print(i)
        pr = Proxy(i)
        print(pr.ip)
        try:
            html = get_page(url, proxy=pr)
            with open('html.get_body', mode='w') as f:
                f.write(html.get_body())
        except HttpCodeException as e:
            print(e)


def get_serp(query, region, num_res=10):
    return site_list(query, region, 0, num_res=num_res)


def site_list(key, region, page=0, site='', search_engine=SE, num_res=10) -> list:
    s_list = []
    if search_engine == 'rambler':
        item_titles = get_rambler(key, region, page, site, num_res)
    elif search_engine == 'yandex':
        item_titles = get_yandex(key, region, page, site)
    else:
        raise SearchEngineException('Unknown search engine')
    # get_serp_items_grab_(params, find_class, captcha_class)
    for h2 in item_titles:
        b = h2.getchildren()
        if len(b):
            url = h2.getchildren()[0].attrib['href']
            s_list.append(url)
    return s_list


def corresponding_page(key, region, site):
    return site_list(key, region, site)[0]


def get_rambler(key, region, page, site='', num_res=10):
    params = ['http', 'nova.rambler.ru', '/search', '', '', '']
    if is_valid_site(site) != 0:
        key += ' site:' + site
    params[4] = urllib.parse.urlencode({
        'query': key,
        'region': region,
        'page': page + 1,
        'correct': 'off',
        'pagelen': num_res,
    })
    return get_serp_items_(params, 'b-serp__list_item_title', 'b-captcha')


def get_yandex(key, region, page, site=''):
    params = ['http', 'yandex.ru', '/search', '', '', '']
    if is_valid_site(site) != 0:
        key += ' site:' + site
    params[4] = urllib.parse.urlencode({
        'text': key,
        'lr': region,
        'page': page,
        'correct': 'off',
    })
    return get_serp_items_(params, "search_engine_tk-item__title clearfix", "b-captcha__image")


def get_serp_items_grab_(params, find_class, captcha_class):
    from grab import Grab, GrabError
    g = Grab()

    for tries in range(0, LIMIT_TRIES_CHOSE_PROXY):
        proxy = IPS2[random.randint(0, len(IPS2) - 1)]
        url = urllib.parse.urlunparse(params)
        g.setup(url=url, proxy=proxy, proxy_type='http', connect_timeout=50, timeout=50)
        with open("cookies{}.txt".format(proxy), mode='w') as f:
            pass
        g.setup(cookiefile="cookies{}.txt".format(proxy))
        print(url)
        if proxy in IPS1:
            g.setup(proxy_userpwd='pervuchin:artem')
        try:
            g.request()
        except GrabError:
            IPS2.remove(proxy)
            print('{} FAIL'.format(proxy))
            continue
        print(g.response.code)
        res = []
        try:
                if g.response.code != 200:
                    IPS2.remove(proxy)
                    raise HttpCodeException('HTTP code is {0},{1}'.format(g.response.code, proxy))
                res = (list(g.response.select('//h2[@class="{}"]/a'.format(find_class))))  #
                print(len(res))
                if len(res) == 0:
                    r = g.response.select('//*[@class="{}"]/img/@src'.format(captcha_class))
                    print(len(r))
                    image_url = r[0].text()
                    if len(r) == 0:
                        IPS2.remove(proxy)
                        raise SearchEngineException("Captcha wasn't found")
                    cpt = message(image_url)
                    params2 = ['http', 'nova.rambler.ru', '/check_captcha', '', '', '']
                    path = 'scroll=1&utm_source=nhp&query={}'.format(params[4]['query'])
                    key = image_url.replace('http://u.captcha.yandex.net/image?key=', '')
                    params2[4] = urllib.parse.urlencode({
                            'path': path,
                            'key': key,
                            'text': cpt,
                    })
                    url = urllib.parse.urlunparse(params2)
                    print(url)
                    #g.setup(url=url, proxy=proxy, proxy_type='http', connect_timeout=50, timeout=50)
                    #g.request()
                    raise SearchEngineException('Captcha!!!')
                return [r.attr('href') for r in res]
        except (HttpCodeException, SearchEngineException, GrabError) as e:
            print(str(e))


def get_serp_items_(params, find_class, captcha_class):
    pr = Proxy()
    ag_num = random.randint(0, len(AGENTS) - 1)
    print(pr.ip, str(ag_num))

    url = urllib.parse.urlunparse(params)
    print(url)
    for tries in range(0, LIMIT_TRIES_CHOSE_PROXY):
        try:
            html = get_page(url, pr, ag_num)
            with open('html.get_body', mode='w', encoding='utf-8') as f:
                f.write(str(html.get_body()))
            html_body = html.get_body()
            res = lxml.html.fromstring(html_body).find_class(find_class)
            while len(res) == 0:
                r = lxml.html.fromstring(html_body).find_class(captcha_class)
                if len(r) == 0:
                    raise SearchEngineException("Captcha wasn't found")
                print('Captcha')
                raise SearchEngineException("SE doesn't answer")
            return res
        except (SearchEngineException, ScriptException, HttpCodeException, lxml.etree.ParserError) as e:
            print(str(e))
            IPS2.remove(pr.prx)
            print(len(IPS2))
            if len(IPS2) == 0:
                raise ProxyException("No one valid proxy")
            pr = Proxy()
            ag_num = random.randint(0, len(AGENTS) - 1)
            print(pr.ip, str(ag_num))


def cut_www(site):
    print(site)
    if site.startswith('http://'):
        site = site[7:]
    if site.startswith('www.'):
        site = site[4:]
    print(site)
    return site


def site_list_grab(key, page, region, num_res=10):
    params = ['http', 'nova.rambler.ru', '/search', '', '', '']
    params[4] = urllib.parse.urlencode({
        'query': key,
        'region': region,
        'page': page + 1,
        'correct': 'off',
        'pagelen': num_res,
    })
    return get_serp_items_grab_(params, 'b-serp__list_item_title', 'b-captcha')


def site_pos_url(key, site, region, num_res=100, max_position=10):
    for pg in range(0, int(math.ceil(max_position / num_res))):
        print("page: {0}".format(pg))
        site = cut_www(site)
        num = 0
        for url in site_list_grab(key, pg, region, num_res):  # l:  #
            num += 1
            if cut_www(url).startswith(site):
                return num_res * pg + num, url
    return max_position + 1, corresponding_page(key, site)


def get_text_from_html(url):
    html_doc = urlopen(url).read()
    soup = BeautifulSoup(html_doc)
    print(chardet.detect(html_doc))
    # soup.originalEncoding
    return soup.get_text()


def write_files_by_query(w_dir, query):
    cur_dir = os.path.join(w_dir, query)
    if os.path.isdir(cur_dir):
        for n in os.listdir(cur_dir):
            os.remove(os.path.join(cur_dir, n))
    else:
        os.mkdir(os.path.join(w_dir, query))
    for pos, s, url in site_list(query):
        print(url)
        name = "{0}_{1}.txt".format(pos, s)
        with open(os.path.join(os.path.join(w_dir, query), name), encoding='utf-8', mode='w') as f_out:
            text = get_text_from_html(url)
            f_out.write(text)


def message(gif_link):
    img = Grab()
    img.go(gif_link)
    img.response.save('yy.gif')
    print(img.response.url())

    root = tkinter.Tk()
    im = tkinter.PhotoImage(file='yy.gif')
    #canvas = tkinter.Canvas(root)

    #canvas.grid(row = 0, column = 0)


    #canvas.create_image(0,0, image=photo)
    captcha = ''
    l = tkinter.Label(root, image=im)
    l.pack()
    ed = tkinter.Entry()
    ed.pack()

    def hello(ev):
        captcha = ed.get()
        global root
        root.destroy()
    btn = tkinter.Button(text='Ok')
    btn.bind("<Button-1>", hello)
    btn.pack()
    root.mainloop()
    return captcha



if __name__ == '__main__':
    site_list_grab('lenovo', 0)
    message('http://u.captcha.yandex.net/image?key=333UUAGLBRygq6zUsChGMbhsHAySvXN1')
    #site_list_grab('Нечеткие отношения', 0)
