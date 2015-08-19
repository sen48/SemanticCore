import re
from urllib.parse import urlparse


def get_hostname(url):
    return urlparse(url).hostname


class SerpItem:
    """
    Соответствует позиции в поисковой выдаче
    """
    URL_COMM_WORDS = ['shop', 'catalog', 'cat/', 'katalog/', 'card', 'product', 'category', 'stock', ]
    URL_INF_WORDS = ['news', 'info', 'blog', 'forum', 'article', 'help', 'wiki', 'topic', 'story', 'post', 'news',
                     'otvet.mail.ru']
    PATTERN_COMM_WORDS = re.compile(r'(?:^|[\s]|[.])(?:'
                                    r'[к]?купить|'
                                    r'прода(?:ть|жа|м|ем)|'
                                    r'товар(?:а|ы|ов)?|руб[. ]|'
                                    r'розниц(?:а|у|ой)|'
                                    r'опт(?:ом)?|'
                                    r'приобр(?:ести|рети|ай)|'
                                    r'заказ(?:а|ы|ать|жи)|'
                                    r'доставк(?:а|у|ой)|'
                                    r'магазин(?:а|е)|'
                                    r'цен(?:а|ы|е)|'
                                    r'акци(?:я|е|и|q)|'
                                    r'каталог|'
                                    r'клиент'
                                    r')(?:$|[\s]|[.])')
    PATTERN_INF_WORDS = re.compile(r'(?:^|[\s]|[.])(?:'
                                   r'wiki.*|'
                                   r'youtube|'
                                   r'обзор|'
                                   r'вики.*|'
                                   r'новост(?:ь|и|ью|ей)|'
                                   r'что такое|'
                                   r'где|'
                                   r'Фото|'
                                   r'видео|'
                                   r'как|'
                                   r'блог|'
                                   r'в картинках|'
                                   r'своими руками|'
                                   r'схема|'
                                   r'ремонт|'
                                   r'какой|'
                                   r'зачем|'
                                   r'почему'
                                   r')(?:$|[\s]|[.])')

    def __init__(self, url_id, url, title, snippet):
        self.url_id = url_id
        self.url = url
        self.title = title
        self.snippet = snippet

    def get_domain(self):
        return get_hostname(self.url)

    def _count_comm_inf_words(self, string):
        """

        :param string: str
        :return:return: str
            возвращается пара целых чисел, первое(второе) равно количеству вхождений в string
            слов-маркеров коммерческости(информационности) запроса.
        """

        comm = len(self.PATTERN_COMM_WORDS.findall(string.lower()))
        inf = len(self.PATTERN_INF_WORDS.findall(string.lower()))
        return comm, inf

    def count_comm_inf_url(self):
        c = sum([w in self.url.lower() for w in self.URL_COMM_WORDS])
        i = sum([w in self.url.lower() for w in self.URL_INF_WORDS])
        return c, i

    def count_comm_inf_title(self):
        """
        :return: (c,i) c - int, i - int
        возвращается пара целых чисел, первое(второе) равно количеству вхождений в title
        слов-маркеров коммерческости(информационности) запроса.
        """
        return self._count_comm_inf_words(self.title)

    def count_comm_inf_snippet(self):
        """
        :return: возвращается пара целых чисел, первое(второе) равно количеству вхождений в snippet
        слов-маркеров коммерческости(информационности) запроса.
        """
        return self._count_comm_inf_words(self.snippet)

    def count_commercial(self):
        """
        :return: возвращается сумма числа вхождений слов-маркеров коммерческости запроса в url,
        title, snippet.
        """
        # количество вхождений слов-маркеров коммерческости запроса в url:
        u = sum([w in self.url.lower() for w in self.URL_COMM_WORDS])
        # количество вхождений слов-маркеров коммерческости запроса в title:
        t = len(self.PATTERN_COMM_WORDS.findall(self.title.lower()))
        # количество вхождений слов-маркеров коммерческости запроса в snippet:
        s = len(self.PATTERN_COMM_WORDS.findall(self.snippet.lower()))
        res = u + t + s
        return res

    def count_informational(self):
        """
        :return: возвращается сумма числа вхождений слов-маркеров информационности запроса в url,
        title, snippet.
        """
        # количество вхождений слов-маркеров информационности запроса в url:
        u = sum([w in self.url.lower() for w in self.URL_INF_WORDS])
        # количество вхождений слов-маркеров информационности запроса в title:
        t = len(self.PATTERN_INF_WORDS.findall(self.title.lower()))
        # количество вхождений слов-маркеров информационности запроса в snippet:
        s = len(self.PATTERN_INF_WORDS.findall(self.snippet.lower()))
        return u + t + s

    def count_commercial_wp(self):
        from web_page_content import WebPageContent
        wp = WebPageContent(self.url)
        return wp.count_commercial()

    def count_informational_wp(self):
        from web_page_content import WebPageContent
        wp = WebPageContent(self.url)
        return wp.count_informational()