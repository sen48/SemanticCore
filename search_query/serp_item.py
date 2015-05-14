import re
from urllib.parse import urlparse


def get_hostname(url):
    return urlparse(url).hostname


class SerpItem:
    URL_COMM_WORDS = ['shop', 'catalog/', 'cat/', 'katalog/', 'card', 'product', 'item']
    URL_INF_WORDS = ['news', 'info', 'blog', 'forum', 'article', 'help', 'wiki']
    PATTERN_COMM_WORDS = re.compile(r'(?:^|[\s])(?:купить|прода(?:ть|жа|м|ем)|товар(?:а|ы|ов)?|руб[. ]|'
                                    r'в розницу|опт(?:ом)?|приобр(?:ести|рети|ай)|заказ(?:а|ы|ать|жи)|'
                                    r'доставк(?:а|у|ой)|магазин(?:а|е)|цен(?:а|ы|е)|'
                                    r'акци(?:я|е|и|q))(?:$|[\s])')
    PATTERN_INF_WORDS = re.compile(r'(?:^|[\s])(?:wiki.*|youtube|обзор|вики.*|новост(?:ь|и|ью|ей)|'
                                   r'что такое|где|Фото|видео)(?:$|[\s])')

    def __init__(self, url_id, url, title, snippet):
        self.url_id = url_id
        self.url = url
        self.title = title
        self.snippet = snippet

    def get_domain(self):
        return get_hostname(self.url)

    def _check_comm_inf_words(self, string):

        """

        :param string: str
        :return:return: str
            возвращается пара целых чисел, первое(второе) равно количеству вхождений в string
            слов-маркеров коммерческости(информационности) запроса.
        """

        comm = len(self.PATTERN_COMM_WORDS.findall(string.lower()))
        inf = len(self.PATTERN_INF_WORDS.findall(string.lower()))
        return comm, inf

    def check_url(self):
        c = sum([w in self.url.lower() for w in self.URL_COMM_WORDS])
        i = sum([w in self.url.lower() for w in self.URL_INF_WORDS])
        return c, i

    def check_title(self):
        """
        :return: (c,i) c - int, i - int
        возвращается пара целых чисел, первое(второе) равно количеству вхождений в title
        слов-маркеров коммерческости(информационности) запроса.
        """
        return self._check_comm_inf_words(self.title)

    def check_snippet(self):
        """
        :return: возвращается пара целых чисел, первое(второе) равно количеству вхождений в snippet
        слов-маркеров коммерческости(информационности) запроса.
        """
        return self._check_comm_inf_words(self.snippet)

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