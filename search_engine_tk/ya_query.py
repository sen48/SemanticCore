import math
import nltk
import re
import search_engine_tk.serp as wrs
from search_engine_tk.serp_metrics import rating_dist


class YaQuery:
    """
    Соответствует запросу в Яндексе. Однозначно определяется строкой поискового запроса и регионом поиска.
    """
    def __init__(self, query, region):
        """
        Parameters
        ----------
        query: str
               Поисковый запрос
        region: int
                Номер региона Яндекс по которому производится поиск
        """
        self.query = query
        self.region = region

    def __eq__(self, other):
        return self.query == other.query and self.region == other.region

    def get_strict_query(self):
        """
        Возвращает строгий запрос для Вордстата Яндекс
        Returns
        -------
        YaQuery, регион остается тот-же, а в запросе перед каждым словом ставится
        восклицательный знак
        """
        return YaQuery('!' + ' !'.join(nltk.word_tokenize(self.query)), self.region)

    def get_serp(self, top):
        """
        Возвращает выдачу Яндекс XML
        Parameters
        ----------
        top: int,
            глубина ТОПа
        Returns
        -------
            tuple of SerpItem
            упорядоченный по позиции в выдаче кортеж объектов типа SerpItem длиной top,
            соответствующий ТОП{top} поисковой выдачи
        """
        print('.', end='.')
        return wrs.read_serp(self.query, self.region, top)

    def get_position_page(self, site):
        """
        Возвращает позицию и страницу, соответствующие данному сайту в поисковой выдаче по данному запросу.
        Parameters
        ----------
        site: str,
                адрес сайта без http:// и www
        Returns
        -------
             пара состоящая из номера позиции и объекта класса SerpItem,
             соответствующего странице сайта в поисковой выдаче. Если сайт не присутствует в ТОП100,
             то номер позиции принимается 200, и выполняется поис наиболее релевантной страницы сайта по
             запросу со словом site. Если такая не найдена, то вместо url, title, snippet в конструкторе SerpItem
             используются пустые строки.
        """
        return wrs.read_url_position(site, self.query, self.region)

    def get_url_ids(self, top):
        """
        Возвращает idшников urlов поисковой выдачи. Обычно используется для расчетов,
        где нужно проверить urlы на раверство, но не нужны сами urlы.
        Parameters
        ----------
        top: int
            глубина ТОПа
        Returns
        -------
                list
                упорядоченный по позиции в выдаче список idшников urlов в БД длиной top,
                соответствующий ТОП{top} поисковой выдачи
        """
        return [item.url_id for item in self.get_serp(top)]

    def get_domains(self, top):
        """
        Возвращает hostname urlов поисковой выдачи.
        Parameters
        ----------
        top: int
            глубина ТОПа
        Returns
        -------
                list
                упорядоченный по позиции в выдаче список hostname urlов длиной top,
                соответствующий ТОП{top} поисковой выдачи
        """
        return [item.get_domain() for item in self.get_serp(top)]

    def get_urls(self, top):
        """
        Возвращает urls поисковой выдачи.
        Parameters
        ----------
        top: int
            глубина ТОПа
        Returns
        -------
                generator
                упорядоченный по позиции в выдаче список urlов в БД длиной top,
                соответствующий ТОП{top} поисковой выдачи
        """
        for item in self.get_serp(top):
            yield item.url

    def count_commercial(self, top):
        """
        Мера коммерческости запроса, чем больше значение, тем более коммерческий запрос. Не нормированна.
        Parameters
        ----------
        top: int
            глубина ТОПа
        Returns
        -------
                int
                сумма числа вхождений слов-маркеров коммерческости запроса в urlы,
                titleы, snippetы ТОП{top} поисковой выдачи по данному запросу
        """
        return sum([item.count_commercial() for item in self.get_serp(top)])

    def count_informational(self, top):
        """
        Мера информационности запроса, чем больше значение, тем более информационный запрос. Не нормированна.
        Parameters
        ----------
        top: int
            глубина ТОПа top: глубина ТОПа
        Returns
        -------
                int
                сумма числа вхождений слов-маркеров информационности запроса в urlы,
                 titleы, snippetы ТОП{n} поисковой выдачи по данному запросу
        """
        return sum([item.count_informational() for item in self.get_serp(top)])

    def competitiveness(self, top):
        """
        Мера конкурентности запроса, чем больше значение, тем больше конкуренция. Принимает значения от 0 до 1.
        Parameters
        ----------
        top: int
            глубина ТОПа
        Returns
        -------
                float
                мера конкурентности запроса, вычесленная с помощтю rating_dist
        """
        serp = self.get_url_ids(top)
        strict_serp = self.get_strict_query().get_url_ids(top)
        return 1 - rating_dist(serp, strict_serp)

    def complexity(self, site, top):
        """
        Мера сложности продвижения, чем больше значение, тем сложнее продвигать. Не нормирована.
        Parameters
        ----------
        top: int
            глубина ТОПа
        Returns
        -------
                float
                мера сложности продвижения сайтом по данному запросу, равная произведению
        меры конкурентности запроса на логорифм позиции, занимаемой сайтом в выдачи
        """
        position = self.get_position_page(site)[0]
        if position == 0:
            position = 200
        return self.competitiveness(top) * math.log(position+1, 2)


def queries_from_file(file, region):
    """
    Возвращает список объектов YaQuery, соответствующих запросам из файла
    Parameters
    ----------
    file: str,
    region: int, регион поиска

    Returns
    -------
    :list of YaQuery
    """
    try:
        queries = []
        for line in open(file, mode='r', encoding='utf-8'):
            line = re.sub(r'[^\w\s]+|[\d]+|•', r' ', line).strip().lower()
            query = YaQuery(line, region)
            if query in queries or len(query.query) == 0:
                continue
            queries.append(query)
        return queries
    except IOError as er:  # Обработка отсутствия файла
        print(u'Can\'t open the "{0}" file'.format(er.filename))
        raise er
