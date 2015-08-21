"""
Строит обратный индекс коллекции текстов и считает значение показателя релевантности документа запросу по формулам из
статьи «Алгоритм текстового ранжирования Яндекса» с РОМИП-2006. http://download.yandex.ru/company/03_yandex.pdf
"""

import math
import collections
import re
from string import punctuation

import pymorphy2
import nltk
from nltk.corpus import stopwords

import text_analysis


punctuation += "«—»"  # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
stop_words = stopwords.words('russian')
stop_words.extend(['это', 'дата', 'смочь', 'хороший', 'нужный',
                   'перед', 'весь', 'хотеть', 'цель', 'сказать', 'ради', 'самый', 'согласно',
                   'около', 'быстрый', 'накануне', 'неужели', 'понимать', 'ввиду', 'против',
                   'близ', 'поперёк', 'никто', 'понять', 'вопреки', 'твой', 'объектный',
                   'вместо', 'идеальный', 'целевой', 'сила', 'благодаря', 'знаешь',
                   'вследствие', 'знать', 'прийти', 'вдоль', 'вокруг', 'мочь', 'предлагать',
                   'наш', 'всей', 'однако', 'очевидно', "намного", "один", "по-прежнему",
                   'суть', 'очень', 'год', 'который', 'usd'])
morph = pymorphy2.MorphAnalyzer()


def _clear_term(term):
    term = re.sub(r'[^\w\s]+|[\d]+|•', r' ', term).strip().lower()
    return term


def _get_pars_from_tokens(tokens):
    """
    По списку слов возвращает список объектов pymorphy2.analyzer.Parse, которые содержат в себе нормальную форму
    слова, а так же информацию о том в какой форме слово представлено в списке tokens. Стоп-слова и знаки препинания
    в результат не добвляются.

    Parameters
    ----------
    tokens: list of str,
        список слов
    Returns
    -------
    pars: list of pymorphy2.analyzer.Parse
    """
    pre_pars = [morph.parse(_clear_term(tok))[0] for tok in tokens]
    pars = []
    for par in pre_pars:
        term = par.normal_form
        if term not in punctuation and len(term) > 1 and term not in stop_words:
            pars.append(par)
    return pars


def _get_par_list(text):
    if isinstance(text, list):
        text = ' '.join(text)
    return _get_pars_from_tokens(nltk.word_tokenize(text))


class Entry:
    """
    Вхождение слова в текст, характеризуется возицией и некоторыми свойствами, отвечающими за форму слова (термина).
    Далее в качестве свойств используются объекты класса (pymorphy2.tagset.OpencorporaTag),
    """
    def __init__(self, position, props=None):
        self.position = position
        self.props = props

    def __str__(self):
        return '<Entry> (position = {}, properties = {})'.format(self.position, self.props)


class PostingList:
    """
    Постинглист термина - это словарь, ключами которого являются id документов коллекции,
    в которых этот термин встречается, а значениями являются словари вида {Зона документа: список вхождений (Entry)
    данного термина в соответствующую зону соответствующего документа}. Зонами могут быть title, h1, body и т.д.
    """
    def __init__(self):
        self.posting_list = collections.defaultdict(lambda: collections.defaultdict(lambda: list()))

    def __contains__(self, item):
        return item in self.posting_list

    def __getitem__(self, doc_id):
        return self.posting_list[doc_id]

    def add(self, doc_id, zone, entry):
        self[doc_id][zone].append(entry)

    def tf(self, doc_id, zone):
        """
        Возвращает число вхождений термина, для которого создан постинглист в соответствующую зону документа с id
        равным doc_id
        Parameters
        ----------
        doc_id: int,
           id документа
        zone: str,
            зона документа, принимает одно из значений 'body', 'title', 'h1'
        Returns
        -------
        tf: int
            число вхождений термина
    """
        return len(self.posting_list[doc_id][zone])

    def entries(self, doc_id, zone):
        return self.posting_list[doc_id][zone]


class InvertedIndex:
    """
    Обратный индекс состоит из
    - словаря, в котором ключами являются термины, а значениями соответствующие постинглисты.
    - словаря с частотами терминов в русском языке
    - таблицы длин документов коллекции
    """

    p_type = 'idf'
    ZONE_COEFFICIENT = {'body': 1, 'title': 2, 'h1': 1.5}

    def __init__(self):
        self.IDF = text_analysis.load_ruscorpra_probabilities()
        self.index = collections.defaultdict(lambda: PostingList())
        self.doc_lens = DocumentLengthTable()

    def doc_ids(self):
        return self.doc_lens.doc_ids()

    def __contains__(self, item):
        return item in self.index

    def __getitem__(self, term):
        return self.index[term]

    def doc_length(self, docid, zone):
        return self.doc_lens.get_length(docid, zone)

    def add(self, term, docid, zone, e):
        self.index[term].add(docid, zone, e)
        self.doc_lens.update(docid, zone)

    def get_document_frequency(self, word, docid):
        """
        количество вхождений слова в документ
        
        Parameters
        ----------
        word: str
        docid: int, id документа

        Returns
        -------
        int, document_frequency
        """

        # morph = pymorphy2.MorphAnalyzer()
        term = morph.parse(word)[0].normal_form
        if term in self.index:
            return self.index[word].tf(docid)
        else:
            return 0

    def get_index_frequency(self, word):
        """
        частота слова в индексе
        """
        return self.IDF.prob(word)

    def total_score(self, doc_id, query):
        """
        Считает значение показателя релевантности документа запросу по формулам из статьи
        «Алгоритм текстового ранжирования Яндекса» с РОМИП-2006
        http://download.yandex.ru/company/03_yandex.pdf
        
        Parameters
        ----------
        docid: int, id документа
        query: str

        Returns
        -------
        float
        """

        res = 0
        #sum([self.w_half_phrase(doc_id, zone, _get_par_list(query)) for zone in ['body', 'title', 'h1']])
        for zone in ['body', 'title', 'h1']:
            sc = self.score(doc_id, zone, query)
            res += self.ZONE_COEFFICIENT[zone] * sc
        print('------------------')
        print('{0:<20}: {1: 3.2f}'.format('total', res))
        print('==================')
        return res

    def score(self, doc_id, zone, query):
        """
        Считает значение показателя релевантности заданной зоны документа.
        
        Parameters
        ----------
        docid: int, id документа
        zone: str, зона документа, принимает одно из значений 'body', 'title', 'h1'
        query:str

        Returns
        -------
        float
        """
        k0 = 0.3
        k1 = 0.1
        k2 = 0.2
        k3 = 0.02
        pars = _get_par_list(query)

        if len(pars) == 1:
            return self.w_single(doc_id, zone, pars)
        w_s = self.w_single(doc_id, zone, pars)
        w_p = self.w_pair(doc_id, zone, pars)
        w_a = self.w_all_words(doc_id, zone, pars)
        w_ph = self.w_phrase(doc_id, zone, pars)
        w_h = self.w_half_phrase(doc_id, zone, pars)
        res = w_s + k0 * w_p + k1 * w_a + k2 * w_ph + k3 * w_h
        print('{7:<20} {0: 3d}: {1: 3.2f} = {2: 3.2f} + k0 * {3: 3.2f} + k1 * {4: 3.2f} + k2 * {5: 3.2f}'
              ' + k3 * {6: 3.2f}'.format(doc_id, res, w_s, w_p, w_a, w_ph, w_h, zone))
        return res

    def w_all_words(self, doc_id, zone, pars):
        """
        W_allwords — вклад вхождения всех терминов (форма слова не важна) из запроса;
        
        Parameters
        ----------
        docid: int,
            id документа
        zone: str,
            зона документа, принимает одно из значений 'body', 'title', 'h1'
        pars: list of pymorphy2.analyzer.Parse

        Returns
        -------
        float
        """
        n_miss = 0
        idfs = 0
        for par in pars:
            if par.normal_form not in self \
                    or doc_id not in self[par.normal_form] \
                    or zone not in self[par.normal_form][doc_id]:
                n_miss += 1
            else:
                for e in self[par.normal_form][doc_id][zone]:
                    if e.props == par.tag:
                        idfs += self.log_p(par.normal_form)
        return idfs * 0.03 ** n_miss

    def w_phrase(self, doc_id, zone, pars):
        """
        Wphrase — вклад вхождения всего запроса (фразы), учитываются формы слов
        TF — число вхождений запроса целиком.
        Считаем сколько раз встречается в тексте самое редкое для данного текста слово запроса.
        
        Parameters
        ----------
        docid: int,
            id документа
        zone: str,
            зона документа, принимает одно из значений 'body', 'title', 'h1'
        pars: list of pymorphy2.analyzer.Parse

        Returns
        -------
        :float
        """
        tf = self.doc_length(doc_id, zone)
        for par in pars:
            term = par.normal_form
            if self[term].tf(doc_id, zone) == 0:
                return 0
            for entry in self[term][doc_id][zone]:
                if entry.props == par.tag:
                    tf = min(tf, self[term].tf(doc_id, zone))
        idfs = sum(self.log_p(par.normal_form) for par in pars)
        return idfs * tf / (1 + tf)

    def w_half_phrase(self, doc_id, zone, pars):
        """
        W_halfphrase — вклад вхождения части запроса, учитываются формы слов
        TF — число вхождений большей половины слов запроса.
        
        Parameters
        ----------
        docid: int,
            id документа
        zone: str,
            зона документа, принимает одно из значений 'body', 'title', 'h1'
        pars: list of pymorphy2.analyzer.Parse

        Returns
        -------
        :float
        """
        tfs = collections.defaultdict(lambda: 0)
        for par in pars:
            term = par.normal_form
            for entry in self[term][doc_id][zone]:
                if entry.props == par.tag:
                    tfs[par] += 1

        non_zero_tfs = sorted(list(filter(lambda u: u[1] > 0, tfs.items())), key=lambda u: u[1], reverse=True)
        counter = len(non_zero_tfs)
        if 2 * counter < len(pars):
            return 0
        tf = non_zero_tfs[round(len(pars)/2)-1][1]
        return sum(self.log_p(par.normal_form) for par, v in non_zero_tfs) * tf / (1 + tf)

    def w_single(self, doc_id, zone, terms):
        """
        W_single — вклад отдельных слов из запроса;
        
        Parameters
        ----------
        docid: int,
            id документа
        zone: str,
            зона документа, принимает одно из значений 'body', 'title', 'h1'
        pars: list of pymorphy2.analyzer.Parse

        Returns
        -------
        :float
        """
        terms = [t.normal_form for t in terms]
        res = 0
        k1 = 1
        k2 = 1 / 350
        for term in terms:
            if term not in self:
                continue
            tf = self[term].tf(doc_id, zone)
            l_p = self.log_p(term)
            res += l_p * tf / (tf + k1 + k2 * self.doc_length(doc_id, zone))
        return res

    def _positions(self, doc_id, zone, term):
        if term not in self:
            return []
        return [e.position for e in self.index[term].entries(doc_id, zone)]

    def w_pair(self, doc_id, zone, pars):
        """
        Parameters
        ----------
        docid: int,
            id документа
        zone: str,
            зона документа, принимает одно из значений 'body', 'title', 'h1'
        pars: list of pymorphy2.analyzer.Parse

        Returns
        -------
        :float
        """
        terms = [t.normal_form for t in pars]
        if len(terms) < 2:
            return 0
        j = 0
        summat = 0
        positions0 = self._positions(doc_id, zone, terms[0])
        positions1 = self._positions(doc_id, zone, terms[1])

        s0 = self.log_p(terms[0])
        s1 = self.log_p(terms[1])
        while j + 2 < len(terms):
            s2 = self.log_p(terms[2])
            positions2 = self._positions(doc_id, zone, terms[j + 2])
            tf = pair_tf(positions0, positions1)
            tf_spec = pair_spec_tf(positions0, positions2)

            summat += (s0 + s1) * tf / (1 + tf)
            summat += (s0 + s2) * tf_spec / (1 + tf_spec)
            j += 1
            positions0 = positions1
            positions1 = positions2
            s0 = s1
            s1 = s2

        tf = pair_tf(positions0, positions1)
        summat += (s0 + s1) * tf / (1 + tf)
        return summat

    def log_p(self, term, p_type='p_on_topic'):
        if p_type == 'p_on_topic':
            p = 1 - math.exp(-1.5 * self.IDF.prob(term))
        elif p_type == 'ICF':
            p = 1 / self.IDF.prob(term)
        else:
            raise Exception('Wrong p_type')
        return abs(math.log(p))


def pair_tf(positions0, positions1):
    """
    TF — количество вхождений пары слов, с учетом весов. Пара учитывается, когда слова запроса встречаются в тексте
    подряд (+1),
    через слово (+0.5) или
    в обратном порядке (+0.5).

    Parameters
    ----------
    positions0: list of int,
        список позиций вхождений первого слова
    positions1: list of int,
        список позиций вхождений второго слова

    Returns
    -------
    :float
        количество вхождений пары слов, с учетом весов
    """
    tf = 0
    for p1 in positions0:
        if p1 + 1 in positions1:
            tf += 1
        if p1 - 1 in positions1:
            tf += 0.5
        if p1 + 2 in positions1:
            tf += 0.5
    return tf


def pair_spec_tf(positions0, positions2):
    """
    Количество вхождений пары слов специальный случай, когда слова, идущие в запросе через одно,
    в тексте встречаются подряд (+0.1).

    Parameters
    ----------
    positions0: list of int,
        список позиций вхождений первого слова
    positions1: list of int,
        список позиций вхождений второго слова

    Returns
    -------
    :float
         количество вхождений пары слов, специальный случай
    """
    tf = 0
    for p1 in positions0:
        if p1 + 1 in positions2:
            tf += 0.1
    return tf


class DocumentLengthTable:
    """
    Таблица длин документов коллекции (в словах). Состоит из словаря, ключами которого выступают idшники документов, а
    значениями - опять словари. Ключами внутренних словарей являются зоны соответствующего документа, а значениями -
    количества слов в тексте соответствующей зоны.
    """

    def __init__(self):
        self.table = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))

    def doc_ids(self):
        """
        Возвращяет idшники документов, длины которых содержатся в таблице

        Returns
        -------
        a set-like object:
        """
        return self.table.keys()

    def update(self, docid, zone):
        """
        Слова их документа в индекс добавляются по очереди, поэтому, при считывании нового слова, длина текущей
        зоны документа увеличивается на 1. Подучается, что DocumentLengthTable хранятся количество обработанных
        слов документа с идентификатором docid в зоне документа zone
        
        Parameters
        ----------
            docid: int, id документа
            zone: str, зона документа, принимает одно из значений 'body', 'title', 'h1'
        """
        self.table[docid][zone] += 1

    def get_length(self, docid, zone=None):
        """
        Возращяет длину зоны документа, есле она указана. Иначе, длину всего документа.
        Длиной документа считается сумма длин всех его зон.

        Parameters
        ----------
        docid: int,
            id документа
        zone: str,
            зона документа, принимает одно из значений 'body', 'title', 'h1'

        Returns
        -------
        :float, длина документа
        """
        if zone:
            return self.table[docid][zone]
        else:
            return sum([len(zone_length) for zone_length in self.table[docid].values])

    def count_average_length(self):
        """
        Вычисляет среднюю длину документа в коллекции в словах. Длиной документа считается сумма длин всех его зон.

        Returns
        -------
        :float,
            средняя длина документа в коллекции
        """
        summat = 0
        for docid in self.table:
            summat += self.get_length(docid)
        return float(summat) / float(len(self.table))


def all_entries(text):
    """
    Возвращяет словарь вида {термин: список вхождений (Entry)
    данного термина в соответствующий текст} для всех слов, встречающихся в тексте, кроме стоп-слов
    
    Parameters
    ----------
    text: str

    Returns
    -------
    entries: dict(),
        словарь вида {термин(str): list of Entry objects}
    """
    entries = dict()
    # morph = pymorphy2.MorphAnalyzer()
    pos = 0
    for tok in nltk.word_tokenize(text):
        pos += 1
        term = morph.parse(tok)
        term_nf = term[0].normal_form
        if term_nf in stop_words or term_nf in punctuation:
            continue
        if term_nf not in entries:
            entries[term_nf] = list()
        entries[term_nf].append(Entry(pos, term[0].tag))
    return entries


def build_idx(corpus_of_readable):
    """
    Строит обратный индекс
    
    Parameters
    ----------
    corpus_of_readable:
        list of text_analysis.Readable objects, коллекция текстов
    Returns
    -------
    idx: InvertedIndex,
        обратный индекс коллекции текстов corpus_of_readable
    """
    idx = InvertedIndex()
    for docid, c in enumerate(corpus_of_readable):
        for zone in ['body', 'title', 'h1']:
            entries = all_entries(c.get_zone_text(zone))
            for term in entries:
                for e in entries[term]:
                    idx.add(term, docid, zone, e)
    return idx


if __name__ == '__main__':

    def readables_by_queries(queries):
        """
        Эта функция была создана для тестирования

        Возвращяет список объектов типа text_analysis.Readable. соответствующих всем элементам ТОП10 поисковой выдачи
        по всем запросам из файла f_name
        
        Parameters
        ----------
        f_name: str,
            имя файла со списком запросов.
        Returns
        -------
        readables: list of text_analysis.Readable,
            тексты всех документов ТОП10 поисковой выдачи  по каждому из запросов из файла f_name
        """
        from text_analysis import Readable
        import search_engine_tk.ya_query as sps
        from web_page_content import WebPageContent

        readables = []
        for query in queries:
            q = sps.YaQuery(query, 213)
            for u in q.get_urls(top=10):
                w = WebPageContent(u).html()
                readables.append(Readable(w))
        return readables

    QUERIES = ['методы кластеризации']
    indx = build_idx(readables_by_queries(QUERIES))
    for i in indx.doc_ids():
        print(indx.total_score(i, 'Методы кластерного анализа'))
