import math
from string import punctuation
import collections
import pickle
import nltk
from nltk.corpus import stopwords
import pymorphy2
import re
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


def _get_terms_from_tokens(tokens):
    pre_terms = [morph.parse(tok)[0].normal_form for tok in tokens]
    terms = []
    for term in pre_terms:
        if term in punctuation or term.isnumeric() or len(term) <= 1:
            continue
        term = _clear_term(term)
        if term not in stop_words and len(term) > 1:
            terms.append(term)
    return terms


def _get_term_list(text):
    if isinstance(text, list):
        text = ' '.join(text)
    return _get_terms_from_tokens(nltk.word_tokenize(text))


def _get_par_from_tokens(tokens):
    """Analyze tokens and return a list of token.tag"""
    # morph = pymorphy2.MorphAnalyzer()
    terms = []
    for tok in tokens:
        tok = _clear_term(tok)
        pr = morph.parse(tok)[0]
        term = pr.normal_form
        if term in punctuation or term.isnumeric() or len(term) <= 1 or term in stop_words:
            continue
        terms.append(pr)
    return terms


def _get_par_list(text):
    if isinstance(text, list):
        text = ' '.join(text)
    return _get_par_from_tokens(nltk.word_tokenize(text))


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
        self.posting_list = collections.defaultdict(lambda: collections.defaultdict(list()))

    def __contains__(self, item):
        return item in self.posting_list

    def __getitem__(self, doc_id):
        return self.posting_list[doc_id]

    def add(self, doc_id, zone, entry):
        if doc_id not in self.posting_list:
            self.posting_list[doc_id] = dict()
        if zone not in self.posting_list[doc_id]:
            self.posting_list[doc_id][zone] = list()
        self.posting_list[doc_id][zone].append(entry)

    def tf(self, doc_id, zone):
        if doc_id not in self.posting_list:
            return 0
        if zone not in self.posting_list[doc_id]:
            return 0
        return len(self.posting_list[doc_id][zone])

    def entries(self, doc_id, zone):
        if doc_id not in self.posting_list:
            return []
        if zone not in self.posting_list[doc_id]:
            return []
        return self.posting_list[doc_id][zone]


class InvertedIndex:
    """
    Обратный индекс состоит из
    - словаря, в котором ключами являются термины, а значениями соответствующие постинглисты.
    - словаря с частотами терминов в русском языке
    - таблицы длин документов коллекции
    """
    p_type = 'idf'

    def __init__(self, idfs='C:\\_Work\\SemanticCore\\bm_25\\NKRL.csv'):
        self.IDF = text_analysis.load_ruscorpra_frqs()
        self.index = dict()
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
        if term not in self.index:
            self.index[term] = PostingList()
        self.index[term].add(docid, zone, e)
        self.doc_lens.update(docid, zone)

    def save(self, name):
        with open('{}.pickle'.format(name), 'wb') as f_out:
            pickle.dump(self, f_out)

    @staticmethod
    def load(name):
        with open('{}.pickle'.format(name), 'rb') as fin:
            return InvertedIndex(pickle.load(fin))

    def get_document_frequency(self, word, docid):
        """
        количество вхождений слова в документ
        :param word: str
        :param docid: int
        :return: int
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

    def score(self, doc_id, zone, query):
        """
        Считает значение показателя релевантности документа запросу по формулам из статьи
        «Алгоритм текстового ранжирования Яндекса» с РОМИП-2006
        http://download.yandex.ru/company/03_yandex.pdf
        :param doc_id:
        :param zone:
        :param query:
        :return:
        """
        k0 = 0.3
        k1 = 0.1
        k2 = 0.2
        k3 = 0.02
        terms = _get_par_list(query)

        if len(terms) == 1:
            return self.w_single(doc_id, zone, terms)
        w_s = self.w_single(doc_id, zone, terms)
        w_p = self.w_pair(doc_id, zone, terms)
        w_a = self.w_all_words(doc_id, zone, terms)
        w_ph = self.w_phrase(doc_id, zone, terms)
        w_h = self.w_half_phrase(doc_id, zone, terms)
        res = w_s + k0 * w_p + k1 * w_a + k2 * w_ph + k3 * w_h
        print('{7:<20} {0: 3d}: {1: 3.2f} = {2: 3.2f} + k0 * {3: 3.2f} + k1 * {4: 3.2f} + k2 * {5: 3.2f}'
              ' + k3 * {6: 3.2f}'.format(doc_id, res, w_s, w_p, w_a, w_ph, w_h, zone))
        return res

    def w_all_words(self, doc_id, zone, terms):
        """
        Wallwords — вклад вхождения всех терминов (форма слова не важна) из запроса;
        :param doc_id:
        :param zone:
        :param terms:
        :return:
        """
        n_miss = 0
        idfs = 0
        for term in terms:
            if term.normal_form not in self \
                    or doc_id not in self[term.normal_form] \
                    or zone not in self[term.normal_form][doc_id]:
                n_miss += 1
            else:
                for e in self[term.normal_form][doc_id][zone]:
                    if e.props == term.tag:
                        idfs += self.log_p(term.normal_form)
        return idfs * 0.03 ** n_miss

    def w_phrase(self, doc_id, zone, terms):
        """
        Wphrase — вклад вхождения всего запроса (фразы), учитываются формы слов
        :param doc_id:
        :param zone:
        :param terms:
        :return:
        """
        tf = self.doc_length(doc_id, zone)
        for term in terms:
            if term.normal_form not in self \
                    or doc_id not in self[term.normal_form] \
                    or zone not in self[term.normal_form][doc_id]:
                continue

            for e in self[term.normal_form][doc_id][zone]:
                if e.props == term.tag:
                    tf = min(tf, len(self[term.normal_form][doc_id][zone]))
        if tf == 0:
            return tf
        idfs = sum(self.log_p(term) for term in terms)
        return idfs * tf / (1 + tf)

    def w_half_phrase(self, doc_id, zone, terms):
        """
        Whalfphrase — вклад вхождения части запроса, учитываются формы слов
        :param doc_id:
        :param zone:
        :param terms:
        :return:
        """
        tf = self.doc_length(doc_id, zone)
        counter = 0
        for term in terms:
            if term.normal_form not in self \
                    or doc_id not in self[term.normal_form] \
                    or zone not in self[term.normal_form][doc_id]:
                continue

            for e in self[term.normal_form][doc_id][zone]:
                if e.props == term.tag:
                    tf = min(tf, len(self[term.normal_form][doc_id][zone]))
                    counter += tf > 0
        if 2 * counter < len(terms):
            return 0
        return sum(self.log_p(term) for term in terms) * tf / (1 + tf)

    ZONE_COEFFICIENT = {'body': 1, 'title': 2, 'h1': 1.5}

    def total_score(self, doc_id, query):
        res = 0
        sum([self.w_half_phrase(doc_id, zone, _get_par_list(query)) for zone in ['body', 'title', 'h1']])
        for zone in ['body', 'title', 'h1']:
            sc = self.score(doc_id, zone, query)
            res += self.ZONE_COEFFICIENT[zone] * sc
        print('------------------')
        print('{0:<20}: {1: 3.2f}'.format('total', res))
        print('==================')
        return res

    def w_single(self, doc_id, zone, terms):
        """
        W_single — вклад отдельных слов из запроса;
        :param doc_id:
        :param zone:
        :param terms:
        :return:
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

    def w_pair(self, doc_id, zone, terms):
        """
        Wpair — вклад пар слов;
        :param doc_id:
        :param zone:
        :param terms:
        :return:
        """
        terms = [t.normal_form for t in terms]
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
            tf = double_tf(positions0, positions1)
            tf_spec = double_spec_tf(positions0, positions2)
            summat += (s0 + s1) * tf / (1 + tf)
            summat += (s0 + s2) * tf_spec / (1 + tf_spec)
            j += 1
            positions0 = positions1
            positions1 = positions2
            s0 = s1
            s1 = s2

        tf = double_tf(positions0, positions1)
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


def double_tf(positions0, positions1):
    tf = 0
    for p1 in positions0:
        if p1 + 1 in positions1:
            tf += 1
        if p1 - 1 in positions1:
            tf += 0.5
        if p1 + 2 in positions1:
            tf += 0.5
    return tf


def double_spec_tf(positions0, positions2):
    tf = 0
    for p1 in positions0:
        if p1 + 1 in positions2:
            tf += 0.1
    return tf


class DocumentLengthTable:
    def __init__(self):
        self.table = dict()

    def __contains__(self, item):
        return item in dict

    def __len__(self):
        return len(self.table)

    def doc_ids(self):
        return self.table.keys()

    def add(self, docid, length):
        self.table[docid] = dict()
        self.table[docid]['body'] = length

    def update(self, docid, zone):
        if docid not in self.table:
            self.table[docid] = dict()
        if zone not in self.table[docid]:
            self.table[docid][zone] = 0
        self.table[docid][zone] += 1

    def get_length(self, docid, zone):
        if docid in self.table and zone in self.table[docid]:
            return self.table[docid][zone]
        else:
            return 0

    def count_average_length(self):
        """
        Вычисляет среднюю длину документа в коллекции в словах
        :return: float
        """
        summat = 0
        for docid in self.table:
            length = 0
            for zone in self.table[docid]:
                length += self.table[docid][zone]
            summat += length
        return float(summat) / float(len(self.table))


def all_entries(text):
    """
    Возвращяет словарь вида {термин: список вхождений (Entry)
    данного термина в соответствующий текст} для всех слов, встречающихся в тексте, кроме стоп-слов
    :param text: str
    :return:
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
    # build inverted index
    :param corpus_of_readable: list of text_analysis.Readable objects
    :return: InvertedIndex
    """
    idx = InvertedIndex()
    for docid, c in enumerate(corpus_of_readable):
        for zone in ['body', 'title', 'h1']:
            entries = all_entries(c.get_zone(zone))
            for term in entries:
                for e in entries[term]:
                    idx.add(term, docid, zone, e)
    return idx


if __name__ == '__main__':

    def readables_by_queries_file(f_name):
        """
        Эта функция была создана для тестирования

        Возвращяет список объектов типа text_analysis.Readable. соответствующих всем элементам ТОП10 поисковой выдачи
        по всем запросам из файла f_name
        :param f_name:
        :return:
        """
        from text_analysis import Readable
        import search_engine_tk.ya_query as sps
        from content import WebPage

        readable_s = []
        for q in sps.queries_from_file(f_name, 213):
            print(q.query)
            for u in q.get_urls(top=10):
                w = WebPage(u).html()
                readable_s.append(Readable(w))
        return readable_s


    indx = build_idx(readables_by_queries_file(f_name='C:\\_Work\\vostok\\2.txt'))
    for i in indx.doc_ids():
        print(indx.total_score(i, 'купить сапоги'))
