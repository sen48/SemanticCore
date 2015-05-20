import math
import os
import sqlite3
import re

import nltk
import pandas
import pymorphy2

#from readability.readability import Document
from bm_25.invdx import Entry, InvertedIndex, DocumentLengthTable
from read.htmls import norm_title
import read
from grab import Grab, GrabError
from nltk.util import bigrams, ngrams
from string import punctuation
from nltk.corpus import stopwords
# from serp_yaxml import site_pos_url
# import write_read_serp
# import random
import readability
import bs4

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

ZONES = {'body': 1, 'title': 2, 'h1': 3, 'h2': 4, 'h3': 5, 'alt': 6}
ZONE_COEFFICIENT = {'body': 1.0, 'title': 2, 'h1': 1.5, 'h2': 1.25, 'h3': 1.125, 'alt': 1.0}
DB_PATH = 'C:/_Work/ContentAnalisys/'
DB_FILE = 'TextsDB'
NUM_DOCS = 38369
TOTAL_LEMMS = 52138
LOG_BASE = 2
INDEX = {}
MIN_DF = 3
MIN_CF = 0.3


class Readable(read.Document):

    def __str__(self):
        return ''

    def text(self):
        soup = bs4.BeautifulSoup(self.summary())
        text = soup.get_text()
        '''text = text.replace('#', '').replace('↵', '').replace('↑', '').replace('°', '').replace('©', '').replace('«',
                                                                                                                 ''). \
            replace('»', '').replace('$', '').replace('*', '').replace(u"\.", "").replace(u"\,", "").replace(u"^", ""). \
            replace(u"|", "").replace(u"—", "").replace(u',', '').replace(u'•', '').replace(u'﴾', '').replace(u'﴿', '')'''
        p = re.compile(r'[\n\r][\n\r \t\a]+')
        text = p.sub('\n', text)
        return text

    def get(self, tag):
        doc = self._html(True)
        res = []
        for e in list(doc.iterfind('.//'+tag)):
            if e.text and len(norm_title(e.text)) != 0:
                res.append(norm_title(e.text))
                continue
            if e.text_content() and len(norm_title(e.text_content())) != 0:
                res.append(norm_title(e.text_content()))
        return res

    def h1(self):
        """


        :return: list of h1 texts
        """
        return self.get('h1')[0]

    def get_all_h2(self):
        return self.get('h2')

    def get_zone(self, zone):
        if zone == 'boby':
            return self.text()
        elif zone == 'title':
            return self.title()
        elif zone == 'h1':
            return self.h1()
        else:
            return self.get(zone)[0]

    def all_entries(self):
        entries = dict()
        morph = pymorphy2.MorphAnalyzer()
        for zone in ['boby', 'title', 'h1']:
            text = self.get_zone(zone)
            pos = 0
            for tok in nltk.word_tokenize(text):
                pos += 1
                term = morph.parse(tok)
                term_nf = term[0].normal_form
                if term_nf in stop_words or term_nf in punctuation:
                    continue
                if term_nf not in entries:
                        entries[term_nf] = list()
                entries[term_nf].append(Entry(pos, zone, term[0].tag))
        return entries


class HttpCodeException(Exception):
    pass


def get_alts(response):
    alts = [a.text() for a in list(response.select('//img/@alt')) if len(a.text()) > 0]
    return alts


def _text_sub(html_doc, rex):
    p = re.compile(rex)
    return p.sub(' ', html_doc)


def _clear_term(term):
    s = term[-1]
    while not s.isalpha():
        term = term[:-1]
        if len(term) > 0:
            s = term[-1]
        else:
            s = 'a'
    if len(term) < 2:
        return term
    s = term[0]
    while not s.isalpha():
        term = term[1:]
        if len(term) > 0:
            s = term[0]
        else:
            s = 'a'
    return term


def _get_terms_from_tokens(tokens):
    morph = pymorphy2.MorphAnalyzer()
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

def log_p(term, p_type):
    if p_type == 'p_on_topic':
        field = 'Freq'
        total = NUM_DOCS
        min_val = MIN_CF
        func = lambda cf, d: 1 - math.exp(-1.5 * cf / d)
    elif p_type == 'IDF':
        field = 'Doc'
        total = NUM_DOCS
        min_val = MIN_DF
        func = lambda df, d: d / df
    elif p_type == 'ICF':
        field = 'Freq'
        total = TOTAL_LEMMS
        min_val = MIN_CF
        func = lambda cf, total_lemms: total_lemms / cf
    else:
        raise Exception('Wrong p_type')

    with sqlite3.connect(os.path.join(DB_PATH, DB_FILE)) as con:
        cur = con.cursor()
        fs = cur.execute('''SELECT ''' + field + ''' FROM freq WHERE Lemma = :term''', {'term': term}).fetchall()
    if len(fs) == 0:
        f = round(min_val * 10 ** 6 / NUM_DOCS)
    else:
        f = round(fs[0][0] * 10 ** 6 / NUM_DOCS)
    p = func(float(f), total)

    return abs(math.log(p))


def term_freq(document, zone, term):
    if term not in document.tfs:
        return 0
    return document.tfs[term][zone]


def w_single(document, zone, terms, p_type):
    assert isinstance(document, Readable)
    res = 0
    k1 = 1
    k2 = 1 / 350
    for term in terms:
        tf = term_freq(document, zone, term)
        l_p = log_p(term, p_type)
        res += l_p * tf / (tf + k1 + k2 * document.length[zone])
    return res


def _sum_p_tf(terms, tf, p_type):
    if tf == 0:
        return 0
    s = 0
    for term in terms:
        s += log_p(term, p_type)
    return s * tf / (1 + tf)


def pair_weight(document, zone, pair, p_type):
    if pair[0] not in document.tfs or pair[1] not in document.tfs:
        return 0
    tf = 0
    positions1 = document.index[pair[0]][zone]
    positions2 = document.index[pair[1]][zone]
    for p1 in positions1:
        if p1 + 1 in positions2:
            tf += 1
        if p1 - 1 in positions2:
            tf += 0.5
        if p1 + 2 in positions2:
            tf += 0.5
    s = log_p(pair[0], p_type) + log_p(pair[1], p_type)
    return s * tf / (1 + tf)


def triple_weight(document, zone, triple, p_type):
    if triple[0] not in document.tfs or triple[2] not in document.tfs:
        return 0
    tf = 0
    positions1 = document.index[triple[0]][zone]
    positions2 = document.index[triple[2]][zone]
    for p1 in positions1:
        if p1 + 1 in positions2:
            tf += 0.1
    s = log_p(triple[0], p_type) + log_p(triple[1], p_type)
    return s * tf / (1 + tf)


def w_pair(document, zone, terms, p_type):
    if len(terms) < 2:
        return 0
    pairs = bigrams(terms)
    triples = ngrams(terms, 3)
    w_p = 0
    for pair in pairs:
        w_p += pair_weight(document, zone, pair, p_type)
    for triple in triples:
        w_p += triple_weight(document, zone, triple, p_type)
    return w_p


def w_all_words(document, zone, terms, p_type):
    n_miss = 0
    idfs = 0
    for term in terms:
        if term not in document.tfs or document.tfs[term][zone] == 0:
            n_miss += 1
        else:
            idfs += log_p(term, p_type)
    return idfs * 0.03 ** n_miss


def w_phrase(document, zone, query, p_type):
    tf = 0
    for sent in document.sent[zone]:
        sent = sent.lower()
        index = sent.find(query.lower())
        if index != -1:
            tf += 1
    tokens = nltk.word_tokenize(query)
    terms = _get_terms_from_tokens(tokens)
    idfs = sum(log_p(term, p_type) for term in terms)
    return idfs * tf / (1 + tf)


def w_half_phrase(document, zone, terms, idfs, p_type):
    idf = sum(idfs.values())
    tf = 0
    for sent in document.sent[zone]:
        sent_terms = _get_term_list(sent.lower())
        term_in_sent = [term for term in terms if term in sent_terms]
        sent_idf = sum([idfs[term] for term in term_in_sent])
        if sent_idf > 0.5 * idf:
            tf += 1
    return sum(log_p(term, p_type) for term in terms) * tf / (1 + tf)


def total_score(document, query, p_type='p_on_topic'):
    res = 0
    for zone in ZONES:
        sc = score(document, zone, query, p_type)
        res += ZONE_COEFFICIENT[zone] * sc
    print('------------------')
    print('{0:<20}: {1: 3.2f}'.format('total', res))
    print('==================')
    return res


def score(document, zone, query, p_type='p_on_topic'):
    k0 = 0.3
    k1 = 0.1
    k2 = 0.2
    k3 = 0.02
    tokens = nltk.word_tokenize(query)
    terms = _get_terms_from_tokens(tokens)
    idfs = {}
    with sqlite3.connect(os.path.join(DB_PATH, DB_FILE)) as con:
        cur = con.cursor()
        for term in terms:
            fs = cur.execute('''SELECT Doc FROM freq WHERE Lemma = :term''', {'term': term}).fetchall()
            if len(fs) == 0:
                idfs[term] = NUM_DOCS / MIN_DF
            else:
                idfs[term] = NUM_DOCS / fs[0][0]
    if len(terms) == 1:
        return w_single(document, zone, terms, p_type)
    w_s = w_single(document, zone, terms, p_type)
    w_p = w_pair(document, zone, terms, p_type)
    w_a = w_all_words(document, zone, terms, p_type)
    w_ph = w_phrase(document, zone, query, p_type)
    w_h = w_half_phrase(document, zone, terms, idfs, p_type)
    res = w_s + k0 * w_p + k1 * w_a + k2 * w_ph + k3 * w_h
    print('{7:<20} {0: 3d}: {1: 3.2f} = {2: 3.2f} + k0 * {3: 3.2f} + k1 * {4: 3.2f} + k2 * {5: 3.2f}'
          ' + k3 * {6: 3.2f}'.format(document.id, res, w_s, w_p, w_a, w_ph, w_h, zone))
    return w_s + k0 * w_p + k1 * w_a + k2 * w_ph + k3 * w_h


def _find_common_terms(documents):
    return [term for term in documents[0].tfs if sum(int(term in doc.tfs) for doc in documents) >= 0.8 * len(documents)]


def build_data_structures(corpus_of_readable, file=None):
    idx = InvertedIndex(file)
    dlt = DocumentLengthTable()
    for docid, c in enumerate(corpus_of_readable):
        # build inverted index
        for e in c:
            entries = c.all_entries()
            for term, entry in entries:
                idx.add(term, docid, entry)
        # build document length table
        length = len(corpus_of_readable[docid]) #FIX:
        dlt.add(docid, length)
    return idx, dlt




if __name__ == '__main__':
    url = 'http://wiki.python.su/%D0%94%D0%BE%D0%BA%D1%83%D0%BC%D0%B5%D0%BD%D1%82%D0%B0%D1%86%D0%B8%D0%B8/BeautifulSoup'
    url = 'http://dvhb.ru/#'
    g = Grab()
    g.setup()
    g.go(url)
    doc = Readable(g.doc.unicode_body())
    print(doc.title())
    print(doc.h1())
    print([e for e in doc.all_entries()])
    print(doc.text())