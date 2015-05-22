import math
import os
import sqlite3
import re
import nltk
import nltk.text
import pymorphy2
from bm_25.invdx import Entry, InvertedIndex, DocumentLengthTable
from read.htmls import norm_title
import read
from nltk.util import bigrams, ngrams
from string import punctuation
from nltk.corpus import stopwords
import bs4


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
        return '<Readable object>'

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


class HttpCodeException(Exception):
    pass


def get_alts(response):
    alts = [a.text() for a in list(response.select('//img/@alt')) if len(a.text()) > 0]
    return alts


def _text_sub(html_doc, rex):
    p = re.compile(rex)
    return p.sub(' ', html_doc)





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



def make_corp(readables, path):
    for i, r in enumerate(readables):
        with open(os.path.join(path, str(i+10) + '.txt'), mode='w', encoding='utf8') as fout:
            fout.write(r.title() + '/n' + r.text())


def make_corp_file():
    import search_query.ya_query as sps
    from search_query.content import WebPage
    readables = []
    for q in sps.queries_from_file('C:\\_Work\\vostok\\000.txt', 213)[1:]:
        print(q.query)
        for u in q.get_urls(10):
            try:
                readables.append(Readable(WebPage(u).html()))
            except:
                print(u)
    make_corp(readables, 'C:\\_Work\\vostok\\corp')


if __name__ == '__main__':
    from nltk.corpus import PlaintextCorpusReader
    corpus_root = 'C:\\_Work\\vostok\\corp'
    wordlists = PlaintextCorpusReader(corpus_root, fileids='.+[.]txt')
    print(len(wordlists.raw()))

    print(wordlists.words('145.txt'))
    wl = []
    for fid in wordlists.fileids():
        for w in list(wordlists.words(fid)):
            if not (w in punctuation or w in stop_words):
                wl.append(w)

    text1 = nltk.text.Text(wl)
    # print(text1.similar("респираторы"))
    # print(text1.common_contexts(["Респиратор", "полумаска"]))
    fdist1 = nltk.FreqDist(text1)
    lapprob = nltk.LaplaceProbDist(fdist1)
    print(lapprob.prob('Полная'))
    print(lapprob.prob('оффшор'))
    # print(fdist1.most_common())
    # print(text1.collocations())
    #fdist1.plot(500, cumulative=False)
    print(nltk.pos_tag(text1))


