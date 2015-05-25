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
            if not (w in punctuation or w in []):
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


