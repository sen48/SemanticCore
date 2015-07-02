import os
import re
import math
import nltk
import nltk.text
import pickle
import numpy
import pymorphy2
from read.htmls import norm_title
import read
from string import punctuation
from nltk.corpus import stopwords
from nltk.corpus import PlaintextCorpusReader
import bs4
from search_query import ya_query

stop_words = stopwords.words('russian')
stop_words.extend(['это', 'дата', 'смочь', 'хороший', 'нужный',
                   'перед', 'весь', 'хотеть', 'цель', 'сказать', 'ради', 'самый', 'согласно',
                   'около', 'быстрый', 'накануне', 'неужели', 'понимать', 'ввиду', 'против',
                   'близ', 'поперёк', 'никто', 'понять', 'вопреки', 'твой', 'объектный',
                   'вместо', 'идеальный', 'целевой', 'сила', 'благодаря', 'знаешь',
                   'вследствие', 'знать', 'прийти', 'вдоль', 'вокруг', 'мочь', 'предлагать',
                   'наш', 'всей', 'однако', 'очевидно', "намного", "один", "по-прежнему",
                   'суть', 'очень', 'год', 'который', 'usd'])

# ZONE_COEFFICIENT = {'body': 1.0, 'title': 2, 'h1': 1.5, 'h2': 1.25, 'h3': 1.125, 'alt': 1.0}


class Readable(read.Document):
    """
    Соотвестсвует основному содержимому web-страницы. Т.е. без воковых меню, шапок и подвалов.
    """

    def __str__(self):
        return '<Readable object> {}'.format(self.title())

    def text(self):
        """
        Извлекает основной текст
        :return: str
        """
        soup = bs4.BeautifulSoup(self.summary())
        text = soup.get_text()
        '''text = text.replace('#', '').replace('↵', '').replace('↑', '').replace('°', '').replace('©', '').
        replace('«', '').
        replace('»', '').replace('$', '').replace('*', '').replace(u"\.", "").replace(u"\,", "").replace(u"^", "").
        replace(u"|", "").replace(u"—", "").replace(u',', '').replace(u'•', '').replace(u'﴾', '').replace(u'﴿', '')'''
        p = re.compile(r'[\n\r][\n\r \t\a]+')
        text = p.sub('\n', text)
        return text

    def _get_tag(self, tag):
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
        return self._get_tag('h1')[0]

    def get_all_h2(self):
        return self._get_tag('h2')

    def get_zone(self, zone):
        if zone == 'body':
            return self.text()
        elif zone == 'title':
            return self.title()
        elif zone == 'h1':
            return self.h1()
        else:
            return self._get_tag(zone)[0]


class HttpCodeException(Exception):
    pass


def get_alts(response):
    alts = [a.text() for a in list(response.select('//img/@alt')) if len(a.text()) > 0]
    return alts


def _text_sub(html_doc, rex):
    p = re.compile(rex)
    return p.sub(' ', html_doc)


def make_plain_text_files(queries_file, path):
    """

    :param path:
    """
    import search_query.ya_query as sps
    i = 0
    for q in sps.queries_from_file(queries_file, 2)[1:]:
        print(q.query)
        make_plain_text_files_urls(q.get_urls(10),path)


def make_plain_text_files_urls(urls, path):

    from search_query.content import WebPage
    for u in urls:
        try:
            r = Readable(WebPage(u).html())
            with open(os.path.join(path, str(hash(u)) + '.txt'), mode='w', encoding='utf8') as f_out:
                f_out.write(r.title() + '/n' + r.text())
        except:
            print(u)


def collocations(corpus_dir):
    word_lists = PlaintextCorpusReader(corpus_dir, fileids='.+[.]txt')
    text = nltk.text.Text([token for sentence in word_lists.sents() for token in sentence if token not in punctuation])
    return text.collocations()

from nltk.compat import Counter

def words_freq(sents):
    morph = pymorphy2.MorphAnalyzer()
    text1 = Counter([morph.parse(tok)[0].normal_form
                            for s in sents
                            for tok in nltk.word_tokenize(s)
                            if tok not in punctuation and tok not in stop_words])
    return nltk.FreqDist(text1)


if __name__ == '__main__':
    '''import search_query.serps as wrs
    semcorefile = 'C:\\_Work\\lightstar\\to_filter.csv'
    region = 213
    #num_res = 10
    queries = [ya_query.YaQuery(q, region) for q in wrs.queries_from_file(semcorefile)]
    comp = []
    for i, q in enumerate(queries):
        try:
            comp.append(q.competitiveness( 10))
        except:
            pass
        print(i, '{', q.query, ':', comp[-1], '}')
    print('done')

    pickle.dump(comp, open("comp", mode='wb'))'''

    comp = pickle.load(open("comp", mode='rb'))


    import pylab as pl
    pl.figure()
    pl.plot([i for i in range(len(comp))], sorted(comp, reverse=True))
    #pl.hist(comp, 50)
    pl.show()



    """corpus_root = 'C:\\_Work\\lightstar\\corp'
    # make_plain_text_files('c:/_Work/lightstar/to_markers_ws.txt',corpus_root)

    corp = PlaintextCorpusReader(corpus_root, fileids='.+[.]txt')
    print('PlaintextCorpusRead')
    text1 = nltk.text.Text([tok for s in corp.sents() for tok in s if tok not in punctuation])
    print('Text')
    print(text1.collocations())
    print(text1.similar("люстра"))
    print(text1.similar("торшер"))
    print(text1.similar("светильник"))
    print(text1.similar("потолочный"))
    print(text1.common_contexts(["люстра", "торшер"]))

    morph = pymorphy2.MorphAnalyzer()

    print('Text_term')
    del text1
    wl = []
    for s in corp.sents():
        for tok in s:
            if tok not in punctuation:
                tm = morph.parse(tok)[0].normal_form
                if tm not in stop_words:
                    wl.append(tm)
    text1 = nltk.text.Text(wl)
    print(text1.collocations())
    print(text1.similar("люстра"))
    print(text1.similar("торшер"))
    print(text1.similar("светильник"))
    print(text1.similar("потолочный"))
    print(text1.common_contexts(["люстра", "торшер"]))

    fdist1 = nltk.FreqDist(text1)
    fdist1.plot(500, cumulative=False)
    # print(nltk.pos_tag(text1))"""
