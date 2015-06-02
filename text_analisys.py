import os
import re
import nltk
import nltk.text
import pymorphy2
from read.htmls import norm_title
import read
from string import punctuation
from nltk.corpus import stopwords
from nltk.corpus import PlaintextCorpusReader
import bs4

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

    def __str__(self):
        return '<Readable object> {}'.format(self.title())

    def text(self):
        soup = bs4.BeautifulSoup(self.summary())
        text = soup.get_text()
        '''text = text.replace('#', '').replace('↵', '').replace('↑', '').replace('°', '').replace('©', '').
        replace('«', '').
        replace('»', '').replace('$', '').replace('*', '').replace(u"\.", "").replace(u"\,", "").replace(u"^", "").
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
        if zone == 'body':
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


def make_corp(readable_docs, path):
    for i, r in enumerate(readable_docs):
        with open(os.path.join(path, str(i) + '.txt'), mode='w', encoding='utf8') as f_out:
            f_out.write(r.title() + '/n' + r.text())


def make_corp_file(path):
    import search_query.ya_query as sps
    from search_query.content import WebPage
    i = 0
    for q in sps.queries_from_file('c:/_Work/lightstar/to_markers_ws.txt', 2)[1:]:
        print(q.query)
        for u in q.get_urls(10):
            try:
                r = Readable(WebPage(u).html())
                with open(os.path.join(path, str(i) + '.txt'), mode='w', encoding='utf8') as f_out:
                    f_out.write(r.title() + '/n' + r.text())
                i += 1
            except:
                print(u)


def collocations(corpus_dir):
    word_lists = PlaintextCorpusReader(corpus_dir, fileids='.+[.]txt')
    text = nltk.text.Text([token for sentence in word_lists.sents() for token in sentence if token not in punctuation])
    return text.collocations()


if __name__ == '__main__':
    corpus_root = 'C:\\_Work\\lightstar\\corp'
    # make_corp_file(corpus_root)
    # print(collocations(corpus_root))

    corp = PlaintextCorpusReader(corpus_root, fileids='.+[.]txt')
    print('PlaintextCorpusRead')
    text1 = nltk.text.Text([tok for s in corp.sents() for tok in s if tok not in punctuation])
    print('Text')
    print(text1.collocations())
    """print(wordlists.words('145.txt'))
    wl = []
    for fid in wordlists.fileids():
        for w in list(wordlists.words(fid)):
            if not (w in punctuation or w in []):
                wl.append(w)
    text1 = nltk.text.Text(wl)"""
    print(text1.similar("люстра"))
    print(text1.similar("торшер"))
    print(text1.similar("светильник"))
    print(text1.similar("потолочный"))
    print(text1.common_contexts(["люстра", "торшер"]))

    morph = pymorphy2.MorphAnalyzer()

    print('Text')

    text1 = nltk.text.Text([tok for s in corp.sents() for tok in s if tok not in punctuation])
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
    # lapprob = nltk.LaplaceProbDist(fdist1)
    # print(lapprob.prob('Полная'))
    # print(lapprob.prob('оффшор'))
    # print(fdist1.most_common())

    fdist1.plot(500, cumulative=False)
    # print(nltk.pos_tag(text1))
