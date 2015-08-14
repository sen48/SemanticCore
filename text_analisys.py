import os
import pickle
import re
import collections
import nltk
import nltk.text
import pymorphy2
from read.htmls import norm_title
import read
from string import punctuation
from nltk.corpus import stopwords
from nltk.compat import Counter
import bs4


stop_words = stopwords.words('russian')


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
        for e in list(doc.iterfind('.//' + tag)):
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
        make_plain_text_files_urls(q.get_urls(10), path)


def make_plain_text_files_urls(urls, path):
    from search_query.content import WebPage

    for u in urls:
        try:
            r = Readable(WebPage(u).html())
            with open(os.path.join(path, str(hash(u)) + '.txt'), mode='w', encoding='utf8') as f_out:
                f_out.write(r.title() + '/n' + r.text())
        except:
            print(u)


def load_ruscorpra_frqs():
    try:
        with open('C:\\_Work\\SemanticCore\\CF.pickled', mode='rb') as pickled:
            cf = pickle.load(pickled)
    except FileNotFoundError :
        print('start')
        morph = pymorphy2.MorphAnalyzer()
        cf = collections.defaultdict(lambda: 0)
        with open('C:\\_Work\\SemanticCore\\1grams-3.txt', mode='r', encoding='utf-8') as grams:
            for line in grams.readlines():
                fr_word = line.split('\t')
                term = morph.parse(fr_word[1][:-1])[0].normal_form
                cf[term] += int(fr_word[0])
        print('done')
        #with open('C:\\_Work\\SemanticCore\\CF.pickled', mode='w') as fout:
        try:
            pf = dict(cf)
            print(pf)
            pickle.dump(pf, open('C:\\_Work\\SemanticCore\\CF.pickled', mode='wb'))
        except pickle.PicklingError as ex:
            print(ex)
    return cf

CF = load_ruscorpra_frqs()


def collocations(sents):
    text = nltk.text.Text([token for sentence in sents for token in nltk.word_tokenize(sentence) if token not in punctuation])
    return text.find_collocations(num=30)


def words_freq(sents, normalize):
    morph = pymorphy2.MorphAnalyzer()
    words = []
    docdist = []
    for s in sents:
        toks = set()
        for tok in nltk.word_tokenize(s):
            tok = re.sub(r'[^\w\s]+|[\d]+', r'', tok).strip()
            tok = tok.lower()
            par = morph.parse(tok)[0]
            if par.normal_form not in stop_words:
                if par.normal_form not in toks:
                    docdist.append(par.normal_form)
                    toks.add(par.normal_form)
                if normalize:
                    words.append(par.normal_form)
                else:
                    words.append(tok)

    text1 = Counter(words)
    text2 = Counter(docdist)

    return nltk.FreqDist(text1), nltk.FreqDist(text2)


if __name__ == '__main__':
    pass

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
