"""
Предназначен для анализа текстов web-страниц.
Реализованна возможность выделять
  - наиболее популярные слова коллекции,
  - наиболее частотные словосочетания
  - наиболее характерных слова
"""

import pickle
import re
import collections
from string import punctuation
import nltk
import nltk.text
import pymorphy2
from nltk.corpus import stopwords
from nltk.compat import Counter
import bs4
from read.htmls import norm_title
import read


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

        Returns
        -------
        : str
        """
        soup = bs4.BeautifulSoup(self.summary())
        text = soup.get_text()
        return re.compile(r'[\n\r][\n\r \t\a]+').sub('\n', text)

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
        Returns
        -------
        : str, concatenation of h1 texts
        """
        return ' '.join(self._get_tag('h1'))

    def get_all_h2(self):
        return self._get_tag('h2')

    def get_zone_text(self, zone):
        """
        Parameters
        ----------
        zone: str,
            зона документа, напимер: 'body', 'title', 'h1'

        Returns
        -------
        : str
        """
        if zone == 'body':
            return self.text()
        elif zone == 'title':
            return self.title()
        elif zone == 'h1':
            return self.h1()
        else:
            return ' '.join(self._get_tag(zone))


def get_alts(response):
    alts = [a.text() for a in list(response.select('//img/@alt')) if len(a.text()) > 0]
    return alts


def load_ruscorpra_probabilities():
    """
    Возвращает вероятности (относительные частоты) слов в русском языке из НКРЯ в виде nltk.ELEProbDist
    (это для сглаживания, т.е чтобы не было нулей),
    которые находятся в файле 1grams.txt
    Чтобы каждый раз не разбирать, маринуется в CF.pickled


    Returns
    -------
    : nltk.ELEProbDist
    """
    try:
        with open('C:\\_Work\\SemanticCore\\text_analysis\\CF.pickled', mode='rb') as pickled:
            cf = pickle.load(pickled)
    except FileNotFoundError:
        print('start')
        morph = pymorphy2.MorphAnalyzer()
        cf = collections.defaultdict(lambda: 0)
        with open('C:\\_Work\\SemanticCore\\text_analysis\\1grams.txt', mode='r', encoding='utf-8') as grams:
            for line in grams.readlines():
                fr_word = line.split('\t')
                term = morph.parse(fr_word[1][:-1])[0].normal_form
                cf[term] += int(fr_word[0])
        print('done')
        try:
            pf = dict(cf)
            print(pf)
            pickle.dump(pf, open('C:\\_Work\\SemanticCore\\CF.pickled', mode='wb'))
        except pickle.PicklingError as ex:
            print(ex)
    return nltk.ELEProbDist(nltk.FreqDist(cf))


def weird(freq_dist, num=50):
    """
    Возвращяет список, состоящий из num наиболее характерных терминов коллекции. Термин считается характерным,
    если в коллекции он встречается часто, а в русском языке - нет. Степень характерности (weirdness) вычисляется как
    отношение частоты в коллекции к частоте в русском языке.
    Parameters
    ----------
    freq_dist: nltk.FreqDist - частоты слов в коллекции
    num: int, число результатов
    Returns
    -------
    : list of str
    """
    weirdness = {}
    for w in freq_dist.keys():
        if w == '':
            continue
        lang_prob_dist = load_ruscorpra_probabilities()
        lang_freq = lang_prob_dist.prob(w)
        weirdness[w] = freq_dist.get(w) / lang_freq
    return sorted(weirdness.keys(), key=lambda k: weirdness[k], reverse=True)[:num]


def collocations(sents, num=30, window_size=2):
    """
    Ищет колокации в тексте
    Parameters
    ----------
    sents: list of str,
        набор текстов (коллекция)
    num: int,
        сколько нужно найти
    window_size:
       int, длина коллокации
    Returns
    -------
    : list of str
    """
    text = nltk.text.Text([token for sentence in sents for token in nltk.word_tokenize(sentence)
                           if token not in punctuation])

    return text.find_collocations(num, window_size)


def words_freq(sents, normalize):
    """
    Считает частоты слов
    Parameters
    ----------
    sents:  list of str,
        набор текстов (коллекция)
    normalize: bool,
        если ИСТИНА, то слова приводятся к нормальной форме
    Returns
    -------
    : nltk.FreqDist
        общее число вхождений слов во все строки.
    : nltk.FreqDist
        количество документов, в которые вошло слво
    """
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
