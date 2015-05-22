# invdx.py
import math
import nltk
from string import punctuation
from nltk.corpus import stopwords
import pickle

import pymorphy2
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


def _read_idfs(file):
    idf = dict()
    for line in open(file, mode='r', encoding='utf-8'):
        p = line.split(';')
        if len(p) != 2:
            raise LookupError('%s is frog' % file)
        word = p[1][:-1]
        if word[0].isdigit():
            continue
        try:
            while not p[0][0].isdigit():
                p[0] = p[0][1:]
            freq = int(p[0])
        except:
            raise LookupError('%s is bad' % file)
        idf[word] = freq

    return nltk.ELEProbDist(nltk.FreqDist(idf))


class Entry:

    def __init__(self, position, zone, props=None):
        self.position = position
        self.zone = zone
        self.props = props

    def __str__(self):
        return '<Entry> (position = {}, zone = {}, properties = {})'.format(self.position, self.zone, self.props)


class PostingList:
    def __init__(self):
        self.posting_list = dict()

    def __getitem__(self, doc_id):
        return self.posting_list[doc_id]

    def add(self, doc_id, entry):
        if doc_id not in self.posting_list:
            self.posting_list[doc_id] = list()
        self.posting_list[doc_id].append(entry)

    def tf(self, doc_id, zone=None):
        if doc_id not in self.posting_list:
            return 0
        if zone:
            tf = 0
            for e in self.posting_list[doc_id]:
                if e.zone == zone:
                    tf += 1
            return tf
        else:
            return len(self.posting_list[doc_id])


class InvertedIndex:
    p_type = 'idf'

    def __init__(self, idfs='C:\\_Work\\SemanticCore\\bm_25\\NKRL.csv'):
        self.IDF = _read_idfs(idfs)
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

    def add(self, term, docid, e):
        if term not in self.index:
            self.index[term] = PostingList()
        self.index[term].add(docid, e)
        print(e)
        self.doc_lens.update(docid, e.zone)

    def save(self, name):
        with open('{}.pickle'.format(name), 'wb') as f:
            pickle.dump(self, f)

    def load(self, name):
        with open('{}.pickle'.format(name), 'rb') as f:
            return InvertedIndex(pickle.load(f))

    # frequency of word in document
    def get_document_frequency(self, word, docid):
        morth = pymorphy2.MorphAnalyzer()
        term = morth.parse(word)[0].normal_form
        if term in self.index:
            return self.index[word].tf(docid)
        else:
            return 0
            #raise LookupError('%s not in index' % str(word))

    #frequency of word in index, i.e. number of documents that contain word
    def get_index_frequency(self, word):
        return self.IDF.freq(word)

    def score(self, doc_id, query):

        return sum([self.w_single(doc_id, zone, _get_term_list(query)) for zone in ['body', 'title', 'h1']])

    def w_single(self, doc_id, zone, terms):
        res = 0
        k1 = 1
        k2 = 1 / 350
        for term in terms:
            tf = self.index[term].tf(doc_id, zone)
            l_p = self.log_p(term)
            res += l_p * tf / (tf + k1 + k2 * self.doc_length(doc_id, zone))
        return res

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


    def triple_weight(self, doc_id, zone, triple, p_type):
        for term in [triple[0], triple[2]]:
            if
        positions1 = self.d.index[triple[0]][doc_id]
        positions2 = self.index[triple[2]][zone]
        for p1 in positions1:
            if p1 + 1 in positions2:
                tf += 0.1
        s = log_p(triple[0], p_type) + log_p(triple[1], p_type)
        return s * tf / (1 + tf)

    def log_p(self, term, p_type='p_on_topic'):
        if p_type == 'p_on_topic':
            p = 1 - math.exp(-1.5 * self.IDF.prob(term))
        elif p_type == 'ICF':
            p = 1 / self.IDF.prob(term)
        else:
            raise Exception('Wrong p_type')
        return abs(math.log(p))

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
            raise LookupError('%s not found in table' % str(docid))

    def get_average_length(self):
        sum = 0
        for length in self.table.values():
            sum += length
        return float(sum) / float(len(self.table))


def all_entries(text, zone):
    entries = dict()
    morph = pymorphy2.MorphAnalyzer()
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
        #print(entries[term_nf])
    return entries


def build_idx(corpus_of_readable):
    idx = InvertedIndex()
    for docid, c in enumerate(corpus_of_readable):
        # build inverted index
        for zone in ['body', 'title', 'h1']:
            entries = all_entries(c.get_zone(zone), zone)
            for term in entries:
                for e in entries[term]:
                    idx.add(term, docid, e)
    return idx


class MyException(Exception):
    pass


def read_wp_html(u):
    try:
        with open('{}.txt'.format(hash(u)), mode='r',  encoding='utf8', errors='ignore') as f:
            return f.read()
    except:
        raise MyException()

def write_wp_html(u,html):
    try:
        with open('{}.txt'.format(hash(u)), mode='w', encoding='utf8', errors='ignore') as f:
            f.write(html)
    except Exception as e:
        print(html.encode('utf8', errors='ignore'))
        print('OOps'+str(e))


def make_corp_file():
    from text_analisys import Readable
    import search_query.ya_query as sps
    from search_query.content import WebPage
    readables = []
    for q in sps.queries_from_file('C:\\_Work\\vostok\\2.txt', 213)[1:]:
        print(q.query)
        for u in q.get_urls(10):
                try:
                    w = read_wp_html(u)
                except MyException:
                    try:
                        w = WebPage(u).html()
                    except:
                        print(u)
                        continue
                    write_wp_html(u, w)
                readables.append(Readable(w))
    return readables



if __name__ == '__main__':
    import pickle
    #try:
    with open('data.pickle', 'rb') as f:
            indx = InvertedIndex()
            indx = pickle.load(f)
    """except Exception as e:
        print(e)
        indx = build_idx(make_corp_file())
        data = indx
        with open('data.pickle', 'wb') as f:
            pickle.dump(data, f)"""

    for i in indx.doc_ids():

        print(indx.score(i, 'купить сапоги'))
