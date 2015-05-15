import nltk
import pymorphy2
from string import punctuation
from nltk.corpus import stopwords
from bm_25 import rank

punctuation += "«—»"  # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
stop_words = stopwords.words('russian')
stop_words.extend(['это', 'дата', 'смочь', 'хороший', 'нужный',
                   'перед', 'весь', 'хотеть', 'цель', 'сказать', 'ради', 'самый', 'согласно',
                   'около', 'быстрый', 'накануне', 'неужели', 'понимать', 'ввиду', 'против',
                   'близ', 'поперёк', 'никто', 'понять', 'вопреки', 'твой', 'объектный',
                   'вместо', 'идеальный', 'целевой', 'сила', 'благодаря', 'знаешь',
                   'вследствие', 'знать', 'прийти', 'вдоль', 'вокруг', 'мочь', 'предлагать',
                   'наш', 'всей', 'однако', 'очевидно', "намного", "один", "по-прежнему",
                   "суть", "очень", "год", "который", 'usd'])


__author__ = 'lvova'

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
    return idf


class TextTable:
    def __init__(self, text):
        self.table = dict()
        #self.words = dict()
        morph = pymorphy2.MorphAnalyzer()
        pos = 0
        tokens = nltk.word_tokenize(text)
        self.length = len(tokens)
        for word in tokens:
            term = morph.parse(word)[0].normal_form
            if term in stop_words:
                continue
            if term in self.table:
                self.table[term] += 1
            else:
                self.table[term] = 1
            '''if word in self.words:
                self.words[word].append(pos)
            else:
                self.words[word] = [pos]'''
            pos += 1

    def get_term_frequency(self, term):
        return self.table[term] if term in self.table else 0


class ScoreCounter:

    def __init__(self, dfs):
        self.DF = _read_idfs(dfs)

    def get_df(self,word):
        if word in self.DF:
            return self.DF[word]
        return 1

    def count(self, text_table, query_table):
        res = 0
        for word in query_table.table:
            tf = text_table.get_term_frequency(word)
            res += rank.score_bm25(1/self.get_df(word), tf, 10, text_table.length, len(self.DF))
        return res
    w_s = w_single(document, zone, terms, p_type)
    w_p = w_pair(document, zone, terms, p_type)
    w_a = w_all_words(document, zone, terms, p_type)
    w_ph = w_phrase(document, zone, query, p_type)
    w_h = w_half_phrase(document, zone, terms, idfs, p_type)

if __name__ == '__main__':
    sc = ScoreCounter('C:\\_Work\\SemanticCore\\bm_25\\NKRL.csv')
    text = TextTable('ываы')
    " Есть и другой подход: попытаться повлиять на поведенческие факторы за счет привлечения аудитории другими способами. Раньше для этого просто использовались «накрутки». Как мы помним, первые системы генерировали ботов, которые совершали необходимые действия в поисковой выдаче и на продвигаемых сайтах. Но поисковики начали внедрение систем, способных отсеивать подобное продвижение и делать его бесполезным, неэффективным, и даже вредным.\n"
    "\n"
    "В качестве ответа, оптимизаторы перешли от роботизированных «накруток» к раздаче заданий всем пользователям подряд, что до сих пор неплохо работает для продвижения, но с учетом разумного применения технологии. \n"
    "\n"
    "В результате, сейчас рынок поведенческих факторов стал динамично развиваться, а поведенческие сервисы стали отдаленно напоминать старые добрые ссылочные биржи. Сервис SERPClick.ru о котором я уже упоминал ранее, как раз позволяет эффективно и, на мой взгляд, гораздо более выгодно влиять на поведенческие факторы для ранжирования сайта в выдаче.\n"

    query = TextTable('ываы')
    print(sc.count(text, query))
    query = TextTable('фактор')
    print(sc.count(text, query))