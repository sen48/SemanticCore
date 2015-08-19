"""
Модуль для раскраски текста в соответствии с "тематичностью" слов.
"""

import re
import pymorphy2

import text_analysis


def _get_par(word, morph):
    tok = re.sub(r'[^\w\s]+|[\d]+|•', r' ', word).strip().lower()
    return morph.parse(tok)[0]


def colorize_text(fdist, text):
    """
    Расскрашивает текст по средствам html разметки.
    Черными останутся служебные части речи (предлоги, частицы, наречия).
    Для остальных слов градиентом от красного до зеленого будет обозначена степень соответствия запросу.
    Зеленый – соответствует, красный – нет.
    Если навести мышку на слово, во всплывающей подсказке будут показаны часть речи и
    численное значение «релевантности».
    :param fdist: nltk.FreqDist - частоты слов
    :param text: srt, текст для раскраски
    :return: str,  текст с html разметкой
    """

    morph = pymorphy2.MorphAnalyzer()
    text = text.replace('•', ' ').replace('\n', ' ')

    res = []
    for word in text.split(' '):
        if '-' in word:
            for w in word.split('-')[:-1]:
                res.append(SpecialWord(w, morph, fdist, '-'))
            res.append(SpecialWord(word.split('-')[-1], morph, fdist))
        else:
            res.append(SpecialWord(word, morph, fdist))
    return make_html(res)


def is_black(par):
    """
    Если рассматриваемое слово не нужно раскрашивать, то есть оно является служебной частью речи
    или стоп-словом, возвтащает TRUE. Иначе  FALSE
    :param par: pymorphy2.analyzer.Parse
    :return: bool
    """
    return not par.tag.is_productive() or par.normal_form in text_analysis.stop_words or par.tag.POS != 'ADVB'


def form_color_title(par, fdist):
    """
    Формирует:
        1)color - строку, означающую цвет, в который нужно раскрасить слово
        2)title - строку, которая содержит частоту, нормальную форму и части речи.
    :param par:  pymorphy2.analyzer.Parse
    :param fdist: nltk.FreqDist - частоты слов
    :return: color: str, строка начинающаяся с # а дальше 16-тиричный код цвета
             title: str, строка состоящая из частоты слова(опционально, если слово раскрашивается в оттенок зеленого),
                    нормальной формы и части речи
    """
    max_freq = fdist.most_common(1)[0][1]
    tok = par.normal_form
    if len(tok) == 0:
        return "#010101", ' '
    elif tok not in fdist.keys():
        if is_black(par):
            color = "#010101"
            title = "{} {}".format(tok, par.tag.POS)
        else:
            color = "#ff0000"
            title = "{} {}".format(tok, par.tag.POS)
    else:
        green = round(100 + 155 * fdist.get(tok) / max_freq)
        color = "#{0:02x}{1:02x}{2}".format(255 - green, green, '00', )
        title = "{:003.4f} {} {}".format(fdist.get(tok), tok, par.tag.POS)
    return color, title


def make_html(special_words):
    """
    Формирует текст с html разметкой из списка special_words
    :param special_words: list of SpecialWord
    :return: str
    """
    html = ''
    for sw in special_words:
        html += '<font color = {} title = {}>{}</font>{}'.format(sw.color, sw.title, sw.word, sw.sep)
    return html


class SpecialWord:
    """
    Служебный класс для расскрашевания слов

    sep: разделитель - это символ между текущим словом и следующим
    color: цвет - это строка начинающаяся с # а дальше 16-тиричный код цвета
    title: строка состоящая из частоты слова(опционально, если слово раскрашивается в оттенок зеленого),
           нормальной формы и части речи
    """

    def __init__(self, word, morph, fdist, sep=' '):
        """
        :param word: str, само слово
        :param morph: pymorphy2.MorphAnalyzer
        :param fdist: nltk.FreqDist - частоты слов коллекции
        :param sep: str, символ, который нежно поставить после слова
        """
        par = _get_par(word, morph)
        color, title = form_color_title(par, fdist)
        self.word = word
        self.color = color
        self.title = title
        self.sep = sep



if __name__ == "__main__":
    def get_texts(query, region):
        """
        Извлекает тексты ТОП10 по заданному запросу
        :param query: запрос
        :param region: регион
        :return: список текстов.
        """
        import search_engine_tk.serp as wrs
        from web_page_content import WebPageContent

        serp_items = wrs.read_serp(query, region, 10)
        pgs = []
        for item in serp_items:
            try:
                pgs.append(WebPageContent(item.url))
            except Exception as e:
                print(e)
                continue
        return [text_analysis.Readable(p.html()).title() + text_analysis.Readable(p.html()).text() for p in pgs]

    def main(queries, analyzed_text='', region=213):
        """
        Возвращяет раскришенный текст, популярные слова, популярные словосочетания,

        :param queries: lift of str, Список запросов
        :param analyzed_text: str, Текст, который нужно раскрасить
        :param region: int, регион запроса
        :return: color_text - str, текст с html разметкой,
                 most_comm - lift of str, список наиболее популярных терминов,
                 coll - lift of str, список наиболее попуолярных словосочетаний
                 weird - lift of str, список наиболее характерных терминов
        """

        collection = []
        for query in queries:
            collection += get_texts(query, region)

        freq_dist, doc_dist = text_analysis.words_freq(collection, normalize=True)

        color_text = colorize_text(doc_dist, analyzed_text)
        most_comm = [str(word[0]) for word in freq_dist.most_common(20) if str(word[0]) != '']
        coll = text_analysis.collocations(collection)
        weird_words = text_analysis.weird(doc_dist)

        return color_text, most_comm, coll, weird_words

    TEXT = '''В интернет-магазине «Троицкая книга» Вы можете в считанные минуты купить облачения и одежду.
     Церковные ткани,
    используемые для пошива изделий, представлены в соответствующем разделе.
    Наш интернет-магазин располагает широким выбором готовых богослужебных облачений (иерейские и диаконские облачения,
     митры, камилавки, требные комплекты, стихари), а также повседневной священнической и монашеской одежды
     (рясы, подрясники).
     Кроме того, у нас Вы найдете безрукавки, брюки и рубашки.
    Если нужного Вам размера не оказалось в наличии или у вас возникли трудности с подбором необходимых
    тканей и размера,
    сразу же звоните нам – мы с радостью Вас проконсультируем и поможем подобрать нужный товар.
    А если Вы живете в Москве,
    то лучше приходите в наш магазин. Лично ознакомьтесь с ассортиментом, примерьте понравившиеся изделия и сделайте
     конечный выбор.
    Мы также предлагаем возможность заказать индивидуальный пошив изделия для православных священников.
    Облачения и одежда
    шьются в нашей мастерской Подворья в Москве в течение 3-х недель, доставляются даже в самые удаленные уголки
    России и мира.
    Облачения и одежда православного священника

    Мы производим священнические облачения и одежду для священнослужителей и монахов в
    собственной мастерской уже 15 лет.
    В нашем ассортименте представлены ткани на любой вкус и кошелек. Вы можете выбрать православные
     облачения как из традиционных тканей
    (шелк, парча русская, парча греческая), так и из более редких.
     Для пошива изделий мы используем хорошие лекала, поэтому,
    если Вы правильно указали свои параметры, можете быть уверены, что купленная вещь отлично сядет на Вас.'''

    QUERIES = ["адронный коллайдер", "бозон Хиггса"]

    txt, comm, col, wierd = main(QUERIES, analyzed_text=TEXT, region=213)
    print(txt)
    print(comm)
    print(col)
    print(wierd)
