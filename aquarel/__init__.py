"""
Модуль для раскраски текста в соответствии с "тематичностью" слов.
"""

import re
import pymorphy2

import text_analysis
import search_engine_tk.serp as wrs
from content import WebPage


def _get_par(word, morph):
    tok = re.sub(r'[^\w\s]+|[\d]+|•', r' ', word).strip().lower()
    return morph.parse(tok)[0]


class SpecialWord:
    """
    sep: разделитель - это символ между текущим словом и следующим
    color: цвет - это строка начинающаяся с # а дальше 16-тиричный код цвета
    title:
    """

    def __init__(self, word, morph, fdist, max_freq, sep=' '):
        par = _get_par(word, morph)
        color, title = color_word(par, fdist, max_freq)
        self.word = word
        self.color = color
        self.title = title
        self.sep = sep


def colorize_text(fdist, text):
    """
    :param fdist:
    :param text:
    :return:  текст с html разметкой

    """

    morph = pymorphy2.MorphAnalyzer()
    text = text.replace('•', ' ').replace('\n', ' ')
    mc = fdist.most_common()[0]
    max_freq = mc[1]

    res = []
    for word in text.split(' '):
        if '-' in word:
            for w in word.split('-')[:-1]:
                res.append(SpecialWord(w, morph, fdist, max_freq, '-'))
            res.append(SpecialWord(word.split('-')[-1], morph, fdist, max_freq))
        else:
            res.append(SpecialWord(word, morph, fdist, max_freq))
    return make_html(res)


def colorable(par):
    """
    если рассматриваемое слово служебная часть речи (предлоги, частицы, наречия) иил стоп-слово возвтащает FALSE
    Инече TRUE
    :param par: par = morph.parse(tok)[0]
    :return:
    """
    return par.tag.is_productive() and par.normal_form not in text_analysis.stop_words and par.tag.POS != 'ADVB'


def color_word(par, fdist, max_freq):
    """

    :param par:
    :param fdist:
    :param max_freq:
    :return: color: строка начинающаяся с # а дальше 16-тиричный код цвета
             title: строка состоящая из частоты слова(опционально, если слово раскрашивается в оттенок зеленого),
                    нормальной формы и части речи
    """

    tok = par.normal_form
    if len(tok) == 0:
        return "#010101", ' '
    elif tok not in fdist.keys():
        if not colorable(par):
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


def make_html(specialwords):
    html = ''
    for sw in specialwords:
        html += '<font color = {} title = {}>{}</font>{}'.format(sw.color, sw.title, sw.word, sw.sep)
    return html


def colorize(queries, analyzed_text='', region=213):
    """
    Серым – будут выделены служебные части речи (предлоги, частицы, наречия).
    Для остальных слов градиентом от красного до зеленого будет обозначена степень соответствия запросу.
    Зеленый – соответствует, красный – нет.
    Если навести мышку на слово, во всплывающей подсказке будут показаны часть речи слова и
    численное значение «релевантности».

    :param queries: Список запросов
    :param analyzed_text: Текст, который нужно раскрасить
    :param region: регион запроса
    :return: color_text - текст с html разметкой,
             most_comm - список наиболее популярных словоформ,
             coll - список наиболее попуолярных словосочетаний
             weird - список наиболее характерных словоформ
    """

    collection = []
    for query in queries:
        collection += get_texts(query, region)

    freq_dist, doc_dist = text_analysis.words_freq(collection, normalize=True)

    color_text = colorize_text(doc_dist, analyzed_text)
    most_comm = [str(word[0]) for word in freq_dist.most_common(20) if str(word[0]) != '']

    coll = ', '.join(text_analysis.collocations(collection))
    weirdness = {}
    for w in freq_dist.keys():
        if w == '':
            continue
        lang_freq = text_analysis.CF.prob(w) #[w] if w in text_analysis.CF.keys() else 2
        weirdness[w] = freq_dist.get(w) / lang_freq
    weird = sorted(weirdness.keys(), key=lambda k: weirdness[k], reverse=True)[:50]

    return color_text, most_comm, coll, weird


def get_texts(query, region):
    """
    Извлекает тексты ТОП10 по заданному запросу
    :param query: запрос
    :param region: регион
    :return: список текстов.
    """

    serp_items = wrs.read_serp(query, region, 10)
    pgs = []
    for item in serp_items:
        try:
            pgs.append(WebPage(item.url))
        except Exception as e:
            print(e)
            continue
    return [text_analysis.Readable(p.html()).title() + text_analysis.Readable(p.html()).text() for p in pgs]


if __name__ == "__main__":
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

    txt, comm, col, wierd = colorize(QUERIES, analyzed_text=TEXT, region=213)
    print(txt)
    print(comm)
    print(col)
    print(wierd)

