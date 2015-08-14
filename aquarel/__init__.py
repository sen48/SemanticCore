import re
import pymorphy2
import text_analisys
import search_query.serps as wrs
from search_query.content import WebPage


def colorize_text(fdist, text, morph):

    """
    Расскрашивает текст text, в соответствии с частотами слов fdist.
    Серым – будут выделены служебные части речи (предлоги, частицы, наречия) и стоп-слова.
    Для остальных слов градиентом от красного до зеленого будет обозначена частота.
    Зеленый – частотное слово, красный – нет.

    :param fdist:
    :ыек text:
    :param morph:
    :return:
    """

    text = text.replace('•', ' ').replace('\n', ' ')
    mc = fdist.most_common()[0]
    max_freq = mc[1]

    res = []
    for word in text.split(' '):
        if '-' in word:
            for w in word.split('-')[:-1]:
                res.append(color_word(w, fdist, morph, max_freq, sep='-'))
            res.append(color_word(word.split('-')[-1], fdist, morph, max_freq))
        else:
            res.append(color_word(word, fdist, morph, max_freq))
    return res


def colorable(par):

    """
    если рассматриваемое слово служебная часть речи (предлоги, частицы, наречия) иил стоп-слово возвтащает FALSE
    Инече TRUE
    :param par: par = morph.parse(tok)[0]
    :return:
    """
    return par.tag.is_productive() and par.normal_form not in text_analisys.stop_words and par.tag.POS != 'ADVB'


def color_word(word, fdist, morph, max_freq, sep=' '):
    tok = re.sub(r'[^\w\s]+|[\d]+|•', r' ', word).strip().lower()

    par = morph.parse(tok)[0]
    tok = par.normal_form
    if len(tok) == 0:
        return ['', "#010101", ' ', sep]
    elif tok not in fdist.keys():
        if not colorable(par):
            color = "#010101"
            title = "{} {}".format(tok, par.tag.POS)
        else:
            color = "#ff0000"
            title = "{} {}".format(tok, par.tag.POS)
    else:
        green = round(100+155*fdist.get(tok) / max_freq)
        color = "#{0:02x}{1:02x}{2}".format(255-green, green, '00',)
        title = "{:003.4f} {} {}".format(fdist.get(tok), tok,  par.tag.POS)
    return [word, color, title, sep]



def colorize(queries, analyzed_text='', region=213, full=False):
    """
    Серым – будут выделены служебные части речи (предлоги, частицы, наречия).
    Для остальных слов градиентом от красного до зеленого будет обозначена степень соответствия запросу.
    Зеленый – соответствует, красный – нет.
    Если навести мышку на слово, во всплывающей подсказке будут показаны часть речи слова и
    численное значение «релевантности».

    :param queries: Список запросов
    :param analyzed_text: Текст, который нужно раскрасить
    :param region: регион запроса
    :param full:  Если TRUE то при анализе учитывются сквозные блоки страниц ТОП10
    :return: color_text - текст с html разметкой,
             most_comm - список наиболее популярных словоформ,
             coll - список наиболее попуолярных словосочетаний
             weird - список наиболее характерных словоформ
    """

    collection = []
    for query in queries:
        collection += get_texts(query, region, full)

    fdist, doc_dist, morph = text_analisys.words_freq(collection, normalize=True)

    color_text = color_text (doc_dist, analyzed_text, morph)
    most_comm = [str(word[0]) for word in fdist.most_common(20) if str(word[0]) != '']

    coll = ', '.join(text_analisys.collocations(collection))
    weirdness = {}
    for k in fdist.keys():
        if k == '':
            continue
        lang_freq = text_analisys.CF[k] if k in text_analisys.CF.keys() else 2
        weirdness[k] = fdist.get(k)/lang_freq
    weird = sorted(weirdness.keys(), key=lambda k: weirdness[k], reverse=True)[:50]

    return color_text, most_comm, coll, weird



def get_texts(query, region, full, ):
    serp_items = wrs.read_serp(query, region, 10)
    pgs = []
    for item in serp_items:
        print(item.url)
        try:
            pgs.append(WebPage(item.url))
        except Exception as e:
            print(e)
            continue

    return [text_analisys.Readable(p.html()).title() + text_analisys.Readable(p.html()).text() for p in pgs]
