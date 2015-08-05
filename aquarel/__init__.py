from functools import reduce
import grab
import pymorphy2
import search_query.serps as wrs
from search_query.content import WebPage
import text_analisys


def run(queries, text='', region=213, full=False):
    collection = []
    for query in queries:
        collection += get_texts(query, region, full)
    normalize = True
    fdist, doc_dist = text_analisys.words_freq(collection, normalize)
    result = '<p>' + text_analisys.color_text(doc_dist, text, normalize) + '</p><br><br><br>'
    most_comm = reduce(lambda x, y: x + y, [str(word[0]) + ', ' for word in fdist.most_common(20)], '')
    result += '<p>Наиболее популярные слова:<br>         ' + \
              most_comm + '</p><br><br><br>'
    coll = reduce(lambda x, y: x + y + ', ', text_analisys.collocations(collection), '')
    result += '<p>Наиболее популярные коллокации:<br>         ' + \
              coll + '</p><br><br><br>'
    weirdness = {}
    for k in fdist.keys():
        lang_freq = text_analisys.CF[k] if k in text_analisys.CF.keys() else 2
        weirdness[k] = fdist.get(k)/lang_freq
    weird = reduce(lambda x, y: x + y + ', ', sorted(weirdness.keys(), key=lambda k: weirdness[k], reverse=True)[:50], '')
    result += '<p>Наиболее тематичные слова:<br>         ' + \
              weird + '</p><br><br><br>'
    return result


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


text = '''В интернет-магазине «Троицкая книга» Вы можете в считанные минуты купить облачения и одежду. Церковные ткани,
используемые для пошива изделий, представлены в соответствующем разделе.
Наш интернет-магазин располагает широким выбором готовых богослужебных облачений (иерейские и диаконские облачения,
 митры, камилавки, требные комплекты, стихари), а также повседневной священнической и монашеской одежды (рясы, подрясники).
 Кроме того, у нас Вы найдете безрукавки, брюки и рубашки.
Если нужного Вам размера не оказалось в наличии или у вас возникли трудности с подбором необходимых тканей и размера,
сразу же звоните нам – мы с радостью Вас проконсультируем и поможем подобрать нужный товар. А если Вы живете в Москве,
то лучше приходите в наш магазин. Лично ознакомьтесь с ассортиментом, примерьте понравившиеся изделия и сделайте конечный выбор.
Мы также предлагаем возможность заказать индивидуальный пошив изделия для православных священников. Облачения и одежда
шьются в нашей мастерской Подворья в Москве в течение 3-х недель, доставляются даже в самые удаленные уголки России и мира.
Облачения и одежда православного священника

Мы производим священнические облачения и одежду для священнослужителей и монахов в собственной мастерской уже 15 лет.
В нашем ассортименте представлены ткани на любой вкус и кошелек. Вы можете выбрать православные облачения как из традиционных тканей
(шелк, парча русская, парча греческая), так и из более редких. Для пошива изделий мы используем хорошие лекала, поэтому,
если Вы правильно указали свои параметры, можете быть уверены, что купленная вещь отлично сядет на Вас.'''
html = run(['одежда священнослужителей',
            'одежда православных священнослужителей',
            'одежда православного священника'], text=text, region=213, full=False)
# , 'одежда православных священнослужителей', 'одежда православного священника',

with open('C:\\Users\\lvova\\Desktop\\1.html', mode='w') as fout:
    fout.write(html)


'''одежда священнослужителей',
'одежда православных священнослужителей',
'одежда православного священника',
'одежда для священнослужителей'''
