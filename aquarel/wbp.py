from functools import reduce
from django.utils.html import escape
import search_query.serps as wrs
from search_query.content import WebPage
import text_analisys

def makebold(fn):
    def wrapped():
        return '''<!DOCTYPE html PUBLIC>
                <html>
                    <head>
                        <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
                    </head>
                    <body>
                        {}
                    </body>
                </html>'''.format(fn())
    return wrapped



def run(queries, analyzed_text='', region=213, full=False):
    collection = []
    for query in queries:
        collection += get_texts(query, region, full)
    normalize = True
    fdist, doc_dist = text_analisys.words_freq(collection, normalize)


    result = '<div>' + text_analisys.color_text(doc_dist, analyzed_text, normalize) + '</div>'
    most_comm = reduce(lambda x, y: x + y, [str(word[0]) + ', ' for word in fdist.most_common(20)], '')
    result += '<h2>Наиболее популярные слова:</h2><p>' + \
              most_comm + '</p>'
    coll = reduce(lambda x, y: x + y + ', ', text_analisys.collocations(collection), '')
    result += '<h2>Наиболее популярные коллокации:</h2><p>' + \
              coll + '</p>'
    weirdness = {}
    for k in fdist.keys():
        lang_freq = text_analisys.CF[k] if k in text_analisys.CF.keys() else 2
        weirdness[k] = fdist.get(k)/lang_freq
    weird = reduce(lambda x, y: x + y + ', ', sorted(weirdness.keys(), key=lambda k: weirdness[k], reverse=True)[:50], '')
    result += '<h2>Наиболее тематичные слова:</h2><p>' + \
              weird + '</p>'
    return result


def colorize(queries, analyzed_text='', region=213, full=False):
    collection = []
    for query in queries:
        collection += get_texts(query, region, full)
    normalize = True
    fdist, doc_dist, parsed, morph = text_analisys.words_freq(collection, normalize)


    color_text = text_analisys.color_words(doc_dist, analyzed_text, morph, normalize, parsed)
    most_comm = ', '.join([str(word[0]) for word in fdist.most_common(20) if str(word[0]) != ''])

    coll = ', '.join(text_analisys.collocations(collection))
    weirdness = {}
    for k in fdist.keys():
        if k == '':
            continue
        lang_freq = text_analisys.CF[k] if k in text_analisys.CF.keys() else 2
        weirdness[k] = fdist.get(k)/lang_freq
    weird = ', '.join(sorted(weirdness.keys(), key=lambda k: weirdness[k], reverse=True)[:50])

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


TEXT = '''В интернет-магазине «Троицкая книга» Вы можете в считанные минуты купить облачения и одежду. Церковные ткани,
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


# , 'одежда православных священнослужителей', 'одежда православного священника',

@makebold
def make_html():
    return run(['адронный коллайдер'], analyzed_text=TEXT, region=213, full=False)

'''html = make_html()
with open('C:\\Users\\lvova\\Desktop\\1.html', mode='w', encoding='utf-8') as fout:
    fout.write(html)'''


import tornado.ioloop
import tornado.web
QUERIES = ["адронный коллайдер", "бозон Хиггса"]

class MainHandler(tornado.web.RequestHandler):

    def get(self):
        self.render("start_template.html", queries='\n'.join(QUERIES), text=TEXT)


    def post(self):
        #self.set_header("Content-Type", "text/plain")
        txt = self.get_argument("message")
        q = escape(self.get_argument("queries"))
        queries = [query for query in q.split('\n') if len(query) > 3]
        color_text, most_comm, coll, weird = colorize(queries, txt)
        self.render("result_template.html", color_text=color_text,
               freq_words=most_comm,
               freq_collocations=coll,
               weird_words=weird)


application = tornado.web.Application([
    (r"/", MainHandler),

])

if __name__ == "__main__":
    application.listen(8888)
    tornado.ioloop.IOLoop.current().start()



'''одежда священнослужителей',
'одежда православных священнослужителей',
'одежда православного священника',
'одежда для священнослужителей'''
