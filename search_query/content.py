import re
import statistics
from zlib import compress
from sys import getsizeof

from grab import Grab
import urllib
import bs4
import bs4.element
import grab
import read
import bm_25.invdx as bm
import text_analisys
import pymorphy2
import nltk
from string import punctuation
from nltk.corpus import stopwords
stop_words = stopwords.words('russian')
stop_words.extend(['это', 'дата', 'смочь', 'хороший', 'нужный',
                   'перед', 'весь', 'хотеть', 'цель', 'сказать', 'ради', 'самый', 'согласно',
                   'около', 'быстрый', 'накануне', 'неужели', 'понимать', 'ввиду', 'против',
                   'близ', 'поперёк', 'никто', 'понять', 'вопреки', 'твой', 'объектный',
                   'вместо', 'идеальный', 'целевой', 'сила', 'благодаря', 'знаешь',
                   'вследствие', 'знать', 'прийти', 'вдоль', 'вокруг', 'мочь', 'предлагать',
                   'наш', 'всей', 'однако', 'очевидно', "намного", "один", "по-прежнему",
                   'суть', 'очень', 'год', 'который', 'usd'])
morth = pymorphy2.MorphAnalyzer()

def _get_attr(t, attribute):
    if not t:
        return []
    if isinstance(t, bs4.element.Tag):
        try:
            c = t['class']
            return c
        except KeyError:
            return []
    c = t.attr(attribute, default='')
    if c != '':
        return c.split(' ')
    else:
        return []


def _acceptable_phone(tel):
    minus_numbers = ['1234567', '99999999', '987654']
    phone_lengths = [12, 10, 11, 7, 6, 5]
    return len(tel) in phone_lengths and all([m not in tel for m in minus_numbers])


def _clear_phone(tel):
    tel = re.sub(pattern=r'\D', repl='', string=tel)
    if len(tel) == 11 and (tel.startswith('7') or tel.startswith('8')):
        tel = tel[1:]
    if len(tel) == 12 and tel.startswith('44'):
        tel = tel[2:]
    return tel



class WebPage:
    PRICE_PATTERN = re.compile(r'''[\d]{1,3}(?:(?:[\s]|[&]nbsp[;]|[,])*[\d]{3})*'''
                               r'''(?:[.,](?:[\d][\d]|[-–]))?(?:[\s]|[&]nbsp[;])?'''
                               r'''(?:руб[.]?|[$]|USD|EUR|RUB|р[.])?''')

    def __init__(self, url):
        self.url = url
        self._g = Grab()
        self._g.setup()
        self._g.go(quote(url))
        self.doc = self._g.doc

    def getTic(self):
        """
        Возвращает тИЦ страницы. Использовать с осторожностью, могут забанить ip
        """
        import urllib
        yurl = 'http://bar-navig.yandex.ru/u?ver=2&show=32&url=%s' % self.url

        f = urllib.urlopen(yurl)
        st = f.read()
        import re
        m = re.search(r'value="([0-9]{1,5})"', st)

        try:
            tic = m.group(1)
        except:
            tic = 0
        return tic

    def is_in_Yaca(self):
        """
        Проверка наличия сайта в яндекс каталоге. Использовать с осторожностью, могут забанить ip
        """
        import urllib
        yurl = 'http://bar-navig.yandex.ru/u?ver=2&show=32&url=%s' % self.url

        f = urllib.urlopen(yurl)
        st = f.read()
        import re
        m = re.search(r'<textinfo>(?P<author>[\W\w]+)</textinfo>', st)

        try:
            yaca = m.group(1)
            print(yaca)
        except:
            yaca = ''

        if len(yaca) > 2:
            return True
        else:
            return False

    def get_phones(self):
        """
        Возвращает список номеров телефонов, встречающихся на странице.
        """
        phone_pattern = re.compile(r'(?:^|[>"\s])(?:т[.]\D*|тел[.]\D*|телефон\D*)'
                                   r'(?:'
                                   r'(?:\d{2,3})[-– ]?(?:\d{2})[-– ]?(?:\d{2})|'
                                   r'(?:\d{3})[-– ](?:\d)[-– ](?:\d{3})|'
                                   r'(?:\d{3})[-– ](?:\d{3})(?:[-– ](?:\d))?'
                                   r')(?:$|[<"\s])|'
                                   r'(?:[+]7|[+]44|[ \^][8]?)(?:'
                                   r'[-– ]?(?:\d{3}|[(]\d{3}[)])'
                                   r'[-– ]?\d{3}[-– ]?\d[-– ]?\d[-– ]?\d[-– ]?\d|'
                                   r'[-– ]?(?:\d{4}|[(]\d{4}[)])'
                                   r'[-– ]?\d{2}[-– ]?\d[-– ]?\d[-– ]?\d[-– ]?\d)')

        numbers = [_clear_phone(ph) for ph in phone_pattern.findall(self.doc.unicode_body())]
        return [tel for tel in numbers if _acceptable_phone(tel)]

    def count_addresses(self):
        """
        Проверяет наличие адреса на странице. 
        Сначала ищет подходящую микроразметку. Если не нашел, ищет по регулярному выражению
        """
        num = len(self.doc.select('//*[@itemtype="http://data-vocabulary.org/Organization"]/*[@class="adr"]')) + \
              len(self.doc.select('//*[@itemtype="http://data-vocabulary.org/Organization"]/*[@class="address"]')) + \
              len(self.doc.select('//*[@class="vcard"]/*[@class="adr"]')) + \
              len(self.doc.select('//*[@itemtype="http://schema.org/Organization"]/*[@class="address"]'))
        if num > 0:
            return num
        p0 = re.compile(r'(?:^|[>,;"\s])'
                        r'(?:[А-Я\d][А-Яа-я ]+ )[\s\n]*'
                        r'(?:[Уу]л[.]|[Пп]р-т|[Пп]р[.]|[Пп]ер[.]|[Нн]аб[.]|'
                        r'[Пп]роспект|[Пп]ереулок|[Уу]лица|[Нн]абережная|[Пп]роезд)'
                        r'(?:[,][\s]*|[\s\n]+)(?:д[.][\s]*|дом[\s]*|)[\d]+[/-]?[А-ЯA-Za-zа-я\d]{0,2}'
                        r'(?:["<]|$|[,]|[\s])')
        p1 = re.compile(r'(?:[">]|^|[,;]|[\s])'
                        r'(?:[Уу]л[.][ ]?|[Пп]р-т |[Пп]р[.][ ]?|[Пп]ер[.][ ]?|[Нн]аб[.][ ]?|'
                        r'[Пп]роспект |[Пп]ереулок |[Уу]лица |[Нн]абережная |[Пп]роезд )'
                        r'(?:[А-Я\d][А-Яа-я ]+)(?:[,][\s]*|[\s\n]+)(?:д[.][\s]*|дом[\s]*|)'
                        r'[\d]+[/-]?[А-ЯA-Za-zа-я\d]{0,2}'
                        r'(?:["<]|$|[,]|[\s])')
        p2 = re.compile(r'Москва[,] [А-Яа-я ]*[,] [0-9]+')
        p3 = re.compile(r'Адрес:[\s\S]*')

        html_body = self.doc.unicode_body()
        return len(p0.findall(html_body) + p1.findall(html_body) + p2.findall(html_body) + p3.findall(html_body))

    def is_search_on_page(self):
        """
        Проверяет наличие поиска по сайту. 
        """
        p = re.compile(r'''(?:^|>|=['"])[\s]*поиск(?: по сайту)?|найти[\s]*(?:$|[<'"])''',
                       flags=re.IGNORECASE)
        if len(p.findall(self.doc.unicode_body())) > 0:
            return True

        return any(['search' in (t.attr('class', default='') + t.attr('id', default='')).lower()
                    for t in self.doc.select('//*')])

    def is_face(self):
        """
        Если страница является "мордой" возвращает True, иначе False. 
        """
        p = re.compile(r'http://(?:[-_a-zA-Z.]+[.])+[a-z]{2,5}[/]?[\s]*$')
        return bool(p.match(self.url))

    def html(self):
        """
        Возвращяет html код страницы
        """
        return self.doc.unicode_body()

    def title(self):
        """
        Возвращяет заголовок страницы, если заголовка нет возвращяет строку 'no-title'
        """
        try:
            return self.doc.select('//title')[0].text()
        except (grab.error.DataNotFound, IndexError):
            return 'no-title'

    def description(self):
        """
        Возвращяет содержимое мета-тега "description", если его нет, возвращяет строку 'no-description'
        """
        try:
            p = self.doc.select('//meta[@name="description"]').node().get('content')
            return p
        except grab.error.DataNotFound:
            return 'no-description'

    #def get_read_doc(self):
    #    return read.Document(self.html())

    def readable(self):
        """
        Возвращяет текстовое содержимое страницы (без сквозного содержимого, содержательну часть) с html разметкой
        """
        return read.Document(self.html()).summary()

    def text(self):
        """
        Возвращяет текстовое содержимое страницы (без сквозного содержимого, содержательнуб часть) без html
        """
        html_doc = read.Document(self.html()).summary()
        soup = bs4.BeautifulSoup(html_doc)
        text = soup.get_text()
        text = text.replace('#', '').replace('↵', '').replace('↑', '').replace('°', '').replace('©', '').replace('«',
                                                                                                                 ''). \
            replace('»', '').replace('$', '').replace('*', '').replace(u"\.", "").replace(u"\,", "").replace(u"^", ""). \
            replace(u"|", "").replace(u"—", "").replace(u',', '').replace(u'•', '').replace(u'﴾', '').replace(u'﴿', '')
        p = re.compile(r'[\n\r][\n\r \t\a]+')
        text = p.sub('\n', text)
        return text

    def text_len(self):
        """
        Длина содержательного текста
        """
        return len(self.text())

    def is_online_consultant_on_page(self):
        """
        Проверяет наличие онлайн консультанта 
        """
        p = re.compile(r'livetex.ru/js/client.js|web.redhelper.ru/service/main.js|static.cloudim.ru/js/chat.js|'
                       r'consultsystems.ru/script/|//code.jivosite.com/script/widget/|'
                       r'//widget.siteheart.com/apps/js/sh.js')
        return len(p.findall(self.html())) > 0

    def is_video_on_page(self):
        """
        Проверяет наличие видео контента
        """
        divs = self.doc.select('//div[@itemtype="http://schema.org/VideoObject"]')
        if len(divs) > 0:
            return True

        video_sources = ['youtube.com/v/', '//videoapi.my.mail.ru/']
        return any([v in g for frame in self.doc.select('//*[@src]')
                    for g in frame.attr('src', default='')
                    for v in video_sources])

    def is_file_on_page(self, ext='pdf'):
        """
        Проверяет наличие ссылки на фаил с расширением ext
        """
        p = re.compile(r'[.]{}$'.format(ext),
                       flags=re.IGNORECASE)
        return any([len(p.findall(r.text())) > 0 for r in self.doc.select('//a/@href')])

    def is_payment_on_page(self):
        """
        Проверяет наличие ссылки способы оплаты|доставки
        """
        p = re.compile(r'''[>=]["]?[\s](?:(?:варианты|способы|условия)[\s]*(?:оплаты|доставки)[\s]*|'''
                       r'''доставка(?:[\s]*[и\\][\s]*оплата)|'''
                       r'''оплата(?:[\s]*[и\\][\s]доставка))[\s]*["<]''',
                       flags=re.IGNORECASE)  # (?: товаров| [Вв]ашего заказа)?)
        return len(p.findall(self.html())) > 0

    def is_pagination_on_page(self):
        """
        Проверяет наличие пагинатора
        """
        p = re.compile(r'^.*?pag(?:inat|e|ing|or)[\s\S]*?$',
                       flags=re.IGNORECASE)
        cl = self.doc.select('//*[@class]')
        len_nums = 500
        res_nums = []
        for c in cl:
            names = _get_attr(c, 'class')
            cl_names = [n for name in names for n in p.findall(name)]
            if len(cl_names):
                refs = self.doc.select('//*[@class="{}"]//a'.format(cl_names[0]))
                num_pattern = re.compile(r'^\s*\d+\s*$')
                nums = []
                for a in refs:
                    pre_nums = num_pattern.findall(a.text())
                    if len(pre_nums):
                        nums.append(int(pre_nums[0]))
                if 0 < len(nums) < len_nums:
                    if 1 and 2 and 3 in nums or 2 and 3 and 4 in nums or all(n in [1, 2] for n in nums):
                        res_nums = [n for i, n in enumerate(nums) if n not in nums[i + 1:]]
                        len_nums = len(res_nums)
        return max(res_nums) if len(res_nums) else 0

    def _card_in_page_class(self):
        soup = bs4.BeautifulSoup(self.html())
        counter = []
        # texts = []
        items = [item for item in soup.findAll() if item.find('img') and item.find('a')]

        groups = []
        for item in items:
            parent = item.parent
            while parent and str(parent.name) not in ('div', 'table', 'html', 'section', 'td'):  # 'li', 'tr'
                parent = parent.parent
            item_class = _get_attr(item, 'class')
            if len(item_class) == 0:
                item_class = [str(item.name)]
            parent = _get_attr(parent, 'class')
            for cl in item_class:
                try:
                    n = groups.index((cl, parent))
                    counter[n] += 1
                except ValueError:
                    groups.append((cl, parent))
                    counter.append(1)
        in_page = 0
        for i, pu in enumerate(groups):
            if counter[i] < 2:
                continue
            if counter[i] > in_page:  # len(common_words) > 0 and
                in_page = counter[i]
        return in_page

    def _find_offer(self):
        divs = self.doc.select('//*[@itemtype="http://schema.org/Offer"]')
        if len(divs) == 0:
            divs = self.doc.select('//*[@itemtype="http://data-vocabulary.org/Offer"]')
        return divs

    def count_variety_on_page(self):
        tags = self._find_offer()
        if len(tags) > 0:
            return len(tags)

        p = re.compile(r'(\d+)(?:[&]nbsp[;]|\s)*(?:[-—]|[&]mdash[;]|по)(?:[&]nbsp[;]|\s)*(\d+) из (\d+)')
        flt = p.findall(self.html())
        for (first, last, total) in flt:
            in_page = int(last) - int(first) + 1
            return in_page

        return self._card_in_page_class()

    def is_timetable_on_page(self):
        p = re.compile(r'(?:[>"]?[сc] ([\d]{1,2}:[\d]{2}) до ([\d]{1,2}:[\d]{2})|'
                       r'([\d]{1,2}:[\d]{2})(?:-| - )([\d]{1,2}:[\d]{2}))',
                       flags=re.IGNORECASE)
        return len(p.findall(self.html())) > 0

    def is_show_num_res_on_page(self):
        p = re.compile(r'(?:(?:количество(?:[&]nbsp[;]|\s)|>)на стр(?:[.]|анице)|'
                       r'(?:выводить|показывать|показать|вывести)(?:[&]nbsp[;]|\s)(?:по|товаров))'
                       r'[:]?(?:[&]nbsp[;]|\s)*(?:$|"|<)',
                       flags=re.IGNORECASE)
        return len(p.findall(self.html())) > 0

    def is_basket_on_page(self):
        p = re.compile(r'(?:^|>|=")(?:[0-9\s\S]*(?:моя |ваша |товар(?:ов|а)?)?корзин[ае](?: пуста)?[:]?[\s\S]*|'
                       r'[\s]?в корзине(?: (?:[0-9]+|(?:пока)? нет)? товар[(]?(?:ов|а)?[)]?)?)(?:$|<|")',
                       flags=re.IGNORECASE)

        refs = self.doc.select('//a')
        for t in refs:
            href = _get_attr(t, 'href')
            if len(re.compile(r'[/=](?:shopping[-_])?(?:cart|basket)[/]?').findall(' '.join(href))) > 0:
                return True
            if len(p.findall(t.text())) > 0:
                return True

        soup = bs4.BeautifulSoup(self.html())
        spansdivs = [item for item in soup.findAll() if item.find('id') and item.find('class')]
        for t in spansdivs:
            ic = _get_attr(t, 'id') + _get_attr(t, 'class')
            if len(re.compile(r'(?:^|[-_])?(?:cart|basket)(?:$|[-_])').findall(' '.join(ic))) > 0:
                return True
                # if len(p.findall(t.text())) > 0:
                # return True
        return False

    def count_buy_buttons(self):
        soup = bs4.BeautifulSoup(self.html())
        spansdivs = [item for item in soup.findAll() if item.find('class')]

        p_ch = re.compile(r'(?:cart|basket)|'  # (?:add|move)?[_-]?(?:to|[2])?(?:[_-]?shopping)?[_-]?
                          r'^buy$|buy(?:[_-]?btn|[_-]?button|[_-]?now)', flags=re.IGNORECASE)
        btn_types = {}
        btn_lables = {}
        p = re.compile(r'^(?:[&]nbsp[;]|\s|["])*'
                       r'(?:купить(?: в один клик)?$|(?:добавить |положить )?в корзину|в корзину)'
                       r'(?:[&]nbsp[;]|\s|["])*$', flags=re.IGNORECASE)
        for t in spansdivs:
            c = _get_attr(t, 'class')
            btn_class = [cc for cc in c if (len(p_ch.findall(cc)) > 0)]
            for cl in btn_class:
                btn_types[cl] = 1 if cl not in btn_types else btn_types[cl] + 1

            l = p.findall(t.text())
            if len(l) > 0:
                btn_lables[t.text()] = 1 if t.text() not in btn_lables else btn_lables[t.text()] + 1

        if len(btn_types) > 0:
            btns = 0
            for tp in btn_types:
                if btn_types[tp] > btns:
                    btns = btn_types[tp]
            return btns
        if len(btn_lables) > 0:
            btns = 0
            for lable in btn_lables:
                if btn_lables[lable] > btns:
                    btns = btn_lables[lable]
            return btns

        submits = self.doc.select('//input[@type="submit"]')
        submits2 = self.doc.select('//button')
        res = 0
        for t in submits or submits2:
            l = p.findall(t.text())
            res += 1 if len(l) > 0 else 0
        return res

    def count_prices(self):
        divs = self._find_offer()
        prices = []
        p_price_html = re.compile(r'''(?:["'>]|[^в][\s]*){}[ <"']'''.format(self.PRICE_PATTERN))
        for div in divs:
            res = p_price_html.findall(div.html())
            if len(res) > 0:
                st = res[0][1:-1].replace(u'\xa0', ' ')
                prices.append(st)
        if len(prices) > 0:
            return len(prices)

        p = re.compile(r'''(?:[Pp]ri[sc]e|[Cc]ena|[Cc]ost)''')
        tags = self.doc.select('//*[@class]')
        prices = []
        temp_prices = {}
        for t in tags:
            for cl in _get_attr(t, 'class'):
                if len(p.findall(cl)) > 0:
                    res = self.PRICE_PATTERN.findall(t.text())
                    if len(res) > 0:
                        st = res[0]
                        if cl not in temp_prices:
                            temp_prices[cl] = []
                        temp_prices[cl].append(st)
                        prices.append(st)
        if len(temp_prices) > 0:
            ind = max(temp_prices, key=lambda k: len(temp_prices[k]))
            prices = temp_prices[ind]

        if len(prices) == 0:
            p_value = re.compile(r'''(?:["'>(]|(?:от|до|[Цц]ена[\s]*[-:]*[\s]*)(?:[\s]|[&]nbsp[;])+)'''
                                 '''{}[) <"']'''.format(self.PRICE_PATTERN))
            prices = p_value.findall(self.html())
        return len(prices)

    def is_advertising_on_page(self):
         p = re.compile(r'(?:adsbygoogle|yandex_ad)', flags=re.IGNORECASE)
         return len(p.findall(self.html())) > 0

    def is_ask_q_on_page(self):
        p = re.compile(r'>[\s]*(?:задать вопрос|отправить сообщение)[\s]*<', flags=re.IGNORECASE)
        return len(p.findall(self.html())) > 0

    def is_callback_on_page(self):
        p = re.compile(r'(?:заказать |обратный )звонок|[>][\s]*перезвонить[\s]*[<]|'
                       r'call[-_]?back[^s]]|звонок с сайта', flags=re.IGNORECASE)
        return len(p.findall(self.html())) > 0

    def is_price_list_on_page(self):
        p = re.compile(r'[>=]["]?прайс(?:[-\s]лист)?[< ]?', flags=re.IGNORECASE)
        return len(p.findall(self.html())) > 0

    def is_catalogue_on_page(self):
        links = self.doc.select('//a[@href]')
        p_hr = re.compile(r'^/catalog(?:ue)?(?:[/]|$)')
        p = re.compile(r'(?:каталог|категории товаров)', flags=re.IGNORECASE)
        for l in links:
            hr = _get_attr(l, 'href')
            if len(p_hr.findall(' '.join(hr))) > 0:
                return True
            if len(p.findall(l.html())) > 0:
                return True
        p = re.compile(r'''(?:[>]|=["'])?(?:каталог|категории товаров)(?:[<]|["'])''', flags=re.IGNORECASE)
        return len(p.findall(self.html())) > 0

    def get_all(self):
        return {'phones': len(self.get_phones()),
                'addresses': self.count_addresses(),
                'buy_buttons': self.count_buy_buttons(),
                'prices': self.count_prices(),
                'variety_on_page': self.count_variety_on_page(),
                'basket': self.is_basket_on_page(),
                'ask_q': self.is_ask_q_on_page(),
                'callback': self.is_callback_on_page(),
                'catalogue': self.is_catalogue_on_page(),
                'timetable': self.is_timetable_on_page(),
                'price_list': self.is_price_list_on_page(),
                'num_res': self.is_show_num_res_on_page(),
                'text len': len(self.text()),
                'face': self.is_face(),
                'pdf': self.is_file_on_page(),
                'online_consultant': self.is_online_consultant_on_page(),
                'pagination': self.is_pagination_on_page(),
                'payment_delivery': self.is_payment_on_page(),
                'search': self.is_search_on_page(),
                'video': self.is_video_on_page()}

    def compare(self, pages):
        m = self.get_all()
        markers = [page.get_all() for page in pages]
        v = _vote(markers, m)
        return v

    def count_commercial(self):
        res = self.is_basket_on_page() > 0
        res += self.is_payment_on_page() > 0
        res += self.is_price_list_on_page() > 0
        res += self._find_offer() > 0
        res += self.count_buy_buttons() > 0
        res += self.count_prices() > 0
        return res

    def count_informational(self):
        res = len(self.text()) > 1000
        res += 3 * (self.is_advertising_on_page() > 0)
        res += self.is_video_on_page() > 0

        return res

    def compare_texts(self, pages):
        texts = [pg.text() for pg in pages]
        text = _text_to_list(self.text())
        res = list()
        for word in _common_words(texts):
            if word not in text:
                res.append(word)
        return res

    def compare_titles(self, pages):
        # TODO: write body
        texts = [pg.title() for pg in pages]
        text = _text_to_list(self.title())
        res = list()
        for word in _common_words(texts):
            if word not in text:
                res.append(word)
        return res

    def compare_descriptions(self, pages):
        # TODO: write body
        texts = [pg.description() for pg in pages]
        text = _text_to_list(self.description())
        res = list()
        for word in _common_words(texts):
            if word not in text:
                res.append(word)
        return res



    def gz_rate(self):
        """
        Расчитывает степень сжимаемости текста

        :return:
        """
        t = self.text()
        text_compressed = compress(bytes(t, 'UTF-8'))
        n = getsizeof(t)
        n_compressed = getsizeof(text_compressed)
        return float(n) / n_compressed


def _text_to_list(text):
    res = list()
    for word in nltk.word_tokenize(text):
        word = morth.parse(word)[0].normal_form
        if word in punctuation or word in stop_words:
            continue
        res.append(word)
    return res


def _common_words(texts):
    corpus = [_text_to_list(text) for text in texts]
    fdist = set([tok for s in corpus for tok in s])

    return [tok for tok in fdist if sum([tok in s for s in corpus]) > 7]


def _accept(values, val):
    mean = statistics.mean(values)
    sigma = statistics.stdev(values)
    return mean - sigma <= val <= mean + sigma


def _vote(marker_dics, m):
    acceptable = {}
    for ind in m:
        values = [markers[ind] for markers in marker_dics]
        val = m[ind]
        #print(ind)
        #print(val)
        #print(values)
        acceptable[ind] = _accept(values, val)
    cols = [k for k in acceptable if not acceptable[k]]
    return cols


def quote(url):
    pref = ''
    if url.startswith('http://'):
        pref = 'http://'
        url = url[7:]
    elif url.startswith('https://'):
        pref = 'https://'
        url = url[8:]
    try:
        new_url = urllib.parse.quote(url, encoding='utf-8', errors='strict')
    except UnicodeDecodeError:
        try:
            new_url = urllib.parse.quote(url, encoding='cp1251', errors='strict')
        except UnicodeDecodeError:
            new_url = urllib.parse.quote(url, encoding='koi8-r', errors='strict')
    return pref + new_url


if __name__ == '__main__':
    import search_query.serps as wrs
    semcorefile = 'C:\\_Work\\vostok\\to_clust_prav.txt'
    queries = ['очки для сварщиков']#wrs.queries_from_file(semcorefile)

    for i, query in enumerate(queries):
        print(query)

        serp_items = wrs.read_serp(query, 213, 10)
        pgs = []
        for item in serp_items:
            print(item.url)
            try:
                pgs.append(WebPage(item.url))
            except grab.error.GrabTooManyRedirectsError:
                continue

        pos, serp_item = wrs.read_url_position('vostok.ru', query, 213)
        print(pos)
        print(serp_item.url)
        pg = WebPage(serp_item.url)
        #print(pg.compare(pgs))


        print([p.text_len() for p in pgs ])
        print(pg.text_len())

        print('text:', pg.compare_texts(pgs))
        print('title:', pg.compare_titles(pgs))
        print('description:', pg.compare_descriptions(pgs))
