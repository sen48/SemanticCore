import random

__author__ = 'lvova'



def ctr(position):
    return 34.16200393 / position - 0.83594252


def split_duration(current_conversion_rate, delta, number_of_variations, visitors_per_day, visitors_in_experiment_rate):

    """
    Функция расчитывает минимальное количество дней, необходимое для получения достоверных результатов при
    сплит тестировании. Уровень значимости 0.05, статистическая мощность 0.8

    :param current_conversion_rate: Показатель конверсии на данный момент в долях.
    :param delta: минимальное изменение, которое хотим отследит. В долях от текущего показателя конверсий
    :param number_of_variations: Число различных вариаций
    :param visitors_per_day: Посетителей в день
    :param visitors_in_experiment_rate: Доля посетителей, участвующих в эксперименте
    """

    p = current_conversion_rate
    effect_size = p * delta

    per_day_observations = visitors_per_day * visitors_in_experiment_rate
    per_variation_result = 16 * p*(1 - p) / (effect_size**2)
    result = per_variation_result * number_of_variations
    result = round(round(result, 0) / per_day_observations)
    if result == 0:
        print("Less than a day")
    else:
        print(str(result) + " day")
    return result


def ws_history_parser():
    import pandas as pd
    import statsmodels.api as sm
    import datetime
    from statsmodels.iolib.table import SimpleTable

    file = open('c:/_Work/wsh.txt', mode='r', encoding='utf-8')
    data = {}

    query = ''
    for i, line in enumerate(file.readlines()):
        div, mod = divmod(i, 25)
        if mod == 0:
            query = line[:-1]
            data[query] = ([], [])
        elif mod <= 24:
            cols = (line.split('\t'))
            d = cols[0].split(' - ')[0].split('.')
            date = datetime.date(int(d[2]), int(d[1]), int(d[0]))
            data[query][0].append(date)
            data[query][1].append(int(cols[1]))

    query = random.choice(list(data.keys()))
    print(query)
    ts = pd.TimeSeries(data[query][1])

    dataset = pd.DataFrame(ts, columns=["test"])
    dataset.index = pd.Index(pd.date_range("2013/07/01", periods = len(data[query][1]), freq = 'M'))



    otg = dataset.test
    print(otg.head())
    import matplotlib.pyplot as plt

    plt.figure(0)
    otg.plot(figsize=(12, 6))
    plt.show()
    itog = otg.describe()
    otg.hist()
    plt.show()
    print(itog)

    import numpy as np
    row = ['JB', 'p-value', 'skew', 'kurtosis']
    jb_test = sm.stats.stattools.jarque_bera(otg)
    a = np.vstack([jb_test])
    itog = SimpleTable(a, row)
    print(itog)

    otg1diff = otg.diff(periods=1).dropna()

    test = sm.tsa.adfuller(otg)
    print('adf: ', test[0])
    print('p-value: ', test[1])
    print('Critical values: ', test[4])
    if test[0] > test[4]['5%']:
        print('есть единичные корни, ряд не стационарен')
    else:
        print('единичных корней нет, ряд стационарен')

    m = otg1diff.index[len(otg1diff.index)/2+1]
    r1 = sm.stats.DescrStatsW(otg1diff[m:])
    r2 = sm.stats.DescrStatsW(otg1diff[:m])

    print('p-value: ', sm.stats.CompareMeans(r1, r2).ttest_ind()[1])

    otg1diff.plot(figsize=(12, 6))
    plt.show()


    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(otg1diff.values.squeeze(), lags=14, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(otg1diff, lags=14, ax=ax2)
    plt.show()

    model = sm.tsa.ARIMA(dataset, order=(6, 0, 0)).fit()
    pred1 = model.predict(start=25, end=26)

    print(model.summary())
    print(pred1)



def eject(line):
    res = []
    for col in line.split(',')[:2]:
        while col.startswith('"'):
            col = col[1:]
        res.append(col)
    if res[1].isdigit():
        res[1] = int(res[1])
    return res[0], res[1]


def parse_ga_report():
    file = open('c:/_Work/0.csv', mode='r', encoding='utf-8')
    columns = None
    res = {}

    for line in file.readlines():
        if line.startswith('"'):
            if not columns:
                columns = eject(line)
                continue
            k, v = eject(line)
            res[k] = v
            print(k,v)

    big = set(res.keys())
    import csv
    small = set()

    with open('c:/_Work/qqq1.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';', quotechar='"')
        for row in spamreader:
             small.add(row[0])

    s = sum(res[k] for k in res if isinstance(res[k],int) and k != '(not set)' and k != '(not provided)')
    koeff = (res['(not set)'] + res['(not provided)'] + s) / s
    freg = {k: koeff * res[k] for k in res if isinstance(res[k], int) and k != '(not set)' and k != '(not provided)'}
    brand = ['троицк',
             'pravmagazin',
             'что ']

    print(sum(freg[k] for k in freg if isinstance(res[k], int) and all(b not in k for b in brand)))
    intersect = big.intersection(small)
    print(sum(freg[k] for k in intersect if isinstance(res[k], int)))

    for k in sorted(intersect, key = lambda k:res[k], reverse=True):
        print(k)


ws_history_parser()