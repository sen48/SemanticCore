"""
Различные расстояния между равными по длине списками
"""

class _RatingDistException(Exception):
    pass


def rating_dist(r1, r2, method='CTR'):
    """
    Расстояние между двумя рейтингами. Для нас, это расстояние между выдачами.
    http://habrahabr.ru/post/239797/

    Parameters
    ----------
    r1: list of urls (str ot int),
        первая выдача, может быть представлена ввиде списка урлов или idшников урлов
    r2: list of urls (str ot int)
        вторая выдача,
    method: str,
        может быть 'CTR' или 'classic',
        'classic' - это формула (5) из http://habrahabr.ru/post/239797/
        'CTR' - формула (10)

    Returns
    -------
    float: расстояние
    """
    n = len(r1)
    if n != len(r2):
        raise _RatingDistException('Ratings has different lengths (Lists of different lengths) {} {}'.format(n, len(r2)))
    if method == 'CTR':
        var = lambda u, v: abs(1.0 / n1 - 1.0 / n2)
        dev = 2 * (sum([1 / i for i in range(1, n + 1)]) - n / (n + 1))
    elif method == 'classic':
        var = lambda u, v: abs(n1 - n2)
        dev = (n + 1) * n
    else:
        raise _RatingDistException('Unknown method: {}'.format(method))

    unique_urls = set(r1).union(set(r2))
    summ = 0
    for s in unique_urls:
        try:
            n1 = r1.tolist().index(s) + 1 if s in r1 else n + 1
            n2 = r2.tolist().index(s) + 1 if s in r2 else n + 1
        except AttributeError:
            n1 = r1.index(s) + 1 if s in r1 else n + 1
            n2 = r2.index(s) + 1 if s in r2 else n + 1
        summ += var(n1, n2)
    return summ / dev


def number_of_common_items(r1, r2):
    """
    Мера схожести двух поисковых выдач. Равна количеству урлов, присутствующих в обеих выдачах.

    Parameters
    ----------
    r1: list of urls (str ot int),
        первая выдача, может быть представлена ввиде списка урлов или idшников урлов,
    r2: list of urls (str ot int)
        вторая выдача,

    Returns
    -------
    float: мера схожести
    """
    return sum([int(i in r1) for i in r2])
