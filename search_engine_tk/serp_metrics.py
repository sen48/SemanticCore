

class RatingDistException(Exception):
    pass


def rating_dist(r1, r2, method='CTR'):
    n = len(r1)
    if n != len(r2):
        raise RatingDistException('Ratings has different lengths (Lists of different lengths) {} {}'.format(n, len(r2)))
    if method == 'CTR':
        var = lambda u, v: abs(1.0 / n1 - 1.0 / n2)
        dev = 2 * (sum([1 / i for i in range(1, n + 1)]) - n / (n + 1))
    elif method == 'classic':
        var = lambda u, v: abs(n1 - n2)
        dev = (n + 1) * n
    else:
        raise RatingDistException('Unknown method: {}'.format(method))

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


number_of_common_urls = lambda u, v: sum([int(i in v) for i in u])