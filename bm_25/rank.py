from math import log, exp

k1 = 2
b = 0.75
R = 0.0


def score_bm25(cf, tf, dl, avdl, total_lemms):
    """
    Вычисляет значение классического BM-25
    :param cf: Частота терма в коллекции
    :param tf: Term frequency in document
    :param N: Число документов в коллекции
    :param dl: Длина документа
    :param avdl: Средняя длина документа в коллекции
    :return:
    """

    k = compute_length_normalization_coefficient(dl, avdl)
    first = log(total_lemms/cf)
    # log(1 - exp(-1.5*cf/dl))
    second = ((k1 + 1) * tf) / (k + tf)
    return first * second


def compute_length_normalization_coefficient(dl, avdl):
    """
    Computes length normalization coefficient
    :param dl: Document length
    :param avdl: Average document length in collection
    :return: length normalization coefficient
    """
    return k1 * ((1 - b) + b * (float(dl) / avdl))



