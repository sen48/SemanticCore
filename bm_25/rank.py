"""
Вычисление классического значения Okapi bm25
"""

from math import log, exp

k1 = 2
b = 0.75
R = 0.0


def score_bm25(cf, tf, dl, avdl, total_lemms):
    """
    Классическое значение bm25
   
    Parameters
    ----------
        cf: numeric, Частота терма в коллекции
        tf: numeric, Term frequency in document
        total_lemms: int, Число терминов в коллекции
        dl: int, Длина документа
        avdl: numeric, Средняя длина документа в коллекции
    Returns
    -------
        score_bm25: numeric, значение bm25
    """

    k = compute_length_normalization_coefficient(dl, avdl)
    first = log(total_lemms/cf)  # log(1 - exp(-1.5*cf/dl))
    second = ((k1 + 1) * tf) / (k + tf)
    return first * second


def compute_length_normalization_coefficient(dl, avdl):
    """
    Computes length normalization coefficient
    
    Parameters
    ----------
        dl: int, Document length
        avdl: numeric, Average document length in collection

    Returns
    -------
        float, length normalization coefficient
    """
    return k1 * ((1 - b) + b * (float(dl) / avdl))



