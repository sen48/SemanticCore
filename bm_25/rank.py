from math import log

k1 = 1.2
k3 = 100
b = 0.75
R = 0.0


def score_bm25(n, f, qf, r, N, dl, avdl):
    """

    :param n: Количество документов, созердащих терм
    :param f: Term frequency in document
    :param qf: Term frequency in query
    :param r: number of relevant documents that contain the term
            R – r = number of relevant documents that do not contain the term
    :param N: Число документов в коллекции
    :param dl: Длина документа
    :param avdl: Средняя длина документа в коллекции
    :return:
    """


    K = compute_K(dl, avdl)
    first = log(( (r + 0.5) / (R - r + 0.5) ) / ( (n - r + 0.5) / (N - n - R + r + 0.5)))
    second = ((k1 + 1) * f) / (K + f)
    third = ((k3 + 1) * qf) / (k3 + qf)
    return first * second * third


def compute_K(dl, avdl):
    """
    Computes length normalization coefficient
    :param dl: Document length
    :param avdl: Average document length in collection
    :return: length normalization coefficient
    """
    return k1 * ((1 - b) + b * (float(dl) / avdl))


def score(text, query):
    k0 = 0.3
    k1 = 0.1
    k2 = 0.2
    k3 = 0.02
    tokens = nltk.word_tokenize(query)
    terms = _get_terms_from_tokens(tokens)
    idfs = {}
    with sqlite3.connect(os.path.join(DB_PATH, DB_FILE)) as con:
        cur = con.cursor()
        for term in terms:
            fs = cur.execute('''SELECT Doc FROM freq WHERE Lemma = :term''', {'term': term}).fetchall()
            if len(fs) == 0:
                idfs[term] = NUM_DOCS / MIN_DF
            else:
                idfs[term] = NUM_DOCS / fs[0][0]
    if len(terms) == 1:
        return w_single(document, zone, terms, p_type)
    w_s = w_single(document, zone, terms, p_type)
    w_p = w_pair(document, zone, terms, p_type)
    w_a = w_all_words(document, zone, terms, p_type)
    w_ph = w_phrase(document, zone, query, p_type)
    w_h = w_half_phrase(document, zone, terms, idfs, p_type)
    res = w_s + k0 * w_p + k1 * w_a + k2 * w_ph + k3 * w_h
    print('{7:<20} {0: 3d}: {1: 3.2f} = {2: 3.2f} + k0 * {3: 3.2f} + k1 * {4: 3.2f} + k2 * {5: 3.2f}'
          ' + k3 * {6: 3.2f}'.format(document.id, res, w_s, w_p, w_a, w_ph, w_h, zone))
    return w_s + k0 * w_p + k1 * w_a + k2 * w_ph + k3 * w_h
