# invdx.py
# An inverted index


def _read_idfs(file):
    idf = dict()
    for line in open(file, mode='r', encoding='utf-8'):
        p = line.split(';')
        if len(p) != 2:
            raise LookupError('%s is frog' % file)
        word = p[1][:-1]
        if word[0].isdigit():
            continue
        try:
            while not p[0][0].isdigit():
                p[0] = p[0][1:]
            freq = int(p[0])
        except:
            raise LookupError('%s is bad' % file)
        idf[word] = freq
        if word == 'ять':
            print('{} {}'.format(word, freq))
    return idf


class InvertedIndex:

    def __init__(self, idfs=None):
        self.glob_idfs = False
        if idfs:
            self.IDF = _read_idfs(idfs)
            self.glob_idfs = True
        self.index = dict()

    def __contains__(self, item):
        return item in self.index

    def __getitem__(self, item):
        return self.index[item]

    def add(self, word, docid):
        if word in self.index:
            if docid in self.index[word]:
                self.index[word][docid] += 1
            else:
                self.index[word][docid] = 1
        else:
            d = dict()
            d[docid] = 1
            self.index[word] = d

    #frequency of word in document
    def get_document_frequency(self, word, docid):
        if word in self.index:
            if docid in self.index[word]:
                return self.index[word][docid]
            else:
                return 0
                #raise LookupError('%s not in document %s' % (str(word), str(docid)))
        else:
            return 0
            #raise LookupError('%s not in index' % str(word))

    #frequency of word in index, i.e. number of documents that contain word
    def get_index_frequency(self, word):
        if self.glob_idfs:
            if word in self.IDF:
                return self.IDF[word]
            else:
                raise LookupError('%s not in IDF file' % word)
        elif word in self.index:
            return len(self.index[word])
        else:
            raise LookupError('%s not in index' % word)


class DocumentLengthTable:
    def __init__(self):
        self.table = dict()

    def __len__(self):
        return len(self.table)

    def add(self, docid, length):
        self.table[docid] = length

    def get_length(self, docid):
        if docid in self.table:
            return self.table[docid]
        else:
            raise LookupError('%s not found in table' % str(docid))

    def get_average_length(self):
        sum = 0
        for length in self.table.values():
            sum += length
        return float(sum) / float(len(self.table))


def build_data_structures(corpus, file=None):
    idx = InvertedIndex(file)
    dlt = DocumentLengthTable()
    for docid, c in enumerate(corpus):
        # build inverted index
        for word in c:
            idx.add(str(word), docid)
        # build document length table
        length = len(corpus[docid])
        dlt.add(docid, length)
    return idx, dlt
