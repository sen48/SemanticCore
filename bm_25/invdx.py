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


class Entry:

    def __init__(self, position, zone, props=None):
        self.position = position
        self.zone = zone
        if props:
            self.props = props


class PostingList:
    def __init__(self):
        self.posting_list = dict()

    def add(self, doc_id, entry):
        if doc_id not in self.posting_list:
            self.posting_list[doc_id] = list()
        self.posting_list[doc_id].append(entry)

    def tf(self, doc_id, zone=None):
        if doc_id not in self.posting_list:
            return 0
        if zone:
            tf = 0
            for e in self.posting_list[doc_id]:
                if e.zone == zone:
                    tf += 1
            return tf
        else:
            return len(self.posting_list[doc_id])


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

    def add(self, term, docid, e):
        if term not in self.index:
            self.index[term] = PostingList()
        self.index[term].add(docid, e)
    # frequency of word in document
    def get_document_frequency(self, word, docid):
        if word in self.index:
            return self.index[word].tf(docid)
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


