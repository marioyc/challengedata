from nltk import word_tokenize
import numpy
import pickle

from utils import get_stop_words

def extract_features(X):
    print "method of feature extraction #1 (with polyglot features)"
    embedding_data = pickle.load(open('data/polyglot-fr.pkl', 'rb'))
    dim = 64
    word_embedding = {}

    for i in range(len(embedding_data[0])):
        word_embedding[ embedding_data[0][i] ] = embedding_data[1][i]

    n = len(X)
    ret = numpy.zeros((n, 2 * dim + 3))

    stop = get_stop_words

    for i in range(n):
        ### review content
        tokens = word_tokenize(X[i]['content'], language='french')
        tokens = [w.lower() for w in tokens]
        tokens = [w for w in tokens if not w in stop]

        embedding = numpy.zeros(dim)
        cont = 0
        for token in tokens:
            if token in word_embedding:
                embedding += word_embedding[token]
                cont += 1
        if cont == 0:
            #print tokens
            pass
        else:
            embedding /= cont

        ret[i, 0:dim] = embedding
        ret[i, dim] = cont

        ### review title
        tokens = word_tokenize(X[i]['title'], language='french')
        tokens = [w.lower() for w in tokens]
        tokens = [w for w in tokens if not w in stop]

        embedding = numpy.zeros(dim)
        cont = 0
        for token in tokens:
            if token in word_embedding:
                embedding += word_embedding[token]
                cont += 1
        if cont == 0:
            #print tokens
            pass
        else:
            embedding /= cont

        ret[i, dim + 1:2 * dim + 1] = embedding
        ret[i, 2 * dim + 1] = cont

        ret[i, 2 * dim + 2] = X[i]['stars']

    return ret
