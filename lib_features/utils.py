from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer
import numpy
import operator
import string

def get_stop_words():
    stop = stopwords.words('french')
    stop += list(string.punctuation)
    stop += ["''", "``"]
    stop = set(stop)
    return stop

def fix_token(model, token):
    while len(token) > 1 and token[0] == "'":
        token = token[1:]
    while len(token) > 1 and token[-1] == "'":
        token = token[:-1]
    if token in model:
        return token
    prefixes = ["l'", "s'", "d'"]
    if token[:2] in prefixes:
        token = token[2:]
        if token in model:
            return token
        else:
            return None
    return None

def TF(tokens, norm_scheme=None, K=None, stemming=False):
    # tokens are assumed to have been extracted from review content or title, or both (concatenated)
    # refer to https://fr.wikipedia.org/wiki/TF-IDF
    # print "Computing TF with scheme: %s" % norm_scheme if (not norm_scheme is None) else "raw frequency"

    if stemming:
        stemmer = FrenchStemmer()
        tokens_copy = [stemmer.stem(token) for token in tokens if not token is None]
    else:
        tokens_copy = [token for token in tokens]

    if len(tokens_copy) == 0 or tokens_copy is None:
        return None

    norm_schemes = ["binary","log","max.5","max"]
    raw_freq = {}
    unique_tokens = numpy.unique(tokens_copy)

    for unique_token in unique_tokens:
        for token in tokens_copy:
            if unique_token in raw_freq:
                raw_freq[unique_token] += 1
            else:
                raw_freq[unique_token] = 1

    # now that we have the raw_freq, different normalization schemes can be used
    if norm_scheme in norm_schemes:
        ret = {}
        if norm_scheme=="binary":
            for key,value in raw_freq.iteritems():
                ret[key] = 1
            return ret
        if norm_scheme=="log":
            for key,value in raw_freq.iteritems():
                ret[key] = 1 + numpy.log(value)
            return ret
        if norm_scheme=="max.5":
            max_freq = float(max([value for value in raw_freq.values()]))
            for key,value in raw_freq.iteritems():
                ret[key] = 0.5 + numpy.log(float(value)/max_freq)
            return ret
        if norm_scheme=="max":
            assert (K>=0 and K<=0), "in TF: max scheme was specified but no valid K was"
            max_freq = float(max([value for value in raw_freq.values()]))
            for key,value in raw_freq.iteritems():
                ret[key] = K + (1-K)*numpy.log(float(value)/max_freq)
            return ret
    else:
        # default is the raw frequency
        return raw_freq

def IDF(X, model=None, stemming=False, dim=None):
    # only retain the dim terms with the most document frequency
    print "Computing IDF"
    n = len(X)
    stop = get_stop_words()
    df = {}

    if stemming:
        stemmer = FrenchStemmer()

    for i in range(n):
        ### review content
        tokens = word_tokenize(X[i]['content'], language='french')
        tokens = [w.lower() for w in tokens]
        tokens = [w for w in tokens if not w in stop]
        tokens = numpy.unique(tokens)

        if not model is None:
            tokens = [fix_token(model, w) for w in tokens]

        if stemming:
            tokens = [stemmer.stem(token) for token in tokens if not token is None]

        for token in tokens:
            if token in df:
                df[token] += 1
            else:
                df[token] = 1

        ### review title
        tokens = word_tokenize(X[i]['title'], language='french')
        tokens = [w.lower() for w in tokens]
        tokens = [w for w in tokens if not w in stop]
        tokens = numpy.unique(tokens)

        if not model is None:
            tokens = [fix_token(model, w) for w in tokens]

        if stemming:
            tokens = [stemmer.stem(token) for token in tokens if not token is None]

        for token in tokens:
            if token in df:
                df[token] += 1
            else:
                df[token] = 1

    word2idf = {}
    if not dim is None:
        df = dict(sorted(df.items(), key=operator.itemgetter(1), reverse=True)[:dim])

    for k,v in df.iteritems():
        word2idf[k] = numpy.log(float(n) / v)

    return word2idf
