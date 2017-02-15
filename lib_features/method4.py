from nltk import word_tokenize
from nltk.stem.snowball import FrenchStemmer
import numpy
import string
import word2vec

from utils import get_stop_words, fix_token, IDF, TF

# "one-hot" tf-idf based on stemmed words, (otherwise we end up with 50k+ words)
def extract_features(X, model_filename, dim=1000, tf_scheme=None, debug=False):
    print "method of feature extraction #4 (pure TF-IDF) (TF_scheme = %s)" % (tf_scheme if (not tf_scheme is None) else "raw_frequency")
    print "model filename: %s" % model_filename
    model = word2vec.load(model_filename)
    n = len(X)
    stop = get_stop_words()

    if debug:
        f = open('feature5.log', 'w')
        f.write('stopwords: ' + str(stop) + '\n')

    # pre-processing to get all the words present in the training corpus (contents and titles, concatenated)
    word2idf = IDF(X,model,stemming=True,dim=dim)
    all_words = word2idf.keys()
    dim = len(all_words)
    print dim # can be very long, need for a stemmer
    stemmer = FrenchStemmer()

    if debug:
        f.write('stemmed word keys: ' + str(all_words) + '\n')

    ret = numpy.zeros((n,2 * dim + 3)) # 3 is for content_sum_weights, title_sum_weights, n_stars

    for i in range(n):
        if debug:
            f.write('id: ' + str(X[i]['id']) + '\n')
        ### review content
        tokens = word_tokenize(X[i]['content'], language='french')
        tokens = [w.lower() for w in tokens]
        tokens = [w for w in tokens if not w in stop]

        if debug:
            f.write('(content) tokens: ' + str(tokens) + '\n')

        tokens = [fix_token(model, w) for w in tokens]
        tokens_model = []

        word2tf = None

        word2tf = TF(tokens,tf_scheme,stemming=True)

        tfidf_scores = {key:0 for key in word2idf}
        sum_weights = 0

        stemmed_tokens = [stemmer.stem(token) for token in tokens if not token is None]
        for token in numpy.unique(stemmed_tokens):
            if (token is not None) and (token in word2idf):
                tokens_model.append(token)
                tfidf_scores[token] += word2tf[token] \
                * word2idf[token]
                sum_weights += 1

        if debug:
            f.write('(content) tokens in model: ' + str(tokens_model) + '\n')
            f.write('(content) sum weights: ' + str(sum_weights) + '\n')

        ret[i, 0:dim] = tfidf_scores.values()
        ret[i,dim] = sum_weights

        ### review title
        tokens = word_tokenize(X[i]['title'], language='french')
        tokens = [w.lower() for w in tokens]
        tokens = [w for w in tokens if not w in stop]

        if debug:
            f.write('(title) tokens: ' + str(tokens) + '\n')

        tokens = [fix_token(model, w) for w in tokens]
        tokens_model = []


        word2tf = TF(tokens,tf_scheme,stemming=True)

        tfidf_scores = {key:0 for key in all_words}
        sum_weights = 0

        stemmed_tokens = [stemmer.stem(token) for token in tokens if not token is None]
        for token in numpy.unique(stemmed_tokens):
            if (token is not None) and (token in word2idf):
                tokens_model.append(token)
                tfidf_scores[token] += word2tf[token] * word2idf[token]
                sum_weights += 1

        if debug:
            f.write('(title) tokens in model: ' + str(tokens_model) + '\n')
            f.write('(title) sum weights: ' + str(sum_weights) + '\n')

        ret[i, dim + 1:2 * dim + 1] = tfidf_scores.values()
        ret[i,2 * dim + 1] = sum_weights

        ret[i, 2 * dim + 2] = X[i]['stars']
        if debug:
            f.write('stars: ' + str(X[i]['stars']) + '\n')

    return ret
