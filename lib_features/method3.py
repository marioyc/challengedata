from nltk import word_tokenize
import numpy
import string
import word2vec

from utils import get_stop_words, fix_token, TF

def extract_features(X, model_filename, dim, use_tfidf=False, word2idf=None, debug=False):
    print "method of feature extraction #3 (TF-IDF weigthed word2vec word embeddings)"
    print "model filename: %s" % model_filename
    model = word2vec.load(model_filename)
    n = len(X)
    ret = numpy.zeros((n,2 * dim + 3)) # 3 is for content_sum_weights, title_sum_weights, n_stars
    stop = get_stop_words()

    if debug:
        f = open('feature4.log', 'w')
        f.write('stopwords: ' + str(stop) + '\n')

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

        embedding = numpy.zeros(dim)
        sum_weights = 0

        word2tf = None
        if use_tfidf:
            word2tf = TF(tokens,"max.5")

        for token in tokens:
            if token is not None:
                tokens_model.append(token)
                if use_tfidf:
                    if token in word2idf:
                        embedding += word2idf[token] * word2tf[token] * model[token]
                        sum_weights += word2idf[token] * word2tf[token]
                    else:
                        #print token
                        pass
                else:
                    embedding += model[token]
                    sum_weights += 1
        if sum_weights == 0:
            #print tokens
            pass
        else:
            embedding /= sum_weights

        if debug:
            f.write('(content) tokens in model: ' + str(tokens_model) + '\n')
            f.write('(content) sum weights: ' + str(sum_weights) + '\n')

        ret[i, 0:dim] = embedding
        ret[i,dim] = sum_weights

        ### review title
        tokens = word_tokenize(X[i]['title'], language='french')
        tokens = [w.lower() for w in tokens]
        tokens = [w for w in tokens if not w in stop]

        if debug:
            f.write('(title) tokens: ' + str(tokens) + '\n')

        tokens = [fix_token(model, w) for w in tokens]
        tokens_model = []

        embedding = numpy.zeros(dim)
        sum_weights = 0

        word2tf = None
        if use_tfidf:
            word2tf = TF(tokens,"max.5")

        for token in tokens:
            if token is not None:
                tokens_model.append(token)
                if use_tfidf:
                    if token in word2idf:
                        embedding += word2idf[token] * word2tf[token] * model[token]
                        sum_weights += word2idf[token] * word2tf[token]
                    else:
                        #print token
                        pass
                else:
                    embedding += model[token]
                    sum_weights += 1
        if sum_weights == 0:
            #print tokens
            pass
        else:
            embedding /= sum_weights

        if debug:
            f.write('(title) tokens in model: ' + str(tokens_model) + '\n')
            f.write('(title) sum weights: ' + str(sum_weights) + '\n')

        ret[i, dim + 1:2 * dim + 1] = embedding
        ret[i,2 * dim + 1] = sum_weights

        ret[i, 2 * dim + 2] = X[i]['stars']
        if debug:
            f.write('stars: ' + str(X[i]['stars']) + '\n')

    return ret
