import argparse
import nltk
import numpy
import os
import pickle
import operator
import string
import word2vec
from nltk.stem.snowball import FrenchStemmer

from load_data import load_data
from lib import method1

stemmer = FrenchStemmer()

def parseArguments():
    parser = argparse.ArgumentParser(description="Train and test prediction of review interest")

    parser.add_argument("output_prefix",
        type=str, help="output prefix")

    parser.add_argument("output_folder",
        type=str, help='output folder for features')

    parser.add_argument("-y",
        help = 'extract train labels as well', action="store_true")

    args = parser.parse_args()
    return args


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

def TF(tokens,norm_scheme=None,K=None,stemming=False):
    # tokens are assumed to have been extracted from review content or title, or both (concatenated)
    # refer to https://fr.wikipedia.org/wiki/TF-IDF
    # print "Computing TF with scheme: %s" % norm_scheme if (not norm_scheme is None) else "raw frequency"

    if stemming:
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

def IDF(X,model=None,stemming=False,dim=None):
    # only retain the dim terms with the most document frequency
    print "Computing IDF"
    n = len(X)
    stop = nltk.corpus.stopwords.words('french')
    stop += list(string.punctuation)
    stop += ["''", "``"]
    stop = set(stop)
    df = {}

    for i in range(n):
        ### review content
        tokens = nltk.word_tokenize(X[i]['content'], language='french')
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
        tokens = nltk.word_tokenize(X[i]['title'], language='french')
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

#### FEATURE EXTRACTION METHODS ####

def feature2(X):
    print "method of feature extraction: feature2 (with polyglot features)"
    embedding_data = pickle.load(open('data/polyglot-fr.pkl', 'rb'))
    word_embedding = {}

    for i in range(len(embedding_data[0])):
        word_embedding[ embedding_data[0][i] ] = embedding_data[1][i]

    n = len(X)
    ret = numpy.zeros((n,131))

    stop = set(nltk.corpus.stopwords.words('french'))

    for i in range(n):
        ### review content
        tokens = nltk.word_tokenize(X[i]['content'], language='french')
        tokens = [w.lower() for w in tokens]
        tokens = [w for w in tokens if not w in stop]

        embedding = numpy.zeros(64)
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

        ret[i, 0:64] = embedding
        ret[i,64] = cont

        ### review title
        tokens = nltk.word_tokenize(X[i]['title'], language='french')
        tokens = [w.lower() for w in tokens]
        tokens = [w for w in tokens if not w in stop]

        embedding = numpy.zeros(64)
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

        ret[i, 65:129] = embedding
        ret[i,129] = cont

        ret[i, 130] = X[i]['stars']

    return ret

def feature3(X, model_filename, dim, use_idf=False, word2idf=None, debug=False, log_filename='feature3'):
    print "method of feature extraction: feature3 (IDF weigthed word2vec word embeddings)"
    print "model filename: %s" % model_filename
    model = word2vec.load(model_filename)

    n = len(X)
    ret = numpy.zeros((n,2 * dim + 3))

    stop = nltk.corpus.stopwords.words('french')
    stop += list(string.punctuation)
    stop += ["''", "``"]

    if debug:
        f = open(log_filename + '.log', 'w')
        f.write('stopwords: ' + str(stop) + '\n')

    stop = set(stop)

    for i in range(n):
        if debug:
            f.write('id: ' + str(X[i]['id']) + '\n')
        ### review content
        tokens = nltk.word_tokenize(X[i]['content'], language='french')
        tokens = [w.lower() for w in tokens]

        if debug:
            f.write('(content) tokens: ' + str(tokens) + '\n')

        tokens = [fix_token(model, w) for w in tokens]
        tokens = [w for w in tokens if not w in stop]
        tokens_model = []

        embedding = numpy.zeros(dim)
        sum_weights = 0
        for token in tokens:
            if token is not None:
                tokens_model.append(token)
                if use_idf:
                    if token in word2idf:
                        embedding += word2idf[token] * model[token]
                        sum_weights += word2idf[token]
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
        tokens = nltk.word_tokenize(X[i]['title'], language='french')
        tokens = [w.lower() for w in tokens]

        if debug:
            f.write('(title) tokens: ' + str(tokens) + '\n')

        tokens = [fix_token(model, w) for w in tokens]
        tokens = [w for w in tokens if not w in stop]
        tokens_model = []

        embedding = numpy.zeros(dim)
        sum_weights = 0
        for token in tokens:
            if token is not None:
                tokens_model.append(token)
                if use_idf:
                    if token in word2idf:
                        embedding += word2idf[token] * model[token]
                        sum_weights += word2idf[token]
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

def feature4(X, model_filename, dim, use_tfidf=False, word2idf=None, debug=False):
    print "method of feature extraction: feature4 (TF-IDF weigthed word2vec word embeddings)"
    print "model filename: %s" % model_filename
    model = word2vec.load(model_filename)

    n = len(X)
    ret = numpy.zeros((n,2 * dim + 3)) # 3 is for content_sum_weights, title_sum_weights, n_stars

    stop = nltk.corpus.stopwords.words('french')
    stop += list(string.punctuation)
    stop += ["''", "``"]
    stop = set(stop)

    if debug:
        f = open('feature4.log', 'w')
        f.write('stopwords: ' + str(nltk.corpus.stopwords.words('french')) + '\n')

    for i in range(n):
        if debug:
            f.write('id: ' + str(X[i]['id']) + '\n')
        ### review content
        tokens = nltk.word_tokenize(X[i]['content'], language='french')
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
        tokens = nltk.word_tokenize(X[i]['title'], language='french')
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

# "one-hot" tf-idf based on stemmed words, (otherwise we end up with 50k+ words)
def feature5(X, model_filename, dim=1000, tf_scheme=None, debug=False):
    print "method of feature extraction: feature5 (pure TF-IDF) (TF_scheme = %s)" % (tf_scheme if (not tf_scheme is None) else "raw_frequency")
    print "model filename: %s" % model_filename
    model = word2vec.load(model_filename)

    n = len(X)

    stop = nltk.corpus.stopwords.words('french')
    stop += list(string.punctuation)
    stop += ["''", "``"]
    stop = set(stop)

    if debug:
        f = open('feature5.log', 'w')
        f.write('stopwords: ' + str(nltk.corpus.stopwords.words('french')) + '\n')

    # pre-processing to get all the words present in the training corpus (contents and titles, concatenated)
    word2idf = IDF(X,model,stemming=True,dim=dim)
    all_words = word2idf.keys()
    dim = len(all_words)
    print dim # can be very long, need for a stemmer
    if debug:
        f.write('stemmed word keys: ' + str(all_words) + '\n')

    ret = numpy.zeros((n,2 * dim + 3)) # 3 is for content_sum_weights, title_sum_weights, n_stars

    for i in range(n):
        if debug:
            f.write('id: ' + str(X[i]['id']) + '\n')
        ### review content
        tokens = nltk.word_tokenize(X[i]['content'], language='french')
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
        tokens = nltk.word_tokenize(X[i]['title'], language='french')
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


def process_input(X):
    embedding_data = pickle.load(open('data/polyglot-fr.pkl', 'rb'))
    word_embedding = {}

    for i in range(len(embedding_data[0])):
        word_embedding[ embedding_data[0][i] ] = embedding_data[1][i]

    n = len(X)
    ret = numpy.zeros((n,65))

    stop = set(nltk.corpus.stopwords.words('french'))

    for i in range(n):
        tokens = nltk.word_tokenize(X[i]['content'], language='french')
        tokens = [w.lower() for w in tokens]
        tokens = [w for w in tokens if not w in stop]

        embedding = numpy.zeros(64)
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

        ret[i, 0:64] = embedding
        ret[i, 64] = X[i]['stars']

    return ret

####################################


def main():
    args = parseArguments()
    output_prefix = args.output_prefix
    output_folder = args.output_folder

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    Xtrain, Xtest, Ytrain = load_data()

    ##### Processing
    word2vec_model = 'data/frWac_non_lem_no_postag_no_phrase_200_skip_cut100.bin'
    emb_dim = 200
    use_tfidf = True
    tf_scheme="max.5"
    debug = True

#    pure tfidf
    Xtrain = feature5(Xtrain, word2vec_model, tf_scheme=tf_scheme, debug=debug)
    Xtest = feature5(Xtest, word2vec_model, tf_scheme=tf_scheme, debug=debug)

    # if use_tfidf:
    #     word2idf = IDF(numpy.concatenate((Xtrain, Xtest), axis=0))
    #     Xtrain = feature4(Xtrain, word2vec_model, emb_dim, use_tfidf=True, word2idf=word2idf, debug=debug)
    #     Xtest = feature4(Xtest, word2vec_model, emb_dim, use_tfidf=True, word2idf=word2idf, debug=debug)
    # else:
    #     Xtrain = feature4(Xtrain, word2vec_model, emb_dim, debug=debug)
    #     Xtest = feature4(Xtest, word2vec_model, emb_dim, debug=debug)

    #if use_tfidf:
    #    word2idf = IDF(numpy.concatenate((Xtrain, Xtest), axis=0), word2vec_model)
    #    Xtrain = feature3(Xtrain, word2vec_model, emb_dim, use_idf=True, word2idf=word2idf, debug=debug, log_filename='feature3_train')
    #    Xtest = feature3(Xtest, word2vec_model, emb_dim, use_idf=True, word2idf=word2idf, debug=debug, log_filename='feature3_test')
    #else:
    #    Xtrain = feature3(Xtrain, word2vec_model, emb_dim, debug=debug, log_filename='feature3_train')
    #    Xtest = feature3(Xtest, word2vec_model, emb_dim, debug=debug, log_filename='feature3_test')

    #####

    numpy.save(os.path.join(output_folder,output_prefix + '_train.npy'),Xtrain)
    print "train features saved at: %s" % os.path.join(output_folder,output_prefix + '_train.npy')
    numpy.save(os.path.join(output_folder,output_prefix + '_test.npy'),Xtest)
    print "test features saved at: %s" % os.path.join(output_folder,output_prefix + '_test.npy')

    if args.y:
        numpy.save(os.path.join(output_folder,'Ytrain.npy'),Ytrain)
        print "train features saved at: %s" % os.path.join(output_folder,'Ytrain.npy')

if __name__ == '__main__':
    main()
