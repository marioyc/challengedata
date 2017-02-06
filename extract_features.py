import nltk
import numpy
import pickle
import word2vec
from load_data import load_data
from lib import method1
import os
import argparse

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

def IDF(X):
    print "Computing IDF"
    n = len(X)
    stop = set(nltk.corpus.stopwords.words('french'))
    df = {}

    for i in range(n):
        ### review content
        tokens = nltk.word_tokenize(X[i]['content'], language='french')
        tokens = [w.lower() for w in tokens]
        tokens = [w for w in tokens if not w in stop]
        tokens = numpy.unique(tokens)

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

        for token in tokens:
            if token in df:
                df[token] += 1
            else:
                df[token] = 1

    word2idf = {}
    for k,v in df.iteritems():
        word2idf[k] = numpy.log(float(n) / v)

    return word2idf

#### FEATURE EXTRACTION METHODS ####

def feature2(X):
    print "method of feature extraction: feature2"
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

def feature3(X, model_filename, dim, use_idf=False, word2idf=None, debug=False):
    print "method of feature extraction: feature3"
    print "model filename: %s" % model_filename
    model = word2vec.load(model_filename)

    n = len(X)
    ret = numpy.zeros((n,2 * dim + 3))

    stop = nltk.corpus.stopwords.words('french')
    stop = set(stop)

    if debug:
        f = open('feature3.log', 'w')
        f.write('stopwords: ' + str(nltk.corpus.stopwords.words('french')) + '\n')

    for i in range(n):
        if debug:
            f.write('id: ' + X[i]['id'] + '\n')
        ### review content
        tokens = nltk.word_tokenize(X[i]['content'], language='french')
        tokens = [w.lower() for w in tokens]
        tokens = [w for w in tokens if not w in stop]
        tokens_model = []

        embedding = numpy.zeros(dim)
        sum_weights = 0
        for token in tokens:
            if token in model:
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
            f.write('(content) tokens: ' + str(tokens) + '\n')
            f.write('(content) tokens in model: ' + str(tokens_model) + '\n')
            f.write('(content) sum weights: ' + str(sum_weights) + '\n')

        ret[i, 0:dim] = embedding
        ret[i,dim] = sum_weights

        ### review title
        tokens = nltk.word_tokenize(X[i]['title'], language='french')
        tokens = [w.lower() for w in tokens]
        tokens = [w for w in tokens if not w in stop]
        tokens_model = []

        embedding = numpy.zeros(dim)
        sum_weights = 0
        for token in tokens:
            if token in model:
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
            f.write('(title) tokens: ' + str(tokens) + '\n')
            f.write('(title) tokens in model: ' + str(tokens_model) + '\n')
            f.write('(title) sum weights: ' + str(sum_weights) + '\n')

        ret[i, dim + 1:2 * dim + 1] = embedding
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
    use_idf = False
    debug = True

    if use_idf:
        word2idf = IDF(numpy.concatenate((Xtrain, Xtest), axis=0))
        Xtrain = feature3(Xtrain, word2vec_model, emb_dim, use_idf=True, word2idf=word2idf, debug=debug)
        Xtest = feature3(Xtest, word2vec_model, emb_dim, use_idf=True, word2idf=word2idf, debug=debug)
    else:
        Xtrain = feature3(Xtrain, word2vec_model, emb_dim, debug=debug)
        Xtest = feature3(Xtest, word2vec_model, emb_dim, debug=debug)
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
