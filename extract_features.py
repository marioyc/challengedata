import nltk
import numpy
import pickle
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


#### FEATURE EXTRACTION METHODS ####

def feature2(X):
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
    Xtrain = feature2(Xtrain)
    Xtest = feature2(Xtest)
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