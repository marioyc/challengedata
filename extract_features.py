import argparse
import numpy
import os
import pickle
import word2vec

from lib_features import *
from load_data import load_data

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

def main():
    args = parseArguments()
    output_prefix = args.output_prefix
    output_folder = args.output_folder

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    Xtrain, Xtest, Ytrain = load_data()

    ##### Processing
    feature_extraction_method = 2
    word2vec_model = 'data/frWac_non_lem_no_postag_no_phrase_500_skip_cut100.bin'
    emb_dim = 500
    use_tfidf = False
    tf_scheme="max.5"
    debug = True

    if feature_extraction_method == 5:
        model = word2vec.load(word2vec_model)
        embeddings_index = {}
        Xtrain, embeddings_index = method5.extract_features(Xtrain, model, embeddings_index, debug=debug, log_filename='feature5_train')
        Xtest, embeddings_index = method5.extract_features(Xtest, model, embeddings_index, debug=debug, log_filename='feature5_test')

        pickle.dump(Xtrain, open(os.path.join(output_folder, output_prefix + '_train.pkl'), 'wb'))
        pickle.dump(Xtest, open(os.path.join(output_folder, output_prefix + '_test.pkl'), 'wb'))

        embeddings_matrix = numpy.zeros((len(embeddings_index) + 1, emb_dim))

        for k,v in embeddings_index.iteritems():
            embeddings_matrix[v] = model[k]

        #pickle.dump(embeddings_index, open(os.path.join(output_folder, 'embeddings_index.pkl'), 'wb'))
        pickle.dump(embeddings_matrix, open(os.path.join(output_folder, 'embeddings_matrix.pkl'), 'wb'))
    else:
        if feature_extraction_method == 1:
            Xtrain = method1.extract_features(Xtrain)
            Xtest = method1.extract_features(Xtest)
        elif feature_extraction_method == 2:
            if use_tfidf:
                word2idf = utils.IDF(numpy.concatenate((Xtrain, Xtest), axis=0), word2vec_model)
                Xtrain = method2.extract_features(Xtrain, word2vec_model, emb_dim, use_idf=True, word2idf=word2idf, debug=debug, log_filename='feature2_train')
                Xtest = method2.extract_features(Xtest, word2vec_model, emb_dim, use_idf=True, word2idf=word2idf, debug=debug, log_filename='feature2_test')
            else:
                Xtrain = method2.extract_features(Xtrain, word2vec_model, emb_dim, debug=debug, log_filename='feature2_train')
                Xtest = method2.extract_features(Xtest, word2vec_model, emb_dim, debug=debug, log_filename='feature2_test')
        elif feature_extraction_method == 3:
            if use_tfidf:
                word2idf = utils.IDF(numpy.concatenate((Xtrain, Xtest), axis=0))
                Xtrain = method3.extract_features(Xtrain, word2vec_model, emb_dim, use_tfidf=True, word2idf=word2idf, debug=debug)
                Xtest = method3.extract_features(Xtest, word2vec_model, emb_dim, use_tfidf=True, word2idf=word2idf, debug=debug)
            else:
                Xtrain = method3.extract_features(Xtrain, word2vec_model, emb_dim, debug=debug)
                Xtest = method3.extract_features(Xtest, word2vec_model, emb_dim, debug=debug)
        elif feature_extraction_method == 4:
            #pure tfidf
            Xtrain = method4.extract_features(Xtrain, word2vec_model, tf_scheme=tf_scheme, debug=debug)
            Xtest = method4.extract_features(Xtest, word2vec_model, tf_scheme=tf_scheme, debug=debug)
        else:
            raise Exception("Invalid feature extraction method")

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
