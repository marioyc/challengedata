import argparse
import numpy
import os
import pickle

from lib_fit import *

def parseArguments():
    parser = argparse.ArgumentParser(description="Train and test prediction of review interest")

    parser.add_argument("input_path",
        type=str, help='output path for the prediction file')

    parser.add_argument("input_prefix",
        type=str, help='prefix of input files')

    parser.add_argument("output_prefix",
        type=str, help='prefix for output and model files')

    parser.add_argument("-X",
        type=int, help='(optional) Indicate the X-validation split percentage (eg. 70) (defaults to 70)')

    args = parser.parse_args()
    return args

def main():
    args = parseArguments()
    input_path = args.input_path
    input_prefix = args.input_prefix
    Xtrain = pickle.load(open(os.path.join(input_path, input_prefix + '_train.pkl'), 'rb'))
    Ytrain = numpy.load(os.path.join(input_path, 'Ytrain.npy'))
    Xtest = pickle.load(open(os.path.join(input_path, input_prefix + '_test.pkl'), 'rb'))
    #embeddings_index = pickle.load(open(os.path.join(input_path, 'embeddings_index.pkl'), 'rb'))
    embeddings_matrix = pickle.load(open(os.path.join(input_path, 'embeddings_matrix.pkl'), 'rb'))
    print embeddings_matrix.shape
    output_prefix = args.output_prefix
    cross_validate = args.X

    if cross_validate is None:
        cross_validate = 70

    assert (cross_validate > 0 and cross_validate < 100), "Error in cross-validation splitting percentage"

    method5.predict(Xtrain, Ytrain, Xtest, embeddings_matrix, output_prefix, False)

if __name__ == '__main__':
    main()
