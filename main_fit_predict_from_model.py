import argparse
import numpy
import os
import pickle

from lib_fit import *

def parseArguments():
    parser = argparse.ArgumentParser(description="Train and test prediction of review interest")

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
    input_prefix = args.input_prefix
    #Xtrain = pickle.load(open(os.path.join('features/', input_prefix + '_train.pkl'), 'rb'))
    Xtrain = numpy.load('features/' + input_prefix + '_train_features_1.npy')
    Ytrain = numpy.load(os.path.join('features/', 'Ytrain.npy'))
    Xtest = numpy.load('features/' + input_prefix + '_test_features_1.npy')
    output_prefix = args.output_prefix
    cross_validate = args.X

    if cross_validate is None:
        cross_validate = 70

    assert (cross_validate > 0 and cross_validate < 100), "Error in cross-validation splitting percentage"

    method6.predict(Xtrain, Ytrain, Xtest, output_prefix, True, X_percentage=cross_validate)

if __name__ == '__main__':
    main()
