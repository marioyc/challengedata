import nltk
import numpy
import pickle
from load_data import load_data
from lib import method1

import argparse

def parseArguments():
    parser = argparse.ArgumentParser(description="Train and test prediction of review interest")

    parser.add_argument("Xtrain",
        type=str, help='input path for train features')

    parser.add_argument("Xtest",
        type=str, help='input path for test features')

    parser.add_argument("Ytrain",
        type=str, help='input path for train labels')

    parser.add_argument("output_path",
        type=str, help='output path for the prediction file')

    parser.add_argument("-X",
        type=int, help='(optional) Indicate the X-validation split percentage (eg. 70) (defaults to 70)')

    args = parser.parse_args()
    return args

def main():
    args = parseArguments()
    output_path = args.output_path
    Xtrain = numpy.load(args.Xtrain)
    Xtest = numpy.load(args.Xtest)
    Ytrain = numpy.load(args.Ytrain)
    cross_validate = args.X

    if cross_validate is None:
        cross_validate = 70

    assert (cross_validate > 0 and cross_validate < 100), "Error in cross-validation splitting percentage"

    method1.predict(Xtrain,Ytrain,Xtest,cross_validate,output_path)


if __name__ == '__main__':
    main()
