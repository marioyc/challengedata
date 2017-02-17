from sklearn.model_selection import train_test_split
from hyperopt import fmin, tpe, hp, Trials
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb

from utils import output_distribution

def score(params):
    global ntrials
    ntrials += 1
    print "(Trial %d)\n" % ntrials
    print "Training with params :"
    print params

    params['max_depth'] = int(params['max_depth'])
    params['n_estimators'] = 3000
    early_stopping_rounds = 10

    eval_set = [(X_train, y_train), (X_test, y_test)]
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=eval_set, eval_metric='auc',
        early_stopping_rounds=early_stopping_rounds, verbose=False)

    evals_result = model.evals_result()
    rounds = len(evals_result['validation_1']['auc']) - early_stopping_rounds
    auc = evals_result['validation_1']['auc'][-early_stopping_rounds - 1]
    print "rounds: %d, AUC: %.6f" % (rounds, auc)
    print "-" * 20
    return -auc


def optimize(X, y, train_percentage):
    global X_train, X_test, y_train, y_test, ntrials
    split_idx = int((float(train_percentage) / float(100)) * len(X))

    X_train = X[:split_idx,:]
    X_test = X[split_idx:,:]

    y_train = y[:split_idx]
    y_test = y[split_idx:]

    output_distribution(y_train)
    output_distribution(y_test)

    space = {
        'max_depth' : hp.quniform('max_depth', 7, 12, 1),
        'learning_rate' : hp.quniform('learning_rate', 0.005, 0.5, 0.005),
        'min_child_weight' : hp.quniform('min_child_weight', 1, 6, 1),
        'gamma' : hp.quniform('gamma', 0.5, 1, 0.05),
        'subsample' : hp.quniform('subsample', 0.5, 1, 0.05),
        'colsample_bytree' : hp.quniform('colsample_bytree', 0.5, 1, 0.05),
        'reg_lambda' : hp.quniform('reg_lambda', 1, 10, 0.5),
    }

    trials = Trials()
    ntrials = 0
    best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=250)
    print best

def predict(Xtrain, Ytrain, Xtest, output_path, cross_validation=False, X_percentage=None):
    params = {
        'n_estimators' : 2000,
        'max_depth' : 11,
        'learning_rate' : 0.01,
        'gamma' : 1,
        'subsample' : 0.8,
        'colsample_bytree' : 0.8,
        'reg_lambda' : 3,
        #'min_child_weight': 4.0,
    }

    print "-------- PARAMETERS --------"
    for k,v in params.iteritems():
        print "%s: %f" % (k, v)
    print "----------------------------"

    if cross_validation:
        split_idx = int((float(X_percentage) / float(100)) * len(Xtrain))

        print "--- DATABASE COMPOSITION ---"
        print "train_train: %d entries" % split_idx
        print "train_test:  %d entries" % (len(Xtrain) - split_idx)
        print "test:        %d entries" % len(Xtest)
        print "----------------------------"

        Xtrain_train = Xtrain[:split_idx,:]
        Xtrain_test = Xtrain[split_idx:,:]

        Ytrain_train = Ytrain[:split_idx]
        Ytrain_test = Ytrain[split_idx:]

        output_distribution(Ytrain_train)
        output_distribution(Ytrain_test)

        eval_set = [(Xtrain_train, Ytrain_train), (Xtrain_test, Ytrain_test)]

        early_stopping_rounds = 10

        model = xgb.XGBClassifier(**params)

        print "-------- fitting on the train_train data"

        model.fit(Xtrain_train, Ytrain_train, eval_set=eval_set, eval_metric='auc',
            early_stopping_rounds=early_stopping_rounds)

        evals_result = model.evals_result()
        rounds = len(evals_result['validation_0']['auc'])
        params['n_estimators'] = rounds - early_stopping_rounds

        plt.figure(1)
        plt.plot(range(0, rounds), evals_result['validation_0']['auc'], 'b', range(0, rounds), evals_result['validation_1']['auc'], 'r')
        plt.ylabel('auc')
        plt.xlabel('round')
        plt.legend(['train', 'validation'], loc='upper left')

        #xgb.plot_importance(model)
        #plt.show()

    print "-------- fitting on the whole train data"

    model = xgb.XGBClassifier(**params)
    model.fit(Xtrain, Ytrain)

    print "-------- predict probabilities"

    prob = np.array(model.predict_proba(Xtest))#, ntree_limit=model.best_ntree_limit))
    prob = prob[:,1]
    f = open(output_path, 'w')
    f.write('ID;TARGET\n')

    pos = 80001
    for p in prob:
        f.write("{0};{1:.10f}\n".format(pos, p))
        pos += 1
    print "Ytest output to : %s" % output_path
