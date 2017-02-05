import numpy as np
import xgboost as xgb

import matplotlib.pyplot as plt

def predict(Xtrain,Ytrain,Xtest,X_percentage,output_path):
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

    dtrain = xgb.DMatrix(Xtrain_train, label=Ytrain_train)
    dval = xgb.DMatrix(Xtrain_test, label=Ytrain_test)
    eval_set = [(Xtrain_train, Ytrain_train), (Xtrain_test, Ytrain_test)]

    params = {
        'n_estimators' : 1000,
        'max_depth' : 8,
        'learning_rate' : 0.01,
        'gamma' : 1,
        'subsample' : 0.8,
        'colsample_bytree' : 0.8,
        #'reg_lambda' : 3,
    }

    print "-------- PARAMETERS --------"
    for k,v in params.iteritems():
        print "%s: %f" % (k, v)
    print "----------------------------"

    model = xgb.XGBClassifier(**params)

    print "-------- fitting on the train_train data"

    model.fit(Xtrain_train, Ytrain_train, eval_set=eval_set, eval_metric='auc', early_stopping_rounds=10)
    evals_result = model.evals_result()

    rounds = len(evals_result['validation_0']['auc'])
    plt.figure(1)
    plt.plot(range(0, rounds), evals_result['validation_0']['auc'], 'b', range(0, rounds), evals_result['validation_1']['auc'], 'r')
    plt.ylabel('auc')
    plt.xlabel('round')
    plt.legend(['train', 'validation'], loc='upper left')

    #xgb.plot_importance(model)
    plt.show()

    print "-------- fitting on the whole train data"

    model = xgb.XGBClassifier(**params)
    model.fit(Xtrain, Ytrain, eval_metric='auc')

    print "-------- predict probabilities"

    prob = np.array(model.predict_proba(Xtest, ntree_limit=model.best_ntree_limit))
    prob = prob[:,1]
    f = open(output_path, 'w')
    f.write('ID;TARGET\n')

    pos = 80001
    for p in prob:
        f.write("{0};{1:.10f}\n".format(pos, p))
        pos += 1
    print "Ytest output to : %s" % output_path
