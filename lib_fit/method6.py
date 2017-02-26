import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb

from utils import output_result

def predict(Xtrain, Ytrain, Xtest, output_prefix, cross_validation=False, X_percentage=None):
    print "--- DATABASE COMPOSITION ---"
    print "train:  %d entries" % len(Xtrain)
    print "test:   %d entries" % len(Xtest)
    print "----------------------------"

    params = {
        'n_estimators' : 2000,
        'max_depth' : 9,
        'learning_rate' : 0.01,
        'gamma' : 1,
        'subsample' : 0.8,
        'colsample_bytree' : 0.8,
        'reg_lambda' : 3,
        #'min_child_weight': 4.0,
    }

    if cross_validation:
        split_idx = int((float(X_percentage) / float(100)) * len(Xtrain))

    	Xtrain_train = Xtrain[:split_idx,:]
    	Xtrain_test = Xtrain[split_idx:,:]

    	Ytrain_train = Ytrain[:split_idx]
    	Ytrain_test = Ytrain[split_idx:]

        eval_set = [(Xtrain_train, Ytrain_train), (Xtrain_test, Ytrain_test)]
        early_stopping_rounds = 10

        model = xgb.XGBClassifier(**params)

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

        plt.show()

    print "-------- fitting on the whole train data"

    model = xgb.XGBClassifier(**params)
    model.fit(Xtrain, Ytrain)

    print "-------- predict probabilities"

    prob = np.array(model.predict_proba(Xtest))
    prob = prob[:,1]
    output_result(prob, "results/" + output_prefix + ".csv")
