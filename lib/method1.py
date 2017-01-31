### logistic regression method
import nltk
import numpy
import pickle
from sklearn.linear_model import LogisticRegression
from public_score_auc import score_function


embedding_data = pickle.load(open('data/polyglot-fr.pkl', 'rb'))
word_embedding = {}

for i in range(len(embedding_data[0])):
    word_embedding[ embedding_data[0][i] ] = embedding_data[1][i]


def process_input(X):
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

def predict(Xtrain,Ytrain,Xtest,X_percentage,output_path):

	split_idx = int((float(X_percentage) / float(100)) * len(Xtrain))

	print "--- DATABASE COMPOSITION ---"
	print "train_train: %d entries" % split_idx
	print "train_test:  %d entries" % (len(Xtrain) - split_idx)
	print "test:        %d entries" % len(Xtest)
	print "----------------------------"

	Ytrain_train = Ytrain[:split_idx]
	Ytrain_test = Ytrain[split_idx:]

	Xtrain_train = Xtrain[:split_idx,:]
	Xtrain_test = Xtrain[split_idx:,:]

	model = LogisticRegression(C=0.1)

	print "-------- fitting on the train_train data"
	model = model.fit(Xtrain_train, Ytrain_train)

	y = model.predict(Xtrain_test)
	prob = model.predict_proba(Xtrain_test)[:,1]

	correct = 0
	for i in range(len(Ytrain_test)):
	    if y[i] == Ytrain_test[i]:
	        correct += 1
	print "Cross-validation -> Precision at .5 threshold: {0} / {1}".format(correct, len(Ytrain_test))
	print "Score on cross validation : %.5f" % score_function(Ytrain_test,prob)

	# fit the model on the whole dataset
	print "-------- fitting on the whole train data"
	model = model.fit(Xtrain, Ytrain)

	prob = model.predict_proba(Xtest)[:,1]

	f = open(output_path, 'w')
	f.write('ID;TARGET\n')

	pos = 80001
	for p in prob:
	    f.write("{0};{1:.10f}\n".format(pos, p))
	    pos += 1
	print "Ytest output to : %s" % output_path