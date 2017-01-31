import nltk
import numpy
import pickle
from sklearn.linear_model import LogisticRegression

from load_data import load_data

Xtrain, Xtest, Ytrain = load_data()

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

Xtrain = process_input(Xtrain)
Xtest = process_input(Xtest)

model = LogisticRegression(C=0.1)

model = model.fit(Xtrain, Ytrain)

y = model.predict(Xtrain)
prob = model.predict_proba(Xtrain)[:,1]

correct = 0
for i in range(len(Ytrain)):
    if y[i] == Ytrain[i]:
        correct += 1
print "Precision: {0} / {1}".format(correct, len(Ytrain))

prob = model.predict_proba(Xtest)[:,1]

f = open('data/output_test.csv', 'w')
f.write('ID;TARGET\n')

pos = 80001
for p in prob:
    f.write("{0};{1:.10f}\n".format(pos, p))
    pos += 1
