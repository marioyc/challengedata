from sklearn.ensemble import RandomForestClassifier
from public_score_auc import score_function

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

	model = RandomForestClassifier(n_estimators=5)

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