import csv

def load_input_file(filename, ninput, firstID):
    f = open(filename, 'r')
    lines = f.readlines()
    aux = ""
    X = []

    for line in lines:
        aux += line

    pos = 0
    starts = []
    for i in range(ninput):
        target = "\n" + str(firstID + i) + ";"
        L = len(target)

        pos = aux.index(target, pos)
        starts.append(pos + 1)
        pos += L
    starts.append(len(aux))

    for i in range(ninput):
        entry = aux[starts[i]:starts[i + 1]]
        tokens = entry.split(';')
        ntokens = len(tokens)

        content = ""
        for j in range(1,ntokens - 3):
            if j > 1:
                content += " "
            content += tokens[j]

        d = {}
        d['id'] = int(tokens[0])
        d['content'] = content.decode('utf-8')
        d['title'] = tokens[-3].decode('utf-8')
        d['stars'] = tokens[-2]
        d['product'] = tokens[-1]

        X.append(d)

    return X

def load_output_file(filename):
    f = open(filename, 'rU')
    reader = csv.reader(f, delimiter=';')
    reader.next()

    Y = []
    useful, not_useful = 0, 0
    for row in reader:
        Y.append(int(row[1]))
        if Y[-1] == 1:
            useful += 1
        else:
            not_useful += 1

    print "Useful reviews: %d" % useful
    print "Not useful reviews: %d" % not_useful

    return Y

def load_data():
    Xtrain = load_input_file('data/input_train.csv', 80000, 1)
    Xtest = load_input_file('data/input_test.csv', 36395, 80001)
    Ytrain = load_output_file('data/output_train.csv')
    assert len(Xtrain) == len(Ytrain)
    return Xtrain, Xtest, Ytrain

if __name__ == '__main__':
    Xtrain, Xtest, Ytrain = load_data()

    for i in range(5):
        print Xtrain[i], Ytrain[i]
    for i in range(5):
        print Xtrain[-i - 1], Ytrain[i]

    for i in range(5):
        print Xtest[i]
    for i in range(5):
        print Xtest[-i - 1]
