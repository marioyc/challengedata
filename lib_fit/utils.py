def output_distribution(y):
    useful, not_useful = 0, 0
    for c in y:
        if c == 1:
            useful += 1
        else:
            not_useful += 1

    print "Useful reviews: %d" % useful
    print "Not useful reviews: %d" % not_useful

def output_result(prob, output_path):
    f = open(output_path, 'w')
    f.write('ID;TARGET\n')

    pos = 80001
    for p in prob:
        f.write("{0};{1:.10f}\n".format(pos, p))
        pos += 1
    print "Ytest output to : %s" % output_path
