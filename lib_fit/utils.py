def output_distribution(y):
    useful, not_useful = 0, 0
    for c in y:
        if c == 1:
            useful += 1
        else:
            not_useful += 1

    print "Useful reviews: %d" % useful
    print "Not useful reviews: %d" % not_useful
