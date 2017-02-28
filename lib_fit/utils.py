import numpy as np

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

def split_data(X):
    content = []
    title = []
    stars = []

    for x in X:
        content.append(x['content'])
        title.append(x['title'])
        stars.append(x['stars'])

    return content, title, np.array(stars)

def get_PR_coordinates(predictions, ground_truth, output_path):
    thr_array = np.linspace(0,1,100)
    pos_array = np.zeros(len(thr_array))
    true_pos_array = np.zeros(len(thr_array))

    for thr_index in range(len(thr_array)):
        thr = thr_array[thr_index]

        filtered_result = np.array(predictions>=thr).astype(int) # 0 or 1
        pos_array[thr_index] += np.sum(filtered_result)
        true_pos_array[thr_index] += np.sum(filtered_result * ground_truth)

    precision = true_pos_array / pos_array
    recall = true_pos_array / np.sum(ground_truth)

    f = open(output_path, 'w')
    for r in recall:
        f.write("%.5f " % r)
    f.write("\n")
    for p in precision:
        f.write("%.5f " % p)
    f.write("\n")
