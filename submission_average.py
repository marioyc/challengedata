import csv

submissions_dir = "results/"
submissions_filename = ["trial10.csv", "trial13.csv"]
output_path = submissions_dir + "average.csv"

prob_list = []
n_submissions = len(submissions_filename)

for filename in submissions_filename:
    f = open(submissions_dir + filename)
    reader = csv.reader(f, delimiter=';')
    reader.next()

    p = []
    for row in reader:
        p.append(float(row[1]))

    prob_list.append(p)

f = open(output_path, 'w')

pos = 80001
n_prob = len(prob_list[0])
for i in range(n_prob):
    p = 0
    for j in range(n_submissions):
        p += prob_list[j][i]
    p /= n_submissions

    f.write("{0};{1:.10f}\n".format(pos, p))
    pos += 1
