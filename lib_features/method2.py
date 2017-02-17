from nltk import word_tokenize
import numpy
import string
import word2vec

from utils import get_stop_words, fix_token

def extract_features(X, model_filename, dim, mix_content_title=True, use_idf=False, word2idf=None, debug=False, log_filename='feature2'):
    print "method of feature extraction #2 (IDF weigthed word2vec word embeddings)"
    print "model filename: %s" % model_filename
    model = word2vec.load(model_filename)
    n = len(X)
    stop = get_stop_words()

    if debug:
        f = open(log_filename + '.log', 'w')
        f.write('stopwords: ' + str(stop) + '\n')

    if mix_content_title:
        ret = numpy.zeros((n, dim + 2))
    else:
        ret = numpy.zeros((n, 2 * dim + 3))

    for i in range(n):
        if debug:
            f.write('id: ' + str(X[i]['id']) + '\n')
        ### review content
        tokens = word_tokenize(X[i]['content'], language='french')
        tokens = [w.lower() for w in tokens]

        if debug:
            f.write('(content) tokens: ' + str(tokens) + '\n')

        tokens = [fix_token(model, w) for w in tokens]
        tokens = [w for w in tokens if not w in stop]
        tokens_model = []

        embedding = numpy.zeros(dim)
        sum_weights = 0
        not_in_model = 0
        total_tokens = len(tokens)
        for token in tokens:
            if token is not None:
                tokens_model.append(token)
                if use_idf:
                    if token in word2idf:
                        embedding += word2idf[token] * model[token]
                        sum_weights += word2idf[token]
                    else:
                        #print token
                        pass
                else:
                    embedding += model[token]
                    sum_weights += 1
            else:
                not_in_model += 1

        if debug:
            f.write('(content) tokens in model: ' + str(tokens_model) + '\n')
            f.write('(content) sum weights: ' + str(sum_weights) + '\n')

        if not mix_content_title:
            if sum_weights == 0:
                #print tokens
                pass
            else:
                embedding /= sum_weights
            ret[i, 0:dim] = embedding
            ret[i, dim] = sum_weights
            #ret[i, dim + 1] = not_in_model / float(total_tokens)

        ### review title
        tokens = word_tokenize(X[i]['title'], language='french')
        tokens = [w.lower() for w in tokens]

        if debug:
            f.write('(title) tokens: ' + str(tokens) + '\n')

        tokens = [fix_token(model, w) for w in tokens]
        tokens = [w for w in tokens if not w in stop]
        tokens_model = []

        if not mix_content_title:
            embedding = numpy.zeros(dim)
            sum_weights = 0
            not_in_model = 0
            total_tokens = len(tokens)
        else:
            total_tokens += len(tokens)

        for token in tokens:
            if token is not None:
                tokens_model.append(token)
                if use_idf:
                    if token in word2idf:
                        embedding += word2idf[token] * model[token]
                        sum_weights += word2idf[token]
                    else:
                        #print token
                        pass
                else:
                    embedding += model[token]
                    sum_weights += 1
            else:
                not_in_model += 1

        if debug:
            f.write('(title) tokens in model: ' + str(tokens_model) + '\n')
            f.write('(title) sum weights: ' + str(sum_weights) + '\n')

        if sum_weights == 0:
            #print tokens
            pass
        else:
            embedding /= sum_weights

        if not mix_content_title:
            ret[i, dim + 1:2 * dim + 1] = embedding
            ret[i, 2 * dim + 1] = sum_weights
            #ret[i, 2 * dim + 3] = not_in_model / float(total_tokens)
            ret[i, 2 * dim + 2] = X[i]['stars']
        else:
            ret[i, 0:dim] = embedding
            ret[i, dim] = sum_weights
            #ret[i, dim + 1] = not_in_model / float(total_tokens)
            ret[i, dim + 1] = X[i]['stars']

        if debug:
            f.write('stars: ' + str(X[i]['stars']) + '\n')

    return ret
