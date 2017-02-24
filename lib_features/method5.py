from nltk import word_tokenize

from utils import get_stop_words, fix_token

def extract_features(X, model, embeddings_index, debug=False, log_filename='feature5'):
    print "method of feature extraction #5 (list of words for RNN)"
    n = len(X)
    stop = get_stop_words()
    ret = []

    if debug:
        f = open(log_filename + '.log', 'w')
        f.write('stopwords' + str(stop) + '\n')

    index = 1
    max_content_len = 0
    max_title_len = 0

    for i in range(n):
        if debug:
            f.write('id: ' + str(X[i]['id']) + '\n')

        review = {}
        review['stars'] = X[i]['stars']
        review['content'] = []
        review['title'] = []

        ### review content
        tokens = word_tokenize(X[i]['content'], language='french')
        tokens = [w.lower() for w in tokens]
        tokens = [w for w in tokens if not w in stop]

        if debug:
            f.write('(content) tokens: ' + str(tokens) + '\n')

        tokens = [fix_token(model, w) for w in tokens]
        tokens = [w for w in tokens if not w in stop]
        tokens_model = []

        for token in tokens:
            if token is not None:
                tokens_model.append(token)

                if token not in embeddings_index:
                    embeddings_index[token] = index
                    index += 1

                review['content'].append(embeddings_index[token])
            #else:
            #    review['content'].append(0)

        content_len = len(review['content'])
        max_content_len = max(max_content_len, content_len)

        if debug:
            f.write('(content) tokens in model: ' + str(tokens_model) + '\n')

        ### review title
        tokens = word_tokenize(X[i]['title'], language='french')
        tokens = [w.lower() for w in tokens]
        tokens = [w for w in tokens if not w in stop]

        if debug:
            f.write('(title) tokens: ' + str(tokens) + '\n')

        tokens = [fix_token(model, w) for w in tokens]
        tokens = [w for w in tokens if not w in stop]
        tokens_model = []

        for token in tokens:
            if token is not None:
                if token not in embeddings_index:
                    embeddings_index[token] = index
                    index += 1

                review['title'].append(embeddings_index[token])
            #else:
            #    review['title'].append(0)

        title_len = len(review['title'])
        max_title_len = max(max_title_len, title_len)

        if debug:
            f.write('(title) tokens in model: ' + str(tokens_model) + '\n')

        ret.append(review)

    print "Maximum length of content: %d" % max_content_len
    print "Maximum length of title: %d" % max_title_len

    return ret, embeddings_index
