from keras.callbacks import Callback
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Input, merge
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
import numpy as np

from utils import output_result

class AUCCallback(Callback):
    def __init__(self):
        super(Callback, self).__init__()

    def on_train_begin(self, logs=None):
        self.params['metrics'] += ['val_auc']

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.model.validation_data[0:3], batch_size=128, verbose=0)
        auc = roc_auc_score(self.model.validation_data[3], y_pred)
        logs['val_auc'] = auc

def load_data(X):
    content = []
    title = []
    stars = []

    for x in X:
        content.append(x['content'])
        title.append(x['title'])
        stars.append(x['stars'])

    return content, title, np.array(stars)

def predict(Xtrain, Ytrain, Xtest, embeddings_matrix, output_path, cross_validation=False):
    Xtrain_content, Xtrain_title, Xtrain_stars = load_data(Xtrain)
    Xtest_content, Xtest_title, Xtest_stars = load_data(Xtest)

    print "--- DATABASE COMPOSITION ---"
    print "train:  %d entries" % len(Xtrain)
    print "test:   %d entries" % len(Xtest)
    print "----------------------------"

    vocab_size = embeddings_matrix.shape[0]
    embed_dim = embeddings_matrix.shape[1]

    maxlen_content  = 350
    Xtrain_content = sequence.pad_sequences(Xtrain_content, maxlen=maxlen_content)
    Xtest_content  = sequence.pad_sequences(Xtest_content, maxlen=maxlen_content)
    maxlen_title = 20
    Xtrain_title = sequence.pad_sequences(Xtrain_title, maxlen=maxlen_title)
    Xtest_title  = sequence.pad_sequences(Xtest_title, maxlen=maxlen_title)

    print "-------- building model"
    nb_filter = 250
    filter_length = 5
    nhid = 128
    hidden_dims = 250

    content = Input(shape=(maxlen_content,))
    title = Input(shape=(maxlen_title,))
    stars = Input(shape=(1,))

    embedding = Embedding(vocab_size,
                        embed_dim,
                        weights=[embeddings_matrix],
                        dropout=0.3)
    x_content = embedding(content)
    x_title = embedding(title)

    conv = Convolution1D(nb_filter=nb_filter,
                            filter_length=filter_length,
                            border_mode='valid',
                            activation='relu',
                            subsample_length=1)
    x_content = conv(x_content)
    x_title = conv(x_title)

    x_content = GlobalMaxPooling1D()(x_content)
    x_content = Dropout(0.5)(x_content)
    x_title = GlobalMaxPooling1D()(x_title)
    x_title = Dropout(0.5)(x_title)

    x = merge([x_content, x_title, stars], mode='concat')
    x = Dense(hidden_dims, activation='relu')(x)
    x = Dropout(0.5)(x)

    output = Dense(1, activation='sigmoid')(x)

    model = Model(input=[content, title, stars], output=[output])
    model.layers[2].trainable = False

    adam = Adam(lr=0.001)

    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()

    print "-------- starting training of the model"

    if cross_validation:
        batch_size       =  64
        nb_epoch         =  20
        auc_callback = AUCCallback()
        history = model.fit([Xtrain_content, Xtrain_title, Xtrain_stars], Ytrain, batch_size=batch_size, nb_epoch=nb_epoch, validation_split=0.3, callbacks=[auc_callback])

        plt.figure(1)
        plt.subplot(1,3,1)
        plt.plot(range(1,nb_epoch + 1), history.history['loss'], 'b', range(1,nb_epoch + 1), history.history['val_loss'], 'r')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')

        plt.subplot(1,3,2)
        plt.plot(range(1,nb_epoch + 1), history.history['acc'], 'b', range(1,nb_epoch + 1), history.history['val_acc'], 'r')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')

        plt.subplot(1,3,3)
        plt.plot(range(1,nb_epoch + 1), history.history['val_auc'], 'r')
        plt.ylabel('auc')
        plt.xlabel('epoch')
        plt.legend(['validation'], loc='upper left')
        plt.show()
    else:
        batch_size = 64
        nb_epoch = 12
        history = model.fit([Xtrain_conten, Xtrain_title, Xtrain_stars], Ytrain, batch_size=batch_size, nb_epoch=nb_epoch)

    model_json = model.to_json()
    with open("models/model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("models/model.h5")

    print "-------- predict probabilities"
    prob = model.predict([Xtest_content, Xtest_title, Xtest_stars], verbose=0)
    prob = prob[:,0]
    output_result(prob, output_path)
