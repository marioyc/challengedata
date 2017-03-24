from keras.callbacks import Callback
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, SpatialDropout1D
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.layers.merge import concatenate
from keras.optimizers import Adam, RMSprop, Adagrad, Adadelta, Nadam
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

from utils import output_result, split_data, get_PR_coordinates

import numpy

class AUCCallback(Callback):
    def __init__(self):
        super(Callback, self).__init__()

    def on_train_begin(self, logs=None):
        self.params['metrics'] += ['val_auc']

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.validation_data[0:3], batch_size=128, verbose=0)
        auc = roc_auc_score(self.validation_data[3], y_pred)
        logs['val_auc'] = auc

def pop_layer(model):
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.output_layers = [model.layers[-1]]
    model.layers[-1].outbound_nodes = []

def predict(Xtrain, Ytrain, Xtest, embeddings_matrix, output_prefix, cross_validation=False):
    Xtrain_content, Xtrain_title, Xtrain_stars = split_data(Xtrain)
    Xtest_content, Xtest_title, Xtest_stars = split_data(Xtest)

    print "--- DATABASE COMPOSITION ---"
    print "train:  %d entries" % len(Xtrain)
    print "test:   %d entries" % len(Xtest)
    print "----------------------------"

    vocab_size = embeddings_matrix.shape[0]
    embed_dim = embeddings_matrix.shape[1]

    maxlen_content = 325
    Xtrain_content = sequence.pad_sequences(Xtrain_content, maxlen=maxlen_content)
    Xtest_content  = sequence.pad_sequences(Xtest_content, maxlen=maxlen_content)
    maxlen_title = 20
    Xtrain_title = sequence.pad_sequences(Xtrain_title, maxlen=maxlen_title)
    Xtest_title  = sequence.pad_sequences(Xtest_title, maxlen=maxlen_title)

    print "-------- building model"
    nb_filter = 150
    hidden_dims = 250

    content = Input(shape=(maxlen_content,))
    title = Input(shape=(maxlen_title,))
    stars = Input(shape=(1,))

    embedding = Embedding(vocab_size,
                        embed_dim,
                        weights=[embeddings_matrix])
    x_content = embedding(content)
    x_title = embedding(title)

    #x_content = SpatialDropout1D(rate=0.3)(x_content)
    #x_title = SpatialDropout1D(rate=0.3)(x_title)

    conv1 = Convolution1D(filters=nb_filter,
                            kernel_size=3,
                            padding='valid',
                            activation='relu')
    x_content1 = conv1(x_content)
    x_title1 = conv1(x_title)
    x_content1 = GlobalMaxPooling1D()(x_content1)
    x_content1 = Dropout(0.5)(x_content1)
    x_title1 = GlobalMaxPooling1D()(x_title1)
    x_title1 = Dropout(0.5)(x_title1)

    conv2 = Convolution1D(filters=nb_filter,
                            kernel_size=5,
                            padding='valid',
                            activation='relu')
    x_content2 = conv2(x_content)
    x_title2 = conv2(x_title)
    x_content2 = GlobalMaxPooling1D()(x_content2)
    x_content2 = Dropout(0.5)(x_content2)
    x_title2 = GlobalMaxPooling1D()(x_title2)
    x_title2 = Dropout(0.5)(x_title2)

    x = concatenate([x_content1, x_title1, x_content2, x_title2, stars])
    x = Dense(hidden_dims, activation='relu')(x)
    x = Dropout(0.5)(x)

    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[content, title, stars], outputs=[output])
    model.layers[2].trainable = False

    #optimizer = Adam(lr=0.0001)
    #optimizer = RMSprop(lr=0.001)
    #optimizer = Adagrad(lr=0.001)
    #optimizer = Adadelta(lr=0.001)
    optimizer = Nadam(lr=0.0001)

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()

    print "-------- starting training of the model"

    if cross_validation:
        batch_size = 64
        nb_epoch = 20
        auc_callback = AUCCallback()
        history = model.fit([Xtrain_content, Xtrain_title, Xtrain_stars], Ytrain, batch_size=batch_size, epochs=nb_epoch, validation_split=0.3, callbacks=[auc_callback])

        f, axarr = plt.subplots(3, sharex=True)
        axarr[0].plot(range(1,nb_epoch + 1), history.history['loss'], 'b', range(1,nb_epoch + 1), history.history['val_loss'], 'r')
        axarr[0].set_ylabel('loss')
        legend0 = axarr[0].legend(['train', 'validation'], loc='upper left')

        axarr[1].plot(range(1,nb_epoch + 1), history.history['acc'], 'b', range(1,nb_epoch + 1), history.history['val_acc'], 'r')
        axarr[1].set_ylabel('accuracy')
        legend1 = axarr[1].legend(['train', 'validation'], loc='upper left')

        axarr[2].plot(range(1,nb_epoch + 1), history.history['val_auc'], 'r')
        axarr[2].set_ylabel('auc')
        axarr[2].set_xlabel('epoch')
        legend2 = axarr[2].legend(['validation'], loc='upper left')

        f.savefig("plots/" + output_prefix + ".png", bbox_extra_artists=(legend0, legend1, legend2), bbox_inches='tight')

        split_idx = int(0.7 * len(Xtrain))
        Xtrain_content = Xtrain_content[split_idx:]
        Xtrain_title = Xtrain_title[split_idx:]
        Xtrain_stars = Xtrain_stars[split_idx:]
        Ytrain = Ytrain[split_idx:]
        prob = model.predict([Xtrain_content, Xtrain_title, Xtrain_stars], verbose=0)
        prob = prob[:,0]
        get_PR_coordinates(prob, Ytrain, "plots/" + output_prefix + "_pr.txt")
    else:
        batch_size = 64
        nb_epoch = 18
        history = model.fit([Xtrain_content, Xtrain_title, Xtrain_stars], Ytrain, batch_size=batch_size, epochs=nb_epoch)

    model_json = model.to_json()
    with open("models/" + output_prefix + ".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("models/" + output_prefix + ".h5")

    print "-------- predict probabilities"
    prob = model.predict([Xtest_content, Xtest_title, Xtest_stars], batch_size=128, verbose=0)
    prob = prob[:,0]
    output_result(prob, "results/" + output_prefix + ".csv")

    print "-------- save intermediate features"
    pop_layer(model)
    pop_layer(model)
    print model.layers
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    train_features = model.predict([Xtrain_content, Xtrain_title, Xtrain_stars], batch_size=128, verbose=0)
    test_features = model.predict([Xtest_content, Xtest_title, Xtest_stars], batch_size=128, verbose=0)
    print train_features.shape, test_features.shape
    numpy.save("features/" + output_prefix + "_train_features_1.npy", train_features)
    numpy.save("features/" + output_prefix + "_test_features_1.npy", test_features)

    pop_layer(model)
    print model.layers
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    train_features = model.predict([Xtrain_content, Xtrain_title, Xtrain_stars], batch_size=128, verbose=0)
    test_features = model.predict([Xtest_content, Xtest_title, Xtest_stars], batch_size=128, verbose=0)
    print train_features.shape, test_features.shape
    numpy.save("features/" + output_prefix + "_train_features_2.npy", train_features)
    numpy.save("features/" + output_prefix + "_test_features_2.npy", test_features)
