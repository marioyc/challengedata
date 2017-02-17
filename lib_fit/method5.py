from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM, SimpleRNN, GRU
from keras.layers import Convolution1D
from keras.optimizers import Adam

import matplotlib.pyplot as plt

def predict(Xtrain, Ytrain, Xtest, embeddings_matrix, X_percentage, output_path):
    split_idx = int((float(X_percentage) / float(100)) * len(Xtrain))

    print "--- DATABASE COMPOSITION ---"
    print "train_train: %d entries" % split_idx
    print "train_test:  %d entries" % (len(Xtrain) - split_idx)
    print "test:        %d entries" % len(Xtest)
    print "----------------------------"

    vocab_size = embeddings_matrix.shape[0]
    embed_dim = embeddings_matrix.shape[1]

    maxlen  = 350
    Xtrain = sequence.pad_sequences(Xtrain, maxlen=maxlen)
    Xtest  = sequence.pad_sequences(Xtest, maxlen=maxlen)

    print "-------- building model"
    nb_filter = 250
    filter_length = 3
    nhid = 128
    hidden_dims = 250

    model = Sequential()
    model.add(Embedding(vocab_size,
                        embed_dim,
                        weights=[embeddings_matrix],
                        input_length=maxlen,
                        dropout=0.3))
    model.layers[0].trianable = False
    model.add(Convolution1D(nb_filter=nb_filter,
                            filter_length=filter_length,
                            border_mode='valid',
                            activation='relu',
                            subsample_length=1))
    model.add(LSTM(nhid, dropout_W=0.2, dropout_U=0.2))

    model.add(Dense(hidden_dims))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print "-------- starting training of the model"
    batch_size       =  64
    nb_epoch         =  7
    history = model.fit(Xtrain, Ytrain, batch_size=batch_size, nb_epoch=nb_epoch, validation_split=0.3)

    plt.figure(1)
    plt.subplot(1,2,1)
    plt.plot(range(1,nb_epoch + 1), history.history['loss'], 'b', range(1,nb_epoch + 1), history.history['val_loss'], 'r')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')

    plt.subplot(1,2,2)
    plt.plot(range(1,nb_epoch + 1), history.history['acc'], 'b', range(1,nb_epoch + 1), history.history['val_acc'], 'r')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
