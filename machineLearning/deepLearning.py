import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, GRU, BatchNormalization, Bidirectional, Embedding, LSTM, Dropout, MaxPooling1D,SpatialDropout1D, Flatten, Conv1D, TimeDistributed
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras import regularizers
from attention import AttentionWeightedAverage
from keras.regularizers import L1L2
from keras import optimizers
from keras.callbacks import EarlyStopping

import re
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class DeepLearning():

    def __init__(self, pathCharEmbed, pathWordEmbed, dataPath):
        self.pathCharEmbed = pathCharEmbed
        self.pathWordEmbed = pathWordEmbed
        self.dataPath = dataPath
        self.embed_dim = 400
        self.preprocess(dataPath)
        self.get_embedding_index(pathWordEmbed)
        self.get_embedding_matrix()
        self.buildModel()

    def load_char_embed(self, path):
        char_embed = {}
        raw = open(path,"r").read().split("\n")
        for line in raw:
            idx = line.split()[0]
            vectorStr = line.split()[1:]
            vector = []
            for elem in vectorStr:
                vector.append(float(elem))
            
            char_embed[idx] = vector

        return char_embed

    def get_embedding_index(self, path):

        embeddings_index = dict()
        f = open(path,"r")
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')

            embeddings_index[word] = coefs
        f.close()

        self.embeddings_index = embeddings_index


    def preprocess(self, path):
        fd = open(path,"r")
        lines = fd.read().strip().split("\n")[1:]
        dictLabels = {}
        for line in lines:
            label = line.split("\t")[1].strip()
            if label not in dictLabels:
                dictLabels[label] = 0
            dictLabels[label] += 1

        fd.close()
        print(dictLabels)

        self.data = pd.read_csv(path, sep='\t')
        self.data['text'] = self.data['text'].apply(lambda x: str(x))

        tokenizer = Tokenizer(split=' ')
        tokenizer.fit_on_texts(self.data['text'].values)
        X = tokenizer.texts_to_sequences(self.data['text'].values)
        self.X = pad_sequences(X)
        self.word_index = tokenizer.word_index

    def get_embedding_matrix(self):
        self.embedding_matrix = np.zeros((len(self.word_index) + 1, self.embed_dim))
        self.char_embed = self.load_char_embed(self.pathCharEmbed)

        for word, i in self.word_index.items():
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                self.embedding_matrix[i] = embedding_vector
            else:
                embedding_vector = []
                for char in word:
                    if char in self.char_embed:
                        embedding_vector.append(self.char_embed[char])
                
                if len(embedding_vector) > 10:
                    self.embedding_matrix[i] = np.sum(embedding_vector,axis=0)

    def buildModel(self):
        batch_size = 64
        validation_size = 500
        epochs = 5
        units_out = 128
        dropout = 0.5
        recurrent_dropout = 0.5

        embedding_pretrained = Embedding(len(self.word_index) + 1, self.embed_dim, weights=[self.embedding_matrix], trainable=False)
        embedding_pretrained_trainable = Embedding(len(self.word_index) + 1, self.embed_dim, weights=[self.embedding_matrix], trainable=True)
        embedding_to_train = Embedding(len(self.word_index) + 1, self.embed_dim, trainable=True)


        model = Sequential()
        model.add(embedding_to_train)
        model.add(Bidirectional(LSTM(units_out, dropout=dropout, recurrent_dropout=recurrent_dropout)))
        model.add(Dense(3,activation='softmax'))

        model.compile(loss = 'categorical_crossentropy', optimizer="adam",metrics = ['acc'])
        print(model.summary())

        self.Y = pd.get_dummies(self.data['sentiment']).values
        X_train, X_test, Y_train, Y_test = train_test_split(self.X,self.Y, test_size = 0.15, random_state = 42)

        X_validate = X_test[-validation_size:]
        Y_validate = Y_test[-validation_size:]
        X_test = X_test[:-validation_size]
        Y_test = Y_test[:-validation_size]

        earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, mode='min')
        callbacks_list = [earlystop]

        history = model.fit(X_train, Y_train, validation_data=(X_validate, Y_validate) ,epochs = epochs, batch_size=batch_size, verbose = 2, shuffle=True, callbacks=callbacks_list)
        self.plot_history(history)
        score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
        print("score: %.2f" % (score))
        print("acc: %.2f" % (acc))



    def plot_history(self, history):
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        x = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(x, acc, 'b', label='Training acc')
        plt.plot(x, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(x, loss, 'b', label='Training loss')
        plt.plot(x, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()


if __name__ == '__main__':
    pathCharEmbed = "/home/joan/Escritorio/Datasets/TASS data/machineLearning/spanish_w2v_embeddings-char.txt"
    pathWordEmbed = "/home/joan/Escritorio/Datasets/TASS data/machineLearning/spanish_w2v_embeddings.txt"
    dataPath = "/home/joan/Escritorio/Datasets/TASS data/extracted_merged.tsv"
    iDL = DeepLearning(pathCharEmbed, pathWordEmbed, dataPath)