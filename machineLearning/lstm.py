import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, GRU, BatchNormalization, Bidirectional, Embedding, LSTM, Dropout, MaxPooling1D,SpatialDropout1D, Flatten, Conv1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras import regularizers
from attention import AttentionWeightedAverage
from keras.regularizers import L1L2
from keras import optimizers
from keras.callbacks import EarlyStopping

import re

#########################
##### PREPROCESSING #####
#########################

data = pd.read_csv("../extracted_general_train.tsv", sep='\t')
data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

print(data[ data['sentiment'] == 'P'].size)
print(data[ data['sentiment'] == 'N'].size)
print(data[ data['sentiment'] == 'NEU'].size)

for idx,row in data.iterrows():
    row[0] = row[0].replace('rt',' ')

tokenizer = Tokenizer(split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X)

word_index = tokenizer.word_index

#################################
### CREATING EMBEDDING MATRIX ###
#################################
embed_dim = 400

embeddings_index = dict()
f = open('/home/joan/Escritorio/spanish_w2v_embeddings.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_matrix = np.zeros((len(word_index) + 1, embed_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
    else:
    	#CHAR EMBEDDINGS
    	pass

''' 

#BRAVO CNN

model = Sequential()
model.add(embedding_layer)
model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D(5))
model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D(5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['acc'])

'''

'''
BRAVO LSTM 1

model = Sequential()
model.add(embedding_layer)
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
#model.add(Dense(2, activation='softmax'))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())
# try using different optimizers and different optimizer configs
model.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['accuracy'])

BRAVO ATTENTION

d = 0.5
rd = 0.5
rnn_units = 30
model = Sequential()
model.add(embedding_layer)
model.add(SpatialDropout1D(d))
model.add(Bidirectional(LSTM(units=rnn_units, return_sequences=True, recurrent_regularizer=L1L2(l1=0.01, l2=0.01))))#, dropout=d, recurrent_dropout=rd)))
model.add(AttentionWeightedAverage())
model.add(Dropout(d))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy', f1])

MIO


model = Sequential()
model.add(Embedding(len(word_index) + 1, embed_dim, weights=[embedding_matrix], trainable=True))
model.add(SpatialDropout1D(0.6))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(3,activation='softmax',kernel_regularizer=regularizers.l2(0.0001)))

model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())


'''
units_out = 128

opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
embedding_layer = Embedding(len(word_index) + 1, embed_dim, weights=[embedding_matrix], trainable=True)
epochs = 20

learning_rate = 0.1
decay_rate = learning_rate / epochs
momentum = 0.8
sgd = optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)


lstm_layer = Bidirectional(LSTM(units_out, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False))
gru_layer = GRU(units_out, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False, reset_after=False)

model = Sequential()
model.add(embedding_layer)
model.add(lstm_layer)
model.add(Dropout(0.6))
model.add(Dense(3, activation='softmax',kernel_regularizer=regularizers.l2(0.00001)))
model.compile(optimizer=sgd,loss='categorical_crossentropy', metrics=['accuracy'])


print(model.summary())

Y = pd.get_dummies(data['sentiment']).values
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.15, random_state = 42)

batch_size = 128

validation_size = 300
X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]


earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, mode='min')

callbacks_list = [earlystop]

model.fit(X_train, Y_train, validation_data=(X_validate, Y_validate) ,epochs = epochs, batch_size=batch_size, verbose = 2, shuffle=True, callbacks=callbacks_list)

score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))


'''
from keras.models import load_model

# Creates a HDF5 file 'my_model.h5'
model.save('sentiment.h5')

# Deletes the existing model
del model  

# Returns a compiled model identical to the previous one
model = load_model('sentiment.h5')
'''
