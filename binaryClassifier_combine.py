#! /usr/bin/python
#coding:utf-8

import numpy as np
import sys
import jieba
import json
import random
import re

from gensim.models import Word2Vec, KeyedVectors
from gensim import corpora
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Bidirectional, LSTM, Embedding, Dense, Activation, BatchNormalization, TimeDistributed, Flatten, merge, RepeatVector, Permute, Lambda, Dropout
from keras.models import Model
from keras.callbacks import Callback, LearningRateScheduler, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import keras.backend as K
import tensorflow as tf

class StaticHistory(Callback):

    def __init__(self, test_data, test_label):
        self.logfile = "/data/ceph/query/binary/scripts/train/logs/binaryLogs"
        self.test_data = test_data
        self.test_label = test_label
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accs = []
        self.lr = []
    def on_epoch_end(self, epoch, logs={}):
        output = self.model.predict(self.test_data, batch_size=128)
        preds = [round(value) for value in output]
        with open(self.logfile, "a") as fwrite:
            fwrite.write("epoch  -->" + str(epoch) + "\n")
            fwrite.write("loss   -->" + str(logs.get("loss")) + "\n")
            fwrite.write("acc    -->" + str(logs.get("acc"))+ "\n")
            fwrite.write(classification_report(self.test_label, preds))

class binaryClassifier(object):

    def __init__(self):
        print("Initalizing...")
        self.posFile = "/data/ceph/query/binary/data/train/positive_test"
        self.negFile = "/data/ceph/query/binary/data/train/negative_test"
        self.savefile = "/data/ceph/query/binary/model/combine_model.hdf5"
        self.char_dictfile = "/data/ceph/query/binary/data/charDict"
        self.word_dictfile = "/data/ceph/query/binary/data/wordDict"
        self.word_features_dim = 50
        self.char_features_dim = 100
        self.char_max_length = 15
        self.word_max_length = 10
        self.char_hidden_dims = 100
        self.word_hidden_dims = 100

    def step_decay(self, epoch):
        if epoch != 0:
            lr = K.get_value(self.model.optimizer.lr)
            K.set_value(self.model.optimizer.lr, lr * 0.5)
            print("lr changed to {}".format(lr * 0.5))
        return K.get_value(self.model.optimizer.lr)

    def filter(self, line):
        newline = re.sub("[0123456789\s.\!\/_,$%^*(+\"\']+", "", line)
        return newline

    def posPreprocessing(self):
        self.posDocs = []
        self.posWords = []
        with open(self.posFile, "r") as f:
            lines = f.readlines()
            for line in lines:
                contents = line.split("\t")
                query = contents[0].decode('utf-8', 'ignore')
                charList = []
                wordList = []
                for index in range(len(query)):
                    charList.append(query[index])
                for word in jieba.cut(self.filter(contents[0])):
                    wordList.append(word)
                self.posWords.append(wordList)
                self.posDocs.append(charList)
        self.poslabel = np.ones(shape=(1, len(self.posDocs)))

    def negPreprocessing(self):
        self.negDocs = []  # as above
        self.negWords = []
        with open(self.negFile, "r") as f:
            lines = f.readlines()
            for line in lines:
                charList = []
                wordList = []
                query = line.strip().decode('utf-8','ignore')
                for index in range(len(query)):
                    charList.append(query[index])
                for word in jieba.cut(self.filter(line.strip())):
                    wordList.append(word)
                self.negDocs.append(charList)
                self.negWords.append(wordList)
        self.neglabel = np.zeros(shape=(1, len(self.negDocs)))

    def mergeLabels(self):
        Labels_np = np.hstack([self.poslabel,self.neglabel])
        self.labels = (Labels_np.reshape(Labels_np.shape[1])).tolist()

    def word2vec(self):
        texts = self.posDocs + self.negDocs
        tokens_texts = self.posWords + self.negWords

        self.w2v_model_char = Word2Vec(texts, self.char_features_dim, min_count=0).wv
        self.w2v_model_word = Word2Vec(tokens_texts, self.word_features_dim, min_count=0).wv

    def getDict(self):
        words_char = self.w2v_model_char.index2word
        self.dictionary_char = {}
        for index, word in enumerate(words_char):
            self.dictionary_char[word] = index

        tokens = self.w2v_model_word.index2word
        self.dictionary_word = {}
        for index, word in enumerate(tokens):
            self.dictionary_word[word] = index


    def saveDict(self):
        with open(self.char_dictfile, "w") as f:
            json.dump(self.dictionary_char, f)
        with open(self.word_dictfile, "w") as f:
            json.dump(self.dictionary_word, f)

    def replaceWordbyID(self):
        self.docs = self.posDocs + self.negDocs
        self.words = self.posWords + self.negWords
        self.chars = []
        self.tokens = []
        for text in self.docs:
            new_text = []
            for word in text:
                wordId = self.dictionary_char[word]
                new_text.append(wordId)
            self.chars.append(new_text)
        for text in self.words:
            new_text = []
            for word in text:
                wordId = self.dictionary_word[word]
                new_text.append(wordId)
            self.tokens.append(new_text)

        del self.docs
        del self.posDocs
        del self.negDocs
        del self.words
        del self.posWords
        del self.negWords

    def splitData(self):

        char_sequences = pad_sequences(sequences=self.chars, maxlen=self.char_max_length)
        word_sequences = pad_sequences(sequences=self.tokens, maxlen=self.word_max_length)
        data = np.hstack((char_sequences, word_sequences))
        self.train_data,self.test_data,self.train_label,self.test_label = train_test_split(data, self.labels, test_size= 0.2)
        self.train_chars = self.train_data[:, :self.char_max_length]
        self.train_words = self.train_data[:, self.char_max_length:]
        self.test_chars = self.test_data[:, :self.char_max_length]
        self.test_words = self.test_data[:, self.char_max_length:]
        del self.labels
        del self.chars
        del self.tokens
        del self.train_data
        del self.test_data

    def build_network(self):
        print("Start to build the DL model")
        ''' chars model '''
        embedder_chars = Embedding(input_dim=len(self.w2v_model_char.index2word),
                              output_dim=self.char_features_dim,
                              weights=[self.w2v_model_char.syn0],
                              trainable=True)
        char_input = Input(shape=(self.char_max_length,), dtype='int32')
        embedded_chars = embedder_chars(char_input)

        x = Bidirectional(LSTM(self.char_hidden_dims,
                               dropout_W=0.2,
                               dropout_U=0.2,
                               return_sequences=True),
                               merge_mode='concat')(embedded_chars)

        attention = TimeDistributed(Dense(1, activation='tanh'))(x)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(self.char_hidden_dims * 2)(attention)  # biDirection
        attention = Permute((2, 1))(attention)

        x = merge([x, attention], mode='mul')
        x = Lambda(lambda xx: K.sum(xx, axis=1))(x)
        x = Dense(self.char_hidden_dims, activation="sigmoid")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)

        ''' word model '''
        embedder_words = Embedding(input_dim=len(self.w2v_model_word.index2word),
                              output_dim=self.word_features_dim,
                              weights=[self.w2v_model_word.syn0],
                              trainable=True)
        word_input = Input(shape=(self.word_max_length,), dtype='int32')
        embedded_words = embedder_words(word_input)

        y = Bidirectional(LSTM(self.word_hidden_dims,
                               dropout_W=0.2,
                               dropout_U=0.2,
                               return_sequences=True),
                               merge_mode='concat')(embedded_words)

        attention = TimeDistributed(Dense(1, activation='tanh'))(y)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(self.word_hidden_dims * 2)(attention)  # biDirection
        attention = Permute((2, 1))(attention)

        y = merge([y, attention], mode='mul')
        y = Lambda(lambda xx: K.sum(xx, axis=1))(y)
        y = Dense(self.word_hidden_dims, activation="sigmoid")(y)
        y = BatchNormalization()(y)
        y = Dropout(0.5)(y)

        merged = merge([x, y], mode='concat', concat_axis=1)
        output = Dense(1, activation='sigmoid', name='binary')(merged)
        self.model = Model(inputs=[char_input, word_input], outputs=output)
        self.model.compile(optimizer="Adadelta",
                           loss="binary_crossentropy",
                           metrics=["accuracy"])
        print(self.model.summary())
        print("Get the model build work Done!")

    def train(self, num_epochs):
        self.posPreprocessing()
        self.negPreprocessing()
        self.mergeLabels()
        self.word2vec()
        self.getDict()
        self.saveDict()
        self.replaceWordbyID()
        self.splitData()
        self.build_network()
        static_history = StaticHistory([self.test_chars, self.test_words], self.test_label)
        lrate = LearningRateScheduler(self.step_decay)
        callback_list = [static_history, lrate]
        self.model.fit([self.train_chars, self.train_words],
             	       self.train_label,
                       batch_size=128,
                       epochs=num_epochs,
                       callbacks=callback_list,
                       verbose=1)
        self.model.save(self.savefile)
        del self.model

if __name__ == "__main__":

    num_epochs = int(sys.argv[1])
    binaryClf = binaryClassifier()
    binaryClf.train(num_epochs)
