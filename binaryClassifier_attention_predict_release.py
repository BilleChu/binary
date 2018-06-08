#! /usr/bin/python
#coding:utf-8

import numpy as np
import json
import sys
import re
import os
import time
import jieba
import gc

from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from sklearn.metrics import classification_report
import keras.backend as K
import tensorflow as tf
sDay = "00000000"
gc.disable()

class binaryClassifier(object):

    def __init__(self):
        print("Initalizing...")
        self.srcfile = "/data/ceph/query/binary/data/predict/ds=" + sDay  
        self.modelfile = "/data/ceph/query/binary/model/combine_model.hdf5"
        self.charfile = "/data/ceph/query/binary/data/charDict"
        self.wordfile = "/data/ceph/query/binary/data/wordDict"
        self.filename = "/data/ceph/query/binary/data/predict/src/src_" + sDay 
        self.highBelief_posfile = "/data/ceph/query/binary/data/predict/result/preds_highBelief_pos_" + sDay
        self.negfile = "/data/ceph/query/binary/data/predict/result/pred_neg_" + sDay
 
        self.word_features_dim = 50
        self.char_features_dim = 100
        self.char_max_length = 15
	self.batch_size = 1024
        self.word_max_length = 10
        self.maxCharsNum = 100

    def unzip(self):
        srcPath = self.srcfile + "/*.gz"
        cmd = "gunzip " + srcPath
        os.system(cmd)
        time.sleep(60)
        cmd = "cat " + self.srcfile + "/* > " + self.filename 
        return os.system(cmd)

    def preprocessing(self):
        print("Processing... & begin to predict...")
        self.corpus = []
        self.tokens = []
        self.queryList = []
        self.empty = "".decode('utf-8')
        with open(self.filename, "r") as f:
            i = 0
            while True:
                line = f.readline()
                if not line:
                    break
                query = line.strip().decode('utf-8', 'ignore')
                charList = []
                wordList = []
                for char in query:
                    if (char in self.dictionary):
                	charList.append(self.dictionary[char])
                for token in jieba.cut(query.strip()):
                    if (token in self.dictionary_word):
                        wordList.append(self.dictionary_word[token])
                if (len(wordList) != 0):
                    self.corpus.append(charList)
                    self.queryList.append(line)
                    self.tokens.append(wordList)
                i += 1
                if i% self.batch_size == 0:
                    self.pad_data()
                    self.predict() 
                    del self.corpus[:]
                    del self.tokens[:]
                    del self.queryList[:]

    def loadDict(self):
        with open(self.charfile, "r") as f:
            self.dictionary = json.load(f)
        with open(self.wordfile, "r") as f:
            self.dictionary_word = json.load(f)

    def pad_data(self):
        self.sequences = pad_sequences(sequences=self.corpus, maxlen=self.char_max_length)
        self.sequences_word = pad_sequences(sequences=self.tokens, maxlen=self.word_max_length)

    def __load_model__(self):
        print("load_model...")
        self.model = load_model(self.modelfile)

    def predict(self):
        fhp = open(self.highBelief_posfile, "a")
        fn = open(self.negfile, "a")
        output = self.model.predict([self.sequences, self.sequences_word], batch_size= self.batch_size)
        for index, value in enumerate(output):
            if (value > 0.9):
                fhp.write(self.queryList[index])
            else:
                fn.write(self.queryList[index])
        fhp.close()
        fn.close()

    def call(self):
        if (self.unzip() == 0):
            time.sleep(60)
            self.loadDict()
            self.__load_model__()
            self.preprocessing()
            print("predict finished")

if __name__ == "__main__":
        sDay = sys.argv[1]
        binaryClf = binaryClassifier()
        binaryClf.batch_size = int(sys.argv[2])
        binaryClf.call()
