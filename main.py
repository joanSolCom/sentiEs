# -*- coding: utf-8 -*-

from machineLearning.classify import SupervisedLearning
from machineLearning.sqlEmbeddings import SQLEmbeddings
import codecs

pathIn = "extracted_general_train.tsv"

raw = codecs.open(pathIn,"r").read().split("\n")
iSQL = SQLEmbeddings()

X = []
Y = []

for row in raw:
    pieces = row.split("\t")
    tweet = pieces[0]

    vector = iSQL.getMsgVector(tweet)
    try:
        label = pieces[1]
        if label not in ["N","P","NEU"]:
            continue
    except IndexError:
        print("no label in tweet",row)
        continue
        
    X.append(vector)
    Y.append(label)


iSup = SupervisedLearning()
iSup.cross_validation(X,Y)
#iSup.getOptimalParams(X,Y)