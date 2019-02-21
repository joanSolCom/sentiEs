import pymysql.cursors
import machineLearning.utils as utils
import numpy as np
from scipy.spatial.distance import cdist
import os
from sklearn import preprocessing
import pymysql.cursors
import pymysql

class SQLEmbeddings:

	def __init__(self, dbname="Embeddings"):
		self.db = pymysql.connect(host='localhost', user='root', password='pany8491', db=dbname, charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)

	def getWordVector(self, word, tableName ="joseembeddings" ,nDims = 400):
		cursor = self.db.cursor()
		dimList = []
		for i in range(1,nDims+1):
			dimList.append("dim"+str(i))

		dims = ",".join(dimList)
		strQuery = "SELECT "+dims+" FROM "+tableName + " WHERE word='"+word+"'"
		try:
			cursor.execute(strQuery)
		except:
			return None
		vector = []

		if cursor.rowcount > 0:
			results = cursor.fetchone()
			for i in range(1,nDims+1):
				strDim = "dim"+str(i)
				dim = results[strDim]
				vector.append(float(dim))
		else:
			#vector = np.zeros(nDims)
			vector = None

		return vector

	def getMsgVector(self, msg, tableName ="joseembeddings" ,nDims = 400, lang="es"):
		cleanMsg = utils.clean_text(msg)
		vectors = []

		for token in cleanMsg:
			vector = self.getWordVector(token,tableName,nDims)
			if vector is not None:
				vectors.append(vector)

		if vectors:
			avgVector = np.mean(vectors,axis=0)
			return avgVector.tolist()
		else:
			return np.zeros(nDims)

	def getWeightedMsgVector(self, msg, dictSeed, tableName ="joseembeddings" ,nDims = 400):
		categoriesSupported = os.listdir("./dictSeeds/")
		if dictSeed not in categoriesSupported:
			return self.getMsgVector(msg)

		cleanMsg = utils.clean_text(msg)
		vectors = []
		dictSeedTokens = open("./dictSeeds/"+dictSeed).read().split("\n")

		i = 0
		while i<len(dictSeedTokens):
			dictSeedTokens[i] = dictSeedTokens[i].strip()
			i+=1

		weights = []
		for token in cleanMsg:
			vector = self.getWordVector(token,tableName,nDims)
			vectors.append(vector)
			weight = 1
			if token in dictSeedTokens:
				idx = dictSeedTokens.index(token)
				weight = 4
				#dictSeedTokens.pop(idx)

			weights.append(weight)

		'''
		for elem in dictSeedTokens:
			vector = self.getWordVector(elem,tableName,nDims)
			vectors.append(vector)
			weights.append(0.2)
		'''

		avgVector = np.average(vectors,axis=0,weights=weights)
		return avgVector.tolist()

	def aggregateVectors(self, A, B):
		C = []
		C.append(A)
		C.append(B)
		C = np.mean(C,axis=0)
		return C.tolist()

	def getNormVector(self, vector):
		return np.linalg.norm(vector)

	def distance(self, A, B, distance = "cosine"):
		return cdist([A],[B],distance)
