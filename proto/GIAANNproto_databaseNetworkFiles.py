"""GIAANNproto_databaseNetworkFiles.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_main.py

# Usage:
see GIAANNproto_main.py

# Description:
GIA ANN proto database Network Files

"""

import torch as pt
import pickle
import os

from GIAANNproto_globalDefs import *


def initialiseDatabaseFiles():
	os.makedirs(observedColumnsDir, exist_ok=True)

def pathExists(pathName):
	if(os.path.exists(pathName)):
		return True
	else:
		return False
	
def loadDictFile(dictFileName):
	with open(dictFileName, 'rb') as fIn:
		dictionary = pickle.load(fIn)
	return dictionary

def loadFeatureNeuronsGlobalFile():
	globalFeatureNeurons = loadTensor(databaseFolder, globalFeatureNeuronsFile)
	return globalFeatureNeurons
	
def saveData(databaseNetworkObject, observedColumnsDict):
	# Save observed columns to disk
	for observedColumn in observedColumnsDict.values():
		observedColumn.saveToDisk()

	# Save global feature neuron arrays if not lowMem
	if not lowMem:
		if(performRedundantCoalesce):
			databaseNetworkObject.globalFeatureNeurons = databaseNetworkObject.globalFeatureNeurons.coalesce()
		saveTensor(databaseNetworkObject.globalFeatureNeurons, databaseFolder, globalFeatureNeuronsFile)

	# Save concept columns dictionary to disk
	with open(conceptColumnsDictFile, 'wb') as fOut:
		pickle.dump(databaseNetworkObject.conceptColumnsDict, fOut)

	# Save concept features dictionary to disk
	with open(conceptFeaturesDictFile, 'wb') as fOut:
		pickle.dump(databaseNetworkObject.conceptFeaturesDict, fOut)


def observedColumnSaveToDisk(self):
	"""
	Save the observed column data to disk.
	"""
	data = {
		'conceptIndex': self.conceptIndex,
		'featureWordToIndex': self.featureWordToIndex,
		'featureIndexToWord': self.featureIndexToWord,
		'nextFeatureIndex': self.nextFeatureIndex
	}
	# Save the data dictionary using pickle
	with open(os.path.join(observedColumnsDir, f"{self.conceptIndex}_data.pkl"), 'wb') as f:
		pickle.dump(data, f)
	# Save the tensors using pt.save
	if(performRedundantCoalesce):
		self.featureConnections = self.featureConnections.coalesce()
		print("self.featureConnections = ", self.featureConnections)
	saveTensor(self.featureConnections, observedColumnsDir, f"{self.conceptIndex}_featureConnections")
	if lowMem:
		if(performRedundantCoalesce):
			self.featureNeurons = self.featureNeurons.coalesce()
			print("self.featureNeurons = ", self.featureNeurons)
		saveTensor(self.featureNeurons, observedColumnsDir, f"{self.conceptIndex}_featureNeurons")

def observedColumnLoadFromDisk(cls, databaseNetworkObject, conceptIndex, lemma, i):
	"""
	Load the observed column data from disk.
	"""
	# Load the data dictionary
	with open(os.path.join(observedColumnsDir, f"{conceptIndex}_data.pkl"), 'rb') as f:
		data = pickle.load(f)
	instance = cls(databaseNetworkObject, conceptIndex, lemma, i)
	instance.featureWordToIndex = data['featureWordToIndex']
	instance.featureIndexToWord = data['featureIndexToWord']
	instance.nextFeatureIndex = data['nextFeatureIndex']
	# Load the tensors
	instance.featureConnections = loadTensor(observedColumnsDir, f"{conceptIndex}_featureConnections")
	if lowMem:
		instance.featureNeurons = loadTensor(observedColumnsDir, f"{conceptIndex}_featureNeurons")
	return instance

def saveTensor(tensor, folderName, fileName):
	pt.save(tensor, os.path.join(folderName, fileName+pytorchTensorFileExtension))

def loadTensor(folderName, fileName):
	tensor = pt.load(os.path.join(folderName, fileName+pytorchTensorFileExtension))	#does not work: , map_location=deviceSparse
	tensor = tensor.to(deviceSparse)
	return tensor
