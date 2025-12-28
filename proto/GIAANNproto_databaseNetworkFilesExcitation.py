"""GIAANNproto_databaseNetworkFilesExcitation.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_main.py

# Usage:
see GIAANNproto_main.py

# Description:
GIA ANN proto database Network Files Excitation

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

def saveDictFile(dictFileName, dictObject):
	# Save dictionary to disk
	with open(dictFileName, 'wb') as fOut:
		pickle.dump(dictObject, fOut)

def loadListFile(listFileName):
	with open(listFileName, 'rb') as fIn:
		listObject = pickle.load(fIn)
	return listObject

def saveListFile(listFileName, listObject):
	# Save list to disk
	with open(listFileName, 'wb') as fOut:
		pickle.dump(listObject, fOut)

def loadFeatureNeuronsGlobalFile():
	globalFeatureNeurons = loadTensor(databaseFolder, globalFeatureNeuronsFile)
	globalFeatureNeurons = adjustPropertyDimensions(globalFeatureNeurons, "globalFeatureNeurons")
	return globalFeatureNeurons

def adjustPropertyDimensions(tensor, tensorName):
	propertyCount = tensor.shape[0]
	if(propertyCount == arrayNumberOfProperties):
		return tensor
	if(arrayIndexPropertiesActivationCreate and not arrayIndexPropertiesActivation and propertyCount == arrayNumberOfProperties - 1):
		return insertPropertyDimension(tensor, arrayIndexPropertiesActivationIndex, arrayNumberOfProperties)
	raise RuntimeError(f"{tensorName} property dimension mismatch: expected {arrayNumberOfProperties}, got {propertyCount}")

def insertPropertyDimension(tensor, insertIndex, targetPropertyCount):
	if(tensor.is_sparse):
		tensor = tensor.coalesce()
		indices = tensor.indices()
		values = tensor.values()
		shiftMask = indices[0] >= insertIndex
		if(shiftMask.any()):
			indices = indices.clone()
			indices[0, shiftMask] += 1
		newSize = list(tensor.size())
		newSize[0] = targetPropertyCount
		return pt.sparse_coo_tensor(indices, values, size=newSize, dtype=tensor.dtype, device=tensor.device).coalesce()
	zerosShape = list(tensor.shape)
	zerosShape[0] = 1
	zerosTensor = pt.zeros(zerosShape, dtype=tensor.dtype, device=tensor.device)
	return pt.cat([tensor[:insertIndex], zerosTensor, tensor[insertIndex:]], dim=0)
	
def saveData(databaseNetworkObject, observedColumnsDict):
	# Save observed columns to disk
	for observedColumn in observedColumnsDict.values():
		observedColumn.saveToDisk()

	# Save global feature neuron arrays if not lowMem
	if not lowMem:
		if(performRedundantCoalesce):
			databaseNetworkObject.globalFeatureNeurons = databaseNetworkObject.globalFeatureNeurons.coalesce()
		saveTensor(databaseNetworkObject.globalFeatureNeurons, databaseFolder, globalFeatureNeuronsFile)

	saveDictFile(conceptColumnsDictFile, databaseNetworkObject.conceptColumnsDict)
	saveDictFile(conceptFeaturesDictFile, databaseNetworkObject.conceptFeaturesDict)

	if(conceptColumnsDelimitByPOS):
		if(detectReferenceSetDelimitersBetweenNouns):
			conceptFeaturesReferenceSetDelimiterProbabilisticDict = dict(enumerate(databaseNetworkObject.conceptFeaturesReferenceSetDelimiterProbabilisticList))
			saveDictFile(conceptFeaturesReferenceSetDelimiterProbabilisticListFile, conceptFeaturesReferenceSetDelimiterProbabilisticDict)
			conceptFeaturesReferenceSetDelimiterDeterministicDict = dict(enumerate(databaseNetworkObject.conceptFeaturesReferenceSetDelimiterDeterministicList))
			saveDictFile(conceptFeaturesReferenceSetDelimiterDeterministicListFile, conceptFeaturesReferenceSetDelimiterDeterministicDict)
		else:
			conceptFeaturesReferenceSetDelimiterDict = dict(enumerate(databaseNetworkObject.conceptFeaturesReferenceSetDelimiterList))
			saveDictFile(conceptFeaturesReferenceSetDelimiterListFile, conceptFeaturesReferenceSetDelimiterDict)
		
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
	instance.featureConnections = adjustPropertyDimensions(loadTensor(observedColumnsDir, f"{conceptIndex}_featureConnections"), f"observedColumn.featureConnections[{conceptIndex}]")
	if lowMem:
		instance.featureNeurons = adjustPropertyDimensions(loadTensor(observedColumnsDir, f"{conceptIndex}_featureNeurons"), f"observedColumn.featureNeurons[{conceptIndex}]")
	return instance

def saveTensor(tensor, folderName, fileName):
	pt.save(tensor, os.path.join(folderName, fileName+pytorchTensorFileExtension))

def loadTensor(folderName, fileName):
	tensor = pt.load(os.path.join(folderName, fileName+pytorchTensorFileExtension))	#does not work: , map_location=deviceSparse
	tensor = tensor.to(deviceSparse)
	return tensor
