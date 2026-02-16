"""GIAANNproto_databaseNetworkFilesExcitation.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 Baxter AI (baxterai.com)

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
import time

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
	globalFeatureNeurons = adjustBranchDimensions(globalFeatureNeurons, "globalFeatureNeurons", expectedRank=5)
	return globalFeatureNeurons

def adjustPropertyDimensions(tensor, tensorName):
	propertyCount = tensor.shape[0]
	result = tensor
	if(debugWorkaroundPreviousUngatedShutdownSaveBug):
		legacyTimePropertyIndex = None
		if(arrayIndexPropertiesStrength):
			legacyTimePropertyIndex = 1
		else:
			legacyTimePropertyIndex = 0
		if(arrayIndexPropertiesPermanence):
			legacyTimePropertyIndex += 1
		if(arrayIndexPropertiesActivationCreate):
			legacyTimePropertyIndex += 1
	if(propertyCount == arrayNumberOfProperties):
		result = tensor
	elif(arrayIndexPropertiesActivationCreate and arrayIndexPropertiesActivationIndex is not None and propertyCount == arrayIndexPropertiesActivationIndex and propertyCount < arrayNumberOfProperties):
		result = insertPropertyDimension(tensor, arrayIndexPropertiesActivationIndex, arrayNumberOfProperties)
	elif(arrayIndexPropertiesTimeCreate and arrayIndexPropertiesTimeIndex is not None and propertyCount == arrayIndexPropertiesTimeIndex and propertyCount < arrayNumberOfProperties):
		result = insertPropertyDimension(tensor, arrayIndexPropertiesTimeIndex, arrayNumberOfProperties)
	elif(arrayIndexPropertiesActivationCreate and not arrayIndexPropertiesActivation and propertyCount == arrayNumberOfProperties - 1):
		result = insertPropertyDimension(tensor, arrayIndexPropertiesActivationIndex, arrayNumberOfProperties)
	elif(debugWorkaroundPreviousUngatedShutdownSaveBug and (not arrayIndexPropertiesTimeCreate) and (legacyTimePropertyIndex is not None) and propertyCount == (arrayNumberOfProperties + 1)):
		result = removePropertyDimension(tensor, legacyTimePropertyIndex, arrayNumberOfProperties, tensorName)
	else:
		raise RuntimeError(f"{tensorName} property dimension mismatch: expected {arrayNumberOfProperties}, got {propertyCount}")
	return result

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

if(debugWorkaroundPreviousUngatedShutdownSaveBug):
	def removePropertyDimension(tensor, removeIndex, targetPropertyCount, tensorName):
		result = tensor
		if(removeIndex < 0 or removeIndex >= tensor.shape[0]):
			raise RuntimeError(f"{tensorName} property remove index out of range: {removeIndex}")
		if(tensor.is_sparse):
			tensor = tensor.coalesce()
			indices = tensor.indices()
			values = tensor.values()
			keepMask = indices[0] != removeIndex
			filteredIndices = indices[:, keepMask]
			filteredValues = values[keepMask]
			if(filteredIndices.numel() > 0):
				filteredIndices = filteredIndices.clone()
				shiftMask = filteredIndices[0] > removeIndex
				if(shiftMask.any()):
					filteredIndices[0, shiftMask] -= 1
			newSize = list(tensor.size())
			newSize[0] = targetPropertyCount
			result = pt.sparse_coo_tensor(filteredIndices, filteredValues, size=newSize, dtype=tensor.dtype, device=tensor.device).coalesce()
		else:
			result = pt.cat([tensor[:removeIndex], tensor[removeIndex+1:]], dim=0)
		return result

def adjustBranchDimensions(tensor, tensorName, expectedRank, branchCount=numberOfDendriticBranches):
	if tensor.dim() == expectedRank:
		return expandBranchDimensions(tensor, tensorName, branchCount)
	if tensor.dim() == expectedRank - 1:
		return insertBranchDimension(tensor, tensorName, insertIndex=1, branchCount=branchCount)
	raise RuntimeError(f"{tensorName} branch dimension mismatch: expected rank {expectedRank} or {expectedRank - 1}, got {tensor.dim()}")

def insertBranchDimension(tensor, tensorName, insertIndex, branchCount):
	if tensor.is_sparse:
		tensor = tensor.coalesce()
		indices = tensor.indices()
		values = tensor.values()
		branchRow = pt.zeros((1, indices.shape[1]), dtype=indices.dtype, device=indices.device)
		newIndices = pt.cat([indices[:insertIndex], branchRow, indices[insertIndex:]], dim=0)
		newSize = list(tensor.size())
		newSize.insert(insertIndex, branchCount)
		return pt.sparse_coo_tensor(newIndices, values, size=newSize, dtype=tensor.dtype, device=tensor.device).coalesce()
	tensor = tensor.unsqueeze(insertIndex)
	return expandBranchDimensions(tensor, tensorName, branchCount)

def expandBranchDimensions(tensor, tensorName, branchCount):
	currentBranches = tensor.size(1)
	if currentBranches == branchCount:
		return tensor
	if branchCount == 1 and currentBranches > 1:
		if tensor.is_sparse:
			tensor = tensor.coalesce()
			indices = tensor.indices()
			values = tensor.values()
			if indices.shape[1] == 0:
				newSize = list(tensor.size())
				newSize[1] = 1
				return pt.sparse_coo_tensor(indices, values, size=newSize, dtype=tensor.dtype, device=tensor.device).coalesce()
			indices = indices.clone()
			indices[1] = 0
			newSize = list(tensor.size())
			newSize[1] = 1
			return pt.sparse_coo_tensor(indices, values, size=newSize, dtype=tensor.dtype, device=tensor.device).coalesce()
		return tensor.sum(dim=1, keepdim=True)
	if currentBranches < branchCount:
		if tensor.is_sparse:
			newSize = list(tensor.size())
			newSize[1] = branchCount
			return pt.sparse_coo_tensor(tensor.indices(), tensor.values(), size=newSize, dtype=tensor.dtype, device=tensor.device).coalesce()
		padShape = list(tensor.shape)
		padShape[1] = branchCount - currentBranches
		padTensor = pt.zeros(padShape, dtype=tensor.dtype, device=tensor.device)
		return pt.cat([tensor, padTensor], dim=1)
	raise RuntimeError(f"{tensorName} branch dimension mismatch: expected {branchCount}, got {currentBranches}")


def saveData(databaseNetworkObject, observedColumnsDict, sequenceCount, forceSaveGlobalState=False):
	saveDataStartTime = None
	saveObservedColumnsStartTime = None
	if(debugPrintTrainSectionTimes):
		saveDataStartTime = time.perf_counter()
	if not forceSaveGlobalState:
		# Save observed columns to disk
		if(debugPrintTrainSectionTimes):
			saveObservedColumnsStartTime = time.perf_counter()
		for observedColumn in observedColumnsDict.values():
			observedColumn.saveToDisk()
		if(debugPrintTrainSectionTimes):
			debugTrainSectionTimesAdd(databaseNetworkObject, "saveData.observedColumn.saveToDisk", time.perf_counter() - saveObservedColumnsStartTime)

	saveGlobalState = ((sequenceCount + 1) % saveGlobalFeatureNeuronsRate == 0) or forceSaveGlobalState
	if(saveGlobalState):
		# Save global feature neuron arrays if not lowMem
		if not lowMem:
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
	if(debugPrintTrainSectionTimes):
		debugTrainSectionTimesAdd(databaseNetworkObject, "saveData.total", time.perf_counter() - saveDataStartTime)
		
def observedColumnSaveToDisk(self):
	"""
	Save the observed column data to disk.
	"""
	if(trainStoreFeatureMapsGlobally):
		data = {
			'conceptIndex': self.conceptIndex,
			'nextFeatureIndex': self.nextFeatureIndex
		}
	else:
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
	saveTensor(self.featureConnections, observedColumnsDir, f"{self.conceptIndex}_featureConnections")
	if lowMem:
		saveTensor(self.featureNeurons, observedColumnsDir, f"{self.conceptIndex}_featureNeurons")

def observedColumnLoadFromDisk(cls, databaseNetworkObject, conceptIndex, lemma, i):
	"""
	Load the observed column data from disk.
	"""
	# Load the data dictionary
	with open(os.path.join(observedColumnsDir, f"{conceptIndex}_data.pkl"), 'rb') as f:
		data = pickle.load(f)
	instance = cls(databaseNetworkObject, conceptIndex, lemma, i)
	if(trainStoreFeatureMapsGlobally):
		instance.featureWordToIndex = databaseNetworkObject.conceptFeaturesDict
		instance.featureIndexToWord = databaseNetworkObject.conceptFeaturesIndexToWordDict
		instance.nextFeatureIndex = len(databaseNetworkObject.conceptFeaturesDict) - 1
	else:
		instance.featureWordToIndex = data['featureWordToIndex']
		instance.featureIndexToWord = data['featureIndexToWord']
		instance.nextFeatureIndex = data['nextFeatureIndex']
		if(debugLimitFeatures):
			instance.featureWordToIndex, instance.featureIndexToWord = applyDebugLimitFeatureIndexMaps(instance.featureWordToIndex, instance.featureIndexToWord, databaseNetworkObject.f, f"observedColumn.featureIndexMaps[{conceptIndex}]")
			if(instance.nextFeatureIndex < 0):
				raise RuntimeError("observedColumnLoadFromDisk error: nextFeatureIndex < 0")
			if(instance.nextFeatureIndex > databaseNetworkObject.f):
				instance.nextFeatureIndex = databaseNetworkObject.f
	# Load the tensors
	instance.featureConnections = adjustPropertyDimensions(loadTensor(observedColumnsDir, f"{conceptIndex}_featureConnections"), f"observedColumn.featureConnections[{conceptIndex}]")
	instance.featureConnections = adjustBranchDimensions(instance.featureConnections, f"observedColumn.featureConnections[{conceptIndex}]", expectedRank=6)
	if(debugLimitFeatures):
		instance.featureConnections = applyDebugLimitFeatureConnectionsTensor(instance.featureConnections, databaseNetworkObject.c, databaseNetworkObject.f, f"observedColumn.featureConnections[{conceptIndex}]")
	if lowMem:
		instance.featureNeurons = adjustPropertyDimensions(loadTensor(observedColumnsDir, f"{conceptIndex}_featureNeurons"), f"observedColumn.featureNeurons[{conceptIndex}]")
		instance.featureNeurons = adjustBranchDimensions(instance.featureNeurons, f"observedColumn.featureNeurons[{conceptIndex}]", expectedRank=4)
		if(debugLimitFeatures):
			instance.featureNeurons = applyDebugLimitFeatureNeuronsTensor(instance.featureNeurons, databaseNetworkObject.f, f"observedColumn.featureNeurons[{conceptIndex}]")
	return instance

def saveTensor(tensor, folderName, fileName):
	pt.save(tensor, os.path.join(folderName, fileName+pytorchTensorFileExtension))

def loadTensor(folderName, fileName):
	if(useGPUsparseStrict and not useGPUsparse):
		tensor = pt.load(os.path.join(folderName, fileName+pytorchTensorFileExtension), map_location=deviceSparse)
	else:
		tensor = pt.load(os.path.join(folderName, fileName+pytorchTensorFileExtension))
	tensor = tensor.to(deviceSparse)
	return tensor


if(debugLimitFeatures):
	def applyDebugLimitGlobalFeatureNeuronsTensor(tensor, cLimit, fLimit, tensorName):
		result = tensor
		if(debugLimitFeatures):
			if(cLimit <= 0 or fLimit <= 0):
				raise RuntimeError(f"{tensorName} debug limit requires positive limits")
			capC = tensor.size(3)
			capF = tensor.size(4)
			if(capC > cLimit):
				capC = cLimit
			if(capF > fLimit):
				capF = fLimit
			if(capC != tensor.size(3) or capF != tensor.size(4)):
				if(tensor.is_sparse):
					tensor = tensor.coalesce()
					indices = tensor.indices()
					values = tensor.values()
					mask = (indices[3] < capC) & (indices[4] < capF)
					indices = indices[:, mask]
					values = values[mask]
					newSize = list(tensor.size())
					newSize[3] = capC
					newSize[4] = capF
					result = pt.sparse_coo_tensor(indices, values, size=newSize, dtype=tensor.dtype, device=tensor.device).coalesce()
				else:
					result = tensor[:, :, :, :capC, :capF]
		return result
	def applyDebugLimitFeatureConnectionsTensor(tensor, cLimit, fLimit, tensorName):
		result = tensor
		if(debugLimitFeatures):
			if(cLimit <= 0 or fLimit <= 0):
				raise RuntimeError(f"{tensorName} debug limit requires positive limits")
			if(tensor.size(3) != tensor.size(5)):
				raise RuntimeError(f"{tensorName} feature dimension mismatch: {tensor.size(3)} vs {tensor.size(5)}")
			capF = tensor.size(3)
			capC = tensor.size(4)
			if(capF > fLimit):
				capF = fLimit
			if(capC > cLimit):
				capC = cLimit
			if(capF != tensor.size(3) or capC != tensor.size(4)):
				if(tensor.is_sparse):
					tensor = tensor.coalesce()
					indices = tensor.indices()
					values = tensor.values()
					mask = (indices[3] < capF) & (indices[4] < capC) & (indices[5] < capF)
					indices = indices[:, mask]
					values = values[mask]
					newSize = list(tensor.size())
					newSize[3] = capF
					newSize[4] = capC
					newSize[5] = capF
					result = pt.sparse_coo_tensor(indices, values, size=newSize, dtype=tensor.dtype, device=tensor.device).coalesce()
				else:
					result = tensor[:, :, :, :capF, :capC, :capF]
		return result
	def applyDebugLimitFeatureNeuronsTensor(tensor, fLimit, tensorName):
		result = tensor
		if(debugLimitFeatures):
			if(fLimit <= 0):
				raise RuntimeError(f"{tensorName} debug limit requires positive limits")
			capF = tensor.size(3)
			if(capF > fLimit):
				capF = fLimit
			if(capF != tensor.size(3)):
				if(tensor.is_sparse):
					tensor = tensor.coalesce()
					indices = tensor.indices()
					values = tensor.values()
					mask = indices[3] < capF
					indices = indices[:, mask]
					values = values[mask]
					newSize = list(tensor.size())
					newSize[3] = capF
					result = pt.sparse_coo_tensor(indices, values, size=newSize, dtype=tensor.dtype, device=tensor.device).coalesce()
				else:
					result = tensor[:, :, :, :capF]
		return result
	def applyDebugLimitFeatureIndexMaps(featureWordToIndex, featureIndexToWord, fLimit, mapName):
		resultFeatureWordToIndex = featureWordToIndex
		resultFeatureIndexToWord = featureIndexToWord
		if(debugLimitFeatures):
			if(fLimit <= 0):
				raise RuntimeError(f"{mapName} debug limit requires positive limits")
			trimmedWordToIndex = {}
			trimmedIndexToWord = {}
			for word, index in featureWordToIndex.items():
				if(index < 0):
					raise RuntimeError(f"{mapName} index < 0")
				if(index < fLimit):
					trimmedWordToIndex[word] = index
			for index, word in featureIndexToWord.items():
				if(index < 0):
					raise RuntimeError(f"{mapName} index < 0")
				if(index < fLimit):
					trimmedIndexToWord[index] = word
			for word, index in trimmedWordToIndex.items():
				if(trimmedIndexToWord.get(index) != word):
					raise RuntimeError(f"{mapName} mismatch for index {index}")
			for index, word in trimmedIndexToWord.items():
				if(trimmedWordToIndex.get(word) != index):
					raise RuntimeError(f"{mapName} mismatch for word {word}")
			resultFeatureWordToIndex = trimmedWordToIndex
			resultFeatureIndexToWord = trimmedIndexToWord
		return resultFeatureWordToIndex, resultFeatureIndexToWord
