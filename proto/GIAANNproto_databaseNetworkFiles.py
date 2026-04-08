"""GIAANNproto_databaseNetworkFiles.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 Baxter AI (baxterai.com)

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
import time
import subprocess

from GIAANNproto_globalDefs import *
import GIAANNproto_debug


def prepareDatabaseFilesStartup():
	if(os.path.isdir(observedColumnsDir)):
		if(not trainLoadExistingDatabase):
			clearDatabaseFiles()
	else:
		os.makedirs(observedColumnsDir, exist_ok=True)
	return

def clearDatabaseFiles():
	clearScriptPath = os.path.join(databaseFolder, "clear.sh")
	if(not os.path.isdir(databaseFolder)):
		raise RuntimeError("clearDatabaseFiles error: missing databaseFolder = " + databaseFolder)
	if(not os.path.isfile(clearScriptPath)):
		raise RuntimeError("clearDatabaseFiles error: missing clear.sh = " + clearScriptPath)
	clearResult = subprocess.run(["bash", clearScriptPath], cwd=databaseFolder, check=False)
	if(clearResult.returncode != 0):
		raise RuntimeError("clearDatabaseFiles error: clear.sh failed with return code " + str(clearResult.returncode))
	return

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

def getObservedColumnFolder(conceptIndex):
	result = os.path.join(observedColumnsDir, observedColumnFolderNamePrefix + str(int(conceptIndex)))
	return result

def getObservedColumnMetadataFile(conceptIndex):
	result = os.path.join(getObservedColumnFolder(conceptIndex), observedColumnMetadataFileName)
	return result

def getObservedColumnFeatureConnectionsFolder(conceptIndex):
	result = os.path.join(getObservedColumnFolder(conceptIndex), observedColumnFeatureConnectionsFolderName)
	return result

def getObservedColumnFeatureNeuronsFileBaseName():
	result = observedColumnFeatureNeuronsTensorName
	return result

def getObservedColumnSourceFeatureConnectionsFileBaseName(sourceFeatureIndex):
	result = observedColumnSourceFeatureConnectionsFileNamePrefix + str(int(sourceFeatureIndex))
	return result

def parseObservedColumnSourceFeatureConnectionsFileBaseName(fileBaseName):
	if(not fileBaseName.startswith(observedColumnSourceFeatureConnectionsFileNamePrefix)):
		raise RuntimeError(f"Invalid observed column source feature file name: {fileBaseName}")
	fileIndexText = fileBaseName[len(observedColumnSourceFeatureConnectionsFileNamePrefix):]
	if(fileIndexText == "" or not fileIndexText.isdigit()):
		raise RuntimeError(f"Invalid observed column source feature file name: {fileBaseName}")
	result = int(fileIndexText)
	return result

def getObservedColumnLegacyMetadataFile(conceptIndex):
	result = os.path.join(observedColumnsDir, str(conceptIndex) + observedColumnLegacyMetadataFileSuffix)
	return result

def getObservedColumnLegacyConnectionsFile(conceptIndex):
	result = os.path.join(observedColumnsDir, str(conceptIndex) + observedColumnLegacyFeatureConnectionsTensorNameSuffix + pytorchTensorFileExtension)
	return result

def getObservedColumnLegacyFeatureNeuronsFile(conceptIndex):
	result = os.path.join(observedColumnsDir, str(conceptIndex) + observedColumnLegacyFeatureNeuronsTensorNameSuffix + pytorchTensorFileExtension)
	return result

def ensureObservedColumnFolderExists(conceptIndex):
	columnFolder = getObservedColumnFolder(conceptIndex)
	connectionsFolder = getObservedColumnFeatureConnectionsFolder(conceptIndex)
	os.makedirs(columnFolder, exist_ok=True)
	os.makedirs(connectionsFolder, exist_ok=True)
	return

def validateObservedColumnStorageFormat(conceptIndex):
	legacyMetadataFile = getObservedColumnLegacyMetadataFile(conceptIndex)
	legacyConnectionsFile = getObservedColumnLegacyConnectionsFile(conceptIndex)
	legacyFeatureNeuronsFile = getObservedColumnLegacyFeatureNeuronsFile(conceptIndex)
	if(pathExists(legacyMetadataFile) or pathExists(legacyConnectionsFile) or pathExists(legacyFeatureNeuronsFile)):
		raise RuntimeError(f"Observed column storage format has changed for conceptIndex={conceptIndex}. Clear and rebuild the database.")
	return

def observedColumnMetadataExists(conceptIndex):
	validateObservedColumnStorageFormat(conceptIndex)
	result = pathExists(getObservedColumnMetadataFile(conceptIndex))
	return result

def listObservedColumnSourceFeatureIndices(conceptIndex):
	validateObservedColumnStorageFormat(conceptIndex)
	connectionsFolder = getObservedColumnFeatureConnectionsFolder(conceptIndex)
	indices = []
	if(pathExists(connectionsFolder)):
		for fileName in os.listdir(connectionsFolder):
			if(not fileName.endswith(pytorchTensorFileExtension)):
				continue
			fileStem = fileName[:-len(pytorchTensorFileExtension)]
			indices.append(parseObservedColumnSourceFeatureConnectionsFileBaseName(fileStem))
	indices.sort()
	return indices

def loadFeatureNeuronsGlobalFile(inferenceMode):
	globalFeatureNeurons = loadTensor(databaseFolder, globalFeatureNeuronsFile)
	globalFeatureNeurons = adjustPropertyDimensions(inferenceMode, globalFeatureNeurons, "globalFeatureNeurons")
	globalFeatureNeurons = adjustBranchDimensions(globalFeatureNeurons, "globalFeatureNeurons", expectedRank=5)
	return globalFeatureNeurons

def getTrainToInferencePropertyIndexMap():
	result = {}
	propertyIndexPairs = [
		(arrayIndexPropertiesStrengthIndexTrain, arrayIndexPropertiesStrengthIndexInference),
		(arrayIndexPropertiesPermanenceIndexTrain, arrayIndexPropertiesPermanenceIndexInference),
		(arrayIndexPropertiesActivationIndexTrain, arrayIndexPropertiesActivationIndexInference),
		(arrayIndexPropertiesTimeIndexTrain, arrayIndexPropertiesTimeIndexInference),
		(arrayIndexPropertiesPosIndexTrain, arrayIndexPropertiesPosIndexInference)
	]
	for trainPropertyIndex, inferencePropertyIndex in propertyIndexPairs:
		if(trainPropertyIndex is not None):
			if(inferencePropertyIndex is None):
				raise RuntimeError(f"getTrainToInferencePropertyIndexMap error: missing inference property index for train property index {trainPropertyIndex}")
			result[int(trainPropertyIndex)] = int(inferencePropertyIndex)
	if(len(result) != arrayNumberOfPropertiesTrain):
		raise RuntimeError(f"getTrainToInferencePropertyIndexMap error: expected {arrayNumberOfPropertiesTrain} mapped train properties, got {len(result)}")
	return result

def remapTrainPropertyDimensionsToInference(tensor, tensorName):
	result = tensor
	propertyIndexMap = None
	useIdentityExpansion = False
	if(tensor.shape[0] != arrayNumberOfPropertiesTrain):
		raise RuntimeError(f"{tensorName} train property dimension mismatch: expected {arrayNumberOfPropertiesTrain}, got {tensor.shape[0]}")
	propertyIndexMap = getTrainToInferencePropertyIndexMap()
	if(arrayIndexPropertiesEfficient):
		useIdentityExpansion = True
		for trainPropertyIndex, inferencePropertyIndex in propertyIndexMap.items():
			if(trainPropertyIndex != inferencePropertyIndex):
				useIdentityExpansion = False
	if(useIdentityExpansion):
		if(tensor.is_sparse):
			newSize = list(tensor.size())
			newSize[0] = arrayNumberOfPropertiesInference
			result = expandSparseTensorSize(tensor, newSize, tensorName)
		else:
			newSize = list(tensor.shape)
			newSize[0] = arrayNumberOfPropertiesInference
			result = pt.zeros(newSize, dtype=tensor.dtype, device=tensor.device)
			result[:tensor.shape[0]] = tensor
	else:
		if(tensor.is_sparse):
			tensor = tensor.coalesce()
			indices = tensor.indices().clone()
			values = tensor.values()
			for trainPropertyIndex, inferencePropertyIndex in propertyIndexMap.items():
				propertyMask = indices[0] == trainPropertyIndex
				if(propertyMask.any()):
					indices[0, propertyMask] = inferencePropertyIndex
			newSize = list(tensor.size())
			newSize[0] = arrayNumberOfPropertiesInference
			result = pt.sparse_coo_tensor(indices, values, size=newSize, dtype=tensor.dtype, device=tensor.device).coalesce()
		else:
			newSize = list(tensor.shape)
			newSize[0] = arrayNumberOfPropertiesInference
			result = pt.zeros(newSize, dtype=tensor.dtype, device=tensor.device)
			for trainPropertyIndex, inferencePropertyIndex in propertyIndexMap.items():
				result[inferencePropertyIndex] = tensor[trainPropertyIndex]
	return result
	
def adjustPropertyDimensions(inferenceMode, tensor, tensorName):
	propertyCount = tensor.shape[0]
	result = tensor
	if(inferenceMode):
		if(propertyCount == arrayNumberOfPropertiesInference):
			result = tensor
		elif(propertyCount == arrayNumberOfPropertiesTrain):
			result = remapTrainPropertyDimensionsToInference(tensor, tensorName)
		else:
			raise RuntimeError(f"{tensorName} property dimension mismatch: expected {arrayNumberOfPropertiesInference}, got {propertyCount}")
		'''
		#legacy branches: refactor these if i) the code changes between train and inference, or ii) arrayIndexProperties boolean flags change between train and inference.
		elif(arrayIndexPropertiesActivationCreateInference and arrayIndexPropertiesActivationIndexInference is not None and propertyCount == arrayIndexPropertiesActivationIndexInference and propertyCount < arrayNumberOfPropertiesInference):
			result = insertPropertyDimension(tensor, arrayIndexPropertiesActivationIndexInference, arrayNumberOfPropertiesInference)
		elif(arrayIndexPropertiesTimeCreateInference and arrayIndexPropertiesTimeIndexInference is not None and propertyCount == arrayIndexPropertiesTimeIndexInference and propertyCount < arrayNumberOfPropertiesInference):
			result = insertPropertyDimension(tensor, arrayIndexPropertiesTimeIndexInference, arrayNumberOfPropertiesInference)
		elif(arrayIndexPropertiesActivationCreateInference and not arrayIndexPropertiesActivation and propertyCount == arrayNumberOfPropertiesInference - 1):
			result = insertPropertyDimension(tensor, arrayIndexPropertiesActivationIndexInference, arrayNumberOfPropertiesInference)
		'''	
	else:
		if(propertyCount == arrayNumberOfPropertiesTrain):
			result = tensor
		else:
			raise RuntimeError(f"{tensorName} property dimension mismatch: expected {arrayNumberOfPropertiesTrain}, got {propertyCount}")
	return result

def expandSparseTensorSize(tensor, newSize, tensorName):
	result = tensor
	currentSize = tuple(result.size())
	targetSize = tuple(newSize)
	if(result.layout != pt.sparse_coo):
		raise RuntimeError(f"{tensorName} sparse expansion error: tensor must be sparse COO")
	if(len(currentSize) != len(targetSize)):
		raise RuntimeError(f"{tensorName} sparse expansion error: rank mismatch {len(currentSize)} vs {len(targetSize)}")
	for dimensionIndex in range(len(currentSize)):
		if(targetSize[dimensionIndex] < currentSize[dimensionIndex]):
			raise RuntimeError(f"{tensorName} sparse expansion error: target size {targetSize} shrinks current size {currentSize}")
	result.sparse_resize_(targetSize, result.sparse_dim(), result.dense_dim())
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

def adjustBranchDimensions(tensor, tensorName, expectedRank, branchCount=numberOfDendriticBranches):
	if tensor.dim() == expectedRank:
		return expandBranchDimensions(tensor, tensorName, branchCount)
	if tensor.dim() == expectedRank - 1:
		return insertBranchDimension(tensor, tensorName, insertIndex=1, branchCount=branchCount)
	raise RuntimeError(f"{tensorName} branch dimension mismatch: expected rank {expectedRank} or {expectedRank - 1}, got {tensor.dim()}")

def ensureFeatureConnectionsSourceTensorCurrentSize(tensor, targetC, targetF, tensorName):
	result = tensor
	if(result.dim() != 5):
		raise RuntimeError(f"{tensorName} rank mismatch: expected 5, got {result.dim()}")
	currentC = result.size(3)
	currentF = result.size(4)
	if(currentC > targetC or currentF > targetF):
		raise RuntimeError(f"{tensorName} size mismatch: stored size {(currentC, currentF)} exceeds current database size {(targetC, targetF)}")
	if(currentC < targetC or currentF < targetF):
		newSize = list(result.size())
		newSize[3] = targetC
		newSize[4] = targetF
		if(result.is_sparse):
			result = expandSparseTensorSize(result, newSize, tensorName)
		else:
			expandedTensor = pt.zeros(newSize, dtype=result.dtype, device=result.device)
			expandedTensor[:, :, :, :currentC, :currentF] = result
			result = expandedTensor
	return result

def ensureFeatureNeuronsTensorCurrentSize(tensor, targetF, tensorName):
	result = tensor
	if(result.dim() != 4):
		raise RuntimeError(f"{tensorName} rank mismatch: expected 4, got {result.dim()}")
	currentF = result.size(3)
	if(currentF > targetF):
		raise RuntimeError(f"{tensorName} size mismatch: stored feature size {currentF} exceeds current database size {targetF}")
	if(currentF < targetF):
		newSize = list(result.size())
		newSize[3] = targetF
		if(result.is_sparse):
			result = expandSparseTensorSize(result, newSize, tensorName)
		else:
			expandedTensor = pt.zeros(newSize, dtype=result.dtype, device=result.device)
			expandedTensor[:, :, :, :currentF] = result
			result = expandedTensor
	return result

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
			return expandSparseTensorSize(tensor, newSize, tensorName)
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
		if(not storeDatabaseInRam):
			# Save observed columns to disk
			if(debugPrintTrainSectionTimes):
				saveObservedColumnsStartTime = time.perf_counter()
			for observedColumn in observedColumnsDict.values():
				saveAllSourceFeatures = False
				observedColumn.saveToDisk(saveAllSourceFeatures)
			if(debugPrintTrainSectionTimes):
				GIAANNproto_debug.debugTrainSectionTimesAdd(databaseNetworkObject, "saveData.observedColumn.saveToDisk", time.perf_counter() - saveObservedColumnsStartTime)
		else:
			printe("GIAANNproto_databaseNetworkFiles:saveData():!forceSaveGlobalState requires !storeDatabaseInRam")
			
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
		GIAANNproto_debug.debugTrainSectionTimesAdd(databaseNetworkObject, "saveData.total", time.perf_counter() - saveDataStartTime)

def generateObservedColumnMetadataData(observedColumn):
	if(trainStoreFeatureMapsGlobally):
		result = {
			'conceptIndex': observedColumn.conceptIndex,
			'nextFeatureIndex': observedColumn.nextFeatureIndex,
			'featureConnectionsFormat': observedColumnFeatureConnectionsFormat
		}
	else:
		result = {
			'conceptIndex': observedColumn.conceptIndex,
			'featureWordToIndex': observedColumn.featureWordToIndex,
			'featureIndexToWord': observedColumn.featureIndexToWord,
			'nextFeatureIndex': observedColumn.nextFeatureIndex,
			'featureConnectionsFormat': observedColumnFeatureConnectionsFormat
		}
	return result

def generateObservedColumnMetadataSignature(metadataData):
	if('featureConnectionsFormat' not in metadataData):
		raise RuntimeError("generateObservedColumnMetadataSignature error: missing featureConnectionsFormat")
	if(trainStoreFeatureMapsGlobally):
		result = (int(metadataData['conceptIndex']), int(metadataData['nextFeatureIndex']), metadataData['featureConnectionsFormat'])
	else:
		result = (int(metadataData['conceptIndex']), tuple(sorted(metadataData['featureWordToIndex'].items())), tuple(sorted(metadataData['featureIndexToWord'].items())), int(metadataData['nextFeatureIndex']), metadataData['featureConnectionsFormat'])
	return result
		
def observedColumnSaveToDisk(self, saveAllSourceFeatures, resizeFeatureTensorsToCurrentSize=False):
	"""
	Save the observed column data to disk.
	"""
	validateObservedColumnStorageFormat(self.conceptIndex)
	ensureObservedColumnFolderExists(self.conceptIndex)
	if(resizeFeatureTensorsToCurrentSize):
		self.ensureRAMdatabaseFeatureTensorSizes()
	
	data = generateObservedColumnMetadataData(self)
	if(optimisationObservedColumnsWriteMetadataCheck):
		metadataSignature = generateObservedColumnMetadataSignature(data)
		metadataNeedsWrite = getattr(self, "savedMetadataSignature", None) != metadataSignature
		if(metadataNeedsWrite):
			with open(getObservedColumnMetadataFile(self.conceptIndex), 'wb') as f:
				pickle.dump(data, f)
			self.savedMetadataSignature = metadataSignature
	else:
		with open(getObservedColumnMetadataFile(self.conceptIndex), 'wb') as f:
			pickle.dump(data, f)
	
	if(saveAllSourceFeatures):
		sourceFeatureIndicesToSave = None
	else:
		if(not storeDatabaseInRam):
			if(self.hasTrainPreparedSourceFeatureIndices()):
				sourceFeatureIndicesToSave = self.getTrainPreparedSourceFeatureIndices()
			elif(optimisationArrayIndexPropertiesEfficientSerialConnections):
				if(len(self.loadedSourceFeatureIndices) == 0):
					sourceFeatureIndicesToSave = []
				else:
					raise RuntimeError("observedColumnSaveToDisk error: optimisationArrayIndexPropertiesEfficientSerialConnections requires trainPreparedSourceFeatureIndices for loaded source feature tensors")
			else:
				raise RuntimeError("observedColumnSaveToDisk(saveAllSourceFeatures) requires self.hasTrainPreparedSourceFeatureIndices() or optimisationArrayIndexPropertiesEfficientSerialConnections")
		else:
			raise RuntimeError("observedColumnSaveToDisk(saveAllSourceFeatures) requires !storeDatabaseInRam")
	self.saveLoadedSourceFeatureConnectionsToDisk(sourceFeatureIndicesToSave)
	
	if lowMem:
		saveTensor(self.featureNeurons, getObservedColumnFolder(self.conceptIndex), getObservedColumnFeatureNeuronsFileBaseName())

def loadObservedColumnSourceFeatureConnectionsTensor(databaseNetworkObject, conceptIndex, sourceFeatureIndex, targetDevice, ensureCurrentSizeOnLoad=False):
	connectionsFolder = getObservedColumnFeatureConnectionsFolder(conceptIndex)
	fileBaseName = getObservedColumnSourceFeatureConnectionsFileBaseName(sourceFeatureIndex)
	tensorName = f"observedColumn.featureConnectionsBySourceFeature[{conceptIndex}][{sourceFeatureIndex}]"
	tensor = adjustPropertyDimensions(databaseNetworkObject.inferenceMode, loadTensor(connectionsFolder, fileBaseName, targetDevice=targetDevice), tensorName)
	tensor = adjustBranchDimensions(tensor, tensorName, expectedRank=5)
	if(debugLimitFeatures):
		tensor = GIAANNproto_debug.applyDebugLimitFeatureConnectionsSourceTensor(tensor, databaseNetworkObject.c, databaseNetworkObject.f, tensorName)
	if(ensureCurrentSizeOnLoad):
		tensor = ensureFeatureConnectionsSourceTensorCurrentSize(tensor, databaseNetworkObject.c, databaseNetworkObject.f, tensorName)
	return tensor

def saveObservedColumnSourceFeatureConnectionsTensor(conceptIndex, sourceFeatureIndex, tensor):
	connectionsFolder = getObservedColumnFeatureConnectionsFolder(conceptIndex)
	fileBaseName = getObservedColumnSourceFeatureConnectionsFileBaseName(sourceFeatureIndex)
	filePath = os.path.join(connectionsFolder, fileBaseName + pytorchTensorFileExtension)
	if(tensor is None):
		raise RuntimeError("saveObservedColumnSourceFeatureConnectionsTensor error: tensor is None")
	if(tensor.is_sparse):
		tensor = tensor.coalesce()
		tensorNNZ = tensor._nnz()
	else:
		tensorNNZ = int(pt.count_nonzero(tensor).item())
	if(tensorNNZ > 0):
		saveTensor(tensor, connectionsFolder, fileBaseName)
	else:
		if(pathExists(filePath)):
			os.remove(filePath)
	return

def observedColumnLoadFromDisk(cls, databaseNetworkObject, conceptIndex, lemma, i, targetDevice=None, loadAllSourceFeatures=False, resizeFeatureTensorsToCurrentSize=False):
	"""
	Load the observed column data from disk.
	"""
	validateObservedColumnStorageFormat(conceptIndex)
	metadataFile = getObservedColumnMetadataFile(conceptIndex)
	with open(metadataFile, 'rb') as f:
		data = pickle.load(f)
	instance = cls(databaseNetworkObject, conceptIndex, lemma, i)
	if(data.get('featureConnectionsFormat') != observedColumnFeatureConnectionsFormat):
		raise RuntimeError(f"Unsupported observed column connection storage format for conceptIndex={conceptIndex}. Clear and rebuild the database.")
	instance.savedMetadataSignature = generateObservedColumnMetadataSignature(data)
	if(trainStoreFeatureMapsGlobally):
		instance.featureWordToIndex = databaseNetworkObject.conceptFeaturesDict
		instance.featureIndexToWord = databaseNetworkObject.conceptFeaturesIndexToWordDict
		instance.nextFeatureIndex = len(databaseNetworkObject.conceptFeaturesDict) - 1
	else:
		instance.featureWordToIndex = data['featureWordToIndex']
		instance.featureIndexToWord = data['featureIndexToWord']
		instance.nextFeatureIndex = data['nextFeatureIndex']
		if(debugLimitFeatures):
			instance.featureWordToIndex, instance.featureIndexToWord = GIAANNproto_debug.applyDebugLimitFeatureIndexMaps(instance.featureWordToIndex, instance.featureIndexToWord, databaseNetworkObject.f, f"observedColumn.featureIndexMaps[{conceptIndex}]")
			if(instance.nextFeatureIndex < 0):
				raise RuntimeError("observedColumnLoadFromDisk error: nextFeatureIndex < 0")
			if(instance.nextFeatureIndex > databaseNetworkObject.f):
				instance.nextFeatureIndex = databaseNetworkObject.f
	if lowMem:
		featureNeuronTargetDevice = targetDevice if targetDevice is not None else deviceDatabase
		instance.featureNeurons = adjustPropertyDimensions(databaseNetworkObject.inferenceMode, loadTensor(getObservedColumnFolder(conceptIndex), getObservedColumnFeatureNeuronsFileBaseName(), targetDevice=featureNeuronTargetDevice), f"observedColumn.featureNeurons[{conceptIndex}]")
		instance.featureNeurons = adjustBranchDimensions(instance.featureNeurons, f"observedColumn.featureNeurons[{conceptIndex}]", expectedRank=4)
		if(debugLimitFeatures):
			instance.featureNeurons = GIAANNproto_debug.applyDebugLimitFeatureNeuronsTensor(instance.featureNeurons, databaseNetworkObject.f, f"observedColumn.featureNeurons[{conceptIndex}]")
		if(resizeFeatureTensorsToCurrentSize):
			instance.featureNeurons = ensureFeatureNeuronsTensorCurrentSize(instance.featureNeurons, databaseNetworkObject.f, f"observedColumn.featureNeurons[{conceptIndex}]")
	if(loadAllSourceFeatures):
		sourceFeatureIndices = listObservedColumnSourceFeatureIndices(conceptIndex)
		loadTargetDevice = targetDevice if targetDevice is not None else deviceDatabase
		instance.loadRequiredSourceFeatureConnections(sourceFeatureIndices, loadTargetDevice, createMissing=False, ensureCurrentSizeOnLoad=resizeFeatureTensorsToCurrentSize)
	return instance

def saveTensor(tensor, folderName, fileName):
	fileIOTensor = tensor
	if(fileIOTensor.device != deviceFileIO):
		fileIOTensor = fileIOTensor.to(deviceFileIO)
	pt.save(fileIOTensor, os.path.join(folderName, fileName+pytorchTensorFileExtension))
	return

def loadTensor(folderName, fileName, targetDevice=None):
	loadDevice = targetDevice if targetDevice is not None else deviceSparse
	tensorPath = os.path.join(folderName, fileName+pytorchTensorFileExtension)
	tensor = pt.load(tensorPath, map_location=deviceFileIO)
	if(tensor.device != loadDevice):
		tensor = tensor.to(loadDevice)
	return tensor
