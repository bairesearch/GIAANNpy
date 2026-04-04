"""GIAANNproto_databaseNetwork.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_main.py

# Usage:
see GIAANNproto_main.py

# Description:
GIA ANN proto database Network

"""

import os
import torch as pt

from GIAANNproto_globalDefs import *
import GIAANNproto_databaseNetworkFiles
from GIAANNproto_databaseNetworkObservedColumn import ObservedColumn, ObservedColumnConnectionBase, ObservedColumnProxy, ObservedColumnStub
import GIAANNproto_sparseTensors

def calculateArrayNumberOfProperties(inferenceMode):
	if(inferenceMode):
		arrayNumberOfProperties = arrayNumberOfPropertiesInference
	else:
		arrayNumberOfProperties = arrayNumberOfPropertiesTrain
	return arrayNumberOfProperties

class DatabaseNetworkClass():
	def __init__(self, inferenceMode, c, f, s, p, conceptColumnsDict, conceptColumnsList, conceptFeaturesDict, conceptFeaturesList, globalFeatureNeurons, conceptFeaturesReferenceSetDelimiterList, conceptFeaturesReferenceSetDelimiterDeterministicList, conceptFeaturesReferenceSetDelimiterProbabilisticList):
		self.c = c
		self.f = f
		self.s = s
		self.p = p
		self.conceptColumnsDict = conceptColumnsDict
		self.conceptColumnsList = conceptColumnsList
		self.conceptFeaturesDict = conceptFeaturesDict
		self.conceptFeaturesList = conceptFeaturesList
		self.conceptFeaturesIndexToWordDict = dict(enumerate(conceptFeaturesList))
		self.globalFeatureNeurons = globalFeatureNeurons
		self.globalFeatureConnections = None
		self.globalFeatureNeuronsBackup = None
		self.globalFeatureConnectionsBackup = None
		self.observedColumnsDictRAM = {} if storeDatabaseInRam else None
		self.observedColumnsRAMLoaded = False if storeDatabaseInRam else None
		if(conceptColumnsDelimitByPOS):
			if(detectReferenceSetDelimitersBetweenNouns):
				self.conceptFeaturesReferenceSetDelimiterDeterministicList = conceptFeaturesReferenceSetDelimiterDeterministicList
				self.conceptFeaturesReferenceSetDelimiterProbabilisticList = conceptFeaturesReferenceSetDelimiterProbabilisticList
			else:
				self.conceptFeaturesReferenceSetDelimiterList = conceptFeaturesReferenceSetDelimiterList
		self.setArrayIndexProperties(inferenceMode)
		self.inferenceMode = inferenceMode

	def setArrayIndexProperties(self, inferenceMode):
		if(inferenceMode):
			self.arrayNumberOfProperties = arrayNumberOfPropertiesInference
			self.arrayIndexPropertiesStrengthIndex = arrayIndexPropertiesStrengthIndexInference
			self.arrayIndexPropertiesPermanenceIndex = arrayIndexPropertiesPermanenceIndexInference
			self.arrayIndexPropertiesActivationIndex = arrayIndexPropertiesActivationIndexInference
			self.arrayIndexPropertiesTimeIndex = arrayIndexPropertiesTimeIndexInference
			self.arrayIndexPropertiesPosIndex = arrayIndexPropertiesPosIndexInference
			self.arrayIndexPropertiesMinWordDistanceIndex = arrayIndexPropertiesMinWordDistanceIndexInference
		else:
			self.arrayNumberOfProperties = arrayNumberOfPropertiesTrain
			self.arrayIndexPropertiesStrengthIndex = arrayIndexPropertiesStrengthIndexTrain
			self.arrayIndexPropertiesPermanenceIndex = arrayIndexPropertiesPermanenceIndexTrain
			self.arrayIndexPropertiesActivationIndex = arrayIndexPropertiesActivationIndexTrain
			self.arrayIndexPropertiesTimeIndex = arrayIndexPropertiesTimeIndexTrain
			self.arrayIndexPropertiesPosIndex = arrayIndexPropertiesPosIndexTrain
			self.arrayIndexPropertiesMinWordDistanceIndex = arrayIndexPropertiesMinWordDistanceIndexTrain
		return
		
def backupGlobalArrays(databaseNetworkObject):
	if(databaseNetworkObject.globalFeatureNeurons is None):
		raise RuntimeError("backupGlobalArrays error: globalFeatureNeurons is None")
	backupDevice = pt.device("cpu")
	databaseNetworkObject.globalFeatureNeuronsBackup = databaseNetworkObject.globalFeatureNeurons.coalesce().to(backupDevice)
	if(databaseNetworkObject.globalFeatureConnections is not None):
		databaseNetworkObject.globalFeatureConnectionsBackup = databaseNetworkObject.globalFeatureConnections.coalesce().to(backupDevice)
	else:
		databaseNetworkObject.globalFeatureConnectionsBackup = None
		
def restoreGlobalArrays(databaseNetworkObject):
	if(databaseNetworkObject.globalFeatureNeuronsBackup is None):
		raise RuntimeError("restoreGlobalArrays error: globalFeatureNeuronsBackup is None")
	databaseNetworkObject.globalFeatureNeurons = databaseNetworkObject.globalFeatureNeuronsBackup.to(deviceSparse)
	if(databaseNetworkObject.globalFeatureConnectionsBackup is not None):
		databaseNetworkObject.globalFeatureConnections = databaseNetworkObject.globalFeatureConnectionsBackup.to(deviceSparse)
	else:
		databaseNetworkObject.globalFeatureConnections = None

def ensureGlobalFeatureNeuronsSize(databaseNetworkObject, updateBackup):
	expanded = False
	if(lowMem):
		raise RuntimeError("ensureGlobalFeatureNeuronsSize error: lowMem must be False")
	if(databaseNetworkObject.globalFeatureNeurons is None):
		raise RuntimeError("ensureGlobalFeatureNeuronsSize error: globalFeatureNeurons is None")
	if(databaseNetworkObject.globalFeatureNeurons.shape[3] < databaseNetworkObject.c or databaseNetworkObject.globalFeatureNeurons.shape[4] < databaseNetworkObject.f):
		newShape = (databaseNetworkObject.arrayNumberOfProperties, numberOfDendriticBranches, arrayNumberOfSegments, databaseNetworkObject.c, databaseNetworkObject.f)
		databaseNetworkObject.globalFeatureNeurons = GIAANNproto_databaseNetworkFiles.expandSparseTensorSize(databaseNetworkObject.globalFeatureNeurons, newShape, "ensureGlobalFeatureNeuronsSize.globalFeatureNeurons")
		expanded = True
	if(updateBackup and databaseNetworkObject.globalFeatureNeuronsBackup is not None):
		if(databaseNetworkObject.globalFeatureNeuronsBackup.shape[3] < databaseNetworkObject.c or databaseNetworkObject.globalFeatureNeuronsBackup.shape[4] < databaseNetworkObject.f):
			newBackupShape = (databaseNetworkObject.arrayNumberOfProperties, numberOfDendriticBranches, arrayNumberOfSegments, databaseNetworkObject.c, databaseNetworkObject.f)
			databaseNetworkObject.globalFeatureNeuronsBackup = GIAANNproto_databaseNetworkFiles.expandSparseTensorSize(databaseNetworkObject.globalFeatureNeuronsBackup, newBackupShape, "ensureGlobalFeatureNeuronsSize.globalFeatureNeuronsBackup")
			expanded = True
	return expanded

# Initialize global feature neuron arrays if lowMem is disabled
if not lowMem:
	def initialiseFeatureNeuronsGlobal(inferenceMode, c, f):
		arrayNumberOfProperties = calculateArrayNumberOfProperties(inferenceMode)
		globalFeatureNeurons = GIAANNproto_sparseTensors.createEmptySparseTensor((arrayNumberOfProperties, numberOfDendriticBranches, arrayNumberOfSegments, c, f))
		return globalFeatureNeurons
		
	def loadFeatureNeuronsGlobal(inferenceMode, c, f):
		if GIAANNproto_databaseNetworkFiles.pathExists(globalFeatureNeuronsFileFull):
			globalFeatureNeurons = GIAANNproto_databaseNetworkFiles.loadFeatureNeuronsGlobalFile(inferenceMode)
			if(debugLimitFeatures):
				globalFeatureNeurons = GIAANNproto_databaseNetworkFiles.applyDebugLimitGlobalFeatureNeuronsTensor(globalFeatureNeurons, c, f, "globalFeatureNeurons")
				if(globalFeatureNeurons.size(3) < c or globalFeatureNeurons.size(4) < f):
					print("globalFeatureNeurons.size(3) = ", globalFeatureNeurons.size(3))
					print("globalFeatureNeurons.size(4) = ", globalFeatureNeurons.size(4))
					raise RuntimeError("loadFeatureNeuronsGlobal error: debugLimitFeatures requires limits that do not exceed saved globalFeatureNeurons dimensions")
		else:
			globalFeatureNeurons = initialiseFeatureNeuronsGlobal(inferenceMode, c, f)
			#print("initialiseFeatureNeuronsGlobal: globalFeatureNeurons = ", globalFeatureNeurons)
		return globalFeatureNeurons
		
def initialiseDatabaseNetwork(inferenceMode):

	conceptColumnsDict = {}  # key: lemma, value: index
	conceptColumnsList = []  # list of concept column names (lemmas)
	c = 0  # current number of concept columns
	conceptFeaturesDict = {}  # key: word, value: index
	conceptFeaturesList = []  # list of concept feature names (words)
	f = 0  # current number of concept features
	conceptFeaturesReferenceSetDelimiterList = []
	conceptFeaturesReferenceSetDelimiterDeterministicList = []
	conceptFeaturesReferenceSetDelimiterProbabilisticList = []
	loadExistingDatabase = inferenceMode or (trainLoadExistingDatabase and GIAANNproto_databaseNetworkFiles.pathExists(conceptColumnsDictFile))

	# Initialize the concept columns dictionary
	if(loadExistingDatabase and GIAANNproto_databaseNetworkFiles.pathExists(conceptColumnsDictFile)):
		conceptColumnsDict = GIAANNproto_databaseNetworkFiles.loadDictFile(conceptColumnsDictFile)
		c = len(conceptColumnsDict)
		conceptColumnsList = list(conceptColumnsDict.keys())
		conceptFeaturesDict = GIAANNproto_databaseNetworkFiles.loadDictFile(conceptFeaturesDictFile)
		f = len(conceptFeaturesDict)
		conceptFeaturesList = list(conceptFeaturesDict.keys())
		if(conceptColumnsDelimitByPOS):
			if(detectReferenceSetDelimitersBetweenNouns):	
				conceptFeaturesReferenceSetDelimiterDeterministicDict = GIAANNproto_databaseNetworkFiles.loadDictFile(conceptFeaturesReferenceSetDelimiterDeterministicListFile)
				conceptFeaturesReferenceSetDelimiterDeterministicList = list(conceptFeaturesReferenceSetDelimiterDeterministicDict.values())
				conceptFeaturesReferenceSetDelimiterProbabilisticDict = GIAANNproto_databaseNetworkFiles.loadDictFile(conceptFeaturesReferenceSetDelimiterProbabilisticListFile)
				conceptFeaturesReferenceSetDelimiterProbabilisticList = list(conceptFeaturesReferenceSetDelimiterProbabilisticDict.values())
			else:
				conceptFeaturesReferenceSetDelimiterDict = GIAANNproto_databaseNetworkFiles.loadDictFile(conceptFeaturesReferenceSetDelimiterListFile)
				conceptFeaturesReferenceSetDelimiterList = list(conceptFeaturesReferenceSetDelimiterDict.values())
		if(debugLimitFeatures):
			conceptColumnsDict = applyDebugLimitIndexDict(conceptColumnsDict, debugLimitFeaturesCMax, "conceptColumnsDict")
			conceptColumnsList = buildIndexListFromDict(conceptColumnsDict, "conceptColumnsDict")
			c = len(conceptColumnsList)
			conceptFeaturesDict = applyDebugLimitIndexDict(conceptFeaturesDict, debugLimitFeaturesFMax, "conceptFeaturesDict")
			conceptFeaturesList = buildIndexListFromDict(conceptFeaturesDict, "conceptFeaturesDict")
			f = len(conceptFeaturesList)
			if(conceptColumnsDelimitByPOS):
				if(detectReferenceSetDelimitersBetweenNouns):
					conceptFeaturesReferenceSetDelimiterDeterministicList = applyDebugLimitList(conceptFeaturesReferenceSetDelimiterDeterministicList, f, "conceptFeaturesReferenceSetDelimiterDeterministicList")
					conceptFeaturesReferenceSetDelimiterProbabilisticList = applyDebugLimitList(conceptFeaturesReferenceSetDelimiterProbabilisticList, f, "conceptFeaturesReferenceSetDelimiterProbabilisticList")
				else:
					conceptFeaturesReferenceSetDelimiterList = applyDebugLimitList(conceptFeaturesReferenceSetDelimiterList, f, "conceptFeaturesReferenceSetDelimiterList")
	else:
		if(useDedicatedConceptNames):
			# Add dummy feature for prime concept neuron (different per concept column)
			conceptFeaturesList.append(variablePrimeConceptFeatureNeuronName)
			conceptFeaturesDict[variablePrimeConceptFeatureNeuronName] = len(conceptFeaturesDict)
			f += 1  # Will be updated dynamically based on c

		if useDedicatedFeatureLists:
			print("error: useDedicatedFeatureLists case not yet coded - need to set f and populate concept_features_list/conceptFeaturesDict etc")
			exit()
			# f = max_num_non_nouns + 1  # Maximum number of non-nouns in an English dictionary, plus the prime concept neuron of each column

		if(conceptColumnsDelimitByPOS):
			#initialise for concept feature;
			if(detectReferenceSetDelimitersBetweenNouns):
				conceptFeaturesReferenceSetDelimiterDeterministicList.append(False)
				conceptFeaturesReferenceSetDelimiterProbabilisticList.append(False)
			else:
				conceptFeaturesReferenceSetDelimiterList.append(False)
	if not lowMem:
		if(loadExistingDatabase):
			globalFeatureNeurons = loadFeatureNeuronsGlobal(inferenceMode, c, f)
		else:
			globalFeatureNeurons = initialiseFeatureNeuronsGlobal(inferenceMode, c, f)
	else:
		globalFeatureNeurons = None

	s = arrayNumberOfSegments
	p = calculateArrayNumberOfProperties(inferenceMode)
		
	databaseNetworkObject = DatabaseNetworkClass(inferenceMode, c, f, s, p, conceptColumnsDict, conceptColumnsList, conceptFeaturesDict, conceptFeaturesList, globalFeatureNeurons, conceptFeaturesReferenceSetDelimiterList, conceptFeaturesReferenceSetDelimiterDeterministicList, conceptFeaturesReferenceSetDelimiterProbabilisticList)
	
	if(printTotalFeatures):
		print("initialiseDatabaseNetwork: c = ", databaseNetworkObject.c, ", f = ", databaseNetworkObject.f)
	
	return databaseNetworkObject
	

def addConceptToConceptColumnsDict(databaseNetworkObject, lemma, conceptsFound, newConceptsAdded):
	conceptsFound = True
	if lemma not in databaseNetworkObject.conceptColumnsDict:
		# Add to concept columns dictionary
		#print("adding concept = ", lemma)
		databaseNetworkObject.conceptColumnsDict[lemma] = databaseNetworkObject.c
		databaseNetworkObject.conceptColumnsList.append(lemma)
		databaseNetworkObject.c += 1
		newConceptsAdded = True
	return conceptsFound, newConceptsAdded
	
def loadOrCreateObservedColumn(databaseNetworkObject, conceptIndex, lemma, i, targetDevice=None, createDeviceCopy=False, requiredSourceFeatureIndices=None, loadAllSourceFeatures=False):
	observedColumn = None
	if(storeDatabaseInRam):
		if(databaseNetworkObject.observedColumnsDictRAM is None):
			raise RuntimeError("loadOrCreateObservedColumn error: observedColumnsDictRAM is None while storeDatabaseInRam is enabled")
		if(not databaseNetworkObject.observedColumnsRAMLoaded):
			raise RuntimeError("loadOrCreateObservedColumn error: storeDatabaseInRam requires observedColumnsRAMLoaded after startup")
		if(lemma in databaseNetworkObject.observedColumnsDictRAM):
			observedColumn = databaseNetworkObject.observedColumnsDictRAM[lemma]
		else:
			observedColumn = ObservedColumn(databaseNetworkObject, conceptIndex, lemma, i)
			databaseNetworkObject.observedColumnsDictRAM[lemma] = observedColumn
		observedColumn.ensureObservedColumnFeatureArraysFeatures(databaseNetworkObject.f)
	else:
		GIAANNproto_databaseNetworkFiles.validateObservedColumnStorageFormat(conceptIndex)
		if GIAANNproto_databaseNetworkFiles.observedColumnMetadataExists(conceptIndex):
			observedColumn = ObservedColumn.loadFromDisk(databaseNetworkObject, conceptIndex, lemma, i, targetDevice=deviceDatabase, loadAllSourceFeatures=loadAllSourceFeatures)
		else:
			observedColumn = ObservedColumn(databaseNetworkObject, conceptIndex, lemma, i)
		observedColumn.ensureObservedColumnFeatureArraysFeatures(databaseNetworkObject.f)
	resultObservedColumn = observedColumn
	if(createDeviceCopy):
		if(not storeDatabaseInRam):
			raise RuntimeError("loadOrCreateObservedColumn error: createDeviceCopy requires storeDatabaseInRam")
		if(targetDevice is None):
			raise RuntimeError("loadOrCreateObservedColumn error: createDeviceCopy requires targetDevice")
		resultObservedColumn = cloneObservedColumnToDevice(databaseNetworkObject, observedColumn, lemma, i, targetDevice, requiredSourceFeatureIndices, loadAllSourceFeatures)
	else:
		loadTargetDevice = targetDevice
		if(loadTargetDevice is None and storeDatabaseInRam and useGPUdatabase != useGPUsparse):
			loadTargetDevice = deviceDatabase
		if(loadAllSourceFeatures):
			sourceFeatureIndices = observedColumn.listStoredSourceFeatureIndices()
			observedColumn.loadRequiredSourceFeatureConnections(sourceFeatureIndices, loadTargetDevice, createMissing=False)
		elif(requiredSourceFeatureIndices is not None):
			observedColumn.loadRequiredSourceFeatureConnections(requiredSourceFeatureIndices, loadTargetDevice, createMissing=False)
	return resultObservedColumn

def loadObservedColumnToRamStartup(databaseNetworkObject, conceptIndex, lemma, i):
	if(not storeDatabaseInRam):
		raise RuntimeError("loadObservedColumnToRamStartup error: storeDatabaseInRam is False")
	if(databaseNetworkObject.observedColumnsRAMLoaded):
		raise RuntimeError("loadObservedColumnToRamStartup error: observedColumnsRAMLoaded is already True")
	GIAANNproto_databaseNetworkFiles.validateObservedColumnStorageFormat(conceptIndex)
	metadataFile = GIAANNproto_databaseNetworkFiles.getObservedColumnMetadataFile(conceptIndex)
	if(GIAANNproto_databaseNetworkFiles.pathExists(metadataFile)):
		result = ObservedColumn.loadFromDisk(databaseNetworkObject, conceptIndex, lemma, i, targetDevice=deviceDatabase, loadAllSourceFeatures=True, resizeFeatureTensorsToCurrentSize=resizeTensorsOnRAMdatabaseLoad)
	else:
		result = ObservedColumn(databaseNetworkObject, conceptIndex, lemma, i)
	return result

def generateGlobalFeatureConnections(databaseNetworkObject):
	conceptColumnsListTemp = []
	for i, (lemma, conceptIndex) in enumerate(databaseNetworkObject.conceptColumnsDict.items()):
		conceptColumn = loadOrCreateObservedColumn(databaseNetworkObject, conceptIndex, lemma, i, targetDevice=deviceSparse, createDeviceCopy=False, loadAllSourceFeatures=True)
		conceptColumnsListTemp.append(conceptColumn)
	globalFeatureConnectionsList = []
	for conceptColumn in conceptColumnsListTemp:
		globalFeatureConnectionsList.append(conceptColumn.materialiseFeatureConnections(loadAllStored=True, targetDevice=deviceSparse))
	databaseNetworkObject.globalFeatureConnections = pt.stack(globalFeatureConnectionsList, dim=3)
	print("generate_global_feature_connections: databaseNetworkObject.global_feature_connections.shape = ", databaseNetworkObject.globalFeatureConnections.shape)

def loadAllColumns(databaseNetworkObject):
	observedColumnsDict = {}
	if(storeDatabaseInRam):
		if(databaseNetworkObject.observedColumnsRAMLoaded):
			if(databaseNetworkObject.observedColumnsDictRAM is None):
				raise RuntimeError("loadAllColumns error: observedColumnsDictRAM is None while observedColumnsRAMLoaded is True")
			observedColumnsDict = databaseNetworkObject.observedColumnsDictRAM
			for observedColumn in observedColumnsDict.values():
				observedColumn.ensureObservedColumnFeatureArraysFeatures(databaseNetworkObject.f)
		else:
			for i, (lemma, conceptIndex) in enumerate(databaseNetworkObject.conceptColumnsDict.items()):
				conceptColumn = loadObservedColumnToRamStartup(databaseNetworkObject, conceptIndex, lemma, i)
				observedColumnsDict[lemma] = conceptColumn
	else:
		for i, (lemma, conceptIndex) in enumerate(databaseNetworkObject.conceptColumnsDict.items()):
			conceptColumn = loadOrCreateObservedColumn(databaseNetworkObject, conceptIndex, lemma, i, targetDevice=deviceDatabase, createDeviceCopy=False, loadAllSourceFeatures=False)
			observedColumnsDict[lemma] = conceptColumn
	return observedColumnsDict

def loadAllObservedColumnsToRam(databaseNetworkObject):
	if(storeDatabaseInRam):
		if(not databaseNetworkObject.observedColumnsRAMLoaded):
			observedColumnsDict = loadAllColumns(databaseNetworkObject)
			databaseNetworkObject.observedColumnsDictRAM = observedColumnsDict
			databaseNetworkObject.observedColumnsRAMLoaded = True
	else:
		raise RuntimeError("loadAllObservedColumnsToRam error: storeDatabaseInRam is False")
	return

def cloneObservedColumnToDevice(databaseNetworkObject, observedColumn, lemma, i, targetDevice, requiredSourceFeatureIndices=None, loadAllSourceFeatures=False):
	if(not storeDatabaseInRam):
		raise RuntimeError("cloneObservedColumnToDevice error: storeDatabaseInRam is False")
	if(targetDevice is None):
		raise RuntimeError("cloneObservedColumnToDevice error: targetDevice is None")
	copiedObservedColumn = ObservedColumnProxy(databaseNetworkObject, observedColumn, lemma, i, targetDevice)
	if(loadAllSourceFeatures):
		sourceFeatureIndices = observedColumn.listStoredSourceFeatureIndices()
		copiedObservedColumn.loadRequiredSourceFeatureConnections(sourceFeatureIndices, targetDevice, createMissing=False)
	elif(requiredSourceFeatureIndices is not None):
		copiedObservedColumn.loadRequiredSourceFeatureConnections(requiredSourceFeatureIndices, targetDevice, createMissing=False)
	if(lowMem):
		copiedObservedColumn.featureNeurons = observedColumn.featureNeurons.to(targetDevice)
	return copiedObservedColumn

def moveObservedColumnsDictConnectionsToDatabaseAfterTrain(observedColumnsDict, inferenceSequenceInPrompt):
	if(useGPUdatabase != useGPUsparse):
		if(not inferenceSequenceInPrompt):
			for observedColumn in observedColumnsDict.values():
				for sourceFeatureIndex in observedColumn.getTrainPreparedSourceFeatureIndices():
					sourceTensor = observedColumn.getFeatureConnectionsForSourceFeature(sourceFeatureIndex, deviceDatabase, createMissing=False)
					observedColumn.setFeatureConnectionsForSourceFeature(sourceFeatureIndex, sourceTensor)
				observedColumn.clearTrainPreparedSourceFeatureIndices()
	return

def prepareObservedColumnsForTrainSequence(observedColumnsDict, requiredSourceFeatureIndicesByObservedColumn):
	if(requiredSourceFeatureIndicesByObservedColumn is None):
		raise RuntimeError("prepareObservedColumnsForTrainSequence error: requiredSourceFeatureIndicesByObservedColumn is None")
	for observedColumn in observedColumnsDict.values():
		conceptIndex = int(observedColumn.conceptIndex)
		if(conceptIndex not in requiredSourceFeatureIndicesByObservedColumn):
			raise RuntimeError(f"prepareObservedColumnsForTrainSequence error: missing required source features for conceptIndex {conceptIndex}")
		requiredSourceFeatureIndices = requiredSourceFeatureIndicesByObservedColumn[conceptIndex]
		if(requiredSourceFeatureIndices is None):
			raise RuntimeError(f"prepareObservedColumnsForTrainSequence error: requiredSourceFeatureIndices is None for conceptIndex {conceptIndex}")
		if(len(requiredSourceFeatureIndices) == 0):
			raise RuntimeError(f"prepareObservedColumnsForTrainSequence error: requiredSourceFeatureIndices is empty for conceptIndex {conceptIndex}")
		observedColumn.prepareRequiredSourceFeatureConnectionsTrain(requiredSourceFeatureIndices, deviceSparse, createMissing=False)
		observedColumn.setTrainPreparedSourceFeatureIndices(requiredSourceFeatureIndices)
	return

def saveAllObservedColumnsToDisk(databaseNetworkObject):
	if(storeDatabaseInRam):
		if(databaseNetworkObject.observedColumnsDictRAM is None):
			raise RuntimeError("saveAllObservedColumnsToDisk error: observedColumnsDictRAM is None")
		for observedColumn in databaseNetworkObject.observedColumnsDictRAM.values():
			saveAllSourceFeatures = True
			observedColumn.saveToDisk(saveAllSourceFeatures, resizeFeatureTensorsToCurrentSize=resizeTensorsOnRAMdatabaseSave)
	else:
		raise RuntimeError("saveAllObservedColumnsToDisk error: storeDatabaseInRam is False")
	return

'''
def getTokenConceptFeatureIndexForSequenceConceptIndex(sequence_observed_columns, words_sequence, concept_mask, sequenceConceptIndex, sequenceWordIndex):
	conceptIndex = sequence_observed_columns.sequence_observed_columns_dict[sequenceConceptIndex].conceptIndex
	if(concept_mask[sequenceWordIndex]):
		feature_index = featureIndexPrimeConceptNeuron
	else:
		feature_index = sequence_observed_columns.featureWordToIndex[words_sequence[sequenceWordIndex]]
	return conceptIndex, feature_index
'''

def getTokenConceptFeatureIndexTensor(sequenceObservedColumns, tokensSequence, conceptMask, sequenceWordIndex, kcMax):
	if(kcMax != 1):
		raise RuntimeError("getTokenConceptFeatureIndexTensor error: kcMax must be 1")
	targetFoundNextColumnIndex, targetPreviousColumnIndex, targetNextColumnIndex, targetFeatureIndex = getTokenConceptFeatureIndex(sequenceObservedColumns, tokensSequence, conceptMask, sequenceWordIndex)
	result = (targetPreviousColumnIndex, targetNextColumnIndex, targetFeatureIndex)
	return result


def getTokenConceptFeatureIndex(sequenceObservedColumns, tokensSequence, conceptMask, sequenceWordIndex):
	databaseNetworkObject = sequenceObservedColumns.databaseNetworkObject
	columnsIndexSequenceWordIndexDict = sequenceObservedColumns.columnsIndexSequenceWordIndexDict
	
	if(conceptMask[sequenceWordIndex]):
		targetFeatureIndex = featureIndexPrimeConceptNeuron
	else:
		word = tokensSequence[sequenceWordIndex].word
		targetFeatureIndex = databaseNetworkObject.conceptFeaturesDict[word]
	assignedColumns = getattr(sequenceObservedColumns, "tokenConceptColumnIndexList", None)
	if(assignedColumns is not None and sequenceWordIndex < len(assignedColumns)):
		assignedColumnIndex = assignedColumns[sequenceWordIndex]
		if(assignedColumnIndex is not None):
			return False, assignedColumnIndex, None, targetFeatureIndex
		else:
			printe("tokenConceptColumnIndexList has not been generated")
	else:
		printe("tokenConceptColumnIndexList has not been generated")

def isFeatureIndexReferenceSetDelimiterDeterministic(databaseNetworkObject, featureIndex):
	if(conceptColumnsDelimitByPOS):
		if(detectReferenceSetDelimitersBetweenNouns):
			isDelimiter = databaseNetworkObject.conceptFeaturesReferenceSetDelimiterDeterministicList[featureIndex]
		else:
			isDelimiter = databaseNetworkObject.conceptFeaturesReferenceSetDelimiterList[featureIndex]
	else:
		printe("conceptColumnsDelimitByPOS is required")
	return isDelimiter

def isFeatureIndexReferenceSetDelimiterProbabilistic(databaseNetworkObject, featureIndex):
	if(conceptColumnsDelimitByPOS):
		if(detectReferenceSetDelimitersBetweenNouns):
			isDelimiterProbabilistic = databaseNetworkObject.conceptFeaturesReferenceSetDelimiterProbabilisticList[featureIndex]
		else:
			isDelimiterProbabilistic = False
	else:
		printe("conceptColumnsDelimitByPOS is required")
	return isDelimiterProbabilistic


if(debugLimitFeatures):
	def buildIndexListFromDict(indexDict, dictName):
		resultList = []
		maxIndex = -1
		for index in indexDict.values():
			if(index > maxIndex):
				maxIndex = index
		if(maxIndex >= 0):
			resultList = [None] * (maxIndex + 1)
			for name, index in indexDict.items():
				if(index < 0):
					raise RuntimeError(f"{dictName} index < 0")
				if(index >= len(resultList)):
					raise RuntimeError(f"{dictName} index out of bounds")
				if(resultList[index] is not None):
					raise RuntimeError(f"{dictName} duplicate index {index}")
				resultList[index] = name
			for name in resultList:
				if(name is None):
					raise RuntimeError(f"{dictName} missing index entry")
		return resultList
	def applyDebugLimitIndexDict(indexDict, maxCount, dictName):
		resultDict = indexDict
		if(debugLimitFeatures):
			if(maxCount <= 0):
				raise RuntimeError(f"{dictName} maxCount must be > 0")
			if(len(indexDict) > maxCount):
				trimmedDict = {}
				for name, index in indexDict.items():
					if(index < 0):
						raise RuntimeError(f"{dictName} index < 0")
					if(index < maxCount):
						trimmedDict[name] = index
				resultDict = trimmedDict
		return resultDict
	def applyDebugLimitList(listObject, maxCount, listName):
		resultList = listObject
		if(debugLimitFeatures):
			if(maxCount <= 0):
				raise RuntimeError(f"{listName} maxCount must be > 0")
			if(len(listObject) < maxCount):
				raise RuntimeError(f"{listName} length {len(listObject)} < expected {maxCount}")
			if(len(listObject) > maxCount):
				resultList = listObject[:maxCount]
		return resultList


#if(printCountTotalParameters):
def debugCountObservedColumnConnections(databaseNetworkObject, conceptIndex, lemma, columnIndex):
	columnConnections = 0
	if(not GIAANNproto_databaseNetworkFiles.observedColumnMetadataExists(conceptIndex)):
		columnConnections = 0
	else:
		sourceFeatureIndices = GIAANNproto_databaseNetworkFiles.listObservedColumnSourceFeatureIndices(conceptIndex)
		for sourceFeatureIndex in sourceFeatureIndices:
			featureConnections = GIAANNproto_databaseNetworkFiles.loadObservedColumnSourceFeatureConnectionsTensor(databaseNetworkObject, conceptIndex, sourceFeatureIndex, deviceDatabase)
			if(featureConnections is None):
				raise RuntimeError("debugCountObservedColumnConnections error: featureConnections is None for conceptIndex = " + str(conceptIndex) + ", sourceFeatureIndex = " + str(sourceFeatureIndex))
			if(databaseNetworkObject.arrayIndexPropertiesStrengthIndex < 0 or databaseNetworkObject.arrayIndexPropertiesStrengthIndex >= featureConnections.shape[0]):
				raise RuntimeError("debugCountObservedColumnConnections error: databaseNetworkObject.arrayIndexPropertiesStrengthIndex out of range")
			columnConnections += countNonZero(featureConnections)
			del featureConnections
	return columnConnections

def printCountTotalParametersRun(databaseNetworkObject):
	assert arrayIndexPropertiesEfficient 	#only databaseNetworkObject.arrayIndexPropertiesStrengthIndex stored in database, all tensors are coalesced
	if(databaseNetworkObject is None):
		raise RuntimeError("printCountTotalParametersRun error: databaseNetworkObject is None")
	if(databaseNetworkObject.arrayIndexPropertiesStrengthIndex is None):
		raise RuntimeError("printCountTotalParametersRun error: databaseNetworkObject.arrayIndexPropertiesStrengthIndex is None")
	totalColumns = len(databaseNetworkObject.conceptColumnsList)
	if(totalColumns <= 0):
		raise RuntimeError("printCountTotalParametersRun error: conceptColumnsList is empty")
	totalConnections = 0
	for columnIndex, lemma in enumerate(databaseNetworkObject.conceptColumnsList):
		#print("columnIndex = ", columnIndex)
		conceptIndex = databaseNetworkObject.conceptColumnsDict.get(lemma)
		if(conceptIndex is None):
			raise RuntimeError("printCountTotalParametersRun error: conceptIndex is None for lemma = " + lemma)
		columnConnections = debugCountObservedColumnConnections(databaseNetworkObject, conceptIndex, lemma, columnIndex)
		totalConnections += columnConnections
	database_pt_size_gb = debugCalculateDatabasePtSizeGiB()
	memory_gb = debugCalculateDatabaseSizeGiB()
	if(printCountTotalParameters):
		print("printCountTotalParameters totalConnections = ", totalConnections)
		print("printCountTotalParameters totalColumns = ", totalColumns)
		print(f"Total .pt size (uncompressed GiB): {database_pt_size_gb:.3f}")
		print(f"Total database size (uncompressed GiB): {memory_gb:.3f}")
	return memory_gb

def debugCalculateDatabasePtSizeGiB():
	if(not os.path.isdir(databaseFolder)):
		raise RuntimeError("debugCalculateDatabasePtSizeGiB error: missing databaseFolder = " + databaseFolder)
	totalPtBytesUncompressed = 0
	for directoryPath, directoryNames, fileNames in os.walk(databaseFolder):
		if(directoryNames is None):
			raise RuntimeError("debugCalculateDatabasePtSizeGiB error: directoryNames is None")
		for fileName in fileNames:
			filePath = os.path.join(directoryPath, fileName)
			if(not os.path.isfile(filePath)):
				raise RuntimeError("debugCalculateDatabasePtSizeGiB error: path is not a file = " + filePath)
			if(fileName.lower().endswith(pytorchTensorFileExtension.lower())):
				totalPtBytesUncompressed += os.path.getsize(filePath)
	totalPtGiB = totalPtBytesUncompressed / (1024 ** 3)
	return totalPtGiB

def debugCalculateDatabaseSizeGiB():
	if(not os.path.isdir(databaseFolder)):
		raise RuntimeError("debugCalculateDatabaseSizeGiB error: missing databaseFolder = " + databaseFolder)
	totalDatabaseBytesUncompressed = 0
	for directoryPath, directoryNames, fileNames in os.walk(databaseFolder):
		if(directoryNames is None):
			raise RuntimeError("debugCalculateDatabaseSizeGiB error: directoryNames is None")
		for fileName in fileNames:
			filePath = os.path.join(directoryPath, fileName)
			if(not os.path.isfile(filePath)):
				raise RuntimeError("debugCalculateDatabaseSizeGiB error: path is not a file = " + filePath)
			totalDatabaseBytesUncompressed += os.path.getsize(filePath)
	totalDatabaseGiB = totalDatabaseBytesUncompressed / (1024 ** 3)
	return totalDatabaseGiB

def countNonZero(t):
	# Works for sparse COO/CSR/CSC/BSR/BSC and dense tensors
	if isinstance(t, pt.Tensor):
		if t.is_sparse or (hasattr(t, "layout") and t.layout in {
			pt.sparse_coo, pt.sparse_csr, pt.sparse_csc, pt.sparse_bsr, pt.sparse_bsc
		}):
			return int(t._nnz())
		return int(pt.count_nonzero(t).item())
	return 0
