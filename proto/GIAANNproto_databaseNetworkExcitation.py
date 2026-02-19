"""GIAANNproto_databaseNetworkExcitation.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_main.py

# Usage:
see GIAANNproto_main.py

# Description:
GIA ANN proto database Network Excitation

"""

import os
import torch as pt

from GIAANNproto_globalDefs import *
import GIAANNproto_databaseNetworkFilesExcitation
import GIAANNproto_sparseTensors

class DatabaseNetworkClass():
	def __init__(self, c, f, s, p, conceptColumnsDict, conceptColumnsList, conceptFeaturesDict, conceptFeaturesList, globalFeatureNeurons, conceptFeaturesReferenceSetDelimiterList, conceptFeaturesReferenceSetDelimiterDeterministicList, conceptFeaturesReferenceSetDelimiterProbabilisticList):
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
		newShape = (arrayNumberOfProperties, numberOfDendriticBranches, arrayNumberOfSegments, databaseNetworkObject.c, databaseNetworkObject.f)
		databaseNetworkObject.globalFeatureNeurons = databaseNetworkObject.globalFeatureNeurons.coalesce()
		databaseNetworkObject.globalFeatureNeurons = pt.sparse_coo_tensor(databaseNetworkObject.globalFeatureNeurons.indices(), databaseNetworkObject.globalFeatureNeurons.values(), size=newShape, dtype=arrayType, device=deviceSparse)
		expanded = True
	if(updateBackup and databaseNetworkObject.globalFeatureNeuronsBackup is not None):
		if(databaseNetworkObject.globalFeatureNeuronsBackup.shape[3] < databaseNetworkObject.c or databaseNetworkObject.globalFeatureNeuronsBackup.shape[4] < databaseNetworkObject.f):
			newBackupShape = (arrayNumberOfProperties, numberOfDendriticBranches, arrayNumberOfSegments, databaseNetworkObject.c, databaseNetworkObject.f)
			databaseNetworkObject.globalFeatureNeuronsBackup = databaseNetworkObject.globalFeatureNeuronsBackup.coalesce()
			backupDevice = databaseNetworkObject.globalFeatureNeuronsBackup.device
			databaseNetworkObject.globalFeatureNeuronsBackup = pt.sparse_coo_tensor(databaseNetworkObject.globalFeatureNeuronsBackup.indices(), databaseNetworkObject.globalFeatureNeuronsBackup.values(), size=newBackupShape, dtype=arrayType, device=backupDevice)
			expanded = True
	return expanded

# Initialize global feature neuron arrays if lowMem is disabled
if not lowMem:
	def initialiseFeatureNeuronsGlobal(c, f):
		globalFeatureNeurons = GIAANNproto_sparseTensors.createEmptySparseTensor((arrayNumberOfProperties, numberOfDendriticBranches, arrayNumberOfSegments, c, f))
		return globalFeatureNeurons
		
	def loadFeatureNeuronsGlobal(c, f):
		if GIAANNproto_databaseNetworkFilesExcitation.pathExists(globalFeatureNeuronsFileFull):
			globalFeatureNeurons = GIAANNproto_databaseNetworkFilesExcitation.loadFeatureNeuronsGlobalFile()
			if(debugLimitFeatures):
				globalFeatureNeurons = GIAANNproto_databaseNetworkFilesExcitation.applyDebugLimitGlobalFeatureNeuronsTensor(globalFeatureNeurons, c, f, "globalFeatureNeurons")
				if(globalFeatureNeurons.size(3) < c or globalFeatureNeurons.size(4) < f):
					print("globalFeatureNeurons.size(3) = ", globalFeatureNeurons.size(3))
					print("globalFeatureNeurons.size(4) = ", globalFeatureNeurons.size(4))
					raise RuntimeError("loadFeatureNeuronsGlobal error: debugLimitFeatures requires limits that do not exceed saved globalFeatureNeurons dimensions")
		else:
			globalFeatureNeurons = initialiseFeatureNeuronsGlobal(c, f)
			#print("initialiseFeatureNeuronsGlobal: globalFeatureNeurons = ", globalFeatureNeurons)
		return globalFeatureNeurons
		
def initialiseDatabaseNetwork():

	conceptColumnsDict = {}  # key: lemma, value: index
	conceptColumnsList = []  # list of concept column names (lemmas)
	c = 0  # current number of concept columns
	conceptFeaturesDict = {}  # key: word, value: index
	conceptFeaturesList = []  # list of concept feature names (words)
	f = 0  # current number of concept features
	conceptFeaturesReferenceSetDelimiterList = []
	conceptFeaturesReferenceSetDelimiterDeterministicList = []
	conceptFeaturesReferenceSetDelimiterProbabilisticList = []

	# Initialize the concept columns dictionary
	if(GIAANNproto_databaseNetworkFilesExcitation.pathExists(conceptColumnsDictFile)):
		conceptColumnsDict = GIAANNproto_databaseNetworkFilesExcitation.loadDictFile(conceptColumnsDictFile)
		c = len(conceptColumnsDict)
		conceptColumnsList = list(conceptColumnsDict.keys())
		conceptFeaturesDict = GIAANNproto_databaseNetworkFilesExcitation.loadDictFile(conceptFeaturesDictFile)
		f = len(conceptFeaturesDict)
		conceptFeaturesList = list(conceptFeaturesDict.keys())
		if(conceptColumnsDelimitByPOS):
			if(detectReferenceSetDelimitersBetweenNouns):	
				conceptFeaturesReferenceSetDelimiterDeterministicDict = GIAANNproto_databaseNetworkFilesExcitation.loadDictFile(conceptFeaturesReferenceSetDelimiterDeterministicListFile)
				conceptFeaturesReferenceSetDelimiterDeterministicList = list(conceptFeaturesReferenceSetDelimiterDeterministicDict.values())
				conceptFeaturesReferenceSetDelimiterProbabilisticDict = GIAANNproto_databaseNetworkFilesExcitation.loadDictFile(conceptFeaturesReferenceSetDelimiterProbabilisticListFile)
				conceptFeaturesReferenceSetDelimiterProbabilisticList = list(conceptFeaturesReferenceSetDelimiterProbabilisticDict.values())
			else:
				conceptFeaturesReferenceSetDelimiterDict = GIAANNproto_databaseNetworkFilesExcitation.loadDictFile(conceptFeaturesReferenceSetDelimiterListFile)
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
		globalFeatureNeurons = loadFeatureNeuronsGlobal(c, f)
	else:
		globalFeatureNeurons = None

	s = arrayNumberOfSegments
	p = arrayNumberOfProperties
		
	databaseNetworkObject = DatabaseNetworkClass(c, f, s, p, conceptColumnsDict, conceptColumnsList, conceptFeaturesDict, conceptFeaturesList, globalFeatureNeurons, conceptFeaturesReferenceSetDelimiterList, conceptFeaturesReferenceSetDelimiterDeterministicList, conceptFeaturesReferenceSetDelimiterProbabilisticList)
	if(useInference and debugPrintTotalFeatures):
		print("c = ", databaseNetworkObject.c)
		print("f = ", databaseNetworkObject.f)
	
	return databaseNetworkObject
	

# Define the ObservedColumn class
class ObservedColumn:
	"""
	Create a class defining observed columns. The observed column class contains an index to the dataset concept column dictionary. The observed column class contains a list of feature connection arrays. The observed column class also contains a list of feature neuron arrays when lowMem mode is enabled.
	"""
	def __init__(self, databaseNetworkObject, conceptIndex, lemma, i):
		self.databaseNetworkObject = databaseNetworkObject
		self.conceptIndex = conceptIndex  # Index to the concept columns dictionary
		self.conceptName = lemma
		self.conceptSequenceWordIndex = i	#not currently used (use SequenceObservedColumns observed_columns_sequence_word_index_dict instead)
		
		if lowMem:
			# If lowMem is enabled, the observed columns contain a list of arrays (pytorch) of f feature neurons, where f is the maximum number of feature neurons per column.
			self.featureNeurons = self.initialiseFeatureNeurons(databaseNetworkObject.f)

		# Map from feature words to indices in feature neurons
		if(trainStoreFeatureMapsGlobally):
			self.featureWordToIndex = databaseNetworkObject.conceptFeaturesDict
			self.featureIndexToWord = databaseNetworkObject.conceptFeaturesIndexToWordDict
			self.nextFeatureIndex = len(databaseNetworkObject.conceptFeaturesDict) - 1
		else:
			self.featureWordToIndex = {}  # Maps feature words to indices
			self.featureIndexToWord = {}  # Maps indices to feature words
			if(useDedicatedConceptNames):
				self.nextFeatureIndex = 1  # Start from 1 since index 0 is reserved for prime concept neuron
				if(useDedicatedConceptNames2):
					self.featureWordToIndex[variablePrimeConceptFeatureNeuronName] = featureIndexPrimeConceptNeuron
					self.featureIndexToWord[featureIndexPrimeConceptNeuron] = variablePrimeConceptFeatureNeuronName
			
		# Store all connections for each source column in a list of integer feature connection arrays, each of size f * c * f, where c is the length of the dictionary of columns, and f is the maximum number of feature neurons.
		self.featureConnections = self.initialiseFeatureConnections(databaseNetworkObject.c, databaseNetworkObject.f) 

		if(not trainStoreFeatureMapsGlobally):
			self.nextFeatureIndex = 0
			for featureIndex in range(1, databaseNetworkObject.f, 1):
				featureWord = databaseNetworkObject.conceptFeaturesList[featureIndex]
				self.featureWordToIndex[featureWord] = featureIndex
				self.featureIndexToWord[featureIndex] = featureWord
				self.nextFeatureIndex += 1
					
	@staticmethod
	def initialiseFeatureNeurons(f):
		featureNeurons = GIAANNproto_sparseTensors.createEmptySparseTensor((arrayNumberOfProperties, numberOfDendriticBranches, arrayNumberOfSegments, f))
		return featureNeurons

	@staticmethod
	def initialiseFeatureConnections(c, f):
		featureConnections = GIAANNproto_sparseTensors.createEmptySparseTensor((arrayNumberOfProperties, numberOfDendriticBranches, arrayNumberOfSegments, f, c, f))
		return featureConnections
	
	def resizeConceptArrays(self, newC):
		loadC = self.featureConnections.shape[4]
		if newC > loadC:
			self.featureConnections = self.featureConnections.coalesce()
			expandedSize = (self.featureConnections.shape[0], self.featureConnections.shape[1], self.featureConnections.shape[2], self.featureConnections.shape[3], newC, self.featureConnections.shape[5])
			self.featureConnections = pt.sparse_coo_tensor(self.featureConnections.indices(), self.featureConnections.values(), size=expandedSize, dtype=arrayType, device=deviceSparse)
		
	def expandFeatureArrays(self, newF):
		loadF = self.featureConnections.shape[3]  # or self.featureConnections.shape[5]
		if newF > loadF:
			self.featureConnections = self.featureConnections.coalesce()
			expandedSizeConnections = (self.featureConnections.shape[0], self.featureConnections.shape[1], self.featureConnections.shape[2], newF, self.featureConnections.shape[4], newF)
			self.featureConnections = pt.sparse_coo_tensor(self.featureConnections.indices(), self.featureConnections.values(), size=expandedSizeConnections, dtype=arrayType, device=deviceSparse)
	
			if lowMem:
				expandedSizeNeurons = (self.featureNeurons.shape[0], self.featureNeurons.shape[1], self.featureNeurons.shape[2], newF)
				self.featureNeurons = self.featureNeurons.coalesce()
				self.featureNeurons = pt.sparse_coo_tensor(self.featureNeurons.indices(), self.featureNeurons.values(), size=expandedSizeNeurons, dtype=arrayType, device=deviceSparse)

			if(trainStoreFeatureMapsGlobally):
				self.nextFeatureIndex = len(self.databaseNetworkObject.conceptFeaturesDict) - 1
			else:
				for featureIndex in range(loadF, newF):
					featureWord = self.databaseNetworkObject.conceptFeaturesList[featureIndex]
					self.featureWordToIndex[featureWord] = featureIndex
					self.featureIndexToWord[featureIndex] = featureWord
					self.nextFeatureIndex += 1

	def saveToDisk(self):
		GIAANNproto_databaseNetworkFilesExcitation.observedColumnSaveToDisk(self)

	@classmethod
	def loadFromDisk(cls, databaseNetworkObject, conceptIndex, lemma, i):
		return GIAANNproto_databaseNetworkFilesExcitation.observedColumnLoadFromDisk(cls, databaseNetworkObject, conceptIndex, lemma, i)
		

class ObservedColumnStub:
	"""
	Minimal observed column placeholder for inference-only sequence indexing.
	"""
	def __init__(self, databaseNetworkObject, conceptIndex, lemma, i):
		self.databaseNetworkObject = databaseNetworkObject
		self.conceptIndex = conceptIndex
		self.conceptName = lemma
		self.conceptSequenceWordIndex = i

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
	
def loadOrCreateObservedColumn(databaseNetworkObject, conceptIndex, lemma, i):
	observedColumnFile = observedColumnsDir + '/' + f"{conceptIndex}_data.pkl"
	observedColumn = None
	if(storeDatabaseInRam):
		if(databaseNetworkObject.observedColumnsDictRAM is None):
			databaseNetworkObject.observedColumnsDictRAM = {}
		if(lemma in databaseNetworkObject.observedColumnsDictRAM):
			observedColumn = databaseNetworkObject.observedColumnsDictRAM[lemma]
		else:
			if(databaseNetworkObject.observedColumnsRAMLoaded):
				observedColumn = ObservedColumn(databaseNetworkObject, conceptIndex, lemma, i)
			else:
				if GIAANNproto_databaseNetworkFilesExcitation.pathExists(observedColumnFile):
					observedColumn = ObservedColumn.loadFromDisk(databaseNetworkObject, conceptIndex, lemma, i)
				else:
					observedColumn = ObservedColumn(databaseNetworkObject, conceptIndex, lemma, i)
			databaseNetworkObject.observedColumnsDictRAM[lemma] = observedColumn
		observedColumn.resizeConceptArrays(databaseNetworkObject.c)
		observedColumn.expandFeatureArrays(databaseNetworkObject.f)
	else:
		if GIAANNproto_databaseNetworkFilesExcitation.pathExists(observedColumnFile):
			observedColumn = ObservedColumn.loadFromDisk(databaseNetworkObject, conceptIndex, lemma, i)
			# Resize connection arrays if c has increased
			observedColumn.resizeConceptArrays(databaseNetworkObject.c)
			# Also expand feature arrays if f has increased
			observedColumn.expandFeatureArrays(databaseNetworkObject.f)
		else:
			observedColumn = ObservedColumn(databaseNetworkObject, conceptIndex, lemma, i)
			# Initialize connection arrays with correct size
			observedColumn.resizeConceptArrays(databaseNetworkObject.c)
			observedColumn.expandFeatureArrays(databaseNetworkObject.f)
	return observedColumn

def generateGlobalFeatureConnections(databaseNetworkObject):
	conceptColumnsListTemp = []
	for i, (lemma, conceptIndex) in enumerate(databaseNetworkObject.conceptColumnsDict.items()):
		conceptColumn = loadOrCreateObservedColumn(databaseNetworkObject, conceptIndex, lemma, i)
		conceptColumnsListTemp.append(conceptColumn)
	globalFeatureConnectionsList = []
	for conceptColumn in conceptColumnsListTemp:
		globalFeatureConnectionsList.append(conceptColumn.featureConnections)
	databaseNetworkObject.globalFeatureConnections = pt.stack(globalFeatureConnectionsList, dim=3)
	print("generate_global_feature_connections: databaseNetworkObject.global_feature_connections.shape = ", databaseNetworkObject.globalFeatureConnections.shape)

def loadAllColumns(databaseNetworkObject):
	observedColumnsDict = {}
	for i, (lemma, conceptIndex) in enumerate(databaseNetworkObject.conceptColumnsDict.items()):
		conceptColumn = loadOrCreateObservedColumn(databaseNetworkObject, conceptIndex, lemma, i)
		observedColumnsDict[lemma] = conceptColumn
	return observedColumnsDict

def loadAllObservedColumnsToRam(databaseNetworkObject):
	if(storeDatabaseInRam):
		observedColumnsDict = loadAllColumns(databaseNetworkObject)
		databaseNetworkObject.observedColumnsDictRAM = observedColumnsDict
		databaseNetworkObject.observedColumnsRAMLoaded = True
	else:
		raise RuntimeError("loadAllObservedColumnsToRam error: storeDatabaseInRam is False")
	return

def saveAllObservedColumnsToDisk(databaseNetworkObject):
	if(storeDatabaseInRam):
		if(databaseNetworkObject.observedColumnsDictRAM is None):
			raise RuntimeError("saveAllObservedColumnsToDisk error: observedColumnsDictRAM is None")
		for observedColumn in databaseNetworkObject.observedColumnsDictRAM.values():
			observedColumn.saveToDisk()
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
