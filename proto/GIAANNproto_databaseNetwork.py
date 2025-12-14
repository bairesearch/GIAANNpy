"""GIAANNproto_databaseNetwork.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

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
		self.globalFeatureNeurons = globalFeatureNeurons
		self.globalFeatureConnections = None #transformerUseInputConnections: initialised during prediction phase
		if(conceptColumnsDelimitByPOS):
			if(detectReferenceSetDelimitersBetweenNouns):
				self.conceptFeaturesReferenceSetDelimiterDeterministicList = conceptFeaturesReferenceSetDelimiterDeterministicList
				self.conceptFeaturesReferenceSetDelimiterProbabilisticList = conceptFeaturesReferenceSetDelimiterProbabilisticList
			else:
				self.conceptFeaturesReferenceSetDelimiterList = conceptFeaturesReferenceSetDelimiterList

def backupGlobalArrays(databaseNetworkObject):
	databaseNetworkObject.globalFeatureNeuronsBackup = databaseNetworkObject.globalFeatureNeurons.clone()
	if(databaseNetworkObject.globalFeatureConnections is not None):
		databaseNetworkObject.globalFeatureConnectionsBackup = databaseNetworkObject.globalFeatureConnections.clone()
	else:
		databaseNetworkObject.globalFeatureConnectionsBackup = None
		
def restoreGlobalArrays(databaseNetworkObject):
	databaseNetworkObject.globalFeatureNeurons = databaseNetworkObject.globalFeatureNeuronsBackup
	databaseNetworkObject.globalFeatureConnections = databaseNetworkObject.globalFeatureConnectionsBackup

# Initialize global feature neuron arrays if lowMem is disabled
if not lowMem:
	def initialiseFeatureNeuronsGlobal(c, f):
		globalFeatureNeurons = GIAANNproto_sparseTensors.createEmptySparseTensor((arrayNumberOfProperties, arrayNumberOfSegments, c, f))
		return globalFeatureNeurons
		
	def loadFeatureNeuronsGlobal(c, f):
		if GIAANNproto_databaseNetworkFiles.pathExists(globalFeatureNeuronsFileFull):
			globalFeatureNeurons = GIAANNproto_databaseNetworkFiles.loadFeatureNeuronsGlobalFile()
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
	if(GIAANNproto_databaseNetworkFiles.pathExists(conceptColumnsDictFile)):
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
	else:
		if(useDedicatedConceptNames):
			# Add dummy feature for concept neuron (different per concept column)
			conceptFeaturesList.append(variableConceptNeuronFeatureName)
			conceptFeaturesDict[variableConceptNeuronFeatureName] = len(conceptFeaturesDict)
			f += 1  # Will be updated dynamically based on c

		if useDedicatedFeatureLists:
			print("error: useDedicatedFeatureLists case not yet coded - need to set f and populate concept_features_list/conceptFeaturesDict etc")
			exit()
			# f = max_num_non_nouns + 1  # Maximum number of non-nouns in an English dictionary, plus the concept neuron of each column

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
		self.featureWordToIndex = {}  # Maps feature words to indices
		self.featureIndexToWord = {}  # Maps indices to feature words
		if(useDedicatedConceptNames):
			self.nextFeatureIndex = 1  # Start from 1 since index 0 is reserved for concept neuron
			if(useDedicatedConceptNames2):
				self.featureWordToIndex[variableConceptNeuronFeatureName] = featureIndexConceptNeuron
				self.featureIndexToWord[featureIndexConceptNeuron] = variableConceptNeuronFeatureName
			
		# Store all connections for each source column in a list of integer feature connection arrays, each of size f * c * f, where c is the length of the dictionary of columns, and f is the maximum number of feature neurons.
		self.featureConnections = self.initialiseFeatureConnections(databaseNetworkObject.c, databaseNetworkObject.f) 

		self.nextFeatureIndex = 0
		for featureIndex in range(1, databaseNetworkObject.f, 1):
			featureWord = databaseNetworkObject.conceptFeaturesList[featureIndex]
			self.featureWordToIndex[featureWord] = featureIndex
			self.featureIndexToWord[featureIndex] = featureWord
			self.nextFeatureIndex += 1
					
	@staticmethod
	def initialiseFeatureNeurons(f):
		featureNeurons = GIAANNproto_sparseTensors.createEmptySparseTensor((arrayNumberOfProperties, arrayNumberOfSegments, f))
		return featureNeurons

	@staticmethod
	def initialiseFeatureConnections(c, f):
		featureConnections = GIAANNproto_sparseTensors.createEmptySparseTensor((arrayNumberOfProperties, arrayNumberOfSegments, f, c, f))
		return featureConnections
	
	def resizeConceptArrays(self, newC):
		loadC = self.featureConnections.shape[3]
		if newC > loadC:
			expandedSize = (self.featureConnections.shape[0], self.featureConnections.shape[1], self.featureConnections.shape[2], newC, self.featureConnections.shape[4])
			self.featureConnections = pt.sparse_coo_tensor(self.featureConnections.indices(), self.featureConnections.values(), size=expandedSize, dtype=arrayType, device=deviceSparse)
		
	def expandFeatureArrays(self, newF):
		loadF = self.featureConnections.shape[2]  # or self.featureConnections.shape[4]
		if newF > loadF:
			self.featureConnections = self.featureConnections.coalesce()
			expandedSizeConnections = (self.featureConnections.shape[0], self.featureConnections.shape[1], newF, self.featureConnections.shape[3], newF)
			self.featureConnections = pt.sparse_coo_tensor(self.featureConnections.indices(), self.featureConnections.values(), size=expandedSizeConnections, dtype=arrayType, device=deviceSparse)
	
			if lowMem:
				expandedSizeNeurons = (self.featureNeurons.shape[0], self.featureNeurons.shape[1], newF)
				self.featureNeurons = self.featureNeurons.coalesce()
				self.featureNeurons = pt.sparse_coo_tensor(self.featureNeurons.indices(), self.featureNeurons.values(), size=expandedSizeNeurons, dtype=arrayType, device=deviceSparse)

			for featureIndex in range(loadF, newF):
				featureWord = self.databaseNetworkObject.conceptFeaturesList[featureIndex]
				self.featureWordToIndex[featureWord] = featureIndex
				self.featureIndexToWord[featureIndex] = featureWord
				self.nextFeatureIndex += 1

	def saveToDisk(self):
		GIAANNproto_databaseNetworkFiles.observedColumnSaveToDisk(self)

	@classmethod
	def loadFromDisk(cls, databaseNetworkObject, conceptIndex, lemma, i):
		return GIAANNproto_databaseNetworkFiles.observedColumnLoadFromDisk(cls, databaseNetworkObject, conceptIndex, lemma, i)
		

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
	if GIAANNproto_databaseNetworkFiles.pathExists(observedColumnFile):
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
	databaseNetworkObject.globalFeatureConnections = pt.stack(globalFeatureConnectionsList, dim=2)
	print("generate_global_feature_connections: databaseNetworkObject.global_feature_connections.shape = ", databaseNetworkObject.globalFeatureConnections.shape)

def loadAllColumns(databaseNetworkObject):
	observedColumnsDict = {}
	for i, (lemma, conceptIndex) in enumerate(databaseNetworkObject.conceptColumnsDict.items()):
		conceptColumn = loadOrCreateObservedColumn(databaseNetworkObject, conceptIndex, lemma, i)
		observedColumnsDict[lemma] = conceptColumn
	return observedColumnsDict

'''
def getTokenConceptFeatureIndexForSequenceConceptIndex(sequence_observed_columns, words_sequence, concept_mask, sequenceConceptIndex, sequenceWordIndex):
	conceptIndex = sequence_observed_columns.sequence_observed_columns_dict[sequenceConceptIndex].conceptIndex
	if(concept_mask[sequenceWordIndex]):
		feature_index = featureIndexConceptNeuron
	else:
		feature_index = sequence_observed_columns.featureWordToIndex[words_sequence[sequenceWordIndex]]
	return conceptIndex, feature_index
'''

def getTokenConceptFeatureIndexTensor(sequenceObservedColumns, tokensSequence, conceptMask, sequenceWordIndex, kcMax):
	targetFoundNextColumnIndex, targetPreviousColumnIndex, targetNextColumnIndex, targetFeatureIndex = getTokenConceptFeatureIndex(sequenceObservedColumns, tokensSequence, conceptMask, sequenceWordIndex)

	if(kcMax == 1 or not targetFoundNextColumnIndex):
		targetConceptColumnsIndices = pt.tensor(targetPreviousColumnIndex).unsqueeze(0)
		targetConceptColumnsFeatureIndices = pt.tensor(targetFeatureIndex).unsqueeze(0).unsqueeze(0)
		targetMultipleSources = False
	elif(kcMax == 2 and targetFoundNextColumnIndex): 
		targetConceptColumnsIndices = pt.tensor([targetPreviousColumnIndex, targetNextColumnIndex])
		targetConceptColumnsFeatureIndices = pt.stack([pt.tensor(targetFeatureIndex).unsqueeze(0), pt.tensor(targetFeatureIndex).unsqueeze(0)], dim=0)
		targetMultipleSources = True
	else:
		printe("getTokenConceptFeatureIndexTensor currently requires kcMax == 1 or 2; corresponding to the number of target columns per token; check conceptColumnsDelimitByConceptFeaturesStart/multipleTargets")

	return targetMultipleSources, targetPreviousColumnIndex, targetNextColumnIndex, targetFeatureIndex, targetConceptColumnsIndices, targetConceptColumnsFeatureIndices

def getTokenConceptFeatureIndex(sequenceObservedColumns, tokensSequence, conceptMask, sequenceWordIndex):
	databaseNetworkObject = sequenceObservedColumns.databaseNetworkObject
	columnsIndexSequenceWordIndexDict = sequenceObservedColumns.columnsIndexSequenceWordIndexDict
	
	if(conceptMask[sequenceWordIndex]):
		targetFeatureIndex = featureIndexConceptNeuron
	else:
		word = tokensSequence[sequenceWordIndex].word
		targetFeatureIndex = databaseNetworkObject.conceptFeaturesDict[word]
	sequenceLen = conceptMask.shape[0]
	foundFeature = False
	conceptFeature = False
	targetFoundNextColumnIndex = False
	targetPreviousColumnIndex = 0
	targetNextColumnIndex = 0
	for i in range(sequenceLen):
		if(foundFeature):
			if(not conceptFeature):
				if(not targetFoundNextColumnIndex):
					if(conceptMask[i] != 0):
						targetNextColumnIndex = columnsIndexSequenceWordIndexDict[i]
						targetFoundNextColumnIndex = True
		else:
			if(conceptMask[i] != 0):
				targetPreviousColumnIndex = columnsIndexSequenceWordIndexDict[i]
		if(i == sequenceWordIndex):
			foundFeature = True
			if(conceptMask[i] != 0):
				conceptFeature = True
	
	return targetFoundNextColumnIndex, targetPreviousColumnIndex, targetNextColumnIndex, targetFeatureIndex

def isFeatureIndexReferenceSetDelimiterDeterministic(databaseNetworkObject, featureIndex):
	if(conceptColumnsDelimitByPOS):
		if(detectReferenceSetDelimitersBetweenNouns):
			isDelimiter = databaseNetworkObject.conceptFeaturesReferenceSetDelimiterDeterministicList[featureIndex]
		else:
			isDelimiter = databaseNetworkObject.conceptFeaturesReferenceSetDelimiterList[featureIndex]
	else:
		isDelimiter = False
	return isDelimiter

def isFeatureIndexReferenceSetDelimiterProbabilistic(databaseNetworkObject, featureIndex):
	if(conceptColumnsDelimitByPOS):
		if(detectReferenceSetDelimitersBetweenNouns):
			isDelimiterProbabilistic = databaseNetworkObject.conceptFeaturesReferenceSetDelimiterProbabilisticList[featureIndex]
		else:
			isDelimiterProbabilistic = False
	else:
		isDelimiterProbabilistic = False
	return isDelimiterProbabilistic
