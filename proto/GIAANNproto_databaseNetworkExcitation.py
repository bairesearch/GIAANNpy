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
		databaseNetworkObject.globalFeatureNeurons = databaseNetworkObject.globalFeatureNeurons.coalesce()
		databaseNetworkObject.globalFeatureNeurons = pt.sparse_coo_tensor(databaseNetworkObject.globalFeatureNeurons.indices(), databaseNetworkObject.globalFeatureNeurons.values(), size=newShape, dtype=arrayType, device=deviceSparse)
		expanded = True
	if(updateBackup and databaseNetworkObject.globalFeatureNeuronsBackup is not None):
		if(databaseNetworkObject.globalFeatureNeuronsBackup.shape[3] < databaseNetworkObject.c or databaseNetworkObject.globalFeatureNeuronsBackup.shape[4] < databaseNetworkObject.f):
			newBackupShape = (databaseNetworkObject.arrayNumberOfProperties, numberOfDendriticBranches, arrayNumberOfSegments, databaseNetworkObject.c, databaseNetworkObject.f)
			databaseNetworkObject.globalFeatureNeuronsBackup = databaseNetworkObject.globalFeatureNeuronsBackup.coalesce()
			backupDevice = databaseNetworkObject.globalFeatureNeuronsBackup.device
			databaseNetworkObject.globalFeatureNeuronsBackup = pt.sparse_coo_tensor(databaseNetworkObject.globalFeatureNeuronsBackup.indices(), databaseNetworkObject.globalFeatureNeuronsBackup.values(), size=newBackupShape, dtype=arrayType, device=backupDevice)
			expanded = True
	return expanded

# Initialize global feature neuron arrays if lowMem is disabled
if not lowMem:
	def initialiseFeatureNeuronsGlobal(inferenceMode, c, f):
		arrayNumberOfProperties = calculateArrayNumberOfProperties(inferenceMode)
		globalFeatureNeurons = GIAANNproto_sparseTensors.createEmptySparseTensor((arrayNumberOfProperties, numberOfDendriticBranches, arrayNumberOfSegments, c, f))
		return globalFeatureNeurons
		
	def loadFeatureNeuronsGlobal(inferenceMode, c, f):
		if GIAANNproto_databaseNetworkFilesExcitation.pathExists(globalFeatureNeuronsFileFull):
			globalFeatureNeurons = GIAANNproto_databaseNetworkFilesExcitation.loadFeatureNeuronsGlobalFile(inferenceMode)
			if(debugLimitFeatures):
				globalFeatureNeurons = GIAANNproto_databaseNetworkFilesExcitation.applyDebugLimitGlobalFeatureNeuronsTensor(globalFeatureNeurons, c, f, "globalFeatureNeurons")
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
	loadExistingDatabase = inferenceMode or (trainLoadExistingDatabase and GIAANNproto_databaseNetworkFilesExcitation.pathExists(conceptColumnsDictFile))

	# Initialize the concept columns dictionary
	if(loadExistingDatabase and GIAANNproto_databaseNetworkFilesExcitation.pathExists(conceptColumnsDictFile)):
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

		self.featureConnectionsBySourceFeature = {}
		self.loadedSourceFeatureIndices = set()

		if(not trainStoreFeatureMapsGlobally):
			self.nextFeatureIndex = 0
			for featureIndex in range(1, databaseNetworkObject.f, 1):
				featureWord = databaseNetworkObject.conceptFeaturesList[featureIndex]
				self.featureWordToIndex[featureWord] = featureIndex
				self.featureIndexToWord[featureIndex] = featureWord
				self.nextFeatureIndex += 1
					
	#@staticmethod
	def initialiseFeatureNeurons(self, f, targetDevice=None):
		deviceTarget = targetDevice if targetDevice is not None else (deviceDatabase if storeDatabaseInRam else deviceSparse)
		indices = pt.empty((4, 0), dtype=pt.long, device=deviceTarget)
		values = pt.empty((0,), dtype=arrayType, device=deviceTarget)
		featureNeurons = pt.sparse_coo_tensor(indices, values, size=(self.databaseNetworkObject.arrayNumberOfProperties, numberOfDendriticBranches, arrayNumberOfSegments, f), dtype=arrayType, device=deviceTarget)
		return featureNeurons

	#@staticmethod
	def initialiseFeatureConnections(self, c, f, targetDevice=None):
		deviceTarget = targetDevice if targetDevice is not None else self.getDefaultConnectionTargetDevice()
		indices = pt.empty((5, 0), dtype=pt.long, device=deviceTarget)
		values = pt.empty((0,), dtype=arrayType, device=deviceTarget)
		featureConnections = pt.sparse_coo_tensor(indices, values, size=(self.databaseNetworkObject.arrayNumberOfProperties, numberOfDendriticBranches, arrayNumberOfSegments, c, f), dtype=arrayType, device=deviceTarget)
		return featureConnections

	def getDefaultConnectionTargetDevice(self):
		deviceTarget = deviceDatabase if storeDatabaseInRam else deviceSparse
		if(len(self.featureConnectionsBySourceFeature) > 0):
			firstTensor = next(iter(self.featureConnectionsBySourceFeature.values()))
			deviceTarget = firstTensor.device
		return deviceTarget

	def getFeatureConnectionsTargetSize(self, c=None, f=None):
		targetC = c if c is not None else self.databaseNetworkObject.c
		targetF = f if f is not None else self.databaseNetworkObject.f
		targetSize = (self.databaseNetworkObject.arrayNumberOfProperties, numberOfDendriticBranches, arrayNumberOfSegments, targetC, targetF)
		return targetSize

	def getMaterialisedFeatureConnectionsTargetSize(self, c=None, f=None):
		targetC = c if c is not None else self.databaseNetworkObject.c
		targetF = f if f is not None else self.databaseNetworkObject.f
		targetSize = (self.databaseNetworkObject.arrayNumberOfProperties, numberOfDendriticBranches, arrayNumberOfSegments, targetF, targetC, targetF)
		return targetSize

	def normaliseSourceFeatureIndex(self, sourceFeatureIndex):
		result = int(sourceFeatureIndex)
		if(result < 0 or result >= self.databaseNetworkObject.f):
			raise RuntimeError(f"ObservedColumn source feature index out of range: {result}")
		return result

	def normaliseSourceFeatureIndices(self, requiredSourceFeatureIndices):
		indicesList = []
		seen = set()
		if(requiredSourceFeatureIndices is not None):
			if(pt.is_tensor(requiredSourceFeatureIndices)):
				rawIndices = requiredSourceFeatureIndices.detach().view(-1).cpu().tolist()
			else:
				rawIndices = list(requiredSourceFeatureIndices)
			for sourceFeatureIndex in rawIndices:
				normalisedSourceFeatureIndex = self.normaliseSourceFeatureIndex(sourceFeatureIndex)
				if(normalisedSourceFeatureIndex not in seen):
					indicesList.append(normalisedSourceFeatureIndex)
					seen.add(normalisedSourceFeatureIndex)
		indicesList.sort()
		return indicesList

	def listStoredSourceFeatureIndices(self):
		storedIndices = GIAANNproto_databaseNetworkFilesExcitation.listObservedColumnSourceFeatureIndices(self.conceptIndex)
		combinedIndices = set(storedIndices)
		for sourceFeatureIndex in self.featureConnectionsBySourceFeature.keys():
			combinedIndices.add(self.normaliseSourceFeatureIndex(sourceFeatureIndex))
		result = sorted(combinedIndices)
		return result

	def loadRequiredSourceFeatureConnections(self, requiredSourceFeatureIndices, targetDevice, createMissing=False):
		resolvedTargetDevice = targetDevice if targetDevice is not None else self.getDefaultConnectionTargetDevice()
		sourceFeatureIndices = self.normaliseSourceFeatureIndices(requiredSourceFeatureIndices)
		for sourceFeatureIndex in sourceFeatureIndices:
			self.getFeatureConnectionsForSourceFeature(sourceFeatureIndex, resolvedTargetDevice, createMissing)
		return

	def getFeatureConnectionsForSourceFeature(self, sourceFeatureIndex, targetDevice=None, createMissing=False):
		normalisedSourceFeatureIndex = self.normaliseSourceFeatureIndex(sourceFeatureIndex)
		resolvedTargetDevice = targetDevice if targetDevice is not None else self.getDefaultConnectionTargetDevice()
		result = self.featureConnectionsBySourceFeature.get(normalisedSourceFeatureIndex)
		if(result is None):
			storedSourceFeatureIndices = self.listStoredSourceFeatureIndices()
			if(normalisedSourceFeatureIndex in storedSourceFeatureIndices):
				result = GIAANNproto_databaseNetworkFilesExcitation.loadObservedColumnSourceFeatureConnectionsTensor(self.databaseNetworkObject, self.conceptIndex, normalisedSourceFeatureIndex, resolvedTargetDevice)
			else:
				if(not createMissing):
					result = self.initialiseFeatureConnections(self.databaseNetworkObject.c, self.databaseNetworkObject.f, resolvedTargetDevice)
				else:
					result = self.initialiseFeatureConnections(self.databaseNetworkObject.c, self.databaseNetworkObject.f, resolvedTargetDevice)
			self.featureConnectionsBySourceFeature[normalisedSourceFeatureIndex] = result
		elif(result.device != resolvedTargetDevice):
			result = result.to(resolvedTargetDevice)
			self.featureConnectionsBySourceFeature[normalisedSourceFeatureIndex] = result
		self.loadedSourceFeatureIndices.add(normalisedSourceFeatureIndex)
		return result

	def setFeatureConnectionsForSourceFeature(self, sourceFeatureIndex, tensor):
		normalisedSourceFeatureIndex = self.normaliseSourceFeatureIndex(sourceFeatureIndex)
		expectedSize = self.getFeatureConnectionsTargetSize()
		if(tensor is None):
			raise RuntimeError("setFeatureConnectionsForSourceFeature error: tensor is None")
		if(tensor.layout != pt.sparse_coo):
			raise RuntimeError("setFeatureConnectionsForSourceFeature error: tensor must be sparse COO")
		if(tensor.dim() != 5):
			raise RuntimeError("setFeatureConnectionsForSourceFeature error: tensor rank must be 5")
		if(tuple(tensor.size()) != tuple(expectedSize)):
			raise RuntimeError(f"setFeatureConnectionsForSourceFeature error: tensor size {tuple(tensor.size())} does not match expected size {tuple(expectedSize)}")
		if(not tensor.is_coalesced()):
			tensor = tensor.coalesce()
		self.featureConnectionsBySourceFeature[normalisedSourceFeatureIndex] = tensor
		self.loadedSourceFeatureIndices.add(normalisedSourceFeatureIndex)
		return

	def saveLoadedSourceFeatureConnectionsToDisk(self):
		sourceFeatureIndices = sorted(self.loadedSourceFeatureIndices)
		for sourceFeatureIndex in sourceFeatureIndices:
			if(sourceFeatureIndex not in self.featureConnectionsBySourceFeature):
				raise RuntimeError(f"saveLoadedSourceFeatureConnectionsToDisk error: missing loaded source feature tensor {sourceFeatureIndex}")
			sourceTensor = self.featureConnectionsBySourceFeature[sourceFeatureIndex]
			GIAANNproto_databaseNetworkFilesExcitation.saveObservedColumnSourceFeatureConnectionsTensor(self.conceptIndex, sourceFeatureIndex, sourceTensor)
		return

	def unloadLoadedSourceFeatureConnections(self, sourceFeatureIndices=None):
		indicesToUnload = self.normaliseSourceFeatureIndices(sourceFeatureIndices) if sourceFeatureIndices is not None else sorted(self.loadedSourceFeatureIndices)
		for sourceFeatureIndex in indicesToUnload:
			if(sourceFeatureIndex in self.featureConnectionsBySourceFeature):
				del self.featureConnectionsBySourceFeature[sourceFeatureIndex]
			if(sourceFeatureIndex in self.loadedSourceFeatureIndices):
				self.loadedSourceFeatureIndices.remove(sourceFeatureIndex)
		return

	def materialiseFeatureConnections(self, loadAllStored=False, targetDevice=None):
		resolvedTargetDevice = targetDevice if targetDevice is not None else self.getDefaultConnectionTargetDevice()
		sourceFeatureIndices = sorted(self.featureConnectionsBySourceFeature.keys())
		if(loadAllStored):
			sourceFeatureIndices = self.listStoredSourceFeatureIndices()
		combinedIndicesList = []
		combinedValuesList = []
		for sourceFeatureIndex in sourceFeatureIndices:
			sourceTensor = self.getFeatureConnectionsForSourceFeature(sourceFeatureIndex, resolvedTargetDevice, createMissing=False)
			sourceTensor = sourceTensor.coalesce()
			if(sourceTensor._nnz() > 0):
				sourceIndices = sourceTensor.indices()
				sourceValues = sourceTensor.values()
				sourceRow = pt.full((1, sourceIndices.shape[1]), sourceFeatureIndex, dtype=pt.long, device=sourceIndices.device)
				materialisedIndices = pt.cat([sourceIndices[0:3], sourceRow, sourceIndices[3:]], dim=0)
				combinedIndicesList.append(materialisedIndices)
				combinedValuesList.append(sourceValues)
		targetSize = self.getMaterialisedFeatureConnectionsTargetSize()
		if(len(combinedIndicesList) > 0):
			combinedIndices = pt.cat(combinedIndicesList, dim=1)
			combinedValues = pt.cat(combinedValuesList, dim=0)
		else:
			combinedIndices = pt.empty((6, 0), dtype=pt.long, device=resolvedTargetDevice)
			combinedValues = pt.empty((0,), dtype=arrayType, device=resolvedTargetDevice)
		result = pt.sparse_coo_tensor(combinedIndices, combinedValues, size=targetSize, dtype=arrayType, device=resolvedTargetDevice).coalesce()
		return result

	def setMaterialisedFeatureConnections(self, featureConnections, sourceFeatureIndices=None):
		if(featureConnections is None):
			raise RuntimeError("setMaterialisedFeatureConnections error: featureConnections is None")
		if(featureConnections.layout != pt.sparse_coo):
			raise RuntimeError("setMaterialisedFeatureConnections error: featureConnections must be sparse COO")
		if(featureConnections.dim() != 6):
			raise RuntimeError("setMaterialisedFeatureConnections error: featureConnections rank must be 6")
		featureConnections = featureConnections.coalesce()
		connectionIndices = featureConnections.indices()
		connectionValues = featureConnections.values()
		resolvedSourceFeatureIndices = self.normaliseSourceFeatureIndices(sourceFeatureIndices) if sourceFeatureIndices is not None else sorted(self.loadedSourceFeatureIndices)
		targetSize = self.getFeatureConnectionsTargetSize()
		for sourceFeatureIndex in resolvedSourceFeatureIndices:
			sourceMask = connectionIndices[3] == sourceFeatureIndex
			if(sourceMask.any()):
				sourceIndices = pt.stack((connectionIndices[0, sourceMask], connectionIndices[1, sourceMask], connectionIndices[2, sourceMask], connectionIndices[4, sourceMask], connectionIndices[5, sourceMask]), dim=0)
				sourceValues = connectionValues[sourceMask]
			else:
				sourceIndices = pt.empty((5, 0), dtype=pt.long, device=featureConnections.device)
				sourceValues = pt.empty((0,), dtype=arrayType, device=featureConnections.device)
			sourceTensor = pt.sparse_coo_tensor(sourceIndices, sourceValues, size=targetSize, dtype=arrayType, device=featureConnections.device).coalesce()
			self.setFeatureConnectionsForSourceFeature(sourceFeatureIndex, sourceTensor)
		return

	def resizeConceptArrays(self, newC):
		for sourceFeatureIndex, sourceTensor in list(self.featureConnectionsBySourceFeature.items()):
			loadC = sourceTensor.shape[3]
			if(newC > loadC):
				sourceTensor = sourceTensor.coalesce()
				expandedSize = (sourceTensor.shape[0], sourceTensor.shape[1], sourceTensor.shape[2], newC, sourceTensor.shape[4])
				sourceTensor = pt.sparse_coo_tensor(sourceTensor.indices(), sourceTensor.values(), size=expandedSize, dtype=arrayType, device=sourceTensor.device).coalesce()
				self.featureConnectionsBySourceFeature[sourceFeatureIndex] = sourceTensor
		return
		
	def expandFeatureArrays(self, newF):
		loadFReference = (self.nextFeatureIndex + 1) if (len(self.featureConnectionsBySourceFeature) == 0 and not trainStoreFeatureMapsGlobally) else None
		for sourceFeatureIndex, sourceTensor in list(self.featureConnectionsBySourceFeature.items()):
			loadF = sourceTensor.shape[4]
			if(loadFReference is None):
				loadFReference = loadF
			if(newF > loadF):
				sourceTensor = sourceTensor.coalesce()
				expandedSizeConnections = (sourceTensor.shape[0], sourceTensor.shape[1], sourceTensor.shape[2], sourceTensor.shape[3], newF)
				sourceTensor = pt.sparse_coo_tensor(sourceTensor.indices(), sourceTensor.values(), size=expandedSizeConnections, dtype=arrayType, device=sourceTensor.device).coalesce()
				self.featureConnectionsBySourceFeature[sourceFeatureIndex] = sourceTensor
		if(loadFReference is None):
			loadFReference = 0
		if lowMem:
			expandedSizeNeurons = (self.featureNeurons.shape[0], self.featureNeurons.shape[1], self.featureNeurons.shape[2], newF)
			self.featureNeurons = self.featureNeurons.coalesce()
			self.featureNeurons = pt.sparse_coo_tensor(self.featureNeurons.indices(), self.featureNeurons.values(), size=expandedSizeNeurons, dtype=arrayType, device=self.featureNeurons.device).coalesce()
		if(trainStoreFeatureMapsGlobally):
			self.nextFeatureIndex = len(self.databaseNetworkObject.conceptFeaturesDict) - 1
		else:
			for featureIndex in range(loadFReference, newF):
				featureWord = self.databaseNetworkObject.conceptFeaturesList[featureIndex]
				self.featureWordToIndex[featureWord] = featureIndex
				self.featureIndexToWord[featureIndex] = featureWord
				self.nextFeatureIndex += 1
		return

	def saveToDisk(self):
		GIAANNproto_databaseNetworkFilesExcitation.observedColumnSaveToDisk(self)
		return

	@classmethod
	def loadFromDisk(cls, databaseNetworkObject, conceptIndex, lemma, i, targetDevice=None, loadAllSourceFeatures=False):
		result = GIAANNproto_databaseNetworkFilesExcitation.observedColumnLoadFromDisk(cls, databaseNetworkObject, conceptIndex, lemma, i, targetDevice=targetDevice, loadAllSourceFeatures=loadAllSourceFeatures)
		return result
		

class ObservedColumnStub:
	"""
	Minimal observed column placeholder for inference-only sequence indexing.
	"""
	def __init__(self, databaseNetworkObject, conceptIndex, lemma, i):
		self.databaseNetworkObject = databaseNetworkObject
		self.conceptIndex = conceptIndex
		self.conceptName = lemma
		self.conceptSequenceWordIndex = i

class ObservedColumnProxy:
	"""
	Observed column proxy with copied tensors and shared feature maps.
	"""
	def __init__(self, databaseNetworkObject, observedColumn, lemma, i, targetDevice):
		self.databaseNetworkObject = databaseNetworkObject
		self.conceptIndex = observedColumn.conceptIndex
		self.conceptName = lemma
		self.conceptSequenceWordIndex = i
		self.featureWordToIndex = observedColumn.featureWordToIndex
		self.featureIndexToWord = observedColumn.featureIndexToWord
		self.nextFeatureIndex = observedColumn.nextFeatureIndex
		self.sourceObservedColumn = observedColumn
		self.proxyTargetDevice = targetDevice
		self.featureConnectionsBySourceFeature = {}
		self.loadedSourceFeatureIndices = set()
		if(lowMem and hasattr(observedColumn, "featureNeurons")):
			self.featureNeurons = observedColumn.featureNeurons.to(targetDevice)

	def getDefaultConnectionTargetDevice(self):
		deviceTarget = self.proxyTargetDevice
		return deviceTarget

	def getFeatureConnectionsTargetSize(self, c=None, f=None):
		targetC = c if c is not None else self.databaseNetworkObject.c
		targetF = f if f is not None else self.databaseNetworkObject.f
		targetSize = (self.databaseNetworkObject.arrayNumberOfProperties, numberOfDendriticBranches, arrayNumberOfSegments, targetC, targetF)
		return targetSize

	def getMaterialisedFeatureConnectionsTargetSize(self, c=None, f=None):
		targetC = c if c is not None else self.databaseNetworkObject.c
		targetF = f if f is not None else self.databaseNetworkObject.f
		targetSize = (self.databaseNetworkObject.arrayNumberOfProperties, numberOfDendriticBranches, arrayNumberOfSegments, targetF, targetC, targetF)
		return targetSize

	def normaliseSourceFeatureIndex(self, sourceFeatureIndex):
		result = int(sourceFeatureIndex)
		if(result < 0 or result >= self.databaseNetworkObject.f):
			raise RuntimeError(f"ObservedColumnProxy source feature index out of range: {result}")
		return result

	def normaliseSourceFeatureIndices(self, requiredSourceFeatureIndices):
		indicesList = []
		seen = set()
		if(requiredSourceFeatureIndices is not None):
			if(pt.is_tensor(requiredSourceFeatureIndices)):
				rawIndices = requiredSourceFeatureIndices.detach().view(-1).cpu().tolist()
			else:
				rawIndices = list(requiredSourceFeatureIndices)
			for sourceFeatureIndex in rawIndices:
				normalisedSourceFeatureIndex = self.normaliseSourceFeatureIndex(sourceFeatureIndex)
				if(normalisedSourceFeatureIndex not in seen):
					indicesList.append(normalisedSourceFeatureIndex)
					seen.add(normalisedSourceFeatureIndex)
		indicesList.sort()
		return indicesList

	def listStoredSourceFeatureIndices(self):
		result = self.sourceObservedColumn.listStoredSourceFeatureIndices()
		return result

	def loadRequiredSourceFeatureConnections(self, requiredSourceFeatureIndices, targetDevice, createMissing=False):
		resolvedTargetDevice = targetDevice if targetDevice is not None else self.proxyTargetDevice
		sourceFeatureIndices = self.normaliseSourceFeatureIndices(requiredSourceFeatureIndices)
		for sourceFeatureIndex in sourceFeatureIndices:
			self.getFeatureConnectionsForSourceFeature(sourceFeatureIndex, resolvedTargetDevice, createMissing)
		return

	def getFeatureConnectionsForSourceFeature(self, sourceFeatureIndex, targetDevice=None, createMissing=False):
		normalisedSourceFeatureIndex = self.normaliseSourceFeatureIndex(sourceFeatureIndex)
		resolvedTargetDevice = targetDevice if targetDevice is not None else self.proxyTargetDevice
		result = self.featureConnectionsBySourceFeature.get(normalisedSourceFeatureIndex)
		if(result is None):
			baseTensor = self.sourceObservedColumn.getFeatureConnectionsForSourceFeature(normalisedSourceFeatureIndex, self.sourceObservedColumn.getDefaultConnectionTargetDevice(), createMissing)
			result = baseTensor.to(resolvedTargetDevice)
			self.featureConnectionsBySourceFeature[normalisedSourceFeatureIndex] = result
		elif(result.device != resolvedTargetDevice):
			result = result.to(resolvedTargetDevice)
			self.featureConnectionsBySourceFeature[normalisedSourceFeatureIndex] = result
		self.loadedSourceFeatureIndices.add(normalisedSourceFeatureIndex)
		return result

	def setFeatureConnectionsForSourceFeature(self, sourceFeatureIndex, tensor):
		normalisedSourceFeatureIndex = self.normaliseSourceFeatureIndex(sourceFeatureIndex)
		expectedSize = self.getFeatureConnectionsTargetSize()
		if(tensor is None):
			raise RuntimeError("ObservedColumnProxy.setFeatureConnectionsForSourceFeature error: tensor is None")
		if(tensor.layout != pt.sparse_coo):
			raise RuntimeError("ObservedColumnProxy.setFeatureConnectionsForSourceFeature error: tensor must be sparse COO")
		if(tuple(tensor.size()) != tuple(expectedSize)):
			raise RuntimeError("ObservedColumnProxy.setFeatureConnectionsForSourceFeature error: tensor size mismatch")
		self.featureConnectionsBySourceFeature[normalisedSourceFeatureIndex] = tensor.coalesce()
		self.loadedSourceFeatureIndices.add(normalisedSourceFeatureIndex)
		return

	def saveLoadedSourceFeatureConnectionsToDisk(self):
		raise RuntimeError("ObservedColumnProxy cannot save source feature connections to disk")

	def unloadLoadedSourceFeatureConnections(self, sourceFeatureIndices=None):
		indicesToUnload = self.normaliseSourceFeatureIndices(sourceFeatureIndices) if sourceFeatureIndices is not None else sorted(self.loadedSourceFeatureIndices)
		for sourceFeatureIndex in indicesToUnload:
			if(sourceFeatureIndex in self.featureConnectionsBySourceFeature):
				del self.featureConnectionsBySourceFeature[sourceFeatureIndex]
			if(sourceFeatureIndex in self.loadedSourceFeatureIndices):
				self.loadedSourceFeatureIndices.remove(sourceFeatureIndex)
		return

	def materialiseFeatureConnections(self, loadAllStored=False, targetDevice=None):
		resolvedTargetDevice = targetDevice if targetDevice is not None else self.proxyTargetDevice
		sourceFeatureIndices = sorted(self.featureConnectionsBySourceFeature.keys())
		if(loadAllStored):
			sourceFeatureIndices = self.listStoredSourceFeatureIndices()
		combinedIndicesList = []
		combinedValuesList = []
		for sourceFeatureIndex in sourceFeatureIndices:
			sourceTensor = self.getFeatureConnectionsForSourceFeature(sourceFeatureIndex, resolvedTargetDevice, createMissing=False)
			sourceTensor = sourceTensor.coalesce()
			if(sourceTensor._nnz() > 0):
				sourceIndices = sourceTensor.indices()
				sourceValues = sourceTensor.values()
				sourceRow = pt.full((1, sourceIndices.shape[1]), sourceFeatureIndex, dtype=pt.long, device=sourceIndices.device)
				materialisedIndices = pt.cat([sourceIndices[0:3], sourceRow, sourceIndices[3:]], dim=0)
				combinedIndicesList.append(materialisedIndices)
				combinedValuesList.append(sourceValues)
		targetSize = self.getMaterialisedFeatureConnectionsTargetSize()
		if(len(combinedIndicesList) > 0):
			combinedIndices = pt.cat(combinedIndicesList, dim=1)
			combinedValues = pt.cat(combinedValuesList, dim=0)
		else:
			combinedIndices = pt.empty((6, 0), dtype=pt.long, device=resolvedTargetDevice)
			combinedValues = pt.empty((0,), dtype=arrayType, device=resolvedTargetDevice)
		result = pt.sparse_coo_tensor(combinedIndices, combinedValues, size=targetSize, dtype=arrayType, device=resolvedTargetDevice).coalesce()
		return result

	def setMaterialisedFeatureConnections(self, featureConnections, sourceFeatureIndices=None):
		if(featureConnections is None):
			raise RuntimeError("ObservedColumnProxy.setMaterialisedFeatureConnections error: featureConnections is None")
		if(featureConnections.layout != pt.sparse_coo):
			raise RuntimeError("ObservedColumnProxy.setMaterialisedFeatureConnections error: featureConnections must be sparse COO")
		if(featureConnections.dim() != 6):
			raise RuntimeError("ObservedColumnProxy.setMaterialisedFeatureConnections error: featureConnections rank must be 6")
		featureConnections = featureConnections.coalesce()
		connectionIndices = featureConnections.indices()
		connectionValues = featureConnections.values()
		resolvedSourceFeatureIndices = self.normaliseSourceFeatureIndices(sourceFeatureIndices) if sourceFeatureIndices is not None else sorted(self.loadedSourceFeatureIndices)
		targetSize = self.getFeatureConnectionsTargetSize()
		for sourceFeatureIndex in resolvedSourceFeatureIndices:
			sourceMask = connectionIndices[3] == sourceFeatureIndex
			if(sourceMask.any()):
				sourceIndices = pt.stack((connectionIndices[0, sourceMask], connectionIndices[1, sourceMask], connectionIndices[2, sourceMask], connectionIndices[4, sourceMask], connectionIndices[5, sourceMask]), dim=0)
				sourceValues = connectionValues[sourceMask]
			else:
				sourceIndices = pt.empty((5, 0), dtype=pt.long, device=featureConnections.device)
				sourceValues = pt.empty((0,), dtype=arrayType, device=featureConnections.device)
			sourceTensor = pt.sparse_coo_tensor(sourceIndices, sourceValues, size=targetSize, dtype=arrayType, device=featureConnections.device).coalesce()
			self.setFeatureConnectionsForSourceFeature(sourceFeatureIndex, sourceTensor)
		return

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
	GIAANNproto_databaseNetworkFilesExcitation.validateObservedColumnStorageFormat(conceptIndex)
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
				if GIAANNproto_databaseNetworkFilesExcitation.observedColumnMetadataExists(conceptIndex):
					observedColumn = ObservedColumn.loadFromDisk(databaseNetworkObject, conceptIndex, lemma, i, targetDevice=deviceDatabase, loadAllSourceFeatures=True)
				else:
					observedColumn = ObservedColumn(databaseNetworkObject, conceptIndex, lemma, i)
			databaseNetworkObject.observedColumnsDictRAM[lemma] = observedColumn
		observedColumn.resizeConceptArrays(databaseNetworkObject.c)
		observedColumn.expandFeatureArrays(databaseNetworkObject.f)
	else:
		if GIAANNproto_databaseNetworkFilesExcitation.observedColumnMetadataExists(conceptIndex):
			observedColumn = ObservedColumn.loadFromDisk(databaseNetworkObject, conceptIndex, lemma, i, targetDevice=deviceDatabase, loadAllSourceFeatures=loadAllSourceFeatures)
			observedColumn.resizeConceptArrays(databaseNetworkObject.c)
			observedColumn.expandFeatureArrays(databaseNetworkObject.f)
		else:
			observedColumn = ObservedColumn(databaseNetworkObject, conceptIndex, lemma, i)
			observedColumn.resizeConceptArrays(databaseNetworkObject.c)
			observedColumn.expandFeatureArrays(databaseNetworkObject.f)
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
	for i, (lemma, conceptIndex) in enumerate(databaseNetworkObject.conceptColumnsDict.items()):
		conceptColumn = loadOrCreateObservedColumn(databaseNetworkObject, conceptIndex, lemma, i, targetDevice=deviceDatabase, createDeviceCopy=False, loadAllSourceFeatures=True)
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
				for sourceFeatureIndex in sorted(observedColumn.loadedSourceFeatureIndices):
					sourceTensor = observedColumn.getFeatureConnectionsForSourceFeature(sourceFeatureIndex, deviceDatabase, createMissing=False)
					observedColumn.setFeatureConnectionsForSourceFeature(sourceFeatureIndex, sourceTensor)
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
		observedColumn.loadRequiredSourceFeatureConnections(requiredSourceFeatureIndices, deviceSparse, createMissing=False)
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


#if(debugCountTotalParameters):
def debugCountObservedColumnConnections(databaseNetworkObject, conceptIndex, lemma, columnIndex):
	columnConnections = 0
	if(not GIAANNproto_databaseNetworkFilesExcitation.observedColumnMetadataExists(conceptIndex)):
		columnConnections = 0
	else:
		sourceFeatureIndices = GIAANNproto_databaseNetworkFilesExcitation.listObservedColumnSourceFeatureIndices(conceptIndex)
		for sourceFeatureIndex in sourceFeatureIndices:
			featureConnections = GIAANNproto_databaseNetworkFilesExcitation.loadObservedColumnSourceFeatureConnectionsTensor(databaseNetworkObject, conceptIndex, sourceFeatureIndex, deviceDatabase)
			if(featureConnections is None):
				raise RuntimeError("debugCountObservedColumnConnections error: featureConnections is None for conceptIndex = " + str(conceptIndex) + ", sourceFeatureIndex = " + str(sourceFeatureIndex))
			if(databaseNetworkObject.arrayIndexPropertiesStrengthIndex < 0 or databaseNetworkObject.arrayIndexPropertiesStrengthIndex >= featureConnections.shape[0]):
				raise RuntimeError("debugCountObservedColumnConnections error: databaseNetworkObject.arrayIndexPropertiesStrengthIndex out of range")
			columnConnections += countNonZero(featureConnections)
			del featureConnections
	return columnConnections

def debugCountTotalParametersRun(databaseNetworkObject):
	assert arrayIndexPropertiesEfficient 	#only databaseNetworkObject.arrayIndexPropertiesStrengthIndex stored in database, all tensors are coalesced
	if(databaseNetworkObject is None):
		raise RuntimeError("debugCountTotalParametersRun error: databaseNetworkObject is None")
	if(databaseNetworkObject.arrayIndexPropertiesStrengthIndex is None):
		raise RuntimeError("debugCountTotalParametersRun error: databaseNetworkObject.arrayIndexPropertiesStrengthIndex is None")
	totalColumns = len(databaseNetworkObject.conceptColumnsList)
	if(totalColumns <= 0):
		raise RuntimeError("debugCountTotalParametersRun error: conceptColumnsList is empty")
	totalConnections = 0
	for columnIndex, lemma in enumerate(databaseNetworkObject.conceptColumnsList):
		#print("columnIndex = ", columnIndex)
		conceptIndex = databaseNetworkObject.conceptColumnsDict.get(lemma)
		if(conceptIndex is None):
			raise RuntimeError("debugCountTotalParametersRun error: conceptIndex is None for lemma = " + lemma)
		columnConnections = debugCountObservedColumnConnections(databaseNetworkObject, conceptIndex, lemma, columnIndex)
		totalConnections += columnConnections
	database_pt_size_gb = debugCalculateDatabasePtSizeGiB()
	memory_gb = debugCalculateDatabaseSizeGiB()
	if(debugCountTotalParameters):
		print("debugCountTotalParameters totalConnections = ", totalConnections)
		print("debugCountTotalParameters totalColumns = ", totalColumns)
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
