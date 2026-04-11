"""GIAANNproto_databaseNetworkObservedColumn.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_main.py

# Usage:
see GIAANNproto_main.py

# Description:
GIA ANN proto database Network Observed Column

"""

import torch as pt
import time

from GIAANNproto_globalDefs import *
import GIAANNproto_debug
import GIAANNproto_databaseNetworkFiles


class ObservedColumnConnectionBase:
	def getObservedColumnErrorName(self):
		result = type(self).__name__
		return result

	def initialiseFeatureConnections(self, c, f, targetDevice=None):
		deviceTarget = targetDevice if targetDevice is not None else self.getDefaultConnectionTargetDevice()
		indices = pt.empty((5, 0), dtype=pt.long, device=deviceTarget)
		values = pt.empty((0,), dtype=arrayType, device=deviceTarget)
		featureConnections = pt.sparse_coo_tensor(indices, values, size=(self.databaseNetworkObject.arrayNumberOfProperties, numberOfDendriticBranches, arrayNumberOfSegments, c, f), dtype=arrayType, device=deviceTarget)
		return featureConnections

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
			raise RuntimeError(f"{self.getObservedColumnErrorName()} source feature index out of range: {result}")
		return result

	def normaliseSourceFeatureIndices(self, requiredSourceFeatureIndices):
		if(optimisationNormaliseSourceFeatureIndicesDisabled):
			indicesList = requiredSourceFeatureIndices	
		else:
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

	def loadRequiredSourceFeatureConnections(self, requiredSourceFeatureIndices, targetDevice, createMissing=False, ensureCurrentSizeOnLoad=False):
		resolvedTargetDevice = targetDevice if targetDevice is not None else self.getDefaultConnectionTargetDevice()
		sourceFeatureIndices = self.normaliseSourceFeatureIndices(requiredSourceFeatureIndices)
		for sourceFeatureIndex in sourceFeatureIndices:
			self.getFeatureConnectionsForSourceFeature(sourceFeatureIndex, resolvedTargetDevice, createMissing, ensureCurrentSizeOnLoad)
		return

	def ensureSourceFeatureArraySizes(self, sourceFeatureIndices):
		resolvedSourceFeatureIndices = self.normaliseSourceFeatureIndices(sourceFeatureIndices)
		for sourceFeatureIndex in resolvedSourceFeatureIndices:
			self.ensureSourceFeatureArraySize(sourceFeatureIndex)
		return

	def prepareRequiredSourceFeatureConnectionsTrain(self, requiredSourceFeatureIndices, targetDevice, createMissing=False):
		if(storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam and useGPUdatabase != useGPUsparse):
			self.prepareRequiredSourceFeatureConnectionsDatabaseInRamCPU(requiredSourceFeatureIndices, targetDevice, createMissing)
		else:
			self.prepareRequiredSourceFeatureConnections(requiredSourceFeatureIndices, targetDevice, createMissing)
		return
	
	def prepareRequiredSourceFeatureConnections(self, requiredSourceFeatureIndices, targetDevice, createMissing=False):
		resolvedTargetDevice = targetDevice if targetDevice is not None else self.getDefaultConnectionTargetDevice()
		resolvedSourceFeatureIndices = self.normaliseSourceFeatureIndices(requiredSourceFeatureIndices)
		self.loadRequiredSourceFeatureConnections(resolvedSourceFeatureIndices, resolvedTargetDevice, createMissing)
		self.ensureSourceFeatureArraySizes(resolvedSourceFeatureIndices)
		return

	def prepareRequiredSourceFeatureConnectionsDatabaseInRamCPU(self, requiredSourceFeatureIndices, targetDevice, createMissing=False):
		resolvedTargetDevice = targetDevice if targetDevice is not None else self.getDefaultConnectionTargetDevice()
		resolvedSourceFeatureIndices = self.normaliseSourceFeatureIndices(requiredSourceFeatureIndices)
		self.loadRequiredSourceFeatureConnections(resolvedSourceFeatureIndices, deviceDatabase, createMissing)
		self.ensureSourceFeatureArraySizes(resolvedSourceFeatureIndices)
		if(resolvedTargetDevice != deviceDatabase):
			self.loadRequiredSourceFeatureConnections(resolvedSourceFeatureIndices, resolvedTargetDevice, createMissing)
		return

	def prepareFeatureConnectionsForSourceFeature(self, sourceFeatureIndex, targetDevice=None, createMissing=False):
		resolvedTargetDevice = targetDevice if targetDevice is not None else self.getDefaultConnectionTargetDevice()
		self.prepareRequiredSourceFeatureConnections([sourceFeatureIndex], resolvedTargetDevice, createMissing)
		result = self.getFeatureConnectionsForSourceFeature(sourceFeatureIndex, resolvedTargetDevice, createMissing)
		return result

	def ensureRAMdatabaseFeatureTensorSizes(self):
		loadedSourceFeatureIndices = sorted(self.featureConnectionsBySourceFeature.keys())
		self.expandFeatureNeuronArraysFeatures(self.databaseNetworkObject.f)
		self.expandFeatureConnectionsArraysConcepts(self.databaseNetworkObject.c, loadedSourceFeatureIndices)
		self.expandFeatureConnectionsArraysFeatures(self.databaseNetworkObject.f, loadedSourceFeatureIndices)
		return

	def setTrainPreparedSourceFeatureIndices(self, sourceFeatureIndices):
		self.trainPreparedSourceFeatureIndices = set(self.normaliseSourceFeatureIndices(sourceFeatureIndices))
		return

	def getTrainPreparedSourceFeatureIndices(self):
		if(not hasattr(self, "trainPreparedSourceFeatureIndices")):
			raise RuntimeError(f"{self.getObservedColumnErrorName()}.getTrainPreparedSourceFeatureIndices error: trainPreparedSourceFeatureIndices missing")
		result = sorted(self.trainPreparedSourceFeatureIndices)
		return result

	def hasTrainPreparedSourceFeatureIndices(self):
		if(not hasattr(self, "trainPreparedSourceFeatureIndices")):
			raise RuntimeError(f"{self.getObservedColumnErrorName()}.hasTrainPreparedSourceFeatureIndices error: trainPreparedSourceFeatureIndices missing")
		result = len(self.trainPreparedSourceFeatureIndices) > 0
		return result

	def clearTrainPreparedSourceFeatureIndices(self):
		if(not hasattr(self, "trainPreparedSourceFeatureIndices")):
			raise RuntimeError(f"{self.getObservedColumnErrorName()}.clearTrainPreparedSourceFeatureIndices error: trainPreparedSourceFeatureIndices missing")
		self.trainPreparedSourceFeatureIndices.clear()
		return

	def requiresExpandFeatureConnectionsArraysConcepts(self, newC, sourceFeatureIndices=None):
		result = False
		resolvedSourceFeatureIndices = self.normaliseSourceFeatureIndices(sourceFeatureIndices) if sourceFeatureIndices is not None else sorted(self.featureConnectionsBySourceFeature.keys())
		for sourceFeatureIndex in resolvedSourceFeatureIndices:
			if(sourceFeatureIndex not in self.featureConnectionsBySourceFeature):
				raise RuntimeError(f"{self.getObservedColumnErrorName()}.requiresExpandFeatureConnectionsArraysConcepts error: missing loaded source feature tensor {sourceFeatureIndex}")
			sourceTensor = self.featureConnectionsBySourceFeature[sourceFeatureIndex]
			if(newC > sourceTensor.shape[3]):
				result = True
		return result

	def expandFeatureConnectionsArraysConcepts(self, newC, sourceFeatureIndices=None):
		resolvedSourceFeatureIndices = self.normaliseSourceFeatureIndices(sourceFeatureIndices) if sourceFeatureIndices is not None else sorted(self.featureConnectionsBySourceFeature.keys())
		for sourceFeatureIndex in resolvedSourceFeatureIndices:
			if(sourceFeatureIndex not in self.featureConnectionsBySourceFeature):
				raise RuntimeError(f"{self.getObservedColumnErrorName()}.expandFeatureConnectionsArraysConcepts error: missing loaded source feature tensor {sourceFeatureIndex}")
			sourceTensor = self.featureConnectionsBySourceFeature[sourceFeatureIndex]
			loadC = sourceTensor.shape[3]
			if(newC > loadC):
				expandedSize = (sourceTensor.shape[0], sourceTensor.shape[1], sourceTensor.shape[2], newC, sourceTensor.shape[4])
				sourceTensor = GIAANNproto_databaseNetworkFiles.expandSparseTensorSize(sourceTensor, expandedSize, f"{self.getObservedColumnErrorName()}.expandFeatureConnectionsArraysConcepts")
				self.featureConnectionsBySourceFeature[sourceFeatureIndex] = sourceTensor
		return

	def requiresExpandFeatureNeuronArraysFeatures(self, newF):
		result = False
		if(not storeDatabaseGlobalFeatureNeuronsInRam):
			if(not hasattr(self, "featureNeurons")):
				raise RuntimeError(f"{self.getObservedColumnErrorName()}.requiresExpandFeatureNeuronArraysFeatures error: featureNeurons missing while storeDatabaseGlobalFeatureNeuronsInRam is False")
			if(newF > self.featureNeurons.shape[3]):
				result = True
		return result

	def expandFeatureNeuronArraysFeatures(self, newF):
		if(not storeDatabaseGlobalFeatureNeuronsInRam):
			if(not hasattr(self, "featureNeurons")):
				raise RuntimeError(f"{self.getObservedColumnErrorName()}.expandFeatureNeuronArraysFeatures error: featureNeurons missing while storeDatabaseGlobalFeatureNeuronsInRam is False")
			if(newF > self.featureNeurons.shape[3]):
				expandedSizeNeurons = (self.featureNeurons.shape[0], self.featureNeurons.shape[1], self.featureNeurons.shape[2], newF)
				self.featureNeurons = GIAANNproto_databaseNetworkFiles.expandSparseTensorSize(self.featureNeurons, expandedSizeNeurons, f"{self.getObservedColumnErrorName()}.expandFeatureNeuronArraysFeatures")
		return

	def requiresExpandFeatureMapsFeatures(self, newF):
		result = False
		if(not trainStoreFeatureMapsGlobally):
			if(not hasattr(self, "nextFeatureIndex")):
				raise RuntimeError(f"{self.getObservedColumnErrorName()}.requiresExpandFeatureMapsFeatures error: nextFeatureIndex missing")
			if(newF > (self.nextFeatureIndex + 1)):
				result = True
		return result

	def expandFeatureMapsFeatures(self, newF):
		if(not trainStoreFeatureMapsGlobally):
			if(not hasattr(self, "nextFeatureIndex")):
				raise RuntimeError(f"{self.getObservedColumnErrorName()}.expandFeatureMapsFeatures error: nextFeatureIndex missing")
			loadFReference = self.nextFeatureIndex + 1
			for featureIndex in range(loadFReference, newF):
				featureWord = self.databaseNetworkObject.conceptFeaturesList[featureIndex]
				self.featureWordToIndex[featureWord] = featureIndex
				self.featureIndexToWord[featureIndex] = featureWord
				self.nextFeatureIndex += 1
		return

	def requiresExpandFeatureConnectionsArraysFeatures(self, newF, sourceFeatureIndices=None):
		result = False
		resolvedSourceFeatureIndices = self.normaliseSourceFeatureIndices(sourceFeatureIndices) if sourceFeatureIndices is not None else sorted(self.featureConnectionsBySourceFeature.keys())
		for sourceFeatureIndex in resolvedSourceFeatureIndices:
			if(sourceFeatureIndex not in self.featureConnectionsBySourceFeature):
				raise RuntimeError(f"{self.getObservedColumnErrorName()}.requiresExpandFeatureConnectionsArraysFeatures error: missing loaded source feature tensor {sourceFeatureIndex}")
			sourceTensor = self.featureConnectionsBySourceFeature[sourceFeatureIndex]
			if(newF > sourceTensor.shape[4]):
				result = True
		return result

	def expandFeatureConnectionsArraysFeatures(self, newF, sourceFeatureIndices=None):
		resolvedSourceFeatureIndices = self.normaliseSourceFeatureIndices(sourceFeatureIndices) if sourceFeatureIndices is not None else sorted(self.featureConnectionsBySourceFeature.keys())
		for sourceFeatureIndex in resolvedSourceFeatureIndices:
			if(sourceFeatureIndex not in self.featureConnectionsBySourceFeature):
				raise RuntimeError(f"{self.getObservedColumnErrorName()}.expandFeatureConnectionsArraysFeatures error: missing loaded source feature tensor {sourceFeatureIndex}")
			sourceTensor = self.featureConnectionsBySourceFeature[sourceFeatureIndex]
			loadF = sourceTensor.shape[4]
			if(newF > loadF):
				expandedSizeConnections = (sourceTensor.shape[0], sourceTensor.shape[1], sourceTensor.shape[2], sourceTensor.shape[3], newF)
				sourceTensor = GIAANNproto_databaseNetworkFiles.expandSparseTensorSize(sourceTensor, expandedSizeConnections, f"{self.getObservedColumnErrorName()}.expandFeatureConnectionsArraysFeatures")
				self.featureConnectionsBySourceFeature[sourceFeatureIndex] = sourceTensor
		return

	def ensureObservedColumnFeatureArraysFeatures(self, newF):
		if(self.requiresExpandFeatureNeuronArraysFeatures(newF)):
			self.expandFeatureNeuronArraysFeatures(newF)
		if(self.requiresExpandFeatureMapsFeatures(newF)):
			self.expandFeatureMapsFeatures(newF)
		return

	def ensureSourceFeatureArraySize(self, sourceFeatureIndex):
		normalisedSourceFeatureIndex = self.normaliseSourceFeatureIndex(sourceFeatureIndex)
		if(normalisedSourceFeatureIndex not in self.featureConnectionsBySourceFeature):
			raise RuntimeError(f"{self.getObservedColumnErrorName()}.ensureSourceFeatureArraySize error: missing loaded source feature tensor {normalisedSourceFeatureIndex}")
		if(self.requiresExpandFeatureConnectionsArraysConcepts(self.databaseNetworkObject.c, [normalisedSourceFeatureIndex])):
			self.expandFeatureConnectionsArraysConcepts(self.databaseNetworkObject.c, [normalisedSourceFeatureIndex])
		if(self.requiresExpandFeatureConnectionsArraysFeatures(self.databaseNetworkObject.f, [normalisedSourceFeatureIndex])):
			self.expandFeatureConnectionsArraysFeatures(self.databaseNetworkObject.f, [normalisedSourceFeatureIndex])
		return

	def setFeatureConnectionsForSourceFeature(self, sourceFeatureIndex, tensor):
		if(debugPrintTrainSectionTimesSourceFeatureConnections):
			debugSectionName = GIAANNproto_debug.getSourceFeatureConnectionsDebugSectionName(self.databaseNetworkObject, "setFeatureConnectionsForSourceFeature")
			debugSectionStartTime = None
			if(debugSectionName is not None):
				debugSectionStartTime = time.perf_counter()
		normalisedSourceFeatureIndex = self.normaliseSourceFeatureIndex(sourceFeatureIndex)
		expectedSize = self.getFeatureConnectionsTargetSize()
		errorName = self.getObservedColumnErrorName()
		if(tensor is None):
			raise RuntimeError(f"{errorName}.setFeatureConnectionsForSourceFeature error: tensor is None")
		if(tensor.layout != pt.sparse_coo):
			raise RuntimeError(f"{errorName}.setFeatureConnectionsForSourceFeature error: tensor must be sparse COO")
		if(tensor.dim() != 5):
			raise RuntimeError(f"{errorName}.setFeatureConnectionsForSourceFeature error: tensor rank must be 5")
		if(tuple(tensor.size()) != tuple(expectedSize)):
			raise RuntimeError(f"{errorName}.setFeatureConnectionsForSourceFeature error: tensor size {tuple(tensor.size())} does not match expected size {tuple(expectedSize)}")
		if(not tensor.is_coalesced()):
			tensor = tensor.coalesce()
		self.featureConnectionsBySourceFeature[normalisedSourceFeatureIndex] = tensor
		self.loadedSourceFeatureIndices.add(normalisedSourceFeatureIndex)
		if(debugPrintTrainSectionTimesSourceFeatureConnections):
			if(debugSectionName is not None):
				GIAANNproto_debug.debugTrainSectionTimesAdd(self.databaseNetworkObject, debugSectionName, time.perf_counter() - debugSectionStartTime)
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


class ObservedColumn(ObservedColumnConnectionBase):
	"""
	Create a class defining observed columns. The observed column class contains an index to the dataset concept column dictionary. The observed column class contains a list of feature connection arrays. The observed column class also contains a list of feature neuron arrays when storeDatabaseGlobalFeatureNeuronsInRam is False.
	"""
	def __init__(self, databaseNetworkObject, conceptIndex, lemma, i):
		self.databaseNetworkObject = databaseNetworkObject
		self.conceptIndex = conceptIndex
		self.conceptName = lemma
		self.conceptSequenceWordIndex = i
		if not storeDatabaseGlobalFeatureNeuronsInRam:
			self.featureNeurons = self.initialiseFeatureNeurons(databaseNetworkObject.f)
		if(trainStoreFeatureMapsGlobally):
			self.featureWordToIndex = databaseNetworkObject.conceptFeaturesDict
			self.featureIndexToWord = databaseNetworkObject.conceptFeaturesIndexToWordDict
			self.nextFeatureIndex = len(databaseNetworkObject.conceptFeaturesDict) - 1
		else:
			self.featureWordToIndex = {}
			self.featureIndexToWord = {}
			if(useDedicatedConceptNames):
				self.nextFeatureIndex = 1
				if(useDedicatedConceptNames2):
					self.featureWordToIndex[variablePrimeConceptFeatureNeuronName] = featureIndexPrimeConceptNeuron
					self.featureIndexToWord[featureIndexPrimeConceptNeuron] = variablePrimeConceptFeatureNeuronName
		self.featureConnectionsBySourceFeature = {}
		self.loadedSourceFeatureIndices = set()
		self.trainPreparedSourceFeatureIndices = set()
		if(optimisationGetFeatureConnectionsForSourceFeatureCache):
			self.storedSourceFeatureIndicesCache = None
		if(not trainStoreFeatureMapsGlobally):
			self.nextFeatureIndex = 0
			for featureIndex in range(1, databaseNetworkObject.f, 1):
				featureWord = databaseNetworkObject.conceptFeaturesList[featureIndex]
				self.featureWordToIndex[featureWord] = featureIndex
				self.featureIndexToWord[featureIndex] = featureWord
				self.nextFeatureIndex += 1

	def initialiseFeatureNeurons(self, f, targetDevice=None):
		deviceTarget = targetDevice if targetDevice is not None else (deviceDatabase if storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam else deviceSparse)
		indices = pt.empty((4, 0), dtype=pt.long, device=deviceTarget)
		values = pt.empty((0,), dtype=arrayType, device=deviceTarget)
		featureNeurons = pt.sparse_coo_tensor(indices, values, size=(self.databaseNetworkObject.arrayNumberOfProperties, numberOfDendriticBranches, arrayNumberOfSegments, f), dtype=arrayType, device=deviceTarget)
		return featureNeurons

	def getDefaultConnectionTargetDevice(self):
		deviceTarget = deviceDatabase if storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam else deviceSparse
		if(len(self.featureConnectionsBySourceFeature) > 0):
			firstTensor = next(iter(self.featureConnectionsBySourceFeature.values()))
			deviceTarget = firstTensor.device
		return deviceTarget

	def listStoredSourceFeatureIndices(self):
		if(storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam and self.databaseNetworkObject.observedColumnsRAMLoaded):
			combinedIndices = set(self.featureConnectionsBySourceFeature.keys())
		elif(optimisationGetFeatureConnectionsForSourceFeatureCache):
			if(self.storedSourceFeatureIndicesCache is None):
				self.storedSourceFeatureIndicesCache = set(GIAANNproto_databaseNetworkFiles.listObservedColumnSourceFeatureIndices(self.conceptIndex))
			combinedIndices = set(self.storedSourceFeatureIndicesCache)
		else:
			combinedIndices = set(GIAANNproto_databaseNetworkFiles.listObservedColumnSourceFeatureIndices(self.conceptIndex))
		for sourceFeatureIndex in self.featureConnectionsBySourceFeature.keys():
			combinedIndices.add(self.normaliseSourceFeatureIndex(sourceFeatureIndex))
		result = sorted(combinedIndices)
		return result

	def getFeatureConnectionsForSourceFeature(self, sourceFeatureIndex, targetDevice=None, createMissing=False, ensureCurrentSizeOnLoad=False):
		if(debugPrintTrainSectionTimesSourceFeatureConnections):
			debugSectionName = GIAANNproto_debug.getSourceFeatureConnectionsDebugSectionName(self.databaseNetworkObject, "getFeatureConnectionsForSourceFeature")
			debugSectionStartTime = None
			if(debugSectionName is not None):
				debugSectionStartTime = time.perf_counter()
		normalisedSourceFeatureIndex = self.normaliseSourceFeatureIndex(sourceFeatureIndex)
		resolvedTargetDevice = targetDevice if targetDevice is not None else self.getDefaultConnectionTargetDevice()
		result = self.featureConnectionsBySourceFeature.get(normalisedSourceFeatureIndex)
		if(result is None):
			if(storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam and self.databaseNetworkObject.observedColumnsRAMLoaded):
				result = self.initialiseFeatureConnections(self.databaseNetworkObject.c, self.databaseNetworkObject.f, resolvedTargetDevice)
			else:
				if(optimisationGetFeatureConnectionsForSourceFeatureCache):
					if(self.storedSourceFeatureIndicesCache is None):
						self.storedSourceFeatureIndicesCache = set(GIAANNproto_databaseNetworkFiles.listObservedColumnSourceFeatureIndices(self.conceptIndex))
					storedSourceFeatureIndices = self.storedSourceFeatureIndicesCache
				else:
					storedSourceFeatureIndices = self.listStoredSourceFeatureIndices()
				if(normalisedSourceFeatureIndex in storedSourceFeatureIndices):
					result = GIAANNproto_databaseNetworkFiles.loadObservedColumnSourceFeatureConnectionsTensor(self.databaseNetworkObject, self.conceptIndex, normalisedSourceFeatureIndex, resolvedTargetDevice, ensureCurrentSizeOnLoad=ensureCurrentSizeOnLoad)
				else:
					result = self.initialiseFeatureConnections(self.databaseNetworkObject.c, self.databaseNetworkObject.f, resolvedTargetDevice)
			self.featureConnectionsBySourceFeature[normalisedSourceFeatureIndex] = result
		elif(result.device != resolvedTargetDevice):
			result = result.to(resolvedTargetDevice)
			self.featureConnectionsBySourceFeature[normalisedSourceFeatureIndex] = result
		self.loadedSourceFeatureIndices.add(normalisedSourceFeatureIndex)
		if(debugPrintTrainSectionTimesSourceFeatureConnections):
			if(debugSectionName is not None):
				GIAANNproto_debug.debugTrainSectionTimesAdd(self.databaseNetworkObject, debugSectionName, time.perf_counter() - debugSectionStartTime)
		return result

	def saveLoadedSourceFeatureConnectionsToDisk(self, sourceFeatureIndices=None):
		if(sourceFeatureIndices is None):
			resolvedSourceFeatureIndices = sorted(self.loadedSourceFeatureIndices)
		else:
			resolvedSourceFeatureIndices = self.normaliseSourceFeatureIndices(sourceFeatureIndices)
		if(optimisationGetFeatureConnectionsForSourceFeatureCache):
			if(self.storedSourceFeatureIndicesCache is None):
				self.storedSourceFeatureIndicesCache = set(GIAANNproto_databaseNetworkFiles.listObservedColumnSourceFeatureIndices(self.conceptIndex))
		for sourceFeatureIndex in resolvedSourceFeatureIndices:
			if(sourceFeatureIndex not in self.featureConnectionsBySourceFeature):
				raise RuntimeError(f"saveLoadedSourceFeatureConnectionsToDisk error: missing loaded source feature tensor {sourceFeatureIndex}")
			sourceTensor = self.featureConnectionsBySourceFeature[sourceFeatureIndex]
			GIAANNproto_databaseNetworkFiles.saveObservedColumnSourceFeatureConnectionsTensor(self.conceptIndex, sourceFeatureIndex, sourceTensor)
			if(optimisationGetFeatureConnectionsForSourceFeatureCache):
				if(sourceTensor.is_sparse):
					sourceTensor = sourceTensor.coalesce()
					tensorNNZ = sourceTensor._nnz()
				else:
					tensorNNZ = int(pt.count_nonzero(sourceTensor).item())
				if(tensorNNZ > 0):
					self.storedSourceFeatureIndicesCache.add(sourceFeatureIndex)
				else:
					self.storedSourceFeatureIndicesCache.discard(sourceFeatureIndex)
		return

	def saveToDisk(self, saveAllSourceFeatures, resizeFeatureTensorsToCurrentSize=False):
		GIAANNproto_databaseNetworkFiles.observedColumnSaveToDisk(self, saveAllSourceFeatures, resizeFeatureTensorsToCurrentSize=resizeFeatureTensorsToCurrentSize)
		return

	@classmethod
	def loadFromDisk(cls, databaseNetworkObject, conceptIndex, lemma, i, targetDevice=None, loadAllSourceFeatures=False, resizeFeatureTensorsToCurrentSize=False, loadFeatureNeurons=True):
		result = GIAANNproto_databaseNetworkFiles.observedColumnLoadFromDisk(cls, databaseNetworkObject, conceptIndex, lemma, i, targetDevice=targetDevice, loadAllSourceFeatures=loadAllSourceFeatures, resizeFeatureTensorsToCurrentSize=resizeFeatureTensorsToCurrentSize, loadFeatureNeurons=loadFeatureNeurons)
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


class ObservedColumnProxy(ObservedColumnConnectionBase):
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
		self.trainPreparedSourceFeatureIndices = set()
		if((not storeDatabaseGlobalFeatureNeuronsInRam) and hasattr(observedColumn, "featureNeurons")):
			self.featureNeurons = observedColumn.featureNeurons.to(targetDevice)

	def getDefaultConnectionTargetDevice(self):
		deviceTarget = self.proxyTargetDevice
		return deviceTarget

	def listStoredSourceFeatureIndices(self):
		result = self.sourceObservedColumn.listStoredSourceFeatureIndices()
		return result

	def getFeatureConnectionsForSourceFeature(self, sourceFeatureIndex, targetDevice=None, createMissing=False, ensureCurrentSizeOnLoad=False):
		normalisedSourceFeatureIndex = self.normaliseSourceFeatureIndex(sourceFeatureIndex)
		resolvedTargetDevice = targetDevice if targetDevice is not None else self.proxyTargetDevice
		result = self.featureConnectionsBySourceFeature.get(normalisedSourceFeatureIndex)
		if(result is None):
			baseTensor = self.sourceObservedColumn.getFeatureConnectionsForSourceFeature(normalisedSourceFeatureIndex, self.sourceObservedColumn.getDefaultConnectionTargetDevice(), createMissing, ensureCurrentSizeOnLoad)
			result = baseTensor.to(resolvedTargetDevice)
			self.featureConnectionsBySourceFeature[normalisedSourceFeatureIndex] = result
		elif(result.device != resolvedTargetDevice):
			result = result.to(resolvedTargetDevice)
			self.featureConnectionsBySourceFeature[normalisedSourceFeatureIndex] = result
		self.loadedSourceFeatureIndices.add(normalisedSourceFeatureIndex)
		return result

	def saveLoadedSourceFeatureConnectionsToDisk(self):
		raise RuntimeError("ObservedColumnProxy cannot save source feature connections to disk")
