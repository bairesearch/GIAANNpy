"""GIAANNnlp_auxiliaryNeuronsAuto.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 BAI Research Pty Ltd (bairesearch.com.au)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN NLP auxiliary neurons auto

"""

import math
import os
import torch as pt

from GIAANNcmn_globalDefs import *
import GIAANNcmn_databaseNetworkFiles


if(auxiliaryNeurons and auxiliaryNeuronsAuto):

	auxiliaryNeuronsAutoReverseSavedTargetTensorPaths = set()
	auxiliaryNeuronsAutoFeatureDatasetCacheByFile = {}

	def getTokenAutoAuxiliaryFeatureIndices(databaseNetworkObject, token, isConcept, conceptIndex, allowNewFeatures=False, registerParent=False):
		import GIAANNnlp_auxiliaryNeuronsSimilarWords
		result = []
		if(isConcept):
			if(GIAANNnlp_auxiliaryNeuronsSimilarWords.tokenHasPrimeConceptSimilarityWord(token)):
				if(auxiliaryNeuronsSimilarWordsAuto and auxiliaryNeuronsSimilarWordsPrimeConceptFeatures):
					result.append(registerTokenAutoAuxiliaryFeature(databaseNetworkObject, token, conceptIndex, auxiliaryNeuronsSimilarWordsFeatureNamePrefixPrimeConcept, True, allowNewFeatures, registerParent))
				if(auxiliaryNeuronsSimilarSubwordAuto and auxiliaryNeuronsSimilarSubwordPrimeConceptFeatures):
					result.append(registerTokenAutoAuxiliaryFeature(databaseNetworkObject, token, conceptIndex, auxiliaryNeuronsSimilarWordsFeatureNamePrefixSubwordPrimeConcept, True, allowNewFeatures, registerParent))
		else:
			if(GIAANNnlp_auxiliaryNeuronsSimilarWords.tokenHasSecondarySimilarityWord(token)):
				if(auxiliaryNeuronsSimilarWordsAuto and auxiliaryNeuronsSimilarWordsSecondaryConceptFeatures):
					result.append(registerTokenAutoAuxiliaryFeature(databaseNetworkObject, token, conceptIndex, auxiliaryNeuronsSimilarWordsFeatureNamePrefixSecondary, False, allowNewFeatures, registerParent))
				if(auxiliaryNeuronsSimilarSubwordAuto and auxiliaryNeuronsSimilarSubwordSecondaryConceptFeatures):
					result.append(registerTokenAutoAuxiliaryFeature(databaseNetworkObject, token, conceptIndex, auxiliaryNeuronsSimilarWordsFeatureNamePrefixSubwordSecondary, False, allowNewFeatures, registerParent))
		return result

	def registerTokenAutoAuxiliaryFeature(databaseNetworkObject, token, conceptIndex, auxiliaryFeaturePrefix, primeConceptFeature, allowNewFeatures, registerParent):
		import GIAANNnlp_auxiliaryNeuronsSimilarWords
		if(primeConceptFeature):
			auxiliaryBaseWord = GIAANNnlp_auxiliaryNeuronsSimilarWords.getTokenPrimeConceptSimilarityWord(token)
		else:
			auxiliaryBaseWord = GIAANNnlp_auxiliaryNeuronsSimilarWords.getTokenSecondarySimilarityWord(token)
		auxiliaryFeatureWord = GIAANNnlp_auxiliaryNeuronsSimilarWords.buildConceptColumnAuxiliaryFeatureName(databaseNetworkObject, auxiliaryFeaturePrefix, conceptIndex, auxiliaryBaseWord)
		result = GIAANNnlp_auxiliaryNeuronsSimilarWords.registerAuxiliaryFeatureWord(databaseNetworkObject, auxiliaryFeatureWord, allowNewFeatures)
		if(registerParent):
			parentKey = GIAANNnlp_auxiliaryNeuronsSimilarWords.buildSimilarityParentKey(auxiliaryFeaturePrefix, auxiliaryBaseWord)
			GIAANNnlp_auxiliaryNeuronsSimilarWords.registerSimilarityParentFeatureWordWeight(databaseNetworkObject, parentKey, auxiliaryFeatureWord, auxiliaryNeuronsSimilarWordsIdentitySimilarity)
		return result

	def initialiseObservedColumnReverseConnectionStorage(observedColumn):
		observedColumn.reverseFeatureConnectionsByTargetFeature = {}
		observedColumn.reverseLoadedTargetFeatureIndices = set()
		observedColumn.reverseTrainPreparedTargetFeatureIndices = set()
		return

	def ensureObservedColumnReverseConnectionStorage(observedColumn):
		if(not hasattr(observedColumn, "reverseFeatureConnectionsByTargetFeature")):
			initialiseObservedColumnReverseConnectionStorage(observedColumn)
		return

	def initialiseReverseFeatureConnections(databaseNetworkObject, targetDevice):
		indices = pt.empty((5, 0), dtype=pt.long, device=targetDevice)
		values = pt.empty((0,), dtype=arrayType, device=targetDevice)
		result = pt.sparse_coo_tensor(indices, values, size=(databaseNetworkObject.arrayNumberOfProperties, multipleDendriticBranchesNumber, arrayNumberOfSegments, databaseNetworkObject.c, databaseNetworkObject.f), dtype=arrayType, device=targetDevice)
		return result

	def normaliseReverseTargetFeatureIndex(databaseNetworkObject, targetFeatureIndex):
		result = int(targetFeatureIndex)
		if(result < 0 or result >= databaseNetworkObject.f):
			raise RuntimeError("normaliseReverseTargetFeatureIndex error: target feature index out of range")
		return result

	def getObservedColumnReverseFeatureConnectionsFolder(conceptIndex):
		result = os.path.join(GIAANNcmn_databaseNetworkFiles.getObservedColumnFolder(conceptIndex), auxiliaryNeuronsAutoReverseConnectionsFolderName)
		return result

	def getObservedColumnReverseTargetFeatureConnectionsFileBaseName(targetFeatureIndex):
		result = auxiliaryNeuronsAutoReverseTargetFeatureConnectionsFileNamePrefix + str(int(targetFeatureIndex))
		return result

	def listObservedColumnReverseTargetFeatureIndices(databaseNetworkObject, conceptIndex):
		result = []
		connectionsFolder = getObservedColumnReverseFeatureConnectionsFolder(conceptIndex)
		if(os.path.isdir(connectionsFolder)):
			for fileName in os.listdir(connectionsFolder):
				if(fileName.startswith(auxiliaryNeuronsAutoReverseTargetFeatureConnectionsFileNamePrefix) and fileName.endswith(pytorchTensorFileExtension)):
					filePath = os.path.join(connectionsFolder, fileName)
					if(databaseNetworkObject.auxiliaryNeuronsSimilarWordsLoadExistingDatabase or filePath in auxiliaryNeuronsAutoReverseSavedTargetTensorPaths):
						targetFeatureIndexString = fileName[len(auxiliaryNeuronsAutoReverseTargetFeatureConnectionsFileNamePrefix):-len(pytorchTensorFileExtension)]
						result.append(int(targetFeatureIndexString))
		result.sort()
		return result

	def loadObservedColumnReverseFeatureConnectionsTensor(databaseNetworkObject, conceptIndex, targetFeatureIndex, targetDevice, ensureCurrentSizeOnLoad=False):
		connectionsFolder = getObservedColumnReverseFeatureConnectionsFolder(conceptIndex)
		fileBaseName = getObservedColumnReverseTargetFeatureConnectionsFileBaseName(targetFeatureIndex)
		tensorName = "observedColumn.reverseFeatureConnectionsByTargetFeature[" + str(conceptIndex) + "][" + str(targetFeatureIndex) + "]"
		tensor = GIAANNcmn_databaseNetworkFiles.adjustPropertyDimensions(databaseNetworkObject.inferenceMode, GIAANNcmn_databaseNetworkFiles.loadTensor(connectionsFolder, fileBaseName, targetDevice=targetDevice), tensorName)
		tensor = GIAANNcmn_databaseNetworkFiles.adjustBranchDimensions(tensor, tensorName, expectedRank=5)
		if(ensureCurrentSizeOnLoad):
			tensor = GIAANNcmn_databaseNetworkFiles.ensureFeatureConnectionsSourceTensorCurrentSize(tensor, databaseNetworkObject.c, databaseNetworkObject.f, tensorName)
		return tensor

	def saveObservedColumnReverseFeatureConnectionsTensor(conceptIndex, targetFeatureIndex, tensor):
		connectionsFolder = getObservedColumnReverseFeatureConnectionsFolder(conceptIndex)
		fileBaseName = getObservedColumnReverseTargetFeatureConnectionsFileBaseName(targetFeatureIndex)
		filePath = os.path.join(connectionsFolder, fileBaseName + pytorchTensorFileExtension)
		if(tensor is None):
			raise RuntimeError("saveObservedColumnReverseFeatureConnectionsTensor error: tensor is None")
		os.makedirs(connectionsFolder, exist_ok=True)
		if(tensor.is_sparse):
			tensor = tensor.coalesce()
			tensorNNZ = tensor._nnz()
		else:
			tensorNNZ = int(pt.count_nonzero(tensor).item())
		if(tensorNNZ > 0):
			GIAANNcmn_databaseNetworkFiles.saveTensor(tensor, connectionsFolder, fileBaseName)
			auxiliaryNeuronsAutoReverseSavedTargetTensorPaths.add(filePath)
		else:
			if(GIAANNcmn_databaseNetworkFiles.pathExists(filePath)):
				os.remove(filePath)
			auxiliaryNeuronsAutoReverseSavedTargetTensorPaths.discard(filePath)
		return

	def getObservedColumnReverseFeatureConnectionsForTargetFeature(observedColumn, targetFeatureIndex, targetDevice=None, createMissing=False, ensureCurrentSizeOnLoad=False):
		ensureObservedColumnReverseConnectionStorage(observedColumn)
		normalisedTargetFeatureIndex = normaliseReverseTargetFeatureIndex(observedColumn.databaseNetworkObject, targetFeatureIndex)
		resolvedTargetDevice = targetDevice if targetDevice is not None else deviceSparse
		result = observedColumn.reverseFeatureConnectionsByTargetFeature.get(normalisedTargetFeatureIndex)
		if(result is None):
			if(hasattr(observedColumn, "sourceObservedColumn")):
				sourceTensor = getObservedColumnReverseFeatureConnectionsForTargetFeature(observedColumn.sourceObservedColumn, normalisedTargetFeatureIndex, observedColumn.sourceObservedColumn.getDefaultConnectionTargetDevice(), createMissing, ensureCurrentSizeOnLoad)
				result = sourceTensor.to(resolvedTargetDevice)
			elif(storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam and observedColumn.databaseNetworkObject.observedColumnsRAMLoaded):
				result = initialiseReverseFeatureConnections(observedColumn.databaseNetworkObject, resolvedTargetDevice)
			else:
				storedTargetFeatureIndices = listObservedColumnReverseTargetFeatureIndices(observedColumn.databaseNetworkObject, observedColumn.conceptIndex)
				if(normalisedTargetFeatureIndex in storedTargetFeatureIndices):
					result = loadObservedColumnReverseFeatureConnectionsTensor(observedColumn.databaseNetworkObject, observedColumn.conceptIndex, normalisedTargetFeatureIndex, resolvedTargetDevice, ensureCurrentSizeOnLoad=ensureCurrentSizeOnLoad)
				else:
					result = initialiseReverseFeatureConnections(observedColumn.databaseNetworkObject, resolvedTargetDevice)
			observedColumn.reverseFeatureConnectionsByTargetFeature[normalisedTargetFeatureIndex] = result
		elif(result.device != resolvedTargetDevice):
			result = result.to(resolvedTargetDevice)
			observedColumn.reverseFeatureConnectionsByTargetFeature[normalisedTargetFeatureIndex] = result
		if(ensureCurrentSizeOnLoad):
			ensureObservedColumnReverseFeatureConnectionSize(observedColumn, normalisedTargetFeatureIndex)
			result = observedColumn.reverseFeatureConnectionsByTargetFeature[normalisedTargetFeatureIndex]
			if(result.device != resolvedTargetDevice):
				result = result.to(resolvedTargetDevice)
				observedColumn.reverseFeatureConnectionsByTargetFeature[normalisedTargetFeatureIndex] = result
		observedColumn.reverseLoadedTargetFeatureIndices.add(normalisedTargetFeatureIndex)
		return result

	def setObservedColumnReverseFeatureConnectionsForTargetFeature(observedColumn, targetFeatureIndex, tensor):
		ensureObservedColumnReverseConnectionStorage(observedColumn)
		normalisedTargetFeatureIndex = normaliseReverseTargetFeatureIndex(observedColumn.databaseNetworkObject, targetFeatureIndex)
		expectedSize = getObservedColumnReverseFeatureConnectionsTargetSize(observedColumn)
		if(tensor is None):
			raise RuntimeError("setObservedColumnReverseFeatureConnectionsForTargetFeature error: tensor is None")
		if(tensor.layout != pt.sparse_coo):
			raise RuntimeError("setObservedColumnReverseFeatureConnectionsForTargetFeature error: tensor must be sparse COO")
		if(tensor.dim() != 5):
			raise RuntimeError("setObservedColumnReverseFeatureConnectionsForTargetFeature error: tensor rank must be 5")
		if(tuple(tensor.size()) != tuple(expectedSize)):
			raise RuntimeError("setObservedColumnReverseFeatureConnectionsForTargetFeature error: tensor size mismatch")
		if(not tensor.is_coalesced()):
			tensor = tensor.coalesce()
		observedColumn.reverseFeatureConnectionsByTargetFeature[normalisedTargetFeatureIndex] = tensor
		observedColumn.reverseLoadedTargetFeatureIndices.add(normalisedTargetFeatureIndex)
		return

	def ensureObservedColumnReverseFeatureConnectionSize(observedColumn, targetFeatureIndex):
		normalisedTargetFeatureIndex = normaliseReverseTargetFeatureIndex(observedColumn.databaseNetworkObject, targetFeatureIndex)
		if(normalisedTargetFeatureIndex not in observedColumn.reverseFeatureConnectionsByTargetFeature):
			raise RuntimeError("ensureObservedColumnReverseFeatureConnectionSize error: missing loaded target feature tensor")
		targetTensor = observedColumn.reverseFeatureConnectionsByTargetFeature[normalisedTargetFeatureIndex]
		expectedSize = getObservedColumnReverseFeatureConnectionsTargetSize(observedColumn)
		if(tuple(targetTensor.size()) != tuple(expectedSize)):
			targetTensor = GIAANNcmn_databaseNetworkFiles.expandSparseTensorSize(targetTensor, expectedSize, "ensureObservedColumnReverseFeatureConnectionSize")
			observedColumn.reverseFeatureConnectionsByTargetFeature[normalisedTargetFeatureIndex] = targetTensor
		return

	def getObservedColumnReverseFeatureConnectionsTargetSize(observedColumn):
		result = (observedColumn.databaseNetworkObject.arrayNumberOfProperties, multipleDendriticBranchesNumber, arrayNumberOfSegments, observedColumn.databaseNetworkObject.c, observedColumn.databaseNetworkObject.f)
		return result

	def saveObservedColumnReverseFeatureConnectionsToDisk(observedColumn, saveAllTargetFeatures):
		ensureObservedColumnReverseConnectionStorage(observedColumn)
		if(saveAllTargetFeatures):
			targetFeatureIndicesToSave = sorted(observedColumn.reverseLoadedTargetFeatureIndices)
		else:
			targetFeatureIndicesToSave = sorted(observedColumn.reverseTrainPreparedTargetFeatureIndices)
		for targetFeatureIndex in targetFeatureIndicesToSave:
			if(targetFeatureIndex not in observedColumn.reverseFeatureConnectionsByTargetFeature):
				raise RuntimeError("saveObservedColumnReverseFeatureConnectionsToDisk error: missing loaded target feature tensor")
			saveObservedColumnReverseFeatureConnectionsTensor(observedColumn.conceptIndex, targetFeatureIndex, observedColumn.reverseFeatureConnectionsByTargetFeature[targetFeatureIndex])
		return

	def loadObservedColumnReverseConnectionsFromDisk(observedColumn, targetDevice=None, loadAllTargetFeatures=False, resizeFeatureTensorsToCurrentSize=False):
		ensureObservedColumnReverseConnectionStorage(observedColumn)
		if(loadAllTargetFeatures):
			targetFeatureIndices = listObservedColumnReverseTargetFeatureIndices(observedColumn.databaseNetworkObject, observedColumn.conceptIndex)
			loadTargetDevice = targetDevice if targetDevice is not None else deviceDatabase
			for targetFeatureIndex in targetFeatureIndices:
				getObservedColumnReverseFeatureConnectionsForTargetFeature(observedColumn, targetFeatureIndex, targetDevice=loadTargetDevice, createMissing=False, ensureCurrentSizeOnLoad=resizeFeatureTensorsToCurrentSize)
		return

	def ensureRAMdatabaseReverseFeatureTensorSizes(observedColumn):
		ensureObservedColumnReverseConnectionStorage(observedColumn)
		for targetFeatureIndex in sorted(observedColumn.reverseFeatureConnectionsByTargetFeature.keys()):
			ensureObservedColumnReverseFeatureConnectionSize(observedColumn, targetFeatureIndex)
		return

	if(trainReverseConnections):
		def moveObservedColumnReverseConnectionsToDatabaseAfterTrain(observedColumn):
			ensureObservedColumnReverseConnectionStorage(observedColumn)
			for targetFeatureIndex in sorted(observedColumn.reverseTrainPreparedTargetFeatureIndices):
				targetTensor = getObservedColumnReverseFeatureConnectionsForTargetFeature(observedColumn, targetFeatureIndex, targetDevice=deviceDatabase, createMissing=False)
				setObservedColumnReverseFeatureConnectionsForTargetFeature(observedColumn, targetFeatureIndex, targetTensor)
			observedColumn.reverseTrainPreparedTargetFeatureIndices.clear()
			return

		def updateReverseFeatureConnectionsFromSequence(sequenceObservedColumns, sequenceObservedColumnsDict):
			if(not arrayIndexPropertiesStrength):
				raise RuntimeError("updateReverseFeatureConnectionsFromSequence error: arrayIndexPropertiesStrength must be enabled")
			if(sequenceObservedColumns.useTrainSparseConnectionsTensor()):
				connectionDeltaSparse = sequenceObservedColumns.extractSequenceConnectionPropertySparse(sequenceObservedColumns.databaseNetworkObject.arrayIndexPropertiesStrengthIndex)
			else:
				connectionDeltaSparse = sequenceObservedColumns.featureConnections[sequenceObservedColumns.databaseNetworkObject.arrayIndexPropertiesStrengthIndex].to_sparse().coalesce()
			if(connectionDeltaSparse._nnz() > 0):
				applyReverseConnectionUpdates(sequenceObservedColumns, sequenceObservedColumnsDict, connectionDeltaSparse.indices(), connectionDeltaSparse.values())
			return

		def applyReverseConnectionUpdates(sequenceObservedColumns, sequenceObservedColumnsDict, connectionIndices, connectionValues):
			if(optimisationArrayIndexPropertiesEfficientSerialConnections):
				applyReverseConnectionUpdatesSerial(sequenceObservedColumns, sequenceObservedColumnsDict, connectionIndices, connectionValues)
			else:
				applyReverseConnectionUpdatesBatched(sequenceObservedColumns, sequenceObservedColumnsDict, connectionIndices, connectionValues)
			return

		def applyReverseConnectionUpdatesSerial(sequenceObservedColumns, sequenceObservedColumnsDict, connectionIndices, connectionValues):
			databaseNetworkObject = sequenceObservedColumns.databaseNetworkObject
			connectionDevice, sourceConceptIndex, sourceFeatureIndex, targetConceptIndex, targetFeatureIndex = buildReverseConnectionMappedTensors(sequenceObservedColumns, connectionIndices)
			targetCombinedKeys = targetConceptIndex * databaseNetworkObject.f + targetFeatureIndex
			sortedTargetCombinedKeys, sortOrder = pt.sort(targetCombinedKeys)
			sortedBranch = connectionIndices[0].index_select(0, sortOrder)
			sortedSegment = connectionIndices[1].index_select(0, sortOrder)
			sortedSourceConceptIndex = sourceConceptIndex.index_select(0, sortOrder)
			sortedSourceFeatureIndex = sourceFeatureIndex.index_select(0, sortOrder)
			sortedValues = connectionValues.index_select(0, sortOrder)
			uniqueTargetCombinedKeys, counts = pt.unique_consecutive(sortedTargetCombinedKeys, return_counts=True)
			starts = pt.cumsum(counts, 0) - counts
			observedColumnsByConceptIndex = sequenceObservedColumns.getObservedColumnsByConceptIndex(sequenceObservedColumnsDict)
			targetSize = (databaseNetworkObject.arrayNumberOfProperties, multipleDendriticBranchesNumber, arrayNumberOfSegments, databaseNetworkObject.c, databaseNetworkObject.f)
			for targetCombinedKey, start, count in zip(uniqueTargetCombinedKeys.tolist(), starts.tolist(), counts.tolist()):
				end = start + count
				targetConceptIndexValue = int(targetCombinedKey // databaseNetworkObject.f)
				targetFeatureIndexValue = int(targetCombinedKey % databaseNetworkObject.f)
				if(targetConceptIndexValue not in observedColumnsByConceptIndex):
					raise RuntimeError("applyReverseConnectionUpdatesSerial error: missing observed column")
				observedColumn = observedColumnsByConceptIndex[targetConceptIndexValue]
				propertyRow = pt.full((count,), databaseNetworkObject.arrayIndexPropertiesStrengthIndex, dtype=pt.long, device=connectionDevice)
				updateIndices = pt.stack((propertyRow, sortedBranch[start:end], sortedSegment[start:end], sortedSourceConceptIndex[start:end], sortedSourceFeatureIndex[start:end]), dim=0)
				updateSparse = pt.sparse_coo_tensor(updateIndices, sortedValues[start:end], size=targetSize, dtype=arrayType, device=connectionDevice)
				targetSparse = getObservedColumnReverseFeatureConnectionsForTargetFeature(observedColumn, targetFeatureIndexValue, targetDevice=connectionDevice, createMissing=False, ensureCurrentSizeOnLoad=True)
				targetSparse = addSparseUpdateNonNegative(targetSparse, updateSparse)
				setObservedColumnReverseFeatureConnectionsForTargetFeature(observedColumn, targetFeatureIndexValue, targetSparse)
				observedColumn.reverseTrainPreparedTargetFeatureIndices.add(targetFeatureIndexValue)
			return

		def applyReverseConnectionUpdatesBatched(sequenceObservedColumns, sequenceObservedColumnsDict, connectionIndices, connectionValues):
			databaseNetworkObject = sequenceObservedColumns.databaseNetworkObject
			connectionDevice, sourceConceptIndex, sourceFeatureIndex, targetConceptIndex, targetFeatureIndex = buildReverseConnectionMappedTensors(sequenceObservedColumns, connectionIndices)
			targetCombinedKeys = targetConceptIndex * databaseNetworkObject.f + targetFeatureIndex
			targetCombinedKeysUnique = pt.unique(targetCombinedKeys, sorted=True)
			targetSize = (databaseNetworkObject.arrayNumberOfProperties, multipleDendriticBranchesNumber, arrayNumberOfSegments, targetCombinedKeysUnique.shape[0], databaseNetworkObject.c, databaseNetworkObject.f)
			observedColumnsByConceptIndex = sequenceObservedColumns.getObservedColumnsByConceptIndex(sequenceObservedColumnsDict)
			targetSparse = gatherReverseConnectionTargetBucketTensor(observedColumnsByConceptIndex, targetCombinedKeysUnique, databaseNetworkObject, connectionDevice)
			updateSparse = buildReverseConnectionTargetBucketUpdateSparse(databaseNetworkObject, connectionIndices, connectionValues, sourceConceptIndex, sourceFeatureIndex, targetConceptIndex, targetFeatureIndex, targetCombinedKeysUnique, targetSize)
			targetSparse = addSparseUpdateNonNegative(targetSparse, updateSparse)
			scatterReverseConnectionTargetBucketTensor(observedColumnsByConceptIndex, targetCombinedKeysUnique, targetSparse)
			return

		def buildReverseConnectionMappedTensors(sequenceObservedColumns, connectionIndices):
			connectionDevice = connectionIndices.device
			conceptIndicesTensor = sequenceObservedColumns.conceptIndicesInSequenceObservedTensor.to(connectionDevice)
			featureIndicesInObserved = sequenceObservedColumns.featureIndicesInObservedTensor.to(connectionDevice)
			sourceConceptIndex = conceptIndicesTensor[connectionIndices[2]]
			sourceFeatureIndex = connectionIndices[3]
			targetConceptIndex = conceptIndicesTensor[connectionIndices[4]]
			targetFeatureIndex = connectionIndices[5]
			if(trainSequenceObservedColumnsUseSequenceFeaturesOnly):
				sourceFeatureIndex = featureIndicesInObserved[sourceFeatureIndex]
				targetFeatureIndex = featureIndicesInObserved[targetFeatureIndex]
			result = connectionDevice, sourceConceptIndex, sourceFeatureIndex, targetConceptIndex, targetFeatureIndex
			return result

		def buildReverseConnectionTargetBucketUpdateSparse(databaseNetworkObject, connectionIndices, connectionValues, sourceConceptIndex, sourceFeatureIndex, targetConceptIndex, targetFeatureIndex, targetCombinedKeysUnique, targetSize):
			targetCombinedKeys = targetConceptIndex * databaseNetworkObject.f + targetFeatureIndex
			targetBucketIndex = pt.searchsorted(targetCombinedKeysUnique, targetCombinedKeys)
			propertyRow = pt.full_like(connectionIndices[1], databaseNetworkObject.arrayIndexPropertiesStrengthIndex)
			updateIndices = pt.stack((propertyRow, connectionIndices[0], connectionIndices[1], targetBucketIndex, sourceConceptIndex, sourceFeatureIndex), dim=0)
			result = pt.sparse_coo_tensor(updateIndices, connectionValues, size=targetSize, dtype=arrayType, device=connectionIndices.device)
			return result

		def gatherReverseConnectionTargetBucketTensor(observedColumnsByConceptIndex, targetCombinedKeysUnique, databaseNetworkObject, targetDevice):
			targetSize = (databaseNetworkObject.arrayNumberOfProperties, multipleDendriticBranchesNumber, arrayNumberOfSegments, targetCombinedKeysUnique.shape[0], databaseNetworkObject.c, databaseNetworkObject.f)
			combinedIndicesList = []
			combinedValuesList = []
			targetConceptIndexList = pt.div(targetCombinedKeysUnique, databaseNetworkObject.f, rounding_mode='floor').detach().cpu().tolist()
			targetFeatureIndexList = pt.remainder(targetCombinedKeysUnique, databaseNetworkObject.f).detach().cpu().tolist()
			for targetBucketIndex, (targetConceptIndexValue, targetFeatureIndexValue) in enumerate(zip(targetConceptIndexList, targetFeatureIndexList)):
				if(int(targetConceptIndexValue) not in observedColumnsByConceptIndex):
					raise RuntimeError("gatherReverseConnectionTargetBucketTensor error: missing observed column")
				observedColumn = observedColumnsByConceptIndex[int(targetConceptIndexValue)]
				targetTensor = getObservedColumnReverseFeatureConnectionsForTargetFeature(observedColumn, int(targetFeatureIndexValue), targetDevice=targetDevice, createMissing=False, ensureCurrentSizeOnLoad=True)
				targetTensor = targetTensor.coalesce()
				if(targetTensor._nnz() > 0):
					targetIndices = targetTensor.indices()
					targetValues = targetTensor.values()
					targetBucketRow = pt.full((1, targetIndices.shape[1]), targetBucketIndex, dtype=pt.long, device=targetIndices.device)
					batchedIndices = pt.cat([targetIndices[0:3], targetBucketRow, targetIndices[3:]], dim=0)
					combinedIndicesList.append(batchedIndices)
					combinedValuesList.append(targetValues)
			if(len(combinedIndicesList) > 0):
				combinedIndices = pt.cat(combinedIndicesList, dim=1)
				combinedValues = pt.cat(combinedValuesList, dim=0)
				result = pt.sparse_coo_tensor(combinedIndices, combinedValues, size=targetSize, dtype=arrayType, device=targetDevice)
			else:
				result = initialiseReverseConnectionBucketTensor(targetSize, targetDevice)
			return result

		def scatterReverseConnectionTargetBucketTensor(observedColumnsByConceptIndex, targetCombinedKeysUnique, targetSparse):
			targetSparse = targetSparse.coalesce()
			targetIndices = targetSparse.indices()
			targetValues = targetSparse.values()
			targetSortedIndices = targetIndices
			targetSortedValues = targetValues
			targetBucketRanges = {}
			if(targetIndices.numel() > 0):
				sortedBucketIndices, sortOrder = pt.sort(targetIndices[3])
				targetSortedIndices = targetIndices[:, sortOrder]
				targetSortedValues = targetValues.index_select(0, sortOrder)
				uniqueBuckets, counts = pt.unique_consecutive(sortedBucketIndices, return_counts=True)
				starts = pt.cumsum(counts, 0) - counts
				for targetBucketIndexValue, start, count in zip(uniqueBuckets.tolist(), starts.tolist(), counts.tolist()):
					targetBucketRanges[int(targetBucketIndexValue)] = (int(start), int(start + count))
			targetConceptIndexList = pt.div(targetCombinedKeysUnique, targetSparse.size(5), rounding_mode='floor').detach().cpu().tolist()
			targetFeatureIndexList = pt.remainder(targetCombinedKeysUnique, targetSparse.size(5)).detach().cpu().tolist()
			targetTensorSize = (targetSparse.size(0), targetSparse.size(1), targetSparse.size(2), targetSparse.size(4), targetSparse.size(5))
			for targetBucketIndex, (targetConceptIndexValue, targetFeatureIndexValue) in enumerate(zip(targetConceptIndexList, targetFeatureIndexList)):
				if(int(targetConceptIndexValue) not in observedColumnsByConceptIndex):
					raise RuntimeError("scatterReverseConnectionTargetBucketTensor error: missing observed column")
				if(targetBucketIndex in targetBucketRanges):
					start, end = targetBucketRanges[targetBucketIndex]
					targetTensorIndices = pt.stack((targetSortedIndices[0, start:end], targetSortedIndices[1, start:end], targetSortedIndices[2, start:end], targetSortedIndices[4, start:end], targetSortedIndices[5, start:end]), dim=0)
					targetTensorValues = targetSortedValues[start:end]
					targetTensor = pt.sparse_coo_tensor(targetTensorIndices, targetTensorValues, size=targetTensorSize, dtype=arrayType, device=targetSparse.device)
				else:
					targetTensor = initialiseReverseConnectionBucketTensor(targetTensorSize, targetSparse.device)
				observedColumn = observedColumnsByConceptIndex[int(targetConceptIndexValue)]
				setObservedColumnReverseFeatureConnectionsForTargetFeature(observedColumn, int(targetFeatureIndexValue), targetTensor)
				observedColumn.reverseTrainPreparedTargetFeatureIndices.add(int(targetFeatureIndexValue))
			return

		def initialiseReverseConnectionBucketTensor(targetSize, targetDevice):
			indices = pt.empty((len(targetSize), 0), dtype=pt.long, device=targetDevice)
			values = pt.empty((0,), dtype=arrayType, device=targetDevice)
			result = pt.sparse_coo_tensor(indices, values, size=targetSize, dtype=arrayType, device=targetDevice)
			return result

		def addSparseUpdateNonNegative(targetSparse, updateSparse):
			if(tuple(targetSparse.size()) != tuple(updateSparse.size())):
				raise RuntimeError("addSparseUpdateNonNegative error: sparse tensor size mismatch")
			result = (targetSparse.coalesce() + updateSparse.coalesce()).coalesce()
			if(result._nnz() > 0):
				result.values().clamp_(min=0)
			return result

	def updateAutoAuxiliaryConnections(databaseNetworkObject, subwordSimilarity=False):
		updateAutoAuxiliaryFeatureConnectionWeights(databaseNetworkObject, subwordSimilarity)
		return

	def updateAutoAuxiliaryFeatureConnectionWeights(databaseNetworkObject, subwordSimilarity):
		import GIAANNnlp_auxiliaryNeuronsSimilarWords
		ensureAutoAuxiliaryFeatureRecords(databaseNetworkObject, subwordSimilarity)
		removeAutoParentKeysForMode(databaseNetworkObject, subwordSimilarity)
		if(getAutoPrimeFeatureEnabled(subwordSimilarity)):
			updateAutoAuxiliaryFeatureConnectionWeightsForFeatureType(databaseNetworkObject, subwordSimilarity, True)
		if(getAutoSecondaryFeatureEnabled(subwordSimilarity)):
			updateAutoAuxiliaryFeatureConnectionWeightsForFeatureType(databaseNetworkObject, subwordSimilarity, False)
		if(auxiliaryNeuronsAutoInference):
			removeAutoParentKeysForMode(databaseNetworkObject, subwordSimilarity)
		databaseNetworkObject.auxiliaryNeuronsSimilarWordsFeatureIndexWeightsByParentWord = GIAANNnlp_auxiliaryNeuronsSimilarWords.buildAuxiliaryFeatureIndexWeightsByParentWord(databaseNetworkObject)
		GIAANNnlp_auxiliaryNeuronsSimilarWords.invalidateDatabaseAuxiliaryInputConnectionCaches(databaseNetworkObject)
		return

	def calculateDynamicAutoPrimeAuxiliaryConceptActivations(databaseNetworkObject, sourceConceptIndex, sourceFeatureIndex, sourceActivationValue, targetDevice):
		import GIAANNnlp_auxiliaryNeuronsSimilarWords
		validateDynamicAutoPrimeInferenceSourceFeature(sourceFeatureIndex)
		validateDynamicAutoInferenceSourceActivation(sourceActivationValue, "calculateDynamicAutoPrimeAuxiliaryConceptActivations")
		normalisedSourceConceptIndex = GIAANNnlp_auxiliaryNeuronsSimilarWords.normaliseAuxiliaryParentMapConceptIndex(databaseNetworkObject, sourceConceptIndex)
		sourceWord = GIAANNnlp_auxiliaryNeuronsSimilarWords.normaliseSimilarityWord(databaseNetworkObject.conceptColumnsList[normalisedSourceConceptIndex])
		connectionWeights = {}
		if(getAutoPrimeFeatureEnabled(False)):
			addDynamicAutoPrimeAuxiliaryFeatureActivationsForDataset(databaseNetworkObject, connectionWeights, sourceWord, getAutoPrimePrefix(False), getAutoAuxiliaryFeatureDatasetFile(False, True))
		if(getAutoPrimeFeatureEnabled(True)):
			addDynamicAutoPrimeSubwordAuxiliaryFeatureActivations(databaseNetworkObject, connectionWeights, sourceWord)
		result = buildDynamicAutoAuxiliaryActivationVector(databaseNetworkObject, connectionWeights, sourceActivationValue, targetDevice)
		return result

	def calculateDynamicAutoSecondaryAuxiliaryFeatureActivations(observedColumn, sourceFeatureIndex, sourceActivationValue, targetDevice):
		import GIAANNnlp_auxiliaryNeuronsSimilarWords
		databaseNetworkObject = observedColumn.databaseNetworkObject
		normalisedConceptIndex = GIAANNnlp_auxiliaryNeuronsSimilarWords.normaliseAuxiliaryParentMapConceptIndex(databaseNetworkObject, observedColumn.conceptIndex)
		normalisedSourceFeatureIndex = int(sourceFeatureIndex)
		validateDynamicAutoSecondaryInferenceSourceFeature(databaseNetworkObject, normalisedSourceFeatureIndex)
		validateDynamicAutoInferenceSourceActivation(sourceActivationValue, "calculateDynamicAutoSecondaryAuxiliaryFeatureActivations")
		sourceWord = GIAANNnlp_auxiliaryNeuronsSimilarWords.normaliseSimilarityWord(databaseNetworkObject.conceptFeaturesList[normalisedSourceFeatureIndex])
		connectionWeights = {}
		if(getAutoSecondaryFeatureEnabled(False)):
			addDynamicAutoSecondaryAuxiliaryFeatureActivationsForDataset(databaseNetworkObject, connectionWeights, sourceWord, normalisedConceptIndex, getAutoSecondaryPrefix(False), getAutoAuxiliaryFeatureDatasetFile(False, False))
		if(getAutoSecondaryFeatureEnabled(True)):
			addDynamicAutoSecondarySubwordAuxiliaryFeatureActivations(databaseNetworkObject, connectionWeights, sourceWord, normalisedConceptIndex)
		result = buildDynamicAutoAuxiliaryActivationVector(databaseNetworkObject, connectionWeights, sourceActivationValue, targetDevice)
		return result

	def validateDynamicAutoPrimeInferenceSourceFeature(sourceFeatureIndex):
		if(int(sourceFeatureIndex) != featureIndexPrimeConceptNeuron):
			raise RuntimeError("validateDynamicAutoPrimeInferenceSourceFeature error: sourceFeatureIndex must be the active prime concept feature")
		return

	def validateDynamicAutoSecondaryInferenceSourceFeature(databaseNetworkObject, sourceFeatureIndex):
		normalisedSourceFeatureIndex = int(sourceFeatureIndex)
		if(normalisedSourceFeatureIndex <= featureIndexPrimeConceptNeuron or normalisedSourceFeatureIndex >= len(databaseNetworkObject.conceptFeaturesList)):
			raise RuntimeError("validateDynamicAutoSecondaryInferenceSourceFeature error: sourceFeatureIndex must be an active secondary feature")
		return

	def validateDynamicAutoInferenceSourceActivation(sourceActivationValue, functionName):
		if(float(sourceActivationValue.item()) <= auxiliaryNeuronsSimilarWordsMinimumSimilarity):
			raise RuntimeError(functionName + " error: dynamic auto inference source activation must be active")
		return

	def addDynamicAutoPrimeAuxiliaryFeatureActivationsForDataset(databaseNetworkObject, connectionWeights, sourceWord, auxiliaryFeaturePrefix, datasetFile):
		import GIAANNnlp_auxiliaryNeuronsSimilarWords
		targetRecords = getAutoAuxiliaryFeatureDatasetTargetRecordsForSourceWord(datasetFile, sourceWord)
		similarityThreshold = GIAANNnlp_auxiliaryNeuronsSimilarWords.getSimilarityThresholdForAuxiliaryFeaturePrefix(auxiliaryFeaturePrefix)
		for targetWord, activationWeight in targetRecords:
			if(activationWeight >= similarityThreshold):
				if(targetWord not in databaseNetworkObject.conceptColumnsDict):
					raise RuntimeError("addDynamicAutoPrimeAuxiliaryFeatureActivationsForDataset error: targetWord missing from conceptColumnsDict")
				targetConceptIndex = databaseNetworkObject.conceptColumnsDict[targetWord]
				auxiliaryFeatureWord = GIAANNnlp_auxiliaryNeuronsSimilarWords.buildConceptColumnAuxiliaryFeatureName(databaseNetworkObject, auxiliaryFeaturePrefix, targetConceptIndex, targetWord)
				if(auxiliaryFeatureWord not in databaseNetworkObject.auxiliaryNeuronsSimilarWordsFeaturesDict):
					raise RuntimeError("addDynamicAutoPrimeAuxiliaryFeatureActivationsForDataset error: target auxiliary feature word missing")
				auxiliaryFeatureIndex = databaseNetworkObject.auxiliaryNeuronsSimilarWordsFeaturesDict[auxiliaryFeatureWord]
				mergeDynamicAutoAuxiliaryFeatureActivation(connectionWeights, auxiliaryFeatureIndex, activationWeight)
		return

	def addDynamicAutoSecondaryAuxiliaryFeatureActivationsForDataset(databaseNetworkObject, connectionWeights, sourceWord, conceptIndex, auxiliaryFeaturePrefix, datasetFile):
		import GIAANNnlp_auxiliaryNeuronsSimilarWords
		targetRecords = getAutoAuxiliaryFeatureDatasetTargetRecordsForSourceWord(datasetFile, sourceWord)
		similarityThreshold = GIAANNnlp_auxiliaryNeuronsSimilarWords.getSimilarityThresholdForAuxiliaryFeaturePrefix(auxiliaryFeaturePrefix)
		for targetWord, activationWeight in targetRecords:
			if(activationWeight >= similarityThreshold):
				if(targetWord not in databaseNetworkObject.conceptFeaturesDict):
					raise RuntimeError("addDynamicAutoSecondaryAuxiliaryFeatureActivationsForDataset error: targetWord missing from conceptFeaturesDict")
				auxiliaryFeatureWord = GIAANNnlp_auxiliaryNeuronsSimilarWords.buildConceptColumnAuxiliaryFeatureName(databaseNetworkObject, auxiliaryFeaturePrefix, conceptIndex, targetWord)
				if(auxiliaryFeatureWord in databaseNetworkObject.auxiliaryNeuronsSimilarWordsFeaturesDict):
					auxiliaryFeatureIndex = databaseNetworkObject.auxiliaryNeuronsSimilarWordsFeaturesDict[auxiliaryFeatureWord]
					mergeDynamicAutoAuxiliaryFeatureActivation(connectionWeights, auxiliaryFeatureIndex, activationWeight)
		return

	def addDynamicAutoPrimeSubwordAuxiliaryFeatureActivations(databaseNetworkObject, connectionWeights, sourceWord):
		import GIAANNnlp_auxiliaryNeuronsSimilarWords
		auxiliaryFeaturePrefix = getAutoPrimePrefix(True)
		similarityThreshold = GIAANNnlp_auxiliaryNeuronsSimilarWords.getSimilarityThresholdForAuxiliaryFeaturePrefix(auxiliaryFeaturePrefix)
		for auxiliaryFeatureWord, auxiliaryFeatureIndex in databaseNetworkObject.auxiliaryNeuronsSimilarWordsFeaturesDict.items():
			if(GIAANNnlp_auxiliaryNeuronsSimilarWords.auxiliaryFeatureWordHasPrefix(auxiliaryFeatureWord, auxiliaryFeaturePrefix)):
				auxiliaryConceptIndex, targetWord = GIAANNnlp_auxiliaryNeuronsSimilarWords.parseConceptColumnAuxiliaryFeatureName(databaseNetworkObject, auxiliaryFeatureWord)
				activationWeight = calculateDynamicAutoSubwordSimilarityWeight(sourceWord, targetWord)
				if(activationWeight >= similarityThreshold):
					mergeDynamicAutoAuxiliaryFeatureActivation(connectionWeights, auxiliaryFeatureIndex, activationWeight)
		return

	def addDynamicAutoSecondarySubwordAuxiliaryFeatureActivations(databaseNetworkObject, connectionWeights, sourceWord, conceptIndex):
		import GIAANNnlp_auxiliaryNeuronsSimilarWords
		auxiliaryFeaturePrefix = getAutoSecondaryPrefix(True)
		similarityThreshold = GIAANNnlp_auxiliaryNeuronsSimilarWords.getSimilarityThresholdForAuxiliaryFeaturePrefix(auxiliaryFeaturePrefix)
		for auxiliaryFeatureWord, auxiliaryFeatureIndex in databaseNetworkObject.auxiliaryNeuronsSimilarWordsFeaturesDict.items():
			if(GIAANNnlp_auxiliaryNeuronsSimilarWords.auxiliaryFeatureWordHasPrefix(auxiliaryFeatureWord, auxiliaryFeaturePrefix)):
				auxiliaryConceptIndex, targetWord = GIAANNnlp_auxiliaryNeuronsSimilarWords.parseConceptColumnAuxiliaryFeatureName(databaseNetworkObject, auxiliaryFeatureWord)
				if(auxiliaryConceptIndex == conceptIndex):
					activationWeight = calculateDynamicAutoSubwordSimilarityWeight(sourceWord, targetWord)
					if(activationWeight >= similarityThreshold):
						mergeDynamicAutoAuxiliaryFeatureActivation(connectionWeights, auxiliaryFeatureIndex, activationWeight)
		return

	def calculateDynamicAutoSubwordSimilarityWeight(sourceWord, targetWord):
		import GIAANNnlp_auxiliaryNeuronsSimilarWords
		normalisedSourceWord = GIAANNnlp_auxiliaryNeuronsSimilarWords.normaliseSimilarityWord(sourceWord)
		normalisedTargetWord = GIAANNnlp_auxiliaryNeuronsSimilarWords.normaliseSimilarityWord(targetWord)
		maxSharedPrefixLength = min(len(normalisedSourceWord), len(normalisedTargetWord))
		sharedPrefixLength = 0
		for characterIndex in range(maxSharedPrefixLength):
			if(normalisedSourceWord[characterIndex] == normalisedTargetWord[characterIndex]):
				sharedPrefixLength += 1
			else:
				break
		if(normalisedSourceWord == normalisedTargetWord):
			result = auxiliaryNeuronsSimilarWordsIdentitySimilarity
		elif(sharedPrefixLength >= int(auxiliaryNeuronsSimilarSubwordPrefixThreshold)):
			result = sharedPrefixLength/max(len(normalisedSourceWord), len(normalisedTargetWord))
		else:
			result = auxiliaryNeuronsSimilarWordsMinimumSimilarity
		return result

	def mergeDynamicAutoAuxiliaryFeatureActivation(connectionWeights, auxiliaryFeatureIndex, activationWeight):
		normalisedAuxiliaryFeatureIndex = int(auxiliaryFeatureIndex)
		normalisedActivationWeight = float(activationWeight)
		currentActivationWeight = connectionWeights.get(normalisedAuxiliaryFeatureIndex)
		if(currentActivationWeight is None or normalisedActivationWeight > currentActivationWeight):
			connectionWeights[normalisedAuxiliaryFeatureIndex] = normalisedActivationWeight
		return

	def buildDynamicAutoAuxiliaryActivationVector(databaseNetworkObject, connectionWeights, sourceActivationValue, targetDevice):
		result = pt.zeros((databaseNetworkObject.fas,), dtype=arrayType, device=targetDevice)
		if(len(connectionWeights) > 0):
			auxiliaryFeatureIndices = []
			activationWeights = []
			for auxiliaryFeatureIndex in sorted(connectionWeights.keys()):
				auxiliaryFeatureIndices.append(auxiliaryFeatureIndex)
				activationWeights.append(connectionWeights[auxiliaryFeatureIndex])
			auxiliaryFeatureIndexTensor = pt.tensor(auxiliaryFeatureIndices, dtype=pt.long, device=targetDevice)
			activationWeightTensor = pt.tensor(activationWeights, dtype=arrayType, device=targetDevice)
			result[auxiliaryFeatureIndexTensor] = activationWeightTensor * sourceActivationValue.to(targetDevice)
		return result

	def getAutoAuxiliaryFeatureDatasetTargetRecordsForSourceWord(datasetFile, sourceWord):
		import GIAANNnlp_auxiliaryNeuronsSimilarWords
		normalisedSourceWord = GIAANNnlp_auxiliaryNeuronsSimilarWords.normaliseSimilarityWord(sourceWord)
		datasetRecordsBySourceWord = loadAutoAuxiliaryFeatureDataset(datasetFile)
		if(normalisedSourceWord in datasetRecordsBySourceWord):
			result = datasetRecordsBySourceWord[normalisedSourceWord]
		else:
			result = []
		return result

	def loadAutoAuxiliaryFeatureDataset(datasetFile):
		global auxiliaryNeuronsAutoFeatureDatasetCacheByFile
		if(datasetFile not in auxiliaryNeuronsAutoFeatureDatasetCacheByFile):
			if(not GIAANNcmn_databaseNetworkFiles.pathExists(datasetFile)):
				raise RuntimeError("loadAutoAuxiliaryFeatureDataset error: missing datasetFile")
			datasetRecordsBySourceWord = {}
			with open(datasetFile, auxiliaryNeuronsAutoFeatureDatasetFileReadMode, encoding=auxiliaryNeuronsAutoFeatureDatasetFileEncoding) as fileObject:
				for lineIndex, line in enumerate(fileObject):
					sourceWord, targetRecords = parseAutoAuxiliaryFeatureDatasetLine(line)
					if(sourceWord in datasetRecordsBySourceWord):
						raise RuntimeError("loadAutoAuxiliaryFeatureDataset error: duplicate sourceWord")
					datasetRecordsBySourceWord[sourceWord] = targetRecords
			auxiliaryNeuronsAutoFeatureDatasetCacheByFile[datasetFile] = datasetRecordsBySourceWord
		result = auxiliaryNeuronsAutoFeatureDatasetCacheByFile[datasetFile]
		return result

	def parseAutoAuxiliaryFeatureDatasetLine(line):
		import GIAANNnlp_auxiliaryNeuronsSimilarWords
		strippedLine = line.strip()
		fields = strippedLine.split(auxiliaryNeuronsAutoFeatureDatasetDelimiter)
		if(len(fields) < auxiliaryNeuronsAutoFeatureDatasetMinimumFields):
			raise RuntimeError("parseAutoAuxiliaryFeatureDatasetLine error: row has insufficient fields")
		if((len(fields) - auxiliaryNeuronsAutoFeatureDatasetSimilarWordStartFieldIndex)%auxiliaryNeuronsAutoFeatureDatasetSimilarWordPairFields != 0):
			raise RuntimeError("parseAutoAuxiliaryFeatureDatasetLine error: row has incomplete similar-word/score pair")
		sourceWord = GIAANNnlp_auxiliaryNeuronsSimilarWords.normaliseSimilarityWord(fields[auxiliaryNeuronsAutoFeatureDatasetSourceWordFieldIndex])
		targetRecords = []
		for fieldIndex in range(auxiliaryNeuronsAutoFeatureDatasetSimilarWordStartFieldIndex, len(fields), auxiliaryNeuronsAutoFeatureDatasetSimilarWordPairFields):
			targetWord = GIAANNnlp_auxiliaryNeuronsSimilarWords.normaliseSimilarityWord(fields[fieldIndex + auxiliaryNeuronsAutoFeatureDatasetSimilarWordOffset])
			activationWeight = GIAANNnlp_auxiliaryNeuronsSimilarWords.normaliseSimilarityWeight(float(fields[fieldIndex + auxiliaryNeuronsAutoFeatureDatasetSimilarityOffset]))
			targetRecords.append((targetWord, activationWeight))
		result = sourceWord, targetRecords
		return result

	def ensureAutoAuxiliaryFeatureRecords(databaseNetworkObject, subwordSimilarity):
		import GIAANNnlp_auxiliaryNeuronsSimilarWords
		if(getAutoPrimeFeatureEnabled(subwordSimilarity)):
			for conceptIndex, conceptName in enumerate(databaseNetworkObject.conceptColumnsList):
				registerAutoAuxiliaryFeatureRecord(databaseNetworkObject, getAutoPrimePrefix(subwordSimilarity), conceptIndex, GIAANNnlp_auxiliaryNeuronsSimilarWords.normaliseSimilarityWord(conceptName))
		if(getAutoSecondaryFeatureEnabled(subwordSimilarity)):
			for featureKey in sorted(buildAutoObservedSecondaryFeatureKeys(databaseNetworkObject)):
				conceptIndex, featureIndex = getAutoFeatureNeuronIndicesFromKey(databaseNetworkObject, featureKey)
				if(featureIndex >= len(databaseNetworkObject.conceptFeaturesList)):
					raise RuntimeError("ensureAutoAuxiliaryFeatureRecords error: featureIndex out of range")
				registerAutoAuxiliaryFeatureRecord(databaseNetworkObject, getAutoSecondaryPrefix(subwordSimilarity), conceptIndex, GIAANNnlp_auxiliaryNeuronsSimilarWords.normaliseSimilarityWord(databaseNetworkObject.conceptFeaturesList[featureIndex]))
		return

	def registerAutoAuxiliaryFeatureRecord(databaseNetworkObject, auxiliaryFeaturePrefix, conceptIndex, auxiliaryBaseWord):
		import GIAANNnlp_auxiliaryNeuronsSimilarWords
		auxiliaryFeatureWord = GIAANNnlp_auxiliaryNeuronsSimilarWords.buildConceptColumnAuxiliaryFeatureName(databaseNetworkObject, auxiliaryFeaturePrefix, conceptIndex, auxiliaryBaseWord)
		GIAANNnlp_auxiliaryNeuronsSimilarWords.registerAuxiliaryFeatureWord(databaseNetworkObject, auxiliaryFeatureWord, True)
		return

	def buildAutoObservedSecondaryFeatureKeys(databaseNetworkObject):
		result = set()
		for conceptIndex in range(databaseNetworkObject.c):
			for featureIndex in GIAANNcmn_databaseNetworkFiles.listObservedColumnSourceFeatureIndices(conceptIndex):
				if(featureIndex != featureIndexPrimeConceptNeuron and autoSecondaryFeatureIndexHasSimilarityWord(databaseNetworkObject, featureIndex)):
					result.add(getAutoFeatureNeuronKey(databaseNetworkObject, conceptIndex, featureIndex))
		return result

	def autoSecondaryFeatureIndexHasSimilarityWord(databaseNetworkObject, featureIndex):
		if(featureIndex < 0 or featureIndex >= len(databaseNetworkObject.conceptFeaturesList)):
			raise RuntimeError("autoSecondaryFeatureIndexHasSimilarityWord error: featureIndex out of range")
		result = str(databaseNetworkObject.conceptFeaturesList[featureIndex]).strip() != auxiliaryNeuronsSimilarWordsFeatureValueEmpty
		return result

	def getAutoPrimeFeatureEnabled(subwordSimilarity):
		if(subwordSimilarity):
			result = auxiliaryNeuronsSimilarSubwordAuto and auxiliaryNeuronsSimilarSubwordPrimeConceptFeatures
		else:
			result = auxiliaryNeuronsSimilarWordsAuto and auxiliaryNeuronsSimilarWordsPrimeConceptFeatures
		return result

	def getAutoSecondaryFeatureEnabled(subwordSimilarity):
		if(subwordSimilarity):
			result = auxiliaryNeuronsSimilarSubwordAuto and auxiliaryNeuronsSimilarSubwordSecondaryConceptFeatures
		else:
			result = auxiliaryNeuronsSimilarWordsAuto and auxiliaryNeuronsSimilarWordsSecondaryConceptFeatures
		return result

	def getAutoPrimePrefix(subwordSimilarity):
		if(subwordSimilarity):
			result = auxiliaryNeuronsSimilarWordsFeatureNamePrefixSubwordPrimeConcept
		else:
			result = auxiliaryNeuronsSimilarWordsFeatureNamePrefixPrimeConcept
		return result

	def getAutoSecondaryPrefix(subwordSimilarity):
		if(subwordSimilarity):
			result = auxiliaryNeuronsSimilarWordsFeatureNamePrefixSubwordSecondary
		else:
			result = auxiliaryNeuronsSimilarWordsFeatureNamePrefixSecondary
		return result

	def getAutoModePrefixes(subwordSimilarity):
		result = [getAutoPrimePrefix(subwordSimilarity), getAutoSecondaryPrefix(subwordSimilarity)]
		return result

	def removeAutoParentKeysForMode(databaseNetworkObject, subwordSimilarity):
		modePrefixes = getAutoModePrefixes(subwordSimilarity)
		parentKeys = list(databaseNetworkObject.auxiliaryNeuronsSimilarWordsFeatureWordWeightsByParentWord.keys())
		for parentKey in parentKeys:
			parentPrefix, parentWord = parseAutoParentKey(parentKey)
			if(parentPrefix in modePrefixes):
				del databaseNetworkObject.auxiliaryNeuronsSimilarWordsFeatureWordWeightsByParentWord[parentKey]
		return

	def parseAutoParentKey(parentKey):
		parts = parentKey.split(auxiliaryNeuronsSimilarWordsFeatureNameDelimiter, 1)
		if(len(parts) != 2):
			raise RuntimeError("parseAutoParentKey error: invalid parentKey")
		parentPrefix = parts[0]
		parentWord = parts[1]
		result = parentPrefix, parentWord
		return result

	def updateAutoAuxiliaryFeatureConnectionWeightsForFeatureType(databaseNetworkObject, subwordSimilarity, primeConceptFeatures):
		records = buildAutoAuxiliaryFeatureRecords(databaseNetworkObject, subwordSimilarity, primeConceptFeatures)
		if(records["rowCount"] > 0):
			if(subwordSimilarity):
				if(primeConceptFeatures):
					similarityMatrix = calculateSubwordSimilaritySparseMatrix(records["words"], deviceSparse)
					registerAutoSimilaritySparseMatrixWeights(databaseNetworkObject, records, similarityMatrix)
				else:
					registerAutoSubwordSecondaryFeatureWeights(databaseNetworkObject, records, deviceSparse)
			else:
				registerAutoConnectionPropagationWeights(databaseNetworkObject, records, deviceDense)
		writeAutoAuxiliaryFeatureDataset(databaseNetworkObject, subwordSimilarity, primeConceptFeatures)
		return

	def buildAutoAuxiliaryFeatureRecords(databaseNetworkObject, subwordSimilarity, primeConceptFeatures):
		import GIAANNnlp_auxiliaryNeuronsSimilarWords
		prefix = getAutoPrimePrefix(subwordSimilarity) if primeConceptFeatures else getAutoSecondaryPrefix(subwordSimilarity)
		conceptIndexList = []
		featureIndexList = []
		auxiliaryFeatureIndexList = []
		wordList = []
		auxiliaryFeatureWordList = []
		for auxiliaryFeatureWord, auxiliaryFeatureIndex in databaseNetworkObject.auxiliaryNeuronsSimilarWordsFeaturesDict.items():
			if(GIAANNnlp_auxiliaryNeuronsSimilarWords.auxiliaryFeatureWordHasPrefix(auxiliaryFeatureWord, prefix)):
				auxiliaryConceptIndex, auxiliaryBaseWord = GIAANNnlp_auxiliaryNeuronsSimilarWords.parseConceptColumnAuxiliaryFeatureName(databaseNetworkObject, auxiliaryFeatureWord)
				if(primeConceptFeatures):
					featureIndex = featureIndexPrimeConceptNeuron
				else:
					if(auxiliaryBaseWord not in databaseNetworkObject.conceptFeaturesDict):
						raise RuntimeError("buildAutoAuxiliaryFeatureRecords error: missing secondary feature word")
					featureIndex = databaseNetworkObject.conceptFeaturesDict[auxiliaryBaseWord]
				conceptIndexList.append(auxiliaryConceptIndex)
				featureIndexList.append(featureIndex)
				auxiliaryFeatureIndexList.append(auxiliaryFeatureIndex)
				wordList.append(auxiliaryBaseWord)
				auxiliaryFeatureWordList.append(auxiliaryFeatureWord)
		rowCount = len(auxiliaryFeatureIndexList)
		result = {"rowCount": rowCount, "prefix": prefix, "primeConceptFeatures": primeConceptFeatures, "conceptIndices": pt.tensor(conceptIndexList, dtype=pt.long, device=deviceSparse), "featureIndices": pt.tensor(featureIndexList, dtype=pt.long, device=deviceSparse), "auxiliaryFeatureIndices": pt.tensor(auxiliaryFeatureIndexList, dtype=pt.long, device=deviceSparse), "words": wordList, "auxiliaryFeatureWords": auxiliaryFeatureWordList}
		validateAutoAuxiliaryFeatureRecords(databaseNetworkObject, result, primeConceptFeatures)
		return result

	def writeAutoAuxiliaryFeatureDataset(databaseNetworkObject, subwordSimilarity, primeConceptFeatures):
		prefix = getAutoPrimePrefix(subwordSimilarity) if primeConceptFeatures else getAutoSecondaryPrefix(subwordSimilarity)
		datasetFile = getAutoAuxiliaryFeatureDatasetFile(subwordSimilarity, primeConceptFeatures)
		sourceWords = buildAutoAuxiliaryFeatureDatasetSourceWords(databaseNetworkObject, primeConceptFeatures)
		targetRecordsBySourceWord = {}
		for sourceWord in sourceWords:
			targetRecordsBySourceWord[sourceWord] = buildAutoAuxiliaryFeatureDatasetTargetRecords(databaseNetworkObject, prefix, sourceWord)
		writeAutoAuxiliaryFeatureDatasetRows(datasetFile, sourceWords, targetRecordsBySourceWord)
		return

	def getAutoAuxiliaryFeatureDatasetFile(subwordSimilarity, primeConceptFeatures):
		if(subwordSimilarity):
			if(primeConceptFeatures):
				result = auxiliaryNeuronsSimilarSubwordPrimeConceptFeaturesDatasetFile
			else:
				result = auxiliaryNeuronsSimilarSubwordSecondaryConceptFeaturesDatasetFile
		else:
			if(primeConceptFeatures):
				result = auxiliaryNeuronsSimilarWordsPrimeConceptFeaturesDatasetFile
			else:
				result = auxiliaryNeuronsSimilarWordsSecondaryConceptFeaturesDatasetFile
		return result

	def buildAutoAuxiliaryFeatureDatasetSourceWords(databaseNetworkObject, primeConceptFeatures):
		import GIAANNnlp_auxiliaryNeuronsSimilarWords
		result = []
		if(primeConceptFeatures):
			for conceptIndex in range(databaseNetworkObject.c):
				result.append(GIAANNnlp_auxiliaryNeuronsSimilarWords.normaliseSimilarityWord(databaseNetworkObject.conceptColumnsList[conceptIndex]))
		else:
			if(databaseNetworkObject.f != len(databaseNetworkObject.conceptFeaturesList)):
				raise RuntimeError("buildAutoAuxiliaryFeatureDatasetSourceWords error: source feature index out of range")
			for featureIndex in range(featureIndexPrimeConceptNeuron+1, databaseNetworkObject.f):
				if(featureIndex < 0 or featureIndex >= len(databaseNetworkObject.conceptFeaturesList)):
					raise RuntimeError("buildAutoAuxiliaryFeatureDatasetSourceWords error: source feature index out of range")
				if(autoSecondaryFeatureIndexHasSimilarityWord(databaseNetworkObject, featureIndex)):
					result.append(GIAANNnlp_auxiliaryNeuronsSimilarWords.normaliseSimilarityWord(databaseNetworkObject.conceptFeaturesList[featureIndex]))
		validateAutoAuxiliaryFeatureDatasetSourceWords(result)
		return result

	def validateAutoAuxiliaryFeatureDatasetSourceWords(sourceWords):
		seenSourceWords = set()
		for sourceWord in sourceWords:
			if(sourceWord in seenSourceWords):
				raise RuntimeError("validateAutoAuxiliaryFeatureDatasetSourceWords error: duplicate source word")
			seenSourceWords.add(sourceWord)
		return

	def buildAutoAuxiliaryFeatureDatasetTargetRecords(databaseNetworkObject, prefix, sourceWord):
		import GIAANNnlp_auxiliaryNeuronsSimilarWords
		similarityThreshold = GIAANNnlp_auxiliaryNeuronsSimilarWords.getSimilarityThresholdForAuxiliaryFeaturePrefix(prefix)
		normalisedSourceWord = GIAANNnlp_auxiliaryNeuronsSimilarWords.normaliseSimilarityWord(sourceWord)
		parentKey = GIAANNnlp_auxiliaryNeuronsSimilarWords.buildSimilarityParentKey(prefix, normalisedSourceWord)
		targetRecordsByWord = {}
		if(parentKey in databaseNetworkObject.auxiliaryNeuronsSimilarWordsFeatureWordWeightsByParentWord):
			for auxiliaryFeatureWord, activationWeight in databaseNetworkObject.auxiliaryNeuronsSimilarWordsFeatureWordWeightsByParentWord[parentKey].items():
				auxiliaryConceptIndex, targetWord = GIAANNnlp_auxiliaryNeuronsSimilarWords.parseConceptColumnAuxiliaryFeatureName(databaseNetworkObject, auxiliaryFeatureWord)
				normalisedActivationWeight = GIAANNnlp_auxiliaryNeuronsSimilarWords.normaliseSimilarityWeight(activationWeight)
				if(targetWord != normalisedSourceWord):
					if(normalisedActivationWeight < similarityThreshold):
						raise RuntimeError("buildAutoAuxiliaryFeatureDatasetTargetRecords error: activationWeight below threshold")
					currentActivationWeight = targetRecordsByWord.get(targetWord)
					if(currentActivationWeight is None or normalisedActivationWeight > currentActivationWeight):
						targetRecordsByWord[targetWord] = normalisedActivationWeight
		result = sorted(targetRecordsByWord.items(), key=lambda item: (-item[1], item[0]))
		return result

	def writeAutoAuxiliaryFeatureDatasetRows(datasetFile, sourceWords, targetRecordsBySourceWord):
		datasetDirectory = os.path.dirname(datasetFile)
		if(datasetDirectory == auxiliaryNeuronsSimilarWordsFeatureValueEmpty):
			raise RuntimeError("writeAutoAuxiliaryFeatureDatasetRows error: dataset directory is empty")
		os.makedirs(datasetDirectory, exist_ok=True)
		temporaryFile = datasetFile + auxiliaryNeuronsAutoFeatureDatasetTempFileSuffix
		with open(temporaryFile, auxiliaryNeuronsAutoFeatureDatasetFileWriteMode, encoding=auxiliaryNeuronsAutoFeatureDatasetFileEncoding) as fileObject:
			for sourceWord in sourceWords:
				fields = [sourceWord]
				for targetWord, activationWeight in targetRecordsBySourceWord[sourceWord]:
					fields.append(targetWord)
					fields.append(auxiliaryNeuronsAutoFeatureDatasetSimilarityFormat.format(activationWeight))
				fileObject.write(auxiliaryNeuronsAutoFeatureDatasetDelimiter.join(fields) + auxiliaryNeuronsAutoFeatureDatasetLineTerminator)
		os.replace(temporaryFile, datasetFile)
		if(datasetFile in auxiliaryNeuronsAutoFeatureDatasetCacheByFile):
			del auxiliaryNeuronsAutoFeatureDatasetCacheByFile[datasetFile]
		return

	def validateAutoAuxiliaryFeatureRecords(databaseNetworkObject, records, primeConceptFeatures):
		if(primeConceptFeatures):
			recordConceptIndices = set(records["conceptIndices"].detach().cpu().tolist())
			for conceptIndex in range(databaseNetworkObject.c):
				if(conceptIndex not in recordConceptIndices):
					raise RuntimeError("validateAutoAuxiliaryFeatureRecords error: missing prime auxiliary feature conceptIndex=" + str(conceptIndex))
		else:
			recordFeatureKeys = set((records["conceptIndices"]*databaseNetworkObject.f + records["featureIndices"]).detach().cpu().tolist())
			for featureKey in sorted(buildAutoObservedSecondaryFeatureKeys(databaseNetworkObject)):
				if(featureKey not in recordFeatureKeys):
					conceptIndex, featureIndex = getAutoFeatureNeuronIndicesFromKey(databaseNetworkObject, featureKey)
					raise RuntimeError("validateAutoAuxiliaryFeatureRecords error: missing secondary auxiliary feature conceptIndex=" + str(conceptIndex) + ", featureIndex=" + str(featureIndex))
		return

	def registerAutoConnectionPropagationWeights(databaseNetworkObject, records, targetDevice):
		if(records["primeConceptFeatures"]):
			registerAutoPrimeConnectionPropagationWeights(databaseNetworkObject, records, targetDevice)
		else:
			registerAutoSecondaryConnectionPropagationWeights(databaseNetworkObject, records, targetDevice)
		return

	def registerAutoPrimeConnectionPropagationWeights(databaseNetworkObject, records, targetDevice):
		recordRowsByFeatureKey = buildAutoRecordRowsByFeatureKey(databaseNetworkObject, records)
		for sourceRowIndex in range(records["rowCount"]):
			print("auxiliaryNeuronsSimilarWordsAuto: columnIndex=", int(records["conceptIndices"][sourceRowIndex].item()), ", featureIndex=", int(records["featureIndices"][sourceRowIndex].item()))
			registerAutoSimilarityRecordWeight(databaseNetworkObject, records, sourceRowIndex, sourceRowIndex, auxiliaryNeuronsSimilarWordsIdentitySimilarity)
			similarFeatureActivations = calculateTwoHopSimilarFeatureActivations(databaseNetworkObject, records, sourceRowIndex, targetDevice)
			similarFeatureActivations = similarFeatureActivations.coalesce()
			if(similarFeatureActivations._nnz() > 0):
				similarFeatureKeys = similarFeatureActivations.indices()[0]
				similarFeatureValues = similarFeatureActivations.values()
				activeMask = similarFeatureValues >= auxiliaryNeuronsSimilarWordsAutoThreshold
				activeFeatureKeys = similarFeatureKeys[activeMask]
				activeFeatureValues = similarFeatureValues[activeMask]
				for activeIndex in range(activeFeatureKeys.shape[0]):
					similarFeatureKey = int(activeFeatureKeys[activeIndex].item())
					if(similarFeatureKey in recordRowsByFeatureKey):
						targetRowIndex = recordRowsByFeatureKey[similarFeatureKey]
						if(autoConnectionPropagationRecordPairAllowed(records, sourceRowIndex, targetRowIndex)):
							activationWeight = float(activeFeatureValues[activeIndex].item())
							registerAutoSimilarityRecordWeight(databaseNetworkObject, records, sourceRowIndex, targetRowIndex, activationWeight)
		return

	def registerAutoSecondaryConnectionPropagationWeights(databaseNetworkObject, records, targetDevice):
		featureRecords = buildAutoSecondaryFeatureIndexRecords(databaseNetworkObject, targetDevice)
		recordRowsByFeatureIndex = buildAutoRecordRowsByFeatureIndex(records)
		recordRowsByFeatureKey = buildAutoRecordRowsByFeatureKey(databaseNetworkObject, records)
		for sourceRowIndex in range(records["rowCount"]):
			registerAutoSimilarityRecordWeight(databaseNetworkObject, records, sourceRowIndex, sourceRowIndex, auxiliaryNeuronsSimilarWordsIdentitySimilarity)
		for sourceFeatureRowIndex in range(featureRecords["rowCount"]):
			sourceFeatureIndex = int(featureRecords["featureIndices"][sourceFeatureRowIndex].item())
			print("auxiliaryNeuronsSimilarWordsAuto: columnIndex=all, featureIndex=", sourceFeatureIndex)
			if(sourceFeatureIndex in recordRowsByFeatureIndex):
				sourceRowIndices = recordRowsByFeatureIndex[sourceFeatureIndex]
				if(auxiliaryNeuronsSimilarWordsSecondaryConceptFeaturesLimit):
					if(autoSecondaryFeatureIndexSharedSourceAllowed(databaseNetworkObject, sourceFeatureIndex, sourceRowIndices)):
						similarFeatureActivations = calculateTwoHopSimilarSecondaryFeatureIndexActivations(databaseNetworkObject, records, sourceFeatureIndex, sourceRowIndices, targetDevice)
					else:
						similarFeatureActivations = createEmptyFeatureIndexActivationSparseVector(databaseNetworkObject, targetDevice)
				else:
					similarFeatureActivations = calculateTwoHopSimilarSecondaryFeatureIndexActivations(databaseNetworkObject, records, sourceFeatureIndex, sourceRowIndices, targetDevice)
			else:
				similarFeatureActivations = createEmptyFeatureIndexActivationSparseVector(databaseNetworkObject, targetDevice)
			similarFeatureActivations = similarFeatureActivations.coalesce()
			if(similarFeatureActivations._nnz() > 0):
				similarFeatureIndices = similarFeatureActivations.indices()[0]
				similarFeatureValues = similarFeatureActivations.values()
				activeMask = similarFeatureValues >= auxiliaryNeuronsSimilarWordsAutoThreshold
				activeFeatureIndices = similarFeatureIndices[activeMask]
				activeFeatureValues = similarFeatureValues[activeMask]
				for activeIndex in range(activeFeatureIndices.shape[0]):
					targetFeatureIndex = int(activeFeatureIndices[activeIndex].item())
					if(targetFeatureIndex in recordRowsByFeatureIndex):
						targetRowIndices = recordRowsByFeatureIndex[targetFeatureIndex]
						if(not auxiliaryNeuronsSimilarWordsSecondaryConceptFeaturesLimit or autoSecondaryFeatureIndexSharedSourceAllowed(databaseNetworkObject, targetFeatureIndex, targetRowIndices)):
							activationWeight = float(activeFeatureValues[activeIndex].item())
							for targetRowIndex in targetRowIndices:
								registerAutoSimilarityFeatureIndexWeight(databaseNetworkObject, records, sourceFeatureIndex, targetRowIndex, recordRowsByFeatureKey, activationWeight)
		return

	if(auxiliaryNeuronsSimilarWordsSecondaryConceptFeaturesLimit):
		def autoSecondaryFeatureIndexSharedSourceAllowed(databaseNetworkObject, sourceFeatureIndex, sourceRowIndices):
			sharedSourceFeatureIndexFraction = autoSecondaryFeatureIndexSharedSourceFraction(databaseNetworkObject, sourceFeatureIndex, sourceRowIndices)
			sourceFeatureColumnCount = len(sourceRowIndices)
			result = sourceFeatureColumnCount >= auxiliaryNeuronsSimilarWordsSecondaryConceptFeaturesMinimumSharedSourceFeatureIndex and sharedSourceFeatureIndexFraction < auxiliaryNeuronsSimilarWordsSecondaryConceptFeaturesMaximumSharedSourceFeatureIndexFraction
			return result

		def autoSecondaryFeatureIndexSharedSourceFraction(databaseNetworkObject, sourceFeatureIndex, sourceRowIndices):
			if(databaseNetworkObject.c < 1):
				raise RuntimeError("autoSecondaryFeatureIndexSharedSourceFraction error: concept column count must be positive")
			if(sourceFeatureIndex <= featureIndexPrimeConceptNeuron or sourceFeatureIndex >= databaseNetworkObject.f):
				raise RuntimeError("autoSecondaryFeatureIndexSharedSourceFraction error: sourceFeatureIndex out of range")
			sourceFeatureColumnCount = len(sourceRowIndices)
			if(sourceFeatureColumnCount < 0 or sourceFeatureColumnCount > databaseNetworkObject.c):
				raise RuntimeError("autoSecondaryFeatureIndexSharedSourceFraction error: source feature column count out of range")
			result = sourceFeatureColumnCount/databaseNetworkObject.c
			return result


	def buildAutoRecordRowsByFeatureKey(databaseNetworkObject, records):
		result = {}
		for rowIndex in range(records["rowCount"]):
			featureKey = getAutoFeatureNeuronKey(databaseNetworkObject, int(records["conceptIndices"][rowIndex].item()), int(records["featureIndices"][rowIndex].item()))
			if(featureKey in result):
				raise RuntimeError("buildAutoRecordRowsByFeatureKey error: duplicate feature key")
			result[featureKey] = rowIndex
		return result

	def buildAutoRecordRowsByFeatureIndex(records):
		result = {}
		for rowIndex in range(records["rowCount"]):
			featureIndex = int(records["featureIndices"][rowIndex].item())
			if(featureIndex not in result):
				result[featureIndex] = []
			result[featureIndex].append(rowIndex)
		return result

	def getAutoFeatureNeuronKey(databaseNetworkObject, conceptIndex, featureIndex):
		if(conceptIndex < 0 or conceptIndex >= databaseNetworkObject.c):
			raise RuntimeError("getAutoFeatureNeuronKey error: conceptIndex out of range")
		if(featureIndex < 0 or featureIndex >= databaseNetworkObject.f):
			raise RuntimeError("getAutoFeatureNeuronKey error: featureIndex out of range")
		result = int(conceptIndex) * databaseNetworkObject.f + int(featureIndex)
		return result

	def getAutoFeatureNeuronIndicesFromKey(databaseNetworkObject, featureKey):
		if(featureKey < 0 or featureKey >= databaseNetworkObject.c * databaseNetworkObject.f):
			raise RuntimeError("getAutoFeatureNeuronIndicesFromKey error: featureKey out of range")
		conceptIndex = int(featureKey // databaseNetworkObject.f)
		featureIndex = int(featureKey % databaseNetworkObject.f)
		result = conceptIndex, featureIndex
		return result

	def autoConnectionPropagationRecordPairAllowed(records, sourceRowIndex, targetRowIndex):
		if(records["primeConceptFeatures"]):
			result = True
		else:
			if(auxiliaryNeuronsSimilarWordsSecondaryConceptFeaturesIdentifySameColumn):
				result = int(records["conceptIndices"][sourceRowIndex].item()) == int(records["conceptIndices"][targetRowIndex].item())
			else:
				result = True
		return result

	def calculateTwoHopSimilarFeatureActivations(databaseNetworkObject, records, sourceRowIndex, targetDevice):
		conceptIndex = int(records["conceptIndices"][sourceRowIndex].item())
		featureIndex = int(records["featureIndices"][sourceRowIndex].item())
		observedColumn = loadObservedColumnForAutoConnectionPropagation(databaseNetworkObject, conceptIndex, targetDevice)
		forwardConnectionTensor = observedColumn.getFeatureConnectionsForSourceFeature(featureIndex, targetDevice=targetDevice, createMissing=False, ensureCurrentSizeOnLoad=True)
		forwardSiblingActivations = buildFeatureActivationVectorFromConnectionTensor(databaseNetworkObject, forwardConnectionTensor, targetDevice, "calculateTwoHopSimilarFeatureActivations")
		reverseConnectionTensor = getObservedColumnReverseFeatureConnectionsForTargetFeature(observedColumn, featureIndex, targetDevice=targetDevice, createMissing=False, ensureCurrentSizeOnLoad=True)
		reverseSiblingActivations = buildFeatureActivationVectorFromConnectionTensor(databaseNetworkObject, reverseConnectionTensor, targetDevice, "calculateTwoHopSimilarFeatureActivations")
		unloadObservedColumnConnectionsForAutoConnectionPropagation(observedColumn, [featureIndex], [featureIndex])
		similarFeatureActivationsForward = propagateLocalSiblingFeatureActivations(databaseNetworkObject, forwardSiblingActivations, True, targetDevice)
		similarFeatureActivationsReverse = propagateLocalSiblingFeatureActivations(databaseNetworkObject, reverseSiblingActivations, False, targetDevice)
		result = addFeatureActivationSparseVectors(databaseNetworkObject, similarFeatureActivationsForward, similarFeatureActivationsReverse, targetDevice)
		result = normaliseFeatureActivationSparseVector(databaseNetworkObject, result, targetDevice)
		return result

	def buildFeatureActivationVectorFromConnectionTensor(databaseNetworkObject, connectionTensor, targetDevice, functionName):
		connectionStrength = getAutoDirectConnectionStrength(databaseNetworkObject, connectionTensor, functionName)
		if(connectionStrength._nnz() > 0):
			indices = connectionStrength.indices()
			values = connectionStrength.values().to(targetDevice)
			if((values < auxiliaryNeuronsSimilarWordsMinimumSimilarity).any()):
				raise RuntimeError(functionName + " error: connection strength values must be non-negative")
			featureKeys = indices[2].to(targetDevice)*databaseNetworkObject.f + indices[3].to(targetDevice)
			result = pt.sparse_coo_tensor(featureKeys.view(1, -1), values, size=(databaseNetworkObject.c * databaseNetworkObject.f,), dtype=arrayType, device=targetDevice).coalesce()
			result = normaliseFeatureActivationSparseVector(databaseNetworkObject, result, targetDevice)
		else:
			result = createEmptyFeatureActivationSparseVector(databaseNetworkObject, targetDevice)
		return result

	def propagateLocalSiblingFeatureActivations(databaseNetworkObject, siblingActivations, fireReverseConnections, targetDevice):
		result = createEmptyFeatureActivationSparseVector(databaseNetworkObject, targetDevice)
		siblingActivations = siblingActivations.coalesce()
		if(siblingActivations._nnz() > 0):
			siblingFeatureKeys = siblingActivations.indices()[0]
			siblingActivationValues = siblingActivations.values()
			activeMask = siblingActivationValues > auxiliaryNeuronsSimilarWordsMinimumSimilarity
			siblingFeatureKeys = siblingFeatureKeys[activeMask]
			siblingActivationValues = siblingActivationValues[activeMask]
			for siblingIndex in range(siblingFeatureKeys.shape[0]):
				siblingFeatureKey = int(siblingFeatureKeys[siblingIndex].item())
				siblingActivationValue = float(siblingActivationValues[siblingIndex].item())
				siblingConceptIndex, siblingFeatureIndex = getAutoFeatureNeuronIndicesFromKey(databaseNetworkObject, siblingFeatureKey)
				siblingObservedColumn = loadObservedColumnForAutoConnectionPropagation(databaseNetworkObject, siblingConceptIndex, targetDevice)
				if(fireReverseConnections):
					connectionTensor = getObservedColumnReverseFeatureConnectionsForTargetFeature(siblingObservedColumn, siblingFeatureIndex, targetDevice=targetDevice, createMissing=False, ensureCurrentSizeOnLoad=True)
				else:
					connectionTensor = siblingObservedColumn.getFeatureConnectionsForSourceFeature(siblingFeatureIndex, targetDevice=targetDevice, createMissing=False, ensureCurrentSizeOnLoad=True)
				connectionActivations = buildFeatureActivationVectorFromConnectionTensor(databaseNetworkObject, connectionTensor, targetDevice, "propagateLocalSiblingFeatureActivations")
				if(fireReverseConnections):
					unloadObservedColumnConnectionsForAutoConnectionPropagation(siblingObservedColumn, None, [siblingFeatureIndex])
				else:
					unloadObservedColumnConnectionsForAutoConnectionPropagation(siblingObservedColumn, [siblingFeatureIndex], None)
				connectionActivations = multiplyFeatureActivationSparseVector(connectionActivations, siblingActivationValue, targetDevice)
				result = addFeatureActivationSparseVectors(databaseNetworkObject, result, connectionActivations, targetDevice)
		result = normaliseFeatureActivationSparseVector(databaseNetworkObject, result, targetDevice)
		return result


	def calculateTwoHopSimilarSecondaryFeatureIndexActivations(databaseNetworkObject, records, sourceFeatureIndex, sourceRowIndices, targetDevice):
		sourceRowIndicesTensor = pt.tensor(sourceRowIndices, dtype=pt.long, device=records["conceptIndices"].device)
		if(sourceRowIndicesTensor.numel() > 0):
			sourceConceptIndices = records["conceptIndices"].index_select(0, sourceRowIndicesTensor).to(targetDevice)
			forwardSiblingActivations = buildSecondarySourceFeatureSiblingActivations(databaseNetworkObject, sourceConceptIndices, sourceFeatureIndex, True, targetDevice)
			reverseSiblingActivations = buildSecondarySourceFeatureSiblingActivations(databaseNetworkObject, sourceConceptIndices, sourceFeatureIndex, False, targetDevice)
			similarFeatureActivationsForward = propagateLocalSiblingFeatureActivations(databaseNetworkObject, forwardSiblingActivations, True, targetDevice)
			similarFeatureActivationsReverse = propagateLocalSiblingFeatureActivations(databaseNetworkObject, reverseSiblingActivations, False, targetDevice)
			similarFeatureActivations = addFeatureActivationSparseVectors(databaseNetworkObject, similarFeatureActivationsForward, similarFeatureActivationsReverse, targetDevice)
			similarFeatureActivations = normaliseFeatureActivationSparseVector(databaseNetworkObject, similarFeatureActivations, targetDevice)
			result = collapseFeatureActivationSparseVectorByFeatureIndex(databaseNetworkObject, similarFeatureActivations, targetDevice)
		else:
			result = createEmptyFeatureIndexActivationSparseVector(databaseNetworkObject, targetDevice)
		return result

	def buildSecondarySourceFeatureSiblingActivations(databaseNetworkObject, sourceConceptIndices, sourceFeatureIndex, fireForwardConnections, targetDevice):
		result = createEmptyFeatureActivationSparseVector(databaseNetworkObject, targetDevice)
		for sourceConceptIndexTensor in sourceConceptIndices:
			sourceConceptIndex = int(sourceConceptIndexTensor.item())
			observedColumn = loadObservedColumnForAutoConnectionPropagation(databaseNetworkObject, sourceConceptIndex, targetDevice)
			if(fireForwardConnections):
				connectionTensor = observedColumn.getFeatureConnectionsForSourceFeature(sourceFeatureIndex, targetDevice=targetDevice, createMissing=False, ensureCurrentSizeOnLoad=True)
			else:
				connectionTensor = getObservedColumnReverseFeatureConnectionsForTargetFeature(observedColumn, sourceFeatureIndex, targetDevice=targetDevice, createMissing=False, ensureCurrentSizeOnLoad=True)
			connectionActivations = buildFeatureActivationVectorFromConnectionTensor(databaseNetworkObject, connectionTensor, targetDevice, "buildSecondarySourceFeatureSiblingActivations")
			if(fireForwardConnections):
				unloadObservedColumnConnectionsForAutoConnectionPropagation(observedColumn, [sourceFeatureIndex], None)
			else:
				unloadObservedColumnConnectionsForAutoConnectionPropagation(observedColumn, None, [sourceFeatureIndex])
			result = addFeatureActivationSparseVectors(databaseNetworkObject, result, connectionActivations, targetDevice)
		result = normaliseFeatureActivationSparseVector(databaseNetworkObject, result, targetDevice)
		return result

	def collapseFeatureActivationSparseVectorByFeatureIndex(databaseNetworkObject, featureActivations, targetDevice):
		featureActivations = featureActivations.coalesce()
		if(tuple(featureActivations.size()) != (databaseNetworkObject.c * databaseNetworkObject.f,)):
			raise RuntimeError("collapseFeatureActivationSparseVectorByFeatureIndex error: feature activation vector size mismatch")
		if(featureActivations._nnz() > 0):
			featureIndices = featureActivations.indices()[0].remainder(databaseNetworkObject.f)
			secondaryMask = featureIndices != featureIndexPrimeConceptNeuron
			featureIndices = featureIndices[secondaryMask]
			values = featureActivations.values()[secondaryMask]
			if(featureIndices.numel() > 0):
				result = pt.sparse_coo_tensor(featureIndices.view(1, -1), values, size=(databaseNetworkObject.f,), dtype=arrayType, device=targetDevice).coalesce()
				result = normaliseFeatureIndexActivationSparseVector(databaseNetworkObject, result, targetDevice)
			else:
				result = createEmptyFeatureIndexActivationSparseVector(databaseNetworkObject, targetDevice)
		else:
			result = createEmptyFeatureIndexActivationSparseVector(databaseNetworkObject, targetDevice)
		return result


	def getAutoDirectConnectionStrength(databaseNetworkObject, connectionTensor, functionName):
		connectionStrength = connectionTensor[databaseNetworkObject.arrayIndexPropertiesStrengthIndex].coalesce()
		if(connectionStrength.dim() != 4):
			raise RuntimeError(functionName + " error: connection strength tensor rank mismatch")
		if(connectionStrength.size(1) <= arrayIndexSegmentLast):
			raise RuntimeError(functionName + " error: direct segment index out of range")
		if(connectionStrength._nnz() > 0):
			indices = connectionStrength.indices()
			values = connectionStrength.values()
			directConnectionMask = indices[1] == arrayIndexSegmentLast
			if(bool(directConnectionMask.any().item())):
				result = pt.sparse_coo_tensor(indices[:, directConnectionMask], values[directConnectionMask], size=connectionStrength.size(), dtype=arrayType, device=connectionStrength.device).coalesce()
			else:
				result = pt.sparse_coo_tensor(pt.empty((4, 0), dtype=pt.long, device=connectionStrength.device), pt.empty((0,), dtype=arrayType, device=connectionStrength.device), size=connectionStrength.size(), dtype=arrayType, device=connectionStrength.device)
		else:
			result = connectionStrength
		return result

	def loadObservedColumnForAutoConnectionPropagation(databaseNetworkObject, conceptIndex, targetDevice):
		if(conceptIndex < 0 or conceptIndex >= databaseNetworkObject.c):
			raise RuntimeError("loadObservedColumnForAutoConnectionPropagation error: conceptIndex out of range")
		conceptName = databaseNetworkObject.conceptColumnsList[conceptIndex]
		if(storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam):
			import GIAANNcmn_databaseNetwork
			result = GIAANNcmn_databaseNetwork.loadOrCreateObservedColumn(databaseNetworkObject, conceptIndex, conceptName, conceptIndex, targetDevice=targetDevice, createDeviceCopy=True, loadAllSourceFeatures=False)
		else:
			from GIAANNcmn_databaseNetworkObservedColumn import ObservedColumn
			if(GIAANNcmn_databaseNetworkFiles.observedColumnMetadataExists(conceptIndex)):
				result = ObservedColumn.loadFromDisk(databaseNetworkObject, conceptIndex, conceptName, conceptIndex, targetDevice=targetDevice, loadAllSourceFeatures=False, resizeFeatureTensorsToCurrentSize=False, loadFeatureNeurons=False)
			else:
				result = ObservedColumn(databaseNetworkObject, conceptIndex, conceptName, conceptIndex)
		return result

	def unloadObservedColumnConnectionsForAutoConnectionPropagation(observedColumn, sourceFeatureIndices, targetFeatureIndices):
		if(observedColumn is None):
			raise RuntimeError("unloadObservedColumnConnectionsForAutoConnectionPropagation error: observedColumn is None")
		if(sourceFeatureIndices is not None):
			observedColumn.unloadLoadedSourceFeatureConnections(sourceFeatureIndices)
		if(targetFeatureIndices is not None):
			ensureObservedColumnReverseConnectionStorage(observedColumn)
			for targetFeatureIndex in targetFeatureIndices:
				normalisedTargetFeatureIndex = normaliseReverseTargetFeatureIndex(observedColumn.databaseNetworkObject, targetFeatureIndex)
				if(normalisedTargetFeatureIndex in observedColumn.reverseFeatureConnectionsByTargetFeature):
					del observedColumn.reverseFeatureConnectionsByTargetFeature[normalisedTargetFeatureIndex]
				observedColumn.reverseLoadedTargetFeatureIndices.discard(normalisedTargetFeatureIndex)
		return

	def createEmptyFeatureActivationSparseVector(databaseNetworkObject, targetDevice):
		indices = pt.empty((1, 0), dtype=pt.long, device=targetDevice)
		values = pt.empty((0,), dtype=arrayType, device=targetDevice)
		result = pt.sparse_coo_tensor(indices, values, size=(databaseNetworkObject.c * databaseNetworkObject.f,), dtype=arrayType, device=targetDevice)
		return result

	def normaliseFeatureActivationSparseVector(databaseNetworkObject, featureActivations, targetDevice):
		featureActivations = featureActivations.coalesce()
		if(featureActivations._nnz() > 0):
			values = featureActivations.values()
			if((values < auxiliaryNeuronsSimilarWordsMinimumSimilarity).any()):
				raise RuntimeError("normaliseFeatureActivationSparseVector error: activation values must be non-negative")
			maxActivation = values.max()
			if(float(maxActivation.item()) > auxiliaryNeuronsSimilarWordsMinimumSimilarity):
				normalisedValues = values/maxActivation
				result = pt.sparse_coo_tensor(featureActivations.indices(), normalisedValues, size=featureActivations.size(), dtype=arrayType, device=targetDevice).coalesce()
			else:
				result = createEmptyFeatureActivationSparseVector(databaseNetworkObject, targetDevice)
		else:
			result = featureActivations
		return result

	def multiplyFeatureActivationSparseVector(featureActivations, activationMultiplier, targetDevice):
		if(activationMultiplier < auxiliaryNeuronsSimilarWordsMinimumSimilarity or activationMultiplier > auxiliaryNeuronsSimilarWordsMaximumSimilarity):
			raise RuntimeError("multiplyFeatureActivationSparseVector error: activationMultiplier out of range")
		featureActivations = featureActivations.coalesce()
		if(featureActivations._nnz() > 0):
			result = pt.sparse_coo_tensor(featureActivations.indices(), featureActivations.values()*activationMultiplier, size=featureActivations.size(), dtype=arrayType, device=targetDevice).coalesce()
		else:
			result = featureActivations
		return result

	def addFeatureActivationSparseVectors(databaseNetworkObject, featureActivations1, featureActivations2, targetDevice):
		if(tuple(featureActivations1.size()) != tuple(featureActivations2.size())):
			raise RuntimeError("addFeatureActivationSparseVectors error: feature activation vector size mismatch")
		result = (featureActivations1.coalesce() + featureActivations2.coalesce()).coalesce()
		if(result._nnz() > 0):
			if((result.values() < auxiliaryNeuronsSimilarWordsMinimumSimilarity).any()):
				raise RuntimeError("addFeatureActivationSparseVectors error: result activation values must be non-negative")
		return result

	def createEmptyFeatureIndexActivationSparseVector(databaseNetworkObject, targetDevice):
		indices = pt.empty((1, 0), dtype=pt.long, device=targetDevice)
		values = pt.empty((0,), dtype=arrayType, device=targetDevice)
		result = pt.sparse_coo_tensor(indices, values, size=(databaseNetworkObject.f,), dtype=arrayType, device=targetDevice)
		return result

	def normaliseFeatureIndexActivationSparseVector(databaseNetworkObject, featureActivations, targetDevice):
		featureActivations = featureActivations.coalesce()
		if(featureActivations._nnz() > 0):
			values = featureActivations.values()
			maxActivation = values.max()
			if(float(maxActivation.item()) < auxiliaryNeuronsSimilarWordsMinimumSimilarity):
				raise RuntimeError("normaliseFeatureIndexActivationSparseVector error: max activation must be non-negative")
			if(float(maxActivation.item()) > auxiliaryNeuronsSimilarWordsMinimumSimilarity):
				normalisedValues = values/maxActivation
				result = pt.sparse_coo_tensor(featureActivations.indices(), normalisedValues, size=featureActivations.size(), dtype=arrayType, device=targetDevice).coalesce()
			else:
				result = createEmptyFeatureIndexActivationSparseVector(databaseNetworkObject, targetDevice)
		else:
			result = featureActivations
		return result

	def registerAutoSimilarityRecordWeight(databaseNetworkObject, records, parentRowIndex, auxiliaryRowIndex, activationWeight):
		import GIAANNnlp_auxiliaryNeuronsSimilarWords
		parentKey = GIAANNnlp_auxiliaryNeuronsSimilarWords.buildSimilarityParentKey(records["prefix"], records["words"][parentRowIndex])
		GIAANNnlp_auxiliaryNeuronsSimilarWords.registerSimilarityParentFeatureWordWeight(databaseNetworkObject, parentKey, records["auxiliaryFeatureWords"][auxiliaryRowIndex], activationWeight)
		return

	def registerAutoSimilarityFeatureIndexWeight(databaseNetworkObject, records, parentFeatureIndex, auxiliaryRowIndex, recordRowsByFeatureKey, activationWeight):
		import GIAANNnlp_auxiliaryNeuronsSimilarWords
		if(parentFeatureIndex < 0 or parentFeatureIndex >= len(databaseNetworkObject.conceptFeaturesList)):
			raise RuntimeError("registerAutoSimilarityFeatureIndexWeight error: parentFeatureIndex out of range")
		auxiliaryConceptIndex = int(records["conceptIndices"][auxiliaryRowIndex].item())
		parentFeatureKey = getAutoFeatureNeuronKey(databaseNetworkObject, auxiliaryConceptIndex, parentFeatureIndex)
		if(parentFeatureKey in recordRowsByFeatureKey):
			parentKey = GIAANNnlp_auxiliaryNeuronsSimilarWords.buildSimilarityParentKey(records["prefix"], databaseNetworkObject.conceptFeaturesList[parentFeatureIndex])
			GIAANNnlp_auxiliaryNeuronsSimilarWords.registerSimilarityParentFeatureWordWeight(databaseNetworkObject, parentKey, records["auxiliaryFeatureWords"][auxiliaryRowIndex], activationWeight)
		return

	def registerAutoSubwordSecondaryFeatureWeights(databaseNetworkObject, records, targetDevice):
		featureRecords = buildAutoSecondaryFeatureIndexRecords(databaseNetworkObject, targetDevice)
		recordRowsByFeatureIndex = buildAutoRecordRowsByFeatureIndex(records)
		recordRowsByFeatureKey = buildAutoRecordRowsByFeatureKey(databaseNetworkObject, records)
		if(featureRecords["rowCount"] > 0):
			similarityMatrix = calculateSubwordSimilaritySparseMatrix(featureRecords["words"], targetDevice)
			activePairs = similarityMatrix.indices().transpose(0, 1)
			activeValues = similarityMatrix.values()
			for pairIndex in range(activePairs.shape[0]):
				parentRowIndex = int(activePairs[pairIndex, 0].item())
				parentFeatureIndex = int(featureRecords["featureIndices"][parentRowIndex].item())
				if(pairIndex == 0 or int(activePairs[pairIndex - 1, 0].item()) != parentRowIndex): print("auxiliaryNeuronsSimilarSubwordAuto: columnIndex=all, featureIndex=", parentFeatureIndex)
				targetRowIndex = int(activePairs[pairIndex, 1].item())
				targetFeatureIndex = int(featureRecords["featureIndices"][targetRowIndex].item())
				if(targetFeatureIndex in recordRowsByFeatureIndex):
					activationWeight = float(activeValues[pairIndex].item())
					for auxiliaryRowIndex in recordRowsByFeatureIndex[targetFeatureIndex]:
						registerAutoSimilarityFeatureIndexWeight(databaseNetworkObject, records, parentFeatureIndex, auxiliaryRowIndex, recordRowsByFeatureKey, activationWeight)
		return

	def buildAutoSecondaryFeatureIndexRecords(databaseNetworkObject, targetDevice):
		featureIndexList = []
		wordList = []
		if(databaseNetworkObject.f != len(databaseNetworkObject.conceptFeaturesList)):
			raise RuntimeError("buildAutoSecondaryFeatureIndexRecords error: feature count mismatch")
		for featureIndex in range(featureIndexPrimeConceptNeuron+1, databaseNetworkObject.f):
			featureWord = databaseNetworkObject.conceptFeaturesList[featureIndex]
			if(featureWord not in databaseNetworkObject.conceptFeaturesDict or databaseNetworkObject.conceptFeaturesDict[featureWord] != featureIndex):
				raise RuntimeError("buildAutoSecondaryFeatureIndexRecords error: feature index map mismatch")
			featureIndexList.append(featureIndex)
			wordList.append(featureWord)
		result = {"rowCount": len(featureIndexList), "featureIndices": pt.tensor(featureIndexList, dtype=pt.long, device=targetDevice), "words": wordList}
		return result

	def calculateSubwordSimilaritySparseMatrix(words, targetDevice):
		wordCount = len(words)
		if(wordCount > 0):
			prefixThreshold = int(auxiliaryNeuronsSimilarSubwordPrefixThreshold)
			similarityThreshold = float(auxiliaryNeuronsSimilarSubwordAutoThreshold)
			validateSubwordSimilaritySparseMatrixParameters(prefixThreshold, similarityThreshold)
			wordLengths = [len(word) for word in words]
			requiredPrefixLengths = calculateSubwordRequiredPrefixLengths(wordLengths, prefixThreshold, similarityThreshold)
			prefixRowsByRequiredLength = buildSubwordPrefixRowsByRequiredLength(words, wordLengths, requiredPrefixLengths)
			prefixRowsByLengthCache = {}
			requiredPrefixLengthValues = sorted(prefixRowsByRequiredLength.keys())
			rowIndices = []
			columnIndices = []
			similarityValues = []
			appendSubwordIdentitySimilarityPairs(wordCount, rowIndices, columnIndices, similarityValues)
			for sourceRowIndex in range(wordCount):
				appendSubwordSimilarityPairsForSource(words, wordLengths, requiredPrefixLengths, prefixRowsByRequiredLength, prefixRowsByLengthCache, requiredPrefixLengthValues, sourceRowIndex, prefixThreshold, similarityThreshold, rowIndices, columnIndices, similarityValues)
			indices, values = buildSubwordSimilaritySparseTensors(rowIndices, columnIndices, similarityValues, targetDevice)
		else:
			indices = pt.empty((2, 0), dtype=pt.long, device=targetDevice)
			values = pt.empty((0,), dtype=arrayType, device=targetDevice)
		result = pt.sparse_coo_tensor(indices, values, size=(wordCount, wordCount), dtype=arrayType, device=targetDevice).coalesce()
		return result

	def validateSubwordSimilaritySparseMatrixParameters(prefixThreshold, similarityThreshold):
		if(prefixThreshold <= 0):
			raise RuntimeError("validateSubwordSimilaritySparseMatrixParameters error: auxiliaryNeuronsSimilarSubwordPrefixThreshold must be positive")
		if(similarityThreshold < auxiliaryNeuronsSimilarWordsMinimumSimilarity or similarityThreshold > auxiliaryNeuronsSimilarWordsMaximumSimilarity):
			raise RuntimeError("validateSubwordSimilaritySparseMatrixParameters error: auxiliaryNeuronsSimilarSubwordAutoThreshold out of range")
		return

	def calculateSubwordRequiredPrefixLengths(wordLengths, prefixThreshold, similarityThreshold):
		result = []
		for wordLength in wordLengths:
			requiredPrefixLength = calculateSubwordRequiredPrefixLength(wordLength, prefixThreshold, similarityThreshold)
			result.append(requiredPrefixLength)
		return result

	def calculateSubwordRequiredPrefixLength(wordLength, prefixThreshold, similarityThreshold):
		if(wordLength < 0):
			raise RuntimeError("calculateSubwordRequiredPrefixLength error: wordLength out of range")
		requiredPrefixLength = max(prefixThreshold, math.ceil(similarityThreshold*wordLength))
		if(wordLength > 0):
			while(requiredPrefixLength > prefixThreshold and (requiredPrefixLength - 1)/wordLength >= similarityThreshold):
				requiredPrefixLength -= 1
			while(requiredPrefixLength < wordLength and requiredPrefixLength/wordLength < similarityThreshold):
				requiredPrefixLength += 1
		result = requiredPrefixLength
		return result

	def buildSubwordPrefixRowsByRequiredLength(words, wordLengths, requiredPrefixLengths):
		result = {}
		for wordIndex, word in enumerate(words):
			requiredPrefixLength = requiredPrefixLengths[wordIndex]
			if(wordLengths[wordIndex] >= requiredPrefixLength):
				if(requiredPrefixLength not in result):
					result[requiredPrefixLength] = {}
				prefix = word[:requiredPrefixLength]
				if(prefix not in result[requiredPrefixLength]):
					result[requiredPrefixLength][prefix] = []
				result[requiredPrefixLength][prefix].append(wordIndex)
		return result

	def appendSubwordIdentitySimilarityPairs(wordCount, rowIndices, columnIndices, similarityValues):
		for wordIndex in range(wordCount):
			rowIndices.append(wordIndex)
			columnIndices.append(wordIndex)
			similarityValues.append(auxiliaryNeuronsSimilarWordsIdentitySimilarity)
		return

	def appendSubwordSimilarityPairsForSource(words, wordLengths, requiredPrefixLengths, prefixRowsByRequiredLength, prefixRowsByLengthCache, requiredPrefixLengthValues, sourceRowIndex, prefixThreshold, similarityThreshold, rowIndices, columnIndices, similarityValues):
		sourceWord = words[sourceRowIndex]
		sourceLength = wordLengths[sourceRowIndex]
		sourceRequiredPrefixLength = requiredPrefixLengths[sourceRowIndex]
		if(sourceLength >= prefixThreshold):
			appendSubwordSimilarityPairsForSourceRequiredLength(words, wordLengths, requiredPrefixLengths, prefixRowsByLengthCache, sourceRowIndex, sourceRequiredPrefixLength, prefixThreshold, similarityThreshold, rowIndices, columnIndices, similarityValues)
			for targetRequiredPrefixLength in requiredPrefixLengthValues:
				if(targetRequiredPrefixLength > sourceRequiredPrefixLength and targetRequiredPrefixLength <= sourceLength):
					sourcePrefix = sourceWord[:targetRequiredPrefixLength]
					if(sourcePrefix in prefixRowsByRequiredLength[targetRequiredPrefixLength]):
						candidateRows = prefixRowsByRequiredLength[targetRequiredPrefixLength][sourcePrefix]
						appendSubwordSimilarityPairsForCandidateRows(words, wordLengths, sourceRowIndex, candidateRows, prefixThreshold, similarityThreshold, rowIndices, columnIndices, similarityValues)
				elif(targetRequiredPrefixLength > sourceLength):
					break
		return

	def appendSubwordSimilarityPairsForSourceRequiredLength(words, wordLengths, requiredPrefixLengths, prefixRowsByLengthCache, sourceRowIndex, sourceRequiredPrefixLength, prefixThreshold, similarityThreshold, rowIndices, columnIndices, similarityValues):
		sourceWord = words[sourceRowIndex]
		sourcePrefix = sourceWord[:sourceRequiredPrefixLength]
		prefixRowsByLength = getSubwordPrefixRowsByLength(words, wordLengths, prefixRowsByLengthCache, sourceRequiredPrefixLength)
		if(sourcePrefix in prefixRowsByLength):
			appendSubwordSimilarityPairsForCandidateRowsWithRequiredPrefixLimit(words, wordLengths, requiredPrefixLengths, sourceRowIndex, prefixRowsByLength[sourcePrefix], sourceRequiredPrefixLength, prefixThreshold, similarityThreshold, rowIndices, columnIndices, similarityValues)
		return

	def appendSubwordSimilarityPairsForCandidateRowsWithRequiredPrefixLimit(words, wordLengths, requiredPrefixLengths, sourceRowIndex, candidateRows, requiredPrefixLengthLimit, prefixThreshold, similarityThreshold, rowIndices, columnIndices, similarityValues):
		for targetRowIndex in candidateRows:
			if(requiredPrefixLengths[targetRowIndex] <= requiredPrefixLengthLimit):
				appendSubwordSimilarityPairForCandidateRow(words, wordLengths, sourceRowIndex, targetRowIndex, prefixThreshold, similarityThreshold, rowIndices, columnIndices, similarityValues)
		return

	def getSubwordPrefixRowsByLength(words, wordLengths, prefixRowsByLengthCache, prefixLength):
		if(prefixLength not in prefixRowsByLengthCache):
			rowsByPrefix = {}
			for wordIndex, word in enumerate(words):
				if(wordLengths[wordIndex] >= prefixLength):
					prefix = word[:prefixLength]
					if(prefix not in rowsByPrefix):
						rowsByPrefix[prefix] = []
					rowsByPrefix[prefix].append(wordIndex)
			prefixRowsByLengthCache[prefixLength] = rowsByPrefix
		result = prefixRowsByLengthCache[prefixLength]
		return result

	def appendSubwordSimilarityPairsForCandidateRows(words, wordLengths, sourceRowIndex, candidateRows, prefixThreshold, similarityThreshold, rowIndices, columnIndices, similarityValues):
		for targetRowIndex in candidateRows:
			appendSubwordSimilarityPairForCandidateRow(words, wordLengths, sourceRowIndex, targetRowIndex, prefixThreshold, similarityThreshold, rowIndices, columnIndices, similarityValues)
		return

	def appendSubwordSimilarityPairForCandidateRow(words, wordLengths, sourceRowIndex, targetRowIndex, prefixThreshold, similarityThreshold, rowIndices, columnIndices, similarityValues):
		if(targetRowIndex != sourceRowIndex):
			sourceWord = words[sourceRowIndex]
			targetWord = words[targetRowIndex]
			sourceLength = wordLengths[sourceRowIndex]
			targetLength = wordLengths[targetRowIndex]
			sharedPrefixLength = calculateSubwordSharedPrefixLength(sourceWord, targetWord)
			if(sharedPrefixLength >= prefixThreshold):
				similarityDenominator = max(sourceLength, targetLength)
				similarityValue = sharedPrefixLength/similarityDenominator
				if(similarityValue >= similarityThreshold):
					rowIndices.append(sourceRowIndex)
					columnIndices.append(targetRowIndex)
					similarityValues.append(similarityValue)
		return

	def calculateSubwordSharedPrefixLength(sourceWord, targetWord):
		sharedPrefixLength = 0
		maxSharedPrefixLength = min(len(sourceWord), len(targetWord))
		for characterIndex in range(maxSharedPrefixLength):
			if(sourceWord[characterIndex] == targetWord[characterIndex]):
				sharedPrefixLength += 1
			else:
				break
		result = sharedPrefixLength
		return result

	def buildSubwordSimilaritySparseTensors(rowIndices, columnIndices, similarityValues, targetDevice):
		if(len(similarityValues) > 0):
			indices = pt.tensor([rowIndices, columnIndices], dtype=pt.long, device=targetDevice)
			values = pt.tensor(similarityValues, dtype=arrayType, device=targetDevice)
		else:
			indices = pt.empty((2, 0), dtype=pt.long, device=targetDevice)
			values = pt.empty((0,), dtype=arrayType, device=targetDevice)
		result = indices, values
		return result

	def registerAutoSimilaritySparseMatrixWeights(databaseNetworkObject, records, similarityMatrix):
		import GIAANNnlp_auxiliaryNeuronsSimilarWords
		if(not similarityMatrix.is_sparse):
			raise RuntimeError("registerAutoSimilaritySparseMatrixWeights error: similarity matrix must be sparse")
		if(similarityMatrix.dim() != 2 or similarityMatrix.shape[0] != records["rowCount"] or similarityMatrix.shape[1] != records["rowCount"]):
			raise RuntimeError("registerAutoSimilaritySparseMatrixWeights error: similarity matrix dimensions mismatch")
		similarityMatrix = similarityMatrix.coalesce()
		activePairs = similarityMatrix.indices().transpose(0, 1)
		activeValues = similarityMatrix.values()
		for pairIndex in range(activePairs.shape[0]):
			parentRowIndex = int(activePairs[pairIndex, 0].item())
			if(pairIndex == 0 or int(activePairs[pairIndex - 1, 0].item()) != parentRowIndex): print("auxiliaryNeuronsSimilarSubwordAuto: columnIndex=", int(records["conceptIndices"][parentRowIndex].item()), ", featureIndex=", int(records["featureIndices"][parentRowIndex].item()))
			auxiliaryRowIndex = int(activePairs[pairIndex, 1].item())
			activationWeight = float(activeValues[pairIndex].item())
			parentKey = GIAANNnlp_auxiliaryNeuronsSimilarWords.buildSimilarityParentKey(records["prefix"], records["words"][parentRowIndex])
			GIAANNnlp_auxiliaryNeuronsSimilarWords.registerSimilarityParentFeatureWordWeight(databaseNetworkObject, parentKey, records["auxiliaryFeatureWords"][auxiliaryRowIndex], activationWeight)
		return
