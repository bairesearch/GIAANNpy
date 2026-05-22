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

import os
import torch as pt

from GIAANNcmn_globalDefs import *
import GIAANNcmn_databaseNetworkFiles


if(auxiliaryNeurons and auxiliaryNeuronsAuto):

	auxiliaryNeuronsAutoReverseSavedTargetTensorPaths = set()

	def getTokenAutoAuxiliaryFeatureIndices(databaseNetworkObject, token, isConcept, conceptIndex, allowNewFeatures=False, registerParent=False):
		import GIAANNnlp_auxiliaryNeuronsSimilarity
		result = []
		if(isConcept):
			if(GIAANNnlp_auxiliaryNeuronsSimilarity.tokenHasPrimeConceptSimilarityWord(token)):
				if(auxiliaryNeuronsSimilarWordsAuto and auxiliaryNeuronsSimilarWordsPrimeConceptFeatures):
					result.append(registerTokenAutoAuxiliaryFeature(databaseNetworkObject, token, conceptIndex, auxiliaryNeuronsSimilarWordsFeatureNamePrefixPrimeConcept, True, allowNewFeatures, registerParent))
				if(auxiliaryNeuronsTokenisationSubwordAuto and auxiliaryNeuronsTokenisationSubwordPrimeConceptFeatures):
					result.append(registerTokenAutoAuxiliaryFeature(databaseNetworkObject, token, conceptIndex, auxiliaryNeuronsSimilarWordsFeatureNamePrefixSubwordPrimeConcept, True, allowNewFeatures, registerParent))
		else:
			if(GIAANNnlp_auxiliaryNeuronsSimilarity.tokenHasSecondarySimilarityWord(token)):
				if(auxiliaryNeuronsSimilarWordsAuto and auxiliaryNeuronsSimilarWordsSecondaryConceptFeatures):
					result.append(registerTokenAutoAuxiliaryFeature(databaseNetworkObject, token, conceptIndex, auxiliaryNeuronsSimilarWordsFeatureNamePrefixSecondary, False, allowNewFeatures, registerParent))
				if(auxiliaryNeuronsTokenisationSubwordAuto and auxiliaryNeuronsTokenisationSubwordSecondaryConceptFeatures):
					result.append(registerTokenAutoAuxiliaryFeature(databaseNetworkObject, token, conceptIndex, auxiliaryNeuronsSimilarWordsFeatureNamePrefixSubwordSecondary, False, allowNewFeatures, registerParent))
		return result

	def registerTokenAutoAuxiliaryFeature(databaseNetworkObject, token, conceptIndex, auxiliaryFeaturePrefix, primeConceptFeature, allowNewFeatures, registerParent):
		import GIAANNnlp_auxiliaryNeuronsSimilarity
		if(primeConceptFeature):
			auxiliaryBaseWord = GIAANNnlp_auxiliaryNeuronsSimilarity.getTokenPrimeConceptSimilarityWord(token)
		else:
			auxiliaryBaseWord = GIAANNnlp_auxiliaryNeuronsSimilarity.getTokenSecondarySimilarityWord(token)
		auxiliaryFeatureWord = GIAANNnlp_auxiliaryNeuronsSimilarity.buildConceptColumnAuxiliaryFeatureName(databaseNetworkObject, auxiliaryFeaturePrefix, conceptIndex, auxiliaryBaseWord)
		result = GIAANNnlp_auxiliaryNeuronsSimilarity.registerAuxiliaryFeatureWord(databaseNetworkObject, auxiliaryFeatureWord, allowNewFeatures)
		if(registerParent):
			parentKey = GIAANNnlp_auxiliaryNeuronsSimilarity.buildSimilarityParentKey(auxiliaryFeaturePrefix, auxiliaryBaseWord)
			GIAANNnlp_auxiliaryNeuronsSimilarity.registerSimilarityParentFeatureWordWeight(databaseNetworkObject, parentKey, auxiliaryFeatureWord, auxiliaryNeuronsSimilarWordsIdentitySimilarity)
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
		result = pt.sparse_coo_tensor(indices, values, size=(databaseNetworkObject.arrayNumberOfProperties, numberOfDendriticBranches, arrayNumberOfSegments, databaseNetworkObject.c, databaseNetworkObject.f), dtype=arrayType, device=targetDevice)
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
		result = (observedColumn.databaseNetworkObject.arrayNumberOfProperties, numberOfDendriticBranches, arrayNumberOfSegments, observedColumn.databaseNetworkObject.c, observedColumn.databaseNetworkObject.f)
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
		databaseNetworkObject = sequenceObservedColumns.databaseNetworkObject
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
		targetSize = (databaseNetworkObject.arrayNumberOfProperties, numberOfDendriticBranches, arrayNumberOfSegments, databaseNetworkObject.c, databaseNetworkObject.f)
		for targetCombinedKey, start, count in zip(uniqueTargetCombinedKeys.tolist(), starts.tolist(), counts.tolist()):
			end = start + count
			targetConceptIndexValue = int(targetCombinedKey // databaseNetworkObject.f)
			targetFeatureIndexValue = int(targetCombinedKey % databaseNetworkObject.f)
			if(targetConceptIndexValue not in observedColumnsByConceptIndex):
				raise RuntimeError("applyReverseConnectionUpdates error: missing observed column")
			observedColumn = observedColumnsByConceptIndex[targetConceptIndexValue]
			propertyRow = pt.full((count,), databaseNetworkObject.arrayIndexPropertiesStrengthIndex, dtype=pt.long, device=connectionDevice)
			updateIndices = pt.stack((propertyRow, sortedBranch[start:end], sortedSegment[start:end], sortedSourceConceptIndex[start:end], sortedSourceFeatureIndex[start:end]), dim=0)
			updateSparse = pt.sparse_coo_tensor(updateIndices, sortedValues[start:end], size=targetSize, dtype=arrayType, device=connectionDevice)
			targetSparse = getObservedColumnReverseFeatureConnectionsForTargetFeature(observedColumn, targetFeatureIndexValue, targetDevice=connectionDevice, createMissing=False, ensureCurrentSizeOnLoad=True)
			targetSparse = addSparseUpdateNonNegative(targetSparse, updateSparse)
			setObservedColumnReverseFeatureConnectionsForTargetFeature(observedColumn, targetFeatureIndexValue, targetSparse)
			observedColumn.reverseTrainPreparedTargetFeatureIndices.add(targetFeatureIndexValue)
		return

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
		import GIAANNnlp_auxiliaryNeuronsSimilarity
		ensureAutoAuxiliaryFeatureRecords(databaseNetworkObject, subwordSimilarity)
		removeAutoParentKeysForMode(databaseNetworkObject, subwordSimilarity)
		if(getAutoPrimeFeatureEnabled(subwordSimilarity)):
			updateAutoAuxiliaryFeatureConnectionWeightsForFeatureType(databaseNetworkObject, subwordSimilarity, True)
		if(getAutoSecondaryFeatureEnabled(subwordSimilarity)):
			updateAutoAuxiliaryFeatureConnectionWeightsForFeatureType(databaseNetworkObject, subwordSimilarity, False)
		databaseNetworkObject.auxiliaryNeuronsSimilarWordsFeatureIndexWeightsByParentWord = GIAANNnlp_auxiliaryNeuronsSimilarity.buildAuxiliaryFeatureIndexWeightsByParentWord(databaseNetworkObject)
		GIAANNnlp_auxiliaryNeuronsSimilarity.invalidateDatabaseAuxiliaryInputConnectionCaches(databaseNetworkObject)
		return

	def ensureAutoAuxiliaryFeatureRecords(databaseNetworkObject, subwordSimilarity):
		import GIAANNnlp_auxiliaryNeuronsSimilarity
		if(getAutoPrimeFeatureEnabled(subwordSimilarity)):
			for conceptIndex, conceptName in enumerate(databaseNetworkObject.conceptColumnsList):
				registerAutoAuxiliaryFeatureRecord(databaseNetworkObject, getAutoPrimePrefix(subwordSimilarity), conceptIndex, GIAANNnlp_auxiliaryNeuronsSimilarity.normaliseSimilarityWord(conceptName))
		if(getAutoSecondaryFeatureEnabled(subwordSimilarity)):
			for featureKey in sorted(buildAutoObservedSecondaryFeatureKeys(databaseNetworkObject)):
				conceptIndex, featureIndex = getAutoFeatureNeuronIndicesFromKey(databaseNetworkObject, featureKey)
				if(featureIndex >= len(databaseNetworkObject.conceptFeaturesList)):
					raise RuntimeError("ensureAutoAuxiliaryFeatureRecords error: featureIndex out of range")
				registerAutoAuxiliaryFeatureRecord(databaseNetworkObject, getAutoSecondaryPrefix(subwordSimilarity), conceptIndex, GIAANNnlp_auxiliaryNeuronsSimilarity.normaliseSimilarityWord(databaseNetworkObject.conceptFeaturesList[featureIndex]))
		return

	def registerAutoAuxiliaryFeatureRecord(databaseNetworkObject, auxiliaryFeaturePrefix, conceptIndex, auxiliaryBaseWord):
		import GIAANNnlp_auxiliaryNeuronsSimilarity
		auxiliaryFeatureWord = GIAANNnlp_auxiliaryNeuronsSimilarity.buildConceptColumnAuxiliaryFeatureName(databaseNetworkObject, auxiliaryFeaturePrefix, conceptIndex, auxiliaryBaseWord)
		GIAANNnlp_auxiliaryNeuronsSimilarity.registerAuxiliaryFeatureWord(databaseNetworkObject, auxiliaryFeatureWord, True)
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
			result = auxiliaryNeuronsTokenisationSubwordAuto and auxiliaryNeuronsTokenisationSubwordPrimeConceptFeatures
		else:
			result = auxiliaryNeuronsSimilarWordsAuto and auxiliaryNeuronsSimilarWordsPrimeConceptFeatures
		return result

	def getAutoSecondaryFeatureEnabled(subwordSimilarity):
		if(subwordSimilarity):
			result = auxiliaryNeuronsTokenisationSubwordAuto and auxiliaryNeuronsTokenisationSubwordSecondaryConceptFeatures
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
		return

	def buildAutoAuxiliaryFeatureRecords(databaseNetworkObject, subwordSimilarity, primeConceptFeatures):
		import GIAANNnlp_auxiliaryNeuronsSimilarity
		prefix = getAutoPrimePrefix(subwordSimilarity) if primeConceptFeatures else getAutoSecondaryPrefix(subwordSimilarity)
		conceptIndexList = []
		featureIndexList = []
		auxiliaryFeatureIndexList = []
		wordList = []
		auxiliaryFeatureWordList = []
		for auxiliaryFeatureWord, auxiliaryFeatureIndex in databaseNetworkObject.auxiliaryNeuronsSimilarWordsFeaturesDict.items():
			if(GIAANNnlp_auxiliaryNeuronsSimilarity.auxiliaryFeatureWordHasPrefix(auxiliaryFeatureWord, prefix)):
				auxiliaryConceptIndex, auxiliaryBaseWord = GIAANNnlp_auxiliaryNeuronsSimilarity.parseConceptColumnAuxiliaryFeatureName(databaseNetworkObject, auxiliaryFeatureWord)
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
		for sourceRowIndex in range(records["rowCount"]):
			registerAutoSimilarityRecordWeight(databaseNetworkObject, records, sourceRowIndex, sourceRowIndex, auxiliaryNeuronsSimilarWordsIdentitySimilarity)
		for sourceFeatureRowIndex in range(featureRecords["rowCount"]):
			sourceFeatureIndex = int(featureRecords["featureIndices"][sourceFeatureRowIndex].item())
			print("auxiliaryNeuronsSimilarWordsAuto: columnIndex=all, featureIndex=", sourceFeatureIndex)
			if(sourceFeatureIndex in recordRowsByFeatureIndex):
				similarFeatureActivations = calculateTwoHopSimilarSecondaryFeatureIndexActivations(databaseNetworkObject, records, sourceFeatureIndex, recordRowsByFeatureIndex[sourceFeatureIndex], targetDevice)
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
						activationWeight = float(activeFeatureValues[activeIndex].item())
						for targetRowIndex in recordRowsByFeatureIndex[targetFeatureIndex]:
							registerAutoSimilarityFeatureIndexWeight(databaseNetworkObject, records, sourceFeatureIndex, targetRowIndex, activationWeight)
		return


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
		import GIAANNnlp_auxiliaryNeuronsSimilarity
		parentKey = GIAANNnlp_auxiliaryNeuronsSimilarity.buildSimilarityParentKey(records["prefix"], records["words"][parentRowIndex])
		GIAANNnlp_auxiliaryNeuronsSimilarity.registerSimilarityParentFeatureWordWeight(databaseNetworkObject, parentKey, records["auxiliaryFeatureWords"][auxiliaryRowIndex], activationWeight)
		return

	def registerAutoSimilarityFeatureIndexWeight(databaseNetworkObject, records, parentFeatureIndex, auxiliaryRowIndex, activationWeight):
		import GIAANNnlp_auxiliaryNeuronsSimilarity
		if(parentFeatureIndex < 0 or parentFeatureIndex >= len(databaseNetworkObject.conceptFeaturesList)):
			raise RuntimeError("registerAutoSimilarityFeatureIndexWeight error: parentFeatureIndex out of range")
		parentKey = GIAANNnlp_auxiliaryNeuronsSimilarity.buildSimilarityParentKey(records["prefix"], databaseNetworkObject.conceptFeaturesList[parentFeatureIndex])
		GIAANNnlp_auxiliaryNeuronsSimilarity.registerSimilarityParentFeatureWordWeight(databaseNetworkObject, parentKey, records["auxiliaryFeatureWords"][auxiliaryRowIndex], activationWeight)
		return

	def registerAutoSubwordSecondaryFeatureWeights(databaseNetworkObject, records, targetDevice):
		featureRecords = buildAutoSecondaryFeatureIndexRecords(databaseNetworkObject, targetDevice)
		recordRowsByFeatureIndex = buildAutoRecordRowsByFeatureIndex(records)
		if(featureRecords["rowCount"] > 0):
			similarityMatrix = calculateSubwordSimilaritySparseMatrix(featureRecords["words"], targetDevice)
			activePairs = similarityMatrix.indices().transpose(0, 1)
			activeValues = similarityMatrix.values()
			for pairIndex in range(activePairs.shape[0]):
				parentRowIndex = int(activePairs[pairIndex, 0].item())
				parentFeatureIndex = int(featureRecords["featureIndices"][parentRowIndex].item())
				if(pairIndex == 0 or int(activePairs[pairIndex - 1, 0].item()) != parentRowIndex): print("auxiliaryNeuronsTokenisationSubwordAuto: columnIndex=all, featureIndex=", parentFeatureIndex)
				targetRowIndex = int(activePairs[pairIndex, 1].item())
				targetFeatureIndex = int(featureRecords["featureIndices"][targetRowIndex].item())
				if(targetFeatureIndex in recordRowsByFeatureIndex):
					activationWeight = float(activeValues[pairIndex].item())
					for auxiliaryRowIndex in recordRowsByFeatureIndex[targetFeatureIndex]:
						registerAutoSimilarityFeatureIndexWeight(databaseNetworkObject, records, parentFeatureIndex, auxiliaryRowIndex, activationWeight)
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
		wordLengths = pt.tensor([len(word) for word in words], dtype=pt.long, device=targetDevice)
		maxWordLength = int(wordLengths.max().item()) if wordCount > 0 else 0
		wordCodes = pt.zeros((wordCount, maxWordLength), dtype=pt.long, device=targetDevice)
		for wordIndex, word in enumerate(words):
			for characterIndex, character in enumerate(word):
				wordCodes[wordIndex, characterIndex] = ord(character)
		if(wordCount > 0):
			batchSize = int(auxiliaryNeuronsTokenisationSubwordAutoSimilarityBatchSize)
			if(batchSize <= 0):
				raise RuntimeError("calculateSubwordSimilaritySparseMatrix error: auxiliaryNeuronsTokenisationSubwordAutoSimilarityBatchSize must be positive")
			indicesList = []
			valuesList = []
			for rowStart in range(0, wordCount, batchSize):
				rowEnd = min(rowStart + batchSize, wordCount)
				batchIndices, batchValues = calculateSubwordSimilaritySparseMatrixRows(wordCodes, wordLengths, rowStart, rowEnd, targetDevice)
				if(batchValues.numel() > 0):
					indicesList.append(batchIndices)
					valuesList.append(batchValues)
			if(len(valuesList) > 0):
				indices = pt.cat(indicesList, dim=1)
				values = pt.cat(valuesList, dim=0)
			else:
				indices = pt.empty((2, 0), dtype=pt.long, device=targetDevice)
				values = pt.empty((0,), dtype=arrayType, device=targetDevice)
		else:
			indices = pt.empty((2, 0), dtype=pt.long, device=targetDevice)
			values = pt.empty((0,), dtype=arrayType, device=targetDevice)
		result = pt.sparse_coo_tensor(indices, values, size=(wordCount, wordCount), dtype=arrayType, device=targetDevice).coalesce()
		return result

	def calculateSubwordSimilaritySparseMatrixRows(wordCodes, wordLengths, rowStart, rowEnd, targetDevice):
		wordCount = wordCodes.shape[0]
		maxWordLength = wordCodes.shape[1]
		rowWordCount = rowEnd - rowStart
		if(rowWordCount <= 0):
			raise RuntimeError("calculateSubwordSimilaritySparseMatrixRows error: row range is empty")
		if(maxWordLength > 0):
			positionIndices = pt.arange(maxWordLength, dtype=pt.long, device=targetDevice)
			rowWordCodes = wordCodes[rowStart:rowEnd]
			rowWordLengths = wordLengths[rowStart:rowEnd]
			validMask = (positionIndices.view(1, 1, maxWordLength) < rowWordLengths.view(-1, 1, 1)) & (positionIndices.view(1, 1, maxWordLength) < wordLengths.view(1, -1, 1))
			characterEqualMask = (rowWordCodes.view(rowWordCount, 1, maxWordLength) == wordCodes.view(1, wordCount, maxWordLength)) & validMask
			prefixMask = characterEqualMask.to(pt.long).cumprod(dim=2).to(pt.bool)
			sharedPrefixLengths = prefixMask.sum(dim=2)
			activeMask = sharedPrefixLengths >= int(auxiliaryNeuronsTokenisationSubwordPrefixThreshold)
			rowLocalIndices = pt.arange(rowWordCount, dtype=pt.long, device=targetDevice)
			diagonalColumnIndices = rowStart + rowLocalIndices
			activeMask[rowLocalIndices, diagonalColumnIndices] = True
			activeRowIndices, activeColumnIndices = pt.nonzero(activeMask, as_tuple=True)
			if(activeRowIndices.numel() > 0):
				activeSharedPrefixLengths = sharedPrefixLengths[activeRowIndices, activeColumnIndices].to(arrayType)
				activeDenominator = pt.maximum(wordLengths[rowStart + activeRowIndices], wordLengths[activeColumnIndices]).clamp(min=1).to(arrayType)
				values = activeSharedPrefixLengths/activeDenominator
				values = pt.maximum(values, pt.full_like(values, auxiliaryNeuronsSimilarWordsThreshold))
				diagonalMask = activeColumnIndices == rowStart + activeRowIndices
				values = pt.where(diagonalMask, pt.full_like(values, auxiliaryNeuronsSimilarWordsIdentitySimilarity), values)
				indices = pt.stack((rowStart + activeRowIndices, activeColumnIndices), dim=0)
			else:
				indices = pt.empty((2, 0), dtype=pt.long, device=targetDevice)
				values = pt.empty((0,), dtype=arrayType, device=targetDevice)
		else:
			rowGlobalIndices = pt.arange(rowStart, rowEnd, dtype=pt.long, device=targetDevice)
			indices = pt.stack((rowGlobalIndices, rowGlobalIndices), dim=0)
			values = pt.full((rowWordCount,), auxiliaryNeuronsSimilarWordsIdentitySimilarity, dtype=arrayType, device=targetDevice)
		return indices, values

	def registerAutoSimilaritySparseMatrixWeights(databaseNetworkObject, records, similarityMatrix):
		import GIAANNnlp_auxiliaryNeuronsSimilarity
		if(not similarityMatrix.is_sparse):
			raise RuntimeError("registerAutoSimilaritySparseMatrixWeights error: similarity matrix must be sparse")
		if(similarityMatrix.dim() != 2 or similarityMatrix.shape[0] != records["rowCount"] or similarityMatrix.shape[1] != records["rowCount"]):
			raise RuntimeError("registerAutoSimilaritySparseMatrixWeights error: similarity matrix dimensions mismatch")
		similarityMatrix = similarityMatrix.coalesce()
		activePairs = similarityMatrix.indices().transpose(0, 1)
		activeValues = similarityMatrix.values()
		for pairIndex in range(activePairs.shape[0]):
			parentRowIndex = int(activePairs[pairIndex, 0].item())
			if(pairIndex == 0 or int(activePairs[pairIndex - 1, 0].item()) != parentRowIndex): print("auxiliaryNeuronsTokenisationSubwordAuto: columnIndex=", int(records["conceptIndices"][parentRowIndex].item()), ", featureIndex=", int(records["featureIndices"][parentRowIndex].item()))
			auxiliaryRowIndex = int(activePairs[pairIndex, 1].item())
			activationWeight = float(activeValues[pairIndex].item())
			parentKey = GIAANNnlp_auxiliaryNeuronsSimilarity.buildSimilarityParentKey(records["prefix"], records["words"][parentRowIndex])
			GIAANNnlp_auxiliaryNeuronsSimilarity.registerSimilarityParentFeatureWordWeight(databaseNetworkObject, parentKey, records["auxiliaryFeatureWords"][auxiliaryRowIndex], activationWeight)
		return
