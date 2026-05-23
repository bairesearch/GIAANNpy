"""GIAANNnlp_auxiliaryNeuronsSimilarWordsStatic.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 BAI Research Pty Ltd (bairesearch.com.au)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN NLP auxiliary neurons similar words static

"""

import os
import torch as pt

from GIAANNcmn_globalDefs import *
import GIAANNcmn_databaseNetworkFiles


if(auxiliaryNeurons and auxiliaryNeuronsSimilar):

	auxiliaryNeuronsSimilarWordsSavedSourceTensorPaths = set()
	auxiliaryNeuronsSimilarWordsDatasetCache = None
	auxiliaryNeuronsSimilarWordsDatasetWordNetValidated = False
	auxiliaryNeuronsSimilarWordsParentWordWeightsCache = {}

	def loadOrCreateDatabaseAuxiliaryFeatureMaps(loadExistingDatabase):
		auxiliarySimilarFeaturesDict = {}
		auxiliarySimilarFeatureWordWeightsByParentWord = {}
		if(loadExistingDatabase):
			auxiliarySimilarFeaturesDict = GIAANNcmn_databaseNetworkFiles.loadDictFile(auxiliaryNeuronsSimilarWordsFeaturesDictFile)
			auxiliarySimilarFeatureWordWeightsByParentWord = GIAANNcmn_databaseNetworkFiles.loadDictFile(auxiliaryNeuronsSimilarWordsFeatureWordWeightsByParentWordFile)
		auxiliarySimilarFeaturesList = buildIndexListFromIndexDict(auxiliarySimilarFeaturesDict, auxiliaryNeuronsSimilarWordsFeaturesDictFileName)
		result = auxiliarySimilarFeaturesDict, auxiliarySimilarFeaturesList, auxiliarySimilarFeatureWordWeightsByParentWord
		return result

	def initialiseDatabaseNetworkAuxiliary(databaseNetworkObject, auxiliarySimilarFeaturesDict, auxiliarySimilarFeaturesList, auxiliarySimilarFeatureWordWeightsByParentWord, auxiliarySimilarLoadExistingDatabase):
		if(not isinstance(auxiliarySimilarLoadExistingDatabase, bool)):
			raise RuntimeError("initialiseDatabaseNetworkAuxiliarySimilarity error: auxiliarySimilarLoadExistingDatabase must be bool")
		validateSimilarWordsConfiguration()
		if(auxiliaryNeuronsSimilarWordsStatic and auxiliaryNeuronsSimilarWordsDataset3):
			ensureSimilarWordsDataset3CompactFile()
		databaseNetworkObject.auxiliaryNeuronsSimilarWordsFeaturesDict = auxiliarySimilarFeaturesDict
		databaseNetworkObject.auxiliaryNeuronsSimilarWordsFeaturesList = auxiliarySimilarFeaturesList
		databaseNetworkObject.auxiliaryNeuronsSimilarWordsFeaturesIndexToWordDict = dict(enumerate(auxiliarySimilarFeaturesList))
		databaseNetworkObject.auxiliaryNeuronsSimilarWordsFeatureWordWeightsByParentWord = auxiliarySimilarFeatureWordWeightsByParentWord
		databaseNetworkObject.auxiliaryNeuronsSimilarWordsFeatureIndexWeightsByParentWord = buildAuxiliaryFeatureIndexWeightsByParentWord(databaseNetworkObject)
		databaseNetworkObject.auxiliaryNeuronsSimilarWordsLoadExistingDatabase = auxiliarySimilarLoadExistingDatabase
		databaseNetworkObject.fas = len(auxiliarySimilarFeaturesList)
		databaseNetworkObject.auxiliaryNeuronsSimilarWordsPrimeInputConnections = None
		databaseNetworkObject.auxiliaryNeuronsSimilarWordsSecondaryInputConnectionsByConceptIndex = {}
		databaseNetworkObject.auxiliaryNeuronsSimilarWordsPrimeOutputConnectionsMaterialised = None
		return

	def saveDatabaseAuxiliaryFeatureMaps(databaseNetworkObject):
		GIAANNcmn_databaseNetworkFiles.saveDictFile(auxiliaryNeuronsSimilarWordsFeaturesDictFile, databaseNetworkObject.auxiliaryNeuronsSimilarWordsFeaturesDict)
		GIAANNcmn_databaseNetworkFiles.saveDictFile(auxiliaryNeuronsSimilarWordsFeatureWordWeightsByParentWordFile, databaseNetworkObject.auxiliaryNeuronsSimilarWordsFeatureWordWeightsByParentWord)
		return

	def initialiseObservedColumnAuxiliaryStorage(observedColumn):
		observedColumn.similarAuxiliaryFeatureConnectionsBySourceFeature = {}
		observedColumn.similarAuxiliaryLoadedSourceFeatureIndices = set()
		observedColumn.similarAuxiliaryTrainPreparedSourceFeatureIndices = set()
		observedColumn.similarAuxiliarySecondaryInputConnections = None
		observedColumn.similarAuxiliarySecondaryOutputConnectionsMaterialised = None
		return

	def initialiseObservedColumnProxyAuxiliaryStorage(observedColumnProxy):
		observedColumnProxy.similarAuxiliaryFeatureConnectionsBySourceFeature = {}
		observedColumnProxy.similarAuxiliaryLoadedSourceFeatureIndices = set()
		observedColumnProxy.similarAuxiliaryTrainPreparedSourceFeatureIndices = set()
		observedColumnProxy.similarAuxiliarySecondaryInputConnections = None
		observedColumnProxy.similarAuxiliarySecondaryOutputConnectionsMaterialised = None
		return

	def prepareObservedColumnsForTrainSequenceAuxiliary(sequenceObservedColumns, observedColumnsDict, allowNewFeatures):
		requiredAuxiliaryFeatureIndicesByObservedColumn = getRequiredAuxiliaryFeatureIndicesByObservedColumn(sequenceObservedColumns, allowNewFeatures)
		for observedColumn in observedColumnsDict.values():
			conceptIndex = int(observedColumn.conceptIndex)
			if(conceptIndex not in requiredAuxiliaryFeatureIndicesByObservedColumn):
				raise RuntimeError("prepareObservedColumnsForTrainSequenceAuxiliarySimilarity error: missing required auxiliary source features")
			prepareObservedColumnAuxiliaryFeatureConnectionsTrain(observedColumn, requiredAuxiliaryFeatureIndicesByObservedColumn[conceptIndex], deviceSparse)
		return

	def trainAuxiliaryFeatureConnections(sequenceObservedColumns, featureNeuronsActive, columnsWordOrder, featureNeuronsWordOrder, conceptIndices, startIndices, endIndices):
		if(arrayIndexPropertiesPos or trainConnectionStrengthPOSdependence):
			raise RuntimeError("trainAuxiliaryFeatureConnectionsSimilarity error: auxiliary POS connection properties are not implemented")
		sourceConceptIndices, sourceAuxiliaryFeatureIndices, sourceWordOrder = buildAuxiliarySourceOccurrenceTensors(sequenceObservedColumns, conceptIndices, startIndices, endIndices, featureNeuronsActive.device)
		connectionIndices, connectionValues = buildAuxiliaryConnectionIndicesAndValues(sequenceObservedColumns, featureNeuronsActive, columnsWordOrder, featureNeuronsWordOrder, sourceConceptIndices, sourceAuxiliaryFeatureIndices, sourceWordOrder)
		if(connectionIndices.numel() > 0):
			applyAuxiliaryConnectionPropertyUpdates(sequenceObservedColumns, connectionIndices, connectionValues, sequenceObservedColumns.databaseNetworkObject.arrayIndexPropertiesStrengthIndex)
			if(arrayIndexPropertiesPermanence):
				permanenceValues = pt.full((connectionValues.shape[0],), z1, dtype=arrayType, device=connectionValues.device)
				applyAuxiliaryConnectionPropertyUpdates(sequenceObservedColumns, connectionIndices, permanenceValues, sequenceObservedColumns.databaseNetworkObject.arrayIndexPropertiesPermanenceIndex)
		return

	def processAuxiliaryFeaturePredictionActivations(databaseNetworkObject, observedColumn, globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, sourceColumnIndex, sourceFeatureIndex, globalFeatureNeuronsTime=None, sequenceWordIndex=None, sequenceColumnIndex=None):
		import GIAANNcmn_predictionActivate
		globalFeatureNeuronsActivationResult = globalFeatureNeuronsActivation
		globalFeatureConnectionsActivationResult = globalFeatureConnectionsActivation
		globalFeatureNeuronsTimeResult = globalFeatureNeuronsTime
		connectionDevice = globalFeatureNeuronsActivationResult.device
		sourceActivation = GIAANNcmn_predictionActivate.calculateFeatureNeuronSourceActivationPredict(databaseNetworkObject, globalFeatureNeuronsActivationResult, sourceColumnIndex, sourceFeatureIndex)
		sourceActivationValue = collapseSourceActivationForAuxiliaryInput(sourceActivation)
		if(float(sourceActivationValue.item()) > auxiliaryNeuronsSimilarWordsMinimumSimilarity):
			if(auxiliaryNeuronsSimilarWordsPrimeConceptFeatures and int(sourceFeatureIndex) == featureIndexPrimeConceptNeuron):
				globalFeatureNeuronsActivationResult, globalFeatureConnectionsActivationResult, globalFeatureNeuronsTimeResult = processAuxiliaryPrimeFeaturePredictionActivations(databaseNetworkObject, globalFeatureNeuronsActivationResult, globalFeatureConnectionsActivationResult, int(sourceColumnIndex), sourceActivationValue, globalFeatureNeuronsTimeResult, sequenceWordIndex, sequenceColumnIndex, connectionDevice)
			if(auxiliaryNeuronsSimilarWordsSecondaryConceptFeatures and int(sourceFeatureIndex) != featureIndexPrimeConceptNeuron):
				globalFeatureNeuronsActivationResult, globalFeatureConnectionsActivationResult, globalFeatureNeuronsTimeResult = processAuxiliarySecondaryFeaturePredictionActivations(databaseNetworkObject, observedColumn, globalFeatureNeuronsActivationResult, globalFeatureConnectionsActivationResult, int(sourceFeatureIndex), sourceActivationValue, globalFeatureNeuronsTimeResult, sequenceWordIndex, sequenceColumnIndex, connectionDevice)
		result = globalFeatureNeuronsActivationResult, globalFeatureConnectionsActivationResult, globalFeatureNeuronsTimeResult
		return result

	def processAuxiliaryPrimeFeaturePredictionActivations(databaseNetworkObject, globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, sourceColumnIndex, sourceActivationValue, globalFeatureNeuronsTime, sequenceWordIndex, sequenceColumnIndex, connectionDevice):
		import GIAANNcmn_predictionActivate
		globalFeatureNeuronsActivationResult = globalFeatureNeuronsActivation
		globalFeatureConnectionsActivationResult = globalFeatureConnectionsActivation
		globalFeatureNeuronsTimeResult = globalFeatureNeuronsTime
		auxiliaryActivations = calculatePrimeAuxiliaryConceptActivations(databaseNetworkObject, sourceColumnIndex, sourceActivationValue, connectionDevice)
		if(auxiliaryActivationVectorHasActiveValues(auxiliaryActivations)):
			materialisedConnections = getPrimeOutputConnectionsMaterialised(databaseNetworkObject, connectionDevice, auxiliaryActivations)
			featureNeuronsTargetActivation = calculateAuxiliaryFeatureTargetActivation(databaseNetworkObject, materialisedConnections, auxiliaryActivations)
			if(sparseTensorHasValues(featureNeuronsTargetActivation)):
				globalFeatureNeuronsActivationResult, globalFeatureConnectionsActivationResult, globalFeatureNeuronsTimeResult = GIAANNcmn_predictionActivate.applyFeatureNeuronsTargetActivationPredict(databaseNetworkObject, globalFeatureNeuronsActivationResult, globalFeatureConnectionsActivationResult, featureNeuronsTargetActivation, globalFeatureNeuronsTimeResult, sequenceWordIndex, sequenceColumnIndex, applySegmentActivations=False)
		result = globalFeatureNeuronsActivationResult, globalFeatureConnectionsActivationResult, globalFeatureNeuronsTimeResult
		return result

	def processAuxiliarySecondaryFeaturePredictionActivations(databaseNetworkObject, observedColumn, globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, sourceFeatureIndex, sourceActivationValue, globalFeatureNeuronsTime, sequenceWordIndex, sequenceColumnIndex, connectionDevice):
		import GIAANNcmn_predictionActivate
		globalFeatureNeuronsActivationResult = globalFeatureNeuronsActivation
		globalFeatureConnectionsActivationResult = globalFeatureConnectionsActivation
		globalFeatureNeuronsTimeResult = globalFeatureNeuronsTime
		auxiliaryActivations = calculateSecondaryAuxiliaryFeatureActivations(observedColumn, sourceFeatureIndex, sourceActivationValue, connectionDevice)
		if(auxiliaryActivationVectorHasActiveValues(auxiliaryActivations)):
			materialisedConnections = getSecondaryOutputConnectionsMaterialised(observedColumn, connectionDevice)
			featureNeuronsTargetActivation = calculateAuxiliaryFeatureTargetActivation(databaseNetworkObject, materialisedConnections, auxiliaryActivations)
			if(sparseTensorHasValues(featureNeuronsTargetActivation)):
				globalFeatureNeuronsActivationResult, globalFeatureConnectionsActivationResult, globalFeatureNeuronsTimeResult = GIAANNcmn_predictionActivate.applyFeatureNeuronsTargetActivationPredict(databaseNetworkObject, globalFeatureNeuronsActivationResult, globalFeatureConnectionsActivationResult, featureNeuronsTargetActivation, globalFeatureNeuronsTimeResult, sequenceWordIndex, sequenceColumnIndex, applySegmentActivations=False)
		result = globalFeatureNeuronsActivationResult, globalFeatureConnectionsActivationResult, globalFeatureNeuronsTimeResult
		return result

	def getConnectedColumnsForAuxiliaryFeatures(observedColumn, parentFeatureIndex, includeFeatureDetails=False):
		databaseNetworkObject = observedColumn.databaseNetworkObject
		targetColumnsList = []
		columnFeatureMap = {}
		activationValue = pt.tensor(auxiliaryNeuronsSimilarWordsIdentitySimilarity, dtype=arrayType, device=deviceSparse)
		if(auxiliaryNeuronsSimilarWordsPrimeConceptFeatures and int(parentFeatureIndex) == featureIndexPrimeConceptNeuron):
			auxiliaryActivations = calculatePrimeAuxiliaryConceptActivations(databaseNetworkObject, int(observedColumn.conceptIndex), activationValue, deviceSparse)
			if(auxiliaryActivationVectorHasActiveValues(auxiliaryActivations)):
				materialisedConnections = getPrimeOutputConnectionsMaterialised(databaseNetworkObject, deviceSparse, auxiliaryActivations)
				auxiliaryTargetColumnsList, auxiliaryColumnFeatureMap = getConnectedColumnsForMaterialisedAuxiliaryConnections(databaseNetworkObject, materialisedConnections, auxiliaryActivations, includeFeatureDetails)
				targetColumnsList.extend(auxiliaryTargetColumnsList)
				if(includeFeatureDetails):
					for columnValue, featureSet in auxiliaryColumnFeatureMap.items():
						columnFeatureMap.setdefault(columnValue, set()).update(featureSet)
		if(auxiliaryNeuronsSimilarWordsSecondaryConceptFeatures and int(parentFeatureIndex) != featureIndexPrimeConceptNeuron):
			auxiliaryActivations = calculateSecondaryAuxiliaryFeatureActivations(observedColumn, int(parentFeatureIndex), activationValue, deviceSparse)
			if(auxiliaryActivationVectorHasActiveValues(auxiliaryActivations)):
				materialisedConnections = getSecondaryOutputConnectionsMaterialised(observedColumn, deviceSparse)
				auxiliaryTargetColumnsList, auxiliaryColumnFeatureMap = getConnectedColumnsForMaterialisedAuxiliaryConnections(databaseNetworkObject, materialisedConnections, auxiliaryActivations, includeFeatureDetails)
				targetColumnsList.extend(auxiliaryTargetColumnsList)
				if(includeFeatureDetails):
					for columnValue, featureSet in auxiliaryColumnFeatureMap.items():
						columnFeatureMap.setdefault(columnValue, set()).update(featureSet)
		targetColumnsList = sorted(set(targetColumnsList))
		result = targetColumnsList, columnFeatureMap
		return result

	def collapseSourceActivationForAuxiliaryInput(sourceActivation):
		if(sourceActivation.dim() > 0):
			result = sourceActivation.sum()
		else:
			result = sourceActivation
		return result

	def calculatePrimeAuxiliaryConceptActivations(databaseNetworkObject, sourceConceptIndex, sourceActivationValue, targetDevice):
		inputConnections = getPrimeInputConnections(databaseNetworkObject, targetDevice)
		sourceVector = pt.zeros((databaseNetworkObject.c, 1), dtype=arrayType, device=targetDevice)
		sourceVector[int(sourceConceptIndex), 0] = sourceActivationValue.to(targetDevice)
		result = pt.sparse.mm(inputConnections.transpose(0, 1).coalesce(), sourceVector).view(-1)
		return result

	def calculateSecondaryAuxiliaryFeatureActivations(observedColumn, sourceFeatureIndex, sourceActivationValue, targetDevice):
		databaseNetworkObject = observedColumn.databaseNetworkObject
		inputConnections = getSecondaryInputConnections(observedColumn, targetDevice)
		sourceVector = pt.zeros((databaseNetworkObject.f, 1), dtype=arrayType, device=targetDevice)
		sourceVector[int(sourceFeatureIndex), 0] = sourceActivationValue.to(targetDevice)
		result = pt.sparse.mm(inputConnections.transpose(0, 1).coalesce(), sourceVector).view(-1)
		return result

	def getPrimeInputConnections(databaseNetworkObject, targetDevice):
		targetSize = (databaseNetworkObject.c, databaseNetworkObject.fas)
		result = databaseNetworkObject.auxiliaryNeuronsSimilarWordsPrimeInputConnections
		if(result is None or tuple(result.size()) != targetSize):
			result = buildPrimeInputConnections(databaseNetworkObject, targetDevice)
			databaseNetworkObject.auxiliaryNeuronsSimilarWordsPrimeInputConnections = result
		elif(result.device != targetDevice):
			result = result.to(targetDevice)
			databaseNetworkObject.auxiliaryNeuronsSimilarWordsPrimeInputConnections = result
		return result

	def buildPrimeInputConnections(databaseNetworkObject, targetDevice):
		connectionWeights = {}
		primeFeaturePrefixes = getPrimeAuxiliaryFeaturePrefixes()
		for parentKey, auxiliaryActivationRecords in databaseNetworkObject.auxiliaryNeuronsSimilarWordsFeatureIndexWeightsByParentWord.items():
			parentPrefix, parentWord = parseSimilarityParentKey(parentKey)
			if(parentPrefix in primeFeaturePrefixes and parentWord in databaseNetworkObject.conceptColumnsDict):
				similarityThreshold = getSimilarityThresholdForAuxiliaryFeaturePrefix(parentPrefix)
				sourceConceptIndex = databaseNetworkObject.conceptColumnsDict[parentWord]
				for auxiliaryConceptIndex, auxiliaryFeatureIndex, activationWeight in auxiliaryActivationRecords:
					if(activationWeight >= similarityThreshold):
						auxiliaryFeatureWord = getAuxiliaryFeatureWordForIndex(databaseNetworkObject, auxiliaryFeatureIndex)
						if(auxiliaryFeatureWordHasAnyPrefix(auxiliaryFeatureWord, primeFeaturePrefixes)):
							mergeSimilarityInputConnection(connectionWeights, sourceConceptIndex, auxiliaryFeatureIndex, activationWeight)
		sourceIndices, auxiliaryIndices, values = unpackSimilarityInputConnectionWeights(connectionWeights)
		result = createSimilarityInputConnectionsSparseTensor(sourceIndices, auxiliaryIndices, values, (databaseNetworkObject.c, databaseNetworkObject.fas), targetDevice)
		return result

	def getSecondaryInputConnections(observedColumn, targetDevice):
		databaseNetworkObject = observedColumn.databaseNetworkObject
		conceptIndex = normaliseAuxiliaryParentMapConceptIndex(databaseNetworkObject, observedColumn.conceptIndex)
		targetSize = (databaseNetworkObject.f, databaseNetworkObject.fas)
		result = databaseNetworkObject.auxiliaryNeuronsSimilarWordsSecondaryInputConnectionsByConceptIndex.get(conceptIndex)
		if(result is None or tuple(result.size()) != targetSize):
			result = buildSecondaryInputConnections(observedColumn, targetDevice)
			databaseNetworkObject.auxiliaryNeuronsSimilarWordsSecondaryInputConnectionsByConceptIndex[conceptIndex] = result
		elif(result.device != targetDevice):
			result = result.to(targetDevice)
			databaseNetworkObject.auxiliaryNeuronsSimilarWordsSecondaryInputConnectionsByConceptIndex[conceptIndex] = result
		return result

	def buildSecondaryInputConnections(observedColumn, targetDevice):
		databaseNetworkObject = observedColumn.databaseNetworkObject
		conceptIndex = normaliseAuxiliaryParentMapConceptIndex(databaseNetworkObject, observedColumn.conceptIndex)
		connectionWeights = {}
		secondaryFeaturePrefixes = getSecondaryAuxiliaryFeaturePrefixes()
		for parentKey, auxiliaryActivationRecords in databaseNetworkObject.auxiliaryNeuronsSimilarWordsFeatureIndexWeightsByParentWord.items():
			parentPrefix, parentWord = parseSimilarityParentKey(parentKey)
			if(parentPrefix in secondaryFeaturePrefixes and parentWord in databaseNetworkObject.conceptFeaturesDict):
				similarityThreshold = getSimilarityThresholdForAuxiliaryFeaturePrefix(parentPrefix)
				sourceFeatureIndex = databaseNetworkObject.conceptFeaturesDict[parentWord]
				for auxiliaryConceptIndex, auxiliaryFeatureIndex, activationWeight in auxiliaryActivationRecords:
					if(auxiliaryConceptIndex == conceptIndex and activationWeight >= similarityThreshold):
						auxiliaryFeatureWord = getAuxiliaryFeatureWordForIndex(databaseNetworkObject, auxiliaryFeatureIndex)
						if(auxiliaryFeatureWordHasAnyPrefix(auxiliaryFeatureWord, secondaryFeaturePrefixes)):
							mergeSimilarityInputConnection(connectionWeights, sourceFeatureIndex, auxiliaryFeatureIndex, activationWeight)
		sourceIndices, auxiliaryIndices, values = unpackSimilarityInputConnectionWeights(connectionWeights)
		result = createSimilarityInputConnectionsSparseTensor(sourceIndices, auxiliaryIndices, values, (databaseNetworkObject.f, databaseNetworkObject.fas), targetDevice)
		return result

	def mergeSimilarityInputConnection(connectionWeights, sourceIndex, auxiliaryIndex, activationWeight):
		connectionKey = (int(sourceIndex), int(auxiliaryIndex))
		currentActivationWeight = connectionWeights.get(connectionKey)
		if(currentActivationWeight is None or activationWeight > currentActivationWeight):
			connectionWeights[connectionKey] = activationWeight
		return

	def unpackSimilarityInputConnectionWeights(connectionWeights):
		sourceIndices = []
		auxiliaryIndices = []
		values = []
		for connectionKey in sorted(connectionWeights.keys()):
			sourceIndex, auxiliaryIndex = connectionKey
			sourceIndices.append(sourceIndex)
			auxiliaryIndices.append(auxiliaryIndex)
			values.append(connectionWeights[connectionKey])
		result = sourceIndices, auxiliaryIndices, values
		return result

	def createSimilarityInputConnectionsSparseTensor(sourceIndices, auxiliaryIndices, values, targetSize, targetDevice):
		if(len(values) > 0):
			indices = pt.tensor([sourceIndices, auxiliaryIndices], dtype=pt.long, device=targetDevice)
			valuesTensor = pt.tensor(values, dtype=arrayType, device=targetDevice)
			result = pt.sparse_coo_tensor(indices, valuesTensor, size=targetSize, dtype=arrayType, device=targetDevice).coalesce()
		else:
			indices = pt.empty((2, 0), dtype=pt.long, device=targetDevice)
			valuesTensor = pt.empty((0,), dtype=arrayType, device=targetDevice)
			result = pt.sparse_coo_tensor(indices, valuesTensor, size=targetSize, dtype=arrayType, device=targetDevice)
		return result

	def getPrimeOutputConnectionsMaterialised(databaseNetworkObject, targetDevice, auxiliaryActivations=None):
		if(auxiliaryActivations is None):
			targetSize = getMaterialisedAuxiliaryFeatureConnectionsTargetSize(databaseNetworkObject, databaseNetworkObject.fas)
			result = databaseNetworkObject.auxiliaryNeuronsSimilarWordsPrimeOutputConnectionsMaterialised
			if(result is None or tuple(result.size()) != targetSize):
				result = buildPrimeOutputConnectionsMaterialised(databaseNetworkObject, targetDevice)
				databaseNetworkObject.auxiliaryNeuronsSimilarWordsPrimeOutputConnectionsMaterialised = result
			elif(result.device != targetDevice):
				result = result.to(targetDevice)
				databaseNetworkObject.auxiliaryNeuronsSimilarWordsPrimeOutputConnectionsMaterialised = result
		else:
			activeAuxiliaryFeatureIndices = getActiveAuxiliaryActivationIndices(auxiliaryActivations)
			result = buildPrimeOutputConnectionsMaterialised(databaseNetworkObject, targetDevice, activeAuxiliaryFeatureIndices)
		return result

	def buildPrimeOutputConnectionsMaterialised(databaseNetworkObject, targetDevice, auxiliaryFeatureIndices=None):
		import GIAANNcmn_databaseNetwork
		indicesList = []
		valuesList = []
		if(auxiliaryFeatureIndices is None):
			for auxiliaryConceptIndex in range(databaseNetworkObject.c):
				conceptName = databaseNetworkObject.conceptColumnsList[auxiliaryConceptIndex]
				observedColumn = GIAANNcmn_databaseNetwork.loadOrCreateObservedColumn(databaseNetworkObject, auxiliaryConceptIndex, conceptName, auxiliaryConceptIndex, targetDevice=targetDevice, createDeviceCopy=False, loadAllSourceFeatures=False)
				appendObservedColumnPrimeAuxiliaryConnectionsMaterialised(indicesList, valuesList, observedColumn, targetDevice)
		else:
			for auxiliaryFeatureIndex in auxiliaryFeatureIndices:
				auxiliaryFeatureWord = getAuxiliaryFeatureWordForIndex(databaseNetworkObject, auxiliaryFeatureIndex)
				if(auxiliaryFeatureWordHasAnyPrefix(auxiliaryFeatureWord, getPrimeAuxiliaryFeaturePrefixes())):
					auxiliaryConceptIndex, auxiliaryBaseWord = parseConceptColumnAuxiliaryFeatureName(databaseNetworkObject, auxiliaryFeatureWord)
					conceptName = databaseNetworkObject.conceptColumnsList[auxiliaryConceptIndex]
					observedColumn = GIAANNcmn_databaseNetwork.loadOrCreateObservedColumn(databaseNetworkObject, auxiliaryConceptIndex, conceptName, auxiliaryConceptIndex, targetDevice=targetDevice, createDeviceCopy=False, loadAllSourceFeatures=False)
					appendObservedColumnAuxiliaryConnectionsMaterialised(indicesList, valuesList, observedColumn, auxiliaryFeatureIndex, auxiliaryFeatureIndex, targetDevice)
		result = buildMaterialisedAuxiliaryFeatureConnections(databaseNetworkObject, indicesList, valuesList, databaseNetworkObject.fas, targetDevice)
		return result

	def getSecondaryOutputConnectionsMaterialised(observedColumn, targetDevice):
		databaseNetworkObject = observedColumn.databaseNetworkObject
		targetSize = getMaterialisedAuxiliaryFeatureConnectionsTargetSize(databaseNetworkObject, databaseNetworkObject.fas)
		ensureObservedColumnAuxiliaryStorage(observedColumn)
		result = observedColumn.similarAuxiliarySecondaryOutputConnectionsMaterialised
		if(result is None or tuple(result.size()) != targetSize):
			indicesList = []
			valuesList = []
			appendObservedColumnSecondaryAuxiliaryConnectionsMaterialised(indicesList, valuesList, observedColumn, targetDevice)
			result = buildMaterialisedAuxiliaryFeatureConnections(databaseNetworkObject, indicesList, valuesList, databaseNetworkObject.fas, targetDevice)
			observedColumn.similarAuxiliarySecondaryOutputConnectionsMaterialised = result
		elif(result.device != targetDevice):
			result = result.to(targetDevice)
			observedColumn.similarAuxiliarySecondaryOutputConnectionsMaterialised = result
		return result

	def appendObservedColumnPrimeAuxiliaryConnectionsMaterialised(indicesList, valuesList, observedColumn, targetDevice):
		sourceFeatureIndices = getObservedColumnAuxiliarySourceFeatureIndicesForMaterialisation(observedColumn)
		primeFeaturePrefixes = getPrimeAuxiliaryFeaturePrefixes()
		for sourceFeatureIndex in sourceFeatureIndices:
			auxiliaryFeatureWord = getAuxiliaryFeatureWordForIndex(observedColumn.databaseNetworkObject, sourceFeatureIndex)
			if(auxiliaryFeatureWordHasAnyPrefix(auxiliaryFeatureWord, primeFeaturePrefixes)):
				auxiliaryConceptIndex, auxiliaryBaseWord = parseConceptColumnAuxiliaryFeatureName(observedColumn.databaseNetworkObject, auxiliaryFeatureWord)
				if(auxiliaryConceptIndex != int(observedColumn.conceptIndex)):
					raise RuntimeError("appendObservedColumnPrimeAuxiliaryConnectionsMaterialised error: auxiliary concept index mismatch")
				appendObservedColumnAuxiliaryConnectionsMaterialised(indicesList, valuesList, observedColumn, sourceFeatureIndex, sourceFeatureIndex, targetDevice)
		return

	def appendObservedColumnSecondaryAuxiliaryConnectionsMaterialised(indicesList, valuesList, observedColumn, targetDevice):
		sourceFeatureIndices = getObservedColumnAuxiliarySourceFeatureIndicesForMaterialisation(observedColumn)
		secondaryFeaturePrefixes = getSecondaryAuxiliaryFeaturePrefixes()
		for sourceFeatureIndex in sourceFeatureIndices:
			auxiliaryFeatureWord = getAuxiliaryFeatureWordForIndex(observedColumn.databaseNetworkObject, sourceFeatureIndex)
			if(auxiliaryFeatureWordHasAnyPrefix(auxiliaryFeatureWord, secondaryFeaturePrefixes)):
				auxiliaryConceptIndex, auxiliaryBaseWord = parseConceptColumnAuxiliaryFeatureName(observedColumn.databaseNetworkObject, auxiliaryFeatureWord)
				if(auxiliaryConceptIndex != int(observedColumn.conceptIndex)):
					raise RuntimeError("appendObservedColumnSecondaryAuxiliaryConnectionsMaterialised error: auxiliary concept index mismatch")
				appendObservedColumnAuxiliaryConnectionsMaterialised(indicesList, valuesList, observedColumn, sourceFeatureIndex, sourceFeatureIndex, targetDevice)
		return

	def appendObservedColumnAuxiliaryConnectionsMaterialised(indicesList, valuesList, observedColumn, sourceFeatureIndex, materialisedSourceIndex, targetDevice):
		sourceTensor = getObservedColumnAuxiliaryFeatureConnectionsForSourceFeature(observedColumn, sourceFeatureIndex, targetDevice=targetDevice, createMissing=False, ensureCurrentSizeOnLoad=True)
		sourceTensor = sourceTensor.coalesce()
		if(sourceTensor._nnz() > 0):
			sourceIndices = sourceTensor.indices()
			materialisedSourceIndices = pt.full((sourceIndices.shape[1],), int(materialisedSourceIndex), dtype=pt.long, device=targetDevice)
			materialisedIndices = pt.stack((sourceIndices[0], sourceIndices[1], sourceIndices[2], materialisedSourceIndices, sourceIndices[3], sourceIndices[4]), dim=0)
			indicesList.append(materialisedIndices)
			valuesList.append(sourceTensor.values())
		return

	def getObservedColumnAuxiliarySourceFeatureIndicesForMaterialisation(observedColumn):
		ensureObservedColumnAuxiliaryStorage(observedColumn)
		sourceFeatureIndices = set(listObservedColumnAuxiliarySourceFeatureIndices(observedColumn.databaseNetworkObject, observedColumn.conceptIndex))
		for sourceFeatureIndex in observedColumn.similarAuxiliaryFeatureConnectionsBySourceFeature.keys():
			sourceFeatureIndices.add(normaliseAuxiliarySourceFeatureIndex(observedColumn.databaseNetworkObject, sourceFeatureIndex))
		result = sorted(sourceFeatureIndices)
		return result

	def getMaterialisedAuxiliaryFeatureConnectionsTargetSize(databaseNetworkObject, sourceDimensionSize):
		result = (databaseNetworkObject.arrayNumberOfProperties, numberOfDendriticBranches, arrayNumberOfSegments, int(sourceDimensionSize), databaseNetworkObject.c, databaseNetworkObject.f)
		return result

	def buildMaterialisedAuxiliaryFeatureConnections(databaseNetworkObject, indicesList, valuesList, sourceDimensionSize, targetDevice):
		targetSize = getMaterialisedAuxiliaryFeatureConnectionsTargetSize(databaseNetworkObject, sourceDimensionSize)
		if(len(indicesList) > 0):
			indices = pt.cat(indicesList, dim=1)
			values = pt.cat(valuesList, dim=0)
			result = pt.sparse_coo_tensor(indices, values, size=targetSize, dtype=arrayType, device=targetDevice).coalesce()
		else:
			indices = pt.empty((6, 0), dtype=pt.long, device=targetDevice)
			values = pt.empty((0,), dtype=arrayType, device=targetDevice)
			result = pt.sparse_coo_tensor(indices, values, size=targetSize, dtype=arrayType, device=targetDevice)
		return result

	def calculateAuxiliaryFeatureTargetActivation(databaseNetworkObject, materialisedConnections, auxiliaryActivations):
		import GIAANNcmn_predictionActivate
		if(inferenceConnectionStrengthPOSdependence):
			raise RuntimeError("calculateAuxiliaryFeatureTargetActivation error: auxiliary POS connection properties are not implemented")
		if(materialisedConnections.layout != pt.sparse_coo):
			raise RuntimeError("calculateAuxiliaryFeatureTargetActivation error: materialisedConnections must be sparse COO")
		featureConnectionsStrength = materialisedConnections[databaseNetworkObject.arrayIndexPropertiesStrengthIndex]
		if(inferenceConnectionsStrengthBoolean):
			featureConnectionsStrength = featureConnectionsStrength.bool().float()
		if(not featureConnectionsStrength.is_sparse):
			raise RuntimeError("calculateAuxiliaryFeatureTargetActivation error: featureConnectionsStrength must be sparse")
		featureConnectionsStrength = featureConnectionsStrength.coalesce()
		if(featureConnectionsStrength._nnz() > 0):
			indices = featureConnectionsStrength.indices()
			values = featureConnectionsStrength.values()
			activationValues = auxiliaryActivations.to(values.device).index_select(0, indices[2])
			activeMask = activationValues > auxiliaryNeuronsSimilarWordsMinimumSimilarity
			if(activeMask.any()):
				activeIndices = indices[:, activeMask]
				activeValues = values[activeMask] * activationValues[activeMask]
				activeTensor = pt.sparse_coo_tensor(activeIndices, activeValues, size=featureConnectionsStrength.size(), dtype=arrayType, device=values.device).coalesce()
				activeTensor = GIAANNcmn_predictionActivate.transformFeatureNeuronsTargetActivationPredict(activeTensor)
				result = collapseAuxiliarySourceDimensionTargetActivation(databaseNetworkObject, activeTensor)
			else:
				result = createEmptyFeatureNeuronsTargetActivation(databaseNetworkObject, featureConnectionsStrength.device)
		else:
			result = createEmptyFeatureNeuronsTargetActivation(databaseNetworkObject, featureConnectionsStrength.device)
		return result

	def collapseAuxiliarySourceDimensionTargetActivation(databaseNetworkObject, sourceTargetActivation):
		sourceTargetActivation = sourceTargetActivation.coalesce()
		if(sourceTargetActivation._nnz() > 0):
			indices = sourceTargetActivation.indices()
			values = sourceTargetActivation.values()
			targetIndices = pt.stack((indices[0], indices[1], indices[3], indices[4]), dim=0)
			result = pt.sparse_coo_tensor(targetIndices, values, size=(numberOfDendriticBranches, arrayNumberOfSegments, databaseNetworkObject.c, databaseNetworkObject.f), dtype=arrayType, device=values.device).coalesce()
		else:
			result = createEmptyFeatureNeuronsTargetActivation(databaseNetworkObject, sourceTargetActivation.device)
		return result

	def createEmptyFeatureNeuronsTargetActivation(databaseNetworkObject, targetDevice):
		indices = pt.empty((4, 0), dtype=pt.long, device=targetDevice)
		values = pt.empty((0,), dtype=arrayType, device=targetDevice)
		result = pt.sparse_coo_tensor(indices, values, size=(numberOfDendriticBranches, arrayNumberOfSegments, databaseNetworkObject.c, databaseNetworkObject.f), dtype=arrayType, device=targetDevice)
		return result

	def getConnectedColumnsForMaterialisedAuxiliaryConnections(databaseNetworkObject, materialisedConnections, auxiliaryActivations, includeFeatureDetails):
		targetColumnsList = []
		columnFeatureMap = {}
		featureConnectionsStrength = materialisedConnections[databaseNetworkObject.arrayIndexPropertiesStrengthIndex]
		featureConnectionsStrength = featureConnectionsStrength.coalesce()
		if(featureConnectionsStrength._nnz() > 0):
			targetColumnIndices = featureConnectionsStrength.indices()
			activationValues = auxiliaryActivations.to(featureConnectionsStrength.device).index_select(0, targetColumnIndices[2])
			activeMask = activationValues > auxiliaryNeuronsSimilarWordsMinimumSimilarity
			if(algorithmMatrixSANImethod=="enforceActivationAcrossSegments" and algorithmMatrixSANIenforceRequirement=="enforceLastSegmentMustBeActive"):
				activeMask = activeMask & (targetColumnIndices[1] == arrayIndexSegmentLast)
			if(activeMask.any()):
				targetColumnIndices = targetColumnIndices[:, activeMask]
				targetColumns = targetColumnIndices[3].unique()
				targetColumnsList.extend(targetColumns.cpu().tolist())
				if(includeFeatureDetails):
					for columnValue, featureValue in zip(targetColumnIndices[3].cpu().tolist(), targetColumnIndices[4].cpu().tolist()):
						columnFeatureMap.setdefault(columnValue, set()).add(featureValue)
		result = targetColumnsList, columnFeatureMap
		return result

	def getActiveAuxiliaryActivationIndices(auxiliaryActivations):
		activeMask = auxiliaryActivations > auxiliaryNeuronsSimilarWordsMinimumSimilarity
		if(activeMask.any()):
			result = pt.nonzero(activeMask, as_tuple=False).view(-1).detach().cpu().tolist()
		else:
			result = []
		return result

	def auxiliaryActivationVectorHasActiveValues(auxiliaryActivations):
		result = auxiliaryActivations.numel() > 0 and bool((auxiliaryActivations > auxiliaryNeuronsSimilarWordsMinimumSimilarity).any().item())
		return result

	def sparseTensorHasValues(tensor):
		if(tensor.layout == pt.sparse_coo):
			result = tensor.coalesce()._nnz() > 0
		else:
			result = bool(pt.count_nonzero(tensor).item() > 0)
		return result

	def getAuxiliaryFeatureWordForIndex(databaseNetworkObject, auxiliaryFeatureIndex):
		if(int(auxiliaryFeatureIndex) not in databaseNetworkObject.auxiliaryNeuronsSimilarWordsFeaturesIndexToWordDict):
			raise RuntimeError("getAuxiliaryFeatureWordForIndex error: auxiliary feature index not found")
		result = databaseNetworkObject.auxiliaryNeuronsSimilarWordsFeaturesIndexToWordDict[int(auxiliaryFeatureIndex)]
		return result

	def auxiliaryFeatureWordHasPrefix(auxiliaryFeatureWord, auxiliaryFeaturePrefix):
		result = auxiliaryFeatureWord.startswith(auxiliaryFeaturePrefix + auxiliaryNeuronsSimilarWordsFeatureNameDelimiter)
		return result

	def auxiliaryFeatureWordHasAnyPrefix(auxiliaryFeatureWord, auxiliaryFeaturePrefixes):
		result = False
		for auxiliaryFeaturePrefix in auxiliaryFeaturePrefixes:
			if(auxiliaryFeatureWordHasPrefix(auxiliaryFeatureWord, auxiliaryFeaturePrefix)):
				result = True
		return result

	def getPrimeAuxiliaryFeaturePrefixes():
		result = []
		if(auxiliaryNeuronsSimilarWordsPrimeConceptFeatures):
			result.append(auxiliaryNeuronsSimilarWordsFeatureNamePrefixPrimeConcept)
		if(auxiliaryNeuronsSimilarSubwordAuto and auxiliaryNeuronsSimilarSubwordPrimeConceptFeatures):
			result.append(auxiliaryNeuronsSimilarWordsFeatureNamePrefixSubwordPrimeConcept)
		return result

	def getSecondaryAuxiliaryFeaturePrefixes():
		result = []
		if(auxiliaryNeuronsSimilarWordsSecondaryConceptFeatures):
			result.append(auxiliaryNeuronsSimilarWordsFeatureNamePrefixSecondary)
		if(auxiliaryNeuronsSimilarSubwordAuto and auxiliaryNeuronsSimilarSubwordSecondaryConceptFeatures):
			result.append(auxiliaryNeuronsSimilarWordsFeatureNamePrefixSubwordSecondary)
		return result

	def ensureRAMdatabaseAuxiliaryFeatureTensorSizes(observedColumn):
		ensureObservedColumnAuxiliaryStorage(observedColumn)
		for sourceFeatureIndex in sorted(observedColumn.similarAuxiliaryFeatureConnectionsBySourceFeature.keys()):
			ensureObservedColumnAuxiliaryFeatureConnectionSize(observedColumn, sourceFeatureIndex)
		return

	def saveObservedColumnAuxiliaryFeatureConnectionsToDisk(observedColumn, saveAllSourceFeatures):
		ensureObservedColumnAuxiliaryStorage(observedColumn)
		if(saveAllSourceFeatures):
			sourceFeatureIndicesToSave = sorted(observedColumn.similarAuxiliaryLoadedSourceFeatureIndices)
		else:
			sourceFeatureIndicesToSave = sorted(observedColumn.similarAuxiliaryTrainPreparedSourceFeatureIndices)
		for sourceFeatureIndex in sourceFeatureIndicesToSave:
			if(sourceFeatureIndex not in observedColumn.similarAuxiliaryFeatureConnectionsBySourceFeature):
				raise RuntimeError("saveObservedColumnAuxiliaryFeatureConnectionsToDiskSimilarity error: missing loaded source feature tensor")
			saveObservedColumnAuxiliaryFeatureConnectionsTensor(observedColumn.conceptIndex, sourceFeatureIndex, observedColumn.similarAuxiliaryFeatureConnectionsBySourceFeature[sourceFeatureIndex])
		return

	def loadObservedColumnAuxiliaryConnectionsFromDisk(observedColumn, targetDevice=None, loadAllSourceFeatures=False, resizeFeatureTensorsToCurrentSize=False):
		ensureObservedColumnAuxiliaryStorage(observedColumn)
		if(loadAllSourceFeatures):
			sourceFeatureIndices = listObservedColumnAuxiliarySourceFeatureIndices(observedColumn.databaseNetworkObject, observedColumn.conceptIndex)
			loadTargetDevice = targetDevice if targetDevice is not None else deviceDatabase
			for sourceFeatureIndex in sourceFeatureIndices:
				getObservedColumnAuxiliaryFeatureConnectionsForSourceFeature(observedColumn, sourceFeatureIndex, targetDevice=loadTargetDevice, createMissing=False, ensureCurrentSizeOnLoad=resizeFeatureTensorsToCurrentSize)
		return

	def moveObservedColumnAuxiliaryConnectionsToDatabaseAfterTrain(observedColumn):
		ensureObservedColumnAuxiliaryStorage(observedColumn)
		for sourceFeatureIndex in sorted(observedColumn.similarAuxiliaryTrainPreparedSourceFeatureIndices):
			sourceTensor = getObservedColumnAuxiliaryFeatureConnectionsForSourceFeature(observedColumn, sourceFeatureIndex, targetDevice=deviceDatabase, createMissing=False)
			setObservedColumnAuxiliaryFeatureConnectionsForSourceFeature(observedColumn, sourceFeatureIndex, sourceTensor)
		observedColumn.similarAuxiliaryTrainPreparedSourceFeatureIndices.clear()
		return

	def getRequiredAuxiliaryFeatureIndicesByObservedColumn(sequenceObservedColumns, allowNewFeatures):
		if(not isinstance(allowNewFeatures, bool)):
			raise RuntimeError("getRequiredAuxiliaryFeatureIndicesByObservedColumnSimilarity error: allowNewFeatures must be bool")
		result = {}
		sequenceObservedColumns.ensureTokenConceptColumnIndexList()
		for observedColumn in sequenceObservedColumns.observedColumnsDict.values():
			result[int(observedColumn.conceptIndex)] = set()
		for tokenIndex, token in enumerate(sequenceObservedColumns.tokens):
			isConcept = tokenIndex in sequenceObservedColumns.columnsIndexSequenceWordIndexDict
			conceptIndex = sequenceObservedColumns.tokenConceptColumnIndexList[tokenIndex]
			if(conceptIndex is None):
				raise RuntimeError("getRequiredAuxiliaryFeatureIndicesByObservedColumnSimilarity error: unassigned token")
			if(int(conceptIndex) not in result):
				raise RuntimeError("getRequiredAuxiliaryFeatureIndicesByObservedColumnSimilarity error: missing observed column")
			for auxiliaryFeatureIndex in getTokenAuxiliaryFeatureIndices(sequenceObservedColumns.databaseNetworkObject, token, isConcept, conceptIndex, allowNewFeatures, registerParent=True):
				result[int(conceptIndex)].add(int(auxiliaryFeatureIndex))
		for conceptIndex, auxiliaryFeatureIndices in result.items():
			result[conceptIndex] = sorted(auxiliaryFeatureIndices)
		sequenceObservedColumns.requiredSimilarAuxiliarySourceFeatureIndicesByObservedColumn = result
		return result

	def getTokenAuxiliaryFeatureIndices(databaseNetworkObject, token, isConcept, conceptIndex, allowNewFeatures=False, registerParent=False):
		result = []
		if(auxiliaryNeuronsAuto):
			import GIAANNnlp_auxiliaryNeuronsAuto
			result = GIAANNnlp_auxiliaryNeuronsAuto.getTokenAutoAuxiliaryFeatureIndices(databaseNetworkObject, token, isConcept, conceptIndex, allowNewFeatures, registerParent)
		else:
			if(isConcept):
				if(auxiliaryNeuronsSimilarWordsPrimeConceptFeatures and tokenHasPrimeConceptSimilarityWord(token)):
					similarityWord = getTokenPrimeConceptSimilarityWord(token)
					auxiliaryFeatureWord = buildConceptColumnAuxiliaryFeatureName(databaseNetworkObject, auxiliaryNeuronsSimilarWordsFeatureNamePrefixPrimeConcept, conceptIndex, similarityWord)
					auxiliaryFeatureIndex = registerAuxiliaryFeatureWord(databaseNetworkObject, auxiliaryFeatureWord, allowNewFeatures)
					if(registerParent):
						registerSimilarityParentFeatureWordWeights(databaseNetworkObject, auxiliaryNeuronsSimilarWordsFeatureNamePrefixPrimeConcept, similarityWord, auxiliaryFeatureWord)
					result.append(auxiliaryFeatureIndex)
			else:
				if(auxiliaryNeuronsSimilarWordsSecondaryConceptFeatures and tokenHasSecondarySimilarityWord(token)):
					similarityWord = getTokenSecondarySimilarityWord(token)
					auxiliaryFeatureWord = buildConceptColumnAuxiliaryFeatureName(databaseNetworkObject, auxiliaryNeuronsSimilarWordsFeatureNamePrefixSecondary, conceptIndex, similarityWord)
					auxiliaryFeatureIndex = registerAuxiliaryFeatureWord(databaseNetworkObject, auxiliaryFeatureWord, allowNewFeatures)
					if(registerParent):
						registerSimilarityParentFeatureWordWeights(databaseNetworkObject, auxiliaryNeuronsSimilarWordsFeatureNamePrefixSecondary, similarityWord, auxiliaryFeatureWord)
					result.append(auxiliaryFeatureIndex)
		return result

	def buildAuxiliarySourceOccurrenceTensors(sequenceObservedColumns, conceptIndices, startIndices, endIndices, targetDevice):
		sourceConceptIndexList = []
		sourceAuxiliaryFeatureIndexList = []
		sourceWordOrderList = []
		tokenSequenceConceptIndexList = buildTokenSequenceConceptIndexList(sequenceObservedColumns.tokens, conceptIndices, startIndices, endIndices)
		for tokenIndex, token in enumerate(sequenceObservedColumns.tokens):
			isConcept = tokenIndex in sequenceObservedColumns.columnsIndexSequenceWordIndexDict
			sequenceConceptIndex = tokenSequenceConceptIndexList[tokenIndex]
			if(sequenceConceptIndex is None):
				raise RuntimeError("buildAuxiliarySourceOccurrenceTensorsSimilarity error: token has no sequence concept index")
			conceptIndex = sequenceObservedColumns.tokenConceptColumnIndexList[tokenIndex]
			if(conceptIndex is None):
				raise RuntimeError("buildAuxiliarySourceOccurrenceTensorsSimilarity error: token has no concept index")
			for auxiliaryFeatureIndex in getTokenAuxiliaryFeatureIndices(sequenceObservedColumns.databaseNetworkObject, token, isConcept, conceptIndex):
				sourceConceptIndexList.append(sequenceConceptIndex)
				sourceAuxiliaryFeatureIndexList.append(auxiliaryFeatureIndex)
				sourceWordOrderList.append(tokenIndex)
		sourceConceptIndices = pt.tensor(sourceConceptIndexList, dtype=pt.long, device=targetDevice)
		sourceAuxiliaryFeatureIndices = pt.tensor(sourceAuxiliaryFeatureIndexList, dtype=pt.long, device=targetDevice)
		sourceWordOrder = pt.tensor(sourceWordOrderList, dtype=pt.long, device=targetDevice)
		result = sourceConceptIndices, sourceAuxiliaryFeatureIndices, sourceWordOrder
		return result

	def buildAuxiliaryConnectionIndicesAndValues(sequenceObservedColumns, featureNeuronsActive, columnsWordOrder, featureNeuronsWordOrder, sourceConceptIndices, sourceAuxiliaryFeatureIndices, sourceWordOrder):
		connectionDevice = featureNeuronsActive.device
		indicesList = []
		valuesList = []
		if(sourceConceptIndices.numel() > 0):
			targetActive = featureNeuronsActive.amax(dim=1)
			targetIndices = pt.nonzero(targetActive > 0, as_tuple=False)
			if(targetIndices.numel() > 0):
				sourceCount = sourceConceptIndices.shape[0]
				targetCount = targetIndices.shape[0]
				branchIndices = targetIndices[:, 0].repeat(sourceCount)
				targetConceptIndices = targetIndices[:, 1].repeat(sourceCount)
				targetFeatureIndices = targetIndices[:, 2].repeat(sourceCount)
				sourceConceptIndicesPair = sourceConceptIndices.repeat_interleave(targetCount)
				sourceAuxiliaryFeatureIndicesPair = sourceAuxiliaryFeatureIndices.repeat_interleave(targetCount)
				sourceWordOrderPair = sourceWordOrder.repeat_interleave(targetCount)
				targetWordOrder = featureNeuronsWordOrder[targetConceptIndices, targetFeatureIndices]
				connectionMask = createAuxiliaryFeatureWordOrderConnectionMask(sourceWordOrderPair, targetWordOrder, sequenceObservedColumns.trainConnectionsIncludeSameTimeIndex)
				sourceColumnWordOrder = columnsWordOrder[sourceConceptIndicesPair]
				targetColumnWordOrder = columnsWordOrder[targetConceptIndices]
				if(debugConnectColumnsToNextColumnsInSequenceOnly):
					connectionMask = connectionMask & pt.logical_and(targetColumnWordOrder >= sourceColumnWordOrder, targetColumnWordOrder <= sourceColumnWordOrder+1)
				else:
					connectionMask = connectionMask & (targetColumnWordOrder >= sourceColumnWordOrder)
				if(connectionMask.any()):
					branchIndices = branchIndices[connectionMask]
					sourceConceptIndicesPair = sourceConceptIndicesPair[connectionMask]
					sourceAuxiliaryFeatureIndicesPair = sourceAuxiliaryFeatureIndicesPair[connectionMask]
					targetConceptIndices = targetConceptIndices[connectionMask]
					targetFeatureIndices = targetFeatureIndices[connectionMask]
					sourceWordOrderPair = sourceWordOrderPair[connectionMask]
					targetWordOrder = targetWordOrder[connectionMask]
					baseValues = pt.ones((branchIndices.shape[0],), dtype=arrayType, device=connectionDevice)
					if(trainConnectionStrengthNormaliseWrtContextLength):
						connectionDistances = pt.abs(targetWordOrder - sourceWordOrderPair).to(baseValues.dtype)
						baseValues = baseValues * (auxiliaryNeuronsSimilarWordsConnectionProximityMultiplier/(connectionDistances + 1))
					appendAuxiliaryConnectionSegmentIndices(indicesList, valuesList, branchIndices, sourceConceptIndicesPair, sourceAuxiliaryFeatureIndicesPair, targetConceptIndices, targetFeatureIndices, sourceWordOrderPair, targetWordOrder, baseValues)
		if(len(indicesList) > 0):
			combinedIndices = pt.cat(indicesList, dim=1)
			combinedValues = pt.cat(valuesList, dim=0)
			sparseSize = (numberOfDendriticBranches, arrayNumberOfSegments, sequenceObservedColumns.cs, sequenceObservedColumns.databaseNetworkObject.fas, sequenceObservedColumns.cs, sequenceObservedColumns.fs)
			connectionSparse = pt.sparse_coo_tensor(combinedIndices, combinedValues, size=sparseSize, dtype=arrayType, device=connectionDevice).coalesce()
			combinedIndices = connectionSparse.indices()
			combinedValues = connectionSparse.values()
		else:
			combinedIndices = pt.empty((6, 0), dtype=pt.long, device=connectionDevice)
			combinedValues = pt.empty((0,), dtype=arrayType, device=connectionDevice)
		result = combinedIndices, combinedValues
		return result

	def applyAuxiliaryConnectionPropertyUpdates(sequenceObservedColumns, connectionIndices, connectionValues, propertyIndex):
		if(connectionIndices.numel() > 0):
			connectionDevice = connectionIndices.device
			databaseNetworkObject = sequenceObservedColumns.databaseNetworkObject
			conceptIndicesTensor = sequenceObservedColumns.conceptIndicesInSequenceObservedTensor.to(connectionDevice)
			featureIndicesInObserved = sequenceObservedColumns.featureIndicesInObservedTensor.to(connectionDevice)
			sourceConceptIndex = conceptIndicesTensor[connectionIndices[2]]
			sourceAuxiliaryFeatureIndex = connectionIndices[3]
			targetConceptIndex = conceptIndicesTensor[connectionIndices[4]]
			targetFeatureIndex = featureIndicesInObserved[connectionIndices[5]]
			sourceCombinedKeys = sourceConceptIndex * databaseNetworkObject.fas + sourceAuxiliaryFeatureIndex
			sortedSourceCombinedKeys, sortOrder = pt.sort(sourceCombinedKeys)
			sortedBranch = connectionIndices[0].index_select(0, sortOrder)
			sortedSegment = connectionIndices[1].index_select(0, sortOrder)
			sortedTargetConceptIndex = targetConceptIndex.index_select(0, sortOrder)
			sortedTargetFeatureIndex = targetFeatureIndex.index_select(0, sortOrder)
			sortedValues = connectionValues.index_select(0, sortOrder)
			uniqueSourceCombinedKeys, counts = pt.unique_consecutive(sortedSourceCombinedKeys, return_counts=True)
			starts = pt.cumsum(counts, 0) - counts
			if(trainSequenceObservedColumnsMatchSequenceWords):
				sequenceObservedColumnsDict = sequenceObservedColumns.sequenceObservedColumnsDict
			else:
				sequenceObservedColumnsDict = sequenceObservedColumns.observedColumnsDict2
			observedColumnsByConceptIndex = sequenceObservedColumns.getObservedColumnsByConceptIndex(sequenceObservedColumnsDict)
			targetSize = (databaseNetworkObject.arrayNumberOfProperties, numberOfDendriticBranches, arrayNumberOfSegments, databaseNetworkObject.c, databaseNetworkObject.f)
			for sourceCombinedKey, start, count in zip(uniqueSourceCombinedKeys.tolist(), starts.tolist(), counts.tolist()):
				end = start + count
				sourceConceptIndexValue = int(sourceCombinedKey // databaseNetworkObject.fas)
				sourceAuxiliaryFeatureIndexValue = int(sourceCombinedKey % databaseNetworkObject.fas)
				if(sourceConceptIndexValue not in observedColumnsByConceptIndex):
					raise RuntimeError("applyAuxiliaryConnectionPropertyUpdatesSimilarity error: missing observed column")
				observedColumn = observedColumnsByConceptIndex[sourceConceptIndexValue]
				if(not storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam):
					if(sourceAuxiliaryFeatureIndexValue not in observedColumn.similarAuxiliaryTrainPreparedSourceFeatureIndices):
						raise RuntimeError("applyAuxiliaryConnectionPropertyUpdatesSimilarity error: source auxiliary feature was not prepared")
				propertyRow = pt.full((count,), propertyIndex, dtype=pt.long, device=connectionDevice)
				updateIndices = pt.stack((propertyRow, sortedBranch[start:end], sortedSegment[start:end], sortedTargetConceptIndex[start:end], sortedTargetFeatureIndex[start:end]), dim=0)
				updateSparse = pt.sparse_coo_tensor(updateIndices, sortedValues[start:end], size=targetSize, dtype=arrayType, device=connectionDevice)
				targetSparse = getObservedColumnAuxiliaryFeatureConnectionsForSourceFeature(observedColumn, sourceAuxiliaryFeatureIndexValue, targetDevice=connectionDevice, createMissing=False)
				targetSparse = addSparseUpdateNonNegative(targetSparse, updateSparse)
				setObservedColumnAuxiliaryFeatureConnectionsForSourceFeature(observedColumn, sourceAuxiliaryFeatureIndexValue, targetSparse)
		return

	def getAuxiliaryFeatureActivationRecordsForParentFeature(databaseNetworkObject, observedColumn, sourceFeatureIndex):
		result = []
		if(observedColumn is None):
			raise RuntimeError("getAuxiliaryFeatureActivationRecordsForParentFeatureSimilarity error: observedColumn is None")
		if(not hasattr(observedColumn, "conceptIndex")):
			raise RuntimeError("getAuxiliaryFeatureActivationRecordsForParentFeatureSimilarity error: observedColumn missing conceptIndex")
		parentKey = getSimilarityParentKeyForSourceFeature(databaseNetworkObject, observedColumn, sourceFeatureIndex)
		if(parentKey in databaseNetworkObject.auxiliaryNeuronsSimilarWordsFeatureIndexWeightsByParentWord):
			result = databaseNetworkObject.auxiliaryNeuronsSimilarWordsFeatureIndexWeightsByParentWord[parentKey]
		return result

	def getObservedColumnForAuxiliaryFeatureActivation(databaseNetworkObject, observedColumn, auxiliaryConceptIndex, targetDevice):
		result = observedColumn
		normalisedConceptIndex = normaliseAuxiliaryParentMapConceptIndex(databaseNetworkObject, auxiliaryConceptIndex)
		if(int(observedColumn.conceptIndex) != normalisedConceptIndex):
			import GIAANNcmn_databaseNetwork
			if(normalisedConceptIndex >= len(databaseNetworkObject.conceptColumnsList)):
				raise RuntimeError("getObservedColumnForAuxiliaryFeatureActivationSimilarity error: auxiliary concept index out of range")
			conceptName = databaseNetworkObject.conceptColumnsList[normalisedConceptIndex]
			result = GIAANNcmn_databaseNetwork.loadOrCreateObservedColumn(databaseNetworkObject, normalisedConceptIndex, conceptName, normalisedConceptIndex, targetDevice=targetDevice, createDeviceCopy=False, loadAllSourceFeatures=False)
		return result

	def getSimilarityParentKeyForSourceFeature(databaseNetworkObject, observedColumn, sourceFeatureIndex):
		result = None
		if(sourceFeatureIndex == featureIndexPrimeConceptNeuron):
			if(auxiliaryNeuronsSimilarWordsPrimeConceptFeatures):
				similarityWord = normaliseSimilarityWord(observedColumn.conceptName)
				result = buildSimilarityParentKey(auxiliaryNeuronsSimilarWordsFeatureNamePrefixPrimeConcept, similarityWord)
		else:
			if(auxiliaryNeuronsSimilarWordsSecondaryConceptFeatures):
				if(sourceFeatureIndex < 0 or sourceFeatureIndex >= len(databaseNetworkObject.conceptFeaturesList)):
					raise RuntimeError("getSimilarityParentKeyForSourceFeature error: sourceFeatureIndex out of range")
				similarityWord = normaliseSimilarityWord(databaseNetworkObject.conceptFeaturesList[sourceFeatureIndex])
				result = buildSimilarityParentKey(auxiliaryNeuronsSimilarWordsFeatureNamePrefixSecondary, similarityWord)
		return result

	def registerAuxiliaryFeatureWord(databaseNetworkObject, auxiliaryFeatureWord, allowNewFeatures):
		result = None
		if(auxiliaryFeatureWord not in databaseNetworkObject.auxiliaryNeuronsSimilarWordsFeaturesDict):
			if(not allowNewFeatures):
				raise RuntimeError("registerAuxiliaryFeatureWordSimilarity error: auxiliary feature word not found while allowNewFeatures is False (" + auxiliaryFeatureWord + ")")
			result = len(databaseNetworkObject.auxiliaryNeuronsSimilarWordsFeaturesDict)
			databaseNetworkObject.auxiliaryNeuronsSimilarWordsFeaturesDict[auxiliaryFeatureWord] = result
			databaseNetworkObject.auxiliaryNeuronsSimilarWordsFeaturesList.append(auxiliaryFeatureWord)
			databaseNetworkObject.auxiliaryNeuronsSimilarWordsFeaturesIndexToWordDict[result] = auxiliaryFeatureWord
			databaseNetworkObject.fas += 1
			invalidateDatabaseAuxiliaryInputConnectionCaches(databaseNetworkObject)
		else:
			result = databaseNetworkObject.auxiliaryNeuronsSimilarWordsFeaturesDict[auxiliaryFeatureWord]
		return result

	def registerSimilarityParentFeatureWordWeights(databaseNetworkObject, auxiliaryFeaturePrefix, auxiliaryBaseWord, auxiliaryFeatureWord):
		if(auxiliaryFeatureWord not in databaseNetworkObject.auxiliaryNeuronsSimilarWordsFeaturesDict):
			raise RuntimeError("registerSimilarityParentFeatureWordWeights error: missing auxiliary feature word " + auxiliaryFeatureWord)
		parentWordWeights = getSimilarityAuxiliaryParentWordWeights(auxiliaryBaseWord)
		for parentWord, activationWeight in parentWordWeights.items():
			parentKey = buildSimilarityParentKey(auxiliaryFeaturePrefix, parentWord)
			registerSimilarityParentFeatureWordWeight(databaseNetworkObject, parentKey, auxiliaryFeatureWord, activationWeight)
		return

	def registerSimilarityParentFeatureWordWeight(databaseNetworkObject, parentKey, auxiliaryFeatureWord, activationWeight):
		parentPrefix, parentWord = parseSimilarityParentKey(parentKey)
		similarityThreshold = getSimilarityThresholdForAuxiliaryFeaturePrefix(parentPrefix)
		normalisedActivationWeight = normaliseSimilarityWeight(activationWeight)
		if(parentKey not in databaseNetworkObject.auxiliaryNeuronsSimilarWordsFeatureWordWeightsByParentWord):
			databaseNetworkObject.auxiliaryNeuronsSimilarWordsFeatureWordWeightsByParentWord[parentKey] = {}
			databaseNetworkObject.auxiliaryNeuronsSimilarWordsFeatureIndexWeightsByParentWord[parentKey] = []
		parentFeatureWords = databaseNetworkObject.auxiliaryNeuronsSimilarWordsFeatureWordWeightsByParentWord[parentKey]
		currentActivationWeight = parentFeatureWords.get(auxiliaryFeatureWord)
		if(normalisedActivationWeight >= similarityThreshold):
			if(currentActivationWeight is None or normalisedActivationWeight > currentActivationWeight):
				parentFeatureWords[auxiliaryFeatureWord] = normalisedActivationWeight
				databaseNetworkObject.auxiliaryNeuronsSimilarWordsFeatureIndexWeightsByParentWord[parentKey] = buildAuxiliaryFeatureIndexWeightListForParentFeatureWords(databaseNetworkObject, parentFeatureWords)
				invalidateDatabaseAuxiliaryInputConnectionCaches(databaseNetworkObject)
		return

	def getSimilarityThresholdForAuxiliaryFeaturePrefix(auxiliaryFeaturePrefix):
		if(auxiliaryNeuronsAuto and auxiliaryNeuronsSimilarSubwordAuto and (auxiliaryFeaturePrefix == auxiliaryNeuronsSimilarWordsFeatureNamePrefixSubwordPrimeConcept or auxiliaryFeaturePrefix == auxiliaryNeuronsSimilarWordsFeatureNamePrefixSubwordSecondary)):
			result = auxiliaryNeuronsSimilarSubwordAutoThreshold
		else:
			result = auxiliaryNeuronsSimilarWordsThreshold
		return result

	def buildAuxiliaryFeatureIndexWeightsByParentWord(databaseNetworkObject):
		result = {}
		for parentKey, auxiliaryFeatureWordWeights in databaseNetworkObject.auxiliaryNeuronsSimilarWordsFeatureWordWeightsByParentWord.items():
			if(not isinstance(auxiliaryFeatureWordWeights, dict)):
				raise RuntimeError("buildAuxiliaryFeatureIndexWeightsByParentWord error: parent map must be dict")
			result[parentKey] = buildAuxiliaryFeatureIndexWeightListForParentFeatureWords(databaseNetworkObject, auxiliaryFeatureWordWeights)
		return result

	def buildAuxiliaryFeatureIndexWeightListForParentFeatureWords(databaseNetworkObject, auxiliaryFeatureWordWeights):
		result = []
		for auxiliaryFeatureWord, activationWeight in auxiliaryFeatureWordWeights.items():
			if(auxiliaryFeatureWord not in databaseNetworkObject.auxiliaryNeuronsSimilarWordsFeaturesDict):
				raise RuntimeError("buildAuxiliaryFeatureIndexWeightListForParentFeatureWords error: missing auxiliary feature word " + auxiliaryFeatureWord)
			auxiliaryConceptIndex, auxiliaryBaseWord = parseConceptColumnAuxiliaryFeatureName(databaseNetworkObject, auxiliaryFeatureWord)
			auxiliaryFeatureIndex = databaseNetworkObject.auxiliaryNeuronsSimilarWordsFeaturesDict[auxiliaryFeatureWord]
			result.append((auxiliaryConceptIndex, auxiliaryFeatureIndex, normaliseSimilarityWeight(activationWeight)))
		result.sort(key=lambda item: (item[0], item[1]))
		return result

	def getSimilarityAuxiliaryParentWordWeights(auxiliaryBaseWord):
		normalisedAuxiliaryBaseWord = normaliseSimilarityWord(auxiliaryBaseWord)
		if(normalisedAuxiliaryBaseWord in auxiliaryNeuronsSimilarWordsParentWordWeightsCache):
			result = auxiliaryNeuronsSimilarWordsParentWordWeightsCache[normalisedAuxiliaryBaseWord]
		else:
			result = {normalisedAuxiliaryBaseWord: auxiliaryNeuronsSimilarWordsIdentitySimilarity}
			similarWordWeights = getSimilarWordWeights(normalisedAuxiliaryBaseWord)
			for parentWord, activationWeight in similarWordWeights.items():
				normalisedParentWord = normaliseSimilarityWord(parentWord)
				normalisedActivationWeight = normaliseSimilarityWeight(activationWeight)
				if(normalisedActivationWeight >= auxiliaryNeuronsSimilarWordsThreshold):
					currentActivationWeight = result.get(normalisedParentWord)
					if(currentActivationWeight is None or normalisedActivationWeight > currentActivationWeight):
						result[normalisedParentWord] = normalisedActivationWeight
			auxiliaryNeuronsSimilarWordsParentWordWeightsCache[normalisedAuxiliaryBaseWord] = result
		return result

	def getSimilarWordWeights(word):
		result = {}
		if(auxiliaryNeuronsSimilarWordsDatasetName == auxiliaryNeuronsSimilarWordsDataset1Name):
			result = getSimilarWordWeightsDatasetWordNet(word)
		elif(auxiliaryNeuronsSimilarWordsDatasetName == auxiliaryNeuronsSimilarWordsDataset2Name):
			result = getSimilarWordWeightsDatasetTextPairs(word)
		elif(auxiliaryNeuronsSimilarWordsDatasetName == auxiliaryNeuronsSimilarWordsDataset3Name):
			result = getSimilarWordWeightsDatasetWord2VecText(word)
		else:
			raise RuntimeError("getSimilarWordWeights error: unsupported auxiliaryNeuronsSimilarWordsDatasetName")
		return result

	def getSimilarWordWeightsDatasetWordNet(word):
		result = {}
		try:
			from nltk.corpus import wordnet as wn
			validateSimilarWordsDatasetWordNet(wn)
			synsets = wn.synsets(word)
		except LookupError as error:
			raise RuntimeError("getSimilarWordWeightsDatasetWordNet error: missing NLTK WordNet dataset") from error
		for synset in synsets:
			for lemma in synset.lemma_names():
				similarWord = normaliseSimilarityDatasetWordNetLemma(lemma)
				if(similarWord != word):
					result[similarWord] = auxiliaryNeuronsSimilarWordsIdentitySimilarity
		return result

	def validateSimilarWordsDatasetWordNet(wn):
		global auxiliaryNeuronsSimilarWordsDatasetWordNetValidated
		if(not auxiliaryNeuronsSimilarWordsDatasetWordNetValidated):
			synsetCount = 0
			for synset in wn.all_synsets():
				synsetCount += 1
				if(synsetCount >= auxiliaryNeuronsSimilarWordsDataset1MinimumSynsets):
					auxiliaryNeuronsSimilarWordsDatasetWordNetValidated = True
					break
			if(not auxiliaryNeuronsSimilarWordsDatasetWordNetValidated):
				raise RuntimeError("getSimilarWordWeightsDatasetWordNet error: WordNet dataset is below production minimum synsets" + auxiliaryNeuronsSimilarWordsDatasetMinimumErrorActualPrefix + str(synsetCount) + auxiliaryNeuronsSimilarWordsDatasetMinimumErrorMinimumPrefix + str(auxiliaryNeuronsSimilarWordsDataset1MinimumSynsets))
		return

	def getSimilarWordWeightsDatasetTextPairs(word):
		global auxiliaryNeuronsSimilarWordsDatasetCache
		if(auxiliaryNeuronsSimilarWordsDatasetCache is None):
			auxiliaryNeuronsSimilarWordsDatasetCache = loadSimilarWordsDatasetTextPairs()
		result = auxiliaryNeuronsSimilarWordsDatasetCache.get(word, {})
		return result

	def loadSimilarWordsDatasetTextPairs():
		result = {}
		rowCount = 0
		uniqueWords = set()
		if(not GIAANNcmn_databaseNetworkFiles.pathExists(auxiliaryNeuronsSimilarWordsDataset2File)):
			raise RuntimeError("loadSimilarWordsDatasetTextPairs error: missing auxiliaryNeuronsSimilarWordsDataset2File = " + auxiliaryNeuronsSimilarWordsDataset2File)
		with open(auxiliaryNeuronsSimilarWordsDataset2File, "r", encoding="utf-8") as fileObject:
			for lineIndex, line in enumerate(fileObject):
				strippedLine = line.strip()
				if(strippedLine != auxiliaryNeuronsSimilarWordsFeatureValueEmpty and not strippedLine.startswith(auxiliaryNeuronsSimilarWordsDataset2CommentPrefix)):
					fields = strippedLine.split(auxiliaryNeuronsSimilarWordsDataset2Delimiter)
					if(len(fields) < auxiliaryNeuronsSimilarWordsDataset2MinimumFields):
						raise RuntimeError("loadSimilarWordsDatasetTextPairs error: line has insufficient fields")
					sourceWord = normaliseSimilarityWord(fields[auxiliaryNeuronsSimilarWordsDataset2SourceWordFieldIndex])
					targetWord = normaliseSimilarityWord(fields[auxiliaryNeuronsSimilarWordsDataset2TargetWordFieldIndex])
					activationWeight = normaliseSimilarityWeight(float(fields[auxiliaryNeuronsSimilarWordsDataset2SimilarityFieldIndex]))
					rowCount += 1
					uniqueWords.add(sourceWord)
					uniqueWords.add(targetWord)
					addSimilarWordsDatasetTextPair(result, sourceWord, targetWord, activationWeight)
					addSimilarWordsDatasetTextPair(result, targetWord, sourceWord, activationWeight)
		validateSimilarWordsDatasetTextPairs(rowCount, uniqueWords)
		return result

	def validateSimilarWordsDatasetTextPairs(rowCount, uniqueWords):
		uniqueWordCount = len(uniqueWords)
		if(rowCount < auxiliaryNeuronsSimilarWordsDataset2MinimumRows):
			raise RuntimeError("loadSimilarWordsDatasetTextPairs error: dataset2 is below production minimum rows" + auxiliaryNeuronsSimilarWordsDatasetMinimumErrorActualPrefix + str(rowCount) + auxiliaryNeuronsSimilarWordsDatasetMinimumErrorMinimumPrefix + str(auxiliaryNeuronsSimilarWordsDataset2MinimumRows))
		if(uniqueWordCount < auxiliaryNeuronsSimilarWordsDataset2MinimumUniqueWords):
			raise RuntimeError("loadSimilarWordsDatasetTextPairs error: dataset2 is below production minimum unique words" + auxiliaryNeuronsSimilarWordsDatasetMinimumErrorActualPrefix + str(uniqueWordCount) + auxiliaryNeuronsSimilarWordsDatasetMinimumErrorMinimumPrefix + str(auxiliaryNeuronsSimilarWordsDataset2MinimumUniqueWords))
		return

	def addSimilarWordsDatasetTextPair(datasetDict, sourceWord, targetWord, activationWeight):
		if(sourceWord not in datasetDict):
			datasetDict[sourceWord] = {}
		currentActivationWeight = datasetDict[sourceWord].get(targetWord)
		if(currentActivationWeight is None or activationWeight > currentActivationWeight):
			datasetDict[sourceWord][targetWord] = activationWeight
		return

	def addSimilarWordsDatasetDirectedPair(datasetDict, sourceWord, targetWord, activationWeight):
		currentActivationWeight = datasetDict[sourceWord].get(targetWord)
		if(currentActivationWeight is None or activationWeight > currentActivationWeight):
			datasetDict[sourceWord][targetWord] = activationWeight
		return

	def getSimilarWordWeightsDatasetWord2VecText(word):
		global auxiliaryNeuronsSimilarWordsDatasetCache
		if(auxiliaryNeuronsSimilarWordsDatasetCache is None):
			auxiliaryNeuronsSimilarWordsDatasetCache = loadSimilarWordsDatasetWord2VecText()
		result = auxiliaryNeuronsSimilarWordsDatasetCache.get(word, {})
		return result

	def loadSimilarWordsDatasetWord2VecText():
		ensureSimilarWordsDataset3CompactFile()
		result = {}
		rowCount = 0
		with open(auxiliaryNeuronsSimilarWordsDataset3File, "r", encoding="utf-8") as fileObject:
			for lineIndex, line in enumerate(fileObject):
				strippedLine = line.strip()
				if(strippedLine != auxiliaryNeuronsSimilarWordsFeatureValueEmpty and not strippedLine.startswith(auxiliaryNeuronsSimilarWordsDataset3CommentPrefix)):
					fields = strippedLine.split(auxiliaryNeuronsSimilarWordsDataset3Delimiter)
					if(len(fields) < auxiliaryNeuronsSimilarWordsDataset3CompactMinimumFields):
						raise RuntimeError("loadSimilarWordsDatasetWord2VecText error: compact row has insufficient fields")
					if((len(fields) - auxiliaryNeuronsSimilarWordsDataset3CompactSimilarWordStartFieldIndex)%auxiliaryNeuronsSimilarWordsDataset3CompactSimilarWordPairFields != 0):
						raise RuntimeError("loadSimilarWordsDatasetWord2VecText error: compact row has incomplete similar-word/score pair")
					sourceWord = normaliseSimilarityWord(fields[auxiliaryNeuronsSimilarWordsDataset3CompactSourceWordFieldIndex])
					if(sourceWord in result):
						raise RuntimeError("loadSimilarWordsDatasetWord2VecText error: duplicate compact source word")
					result[sourceWord] = {}
					rowCount += 1
					for fieldIndex in range(auxiliaryNeuronsSimilarWordsDataset3CompactSimilarWordStartFieldIndex, len(fields), auxiliaryNeuronsSimilarWordsDataset3CompactSimilarWordPairFields):
						targetWord = normaliseSimilarityWord(fields[fieldIndex + auxiliaryNeuronsSimilarWordsDataset3CompactSimilarWordOffset])
						activationWeight = normaliseSimilarityWeight(float(fields[fieldIndex + auxiliaryNeuronsSimilarWordsDataset3CompactSimilarityOffset]))
						addSimilarWordsDatasetDirectedPair(result, sourceWord, targetWord, activationWeight)
		validateSimilarWordsDataset3CompactRows(rowCount)
		return result

	def ensureSimilarWordsDataset3CompactFile():
		if(not GIAANNcmn_databaseNetworkFiles.pathExists(auxiliaryNeuronsSimilarWordsDataset3File)):
			generateSimilarWordsDataset3CompactFile()
		return

	def generateSimilarWordsDataset3CompactFile():
		try:
			from nltk.corpus import wordnet as wn
		except LookupError as error:
			raise RuntimeError("generateSimilarWordsDataset3CompactFile error: missing NLTK WordNet dataset") from error
		print(auxiliaryNeuronsSimilarWordsDataset3GenerateStartMessage + auxiliaryNeuronsSimilarWordsDataset3File)
		candidateMap = buildSimilarWordsDataset3WordNetCandidateMap(wn)
		wordVectors = loadSimilarWordsDataset3SourceVectors(set(candidateMap.keys()))
		os.makedirs(auxiliaryNeuronsSimilarWordsDatasetFolder, exist_ok=True)
		temporaryFile = auxiliaryNeuronsSimilarWordsDataset3File + auxiliaryNeuronsSimilarWordsDataset3TempFileSuffix
		rowCount = 0
		with open(temporaryFile, "w", encoding="utf-8") as fileObject:
			for sourceWord in sorted(candidateMap.keys()):
				records = calculateSimilarWordsDataset3CompactRecords(sourceWord, candidateMap[sourceWord], wordVectors)
				fields = [sourceWord]
				for targetWord, activationWeight in records:
					fields.append(targetWord)
					fields.append(auxiliaryNeuronsSimilarWordsDataset3SimilarityFormat.format(activationWeight))
				fileObject.write(auxiliaryNeuronsSimilarWordsDataset3Delimiter.join(fields) + "\n")
				rowCount += 1
		validateSimilarWordsDataset3CompactRows(rowCount)
		os.replace(temporaryFile, auxiliaryNeuronsSimilarWordsDataset3File)
		print(auxiliaryNeuronsSimilarWordsDataset3GenerateFinishMessage + str(rowCount))
		return

	def buildSimilarWordsDataset3WordNetCandidateMap(wn):
		result = {}
		validateSimilarWordsDatasetWordNet(wn)
		for partOfSpeech in auxiliaryNeuronsSimilarWordsDataset3WordNetPOSList:
			for synset in wn.all_synsets(pos=partOfSpeech):
				wordList = buildSimilarWordsDataset3WordNetSynsetWordList(synset)
				addSimilarWordsDataset3WordNetCandidateWords(result, wordList)
		validateSimilarWordsDataset3CompactRows(len(result))
		return result

	def buildSimilarWordsDataset3WordNetSynsetWordList(synset):
		result = []
		wordSet = set()
		for lemmaName in synset.lemma_names():
			word = normaliseSimilarityDatasetWordNetLemma(lemmaName)
			if(word not in wordSet):
				wordSet.add(word)
				result.append(word)
		return result

	def addSimilarWordsDataset3WordNetCandidateWords(candidateMap, wordList):
		for sourceWord in wordList:
			if(sourceWord not in candidateMap):
				candidateMap[sourceWord] = set()
			for targetWord in wordList:
				if(targetWord != sourceWord):
					candidateMap[sourceWord].add(targetWord)
		return

	def loadSimilarWordsDataset3SourceVectors(requiredWords):
		ensureSimilarWordsDataset3SourceFile()
		result = {}
		sourceWordsByNormalisedWord = {}
		expectedVectorLength = None
		with open(auxiliaryNeuronsSimilarWordsDataset3SourceFile, "r", encoding="utf-8") as fileObject:
			for lineIndex, line in enumerate(fileObject):
				strippedLine = line.strip()
				if(strippedLine != auxiliaryNeuronsSimilarWordsFeatureValueEmpty and not strippedLine.startswith(auxiliaryNeuronsSimilarWordsDataset3CommentPrefix)):
					fields = strippedLine.split()
					if(len(fields) < auxiliaryNeuronsSimilarWordsDataset3SourceMinimumFields):
						raise RuntimeError("loadSimilarWordsDataset3SourceVectors error: source row has insufficient fields")
					if(expectedVectorLength is None and len(fields) == auxiliaryNeuronsSimilarWordsDataset3SourceHeaderFieldCount and fields[0].isdigit() and fields[1].isdigit()):
						expectedVectorLength = int(fields[1])
					else:
						sourceWord = fields[auxiliaryNeuronsSimilarWordsDataset3SourceWordFieldIndex]
						word = normaliseSimilarityDatasetWordNetLemma(sourceWord)
						vectorFields = fields[auxiliaryNeuronsSimilarWordsDataset3SourceVectorStartFieldIndex:]
						if(expectedVectorLength is None):
							expectedVectorLength = len(vectorFields)
						if(len(vectorFields) != expectedVectorLength):
							raise RuntimeError("loadSimilarWordsDataset3SourceVectors error: source vector length mismatch")
						if(word in requiredWords):
							storeSimilarWordsDataset3SourceVector(result, sourceWordsByNormalisedWord, word, sourceWord, vectorFields)
		validateSimilarWordsDataset3SourceVectors(result, expectedVectorLength)
		return result

	def storeSimilarWordsDataset3SourceVector(wordVectors, sourceWordsByNormalisedWord, normalisedWord, sourceWord, vectorFields):
		if(normalisedWord not in wordVectors or isSimilarWordsDataset3SourceWordPreferred(sourceWord, sourceWordsByNormalisedWord[normalisedWord], normalisedWord)):
			wordVectors[normalisedWord] = normaliseSimilarWordsDataset3SourceVector([float(field) for field in vectorFields])
			sourceWordsByNormalisedWord[normalisedWord] = str(sourceWord)
		return

	def isSimilarWordsDataset3SourceWordPreferred(sourceWord, currentSourceWord, normalisedWord):
		result = False
		if(auxiliaryNeuronsSimilarWordsDataset3PreferExactSourceWord):
			sourceWordExact = str(sourceWord) == normalisedWord
			currentSourceWordExact = str(currentSourceWord) == normalisedWord
			if(sourceWordExact and not currentSourceWordExact):
				result = True
		return result

	def ensureSimilarWordsDataset3SourceFile():
		if(not GIAANNcmn_databaseNetworkFiles.pathExists(auxiliaryNeuronsSimilarWordsDataset3SourceFile)):
			if(auxiliaryNeuronsSimilarWordsDataset3SourceDownload):
				downloadSimilarWordsDataset3SourceFile()
			if(not GIAANNcmn_databaseNetworkFiles.pathExists(auxiliaryNeuronsSimilarWordsDataset3SourceFile)):
				raise RuntimeError("loadSimilarWordsDatasetWord2VecText error: missing auxiliaryNeuronsSimilarWordsDataset3SourceFile = " + auxiliaryNeuronsSimilarWordsDataset3SourceFile)
		return

	def downloadSimilarWordsDataset3SourceFile():
		import urllib.request
		os.makedirs(auxiliaryNeuronsSimilarWordsDatasetFolder, exist_ok=True)
		try:
			if(not GIAANNcmn_databaseNetworkFiles.pathExists(auxiliaryNeuronsSimilarWordsDataset3SourceDownloadArchiveFile)):
				urllib.request.urlretrieve(auxiliaryNeuronsSimilarWordsDataset3SourceDownloadURL, auxiliaryNeuronsSimilarWordsDataset3SourceDownloadArchiveFile)
			extractSimilarWordsDataset3SourceFile()
		except Exception as exception:
			raise RuntimeError("downloadSimilarWordsDataset3SourceFile error: failed to download source embeddings") from exception
		return

	def extractSimilarWordsDataset3SourceFile():
		import zipfile
		with zipfile.ZipFile(auxiliaryNeuronsSimilarWordsDataset3SourceDownloadArchiveFile, "r") as archiveObject:
			if(auxiliaryNeuronsSimilarWordsDataset3SourceDownloadArchiveMemberName not in archiveObject.namelist()):
				raise RuntimeError("extractSimilarWordsDataset3SourceFile error: missing archive member = " + auxiliaryNeuronsSimilarWordsDataset3SourceDownloadArchiveMemberName)
			archiveObject.extract(auxiliaryNeuronsSimilarWordsDataset3SourceDownloadArchiveMemberName, auxiliaryNeuronsSimilarWordsDatasetFolder)
		extractedFile = os.path.join(auxiliaryNeuronsSimilarWordsDatasetFolder, auxiliaryNeuronsSimilarWordsDataset3SourceDownloadArchiveMemberName)
		os.replace(extractedFile, auxiliaryNeuronsSimilarWordsDataset3SourceFile)
		return

	def normaliseSimilarWordsDataset3SourceVector(vectorValues):
		vector = pt.tensor(vectorValues, dtype=arrayType, device=pt.device("cpu"))
		vectorNorm = pt.linalg.vector_norm(vector)
		if(float(vectorNorm.item()) <= auxiliaryNeuronsSimilarWordsDataset3Epsilon):
			raise RuntimeError("normaliseSimilarWordsDataset3SourceVector error: zero source vector detected")
		result = vector/vectorNorm
		return result

	def calculateSimilarWordsDataset3CompactRecords(sourceWord, candidateWords, wordVectors):
		result = []
		sourceVector = wordVectors.get(sourceWord)
		if(sourceVector is not None):
			for targetWord in candidateWords:
				targetVector = wordVectors.get(targetWord)
				if(targetVector is not None):
					activationWeight = float(pt.dot(sourceVector, targetVector).item())
					if(activationWeight >= auxiliaryNeuronsSimilarWordsThreshold):
						result.append((targetWord, normaliseSimilarityWeight(activationWeight)))
		result.sort(key=lambda item: (-item[1], item[0]))
		if(len(result) > auxiliaryNeuronsSimilarWordsDataset3maxNumberSimilarWords):
			result = result[:auxiliaryNeuronsSimilarWordsDataset3maxNumberSimilarWords]
		return result

	def validateSimilarWordsDataset3CompactRows(rowCount):
		if(rowCount < auxiliaryNeuronsSimilarWordsDataset3MinimumWords):
			raise RuntimeError("loadSimilarWordsDatasetWord2VecText error: dataset3 is below production minimum words" + auxiliaryNeuronsSimilarWordsDatasetMinimumErrorActualPrefix + str(rowCount) + auxiliaryNeuronsSimilarWordsDatasetMinimumErrorMinimumPrefix + str(auxiliaryNeuronsSimilarWordsDataset3MinimumWords))
		return

	def validateSimilarWordsDataset3SourceVectors(wordVectors, vectorLength):
		if(len(wordVectors) < auxiliaryNeuronsSimilarWordsDataset3MinimumWords):
			raise RuntimeError("loadSimilarWordsDatasetWord2VecText error: dataset3 is below production minimum words" + auxiliaryNeuronsSimilarWordsDatasetMinimumErrorActualPrefix + str(len(wordVectors)) + auxiliaryNeuronsSimilarWordsDatasetMinimumErrorMinimumPrefix + str(auxiliaryNeuronsSimilarWordsDataset3MinimumWords))
		if(vectorLength < auxiliaryNeuronsSimilarWordsDataset3MinimumVectorLength):
			raise RuntimeError("loadSimilarWordsDatasetWord2VecText error: dataset3 source vectors are below production minimum vector length" + auxiliaryNeuronsSimilarWordsDatasetMinimumErrorActualPrefix + str(vectorLength) + auxiliaryNeuronsSimilarWordsDatasetMinimumErrorMinimumPrefix + str(auxiliaryNeuronsSimilarWordsDataset3MinimumVectorLength))
		return

	def normaliseSimilarityDatasetWordNetLemma(lemmaName):
		result = normaliseSimilarityWord(str(lemmaName).replace("_", " "))
		return result

	def buildTokenSequenceConceptIndexList(tokens, conceptIndices, startIndices, endIndices):
		result = [None]*len(tokens)
		conceptIndicesList = conceptIndices.tolist()
		startList = startIndices.tolist()
		endList = endIndices.tolist()
		for sequenceConceptIndex, conceptWordIndex in enumerate(conceptIndicesList):
			startIndex = max(0, int(startList[sequenceConceptIndex]))
			endIndex = min(len(tokens), int(endList[sequenceConceptIndex]))
			for tokenIndex in range(startIndex, endIndex):
				result[tokenIndex] = sequenceConceptIndex
		return result

	def initialiseAuxiliaryFeatureConnections(databaseNetworkObject, targetDevice):
		indices = pt.empty((5, 0), dtype=pt.long, device=targetDevice)
		values = pt.empty((0,), dtype=arrayType, device=targetDevice)
		result = pt.sparse_coo_tensor(indices, values, size=(databaseNetworkObject.arrayNumberOfProperties, numberOfDendriticBranches, arrayNumberOfSegments, databaseNetworkObject.c, databaseNetworkObject.f), dtype=arrayType, device=targetDevice)
		return result

	def getObservedColumnAuxiliaryFeatureConnectionsForSourceFeature(observedColumn, sourceFeatureIndex, targetDevice=None, createMissing=False, ensureCurrentSizeOnLoad=False):
		ensureObservedColumnAuxiliaryStorage(observedColumn)
		normalisedSourceFeatureIndex = normaliseAuxiliarySourceFeatureIndex(observedColumn.databaseNetworkObject, sourceFeatureIndex)
		resolvedTargetDevice = targetDevice if targetDevice is not None else deviceSparse
		result = observedColumn.similarAuxiliaryFeatureConnectionsBySourceFeature.get(normalisedSourceFeatureIndex)
		if(result is None):
			if(hasattr(observedColumn, "sourceObservedColumn")):
				sourceTensor = getObservedColumnAuxiliaryFeatureConnectionsForSourceFeature(observedColumn.sourceObservedColumn, normalisedSourceFeatureIndex, observedColumn.sourceObservedColumn.getDefaultConnectionTargetDevice(), createMissing, ensureCurrentSizeOnLoad)
				result = sourceTensor.to(resolvedTargetDevice)
			elif(storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam and observedColumn.databaseNetworkObject.observedColumnsRAMLoaded):
				result = initialiseAuxiliaryFeatureConnections(observedColumn.databaseNetworkObject, resolvedTargetDevice)
			else:
				storedSourceFeatureIndices = listObservedColumnAuxiliarySourceFeatureIndices(observedColumn.databaseNetworkObject, observedColumn.conceptIndex)
				if(normalisedSourceFeatureIndex in storedSourceFeatureIndices):
					result = loadObservedColumnAuxiliaryFeatureConnectionsTensor(observedColumn.databaseNetworkObject, observedColumn.conceptIndex, normalisedSourceFeatureIndex, resolvedTargetDevice, ensureCurrentSizeOnLoad=ensureCurrentSizeOnLoad)
				else:
					result = initialiseAuxiliaryFeatureConnections(observedColumn.databaseNetworkObject, resolvedTargetDevice)
			observedColumn.similarAuxiliaryFeatureConnectionsBySourceFeature[normalisedSourceFeatureIndex] = result
		elif(result.device != resolvedTargetDevice):
			result = result.to(resolvedTargetDevice)
			observedColumn.similarAuxiliaryFeatureConnectionsBySourceFeature[normalisedSourceFeatureIndex] = result
		if(ensureCurrentSizeOnLoad):
			ensureObservedColumnAuxiliaryFeatureConnectionSize(observedColumn, normalisedSourceFeatureIndex)
			result = observedColumn.similarAuxiliaryFeatureConnectionsBySourceFeature[normalisedSourceFeatureIndex]
			if(result.device != resolvedTargetDevice):
				result = result.to(resolvedTargetDevice)
				observedColumn.similarAuxiliaryFeatureConnectionsBySourceFeature[normalisedSourceFeatureIndex] = result
		observedColumn.similarAuxiliaryLoadedSourceFeatureIndices.add(normalisedSourceFeatureIndex)
		return result

	def setObservedColumnAuxiliaryFeatureConnectionsForSourceFeature(observedColumn, sourceFeatureIndex, tensor):
		ensureObservedColumnAuxiliaryStorage(observedColumn)
		normalisedSourceFeatureIndex = normaliseAuxiliarySourceFeatureIndex(observedColumn.databaseNetworkObject, sourceFeatureIndex)
		expectedSize = getObservedColumnAuxiliaryFeatureConnectionsTargetSize(observedColumn)
		if(tensor is None):
			raise RuntimeError("setObservedColumnAuxiliaryFeatureConnectionsForSourceFeatureSimilarity error: tensor is None")
		if(tensor.layout != pt.sparse_coo):
			raise RuntimeError("setObservedColumnAuxiliaryFeatureConnectionsForSourceFeatureSimilarity error: tensor must be sparse COO")
		if(tensor.dim() != 5):
			raise RuntimeError("setObservedColumnAuxiliaryFeatureConnectionsForSourceFeatureSimilarity error: tensor rank must be 5")
		if(tuple(tensor.size()) != tuple(expectedSize)):
			raise RuntimeError("setObservedColumnAuxiliaryFeatureConnectionsForSourceFeatureSimilarity error: tensor size mismatch")
		if(not tensor.is_coalesced()):
			tensor = tensor.coalesce()
		observedColumn.similarAuxiliaryFeatureConnectionsBySourceFeature[normalisedSourceFeatureIndex] = tensor
		observedColumn.similarAuxiliaryLoadedSourceFeatureIndices.add(normalisedSourceFeatureIndex)
		invalidateObservedColumnAuxiliaryMaterialisedConnections(observedColumn)
		return

	def prepareObservedColumnAuxiliaryFeatureConnectionsTrain(observedColumn, requiredSourceFeatureIndices, targetDevice):
		ensureObservedColumnAuxiliaryStorage(observedColumn)
		resolvedSourceFeatureIndices = normaliseAuxiliarySourceFeatureIndices(observedColumn.databaseNetworkObject, requiredSourceFeatureIndices)
		for sourceFeatureIndex in resolvedSourceFeatureIndices:
			getObservedColumnAuxiliaryFeatureConnectionsForSourceFeature(observedColumn, sourceFeatureIndex, targetDevice=targetDevice, createMissing=False)
			ensureObservedColumnAuxiliaryFeatureConnectionSize(observedColumn, sourceFeatureIndex)
		observedColumn.similarAuxiliaryTrainPreparedSourceFeatureIndices = set(resolvedSourceFeatureIndices)
		return

	def ensureObservedColumnAuxiliaryFeatureConnectionSize(observedColumn, sourceFeatureIndex):
		normalisedSourceFeatureIndex = normaliseAuxiliarySourceFeatureIndex(observedColumn.databaseNetworkObject, sourceFeatureIndex)
		if(normalisedSourceFeatureIndex not in observedColumn.similarAuxiliaryFeatureConnectionsBySourceFeature):
			raise RuntimeError("ensureObservedColumnAuxiliaryFeatureConnectionSizeSimilarity error: missing loaded source feature tensor")
		sourceTensor = observedColumn.similarAuxiliaryFeatureConnectionsBySourceFeature[normalisedSourceFeatureIndex]
		expectedSize = getObservedColumnAuxiliaryFeatureConnectionsTargetSize(observedColumn)
		if(tuple(sourceTensor.size()) != tuple(expectedSize)):
			sourceTensor = GIAANNcmn_databaseNetworkFiles.expandSparseTensorSize(sourceTensor, expectedSize, "ensureObservedColumnAuxiliaryFeatureConnectionSizeSimilarity")
			observedColumn.similarAuxiliaryFeatureConnectionsBySourceFeature[normalisedSourceFeatureIndex] = sourceTensor
		return

	def getObservedColumnAuxiliaryFeatureConnectionsTargetSize(observedColumn):
		result = (observedColumn.databaseNetworkObject.arrayNumberOfProperties, numberOfDendriticBranches, arrayNumberOfSegments, observedColumn.databaseNetworkObject.c, observedColumn.databaseNetworkObject.f)
		return result

	def getObservedColumnAuxiliaryFeatureConnectionsFolder(conceptIndex):
		result = os.path.join(GIAANNcmn_databaseNetworkFiles.getObservedColumnFolder(conceptIndex), auxiliaryNeuronsSimilarWordsConnectionsFolderName)
		return result

	def getObservedColumnAuxiliarySourceFeatureConnectionsFileBaseName(sourceFeatureIndex):
		result = auxiliaryNeuronsSimilarWordsSourceFeatureConnectionsFileNamePrefix + str(int(sourceFeatureIndex))
		return result

	def listObservedColumnAuxiliarySourceFeatureIndices(databaseNetworkObject, conceptIndex):
		result = []
		connectionsFolder = getObservedColumnAuxiliaryFeatureConnectionsFolder(conceptIndex)
		if(os.path.isdir(connectionsFolder)):
			for fileName in os.listdir(connectionsFolder):
				if(fileName.startswith(auxiliaryNeuronsSimilarWordsSourceFeatureConnectionsFileNamePrefix) and fileName.endswith(pytorchTensorFileExtension)):
					filePath = os.path.join(connectionsFolder, fileName)
					if(databaseNetworkObject.auxiliaryNeuronsSimilarWordsLoadExistingDatabase or filePath in auxiliaryNeuronsSimilarWordsSavedSourceTensorPaths):
						sourceFeatureIndexString = fileName[len(auxiliaryNeuronsSimilarWordsSourceFeatureConnectionsFileNamePrefix):-len(pytorchTensorFileExtension)]
						result.append(int(sourceFeatureIndexString))
		result.sort()
		return result

	def loadObservedColumnAuxiliaryFeatureConnectionsTensor(databaseNetworkObject, conceptIndex, sourceFeatureIndex, targetDevice, ensureCurrentSizeOnLoad=False):
		connectionsFolder = getObservedColumnAuxiliaryFeatureConnectionsFolder(conceptIndex)
		fileBaseName = getObservedColumnAuxiliarySourceFeatureConnectionsFileBaseName(sourceFeatureIndex)
		tensorName = "observedColumn.similarAuxiliaryFeatureConnectionsBySourceFeature[" + str(conceptIndex) + "][" + str(sourceFeatureIndex) + "]"
		tensor = GIAANNcmn_databaseNetworkFiles.adjustPropertyDimensions(databaseNetworkObject.inferenceMode, GIAANNcmn_databaseNetworkFiles.loadTensor(connectionsFolder, fileBaseName, targetDevice=targetDevice), tensorName)
		tensor = GIAANNcmn_databaseNetworkFiles.adjustBranchDimensions(tensor, tensorName, expectedRank=5)
		if(ensureCurrentSizeOnLoad):
			tensor = GIAANNcmn_databaseNetworkFiles.ensureFeatureConnectionsSourceTensorCurrentSize(tensor, databaseNetworkObject.c, databaseNetworkObject.f, tensorName)
		return tensor

	def saveObservedColumnAuxiliaryFeatureConnectionsTensor(conceptIndex, sourceFeatureIndex, tensor):
		connectionsFolder = getObservedColumnAuxiliaryFeatureConnectionsFolder(conceptIndex)
		fileBaseName = getObservedColumnAuxiliarySourceFeatureConnectionsFileBaseName(sourceFeatureIndex)
		filePath = os.path.join(connectionsFolder, fileBaseName + pytorchTensorFileExtension)
		if(tensor is None):
			raise RuntimeError("saveObservedColumnAuxiliaryFeatureConnectionsTensorSimilarity error: tensor is None")
		os.makedirs(connectionsFolder, exist_ok=True)
		if(tensor.is_sparse):
			tensor = tensor.coalesce()
			tensorNNZ = tensor._nnz()
		else:
			tensorNNZ = int(pt.count_nonzero(tensor).item())
		if(tensorNNZ > 0):
			GIAANNcmn_databaseNetworkFiles.saveTensor(tensor, connectionsFolder, fileBaseName)
			auxiliaryNeuronsSimilarWordsSavedSourceTensorPaths.add(filePath)
		else:
			if(GIAANNcmn_databaseNetworkFiles.pathExists(filePath)):
				os.remove(filePath)
			auxiliaryNeuronsSimilarWordsSavedSourceTensorPaths.discard(filePath)
		return

	def ensureObservedColumnAuxiliaryStorage(observedColumn):
		if(not hasattr(observedColumn, "similarAuxiliaryFeatureConnectionsBySourceFeature")):
			initialiseObservedColumnAuxiliaryStorage(observedColumn)
		else:
			if(not hasattr(observedColumn, "similarAuxiliarySecondaryInputConnections")):
				observedColumn.similarAuxiliarySecondaryInputConnections = None
			if(not hasattr(observedColumn, "similarAuxiliarySecondaryOutputConnectionsMaterialised")):
				observedColumn.similarAuxiliarySecondaryOutputConnectionsMaterialised = None
		return

	def invalidateObservedColumnAuxiliaryMaterialisedConnections(observedColumn):
		ensureObservedColumnAuxiliaryStorage(observedColumn)
		observedColumn.similarAuxiliarySecondaryOutputConnectionsMaterialised = None
		databaseNetworkObject = observedColumn.databaseNetworkObject
		if(hasattr(databaseNetworkObject, "auxiliaryNeuronsSimilarWordsPrimeOutputConnectionsMaterialised")):
			databaseNetworkObject.auxiliaryNeuronsSimilarWordsPrimeOutputConnectionsMaterialised = None
		return

	def invalidateDatabaseAuxiliaryInputConnectionCaches(databaseNetworkObject):
		if(hasattr(databaseNetworkObject, "auxiliaryNeuronsSimilarWordsPrimeInputConnections")):
			databaseNetworkObject.auxiliaryNeuronsSimilarWordsPrimeInputConnections = None
		if(hasattr(databaseNetworkObject, "auxiliaryNeuronsSimilarWordsSecondaryInputConnectionsByConceptIndex")):
			databaseNetworkObject.auxiliaryNeuronsSimilarWordsSecondaryInputConnectionsByConceptIndex = {}
		return

	def normaliseAuxiliarySourceFeatureIndex(databaseNetworkObject, sourceFeatureIndex):
		result = int(sourceFeatureIndex)
		if(result < 0 or result >= databaseNetworkObject.fas):
			raise RuntimeError("normaliseAuxiliarySourceFeatureIndexSimilarity error: source feature index out of range")
		return result

	def normaliseAuxiliarySourceFeatureIndices(databaseNetworkObject, sourceFeatureIndices):
		result = []
		seen = set()
		if(sourceFeatureIndices is not None):
			if(pt.is_tensor(sourceFeatureIndices)):
				rawSourceFeatureIndices = sourceFeatureIndices.detach().view(-1).cpu().tolist()
			else:
				rawSourceFeatureIndices = list(sourceFeatureIndices)
			for sourceFeatureIndex in rawSourceFeatureIndices:
				normalisedSourceFeatureIndex = normaliseAuxiliarySourceFeatureIndex(databaseNetworkObject, sourceFeatureIndex)
				if(normalisedSourceFeatureIndex not in seen):
					result.append(normalisedSourceFeatureIndex)
					seen.add(normalisedSourceFeatureIndex)
		result.sort()
		return result

	def appendAuxiliaryConnectionSegmentIndices(indicesList, valuesList, branchIndices, sourceConceptIndices, sourceAuxiliaryFeatureIndices, targetConceptIndices, targetFeatureIndices, sourceWordOrder, targetWordOrder, baseValues):
		if(sourceConceptIndices.numel() > 0):
			if(useSANIcolumns):
				conceptDistances = pt.abs(targetConceptIndices - sourceConceptIndices)
				connectionsSegmentIndex = arrayNumberOfSegments - conceptDistances - 1
				connectionsSegmentIndex = pt.clamp(connectionsSegmentIndex, min=0, max=arrayNumberOfSegments-1).long()
				appendAuxiliaryConnectionIndexGroup(indicesList, valuesList, branchIndices, connectionsSegmentIndex, sourceConceptIndices, sourceAuxiliaryFeatureIndices, targetConceptIndices, targetFeatureIndices, baseValues)
			elif(useSANIfeatures):
				relativeDistance = targetWordOrder - sourceWordOrder
				if(SANIfeaturesLinkFirstSegmentToAllPriorTrainSeqTokens):
					relativeDistance = pt.clamp(relativeDistance, min=1)
					connectionsSegmentIndex = arrayNumberOfSegments - relativeDistance
					connectionsSegmentIndex = connectionsSegmentIndex.clamp(min=0, max=arrayNumberOfSegments-1).long()
					appendAuxiliaryConnectionIndexGroup(indicesList, valuesList, branchIndices, connectionsSegmentIndex, sourceConceptIndices, sourceAuxiliaryFeatureIndices, targetConceptIndices, targetFeatureIndices, baseValues)
				else:
					relativeDistance = pt.clamp(relativeDistance, min=1)
					validDistanceMask = relativeDistance <= arrayNumberOfSegments
					if(validDistanceMask.any()):
						connectionsSegmentIndex = arrayNumberOfSegments - relativeDistance
						connectionsSegmentIndex = connectionsSegmentIndex.clamp(min=0, max=arrayNumberOfSegments-1).long()
						appendAuxiliaryConnectionIndexGroup(indicesList, valuesList, branchIndices[validDistanceMask], connectionsSegmentIndex[validDistanceMask], sourceConceptIndices[validDistanceMask], sourceAuxiliaryFeatureIndices[validDistanceMask], targetConceptIndices[validDistanceMask], targetFeatureIndices[validDistanceMask], baseValues[validDistanceMask])
			elif(useSANIfeaturesAndColumns):
				appendAuxiliaryFeatureAndColumnSegmentIndices(indicesList, valuesList, branchIndices, sourceConceptIndices, sourceAuxiliaryFeatureIndices, targetConceptIndices, targetFeatureIndices, sourceWordOrder, targetWordOrder, baseValues)
			else:
				connectionsSegmentIndex = pt.zeros((branchIndices.shape[0],), dtype=pt.long, device=branchIndices.device)
				appendAuxiliaryConnectionIndexGroup(indicesList, valuesList, branchIndices, connectionsSegmentIndex, sourceConceptIndices, sourceAuxiliaryFeatureIndices, targetConceptIndices, targetFeatureIndices, baseValues)
		return

	def appendAuxiliaryFeatureAndColumnSegmentIndices(indicesList, valuesList, branchIndices, sourceConceptIndices, sourceAuxiliaryFeatureIndices, targetConceptIndices, targetFeatureIndices, sourceWordOrder, targetWordOrder, baseValues):
		relativeDistance = targetWordOrder - sourceWordOrder
		featureSegmentsOffset = arrayNumberOfSegmentsColumnDistance
		if(SANIfeaturesLinkFirstSegmentToAllPriorTrainSeqTokens):
			relativeDistanceFeature = pt.clamp(relativeDistance, min=1, max=arrayNumberOfSegmentsFeatureDistance)
			featureSegmentIndex = featureSegmentsOffset + arrayNumberOfSegmentsFeatureDistance - relativeDistanceFeature
			featureSegmentIndex = featureSegmentIndex.clamp(min=featureSegmentsOffset, max=arrayNumberOfSegments-1).long()
			appendAuxiliaryConnectionIndexGroup(indicesList, valuesList, branchIndices, featureSegmentIndex, sourceConceptIndices, sourceAuxiliaryFeatureIndices, targetConceptIndices, targetFeatureIndices, baseValues)
		else:
			relativeDistanceFeature = pt.clamp(relativeDistance, min=1)
			validFeatureDistanceMask = relativeDistanceFeature <= arrayNumberOfSegmentsFeatureDistance
			if(validFeatureDistanceMask.any()):
				featureSegmentIndex = featureSegmentsOffset + arrayNumberOfSegmentsFeatureDistance - relativeDistanceFeature
				featureSegmentIndex = featureSegmentIndex.clamp(min=featureSegmentsOffset, max=arrayNumberOfSegments-1).long()
				appendAuxiliaryConnectionIndexGroup(indicesList, valuesList, branchIndices[validFeatureDistanceMask], featureSegmentIndex[validFeatureDistanceMask], sourceConceptIndices[validFeatureDistanceMask], sourceAuxiliaryFeatureIndices[validFeatureDistanceMask], targetConceptIndices[validFeatureDistanceMask], targetFeatureIndices[validFeatureDistanceMask], baseValues[validFeatureDistanceMask])
		if(arrayNumberOfSegmentsColumnDistance > 0):
			conceptDistances = pt.abs(targetConceptIndices - sourceConceptIndices)
			if(useSANIfeaturesAndColumnsInternal):
				columnSegmentIndex = arrayNumberOfSegmentsColumnDistance - conceptDistances - 1
			else:
				columnSegmentIndex = arrayNumberOfSegmentsColumnDistance - conceptDistances
			columnSegmentIndex = columnSegmentIndex.clamp(min=0, max=arrayNumberOfSegmentsColumnDistance-1).long()
			validColumnMask = pt.ones((branchIndices.shape[0],), dtype=pt.bool, device=branchIndices.device)
			if(not useSANIfeaturesAndColumnsInternal):
				validColumnMask = conceptDistances > 0
			if(validColumnMask.any()):
				appendAuxiliaryConnectionIndexGroup(indicesList, valuesList, branchIndices[validColumnMask], columnSegmentIndex[validColumnMask], sourceConceptIndices[validColumnMask], sourceAuxiliaryFeatureIndices[validColumnMask], targetConceptIndices[validColumnMask], targetFeatureIndices[validColumnMask], baseValues[validColumnMask])
		return

	def appendAuxiliaryConnectionIndexGroup(indicesList, valuesList, branchIndices, segmentIndices, sourceConceptIndices, sourceAuxiliaryFeatureIndices, targetConceptIndices, targetFeatureIndices, values):
		indicesList.append(pt.stack((branchIndices, segmentIndices, sourceConceptIndices, sourceAuxiliaryFeatureIndices, targetConceptIndices, targetFeatureIndices), dim=0))
		valuesList.append(values)
		return

	def createAuxiliaryFeatureWordOrderConnectionMask(sourceWordOrder, targetWordOrder, trainConnectionsIncludeSameTimeIndex):
		if(not isinstance(trainConnectionsIncludeSameTimeIndex, bool)):
			raise RuntimeError("createAuxiliaryFeatureWordOrderConnectionMaskSimilarity error: trainConnectionsIncludeSameTimeIndex must be a bool")
		if(debugConnectNodesToNextNodesInSequenceOnly):
			wordOrderUpperBound = sourceWordOrder + 1
			if(trainConnectionsIncludeSameTimeIndex):
				result = pt.logical_and(targetWordOrder >= sourceWordOrder, targetWordOrder <= wordOrderUpperBound)
			else:
				result = pt.logical_and(targetWordOrder > sourceWordOrder, targetWordOrder <= wordOrderUpperBound)
		else:
			if(trainConnectionsIncludeSameTimeIndex):
				result = targetWordOrder >= sourceWordOrder
			else:
				result = targetWordOrder > sourceWordOrder
		return result

	def addSparseUpdateNonNegative(targetSparse, updateSparse):
		result = (targetSparse.coalesce() + updateSparse.coalesce()).coalesce()
		if(result._nnz() > 0):
			resultValues = result.values()
			resultValues.clamp_(min=0)
		return result

	def buildAuxiliaryFeatureName(auxiliaryFeaturePrefix, auxiliaryFeatureValue):
		result = auxiliaryFeaturePrefix + auxiliaryNeuronsSimilarWordsFeatureNameDelimiter + auxiliaryFeatureValue
		return result

	def buildConceptColumnAuxiliaryFeatureName(databaseNetworkObject, auxiliaryFeaturePrefix, conceptIndex, auxiliaryFeatureValue):
		normalisedConceptIndex = normaliseAuxiliaryParentMapConceptIndex(databaseNetworkObject, conceptIndex)
		auxiliaryFeatureName = buildAuxiliaryFeatureName(auxiliaryFeaturePrefix, auxiliaryFeatureValue)
		auxiliaryFeatureNamePrefix = auxiliaryFeaturePrefix + auxiliaryNeuronsSimilarWordsFeatureNameDelimiter
		if(not auxiliaryFeatureName.startswith(auxiliaryFeatureNamePrefix)):
			raise RuntimeError("buildConceptColumnAuxiliaryFeatureNameSimilarity error: auxiliaryFeatureName missing prefix")
		result = auxiliaryFeatureNamePrefix + str(normalisedConceptIndex) + auxiliaryNeuronsSimilarWordsFeatureNameDelimiter + auxiliaryFeatureName[len(auxiliaryFeatureNamePrefix):]
		return result

	def parseConceptColumnAuxiliaryFeatureName(databaseNetworkObject, auxiliaryFeatureWord):
		parts = auxiliaryFeatureWord.split(auxiliaryNeuronsSimilarWordsFeatureNameDelimiter, auxiliaryNeuronsSimilarWordsScopedFeatureNameParts-1)
		if(len(parts) != auxiliaryNeuronsSimilarWordsScopedFeatureNameParts):
			raise RuntimeError("parseConceptColumnAuxiliaryFeatureName error: invalid auxiliary feature word")
		auxiliaryConceptIndex = normaliseAuxiliaryParentMapConceptIndex(databaseNetworkObject, parts[1])
		auxiliaryBaseWord = normaliseSimilarityWord(parts[2])
		result = auxiliaryConceptIndex, auxiliaryBaseWord
		return result

	def buildSimilarityParentKey(auxiliaryFeaturePrefix, parentWord):
		result = auxiliaryFeaturePrefix + auxiliaryNeuronsSimilarWordsFeatureNameDelimiter + normaliseSimilarityWord(parentWord)
		return result

	def parseSimilarityParentKey(parentKey):
		parts = parentKey.split(auxiliaryNeuronsSimilarWordsFeatureNameDelimiter, 1)
		if(len(parts) != 2):
			raise RuntimeError("parseSimilarityParentKey error: invalid parentKey")
		parentPrefix = parts[0]
		parentWord = normaliseSimilarityWord(parts[1])
		result = parentPrefix, parentWord
		return result

	def tokenHasPrimeConceptSimilarityWord(token):
		if(token.lemma is None):
			raise RuntimeError("tokenHasPrimeConceptSimilarityWord error: token lemma is None")
		result = str(token.lemma).strip() != auxiliaryNeuronsSimilarWordsFeatureValueEmpty
		return result

	def tokenHasSecondarySimilarityWord(token):
		if(token.word is None):
			raise RuntimeError("tokenHasSecondarySimilarityWord error: token word is None")
		result = str(token.word).strip() != auxiliaryNeuronsSimilarWordsFeatureValueEmpty
		return result

	def getTokenPrimeConceptSimilarityWord(token):
		if(not tokenHasPrimeConceptSimilarityWord(token)):
			raise RuntimeError("getTokenPrimeConceptSimilarityWord error: token lemma is empty")
		result = normaliseSimilarityWord(token.lemma)
		return result

	def getTokenSecondarySimilarityWord(token):
		if(not tokenHasSecondarySimilarityWord(token)):
			raise RuntimeError("getTokenSecondarySimilarityWord error: token word is empty")
		result = normaliseSimilarityWord(token.word)
		return result

	def normaliseSimilarityWord(word):
		if(word is None):
			raise RuntimeError("normaliseSimilarityWord error: word is None")
		result = str(word).strip().lower()
		if(result == auxiliaryNeuronsSimilarWordsFeatureValueEmpty):
			raise RuntimeError("normaliseSimilarityWord error: word is empty")
		return result

	def normaliseSimilarityWeight(activationWeight):
		result = float(activationWeight)
		if(result < auxiliaryNeuronsSimilarWordsMinimumSimilarity or result > auxiliaryNeuronsSimilarWordsMaximumSimilarity):
			raise RuntimeError("normaliseSimilarityWeight error: activationWeight out of range")
		return result

	def normaliseAuxiliaryParentMapConceptIndex(databaseNetworkObject, conceptIndex):
		if(isinstance(conceptIndex, bool)):
			raise RuntimeError("normaliseAuxiliaryParentMapConceptIndexSimilarity error: conceptIndex must not be bool")
		try:
			result = int(conceptIndex)
		except (TypeError, ValueError):
			raise RuntimeError("normaliseAuxiliaryParentMapConceptIndexSimilarity error: conceptIndex must be int")
		if(result < 0 or result >= databaseNetworkObject.c):
			raise RuntimeError("normaliseAuxiliaryParentMapConceptIndexSimilarity error: conceptIndex out of range")
		return result

	def buildIndexListFromIndexDict(indexDict, mapName):
		result = [None]*len(indexDict)
		for key, index in indexDict.items():
			if(not isinstance(index, int) or isinstance(index, bool)):
				raise RuntimeError("buildIndexListFromIndexDictSimilarity error: index must be int for " + mapName)
			if(index < 0 or index >= len(indexDict)):
				raise RuntimeError("buildIndexListFromIndexDictSimilarity error: index out of range for " + mapName)
			if(result[index] is not None):
				raise RuntimeError("buildIndexListFromIndexDictSimilarity error: duplicate index for " + mapName)
			result[index] = key
		for value in result:
			if(value is None):
				raise RuntimeError("buildIndexListFromIndexDictSimilarity error: missing index for " + mapName)
		return result

	def validateSimilarWordsConfiguration():
		if(not auxiliaryNeuronsSimilarWordsPrimeConceptFeatures and not auxiliaryNeuronsSimilarWordsSecondaryConceptFeatures and not auxiliaryNeuronsSimilarSubwordAuto):
			raise RuntimeError("validateSimilarWordsConfiguration error: no auxiliary similarity feature suboption is enabled")
		if(auxiliaryNeuronsSimilarWordsThreshold < auxiliaryNeuronsSimilarWordsMinimumSimilarity or auxiliaryNeuronsSimilarWordsThreshold > auxiliaryNeuronsSimilarWordsMaximumSimilarity):
			raise RuntimeError("validateSimilarWordsConfiguration error: auxiliaryNeuronsSimilarWordsThreshold out of range")
		if(auxiliaryNeuronsSimilarSubwordAuto):
			if(auxiliaryNeuronsSimilarSubwordAutoThreshold < auxiliaryNeuronsSimilarWordsMinimumSimilarity or auxiliaryNeuronsSimilarSubwordAutoThreshold > auxiliaryNeuronsSimilarWordsMaximumSimilarity):
				raise RuntimeError("validateSimilarWordsConfiguration error: auxiliaryNeuronsSimilarSubwordAutoThreshold out of range")
		if(auxiliaryNeuronsAuto and auxiliaryNeuronsSimilarWordsAuto and auxiliaryNeuronsSimilarWordsSecondaryConceptFeatures):
			if(auxiliaryNeuronsSimilarWordsSecondaryConceptFeaturesLimit):
				if(auxiliaryNeuronsSimilarWordsSecondaryConceptFeaturesMaximumSharedSourceFeatureIndex < 0.0 or auxiliaryNeuronsSimilarWordsSecondaryConceptFeaturesMaximumSharedSourceFeatureIndex > 1.0):
					raise RuntimeError("validateSimilarWordsConfiguration error: auxiliaryNeuronsSimilarWordsSecondaryConceptFeaturesMaximumSharedSourceFeatureIndex out of range")
		if(auxiliaryNeuronsSimilarWordsStatic):
			if(auxiliaryNeuronsSimilarWordsDatasetName not in (auxiliaryNeuronsSimilarWordsDataset1Name, auxiliaryNeuronsSimilarWordsDataset2Name, auxiliaryNeuronsSimilarWordsDataset3Name)):
				raise RuntimeError("validateSimilarWordsConfiguration error: unsupported auxiliaryNeuronsSimilarWordsDatasetName")
			if(auxiliaryNeuronsSimilarWordsDataset3 and auxiliaryNeuronsSimilarWordsDataset3maxNumberSimilarWords <= 0):
				raise RuntimeError("validateSimilarWordsConfiguration error: auxiliaryNeuronsSimilarWordsDataset3maxNumberSimilarWords must be positive")
		return
