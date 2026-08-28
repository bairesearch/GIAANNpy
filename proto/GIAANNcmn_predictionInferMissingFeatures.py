"""GIAANNcmn_predictionInferMissingFeatures.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 BAI Research Pty Ltd (bairesearch.com.au)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_prediction.py

# Description:
GIA ANN common prediction missing-feature inference

"""

import math
import torch as pt

from GIAANNcmn_globalDefs import *
import GIAANNcmn_databaseNetwork
import GIAANNcmn_inferenceDuringTrain
import GIAANNcmn_predictionActivate

def getTrainedMissingFeatureSourceColumnIndices(databaseNetworkObject, globalFeatureNeuronsStrength, sourceColumnIndex, sourceFeatureIndex):
	result = None
	if(inferenceInferMissingFeatures):
		if(databaseNetworkObject is None):
			raise RuntimeError("getTrainedMissingFeatureSourceColumnIndices error: databaseNetworkObject is None")
		if(globalFeatureNeuronsStrength is None):
			raise RuntimeError("getTrainedMissingFeatureSourceColumnIndices error: globalFeatureNeuronsStrength is None")
		if(globalFeatureNeuronsStrength.dim() != inferenceInferMissingFeaturesNeuronTensorRank):
			raise RuntimeError("getTrainedMissingFeatureSourceColumnIndices error: globalFeatureNeuronsStrength rank is invalid")
		sourceColumnIndex = int(sourceColumnIndex)
		sourceFeatureIndex = int(sourceFeatureIndex)
		if(sourceColumnIndex < arrayIndexSegmentFirst or sourceColumnIndex >= databaseNetworkObject.c):
			raise RuntimeError("getTrainedMissingFeatureSourceColumnIndices error: sourceColumnIndex is out of range")
		if(sourceFeatureIndex < arrayIndexSegmentFirst or sourceFeatureIndex >= databaseNetworkObject.f):
			raise RuntimeError("getTrainedMissingFeatureSourceColumnIndices error: sourceFeatureIndex is out of range")
		if(globalFeatureNeuronsStrength.shape[inferenceInferMissingFeaturesConceptDimension] < databaseNetworkObject.c or globalFeatureNeuronsStrength.shape[inferenceInferMissingFeaturesFeatureDimension] < databaseNetworkObject.f):
			raise RuntimeError("getTrainedMissingFeatureSourceColumnIndices error: globalFeatureNeuronsStrength is smaller than the database dimensions")
		if(globalFeatureNeuronsStrength.is_sparse):
			strengthSparse = globalFeatureNeuronsStrength.coalesce()
			strengthIndices = strengthSparse.indices()
			strengthValues = strengthSparse.values()
			featureMask = strengthIndices[inferenceInferMissingFeaturesFeatureDimension] == sourceFeatureIndex
			relevantStrengthValues = strengthValues[featureMask]
			if(relevantStrengthValues.numel() > arrayIndexSegmentFirst):
				if(not bool(pt.all(pt.isfinite(relevantStrengthValues)).item())):
					raise RuntimeError("getTrainedMissingFeatureSourceColumnIndices error: relevant neuron strengths must be finite")
				if(bool(pt.any(relevantStrengthValues < inferenceInferMissingFeaturesMinimumActivation).item())):
					raise RuntimeError("getTrainedMissingFeatureSourceColumnIndices error: relevant neuron strengths must be non-negative")
			trainedMask = featureMask & (strengthValues > inferenceInferMissingFeaturesMinimumActivation)
			trainedColumnIndices = strengthIndices[inferenceInferMissingFeaturesConceptDimension, trainedMask].unique(sorted=True)
		else:
			featureStrength = globalFeatureNeuronsStrength.select(inferenceInferMissingFeaturesFeatureDimension, sourceFeatureIndex)
			if(not bool(pt.all(pt.isfinite(featureStrength)).item())):
				raise RuntimeError("getTrainedMissingFeatureSourceColumnIndices error: relevant neuron strengths must be finite")
			if(bool(pt.any(featureStrength < inferenceInferMissingFeaturesMinimumActivation).item())):
				raise RuntimeError("getTrainedMissingFeatureSourceColumnIndices error: relevant neuron strengths must be non-negative")
			trainedColumnMask = (featureStrength > inferenceInferMissingFeaturesMinimumActivation).reshape(-1, featureStrength.shape[-1]).any(dim=arrayIndexSegmentFirst)
			trainedColumnIndices = pt.nonzero(trainedColumnMask, as_tuple=False).reshape(-1)
		currentFeatureIsTrained = bool(pt.any(trainedColumnIndices == sourceColumnIndex).item())
		if(currentFeatureIsTrained):
			result = None
		else:
			if(sourceFeatureIndex == featureIndexPrimeConceptNeuron):
				trainedColumnIndices = trainedColumnIndices[:arrayIndexSegmentFirst]
			result = trainedColumnIndices.to(dtype=pt.long, device=globalFeatureNeuronsStrength.device)
	else:
		raise RuntimeError("getTrainedMissingFeatureSourceColumnIndices error: inferenceInferMissingFeatures is disabled")
	return result

def processMissingFeaturePredictionActivations(databaseNetworkObject, observedColumnsDict, globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, sourceFeatureIndex, trainedSourceColumnIndices, globalFeatureNeuronsTime, activationSequenceWordIndex, activationSequenceColumnIndex, sequenceWordIndex):
	result = None
	if(inferenceInferMissingFeatures):
		activeSourceColumns, activeSourceActivations, totalSourceActivation = getActiveMissingFeatureSources(databaseNetworkObject, globalFeatureNeuronsActivation, sourceFeatureIndex, trainedSourceColumnIndices)
		if(len(activeSourceColumns) > arrayIndexSegmentFirst):
			combinedTargetActivation = calculateCombinedMissingFeatureTargetActivation(databaseNetworkObject, observedColumnsDict, globalFeatureNeuronsActivation, sourceFeatureIndex, activeSourceColumns, activeSourceActivations, totalSourceActivation, sequenceWordIndex)
			if(combinedTargetActivation is not None):
				globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, globalFeatureNeuronsTime = GIAANNcmn_predictionActivate.processFeatureNeuronsTargetActivationPredict(databaseNetworkObject, globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, combinedTargetActivation, globalFeatureNeuronsTime, activationSequenceWordIndex, activationSequenceColumnIndex)
		result = globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, globalFeatureNeuronsTime, activeSourceColumns
	else:
		raise RuntimeError("processMissingFeaturePredictionActivations error: inferenceInferMissingFeatures is disabled")
	return result

def processMissingFeaturePredictionActivationsEnforceLastSegment(databaseNetworkObject, observedColumnsDict, globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, sourceFeatureIndex, trainedSourceColumnIndices, somaActivationFromLastSegmentKeys, globalFeatureNeuronsTime, activationSequenceWordIndex, activationSequenceColumnIndex, sequenceWordIndex):
	result = None
	if(inferenceInferMissingFeatures):
		if(not inferenceLeakyIntegrateAndFire or algorithmMatrixSANIenforceRequirement != "enforceLastSegmentMustBeActive"):
			raise RuntimeError("processMissingFeaturePredictionActivationsEnforceLastSegment error: requires inferenceLeakyIntegrateAndFire enforceLastSegmentMustBeActive")
		activeSourceColumns, activeSourceActivations, totalSourceActivation = getActiveMissingFeatureSources(databaseNetworkObject, globalFeatureNeuronsActivation, sourceFeatureIndex, trainedSourceColumnIndices)
		if(len(activeSourceColumns) > arrayIndexSegmentFirst):
			combinedTargetActivation = calculateCombinedMissingFeatureTargetActivation(databaseNetworkObject, observedColumnsDict, globalFeatureNeuronsActivation, sourceFeatureIndex, activeSourceColumns, activeSourceActivations, totalSourceActivation, sequenceWordIndex)
			if(combinedTargetActivation is not None):
				combinedTargetActivation = GIAANNcmn_predictionActivate.transformFeatureNeuronsTargetActivationPredict(combinedTargetActivation)
				combinedTargetActivation, somaActivationFromLastSegmentKeys = GIAANNcmn_predictionActivate.mergeLeakyIntegrateAndFireCurrentSomaActivationKeys(somaActivationFromLastSegmentKeys, combinedTargetActivation)
				globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, globalFeatureNeuronsTime = GIAANNcmn_predictionActivate.applyFeatureNeuronsTargetActivationPredict(databaseNetworkObject, globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, combinedTargetActivation, globalFeatureNeuronsTime, activationSequenceWordIndex, activationSequenceColumnIndex)
		result = globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, globalFeatureNeuronsTime, somaActivationFromLastSegmentKeys, activeSourceColumns
	else:
		raise RuntimeError("processMissingFeaturePredictionActivationsEnforceLastSegment error: inferenceInferMissingFeatures is disabled")
	return result

def buildMissingFeatureSourceTensors(activeSourceColumns, sourceFeatureIndex, device):
	result = None
	if(inferenceInferMissingFeatures):
		if(activeSourceColumns is None or len(activeSourceColumns) == arrayIndexSegmentFirst):
			raise RuntimeError("buildMissingFeatureSourceTensors error: activeSourceColumns must not be empty")
		conceptColumnIndices = pt.tensor(activeSourceColumns, dtype=pt.long, device=device)
		conceptColumnFeatureIndices = pt.full_like(conceptColumnIndices.unsqueeze(inferenceInferMissingFeaturesCandidateFeatureDimension), int(sourceFeatureIndex))
		result = conceptColumnIndices, conceptColumnFeatureIndices
	else:
		raise RuntimeError("buildMissingFeatureSourceTensors error: inferenceInferMissingFeatures is disabled")
	return result

def getActiveMissingFeatureSources(databaseNetworkObject, globalFeatureNeuronsActivation, sourceFeatureIndex, trainedSourceColumnIndices):
	result = None
	if(inferenceInferMissingFeatures):
		if(trainedSourceColumnIndices is None):
			raise RuntimeError("getActiveMissingFeatureSources error: trainedSourceColumnIndices is None")
		if(inferenceLeakyIntegrateAndFire):
			activeSourceColumns, activeSourceActivations, totalSourceActivation = getActiveMissingFeatureSourcesLeakyIntegrateAndFire(globalFeatureNeuronsActivation, sourceFeatureIndex, trainedSourceColumnIndices)
		else:
			activeSourceColumns = []
			activeSourceActivations = []
			totalSourceActivation = pt.zeros((), dtype=globalFeatureNeuronsActivation.dtype, device=globalFeatureNeuronsActivation.device)
			for sourceColumnIndexTensor in trainedSourceColumnIndices:
				sourceColumnIndex = int(sourceColumnIndexTensor.item())
				sourceActivation = GIAANNcmn_predictionActivate.calculateFeatureNeuronSourceActivationPredict(databaseNetworkObject, globalFeatureNeuronsActivation, sourceColumnIndex, sourceFeatureIndex)
				if(not isinstance(sourceActivation, pt.Tensor)):
					raise RuntimeError("getActiveMissingFeatureSources error: source activation must be a tensor")
				if(sourceActivation.numel() == arrayIndexSegmentFirst):
					raise RuntimeError("getActiveMissingFeatureSources error: source activation must not be empty")
				if(not bool(pt.all(pt.isfinite(sourceActivation)).item())):
					raise RuntimeError("getActiveMissingFeatureSources error: source activation must be finite")
				if(bool(pt.any(sourceActivation < inferenceInferMissingFeaturesMinimumActivation).item())):
					raise RuntimeError("getActiveMissingFeatureSources error: source activation must be non-negative")
				sourceActivationTotal = sourceActivation.sum()
				if(bool((sourceActivationTotal > inferenceInferMissingFeaturesMinimumActivation).item())):
					activeSourceColumns.append(sourceColumnIndex)
					activeSourceActivations.append(sourceActivation)
					totalSourceActivation = totalSourceActivation + sourceActivationTotal
		if(not bool(pt.isfinite(totalSourceActivation).item())):
			raise RuntimeError("getActiveMissingFeatureSources error: combined source activation must be finite")
		if(len(activeSourceColumns) == arrayIndexSegmentFirst):
			if(bool((totalSourceActivation != inferenceInferMissingFeaturesMinimumActivation).item())):
				raise RuntimeError("getActiveMissingFeatureSources error: inactive combined source activation must be zero")
		elif(not bool((totalSourceActivation > inferenceInferMissingFeaturesMinimumActivation).item())):
			raise RuntimeError("getActiveMissingFeatureSources error: active combined source activation must be positive")
		result = activeSourceColumns, activeSourceActivations, totalSourceActivation
	else:
		raise RuntimeError("getActiveMissingFeatureSources error: inferenceInferMissingFeatures is disabled")
	return result

def getActiveMissingFeatureSourcesLeakyIntegrateAndFire(globalFeatureNeuronsActivation, sourceFeatureIndex, trainedSourceColumnIndices):
	result = None
	if(inferenceInferMissingFeatures):
		if(not inferenceLeakyIntegrateAndFire):
			raise RuntimeError("getActiveMissingFeatureSourcesLeakyIntegrateAndFire error: inferenceLeakyIntegrateAndFire is disabled")
		somaActivation = GIAANNcmn_predictionActivate.calculateLeakyIntegrateAndFireSomaActivation(globalFeatureNeuronsActivation).coalesce()
		somaIndices = somaActivation.indices()
		somaValues = somaActivation.values()
		featureMask = somaIndices[inferenceLeakyIntegrateAndFireSomaActivationFeatureDimension] == int(sourceFeatureIndex)
		relevantSomaValues = somaValues[featureMask]
		if(relevantSomaValues.numel() > arrayIndexSegmentFirst):
			if(not bool(pt.all(pt.isfinite(relevantSomaValues)).item())):
				raise RuntimeError("getActiveMissingFeatureSourcesLeakyIntegrateAndFire error: relevant soma activations must be finite")
			if(bool(pt.any(relevantSomaValues < inferenceInferMissingFeaturesMinimumActivation).item())):
				raise RuntimeError("getActiveMissingFeatureSourcesLeakyIntegrateAndFire error: relevant soma activations must be non-negative")
		activeMask = featureMask & (somaValues >= inferenceLeakyIntegrateAndFireSomaActivationThreshold)
		activeColumnIndices = somaIndices[inferenceLeakyIntegrateAndFireSomaActivationConceptDimension, activeMask]
		if(inferenceInferMissingFeaturesCandidateSourceWeighted):
			activeSourceActivationValues = somaValues[activeMask]
		if(activeColumnIndices.numel() > arrayIndexSegmentFirst):
			trainedSourceColumnIndicesDevice = trainedSourceColumnIndices.to(device=activeColumnIndices.device)
			trainedSourceMask = pt.isin(activeColumnIndices, trainedSourceColumnIndicesDevice)
			activeColumnIndices = activeColumnIndices[trainedSourceMask]
			if(inferenceInferMissingFeaturesCandidateSourceWeighted):
				activeSourceActivationValues = activeSourceActivationValues[trainedSourceMask]
		activeSourceColumns = activeColumnIndices.tolist()
		if(not inferenceInferMissingFeaturesCandidateSourceWeighted):
			activeSourceActivationValues = pt.full((activeColumnIndices.shape[0],), inferenceInferMissingFeaturesNormalisedActivationTotal, dtype=globalFeatureNeuronsActivation.dtype, device=globalFeatureNeuronsActivation.device)
		activeSourceActivations = list(activeSourceActivationValues.unbind())
		totalSourceActivation = activeSourceActivationValues.sum()
		result = activeSourceColumns, activeSourceActivations, totalSourceActivation
	else:
		raise RuntimeError("getActiveMissingFeatureSourcesLeakyIntegrateAndFire error: inferenceInferMissingFeatures is disabled")
	return result

def calculateCombinedMissingFeatureTargetActivation(databaseNetworkObject, observedColumnsDict, globalFeatureNeuronsActivation, sourceFeatureIndex, activeSourceColumns, activeSourceActivations, totalSourceActivation, sequenceWordIndex):
	result = None
	if(inferenceInferMissingFeatures):
		if(len(activeSourceColumns) == arrayIndexSegmentFirst or len(activeSourceColumns) != len(activeSourceActivations)):
			raise RuntimeError("calculateCombinedMissingFeatureTargetActivation error: active source columns and activations must be non-empty and have equal length")
		normalisationMultiplier = inferenceInferMissingFeaturesNormalisedActivationTotal / float(totalSourceActivation.item())
		if(not math.isfinite(normalisationMultiplier) or normalisationMultiplier <= inferenceInferMissingFeaturesMinimumActivation):
			raise RuntimeError("calculateCombinedMissingFeatureTargetActivation error: normalisation multiplier must be finite and positive")
		combinedTargetActivation = None
		for sourceColumnIndex, sourceActivation in zip(activeSourceColumns, activeSourceActivations):
			observedColumn = loadMissingFeatureSourceObservedColumn(databaseNetworkObject, observedColumnsDict, sourceColumnIndex, sourceFeatureIndex, sequenceWordIndex)
			featureConnections = observedColumn.prepareFeatureConnectionsForSourceFeature(sourceFeatureIndex, targetDevice=globalFeatureNeuronsActivation.device, createMissing=False)
			featureNeuronsTargetActivation = calculateMissingFeatureTargetActivation(databaseNetworkObject, featureConnections, sourceColumnIndex, sourceFeatureIndex, sourceActivation, normalisationMultiplier)
			if(combinedTargetActivation is None):
				combinedTargetActivation = featureNeuronsTargetActivation
			else:
				combinedTargetActivation = combinedTargetActivation + featureNeuronsTargetActivation
		if(combinedTargetActivation is None):
			raise RuntimeError("calculateCombinedMissingFeatureTargetActivation error: no target activation was generated")
		if(combinedTargetActivation.is_sparse):
			combinedTargetActivation = combinedTargetActivation.coalesce()
			combinedTargetValues = combinedTargetActivation.values()
		else:
			combinedTargetValues = combinedTargetActivation
		if(not bool(pt.all(pt.isfinite(combinedTargetValues)).item())):
			raise RuntimeError("calculateCombinedMissingFeatureTargetActivation error: combined target activation must be finite")
		if(bool(pt.any(combinedTargetValues < inferenceInferMissingFeaturesMinimumActivation).item())):
			raise RuntimeError("calculateCombinedMissingFeatureTargetActivation error: combined target activation must be non-negative")
		if(combinedTargetValues.numel() > arrayIndexSegmentFirst and bool(pt.any(combinedTargetValues > inferenceInferMissingFeaturesMinimumActivation).item())):
			result = combinedTargetActivation
	else:
		raise RuntimeError("calculateCombinedMissingFeatureTargetActivation error: inferenceInferMissingFeatures is disabled")
	return result

def loadMissingFeatureSourceObservedColumn(databaseNetworkObject, observedColumnsDict, sourceColumnIndex, sourceFeatureIndex, sequenceWordIndex):
	result = None
	if(inferenceInferMissingFeatures):
		if(observedColumnsDict is None):
			raise RuntimeError("loadMissingFeatureSourceObservedColumn error: observedColumnsDict is None")
		sourceColumnName = databaseNetworkObject.conceptColumnsList[sourceColumnIndex]
		observedColumn = observedColumnsDict.get(sourceColumnName)
		if(observedColumn is None):
			observedColumn = GIAANNcmn_databaseNetwork.loadOrCreateObservedColumn(databaseNetworkObject, sourceColumnIndex, sourceColumnName, sequenceWordIndex, deviceLoadColumnInference, deviceLoadColumnInferenceCopy, requiredSourceFeatureIndices=[sourceFeatureIndex])
			if(not inferenceOnlyRetainPredictedTargetObservedColumn):
				observedColumnsDict[sourceColumnName] = observedColumn
		result = observedColumn
	else:
		raise RuntimeError("loadMissingFeatureSourceObservedColumn error: inferenceInferMissingFeatures is disabled")
	return result

def calculateMissingFeatureTargetActivation(databaseNetworkObject, featureConnections, sourceColumnIndex, sourceFeatureIndex, sourceActivation, normalisationMultiplier):
	result = None
	if(inferenceInferMissingFeatures):
		featureConnectionsStrengthStored = featureConnections[databaseNetworkObject.arrayIndexPropertiesStrengthIndex]
		featureConnectionsStrength = featureConnectionsStrengthStored
		if(inferenceConnectionStrengthPOSdependence):
			featureConnectionsPos = featureConnections[databaseNetworkObject.arrayIndexPropertiesPosIndex]
			featureConnectionsStrength = GIAANNcmn_predictionActivate.applyConnectionStrengthPOSdependenceInference(databaseNetworkObject, featureConnectionsStrength, featureConnectionsPos, sourceColumnIndex)
		featureConnectionsStrengthRaw = featureConnectionsStrength
		if(inferenceConnectionsStrengthBoolean):
			featureConnectionsStrength = featureConnectionsStrength.bool().float()
		if(sourceActivation.dim() > arrayIndexSegmentFirst):
			sourceActivation = sourceActivation.reshape(-1)
		featureNeuronsTargetActivation = GIAANNcmn_predictionActivate.calculateFeatureNeuronsTargetActivationPredict(featureConnectionsStrength, sourceActivation)
		if(inferenceDuringTrainAdjustSynapseStrengthBiasTimingCalculations and inferenceConnectionsStrengthBoolean):
			featureNeuronsTargetActivation = GIAANNcmn_predictionActivate.calculateFeatureNeuronsTargetActivationPredict(featureConnectionsStrengthRaw, sourceActivation)
		featureNeuronsTargetActivation = featureNeuronsTargetActivation * normalisationMultiplier
		if(inferenceDuringTrainAdjustSynapseStrength and inferenceDuringTrainAdjustSynapseStrengthDecrementInference):
			GIAANNcmn_inferenceDuringTrain.updateInferenceDuringTrainConnectionsActive(databaseNetworkObject, featureNeuronsTargetActivation, featureConnectionsStrengthStored, sourceColumnIndex, sourceFeatureIndex)
		result = featureNeuronsTargetActivation
	else:
		raise RuntimeError("calculateMissingFeatureTargetActivation error: inferenceInferMissingFeatures is disabled")
	return result
