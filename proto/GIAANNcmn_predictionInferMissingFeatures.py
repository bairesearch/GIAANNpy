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
import os
import torch as pt

from GIAANNcmn_globalDefs import *
import GIAANNcmn_databaseNetwork
import GIAANNcmn_databaseNetworkFiles
import GIAANNcmn_inferenceDuringTrain
import GIAANNcmn_predictionActivate

def getTrainedMissingFeatureSourceColumnIndices(databaseNetworkObject, globalFeatureNeuronsStrength, sourceColumnIndex, sourceFeatureIndex):
	result = None
	if(inferenceInferMissingFeatures):
		if(databaseNetworkObject is None):
			raise RuntimeError("getTrainedMissingFeatureSourceColumnIndices error: databaseNetworkObject is None")
		if(globalFeatureNeuronsStrength is None):
			raise RuntimeError("getTrainedMissingFeatureSourceColumnIndices error: globalFeatureNeuronsStrength is None")
		sourceColumnIndex = int(sourceColumnIndex)
		sourceFeatureIndex = int(sourceFeatureIndex)
		if(sourceColumnIndex < arrayIndexSegmentFirst or sourceColumnIndex >= databaseNetworkObject.c):
			raise RuntimeError("getTrainedMissingFeatureSourceColumnIndices error: sourceColumnIndex is out of range")
		if(sourceFeatureIndex < arrayIndexSegmentFirst or sourceFeatureIndex >= databaseNetworkObject.f):
			raise RuntimeError("getTrainedMissingFeatureSourceColumnIndices error: sourceFeatureIndex is out of range")
		if(inferenceInferMissingFeaturesUpdate7UsePersistedSourceConnections):
			if(not inferenceLeakyIntegrateAndFire):
				raise RuntimeError("getTrainedMissingFeatureSourceColumnIndices error: persisted source connection discovery requires inferenceLeakyIntegrateAndFire")
			currentFeatureIsTrained = missingFeatureSourceConnectionsExist(sourceColumnIndex, sourceFeatureIndex)
			trainedColumnIndices = pt.empty((arrayIndexSegmentFirst,), dtype=pt.long, device=globalFeatureNeuronsStrength.device)
		else:
			if(globalFeatureNeuronsStrength.dim() != inferenceInferMissingFeaturesNeuronTensorRank):
				raise RuntimeError("getTrainedMissingFeatureSourceColumnIndices error: globalFeatureNeuronsStrength rank is invalid")
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
		if(currentFeatureIsTrained or (inferenceInferMissingFeaturesUpdate7UsePersistedSourceConnections and sourceFeatureIndex == featureIndexPrimeConceptNeuron)):
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
		missingFeaturePropagationApplied = False
		activeSourceColumns, activeSourceFeatureIndices, activeSourceActivations, activeSourceScores, totalSourceActivation, sourceConfidence = getActiveMissingFeatureSources(databaseNetworkObject, globalFeatureNeuronsActivation, sourceFeatureIndex, trainedSourceColumnIndices)
		constraintSourceColumns, constraintSourceFeatureIndices = selectMissingFeatureConstraintSources(activeSourceColumns, activeSourceFeatureIndices, activeSourceScores)
		if(len(activeSourceColumns) > arrayIndexSegmentFirst):
			combinedTargetActivation, combinedTargetActivationTransformed = calculateCombinedMissingFeatureTargetActivation(databaseNetworkObject, observedColumnsDict, globalFeatureNeuronsActivation, activeSourceColumns, activeSourceFeatureIndices, activeSourceActivations, totalSourceActivation, sourceConfidence, sequenceWordIndex)
			if(combinedTargetActivation is not None):
				if(inferenceInferMissingFeaturesUpdate6RetainCandidateActivations):
					missingFeatureCandidateActivations = captureMissingFeatureCandidateActivations(globalFeatureNeuronsActivation, activeSourceColumns, activeSourceFeatureIndices)
				if(combinedTargetActivationTransformed):
					globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, globalFeatureNeuronsTime = GIAANNcmn_predictionActivate.applyFeatureNeuronsTargetActivationPredict(databaseNetworkObject, globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, combinedTargetActivation, globalFeatureNeuronsTime, activationSequenceWordIndex, activationSequenceColumnIndex)
				else:
					globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, globalFeatureNeuronsTime = GIAANNcmn_predictionActivate.processFeatureNeuronsTargetActivationPredict(databaseNetworkObject, globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, combinedTargetActivation, globalFeatureNeuronsTime, activationSequenceWordIndex, activationSequenceColumnIndex)
				if(inferenceInferMissingFeaturesUpdate6RetainCandidateActivations):
					globalFeatureNeuronsActivation = restoreMissingFeatureCandidateActivations(globalFeatureNeuronsActivation, missingFeatureCandidateActivations, activeSourceColumns, activeSourceFeatureIndices)
				missingFeaturePropagationApplied = True
		result = globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, globalFeatureNeuronsTime, activeSourceColumns, activeSourceFeatureIndices, constraintSourceColumns, constraintSourceFeatureIndices, missingFeaturePropagationApplied
	else:
		raise RuntimeError("processMissingFeaturePredictionActivations error: inferenceInferMissingFeatures is disabled")
	return result

def processMissingFeaturePredictionActivationsEnforceLastSegment(databaseNetworkObject, observedColumnsDict, globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, sourceFeatureIndex, trainedSourceColumnIndices, somaActivationFromLastSegmentKeys, globalFeatureNeuronsTime, activationSequenceWordIndex, activationSequenceColumnIndex, sequenceWordIndex):
	result = None
	if(inferenceInferMissingFeatures):
		if(not inferenceLeakyIntegrateAndFire or algorithmMatrixSANIenforceRequirement != "enforceLastSegmentMustBeActive"):
			raise RuntimeError("processMissingFeaturePredictionActivationsEnforceLastSegment error: requires inferenceLeakyIntegrateAndFire enforceLastSegmentMustBeActive")
		missingFeaturePropagationApplied = False
		activeSourceColumns, activeSourceFeatureIndices, activeSourceActivations, activeSourceScores, totalSourceActivation, sourceConfidence = getActiveMissingFeatureSources(databaseNetworkObject, globalFeatureNeuronsActivation, sourceFeatureIndex, trainedSourceColumnIndices)
		constraintSourceColumns, constraintSourceFeatureIndices = selectMissingFeatureConstraintSources(activeSourceColumns, activeSourceFeatureIndices, activeSourceScores)
		if(len(activeSourceColumns) > arrayIndexSegmentFirst):
			combinedTargetActivation, combinedTargetActivationTransformed = calculateCombinedMissingFeatureTargetActivation(databaseNetworkObject, observedColumnsDict, globalFeatureNeuronsActivation, activeSourceColumns, activeSourceFeatureIndices, activeSourceActivations, totalSourceActivation, sourceConfidence, sequenceWordIndex)
			if(combinedTargetActivation is not None):
				if(inferenceInferMissingFeaturesUpdate6RetainCandidateActivations):
					missingFeatureCandidateActivations = captureMissingFeatureCandidateActivations(globalFeatureNeuronsActivation, activeSourceColumns, activeSourceFeatureIndices)
				if(not combinedTargetActivationTransformed):
					combinedTargetActivation = GIAANNcmn_predictionActivate.transformFeatureNeuronsTargetActivationPredict(combinedTargetActivation)
				combinedTargetActivation, somaActivationFromLastSegmentKeys = GIAANNcmn_predictionActivate.mergeLeakyIntegrateAndFireCurrentSomaActivationKeys(somaActivationFromLastSegmentKeys, combinedTargetActivation)
				globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, globalFeatureNeuronsTime = GIAANNcmn_predictionActivate.applyFeatureNeuronsTargetActivationPredict(databaseNetworkObject, globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, combinedTargetActivation, globalFeatureNeuronsTime, activationSequenceWordIndex, activationSequenceColumnIndex)
				if(inferenceInferMissingFeaturesUpdate6RetainCandidateActivations):
					globalFeatureNeuronsActivation = restoreMissingFeatureCandidateActivations(globalFeatureNeuronsActivation, missingFeatureCandidateActivations, activeSourceColumns, activeSourceFeatureIndices)
				missingFeaturePropagationApplied = True
		result = globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, globalFeatureNeuronsTime, somaActivationFromLastSegmentKeys, activeSourceColumns, activeSourceFeatureIndices, constraintSourceColumns, constraintSourceFeatureIndices, missingFeaturePropagationApplied
	else:
		raise RuntimeError("processMissingFeaturePredictionActivationsEnforceLastSegment error: inferenceInferMissingFeatures is disabled")
	return result

def selectMissingFeatureConstraintSources(activeSourceColumns, activeSourceFeatureIndices, activeSourceScores):
	result = None
	if(inferenceInferMissingFeatures):
		if(len(activeSourceColumns) != len(activeSourceFeatureIndices) or len(activeSourceColumns) != len(activeSourceScores)):
			raise RuntimeError("selectMissingFeatureConstraintSources error: source columns, features, and scores must have equal length")
		resultColumns = list(activeSourceColumns)
		resultFeatureIndices = list(activeSourceFeatureIndices)
		if(inferenceInferMissingFeaturesUpdate5SelectTopKConstraintSources and len(activeSourceColumns) > arrayIndexSegmentFirst):
			sourceScoreValues = pt.stack(activeSourceScores)
			selectedSourceIndices = selectTopMissingFeatureSourceIndices(sourceScoreValues, inferenceInferMissingFeaturesUpdate5ConstraintCandidateTopK)
			resultColumns = [activeSourceColumns[int(sourceIndex.item())] for sourceIndex in selectedSourceIndices]
			resultFeatureIndices = [activeSourceFeatureIndices[int(sourceIndex.item())] for sourceIndex in selectedSourceIndices]
		result = resultColumns, resultFeatureIndices
	else:
		raise RuntimeError("selectMissingFeatureConstraintSources error: inferenceInferMissingFeatures is disabled")
	return result

def buildMissingFeatureSourceTensors(activeSourceColumns, activeSourceFeatureIndices, device):
	result = None
	if(inferenceInferMissingFeatures):
		if(activeSourceColumns is None or activeSourceFeatureIndices is None or len(activeSourceColumns) == arrayIndexSegmentFirst or len(activeSourceColumns) != len(activeSourceFeatureIndices)):
			raise RuntimeError("buildMissingFeatureSourceTensors error: source columns and features must be non-empty and have equal length")
		conceptColumnIndices = pt.tensor(activeSourceColumns, dtype=pt.long, device=device)
		conceptColumnFeatureIndices = pt.tensor(activeSourceFeatureIndices, dtype=pt.long, device=device).unsqueeze(inferenceInferMissingFeaturesCandidateFeatureDimension)
		result = conceptColumnIndices, conceptColumnFeatureIndices
	else:
		raise RuntimeError("buildMissingFeatureSourceTensors error: inferenceInferMissingFeatures is disabled")
	return result

def captureMissingFeatureCandidateActivations(globalFeatureNeuronsActivation, activeSourceColumns, activeSourceFeatureIndices):
	result = None
	if(inferenceInferMissingFeatures and inferenceInferMissingFeaturesUpdate6RetainCandidateActivations):
		globalFeatureNeuronsActivationSparse, candidateActivationMask = getMissingFeatureCandidateActivationMask(globalFeatureNeuronsActivation, activeSourceColumns, activeSourceFeatureIndices)
		result = pt.sparse_coo_tensor(globalFeatureNeuronsActivationSparse.indices()[:, candidateActivationMask], globalFeatureNeuronsActivationSparse.values()[candidateActivationMask], size=globalFeatureNeuronsActivationSparse.size(), dtype=globalFeatureNeuronsActivationSparse.dtype, device=globalFeatureNeuronsActivationSparse.device).coalesce()
	else:
		raise RuntimeError("captureMissingFeatureCandidateActivations error: inferenceInferMissingFeaturesUpdate6RetainCandidateActivations is disabled")
	return result

def restoreMissingFeatureCandidateActivations(globalFeatureNeuronsActivation, missingFeatureCandidateActivations, activeSourceColumns, activeSourceFeatureIndices):
	result = None
	if(inferenceInferMissingFeatures and inferenceInferMissingFeaturesUpdate6RetainCandidateActivations):
		globalFeatureNeuronsActivationSparse, candidateActivationMask = getMissingFeatureCandidateActivationMask(globalFeatureNeuronsActivation, activeSourceColumns, activeSourceFeatureIndices)
		if(missingFeatureCandidateActivations is None or not missingFeatureCandidateActivations.is_sparse):
			raise RuntimeError("restoreMissingFeatureCandidateActivations error: captured candidate activation must be sparse")
		missingFeatureCandidateActivationsSparse = missingFeatureCandidateActivations.coalesce()
		if(missingFeatureCandidateActivationsSparse.size() != globalFeatureNeuronsActivationSparse.size() or missingFeatureCandidateActivationsSparse.dtype != globalFeatureNeuronsActivationSparse.dtype or missingFeatureCandidateActivationsSparse.device != globalFeatureNeuronsActivationSparse.device):
			raise RuntimeError("restoreMissingFeatureCandidateActivations error: captured candidate activation tensor metadata does not match global activation")
		if(missingFeatureCandidateActivationsSparse._nnz() > arrayIndexSegmentFirst):
			capturedCandidateActivationSparse, capturedCandidateActivationMask = getMissingFeatureCandidateActivationMask(missingFeatureCandidateActivationsSparse, activeSourceColumns, activeSourceFeatureIndices)
			if(not bool(pt.all(capturedCandidateActivationMask).item())):
				raise RuntimeError("restoreMissingFeatureCandidateActivations error: captured activation contains non-candidate coordinates")
			missingFeatureCandidateActivationsSparse = capturedCandidateActivationSparse
		retainedActivationIndices = globalFeatureNeuronsActivationSparse.indices()[:, pt.logical_not(candidateActivationMask)]
		retainedActivationValues = globalFeatureNeuronsActivationSparse.values()[pt.logical_not(candidateActivationMask)]
		restoredActivationIndices = pt.cat((retainedActivationIndices, missingFeatureCandidateActivationsSparse.indices()), dim=1)
		restoredActivationValues = pt.cat((retainedActivationValues, missingFeatureCandidateActivationsSparse.values()), dim=0)
		result = pt.sparse_coo_tensor(restoredActivationIndices, restoredActivationValues, size=globalFeatureNeuronsActivationSparse.size(), dtype=globalFeatureNeuronsActivationSparse.dtype, device=globalFeatureNeuronsActivationSparse.device).coalesce()
	else:
		raise RuntimeError("restoreMissingFeatureCandidateActivations error: inferenceInferMissingFeaturesUpdate6RetainCandidateActivations is disabled")
	return result

def getMissingFeatureCandidateActivationMask(globalFeatureNeuronsActivation, activeSourceColumns, activeSourceFeatureIndices):
	result = None
	if(inferenceInferMissingFeatures and inferenceInferMissingFeaturesUpdate6RetainCandidateActivations):
		if(globalFeatureNeuronsActivation is None or not globalFeatureNeuronsActivation.is_sparse):
			raise RuntimeError("getMissingFeatureCandidateActivationMask error: global activation must be sparse")
		if(globalFeatureNeuronsActivation.dim() != inferenceInferMissingFeaturesNeuronTensorRank):
			raise RuntimeError("getMissingFeatureCandidateActivationMask error: global activation rank is invalid")
		if(activeSourceColumns is None or activeSourceFeatureIndices is None or len(activeSourceColumns) == arrayIndexSegmentFirst or len(activeSourceColumns) != len(activeSourceFeatureIndices)):
			raise RuntimeError("getMissingFeatureCandidateActivationMask error: source columns and features must be non-empty and have equal length")
		globalFeatureNeuronsActivationSparse = globalFeatureNeuronsActivation.coalesce()
		candidateColumnIndices = pt.tensor(activeSourceColumns, dtype=pt.long, device=globalFeatureNeuronsActivationSparse.device)
		candidateFeatureIndices = pt.tensor(activeSourceFeatureIndices, dtype=pt.long, device=globalFeatureNeuronsActivationSparse.device)
		if(bool(pt.any(candidateColumnIndices < arrayIndexSegmentFirst).item()) or bool(pt.any(candidateColumnIndices >= globalFeatureNeuronsActivationSparse.shape[inferenceInferMissingFeaturesConceptDimension]).item())):
			raise RuntimeError("getMissingFeatureCandidateActivationMask error: source column index is out of range")
		if(bool(pt.any(candidateFeatureIndices < arrayIndexSegmentFirst).item()) or bool(pt.any(candidateFeatureIndices >= globalFeatureNeuronsActivationSparse.shape[inferenceInferMissingFeaturesFeatureDimension]).item())):
			raise RuntimeError("getMissingFeatureCandidateActivationMask error: source feature index is out of range")
		candidateKeys = candidateColumnIndices*int(globalFeatureNeuronsActivationSparse.shape[inferenceInferMissingFeaturesFeatureDimension]) + candidateFeatureIndices
		activationIndices = globalFeatureNeuronsActivationSparse.indices()
		activationKeys = activationIndices[inferenceInferMissingFeaturesConceptDimension]*int(globalFeatureNeuronsActivationSparse.shape[inferenceInferMissingFeaturesFeatureDimension]) + activationIndices[inferenceInferMissingFeaturesFeatureDimension]
		candidateActivationMask = pt.isin(activationKeys, candidateKeys)
		result = globalFeatureNeuronsActivationSparse, candidateActivationMask
	else:
		raise RuntimeError("getMissingFeatureCandidateActivationMask error: inferenceInferMissingFeaturesUpdate6RetainCandidateActivations is disabled")
	return result

def getActiveMissingFeatureSources(databaseNetworkObject, globalFeatureNeuronsActivation, sourceFeatureIndex, trainedSourceColumnIndices):
	result = None
	if(inferenceInferMissingFeatures):
		if(trainedSourceColumnIndices is None):
			raise RuntimeError("getActiveMissingFeatureSources error: trainedSourceColumnIndices is None")
		if(inferenceInferMissingFeaturesUpdateEnabledCount > arrayIndexSegmentFirst and not inferenceLeakyIntegrateAndFire):
			raise RuntimeError("getActiveMissingFeatureSources error: inferenceInferMissingFeaturesUpdate experiments require inferenceLeakyIntegrateAndFire")
		if(inferenceInferMissingFeaturesUpdate7UsePersistedSourceConnections and not inferenceLeakyIntegrateAndFire):
			raise RuntimeError("getActiveMissingFeatureSources error: persisted source connection discovery requires inferenceLeakyIntegrateAndFire")
		if(inferenceLeakyIntegrateAndFire):
			activeSourceColumns, activeSourceFeatureIndices, activeSourceActivations, activeSourceScores, totalSourceActivation, sourceConfidence = getActiveMissingFeatureSourcesLeakyIntegrateAndFire(globalFeatureNeuronsActivation, sourceFeatureIndex, trainedSourceColumnIndices)
		else:
			activeSourceColumns = []
			activeSourceFeatureIndices = []
			activeSourceActivations = []
			activeSourceScores = []
			totalSourceActivation = pt.zeros((), dtype=globalFeatureNeuronsActivation.dtype, device=globalFeatureNeuronsActivation.device)
			sourceConfidence = pt.full((), inferenceInferMissingFeaturesNormalisedActivationTotal, dtype=globalFeatureNeuronsActivation.dtype, device=globalFeatureNeuronsActivation.device)
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
					activeSourceFeatureIndices.append(sourceFeatureIndex)
					activeSourceActivations.append(sourceActivation)
					activeSourceScores.append(sourceActivationTotal)
					totalSourceActivation = totalSourceActivation + sourceActivationTotal
		if(not bool(pt.isfinite(totalSourceActivation).item())):
			raise RuntimeError("getActiveMissingFeatureSources error: combined source activation must be finite")
		if(len(activeSourceColumns) == arrayIndexSegmentFirst):
			if(bool((totalSourceActivation != inferenceInferMissingFeaturesMinimumActivation).item())):
				raise RuntimeError("getActiveMissingFeatureSources error: inactive combined source activation must be zero")
		elif(not bool((totalSourceActivation > inferenceInferMissingFeaturesMinimumActivation).item())):
			raise RuntimeError("getActiveMissingFeatureSources error: active combined source activation must be positive")
		result = activeSourceColumns, activeSourceFeatureIndices, activeSourceActivations, activeSourceScores, totalSourceActivation, sourceConfidence
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
		if(inferenceInferMissingFeaturesUpdate10UseGlobalFeaturePredictions):
			featureMask = pt.ones_like(somaValues, dtype=pt.bool)
		else:
			featureMask = somaIndices[inferenceLeakyIntegrateAndFireSomaActivationFeatureDimension] == int(sourceFeatureIndex)
		relevantSomaValues = somaValues[featureMask]
		if(relevantSomaValues.numel() > arrayIndexSegmentFirst):
			if(not bool(pt.all(pt.isfinite(relevantSomaValues)).item())):
				raise RuntimeError("getActiveMissingFeatureSourcesLeakyIntegrateAndFire error: relevant soma activations must be finite")
			if(bool(pt.any(relevantSomaValues < inferenceInferMissingFeaturesMinimumActivation).item())):
				raise RuntimeError("getActiveMissingFeatureSourcesLeakyIntegrateAndFire error: relevant soma activations must be non-negative")
		if(inferenceInferMissingFeaturesUpdate9UsePartialSourceActivations):
			activeMask = featureMask & (somaValues > inferenceInferMissingFeaturesMinimumActivation)
		else:
			activeMask = featureMask & (somaValues >= inferenceLeakyIntegrateAndFireSomaActivationThreshold)
		activeColumnIndices = somaIndices[inferenceLeakyIntegrateAndFireSomaActivationConceptDimension, activeMask]
		activeFeatureIndices = somaIndices[inferenceLeakyIntegrateAndFireSomaActivationFeatureDimension, activeMask]
		activeSourceScoreValues = somaValues[activeMask]
		if(activeColumnIndices.numel() > arrayIndexSegmentFirst):
			if(inferenceInferMissingFeaturesUpdate7UsePersistedSourceConnections):
				trainedSourceMask = pt.tensor([missingFeatureSourceConnectionsExist(sourceColumnIndex, activeSourceFeatureIndex) for sourceColumnIndex, activeSourceFeatureIndex in zip(activeColumnIndices.tolist(), activeFeatureIndices.tolist())], dtype=pt.bool, device=activeColumnIndices.device)
			else:
				trainedSourceColumnIndicesDevice = trainedSourceColumnIndices.to(device=activeColumnIndices.device)
				trainedSourceMask = pt.isin(activeColumnIndices, trainedSourceColumnIndicesDevice)
			activeColumnIndices = activeColumnIndices[trainedSourceMask]
			activeFeatureIndices = activeFeatureIndices[trainedSourceMask]
			activeSourceScoreValues = activeSourceScoreValues[trainedSourceMask]
		if(inferenceInferMissingFeaturesUpdate10UseGlobalFeaturePredictions and activeColumnIndices.numel() > arrayIndexSegmentFirst):
			selectedSourceIndices = selectTopMissingFeatureSourceIndices(activeSourceScoreValues, inferenceInferMissingFeaturesUpdate10GlobalFeaturePredictionTopK)
			activeColumnIndices = activeColumnIndices[selectedSourceIndices]
			activeFeatureIndices = activeFeatureIndices[selectedSourceIndices]
			activeSourceScoreValues = activeSourceScoreValues[selectedSourceIndices]
		activeColumnIndices, activeFeatureIndices, activeSourceScoreValues, sourceConfidence = applyMissingFeatureSourceSelectionUpdates(activeColumnIndices, activeFeatureIndices, activeSourceScoreValues)
		activeSourceColumns = activeColumnIndices.tolist()
		activeSourceFeatureIndices = activeFeatureIndices.tolist()
		if(inferenceInferMissingFeaturesCandidateSourceWeighted or inferenceInferMissingFeaturesUpdate3AbsoluteSourceConfidence or inferenceInferMissingFeaturesUpdate4TransformCandidateSignals):
			activeSourceActivationValues = activeSourceScoreValues
		else:
			activeSourceActivationValues = pt.full((activeColumnIndices.shape[0],), inferenceInferMissingFeaturesNormalisedActivationTotal, dtype=globalFeatureNeuronsActivation.dtype, device=globalFeatureNeuronsActivation.device)
		activeSourceActivations = list(activeSourceActivationValues.unbind())
		activeSourceScores = list(activeSourceScoreValues.unbind())
		totalSourceActivation = activeSourceActivationValues.sum()
		result = activeSourceColumns, activeSourceFeatureIndices, activeSourceActivations, activeSourceScores, totalSourceActivation, sourceConfidence
	else:
		raise RuntimeError("getActiveMissingFeatureSourcesLeakyIntegrateAndFire error: inferenceInferMissingFeatures is disabled")
	return result

def missingFeatureSourceConnectionsExist(sourceColumnIndex, sourceFeatureIndex):
	result = None
	if(inferenceInferMissingFeatures and inferenceInferMissingFeaturesUpdate7UsePersistedSourceConnections):
		sourceConnectionsFolder = GIAANNcmn_databaseNetworkFiles.getObservedColumnFeatureConnectionsFolder(sourceColumnIndex)
		sourceConnectionsFileName = GIAANNcmn_databaseNetworkFiles.getObservedColumnSourceFeatureConnectionsFileBaseName(sourceFeatureIndex) + pytorchTensorFileExtension
		sourceConnectionsFilePath = os.path.join(sourceConnectionsFolder, sourceConnectionsFileName)
		result = GIAANNcmn_databaseNetworkFiles.pathExists(sourceConnectionsFilePath)
	else:
		raise RuntimeError("missingFeatureSourceConnectionsExist error: persisted source connection discovery is disabled")
	return result

def applyMissingFeatureSourceSelectionUpdates(activeColumnIndices, activeFeatureIndices, activeSourceScoreValues):
	result = None
	if(inferenceInferMissingFeatures):
		if(activeColumnIndices.dim() != 1 or activeFeatureIndices.dim() != 1 or activeSourceScoreValues.dim() != 1 or activeColumnIndices.shape[0] != activeFeatureIndices.shape[0] or activeColumnIndices.shape[0] != activeSourceScoreValues.shape[0]):
			raise RuntimeError("applyMissingFeatureSourceSelectionUpdates error: source columns, features, and scores must be aligned vectors")
		selectedSourceIndices = pt.arange(activeColumnIndices.shape[0], dtype=pt.long, device=activeColumnIndices.device)
		sourceConfidence = pt.full((), inferenceInferMissingFeaturesNormalisedActivationTotal, dtype=activeSourceScoreValues.dtype, device=activeSourceScoreValues.device)
		if(inferenceInferMissingFeaturesUpdate2SourceConfidence and activeColumnIndices.numel() > arrayIndexSegmentFirst):
			sourceConfidence = calculateMissingFeatureSourceConfidence(activeSourceScoreValues)
		if(inferenceInferMissingFeaturesUpdate1SelectTopKCandidates and selectedSourceIndices.numel() > arrayIndexSegmentFirst):
			selectedSourceIndicesRanked = selectTopMissingFeatureSourceIndices(activeSourceScoreValues[selectedSourceIndices], inferenceInferMissingFeaturesUpdate1CandidateTopK)
			selectedSourceIndices = selectedSourceIndices[selectedSourceIndicesRanked]
		activeColumnIndices = activeColumnIndices[selectedSourceIndices]
		activeFeatureIndices = activeFeatureIndices[selectedSourceIndices]
		activeSourceScoreValues = activeSourceScoreValues[selectedSourceIndices]
		result = activeColumnIndices, activeFeatureIndices, activeSourceScoreValues, sourceConfidence
	else:
		raise RuntimeError("applyMissingFeatureSourceSelectionUpdates error: inferenceInferMissingFeatures is disabled")
	return result

def calculateMissingFeatureSourceConfidence(activeSourceScoreValues):
	result = None
	if(inferenceInferMissingFeatures and inferenceInferMissingFeaturesUpdate2SourceConfidence):
		if(activeSourceScoreValues.dim() != 1 or activeSourceScoreValues.numel() == arrayIndexSegmentFirst):
			raise RuntimeError("calculateMissingFeatureSourceConfidence error: source scores must be a non-empty vector")
		if(not bool(pt.all(pt.isfinite(activeSourceScoreValues)).item())):
			raise RuntimeError("calculateMissingFeatureSourceConfidence error: source scores must be finite")
		if(bool(pt.any(activeSourceScoreValues <= inferenceInferMissingFeaturesMinimumActivation).item())):
			raise RuntimeError("calculateMissingFeatureSourceConfidence error: source scores must be positive")
		rankedSourceIndices = selectTopMissingFeatureSourceIndices(activeSourceScoreValues, activeSourceScoreValues.shape[0])
		winnerActivation = activeSourceScoreValues[rankedSourceIndices[arrayIndexSegmentFirst]]
		absoluteConfidence = pt.clamp(winnerActivation / inferenceInferMissingFeaturesUpdate2FullConfidenceActivation, max=inferenceInferMissingFeaturesNormalisedActivationTotal)
		marginConfidence = pt.full((), inferenceInferMissingFeaturesNormalisedActivationTotal, dtype=activeSourceScoreValues.dtype, device=activeSourceScoreValues.device)
		if(activeSourceScoreValues.shape[0] > inferenceInferMissingFeaturesUpdate2RunnerUpRank):
			runnerUpActivation = activeSourceScoreValues[rankedSourceIndices[inferenceInferMissingFeaturesUpdate2RunnerUpRank]]
			marginRatio = winnerActivation / runnerUpActivation
			marginConfidence = pt.clamp(marginRatio / inferenceInferMissingFeaturesUpdate2FullConfidenceMarginRatio, max=inferenceInferMissingFeaturesNormalisedActivationTotal)
		result = absoluteConfidence * marginConfidence
		if(not bool(pt.isfinite(result).item()) or not bool((result > inferenceInferMissingFeaturesMinimumActivation).item()) or not bool((result <= inferenceInferMissingFeaturesNormalisedActivationTotal).item())):
			raise RuntimeError("calculateMissingFeatureSourceConfidence error: confidence must be finite and within (minimum activation, normalised activation total]")
	else:
		raise RuntimeError("calculateMissingFeatureSourceConfidence error: inferenceInferMissingFeaturesUpdate2SourceConfidence is disabled")
	return result

def selectTopMissingFeatureSourceIndices(activeSourceScoreValues, candidateTopK):
	result = None
	if(inferenceInferMissingFeatures):
		if(activeSourceScoreValues.dim() != 1):
			raise RuntimeError("selectTopMissingFeatureSourceIndices error: source scores must be a vector")
		if(candidateTopK <= arrayIndexSegmentFirst):
			raise RuntimeError("selectTopMissingFeatureSourceIndices error: candidateTopK must be positive")
		selectedCandidateCount = min(candidateTopK, activeSourceScoreValues.shape[0])
		rankedSourceIndices = pt.argsort(activeSourceScoreValues, descending=True, stable=True)
		result = rankedSourceIndices[:selectedCandidateCount]
	else:
		raise RuntimeError("selectTopMissingFeatureSourceIndices error: inferenceInferMissingFeatures is disabled")
	return result

def calculateCombinedMissingFeatureTargetActivation(databaseNetworkObject, observedColumnsDict, globalFeatureNeuronsActivation, activeSourceColumns, activeSourceFeatureIndices, activeSourceActivations, totalSourceActivation, sourceConfidence, sequenceWordIndex):
	result = None, False
	if(inferenceInferMissingFeatures):
		if(len(activeSourceColumns) == arrayIndexSegmentFirst or len(activeSourceColumns) != len(activeSourceFeatureIndices) or len(activeSourceColumns) != len(activeSourceActivations)):
			raise RuntimeError("calculateCombinedMissingFeatureTargetActivation error: active source columns, features, and activations must be non-empty and have equal length")
		absoluteSourceConfidence = inferenceInferMissingFeaturesNormalisedActivationTotal
		if(inferenceInferMissingFeaturesUpdate3AbsoluteSourceConfidence):
			absoluteSourceConfidence = min(max(float(sourceActivation.item()) for sourceActivation in activeSourceActivations), inferenceInferMissingFeaturesUpdate3MaximumConfidence)
		if(inferenceInferMissingFeaturesUpdate2SourceConfidence):
			if(not isinstance(sourceConfidence, pt.Tensor) or sourceConfidence.dim() != inferenceInferMissingFeaturesConfidenceTensorRank or not bool(pt.isfinite(sourceConfidence).item()) or not bool((sourceConfidence > inferenceInferMissingFeaturesMinimumActivation).item()) or not bool((sourceConfidence <= inferenceInferMissingFeaturesNormalisedActivationTotal).item())):
				raise RuntimeError("calculateCombinedMissingFeatureTargetActivation error: source confidence must be a finite scalar within (minimum activation, normalised activation total]")
			absoluteSourceConfidence = absoluteSourceConfidence * float(sourceConfidence.item())
		normalisationMultiplier = absoluteSourceConfidence / float(totalSourceActivation.item())
		if(not math.isfinite(normalisationMultiplier) or normalisationMultiplier <= inferenceInferMissingFeaturesMinimumActivation):
			raise RuntimeError("calculateCombinedMissingFeatureTargetActivation error: normalisation multiplier must be finite and positive")
		combinedTargetActivation = None
		for sourceColumnIndex, sourceFeatureIndex, sourceActivation in zip(activeSourceColumns, activeSourceFeatureIndices, activeSourceActivations):
			observedColumn = loadMissingFeatureSourceObservedColumn(databaseNetworkObject, observedColumnsDict, sourceColumnIndex, sourceFeatureIndex, sequenceWordIndex)
			featureConnections = observedColumn.prepareFeatureConnectionsForSourceFeature(sourceFeatureIndex, targetDevice=globalFeatureNeuronsActivation.device, createMissing=False)
			if(inferenceInferMissingFeaturesUpdate4TransformCandidateSignals):
				candidateSourceActivation = pt.full((), inferenceInferMissingFeaturesUpdate4SourceActivation, dtype=sourceActivation.dtype, device=sourceActivation.device)
				featureNeuronsTargetActivation = calculateMissingFeatureTargetActivation(databaseNetworkObject, featureConnections, sourceColumnIndex, sourceFeatureIndex, candidateSourceActivation, inferenceInferMissingFeaturesNormalisedActivationTotal)
				featureNeuronsTargetActivation = GIAANNcmn_predictionActivate.transformFeatureNeuronsTargetActivationPredict(featureNeuronsTargetActivation)
				candidateProbability = absoluteSourceConfidence * float(sourceActivation.item()) / float(totalSourceActivation.item())
				if(not math.isfinite(candidateProbability) or candidateProbability <= inferenceInferMissingFeaturesMinimumActivation):
					raise RuntimeError("calculateCombinedMissingFeatureTargetActivation error: candidate probability must be finite and positive")
				featureNeuronsTargetActivation = featureNeuronsTargetActivation * candidateProbability
			else:
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
			result = combinedTargetActivation, inferenceInferMissingFeaturesUpdate4TransformCandidateSignals
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
