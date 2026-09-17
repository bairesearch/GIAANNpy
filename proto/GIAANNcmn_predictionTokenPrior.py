"""Rescore directly connected LIF candidates using trained token-transition evidence."""

from array import array
from numbers import Integral
import math
import torch as pt

from GIAANNcmn_globalDefs import *
import GIAANNcmn_databaseNetworkFiles
import GIAANNcmn_predictionConstraints


def scoreLIFCandidatesWithTokenPrior(databaseNetworkObject, columnIndices, featureIndices, activationValues, constraintState):
	result = None
	if(inferenceReviewPatch15TokenTransitionPrior):
		validateLIFTokenPriorDatabase(databaseNetworkObject)
		if(not all(isinstance(vector, pt.Tensor) and vector.dim() == inferenceLIFTokenPriorCandidateRank for vector in (columnIndices, featureIndices, activationValues)) or columnIndices.shape != activationValues.shape or featureIndices.shape != activationValues.shape or not activationValues.is_floating_point() or columnIndices.dtype != pt.long or featureIndices.dtype != pt.long or columnIndices.device != activationValues.device or featureIndices.device != activationValues.device):
			raise RuntimeError(inferenceLIFTokenPriorInvalidCandidates)
		if(not pt.isfinite(activationValues).all() or (activationValues <= inferenceLIFTokenPriorMinimumStrength).any() or (columnIndices < arrayIndexSegmentFirst).any() or (columnIndices >= databaseNetworkObject.c).any() or (featureIndices < arrayIndexSegmentFirst).any() or (featureIndices >= databaseNetworkObject.f).any()):
			raise RuntimeError(inferenceLIFTokenPriorInvalidCandidates)
		#The caller has already applied the exact-source connectivity and LIF firing filters.
		columnIndices, featureIndices, activationValues = GIAANNcmn_predictionConstraints.filterColumnFeatureCandidatesByConstraint(databaseNetworkObject, columnIndices, featureIndices, activationValues, constraintState)
		if(columnIndices is not None and activationValues.numel() > inferenceLIFTokenPriorSingleCandidateCount):
			sourceColumnIndex, sourceFeatureIndex = databaseNetworkObject.inferenceLIFTokenPriorSource
			sourceToken = getLIFTokenPriorToken(databaseNetworkObject, sourceColumnIndex, sourceFeatureIndex)
			tokenPrior = getLIFTokenTransitionPrior(databaseNetworkObject, sourceToken)
			targetTokens = [getLIFTokenPriorToken(databaseNetworkObject, columnIndex, featureIndex) for columnIndex, featureIndex in zip(columnIndices.tolist(), featureIndices.tolist())]
			if(any(targetToken not in tokenPrior for targetToken in targetTokens)):
				raise RuntimeError(inferenceLIFTokenPriorMissingEvidence)
			priorStrengths = pt.tensor([tokenPrior[targetToken] for targetToken in targetTokens], dtype=activationValues.dtype, device=activationValues.device)
			if(not pt.isfinite(priorStrengths).all() or (priorStrengths <= inferenceLIFTokenPriorMinimumStrength).any()):
				raise RuntimeError(inferenceLIFTokenPriorMissingEvidence)
			#A bounded prior calibrates the readout without changing any live dendritic or soma activation.
			priorGain = (priorStrengths / priorStrengths.max()).pow(inferenceLIFTokenPriorExponent)
			activationValues = activationValues * (inferenceLIFTokenPriorMinimumGain + (inferenceLIFTokenPriorMaximumGain-inferenceLIFTokenPriorMinimumGain)*priorGain)
			if(not pt.isfinite(activationValues).all() or (activationValues <= inferenceLIFTokenPriorMinimumStrength).any()):
				raise RuntimeError(inferenceLIFTokenPriorInvalidCandidates)
		result = columnIndices, featureIndices, activationValues
	else:
		raise RuntimeError(inferenceLIFTokenPriorInvalidConfiguration)
	return result


def getLIFTokenTransitionPrior(databaseNetworkObject, sourceToken):
	result = None
	if(inferenceReviewPatch15TokenTransitionPrior):
		if(sourceToken not in databaseNetworkObject.inferenceLIFTokenPriorCache):
			if(databaseNetworkObject.inferenceLIFTokenPriorSourceIndex is None):
				databaseNetworkObject.inferenceLIFTokenPriorSourceIndex = buildLIFTokenPriorSourceIndex(databaseNetworkObject)
			databaseNetworkObject.inferenceLIFTokenPriorCache[sourceToken] = calculateLIFTokenTransitionPrior(databaseNetworkObject, sourceToken)
		result = databaseNetworkObject.inferenceLIFTokenPriorCache[sourceToken]
	else:
		raise RuntimeError(inferenceLIFTokenPriorInvalidConfiguration)
	return result


def buildLIFTokenPriorSourceIndex(databaseNetworkObject):
	result = None
	if(inferenceReviewPatch15TokenTransitionPrior):
		sourceFeatureStride = databaseNetworkObject.f
		sourcesByToken = {}
		if(storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam):
			observedColumns = sorted(databaseNetworkObject.observedColumnsDictRAM.values(), key=lambda observedColumn: observedColumn.conceptIndex)
			for observedColumn in observedColumns:
				indexLIFTokenPriorSources(databaseNetworkObject, sourcesByToken, sourceFeatureStride, observedColumn.conceptIndex, observedColumn.listStoredSourceFeatureIndices())
		else:
			for sourceColumnIndex in GIAANNcmn_databaseNetworkFiles.listPersistedObservedColumnConceptIndices():
				indexLIFTokenPriorSources(databaseNetworkObject, sourcesByToken, sourceFeatureStride, sourceColumnIndex, GIAANNcmn_databaseNetworkFiles.listObservedColumnSourceFeatureIndices(sourceColumnIndex))
		result = sourceFeatureStride, sourcesByToken
	else:
		raise RuntimeError(inferenceLIFTokenPriorInvalidConfiguration)
	return result


def indexLIFTokenPriorSources(databaseNetworkObject, sourcesByToken, sourceFeatureStride, sourceColumnIndex, sourceFeatureIndices):
	if(inferenceReviewPatch15TokenTransitionPrior):
		if(not isinstance(sourceFeatureStride, Integral) or isinstance(sourceFeatureStride, bool) or sourceFeatureStride != databaseNetworkObject.f):
			raise RuntimeError(inferenceLIFTokenPriorInvalidDatabase)
		for sourceFeatureIndex in sourceFeatureIndices:
			sourceToken = getLIFTokenPriorToken(databaseNetworkObject, sourceColumnIndex, sourceFeatureIndex)
			if(sourceToken not in sourcesByToken):
				sourcesByToken[sourceToken] = array(inferenceLIFTokenPriorSourceIndexTypeCode)
			sourcesByToken[sourceToken].append(sourceColumnIndex*sourceFeatureStride + sourceFeatureIndex)
	else:
		raise RuntimeError(inferenceLIFTokenPriorInvalidConfiguration)
	return


def calculateLIFTokenTransitionPrior(databaseNetworkObject, sourceToken):
	result = None
	if(inferenceReviewPatch15TokenTransitionPrior):
		sourceFeatureStride, sourcesByToken = databaseNetworkObject.inferenceLIFTokenPriorSourceIndex
		if(sourceToken not in sourcesByToken):
			raise RuntimeError(inferenceLIFTokenPriorMissingEvidence)
		targetStrengths = {}
		for sourceKey in sourcesByToken[sourceToken]:
			sourceColumnIndex, sourceFeatureIndex = divmod(sourceKey, sourceFeatureStride)
			if(storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam):
				observedColumn = databaseNetworkObject.observedColumnsDictRAM[databaseNetworkObject.conceptColumnsList[sourceColumnIndex]]
				sourceConnections = observedColumn.getFeatureConnectionsForSourceFeature(sourceFeatureIndex, targetDevice=inferenceLIFTokenPriorLoadDevice, createMissing=False)
			else:
				sourceConnections = GIAANNcmn_databaseNetworkFiles.loadObservedColumnSourceFeatureConnectionsTensor(databaseNetworkObject, sourceColumnIndex, sourceFeatureIndex, inferenceLIFTokenPriorLoadDevice)
			accumulateLIFTokenPriorStrengths(databaseNetworkObject, sourceConnections, targetStrengths)
		result = targetStrengths
	else:
		raise RuntimeError(inferenceLIFTokenPriorInvalidConfiguration)
	return result


def accumulateLIFTokenPriorStrengths(databaseNetworkObject, sourceConnections, targetStrengths):
	if(inferenceReviewPatch15TokenTransitionPrior):
		if(not isinstance(sourceConnections, pt.Tensor) or sourceConnections.layout != pt.sparse_coo or sourceConnections.dim() != inferenceLIFTokenPriorConnectionRank or sourceConnections.sparse_dim() != inferenceLIFTokenPriorConnectionRank or not sourceConnections.is_floating_point()):
			raise RuntimeError(inferenceLIFTokenPriorInvalidConnections)
		if(sourceConnections.shape[inferenceLIFTokenPriorPropertyDimension] != databaseNetworkObject.arrayNumberOfProperties or sourceConnections.shape[inferenceLIFTokenPriorBranchDimension] != multipleDendriticBranchesNumber or sourceConnections.shape[inferenceLIFTokenPriorSegmentDimension] != arrayNumberOfSegments or sourceConnections.shape[inferenceLIFTokenPriorColumnDimension] > databaseNetworkObject.c or sourceConnections.shape[inferenceLIFTokenPriorFeatureDimension] > databaseNetworkObject.f):
			raise RuntimeError(inferenceLIFTokenPriorInvalidConnections)
		rawIndices = sourceConnections._indices()
		rawValues = sourceConnections._values()
		connectionShape = pt.tensor(sourceConnections.size(), dtype=rawIndices.dtype, device=rawIndices.device).unsqueeze(inferenceLIFTokenPriorEntryDimension)
		rawDirectMask = (rawIndices[inferenceLIFTokenPriorPropertyDimension] == databaseNetworkObject.arrayIndexPropertiesStrengthIndex) & (rawIndices[inferenceLIFTokenPriorSegmentDimension] == arrayIndexSegmentSoma)
		if((rawIndices < arrayIndexSegmentFirst).any() or (rawIndices >= connectionShape).any() or not pt.isfinite(rawValues[rawDirectMask]).all() or (rawValues[rawDirectMask] < inferenceLIFTokenPriorMinimumStrength).any()):
			raise RuntimeError(inferenceLIFTokenPriorInvalidConnections)
		sourceConnections = sourceConnections.coalesce()
		indices = sourceConnections.indices()
		values = sourceConnections.values()
		directMask = (indices[inferenceLIFTokenPriorPropertyDimension] == databaseNetworkObject.arrayIndexPropertiesStrengthIndex) & (indices[inferenceLIFTokenPriorSegmentDimension] == arrayIndexSegmentSoma)
		if(not pt.isfinite(values[directMask]).all() or (values[directMask] < inferenceLIFTokenPriorMinimumStrength).any()):
			raise RuntimeError(inferenceLIFTokenPriorInvalidConnections)
		positiveMask = directMask & (values > inferenceLIFTokenPriorMinimumStrength)
		for columnIndex, featureIndex, strength in zip(indices[inferenceLIFTokenPriorColumnDimension, positiveMask].tolist(), indices[inferenceLIFTokenPriorFeatureDimension, positiveMask].tolist(), values[positiveMask].tolist()):
			targetToken = getLIFTokenPriorToken(databaseNetworkObject, columnIndex, featureIndex)
			totalStrength = targetStrengths.get(targetToken, inferenceLIFTokenPriorMinimumStrength) + strength
			if(not math.isfinite(totalStrength)):
				raise RuntimeError(inferenceLIFTokenPriorInvalidConnections)
			targetStrengths[targetToken] = totalStrength
	else:
		raise RuntimeError(inferenceLIFTokenPriorInvalidConfiguration)
	return


def getLIFTokenPriorToken(databaseNetworkObject, columnIndex, featureIndex):
	result = None
	if(inferenceReviewPatch15TokenTransitionPrior):
		if(not isinstance(columnIndex, Integral) or isinstance(columnIndex, bool) or not isinstance(featureIndex, Integral) or isinstance(featureIndex, bool) or columnIndex < arrayIndexSegmentFirst or columnIndex >= databaseNetworkObject.c or featureIndex < arrayIndexSegmentFirst or featureIndex >= databaseNetworkObject.f):
			raise RuntimeError(inferenceLIFTokenPriorInvalidNeuron)
		if(featureIndex == featureIndexPrimeConceptNeuron):
			result = databaseNetworkObject.conceptColumnsList[columnIndex]
		else:
			result = databaseNetworkObject.conceptFeaturesList[featureIndex]
	else:
		raise RuntimeError(inferenceLIFTokenPriorInvalidConfiguration)
	return result


def validateLIFTokenPriorDatabase(databaseNetworkObject):
	if(inferenceReviewPatch15TokenTransitionPrior):
		if(not inferenceLeakyIntegrateAndFire or not predictionEnsureConnectedToPreviousPrediction or not enforceDirectConnectionsSANI or inferenceBeamSearch or (inferenceInferMissingFeatures and inferenceInferMissingFeaturesUpdate10UseGlobalFeaturePredictions) or (auxiliaryNeurons and auxiliaryNeuronsSimilar) or useTrainDuringInference or inferenceTrainFirstSequences or not isinstance(inferenceLIFTokenPriorExponent, (int, float)) or isinstance(inferenceLIFTokenPriorExponent, bool) or not math.isfinite(inferenceLIFTokenPriorExponent) or inferenceLIFTokenPriorExponent <= inferenceLIFTokenPriorMinimumStrength or inferenceLIFTokenPriorExponent > inferenceLIFTokenPriorMaximumExponent or not isinstance(inferenceLIFTokenPriorMinimumGain, (int, float)) or isinstance(inferenceLIFTokenPriorMinimumGain, bool) or not math.isfinite(inferenceLIFTokenPriorMinimumGain) or inferenceLIFTokenPriorMinimumGain < inferenceLIFTokenPriorMinimumStrength or inferenceLIFTokenPriorMinimumGain > inferenceLIFTokenPriorMaximumGain):
			raise RuntimeError(inferenceLIFTokenPriorInvalidConfiguration)
		if(databaseNetworkObject is None or not databaseNetworkObject.inferenceMode or len(databaseNetworkObject.conceptColumnsList) != databaseNetworkObject.c or len(databaseNetworkObject.conceptFeaturesList) != databaseNetworkObject.f or databaseNetworkObject.arrayIndexPropertiesStrengthIndex is None or databaseNetworkObject.inferenceLIFTokenPriorSource is None):
			raise RuntimeError(inferenceLIFTokenPriorInvalidDatabase)
		if(storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam and (not databaseNetworkObject.observedColumnsRAMLoaded or databaseNetworkObject.observedColumnsDictRAM is None)):
			raise RuntimeError(inferenceLIFTokenPriorInvalidDatabase)
	else:
		raise RuntimeError(inferenceLIFTokenPriorInvalidConfiguration)
	return
