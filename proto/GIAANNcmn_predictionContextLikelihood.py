"""GIAANNcmn_predictionContextLikelihood.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 BAI Research Pty Ltd (bairesearch.com.au)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN common prediction Context Likelihood

LIF readout using trained evidence from the configured source representations.

"""


from collections import OrderedDict
import math
import torch as pt

from GIAANNcmn_globalDefs import *
import GIAANNcmn_databaseNetworkFiles
import GIAANNcmn_predictionConstraints
if(inferenceColumnLikelihood):
	import GIAANNcmn_predictionColumnLikelihood


def initialiseContextLikelihoodSequence(databaseNetworkObject):
	if(inferenceContextLikelihood):
		if(not databaseNetworkObject.inferenceMode or databaseNetworkObject.c <= inferenceContextLikelihoodZero or databaseNetworkObject.f <= inferenceContextLikelihoodZero):
			raise RuntimeError(inferenceContextLikelihoodInvalidState)
		databaseNetworkObject.contextLikelihoodHistory = []
		databaseNetworkObject.contextLikelihoodColumnOrigins = {}
		databaseNetworkObject.contextLikelihoodPosition = -inferenceContextLikelihoodOne
		databaseNetworkObject.contextLikelihoodShape = (databaseNetworkObject.c, databaseNetworkObject.f)
		if(inferenceColumnLikelihood):
			GIAANNcmn_predictionColumnLikelihood.initialiseColumnLikelihoodSequence(databaseNetworkObject)
		if(not hasattr(databaseNetworkObject, inferenceContextLikelihoodFrequenciesAttribute)):
			databaseNetworkObject.contextLikelihoodFrequencies = OrderedDict()
	return


def prepareContextLikelihoodStep(databaseNetworkObject, sequenceWordIndex):
	if(inferenceContextLikelihood):
		validateContextLikelihoodState(databaseNetworkObject)
		if(type(sequenceWordIndex) is not int or sequenceWordIndex != databaseNetworkObject.contextLikelihoodPosition + inferenceContextLikelihoodOne):
			raise RuntimeError(inferenceContextLikelihoodInvalidState)
		databaseNetworkObject.contextLikelihoodPosition = sequenceWordIndex
		# An empty frame denotes an observed source with no stored outgoing evidence.
		databaseNetworkObject.contextLikelihoodHistory.append(None)
		del databaseNetworkObject.contextLikelihoodHistory[:-inferenceContextLikelihoodHistoryLength]
	return


def recordContextLikelihoodProxySources(databaseNetworkObject, sources, sequenceWordIndex):
	if(inferenceContextLikelihood):
		if(not inferenceInferMissingFeatures or predictionEnsureConnectedToPreviousPredictionStrict or not sources):
			raise RuntimeError(inferenceContextLikelihoodInvalidProxies)
		validateContextLikelihoodState(databaseNetworkObject)
		combinedConnections = None
		origins = set()
		neurons = set()
		totalWeight = inferenceContextLikelihoodZero
		for sourceColumnIndex, sourceFeatureIndex, sourceWeight, connections in sources:
			if(type(sourceColumnIndex) is not int or type(sourceFeatureIndex) is not int or not inferenceContextLikelihoodZero <= sourceColumnIndex < databaseNetworkObject.c or not inferenceContextLikelihoodZero <= sourceFeatureIndex < databaseNetworkObject.f or (sourceColumnIndex, sourceFeatureIndex) in neurons or type(sourceWeight) not in (int, float) or not math.isfinite(sourceWeight) or sourceWeight <= inferenceContextLikelihoodZero):
				raise RuntimeError(inferenceContextLikelihoodInvalidProxies)
			neurons.add((sourceColumnIndex, sourceFeatureIndex))
			totalWeight += sourceWeight
			validateContextLikelihoodConnections(databaseNetworkObject, connections)
			weightedConnections = connections.to(pt.float64) * sourceWeight
			if(inferenceInferMissingFeaturesSemantic and inferenceInferMissingFeaturesSemanticProjectColumns):
				import GIAANNnlp_predictionSemantic
				weightedConnections = GIAANNnlp_predictionSemantic.projectSemanticDonorTargets(databaseNetworkObject, weightedConnections, sourceColumnIndex)
				origins.add(sourceColumnIndex)
			if(combinedConnections is None):
				combinedConnections = weightedConnections
			else:
				combinedConnections = combinedConnections + weightedConnections
		if(not math.isclose(totalWeight, inferenceContextLikelihoodOne, rel_tol=inferenceContextLikelihoodProxyWeightTolerance, abs_tol=inferenceContextLikelihoodProxyWeightTolerance)):
			raise RuntimeError(inferenceContextLikelihoodInvalidProxies)
		recordContextLikelihoodSource(databaseNetworkObject, combinedConnections.coalesce(), sequenceWordIndex)
		if(origins):
			observedColumnIndex, observedFeatureIndex = databaseNetworkObject.inferenceSemanticObservedSource
			databaseNetworkObject.contextLikelihoodColumnOrigins.setdefault(observedColumnIndex, set()).update(origins)
	return


def recordContextLikelihoodSource(databaseNetworkObject, connections, sequenceWordIndex):
	if(inferenceContextLikelihood):
		validateContextLikelihoodState(databaseNetworkObject)
		if(type(sequenceWordIndex) is not int or sequenceWordIndex != databaseNetworkObject.contextLikelihoodPosition or not databaseNetworkObject.contextLikelihoodHistory or databaseNetworkObject.contextLikelihoodHistory[-inferenceContextLikelihoodOne] is not None):
			raise RuntimeError(inferenceContextLikelihoodInvalidState)
		validateContextLikelihoodConnections(databaseNetworkObject, connections)
		connections = connections.coalesce()
		indices = connections.indices()
		values = connections.values()
		segmentIndices = indices[inferenceContextLikelihoodSegmentDimension]
		mask = (segmentIndices >= arrayNumberOfSegmentsColumnDistance) & (values > inferenceContextLikelihoodZero)
		segmentIndices = segmentIndices[mask]
		keys = (segmentIndices * databaseNetworkObject.c + indices[inferenceContextLikelihoodColumnDimension, mask]) * databaseNetworkObject.f + indices[inferenceContextLikelihoodFeatureDimension, mask]
		# Undo the training distance multiplier to recover comparable occurrence counts.
		distances = (arrayIndexSegmentSoma - segmentIndices + inferenceContextLikelihoodOne).to(pt.float64)
		counts = values[mask].to(pt.float64) * (distances + trainConnectionStrengthProximityDistanceOffset).pow(trainConnectionStrengthProximityExponent) / trainConnectionStrengthProximityScale
		if(not pt.isfinite(counts).all() or (counts <= inferenceContextLikelihoodZero).any()):
			raise RuntimeError(inferenceContextLikelihoodInvalidConnections)
		databaseNetworkObject.contextLikelihoodHistory[-inferenceContextLikelihoodOne] = keys, counts
		if(inferenceColumnLikelihood):
			GIAANNcmn_predictionColumnLikelihood.recordColumnLikelihoodSource(databaseNetworkObject, connections, sequenceWordIndex)
	return


def scoreContextLikelihoodCandidates(databaseNetworkObject, columnIndices, featureIndices, activationValues, constraintState):
	result = columnIndices, featureIndices, activationValues
	if(inferenceContextLikelihood):
		validateContextLikelihoodState(databaseNetworkObject)
		validateContextLikelihoodCandidates(databaseNetworkObject, columnIndices, featureIndices, activationValues)
		columnIndices, featureIndices, activationValues = GIAANNcmn_predictionConstraints.filterColumnFeatureCandidatesByConstraint(databaseNetworkObject, columnIndices, featureIndices, activationValues, constraintState)
		result = columnIndices, featureIndices, activationValues
		if(columnIndices is not None and columnIndices.numel() > inferenceContextLikelihoodZero):
			counts = gatherContextLikelihoodCounts(databaseNetworkObject, columnIndices, featureIndices)
			frequencies = gatherContextLikelihoodFrequencies(databaseNetworkObject, columnIndices, featureIndices)
			if((predictionEnsureConnectedToPreviousPrediction and (counts[inferenceContextLikelihoodZero] <= inferenceContextLikelihoodZero).any()) or (frequencies <= inferenceContextLikelihoodZero).any()):
				raise RuntimeError(inferenceContextLikelihoodInvalidEvidence)
			baseProbabilities = frequencies / frequencies.sum()
			pseudoCounts = inferenceContextLikelihoodSmoothing * baseProbabilities
			# Indirect candidates may have zero immediate evidence; the same smoothing formula remains defined.
			logScores = (inferenceContextLikelihoodOne - inferenceContextLikelihoodPriorWeight) * activationValues.to(pt.float64).log() + inferenceContextLikelihoodPriorWeight * (counts[inferenceContextLikelihoodZero] + pseudoCounts).log()
			contextCounts = counts[inferenceContextLikelihoodOne:]
			contextProbabilities = (contextCounts + pseudoCounts) / (contextCounts.sum(dim=inferenceContextLikelihoodOne, keepdim=True) + inferenceContextLikelihoodSmoothing)
			logScores += inferenceContextLikelihoodContextWeight * (contextProbabilities / baseProbabilities).log().sum(dim=inferenceContextLikelihoodZero)
			if(inferenceColumnLikelihood):
				logScores += inferenceColumnLikelihoodWeight * GIAANNcmn_predictionColumnLikelihood.calculateColumnLikelihoodEvidence(databaseNetworkObject, columnIndices, featureIndices, baseProbabilities)
			# The shared scale cancels in token aggregation; keep the live LIF state untouched.
			scores = (logScores - logScores.max()).exp()
			if(inferenceLikelihoodCalibration):
				scores = scores.pow(inferenceLikelihoodCalibrationExponent)
			if(not pt.isfinite(scores).all() or (scores <= inferenceContextLikelihoodZero).any()):
				raise RuntimeError(inferenceContextLikelihoodInvalidEvidence)
			result = columnIndices, featureIndices, scores
	return result


def gatherContextLikelihoodCounts(databaseNetworkObject, columnIndices, featureIndices):
	if(inferenceContextLikelihood):
		result = pt.zeros((inferenceContextLikelihoodHistoryLength, columnIndices.numel()), dtype=pt.float64, device=columnIndices.device)
		for offset, frame in enumerate(reversed(databaseNetworkObject.contextLikelihoodHistory)):
			if(frame is not None):
				keys, counts = frame
				candidateKeys = ((arrayIndexSegmentSoma - offset) * databaseNetworkObject.c + columnIndices) * databaseNetworkObject.f + featureIndices
				if(keys.numel() > inferenceContextLikelihoodZero):
					positions = pt.searchsorted(keys, candidateKeys).clamp(max=keys.numel() - inferenceContextLikelihoodOne)
					present = keys[positions] == candidateKeys
					result[offset, present] = counts[positions[present]]
	else:
		raise RuntimeError(inferenceContextLikelihoodInvalidState)
	return result


def gatherContextLikelihoodFrequencies(databaseNetworkObject, columnIndices, featureIndices):
	if(inferenceContextLikelihood):
		result = pt.empty(columnIndices.shape, dtype=pt.float64, device=columnIndices.device)
		for columnIndex in columnIndices.unique().tolist():
			mask = columnIndices == columnIndex
			candidateKeys = featureIndices[mask]
			origins = databaseNetworkObject.contextLikelihoodColumnOrigins.get(columnIndex)
			if(origins is None):
				sourceColumns = [columnIndex]
			else:
				# Projected neurons inherit trained frequency evidence from their configured donor representations.
				sourceColumns = set(origins)
				if(not GIAANNcmn_databaseNetworkFiles.observedColumnHasConsistentPersistedMetadata(columnIndex)):
					raise RuntimeError(inferenceContextLikelihoodInvalidEvidence)
				if(GIAANNcmn_databaseNetworkFiles.observedColumnMetadataExists(columnIndex)):
					sourceColumns.add(columnIndex)
				sourceColumns = sorted(sourceColumns)
			candidateFrequencies = pt.zeros(candidateKeys.shape, dtype=pt.float64, device=columnIndices.device)
			for sourceColumnIndex in sourceColumns:
				keys, frequencies = getContextLikelihoodColumnFrequencies(databaseNetworkObject, sourceColumnIndex)
				if(keys.numel() > inferenceContextLikelihoodZero):
					positions = pt.searchsorted(keys, candidateKeys).clamp(max=keys.numel() - inferenceContextLikelihoodOne)
					present = keys[positions] == candidateKeys
					candidateFrequencies[present] = pt.maximum(candidateFrequencies[present], frequencies[positions[present]])
			if((candidateFrequencies <= inferenceContextLikelihoodZero).any()):
				raise RuntimeError(inferenceContextLikelihoodInvalidEvidence)
			result[mask] = candidateFrequencies
	else:
		raise RuntimeError(inferenceContextLikelihoodInvalidState)
	return result


def getContextLikelihoodColumnFrequencies(databaseNetworkObject, columnIndex):
	if(inferenceContextLikelihood):
		cache = databaseNetworkObject.contextLikelihoodFrequencies
		if(columnIndex not in cache):
			cache[columnIndex] = loadContextLikelihoodColumnFrequencies(databaseNetworkObject, columnIndex)
			if(len(cache) > inferenceContextLikelihoodFrequencyCacheColumns):
				cache.popitem(last=False)
		cache.move_to_end(columnIndex)
		result = cache[columnIndex]
	else:
		raise RuntimeError(inferenceContextLikelihoodInvalidState)
	return result


def loadContextLikelihoodColumnFrequencies(databaseNetworkObject, columnIndex):
	if(inferenceContextLikelihood):
		if(type(columnIndex) is not int or not inferenceContextLikelihoodZero <= columnIndex < databaseNetworkObject.c):
			raise RuntimeError(inferenceContextLikelihoodInvalidEvidence)
		neurons = GIAANNcmn_databaseNetworkFiles.loadObservedColumnFeatureNeuronsTensor(databaseNetworkObject, columnIndex, targetDevice=inferenceContextLikelihoodDevice).coalesce()
		if(neurons.dim() != inferenceContextLikelihoodNeuronRank or not neurons.is_floating_point()):
			raise RuntimeError(inferenceContextLikelihoodInvalidEvidence)
		indices = neurons.indices()
		values = neurons.values()
		mask = indices[inferenceContextLikelihoodPropertyDimension] == databaseNetworkObject.arrayIndexPropertiesStrengthIndex
		values = values[mask].to(pt.float64)
		if(not pt.isfinite(values).all() or (values < inferenceContextLikelihoodZero).any()):
			raise RuntimeError(inferenceContextLikelihoodInvalidEvidence)
		keys, inverse = pt.unique(indices[inferenceContextLikelihoodNeuronFeatureDimension, mask], sorted=True, return_inverse=True)
		if((keys < inferenceContextLikelihoodZero).any() or (keys >= databaseNetworkObject.f).any()):
			raise RuntimeError(inferenceContextLikelihoodInvalidEvidence)
		frequencies = pt.zeros(keys.shape, dtype=pt.float64)
		frequencies.scatter_reduce_(inferenceContextLikelihoodZero, inverse, values, reduce=inferenceContextLikelihoodMaximumReduction)
		result = keys, frequencies
	else:
		raise RuntimeError(inferenceContextLikelihoodInvalidState)
	return result


def validateContextLikelihoodState(databaseNetworkObject):
	if(inferenceContextLikelihood):
		if(not databaseNetworkObject.inferenceMode or not hasattr(databaseNetworkObject, inferenceContextLikelihoodHistoryAttribute) or databaseNetworkObject.contextLikelihoodShape != (databaseNetworkObject.c, databaseNetworkObject.f) or arrayNumberOfSegments * databaseNetworkObject.c * databaseNetworkObject.f > pt.iinfo(pt.long).max):
			raise RuntimeError(inferenceContextLikelihoodInvalidState)
	return


def validateContextLikelihoodConnections(databaseNetworkObject, connections):
	if(inferenceContextLikelihood):
		if(not isinstance(connections, pt.Tensor) or connections.layout != pt.sparse_coo or connections.dim() != inferenceContextLikelihoodConnectionRank or not connections.is_floating_point() or connections.device.type != inferenceContextLikelihoodDevice or tuple(connections.shape) != (inferenceContextLikelihoodOne, arrayNumberOfSegments, databaseNetworkObject.c, databaseNetworkObject.f)):
			raise RuntimeError(inferenceContextLikelihoodInvalidConnections)
		connections = connections.coalesce()
		indices = connections.indices()
		values = connections.values()
		if(not pt.isfinite(values).all() or (values < inferenceContextLikelihoodZero).any()):
			raise RuntimeError(inferenceContextLikelihoodInvalidConnections)
		for dimension, size in enumerate(connections.shape):
			if((indices[dimension] < inferenceContextLikelihoodZero).any() or (indices[dimension] >= size).any()):
				raise RuntimeError(inferenceContextLikelihoodInvalidConnections)
	return


def validateContextLikelihoodCandidates(databaseNetworkObject, columnIndices, featureIndices, activationValues):
	if(inferenceContextLikelihood):
		if(not all(isinstance(vector, pt.Tensor) and vector.dim() == inferenceContextLikelihoodOne and vector.device.type == inferenceContextLikelihoodDevice for vector in (columnIndices, featureIndices, activationValues)) or columnIndices.shape != featureIndices.shape or columnIndices.shape != activationValues.shape or columnIndices.dtype != pt.long or featureIndices.dtype != pt.long or not activationValues.is_floating_point()):
			raise RuntimeError(inferenceContextLikelihoodInvalidEvidence)
		if(not pt.isfinite(activationValues).all() or (activationValues <= inferenceContextLikelihoodZero).any() or (columnIndices < inferenceContextLikelihoodZero).any() or (columnIndices >= databaseNetworkObject.c).any() or (featureIndices < inferenceContextLikelihoodZero).any() or (featureIndices >= databaseNetworkObject.f).any()):
			raise RuntimeError(inferenceContextLikelihoodInvalidEvidence)
	return
