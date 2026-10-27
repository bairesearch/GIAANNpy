"""GIAANNcmn_predictionSequenceContext.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 BAI Research Pty Ltd (bairesearch.com.au)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN common prediction Sequence Context

Causal joint-context readout over existing eligible GIAANN neurons

"""

import numpy as np
import torch as pt

from GIAANNcmn_globalDefs import *
import GIAANNcmn_sequenceContext
import GIAANNcmn_predictionConstraints
import GIAANNcmn_predictionTokenAggregation


def initialiseSequenceContextInference(databaseNetworkObject):
	if(useSequenceContext):
		if(not databaseNetworkObject.inferenceMode or databaseNetworkObject.f <= sequenceContextOne or len(databaseNetworkObject.conceptFeaturesList) != databaseNetworkObject.f):
			raise RuntimeError(sequenceContextInvalidState)
		if(not hasattr(databaseNetworkObject, sequenceContextModelAttribute)):
			databaseNetworkObject.sequenceContextModel = GIAANNcmn_sequenceContext.loadSequenceContextModel(databaseNetworkObject)
		databaseNetworkObject.sequenceContextHistory = []
		databaseNetworkObject.sequenceContextPosition = -sequenceContextOne
		databaseNetworkObject.sequenceContextVocabularySize = databaseNetworkObject.f
	return


def prepareSequenceContextStep(databaseNetworkObject, sequenceWordIndex, sourceColumnIndex, sourceFeatureIndex):
	if(useSequenceContext):
		validateSequenceContextState(databaseNetworkObject)
		if(type(sequenceWordIndex) is not int or sequenceWordIndex != databaseNetworkObject.sequenceContextPosition + sequenceContextOne):
			raise RuntimeError(sequenceContextInvalidState)
		# Position zero has no preceding observation. Every subsequent source is already observed or generated.
		if(sequenceWordIndex > sequenceContextZero):
			if(type(sourceColumnIndex) is not int or type(sourceFeatureIndex) is not int or not sequenceContextZero <= sourceColumnIndex < databaseNetworkObject.c or not sequenceContextZero <= sourceFeatureIndex < databaseNetworkObject.f):
				raise ValueError(sequenceContextInvalidState)
			tokenIndex = sourceFeatureIndex
			if(sourceFeatureIndex == featureIndexPrimeConceptNeuron):
				tokenIndex = databaseNetworkObject.conceptFeaturesDict[databaseNetworkObject.conceptColumnsList[sourceColumnIndex]]
			if(not sequenceContextZero < tokenIndex < databaseNetworkObject.f):
				raise ValueError(sequenceContextInvalidState)
			databaseNetworkObject.sequenceContextHistory.append(tokenIndex)
			del databaseNetworkObject.sequenceContextHistory[:-sequenceContextMaximumLength]
		databaseNetworkObject.sequenceContextPosition = sequenceWordIndex
	return


def scoreSequenceContextCandidates(databaseNetworkObject, columnIndices, featureIndices, activationValues, constraintState):
	result = columnIndices, featureIndices, activationValues
	if(useSequenceContext):
		validateSequenceContextState(databaseNetworkObject)
		GIAANNcmn_predictionTokenAggregation.validateLIFTokenAggregationCandidates(databaseNetworkObject, columnIndices, featureIndices, activationValues)
		columnIndices, featureIndices, activationValues = GIAANNcmn_predictionConstraints.filterColumnFeatureCandidatesByConstraint(databaseNetworkObject, columnIndices, featureIndices, activationValues, constraintState)
		result = columnIndices, featureIndices, activationValues
		if(columnIndices is not None and columnIndices.numel() > sequenceContextZero and len(databaseNetworkObject.sequenceContextHistory) >= sequenceContextMinimumLength):
			tokenIndices = featureIndices.clone()
			primeMask = featureIndices == featureIndexPrimeConceptNeuron
			if(primeMask.any()):
				tokenIndices[primeMask] = pt.tensor([databaseNetworkObject.conceptFeaturesDict[databaseNetworkObject.conceptColumnsList[column]] for column in columnIndices[primeMask].tolist()], dtype=pt.long)
			uniqueTokens, inverse = pt.unique(tokenIndices, sorted=True, return_inverse=True)
			values = activationValues.to(pt.float64)
			totals = pt.zeros(uniqueTokens.shape, dtype=pt.float64)
			totals.scatter_add_(sequenceContextZero, inverse, values)
			probabilities = (totals / totals.sum()).numpy()
			evidence = gatherSequenceContextEvidence(databaseNetworkObject.sequenceContextModel, databaseNetworkObject.sequenceContextHistory, uniqueTokens.numpy())
			changed = False
			for offset, counts in enumerate(evidence):
				if(counts.sum() > sequenceContextZero):
					alpha = max(sequenceContextMinimumSmoothing, sequenceContextInitialSmoothing * sequenceContextSmoothingDecay ** offset)
					probabilities = (counts + alpha * probabilities) / (counts.sum() + alpha)
					changed = True
			if(changed):
				# Preserve each token's relative neuron scores and all prior connectivity/firing constraints.
				scores = pt.from_numpy(probabilities)[inverse] * (values / totals[inverse])
				if(not pt.isfinite(scores).all() or (scores <= sequenceContextZero).any()):
					raise RuntimeError(sequenceContextInvalidCandidates)
				result = columnIndices, featureIndices, scores
	return result


def gatherSequenceContextEvidence(model, history, candidateTokens):
	result = []
	if(useSequenceContext):
		if(not sequenceContextMinimumLength <= len(history) <= sequenceContextMaximumLength or any(type(token) is not int or not sequenceContextZero < token < sequenceContextOne << sequenceContextBits for token in history) or candidateTokens.ndim != sequenceContextOne or not len(candidateTokens) or not np.issubdtype(candidateTokens.dtype, np.integer) or candidateTokens.min() <= sequenceContextZero or candidateTokens.max() >= sequenceContextOne << sequenceContextBits or (candidateTokens[sequenceContextOne:] <= candidateTokens[:-sequenceContextOne]).any()):
			raise ValueError(sequenceContextInvalidState)
		key = sequenceContextZero
		for token in history[-sequenceContextMinimumLength:]:
			key = (key << sequenceContextBits) | token
		location = int(np.searchsorted(model.keys, np.uint64(key)))
		matches = np.empty(sequenceContextZero, dtype=np.int64)
		starts = np.empty(sequenceContextZero, dtype=np.int64)
		if(location < len(model.keys) and model.keys[location] == key):
			matches = model.positions[int(model.offsets[location]):int(model.offsets[location + sequenceContextOne])].astype(np.int64)
			starts = model.boundaries[np.searchsorted(model.boundaries, matches.astype(np.uint64), side=sequenceContextSearchRight) - sequenceContextOne].astype(np.int64)
		for length in range(sequenceContextMinimumLength, len(history) + sequenceContextOne):
			if(length > sequenceContextMinimumLength):
				withinSequence = matches - starts >= length
				matches = matches[withinSequence]
				starts = starts[withinSequence]
				equal = model.tokens[matches - length] == history[-length]
				matches = matches[equal]
				starts = starts[equal]
			counts = np.zeros(len(candidateTokens), dtype=sequenceContextFloatDtype)
			if(len(matches)):
				outcomes, occurrences = np.unique(model.tokens[matches], return_counts=True)
				indices = np.searchsorted(candidateTokens, outcomes).clip(max=len(candidateTokens) - sequenceContextOne)
				present = candidateTokens[indices] == outcomes
				counts[indices[present]] = occurrences[present]
			result.append(counts)
	return result


def validateSequenceContextState(databaseNetworkObject):
	if(useSequenceContext):
		if(not databaseNetworkObject.inferenceMode or not hasattr(databaseNetworkObject, sequenceContextModelAttribute) or not hasattr(databaseNetworkObject, sequenceContextHistoryAttribute) or databaseNetworkObject.f != databaseNetworkObject.sequenceContextVocabularySize):
			raise RuntimeError(sequenceContextInvalidState)
	return
