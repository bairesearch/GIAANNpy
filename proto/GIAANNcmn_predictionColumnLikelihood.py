"""GIAANNcmn_predictionColumnLikelihood.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 BAI Research Pty Ltd (bairesearch.com.au)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN common prediction Column Likelihood

Causal readout of trained column-segment evidence beyond the feature receptive field.

"""


import torch as pt
from GIAANNcmn_globalDefs import *


def initialiseColumnLikelihoodSequence(databaseNetworkObject):
	if(inferenceColumnLikelihood):
		if(not databaseNetworkObject.inferenceMode or type(databaseNetworkObject.c) is not int or type(databaseNetworkObject.f) is not int or databaseNetworkObject.c <= inferenceColumnLikelihoodZero or databaseNetworkObject.f <= inferenceColumnLikelihoodZero):
			raise RuntimeError(inferenceColumnLikelihoodInvalidState)
		databaseNetworkObject.columnLikelihoodHistory = []
		databaseNetworkObject.columnLikelihoodColumn = None
		databaseNetworkObject.columnLikelihoodEpoch = inferenceColumnLikelihoodZero
		databaseNetworkObject.columnLikelihoodPosition = -inferenceColumnLikelihoodOne
		databaseNetworkObject.columnLikelihoodRecordedPosition = -inferenceColumnLikelihoodOne
	return


def prepareColumnLikelihoodStep(databaseNetworkObject, sequenceWordIndex, columnIndex):
	if(inferenceColumnLikelihood):
		if(type(sequenceWordIndex) is not int or sequenceWordIndex != databaseNetworkObject.columnLikelihoodPosition + inferenceColumnLikelihoodOne or type(columnIndex) is not int or not inferenceColumnLikelihoodZero <= columnIndex < databaseNetworkObject.c):
			raise RuntimeError(inferenceColumnLikelihoodInvalidState)
		if(databaseNetworkObject.columnLikelihoodColumn is not None and columnIndex != databaseNetworkObject.columnLikelihoodColumn):
			databaseNetworkObject.columnLikelihoodEpoch += inferenceColumnLikelihoodOne
		databaseNetworkObject.columnLikelihoodColumn = columnIndex
		databaseNetworkObject.columnLikelihoodPosition = sequenceWordIndex
		databaseNetworkObject.columnLikelihoodHistory = [frame for frame in databaseNetworkObject.columnLikelihoodHistory if databaseNetworkObject.columnLikelihoodEpoch - frame[inferenceColumnLikelihoodEpochField] < arrayNumberOfSegmentsColumnDistance]
	return


def recordColumnLikelihoodSource(databaseNetworkObject, connections, sequenceWordIndex):
	if(inferenceColumnLikelihood):
		if(type(sequenceWordIndex) is not int or sequenceWordIndex != databaseNetworkObject.columnLikelihoodPosition or sequenceWordIndex <= databaseNetworkObject.columnLikelihoodRecordedPosition):
			raise RuntimeError(inferenceColumnLikelihoodInvalidState)
		import GIAANNcmn_predictionContextLikelihood
		GIAANNcmn_predictionContextLikelihood.validateContextLikelihoodConnections(databaseNetworkObject, connections)
		connections = connections.coalesce()
		indices = connections.indices()
		values = connections.values()
		mask = (indices[inferenceContextLikelihoodSegmentDimension] < arrayNumberOfSegmentsColumnDistance) & (values > inferenceColumnLikelihoodZero)
		keys = (indices[inferenceContextLikelihoodSegmentDimension, mask] * databaseNetworkObject.c + indices[inferenceContextLikelihoodColumnDimension, mask]) * databaseNetworkObject.f + indices[inferenceContextLikelihoodFeatureDimension, mask]
		weights = values[mask].to(pt.float64)
		databaseNetworkObject.columnLikelihoodHistory.append((databaseNetworkObject.columnLikelihoodEpoch, sequenceWordIndex, keys, weights))
		databaseNetworkObject.columnLikelihoodRecordedPosition = sequenceWordIndex
	return


def calculateColumnLikelihoodEvidence(databaseNetworkObject, columnIndices, featureIndices, baseProbabilities):
	if(inferenceColumnLikelihood):
		import GIAANNcmn_predictionContextLikelihood
		GIAANNcmn_predictionContextLikelihood.validateContextLikelihoodCandidates(databaseNetworkObject, columnIndices, featureIndices, baseProbabilities)
		if(databaseNetworkObject.columnLikelihoodPosition != databaseNetworkObject.contextLikelihoodPosition or columnIndices.shape != featureIndices.shape or columnIndices.shape != baseProbabilities.shape or not pt.isfinite(baseProbabilities).all() or (baseProbabilities <= inferenceColumnLikelihoodZero).any()):
			raise RuntimeError(inferenceColumnLikelihoodInvalidState)
		result = pt.zeros_like(baseProbabilities)
		frameCount = inferenceColumnLikelihoodZero
		for epoch, position, keys, weights in databaseNetworkObject.columnLikelihoodHistory:
			# Feature-distance evidence is already scored by the local readout; use only older sources here.
			if(databaseNetworkObject.columnLikelihoodPosition - position >= inferenceContextLikelihoodHistoryLength and keys.numel() > inferenceColumnLikelihoodZero):
				distances = databaseNetworkObject.columnLikelihoodEpoch - epoch + (columnIndices != databaseNetworkObject.columnLikelihoodColumn).long()
				segments = arrayIndexSegmentLastColumn - distances
				query = (segments * databaseNetworkObject.c + columnIndices) * databaseNetworkObject.f + featureIndices
				locations = pt.searchsorted(keys, query).clamp(max=keys.numel() - inferenceColumnLikelihoodOne)
				present = (segments >= inferenceColumnLikelihoodZero) & (keys[locations] == query)
				evidence = pt.where(present, weights[locations], inferenceColumnLikelihoodZero)
				result += pt.log1p(evidence / baseProbabilities)
				frameCount += inferenceColumnLikelihoodOne
		# Averaging bounds the gain from longer columns without changing the underlying graph or LIF state.
		result /= max(frameCount, inferenceColumnLikelihoodOne)
		if(not pt.isfinite(result).all() or (result < inferenceColumnLikelihoodZero).any()):
			raise RuntimeError(inferenceColumnLikelihoodInvalidEvidence)
	else:
		raise RuntimeError(inferenceColumnLikelihoodInvalidConfiguration)
	return result
