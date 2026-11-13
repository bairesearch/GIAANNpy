"""GIAANNcmn_contextRoutedBranchesCounts.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 BAI Research Pty Ltd (bairesearch.com.au)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN common context Routed Branches Counts

Read context-specific occurrence counts from native dendritic branch synapses.

"""


import torch as pt
import GIAANNcmn_contextRoutedBranches
from GIAANNcmn_globalDefs import *


def prepareInference(database):
	if(useContextRoutedBranches and inferenceContextRoutedBranchesCounts):
		database.contextRoutedBranchesCounts = (pt.empty(contextRoutedBranchesZero, dtype=pt.long), pt.empty(contextRoutedBranchesZero, dtype=pt.float64))
	return


def recordSource(database, connections):
	if(useContextRoutedBranches and inferenceContextRoutedBranchesCounts):
		if(not connections.is_sparse or connections.dim() != inferenceLeakyIntegrateAndFireNeuronTensorRank or connections.shape[contextRoutedBranchesZero] != multipleDendriticBranchesNumber):
			raise RuntimeError(contextRoutedBranchesInvalidInput)
		connections = connections.coalesce()
		indices,values = connections.indices(),connections.values()
		if(not pt.isfinite(values).all() or (values < contextRoutedBranchesZero).any()):
			raise RuntimeError(contextRoutedBranchesInvalidInput)
		mask = (indices[inferenceContextLikelihoodSegmentDimension] == arrayIndexSegmentSoma) & (values > contextRoutedBranchesZero)
		selectedIndices = pt.stack((indices[contextRoutedBranchesZero,mask],indices[inferenceContextLikelihoodColumnDimension,mask],indices[inferenceContextLikelihoodFeatureDimension,mask]))
		selected = GIAANNcmn_contextRoutedBranches.selectBranches(database, selectedIndices)
		keys = selectedIndices[contextRoutedBranchesOne,selected]*database.f+selectedIndices[-contextRoutedBranchesOne,selected]
		# The soma segment records source distance one; undo precisely the native training multiplier.
		counts = values[mask][selected].to(pt.float64)*(contextRoutedBranchesOne+trainConnectionStrengthProximityDistanceOffset)**trainConnectionStrengthProximityExponent/trainConnectionStrengthProximityScale
		keys,order = keys.sort()
		counts = counts[order]
		if(not pt.isfinite(counts).all() or (counts <= contextRoutedBranchesZero).any() or (keys[contextRoutedBranchesOne:] == keys[:-contextRoutedBranchesOne]).any()):
			raise RuntimeError(contextRoutedBranchesInvalidInput)
		database.contextRoutedBranchesCounts = keys,counts
	return


def scoreCandidates(database, columns, features, scores):
	result = scores
	if(useContextRoutedBranches and inferenceContextRoutedBranchesCounts):
		if(columns.dim() != contextRoutedBranchesOne or columns.shape != features.shape or columns.shape != scores.shape or columns.dtype != pt.long or features.dtype != pt.long or (columns < contextRoutedBranchesZero).any() or (columns >= database.c).any() or (features < contextRoutedBranchesZero).any() or (features >= database.f).any() or not pt.isfinite(scores).all() or (scores <= contextRoutedBranchesZero).any()):
			raise RuntimeError(contextRoutedBranchesInvalidInput)
		keys,counts = database.contextRoutedBranchesCounts
		if(keys.numel() > contextRoutedBranchesZero and columns.numel() > contextRoutedBranchesZero):
			candidateKeys = columns*database.f+features
			positions = pt.searchsorted(keys, candidateKeys).clamp(max=keys.numel()-contextRoutedBranchesOne)
			present = keys[positions] == candidateKeys
			if(bool(present.any().item())):
				selectedCounts = pt.zeros_like(scores, dtype=pt.float64)
				selectedCounts[present] = counts[positions[present]]
				result = selectedCounts+contextRoutedBranchesSmoothing*(scores/scores.sum())
		if(not pt.isfinite(result).all() or (result <= contextRoutedBranchesZero).any()):
			raise RuntimeError(contextRoutedBranchesInvalidInput)
	return result
