"""GIAANNcmn_branchVerificationFractional.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 BAI Research Pty Ltd (bairesearch.com.au)

MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN common branch Verification Fractional

Bound the distinct branch owners of each incoming segment connection, including pending updates.

"""

import math
import torch as pt
from GIAANNcmn_globalDefs import *


def filterTensor(existing, updates, strengthIndex):
	result = updates
	if(trainVerifyConnectionNonexistentAcrossBranchesFractional):
		if(not pt.is_tensor(existing) or not pt.is_tensor(updates) or existing.layout != pt.sparse_coo or updates.layout != pt.sparse_coo):
			raise ValueError(branchVerifyFractionalInvalidInput)
		if(tuple(existing.size()) != tuple(updates.size()) or existing.dim() not in (branchVerifySourceRank, branchVerifyProspectiveRank) or existing.device.type != branchVerifyCPU or updates.device.type != branchVerifyCPU or existing.dtype != pt.float32 or updates.dtype != pt.float32 or existing.size(branchVerifyPropertyAxis) != branchVerifyOne or existing.size(branchVerifyBranchAxis) != multipleDendriticBranchesNumber or existing.size(branchVerifySegmentAxis) != arrayNumberOfSegments or type(strengthIndex) is not int or strengthIndex != branchVerifyZero):
			raise ValueError(branchVerifyFractionalInvalidInput)
		existing, updates = existing.coalesce(), updates.coalesce()
		if(not bool(pt.isfinite(existing.values()).all()) or not bool(pt.isfinite(updates.values()).all())):
			raise ValueError(branchVerifyFractionalInvalidInput)
		indices, values = updates.indices(), updates.values()
		positive = existing.values() > branchVerifyZero
		existingIndices = existing.indices()[:, positive]
		index = createIndex(connectionKeys(existingIndices, existing.size()), existingIndices[branchVerifyBranchAxis])
		keep = keepUpdates(index, connectionKeys(indices, updates.size()), indices[branchVerifyBranchAxis], values > branchVerifyZero)
		result = pt.sparse_coo_tensor(indices[:, keep], values[keep], updates.size(), dtype=updates.dtype).coalesce()
	return result


def keepUpdates(index, keys, branches, positive):
	result = None
	if(trainVerifyConnectionNonexistentAcrossBranchesFractional):
		validateConnections(keys, branches)
		if(not pt.is_tensor(positive) or positive.dtype != pt.bool or positive.device.type != branchVerifyCPU or positive.shape != keys.shape):
			raise ValueError(branchVerifyFractionalInvalidInput)
		existingPairs, existingKeys, existingCounts = index
		pairs = keys*multipleDendriticBranchesNumber + branches
		locations, existing = lookup(existingPairs, pairs)
		# Reserve capacity for each distinct new branch once, even when several occurrences propose it.
		newPairs = pt.unique(pairs[positive & ~existing], sorted=True)
		newKeys = pt.div(newPairs, multipleDendriticBranchesNumber, rounding_mode=branchVerifyRounding)
		uniqueKeys, counts = pt.unique_consecutive(newKeys, return_counts=True)
		locations, found = lookup(existingKeys, uniqueKeys)
		available = pt.full_like(uniqueKeys, branchVerifyFractionalMaximumOwners)
		available[found] -= existingCounts[locations[found]]
		starts = counts.cumsum(branchVerifyZero) - counts
		ranks = pt.arange(newPairs.numel(), dtype=pt.long) - pt.repeat_interleave(starts, counts)
		# Lowest proposed branch IDs win simultaneous competition for remaining slots.
		admitted = newPairs[ranks < pt.repeat_interleave(available, counts)]
		locations, accepted = lookup(admitted, pairs)
		result = existing | ~positive | accepted
	return result


def createIndex(keys, branches):
	result = None
	if(trainVerifyConnectionNonexistentAcrossBranchesFractional):
		validateConnections(keys, branches)
		pairs = pt.unique(keys*multipleDendriticBranchesNumber + branches, sorted=True)
		uniqueKeys, counts = pt.unique_consecutive(pt.div(pairs, multipleDendriticBranchesNumber, rounding_mode=branchVerifyRounding), return_counts=True)
		if(bool(pt.any(counts > branchVerifyFractionalMaximumOwners))):
			raise RuntimeError(branchVerifyFractionalExistingOverLimit)
		result = pairs, uniqueKeys, counts
	return result


def connectionKeys(indices, shape):
	result = None
	if(trainVerifyConnectionNonexistentAcrossBranchesFractional):
		if(not pt.is_tensor(indices) or indices.dtype != pt.long or indices.device.type != branchVerifyCPU or indices.dim() != branchVerifyTwo or len(shape) not in (branchVerifySourceRank, branchVerifyProspectiveRank) or indices.size(branchVerifyZero) != len(shape) or any(size <= branchVerifyZero for size in shape) or math.prod(shape[branchVerifySegmentAxis:])*multipleDendriticBranchesNumber > branchVerifyInt64Maximum):
			raise ValueError(branchVerifyFractionalInvalidInput)
		if(shape[branchVerifyPropertyAxis] != branchVerifyOne or shape[branchVerifyBranchAxis] != multipleDendriticBranchesNumber or shape[branchVerifySegmentAxis] != arrayNumberOfSegments):
			raise ValueError(branchVerifyFractionalInvalidInput)
		if(bool(pt.any(indices < branchVerifyZero)) or bool(pt.any(indices >= pt.tensor(shape, dtype=pt.long).unsqueeze(branchVerifyOne)))):
			raise ValueError(branchVerifyFractionalInvalidInput)
		if(len(shape) == branchVerifySourceRank):
			result = (indices[branchVerifySegmentAxis]*shape[branchVerifyTargetConceptAxis] + indices[branchVerifyTargetConceptAxis])*shape[branchVerifyTargetFeatureAxis] + indices[branchVerifyTargetFeatureAxis]
		else:
			result = ((indices[branchVerifySegmentAxis]*shape[branchVerifyTargetConceptAxis] + indices[branchVerifyTargetConceptAxis])*shape[branchVerifyTargetFeatureAxis] + indices[branchVerifyTargetFeatureAxis])*shape[branchVerifyProspectiveTargetFeatureAxis] + indices[branchVerifyProspectiveTargetFeatureAxis]
	return result


def validateConnections(keys, branches):
	if(trainVerifyConnectionNonexistentAcrossBranchesFractional):
		if(not pt.is_tensor(keys) or not pt.is_tensor(branches) or keys.dtype != pt.long or branches.dtype != pt.long or keys.device.type != branchVerifyCPU or branches.device.type != branchVerifyCPU or keys.dim() != branchVerifyOne or branches.shape != keys.shape):
			raise ValueError(branchVerifyFractionalInvalidInput)
		if(bool(pt.any(keys < branchVerifyZero)) or bool(pt.any(keys > (branchVerifyInt64Maximum-multipleDendriticBranchesNumber+branchVerifyOne)//multipleDendriticBranchesNumber)) or bool(pt.any(branches < branchVerifyZero)) or bool(pt.any(branches >= multipleDendriticBranchesNumber))):
			raise ValueError(branchVerifyFractionalInvalidInput)
	return


def lookup(keys, query):
	result = None
	if(trainVerifyConnectionNonexistentAcrossBranchesFractional):
		positions = pt.searchsorted(keys, query)
		found = positions < keys.numel()
		found[found.clone()] = keys[positions[found]] == query[found]
		result = positions, found
	return result
