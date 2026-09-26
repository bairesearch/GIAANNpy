"""GIAANNcmn_predictionSparse.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 BAI Research Pty Ltd (bairesearch.com.au)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN common prediction sparse activation lookups and updates for inference.

"""

import torch as pt

from GIAANNcmn_globalDefs import *


def selectInferenceCandidateActivationState(stateFeatures, eligibleKeys):
	result = None
	if(optimiseInferenceCandidateLookup):
		activation = validateInferenceCandidateActivationState(stateFeatures, eligibleKeys)
		indices = activation.indices()
		values = activation.values()
		selectedPositions = pt.empty((inferenceCandidateLookupZero,), dtype=pt.long, device=activation.device)
		if(activation._nnz() > inferenceCandidateLookupZero and eligibleKeys.numel() > inferenceCandidateLookupZero):
			maxConcepts = activation.shape[inferenceLeakyIntegrateAndFireConceptDimension]
			maxFeatures = activation.shape[inferenceLeakyIntegrateAndFireFeatureDimension]
			neuronsPerBranch = maxConcepts*maxFeatures
			keys = pt.unique_consecutive(eligibleKeys)
			branches = pt.div(keys, neuronsPerBranch, rounding_mode=inferenceCandidateLookupRoundingMode)
			neurons = pt.remainder(keys, neuronsPerBranch)
			segments = [arrayIndexSegmentSoma]
			if(useSANIcolumns or useSANIfeaturesAndColumns):
				segments.append(arrayIndexSegmentLastColumn)
				if(inferenceReviewPatch11ProspectiveColumnScoring and arrayIndexSegmentLastColumn > inferenceCandidateLookupZero):
					segments.append(arrayIndexSegmentLastColumn-inferenceCandidateLookupStep)
			segmentIndices = pt.tensor(segments, dtype=pt.long, device=activation.device)
			queryKeys = (branches.unsqueeze(inferenceCandidateLookupVectorDimension)*arrayNumberOfSegments+segmentIndices.unsqueeze(inferenceCandidateLookupEntryDimension))*neuronsPerBranch+neurons.unsqueeze(inferenceCandidateLookupVectorDimension)
			#Coalesced COO coordinates already have the same order as these scalar keys.
			stateKeys = ((indices[inferenceLeakyIntegrateAndFireBranchDimension]*arrayNumberOfSegments+indices[inferenceLeakyIntegrateAndFireSegmentDimension])*maxConcepts+indices[inferenceLeakyIntegrateAndFireConceptDimension])*maxFeatures+indices[inferenceLeakyIntegrateAndFireFeatureDimension]
			positions = pt.searchsorted(stateKeys, queryKeys)
			inRange = positions < stateKeys.numel()
			positions = positions.clamp(max=stateKeys.numel()-inferenceCandidateLookupStep)
			matches = inRange & (stateKeys[positions] == queryKeys)
			#Queries are unique; sorting their matched positions preserves canonical COO order.
			selectedPositions = pt.sort(positions[matches]).values
		result = pt.sparse_coo_tensor(indices.index_select(inferenceCandidateLookupEntryDimension, selectedPositions), values.index_select(inferenceCandidateLookupVectorDimension, selectedPositions), size=activation.size(), dtype=activation.dtype, device=activation.device, is_coalesced=True)
	else:
		raise RuntimeError(inferenceCandidateLookupInvalidConfiguration)
	return result


def addInferenceSparseSingleActivation(stateFeatures, updateIndices, updateValues):
	result = None
	if(optimiseInferenceSparseBurst):
		if(not inferenceLeakyIntegrateAndFire or not isinstance(stateFeatures, pt.Tensor) or not stateFeatures.is_sparse or not stateFeatures.is_floating_point() or stateFeatures.dim() != inferenceLeakyIntegrateAndFireNeuronTensorRank or stateFeatures.sparse_dim() != inferenceLeakyIntegrateAndFireNeuronTensorRank or stateFeatures.device.type != inferenceSparseBurstDeviceType):
			raise RuntimeError(inferenceSparseBurstInvalidState)
		if(not isinstance(updateIndices, pt.Tensor) or not isinstance(updateValues, pt.Tensor) or updateIndices.layout != pt.strided or updateValues.layout != pt.strided or updateIndices.shape != (inferenceLeakyIntegrateAndFireNeuronTensorRank, inferenceSparseBurstSingleEntry) or updateValues.shape != (inferenceSparseBurstSingleEntry,) or updateIndices.dtype != pt.long or updateValues.dtype != stateFeatures.dtype or updateIndices.device != stateFeatures.device or updateValues.device != stateFeatures.device):
			raise RuntimeError(inferenceSparseBurstInvalidUpdate)
		query = updateIndices[:, inferenceSparseBurstZero]
		shape = pt.tensor(stateFeatures.size(), dtype=pt.long, device=stateFeatures.device)
		if(bool(pt.any(query < inferenceSparseBurstZero).item()) or bool(pt.any(query >= shape).item()) or not bool(pt.all(pt.isfinite(updateValues)).item()) or bool(pt.any(updateValues < inferenceSparseBurstZero).item())):
			raise RuntimeError(inferenceSparseBurstInvalidUpdate)
		activation = stateFeatures.coalesce()
		indices = activation.indices()
		values = activation.values()
		start = inferenceSparseBurstZero
		end = activation._nnz()
		#Narrow a lexicographically sorted COO range one coordinate at a time without encoding every stored entry.
		for dimension in range(inferenceLeakyIntegrateAndFireNeuronTensorRank):
			coordinates = indices[dimension, start:end].contiguous()
			lower = int(pt.searchsorted(coordinates, query[dimension]).item())
			upper = int(pt.searchsorted(coordinates, query[dimension], right=True).item())
			end = start+upper
			start += lower
		if(end > start):
			if(end-start != inferenceSparseBurstSingleEntry):
				raise RuntimeError(inferenceSparseBurstInvalidState)
			resultIndices = indices
			resultValues = values.clone()
			resultValues[start] += updateValues[inferenceSparseBurstZero]
		else:
			resultIndices = pt.cat((indices[:, :start], updateIndices, indices[:, start:]), dim=inferenceSparseBurstEntryDimension)
			resultValues = pt.cat((values[:start], updateValues, values[start:]), dim=inferenceSparseBurstVectorDimension)
		result = pt.sparse_coo_tensor(resultIndices, resultValues, size=activation.size(), dtype=activation.dtype, device=activation.device, is_coalesced=True)
	else:
		raise RuntimeError(inferenceSparseBurstInvalidState)
	return result


def validateInferenceCandidateActivationState(stateFeatures, eligibleKeys):
	result = None
	if(optimiseInferenceCandidateLookup):
		if(not inferenceLeakyIntegrateAndFire or multipleDendriticBranchesBinaryTree or algorithmMatrixSANIenforceRequirement != inferenceCandidateLookupRequiredCondition):
			raise RuntimeError(inferenceCandidateLookupInvalidConfiguration)
		if(not isinstance(stateFeatures, pt.Tensor) or not stateFeatures.is_sparse or not stateFeatures.is_floating_point() or stateFeatures.dim() != inferenceLeakyIntegrateAndFireNeuronTensorRank or stateFeatures.sparse_dim() != inferenceLeakyIntegrateAndFireNeuronTensorRank):
			raise RuntimeError(inferenceCandidateLookupInvalidState)
		if(stateFeatures.shape[inferenceLeakyIntegrateAndFireBranchDimension] != multipleDendriticBranchesNumber or stateFeatures.shape[inferenceLeakyIntegrateAndFireSegmentDimension] != arrayNumberOfSegments or min(stateFeatures.shape) <= inferenceCandidateLookupZero):
			raise RuntimeError(inferenceCandidateLookupInvalidState)
		if(arrayIndexSegmentSoma != arrayNumberOfSegments-inferenceCandidateLookupStep):
			raise RuntimeError(inferenceCandidateLookupInvalidState)
		if(useSANIcolumns or useSANIfeaturesAndColumns):
			if(arrayIndexSegmentLastColumn < inferenceCandidateLookupZero or arrayIndexSegmentLastColumn >= arrayIndexSegmentSoma):
				raise RuntimeError(inferenceCandidateLookupInvalidState)
		maxConcepts = stateFeatures.shape[inferenceLeakyIntegrateAndFireConceptDimension]
		maxFeatures = stateFeatures.shape[inferenceLeakyIntegrateAndFireFeatureDimension]
		keyLimit = multipleDendriticBranchesNumber*maxConcepts*maxFeatures
		if(keyLimit*arrayNumberOfSegments > pt.iinfo(pt.long).max):
			raise RuntimeError(inferenceCandidateLookupKeyOverflow)
		if(not isinstance(eligibleKeys, pt.Tensor) or eligibleKeys.dim() != inferenceCandidateLookupStep or eligibleKeys.dtype != pt.long or eligibleKeys.device != stateFeatures.device):
			raise RuntimeError(inferenceCandidateLookupInvalidKeys)
		if(eligibleKeys.numel() > inferenceCandidateLookupZero):
			if(bool(pt.any(eligibleKeys < inferenceCandidateLookupZero).item()) or bool(pt.any(eligibleKeys >= keyLimit).item()) or bool(pt.any(eligibleKeys[inferenceCandidateLookupStep:] < eligibleKeys[:-inferenceCandidateLookupStep]).item())):
				raise RuntimeError(inferenceCandidateLookupInvalidKeys)
		result = stateFeatures.coalesce()
		if(result._nnz() > inferenceCandidateLookupZero):
			if(not bool(pt.all(pt.isfinite(result.values())).item()) or bool(pt.any(result.values() < inferenceCandidateLookupZero).item())):
				raise RuntimeError(inferenceCandidateLookupInvalidState)
			shape = pt.tensor(result.shape, dtype=pt.long, device=result.device)
			if(bool(pt.any(result.indices().amin(dim=inferenceCandidateLookupEntryDimension) < inferenceCandidateLookupZero).item()) or bool(pt.any(result.indices().amax(dim=inferenceCandidateLookupEntryDimension) >= shape).item())):
				raise RuntimeError(inferenceCandidateLookupInvalidState)
	else:
		raise RuntimeError(inferenceCandidateLookupInvalidConfiguration)
	return result
