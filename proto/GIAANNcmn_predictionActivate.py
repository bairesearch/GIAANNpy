"""GIAANNcmn_predictionActivate.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 BAI Research Pty Ltd (bairesearch.com.au)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN common Prediction Activate

"""

import torch as pt

from GIAANNcmn_globalDefs import *
import GIAANNcmn_sparseTensors
import GIAANNcmn_inferenceDuringTrain


def decrementActivationDense(featureNeuronsActivation, activationDecrement):
	if(inferenceDecrementActivationsNonlinear):
		featureNeuronsActivation = featureNeuronsActivation * (1-activationDecrement)
	else:
		featureNeuronsActivation = featureNeuronsActivation - activationDecrementPerPredictedSequence
	return featureNeuronsActivation

def decrementActivation(featureNeuronsActivation, activationDecrement):
	if(inferenceDecrementActivationsNonlinear):
		featureNeuronsActivation = featureNeuronsActivation * (1-activationDecrement)
	else:
		featureNeuronsActivation = GIAANNcmn_sparseTensors.subtractValueFromSparseTensorValues(featureNeuronsActivation, activationDecrementPerPredictedSequence)
	return featureNeuronsActivation

if(inferenceSegmentActivationsBoolean):
	def applySegmentActivationsBooleanFeatureSegmentsOnly(globalFeatureNeuronsActivation):
		featureSegmentStart = arrayNumberOfSegmentsColumnDistance
		if featureSegmentStart <= 0:
			printe("useSANIfeaturesAndColumns and no feature segments")
		if globalFeatureNeuronsActivation.is_sparse:
			sparseActivation = globalFeatureNeuronsActivation.coalesce()
			if featureSegmentStart >= sparseActivation.size(1):
				printe("feature segments start beyond the last segment")
			indices = sparseActivation.indices()
			values = sparseActivation.values()
			mask = indices[1] >= featureSegmentStart
			if pt.any(mask):
				#entries in feature segments
				values = values.clone()
				values[mask] = (values[mask] > 0).to(values.dtype)
				return pt.sparse_coo_tensor(indices, values, sparseActivation.size(), device=sparseActivation.device)
			else:
				return sparseActivation
		else:
			globalFeatureNeuronsActivation = globalFeatureNeuronsActivation.clone()
			if featureSegmentStart < globalFeatureNeuronsActivation.size(1):
				globalFeatureNeuronsActivation[:, featureSegmentStart:] = (globalFeatureNeuronsActivation[:, featureSegmentStart:] > 0).to(globalFeatureNeuronsActivation.dtype)
			return globalFeatureNeuronsActivation

	def applySegmentActivationsBoolean(globalFeatureNeuronsActivation):
		if(inferenceSegmentActivationsBooleanFeatureSegmentsOnly):
			if(useSANIcolumns):
				return globalFeatureNeuronsActivation
			elif(useSANIfeatures):
				return globalFeatureNeuronsActivation.bool().float()
			elif(useSANIfeaturesAndColumns):
				return applySegmentActivationsBooleanFeatureSegmentsOnly(globalFeatureNeuronsActivation)
		else:
			return globalFeatureNeuronsActivation.bool().float()

def applySequentialActivationDense(globalFeatureNeuronsActivation, featureNeuronsTargetActivation):
	if(multipleDendriticBranchesBinaryTree):
		resultActivation, featureNeuronsTargetActivationApplied = applySequentialActivationDenseBinaryTree(globalFeatureNeuronsActivation, featureNeuronsTargetActivation)
	else:
		globalFeatureNeuronsActivationDense = globalFeatureNeuronsActivation.to_dense()
		featureNeuronsTargetActivationDense = featureNeuronsTargetActivation
		if(featureNeuronsTargetActivationDense.is_sparse):
			featureNeuronsTargetActivationDense = featureNeuronsTargetActivationDense.to_dense()
		featureNeuronsTargetActivationAppliedDense = pt.zeros_like(featureNeuronsTargetActivationDense)
		for branchIndex in range(globalFeatureNeuronsActivationDense.shape[0]):
			branchActivation = globalFeatureNeuronsActivationDense[branchIndex]
			branchTargetActivation = featureNeuronsTargetActivationDense[branchIndex]
			branchTargetActivationApplied = featureNeuronsTargetActivationAppliedDense[branchIndex]
			if(useSANIfeaturesAndColumns):
				featureSegmentsOffset = arrayNumberOfSegmentsColumnDistance
				assert featureSegmentsOffset >= 0 and featureSegmentsOffset < arrayNumberOfSegments
				previousFeatureChannelActivation = branchActivation[featureSegmentsOffset:arrayNumberOfSegments-1] > 0 if featureSegmentsOffset+1 < arrayNumberOfSegments else None
				if(enforceSequentialActivationFeatureSegmentsOnly):
					branchActivation[:featureSegmentsOffset+1] += branchTargetActivation[:featureSegmentsOffset+1]
					branchTargetActivationApplied[:featureSegmentsOffset+1] = branchTargetActivation[:featureSegmentsOffset+1]
				else:
					if(arrayIndexSegmentInternalColumn < arrayIndexSegmentFirst or arrayIndexSegmentInternalColumn >= featureSegmentsOffset):
						raise RuntimeError("applySequentialActivationDense error: arrayIndexSegmentInternalColumn out of range")
					if(arrayIndexSegmentInternalColumn > arrayIndexSegmentFirst+1):
						previousConceptChannelActivation = branchActivation[arrayIndexSegmentFirst:arrayIndexSegmentInternalColumn-1] > 0
						branchActivation[arrayIndexSegmentFirst+1:arrayIndexSegmentInternalColumn] += branchTargetActivation[arrayIndexSegmentFirst+1:arrayIndexSegmentInternalColumn] * previousConceptChannelActivation
						branchTargetActivationApplied[arrayIndexSegmentFirst+1:arrayIndexSegmentInternalColumn] = branchTargetActivation[arrayIndexSegmentFirst+1:arrayIndexSegmentInternalColumn] * previousConceptChannelActivation
					branchActivation[arrayIndexSegmentFirst] += branchTargetActivation[arrayIndexSegmentFirst]
					branchTargetActivationApplied[arrayIndexSegmentFirst] = branchTargetActivation[arrayIndexSegmentFirst]
					if(arrayIndexSegmentInternalColumn > arrayIndexSegmentFirst):
						branchActivation[arrayIndexSegmentInternalColumn] += branchTargetActivation[arrayIndexSegmentInternalColumn]
						branchTargetActivationApplied[arrayIndexSegmentInternalColumn] = branchTargetActivation[arrayIndexSegmentInternalColumn]
					branchActivation[featureSegmentsOffset] += branchTargetActivation[featureSegmentsOffset]
					branchTargetActivationApplied[featureSegmentsOffset] = branchTargetActivation[featureSegmentsOffset]
				if(previousFeatureChannelActivation is not None):
					branchActivation[featureSegmentsOffset+1:] += branchTargetActivation[featureSegmentsOffset+1:] * previousFeatureChannelActivation
					branchTargetActivationApplied[featureSegmentsOffset+1:] = branchTargetActivation[featureSegmentsOffset+1:] * previousFeatureChannelActivation
			else:
				previousChannelActivation = branchActivation[:-1] > 0
				branchActivation[1:] += branchTargetActivation[1:] * previousChannelActivation
				branchActivation[0] += branchTargetActivation[0]
				branchTargetActivationApplied[1:] = branchTargetActivation[1:] * previousChannelActivation
				branchTargetActivationApplied[0] = branchTargetActivation[0]
			globalFeatureNeuronsActivationDense[branchIndex] = branchActivation
			featureNeuronsTargetActivationAppliedDense[branchIndex] = branchTargetActivationApplied
		resultActivation = globalFeatureNeuronsActivationDense.to_sparse_coo()
		if(featureNeuronsTargetActivation.is_sparse):
			featureNeuronsTargetActivationApplied = featureNeuronsTargetActivationAppliedDense.to_sparse_coo()
		else:
			featureNeuronsTargetActivationApplied = featureNeuronsTargetActivationAppliedDense
	return resultActivation, featureNeuronsTargetActivationApplied

def applySequentialActivationSparse(globalFeatureNeuronsActivation, featureNeuronsTargetActivation):
	resultActivation = globalFeatureNeuronsActivation
	featureNeuronsTargetActivationApplied = featureNeuronsTargetActivation
	if(multipleDendriticBranchesBinaryTree):
		resultActivation, featureNeuronsTargetActivationApplied = applySequentialActivationSparseBinaryTree(globalFeatureNeuronsActivation, featureNeuronsTargetActivation)
	else:
		if(featureNeuronsTargetActivation is None):
			raise RuntimeError("applySequentialActivationSparse: featureNeuronsTargetActivation is None")
		if(not globalFeatureNeuronsActivation.is_sparse):
			raise RuntimeError("applySequentialActivationSparse: globalFeatureNeuronsActivation must be sparse")
		activationSparse = globalFeatureNeuronsActivation.coalesce()
		targetSparse = featureNeuronsTargetActivation
		if(not targetSparse.is_sparse):
			targetSparse = targetSparse.to_sparse_coo()
		targetSparse = targetSparse.coalesce()
		if(targetSparse._nnz() == 0):
			featureNeuronsTargetActivationApplied = targetSparse
		else:
			indices = targetSparse.indices()
			values = targetSparse.values()
			segmentIndices = indices[1]
			allowedMask = pt.ones((segmentIndices.shape[0],), dtype=pt.bool, device=segmentIndices.device)
			if(useSANIfeaturesAndColumns):
				featureSegmentStart = arrayNumberOfSegmentsColumnDistance
				if(featureSegmentStart < 0 or featureSegmentStart >= arrayNumberOfSegments):
					raise RuntimeError("applySequentialActivationSparse: featureSegmentStart out of range")
				if(enforceSequentialActivationFeatureSegmentsOnly):
					requireCheck = segmentIndices > featureSegmentStart
				else:
					if(arrayIndexSegmentInternalColumn < arrayIndexSegmentFirst or arrayIndexSegmentInternalColumn >= featureSegmentStart):
						raise RuntimeError("applySequentialActivationSparse: arrayIndexSegmentInternalColumn out of range")
					requireCheck = (segmentIndices != arrayIndexSegmentFirst) & (segmentIndices != featureSegmentStart) & (segmentIndices != arrayIndexSegmentInternalColumn)
				if(requireCheck.any()):
					checkIndices = pt.nonzero(requireCheck, as_tuple=False).view(-1)
					prevIndices = indices.index_select(1, checkIndices).clone()
					prevIndices[1] = prevIndices[1] - 1
					prevValues = gatherSparseTensorValuesAtIndices(activationSparse, prevIndices, values.dtype)
					allowedMask[checkIndices] = prevValues > 0
			else:
				requireCheck = segmentIndices > 0
				if(requireCheck.any()):
					checkIndices = pt.nonzero(requireCheck, as_tuple=False).view(-1)
					prevIndices = indices.index_select(1, checkIndices).clone()
					prevIndices[1] = prevIndices[1] - 1
					prevValues = gatherSparseTensorValuesAtIndices(activationSparse, prevIndices, values.dtype)
					allowedMask[checkIndices] = prevValues > 0
			if(allowedMask.any()):
				allowedIndices = indices[:, allowedMask]
				allowedValues = values[allowedMask]
				featureNeuronsTargetActivationApplied = pt.sparse_coo_tensor(allowedIndices, allowedValues, size=targetSparse.size(), device=targetSparse.device).coalesce()
			else:
				emptyIndices = pt.empty((indices.shape[0], 0), dtype=indices.dtype, device=indices.device)
				emptyValues = pt.empty((0,), dtype=values.dtype, device=values.device)
				featureNeuronsTargetActivationApplied = pt.sparse_coo_tensor(emptyIndices, emptyValues, size=targetSparse.size(), device=targetSparse.device).coalesce()
		resultActivation = activationSparse + featureNeuronsTargetActivationApplied
		resultActivation = resultActivation.coalesce()
	return resultActivation, featureNeuronsTargetActivationApplied

def applySequentialActivationDenseBinaryTree(globalFeatureNeuronsActivation, featureNeuronsTargetActivation):
	resultActivation = None
	featureNeuronsTargetActivationApplied = None
	if(multipleDendriticBranchesBinaryTree):
		if(globalFeatureNeuronsActivation.dim() != 4 or featureNeuronsTargetActivation.dim() != 4):
			raise RuntimeError("applySequentialActivationDenseBinaryTree error: activation tensors must be rank 4")
		if(globalFeatureNeuronsActivation.shape[0] != multipleDendriticBranchesNumber or globalFeatureNeuronsActivation.shape[1] != arrayNumberOfSegments or featureNeuronsTargetActivation.shape != globalFeatureNeuronsActivation.shape):
			raise RuntimeError("applySequentialActivationDenseBinaryTree error: activation tensor shapes are invalid")
		if(multipleDendriticBranchesNumber % multipleDendriticBranchesBinaryTreeBranchingFactor != arrayIndexSegmentFirst):
			raise RuntimeError("applySequentialActivationDenseBinaryTree error: branch count is invalid")
		globalFeatureNeuronsActivationDense = globalFeatureNeuronsActivation.to_dense()
		featureNeuronsTargetActivationSparse = featureNeuronsTargetActivation.is_sparse
		featureNeuronsTargetActivationDense = featureNeuronsTargetActivation.to_dense() if featureNeuronsTargetActivationSparse else featureNeuronsTargetActivation
		featureNeuronsTargetActivationAppliedDense = pt.zeros_like(featureNeuronsTargetActivationDense)
		featureNeuronsTargetActivationAppliedDense[:, arrayIndexSegmentFirst] = featureNeuronsTargetActivationDense[:, arrayIndexSegmentFirst]
		if(arrayNumberOfSegments > 1):
			parentBranchCount = multipleDendriticBranchesNumber//multipleDendriticBranchesBinaryTreeBranchingFactor
			previousSegmentActivation = pt.zeros_like(featureNeuronsTargetActivationDense[:, 1:])
			previousSegmentActivation[:parentBranchCount] = (globalFeatureNeuronsActivationDense.reshape(parentBranchCount, multipleDendriticBranchesBinaryTreeBranchingFactor, arrayNumberOfSegments, *globalFeatureNeuronsActivationDense.shape[2:])[:, :, :-1] > 0).any(dim=1)
			branchIndices = pt.arange(multipleDendriticBranchesNumber, dtype=pt.long, device=globalFeatureNeuronsActivationDense.device).unsqueeze(1)
			segmentIndices = pt.arange(1, arrayNumberOfSegments, dtype=pt.long, device=globalFeatureNeuronsActivationDense.device).unsqueeze(0)
			validBranchCounts = multipleDendriticBranchesNumber//pt.pow(pt.full_like(segmentIndices, multipleDendriticBranchesBinaryTreeBranchingFactor), segmentIndices)
			validBranchMask = branchIndices < validBranchCounts
			if(useSANIfeaturesAndColumns and enforceSequentialActivationFeatureSegmentsOnly):
				featureSegmentStart = arrayNumberOfSegmentsColumnDistance
				if(featureSegmentStart <= arrayIndexSegmentFirst or featureSegmentStart >= arrayNumberOfSegments):
					raise RuntimeError("applySequentialActivationDenseBinaryTree error: featureSegmentStart out of range")
				if(featureSegmentStart > arrayIndexSegmentFirst+1):
					featureNeuronsTargetActivationAppliedDense[:, arrayIndexSegmentFirst+1:featureSegmentStart] = featureNeuronsTargetActivationDense[:, arrayIndexSegmentFirst+1:featureSegmentStart]*validBranchMask[:, :featureSegmentStart-1].view(multipleDendriticBranchesNumber, featureSegmentStart-1, *([1]*(featureNeuronsTargetActivationDense.dim()-2)))
				featureNeuronsTargetActivationAppliedDense[:, featureSegmentStart] = featureNeuronsTargetActivationDense[:, featureSegmentStart]*validBranchMask[:, featureSegmentStart-1].view(multipleDendriticBranchesNumber, *([1]*(featureNeuronsTargetActivationDense.dim()-2)))
				if(featureSegmentStart+1 < arrayNumberOfSegments):
					featureNeuronsTargetActivationAppliedDense[:, featureSegmentStart+1:] = featureNeuronsTargetActivationDense[:, featureSegmentStart+1:]*previousSegmentActivation[:, featureSegmentStart:]*validBranchMask[:, featureSegmentStart:].view(multipleDendriticBranchesNumber, arrayNumberOfSegments-featureSegmentStart-1, *([1]*(featureNeuronsTargetActivationDense.dim()-2)))
			else:
				featureNeuronsTargetActivationAppliedDense[:, 1:] = featureNeuronsTargetActivationDense[:, 1:]*previousSegmentActivation*validBranchMask.view(multipleDendriticBranchesNumber, arrayNumberOfSegments-1, *([1]*(featureNeuronsTargetActivationDense.dim()-2)))
				if(useSANIfeaturesAndColumns):
					featureSegmentStart = arrayNumberOfSegmentsColumnDistance
					if(featureSegmentStart <= arrayIndexSegmentFirst or featureSegmentStart >= arrayNumberOfSegments):
						raise RuntimeError("applySequentialActivationDenseBinaryTree error: featureSegmentStart out of range")
					if(arrayIndexSegmentInternalColumn < arrayIndexSegmentFirst or arrayIndexSegmentInternalColumn >= featureSegmentStart):
						raise RuntimeError("applySequentialActivationDenseBinaryTree error: arrayIndexSegmentInternalColumn out of range")
					if(arrayIndexSegmentInternalColumn > arrayIndexSegmentFirst):
						featureNeuronsTargetActivationAppliedDense[:, arrayIndexSegmentInternalColumn] = featureNeuronsTargetActivationDense[:, arrayIndexSegmentInternalColumn]*validBranchMask[:, arrayIndexSegmentInternalColumn-1].view(multipleDendriticBranchesNumber, *([1]*(featureNeuronsTargetActivationDense.dim()-2)))
					featureNeuronsTargetActivationAppliedDense[:, featureSegmentStart] = featureNeuronsTargetActivationDense[:, featureSegmentStart]*validBranchMask[:, featureSegmentStart-1].view(multipleDendriticBranchesNumber, *([1]*(featureNeuronsTargetActivationDense.dim()-2)))
		resultActivation = (globalFeatureNeuronsActivationDense + featureNeuronsTargetActivationAppliedDense).to_sparse_coo()
		featureNeuronsTargetActivationApplied = featureNeuronsTargetActivationAppliedDense.to_sparse_coo() if featureNeuronsTargetActivationSparse else featureNeuronsTargetActivationAppliedDense
	else:
		raise RuntimeError("applySequentialActivationDenseBinaryTree error: requires multipleDendriticBranchesBinaryTree")
	return resultActivation, featureNeuronsTargetActivationApplied

def applySequentialActivationSparseBinaryTree(globalFeatureNeuronsActivation, featureNeuronsTargetActivation):
	resultActivation = None
	featureNeuronsTargetActivationApplied = None
	if(multipleDendriticBranchesBinaryTree):
		if(featureNeuronsTargetActivation is None):
			raise RuntimeError("applySequentialActivationSparseBinaryTree error: featureNeuronsTargetActivation is None")
		if(not globalFeatureNeuronsActivation.is_sparse):
			raise RuntimeError("applySequentialActivationSparseBinaryTree error: globalFeatureNeuronsActivation must be sparse")
		if(globalFeatureNeuronsActivation.dim() != 4 or globalFeatureNeuronsActivation.shape[0] != multipleDendriticBranchesNumber or globalFeatureNeuronsActivation.shape[1] != arrayNumberOfSegments):
			raise RuntimeError("applySequentialActivationSparseBinaryTree error: globalFeatureNeuronsActivation shape is invalid")
		activationSparse = globalFeatureNeuronsActivation.coalesce()
		targetSparse = featureNeuronsTargetActivation.coalesce() if featureNeuronsTargetActivation.is_sparse else featureNeuronsTargetActivation.to_sparse_coo().coalesce()
		if(targetSparse.size() != activationSparse.size()):
			raise RuntimeError("applySequentialActivationSparseBinaryTree error: activation tensor shapes are invalid")
		if(targetSparse._nnz() == arrayIndexSegmentFirst):
			featureNeuronsTargetActivationApplied = targetSparse
		else:
			indices = targetSparse.indices()
			values = targetSparse.values()
			segmentIndices = indices[1]
			if(bool(pt.any(segmentIndices < arrayIndexSegmentFirst).item()) or bool(pt.any(segmentIndices >= arrayNumberOfSegments).item())):
				raise RuntimeError("applySequentialActivationSparseBinaryTree error: target segment index out of range")
			allowedMask = pt.ones((segmentIndices.shape[0],), dtype=pt.bool, device=segmentIndices.device)
			requireCheck = calculateBinaryTreeSequentialActivationRequiredMask(segmentIndices)
			if(requireCheck.any()):
				checkIndices = pt.nonzero(requireCheck, as_tuple=False).view(-1)
				previousIndices = buildBinaryTreePreviousSegmentIndices(indices.index_select(1, checkIndices), targetSparse.size())
				previousValues = gatherSparseTensorValuesAtIndices(activationSparse, previousIndices, values.dtype).view(checkIndices.shape[0], multipleDendriticBranchesBinaryTreeBranchingFactor)
				allowedMask[checkIndices] = pt.any(previousValues > 0, dim=1)
			if(allowedMask.any()):
				featureNeuronsTargetActivationApplied = pt.sparse_coo_tensor(indices[:, allowedMask], values[allowedMask], size=targetSparse.size(), device=targetSparse.device).coalesce()
			else:
				emptyIndices = pt.empty((indices.shape[0], 0), dtype=indices.dtype, device=indices.device)
				emptyValues = pt.empty((0,), dtype=values.dtype, device=values.device)
				featureNeuronsTargetActivationApplied = pt.sparse_coo_tensor(emptyIndices, emptyValues, size=targetSparse.size(), device=targetSparse.device).coalesce()
		resultActivation = (activationSparse + featureNeuronsTargetActivationApplied).coalesce()
	else:
		raise RuntimeError("applySequentialActivationSparseBinaryTree error: requires multipleDendriticBranchesBinaryTree")
	return resultActivation, featureNeuronsTargetActivationApplied

def buildBinaryTreePreviousSegmentIndices(indices, activationSize):
	result = None
	if(multipleDendriticBranchesBinaryTree):
		if(indices.dim() != 2 or indices.shape[0] != len(activationSize) or len(activationSize) != 4):
			raise RuntimeError("buildBinaryTreePreviousSegmentIndices error: indices or activationSize is invalid")
		if(indices.shape[1] == arrayIndexSegmentFirst):
			raise RuntimeError("buildBinaryTreePreviousSegmentIndices error: indices must not be empty")
		if(bool(pt.any(indices[1] <= arrayIndexSegmentFirst).item()) or bool(pt.any(indices[1] >= arrayNumberOfSegments).item())):
			raise RuntimeError("buildBinaryTreePreviousSegmentIndices error: segment indices must be in the non-root range")
		validBranchCounts = multipleDendriticBranchesNumber//pt.pow(pt.full_like(indices[1], multipleDendriticBranchesBinaryTreeBranchingFactor), indices[1])
		if(bool(pt.any(indices[0] < arrayIndexSegmentFirst).item()) or bool(pt.any(indices[0] >= validBranchCounts).item())):
			raise RuntimeError("buildBinaryTreePreviousSegmentIndices error: branch index is invalid for a non-root segment")
		childOffsets = pt.arange(multipleDendriticBranchesBinaryTreeBranchingFactor, dtype=indices.dtype, device=indices.device)
		result = indices.repeat_interleave(multipleDendriticBranchesBinaryTreeBranchingFactor, dim=1)
		result[0] = (indices[0].unsqueeze(1)*multipleDendriticBranchesBinaryTreeBranchingFactor + childOffsets.unsqueeze(0)).reshape(-1)
		result[1] = result[1] - 1
	else:
		raise RuntimeError("buildBinaryTreePreviousSegmentIndices error: requires multipleDendriticBranchesBinaryTree")
	return result

def calculateBinaryTreeSequentialActivationRequiredMask(segmentIndices):
	result = None
	featureSegmentStart = None
	if(multipleDendriticBranchesBinaryTree):
		if(not pt.is_tensor(segmentIndices) or segmentIndices.dim() != 1):
			raise RuntimeError("calculateBinaryTreeSequentialActivationRequiredMask error: segmentIndices must be a rank 1 tensor")
		if(bool(pt.any(segmentIndices < arrayIndexSegmentFirst).item()) or bool(pt.any(segmentIndices >= arrayNumberOfSegments).item())):
			raise RuntimeError("calculateBinaryTreeSequentialActivationRequiredMask error: segment index out of range")
		result = segmentIndices > arrayIndexSegmentFirst
		if(useSANIfeaturesAndColumns):
			featureSegmentStart = arrayNumberOfSegmentsColumnDistance
			if(featureSegmentStart <= arrayIndexSegmentFirst or featureSegmentStart >= arrayNumberOfSegments):
				raise RuntimeError("calculateBinaryTreeSequentialActivationRequiredMask error: featureSegmentStart out of range")
			if(enforceSequentialActivationFeatureSegmentsOnly):
				result = segmentIndices > featureSegmentStart
			else:
				if(arrayIndexSegmentInternalColumn < arrayIndexSegmentFirst or arrayIndexSegmentInternalColumn >= featureSegmentStart):
					raise RuntimeError("calculateBinaryTreeSequentialActivationRequiredMask error: arrayIndexSegmentInternalColumn out of range")
				result = result & (segmentIndices != featureSegmentStart) & (segmentIndices != arrayIndexSegmentInternalColumn)
	else:
		raise RuntimeError("calculateBinaryTreeSequentialActivationRequiredMask error: requires multipleDendriticBranchesBinaryTree")
	return result

def calculateSequenceColumnIndex(conceptMask, sequenceWordIndex):
	result = None
	if(conceptMask is None):
		raise RuntimeError("calculateSequenceColumnIndex: conceptMask is None")
	if(sequenceWordIndex < 0 or sequenceWordIndex >= conceptMask.shape[0]):
		raise RuntimeError("calculateSequenceColumnIndex: sequenceWordIndex out of range")
	sequenceColumnIndexTensor = pt.sum(conceptMask[:sequenceWordIndex+1].to(pt.long))
	result = int(sequenceColumnIndexTensor.item())
	return result

def buildSparseIndexKeys(indices, size):
	if(indices.dim() != 2):
		raise RuntimeError("buildSparseIndexKeys: indices must be 2D")
	if(len(size) != indices.shape[0]):
		raise RuntimeError("buildSparseIndexKeys: size mismatch")
	strides = pt.tensor([size[1]*size[2]*size[3], size[2]*size[3], size[3], 1], dtype=pt.long, device=indices.device)
	keys = (indices * strides.unsqueeze(1)).sum(dim=0)
	return keys

def gatherSparseTensorValuesAtIndices(tensor, indices, dtype):
	result = None
	if(tensor is None):
		result = pt.zeros((indices.shape[1],), dtype=dtype, device=indices.device)
	else:
		if(not tensor.is_sparse):
			result = tensor[indices[0], indices[1], indices[2], indices[3]].to(dtype)
		else:
			tensor = tensor.coalesce()
			if(tensor._nnz() == 0):
				result = pt.zeros((indices.shape[1],), dtype=dtype, device=indices.device)
			else:
				size = tensor.size()
				if(len(size) != indices.shape[0]):
					raise RuntimeError("gatherSparseTensorValuesAtIndices: tensor/index size mismatch")
				tensorKeys = buildSparseIndexKeys(tensor.indices(), size)
				queryKeys = buildSparseIndexKeys(indices, size)
				sortedTensorKeys, sortOrder = pt.sort(tensorKeys)
				sortedTensorValues = tensor.values().index_select(0, sortOrder)
				positions = pt.searchsorted(sortedTensorKeys, queryKeys)
				valid = positions < sortedTensorKeys.shape[0]
				safePositions = positions.clamp(max=sortedTensorKeys.shape[0]-1)
				matches = valid & (sortedTensorKeys[safePositions] == queryKeys)
				result = pt.zeros((queryKeys.shape[0],), dtype=sortedTensorValues.dtype, device=sortedTensorValues.device)
				if(matches.any()):
					result[matches] = sortedTensorValues.index_select(0, safePositions[matches])
				if(result.dtype != dtype):
					result = result.to(dtype)
	return result

def gatherSparseTensorPresenceAtIndices(tensor, indices):
	result = None
	if(tensor is None):
		raise RuntimeError("gatherSparseTensorPresenceAtIndices: tensor is None")
	else:
		if(not tensor.is_sparse):
			result = pt.ones((indices.shape[1],), dtype=pt.bool, device=indices.device)
		else:
			tensor = tensor.coalesce()
			if(tensor._nnz() == 0):
				result = pt.zeros((indices.shape[1],), dtype=pt.bool, device=indices.device)
			else:
				size = tensor.size()
				if(len(size) != indices.shape[0]):
					raise RuntimeError("gatherSparseTensorPresenceAtIndices: tensor/index size mismatch")
				tensorKeys = buildSparseIndexKeys(tensor.indices(), size)
				queryKeys = buildSparseIndexKeys(indices, size)
				sortedTensorKeys = pt.sort(tensorKeys).values
				positions = pt.searchsorted(sortedTensorKeys, queryKeys)
				valid = positions < sortedTensorKeys.shape[0]
				safePositions = positions.clamp(max=sortedTensorKeys.shape[0]-1)
				result = valid & (sortedTensorKeys[safePositions] == queryKeys)
	return result

def replaceSparseTensorValuesAtIndices(sparseTensor, updateIndices, updateValues):
	result = sparseTensor
	if(sparseTensor is None):
		raise RuntimeError("replaceSparseTensorValuesAtIndices: sparseTensor is None")
	if(updateIndices is None):
		result = sparseTensor
	else:
		if(updateIndices.numel() == 0):
			result = sparseTensor
		else:
			sparseTensor = sparseTensor.coalesce()
			updateIndices = updateIndices.to(sparseTensor.device)
			updateValues = updateValues.to(sparseTensor.device)
			if(updateValues.dtype != sparseTensor.values().dtype):
				updateValues = updateValues.to(sparseTensor.values().dtype)
			size = sparseTensor.size()
			if(len(size) != updateIndices.shape[0]):
				raise RuntimeError("replaceSparseTensorValuesAtIndices: updateIndices size mismatch")
			existingIndices = sparseTensor.indices()
			existingValues = sparseTensor.values()
			if(existingIndices.numel() == 0):
				newIndices = updateIndices
				newValues = updateValues
			else:
				existingKeys = buildSparseIndexKeys(existingIndices, size)
				updateKeys = buildSparseIndexKeys(updateIndices, size)
				sortedUpdateKeys, updateOrder = pt.sort(updateKeys)
				positions = pt.searchsorted(sortedUpdateKeys, existingKeys)
				valid = positions < sortedUpdateKeys.shape[0]
				safePositions = positions.clamp(max=sortedUpdateKeys.shape[0]-1)
				matches = valid & (sortedUpdateKeys[safePositions] == existingKeys)
				keepMask = ~matches
				newIndices = pt.cat([existingIndices[:, keepMask], updateIndices], dim=1)
				newValues = pt.cat([existingValues[keepMask], updateValues], dim=0)
			result = pt.sparse_coo_tensor(newIndices, newValues, size=size, device=sparseTensor.device).coalesce()
	return result

def buildSegmentTimeValues(segmentIndices, sequenceWordIndex, sequenceColumnIndex, dtype, device):
	result = None
	if(useSANIfeaturesAndColumns):
		if(sequenceColumnIndex is None):
			raise RuntimeError("buildSegmentTimeValues: sequenceColumnIndex is required for column segments")
		if(sequenceWordIndex < 0):
			raise RuntimeError("buildSegmentTimeValues: sequenceWordIndex must be >= 0 for source time")
		if(sequenceColumnIndex < 0):
			raise RuntimeError("buildSegmentTimeValues: sequenceColumnIndex must be >= 0 for source time")
		result = pt.zeros_like(segmentIndices, dtype=dtype, device=device)
		featureMask = segmentIndices >= arrayNumberOfSegmentsColumnDistance
		if(featureMask.any()):
			result[featureMask] = float(sequenceWordIndex)
		if((~featureMask).any()):
			result[~featureMask] = float(sequenceColumnIndex)
	elif(useSANIfeatures):
		if(sequenceWordIndex < 0):
			raise RuntimeError("buildSegmentTimeValues: sequenceWordIndex must be >= 0 for source time")
		result = pt.full_like(segmentIndices, float(sequenceWordIndex), dtype=dtype, device=device)
	elif(useSANIcolumns):
		if(sequenceColumnIndex is None):
			raise RuntimeError("buildSegmentTimeValues: sequenceColumnIndex is required for column segments")
		if(sequenceColumnIndex < 0):
			raise RuntimeError("buildSegmentTimeValues: sequenceColumnIndex must be >= 0 for source time")
		result = pt.full_like(segmentIndices, float(sequenceColumnIndex), dtype=dtype, device=device)
	else:
		raise RuntimeError("buildSegmentTimeValues: useSANI feature mode not configured")
	return result

def computeTimePenaltyForSegmentGroup(segmentIndices, storedTimes, currentTimeValue, numberSegments, segmentIndexOffset):
	localSegmentIndex = (segmentIndices - segmentIndexOffset).to(storedTimes.dtype)
	offsetValues = (float(numberSegments) - localSegmentIndex)
	currentTimeValues = pt.full_like(storedTimes, float(currentTimeValue))
	penaltyValues = pt.abs(currentTimeValues - storedTimes - offsetValues)
	return penaltyValues

def computeTimePenaltyForSegments(segmentIndices, storedTimes, sequenceWordIndex, sequenceColumnIndex):
	result = pt.zeros_like(storedTimes)
	if(useSANIfeaturesAndColumns):
		if(sequenceColumnIndex is None):
			raise RuntimeError("computeTimePenaltyForSegments: sequenceColumnIndex is required for column segments")
		featureMask = segmentIndices >= arrayNumberOfSegmentsColumnDistance
		columnMask = ~featureMask
		if(featureMask.any()):
			result[featureMask] = computeTimePenaltyForSegmentGroup(segmentIndices[featureMask], storedTimes[featureMask], sequenceWordIndex, arrayNumberOfSegmentsFeatureDistance, arrayNumberOfSegmentsColumnDistance)
		if(columnMask.any()):
			result[columnMask] = computeTimePenaltyForSegmentGroup(segmentIndices[columnMask], storedTimes[columnMask], sequenceColumnIndex, arrayNumberOfSegmentsColumnDistance, 0)
	elif(useSANIfeatures):
		result = computeTimePenaltyForSegmentGroup(segmentIndices, storedTimes, sequenceWordIndex, arrayNumberOfSegments, 0)
	elif(useSANIcolumns):
		if(sequenceColumnIndex is None):
			raise RuntimeError("computeTimePenaltyForSegments: sequenceColumnIndex is required for column segments")
		result = computeTimePenaltyForSegmentGroup(segmentIndices, storedTimes, sequenceColumnIndex, arrayNumberOfSegments, 0)
	else:
		raise RuntimeError("computeTimePenaltyForSegments: useSANI feature mode not configured")
	return result

def applyExactTimeActivationConstraint(featureNeuronsTargetActivation, globalFeatureNeuronsTime, sequenceWordIndex, sequenceColumnIndex):
	result = featureNeuronsTargetActivation
	if(inferenceUseNeuronFeaturePropertiesTimeExact):
		if(not inferenceUseNeuronFeaturePropertiesTime):
			raise RuntimeError("applyExactTimeActivationConstraint: inferenceUseNeuronFeaturePropertiesTime is required")
		if(not useSANI):
			raise RuntimeError("applyExactTimeActivationConstraint: useSANI is required")
		if(globalFeatureNeuronsTime is None):
			raise RuntimeError("applyExactTimeActivationConstraint: globalFeatureNeuronsTime is None")
		activationSparse = featureNeuronsTargetActivation
		wasDense = False
		if(not activationSparse.is_sparse):
			activationSparse = activationSparse.to_sparse_coo()
			wasDense = True
		activationSparse = activationSparse.coalesce()
		if(activationSparse._nnz() == 0):
			result = activationSparse
		else:
			indices = activationSparse.indices()
			values = activationSparse.values()
			segmentIndices = indices[1]
			if(multipleDendriticBranchesBinaryTree):
				requiresCheck = calculateBinaryTreeSequentialActivationRequiredMask(segmentIndices)
			else:
				requiresCheck = pt.zeros((segmentIndices.shape[0],), dtype=pt.bool, device=segmentIndices.device)
				if(useSANIfeaturesAndColumns):
					featureMask = segmentIndices >= arrayNumberOfSegmentsColumnDistance
					columnMask = ~featureMask
					if(enforceSequentialActivationFeatureSegmentsOnly):
						if(featureMask.any()):
							requiresCheck = requiresCheck | (featureMask & (segmentIndices > arrayNumberOfSegmentsColumnDistance))
					else:
						if(columnMask.any()):
							if(arrayIndexSegmentInternalColumn < arrayIndexSegmentFirst or arrayIndexSegmentInternalColumn >= arrayNumberOfSegmentsColumnDistance):
								raise RuntimeError("applyExactTimeActivationConstraint: arrayIndexSegmentInternalColumn out of range")
							requiresCheck = requiresCheck | (columnMask & (segmentIndices > arrayIndexSegmentFirst) & (segmentIndices != arrayIndexSegmentInternalColumn))
						if(featureMask.any()):
							requiresCheck = requiresCheck | (featureMask & (segmentIndices > arrayNumberOfSegmentsColumnDistance))
				else:
					requiresCheck = segmentIndices > 0
			allowedMask = pt.ones((segmentIndices.shape[0],), dtype=pt.bool, device=segmentIndices.device)
			if(requiresCheck.any()):
				checkIndices = pt.nonzero(requiresCheck, as_tuple=False).view(-1)
				localSegmentIndices = segmentIndices.index_select(0, checkIndices)
				if(multipleDendriticBranchesBinaryTree):
					previousIndices = buildBinaryTreePreviousSegmentIndices(indices.index_select(1, checkIndices), activationSparse.size())
					previousStoredTimes = gatherSparseTensorValuesAtIndices(globalFeatureNeuronsTime, previousIndices, values.dtype).view(checkIndices.shape[0], multipleDendriticBranchesBinaryTreeBranchingFactor)
					previousStoredTimesExist = gatherSparseTensorPresenceAtIndices(globalFeatureNeuronsTime, previousIndices).view(checkIndices.shape[0], multipleDendriticBranchesBinaryTreeBranchingFactor)
					currentTimeValues = pt.zeros((checkIndices.shape[0],), dtype=values.dtype, device=values.device)
					if(useSANIfeaturesAndColumns):
						featureMaskCheck = localSegmentIndices >= arrayNumberOfSegmentsColumnDistance
						if(featureMaskCheck.any()):
							currentTimeValues[featureMaskCheck] = float(sequenceWordIndex)
						if((~featureMaskCheck).any()):
							if(sequenceColumnIndex is None):
								raise RuntimeError("applyExactTimeActivationConstraint: sequenceColumnIndex is required for column segments")
							currentTimeValues[~featureMaskCheck] = float(sequenceColumnIndex)
					elif(useSANIfeatures):
						currentTimeValues = pt.full_like(currentTimeValues, float(sequenceWordIndex))
					elif(useSANIcolumns):
						if(sequenceColumnIndex is None):
							raise RuntimeError("applyExactTimeActivationConstraint: sequenceColumnIndex is required for column segments")
						currentTimeValues = pt.full_like(currentTimeValues, float(sequenceColumnIndex))
					else:
						raise RuntimeError("applyExactTimeActivationConstraint: useSANI feature mode not configured")
					allowedMask[checkIndices] = pt.any(previousStoredTimesExist & ((currentTimeValues.unsqueeze(1) - previousStoredTimes) == 1), dim=1)
				else:
					prevIndices = indices.index_select(1, checkIndices).clone()
					prevSegmentIndices = localSegmentIndices - 1
					prevIndices[1] = prevSegmentIndices
					prevStoredTimes = gatherSparseTensorValuesAtIndices(globalFeatureNeuronsTime, prevIndices, values.dtype)
					prevStoredTimesExist = gatherSparseTensorPresenceAtIndices(globalFeatureNeuronsTime, prevIndices)
					currentTimeValues = pt.zeros_like(prevStoredTimes)
					if(useSANIfeaturesAndColumns):
						featureMaskCheck = localSegmentIndices >= arrayNumberOfSegmentsColumnDistance
						if(featureMaskCheck.any()):
							currentTimeValues[featureMaskCheck] = float(sequenceWordIndex)
						if((~featureMaskCheck).any()):
							if(sequenceColumnIndex is None):
								raise RuntimeError("applyExactTimeActivationConstraint: sequenceColumnIndex is required for column segments")
							currentTimeValues[~featureMaskCheck] = float(sequenceColumnIndex)
					elif(useSANIfeatures):
						currentTimeValues = pt.full_like(prevStoredTimes, float(sequenceWordIndex))
					elif(useSANIcolumns):
						if(sequenceColumnIndex is None):
							raise RuntimeError("applyExactTimeActivationConstraint: sequenceColumnIndex is required for column segments")
						currentTimeValues = pt.full_like(prevStoredTimes, float(sequenceColumnIndex))
					else:
						raise RuntimeError("applyExactTimeActivationConstraint: useSANI feature mode not configured")
					allowedMask[checkIndices] = prevStoredTimesExist & ((currentTimeValues - prevStoredTimes) == 1)
			keepIndices = pt.nonzero(allowedMask, as_tuple=False).view(-1)
			if(keepIndices.numel() == 0):
				emptyIndices = pt.empty((indices.shape[0], 0), dtype=indices.dtype, device=indices.device)
				emptyValues = pt.empty((0,), dtype=values.dtype, device=values.device)
				result = pt.sparse_coo_tensor(emptyIndices, emptyValues, size=activationSparse.size(), device=activationSparse.device).coalesce()
			else:
				keptIndices = indices.index_select(1, keepIndices)
				keptValues = values.index_select(0, keepIndices)
				result = pt.sparse_coo_tensor(keptIndices, keptValues, size=activationSparse.size(), device=activationSparse.device).coalesce()
		if(wasDense):
			result = result.to_dense()
	return result

def applyTimeBasedActivationModifier(globalFeatureNeuronsActivation, globalFeatureNeuronsTime, sequenceWordIndex, sequenceColumnIndex):
	result = globalFeatureNeuronsActivation
	if(inferenceUseNeuronFeaturePropertiesTime):
		if(not useSANI):
			raise RuntimeError("applyTimeBasedActivationModifier: useSANI is required")
		if(globalFeatureNeuronsTime is None):
			raise RuntimeError("applyTimeBasedActivationModifier: globalFeatureNeuronsTime is None")
		activationSparse = globalFeatureNeuronsActivation
		if(not activationSparse.is_sparse):
			activationSparse = activationSparse.to_sparse_coo()
		activationSparse = activationSparse.coalesce()
		if(inferenceUseNeuronFeaturePropertiesTimeExact):
			# spec step (b): skipped when inferenceUseNeuronFeaturePropertiesTimeExact is enabled.
			result = activationSparse
		elif(activationSparse._nnz() == 0):
			result = activationSparse
		else:
			indices = activationSparse.indices()
			values = activationSparse.values()
			storedTimes = gatherSparseTensorValuesAtIndices(globalFeatureNeuronsTime, indices, values.dtype)
			segmentIndices = indices[1]
			penaltyValues = computeTimePenaltyForSegments(segmentIndices, storedTimes, sequenceWordIndex, sequenceColumnIndex)
			if(inferenceSegmentTimingMultipicativeBias):
				biasFactorValues = pt.full_like(values, inferenceSegmentTimingMultipicativeBiasFactor)
				modifiedValues = values*pt.pow(biasFactorValues, penaltyValues)
			else:
				modifiedValues = values - penaltyValues
			result = pt.sparse_coo_tensor(indices, modifiedValues, size=activationSparse.size(), device=activationSparse.device).coalesce()
	return result

def updateTimeValuesFromActivation(globalFeatureNeuronsTime, featureNeuronsTargetActivation, sequenceWordIndex, sequenceColumnIndex):
	result = globalFeatureNeuronsTime
	if(inferenceUseNeuronFeaturePropertiesTime):
		if(not useSANI):
			raise RuntimeError("updateTimeValuesFromActivation: useSANI is required")
		if(globalFeatureNeuronsTime is None):
			raise RuntimeError("updateTimeValuesFromActivation: globalFeatureNeuronsTime is None")
		if(featureNeuronsTargetActivation is None):
			result = globalFeatureNeuronsTime
		else:
			activationSparse = featureNeuronsTargetActivation
			if(not activationSparse.is_sparse):
				activationSparse = activationSparse.to_sparse_coo()
			activationSparse = activationSparse.coalesce()
			if(activationSparse._nnz() == 0):
				result = globalFeatureNeuronsTime
			else:
				updateIndices = activationSparse.indices()
				segmentIndices = updateIndices[1]
				updateValues = buildSegmentTimeValues(segmentIndices, sequenceWordIndex, sequenceColumnIndex, globalFeatureNeuronsTime.dtype, activationSparse.device)
				# spec step (a): store last timeValue for activated segments during each prediction step.
				result = replaceSparseTensorValuesAtIndices(globalFeatureNeuronsTime, updateIndices, updateValues)
	return result


#first dim cs1 restricted to a single token
def processFeaturesActivePredictSingle(databaseNetworkObject, globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, sequenceObservedColumnsPrediction, sourceColumnIndex, sourceFeatureIndex, globalFeatureNeuronsTime=None, sequenceWordIndex=None, sequenceColumnIndex=None):
	if(0 not in sequenceObservedColumnsPrediction.observedColumnsSequenceWordIndexDict):
		raise RuntimeError("processFeaturesActivePredictSingle error: missing observed column sequence index 0")
	observedColumn = sequenceObservedColumnsPrediction.observedColumnsSequenceWordIndexDict[0]
	connectionDevice = globalFeatureNeuronsActivation.device
	featureConnections = observedColumn.prepareFeatureConnectionsForSourceFeature(sourceFeatureIndex, targetDevice=connectionDevice, createMissing=False)
	result = processFeaturesActivePredict(databaseNetworkObject, globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, featureConnections, sourceColumnIndex, sourceFeatureIndex, globalFeatureNeuronsTime, sequenceWordIndex, sequenceColumnIndex)
	return result

def calculateFeatureNeuronSourceActivationPredict(databaseNetworkObject, globalFeatureNeuronsActivation, sourceColumnIndex, sourceFeatureIndex):
	result = None
	if(multipleDendriticBranchesBinaryTree):
		result = calculateFeatureNeuronSourceActivationPredictBinaryTree(globalFeatureNeuronsActivation, sourceColumnIndex, sourceFeatureIndex)
	else:
		result = calculateFeatureNeuronSourceActivationPredictNonBinary(globalFeatureNeuronsActivation, sourceColumnIndex, sourceFeatureIndex)
	return result

def calculateFeatureNeuronSourceActivationPredictBinaryTree(globalFeatureNeuronsActivation, sourceColumnIndex, sourceFeatureIndex):
	result = None
	if(multipleDendriticBranchesBinaryTree):
		if(globalFeatureNeuronsActivation is None):
			raise RuntimeError("calculateFeatureNeuronSourceActivationPredictBinaryTree error: globalFeatureNeuronsActivation is None")
		if(globalFeatureNeuronsActivation.dim() != 4 or globalFeatureNeuronsActivation.shape[0] != multipleDendriticBranchesNumber or globalFeatureNeuronsActivation.shape[1] != arrayNumberOfSegments):
			raise RuntimeError("calculateFeatureNeuronSourceActivationPredictBinaryTree error: globalFeatureNeuronsActivation shape is invalid")
		sourceColumnIndex = int(sourceColumnIndex)
		sourceFeatureIndex = int(sourceFeatureIndex)
		if(sourceColumnIndex < arrayIndexSegmentFirst or sourceColumnIndex >= globalFeatureNeuronsActivation.shape[2] or sourceFeatureIndex < arrayIndexSegmentFirst or sourceFeatureIndex >= globalFeatureNeuronsActivation.shape[3]):
			raise RuntimeError("calculateFeatureNeuronSourceActivationPredictBinaryTree error: source neuron index out of range")
		activationSparse = globalFeatureNeuronsActivation.coalesce() if globalFeatureNeuronsActivation.is_sparse else globalFeatureNeuronsActivation.to_sparse_coo().coalesce()
		activationIndices = activationSparse.indices()
		activationValues = activationSparse.values()
		sourceMask = (activationIndices[2] == sourceColumnIndex) & (activationIndices[3] == sourceFeatureIndex)
		if(not sourceMask.any()):
			result = pt.zeros((), dtype=activationValues.dtype, device=activationValues.device)
		else:
			sourceActivationSparse = pt.sparse_coo_tensor(activationIndices[:, sourceMask], activationValues[sourceMask], size=activationSparse.size(), dtype=activationSparse.dtype, device=activationSparse.device).coalesce()
			segmentActivations = []
			for segmentIndex in range(arrayNumberOfSegments):
				segmentActivations.append(GIAANNcmn_sparseTensors.projectBinaryTreeSegmentActivation(sourceActivationSparse, segmentIndex))
			combinedIndices = pt.empty((activationSparse.dim()-1, arrayIndexSegmentFirst), dtype=pt.long, device=activationSparse.device)
			combinedValues = pt.empty((arrayIndexSegmentFirst,), dtype=activationSparse.dtype, device=activationSparse.device)
			if(len(segmentActivations) > arrayIndexSegmentFirst):
				combinedIndices = pt.cat([segmentActivation.indices() for segmentActivation in segmentActivations], dim=1)
				combinedValues = pt.cat([segmentActivation.values() for segmentActivation in segmentActivations], dim=0)
			featureNeuronsActive = pt.sparse_coo_tensor(combinedIndices, combinedValues, size=(multipleDendriticBranchesNumber, *activationSparse.size()[2:]), dtype=activationSparse.dtype, device=activationSparse.device).coalesce()
			if(useSANI):
				if(algorithmMatrixSANImethod=="enforceActivationAcrossSegments"):
					if(enforceActivationAcrossSegmentsIgnoreInternalColumn):
						lastSegmentConstraint = arrayIndexSegmentAdjacentColumn
					else:
						lastSegmentConstraint = arrayIndexSegmentLast
					if(lastSegmentConstraint < arrayIndexSegmentFirst or lastSegmentConstraint >= arrayNumberOfSegments):
						raise RuntimeError("calculateFeatureNeuronSourceActivationPredictBinaryTree error: lastSegmentConstraint out of range")
					if(algorithmMatrixSANIenforceRequirement=="enforceLastSegmentMustBeActive"):
						featureNeuronsActive = GIAANNcmn_sparseTensors.selectAindicesContainedInBBinaryTree(featureNeuronsActive, segmentActivations[lastSegmentConstraint])
					elif(algorithmMatrixSANIenforceRequirement=="enforceAllSegmentsMustBeActive"):
						for segmentIndex in GIAANNcmn_sparseTensors.calculateAllSegmentConstraintIndexRange(lastSegmentConstraint):
							featureNeuronsActive = GIAANNcmn_sparseTensors.selectAindicesContainedInBBinaryTree(featureNeuronsActive, segmentActivations[segmentIndex])
					elif(algorithmMatrixSANIenforceRequirement!="enforceAnySegmentMustBeActive"):
						raise RuntimeError("calculateFeatureNeuronSourceActivationPredictBinaryTree error: algorithmMatrixSANIenforceRequirement is invalid")
				elif(algorithmMatrixSANImethod!="doNotEnforceActivationAcrossSegments"):
					raise RuntimeError("calculateFeatureNeuronSourceActivationPredictBinaryTree error: algorithmMatrixSANImethod is invalid")
			else:
				featureNeuronsActive = segmentActivations[arrayIndexSegmentLast]
			if(featureNeuronsActive._nnz() == arrayIndexSegmentFirst):
				result = pt.zeros((), dtype=activationValues.dtype, device=activationValues.device)
			else:
				result = featureNeuronsActive.values().max()
			if(inferenceSourceActivationsBoolean):
				result = (result > arrayIndexSegmentFirst).to(result.dtype)
	else:
		raise RuntimeError("calculateFeatureNeuronSourceActivationPredictBinaryTree error: requires multipleDendriticBranchesBinaryTree")
	return result

def calculateFeatureNeuronSourceActivationPredictNonBinary(globalFeatureNeuronsActivation, sourceColumnIndex, sourceFeatureIndex):
	featureNeuronsActive = GIAANNcmn_sparseTensors.neuronActivationSparse(globalFeatureNeuronsActivation, algorithmMatrixSANImethod)
	
	sourceColumnIndex = int(sourceColumnIndex)
	sourceFeatureIndex = int(sourceFeatureIndex)
	if(featureNeuronsActive.is_sparse):
		featureNeuronsActive = featureNeuronsActive.coalesce()
		if(featureNeuronsActive.dim() == 3):
			branchCount = featureNeuronsActive.size(0)
			indices = featureNeuronsActive.indices()
			values = featureNeuronsActive.values()
			mask = (indices[1] == sourceColumnIndex) & (indices[2] == sourceFeatureIndex)
			featureNeuronsActiveDense = pt.zeros((branchCount,), dtype=values.dtype, device=values.device)
			if(mask.any()):
				branchIndices = indices[0, mask]
				featureNeuronsActiveDense.index_add_(0, branchIndices, values[mask])
			featureNeuronsActive = featureNeuronsActiveDense
		elif(featureNeuronsActive.dim() == 2):
			indices = featureNeuronsActive.indices()
			values = featureNeuronsActive.values()
			mask = (indices[0] == sourceColumnIndex) & (indices[1] == sourceFeatureIndex)
			if(mask.any()):
				featureNeuronsActive = values[mask].sum()
			else:
				featureNeuronsActive = pt.zeros((), dtype=values.dtype, device=values.device)
		else:
			featureNeuronsActive = featureNeuronsActive.to_dense()
			if(featureNeuronsActive.dim() == 3):
				featureNeuronsActive = featureNeuronsActive[:, sourceColumnIndex, sourceFeatureIndex]
			else:
				featureNeuronsActive = featureNeuronsActive[sourceColumnIndex, sourceFeatureIndex]
	else:
		if(featureNeuronsActive.dim() == 3):
			featureNeuronsActive = featureNeuronsActive[:, sourceColumnIndex, sourceFeatureIndex]
		else:
			featureNeuronsActive = featureNeuronsActive[sourceColumnIndex, sourceFeatureIndex]
	if(inferenceSourceActivationsBoolean):
		featureNeuronsActive = (featureNeuronsActive > 0).to(featureNeuronsActive.dtype)	#ensure the source activation signal is binary (even with useSANI)
	if(multipleDendriticBranches and featureNeuronsActive.dim() == 1):
		featureNeuronsActive = featureNeuronsActive.sum()
	result = featureNeuronsActive
	return result

def transformFeatureNeuronsTargetActivationPredict(featureNeuronsTargetActivation):
	if(inferenceActivationFunction):
		result = activationFunction(featureNeuronsTargetActivation)
		#print("featureNeuronsTargetActivation = ", featureNeuronsTargetActivation)
	else:
		result = featureNeuronsTargetActivation*j1
	return result

def applyFeatureNeuronsTargetActivationPredict(databaseNetworkObject, globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, featureNeuronsTargetActivation, globalFeatureNeuronsTime=None, sequenceWordIndex=None, sequenceColumnIndex=None, applySegmentActivations=True):
	if(inferenceUseNeuronFeaturePropertiesTimeExact):
		# spec step (a): only allow segment activation when the time difference to the previous segment is exactly 1.
		featureNeuronsTargetActivation = applyExactTimeActivationConstraint(featureNeuronsTargetActivation, globalFeatureNeuronsTime, sequenceWordIndex, sequenceColumnIndex)

	featureNeuronsTargetActivationApplied = featureNeuronsTargetActivation

	#update the activations of the target nodes;
	if(useSANI):
		if(algorithmMatrixSANImethod=="enforceActivationAcrossSegments"):
			if(enforceSequentialActivation):
				if(inferenceApplySequentialActivationSparse):
					globalFeatureNeuronsActivation, featureNeuronsTargetActivationApplied = applySequentialActivationSparse(globalFeatureNeuronsActivation, featureNeuronsTargetActivation)
				else:
					globalFeatureNeuronsActivation, featureNeuronsTargetActivationApplied = applySequentialActivationDense(globalFeatureNeuronsActivation, featureNeuronsTargetActivation)
			else:
				globalFeatureNeuronsActivation += featureNeuronsTargetActivation
		elif(algorithmMatrixSANImethod=="doNotEnforceActivationAcrossSegments"):
			globalFeatureNeuronsActivation += featureNeuronsTargetActivation
	else:
		globalFeatureNeuronsActivation += featureNeuronsTargetActivation
	applySegmentActivationsBooleanCurrent = inferenceSegmentActivationsBoolean and applySegmentActivations
	if(inferenceDuringTrainAdjustSynapseStrengthBiasTimingCalculations):
		applySegmentActivationsBooleanCurrent = False
	if(applySegmentActivationsBooleanCurrent):
		globalFeatureNeuronsActivation = applySegmentActivationsBoolean(globalFeatureNeuronsActivation)
	if(inferenceUseNeuronFeaturePropertiesTime):
		# spec step (a): store last timeValue for activated segments during each prediction step
		globalFeatureNeuronsTime = updateTimeValuesFromActivation(globalFeatureNeuronsTime, featureNeuronsTargetActivationApplied, sequenceWordIndex, sequenceColumnIndex)
	result = globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, globalFeatureNeuronsTime
	return result

def processFeatureNeuronsTargetActivationPredict(databaseNetworkObject, globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, featureNeuronsTargetActivation, globalFeatureNeuronsTime=None, sequenceWordIndex=None, sequenceColumnIndex=None):
	featureNeuronsTargetActivation = transformFeatureNeuronsTargetActivationPredict(featureNeuronsTargetActivation)
	result = applyFeatureNeuronsTargetActivationPredict(databaseNetworkObject, globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, featureNeuronsTargetActivation, globalFeatureNeuronsTime, sequenceWordIndex, sequenceColumnIndex)
	return result

def processFeaturesActivePredict(databaseNetworkObject, globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, featureConnections, sourceColumnIndex, sourceFeatureIndex, globalFeatureNeuronsTime=None, sequenceWordIndex=None, sequenceColumnIndex=None, sourceActivationMultiplier=None):
	featureNeuronsActive = calculateFeatureNeuronSourceActivationPredict(databaseNetworkObject, globalFeatureNeuronsActivation, sourceColumnIndex, sourceFeatureIndex)

	#target neuron activation dependence on connection strength;
	featureConnectionsStrengthStored = featureConnections[databaseNetworkObject.arrayIndexPropertiesStrengthIndex]
	featureConnectionsStrength = featureConnectionsStrengthStored
	if(inferenceConnectionStrengthPOSdependence):
		featureConnectionsPos = featureConnections[databaseNetworkObject.arrayIndexPropertiesPosIndex]
	if(inferenceConnectionStrengthPOSdependence):
		featureConnectionsStrength = applyConnectionStrengthPOSdependenceInference(databaseNetworkObject, featureConnectionsStrength, featureConnectionsPos, sourceColumnIndex)
	featureConnectionsStrengthRaw = featureConnectionsStrength
	if(inferenceConnectionsStrengthBoolean):
		featureConnectionsStrength = featureConnectionsStrength.bool().float()
	
	if(featureNeuronsActive.dim() > 0):
		featureNeuronsActive = featureNeuronsActive.reshape(-1)
	featureNeuronsTargetActivation = calculateFeatureNeuronsTargetActivationPredict(featureConnectionsStrength, featureNeuronsActive)
	if(inferenceDuringTrainAdjustSynapseStrengthBiasTimingCalculations):
		if(inferenceConnectionsStrengthBoolean):
			featureNeuronsTargetActivation = calculateFeatureNeuronsTargetActivationPredict(featureConnectionsStrengthRaw, featureNeuronsActive)

	if(sourceActivationMultiplier is not None):
		sourceActivationMultiplier = float(sourceActivationMultiplier)
		if(sourceActivationMultiplier < auxiliaryNeuronsSimilarWordsMinimumSimilarity or sourceActivationMultiplier > auxiliaryNeuronsSimilarWordsMaximumSimilarity):
			raise RuntimeError("processFeaturesActivePredict error: sourceActivationMultiplier out of range")
		featureNeuronsTargetActivation = featureNeuronsTargetActivation * sourceActivationMultiplier
	if(inferenceDuringTrainAdjustSynapseStrength):
		if(inferenceDuringTrainAdjustSynapseStrengthDecrementInference):
			GIAANNcmn_inferenceDuringTrain.updateInferenceDuringTrainConnectionsActive(databaseNetworkObject, featureNeuronsTargetActivation, featureConnectionsStrengthStored, sourceColumnIndex, sourceFeatureIndex)
	result = processFeatureNeuronsTargetActivationPredict(databaseNetworkObject, globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, featureNeuronsTargetActivation, globalFeatureNeuronsTime, sequenceWordIndex, sequenceColumnIndex)
	return result

def calculateFeatureNeuronsTargetActivationPredict(featureConnectionsStrength, featureNeuronsActive):
	result = None
	if(featureConnectionsStrength.is_sparse):
		if(featureNeuronsActive.dim() == 0):
			branchCount = featureConnectionsStrength.size(0)
			branchValues = pt.full((branchCount,), featureNeuronsActive.item(), dtype=featureNeuronsActive.dtype, device=featureNeuronsActive.device)
			result = GIAANNcmn_sparseTensors.scaleSparseTensorByBranchValues(featureConnectionsStrength, branchValues)
		else:
			result = GIAANNcmn_sparseTensors.scaleSparseTensorByBranchValues(featureConnectionsStrength, featureNeuronsActive)
	else:
		if(featureNeuronsActive.dim() == 0):
			result = featureConnectionsStrength * featureNeuronsActive
		else:
			result = featureConnectionsStrength * featureNeuronsActive.view(-1, 1, 1, 1)
	return result

def selectActivatedBranchIndex(globalFeatureNeuronsActivation, columnIndex, featureIndex):
	result = arrayIndexSegmentFirst
	if(multipleDendriticBranchesBinaryTree):
		result = selectActivatedBinaryTreeRootBranchIndex(globalFeatureNeuronsActivation, columnIndex, featureIndex)
	elif(not multipleDendriticBranches):
		result = arrayIndexSegmentFirst
	elif(globalFeatureNeuronsActivation is None):
		result = arrayIndexSegmentFirst
	elif(globalFeatureNeuronsActivation.is_sparse):
		if(globalFeatureNeuronsActivation._nnz() != arrayIndexSegmentFirst):
			indices = globalFeatureNeuronsActivation._indices()
			values = globalFeatureNeuronsActivation._values()
			mask = (indices[2] == columnIndex) & (indices[3] == featureIndex)
			if(pt.any(mask)):
				selectedActivation = pt.sparse_coo_tensor(indices[:2, mask], values[mask], size=globalFeatureNeuronsActivation.size()[:2], dtype=values.dtype, device=values.device).coalesce()
				branchIndices = selectedActivation.indices()[0]
				uniqueBranchIndices, inverseBranchIndices = pt.unique(branchIndices, sorted=True, return_inverse=True)
				branchScores = pt.zeros((uniqueBranchIndices.shape[0],), dtype=values.dtype, device=values.device)
				branchScores.index_add_(arrayIndexSegmentFirst, inverseBranchIndices, selectedActivation.values())
				result = int(uniqueBranchIndices[pt.argmax(branchScores)].item())
	else:
		activationSlice = globalFeatureNeuronsActivation[:, :, columnIndex, featureIndex]
		if(activationSlice.numel() != arrayIndexSegmentFirst):
			branchScores = activationSlice.sum(dim=1)
			result = int(pt.argmax(branchScores).item())
	return result

def selectActivatedBinaryTreeRootBranchIndex(globalFeatureNeuronsActivation, columnIndex, featureIndex):
	result = arrayIndexSegmentFirst
	if(multipleDendriticBranchesBinaryTree):
		if(globalFeatureNeuronsActivation is None):
			result = arrayIndexSegmentFirst
		else:
			featureNeuronsActive = GIAANNcmn_sparseTensors.neuronActivationSparse(globalFeatureNeuronsActivation, algorithmMatrixSANImethod)
			if(featureNeuronsActive.is_sparse):
				featureNeuronsActive = featureNeuronsActive.coalesce()
				featureIndices = featureNeuronsActive.indices()
				featureValues = featureNeuronsActive.values()
				featureMask = (featureIndices[1] == columnIndex) & (featureIndices[2] == featureIndex)
				branchScores = pt.zeros((multipleDendriticBranchesNumber,), dtype=featureValues.dtype, device=featureValues.device)
				if(featureMask.any()):
					branchScores.index_add_(arrayIndexSegmentFirst, featureIndices[0, featureMask], featureValues[featureMask])
					result = int(pt.argmax(branchScores).item())
			else:
				if(featureNeuronsActive.dim() != 3):
					raise RuntimeError("selectActivatedBinaryTreeRootBranchIndex error: feature activation rank is invalid")
				if(columnIndex < arrayIndexSegmentFirst or columnIndex >= featureNeuronsActive.shape[1] or featureIndex < arrayIndexSegmentFirst or featureIndex >= featureNeuronsActive.shape[2]):
					raise RuntimeError("selectActivatedBinaryTreeRootBranchIndex error: target neuron index out of range")
				result = int(pt.argmax(featureNeuronsActive[:, columnIndex, featureIndex]).item())
	else:
		raise RuntimeError("selectActivatedBinaryTreeRootBranchIndex error: requires multipleDendriticBranchesBinaryTree")
	return result

def applyConnectionStrengthPOSdependenceInference(databaseNetworkObject, featureConnectionsStrength, featureConnectionsPos, sourceConceptIndex):
	posLookup = getConnectionStrengthPOSdependenceLookup(databaseNetworkObject)
	if posLookup:
		featureConnectionsStrength = featureConnectionsStrength.coalesce()
		if featureConnectionsStrength._nnz() == 0:
			printe("featureConnectionsStrength._nnz() == 0")
			#return featureConnectionsStrength
		if featureConnectionsPos is None:
			printe("featureConnectionsPos is None")
			#return featureConnectionsStrength
		featureConnectionsPos = featureConnectionsPos.coalesce()
		if featureConnectionsPos._nnz() == 0:
			printe("featureConnectionsPos._nnz() == 0")
			#return featureConnectionsStrength
		strengthIndices = featureConnectionsStrength.indices()
		strengthValues = featureConnectionsStrength.values()
		posIndices = featureConnectionsPos.indices()
		posValues = featureConnectionsPos.values()
		if strengthIndices.shape[1] == posIndices.shape[1] and pt.equal(strengthIndices, posIndices):
			alignedPosValues = posValues
		else:
			posIndicesCPU = posIndices.cpu()
			posValuesCPU = posValues.cpu()
			posIndexMap = {tuple(posIndicesCPU[:, idx].tolist()): posValuesCPU[idx].item() for idx in range(posIndicesCPU.shape[1])}
			strengthIndicesCPU = strengthIndices.cpu()
			alignedPosList = []
			for idx in range(strengthIndicesCPU.shape[1]):
				key = tuple(strengthIndicesCPU[:, idx].tolist())
				alignedPosList.append(posIndexMap.get(key, 0.0))
			alignedPosValues = pt.tensor(alignedPosList, dtype=posValues.dtype, device=posValues.device)
		alignedPosValues = alignedPosValues.long()
		if(connectionStrengthPOSdependenceExternal and sourceConceptIndex is not None):
			scopeMask = (strengthIndices[2] != sourceConceptIndex)
		else:
			scopeMask = pt.ones(strengthIndices.shape[1], dtype=pt.bool, device=strengthValues.device)
		if not pt.any(scopeMask):
			return featureConnectionsStrength
		else:
			scaleTensor = pt.ones_like(strengthValues)
			for posIndex, scaleValue in posLookup:
				if scaleValue == 1:
					continue
				posMask = (alignedPosValues == posIndex) & scopeMask
				if pt.any(posMask):
					scaleTensor[posMask] = scaleValue
			strengthValues *= scaleTensor
	return featureConnectionsStrength

def activationFunction(x):
	'''
	A non-linear activation function similar to a sigmoid that outputs from 0 to +1, but the slope of the function goes to 0 at approx 50 instead of 5. 
	The function outputs 0 when the input is 0. All input will be positive. 
	'''
	if x.is_sparse:
		indices = x._indices()
		values = x._values()
		transformedValues = hybridActivation(values)
		z = pt.sparse_coo_tensor(indices, transformedValues, x.size(), device=x.device)
	else:
		z = hybridActivation(x)
	return z

def hybridActivation(x, scale=100.0):
	#print("x = ", x)
	f = (pt.sigmoid(x / scale) - 0.5 ) * 2.0
	#print("f = ", f)
	return f
