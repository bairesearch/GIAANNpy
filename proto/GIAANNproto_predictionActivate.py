"""GIAANNproto_predictionActivate.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_main.py

# Usage:
see GIAANNproto_main.py

# Description:
GIA ANN proto Prediction Activate

"""

import torch as pt

from GIAANNproto_globalDefs import *
import GIAANNproto_sparseTensors


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
		featureNeuronsActivation = GIAANNproto_sparseTensors.subtractValueFromSparseTensorValues(featureNeuronsActivation, activationDecrementPerPredictedSequence)
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
			# For useSANIfeaturesAndColumns, enforce sequential activation independently for:
			# a) concept/column segments and b) feature segments.
			featureSegmentsOffset = arrayNumberOfSegmentsColumnDistance
			assert featureSegmentsOffset >= 0 and featureSegmentsOffset < arrayNumberOfSegments
			previousConceptChannelActivation = branchActivation[:featureSegmentsOffset-1] > 0 if featureSegmentsOffset > 1 else None
			previousFeatureChannelActivation = branchActivation[featureSegmentsOffset:arrayNumberOfSegments-1] > 0 if featureSegmentsOffset+1 < arrayNumberOfSegments else None
			if(previousConceptChannelActivation is not None):
				branchActivation[1:featureSegmentsOffset] += branchTargetActivation[1:featureSegmentsOffset] * previousConceptChannelActivation
				branchTargetActivationApplied[1:featureSegmentsOffset] = branchTargetActivation[1:featureSegmentsOffset] * previousConceptChannelActivation
			if(previousFeatureChannelActivation is not None):
				branchActivation[featureSegmentsOffset+1:] += branchTargetActivation[featureSegmentsOffset+1:] * previousFeatureChannelActivation
				branchTargetActivationApplied[featureSegmentsOffset+1:] = branchTargetActivation[featureSegmentsOffset+1:] * previousFeatureChannelActivation
			branchActivation[0] += branchTargetActivation[0]
			branchActivation[featureSegmentsOffset] += branchTargetActivation[featureSegmentsOffset]
			branchTargetActivationApplied[0] = branchTargetActivation[0]
			branchTargetActivationApplied[featureSegmentsOffset] = branchTargetActivation[featureSegmentsOffset]
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
			requireCheck = (segmentIndices != 0) & (segmentIndices != featureSegmentStart)
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
			requiresCheck = pt.zeros((segmentIndices.shape[0],), dtype=pt.bool, device=segmentIndices.device)
			if(useSANIfeaturesAndColumns):
				featureMask = segmentIndices >= arrayNumberOfSegmentsColumnDistance
				columnMask = ~featureMask
				if(columnMask.any()):
					requiresCheck = requiresCheck | (columnMask & (segmentIndices > 0))
				if(featureMask.any()):
					requiresCheck = requiresCheck | (featureMask & (segmentIndices > arrayNumberOfSegmentsColumnDistance))
			else:
				requiresCheck = segmentIndices > 0
			allowedMask = pt.ones((segmentIndices.shape[0],), dtype=pt.bool, device=segmentIndices.device)
			if(requiresCheck.any()):
				checkIndices = pt.nonzero(requiresCheck, as_tuple=False).view(-1)
				localSegmentIndices = segmentIndices.index_select(0, checkIndices)
				prevIndices = indices.index_select(1, checkIndices).clone()
				prevSegmentIndices = localSegmentIndices - 1
				prevIndices[1] = prevSegmentIndices
				prevStoredTimes = gatherSparseTensorValuesAtIndices(globalFeatureNeuronsTime, prevIndices, values.dtype)
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
				allowedMask[checkIndices] = (currentTimeValues - prevStoredTimes) == 1
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

#first dim cs1 restricted to a candiate set of tokens.
def processFeaturesActivePredictMulti(databaseNetworkObject, globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, sequenceObservedColumnsPrediction, conceptColumnsIndices, conceptColumnsFeatureIndices, globalFeatureNeuronsTime=None, sequenceWordIndex=None, sequenceColumnIndex=None):
	#print("processFeaturesActivePredictMulti:")
	for conceptIndex in range(conceptColumnsIndices.shape[0]):
		conceptColumnsIndicesSource = conceptColumnsIndices[conceptIndex].unsqueeze(dim=0)
		conceptColumnsFeatureIndicesSource = conceptColumnsFeatureIndices[conceptIndex].unsqueeze(dim=0)
		sourceConceptIndexValue = conceptColumnsIndicesSource.squeeze().item()
		featureConnections = GIAANNproto_sparseTensors.sliceSparseTensor(sequenceObservedColumnsPrediction.featureConnections, 3, conceptIndex)	#sequence concept index dimension	#CHECKTHIS
		globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, globalFeatureNeuronsTime = processFeaturesActivePredict(databaseNetworkObject, globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, featureConnections, conceptColumnsIndicesSource, conceptColumnsFeatureIndicesSource, sourceConceptIndexValue, globalFeatureNeuronsTime, sequenceWordIndex, sequenceColumnIndex)
	
	return globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, globalFeatureNeuronsTime
	
#first dim cs1 restricted to a single token
def processFeaturesActivePredictSingle(databaseNetworkObject, globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, sequenceObservedColumnsPrediction, conceptColumnsIndices, conceptColumnsFeatureIndices, globalFeatureNeuronsTime=None, sequenceWordIndex=None, sequenceColumnIndex=None):
	featureConnections = GIAANNproto_sparseTensors.sliceSparseTensor(sequenceObservedColumnsPrediction.featureConnections, 3, 0)	#sequence concept index dimension
	sourceConceptIndexValue = conceptColumnsIndices.squeeze().item()
	return processFeaturesActivePredict(databaseNetworkObject, globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, featureConnections, conceptColumnsIndices, conceptColumnsFeatureIndices, sourceConceptIndexValue, globalFeatureNeuronsTime, sequenceWordIndex, sequenceColumnIndex)

def processFeaturesActivePredict(databaseNetworkObject, globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, featureConnections, conceptColumnsIndices, conceptColumnsFeatureIndices, sourceConceptIndex=None, globalFeatureNeuronsTime=None, sequenceWordIndex=None, sequenceColumnIndex=None):
		
	featureNeuronsActive = GIAANNproto_sparseTensors.neuronActivationSparse(globalFeatureNeuronsActivation, algorithmMatrixSANImethod)
	
	sourceColumnIndex = conceptColumnsIndices.squeeze().item()
	sourceFeatureIndex = conceptColumnsFeatureIndices.squeeze().squeeze().item()
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
		# Collapse branch-local source activations so each target branch receives the same drive.
		featureNeuronsActive = featureNeuronsActive.sum()

	#target neuron activation dependence on connection strength;
	featureConnectionsStrength = featureConnections[arrayIndexPropertiesStrengthIndex]
	if(inferenceConnectionStrengthPOSdependence):
		featureConnectionsPos = featureConnections[arrayIndexPropertiesPosIndex]
	featureConnectionsStrength = GIAANNproto_sparseTensors.sliceSparseTensor(featureConnectionsStrength, 2, sourceFeatureIndex)
	if(inferenceConnectionStrengthPOSdependence):
		featureConnectionsPos = GIAANNproto_sparseTensors.sliceSparseTensor(featureConnectionsPos, 2, sourceFeatureIndex)
		featureConnectionsStrength = applyConnectionStrengthPOSdependenceInference(databaseNetworkObject, featureConnectionsStrength, featureConnectionsPos, sourceConceptIndex)
	if(inferenceConnectionsStrengthBoolean):
		featureConnectionsStrength = featureConnectionsStrength.bool().float()
	
	if(featureNeuronsActive.dim() > 0):
		featureNeuronsActive = featureNeuronsActive.reshape(-1)
	if featureConnectionsStrength.is_sparse:
		if(featureNeuronsActive.dim() == 0):
			branchCount = featureConnectionsStrength.size(0)
			branchValues = pt.full((branchCount,), featureNeuronsActive.item(), dtype=featureNeuronsActive.dtype, device=featureNeuronsActive.device)
			featureNeuronsTargetActivation = GIAANNproto_sparseTensors.scaleSparseTensorByBranchValues(featureConnectionsStrength, branchValues)
		else:
			featureNeuronsTargetActivation = GIAANNproto_sparseTensors.scaleSparseTensorByBranchValues(featureConnectionsStrength, featureNeuronsActive)
	else:
		if(featureNeuronsActive.dim() == 0):
			featureNeuronsTargetActivation = featureConnectionsStrength * featureNeuronsActive
		else:
			featureNeuronsTargetActivation = featureConnectionsStrength * featureNeuronsActive.view(-1, 1, 1, 1)

	if(inferenceActivationFunction):
		featureNeuronsTargetActivation = activationFunction(featureNeuronsTargetActivation)
		#print("featureNeuronsTargetActivation = ", featureNeuronsTargetActivation)
	else:
		featureNeuronsTargetActivation = featureNeuronsTargetActivation*j1
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
	if(inferenceSegmentActivationsBoolean):
		globalFeatureNeuronsActivation = applySegmentActivationsBoolean(globalFeatureNeuronsActivation)
	if(inferenceUseNeuronFeaturePropertiesTime):
		# spec step (a): store last timeValue for activated segments during each prediction step
		globalFeatureNeuronsTime = updateTimeValuesFromActivation(globalFeatureNeuronsTime, featureNeuronsTargetActivationApplied, sequenceWordIndex, sequenceColumnIndex)
		
	if(transformerUseInputConnections):
		featureNeuronsTargetActivation = GIAANNproto_sparseTensors.expand_sparse_tensor(featureNeuronsTargetActivation, 2, conceptColumnsIndices.squeeze(), new_dim_size=databaseNetworkObject.c)
		featureNeuronsTargetActivation = GIAANNproto_sparseTensors.expand_sparse_tensor(featureNeuronsTargetActivation, 3, conceptColumnsFeatureIndices.squeeze(), new_dim_size=databaseNetworkObject.f)
		globalFeatureConnectionsActivation = globalFeatureConnectionsActivation + featureNeuronsTargetActivation

	return globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, globalFeatureNeuronsTime

def selectActivatedBranchIndex(globalFeatureNeuronsActivation, columnIndex, featureIndex):
	if(not multipleDendriticBranches):
		return 0
	if(globalFeatureNeuronsActivation is None):
		return 0
	if(globalFeatureNeuronsActivation.is_sparse):
		sparseActivation = globalFeatureNeuronsActivation.coalesce()
		if(sparseActivation._nnz() == 0):
			return 0
		indices = sparseActivation.indices()
		values = sparseActivation.values()
		mask = (indices[2] == columnIndex) & (indices[3] == featureIndex)
		if(not pt.any(mask)):
			return 0
		branchIndices = indices[0, mask].tolist()
		branchValues = values[mask].tolist()
		branchScores = {}
		for branchIndex, value in zip(branchIndices, branchValues):
			branchScores[branchIndex] = branchScores.get(branchIndex, 0.0) + float(value)
		bestBranch = max(branchScores, key=branchScores.get)
		return int(bestBranch)
	activationSlice = globalFeatureNeuronsActivation[:, :, columnIndex, featureIndex]
	if(activationSlice.numel() == 0):
		return 0
	branchScores = activationSlice.sum(dim=1)
	bestBranch = int(pt.argmax(branchScores).item())
	return bestBranch

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

def computeConnectionMinWordDistanceMask(observedColumn, sourceFeatureIndex, targetIndices, requiredDistance=1.0):
	if(enforceDirectConnectionsMinWordDistance):
		if(targetIndices is None or targetIndices.shape[1] == 0):
			printe("(targetIndices is None or targetIndices.shape[1] == 0)")
			#return None
		featureConnectionsMinWordDistance = observedColumn.featureConnections[arrayIndexPropertiesMinWordDistanceIndex]
		featureConnectionsMinWordDistance = GIAANNproto_sparseTensors.sliceSparseTensor(featureConnectionsMinWordDistance, 2, sourceFeatureIndex)
		featureConnectionsMinWordDistance = featureConnectionsMinWordDistance.coalesce()
		if(featureConnectionsMinWordDistance._nnz() == 0):
			printe("(featureConnectionsMinWordDistance._nnz() == 0)")
			#return pt.zeros(targetIndices.shape[1], dtype=pt.bool, device=targetIndices.device)
		minIndices = featureConnectionsMinWordDistance.indices()
		minValues = featureConnectionsMinWordDistance.values()
		minDistanceLookup = {}
		for idx in range(minValues.shape[0]):
			columnValue = int(minIndices[2, idx].item())
			featureValue = int(minIndices[3, idx].item())
			distanceValue = float(minValues[idx].item())
			key = (columnValue, featureValue)
			if(key not in minDistanceLookup or distanceValue < minDistanceLookup[key]):
				minDistanceLookup[key] = distanceValue
		maskList = []
		for idx in range(targetIndices.shape[1]):
			columnValue = int(targetIndices[2, idx].item())
			featureValue = int(targetIndices[3, idx].item())
			distanceValue = minDistanceLookup.get((columnValue, featureValue))
			if(distanceValue is None):
				maskList.append(False)
			else:
				maskList.append(abs(distanceValue - requiredDistance) < 1e-4)
		if(len(maskList) == 0):
			mask = pt.zeros(0, dtype=pt.bool, device=targetIndices.device)
		else:
			mask = pt.tensor(maskList, dtype=pt.bool, device=targetIndices.device)
		if(debugPrintMinWordDistanceDetails):
			printMinWordDistanceDetails(observedColumn, sourceFeatureIndex, targetIndices, mask, minDistanceLookup)
		#print("mask = ", mask)
	else:
		mask = None
	return mask
	
def printMinWordDistanceDetails(observedColumn, sourceFeatureIndex, targetIndices, mask, minDistanceLookup):
	databaseNetworkObject = getattr(observedColumn, "databaseNetworkObject", None)
	sourceColumnName = getattr(observedColumn, "conceptName", "<unknown>")
	if(databaseNetworkObject is not None and 0 <= sourceFeatureIndex < len(databaseNetworkObject.conceptFeaturesList)):
		sourceFeatureName = databaseNetworkObject.conceptFeaturesList[sourceFeatureIndex]
	else:
		sourceFeatureName = f"<feature:{sourceFeatureIndex}>"
	for idx in range(targetIndices.shape[1]):
		columnValue = int(targetIndices[2, idx].item())
		featureValue = int(targetIndices[2, idx].item())
		distanceValue = minDistanceLookup.get((columnValue, featureValue))
		keepConnection = (mask[idx].item() == 1) if (mask is not None and idx < mask.shape[0]) else False
		columnName = f"<column:{columnValue}>"
		featureName = f"<feature:{featureValue}>"
		if(databaseNetworkObject is not None):
			if(0 <= columnValue < len(databaseNetworkObject.conceptColumnsList)):
				columnName = databaseNetworkObject.conceptColumnsList[columnValue]
			if(0 <= featureValue < len(databaseNetworkObject.conceptFeaturesList)):
				featureName = databaseNetworkObject.conceptFeaturesList[featureValue]
		print(f"debugMinDistance: source {sourceColumnName}:{sourceFeatureName} -> target {columnName}:{featureName} distance={distanceValue} keep={keepConnection}")
