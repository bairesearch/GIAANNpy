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
		if(activationSparse._nnz() == 0):
			result = activationSparse
		else:
			indices = activationSparse.indices()
			values = activationSparse.values()
			storedTimes = gatherSparseTensorValuesAtIndices(globalFeatureNeuronsTime, indices, values.dtype)
			segmentIndices = indices[1]
			penaltyValues = computeTimePenaltyForSegments(segmentIndices, storedTimes, sequenceWordIndex, sequenceColumnIndex)
			modifiedValues = values - penaltyValues
			if(inferenceUseNeuronFeaturePropertiesTimeExact):
				# spec step (b): require exact time match for all segments of a neuron when inferenceUseNeuronFeaturePropertiesTimeExact is enabled.
				groupIndices = indices[[0, 2, 3], :].t()
				uniqueGroups, groupInverse = pt.unique(groupIndices, dim=0, return_inverse=True)
				mismatchValues = (penaltyValues != 0).to(pt.int64)
				groupMismatch = pt.zeros((uniqueGroups.shape[0],), dtype=mismatchValues.dtype, device=mismatchValues.device)
				groupMismatch.index_add_(0, groupInverse, mismatchValues)
				groupExactMask = groupMismatch == 0
				modifiedValues = values * groupExactMask[groupInverse].to(values.dtype)
			if(debugInferenceUseNeuronFeaturePropertiesTime and sequenceWordIndex >= debugInferenceUseNeuronFeaturePropertiesTimeMinSequenceWordIndex):
				# spec step (b): debug time-based activation modifier per segment.
				penaltyMin = float(penaltyValues.min().item())
				penaltyMean = float(penaltyValues.mean().item())
				penaltyMax = float(penaltyValues.max().item())
				modifiedMin = float(modifiedValues.min().item())
				modifiedMean = float(modifiedValues.mean().item())
				modifiedMax = float(modifiedValues.max().item())
				print(f"debugInferenceUseNeuronFeaturePropertiesTime: sequenceWordIndex={sequenceWordIndex}, sequenceColumnIndex={sequenceColumnIndex}, nnz={int(values.shape[0])}, penaltyMin={penaltyMin}, penaltyMean={penaltyMean}, penaltyMax={penaltyMax}, modifiedMin={modifiedMin}, modifiedMean={modifiedMean}, modifiedMax={modifiedMax}")
				offsetValues = pt.zeros_like(storedTimes)
				currentTimeValues = pt.zeros_like(storedTimes)
				if(useSANIfeaturesAndColumns):
					featureMask = segmentIndices >= arrayNumberOfSegmentsColumnDistance
					if(featureMask.any()):
						localSegmentIndex = (segmentIndices[featureMask] - arrayNumberOfSegmentsColumnDistance).to(storedTimes.dtype)
						offsetValues[featureMask] = float(arrayNumberOfSegmentsFeatureDistance) - localSegmentIndex
						currentTimeValues[featureMask] = float(sequenceWordIndex)
					if((~featureMask).any()):
						localSegmentIndex = segmentIndices[~featureMask].to(storedTimes.dtype)
						offsetValues[~featureMask] = float(arrayNumberOfSegmentsColumnDistance) - localSegmentIndex
						currentTimeValues[~featureMask] = float(sequenceColumnIndex)
				elif(useSANIfeatures):
					localSegmentIndex = segmentIndices.to(storedTimes.dtype)
					offsetValues = float(arrayNumberOfSegments) - localSegmentIndex
					currentTimeValues = pt.full_like(storedTimes, float(sequenceWordIndex))
				elif(useSANIcolumns):
					localSegmentIndex = segmentIndices.to(storedTimes.dtype)
					offsetValues = float(arrayNumberOfSegments) - localSegmentIndex
					currentTimeValues = pt.full_like(storedTimes, float(sequenceColumnIndex))
				else:
					raise RuntimeError("applyTimeBasedActivationModifier: useSANI feature mode not configured")
				sampleIndex = int(pt.argmax(penaltyValues).item())
				sampleBranch = int(indices[0, sampleIndex].item())
				sampleSegment = int(segmentIndices[sampleIndex].item())
				sampleColumn = int(indices[2, sampleIndex].item())
				sampleFeature = int(indices[3, sampleIndex].item())
				sampleActivation = float(values[sampleIndex].item())
				sampleStoredTime = float(storedTimes[sampleIndex].item())
				sampleCurrentTime = float(currentTimeValues[sampleIndex].item())
				sampleOffset = float(offsetValues[sampleIndex].item())
				samplePenalty = float(penaltyValues[sampleIndex].item())
				sampleModified = float(modifiedValues[sampleIndex].item())
				print(f"debugInferenceUseNeuronFeaturePropertiesTime: sample=penaltyMax, branch={sampleBranch}, segment={sampleSegment}, column={sampleColumn}, feature={sampleFeature}, activation={sampleActivation}, storedTime={sampleStoredTime}, currentTime={sampleCurrentTime}, offset={sampleOffset}, penalty={samplePenalty}, modified={sampleModified}")
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
	if(inferencePredictiveNetwork and not useGPUsparse):
		conceptColumnsFeatureIndices = conceptColumnsFeatureIndices.to(deviceSparse)
	featureConnectionsStrength = GIAANNproto_sparseTensors.sliceSparseTensor(featureConnectionsStrength, 2, sourceFeatureIndex)
	if(inferenceConnectionStrengthPOSdependence):
		featureConnectionsPos = GIAANNproto_sparseTensors.sliceSparseTensor(featureConnectionsPos, 2, sourceFeatureIndex)
		featureConnectionsStrength = applyConnectionStrengthPOSdependenceInference(databaseNetworkObject, featureConnectionsStrength, featureConnectionsPos, sourceConceptIndex)
	if(inferenceConnectionsStrengthBoolean):
		featureConnectionsStrength = featureConnectionsStrength.bool().float()
	if(inferenceUseNeuronFeaturePropertiesTime and debugInferenceUseNeuronFeaturePropertiesTimeTargeted):
		debugSequenceWordIndex = debugInferenceUseNeuronFeaturePropertiesTimeTargetSequenceWordIndex
		if(debugSequenceWordIndex is None or sequenceWordIndex == debugSequenceWordIndex):
			if(debugInferenceUseNeuronFeaturePropertiesTimeTargetColumnName is None or debugInferenceUseNeuronFeaturePropertiesTimeTargetFeatureNames is None):
				raise RuntimeError("processFeaturesActivePredict: debug target column/feature names not configured")
			if(debugInferenceUseNeuronFeaturePropertiesTimeTargetColumnName not in databaseNetworkObject.conceptColumnsList):
				raise RuntimeError("processFeaturesActivePredict: debug target column not found")
			debugTargetColumnIndex = databaseNetworkObject.conceptColumnsList.index(debugInferenceUseNeuronFeaturePropertiesTimeTargetColumnName)
			if(sourceConceptIndex == debugTargetColumnIndex and sourceFeatureIndex == featureIndexConceptNeuron):
				debugTargetFeatureIndices = [databaseNetworkObject.conceptFeaturesList.index(featureName) if featureName in databaseNetworkObject.conceptFeaturesList else -1 for featureName in debugInferenceUseNeuronFeaturePropertiesTimeTargetFeatureNames]
				if(-1 in debugTargetFeatureIndices):
					raise RuntimeError("processFeaturesActivePredict: debug target feature not found")
				debugDevice = featureConnectionsStrength.device
				debugBranchCount = numberOfDendriticBranches if multipleDendriticBranches else 1
				debugBranchRange = pt.arange(debugBranchCount, dtype=pt.long, device=debugDevice)
				debugSegmentRange = pt.arange(arrayNumberOfSegments, dtype=pt.long, device=debugDevice)
				debugBranchIndices = debugBranchRange.repeat_interleave(arrayNumberOfSegments)
				debugSegmentIndices = debugSegmentRange.repeat(debugBranchCount)
				debugValueDtype = featureConnectionsStrength.values().dtype if featureConnectionsStrength.is_sparse else featureConnectionsStrength.dtype
				for debugFeatureIndex, debugFeatureName in zip(debugTargetFeatureIndices, debugInferenceUseNeuronFeaturePropertiesTimeTargetFeatureNames):
					debugColumnIndices = pt.full_like(debugSegmentIndices, debugTargetColumnIndex)
					debugFeatureIndices = pt.full_like(debugSegmentIndices, debugFeatureIndex)
					debugIndices = pt.stack([debugBranchIndices, debugSegmentIndices, debugColumnIndices, debugFeatureIndices], dim=0)
					debugConnectionValues = gatherSparseTensorValuesAtIndices(featureConnectionsStrength, debugIndices, debugValueDtype)
					debugConnectionByBranch = debugConnectionValues.view(debugBranchCount, arrayNumberOfSegments).sum(dim=1)
					debugConnectionBySegment = debugConnectionValues.view(debugBranchCount, arrayNumberOfSegments).max(dim=0).values
					print(f"debugInferenceUseNeuronFeaturePropertiesTimeTargetSource: sequenceWordIndex={sequenceWordIndex}, sourceColumn={debugInferenceUseNeuronFeaturePropertiesTimeTargetColumnName}({debugTargetColumnIndex}), sourceFeature={sourceFeatureIndex}, targetFeature={debugFeatureName}({debugFeatureIndex}), connectionByBranch={debugConnectionByBranch.tolist()}, connectionBySegmentMax={debugConnectionBySegment.tolist()}")
	
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

	featureNeuronsTargetActivationApplied = featureNeuronsTargetActivation

	#update the activations of the target nodes;
	if(useSANI):
		if(algorithmMatrixSANImethod=="enforceActivationAcrossSegments"):
			if(enforceSequentialActivation):
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
				globalFeatureNeuronsActivation = globalFeatureNeuronsActivationDense.to_sparse_coo()
				if(featureNeuronsTargetActivation.is_sparse):
					featureNeuronsTargetActivationApplied = featureNeuronsTargetActivationAppliedDense.to_sparse_coo()
				else:
					featureNeuronsTargetActivationApplied = featureNeuronsTargetActivationAppliedDense
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
