"""GIAANNcmn_inferenceDuringTrain.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 BAI Research Pty Ltd (bairesearch.com.au)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN common train-during-inference helpers

"""

import torch as pt

from GIAANNcmn_globalDefs import *
import GIAANNcmn_sparseTensors


def initialiseConnectionsActive(databaseNetworkObject):
	result = None
	if(not useTrainDuringInference or executionMode!="trainDuringInference"):
		raise RuntimeError("initialiseConnectionsActive error: requires executionMode trainDuringInference")
	validateDatabaseNetworkObject(databaseNetworkObject, "initialiseConnectionsActive")
	databaseNetworkObject.inferenceDuringTrainConnectionsActive = {}
	return result

def updateInferenceDuringTrainConnectionsActive(databaseNetworkObject, featureNeuronsTargetActivation, featureConnectionsStrengthStored, sourceColumnIndex, sourceFeatureIndex):
	result = None
	if(inferenceDuringTrainAdjustSynapseStrength):
		if(inferenceDuringTrainAdjustSynapseStrengthDecrementInference):
			if(not useTrainDuringInference or executionMode!="trainDuringInference"):
				raise RuntimeError("updateInferenceDuringTrainConnectionsActive error: requires executionMode trainDuringInference")
			getConnectionsActiveDictionary(databaseNetworkObject, "updateInferenceDuringTrainConnectionsActive")
			if(featureNeuronsTargetActivation is None):
				raise RuntimeError("updateInferenceDuringTrainConnectionsActive error: featureNeuronsTargetActivation is None")
			if(featureConnectionsStrengthStored is None):
				raise RuntimeError("updateInferenceDuringTrainConnectionsActive error: featureConnectionsStrengthStored is None")
			if(not pt.is_tensor(featureNeuronsTargetActivation) or not pt.is_tensor(featureConnectionsStrengthStored)):
				raise RuntimeError("updateInferenceDuringTrainConnectionsActive error: tensors are required")
			activationSparse = featureNeuronsTargetActivation
			if(not activationSparse.is_sparse):
				activationSparse = activationSparse.to_sparse_coo()
			activationSparse = activationSparse.coalesce()
			if(activationSparse.dim() != inferenceDuringTrainTargetConnectionTensorRank):
				raise RuntimeError("updateInferenceDuringTrainConnectionsActive error: activation tensor rank invalid")
			if(activationSparse._nnz() > 0):
				activationIndices = activationSparse.indices()
				activationValues = activationSparse.values()
				activationMask = activationValues > inferenceDuringTrainConnectionStrengthActiveThreshold
				if(activationMask.any()):
					activeIndices = activationIndices[:, activationMask]
					connectionStrengthValues = GIAANNcmn_sparseTensors.gatherSparseTensorValuesAtIndices(featureConnectionsStrengthStored, activeIndices, activationValues.dtype)
					connectionStrengthMask = connectionStrengthValues > inferenceDuringTrainConnectionStrengthActiveThreshold
					if(connectionStrengthMask.any()):
						filteredActiveIndices = activeIndices[:, connectionStrengthMask]
						filteredConnectionStrengthValues = connectionStrengthValues[connectionStrengthMask]
						connectionIndices = pt.stack((filteredActiveIndices[inferenceDuringTrainTargetConnectionIndexBranch], filteredActiveIndices[inferenceDuringTrainTargetConnectionIndexSegment], filteredActiveIndices[inferenceDuringTrainTargetConnectionIndexTargetConcept], filteredActiveIndices[inferenceDuringTrainTargetConnectionIndexTargetFeature]), dim=0)
						connectionSize = getSourceFeatureTensorSize(databaseNetworkObject, "updateInferenceDuringTrainConnectionsActive")
						connectionSparse = pt.sparse_coo_tensor(connectionIndices, filteredConnectionStrengthValues, size=connectionSize, dtype=arrayType, device=filteredConnectionStrengthValues.device).coalesce()
						updateSourceFeatureConnectionsActive(databaseNetworkObject, int(sourceColumnIndex), int(sourceFeatureIndex), connectionSparse, "updateInferenceDuringTrainConnectionsActive")
	return result

def applyInferenceDuringTrainAdjustSynapseStrengthDecrementDense(sequenceObservedColumns, featureConnectionsActive, featureConnectionsStrengthUpdate, cs, fs):
	result = featureConnectionsStrengthUpdate
	if(inferenceDuringTrainAdjustSynapseStrength):
		if(inferenceDuringTrainAdjustSynapseStrengthDecrementInference):
			if(featureConnectionsActive is None):
				raise RuntimeError("applyInferenceDuringTrainAdjustSynapseStrengthDecrementDense error: featureConnectionsActive is None")
			if(featureConnectionsStrengthUpdate is None):
				raise RuntimeError("applyInferenceDuringTrainAdjustSynapseStrengthDecrementDense error: featureConnectionsStrengthUpdate is None")
			if(not pt.is_tensor(featureConnectionsActive) or not pt.is_tensor(featureConnectionsStrengthUpdate)):
				raise RuntimeError("applyInferenceDuringTrainAdjustSynapseStrengthDecrementDense error: tensors are required")
			if(featureConnectionsActive.shape != featureConnectionsStrengthUpdate.shape):
				raise RuntimeError("applyInferenceDuringTrainAdjustSynapseStrengthDecrementDense error: tensor shape mismatch")
			connectionActiveSparse = featureConnectionsActive.to_sparse_coo().coalesce()
			connectionDecrementValues = calculateInferenceDuringTrainAdjustSynapseStrengthDecrementValues(sequenceObservedColumns, connectionActiveSparse)
			connectionActiveIndices = connectionActiveSparse.indices()
			decrementMask = connectionDecrementValues > inferenceDuringTrainConnectionStrengthActiveThreshold
			if(decrementMask.any()):
				connectionDecrementSparse = pt.sparse_coo_tensor(connectionActiveIndices[:, decrementMask], connectionDecrementValues[decrementMask], size=(multipleDendriticBranchesNumber, arrayNumberOfSegments, cs, fs, cs, fs), dtype=arrayType, device=featureConnectionsStrengthUpdate.device).coalesce()
				result = featureConnectionsStrengthUpdate - connectionDecrementSparse.to_dense()
	return result

def applyInferenceDuringTrainAdjustSynapseStrengthDecrementSparse(sequenceObservedColumns, connectionActiveSparse, strengthValues):
	result = strengthValues
	if(inferenceDuringTrainAdjustSynapseStrength):
		if(inferenceDuringTrainAdjustSynapseStrengthDecrementInference):
			if(strengthValues is None):
				raise RuntimeError("applyInferenceDuringTrainAdjustSynapseStrengthDecrementSparse error: strengthValues is None")
			connectionDecrementValues = calculateInferenceDuringTrainAdjustSynapseStrengthDecrementValues(sequenceObservedColumns, connectionActiveSparse)
			if(connectionDecrementValues.shape != strengthValues.shape):
				raise RuntimeError("applyInferenceDuringTrainAdjustSynapseStrengthDecrementSparse error: decrement shape mismatch")
			result = strengthValues - connectionDecrementValues
	return result

def calculateInferenceDuringTrainAdjustSynapseStrengthDecrementValues(sequenceObservedColumns, connectionActiveSparse):
	result = None
	if(inferenceDuringTrainAdjustSynapseStrength):
		if(not useTrainDuringInference or executionMode!="trainDuringInference"):
			raise RuntimeError("calculateInferenceDuringTrainAdjustSynapseStrengthDecrementValues error: requires executionMode trainDuringInference")
		if(sequenceObservedColumns is None):
			raise RuntimeError("calculateInferenceDuringTrainAdjustSynapseStrengthDecrementValues error: sequenceObservedColumns is None")
		if(not hasattr(sequenceObservedColumns, "databaseNetworkObject")):
			raise RuntimeError("calculateInferenceDuringTrainAdjustSynapseStrengthDecrementValues error: databaseNetworkObject is unavailable")
		if(connectionActiveSparse is None):
			raise RuntimeError("calculateInferenceDuringTrainAdjustSynapseStrengthDecrementValues error: connectionActiveSparse is None")
		if(not pt.is_tensor(connectionActiveSparse) or connectionActiveSparse.layout != pt.sparse_coo):
			raise RuntimeError("calculateInferenceDuringTrainAdjustSynapseStrengthDecrementValues error: connectionActiveSparse must be sparse COO")
		connectionActiveSparse = connectionActiveSparse.coalesce()
		if(connectionActiveSparse.dim() != inferenceDuringTrainConnectionTensorRank):
			raise RuntimeError("calculateInferenceDuringTrainAdjustSynapseStrengthDecrementValues error: connectionActiveSparse rank invalid")
		connectionActiveIndices = connectionActiveSparse.indices()
		connectionActiveValues = connectionActiveSparse.values()
		result = pt.zeros_like(connectionActiveValues)
		if(connectionActiveIndices.numel() > 0):
			databaseNetworkObject = sequenceObservedColumns.databaseNetworkObject
			getConnectionsActiveDictionary(databaseNetworkObject, "calculateInferenceDuringTrainAdjustSynapseStrengthDecrementValues")
			conceptIndicesTensor = sequenceObservedColumns.conceptIndicesInSequenceObservedTensor.to(connectionActiveIndices.device)
			featureIndicesInObservedTensor = sequenceObservedColumns.featureIndicesInObservedTensor.to(connectionActiveIndices.device)
			sourceConceptIndices = conceptIndicesTensor[connectionActiveIndices[inferenceDuringTrainConnectionIndexSourceConcept]]
			targetConceptIndices = conceptIndicesTensor[connectionActiveIndices[inferenceDuringTrainConnectionIndexTargetConcept]]
			sourceFeatureIndices = connectionActiveIndices[inferenceDuringTrainConnectionIndexSourceFeature]
			targetFeatureIndices = connectionActiveIndices[inferenceDuringTrainConnectionIndexTargetFeature]
			if(trainSequenceObservedColumnsUseSequenceFeaturesOnly):
				sourceFeatureIndices = featureIndicesInObservedTensor[sourceFeatureIndices]
				targetFeatureIndices = featureIndicesInObservedTensor[targetFeatureIndices]
			inferenceConnectionStrengthValues = gatherConnectionsActiveValues(databaseNetworkObject, connectionActiveIndices[inferenceDuringTrainConnectionIndexBranch], connectionActiveIndices[inferenceDuringTrainConnectionIndexSegment], sourceConceptIndices, sourceFeatureIndices, targetConceptIndices, targetFeatureIndices, connectionActiveValues.dtype)
			inferenceConnectionMask = inferenceConnectionStrengthValues > inferenceDuringTrainConnectionStrengthActiveThreshold
			if(inferenceConnectionMask.any()):
				if(inferenceDuringTrainDecrementNonlinear):
					connectionDecrementValues = inferenceConnectionStrengthValues[inferenceConnectionMask] * inferenceDuringTrainDecrement
				else:
					connectionDecrementValues = pt.full_like(inferenceConnectionStrengthValues[inferenceConnectionMask], inferenceDuringTrainDecrement)
				result[inferenceConnectionMask] = connectionDecrementValues
	return result

def updateSourceFeatureConnectionsActive(databaseNetworkObject, sourceColumnIndex, sourceFeatureIndex, connectionSparse, callerName):
	result = None
	if(connectionSparse is None):
		raise RuntimeError(callerName + " error: connectionSparse is None")
	validateSourceFeatureConnectionsActiveTensor(connectionSparse, databaseNetworkObject, callerName)
	sourceTensor = getSourceFeatureConnectionsActive(databaseNetworkObject, sourceColumnIndex, sourceFeatureIndex, connectionSparse.device, True, callerName)
	sourceTensor = GIAANNcmn_sparseTensors.maximumSparseTensorValues(sourceTensor, connectionSparse)
	setSourceFeatureConnectionsActive(databaseNetworkObject, sourceColumnIndex, sourceFeatureIndex, sourceTensor, callerName)
	return result

def gatherConnectionsActiveValues(databaseNetworkObject, branchIndices, segmentIndices, sourceConceptIndices, sourceFeatureIndices, targetConceptIndices, targetFeatureIndices, dtype):
	result = None
	callerName = "gatherConnectionsActiveValues"
	validateDatabaseNetworkObject(databaseNetworkObject, callerName)
	if(dtype is None):
		raise RuntimeError(callerName + " error: dtype is None")
	expectedShape = validateGatherIndexTensor(branchIndices, callerName, "branchIndices")
	validateGatherIndexTensorShape(segmentIndices, expectedShape, callerName, "segmentIndices")
	validateGatherIndexTensorShape(sourceConceptIndices, expectedShape, callerName, "sourceConceptIndices")
	validateGatherIndexTensorShape(sourceFeatureIndices, expectedShape, callerName, "sourceFeatureIndices")
	validateGatherIndexTensorShape(targetConceptIndices, expectedShape, callerName, "targetConceptIndices")
	validateGatherIndexTensorShape(targetFeatureIndices, expectedShape, callerName, "targetFeatureIndices")
	getConnectionsActiveDictionary(databaseNetworkObject, callerName)
	result = pt.zeros(expectedShape, dtype=dtype, device=branchIndices.device)
	if(branchIndices.shape[0] > inferenceDuringTrainMinimumIndex):
		sourceCombinedKeys = sourceConceptIndices*databaseNetworkObject.f + sourceFeatureIndices
		sourceCombinedKeysUnique = pt.unique(sourceCombinedKeys, sorted=True)
		for sourceCombinedKey in sourceCombinedKeysUnique.detach().cpu().tolist():
			sourceMask = sourceCombinedKeys == int(sourceCombinedKey)
			if(sourceMask.any()):
				sourceConceptIndex = int(sourceConceptIndices[sourceMask][inferenceDuringTrainFirstTensorElementIndex].item())
				sourceFeatureIndex = int(sourceFeatureIndices[sourceMask][inferenceDuringTrainFirstTensorElementIndex].item())
				sourceTensor = getSourceFeatureConnectionsActive(databaseNetworkObject, sourceConceptIndex, sourceFeatureIndex, branchIndices.device, False, callerName)
				if(sourceTensor is not None):
					queryIndices = pt.stack((branchIndices[sourceMask], segmentIndices[sourceMask], targetConceptIndices[sourceMask], targetFeatureIndices[sourceMask]), dim=0)
					result[sourceMask] = GIAANNcmn_sparseTensors.gatherSparseTensorValuesAtIndices(sourceTensor, queryIndices, dtype)
	return result

def validateDatabaseNetworkObject(databaseNetworkObject, callerName):
	result = None
	if(databaseNetworkObject is None):
		raise RuntimeError(callerName + " error: databaseNetworkObject is None")
	if(databaseNetworkObject.c < inferenceDuringTrainMinimumIndex or databaseNetworkObject.f < inferenceDuringTrainMinimumIndex):
		raise RuntimeError(callerName + " error: database dimensions must be non-negative")
	return result

def getConnectionsActiveDictionary(databaseNetworkObject, callerName):
	result = None
	validateDatabaseNetworkObject(databaseNetworkObject, callerName)
	if(not hasattr(databaseNetworkObject, "inferenceDuringTrainConnectionsActive")):
		raise RuntimeError(callerName + " error: inferenceDuringTrainConnectionsActive has not been initialised")
	if(not isinstance(databaseNetworkObject.inferenceDuringTrainConnectionsActive, dict)):
		raise RuntimeError(callerName + " error: inferenceDuringTrainConnectionsActive must be a dict")
	result = databaseNetworkObject.inferenceDuringTrainConnectionsActive
	return result

def createSourceFeatureKey(databaseNetworkObject, sourceColumnIndex, sourceFeatureIndex, callerName):
	result = None
	validateDatabaseNetworkObject(databaseNetworkObject, callerName)
	sourceColumnIndex = int(sourceColumnIndex)
	sourceFeatureIndex = int(sourceFeatureIndex)
	if(sourceColumnIndex < inferenceDuringTrainMinimumIndex or sourceColumnIndex >= databaseNetworkObject.c):
		raise RuntimeError(callerName + " error: source column index out of range")
	if(sourceFeatureIndex < inferenceDuringTrainMinimumIndex or sourceFeatureIndex >= databaseNetworkObject.f):
		raise RuntimeError(callerName + " error: source feature index out of range")
	result = (sourceColumnIndex, sourceFeatureIndex)
	return result

def getSourceFeatureTensorSize(databaseNetworkObject, callerName):
	result = None
	validateDatabaseNetworkObject(databaseNetworkObject, callerName)
	result = (multipleDendriticBranchesNumber, arrayNumberOfSegments, databaseNetworkObject.c, databaseNetworkObject.f)
	return result

def validateSourceFeatureConnectionsActiveTensor(sourceTensor, databaseNetworkObject, callerName):
	result = None
	expectedSize = getSourceFeatureTensorSize(databaseNetworkObject, callerName)
	if(sourceTensor is None):
		raise RuntimeError(callerName + " error: sourceTensor is None")
	if(not pt.is_tensor(sourceTensor)):
		raise RuntimeError(callerName + " error: sourceTensor must be a tensor")
	if(sourceTensor.layout != pt.sparse_coo):
		raise RuntimeError(callerName + " error: sourceTensor must be sparse COO")
	if(sourceTensor.dim() != inferenceDuringTrainTargetConnectionTensorRank):
		raise RuntimeError(callerName + " error: sourceTensor rank invalid")
	if(tuple(sourceTensor.size()) != tuple(expectedSize)):
		raise RuntimeError(callerName + " error: sourceTensor size mismatch")
	return result

def createEmptySourceFeatureConnectionsActiveTensor(databaseNetworkObject, targetDevice, callerName):
	result = GIAANNcmn_sparseTensors.createEmptySparseTensor(getSourceFeatureTensorSize(databaseNetworkObject, callerName))
	if(targetDevice is not None and result.device != targetDevice):
		result = result.to(targetDevice)
	return result

def getSourceFeatureConnectionsActive(databaseNetworkObject, sourceColumnIndex, sourceFeatureIndex, targetDevice, createMissing, callerName):
	result = None
	if(not isinstance(createMissing, bool)):
		raise RuntimeError(callerName + " error: createMissing must be bool")
	connectionsActive = getConnectionsActiveDictionary(databaseNetworkObject, callerName)
	sourceKey = createSourceFeatureKey(databaseNetworkObject, sourceColumnIndex, sourceFeatureIndex, callerName)
	result = connectionsActive.get(sourceKey)
	if(result is None):
		if(createMissing):
			result = createEmptySourceFeatureConnectionsActiveTensor(databaseNetworkObject, targetDevice, callerName)
			connectionsActive[sourceKey] = result
	else:
		validateSourceFeatureConnectionsActiveTensor(result, databaseNetworkObject, callerName)
		if(targetDevice is not None and result.device != targetDevice):
			result = result.to(targetDevice)
			connectionsActive[sourceKey] = result
	return result

def setSourceFeatureConnectionsActive(databaseNetworkObject, sourceColumnIndex, sourceFeatureIndex, sourceTensor, callerName):
	result = None
	connectionsActive = getConnectionsActiveDictionary(databaseNetworkObject, callerName)
	sourceKey = createSourceFeatureKey(databaseNetworkObject, sourceColumnIndex, sourceFeatureIndex, callerName)
	validateSourceFeatureConnectionsActiveTensor(sourceTensor, databaseNetworkObject, callerName)
	connectionsActive[sourceKey] = sourceTensor
	return result

def validateGatherIndexTensor(indexTensor, callerName, tensorName):
	result = None
	if(indexTensor is None):
		raise RuntimeError(callerName + " error: " + tensorName + " is None")
	if(not pt.is_tensor(indexTensor)):
		raise RuntimeError(callerName + " error: " + tensorName + " must be a tensor")
	if(indexTensor.dim() != inferenceDuringTrainGatherIndexTensorRank):
		raise RuntimeError(callerName + " error: " + tensorName + " rank invalid")
	result = tuple(indexTensor.shape)
	return result

def validateGatherIndexTensorShape(indexTensor, expectedShape, callerName, tensorName):
	result = validateGatherIndexTensor(indexTensor, callerName, tensorName)
	if(result != tuple(expectedShape)):
		raise RuntimeError(callerName + " error: " + tensorName + " shape mismatch")
	return result
