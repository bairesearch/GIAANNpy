"""GIAANNproto_databaseNetworkTrain.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_main.py

# Usage:
see GIAANNproto_main.py

# Description:
GIA ANN proto database Network Train

"""

import torch as pt
import time

from GIAANNproto_globalDefs import *
import GIAANNproto_debug
import GIAANNproto_sparseTensors
import GIAANNproto_sequenceConcepts

	
def trainConceptWords(sequenceObservedColumns, sequenceIndex, sequence, tokens):
	trainConceptWordsStartTime = None
	if(debugPrintTrainSectionTimes):
		trainConceptWordsStartTime = time.perf_counter()
	result = GIAANNproto_sequenceConcepts.processConceptWords(sequenceObservedColumns, sequenceIndex, sequence, tokens)
	if(printTrainSequenceConceptAssignment):
		print(f"Processing sequenceCount: {sequenceIndex}, {sequenceObservedColumns.sentenceWithConceptAssignment}")	
		if(printTrainSequenceConceptAssignmentByLine):
			print("")	
	if(result is None):
		return False
	conceptIndices, startIndices, endIndices = result
	featureNeuronsActive, cs, fs, sequenceConceptIndexMask, columnsWordOrder, featureNeuronsWordOrder, featureNeuronsPos, featureNeuronsSegmentMask = GIAANNproto_sequenceConcepts.processFeatures(sequenceObservedColumns, sequenceIndex, sequence, tokens, conceptIndices, startIndices, endIndices)

	featureConnectionsActive, featureConnectionsSegmentMask = processFeaturesActiveTrain(sequenceObservedColumns, featureNeuronsActive, cs, fs, sequenceConceptIndexMask, columnsWordOrder, featureNeuronsWordOrder, featureNeuronsPos, featureNeuronsSegmentMask, sequenceIndex)
	if(debugPrintTrainSectionTimes):
		GIAANNproto_debug.debugTrainSectionTimesAdd(sequenceObservedColumns.databaseNetworkObject, "trainConceptWords", time.perf_counter() - trainConceptWordsStartTime)

	return True

#first dim cs1 pertains to every concept node in sequence
def processFeaturesActiveTrain(sequenceObservedColumns, featureNeuronsActive, cs, fs, sequenceConceptIndexMask, columnsWordOrder, featureNeuronsWordOrder, featureNeuronsPos, featureNeuronsSegmentMask, sequenceIndex):
	databaseNetworkObject = sequenceObservedColumns.databaseNetworkObject
	processFeaturesActiveTrainStartTime = None
	if(debugPrintTrainSectionTimes):
		processFeaturesActiveTrainStartTime = time.perf_counter()
	useSparseSequenceNeurons = trainSparseNeuronsTensor
	useSparseSequenceConnections = trainSparseConnectionsTensor
	if(useSparseSequenceNeurons):
		if(not useSparseSequenceConnections):
			raise RuntimeError("processFeaturesActiveTrain error: trainSparseNeuronsTensor requires trainSparseConnectionsTensor")
		if(trainDecreasePermanenceOfInactiveFeatureNeuronsAndConnections and arrayIndexPropertiesPermanence):
			raise RuntimeError("processFeaturesActiveTrain error: trainSparseNeuronsTensor does not support trainDecreasePermanenceOfInactiveFeatureNeuronsAndConnections")
	if(useSparseSequenceConnections):
		if(trainDecreasePermanenceOfInactiveFeatureNeuronsAndConnections and arrayIndexPropertiesPermanence):
			raise RuntimeError("processFeaturesActiveTrain error: trainSparseConnectionsTensor does not support trainDecreasePermanenceOfInactiveFeatureNeuronsAndConnections")
	if(trainDecreasePermanenceOfInactiveFeatureNeuronsAndConnections and arrayIndexPropertiesPermanence):
		featureNeuronsActiveUnion = featureNeuronsActive.amax(dim=(0, 1))
		featureNeuronsInactiveUnion = 1 - featureNeuronsActiveUnion
	if(useSparseSequenceNeurons):
		processFeaturesActiveTrainSparseNeurons(sequenceObservedColumns, featureNeuronsActive, cs, fs, featureNeuronsPos)
	else:
		featureNeuronsInactive = 1 - featureNeuronsActive
		if(arrayIndexPropertiesStrength):
			sequenceObservedColumns.featureNeurons[databaseNetworkObject.arrayIndexPropertiesStrengthIndex, :, :, :, :] += featureNeuronsActive
		if(arrayIndexPropertiesPermanence):
			sequenceObservedColumns.featureNeurons[databaseNetworkObject.arrayIndexPropertiesPermanenceIndex, :, :, :, :] += featureNeuronsActive*z1
		if(arrayIndexPropertiesActivation):
			sequenceObservedColumns.featureNeurons[databaseNetworkObject.arrayIndexPropertiesActivationIndex, :, :, :, :] = 0
		if(arrayIndexPropertiesTime):
			sequenceObservedColumns.featureNeurons[databaseNetworkObject.arrayIndexPropertiesTimeIndex, :, :, :, :] = 0
			#OLD inferenceUseNeuronFeaturePropertiesTime=False: sequenceObservedColumns.featureNeurons[databaseNetworkObject.arrayIndexPropertiesTimeIndex, :, :, :, :] = featureNeuronsInactive*sequenceObservedColumns.featureNeurons[databaseNetworkObject.arrayIndexPropertiesTimeIndex] + featureNeuronsActive*sequenceIndex
		if(arrayIndexPropertiesPos):
			sequenceObservedColumns.featureNeurons[databaseNetworkObject.arrayIndexPropertiesPosIndex, :, :, :, :] = featureNeuronsInactive*sequenceObservedColumns.featureNeurons[databaseNetworkObject.arrayIndexPropertiesPosIndex] + featureNeuronsActive*featureNeuronsPos

	if(useSparseSequenceConnections):
		featureConnectionsActive = None
		featureConnectionsSegmentMask = None
		processFeaturesActiveTrainSparseConnections(sequenceObservedColumns, featureNeuronsActive, cs, fs, columnsWordOrder, featureNeuronsWordOrder, featureNeuronsPos)
	else:
		featureConnectionsActive, featureConnectionsSegmentMask = processFeaturesActiveTrainDenseConnections(databaseNetworkObject, sequenceObservedColumns, featureNeuronsActive, cs, fs, columnsWordOrder, featureNeuronsWordOrder, featureNeuronsPos, useSparseSequenceConnections)

	if(trainDecreasePermanenceOfInactiveFeatureNeuronsAndConnections and arrayIndexPropertiesPermanence):
		decreasePermanenceActive(sequenceObservedColumns, featureNeuronsActiveUnion, featureNeuronsInactiveUnion, sequenceConceptIndexMask, featureNeuronsSegmentMask, featureConnectionsSegmentMask)

	if(arrayIndexPropertiesStrength):
		applyTrainConnectionStrengthLimits(sequenceObservedColumns)
	if(debugPrintTrainSectionTimes):
		GIAANNproto_debug.debugTrainSectionTimesAdd(sequenceObservedColumns.databaseNetworkObject, "processFeaturesActiveTrain", time.perf_counter() - processFeaturesActiveTrainStartTime)

	return featureConnectionsActive, featureConnectionsSegmentMask

def processFeaturesActiveTrainSparseNeurons(sequenceObservedColumns, featureNeuronsActive, cs, fs, featureNeuronsPos):
	databaseNetworkObject = sequenceObservedColumns.databaseNetworkObject
	featureNeuronsActiveSparse = featureNeuronsActive.to_sparse().coalesce()
	featureNeuronIndices = featureNeuronsActiveSparse.indices()
	featureNeuronValues = featureNeuronsActiveSparse.values()
	if(arrayIndexPropertiesStrength):
		strengthSparse = buildSequenceFeaturePropertySparse(featureNeuronIndices, featureNeuronValues, cs, fs)
		addSequenceFeatureNeuronsProperty(sequenceObservedColumns, databaseNetworkObject.arrayIndexPropertiesStrengthIndex, strengthSparse)
	if(arrayIndexPropertiesPermanence):
		permanenceValues = pt.full((featureNeuronValues.shape[0],), z1, dtype=arrayType, device=featureNeuronValues.device)
		permanenceSparse = buildSequenceFeaturePropertySparse(featureNeuronIndices, permanenceValues, cs, fs)
		addSequenceFeatureNeuronsProperty(sequenceObservedColumns, databaseNetworkObject.arrayIndexPropertiesPermanenceIndex, permanenceSparse)
	if(arrayIndexPropertiesActivation):
		setSequenceFeatureNeuronsProperty(sequenceObservedColumns, databaseNetworkObject.arrayIndexPropertiesActivationIndex, None)
	if(arrayIndexPropertiesTime):
		setSequenceFeatureNeuronsProperty(sequenceObservedColumns, databaseNetworkObject.arrayIndexPropertiesTimeIndex, None)
	if(arrayIndexPropertiesPos):
		posValues = featureNeuronValues
		if(featureNeuronIndices.numel() > 0):
			posValues = featureNeuronsPos[featureNeuronIndices[2], featureNeuronIndices[3]].to(featureNeuronValues.dtype)
		posSparse = buildSequenceFeaturePropertySparse(featureNeuronIndices, posValues, cs, fs)
		setSequenceFeatureNeuronsProperty(sequenceObservedColumns, databaseNetworkObject.arrayIndexPropertiesPosIndex, posSparse)
	return

def processFeaturesActiveTrainDenseConnections(databaseNetworkObject, sequenceObservedColumns, featureNeuronsActive, cs, fs, columnsWordOrder, featureNeuronsWordOrder, featureNeuronsPos, useSparseSequenceConnections):

	if(useSANI):
		featureConnectionsActive = None
		featureConnectionsSegmentMask = None
		for segmentIndex in range(arrayNumberOfSegments):
			segmentActive = featureNeuronsActive[:, segmentIndex]
			if not pt.any(segmentActive):
				continue
			segmentConnectionsActive, segmentMask = createFeatureConnectionsActiveTrain(segmentActive, cs, fs, columnsWordOrder, featureNeuronsWordOrder)
			if featureConnectionsActive is None:
				featureConnectionsActive = segmentConnectionsActive
				featureConnectionsSegmentMask = segmentMask
			else:
				featureConnectionsActive = pt.maximum(featureConnectionsActive, segmentConnectionsActive)
				featureConnectionsSegmentMask = pt.logical_or(featureConnectionsSegmentMask, segmentMask)
		if featureConnectionsActive is None:
			printe("processFeaturesActiveTrain() error: featureConnectionsActive is None")
			#featureConnectionsActive = pt.zeros((arrayNumberOfSegments, cs, fs, cs, fs), dtype=arrayType)
			#featureConnectionsSegmentMask = pt.zeros_like(featureConnectionsActive, dtype=pt.bool)
	else:
		featureConnectionsActive, featureConnectionsSegmentMask = createFeatureConnectionsActiveTrain(featureNeuronsActive[:, arrayIndexSegmentLast], cs, fs, columnsWordOrder, featureNeuronsWordOrder)

	featureConnectionsPos = None
	if(arrayIndexPropertiesPos or (arrayIndexPropertiesStrength and trainConnectionStrengthPOSdependence)):
		featureConnectionsPos = featureNeuronsPos.view(1, 1, cs, fs, 1, 1).expand(numberOfDendriticBranches, arrayNumberOfSegments, cs, fs, cs, fs)

	featureConnectionsInactive = None
	if((arrayIndexPropertiesTime or arrayIndexPropertiesPos) and not useSparseSequenceConnections):
		featureConnectionsInactive = 1 - featureConnectionsActive

	if(arrayIndexPropertiesStrength):
		if(trainConnectionStrengthNormaliseWrtContextLength):
			featureNeuronsWordOrder1d = featureNeuronsWordOrder.flatten()
			featureConnectionsDistances = pt.abs(featureNeuronsWordOrder1d.unsqueeze(1) - featureNeuronsWordOrder1d).reshape(cs, fs, cs, fs)
			featureConnectionsProximity = 1/(featureConnectionsDistances + 1) * 10
			featureConnectionsProximity.unsqueeze(0)
			featureConnectionsStrengthUpdate = featureConnectionsActive*featureConnectionsProximity
		else:
			featureConnectionsStrengthUpdate = featureConnectionsActive

		csIndices1 = None
		csIndices2 = None
		if(trainConnectionStrengthIncreaseColumnInternal or trainConnectionStrengthPOSdependence):
			csIndices1 = pt.arange(cs).view(1, 1, cs, 1, 1, 1).expand(numberOfDendriticBranches, arrayNumberOfSegments, cs, fs, cs, fs)
			csIndices2 = pt.arange(cs).view(1, 1, 1, 1, cs, 1).expand(numberOfDendriticBranches, arrayNumberOfSegments, cs, fs, cs, fs)

		if(trainConnectionStrengthIncreaseColumnInternal):
			columnInternalConnectionsMask = (csIndices1 == csIndices2)
			columnInternalConnectionsMaskOff = pt.logical_not(columnInternalConnectionsMask)
			featureConnectionsStrengthUpdate = columnInternalConnectionsMask.float()*featureConnectionsStrengthUpdate*trainIncreaseColumnInternalConnectionsStrengthModifier + columnInternalConnectionsMaskOff.float()*featureConnectionsStrengthUpdate

		if(trainConnectionStrengthPOSdependence):
			featureConnectionsStrengthUpdate = applyConnectionStrengthPOSdependenceTrain(sequenceObservedColumns, featureConnectionsStrengthUpdate, featureConnectionsPos, csIndices1, csIndices2)

		addSequenceFeatureConnectionsProperty(sequenceObservedColumns, databaseNetworkObject.arrayIndexPropertiesStrengthIndex, featureConnectionsStrengthUpdate)
	if(arrayIndexPropertiesPermanence):
		addSequenceFeatureConnectionsProperty(sequenceObservedColumns, databaseNetworkObject.arrayIndexPropertiesPermanenceIndex, featureConnectionsActive*z1)
	if(arrayIndexPropertiesActivation):
		setSequenceFeatureConnectionsProperty(sequenceObservedColumns, databaseNetworkObject.arrayIndexPropertiesActivationIndex, None)
	if(arrayIndexPropertiesTime):
		setSequenceFeatureConnectionsProperty(sequenceObservedColumns, databaseNetworkObject.arrayIndexPropertiesTimeIndex, None)
		#OLD inferenceUseNeuronFeaturePropertiesTime=False: sequenceObservedColumns.featureConnections[databaseNetworkObject.arrayIndexPropertiesTimeIndex, :, :, :, :, :, :] = featureConnectionsInactive*sequenceObservedColumns.featureConnections[databaseNetworkObject.arrayIndexPropertiesTimeIndex] + featureConnectionsActive*sequenceIndex
	if(arrayIndexPropertiesPos):
		sequenceObservedColumns.featureConnections[databaseNetworkObject.arrayIndexPropertiesPosIndex, :, :, :, :, :, :] = featureConnectionsInactive*sequenceObservedColumns.featureConnections[databaseNetworkObject.arrayIndexPropertiesPosIndex] + featureConnectionsActive*featureConnectionsPos

	return featureConnectionsActive, featureConnectionsSegmentMask

def processFeaturesActiveTrainSparseConnections(sequenceObservedColumns, featureNeuronsActive, cs, fs, columnsWordOrder, featureNeuronsWordOrder, featureNeuronsPos):
	databaseNetworkObject = sequenceObservedColumns.databaseNetworkObject
	connectionActiveSparse = createFeatureConnectionsActiveTrainSparse(featureNeuronsActive, cs, fs, columnsWordOrder, featureNeuronsWordOrder)
	connectionActiveIndices = connectionActiveSparse.indices()
	connectionActiveValues = connectionActiveSparse.values()
	if(featureNeuronsWordOrder.device != connectionActiveIndices.device):
		featureNeuronsWordOrder = featureNeuronsWordOrder.to(connectionActiveIndices.device)
	if(featureNeuronsPos.device != connectionActiveIndices.device):
		featureNeuronsPos = featureNeuronsPos.to(connectionActiveIndices.device)
	sourceConceptIndices = None
	sourceFeatureIndices = None
	targetConceptIndices = None
	targetFeatureIndices = None
	if(connectionActiveIndices.numel() > 0):
		sourceConceptIndices = connectionActiveIndices[2]
		sourceFeatureIndices = connectionActiveIndices[3]
		targetConceptIndices = connectionActiveIndices[4]
		targetFeatureIndices = connectionActiveIndices[5]
	if(arrayIndexPropertiesStrength):
		strengthValues = connectionActiveValues
		if(connectionActiveIndices.numel() > 0):
			if(trainConnectionStrengthNormaliseWrtContextLength):
				sourceWordOrder = featureNeuronsWordOrder[sourceConceptIndices, sourceFeatureIndices].to(connectionActiveValues.dtype)
				targetWordOrder = featureNeuronsWordOrder[targetConceptIndices, targetFeatureIndices].to(connectionActiveValues.dtype)
				connectionDistances = pt.abs(targetWordOrder - sourceWordOrder)
				connectionProximity = 1/(connectionDistances + 1) * 10
				strengthValues = strengthValues * connectionProximity
			if(trainConnectionStrengthIncreaseColumnInternal):
				internalConnectionMask = sourceConceptIndices == targetConceptIndices
				if(internalConnectionMask.any()):
					strengthValues = strengthValues.clone()
					strengthValues[internalConnectionMask] = strengthValues[internalConnectionMask] * trainIncreaseColumnInternalConnectionsStrengthModifier
			if(trainConnectionStrengthPOSdependence):
				strengthValues = applyConnectionStrengthPOSdependenceTrainSparse(sequenceObservedColumns, strengthValues, featureNeuronsPos, sourceConceptIndices, sourceFeatureIndices, targetConceptIndices)
		strengthSparse = buildSequenceConnectionPropertySparse(connectionActiveIndices, strengthValues, cs, fs)
		addSequenceFeatureConnectionsProperty(sequenceObservedColumns, databaseNetworkObject.arrayIndexPropertiesStrengthIndex, strengthSparse)
	if(arrayIndexPropertiesPermanence):
		permanenceValues = pt.full((connectionActiveValues.shape[0],), z1, dtype=arrayType, device=connectionActiveValues.device)
		permanenceSparse = buildSequenceConnectionPropertySparse(connectionActiveIndices, permanenceValues, cs, fs)
		addSequenceFeatureConnectionsProperty(sequenceObservedColumns, databaseNetworkObject.arrayIndexPropertiesPermanenceIndex, permanenceSparse)
	if(arrayIndexPropertiesActivation):
		setSequenceFeatureConnectionsProperty(sequenceObservedColumns, databaseNetworkObject.arrayIndexPropertiesActivationIndex, None)
	if(arrayIndexPropertiesTime):
		setSequenceFeatureConnectionsProperty(sequenceObservedColumns, databaseNetworkObject.arrayIndexPropertiesTimeIndex, None)
	if(arrayIndexPropertiesPos):
		posValues = connectionActiveValues
		if(connectionActiveIndices.numel() > 0):
			posValues = featureNeuronsPos[sourceConceptIndices, sourceFeatureIndices].to(connectionActiveValues.dtype)
		posSparse = buildSequenceConnectionPropertySparse(connectionActiveIndices, posValues, cs, fs)
		setSequenceFeatureConnectionsProperty(sequenceObservedColumns, databaseNetworkObject.arrayIndexPropertiesPosIndex, posSparse)
	return

def createFeatureConnectionsActiveTrainSparse(featureNeuronsActive, cs, fs, columnsWordOrder, featureNeuronsWordOrder):
	connectionTargetSize = (numberOfDendriticBranches, arrayNumberOfSegments, cs, fs, cs, fs)
	connectionDevice = featureNeuronsActive.device
	combinedIndices = pt.empty((len(connectionTargetSize), 0), dtype=pt.long, device=connectionDevice)
	combinedValues = pt.empty((0,), dtype=arrayType, device=connectionDevice)
	indicesList = []
	if(useSANI):
		for segmentIndex in range(arrayNumberOfSegments):
			segmentActive = featureNeuronsActive[:, segmentIndex]
			if(pt.any(segmentActive)):
				segmentConnectionIndices = createFeatureConnectionsActiveTrainSparseSegment(segmentActive, cs, fs, columnsWordOrder, featureNeuronsWordOrder)
				if(segmentConnectionIndices.numel() > 0):
					indicesList.append(segmentConnectionIndices)
	else:
		segmentConnectionIndices = createFeatureConnectionsActiveTrainSparseSegment(featureNeuronsActive[:, arrayIndexSegmentLast], cs, fs, columnsWordOrder, featureNeuronsWordOrder)
		if(segmentConnectionIndices.numel() > 0):
			indicesList.append(segmentConnectionIndices)
	if(len(indicesList) > 0):
		combinedIndices = pt.cat(indicesList, dim=1)
		combinedValues = pt.ones((combinedIndices.shape[1],), dtype=arrayType, device=connectionDevice)
	result = pt.sparse_coo_tensor(combinedIndices, combinedValues, size=connectionTargetSize, dtype=arrayType, device=connectionDevice).coalesce()
	if(result._nnz() > 0):
		result.values().clamp_(max=1.0)
	return result

def createFeatureConnectionsActiveTrainSparseSegment(segmentActive, cs, fs, columnsWordOrder, featureNeuronsWordOrder):
	connectionDevice = segmentActive.device
	combinedIndices = pt.empty((6, 0), dtype=pt.long, device=connectionDevice)
	indicesList = []
	if(multipleDendriticBranches):
		sourceActive = segmentActive.amax(dim=0)
		sourceIndices = pt.nonzero(sourceActive > 0, as_tuple=False)
		repeatedFeatureMask = (segmentActive > 0).sum(dim=0) > 1
		for branchIndex in range(segmentActive.shape[0]):
			targetIndices = pt.nonzero(segmentActive[branchIndex] > 0, as_tuple=False)
			branchConnectionIndices = createFeatureConnectionsActiveTrainSparseBranch(branchIndex, sourceIndices, targetIndices, repeatedFeatureMask, columnsWordOrder, featureNeuronsWordOrder)
			if(branchConnectionIndices.numel() > 0):
				indicesList.append(branchConnectionIndices)
	else:
		sourceIndices = pt.nonzero(segmentActive[0] > 0, as_tuple=False)
		targetIndices = sourceIndices
		branchConnectionIndices = createFeatureConnectionsActiveTrainSparseBranch(0, sourceIndices, targetIndices, None, columnsWordOrder, featureNeuronsWordOrder)
		if(branchConnectionIndices.numel() > 0):
			indicesList.append(branchConnectionIndices)
	if(len(indicesList) > 0):
		combinedIndices = pt.cat(indicesList, dim=1)
	return combinedIndices

def createFeatureConnectionsActiveTrainSparseBranch(branchIndex, sourceIndices, targetIndices, repeatedFeatureMask, columnsWordOrder, featureNeuronsWordOrder):
	connectionDevice = sourceIndices.device
	result = pt.empty((6, 0), dtype=pt.long, device=connectionDevice)
	if(sourceIndices.shape[0] > 0 and targetIndices.shape[0] > 0):
		sourceCount = sourceIndices.shape[0]
		targetCount = targetIndices.shape[0]
		sourceConceptIndices = sourceIndices[:, 0].repeat_interleave(targetCount)
		sourceFeatureIndices = sourceIndices[:, 1].repeat_interleave(targetCount)
		targetConceptIndices = targetIndices[:, 0].repeat(sourceCount)
		targetFeatureIndices = targetIndices[:, 1].repeat(sourceCount)
		sourceWordOrder = featureNeuronsWordOrder[sourceConceptIndices, sourceFeatureIndices]
		targetWordOrder = featureNeuronsWordOrder[targetConceptIndices, targetFeatureIndices]
		connectionMask = pt.ones((sourceConceptIndices.shape[0],), dtype=pt.bool, device=connectionDevice)
		if(featureNeuronsWordOrder is not None):
			if(debugConnectNodesToNextNodesInSequenceOnly):
				wordOrderUpperBound = sourceWordOrder + 1
				connectionMask = connectionMask & pt.logical_and(targetWordOrder > sourceWordOrder, targetWordOrder <= wordOrderUpperBound)
			else:
				connectionMask = connectionMask & (targetWordOrder > sourceWordOrder)
		if(columnsWordOrder is not None):
			sourceColumnWordOrder = columnsWordOrder[sourceConceptIndices]
			targetColumnWordOrder = columnsWordOrder[targetConceptIndices]
			if(debugConnectColumnsToNextColumnsInSequenceOnly):
				connectionMask = connectionMask & pt.logical_and(targetColumnWordOrder >= sourceColumnWordOrder, targetColumnWordOrder <= sourceColumnWordOrder+1)
			else:
				connectionMask = connectionMask & (targetColumnWordOrder >= sourceColumnWordOrder)
		selfMask = (sourceConceptIndices == targetConceptIndices) & (sourceFeatureIndices == targetFeatureIndices)
		connectionMask = connectionMask & pt.logical_not(selfMask)
		if(repeatedFeatureMask is not None):
			repeatedSourceMask = repeatedFeatureMask[sourceConceptIndices, sourceFeatureIndices]
			connectionMask = connectionMask | (selfMask & repeatedSourceMask)
		if(connectionMask.any()):
			sourceConceptIndices = sourceConceptIndices[connectionMask]
			sourceFeatureIndices = sourceFeatureIndices[connectionMask]
			targetConceptIndices = targetConceptIndices[connectionMask]
			targetFeatureIndices = targetFeatureIndices[connectionMask]
			sourceWordOrder = sourceWordOrder[connectionMask]
			targetWordOrder = targetWordOrder[connectionMask]
			branchIndices = pt.full((sourceConceptIndices.shape[0],), branchIndex, dtype=pt.long, device=connectionDevice)
			result = assignFeatureConnectionsToTargetSegmentsSparse(branchIndices, sourceConceptIndices, sourceFeatureIndices, targetConceptIndices, targetFeatureIndices, sourceWordOrder, targetWordOrder)
	return result

def assignFeatureConnectionsToTargetSegmentsSparse(branchIndices, sourceConceptIndices, sourceFeatureIndices, targetConceptIndices, targetFeatureIndices, sourceWordOrder, targetWordOrder):
	connectionDevice = branchIndices.device
	indicesList = []
	if(useSANIcolumns):
		conceptDistances = pt.abs(targetConceptIndices - sourceConceptIndices)
		connectionsSegmentIndex = arrayNumberOfSegments - conceptDistances - 1
		connectionsSegmentIndex = pt.clamp(connectionsSegmentIndex, min=0)
		indicesList.append(pt.stack((branchIndices, connectionsSegmentIndex.long(), sourceConceptIndices, sourceFeatureIndices, targetConceptIndices, targetFeatureIndices), dim=0))
	elif(useSANIfeatures):
		relativeDistance = targetWordOrder - sourceWordOrder
		if(SANIfeaturesLinkFirstSegmentToAllPriorTrainSeqTokens):
			relativeDistance = pt.clamp(relativeDistance, min=1)
			connectionsSegmentIndex = arrayNumberOfSegments - relativeDistance
			connectionsSegmentIndex = connectionsSegmentIndex.clamp(min=0, max=arrayNumberOfSegments-1).long()
			indicesList.append(pt.stack((branchIndices, connectionsSegmentIndex, sourceConceptIndices, sourceFeatureIndices, targetConceptIndices, targetFeatureIndices), dim=0))
		else:
			relativeDistance = pt.clamp(relativeDistance, min=1)
			validDistanceMask = relativeDistance <= arrayNumberOfSegments
			if(validDistanceMask.any()):
				connectionsSegmentIndex = arrayNumberOfSegments - relativeDistance
				connectionsSegmentIndex = connectionsSegmentIndex.clamp(min=0, max=arrayNumberOfSegments-1).long()
				indicesList.append(pt.stack((branchIndices[validDistanceMask], connectionsSegmentIndex[validDistanceMask], sourceConceptIndices[validDistanceMask], sourceFeatureIndices[validDistanceMask], targetConceptIndices[validDistanceMask], targetFeatureIndices[validDistanceMask]), dim=0))
	elif(useSANIfeaturesAndColumns):
		relativeDistance = targetWordOrder - sourceWordOrder
		featureSegmentsOffset = arrayNumberOfSegmentsColumnDistance
		if(SANIfeaturesLinkFirstSegmentToAllPriorTrainSeqTokens):
			relativeDistanceFeature = pt.clamp(relativeDistance, min=1, max=arrayNumberOfSegmentsFeatureDistance)
			featureSegmentIndex = featureSegmentsOffset + arrayNumberOfSegmentsFeatureDistance - relativeDistanceFeature
			featureSegmentIndex = featureSegmentIndex.clamp(min=featureSegmentsOffset, max=arrayNumberOfSegments-1).long()
			indicesList.append(pt.stack((branchIndices, featureSegmentIndex, sourceConceptIndices, sourceFeatureIndices, targetConceptIndices, targetFeatureIndices), dim=0))
		else:
			relativeDistanceFeature = pt.clamp(relativeDistance, min=1)
			validFeatureDistanceMask = relativeDistanceFeature <= arrayNumberOfSegmentsFeatureDistance
			if(validFeatureDistanceMask.any()):
				featureSegmentIndex = featureSegmentsOffset + arrayNumberOfSegmentsFeatureDistance - relativeDistanceFeature
				featureSegmentIndex = featureSegmentIndex.clamp(min=featureSegmentsOffset, max=arrayNumberOfSegments-1).long()
				indicesList.append(pt.stack((branchIndices[validFeatureDistanceMask], featureSegmentIndex[validFeatureDistanceMask], sourceConceptIndices[validFeatureDistanceMask], sourceFeatureIndices[validFeatureDistanceMask], targetConceptIndices[validFeatureDistanceMask], targetFeatureIndices[validFeatureDistanceMask]), dim=0))
		if(arrayNumberOfSegmentsColumnDistance > 0):
			conceptDistances = pt.abs(targetConceptIndices - sourceConceptIndices)
			if(useSANIfeaturesAndColumnsInternal):
				columnSegmentIndex = arrayNumberOfSegmentsColumnDistance - conceptDistances - 1
			else:
				columnSegmentIndex = arrayNumberOfSegmentsColumnDistance - conceptDistances
			columnSegmentIndex = columnSegmentIndex.clamp(min=0, max=arrayNumberOfSegmentsColumnDistance-1).long()
			validColumnMask = pt.ones((branchIndices.shape[0],), dtype=pt.bool, device=connectionDevice)
			if(not useSANIfeaturesAndColumnsInternal):
				validColumnMask = conceptDistances > 0
			if(validColumnMask.any()):
				indicesList.append(pt.stack((branchIndices[validColumnMask], columnSegmentIndex[validColumnMask], sourceConceptIndices[validColumnMask], sourceFeatureIndices[validColumnMask], targetConceptIndices[validColumnMask], targetFeatureIndices[validColumnMask]), dim=0))
	else:
		segmentIndices = pt.zeros((branchIndices.shape[0],), dtype=pt.long, device=connectionDevice)
		indicesList.append(pt.stack((branchIndices, segmentIndices, sourceConceptIndices, sourceFeatureIndices, targetConceptIndices, targetFeatureIndices), dim=0))
	result = pt.empty((6, 0), dtype=pt.long, device=connectionDevice)
	if(len(indicesList) > 0):
		result = pt.cat(indicesList, dim=1)
	return result

def buildSequenceFeaturePropertySparse(featureIndices, featureValues, cs, fs):
	targetSize = (numberOfDendriticBranches, arrayNumberOfSegments, cs, fs)
	targetDevice = deviceSparse
	sparseIndices = featureIndices
	sparseValues = featureValues
	if(sparseIndices.numel() == 0):
		sparseIndices = pt.empty((len(targetSize), 0), dtype=pt.long, device=targetDevice)
		sparseValues = pt.empty((0,), dtype=arrayType, device=targetDevice)
	else:
		if(sparseIndices.device != targetDevice):
			sparseIndices = sparseIndices.to(targetDevice)
		if(sparseValues.device != targetDevice):
			sparseValues = sparseValues.to(targetDevice)
	result = pt.sparse_coo_tensor(sparseIndices, sparseValues, size=targetSize, dtype=arrayType, device=targetDevice).coalesce()
	return result

def buildSequenceConnectionPropertySparse(connectionIndices, connectionValues, cs, fs):
	targetSize = (numberOfDendriticBranches, arrayNumberOfSegments, cs, fs, cs, fs)
	targetDevice = deviceSparse
	sparseIndices = connectionIndices
	sparseValues = connectionValues
	if(sparseIndices.numel() == 0):
		sparseIndices = pt.empty((len(targetSize), 0), dtype=pt.long, device=targetDevice)
		sparseValues = pt.empty((0,), dtype=arrayType, device=targetDevice)
	else:
		if(sparseIndices.device != targetDevice):
			sparseIndices = sparseIndices.to(targetDevice)
		if(sparseValues.device != targetDevice):
			sparseValues = sparseValues.to(targetDevice)
	result = pt.sparse_coo_tensor(sparseIndices, sparseValues, size=targetSize, dtype=arrayType, device=targetDevice).coalesce()
	return result

def applyTrainConnectionStrengthLimits(sequenceObservedColumns):
	databaseNetworkObject = sequenceObservedColumns.databaseNetworkObject
	if(trainSparseConnectionsTensor):
		if(trainConnectionStrengthLimitMax):
			sequenceObservedColumns.transformSequenceConnectionPropertyValues(databaseNetworkObject.arrayIndexPropertiesStrengthIndex, "clampMax", 1.0)
		if(trainConnectionStrengthLimitTanh):
			sequenceObservedColumns.transformSequenceConnectionPropertyValues(databaseNetworkObject.arrayIndexPropertiesStrengthIndex, "tanh")
	else:
		if(trainConnectionStrengthLimitMax):
			sequenceObservedColumns.featureConnections[databaseNetworkObject.arrayIndexPropertiesStrengthIndex] = sequenceObservedColumns.featureConnections[databaseNetworkObject.arrayIndexPropertiesStrengthIndex].clamp(max=1.0)
		if(trainConnectionStrengthLimitTanh):
			sequenceObservedColumns.featureConnections[databaseNetworkObject.arrayIndexPropertiesStrengthIndex] = pt.tanh(sequenceObservedColumns.featureConnections[databaseNetworkObject.arrayIndexPropertiesStrengthIndex])
	return

def addSequenceFeatureNeuronsProperty(sequenceObservedColumns, propertyIndex, propertyTensor):
	if(trainSparseNeuronsTensor):
		sequenceObservedColumns.addSequenceFeaturePropertyUpdate(propertyIndex, propertyTensor)
	else:
		sequenceObservedColumns.featureNeurons[propertyIndex, :, :, :, :] += propertyTensor
	return

def setSequenceFeatureNeuronsProperty(sequenceObservedColumns, propertyIndex, propertyTensor):
	if(trainSparseNeuronsTensor):
		sequenceObservedColumns.setSequenceFeaturePropertyUpdate(propertyIndex, propertyTensor)
	else:
		if(propertyTensor is None):
			sequenceObservedColumns.featureNeurons[propertyIndex, :, :, :, :] = 0
		else:
			sequenceObservedColumns.featureNeurons[propertyIndex, :, :, :, :] = propertyTensor
	return

def addSequenceFeatureConnectionsProperty(sequenceObservedColumns, propertyIndex, propertyTensor):
	if(trainSparseConnectionsTensor):
		sequenceObservedColumns.addSequenceConnectionPropertyUpdate(propertyIndex, propertyTensor)
	else:
		sequenceObservedColumns.featureConnections[propertyIndex, :, :, :, :, :, :] += propertyTensor
	return

def setSequenceFeatureConnectionsProperty(sequenceObservedColumns, propertyIndex, propertyTensor):
	if(trainSparseConnectionsTensor):
		sequenceObservedColumns.setSequenceConnectionPropertyUpdate(propertyIndex, propertyTensor)
	else:
		if(propertyTensor is None):
			sequenceObservedColumns.featureConnections[propertyIndex, :, :, :, :, :, :] = 0
		else:
			sequenceObservedColumns.featureConnections[propertyIndex, :, :, :, :, :, :] = propertyTensor
	return

def createFeatureConnectionsActiveTrain(featureNeuronsActive, cs, fs, columnsWordOrder, featureNeuronsWordOrder):

	if featureNeuronsActive.dim() == 3:
		branchCount = featureNeuronsActive.shape[0]
		if(multipleDendriticBranches):
			sourceActive = featureNeuronsActive.amax(dim=0)
			sourceActive1d = sourceActive.view(1, cs*fs, 1)
			targetActive1d = featureNeuronsActive.view(branchCount, cs*fs).unsqueeze(1)
			featureConnectionsActive = (sourceActive1d * targetActive1d).view(branchCount, cs, fs, cs, fs)
		else:
			featureNeuronsActive1d = featureNeuronsActive.view(branchCount, cs*fs)
			featureConnectionsActive = pt.matmul(featureNeuronsActive1d.unsqueeze(2), featureNeuronsActive1d.unsqueeze(1)).view(branchCount, cs, fs, cs, fs)
	else:
		featureNeuronsActive1d = featureNeuronsActive.view(cs*fs)
		featureConnectionsActive = pt.matmul(featureNeuronsActive1d.unsqueeze(1), featureNeuronsActive1d.unsqueeze(0)).view(cs, fs, cs, fs)

	if(featureNeuronsWordOrder is not None):
		featureNeuronsWordOrderExpanded1 = featureNeuronsWordOrder.view(cs, fs, 1, 1).expand(cs, fs, cs, fs)
		featureNeuronsWordOrderExpanded2 = featureNeuronsWordOrder.view(1, 1, cs, fs).expand(cs, fs, cs, fs)
		if(debugConnectNodesToNextNodesInSequenceOnly):
			wordOrderUpperBound = featureNeuronsWordOrderExpanded1 + 1
			wordOrderMask = pt.logical_and(featureNeuronsWordOrderExpanded2 > featureNeuronsWordOrderExpanded1, featureNeuronsWordOrderExpanded2 <= wordOrderUpperBound)
		else:
			wordOrderMask = featureNeuronsWordOrderExpanded2 > featureNeuronsWordOrderExpanded1
		featureConnectionsActive = featureConnectionsActive * wordOrderMask
	if(columnsWordOrder is not None):
		columnsWordOrderExpanded1 = columnsWordOrder.view(cs, 1, 1, 1).expand(cs, fs, cs, fs)
		columnsWordOrderExpanded2 = columnsWordOrder.view(1, 1, cs, 1).expand(cs, fs, cs, fs)
		if(debugConnectColumnsToNextColumnsInSequenceOnly):
			columnsWordOrderMask = pt.logical_and(columnsWordOrderExpanded2 >= columnsWordOrderExpanded1, columnsWordOrderExpanded2 <= columnsWordOrderExpanded1+1)
		else:
			columnsWordOrderMask = columnsWordOrderExpanded2 >= columnsWordOrderExpanded1
		featureConnectionsActive = featureConnectionsActive * columnsWordOrderMask
	
	csIndices1 = pt.arange(cs).view(cs, 1, 1, 1).expand(cs, fs, cs, fs)
	csIndices2 = pt.arange(cs).view(1, 1, cs, 1).expand(cs, fs, cs, fs)
	fsIndices1 = pt.arange(fs).view(1, fs, 1, 1).expand(cs, fs, cs, fs)
	fsIndices2 = pt.arange(fs).view(1, 1, 1, fs).expand(cs, fs, cs, fs)
	identityMask = (csIndices1 != csIndices2) | (fsIndices1 != fsIndices2)
	if(multipleDendriticBranches and featureNeuronsActive.dim() == 3):
		featureBranchCounts = (featureNeuronsActive > 0).sum(dim=0)
		repeatedFeatureMask = featureBranchCounts > 1
		repeatedFeatureMaskExpanded = repeatedFeatureMask.view(cs, fs, 1, 1).expand(cs, fs, cs, fs)
		selfMask = (csIndices1 == csIndices2) & (fsIndices1 == fsIndices2)
		identityMask = identityMask | (selfMask & repeatedFeatureMaskExpanded)
	featureConnectionsActive = featureConnectionsActive * identityMask

	if(useSANI):
		featureConnectionsActive, featureConnectionsSegmentMask = assignFeatureConnectionsToTargetSegments(featureConnectionsActive, cs, fs, featureNeuronsWordOrder)
	else:
		if featureConnectionsActive.dim() == 5:
			featureConnectionsActive = featureConnectionsActive.unsqueeze(1)
		else:
			featureConnectionsActive = featureConnectionsActive.unsqueeze(0)
		featureConnectionsSegmentMask = pt.ones_like(featureConnectionsActive, dtype=pt.bool)
	
	return featureConnectionsActive, featureConnectionsSegmentMask

def assignFeatureConnectionsToTargetSegments(featureConnectionsActive, cs, fs, featureNeuronsWordOrder):
	hasBranchDim = featureConnectionsActive.dim() == 5
	branchCount = featureConnectionsActive.shape[0] if hasBranchDim else 1
	if(useSANIcolumns):
		conceptNeuronsConceptOrder1d = pt.arange(cs)
		conceptNeuronsDistances = pt.abs(conceptNeuronsConceptOrder1d.unsqueeze(1) - conceptNeuronsConceptOrder1d).reshape(cs, cs)
		connectionsSegmentIndex = arrayNumberOfSegments-conceptNeuronsDistances-1
		connectionsSegmentIndex = pt.clamp(connectionsSegmentIndex, min=0)
		featureConnectionsSegmentMask = pt.zeros((arrayNumberOfSegments, cs, cs), dtype=pt.bool)
		featureConnectionsSegmentMask = featureConnectionsSegmentMask.scatter_(0, connectionsSegmentIndex.unsqueeze(0), True)
		featureConnectionsSegmentMask = featureConnectionsSegmentMask.view(arrayNumberOfSegments, cs, 1, cs, 1).expand(arrayNumberOfSegments, cs, fs, cs, fs)
	elif(useSANIfeatures):
		device = featureConnectionsActive.device
		wordOrderTensor = featureNeuronsWordOrder.to(device)
		wordOrderSource = wordOrderTensor.view(cs, fs, 1, 1).expand(cs, fs, cs, fs)
		wordOrderTarget = wordOrderTensor.view(1, 1, cs, fs).expand(cs, fs, cs, fs)
		relativeDistance = (wordOrderTarget - wordOrderSource)
		relativeDistance = pt.clamp(relativeDistance, min=1)
		if(SANIfeaturesLinkFirstSegmentToAllPriorTrainSeqTokens):
			connectionsSegmentIndex = arrayNumberOfSegments - relativeDistance
			connectionsSegmentIndex = connectionsSegmentIndex.clamp(min=0, max=arrayNumberOfSegments-1).long()
			featureConnectionsSegmentMask = pt.zeros((arrayNumberOfSegments, cs, fs, cs, fs), dtype=pt.bool, device=device)
			featureConnectionsSegmentMask.scatter_(0, connectionsSegmentIndex.unsqueeze(0), True)
		else:
			validDistanceMask = (relativeDistance <= arrayNumberOfSegments)
			connectionsSegmentIndex = arrayNumberOfSegments - relativeDistance
			connectionsSegmentIndex = connectionsSegmentIndex.clamp(min=0, max=arrayNumberOfSegments-1).long()
			featureConnectionsSegmentMask = pt.zeros((arrayNumberOfSegments, cs, fs, cs, fs), dtype=pt.bool, device=device)
			featureConnectionsSegmentMask.scatter_(0, connectionsSegmentIndex.unsqueeze(0), True)
			featureConnectionsSegmentMask = featureConnectionsSegmentMask & validDistanceMask.unsqueeze(0)
	elif(useSANIfeaturesAndColumns):
		device = featureConnectionsActive.device
		wordOrderTensor = featureNeuronsWordOrder.to(device)
		wordOrderSource = wordOrderTensor.view(cs, fs, 1, 1).expand(cs, fs, cs, fs)
		wordOrderTarget = wordOrderTensor.view(1, 1, cs, fs).expand(cs, fs, cs, fs)
		relativeDistance = (wordOrderTarget - wordOrderSource)
		if(SANIfeaturesLinkFirstSegmentToAllPriorTrainSeqTokens):
			relativeDistance = pt.clamp(relativeDistance, min=1, max=arrayNumberOfSegmentsFeatureDistance)
			featureSegmentsOffset = arrayNumberOfSegmentsColumnDistance
			featureSegmentIndex = featureSegmentsOffset + arrayNumberOfSegmentsFeatureDistance - relativeDistance
			featureSegmentIndex = featureSegmentIndex.clamp(min=featureSegmentsOffset, max=arrayNumberOfSegments-1).long()
			featureConnectionsSegmentMask = pt.zeros((arrayNumberOfSegments, cs, fs, cs, fs), dtype=pt.bool, device=device)
			featureConnectionsSegmentMask.scatter_(0, featureSegmentIndex.unsqueeze(0), True)
		else:
			relativeDistance = pt.clamp(relativeDistance, min=1)
			validDistanceMask = (relativeDistance <= arrayNumberOfSegmentsFeatureDistance)
			featureSegmentsOffset = arrayNumberOfSegmentsColumnDistance
			featureSegmentIndex = featureSegmentsOffset + arrayNumberOfSegmentsFeatureDistance - relativeDistance
			featureSegmentIndex = featureSegmentIndex.clamp(min=featureSegmentsOffset, max=arrayNumberOfSegments-1).long()
			featureConnectionsSegmentMask = pt.zeros((arrayNumberOfSegments, cs, fs, cs, fs), dtype=pt.bool, device=device)
			featureConnectionsSegmentMask.scatter_(0, featureSegmentIndex.unsqueeze(0), True)
			featureConnectionsSegmentMask = featureConnectionsSegmentMask & validDistanceMask.unsqueeze(0)

		conceptNeuronsConceptOrder1d = pt.arange(cs, device=device)
		conceptNeuronsDistances = pt.abs(conceptNeuronsConceptOrder1d.unsqueeze(1) - conceptNeuronsConceptOrder1d).reshape(cs, cs)
		conceptNeuronsDistances = conceptNeuronsDistances.view(cs, 1, cs, 1).expand(cs, fs, cs, fs)
		if(arrayNumberOfSegmentsColumnDistance > 0):
			if(useSANIfeaturesAndColumnsInternal):
				# Include internal column as the most proximal concept segment.
				columnSegmentIndex = arrayNumberOfSegmentsColumnDistance - conceptNeuronsDistances - 1
			else:
				# External columns only: exclude the internal column from concept segments.
				columnSegmentIndex = arrayNumberOfSegmentsColumnDistance - conceptNeuronsDistances
			columnSegmentIndex = columnSegmentIndex.clamp(min=0, max=arrayNumberOfSegmentsColumnDistance-1).long()
			columnSegmentMask = pt.zeros((arrayNumberOfSegments, cs, fs, cs, fs), dtype=pt.bool, device=device)
			columnSegmentMask.scatter_(0, columnSegmentIndex.unsqueeze(0), True)
			if(not useSANIfeaturesAndColumnsInternal):
				externalColumnMask = (conceptNeuronsDistances > 0)
				columnSegmentMask = columnSegmentMask & externalColumnMask.unsqueeze(0)	#exclude internal column
			featureConnectionsSegmentMask = featureConnectionsSegmentMask | columnSegmentMask

	if(hasBranchDim):
		featureConnectionsSegmentMask = featureConnectionsSegmentMask.unsqueeze(0).expand(branchCount, -1, -1, -1, -1, -1)
		featureConnectionsActive = featureConnectionsSegmentMask * featureConnectionsActive.unsqueeze(1)
	else:
		featureConnectionsActive = featureConnectionsSegmentMask * featureConnectionsActive.unsqueeze(0)

	return featureConnectionsActive, featureConnectionsSegmentMask


def decreasePermanenceActive(sequenceObservedColumns, featureNeuronsActive, featureNeuronsInactive, sequenceConceptIndexMask, featureNeuronsSegmentMask, featureConnectionsSegmentMask):
	databaseNetworkObject = sequenceObservedColumns.databaseNetworkObject

	if(trainSequenceObservedColumnsMatchSequenceWords):
		featureNeuronsInactive = featureNeuronsInactive*sequenceConceptIndexMask
	
	cs = sequenceObservedColumns.cs
	fs = sequenceObservedColumns.fs 
	
	featureNeuronsDecrease = featureNeuronsInactive * z2
	featureNeuronsDecrease = featureNeuronsDecrease * featureNeuronsSegmentMask.unsqueeze(0).unsqueeze(3)
	sequenceObservedColumns.featureNeurons[databaseNetworkObject.arrayIndexPropertiesPermanenceIndex] -= featureNeuronsDecrease
	sequenceObservedColumns.featureNeurons[databaseNetworkObject.arrayIndexPropertiesPermanenceIndex] = pt.clamp(sequenceObservedColumns.featureNeurons[databaseNetworkObject.arrayIndexPropertiesPermanenceIndex], min=0)

	featureNeuronsAll = pt.ones((cs, fs), dtype=arrayType)
	branchCount = featureNeuronsActive.shape[0]
	featureNeuronsAll1d = featureNeuronsAll.view(1, cs*fs).expand(branchCount, -1)
	featureNeuronsActive1d = featureNeuronsActive.view(branchCount, cs*fs)
	featureNeuronsInactive1d = featureNeuronsInactive.view(branchCount, cs*fs)
	 
	featureConnectionsDecrease1 = pt.matmul(featureNeuronsInactive1d.unsqueeze(2), featureNeuronsAll1d.unsqueeze(1)).view(branchCount, cs, fs, cs, fs)
	featureConnectionsDecrease1 = featureConnectionsDecrease1.unsqueeze(1) * featureConnectionsSegmentMask
	sequenceObservedColumns.featureConnections[databaseNetworkObject.arrayIndexPropertiesPermanenceIndex] -= featureConnectionsDecrease1
	sequenceObservedColumns.featureConnections[databaseNetworkObject.arrayIndexPropertiesPermanenceIndex] = pt.clamp(sequenceObservedColumns.featureConnections[databaseNetworkObject.arrayIndexPropertiesPermanenceIndex], min=0)
	
	featureConnectionsDecrease2 = pt.matmul(featureNeuronsActive1d.unsqueeze(2), featureNeuronsInactive1d.unsqueeze(1)).view(branchCount, cs, fs, cs, fs)
	featureConnectionsDecrease2 = featureConnectionsDecrease2.unsqueeze(1) * featureConnectionsSegmentMask
	sequenceObservedColumns.featureConnections[databaseNetworkObject.arrayIndexPropertiesPermanenceIndex] -= featureConnectionsDecrease2
	sequenceObservedColumns.featureConnections[databaseNetworkObject.arrayIndexPropertiesPermanenceIndex] = pt.clamp(sequenceObservedColumns.featureConnections[databaseNetworkObject.arrayIndexPropertiesPermanenceIndex], min=0)
 
def applyConnectionStrengthPOSdependenceTrain(sequenceObservedColumns, featureConnectionsStrengthUpdate, featureConnectionsPos, csIndicesSource, csIndicesTarget):
	posLookup = getConnectionStrengthPOSdependenceLookup(sequenceObservedColumns.databaseNetworkObject)
	if not posLookup:
		return featureConnectionsStrengthUpdate
	if(connectionStrengthPOSdependenceExternal):
		scopeMask = (csIndicesSource != csIndicesTarget)
	else:
		scopeMask = pt.ones_like(csIndicesSource, dtype=pt.bool)
	featureConnectionsPosLong = featureConnectionsPos.long()
	for posIndex, scaleValue in posLookup:
		if scaleValue == 1:
			continue
		posMask = (featureConnectionsPosLong == posIndex) & scopeMask
		if pt.any(posMask):
			posMaskFloat = posMask.to(featureConnectionsStrengthUpdate.dtype)
			featureConnectionsStrengthUpdate = featureConnectionsStrengthUpdate + (scaleValue - 1.0) * featureConnectionsStrengthUpdate * posMaskFloat
	return featureConnectionsStrengthUpdate

def applyConnectionStrengthPOSdependenceTrainSparse(sequenceObservedColumns, featureConnectionsStrengthValues, featureNeuronsPos, sourceConceptIndices, sourceFeatureIndices, targetConceptIndices):
	posLookup = getConnectionStrengthPOSdependenceLookup(sequenceObservedColumns.databaseNetworkObject)
	result = featureConnectionsStrengthValues
	if posLookup:
		if(connectionStrengthPOSdependenceExternal):
			scopeMask = sourceConceptIndices != targetConceptIndices
		else:
			scopeMask = pt.ones_like(sourceConceptIndices, dtype=pt.bool)
		sourcePosLong = featureNeuronsPos[sourceConceptIndices, sourceFeatureIndices].long()
		for posIndex, scaleValue in posLookup:
			if scaleValue == 1:
				continue
			posMask = (sourcePosLong == posIndex) & scopeMask
			if pt.any(posMask):
				if(result is featureConnectionsStrengthValues):
					result = featureConnectionsStrengthValues.clone()
				result[posMask] = result[posMask] * scaleValue
	return result

def getConnectionStrengthPOSdependenceLookup(databaseNetworkObject):
	if not hasattr(databaseNetworkObject, "connectionStrengthPOSdependenceLookup"):
		posLookup = []
		for posType, value in zip(connectionStrengthPOSdependenceTypes, connectionStrengthPOSdependenceValues):
			posIndex = posStringToPosInt(databaseNetworkObject.nlp, posType)
			posLookup.append((posIndex, float(value)))
		databaseNetworkObject.connectionStrengthPOSdependenceLookup = posLookup
	return databaseNetworkObject.connectionStrengthPOSdependenceLookup
