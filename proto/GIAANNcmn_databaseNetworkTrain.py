"""GIAANNcmn_databaseNetworkTrain.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 BAI Research Pty Ltd (bairesearch.com.au)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN common database Network Train

"""

import torch as pt
import time

from GIAANNcmn_globalDefs import *
import GIAANNcmn_debug
import GIAANNcmn_sparseTensors
import GIAANNnlp_sequenceConcepts
if(auxiliaryNeurons and auxiliaryNeuronsSimilar):
	import GIAANNnlp_auxiliaryNeuronsSimilarWords

	
def trainConceptWords(sequenceObservedColumns, sequenceIndex, sequence, tokens, inferenceSuccessfulPredictionMask=None):
	trainConceptWordsStartTime = None
	if(debugPrintTrainSectionTimes):
		trainConceptWordsStartTime = time.perf_counter()
	result = GIAANNnlp_sequenceConcepts.processConceptWords(sequenceObservedColumns, sequenceIndex, sequence, tokens)
	if(printSequenceConceptAssignment):
		print(f"Processing sequenceCount: {sequenceIndex}, {sequenceObservedColumns.sentenceWithConceptAssignment}")	
		if(printSequenceConceptAssignmentByLine):
			print("")	
	trainActive = False
	if(result is not None):
		conceptIndices, startIndices, endIndices = result
		featureNeuronsActive, cs, fs, sequenceConceptIndexMask, columnsWordOrder, featureNeuronsWordOrder, featureNeuronsPos, featureNeuronsSegmentMask = GIAANNnlp_sequenceConcepts.processFeatures(sequenceObservedColumns, sequenceIndex, sequence, tokens, conceptIndices, startIndices, endIndices)
		trainActive = True
		featureNeuronsTargetMask = None
		if(useTrainDuringInference):
			featureNeuronsTargetMask = createTrainDuringInferenceFeatureNeuronsTargetMask(tokens, inferenceSuccessfulPredictionMask, featureNeuronsWordOrder, featureNeuronsActive)
			trainActive = bool(pt.any(featureNeuronsActive*featureNeuronsTargetMask).item())
		if(trainActive):
			featureConnectionsActive, featureConnectionsSegmentMask = processFeaturesActiveTrain(sequenceObservedColumns, featureNeuronsActive, cs, fs, sequenceConceptIndexMask, columnsWordOrder, featureNeuronsWordOrder, featureNeuronsPos, featureNeuronsSegmentMask, sequenceIndex, featureNeuronsTargetMask)
			if(auxiliaryNeurons and auxiliaryNeuronsSimilar):
				if(useTrainDuringInference):
					featureNeuronsActive = featureNeuronsActive*featureNeuronsTargetMask
				GIAANNnlp_auxiliaryNeuronsSimilarWords.trainAuxiliaryFeatureConnections(sequenceObservedColumns, featureNeuronsActive, columnsWordOrder, featureNeuronsWordOrder, conceptIndices, startIndices, endIndices)
	if(debugPrintTrainSectionTimes):
		GIAANNcmn_debug.debugTrainSectionTimesAdd(sequenceObservedColumns.databaseNetworkObject, "trainConceptWords", time.perf_counter() - trainConceptWordsStartTime)

	return trainActive

def createTrainDuringInferenceFeatureNeuronsTargetMask(tokens, inferenceSuccessfulPredictionMask, featureNeuronsWordOrder, featureNeuronsActive):
	result = None
	trainTargetFeaturesMask = None
	if(inferenceSuccessfulPredictionMask is None):
		raise RuntimeError("createTrainDuringInferenceFeatureNeuronsTargetMask error: inferenceSuccessfulPredictionMask is None")
	if(not pt.is_tensor(inferenceSuccessfulPredictionMask)):
		raise RuntimeError("createTrainDuringInferenceFeatureNeuronsTargetMask error: inferenceSuccessfulPredictionMask must be a tensor")
	if(inferenceSuccessfulPredictionMask.dtype != pt.bool):
		raise RuntimeError("createTrainDuringInferenceFeatureNeuronsTargetMask error: inferenceSuccessfulPredictionMask must be bool")
	if(inferenceSuccessfulPredictionMask.dim() != 1):
		raise RuntimeError("createTrainDuringInferenceFeatureNeuronsTargetMask error: inferenceSuccessfulPredictionMask rank must be 1")
	if(int(inferenceSuccessfulPredictionMask.shape[0]) != len(tokens)):
		raise RuntimeError("createTrainDuringInferenceFeatureNeuronsTargetMask error: inferenceSuccessfulPredictionMask length must match tokens length")
	if(numSeedTokens > 0 and bool(pt.any(inferenceSuccessfulPredictionMask[:numSeedTokens]).item())):
		raise RuntimeError("createTrainDuringInferenceFeatureNeuronsTargetMask error: prompt tokens must be False")
	if(featureNeuronsWordOrder is None):
		raise RuntimeError("createTrainDuringInferenceFeatureNeuronsTargetMask error: featureNeuronsWordOrder is None")
	if(not pt.is_tensor(featureNeuronsWordOrder)):
		raise RuntimeError("createTrainDuringInferenceFeatureNeuronsTargetMask error: featureNeuronsWordOrder must be a tensor")
	if(not pt.is_tensor(featureNeuronsActive)):
		raise RuntimeError("createTrainDuringInferenceFeatureNeuronsTargetMask error: featureNeuronsActive must be a tensor")
	if(featureNeuronsWordOrder.dim() != 2):
		raise RuntimeError("createTrainDuringInferenceFeatureNeuronsTargetMask error: featureNeuronsWordOrder rank must be 2")
	if(featureNeuronsActive.dim() != 4):
		raise RuntimeError("createTrainDuringInferenceFeatureNeuronsTargetMask error: featureNeuronsActive rank must be 4")
	if(int(featureNeuronsWordOrder.shape[0]) != int(featureNeuronsActive.shape[2]) or int(featureNeuronsWordOrder.shape[1]) != int(featureNeuronsActive.shape[3])):
		raise RuntimeError("createTrainDuringInferenceFeatureNeuronsTargetMask error: featureNeuronsWordOrder shape must match featureNeuronsActive concept and feature dimensions")
	if(featureNeuronsWordOrder.numel() > 0):
		if(bool(pt.any(featureNeuronsWordOrder < 0).item()) or bool(pt.any(featureNeuronsWordOrder >= inferenceSuccessfulPredictionMask.shape[0]).item())):
			raise RuntimeError("createTrainDuringInferenceFeatureNeuronsTargetMask error: featureNeuronsWordOrder contains out of range token indices")
	trainTargetFeaturesMask = pt.logical_not(inferenceSuccessfulPredictionMask)
	if(numSeedTokens > 0):
		trainTargetFeaturesMask[:numSeedTokens] = False
	result = trainTargetFeaturesMask.to(device=featureNeuronsWordOrder.device)[featureNeuronsWordOrder].to(device=featureNeuronsActive.device, dtype=featureNeuronsActive.dtype).unsqueeze(0).unsqueeze(0)
	return result

def getTrainConnectionsIncludeSameTimeIndex(sequenceObservedColumns):
	result = None
	if(not hasattr(sequenceObservedColumns, "trainConnectionsIncludeSameTimeIndex")):
		raise RuntimeError("getTrainConnectionsIncludeSameTimeIndex error: sequenceObservedColumns missing trainConnectionsIncludeSameTimeIndex")
	if(not isinstance(sequenceObservedColumns.trainConnectionsIncludeSameTimeIndex, bool)):
		raise RuntimeError("getTrainConnectionsIncludeSameTimeIndex error: trainConnectionsIncludeSameTimeIndex must be a bool")
	result = sequenceObservedColumns.trainConnectionsIncludeSameTimeIndex
	return result

def getTrainConnectionsUseSpatialDistance(sequenceObservedColumns):
	result = None
	if(not hasattr(sequenceObservedColumns, "trainConnectionsUseSpatialDistance")):
		raise RuntimeError("getTrainConnectionsUseSpatialDistance error: sequenceObservedColumns missing trainConnectionsUseSpatialDistance")
	if(not isinstance(sequenceObservedColumns.trainConnectionsUseSpatialDistance, bool)):
		raise RuntimeError("getTrainConnectionsUseSpatialDistance error: trainConnectionsUseSpatialDistance must be a bool")
	result = sequenceObservedColumns.trainConnectionsUseSpatialDistance
	return result

def getTrainConnectionsUseSpatialAxis(sequenceObservedColumns):
	result = None
	if(not hasattr(sequenceObservedColumns, "trainConnectionsUseSpatialAxis")):
		raise RuntimeError("getTrainConnectionsUseSpatialAxis error: sequenceObservedColumns missing trainConnectionsUseSpatialAxis")
	if(not isinstance(sequenceObservedColumns.trainConnectionsUseSpatialAxis, bool)):
		raise RuntimeError("getTrainConnectionsUseSpatialAxis error: trainConnectionsUseSpatialAxis must be a bool")
	result = sequenceObservedColumns.trainConnectionsUseSpatialAxis
	return result

def getTrainConnectionsUseSpatialAxes(sequenceObservedColumns):
	result = None
	if(not hasattr(sequenceObservedColumns, "trainConnectionsUseSpatialAxes")):
		raise RuntimeError("getTrainConnectionsUseSpatialAxes error: sequenceObservedColumns missing trainConnectionsUseSpatialAxes")
	if(not isinstance(sequenceObservedColumns.trainConnectionsUseSpatialAxes, bool)):
		raise RuntimeError("getTrainConnectionsUseSpatialAxes error: trainConnectionsUseSpatialAxes must be a bool")
	result = sequenceObservedColumns.trainConnectionsUseSpatialAxes
	return result

def getImageSequenceEncodeAxesSourceColumnIndex(cs):
	result = None
	if(not isinstance(modalityORimageSequenceEncodeAxesSourceColumnIndex, int) or isinstance(modalityORimageSequenceEncodeAxesSourceColumnIndex, bool)):
		raise RuntimeError("getImageSequenceEncodeAxesSourceColumnIndex error: modalityORimageSequenceEncodeAxesSourceColumnIndex must be an int")
	if(not isinstance(modalityORimageSequenceEncodeAxesColumnRandom, bool)):
		raise RuntimeError("getImageSequenceEncodeAxesSourceColumnIndex error: modalityORimageSequenceEncodeAxesColumnRandom must be a bool")
	if(not isinstance(cs, int) or isinstance(cs, bool)):
		raise RuntimeError("getImageSequenceEncodeAxesSourceColumnIndex error: cs must be an int")
	if(cs <= 0):
		raise RuntimeError("getImageSequenceEncodeAxesSourceColumnIndex error: cs must be > 0")
	if(modalityORimageSequenceEncodeAxesSourceColumnIndex < 0 or modalityORimageSequenceEncodeAxesSourceColumnIndex >= cs):
		raise RuntimeError("getImageSequenceEncodeAxesSourceColumnIndex error: modalityORimageSequenceEncodeAxesSourceColumnIndex out of range")
	if(modalityORimageSequenceEncodeAxesColumnRandom):
		if(not isinstance(modalityORnumberOfColumnsVS, int) or isinstance(modalityORnumberOfColumnsVS, bool)):
			raise RuntimeError("getImageSequenceEncodeAxesSourceColumnIndex error: modalityORnumberOfColumnsVS must be an int")
		if(modalityORnumberOfColumnsVS <= 0):
			raise RuntimeError("getImageSequenceEncodeAxesSourceColumnIndex error: modalityORnumberOfColumnsVS must be > 0")
		if(cs != int(modalityORnumberOfColumnsVS)):
			raise RuntimeError("getImageSequenceEncodeAxesSourceColumnIndex error: cs must equal modalityORnumberOfColumnsVS when modalityORimageSequenceEncodeAxesColumnRandom")
	result = int(modalityORimageSequenceEncodeAxesSourceColumnIndex)
	return result

def getImageSequenceEncodeAxesTargetColumnIndex(cs):
	result = None
	if(modalityORimageSequenceEncodeAxesColumnRandom):
		raise RuntimeError("getImageSequenceEncodeAxesTargetColumnIndex error: fixed target column index is not defined when modalityORimageSequenceEncodeAxesColumnRandom")
	if(not isinstance(modalityORimageSequenceEncodeAxesTargetColumnIndex, int) or isinstance(modalityORimageSequenceEncodeAxesTargetColumnIndex, bool)):
		raise RuntimeError("getImageSequenceEncodeAxesTargetColumnIndex error: modalityORimageSequenceEncodeAxesTargetColumnIndex must be an int")
	if(not isinstance(cs, int) or isinstance(cs, bool)):
		raise RuntimeError("getImageSequenceEncodeAxesTargetColumnIndex error: cs must be an int")
	if(cs <= 0):
		raise RuntimeError("getImageSequenceEncodeAxesTargetColumnIndex error: cs must be > 0")
	if(modalityORimageSequenceEncodeAxesTargetColumnIndex < 0 or modalityORimageSequenceEncodeAxesTargetColumnIndex >= cs):
		raise RuntimeError("getImageSequenceEncodeAxesTargetColumnIndex error: modalityORimageSequenceEncodeAxesTargetColumnIndex out of range")
	result = int(modalityORimageSequenceEncodeAxesTargetColumnIndex)
	return result

def getSequenceConceptFieldCoordinates(sequenceObservedColumns, targetDevice):
	result = None
	fieldXTensor = None
	fieldYTensor = None
	if(getTrainConnectionsUseSpatialDistance(sequenceObservedColumns)):
		if(not hasattr(sequenceObservedColumns, "sequenceConceptFieldXTensor") or not hasattr(sequenceObservedColumns, "sequenceConceptFieldYTensor")):
			raise RuntimeError("getSequenceConceptFieldCoordinates error: sequenceObservedColumns missing sequence concept field coordinate tensors")
		fieldXTensor = sequenceObservedColumns.sequenceConceptFieldXTensor
		fieldYTensor = sequenceObservedColumns.sequenceConceptFieldYTensor
		if(fieldXTensor is None or fieldYTensor is None):
			raise RuntimeError("getSequenceConceptFieldCoordinates error: sequence concept field coordinate tensors must not be None")
		if(not pt.is_tensor(fieldXTensor) or not pt.is_tensor(fieldYTensor)):
			raise RuntimeError("getSequenceConceptFieldCoordinates error: sequence concept field coordinate tensors must be tensors")
		if(fieldXTensor.dim() != 1 or fieldYTensor.dim() != 1):
			raise RuntimeError("getSequenceConceptFieldCoordinates error: sequence concept field coordinate tensors must be rank 1")
		if(int(fieldXTensor.shape[0]) != int(sequenceObservedColumns.cs) or int(fieldYTensor.shape[0]) != int(sequenceObservedColumns.cs)):
			raise RuntimeError("getSequenceConceptFieldCoordinates error: sequence concept field coordinate tensor lengths must equal cs")
		result = (fieldXTensor.to(device=targetDevice), fieldYTensor.to(device=targetDevice))
	else:
		raise RuntimeError("getSequenceConceptFieldCoordinates error: requires trainConnectionsUseSpatialDistance")
	return result

def getSequenceConceptAxisCoordinates(sequenceObservedColumns, targetDevice):
	result = None
	axisXTensor = None
	axisYTensor = None
	if(getTrainConnectionsUseSpatialAxis(sequenceObservedColumns)):
		if(not hasattr(sequenceObservedColumns, "sequenceConceptAxisXTensor") or not hasattr(sequenceObservedColumns, "sequenceConceptAxisYTensor")):
			raise RuntimeError("getSequenceConceptAxisCoordinates error: sequenceObservedColumns missing sequence concept axis coordinate tensors")
		axisXTensor = sequenceObservedColumns.sequenceConceptAxisXTensor
		axisYTensor = sequenceObservedColumns.sequenceConceptAxisYTensor
		if(axisXTensor is None or axisYTensor is None):
			raise RuntimeError("getSequenceConceptAxisCoordinates error: sequence concept axis coordinate tensors must not be None")
		if(not pt.is_tensor(axisXTensor) or not pt.is_tensor(axisYTensor)):
			raise RuntimeError("getSequenceConceptAxisCoordinates error: sequence concept axis coordinate tensors must be tensors")
		if(axisXTensor.dim() != 1 or axisYTensor.dim() != 1):
			raise RuntimeError("getSequenceConceptAxisCoordinates error: sequence concept axis coordinate tensors must be rank 1")
		if(int(axisXTensor.shape[0]) != int(sequenceObservedColumns.cs) or int(axisYTensor.shape[0]) != int(sequenceObservedColumns.cs)):
			raise RuntimeError("getSequenceConceptAxisCoordinates error: sequence concept axis coordinate tensor lengths must equal cs")
		result = (axisXTensor.to(device=targetDevice), axisYTensor.to(device=targetDevice))
	else:
		raise RuntimeError("getSequenceConceptAxisCoordinates error: requires trainConnectionsUseSpatialAxis")
	return result

def createFeatureWordOrderConnectionMask(sourceWordOrder, targetWordOrder, trainConnectionsIncludeSameTimeIndex):
	result = None
	if(not isinstance(trainConnectionsIncludeSameTimeIndex, bool)):
		raise RuntimeError("createFeatureWordOrderConnectionMask error: trainConnectionsIncludeSameTimeIndex must be a bool")
	if(debugConnectNodesToNextNodesInSequenceOnly):
		wordOrderUpperBound = sourceWordOrder + 1
		if(trainConnectionsIncludeSameTimeIndex):
			result = pt.logical_and(targetWordOrder >= sourceWordOrder, targetWordOrder <= wordOrderUpperBound)
		else:
			result = pt.logical_and(targetWordOrder > sourceWordOrder, targetWordOrder <= wordOrderUpperBound)
	else:
		if(trainConnectionsIncludeSameTimeIndex):
			result = targetWordOrder >= sourceWordOrder
		else:
			result = targetWordOrder > sourceWordOrder
	return result

#first dim cs1 pertains to every concept node in sequence
def processFeaturesActiveTrain(sequenceObservedColumns, featureNeuronsActive, cs, fs, sequenceConceptIndexMask, columnsWordOrder, featureNeuronsWordOrder, featureNeuronsPos, featureNeuronsSegmentMask, sequenceIndex, featureNeuronsTargetMask=None):
	databaseNetworkObject = sequenceObservedColumns.databaseNetworkObject
	processFeaturesActiveTrainStartTime = None
	if(debugPrintTrainSectionTimes):
		processFeaturesActiveTrainStartTime = time.perf_counter()
	featureNeuronsActiveTarget = featureNeuronsActive
	if(useTrainDuringInference):
		if(featureNeuronsTargetMask is None):
			raise RuntimeError("processFeaturesActiveTrain error: featureNeuronsTargetMask is None")
		featureNeuronsActiveTarget = featureNeuronsActive*featureNeuronsTargetMask
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
		featureNeuronsActiveUnion = featureNeuronsActiveTarget.amax(dim=(0, 1))
		featureNeuronsInactiveUnion = 1 - featureNeuronsActiveUnion
		if(useTrainDuringInference):
			featureNeuronsInactiveUnion = featureNeuronsInactiveUnion*featureNeuronsTargetMask.squeeze(0).squeeze(0)
	if(useSparseSequenceNeurons):
		processFeaturesActiveTrainSparseNeurons(sequenceObservedColumns, featureNeuronsActiveTarget, cs, fs, featureNeuronsPos)
	else:
		featureNeuronsInactive = 1 - featureNeuronsActiveTarget
		if(useTrainDuringInference):
			featureNeuronsInactive = featureNeuronsInactive*featureNeuronsTargetMask
		if(arrayIndexPropertiesStrength):
			sequenceObservedColumns.featureNeurons[databaseNetworkObject.arrayIndexPropertiesStrengthIndex, :, :, :, :] += featureNeuronsActiveTarget
		if(arrayIndexPropertiesPermanence):
			sequenceObservedColumns.featureNeurons[databaseNetworkObject.arrayIndexPropertiesPermanenceIndex, :, :, :, :] += featureNeuronsActiveTarget*z1
		if(arrayIndexPropertiesActivation):
			sequenceObservedColumns.featureNeurons[databaseNetworkObject.arrayIndexPropertiesActivationIndex, :, :, :, :] = 0
		if(arrayIndexPropertiesTime):
			sequenceObservedColumns.featureNeurons[databaseNetworkObject.arrayIndexPropertiesTimeIndex, :, :, :, :] = 0
			#OLD inferenceUseNeuronFeaturePropertiesTime=False: sequenceObservedColumns.featureNeurons[databaseNetworkObject.arrayIndexPropertiesTimeIndex, :, :, :, :] = featureNeuronsInactive*sequenceObservedColumns.featureNeurons[databaseNetworkObject.arrayIndexPropertiesTimeIndex] + featureNeuronsActive*sequenceIndex
		if(arrayIndexPropertiesPos):
			sequenceObservedColumns.featureNeurons[databaseNetworkObject.arrayIndexPropertiesPosIndex, :, :, :, :] = featureNeuronsInactive*sequenceObservedColumns.featureNeurons[databaseNetworkObject.arrayIndexPropertiesPosIndex] + featureNeuronsActiveTarget*featureNeuronsPos

	if(useSparseSequenceConnections):
		featureConnectionsActive = None
		featureConnectionsSegmentMask = None
		processFeaturesActiveTrainSparseConnections(sequenceObservedColumns, featureNeuronsActive, cs, fs, columnsWordOrder, featureNeuronsWordOrder, featureNeuronsPos, featureNeuronsTargetMask)
	else:
		featureConnectionsActive, featureConnectionsSegmentMask = processFeaturesActiveTrainDenseConnections(databaseNetworkObject, sequenceObservedColumns, featureNeuronsActive, cs, fs, columnsWordOrder, featureNeuronsWordOrder, featureNeuronsPos, useSparseSequenceConnections, featureNeuronsTargetMask)

	if(trainDecreasePermanenceOfInactiveFeatureNeuronsAndConnections and arrayIndexPropertiesPermanence):
		decreasePermanenceActive(sequenceObservedColumns, featureNeuronsActiveUnion, featureNeuronsInactiveUnion, sequenceConceptIndexMask, featureNeuronsSegmentMask, featureConnectionsSegmentMask)

	if(arrayIndexPropertiesStrength):
		applyTrainConnectionStrengthLimits(sequenceObservedColumns)
	if(debugPrintTrainSectionTimes):
		GIAANNcmn_debug.debugTrainSectionTimesAdd(sequenceObservedColumns.databaseNetworkObject, "processFeaturesActiveTrain", time.perf_counter() - processFeaturesActiveTrainStartTime)

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

def processFeaturesActiveTrainDenseConnections(databaseNetworkObject, sequenceObservedColumns, featureNeuronsActive, cs, fs, columnsWordOrder, featureNeuronsWordOrder, featureNeuronsPos, useSparseSequenceConnections, featureNeuronsTargetMask=None):

	trainConnectionsIncludeSameTimeIndex = getTrainConnectionsIncludeSameTimeIndex(sequenceObservedColumns)
	if(getTrainConnectionsUseSpatialAxes(sequenceObservedColumns)):
		featureConnectionsActive, featureConnectionsSegmentMask = createFeatureConnectionsActiveTrainSpatialAxes(featureNeuronsActive, cs, fs, featureNeuronsWordOrder, trainConnectionsIncludeSameTimeIndex, sequenceObservedColumns)
	elif(useSANI):
		featureConnectionsActive = None
		featureConnectionsSegmentMask = None
		for segmentIndex in range(arrayNumberOfSegments):
			segmentActive = featureNeuronsActive[:, segmentIndex]
			if not pt.any(segmentActive):
				continue
			segmentConnectionsActive, segmentMask = createFeatureConnectionsActiveTrain(segmentActive, cs, fs, columnsWordOrder, featureNeuronsWordOrder, trainConnectionsIncludeSameTimeIndex, sequenceObservedColumns)
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
		featureConnectionsActive, featureConnectionsSegmentMask = createFeatureConnectionsActiveTrain(featureNeuronsActive[:, arrayIndexSegmentLast], cs, fs, columnsWordOrder, featureNeuronsWordOrder, trainConnectionsIncludeSameTimeIndex, sequenceObservedColumns)

	if(useTrainDuringInference):
		featureConnectionsActive, featureConnectionsSegmentMask = applyTrainDuringInferenceFeatureConnectionsTargetMask(featureConnectionsActive, featureConnectionsSegmentMask, featureNeuronsTargetMask)

	featureConnectionsPos = None
	if(arrayIndexPropertiesPos or (arrayIndexPropertiesStrength and trainConnectionStrengthPOSdependence)):
		featureConnectionsPos = featureNeuronsPos.view(1, 1, cs, fs, 1, 1).expand(numberOfDendriticBranches, arrayNumberOfSegments, cs, fs, cs, fs)

	featureConnectionsInactive = None
	if((arrayIndexPropertiesTime or arrayIndexPropertiesPos) and not useSparseSequenceConnections):
		featureConnectionsInactive = 1 - featureConnectionsActive

	if(arrayIndexPropertiesStrength):
		if(trainConnectionStrengthNormaliseWrtContextLength):
			if(getTrainConnectionsUseSpatialAxes(sequenceObservedColumns) or getTrainConnectionsUseSpatialAxis(sequenceObservedColumns)):
				featureConnectionsDistances = calculateFeatureConnectionsActiveSegmentIndexTensor(featureConnectionsActive)
			elif(getTrainConnectionsUseSpatialDistance(sequenceObservedColumns)):
				featureConnectionsDistances = calculateFeatureConnectionsSpatialDistanceTensor(sequenceObservedColumns, cs, fs, featureConnectionsActive.device)
			else:
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

def applyTrainDuringInferenceFeatureConnectionsTargetMask(featureConnectionsActive, featureConnectionsSegmentMask, featureNeuronsTargetMask):
	featureConnectionsActiveResult = None
	featureConnectionsSegmentMaskResult = None
	connectionTargetMask = None
	if(featureConnectionsActive is None):
		raise RuntimeError("applyTrainDuringInferenceFeatureConnectionsTargetMask error: featureConnectionsActive is None")
	if(featureConnectionsSegmentMask is None):
		raise RuntimeError("applyTrainDuringInferenceFeatureConnectionsTargetMask error: featureConnectionsSegmentMask is None")
	if(featureNeuronsTargetMask is None):
		raise RuntimeError("applyTrainDuringInferenceFeatureConnectionsTargetMask error: featureNeuronsTargetMask is None")
	if(not pt.is_tensor(featureConnectionsActive)):
		raise RuntimeError("applyTrainDuringInferenceFeatureConnectionsTargetMask error: featureConnectionsActive must be a tensor")
	if(not pt.is_tensor(featureConnectionsSegmentMask)):
		raise RuntimeError("applyTrainDuringInferenceFeatureConnectionsTargetMask error: featureConnectionsSegmentMask must be a tensor")
	if(not pt.is_tensor(featureNeuronsTargetMask)):
		raise RuntimeError("applyTrainDuringInferenceFeatureConnectionsTargetMask error: featureNeuronsTargetMask must be a tensor")
	if(featureConnectionsActive.dim() != 6):
		raise RuntimeError("applyTrainDuringInferenceFeatureConnectionsTargetMask error: featureConnectionsActive rank must be 6")
	if(featureConnectionsSegmentMask.shape != featureConnectionsActive.shape):
		raise RuntimeError("applyTrainDuringInferenceFeatureConnectionsTargetMask error: featureConnectionsSegmentMask shape mismatch")
	if(featureNeuronsTargetMask.dim() != 4):
		raise RuntimeError("applyTrainDuringInferenceFeatureConnectionsTargetMask error: featureNeuronsTargetMask rank must be 4")
	if(int(featureNeuronsTargetMask.shape[2]) != int(featureConnectionsActive.shape[4]) or int(featureNeuronsTargetMask.shape[3]) != int(featureConnectionsActive.shape[5])):
		raise RuntimeError("applyTrainDuringInferenceFeatureConnectionsTargetMask error: featureNeuronsTargetMask target shape mismatch")
	connectionTargetMask = featureNeuronsTargetMask.to(device=featureConnectionsActive.device, dtype=featureConnectionsActive.dtype).view(1, 1, 1, 1, int(featureConnectionsActive.shape[4]), int(featureConnectionsActive.shape[5]))
	featureConnectionsActiveResult = featureConnectionsActive*connectionTargetMask
	featureConnectionsSegmentMaskResult = pt.logical_and(featureConnectionsSegmentMask, connectionTargetMask.to(dtype=pt.bool))
	return featureConnectionsActiveResult, featureConnectionsSegmentMaskResult

def applyTrainDuringInferenceFeatureConnectionsSparseTargetMask(connectionActiveSparse, featureNeuronsTargetMask):
	result = None
	connectionActiveSparseCoalesced = None
	connectionActiveIndices = None
	connectionActiveValues = None
	filteredConnectionActiveIndices = None
	filteredConnectionActiveValues = None
	targetMask = None
	targetKeepMask = None
	if(connectionActiveSparse is None):
		raise RuntimeError("applyTrainDuringInferenceFeatureConnectionsSparseTargetMask error: connectionActiveSparse is None")
	if(featureNeuronsTargetMask is None):
		raise RuntimeError("applyTrainDuringInferenceFeatureConnectionsSparseTargetMask error: featureNeuronsTargetMask is None")
	if(not pt.is_tensor(connectionActiveSparse)):
		raise RuntimeError("applyTrainDuringInferenceFeatureConnectionsSparseTargetMask error: connectionActiveSparse must be a tensor")
	if(not pt.is_tensor(featureNeuronsTargetMask)):
		raise RuntimeError("applyTrainDuringInferenceFeatureConnectionsSparseTargetMask error: featureNeuronsTargetMask must be a tensor")
	if(connectionActiveSparse.layout != pt.sparse_coo):
		raise RuntimeError("applyTrainDuringInferenceFeatureConnectionsSparseTargetMask error: connectionActiveSparse must be sparse COO")
	if(featureNeuronsTargetMask.dim() != 4):
		raise RuntimeError("applyTrainDuringInferenceFeatureConnectionsSparseTargetMask error: featureNeuronsTargetMask rank must be 4")
	connectionActiveSparseCoalesced = connectionActiveSparse.coalesce()
	if(connectionActiveSparseCoalesced.dim() != 6):
		raise RuntimeError("applyTrainDuringInferenceFeatureConnectionsSparseTargetMask error: connectionActiveSparse rank must be 6")
	if(int(featureNeuronsTargetMask.shape[2]) != int(connectionActiveSparseCoalesced.shape[4]) or int(featureNeuronsTargetMask.shape[3]) != int(connectionActiveSparseCoalesced.shape[5])):
		raise RuntimeError("applyTrainDuringInferenceFeatureConnectionsSparseTargetMask error: featureNeuronsTargetMask target shape mismatch")
	connectionActiveIndices = connectionActiveSparseCoalesced.indices()
	connectionActiveValues = connectionActiveSparseCoalesced.values()
	filteredConnectionActiveIndices = connectionActiveIndices
	filteredConnectionActiveValues = connectionActiveValues
	if(connectionActiveIndices.numel() > 0):
		targetMask = featureNeuronsTargetMask.squeeze(0).squeeze(0).to(device=connectionActiveIndices.device, dtype=pt.bool)
		targetKeepMask = targetMask[connectionActiveIndices[4], connectionActiveIndices[5]]
		filteredConnectionActiveIndices = connectionActiveIndices[:, targetKeepMask]
		filteredConnectionActiveValues = connectionActiveValues[targetKeepMask]
	result = pt.sparse_coo_tensor(filteredConnectionActiveIndices, filteredConnectionActiveValues, size=connectionActiveSparseCoalesced.size(), dtype=connectionActiveSparseCoalesced.dtype, device=connectionActiveSparseCoalesced.device).coalesce()
	return result

def processFeaturesActiveTrainSparseConnections(sequenceObservedColumns, featureNeuronsActive, cs, fs, columnsWordOrder, featureNeuronsWordOrder, featureNeuronsPos, featureNeuronsTargetMask=None):
	databaseNetworkObject = sequenceObservedColumns.databaseNetworkObject
	trainConnectionsIncludeSameTimeIndex = getTrainConnectionsIncludeSameTimeIndex(sequenceObservedColumns)
	if(getTrainConnectionsUseSpatialAxes(sequenceObservedColumns)):
		connectionActiveSparse = createFeatureConnectionsActiveTrainSpatialAxesSparse(featureNeuronsActive, cs, fs, featureNeuronsWordOrder, trainConnectionsIncludeSameTimeIndex, sequenceObservedColumns)
	else:
		connectionActiveSparse = createFeatureConnectionsActiveTrainSparse(featureNeuronsActive, cs, fs, columnsWordOrder, featureNeuronsWordOrder, trainConnectionsIncludeSameTimeIndex, sequenceObservedColumns)
	if(useTrainDuringInference):
		connectionActiveSparse = applyTrainDuringInferenceFeatureConnectionsSparseTargetMask(connectionActiveSparse, featureNeuronsTargetMask)
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
				if(getTrainConnectionsUseSpatialAxes(sequenceObservedColumns) or getTrainConnectionsUseSpatialAxis(sequenceObservedColumns)):
					connectionDistances = connectionActiveIndices[1].to(connectionActiveValues.dtype)
				elif(getTrainConnectionsUseSpatialDistance(sequenceObservedColumns)):
					connectionDistances = calculateSparseFeatureConnectionsSpatialDistanceTensor(sequenceObservedColumns, sourceConceptIndices, targetConceptIndices, connectionActiveValues.device).to(connectionActiveValues.dtype)
				else:
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

def createFeatureConnectionsActiveTrainSpatialAxes(featureNeuronsActive, cs, fs, featureNeuronsWordOrder, trainConnectionsIncludeSameTimeIndex, sequenceObservedColumns):
	result = None
	connectionTargetSize = None
	connectionIndices = None
	connectionValues = None
	connectionSparse = None
	featureConnectionsActive = None
	featureConnectionsSegmentMask = None
	connectionDevice = None
	if(getTrainConnectionsUseSpatialAxes(sequenceObservedColumns)):
		connectionTargetSize = (numberOfDendriticBranches, arrayNumberOfSegments, cs, fs, cs, fs)
		connectionDevice = featureNeuronsActive.device
		connectionIndices = calculateFeatureConnectionsActiveTrainSpatialAxesSparseIndices(featureNeuronsActive, cs, fs, featureNeuronsWordOrder, trainConnectionsIncludeSameTimeIndex, sequenceObservedColumns)
		connectionValues = pt.ones((connectionIndices.shape[1],), dtype=arrayType, device=connectionDevice)
		connectionSparse = pt.sparse_coo_tensor(connectionIndices, connectionValues, size=connectionTargetSize, dtype=arrayType, device=connectionDevice).coalesce()
		if(connectionSparse._nnz() > 0):
			connectionSparse.values().clamp_(max=1.0)
		featureConnectionsActive = connectionSparse.to_dense()
		featureConnectionsSegmentMask = calculateFeatureConnectionsSpatialAxesSegmentMaskTensor(sequenceObservedColumns, cs, fs, connectionDevice)
		result = (featureConnectionsActive, featureConnectionsSegmentMask)
	else:
		raise RuntimeError("createFeatureConnectionsActiveTrainSpatialAxes error: requires trainConnectionsUseSpatialAxes")
	return result

def createFeatureConnectionsActiveTrainSpatialAxesSparse(featureNeuronsActive, cs, fs, featureNeuronsWordOrder, trainConnectionsIncludeSameTimeIndex, sequenceObservedColumns):
	result = None
	connectionTargetSize = None
	connectionIndices = None
	connectionValues = None
	connectionDevice = None
	if(getTrainConnectionsUseSpatialAxes(sequenceObservedColumns)):
		connectionTargetSize = (numberOfDendriticBranches, arrayNumberOfSegments, cs, fs, cs, fs)
		connectionDevice = featureNeuronsActive.device
		connectionIndices = calculateFeatureConnectionsActiveTrainSpatialAxesSparseIndices(featureNeuronsActive, cs, fs, featureNeuronsWordOrder, trainConnectionsIncludeSameTimeIndex, sequenceObservedColumns)
		connectionValues = pt.ones((connectionIndices.shape[1],), dtype=arrayType, device=connectionDevice)
		result = pt.sparse_coo_tensor(connectionIndices, connectionValues, size=connectionTargetSize, dtype=arrayType, device=connectionDevice).coalesce()
		if(result._nnz() > 0):
			result.values().clamp_(max=1.0)
	else:
		raise RuntimeError("createFeatureConnectionsActiveTrainSpatialAxesSparse error: requires trainConnectionsUseSpatialAxes")
	return result

def calculateFeatureConnectionsActiveTrainSpatialAxesSparseIndices(featureNeuronsActive, cs, fs, featureNeuronsWordOrder, trainConnectionsIncludeSameTimeIndex, sequenceObservedColumns):
	result = None
	connectionDevice = None
	targetActive = None
	sourceActive = None
	sourceIndices = None
	targetIndices = None
	sourceCount = None
	targetCount = None
	sourceConceptIndices = None
	sourceFeatureIndices = None
	targetConceptIndices = None
	targetFeatureIndices = None
	branchIndices = None
	sourceWordOrder = None
	targetWordOrder = None
	connectionMask = None
	selfMask = None
	repeatedFeatureMask = None
	repeatedSourceMask = None
	connectionsSegmentIndex = None
	axesSourceColumnIndex = None
	axesTargetColumnIndex = None
	encodedSourceConceptIndices = None
	encodedTargetConceptIndices = None
	featureAxisMaskTensor = None
	featureCentralColumnMaskTensor = None
	if(getTrainConnectionsUseSpatialAxes(sequenceObservedColumns)):
		if(not pt.is_tensor(featureNeuronsActive)):
			raise RuntimeError("calculateFeatureConnectionsActiveTrainSpatialAxesSparseIndices error: featureNeuronsActive must be a tensor")
		if(featureNeuronsActive.dim() != 4):
			raise RuntimeError("calculateFeatureConnectionsActiveTrainSpatialAxesSparseIndices error: featureNeuronsActive rank must be 4")
		if(int(featureNeuronsActive.shape[2]) != int(cs) or int(featureNeuronsActive.shape[3]) != int(fs)):
			raise RuntimeError("calculateFeatureConnectionsActiveTrainSpatialAxesSparseIndices error: featureNeuronsActive dimensions mismatch")
		if(not pt.is_tensor(featureNeuronsWordOrder)):
			raise RuntimeError("calculateFeatureConnectionsActiveTrainSpatialAxesSparseIndices error: featureNeuronsWordOrder must be a tensor")
		if(featureNeuronsWordOrder.dim() != 2 or int(featureNeuronsWordOrder.shape[0]) != int(cs) or int(featureNeuronsWordOrder.shape[1]) != int(fs)):
			raise RuntimeError("calculateFeatureConnectionsActiveTrainSpatialAxesSparseIndices error: featureNeuronsWordOrder dimensions mismatch")
		connectionDevice = featureNeuronsActive.device
		targetActive = featureNeuronsActive.amax(dim=1) > 0
		featureAxisMaskTensor = getImageAxesFeatureAxisMask(sequenceObservedColumns, connectionDevice)
		sourceActive = featureAxisMaskTensor.view(1, fs)
		featureCentralColumnMaskTensor = getImageAxesFeatureCentralColumnMask(sequenceObservedColumns, connectionDevice)
		targetActive = targetActive & featureCentralColumnMaskTensor.view(1, 1, fs)
		sourceIndices = pt.nonzero(sourceActive, as_tuple=False)
		targetIndices = pt.nonzero(targetActive, as_tuple=False)
		result = pt.empty((6, 0), dtype=pt.long, device=connectionDevice)
		if(sourceIndices.shape[0] > 0 and targetIndices.shape[0] > 0):
			sourceCount = int(sourceIndices.shape[0])
			targetCount = int(targetIndices.shape[0])
			sourceConceptIndices = sourceIndices[:, 0].repeat_interleave(targetCount)
			sourceFeatureIndices = sourceIndices[:, 1].repeat_interleave(targetCount)
			branchIndices = targetIndices[:, 0].repeat(sourceCount)
			targetConceptIndices = targetIndices[:, 1].repeat(sourceCount)
			targetFeatureIndices = targetIndices[:, 2].repeat(sourceCount)
			sourceWordOrder = featureNeuronsWordOrder[sourceConceptIndices, sourceFeatureIndices]
			targetWordOrder = featureNeuronsWordOrder[targetConceptIndices, targetFeatureIndices]
			connectionMask = createFeatureWordOrderConnectionMask(sourceWordOrder, targetWordOrder, trainConnectionsIncludeSameTimeIndex)
			selfMask = (sourceConceptIndices == targetConceptIndices) & (sourceFeatureIndices == targetFeatureIndices)
			repeatedFeatureMask = (targetActive > 0).sum(dim=0) > 1
			repeatedSourceMask = repeatedFeatureMask[sourceConceptIndices, sourceFeatureIndices]
			connectionMask = (connectionMask & pt.logical_not(selfMask)) | (selfMask & repeatedSourceMask)
			if(connectionMask.any()):
				sourceFeatureIndices = sourceFeatureIndices[connectionMask]
				targetFeatureIndices = targetFeatureIndices[connectionMask]
				targetConceptIndices = targetConceptIndices[connectionMask]
				branchIndices = branchIndices[connectionMask]
				connectionsSegmentIndex = calculateFeatureConnectionsSpatialAxesFeatureSegmentIndexTensor(sequenceObservedColumns, sourceFeatureIndices, targetFeatureIndices, connectionDevice)
				axesSourceColumnIndex = getImageSequenceEncodeAxesSourceColumnIndex(cs)
				encodedSourceConceptIndices = pt.full_like(sourceFeatureIndices, axesSourceColumnIndex)
				if(modalityORimageSequenceEncodeAxesColumnRandom):
					encodedTargetConceptIndices = targetConceptIndices
				else:
					axesTargetColumnIndex = getImageSequenceEncodeAxesTargetColumnIndex(cs)
					encodedTargetConceptIndices = pt.full_like(targetFeatureIndices, axesTargetColumnIndex)
				result = pt.stack((branchIndices, connectionsSegmentIndex, encodedSourceConceptIndices, sourceFeatureIndices, encodedTargetConceptIndices, targetFeatureIndices), dim=0)
	else:
		raise RuntimeError("calculateFeatureConnectionsActiveTrainSpatialAxesSparseIndices error: requires trainConnectionsUseSpatialAxes")
	return result

def calculateFeatureConnectionsSpatialAxesSegmentMaskTensor(sequenceObservedColumns, cs, fs, targetDevice):
	result = None
	segmentIndexTensor = None
	segmentMaskCollapsed = None
	axesSourceColumnIndex = None
	axesTargetColumnIndex = None
	featureAxisMaskTensor = None
	featureCentralColumnMaskTensor = None
	if(getTrainConnectionsUseSpatialAxes(sequenceObservedColumns)):
		axesSourceColumnIndex = getImageSequenceEncodeAxesSourceColumnIndex(cs)
		segmentIndexTensor = calculateFeatureConnectionsSpatialAxesFeatureSegmentIndexMatrix(sequenceObservedColumns, fs, targetDevice)
		segmentMaskCollapsed = pt.zeros((arrayNumberOfSegments, fs, fs), dtype=pt.bool, device=targetDevice)
		segmentMaskCollapsed.scatter_(0, segmentIndexTensor.unsqueeze(0), True)
		featureAxisMaskTensor = getImageAxesFeatureAxisMask(sequenceObservedColumns, targetDevice)
		segmentMaskCollapsed = segmentMaskCollapsed & featureAxisMaskTensor.view(1, fs, 1)
		featureCentralColumnMaskTensor = getImageAxesFeatureCentralColumnMask(sequenceObservedColumns, targetDevice)
		segmentMaskCollapsed = segmentMaskCollapsed & featureCentralColumnMaskTensor.view(1, 1, fs)
		result = pt.zeros((numberOfDendriticBranches, arrayNumberOfSegments, cs, fs, cs, fs), dtype=pt.bool, device=targetDevice)
		if(modalityORimageSequenceEncodeAxesColumnRandom):
			result[:, :, axesSourceColumnIndex, :, :, :] = segmentMaskCollapsed.view(1, int(arrayNumberOfSegments), int(fs), 1, int(fs)).expand(int(numberOfDendriticBranches), int(arrayNumberOfSegments), int(fs), int(cs), int(fs))
		else:
			axesTargetColumnIndex = getImageSequenceEncodeAxesTargetColumnIndex(cs)
			result[:, :, axesSourceColumnIndex, :, axesTargetColumnIndex, :] = segmentMaskCollapsed
	else:
		raise RuntimeError("calculateFeatureConnectionsSpatialAxesSegmentMaskTensor error: requires trainConnectionsUseSpatialAxes")
	return result

def calculateFeatureConnectionsSpatialAxesFeatureSegmentIndexMatrix(sequenceObservedColumns, fs, targetDevice):
	result = None
	sourceFeatureIndices = None
	targetFeatureIndices = None
	if(getTrainConnectionsUseSpatialAxes(sequenceObservedColumns)):
		sourceFeatureIndices = pt.arange(fs, device=targetDevice, dtype=pt.long).view(fs, 1).expand(fs, fs).reshape(-1)
		targetFeatureIndices = pt.arange(fs, device=targetDevice, dtype=pt.long).view(1, fs).expand(fs, fs).reshape(-1)
		result = calculateFeatureConnectionsSpatialAxesFeatureSegmentIndexTensor(sequenceObservedColumns, sourceFeatureIndices, targetFeatureIndices, targetDevice).view(fs, fs)
	else:
		raise RuntimeError("calculateFeatureConnectionsSpatialAxesFeatureSegmentIndexMatrix error: requires trainConnectionsUseSpatialAxes")
	return result

def calculateFeatureConnectionsSpatialAxesFeatureSegmentIndexTensor(sequenceObservedColumns, sourceFeatureIndices, targetFeatureIndices, targetDevice):
	result = None
	fieldXTensor = None
	fieldYTensor = None
	centralFieldX = None
	centralFieldY = None
	deltaX = None
	deltaY = None
	distanceSquared = None
	if(getTrainConnectionsUseSpatialAxes(sequenceObservedColumns)):
		if(not pt.is_tensor(sourceFeatureIndices) or not pt.is_tensor(targetFeatureIndices)):
			raise RuntimeError("calculateFeatureConnectionsSpatialAxesFeatureSegmentIndexTensor error: sourceFeatureIndices/targetFeatureIndices must be tensors")
		if(sourceFeatureIndices.shape != targetFeatureIndices.shape):
			raise RuntimeError("calculateFeatureConnectionsSpatialAxesFeatureSegmentIndexTensor error: sourceFeatureIndices/targetFeatureIndices shape mismatch")
		fieldXTensor, fieldYTensor = getImageAxesFeatureFieldCoordinates(sequenceObservedColumns, targetDevice)
		centralFieldX, centralFieldY = getImageAxesCentralFieldCoordinates(sequenceObservedColumns)
		deltaX = pt.abs(centralFieldX - fieldXTensor[sourceFeatureIndices])
		deltaY = pt.abs(centralFieldY - fieldYTensor[sourceFeatureIndices])
		distanceSquared = (deltaX*deltaX + deltaY*deltaY).to(arrayType)
		result = pt.ceil(pt.sqrt(distanceSquared)).long()
		if(bool(pt.any(result < 0).item()) or bool(pt.any(result >= arrayNumberOfSegments).item())):
			raise RuntimeError("calculateFeatureConnectionsSpatialAxesFeatureSegmentIndexTensor error: calculated segment index out of range")
	else:
		raise RuntimeError("calculateFeatureConnectionsSpatialAxesFeatureSegmentIndexTensor error: requires trainConnectionsUseSpatialAxes")
	return result

def getImageAxesFeatureFieldCoordinates(sequenceObservedColumns, targetDevice):
	result = None
	fieldXTensor = None
	fieldYTensor = None
	if(getTrainConnectionsUseSpatialAxes(sequenceObservedColumns)):
		if(not hasattr(sequenceObservedColumns, "imageAxesFeatureFieldXTensor") or not hasattr(sequenceObservedColumns, "imageAxesFeatureFieldYTensor")):
			raise RuntimeError("getImageAxesFeatureFieldCoordinates error: sequenceObservedColumns missing image axes feature field coordinate tensors")
		fieldXTensor = sequenceObservedColumns.imageAxesFeatureFieldXTensor
		fieldYTensor = sequenceObservedColumns.imageAxesFeatureFieldYTensor
		if(fieldXTensor is None or fieldYTensor is None):
			raise RuntimeError("getImageAxesFeatureFieldCoordinates error: feature field coordinate tensors must not be None")
		if(not pt.is_tensor(fieldXTensor) or not pt.is_tensor(fieldYTensor)):
			raise RuntimeError("getImageAxesFeatureFieldCoordinates error: feature field coordinate tensors must be tensors")
		if(fieldXTensor.dim() != 1 or fieldYTensor.dim() != 1):
			raise RuntimeError("getImageAxesFeatureFieldCoordinates error: feature field coordinate tensors must be rank 1")
		if(int(fieldXTensor.shape[0]) != int(sequenceObservedColumns.fs) or int(fieldYTensor.shape[0]) != int(sequenceObservedColumns.fs)):
			raise RuntimeError("getImageAxesFeatureFieldCoordinates error: feature field coordinate tensor lengths must equal fs")
		result = (fieldXTensor.to(device=targetDevice), fieldYTensor.to(device=targetDevice))
	else:
		raise RuntimeError("getImageAxesFeatureFieldCoordinates error: requires trainConnectionsUseSpatialAxes")
	return result

def getImageAxesFeatureAxisMask(sequenceObservedColumns, targetDevice):
	result = None
	featureAxisMaskTensor = None
	if(getTrainConnectionsUseSpatialAxes(sequenceObservedColumns)):
		if(not hasattr(sequenceObservedColumns, "imageAxesFeatureAxisMaskTensor")):
			raise RuntimeError("getImageAxesFeatureAxisMask error: sequenceObservedColumns missing imageAxesFeatureAxisMaskTensor")
		featureAxisMaskTensor = sequenceObservedColumns.imageAxesFeatureAxisMaskTensor
		if(featureAxisMaskTensor is None):
			raise RuntimeError("getImageAxesFeatureAxisMask error: imageAxesFeatureAxisMaskTensor must not be None")
		if(not pt.is_tensor(featureAxisMaskTensor)):
			raise RuntimeError("getImageAxesFeatureAxisMask error: imageAxesFeatureAxisMaskTensor must be a tensor")
		if(featureAxisMaskTensor.dim() != 1):
			raise RuntimeError("getImageAxesFeatureAxisMask error: imageAxesFeatureAxisMaskTensor must be rank 1")
		if(int(featureAxisMaskTensor.shape[0]) != int(sequenceObservedColumns.fs)):
			raise RuntimeError("getImageAxesFeatureAxisMask error: imageAxesFeatureAxisMaskTensor length must equal fs")
		result = featureAxisMaskTensor.to(device=targetDevice, dtype=pt.bool)
	else:
		raise RuntimeError("getImageAxesFeatureAxisMask error: requires trainConnectionsUseSpatialAxes")
	return result


def getImageAxesFeatureCentralColumnMask(sequenceObservedColumns, targetDevice):
	result = None
	featureCentralColumnMaskTensor = None
	if(getTrainConnectionsUseSpatialAxes(sequenceObservedColumns)):
		if(not hasattr(sequenceObservedColumns, "imageAxesFeatureCentralColumnMaskTensor")):
			raise RuntimeError("getImageAxesFeatureCentralColumnMask error: sequenceObservedColumns missing imageAxesFeatureCentralColumnMaskTensor")
		featureCentralColumnMaskTensor = sequenceObservedColumns.imageAxesFeatureCentralColumnMaskTensor
		if(featureCentralColumnMaskTensor is None):
			raise RuntimeError("getImageAxesFeatureCentralColumnMask error: imageAxesFeatureCentralColumnMaskTensor must not be None")
		if(not pt.is_tensor(featureCentralColumnMaskTensor)):
			raise RuntimeError("getImageAxesFeatureCentralColumnMask error: imageAxesFeatureCentralColumnMaskTensor must be a tensor")
		if(featureCentralColumnMaskTensor.dim() != 1):
			raise RuntimeError("getImageAxesFeatureCentralColumnMask error: imageAxesFeatureCentralColumnMaskTensor must be rank 1")
		if(int(featureCentralColumnMaskTensor.shape[0]) != int(sequenceObservedColumns.fs)):
			raise RuntimeError("getImageAxesFeatureCentralColumnMask error: imageAxesFeatureCentralColumnMaskTensor length must equal fs")
		result = featureCentralColumnMaskTensor.to(device=targetDevice, dtype=pt.bool)
	else:
		raise RuntimeError("getImageAxesFeatureCentralColumnMask error: requires trainConnectionsUseSpatialAxes")
	return result


def getImageAxesCentralFieldCoordinates(sequenceObservedColumns):
	result = None
	centralFieldX = None
	centralFieldY = None
	if(getTrainConnectionsUseSpatialAxes(sequenceObservedColumns)):
		if(not hasattr(sequenceObservedColumns, "imageAxesCentralFieldX") or not hasattr(sequenceObservedColumns, "imageAxesCentralFieldY")):
			raise RuntimeError("getImageAxesCentralFieldCoordinates error: sequenceObservedColumns missing image axes central field coordinates")
		centralFieldX = sequenceObservedColumns.imageAxesCentralFieldX
		centralFieldY = sequenceObservedColumns.imageAxesCentralFieldY
		if(not isinstance(centralFieldX, int) or isinstance(centralFieldX, bool) or not isinstance(centralFieldY, int) or isinstance(centralFieldY, bool)):
			raise RuntimeError("getImageAxesCentralFieldCoordinates error: image axes central field coordinates must be ints")
		if(centralFieldX < 0 or centralFieldX >= int(modalityORimageSequenceEncodeDistanceFieldSegments) or centralFieldY < 0 or centralFieldY >= int(modalityORimageSequenceEncodeDistanceFieldSegments)):
			raise RuntimeError("getImageAxesCentralFieldCoordinates error: image axes central field coordinates out of range")
		result = (centralFieldX, centralFieldY)
	else:
		raise RuntimeError("getImageAxesCentralFieldCoordinates error: requires trainConnectionsUseSpatialAxes")
	return result

def createFeatureConnectionsActiveTrainSparse(featureNeuronsActive, cs, fs, columnsWordOrder, featureNeuronsWordOrder, trainConnectionsIncludeSameTimeIndex, sequenceObservedColumns):
	connectionTargetSize = (numberOfDendriticBranches, arrayNumberOfSegments, cs, fs, cs, fs)
	connectionDevice = featureNeuronsActive.device
	combinedIndices = pt.empty((len(connectionTargetSize), 0), dtype=pt.long, device=connectionDevice)
	combinedValues = pt.empty((0,), dtype=arrayType, device=connectionDevice)
	indicesList = []
	if(useSANI):
		for segmentIndex in range(arrayNumberOfSegments):
			segmentActive = featureNeuronsActive[:, segmentIndex]
			if(pt.any(segmentActive)):
				segmentConnectionIndices = createFeatureConnectionsActiveTrainSparseSegment(segmentActive, cs, fs, columnsWordOrder, featureNeuronsWordOrder, trainConnectionsIncludeSameTimeIndex, sequenceObservedColumns)
				if(segmentConnectionIndices.numel() > 0):
					indicesList.append(segmentConnectionIndices)
	else:
		segmentConnectionIndices = createFeatureConnectionsActiveTrainSparseSegment(featureNeuronsActive[:, arrayIndexSegmentLast], cs, fs, columnsWordOrder, featureNeuronsWordOrder, trainConnectionsIncludeSameTimeIndex, sequenceObservedColumns)
		if(segmentConnectionIndices.numel() > 0):
			indicesList.append(segmentConnectionIndices)
	if(len(indicesList) > 0):
		combinedIndices = pt.cat(indicesList, dim=1)
		if(getTrainConnectionsUseSpatialAxes(sequenceObservedColumns)):
			combinedIndices = collapseFeatureConnectionsSpatialAxesSparseIndices(combinedIndices, cs)
		combinedValues = pt.ones((combinedIndices.shape[1],), dtype=arrayType, device=connectionDevice)
	result = pt.sparse_coo_tensor(combinedIndices, combinedValues, size=connectionTargetSize, dtype=arrayType, device=connectionDevice).coalesce()
	if(result._nnz() > 0):
		result.values().clamp_(max=1.0)
	return result

def createFeatureConnectionsActiveTrainSparseSegment(segmentActive, cs, fs, columnsWordOrder, featureNeuronsWordOrder, trainConnectionsIncludeSameTimeIndex, sequenceObservedColumns):
	connectionDevice = segmentActive.device
	combinedIndices = pt.empty((6, 0), dtype=pt.long, device=connectionDevice)
	indicesList = []
	if(multipleDendriticBranches):
		sourceActive = segmentActive.amax(dim=0)
		sourceIndices = pt.nonzero(sourceActive > 0, as_tuple=False)
		repeatedFeatureMask = (segmentActive > 0).sum(dim=0) > 1
		for branchIndex in range(segmentActive.shape[0]):
			targetIndices = pt.nonzero(segmentActive[branchIndex] > 0, as_tuple=False)
			branchConnectionIndices = createFeatureConnectionsActiveTrainSparseBranch(branchIndex, sourceIndices, targetIndices, repeatedFeatureMask, columnsWordOrder, featureNeuronsWordOrder, trainConnectionsIncludeSameTimeIndex, sequenceObservedColumns)
			if(branchConnectionIndices.numel() > 0):
				indicesList.append(branchConnectionIndices)
	else:
		sourceIndices = pt.nonzero(segmentActive[0] > 0, as_tuple=False)
		targetIndices = sourceIndices
		branchConnectionIndices = createFeatureConnectionsActiveTrainSparseBranch(0, sourceIndices, targetIndices, None, columnsWordOrder, featureNeuronsWordOrder, trainConnectionsIncludeSameTimeIndex, sequenceObservedColumns)
		if(branchConnectionIndices.numel() > 0):
			indicesList.append(branchConnectionIndices)
	if(len(indicesList) > 0):
		combinedIndices = pt.cat(indicesList, dim=1)
	return combinedIndices

def createFeatureConnectionsActiveTrainSparseBranch(branchIndex, sourceIndices, targetIndices, repeatedFeatureMask, columnsWordOrder, featureNeuronsWordOrder, trainConnectionsIncludeSameTimeIndex, sequenceObservedColumns):
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
			connectionMask = connectionMask & createFeatureWordOrderConnectionMask(sourceWordOrder, targetWordOrder, trainConnectionsIncludeSameTimeIndex)
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
			result = assignFeatureConnectionsToTargetSegmentsSparse(branchIndices, sourceConceptIndices, sourceFeatureIndices, targetConceptIndices, targetFeatureIndices, sourceWordOrder, targetWordOrder, sequenceObservedColumns)
	return result

def assignFeatureConnectionsToTargetSegmentsSparse(branchIndices, sourceConceptIndices, sourceFeatureIndices, targetConceptIndices, targetFeatureIndices, sourceWordOrder, targetWordOrder, sequenceObservedColumns):
	connectionDevice = branchIndices.device
	indicesList = []
	if(getTrainConnectionsUseSpatialDistance(sequenceObservedColumns)):
		connectionsSegmentIndex = calculateSparseFeatureConnectionsSpatialDistanceTensor(sequenceObservedColumns, sourceConceptIndices, targetConceptIndices, connectionDevice)
		if(getTrainConnectionsUseSpatialAxis(sequenceObservedColumns)):
			indicesList.append(expandFeatureConnectionsSpatialAxisSparse(sequenceObservedColumns, branchIndices, connectionsSegmentIndex, sourceConceptIndices, sourceFeatureIndices, targetConceptIndices, targetFeatureIndices, connectionDevice))
		else:
			indicesList.append(pt.stack((branchIndices, connectionsSegmentIndex, sourceConceptIndices, sourceFeatureIndices, targetConceptIndices, targetFeatureIndices), dim=0))
	elif(useSANIcolumns):
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

def createFeatureConnectionsActiveTrain(featureNeuronsActive, cs, fs, columnsWordOrder, featureNeuronsWordOrder, trainConnectionsIncludeSameTimeIndex, sequenceObservedColumns):

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
		wordOrderMask = createFeatureWordOrderConnectionMask(featureNeuronsWordOrderExpanded1, featureNeuronsWordOrderExpanded2, trainConnectionsIncludeSameTimeIndex)
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
		featureConnectionsActive, featureConnectionsSegmentMask = assignFeatureConnectionsToTargetSegments(featureConnectionsActive, cs, fs, featureNeuronsWordOrder, sequenceObservedColumns)
	else:
		if featureConnectionsActive.dim() == 5:
			featureConnectionsActive = featureConnectionsActive.unsqueeze(1)
		else:
			featureConnectionsActive = featureConnectionsActive.unsqueeze(0)
		featureConnectionsSegmentMask = pt.ones_like(featureConnectionsActive, dtype=pt.bool)
	
	return featureConnectionsActive, featureConnectionsSegmentMask

def assignFeatureConnectionsToTargetSegments(featureConnectionsActive, cs, fs, featureNeuronsWordOrder, sequenceObservedColumns):
	hasBranchDim = featureConnectionsActive.dim() == 5
	branchCount = featureConnectionsActive.shape[0] if hasBranchDim else 1
	if(getTrainConnectionsUseSpatialDistance(sequenceObservedColumns)):
		featureConnectionsSegmentMask = calculateFeatureConnectionsSpatialDistanceSegmentMask(sequenceObservedColumns, cs, fs, featureConnectionsActive.device)
	elif(useSANIcolumns):
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
	if(getTrainConnectionsUseSpatialAxis(sequenceObservedColumns)):
		featureConnectionsActive, featureConnectionsSegmentMask = expandFeatureConnectionsSpatialAxis(sequenceObservedColumns, featureConnectionsActive, featureConnectionsSegmentMask, cs)

	return featureConnectionsActive, featureConnectionsSegmentMask


def calculateFeatureConnectionsSpatialDistanceSegmentMask(sequenceObservedColumns, cs, fs, targetDevice):
	result = None
	connectionsSegmentIndex = None
	if(getTrainConnectionsUseSpatialDistance(sequenceObservedColumns)):
		connectionsSegmentIndex = calculateFeatureConnectionsSpatialDistanceTensor(sequenceObservedColumns, cs, fs, targetDevice).long()
		if(bool(pt.any(connectionsSegmentIndex < 0).item()) or bool(pt.any(connectionsSegmentIndex >= arrayNumberOfSegments).item())):
			raise RuntimeError("calculateFeatureConnectionsSpatialDistanceSegmentMask error: calculated segment index out of range")
		result = pt.zeros((arrayNumberOfSegments, cs, fs, cs, fs), dtype=pt.bool, device=targetDevice)
		result.scatter_(0, connectionsSegmentIndex.unsqueeze(0), True)
	else:
		raise RuntimeError("calculateFeatureConnectionsSpatialDistanceSegmentMask error: requires trainConnectionsUseSpatialDistance")
	return result


def calculateFeatureConnectionsSpatialAxisSourceMask(sequenceObservedColumns, cs, targetDevice):
	result = None
	axisXTensor = None
	axisYTensor = None
	axisSourceX = None
	axisSourceY = None
	axisExpandedX = None
	axisExpandedY = None
	if(getTrainConnectionsUseSpatialAxis(sequenceObservedColumns)):
		axisXTensor, axisYTensor = getSequenceConceptAxisCoordinates(sequenceObservedColumns, targetDevice)
		if(int(axisXTensor.shape[0]) != int(cs) or int(axisYTensor.shape[0]) != int(cs)):
			raise RuntimeError("calculateFeatureConnectionsSpatialAxisSourceMask error: sequence concept axis coordinate tensor lengths must equal cs")
		axisSourceX = axisXTensor.view(cs, 1).expand(cs, cs)
		axisSourceY = axisYTensor.view(cs, 1).expand(cs, cs)
		axisExpandedX = axisXTensor.view(1, cs).expand(cs, cs)
		axisExpandedY = axisYTensor.view(1, cs).expand(cs, cs)
		result = (axisSourceX == axisExpandedX) | (axisSourceY == axisExpandedY)
	else:
		raise RuntimeError("calculateFeatureConnectionsSpatialAxisSourceMask error: requires trainConnectionsUseSpatialAxis")
	return result


def expandFeatureConnectionsSpatialAxis(sequenceObservedColumns, featureConnectionsActive, featureConnectionsSegmentMask, cs):
	result = None
	axisSourceMask = None
	if(getTrainConnectionsUseSpatialAxis(sequenceObservedColumns)):
		if(featureConnectionsActive.device != featureConnectionsSegmentMask.device):
			raise RuntimeError("expandFeatureConnectionsSpatialAxis error: featureConnectionsActive and featureConnectionsSegmentMask device mismatch")
		axisSourceMask = calculateFeatureConnectionsSpatialAxisSourceMask(sequenceObservedColumns, cs, featureConnectionsActive.device)
		featureConnectionsActive = expandFeatureConnectionsSpatialAxisTensor(featureConnectionsActive, axisSourceMask)
		featureConnectionsSegmentMask = expandFeatureConnectionsSpatialAxisMask(featureConnectionsSegmentMask, axisSourceMask)
		result = (featureConnectionsActive, featureConnectionsSegmentMask)
	else:
		raise RuntimeError("expandFeatureConnectionsSpatialAxis error: requires trainConnectionsUseSpatialAxis")
	return result


def expandFeatureConnectionsSpatialAxisTensor(featureConnectionsTensor, axisSourceMask):
	result = None
	axisSourceMaskType = None
	if(not pt.is_tensor(featureConnectionsTensor)):
		raise RuntimeError("expandFeatureConnectionsSpatialAxisTensor error: featureConnectionsTensor must be a tensor")
	if(not pt.is_tensor(axisSourceMask)):
		raise RuntimeError("expandFeatureConnectionsSpatialAxisTensor error: axisSourceMask must be a tensor")
	if(featureConnectionsTensor.dim() == 6):
		axisSourceMaskType = axisSourceMask.to(dtype=featureConnectionsTensor.dtype)
		result = pt.einsum("bsaftg,ae->bseftg", featureConnectionsTensor, axisSourceMaskType).clamp(max=1)
	elif(featureConnectionsTensor.dim() == 5):
		axisSourceMaskType = axisSourceMask.to(dtype=featureConnectionsTensor.dtype)
		result = pt.einsum("saftg,ae->seftg", featureConnectionsTensor, axisSourceMaskType).clamp(max=1)
	else:
		raise RuntimeError("expandFeatureConnectionsSpatialAxisTensor error: featureConnectionsTensor rank must be 5 or 6")
	return result


def expandFeatureConnectionsSpatialAxisMask(featureConnectionsSegmentMask, axisSourceMask):
	result = None
	featureConnectionsSegmentMaskType = None
	axisSourceMaskType = None
	if(not pt.is_tensor(featureConnectionsSegmentMask)):
		raise RuntimeError("expandFeatureConnectionsSpatialAxisMask error: featureConnectionsSegmentMask must be a tensor")
	if(not pt.is_tensor(axisSourceMask)):
		raise RuntimeError("expandFeatureConnectionsSpatialAxisMask error: axisSourceMask must be a tensor")
	featureConnectionsSegmentMaskType = featureConnectionsSegmentMask.to(dtype=arrayType)
	axisSourceMaskType = axisSourceMask.to(dtype=arrayType)
	if(featureConnectionsSegmentMask.dim() == 6):
		result = pt.einsum("bsaftg,ae->bseftg", featureConnectionsSegmentMaskType, axisSourceMaskType) > 0
	elif(featureConnectionsSegmentMask.dim() == 5):
		result = pt.einsum("saftg,ae->seftg", featureConnectionsSegmentMaskType, axisSourceMaskType) > 0
	else:
		raise RuntimeError("expandFeatureConnectionsSpatialAxisMask error: featureConnectionsSegmentMask rank must be 5 or 6")
	return result


def calculateFeatureConnectionsActiveSegmentIndexTensor(featureConnectionsActive):
	result = None
	segmentIndexTensor = None
	if(not pt.is_tensor(featureConnectionsActive)):
		raise RuntimeError("calculateFeatureConnectionsActiveSegmentIndexTensor error: featureConnectionsActive must be a tensor")
	if(featureConnectionsActive.dim() == 6):
		segmentIndexTensor = pt.arange(arrayNumberOfSegments, device=featureConnectionsActive.device, dtype=arrayType)
		result = segmentIndexTensor.view(1, arrayNumberOfSegments, 1, 1, 1, 1)
	elif(featureConnectionsActive.dim() == 5):
		segmentIndexTensor = pt.arange(arrayNumberOfSegments, device=featureConnectionsActive.device, dtype=arrayType)
		result = segmentIndexTensor.view(arrayNumberOfSegments, 1, 1, 1, 1)
	else:
		raise RuntimeError("calculateFeatureConnectionsActiveSegmentIndexTensor error: featureConnectionsActive rank must be 5 or 6")
	return result


def collapseFeatureConnectionsSpatialAxesTensor(featureConnectionsTensor, cs):
	result = None
	collapsedFeatureConnectionsTensor = None
	axesSourceColumnIndex = None
	axesTargetColumnIndex = None
	if(not pt.is_tensor(featureConnectionsTensor)):
		raise RuntimeError("collapseFeatureConnectionsSpatialAxesTensor error: featureConnectionsTensor must be a tensor")
	if(not isinstance(cs, int) or isinstance(cs, bool)):
		raise RuntimeError("collapseFeatureConnectionsSpatialAxesTensor error: cs must be an int")
	if(cs <= 0):
		raise RuntimeError("collapseFeatureConnectionsSpatialAxesTensor error: cs must be > 0")
	axesSourceColumnIndex = getImageSequenceEncodeAxesSourceColumnIndex(cs)
	if(featureConnectionsTensor.dim() == 6):
		if(int(featureConnectionsTensor.shape[2]) != cs or int(featureConnectionsTensor.shape[4]) != cs):
			raise RuntimeError("collapseFeatureConnectionsSpatialAxesTensor error: featureConnectionsTensor concept dimensions must equal cs")
		result = pt.zeros_like(featureConnectionsTensor)
		if(modalityORimageSequenceEncodeAxesColumnRandom):
			collapsedFeatureConnectionsTensor = featureConnectionsTensor.amax(dim=2)
			result[:, :, axesSourceColumnIndex, :, :, :] = collapsedFeatureConnectionsTensor
		else:
			axesTargetColumnIndex = getImageSequenceEncodeAxesTargetColumnIndex(cs)
			collapsedFeatureConnectionsTensor = featureConnectionsTensor.amax(dim=(2, 4))
			result[:, :, axesSourceColumnIndex, :, axesTargetColumnIndex, :] = collapsedFeatureConnectionsTensor
	elif(featureConnectionsTensor.dim() == 5):
		if(int(featureConnectionsTensor.shape[1]) != cs or int(featureConnectionsTensor.shape[3]) != cs):
			raise RuntimeError("collapseFeatureConnectionsSpatialAxesTensor error: featureConnectionsTensor concept dimensions must equal cs")
		result = pt.zeros_like(featureConnectionsTensor)
		if(modalityORimageSequenceEncodeAxesColumnRandom):
			collapsedFeatureConnectionsTensor = featureConnectionsTensor.amax(dim=1)
			result[:, axesSourceColumnIndex, :, :, :] = collapsedFeatureConnectionsTensor
		else:
			axesTargetColumnIndex = getImageSequenceEncodeAxesTargetColumnIndex(cs)
			collapsedFeatureConnectionsTensor = featureConnectionsTensor.amax(dim=(1, 3))
			result[:, axesSourceColumnIndex, :, axesTargetColumnIndex, :] = collapsedFeatureConnectionsTensor
	else:
		raise RuntimeError("collapseFeatureConnectionsSpatialAxesTensor error: featureConnectionsTensor rank must be 5 or 6")
	return result


def collapseFeatureConnectionsSpatialAxesSparseIndices(connectionIndices, cs):
	result = None
	axesSourceColumnIndex = None
	axesTargetColumnIndex = None
	if(not pt.is_tensor(connectionIndices)):
		raise RuntimeError("collapseFeatureConnectionsSpatialAxesSparseIndices error: connectionIndices must be a tensor")
	if(not isinstance(cs, int) or isinstance(cs, bool)):
		raise RuntimeError("collapseFeatureConnectionsSpatialAxesSparseIndices error: cs must be an int")
	if(cs <= 0):
		raise RuntimeError("collapseFeatureConnectionsSpatialAxesSparseIndices error: cs must be > 0")
	if(connectionIndices.dim() != 2):
		raise RuntimeError("collapseFeatureConnectionsSpatialAxesSparseIndices error: connectionIndices rank must be 2")
	if(int(connectionIndices.shape[0]) != 6):
		raise RuntimeError("collapseFeatureConnectionsSpatialAxesSparseIndices error: connectionIndices first dimension must equal 6")
	axesSourceColumnIndex = getImageSequenceEncodeAxesSourceColumnIndex(cs)
	result = connectionIndices.clone()
	if(result.numel() > 0):
		result[2] = axesSourceColumnIndex
		if(not modalityORimageSequenceEncodeAxesColumnRandom):
			axesTargetColumnIndex = getImageSequenceEncodeAxesTargetColumnIndex(cs)
			result[4] = axesTargetColumnIndex
	return result


def expandFeatureConnectionsSpatialAxisSparse(sequenceObservedColumns, branchIndices, connectionsSegmentIndex, sourceConceptIndices, sourceFeatureIndices, targetConceptIndices, targetFeatureIndices, targetDevice):
	result = None
	axisSourceMask = None
	connectionAxisMask = None
	connectionIndices = None
	expandedSourceConceptIndices = None
	if(getTrainConnectionsUseSpatialAxis(sequenceObservedColumns)):
		if(not pt.is_tensor(branchIndices) or not pt.is_tensor(connectionsSegmentIndex) or not pt.is_tensor(sourceConceptIndices) or not pt.is_tensor(sourceFeatureIndices) or not pt.is_tensor(targetConceptIndices) or not pt.is_tensor(targetFeatureIndices)):
			raise RuntimeError("expandFeatureConnectionsSpatialAxisSparse error: connection index inputs must be tensors")
		if(branchIndices.shape != connectionsSegmentIndex.shape or branchIndices.shape != sourceConceptIndices.shape or branchIndices.shape != sourceFeatureIndices.shape or branchIndices.shape != targetConceptIndices.shape or branchIndices.shape != targetFeatureIndices.shape):
			raise RuntimeError("expandFeatureConnectionsSpatialAxisSparse error: connection index input shape mismatch")
		axisSourceMask = calculateFeatureConnectionsSpatialAxisSourceMask(sequenceObservedColumns, int(sequenceObservedColumns.cs), targetDevice)
		connectionAxisMask = axisSourceMask[sourceConceptIndices]
		connectionIndices, expandedSourceConceptIndices = pt.nonzero(connectionAxisMask, as_tuple=True)
		if(connectionIndices.numel() > 0):
			result = pt.stack((branchIndices[connectionIndices], connectionsSegmentIndex[connectionIndices], expandedSourceConceptIndices, sourceFeatureIndices[connectionIndices], targetConceptIndices[connectionIndices], targetFeatureIndices[connectionIndices]), dim=0)
		else:
			result = pt.empty((6, 0), dtype=pt.long, device=targetDevice)
	else:
		raise RuntimeError("expandFeatureConnectionsSpatialAxisSparse error: requires trainConnectionsUseSpatialAxis")
	return result


def calculateFeatureConnectionsSpatialDistanceTensor(sequenceObservedColumns, cs, fs, targetDevice):
	result = None
	fieldXTensor = None
	fieldYTensor = None
	sourceFieldX = None
	sourceFieldY = None
	targetFieldX = None
	targetFieldY = None
	deltaX = None
	deltaY = None
	distanceSquared = None
	if(getTrainConnectionsUseSpatialDistance(sequenceObservedColumns)):
		fieldXTensor, fieldYTensor = getSequenceConceptFieldCoordinates(sequenceObservedColumns, targetDevice)
		if(int(fieldXTensor.shape[0]) != int(cs) or int(fieldYTensor.shape[0]) != int(cs)):
			raise RuntimeError("calculateFeatureConnectionsSpatialDistanceTensor error: sequence concept field coordinate tensor lengths must equal cs")
		sourceFieldX = fieldXTensor.view(cs, 1, 1, 1).expand(cs, fs, cs, fs)
		sourceFieldY = fieldYTensor.view(cs, 1, 1, 1).expand(cs, fs, cs, fs)
		targetFieldX = fieldXTensor.view(1, 1, cs, 1).expand(cs, fs, cs, fs)
		targetFieldY = fieldYTensor.view(1, 1, cs, 1).expand(cs, fs, cs, fs)
		deltaX = pt.abs(targetFieldX - sourceFieldX)
		deltaY = pt.abs(targetFieldY - sourceFieldY)
		distanceSquared = (deltaX*deltaX + deltaY*deltaY).to(arrayType)
		result = pt.ceil(pt.sqrt(distanceSquared)).long()
	else:
		raise RuntimeError("calculateFeatureConnectionsSpatialDistanceTensor error: requires trainConnectionsUseSpatialDistance")
	return result


def calculateSparseFeatureConnectionsSpatialDistanceTensor(sequenceObservedColumns, sourceConceptIndices, targetConceptIndices, targetDevice):
	result = None
	fieldXTensor = None
	fieldYTensor = None
	deltaX = None
	deltaY = None
	distanceSquared = None
	if(getTrainConnectionsUseSpatialDistance(sequenceObservedColumns)):
		if(not pt.is_tensor(sourceConceptIndices) or not pt.is_tensor(targetConceptIndices)):
			raise RuntimeError("calculateSparseFeatureConnectionsSpatialDistanceTensor error: sourceConceptIndices/targetConceptIndices must be tensors")
		if(sourceConceptIndices.shape != targetConceptIndices.shape):
			raise RuntimeError("calculateSparseFeatureConnectionsSpatialDistanceTensor error: sourceConceptIndices/targetConceptIndices shape mismatch")
		fieldXTensor, fieldYTensor = getSequenceConceptFieldCoordinates(sequenceObservedColumns, targetDevice)
		deltaX = pt.abs(fieldXTensor[targetConceptIndices] - fieldXTensor[sourceConceptIndices])
		deltaY = pt.abs(fieldYTensor[targetConceptIndices] - fieldYTensor[sourceConceptIndices])
		distanceSquared = (deltaX*deltaX + deltaY*deltaY).to(arrayType)
		result = pt.ceil(pt.sqrt(distanceSquared)).long()
		if(bool(pt.any(result < 0).item()) or bool(pt.any(result >= arrayNumberOfSegments).item())):
			raise RuntimeError("calculateSparseFeatureConnectionsSpatialDistanceTensor error: calculated segment index out of range")
	else:
		raise RuntimeError("calculateSparseFeatureConnectionsSpatialDistanceTensor error: requires trainConnectionsUseSpatialDistance")
	return result


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
