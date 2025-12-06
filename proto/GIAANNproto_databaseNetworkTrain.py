"""GIAANNproto_databaseNetworkTrain.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

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

from GIAANNproto_globalDefs import *
import GIAANNproto_sparseTensors
import GIAANNproto_sequenceConcepts

def trainConceptWords(sequenceObservedColumns, sequenceIndex, sequence, tokens):
	conceptIndices, startIndices, endIndices = GIAANNproto_sequenceConcepts.processConceptWords(sequenceObservedColumns, sequenceIndex, sequence, tokens)
	featureNeuronsActive, cs, fs, sequenceConceptIndexMask, columnsWordOrder, featureNeuronsWordOrder, featureNeuronsPos, featureNeuronsSegmentMask = GIAANNproto_sequenceConcepts.processFeatures(sequenceObservedColumns, sequenceIndex, sequence, tokens, conceptIndices, startIndices, endIndices)

	processFeaturesActiveTrain(sequenceObservedColumns, featureNeuronsActive, cs, fs, sequenceConceptIndexMask, columnsWordOrder, featureNeuronsWordOrder, featureNeuronsPos, featureNeuronsSegmentMask, sequenceIndex)

#first dim cs1 pertains to every concept node in sequence
def processFeaturesActiveTrain(sequenceObservedColumns, featureNeuronsActive, cs, fs, sequenceConceptIndexMask, columnsWordOrder, featureNeuronsWordOrder, featureNeuronsPos, featureNeuronsSegmentMask, sequenceIndex):
	featureNeuronsInactive = 1 - featureNeuronsActive
		
	sequenceObservedColumns.featureNeurons[arrayIndexPropertiesStrength, :, :, :] += featureNeuronsActive
	sequenceObservedColumns.featureNeurons[arrayIndexPropertiesPermanence, :, :, :] += featureNeuronsActive*z1
	sequenceObservedColumns.featureNeurons[arrayIndexPropertiesActivation, :, :, :] = 0
	if(inferenceUseNeuronFeaturePropertiesTime):
		sequenceObservedColumns.featureNeurons[arrayIndexPropertiesTime, :, :, :] = 0
	else:
		sequenceObservedColumns.featureNeurons[arrayIndexPropertiesTime, :, :, :] = featureNeuronsInactive*sequenceObservedColumns.featureNeurons[arrayIndexPropertiesTime] + featureNeuronsActive*sequenceIndex
	sequenceObservedColumns.featureNeurons[arrayIndexPropertiesPos, :, :, :] = featureNeuronsInactive*sequenceObservedColumns.featureNeurons[arrayIndexPropertiesPos] + featureNeuronsActive*featureNeuronsPos

	featureConnectionsActive, featureConnectionsSegmentMask = createFeatureConnectionsActiveTrain(featureNeuronsActive[arrayIndexSegmentInternalColumn], cs, fs, columnsWordOrder, featureNeuronsWordOrder)
	
	if(debugPrintNeuronActivations8):
		featureConnectionsActive = featureConnectionsActive*1000
	
	featureConnectionsPos = featureNeuronsPos.view(1, cs, fs, 1, 1).expand(arrayNumberOfSegments, cs, fs, cs, fs)

	featureConnectionsInactive = 1 - featureConnectionsActive

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
		csIndices1 = pt.arange(cs).view(1, cs, 1, 1, 1).expand(arrayNumberOfSegments, cs, fs, cs, fs)
		csIndices2 = pt.arange(cs).view(1, 1, 1, cs, 1).expand(arrayNumberOfSegments, cs, fs, cs, fs)

	if(trainConnectionStrengthIncreaseColumnInternal):
		columnInternalConnectionsMask = (csIndices1 == csIndices2)
		columnInternalConnectionsMaskOff = pt.logical_not(columnInternalConnectionsMask)
		featureConnectionsStrengthUpdate = columnInternalConnectionsMask.float()*featureConnectionsStrengthUpdate*trainIncreaseColumnInternalConnectionsStrengthModifier + columnInternalConnectionsMaskOff.float()*featureConnectionsStrengthUpdate

	if(trainConnectionStrengthPOSdependence):
		featureConnectionsStrengthUpdate = applyConnectionStrengthPOSdependenceTrain(sequenceObservedColumns, featureConnectionsStrengthUpdate, featureConnectionsPos, csIndices1, csIndices2)

	sequenceObservedColumns.featureConnections[arrayIndexPropertiesStrength, :, :, :, :, :] += featureConnectionsStrengthUpdate
	sequenceObservedColumns.featureConnections[arrayIndexPropertiesPermanence, :, :, :, :, :] += featureConnectionsActive*z1
	sequenceObservedColumns.featureConnections[arrayIndexPropertiesActivation, :, :, :, :, :] = 0
	if(inferenceUseNeuronFeaturePropertiesTime):
		sequenceObservedColumns.featureConnections[arrayIndexPropertiesTime, :, :, :, :, :] = 0
	else:
		sequenceObservedColumns.featureConnections[arrayIndexPropertiesTime, :, :, :, :, :] = featureConnectionsInactive*sequenceObservedColumns.featureConnections[arrayIndexPropertiesTime] + featureConnectionsActive*sequenceIndex
	sequenceObservedColumns.featureConnections[arrayIndexPropertiesPos, :, :, :, :, :] = featureConnectionsInactive*sequenceObservedColumns.featureConnections[arrayIndexPropertiesPos] + featureConnectionsActive*featureConnectionsPos
	if(arrayIndexPropertiesMinWordDistance):
		updateConnectionMinWordDistances(sequenceObservedColumns, featureConnectionsActive, featureNeuronsWordOrder)

	if(trainDecreasePermanenceOfInactiveFeatureNeuronsAndConnections):
		decreasePermanenceActive(sequenceObservedColumns, featureNeuronsActive[arrayIndexSegmentInternalColumn], featureNeuronsInactive[arrayIndexSegmentInternalColumn], sequenceConceptIndexMask, featureNeuronsSegmentMask, featureConnectionsSegmentMask)
	

def updateConnectionMinWordDistances(sequenceObservedColumns, featureConnectionsActive, featureNeuronsWordOrder):
	if(featureNeuronsWordOrder is None):
		return
	cs = sequenceObservedColumns.cs
	fs = sequenceObservedColumns.fs
	device = featureConnectionsActive.device
	wordOrderTensor = featureNeuronsWordOrder.to(device)
	wordOrderSource = wordOrderTensor.view(cs, fs, 1, 1).expand(cs, fs, cs, fs)
	wordOrderTarget = wordOrderTensor.view(1, 1, cs, fs).expand(cs, fs, cs, fs)
	wordDistances = pt.abs(wordOrderTarget - wordOrderSource).to(arrayType)
	wordDistances = wordDistances.unsqueeze(0).expand(arrayNumberOfSegments, cs, fs, cs, fs)
	wordDistances = wordDistances * featureConnectionsActive
	updateMask = featureConnectionsActive > 0
	validDistanceMask = updateMask & (wordDistances > 0)
	connectionMinDistances = sequenceObservedColumns.featureConnections[arrayIndexPropertiesMinWordDistanceIndex]
	firstWriteMask = validDistanceMask & (connectionMinDistances == 0)
	connectionMinDistances = pt.where(firstWriteMask, wordDistances, connectionMinDistances)
	smallerMask = validDistanceMask & (connectionMinDistances > 0) & (wordDistances < connectionMinDistances)
	connectionMinDistances = pt.where(smallerMask, wordDistances, connectionMinDistances)
	sequenceObservedColumns.featureConnections[arrayIndexPropertiesMinWordDistanceIndex] = connectionMinDistances


def createFeatureConnectionsActiveTrain(featureNeuronsActive, cs, fs, columnsWordOrder, featureNeuronsWordOrder):

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
	featureConnectionsActive = featureConnectionsActive * identityMask

	if(useSANI):
		featureConnectionsActive, featureConnectionsSegmentMask = assignFeatureConnectionsToTargetSegments(featureConnectionsActive, cs, fs)
	else:
		featureConnectionsActive = featureConnectionsActive.unsqueeze(0)
		featureConnectionsSegmentMask = pt.ones_like(featureConnectionsActive)
	
	return featureConnectionsActive, featureConnectionsSegmentMask

def assignFeatureConnectionsToTargetSegments(featureConnectionsActive, cs, fs):

	conceptNeuronsConceptOrder1d = pt.arange(cs)
	conceptNeuronsDistances = pt.abs(conceptNeuronsConceptOrder1d.unsqueeze(1) - conceptNeuronsConceptOrder1d).reshape(cs, cs)
	connectionsSegmentIndex = arrayNumberOfSegments-conceptNeuronsDistances-1
	connectionsSegmentIndex = pt.clamp(connectionsSegmentIndex, min=0)
	
	featureConnectionsSegmentMask = pt.zeros((arrayNumberOfSegments, cs, cs), dtype=pt.bool)
	featureConnectionsSegmentMask = featureConnectionsSegmentMask.scatter_(0, connectionsSegmentIndex.unsqueeze(0), True)
	featureConnectionsSegmentMask = featureConnectionsSegmentMask.view(arrayNumberOfSegments, cs, 1, cs, 1).expand(arrayNumberOfSegments, cs, fs, cs, fs)
	
	featureConnectionsActive = featureConnectionsSegmentMask * featureConnectionsActive.unsqueeze(0)
	
	return featureConnectionsActive, featureConnectionsSegmentMask
		
def decreasePermanenceActive(sequenceObservedColumns, featureNeuronsActive, featureNeuronsInactive, sequenceConceptIndexMask, featureNeuronsSegmentMask, featureConnectionsSegmentMask):

	if(trainSequenceObservedColumnsMatchSequenceWords):
		featureNeuronsInactive = featureNeuronsInactive*sequenceConceptIndexMask
	
	cs = sequenceObservedColumns.cs
	fs = sequenceObservedColumns.fs 
	
	featureNeuronsDecrease = featureNeuronsInactive.unsqueeze(0)*z2 * featureNeuronsSegmentMask.unsqueeze(2)
	sequenceObservedColumns.featureNeurons[arrayIndexPropertiesPermanence, :, :, :] -= featureNeuronsDecrease
	sequenceObservedColumns.featureNeurons[arrayIndexPropertiesPermanence] = pt.clamp(sequenceObservedColumns.featureNeurons[arrayIndexPropertiesPermanence], min=0)

	featureNeuronsAll = pt.ones((cs, fs), dtype=arrayType)
	featureNeuronsAll1d = featureNeuronsAll.view(cs*fs)
	featureNeuronsActive1d = featureNeuronsActive.view(cs*fs)
	featureNeuronsInactive1d = featureNeuronsInactive.view(cs*fs)
	 
	featureConnectionsDecrease1 = pt.matmul(featureNeuronsInactive1d.unsqueeze(1), featureNeuronsAll1d.unsqueeze(0)).view(cs, fs, cs, fs)
	featureConnectionsDecrease1 = featureConnectionsDecrease1.unsqueeze(0)*featureConnectionsSegmentMask
	sequenceObservedColumns.featureConnections[arrayIndexPropertiesPermanence, :, :, :, :, :] -= featureConnectionsDecrease1
	sequenceObservedColumns.featureConnections[arrayIndexPropertiesPermanence] = pt.clamp(sequenceObservedColumns.featureConnections[arrayIndexPropertiesPermanence], min=0)
	
	featureConnectionsDecrease2 = pt.matmul(featureNeuronsActive1d.unsqueeze(1), featureNeuronsInactive1d.unsqueeze(0)).view(cs, fs, cs, fs)
	featureConnectionsDecrease2 = featureConnectionsDecrease2.unsqueeze(0)*featureConnectionsSegmentMask
	sequenceObservedColumns.featureConnections[arrayIndexPropertiesPermanence, :, :, :, :, :] -= featureConnectionsDecrease2
	sequenceObservedColumns.featureConnections[arrayIndexPropertiesPermanence] = pt.clamp(sequenceObservedColumns.featureConnections[arrayIndexPropertiesPermanence], min=0)
 
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

def getConnectionStrengthPOSdependenceLookup(databaseNetworkObject):
	if not hasattr(databaseNetworkObject, "connectionStrengthPOSdependenceLookup"):
		posLookup = []
		for posType, value in zip(connectionStrengthPOSdependenceTypes, connectionStrengthPOSdependenceValues):
			posIndex = posStringToPosInt(databaseNetworkObject.nlp, posType)
			posLookup.append((posIndex, float(value)))
		databaseNetworkObject.connectionStrengthPOSdependenceLookup = posLookup
	return databaseNetworkObject.connectionStrengthPOSdependenceLookup


