"""GIAANNproto_databaseNetworkTrainInhibition.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_main.py

# Usage:
see GIAANNproto_main.py

# Description:
GIA ANN proto database Network Inhibition

"""

import torch as pt

from GIAANNproto_globalDefs import *
import GIAANNproto_sequenceObservedColumnsInhibition
import GIAANNproto_databaseNetworkExcitation
import GIAANNproto_sparseTensors

def processFeaturesInactiveTrain(sequenceObservedColumns, featureNeuronsActive, cs, fs, sequenceConceptIndexMask, columnsWordOrder, featureNeuronsWordOrder, featureNeuronsPos, featureNeuronsSegmentMask, sequenceIndex, featureConnectionsActive, featureConnectionsSegmentMask):
	if(featureConnectionsActive is None):
		return None
	connectionMask = featureConnectionsActive > 0
	if(connectionMask.ndim == 4):
		activeConnections = connectionMask
	else:
		activeConnections = pt.any(connectionMask, dim=0)
	if(not pt.any(activeConnections)):
		return None

	directConnectionTargetsMask = None
	directConnectionSegmentsMask = None
	if(enforceDirectConnections):
		directConnectionTargetsMask, directConnectionSegmentsMask = identifyDirectConnectionCandidates(sequenceObservedColumns)
		if(directConnectionTargetsMask is None or directConnectionSegmentsMask is None):
			return None

	featureNeuronsActiveInhibitory = featureNeuronsActive.clone()
	featureNeuronsWordOrderInhibitory = featureNeuronsWordOrder.clone()
	featureNeuronsPosInhibitory = featureNeuronsPos.clone()
	inhibitionBuffer = GIAANNproto_sequenceObservedColumnsInhibition.createSequenceBuffer(sequenceObservedColumns)
	activatedInhibitoryMap = set()
	sequenceObservedInhibitoryNeurons = []

	activeConnectionIndices = pt.nonzero(activeConnections, as_tuple=False)
	if(activeConnectionIndices.numel() == 0):
		return None
	activeConnectionIndexList = activeConnectionIndices.tolist()

	previousSourceMap = None
	if(featureNeuronsWordOrder is not None or columnsWordOrder is not None):
		previousSourceMap = {}
		for connectionIndex in activeConnectionIndexList:
			sourceColumnIndex = int(connectionIndex[0])
			sourceFeatureIndex = int(connectionIndex[1])
			targetColumnIndex = int(connectionIndex[2])
			targetFeatureIndex = int(connectionIndex[3])
			sourceOrder = getFeatureWordOrderValue(featureNeuronsWordOrder, columnsWordOrder, sourceColumnIndex, sourceFeatureIndex)
			targetOrder = getFeatureWordOrderValue(featureNeuronsWordOrder, columnsWordOrder, targetColumnIndex, targetFeatureIndex)
			if(sourceOrder is None or targetOrder is None or targetOrder <= sourceOrder):
				continue
			key = (targetColumnIndex, targetFeatureIndex)
			entry = previousSourceMap.get(key)
			if(entry is None or sourceOrder > entry["order"]):
				previousSourceMap[key] = {"order": sourceOrder, "sources": {(sourceColumnIndex, sourceFeatureIndex)}}
			elif(sourceOrder == entry["order"]):
				entry["sources"].add((sourceColumnIndex, sourceFeatureIndex))

	conceptIndicesTensor = getattr(sequenceObservedColumns, "conceptIndicesInSequenceObservedTensor", None)
	featureIndicesTensor = getattr(sequenceObservedColumns, "featureIndicesInObservedTensor", None)
	globalToLocalColumnMap = buildGlobalToLocalColumnMap(conceptIndicesTensor)
	observedColumnCache = {}
	for connectionIndex in activeConnectionIndexList:
		sourceColumnIndex = int(connectionIndex[0])
		sourceFeatureIndex = int(connectionIndex[1])
		targetColumnIndex = int(connectionIndex[2])
		targetFeatureIndex = int(connectionIndex[3])
		globalTargetColumnIndex = getGlobalColumnIndex(conceptIndicesTensor, targetColumnIndex)
		globalTargetFeatureIndex = getGlobalFeatureIndex(featureIndicesTensor, targetFeatureIndex)
		globalSourceColumnIndex = getGlobalColumnIndex(conceptIndicesTensor, sourceColumnIndex)
		globalSourceFeatureIndex = getGlobalFeatureIndex(featureIndicesTensor, sourceFeatureIndex)

		if(enforceDirectConnections):
			if(sourceColumnIndex >= directConnectionTargetsMask.shape[0] or sourceFeatureIndex >= directConnectionTargetsMask.shape[1]):
				printe("processFeaturesInactiveTrain error: sourceColumnIndex >= directConnectionTargetsMask.shape[0] or sourceFeatureIndex >= directConnectionTargetsMask.shape[1]")
			candidateMask = directConnectionTargetsMask[sourceColumnIndex, sourceFeatureIndex].clone().to(dtype=pt.bool)
		else:
			candidateMask = activeConnections[sourceColumnIndex, sourceFeatureIndex].clone().to(dtype=pt.bool)
		if(targetColumnIndex < candidateMask.shape[0] and targetFeatureIndex < candidateMask.shape[1]):
			candidateMask[targetColumnIndex, targetFeatureIndex] = False

		inhibitoryFeatureIndex = selectInhibitoryNeuron(sequenceObservedColumns, globalTargetColumnIndex, globalTargetFeatureIndex, targetFeatureIndex)

		key = (globalTargetColumnIndex, globalTargetFeatureIndex)
		if(key not in activatedInhibitoryMap):
			activateInhibitoryNeuron(featureNeuronsActiveInhibitory, featureNeuronsWordOrderInhibitory, featureNeuronsPosInhibitory, featureNeuronsActive, featureNeuronsWordOrder, featureNeuronsPos, targetColumnIndex, targetFeatureIndex, inhibitoryFeatureIndex)
			activatedInhibitoryMap.add(key)

		connectionSegmentsMask = extractConnectionSegmentsMask(featureConnectionsSegmentMask, sourceColumnIndex, sourceFeatureIndex, targetColumnIndex, targetFeatureIndex)
		sourcePosValue = float(featureNeuronsPos[targetColumnIndex, targetFeatureIndex].item()) if featureNeuronsPos is not None else 0.0
		sequenceObservedInhibitoryNeurons.append({
			"sourceColumnIndex": sourceColumnIndex,
			"sourceFeatureIndex": sourceFeatureIndex,
			"targetColumnIndex": targetColumnIndex,
			"targetFeatureIndex": targetFeatureIndex,
			"targetColumnConceptIndex": globalTargetColumnIndex,
			"targetFeatureConceptIndex": globalTargetFeatureIndex,
			"inhibitoryFeatureIndex": inhibitoryFeatureIndex,
			"segmentsMask": connectionSegmentsMask,
			"sourcePosValue": sourcePosValue
		})

		if(previousSourceMap is not None):
			prevEntry = previousSourceMap.get((targetColumnIndex, targetFeatureIndex))
			if(prevEntry is not None and (sourceColumnIndex, sourceFeatureIndex) not in prevEntry["sources"]):
				continue

		processedCandidateSet = set()
		candidateIndices = pt.nonzero(candidateMask, as_tuple=False)
		if(candidateIndices.numel() > 0):
			if(enforceDirectConnections):
				sourceSegmentsTensor = directConnectionSegmentsMask
			else:
				sourceSegmentsTensor = featureConnectionsActive
				for candidateIndex in candidateIndices:
					candidateColumnIndex, candidateFeatureIndex = candidateIndex.tolist()
					globalCandidateColumnIndex = getGlobalColumnIndex(conceptIndicesTensor, candidateColumnIndex)
					globalCandidateFeatureIndex = getGlobalFeatureIndex(featureIndicesTensor, candidateFeatureIndex)
					if(globalCandidateColumnIndex == globalTargetColumnIndex and globalCandidateFeatureIndex == globalTargetFeatureIndex):
						continue	# only wire inhibitory outputs to alternate prediction candidates
					processedCandidateSet.add((globalCandidateColumnIndex, globalCandidateFeatureIndex))
					if(sourceSegmentsTensor is None):
						segmentsMask = None
					else:
						if(sourceColumnIndex >= sourceSegmentsTensor.shape[1] or sourceFeatureIndex >= sourceSegmentsTensor.shape[2]):
							printe("processFeaturesInactiveTrain error: sourceColumnIndex >= sourceSegmentsTensor.shape[1] or sourceFeatureIndex >= sourceSegmentsTensor.shape[2]")
						segmentsMask = sourceSegmentsTensor[:, sourceColumnIndex, sourceFeatureIndex, candidateColumnIndex, candidateFeatureIndex]
					updateInhibitoryConnection(inhibitionBuffer, targetColumnIndex, inhibitoryFeatureIndex, candidateColumnIndex, candidateFeatureIndex, segmentsMask, sequenceIndex, sourcePosValue)

		globalAlternateCandidates = collectGlobalAlternateCandidates(sequenceObservedColumns, observedColumnCache, globalSourceColumnIndex, globalSourceFeatureIndex)
		if(globalAlternateCandidates):
			for candidateEntry in globalAlternateCandidates:
				globalCandidateColumnIndex = candidateEntry["targetColumnIndex"]
				globalCandidateFeatureIndex = candidateEntry["targetFeatureIndex"]
				if(globalCandidateColumnIndex == globalTargetColumnIndex and globalCandidateFeatureIndex == globalTargetFeatureIndex):
					continue
				candidateKey = (globalCandidateColumnIndex, globalCandidateFeatureIndex)
				if(candidateKey in processedCandidateSet):
					continue
				localCandidateColumnIndex = globalToLocalColumnMap.get(globalCandidateColumnIndex)
				if(localCandidateColumnIndex is not None):
					localCandidateFeatureIndex = getLocalFeatureIndex(sequenceObservedColumns, localCandidateColumnIndex, globalCandidateFeatureIndex)
					if(localCandidateFeatureIndex is not None):
						segmentsMask = candidateEntry["segmentsMask"]
						updateInhibitoryConnection(inhibitionBuffer, targetColumnIndex, inhibitoryFeatureIndex, localCandidateColumnIndex, localCandidateFeatureIndex, segmentsMask, sequenceIndex, sourcePosValue)
						processedCandidateSet.add(candidateKey)
						continue
				recordExternalInhibitoryConnection(inhibitionBuffer, globalTargetColumnIndex, inhibitoryFeatureIndex, globalCandidateColumnIndex, globalCandidateFeatureIndex, candidateEntry["segmentsMask"], sequenceIndex, sourcePosValue)
				processedCandidateSet.add(candidateKey)

	if(len(sequenceObservedInhibitoryNeurons) == 0):
		return None

	applyInhibitoryConnectionStrengthLimits(inhibitionBuffer)
	return featureNeuronsActiveInhibitory, featureNeuronsWordOrderInhibitory, featureNeuronsPosInhibitory, inhibitionBuffer, sequenceObservedInhibitoryNeurons

def activateInhibitoryNeuron(featureNeuronsActiveInhibitory, featureNeuronsWordOrderInhibitory, featureNeuronsPosInhibitory, featureNeuronsActive, featureNeuronsWordOrder, featureNeuronsPos, columnIndex, baseFeatureIndex, inhibitoryFeatureIndex):
	segmentActivation = featureNeuronsActive[:, columnIndex, baseFeatureIndex]
	if(not pt.any(segmentActivation)):
		segmentActivation = pt.zeros_like(segmentActivation)
		segmentActivation[arrayIndexSegmentFirst] = 1
	else:
		segmentActivation = segmentActivation.clone()
	featureNeuronsActiveInhibitory[:, columnIndex, inhibitoryFeatureIndex] = segmentActivation
	featureNeuronsWordOrderInhibitory[columnIndex, inhibitoryFeatureIndex] = featureNeuronsWordOrder[columnIndex, baseFeatureIndex]
	featureNeuronsPosInhibitory[columnIndex, inhibitoryFeatureIndex] = featureNeuronsPos[columnIndex, baseFeatureIndex]

def updateInhibitoryConnection(inhibitionBuffer, inhibitoryColumnIndex, inhibitoryFeatureIndex, candidateColumnIndex, candidateFeatureIndex, segmentsMask, sequenceIndex, sourcePosValue):
	if(segmentsMask is None or not pt.any(segmentsMask)):
		segmentsMask = pt.zeros(arrayNumberOfSegments, dtype=arrayType)
		segmentsMask[arrayIndexSegmentFirst] = 1
	activeSegmentIndices = pt.nonzero(segmentsMask, as_tuple=False)
	if(activeSegmentIndices.numel() == 0):
		activeSegmentIndices = pt.tensor([[arrayIndexSegmentFirst]], dtype=pt.long)
	for segmentTensor in activeSegmentIndices:
		segmentIndex = int(segmentTensor.item())
		if(arrayIndexPropertiesStrength):
			strengthIndex = (arrayIndexPropertiesStrengthIndex, segmentIndex, inhibitoryColumnIndex, inhibitoryFeatureIndex, candidateColumnIndex, candidateFeatureIndex)
			inhibitionBuffer.featureConnections[strengthIndex] += inhibitoryConnectionStrengthIncrement
		if(arrayIndexPropertiesPermanence):
			permanenceIndex = (arrayIndexPropertiesPermanenceIndex, segmentIndex, inhibitoryColumnIndex, inhibitoryFeatureIndex, candidateColumnIndex, candidateFeatureIndex)
			inhibitionBuffer.featureConnections[permanenceIndex] += z1
		if(arrayIndexPropertiesActivation):
			activationIndex = (arrayIndexPropertiesActivationIndex, segmentIndex, inhibitoryColumnIndex, inhibitoryFeatureIndex, candidateColumnIndex, candidateFeatureIndex)
			inhibitionBuffer.featureConnections[activationIndex] = 0
		if(arrayIndexPropertiesTime):
			timeIndex = (arrayIndexPropertiesTimeIndex, segmentIndex, inhibitoryColumnIndex, inhibitoryFeatureIndex, candidateColumnIndex, candidateFeatureIndex)
			if(inferenceUseNeuronFeaturePropertiesTime):
				inhibitionBuffer.featureConnections[timeIndex] = 0
			else:
				inhibitionBuffer.featureConnections[timeIndex] = sequenceIndex
		if(arrayIndexPropertiesPos):
			posIndex = (arrayIndexPropertiesPosIndex, segmentIndex, inhibitoryColumnIndex, inhibitoryFeatureIndex, candidateColumnIndex, candidateFeatureIndex)
			inhibitionBuffer.featureConnections[posIndex] = sourcePosValue
	#segmentsList = activeSegmentIndices.squeeze(1).tolist() if activeSegmentIndices.dim() > 1 else [int(activeSegmentIndices.item())]
	#print(f"inhibitory output connection created: column {inhibitoryColumnIndex}, feature {inhibitoryFeatureIndex} -> column {candidateColumnIndex}, feature {candidateFeatureIndex}, segments {segmentsList}")

def applyInhibitoryConnectionStrengthLimits(inhibitionBuffer):
	if(arrayIndexPropertiesStrength):
		if(trainConnectionStrengthLimitMax):
			inhibitionBuffer.featureConnections[arrayIndexPropertiesStrengthIndex] = inhibitionBuffer.featureConnections[arrayIndexPropertiesStrengthIndex].clamp(max=1.0)
		if(trainConnectionStrengthLimitTanh):
			inhibitionBuffer.featureConnections[arrayIndexPropertiesStrengthIndex] = pt.tanh(inhibitionBuffer.featureConnections[arrayIndexPropertiesStrengthIndex])

def extractConnectionSegmentsMask(featureConnectionsSegmentMask, sourceColumnIndex, sourceFeatureIndex, targetColumnIndex, targetFeatureIndex):
	if(featureConnectionsSegmentMask is None):
		return None
	if(featureConnectionsSegmentMask.ndim == 4):
		return None
	maskSlice = featureConnectionsSegmentMask[:, sourceColumnIndex, sourceFeatureIndex, targetColumnIndex, targetFeatureIndex]
	if(maskSlice.numel() == 0):
		return None
	return maskSlice.clone().to(dtype=pt.bool, device=maskSlice.device)

def selectInhibitoryNeuron(sequenceObservedColumns, globalColumnIndex, globalFeatureIndex, localFeatureIndex):
	associationSet = getInhibitoryAssociationSet(sequenceObservedColumns.databaseNetworkObject)
	associationKey = (globalColumnIndex, globalFeatureIndex)
	if(associationKey not in associationSet):
		associationSet.add(associationKey)
	return localFeatureIndex

def getInhibitoryAssociationSet(databaseNetworkObject):
	if(not hasattr(databaseNetworkObject, "inhibitoryNeuronAssociations")):
		databaseNetworkObject.inhibitoryNeuronAssociations = set()
	return databaseNetworkObject.inhibitoryNeuronAssociations

def getGlobalColumnIndex(conceptIndicesTensor, localColumnIndex):
	if(conceptIndicesTensor is None):
		return localColumnIndex
	if(localColumnIndex < 0 or localColumnIndex >= conceptIndicesTensor.shape[0]):
		return localColumnIndex
	return int(conceptIndicesTensor[localColumnIndex].item())

def getGlobalFeatureIndex(featureIndicesTensor, localFeatureIndex):
	if(featureIndicesTensor is None):
		return localFeatureIndex
	if(localFeatureIndex < 0 or localFeatureIndex >= featureIndicesTensor.shape[0]):
		return localFeatureIndex
	return int(featureIndicesTensor[localFeatureIndex].item())

def getFeatureWordOrderValue(featureNeuronsWordOrder, columnsWordOrder, columnIndex, featureIndex):
	if(featureNeuronsWordOrder is not None):
		if(columnIndex >= 0 and columnIndex < featureNeuronsWordOrder.shape[0] and featureIndex >= 0 and featureIndex < featureNeuronsWordOrder.shape[1]):
			return int(featureNeuronsWordOrder[columnIndex, featureIndex].item())
	if(columnsWordOrder is not None):
		if(columnIndex >= 0 and columnIndex < columnsWordOrder.shape[0]):
			return int(columnsWordOrder[columnIndex].item())
	return None

def buildGlobalToLocalColumnMap(conceptIndicesTensor):
	columnMap = {}
	if(conceptIndicesTensor is None):
		return columnMap
	for localIndex in range(conceptIndicesTensor.shape[0]):
		globalIndex = int(conceptIndicesTensor[localIndex].item())
		columnMap[globalIndex] = localIndex
	return columnMap

def getLocalFeatureIndex(sequenceObservedColumns, localColumnIndex, globalFeatureIndex):
	columnFeatureMaps = getattr(sequenceObservedColumns, "columnFeatureLocalIndices", None)
	if(columnFeatureMaps is None):
		return None
	if(localColumnIndex < 0 or localColumnIndex >= len(columnFeatureMaps)):
		return None
	columnMap = columnFeatureMaps[localColumnIndex]
	if(columnMap is None):
		return None
	localEntries = columnMap.get(globalFeatureIndex)
	if(localEntries is None or len(localEntries) == 0):
		return None
	return localEntries[0]

def collectGlobalAlternateCandidates(sequenceObservedColumns, observedColumnCache, globalSourceColumnIndex, globalSourceFeatureIndex):
	if(globalSourceColumnIndex is None or globalSourceFeatureIndex is None):
		return []
	observedColumn = getObservedColumnForConcept(sequenceObservedColumns, observedColumnCache, globalSourceColumnIndex)
	if(observedColumn is None):
		return []
	strengthTensor = observedColumn.featureConnections[arrayIndexPropertiesStrengthIndex]
	strengthSlice = GIAANNproto_sparseTensors.sliceSparseTensor(strengthTensor, 1, globalSourceFeatureIndex)
	strengthSlice = strengthSlice.coalesce()
	if(strengthSlice._nnz() == 0):
		return []
	segmentsDict = {}
	indices = strengthSlice.indices()
	for idx in range(indices.shape[1]):
		segmentIndex = int(indices[0, idx].item())
		targetColumnIndex = int(indices[1, idx].item())
		targetFeatureIndex = int(indices[2, idx].item())
		key = (targetColumnIndex, targetFeatureIndex)
		if(key not in segmentsDict):
			segmentsDict[key] = pt.zeros(arrayNumberOfSegments, dtype=pt.bool)
		segmentsDict[key][segmentIndex] = True
	if(enforceDirectConnectionsMinWordDistance and arrayIndexPropertiesMinWordDistance):
		minDistanceLookup = buildMinDistanceLookup(observedColumn, globalSourceFeatureIndex)
	else:
		minDistanceLookup = None
	candidates = []
	for key, segmentsMask in segmentsDict.items():
		if(not connectionSatisfiesGlobalDirectConstraint(segmentsMask, key, minDistanceLookup)):
			continue
		candidates.append({
			"targetColumnIndex": key[0],
			"targetFeatureIndex": key[1],
			"segmentsMask": segmentsMask.clone().to(dtype=pt.bool)
		})
	return candidates

def getObservedColumnForConcept(sequenceObservedColumns, observedColumnCache, globalColumnIndex):
	if(globalColumnIndex in observedColumnCache):
		return observedColumnCache[globalColumnIndex]
	if(globalColumnIndex < 0 or globalColumnIndex >= sequenceObservedColumns.databaseNetworkObject.c):
		return None
	lemma = sequenceObservedColumns.databaseNetworkObject.conceptColumnsList[globalColumnIndex]
	observedColumn = sequenceObservedColumns.observedColumnsDict.get(lemma)
	if(observedColumn is None):
		observedColumn = GIAANNproto_databaseNetworkExcitation.loadOrCreateObservedColumn(sequenceObservedColumns.databaseNetworkObject, globalColumnIndex, lemma, globalColumnIndex)
	if(observedColumn is not None):
		observedColumnCache[globalColumnIndex] = observedColumn
	return observedColumn

def buildMinDistanceLookup(observedColumn, sourceFeatureIndex):
	minDistances = {}
	minTensor = observedColumn.featureConnections[arrayIndexPropertiesMinWordDistanceIndex]
	minSlice = GIAANNproto_sparseTensors.sliceSparseTensor(minTensor, 1, sourceFeatureIndex)
	minSlice = minSlice.coalesce()
	if(minSlice._nnz() == 0):
		return minDistances
	indices = minSlice.indices()
	values = minSlice.values()
	for idx in range(indices.shape[1]):
		columnIndex = int(indices[1, idx].item())
		featureIndex = int(indices[2, idx].item())
		value = float(values[idx].item())
		key = (columnIndex, featureIndex)
		if(key not in minDistances or value < minDistances[key]):
			minDistances[key] = value
	return minDistances

def connectionSatisfiesGlobalDirectConstraint(segmentsMask, key, minDistanceLookup):
	if(not enforceDirectConnections):
		return True
	if(enforceDirectConnectionsMinWordDistance and arrayIndexPropertiesMinWordDistance):
		if(minDistanceLookup is None):
			return False
		distance = minDistanceLookup.get(key)
		return (distance is not None) and (distance > 0) and (abs(distance - 1.0) < 1e-4)
	if(enforceDirectConnectionsSANI):
		lastSegmentIndex = max(0, min(arrayIndexSegmentLast, arrayNumberOfSegments-1))
		return bool(segmentsMask[lastSegmentIndex].item())
	return True

def recordExternalInhibitoryConnection(inhibitionBuffer, inhibitoryColumnConceptIndex, inhibitoryFeatureIndex, candidateColumnConceptIndex, candidateFeatureConceptIndex, segmentsMask, sequenceIndex, sourcePosValue):
	if(segmentsMask is None or not pt.any(segmentsMask)):
		maskTensor = pt.zeros(arrayNumberOfSegments, dtype=pt.bool)
		maskTensor[arrayIndexSegmentFirst] = True
	else:
		maskTensor = segmentsMask.clone().to(dtype=pt.bool)
	updateEntry = {
		"targetColumnConceptIndex": inhibitoryColumnConceptIndex,
		"inhibitoryFeatureIndex": inhibitoryFeatureIndex,
		"candidateColumnConceptIndex": candidateColumnConceptIndex,
		"candidateFeatureConceptIndex": candidateFeatureConceptIndex,
		"segmentsMask": maskTensor.cpu(),
		"sequenceIndex": sequenceIndex,
		"sourcePosValue": sourcePosValue
	}
	inhibitionBuffer.externalOutputUpdates.append(updateEntry)

# Identify direct source->target pairs that satisfy enforceDirectConnections.
def identifyDirectConnectionCandidates(sequenceObservedColumns):
	featureConnections = getattr(sequenceObservedColumns, "featureConnections", None)
	if(featureConnections is None):
		#print("identifyDirectConnectionCandidates error: featureConnections unavailable in sequenceObservedColumns")
		return None, None
	directSegmentsMask = None

	if(enforceDirectConnectionsMinWordDistance and arrayIndexPropertiesMinWordDistance):
		minDistances = featureConnections[arrayIndexPropertiesMinWordDistanceIndex]
		if(minDistances is None):
			#print("identifyDirectConnectionCandidates error: min word distance tensor unavailable")
			return None, None
		distanceMask = (minDistances > 0) & (pt.abs(minDistances - 1.0) < 1e-4)
		if not pt.any(distanceMask):
			#print("identifyDirectConnectionCandidates error: no direct connections recorded for min word distance enforcement")
			return None, None
		directSegmentsMask = distanceMask.to(dtype=pt.bool)
	elif(enforceDirectConnectionsSANI and arrayNumberOfSegments > 0):
		strengthTensor = featureConnections[arrayIndexPropertiesStrengthIndex]
		if(strengthTensor is None):
			#print("identifyDirectConnectionCandidates error: strength tensor unavailable for SANI enforcement")
			return None, None
		directSegmentsMask = pt.zeros_like(strengthTensor, dtype=pt.bool)
		lastSegmentIndex = max(0, min(arrayIndexSegmentLast, arrayNumberOfSegments-1))
		directSegmentsMask[lastSegmentIndex] = strengthTensor[lastSegmentIndex] > 0
		if not pt.any(directSegmentsMask):
			#print("identifyDirectConnectionCandidates error: no direct SANI connections recorded")
			return None, None
	else:
		#print("identifyDirectConnectionCandidates error: enforceDirectConnections requires enforceDirectConnectionsMinWordDistance or enforceDirectConnectionsSANI")
		return None, None

	directTargetsMask = pt.any(directSegmentsMask, dim=0)
	if(not pt.any(directTargetsMask)):
		#print("identifyDirectConnectionCandidates error: direct connection target mask empty")
		return None, None

	return directTargetsMask.to(dtype=pt.bool), directSegmentsMask
