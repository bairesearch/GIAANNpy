"""GIAANNproto_predictionConstraints.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_main.py

# Usage:
see GIAANNproto_main.py

# Description:
GIA ANN proto prediction constraint helpers

"""

import torch as pt

from GIAANNproto_globalDefs import *
import GIAANNproto_databaseNetworkExcitation
import GIAANNproto_sparseTensors
import GIAANNproto_predictionActivate

def buildAllowedColumnsSet(allowedColumnsTensor):
	allowedSet = None
	if(allowedColumnsTensor is not None and allowedColumnsTensor.numel() > 0):
		columnList = allowedColumnsTensor.cpu().tolist()
		if(len(columnList) > 0):
			allowedSet = set(columnList)
	return allowedSet

def createConstraintState(allowedColumnsTensor, constraintMode):
	constraintState = None
	if(constraintMode in ["internal", "external", "delimiter"]):
		allowedSet = buildAllowedColumnsSet(allowedColumnsTensor)
		if(allowedSet is not None and len(allowedSet) > 0):
			constraintState = {"columns": allowedSet, "mode": constraintMode}
	return constraintState

def constraintAllowsColumn(columnIndex, constraintState):
	allowed = True
	if(constraintState is not None):
		allowedSet = constraintState.get("columns")
		constraintMode = constraintState.get("mode")
		if(allowedSet is not None and constraintMode is not None and len(allowedSet) > 0):
			if(constraintMode == "internal"):
				allowed = columnIndex in allowedSet
			elif(constraintMode == "external"):
				allowed = columnIndex not in allowedSet
			elif(constraintMode == "delimiter"):
				allowed = True
	return allowed

def constraintAllowsNode(databaseNetworkObject, columnIndex, featureIndex, constraintState):
	allowed = constraintAllowsColumn(columnIndex, constraintState)
	if(allowed and constraintState is not None):
		allowedSet = constraintState.get("columns")
		constraintMode = constraintState.get("mode")
		if(constraintMode == "delimiter" and allowedSet is not None and columnIndex in allowedSet):
			if(featureIndex is None):
				allowed = False
			else:
				isDeterministicDelimiter = GIAANNproto_databaseNetworkExcitation.isFeatureIndexReferenceSetDelimiterDeterministic(databaseNetworkObject, int(featureIndex))
				isProbabilisticDelimiter = GIAANNproto_databaseNetworkExcitation.isFeatureIndexReferenceSetDelimiterProbabilistic(databaseNetworkObject, int(featureIndex))
				allowed = (isDeterministicDelimiter or isProbabilisticDelimiter)
	return allowed

def filterColumnFeatureCandidatesByConstraint(databaseNetworkObject, columnIndices, featureIndices, activationValues, constraintState):
	filteredColumns = columnIndices
	filteredFeatures = featureIndices
	filteredValues = activationValues
	if(constraintState is not None):
		if(columnIndices is None or featureIndices is None or activationValues is None):
			filteredColumns = None
			filteredFeatures = None
			filteredValues = None
		else:
			indicesToKeep = []
			for idx in range(columnIndices.shape[0]):
				columnIndex = int(columnIndices[idx].item())
				featureIndex = int(featureIndices[idx].item())
				if(constraintAllowsNode(databaseNetworkObject, columnIndex, featureIndex, constraintState)):
					indicesToKeep.append(idx)
			if(len(indicesToKeep) > 0):
				indexTensor = pt.tensor(indicesToKeep, dtype=pt.long, device=columnIndices.device)
				filteredColumns = columnIndices.index_select(0, indexTensor)
				filteredFeatures = featureIndices.index_select(0, indexTensor)
				filteredValues = activationValues.index_select(0, indexTensor)
			else:
				filteredColumns = None
				filteredFeatures = None
				filteredValues = None
	return filteredColumns, filteredFeatures, filteredValues


def filterColumnFeatureCandidatesByConnectedColumns(columnIndices, featureIndices, activationValues, connectedColumnsTensor, connectedColumnsFeatures=None):
	if(connectedColumnsTensor is None):
		return columnIndices, featureIndices, activationValues
	if(columnIndices is None or featureIndices is None or activationValues is None):
		return None, None, None
	if(connectedColumnsTensor.numel() == 0):
		return None, None, None
	if(columnIndices.numel() == 0):
		return None, None, None
	device = columnIndices.device
	connectedColumnsDevice = connectedColumnsTensor.to(device)
	if(connectedColumnsDevice.numel() == 0):
		return None, None, None
	allowedColumnsSet = set(connectedColumnsDevice.tolist())
	selectedIndices = []
	if(connectedColumnsFeatures is not None and len(connectedColumnsFeatures) > 0):
		for idx in range(columnIndices.shape[0]):
			columnValue = int(columnIndices[idx].item())
			if(columnValue not in allowedColumnsSet):
				continue
			featureAllowed = connectedColumnsFeatures.get(columnValue)
			if(featureAllowed is None or len(featureAllowed) == 0):
				continue
			featureValue = int(featureIndices[idx].item())
			if(featureValue not in featureAllowed):
				continue
			selectedIndices.append(idx)
	else:
		comparison = columnIndices.unsqueeze(0) == connectedColumnsDevice.unsqueeze(1)
		mask = comparison.any(dim=0)
		selectedIndices = pt.nonzero(mask, as_tuple=False).view(-1).tolist()
	if(len(selectedIndices) == 0):
		return None, None, None
	indexTensor = pt.tensor(selectedIndices, dtype=pt.long, device=columnIndices.device)
	return columnIndices.index_select(0, indexTensor), featureIndices.index_select(0, indexTensor), activationValues.index_select(0, indexTensor)

def aggregateSparseColumnFeatureValues(sparseTensor, maxFeatures):
	aggregatedColumns = None
	aggregatedFeatures = None
	aggregatedValues = None
	if(sparseTensor is not None):
		workingTensor = sparseTensor
		if(workingTensor.dim() == 4):
			if(multipleDendriticBranches):
				if(workingTensor.is_sparse):
					workingTensor = GIAANNproto_sparseTensors.reduceSparseBranchMax(workingTensor)
				else:
					workingTensor = workingTensor.max(dim=0).values
			else:
				workingTensor = workingTensor.sum(dim=0)
		if(workingTensor.dim() not in [2, 3]):
			raise RuntimeError("aggregateSparseColumnFeatureValues: unsupported tensor dimensions")
		if(not workingTensor.is_sparse):
			workingTensor = workingTensor.to_sparse()
		workingTensor = workingTensor.coalesce()
		if(workingTensor._nnz() > 0):
			indices = workingTensor.indices()
			values = workingTensor.values()
			if(workingTensor.dim() == 2):
				columnIndices = indices[0]
				featureIndices = indices[1]
			else:
				columnIndices = indices[1]
				featureIndices = indices[2]
			keys = columnIndices * maxFeatures + featureIndices
			uniqueKeys, inverseIndices = pt.unique(keys, return_inverse=True)
			aggregatedValues = pt.zeros((uniqueKeys.shape[0],), dtype=values.dtype, device=values.device)
			aggregatedValues.scatter_add_(0, inverseIndices, values)
			aggregatedColumns = uniqueKeys // maxFeatures
			aggregatedFeatures = uniqueKeys % maxFeatures
	return aggregatedColumns, aggregatedFeatures, aggregatedValues


def clearPredictionTensors(conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, connectedColumns=None):
	conceptColumnsIndicesOut = conceptColumnsIndicesPred
	conceptColumnsFeatureIndicesOut = conceptColumnsFeatureIndicesPred
	if(conceptColumnsIndicesPred is not None):
		conceptColumnsIndicesOut = conceptColumnsIndicesPred[:0]
	elif(connectedColumns is not None):
		conceptColumnsIndicesOut = connectedColumns[:0]
	else:
		conceptColumnsIndicesOut = pt.empty(0, dtype=pt.long)
	if(conceptColumnsFeatureIndicesPred is not None):
		conceptColumnsFeatureIndicesOut = conceptColumnsFeatureIndicesPred[:0]
	return conceptColumnsIndicesOut, conceptColumnsFeatureIndicesOut


def rowHasAllowedFeature(conceptColumnsFeatureIndicesPred, rowIndex, allowedFeaturesSet):
	allowed = False
	if(conceptColumnsFeatureIndicesPred is not None):
		if(conceptColumnsFeatureIndicesPred.dim() == 1):
			if(rowIndex < conceptColumnsFeatureIndicesPred.shape[0]):
				featureValue = int(conceptColumnsFeatureIndicesPred[rowIndex].item())
				if(featureValue == featureIndexPrimeConceptNeuron or featureValue in allowedFeaturesSet):
					allowed = True
		else:
			if(rowIndex < conceptColumnsFeatureIndicesPred.shape[0]):
				rowTensor = conceptColumnsFeatureIndicesPred[rowIndex]
				if(rowTensor is not None and rowTensor.numel() > 0):
					rowValues = rowTensor.view(-1)
					for value in rowValues:
						featureValue = int(value.item())
						if(featureValue == featureIndexPrimeConceptNeuron or featureValue in allowedFeaturesSet):
							allowed = True
	return allowed


def applyConnectedColumnsConstraint(conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, connectedColumns, connectedColumnsFeatures=None):
	conceptColumnsIndicesOut = conceptColumnsIndicesPred
	conceptColumnsFeatureIndicesOut = conceptColumnsFeatureIndicesPred
	if(connectedColumns is not None):
		if(connectedColumns.numel() == 0):
			conceptColumnsIndicesOut, conceptColumnsFeatureIndicesOut = clearPredictionTensors(conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, connectedColumns)
		elif(conceptColumnsIndicesPred is None or conceptColumnsIndicesPred.numel() == 0):
			conceptColumnsIndicesOut = conceptColumnsIndicesPred
			conceptColumnsFeatureIndicesOut = conceptColumnsFeatureIndicesPred
		else:
			connectedList = connectedColumns.view(-1).tolist()
			if(len(connectedList) == 0):
				conceptColumnsIndicesOut, conceptColumnsFeatureIndicesOut = clearPredictionTensors(conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, connectedColumns)
			else:
				allowedSet = set(connectedList)
				if(len(allowedSet) == 0):
					conceptColumnsIndicesOut, conceptColumnsFeatureIndicesOut = clearPredictionTensors(conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, connectedColumns)
				else:
					predictedColumnsList = conceptColumnsIndicesPred.cpu().tolist()
					indicesToKeep = []
					for idx, columnValue in enumerate(predictedColumnsList):
						if(columnValue not in allowedSet):
							continue
						if(connectedColumnsFeatures is not None):
							allowedFeatureSet = connectedColumnsFeatures.get(int(columnValue))
							if(allowedFeatureSet is None or len(allowedFeatureSet) == 0):
								continue
							if(not rowHasAllowedFeature(conceptColumnsFeatureIndicesPred, idx, allowedFeatureSet)):
								continue
						indicesToKeep.append(idx)
					if(len(indicesToKeep) == 0):
						conceptColumnsIndicesOut, conceptColumnsFeatureIndicesOut = clearPredictionTensors(conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, connectedColumns)
					else:
						indexTensor = pt.tensor(indicesToKeep, dtype=pt.long, device=conceptColumnsIndicesPred.device)
						conceptColumnsIndicesOut = conceptColumnsIndicesPred.index_select(0, indexTensor)
						if(conceptColumnsFeatureIndicesPred is not None and conceptColumnsFeatureIndicesPred.shape[0] >= indexTensor.shape[0]):
							conceptColumnsFeatureIndicesOut = conceptColumnsFeatureIndicesPred.index_select(0, indexTensor)
	return conceptColumnsIndicesOut, conceptColumnsFeatureIndicesOut


def getObservedColumn(databaseNetworkObject, observedColumnsDict, columnIndex):
	result = None
	if(columnIndex < 0 or columnIndex >= len(databaseNetworkObject.conceptColumnsList)):
		result = None
	else:
		columnLemma = databaseNetworkObject.conceptColumnsList[columnIndex]
		observedColumn = observedColumnsDict.get(columnLemma)
		if(observedColumn is None):
			observedColumn = GIAANNproto_databaseNetworkExcitation.loadOrCreateObservedColumn(databaseNetworkObject, columnIndex, columnLemma, columnIndex)
		clearObservedColumns = inferenceOnlyRetainPredictedTargetObservedColumn
		if(clearObservedColumns and inferenceBeamSearch and not inferenceOnlyRetainPredictedTargetObservedColumnBeamSearch):
			clearObservedColumns = False
		if(clearObservedColumns):
			if(observedColumnsDict is None):
				raise RuntimeError("getObservedColumn error: observedColumnsDict is None")
			observedColumnsDict.clear()
		observedColumnsDict[columnLemma] = observedColumn
		result = observedColumn
	return result


def getConnectedColumnsForFeature(observedColumn, featureIndex, includeFeatureDetails=False):
	if(featureIndex is None or featureIndex < 0):
		return [], {} if includeFeatureDetails else None
	featureConnectionsStrength = observedColumn.featureConnections[arrayIndexPropertiesStrengthIndex]
	featureConnectionsStrength = GIAANNproto_sparseTensors.sliceSparseTensor(featureConnectionsStrength, 2, featureIndex)
	featureConnectionsStrength = featureConnectionsStrength.coalesce()
	if(featureConnectionsStrength._nnz() == 0):
		return [], {} if includeFeatureDetails else None
	targetColumnIndices = featureConnectionsStrength.indices()
	if(targetColumnIndices.shape[1] == 0):
		return [], {} if includeFeatureDetails else None
	minWordDistanceMask = GIAANNproto_predictionActivate.computeConnectionMinWordDistanceMask(observedColumn, featureIndex, targetColumnIndices)
	if(minWordDistanceMask is not None):
		if(minWordDistanceMask.sum().item() == 0):
			return [], {} if includeFeatureDetails else None
		targetColumnIndices = targetColumnIndices[:, minWordDistanceMask]
	elif(enforceDirectConnections and enforceDirectConnectionsSANI):
		lastSegmentMask = targetColumnIndices[1] == arrayIndexSegmentLast
		targetColumnIndices = targetColumnIndices[:, lastSegmentMask]
	targetColumns = targetColumnIndices[2].unique()
	targetColumnsList = targetColumns.cpu().tolist()
	if(includeFeatureDetails):
		targetFeatures = targetColumnIndices[3].cpu().tolist()
		columnFeatureMap = {}
		for columnValue, featureValue in zip(targetColumnIndices[2].tolist(), targetFeatures):
			columnFeatureMap.setdefault(columnValue, set()).add(featureValue)
		return targetColumnsList, columnFeatureMap
	else:
		return targetColumnsList, None


def buildConnectedColumnsLookup(databaseNetworkObject, observedColumnsDict, columnFeaturePairs, device, dtype):
	if(columnFeaturePairs is None or len(columnFeaturePairs) == 0):
		return None, None
	connectedColumnsSet = set()
	if(debugConnectNodesToNextNodesInSequenceOnly or enforceDirectConnectionsMinWordDistance or enforceDirectConnectionsSANI):
		connectedColumnsFeatures = {}
	else:
		connectedColumnsFeatures = None
	for columnIndex, featureIndex in columnFeaturePairs:
		observedColumn = getObservedColumn(databaseNetworkObject, observedColumnsDict, columnIndex)
		if(observedColumn is None):
			continue
		targetColumns, targetColumnFeatures = getConnectedColumnsForFeature(observedColumn, featureIndex, includeFeatureDetails=(connectedColumnsFeatures is not None))
		connectedColumnsSet.update(targetColumns)
		if(connectedColumnsFeatures is not None and targetColumnFeatures is not None):
			for targetColumnIndex, featureSet in targetColumnFeatures.items():
				if(targetColumnIndex < 0 or targetColumnIndex >= databaseNetworkObject.c):
					continue
				columnFeatureSet = connectedColumnsFeatures.setdefault(targetColumnIndex, set())
				columnFeatureSet.update(featureSet)
	if(len(connectedColumnsSet) == 0):
		emptyTensor = pt.empty(0, dtype=dtype, device=device)
		return emptyTensor, ({} if connectedColumnsFeatures is not None else None)
	validColumns = [col for col in connectedColumnsSet if col >= 0 and col < databaseNetworkObject.c]
	if(len(validColumns) == 0):
		emptyTensor = pt.empty(0, dtype=dtype, device=device)
		return emptyTensor, ({} if connectedColumnsFeatures is not None else None)
	validColumns.sort()
	connectedTensor = pt.tensor(validColumns, dtype=dtype, device=device)
	if(connectedColumnsFeatures is not None):
		filteredFeatureMap = {}
		for columnIndex in validColumns:
			featureSet = connectedColumnsFeatures.get(columnIndex, set())
			if(len(featureSet) > 0):
				filteredFeatureMap[columnIndex] = set(featureSet)
		return connectedTensor, filteredFeatureMap
	else:
		return connectedTensor, None


def buildConnectedColumnsLookupFromPrediction(databaseNetworkObject, observedColumnsDict, conceptColumnsIndices, conceptColumnsFeatureIndices):
	if(not predictionEnsureConnectedToPreviousPrediction and not enforceDirectConnectionsMinWordDistance):
		return None, None
	if(conceptColumnsIndices is None or conceptColumnsFeatureIndices is None):
		return None, None
	if(conceptColumnsIndices.numel() == 0 or conceptColumnsFeatureIndices.numel() == 0):
		device = conceptColumnsIndices.device
		dtype = conceptColumnsIndices.dtype
		return pt.empty(0, dtype=dtype, device=device), None
	columnFeaturePairs = []
	for rowIndex in range(conceptColumnsIndices.shape[0]):
		columnIndex = int(conceptColumnsIndices[rowIndex].item())
		rowTensor = conceptColumnsFeatureIndices[rowIndex]
		if(rowTensor is None or rowTensor.numel() == 0):
			continue
		featureValues = rowTensor.detach().view(-1).tolist()
		for featureValue in featureValues:
			columnFeaturePairs.append((columnIndex, int(featureValue)))
	device = conceptColumnsIndices.device
	dtype = conceptColumnsIndices.dtype
	return buildConnectedColumnsLookup(databaseNetworkObject, observedColumnsDict, columnFeaturePairs, device, dtype)


def buildConnectedColumnsLookupForBeamNodes(databaseNetworkObject, observedColumnsDict, nodes):
	if(nodes is None or len(nodes) == 0):
		return None, None
	return buildConnectedColumnsLookup(databaseNetworkObject, observedColumnsDict, nodes, deviceSparse, pt.long)

def getFirstFeatureValue(conceptColumnsFeatureIndicesPred, rowIndex):
	featureValue = None
	if(conceptColumnsFeatureIndicesPred is not None):
		if(conceptColumnsFeatureIndicesPred.dim() == 1):
			if(rowIndex < conceptColumnsFeatureIndicesPred.shape[0]):
				featureValue = conceptColumnsFeatureIndicesPred[rowIndex].item()
		else:
			if(rowIndex < conceptColumnsFeatureIndicesPred.shape[0]):
				rowTensor = conceptColumnsFeatureIndicesPred[rowIndex]
				if(rowTensor.numel() > 0):
					featureValue = rowTensor.view(-1)[0].item()
	return featureValue


def applyColumnConstraintToPredictions(databaseNetworkObject, conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, allowedColumns, constraintMode, connectedColumns=None, connectedColumnsFeatures=None):
	conceptColumnsIndicesOut = conceptColumnsIndicesPred
	conceptColumnsFeatureIndicesOut = conceptColumnsFeatureIndicesPred
	applyConnectedConstraint = True
	if(conceptColumnsIndicesPred is not None and constraintMode is not None):
		if(allowedColumns is None or allowedColumns.numel() == 0):
			applyConnectedConstraint = True
		else:
			allowedSet = set(allowedColumns.cpu().tolist())
			if(len(allowedSet) == 0):
				applyConnectedConstraint = True
			else:
				predictedColumnsList = conceptColumnsIndicesPred.cpu().tolist()
				constraintState = createConstraintState(allowedColumns, constraintMode)
				if(constraintMode == "internal"):
					indicesToKeep = [idx for idx, columnValue in enumerate(predictedColumnsList) if constraintAllowsColumn(columnValue, constraintState)]
					if(len(indicesToKeep) == 0):
						repeatFactor = max(1, (conceptColumnsIndicesPred.shape[0] + allowedColumns.shape[0] - 1) // allowedColumns.shape[0])
						replacementColumns = allowedColumns.repeat(repeatFactor)[:conceptColumnsIndicesPred.shape[0]]
						conceptColumnsIndicesOut = replacementColumns
						applyConnectedConstraint = False
					else:
						indexTensor = pt.tensor(indicesToKeep, dtype=pt.long, device=conceptColumnsIndicesPred.device)
						conceptColumnsIndicesOut = conceptColumnsIndicesPred.index_select(0, indexTensor)
						if(conceptColumnsFeatureIndicesPred is not None and conceptColumnsFeatureIndicesPred.shape[0] >= indexTensor.shape[0]):
							conceptColumnsFeatureIndicesOut = conceptColumnsFeatureIndicesPred.index_select(0, indexTensor)
				elif(constraintMode == "external"):
					indicesToKeep = [idx for idx, columnValue in enumerate(predictedColumnsList) if constraintAllowsColumn(columnValue, constraintState)]
					if(len(indicesToKeep) > 0):
						indexTensor = pt.tensor(indicesToKeep, dtype=pt.long, device=conceptColumnsIndicesPred.device)
						conceptColumnsIndicesOut = conceptColumnsIndicesPred.index_select(0, indexTensor)
						if(conceptColumnsFeatureIndicesPred is not None and conceptColumnsFeatureIndicesPred.shape[0] >= indexTensor.shape[0]):
							conceptColumnsFeatureIndicesOut = conceptColumnsFeatureIndicesPred.index_select(0, indexTensor)
						applyConnectedConstraint = False
					else:
						if(len(allowedSet) >= databaseNetworkObject.c):
							raise RuntimeError("applyColumnConstraintToPredictions: external constraint requires columns outside allowed set, but no columns are available.")
						raise RuntimeError("applyColumnConstraintToPredictions: external constraint removed all predicted columns; no eligible external predictions remain.")
				elif(constraintMode == "delimiter"):
					debugDelimiterCandidates = []
					indicesToKeep = []
					for idx, columnValue in enumerate(predictedColumnsList):
						featureValue = getFirstFeatureValue(conceptColumnsFeatureIndicesPred, idx)
						allowedCandidate = constraintAllowsNode(databaseNetworkObject, columnValue, featureValue, constraintState)
						if(allowedCandidate):
							indicesToKeep.append(idx)
					if(len(indicesToKeep) == 0):
						raise RuntimeError("applyColumnConstraintToPredictions: delimiter constraint removed all predictions; unable to find external/delimiter-compatible columns.")
					indexTensor = pt.tensor(indicesToKeep, dtype=pt.long, device=conceptColumnsIndicesPred.device)
					conceptColumnsIndicesOut = conceptColumnsIndicesPred.index_select(0, indexTensor)
					if(conceptColumnsFeatureIndicesPred is not None and conceptColumnsFeatureIndicesPred.shape[0] >= indexTensor.shape[0]):
						conceptColumnsFeatureIndicesOut = conceptColumnsFeatureIndicesPred.index_select(0, indexTensor)
	if(applyConnectedConstraint):
		conceptColumnsIndicesOut, conceptColumnsFeatureIndicesOut = applyConnectedColumnsConstraint(conceptColumnsIndicesOut, conceptColumnsFeatureIndicesOut, connectedColumns, connectedColumnsFeatures)
	return conceptColumnsIndicesOut, conceptColumnsFeatureIndicesOut


class InferenceStopSequenceNoPredictionCandidatesAvailable(Exception):
	pass

def raiseOrStopPredictionConnectivityError(sequenceWordIndex, wordPredictionIndex, tokensSequence, reason):
	if(debugTerminateInferenceOnNoPredictionCandidatesAvailable):
		raisePredictionConnectivityError(sequenceWordIndex, wordPredictionIndex, tokensSequence, reason)
	targetWord = getTargetWordForSequenceIndex(tokensSequence, sequenceWordIndex)
	message = f"predictionEnsureConnectedToPreviousPrediction violation: {reason}. sequenceWordIndex={sequenceWordIndex}, wordPredictionIndex={wordPredictionIndex}, targetWord='{targetWord}'"
	print(message)
	raise InferenceStopSequenceNoPredictionCandidatesAvailable(message)

def raisePredictionConnectivityError(sequenceWordIndex, wordPredictionIndex, tokensSequence, reason):
	targetWord = getTargetWordForSequenceIndex(tokensSequence, sequenceWordIndex)
	message = f"predictionEnsureConnectedToPreviousPrediction violation: {reason}. sequenceWordIndex={sequenceWordIndex}, wordPredictionIndex={wordPredictionIndex}, targetWord='{targetWord}'"
	raise RuntimeError(message)

def getTargetWordForSequenceIndex(tokensSequence, sequenceWordIndex):
	if(tokensSequence is None):
		return "<unknown>"
	if(sequenceWordIndex < 0 or sequenceWordIndex >= len(tokensSequence)):
		return "<unknown>"
	return tokensSequence[sequenceWordIndex].word
