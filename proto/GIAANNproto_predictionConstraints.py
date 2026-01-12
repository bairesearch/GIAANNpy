"""GIAANNproto_predictionConstraints.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_predictionConstraints.py

# Usage:
see GIAANNproto_predictionConstraints.py

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
				if(featureValue == featureIndexConceptNeuron or featureValue in allowedFeaturesSet):
					allowed = True
		else:
			if(rowIndex < conceptColumnsFeatureIndicesPred.shape[0]):
				rowTensor = conceptColumnsFeatureIndicesPred[rowIndex]
				if(rowTensor is not None and rowTensor.numel() > 0):
					rowValues = rowTensor.view(-1)
					for value in rowValues:
						featureValue = int(value.item())
						if(featureValue == featureIndexConceptNeuron or featureValue in allowedFeaturesSet):
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
	if(columnIndex < 0 or columnIndex >= len(databaseNetworkObject.conceptColumnsList)):
		return None
	columnLemma = databaseNetworkObject.conceptColumnsList[columnIndex]
	if(columnLemma in observedColumnsDict):
		return observedColumnsDict[columnLemma]
	observedColumn = GIAANNproto_databaseNetworkExcitation.loadOrCreateObservedColumn(databaseNetworkObject, columnIndex, columnLemma, columnIndex)
	observedColumnsDict[columnLemma] = observedColumn
	return observedColumn


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


def enforceMinimumPredictionActivationThreshold(conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, activationValues):
	conceptColumnsIndicesOut = conceptColumnsIndicesPred
	conceptColumnsFeatureIndicesOut = conceptColumnsFeatureIndicesPred
	if(minimumPredictionActivationThreshold > 0 and conceptColumnsIndicesPred is not None and conceptColumnsIndicesPred.numel() > 0 and activationValues is not None):
		if(activationValues.numel() == 0):
			conceptColumnsIndicesOut = conceptColumnsIndicesPred[:0]
			if(conceptColumnsFeatureIndicesPred is not None):
				conceptColumnsFeatureIndicesOut = conceptColumnsFeatureIndicesPred[:0]
		else:
			if(activationValues.dim() == 1):
				activeMask = activationValues >= minimumPredictionActivationThreshold
			else:
				activeMask = (activationValues >= minimumPredictionActivationThreshold).all(dim=1)
			if(activeMask.sum().item() == 0):
				conceptColumnsIndicesOut = conceptColumnsIndicesPred[:0]
				if(conceptColumnsFeatureIndicesPred is not None):
					conceptColumnsFeatureIndicesOut = conceptColumnsFeatureIndicesPred[:0]
			else:
				indexTensor = pt.nonzero(activeMask, as_tuple=False).view(-1)
				conceptColumnsIndicesOut = conceptColumnsIndicesPred.index_select(0, indexTensor)
				if(conceptColumnsFeatureIndicesPred is not None and conceptColumnsFeatureIndicesPred.shape[0] > 0):
					conceptColumnsFeatureIndicesOut = conceptColumnsFeatureIndicesPred.index_select(0, indexTensor)
	return conceptColumnsIndicesOut, conceptColumnsFeatureIndicesOut


def buildSparseColumnFeatureTensor(columnIndices, featureIndices, values, columnsCount, featuresCount, device):
	sparseTensor = None
	if(columnIndices is not None and featureIndices is not None and values is not None):
		if(columnIndices.numel() > 0 and values.numel() > 0):
			indices = pt.stack([columnIndices, featureIndices], dim=0)
			sparseTensor = pt.sparse_coo_tensor(indices, values, size=(columnsCount, featuresCount), device=device).coalesce()
	return sparseTensor


def selectMostActiveFeature(sequenceObservedColumns, globalFeatureNeuronsActivation, globalFeatureNeuronsStrength, globalFeatureNeuronsTime, tokensSequence, wordPredictionIndex, sequenceWordIndex, conceptMask, allowedColumns=None, constraintMode=None, conceptActivationState=None, connectedColumns=None, connectedColumnsFeatures=None):
	databaseNetworkObject = sequenceObservedColumns.databaseNetworkObject
	#generate targets;
	targetMultipleSources, targetPreviousColumnIndex, targetNextColumnIndex, targetFeatureIndex, targetConceptColumnsIndices, targetConceptColumnsFeatureIndices = GIAANNproto_databaseNetworkExcitation.getTokenConceptFeatureIndexTensor(sequenceObservedColumns, tokensSequence, conceptMask, sequenceWordIndex, kcNetwork)

	globalFeatureNeuronsActivationSelection = globalFeatureNeuronsActivation
	if(inferenceUseNeuronFeaturePropertiesTime):
		sequenceColumnIndex = None
		if(useSANIcolumns or useSANIfeaturesAndColumns):
			sequenceColumnIndex = GIAANNproto_predictionActivate.calculateSequenceColumnIndex(conceptMask, sequenceWordIndex)
		# spec step (b): apply time-based activation modifier during feature selection
		globalFeatureNeuronsActivationSelection = GIAANNproto_predictionActivate.applyTimeBasedActivationModifier(globalFeatureNeuronsActivationSelection, globalFeatureNeuronsTime, sequenceWordIndex, sequenceColumnIndex)

	globalFeatureNeuronsActivationAllSegments = globalFeatureNeuronsActivationSelection.sum(dim=1)	#sum across all segments
	if(multipleDendriticBranches):
		if(globalFeatureNeuronsActivationAllSegments.is_sparse):
			globalFeatureNeuronsActivationAllSegments = GIAANNproto_sparseTensors.reduceSparseBranchMax(globalFeatureNeuronsActivationAllSegments)
		else:
			globalFeatureNeuronsActivationAllSegments = globalFeatureNeuronsActivationAllSegments.max(dim=0).values
	else:
		globalFeatureNeuronsActivationAllSegments = globalFeatureNeuronsActivationAllSegments.sum(dim=0)	#sum across all branches
	globalFeatureNeuronsStrengthAllSegments = globalFeatureNeuronsStrength.sum(dim=1)	#sum across all segments
	if(multipleDendriticBranches):
		if(globalFeatureNeuronsStrengthAllSegments.is_sparse):
			globalFeatureNeuronsStrengthAllSegments = GIAANNproto_sparseTensors.reduceSparseBranchMax(globalFeatureNeuronsStrengthAllSegments)
		else:
			globalFeatureNeuronsStrengthAllSegments = globalFeatureNeuronsStrengthAllSegments.max(dim=0).values
	else:
		globalFeatureNeuronsStrengthAllSegments = globalFeatureNeuronsStrengthAllSegments.sum(dim=0)	#sum across all branches
	if(useSANI and algorithmMatrixSANImethod=="enforceActivationAcrossSegments" and algorithmMatrixSANIenforceRequirement=="enforceLastSegmentMustBeActive"):
		# Patch: selection ignored last-segment gating, allowing nodes without last-segment activation to fire.
		if(enforceActivationAcrossSegmentsIgnoreInternalColumn):
			lastSegmentConstraint = arrayIndexSegmentAdjacentColumn
		else:
			lastSegmentConstraint = arrayIndexSegmentLast
		hasBranchDim = (globalFeatureNeuronsActivationSelection.dim() == 4)
		if(globalFeatureNeuronsActivationSelection.is_sparse):
			if(hasBranchDim):
				lastSegmentActivation = GIAANNproto_sparseTensors.sliceSparseTensor(globalFeatureNeuronsActivationSelection, 1, lastSegmentConstraint)
			else:
				lastSegmentActivation = GIAANNproto_sparseTensors.sliceSparseTensor(globalFeatureNeuronsActivationSelection, 0, lastSegmentConstraint)
		else:
			if(hasBranchDim):
				lastSegmentActivation = globalFeatureNeuronsActivationSelection[:, lastSegmentConstraint]
			else:
				lastSegmentActivation = globalFeatureNeuronsActivationSelection[lastSegmentConstraint]
		if(globalFeatureNeuronsActivationAllSegments.is_sparse):
			if(multipleDendriticBranches and lastSegmentActivation.dim() == 3):
				lastSegmentActivationCollapsed = GIAANNproto_sparseTensors.reduceSparseBranchMax(lastSegmentActivation)
			else:
				lastSegmentActivationCollapsed = GIAANNproto_sparseTensors.collapseSparseBranchDimension(lastSegmentActivation)
			globalFeatureNeuronsActivationAllSegments = GIAANNproto_sparseTensors.selectAindicesContainedInB(globalFeatureNeuronsActivationAllSegments, lastSegmentActivationCollapsed)
			if(globalFeatureNeuronsActivationAllSegments._nnz() == 0):
				raise RuntimeError("selectMostActiveFeature error: enforceLastSegmentMustBeActive requires active last-segment nodes, but none are active.")
		else:
			lastSegmentMask = (lastSegmentActivation.to_dense() > 0).any(dim=0)
			globalFeatureNeuronsActivationAllSegments = globalFeatureNeuronsActivationAllSegments * lastSegmentMask
			if(not (globalFeatureNeuronsActivationAllSegments > 0).any().item()):
				raise RuntimeError("selectMostActiveFeature error: enforceLastSegmentMustBeActive requires active last-segment nodes, but none are active.")

	if(conceptColumnsDelimitByPOS and constraintMode == "delimiter"):
		constraintState = createConstraintState(allowedColumns, constraintMode)
		if(constraintState is not None):
			columnIndices, featureIndices, activationValues = aggregateSparseColumnFeatureValues(globalFeatureNeuronsActivationAllSegments, databaseNetworkObject.f)
			columnIndices, featureIndices, activationValues = filterColumnFeatureCandidatesByConstraint(databaseNetworkObject, columnIndices, featureIndices, activationValues, constraintState)
			if(columnIndices is None or featureIndices is None or activationValues is None or activationValues.numel() == 0):
				raise RuntimeError("applyColumnConstraintToPredictions: delimiter constraint removed all predictions; unable to find external/delimiter-compatible columns.")
			filteredActivationTensor = buildSparseColumnFeatureTensor(columnIndices, featureIndices, activationValues, databaseNetworkObject.c, databaseNetworkObject.f, globalFeatureNeuronsActivationAllSegments.device)
			if(filteredActivationTensor is None):
				raise RuntimeError("selectMostActiveFeature: delimiter constraint produced no activation candidates")
			globalFeatureNeuronsActivationAllSegments = filteredActivationTensor
	if(connectedColumns is not None):
		columnIndices, featureIndices, activationValues = aggregateSparseColumnFeatureValues(globalFeatureNeuronsActivationAllSegments, databaseNetworkObject.f)
		columnIndices, featureIndices, activationValues = filterColumnFeatureCandidatesByConnectedColumns(columnIndices, featureIndices, activationValues, connectedColumns, connectedColumnsFeatures)
		if(columnIndices is None or featureIndices is None or activationValues is None or activationValues.numel() == 0):
			raise RuntimeError("selectMostActiveFeature: connected columns constraint removed all predictions.")
		filteredActivationTensor = buildSparseColumnFeatureTensor(columnIndices, featureIndices, activationValues, databaseNetworkObject.c, databaseNetworkObject.f, globalFeatureNeuronsActivationAllSegments.device)
		if(filteredActivationTensor is None):
			raise RuntimeError("selectMostActiveFeature: connected columns constraint produced no activation candidates")
		globalFeatureNeuronsActivationAllSegments = filteredActivationTensor

	#topk column selection;
	conceptColumnsActivation = pt.sum(globalFeatureNeuronsActivationAllSegments, dim=1)	#sum across all feature activations in columns
	conceptColumnsActivation = conceptColumnsActivation.to_dense()	#convert to dense tensor (required for topk)
	if(inferenceNormaliseColumnSelectionByFeatureConnections):
		conceptColumnsActivationTotalConnections = pt.sum(globalFeatureNeuronsStrengthAllSegments, dim=1)	#sum across all feature activations in columns
		conceptColumnsActivationTotalConnections = conceptColumnsActivationTotalConnections.to_dense()
		if(not inferenceNormaliseColumnSelectionByFeatureConnectionsStrength):
			conceptColumnsActivationTotalConnections = (conceptColumnsActivationTotalConnections > 0).float()
		conceptColumnsActivation = conceptColumnsActivation / conceptColumnsActivationTotalConnections

	if(conceptColumnsDelimitByPOS):
		columnIndexLookup = pt.arange(conceptColumnsActivation.shape[0], dtype=pt.long, device=conceptColumnsActivation.device)
		if(allowedColumns is not None and allowedColumns.numel() > 0):
			allowedColumnsDevice = allowedColumns.to(columnIndexLookup.device)
		else:
			allowedColumnsDevice = None
		if(allowedColumnsDevice is not None and constraintMode is not None):
			if(constraintMode == "internal"):
				columnIndexLookup = allowedColumnsDevice
				columnActivationValues = conceptColumnsActivation.index_select(0, columnIndexLookup)
			elif(constraintMode == "external"):
				mask = pt.ones(conceptColumnsActivation.shape[0], dtype=pt.bool, device=conceptColumnsActivation.device)
				mask.scatter_(0, allowedColumnsDevice, False)
				filteredIndices = columnIndexLookup[mask]
				filteredActivations = conceptColumnsActivation[mask]
				if(filteredIndices.numel() > 0):
					columnIndexLookup = filteredIndices
					columnActivationValues = filteredActivations
				else:
					columnActivationValues = conceptColumnsActivation
			else:
				columnActivationValues = conceptColumnsActivation
		else:
			columnActivationValues = conceptColumnsActivation
		if(columnIndexLookup.numel() == 0):
			columnIndexLookup = pt.arange(conceptColumnsActivation.shape[0], dtype=pt.long, device=conceptColumnsActivation.device)
			columnActivationValues = conceptColumnsActivation
		if(kcDynamic):
			activeMask = columnActivationValues > kcActivationThreshold
			columnActivationValues = columnActivationValues[activeMask]
			columnIndexLookup = columnIndexLookup[activeMask]
		kcAvailable = columnActivationValues.shape[0]
		if(kcDynamic and kcAvailable < 1):
			print("selectMostActiveFeature kcDynamic error: kc < 1; cannot continue to predict columns; consider disabling kcDynamic for debug")
			exit()
		if(kcAvailable == 0):
			columnIndexLookup = pt.arange(conceptColumnsActivation.shape[0], dtype=pt.long, device=conceptColumnsActivation.device)
			columnActivationValues = conceptColumnsActivation
			kcAvailable = columnActivationValues.shape[0]
		topkCount = min(kcMax, kcAvailable)
		if(topkCount < 1):
			topkCount = 1
		conceptColumnsActivationTopkConcepts = pt.topk(columnActivationValues, topkCount)
		kc = len(conceptColumnsActivationTopkConcepts.indices)
		selectedColumnIndices = columnIndexLookup.index_select(0, conceptColumnsActivationTopkConcepts.indices)
	else:
		if(kcDynamic):
			conceptColumnsActivation = conceptColumnsActivation[conceptColumnsActivation > kcActivationThreshold]	#select kcMax columns above threshold
		conceptColumnsActivationTopkConcepts = pt.topk(conceptColumnsActivation, kcMax)
		kc = len(conceptColumnsActivationTopkConcepts.indices)
		if(kcDynamic and kc < 1):
			print("selectMostActiveFeature kcDynamic error: kc < 1; cannot continue to predict columns; consider disabling kcDynamic for debug")
			exit()
		selectedColumnIndices = conceptColumnsActivationTopkConcepts.indices

	#top feature selection;
	if(kc==1):
		topkConceptColumnsActivation = globalFeatureNeuronsActivationAllSegments[selectedColumnIndices[0]].unsqueeze(0)	#select topk concept indices
	else:
		topkConceptColumnsActivation = GIAANNproto_sparseTensors.sliceSparseTensorMulti(globalFeatureNeuronsActivationAllSegments, 0, selectedColumnIndices)	#select topk concept indices
	topkConceptColumnsActivation = topkConceptColumnsActivation.to_dense()
	if(inferenceNormaliseFeatureSelectionByFeatureConnections):
		if(kc==1):
			topkConceptColumnsStrength = globalFeatureNeuronsStrengthAllSegments[selectedColumnIndices[0]].unsqueeze(0)	#select topk concept indices
		else:
			topkConceptColumnsStrength = GIAANNproto_sparseTensors.sliceSparseTensorMulti(globalFeatureNeuronsStrengthAllSegments, 0, selectedColumnIndices)	#select topk concept indices
		topkConceptColumnsStrength = topkConceptColumnsStrength.to_dense()
		if(not inferenceNormaliseFeatureSelectionByFeatureConnectionsStrength):
			topkConceptColumnsStrength = (topkConceptColumnsStrength > 0).float()
		topkConceptColumnsActivation = topkConceptColumnsActivation / topkConceptColumnsStrength
	topkConceptColumnsActivationTopkFeatures = pt.topk(topkConceptColumnsActivation, kf, dim=1)

	conceptColumnsIndicesPred = selectedColumnIndices
	conceptColumnsFeatureIndicesPred = topkConceptColumnsActivationTopkFeatures.indices
	conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred = enforceMinimumPredictionActivationThreshold(conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, topkConceptColumnsActivationTopkFeatures.values)
	conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred = applyColumnConstraintToPredictions(databaseNetworkObject, conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, allowedColumns, constraintMode, connectedColumns, connectedColumnsFeatures)
	
	if(inferenceUseNextTokenPredictionsOrTargetsToActivateNextColumnFeatures):
		conceptColumnsIndicesNext = conceptColumnsIndicesPred
		conceptColumnsFeatureIndicesNext = conceptColumnsFeatureIndicesPred
		if(kc > 1 or kf > 1):
			multipleSourcesNext = True
		else:
			multipleSourcesNext = False
	else:
		#while exclusively training predictive network; use targets rather than next token predictions when activating database network
		conceptColumnsIndicesNext = targetConceptColumnsIndices
		conceptColumnsFeatureIndicesNext = targetConceptColumnsFeatureIndices
		multipleSourcesNext = targetMultipleSources
		if(multipleSourcesNext):
			kc = 2
		else:
			kc = 1
		assert kf==1
	conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext = applyColumnConstraintToPredictions(databaseNetworkObject, conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext, allowedColumns, constraintMode, connectedColumns, connectedColumnsFeatures)
	if(conceptColumnsIndicesNext is None or conceptColumnsIndicesNext.numel() == 0):
		multipleSourcesNext = False
		kc = 0
	else:
		kc = conceptColumnsIndicesNext.shape[0]
		
	return conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext, multipleSourcesNext, kc, conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, targetMultipleSources, targetPreviousColumnIndex, targetNextColumnIndex

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
