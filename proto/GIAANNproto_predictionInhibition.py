"""GIAANNproto_predictionInhibition.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_main.py

# Usage:
see GIAANNproto_main.py

# Description:
GIA ANN proto prediction inhibition helpers

"""

import torch as pt

from GIAANNproto_globalDefs import *
import GIAANNproto_sparseTensors
import GIAANNproto_predictionActivate
import GIAANNproto_databaseNetworkFilesInhibition

def applyInferenceInhibition(databaseNetworkObject, globalFeatureNeuronsActivation, conceptColumnsIndicesCurrent, conceptColumnsFeatureIndicesCurrent, conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext):
	if(conceptColumnsIndicesCurrent is None or conceptColumnsIndicesCurrent.numel() == 0):
		return globalFeatureNeuronsActivation
	if(conceptColumnsIndicesNext is None or conceptColumnsIndicesNext.numel() == 0):
		return globalFeatureNeuronsActivation
	inhibitoryColumns = loadInhibitoryColumns(databaseNetworkObject, conceptColumnsIndicesNext)
	if(len(inhibitoryColumns) == 0):
		return globalFeatureNeuronsActivation
	contextFeatureLookup = createFeatureLookup(conceptColumnsIndicesCurrent, conceptColumnsFeatureIndicesCurrent)
	if(len(contextFeatureLookup) == 0):
		return globalFeatureNeuronsActivation
	connectionTensors = buildInhibitoryInputConnectionTensors(databaseNetworkObject, inhibitoryColumns, contextFeatureLookup)
	if(len(connectionTensors) == 0):
		return globalFeatureNeuronsActivation
	activateInhibitoryNeurons(databaseNetworkObject, connectionTensors, contextFeatureLookup, globalFeatureNeuronsActivation)
	predictedFeatureLookup = createFeatureLookup(conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext)
	if(len(predictedFeatureLookup) == 0):
		return globalFeatureNeuronsActivation
	inhibitoryEffects = computeInhibitoryOutputEffects(databaseNetworkObject, inhibitoryColumns, predictedFeatureLookup)
	globalFeatureNeuronsActivation = applyInhibitoryEffect(databaseNetworkObject, globalFeatureNeuronsActivation, inhibitoryEffects)
	return globalFeatureNeuronsActivation


def loadInhibitoryColumns(databaseNetworkObject, conceptColumnsIndices):
	inhibitoryColumns = {}
	for conceptIndexTensor in conceptColumnsIndices:
		columnIndex = int(conceptIndexTensor.item())
		if(columnIndex in inhibitoryColumns):
			continue
		if(columnIndex < 0 or columnIndex >= databaseNetworkObject.c):
			continue
		lemma = databaseNetworkObject.conceptColumnsList[columnIndex]
		inhibitoryColumn = GIAANNproto_databaseNetworkFilesInhibition.getInhibitoryObservedColumn(databaseNetworkObject, columnIndex, lemma)
		if(inhibitoryColumn is None):
			continue
		inhibitoryColumns[columnIndex] = inhibitoryColumn
	return inhibitoryColumns


def createFeatureLookup(conceptColumnsIndices, conceptColumnsFeatureIndices):
	featureLookup = {}
	if(conceptColumnsIndices is None or conceptColumnsFeatureIndices is None):
		return featureLookup
	if(conceptColumnsIndices.numel() == 0 or conceptColumnsFeatureIndices.numel() == 0):
		return featureLookup
	for idx, conceptIndexTensor in enumerate(conceptColumnsIndices):
		columnIndex = int(conceptIndexTensor.squeeze().item())
		if(idx >= conceptColumnsFeatureIndices.shape[0]):
			continue
		featureTensor = conceptColumnsFeatureIndices[idx]
		if(featureTensor.numel() == 0):
			continue
		featureIndex = int(featureTensor.view(-1)[0].item())
		featureLookup[columnIndex] = featureIndex
	return featureLookup


def buildInhibitoryInputConnectionTensors(databaseNetworkObject, inhibitoryColumns, contextFeatureLookup):
	connectionEntries = {}
	for sourceColumnIndex in contextFeatureLookup.keys():
		connectionEntries[sourceColumnIndex] = []
	for targetColumnIndex, inhibitoryColumn in inhibitoryColumns.items():
		connectionsInput = getattr(inhibitoryColumn, "featureConnectionsInput", None)
		if(connectionsInput is None):
			continue
		connectionsInput = connectionsInput.coalesce()
		if(connectionsInput._nnz() == 0):
			continue
		indices = connectionsInput.indices()
		values = connectionsInput.values()
		for sourceColumnIndex, sourceFeatureIndex in contextFeatureLookup.items():
			columnMask = (indices[3] == sourceColumnIndex)
			if(not pt.any(columnMask)):
				continue
			featureMask = (indices[4] == sourceFeatureIndex)
			combinedMask = columnMask & featureMask
			if(not pt.any(combinedMask)):
				continue
			filteredIndices = indices[:, combinedMask]
			filteredValues = values[combinedMask]
			newIndices = pt.stack((
				filteredIndices[0],
				filteredIndices[1],
				filteredIndices[4],
				pt.full_like(filteredIndices[0], targetColumnIndex),
				filteredIndices[2],
			), dim=0)
			connectionEntries[sourceColumnIndex].append((newIndices, filteredValues))
	connectionTensors = {}
	shape = (arrayNumberOfProperties, arrayNumberOfSegments, databaseNetworkObject.f, databaseNetworkObject.c, databaseNetworkObject.f)
	for sourceColumnIndex, entryList in connectionEntries.items():
		if(len(entryList) == 0):
			continue
		combinedIndices = pt.cat([entry[0] for entry in entryList], dim=1)
		combinedValues = pt.cat([entry[1] for entry in entryList])
		connectionTensors[sourceColumnIndex] = pt.sparse_coo_tensor(combinedIndices, combinedValues, size=shape, dtype=arrayType, device=deviceSparse)
	return connectionTensors


def activateInhibitoryNeurons(databaseNetworkObject, connectionTensors, contextFeatureLookup, globalFeatureNeuronsActivation):
	if(len(connectionTensors) == 0):
		return
	if(not hasattr(databaseNetworkObject, "globalInhibitoryNeuronsActivation") or databaseNetworkObject.globalInhibitoryNeuronsActivation is None):
		databaseNetworkObject.globalInhibitoryNeuronsActivation = GIAANNproto_sparseTensors.createEmptySparseTensor((arrayNumberOfSegments, databaseNetworkObject.c, databaseNetworkObject.f))
	for sourceColumnIndex, connectionTensor in connectionTensors.items():
		sourceFeatureIndex = contextFeatureLookup.get(sourceColumnIndex)
		if(sourceFeatureIndex is None):
			continue
		sourceActivationValue = getSparseValue(globalFeatureNeuronsActivation, [arrayIndexSegmentFirst, sourceColumnIndex, sourceFeatureIndex])
		if(sourceActivationValue == 0):
			sourceActivationValue = j1
		sourceTensor = createSingleActivationTensor(databaseNetworkObject, sourceColumnIndex, sourceFeatureIndex, sourceActivationValue)
		conceptTensor = pt.tensor([sourceColumnIndex], dtype=pt.long, device=deviceSparse)
		featureTensor = pt.tensor([[sourceFeatureIndex]], dtype=pt.long, device=deviceSparse)
		sourceTensor, _ = GIAANNproto_predictionActivate.processFeaturesActivePredict(databaseNetworkObject, sourceTensor, None, connectionTensor, conceptTensor, featureTensor, sourceConceptIndex=sourceColumnIndex)
		sourceTensor = removeActivationAtIndex(sourceTensor, [arrayIndexSegmentFirst, sourceColumnIndex, sourceFeatureIndex], sourceActivationValue)
		databaseNetworkObject.globalInhibitoryNeuronsActivation = databaseNetworkObject.globalInhibitoryNeuronsActivation + sourceTensor
	databaseNetworkObject.globalInhibitoryNeuronsActivation = databaseNetworkObject.globalInhibitoryNeuronsActivation.coalesce()


def computeInhibitoryOutputEffects(databaseNetworkObject, inhibitoryColumns, predictedFeatureLookup):
	inhibitoryEffects = None
	if(not hasattr(databaseNetworkObject, "globalInhibitoryNeuronsActivation") or databaseNetworkObject.globalInhibitoryNeuronsActivation is None):
		return inhibitoryEffects
	for targetColumnIndex, inhibitoryColumn in inhibitoryColumns.items():
		inhibitoryFeatureIndex = predictedFeatureLookup.get(targetColumnIndex)
		if(inhibitoryFeatureIndex is None):
			continue
		inhibitoryActivationValue = getSparseValue(databaseNetworkObject.globalInhibitoryNeuronsActivation, [arrayIndexSegmentFirst, targetColumnIndex, inhibitoryFeatureIndex])
		if(inhibitoryActivationValue == 0):
			continue
		sourceTensor = createSingleActivationTensor(databaseNetworkObject, targetColumnIndex, inhibitoryFeatureIndex, inhibitoryActivationValue)
		conceptTensor = pt.tensor([targetColumnIndex], dtype=pt.long, device=deviceSparse)
		featureTensor = pt.tensor([[inhibitoryFeatureIndex]], dtype=pt.long, device=deviceSparse)
		sourceTensor, _ = GIAANNproto_predictionActivate.processFeaturesActivePredict(databaseNetworkObject, sourceTensor, None, inhibitoryColumn.featureConnectionsOutput, conceptTensor, featureTensor, sourceConceptIndex=targetColumnIndex)
		sourceTensor = removeActivationAtIndex(sourceTensor, [arrayIndexSegmentFirst, targetColumnIndex, inhibitoryFeatureIndex], inhibitoryActivationValue)
		if(inhibitoryEffects is None):
			inhibitoryEffects = sourceTensor
		else:
			inhibitoryEffects = inhibitoryEffects + sourceTensor
	if(inhibitoryEffects is not None):
		inhibitoryEffects = inhibitoryEffects.coalesce()
	return inhibitoryEffects


def applyInhibitoryEffect(databaseNetworkObject, globalFeatureNeuronsActivation, inhibitoryEffects):
	if(inhibitoryEffects is None):
		return globalFeatureNeuronsActivation
	inhibitoryEffects = inhibitoryEffects.coalesce()
	effectIndices = inhibitoryEffects.indices()
	effectValues = inhibitoryEffects.values()
	for idx in range(effectIndices.shape[1]):
		dimensions = effectIndices[:, idx].tolist()
		value = effectValues[idx].item()
		if(debugPrintInferenceInhibition):
			segmentIndex = dimensions[0]
			columnIndex = dimensions[1]
			featureIndex = dimensions[2]
			columnName = getColumnName(databaseNetworkObject, columnIndex)
			featureName = getFeatureName(databaseNetworkObject, featureIndex)
			print("inhibitionApplied: segment=", segmentIndex, ", column=", columnName, ", feature=", featureName, ", amount=", value)
		globalFeatureNeuronsActivation = GIAANNproto_sparseTensors.addElementValueToSparseTensor(globalFeatureNeuronsActivation, dimensions, -value)
	globalFeatureNeuronsActivation = globalFeatureNeuronsActivation.coalesce()
	globalFeatureNeuronsActivation._values().clamp_(min=0)
	return globalFeatureNeuronsActivation


def createSingleActivationTensor(databaseNetworkObject, columnIndex, featureIndex, activationValue):
	shape = (arrayNumberOfSegments, databaseNetworkObject.c, databaseNetworkObject.f)
	activationTensor = GIAANNproto_sparseTensors.createEmptySparseTensor(shape)
	dimensions = [arrayIndexSegmentFirst, columnIndex, featureIndex]
	activationTensor = GIAANNproto_sparseTensors.addElementValueToSparseTensor(activationTensor, dimensions, activationValue)
	return activationTensor


def removeActivationAtIndex(tensorSparse, dimensions, value):
	return GIAANNproto_sparseTensors.addElementValueToSparseTensor(tensorSparse, dimensions, -value)


def getSparseValue(sparseTensor, dimensions):
	if(sparseTensor is None):
		return 0.0
	sparseTensor = sparseTensor.coalesce()
	if(sparseTensor._nnz() == 0):
		return 0.0
	indices = sparseTensor.indices()
	values = sparseTensor.values()
	targetIndex = pt.tensor(dimensions, dtype=indices.dtype, device=indices.device).unsqueeze(1)
	mask = (indices == targetIndex).all(dim=0)
	if(not pt.any(mask)):
		return 0.0
	indexTensor = mask.nonzero(as_tuple=False)
	if(indexTensor.numel() == 0):
		return 0.0
	valueIndex = indexTensor[0].item()
	return values[valueIndex].item()


def getColumnName(databaseNetworkObject, columnIndex):
	if(columnIndex < 0 or columnIndex >= len(databaseNetworkObject.conceptColumnsList)):
		return f"<column_{columnIndex}>"
	return databaseNetworkObject.conceptColumnsList[columnIndex]


def getFeatureName(databaseNetworkObject, featureIndex):
	if(featureIndex < 0 or featureIndex >= len(databaseNetworkObject.conceptFeaturesList)):
		return f"<feature_{featureIndex}>"
	return databaseNetworkObject.conceptFeaturesList[featureIndex]
