"""GIAANNproto_sequenceObservedColumnsInhibition.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_main.py

# Usage:
see GIAANNproto_main.py

# Description:
GIA ANN proto sequence Observed Columns Inhibition

"""

import os
import torch as pt

from GIAANNproto_globalDefs import *
import GIAANNproto_sparseTensors
import GIAANNproto_databaseNetworkInhibition
import GIAANNproto_databaseNetworkFilesInhibition

class SequenceObservedColumnsInhibitionBuffer:
	def __init__(self, sequenceObservedColumns):
		self.databaseNetworkObject = sequenceObservedColumns.databaseNetworkObject
		self.cs = sequenceObservedColumns.cs
		self.fs = sequenceObservedColumns.fs
		self.featureNeurons = pt.zeros_like(sequenceObservedColumns.featureNeurons)
		self.featureNeuronsOriginal = pt.zeros_like(sequenceObservedColumns.featureNeurons)
		self.featureConnectionsOutput = pt.zeros_like(sequenceObservedColumns.featureConnections)
		self.featureConnectionsOutputOriginal = pt.zeros_like(sequenceObservedColumns.featureConnections)
		self.featureConnectionsInput = pt.zeros_like(sequenceObservedColumns.featureConnections)
		self.featureConnectionsInputOriginal = pt.zeros_like(sequenceObservedColumns.featureConnections)
		self.featureConnections = self.featureConnectionsOutput
		self.featureConnectionsOriginal = self.featureConnectionsOutputOriginal
		self.connectionMode = "output"
		self.featureNeuronChanges = [None]*self.cs
		self.baseSequenceObservedColumns = sequenceObservedColumns
		#store output updates targeting columns/features that never appear in the local sequence buffer
		self.externalOutputUpdates = []

	def switchConnectionMode(self, mode):
		if(mode == "input"):
			self.featureConnections = self.featureConnectionsInput
			self.featureConnectionsOriginal = self.featureConnectionsInputOriginal
		else:
			self.featureConnections = self.featureConnectionsOutput
			self.featureConnectionsOriginal = self.featureConnectionsOutputOriginal
		self.connectionMode = mode

def createSequenceBuffer(sequenceObservedColumns):
	return SequenceObservedColumnsInhibitionBuffer(sequenceObservedColumns)

def applySequenceUpdates(sequenceObservedColumnsBase, inhibitionBuffer):
	featureNeuronsDelta = inhibitionBuffer.featureNeurons - inhibitionBuffer.featureNeuronsOriginal
	featureConnectionsOutputDelta = inhibitionBuffer.featureConnectionsOutput - inhibitionBuffer.featureConnectionsOutputOriginal
	featureConnectionsInputDelta = inhibitionBuffer.featureConnectionsInput - inhibitionBuffer.featureConnectionsInputOriginal

	if not pt.any(featureNeuronsDelta) and not pt.any(featureConnectionsOutputDelta) and not pt.any(featureConnectionsInputDelta):
		return

	featureNeuronsSparse = featureNeuronsDelta.to_sparse()
	featureConnectionsOutputSparse = featureConnectionsOutputDelta.to_sparse()
	featureConnectionsInputSparse = featureConnectionsInputDelta.to_sparse()
	if(performRedundantCoalesce):
		featureNeuronsSparse = featureNeuronsSparse.coalesce()
		featureConnectionsOutputSparse = featureConnectionsOutputSparse.coalesce()
		featureConnectionsInputSparse = featureConnectionsInputSparse.coalesce()

	if(trainSequenceObservedColumnsMatchSequenceWords):
		sequenceObservedColumnsDict = sequenceObservedColumnsBase.sequenceObservedColumnsDict
	else:
		sequenceObservedColumnsDict = sequenceObservedColumnsBase.observedColumnsDict2

	featureIndicesInObserved, fIdxTensor = sequenceObservedColumnsBase.getObservedColumnFeatureIndices()

	inhibitoryColumnsCache = {}

	def getInhibitoryColumn(conceptIndex, lemma):
		if(conceptIndex not in inhibitoryColumnsCache):
			inhibitoryColumnsCache[conceptIndex] = GIAANNproto_databaseNetworkFilesInhibition.getInhibitoryObservedColumn(sequenceObservedColumnsBase.databaseNetworkObject, conceptIndex, lemma)
		return inhibitoryColumnsCache[conceptIndex]

	for cIdx, observedColumn in sequenceObservedColumnsDict.items():
		conceptIndex = observedColumn.conceptIndex
		lemma = observedColumn.conceptName
		inhibitoryColumn = getInhibitoryColumn(conceptIndex, lemma)

		# feature neurons
		GIAANNproto_sparseTensors.insertSequenceObservedColumnIntoObservedColumnFeatures(None, cIdx, fIdxTensor, featureIndicesInObserved, featureNeuronsSparse, inhibitoryColumn, True)

		# feature connections output
		inhibitoryColumn.featureConnectionsOutput = GIAANNproto_sparseTensors.insertSequenceObservedColumnIntoObservedColumnConnections(sequenceObservedColumnsBase, cIdx, fIdxTensor, featureIndicesInObserved, featureConnectionsOutputSparse, inhibitoryColumn.featureConnectionsOutput, featureConnectionsOutput=True)

		# feature connections input
		inhibitoryColumn.featureConnectionsInput = GIAANNproto_sparseTensors.insertSequenceObservedColumnIntoObservedColumnConnections(sequenceObservedColumnsBase, cIdx, fIdxTensor, featureIndicesInObserved, featureConnectionsInputSparse, inhibitoryColumn.featureConnectionsInput, featureConnectionsOutput=False)

	if(len(inhibitionBuffer.externalOutputUpdates) > 0):
		applyExternalOutputUpdates(sequenceObservedColumnsBase, inhibitionBuffer, inhibitoryColumnsCache)

	for inhibitoryColumn in inhibitoryColumnsCache.values():
		GIAANNproto_databaseNetworkFilesInhibition.saveObservedColumnInhibition(inhibitoryColumn)

def applyExternalOutputUpdates(sequenceObservedColumnsBase, inhibitionBuffer, inhibitoryColumnsCache):
	databaseNetworkObject = sequenceObservedColumnsBase.databaseNetworkObject
	for updateEntry in inhibitionBuffer.externalOutputUpdates:
		targetConceptIndex = updateEntry["targetColumnConceptIndex"]
		if(targetConceptIndex not in inhibitoryColumnsCache):
			lemma = databaseNetworkObject.conceptColumnsList[targetConceptIndex]
			inhibitoryColumnsCache[targetConceptIndex] = GIAANNproto_databaseNetworkFilesInhibition.getInhibitoryObservedColumn(databaseNetworkObject, targetConceptIndex, lemma)
		inhibitoryColumn = inhibitoryColumnsCache[targetConceptIndex]
		applyExternalOutputUpdateToColumn(inhibitoryColumn, updateEntry)

def applyExternalOutputUpdateToColumn(inhibitoryColumn, updateEntry):
	segmentsMask = updateEntry["segmentsMask"]
	if(segmentsMask is None or not pt.any(segmentsMask)):
		segmentsMask = pt.zeros(arrayNumberOfSegments, dtype=pt.bool)
		segmentsMask[arrayIndexSegmentFirst] = True
	else:
		segmentsMask = segmentsMask.to(dtype=pt.bool)
	activeSegmentIndices = pt.nonzero(segmentsMask, as_tuple=False)
	if(activeSegmentIndices.numel() == 0):
		activeSegmentIndices = pt.tensor([[arrayIndexSegmentFirst]], dtype=pt.long)
	for segmentTensor in activeSegmentIndices:
		segmentIndex = int(segmentTensor.item())
		if(arrayIndexPropertiesStrength):
			inhibitoryColumn.featureConnectionsOutput = addConnectionPropertyValue(inhibitoryColumn.featureConnectionsOutput, arrayIndexPropertiesStrengthIndex, segmentIndex, updateEntry["inhibitoryFeatureIndex"], updateEntry["candidateColumnConceptIndex"], updateEntry["candidateFeatureConceptIndex"], inhibitoryConnectionStrengthIncrement)
		if(arrayIndexPropertiesPermanence):
			inhibitoryColumn.featureConnectionsOutput = addConnectionPropertyValue( inhibitoryColumn.featureConnectionsOutput, arrayIndexPropertiesPermanenceIndex, segmentIndex, updateEntry["inhibitoryFeatureIndex"], updateEntry["candidateColumnConceptIndex"], updateEntry["candidateFeatureConceptIndex"], z1)
		if(arrayIndexPropertiesActivation):
			inhibitoryColumn.featureConnectionsOutput = setConnectionPropertyValue( inhibitoryColumn.featureConnectionsOutput, arrayIndexPropertiesActivationIndex, segmentIndex, updateEntry["inhibitoryFeatureIndex"], updateEntry["candidateColumnConceptIndex"], updateEntry["candidateFeatureConceptIndex"], 0.0)
		if(arrayIndexPropertiesTime):
			if(inferenceUseNeuronFeaturePropertiesTime):
				timeValue = 0.0
			else:
				timeValue = float(updateEntry["sequenceIndex"])
			inhibitoryColumn.featureConnectionsOutput = setConnectionPropertyValue( inhibitoryColumn.featureConnectionsOutput, arrayIndexPropertiesTimeIndex, segmentIndex, updateEntry["inhibitoryFeatureIndex"], updateEntry["candidateColumnConceptIndex"], updateEntry["candidateFeatureConceptIndex"], timeValue)
		if(arrayIndexPropertiesPos):
			inhibitoryColumn.featureConnectionsOutput = setConnectionPropertyValue( inhibitoryColumn.featureConnectionsOutput, arrayIndexPropertiesPosIndex, segmentIndex, updateEntry["inhibitoryFeatureIndex"], updateEntry["candidateColumnConceptIndex"], updateEntry["candidateFeatureConceptIndex"], float(updateEntry["sourcePosValue"]))

def addConnectionPropertyValue(tensorSparse, propertyIndex, segmentIndex, inhibitoryFeatureIndex, candidateColumnIndex, candidateFeatureIndex, incrementValue):
	dimensions = [propertyIndex, segmentIndex, inhibitoryFeatureIndex, candidateColumnIndex, candidateFeatureIndex]
	return GIAANNproto_sparseTensors.addElementValueToSparseTensor(tensorSparse, dimensions, incrementValue)

def setConnectionPropertyValue(tensorSparse, propertyIndex, segmentIndex, inhibitoryFeatureIndex, candidateColumnIndex, candidateFeatureIndex, newValue):
	dimensions = [propertyIndex, segmentIndex, inhibitoryFeatureIndex, candidateColumnIndex, candidateFeatureIndex]
	currentValue = getSparseTensorValue(tensorSparse, dimensions)
	delta = newValue - currentValue
	if(abs(delta) < 1e-4):
		return tensorSparse
	return GIAANNproto_sparseTensors.addElementValueToSparseTensor(tensorSparse, dimensions, delta)

def getSparseTensorValue(tensorSparse, dimensions):
	tensorSparse = tensorSparse.coalesce()
	if(tensorSparse._nnz() == 0):
		return 0.0
	indices = tensorSparse.indices()
	values = tensorSparse.values()
	targetIndex = pt.tensor(dimensions, dtype=indices.dtype, device=indices.device).unsqueeze(1)
	mask = (indices == targetIndex).all(dim=0)
	if(not pt.any(mask)):
		return 0.0
	indexTensor = mask.nonzero(as_tuple=False)
	if(indexTensor.numel() == 0):
		return 0.0
	valueIndex = indexTensor[0].item()
	return float(values[valueIndex].item())
