"""GIAANNproto_databaseNetworkInhibitionStorage.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_main.py

# Usage:
see GIAANNproto_main.py

# Description:
GIA ANN proto database Network Inhibition Storage

Utility module providing isolated storage for inhibitory neurons and connections.
All excitatory database structures remain untouched; inhibitory updates are kept in
parallel tensors saved under a dedicated directory.

"""

import os
import torch as pt

from GIAANNproto_globalDefs import *
import GIAANNproto_sparseTensors

inhibitoryObservedColumnsDir = os.path.join(databaseFolder, "observedColumnsInhibitory")


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

	def switchConnectionMode(self, mode):
		if(mode == "input"):
			self.featureConnections = self.featureConnectionsInput
			self.featureConnectionsOriginal = self.featureConnectionsInputOriginal
		else:
			self.featureConnections = self.featureConnectionsOutput
			self.featureConnectionsOriginal = self.featureConnectionsOutputOriginal
		self.connectionMode = mode


class InhibitoryObservedColumn:
	def __init__(self, databaseNetworkObject, conceptIndex, lemma):
		self.databaseNetworkObject = databaseNetworkObject
		self.conceptIndex = conceptIndex
		self.conceptName = lemma
		self.featureNeurons = GIAANNproto_sparseTensors.createEmptySparseTensor((arrayNumberOfProperties, arrayNumberOfSegments, databaseNetworkObject.f))
		self.featureConnectionsOutput = GIAANNproto_sparseTensors.createEmptySparseTensor((arrayNumberOfProperties, arrayNumberOfSegments, databaseNetworkObject.f, databaseNetworkObject.c, databaseNetworkObject.f))
		self.featureConnectionsInput = GIAANNproto_sparseTensors.createEmptySparseTensor((arrayNumberOfProperties, arrayNumberOfSegments, databaseNetworkObject.f, databaseNetworkObject.c, databaseNetworkObject.f))

	def resizeConceptArrays(self, newC):
		loadC = self.featureConnectionsOutput.shape[3]
		if newC > loadC:
			self.featureConnectionsOutput = self.featureConnectionsOutput.coalesce()
			expandedSizeOutput = (self.featureConnectionsOutput.shape[0], self.featureConnectionsOutput.shape[1], self.featureConnectionsOutput.shape[2], newC, self.featureConnectionsOutput.shape[4])
			self.featureConnectionsOutput = pt.sparse_coo_tensor(self.featureConnectionsOutput.indices(), self.featureConnectionsOutput.values(), size=expandedSizeOutput, dtype=arrayType, device=deviceSparse)
			self.featureConnectionsInput = self.featureConnectionsInput.coalesce()
			expandedSizeInput = (self.featureConnectionsInput.shape[0], self.featureConnectionsInput.shape[1], self.featureConnectionsInput.shape[2], newC, self.featureConnectionsInput.shape[4])
			self.featureConnectionsInput = pt.sparse_coo_tensor(self.featureConnectionsInput.indices(), self.featureConnectionsInput.values(), size=expandedSizeInput, dtype=arrayType, device=deviceSparse)

	def expandFeatureArrays(self, newF):
		loadF = self.featureConnectionsOutput.shape[2]
		if newF > loadF:
			self.featureConnectionsOutput = self.featureConnectionsOutput.coalesce()
			expandedSizeConnectionsOut = (self.featureConnectionsOutput.shape[0], self.featureConnectionsOutput.shape[1], newF, self.featureConnectionsOutput.shape[3], newF)
			self.featureConnectionsOutput = pt.sparse_coo_tensor(self.featureConnectionsOutput.indices(), self.featureConnectionsOutput.values(), size=expandedSizeConnectionsOut, dtype=arrayType, device=deviceSparse)

			self.featureConnectionsInput = self.featureConnectionsInput.coalesce()
			expandedSizeConnectionsIn = (self.featureConnectionsInput.shape[0], self.featureConnectionsInput.shape[1], newF, self.featureConnectionsInput.shape[3], newF)
			self.featureConnectionsInput = pt.sparse_coo_tensor(self.featureConnectionsInput.indices(), self.featureConnectionsInput.values(), size=expandedSizeConnectionsIn, dtype=arrayType, device=deviceSparse)

			self.featureNeurons = self.featureNeurons.coalesce()
			expandedSizeNeurons = (self.featureNeurons.shape[0], self.featureNeurons.shape[1], newF)
			self.featureNeurons = pt.sparse_coo_tensor(self.featureNeurons.indices(), self.featureNeurons.values(), size=expandedSizeNeurons, dtype=arrayType, device=deviceSparse)

	def save(self):
		os.makedirs(inhibitoryObservedColumnsDir, exist_ok=True)
		featureNeuronsPath = os.path.join(inhibitoryObservedColumnsDir, f"{self.conceptIndex}_inhib_featureNeurons{pytorchTensorFileExtension}")
		featureConnectionsOutputPath = os.path.join(inhibitoryObservedColumnsDir, f"{self.conceptIndex}_inhib_featureConnectionsOutput{pytorchTensorFileExtension}")
		featureConnectionsInputPath = os.path.join(inhibitoryObservedColumnsDir, f"{self.conceptIndex}_inhib_featureConnectionsInput{pytorchTensorFileExtension}")
		pt.save(self.featureNeurons.coalesce(), featureNeuronsPath)
		pt.save(self.featureConnectionsOutput.coalesce(), featureConnectionsOutputPath)
		pt.save(self.featureConnectionsInput.coalesce(), featureConnectionsInputPath)


def createSequenceBuffer(sequenceObservedColumns):
	return SequenceObservedColumnsInhibitionBuffer(sequenceObservedColumns)


def getInhibitoryObservedColumn(databaseNetworkObject, conceptIndex, lemma):
	if not hasattr(databaseNetworkObject, "inhibitoryObservedColumnsDict"):
		databaseNetworkObject.inhibitoryObservedColumnsDict = {}
	columnsDict = databaseNetworkObject.inhibitoryObservedColumnsDict
	if conceptIndex in columnsDict:
		columnObject = columnsDict[conceptIndex]
		columnObject.resizeConceptArrays(databaseNetworkObject.c)
		columnObject.expandFeatureArrays(databaseNetworkObject.f)
		return columnObject
	column = loadInhibitoryObservedColumn(databaseNetworkObject, conceptIndex, lemma)
	columnsDict[conceptIndex] = column
	return column


def loadInhibitoryObservedColumn(databaseNetworkObject, conceptIndex, lemma):
	featureNeuronsPath = os.path.join(inhibitoryObservedColumnsDir, f"{conceptIndex}_inhib_featureNeurons{pytorchTensorFileExtension}")
	featureConnectionsOutputPath = os.path.join(inhibitoryObservedColumnsDir, f"{conceptIndex}_inhib_featureConnectionsOutput{pytorchTensorFileExtension}")
	featureConnectionsInputPath = os.path.join(inhibitoryObservedColumnsDir, f"{conceptIndex}_inhib_featureConnectionsInput{pytorchTensorFileExtension}")
	legacyConnectionsPath = os.path.join(inhibitoryObservedColumnsDir, f"{conceptIndex}_inhib_featureConnections{pytorchTensorFileExtension}")
	if os.path.exists(featureNeuronsPath) and (os.path.exists(featureConnectionsOutputPath) or os.path.exists(legacyConnectionsPath)):
		featureNeurons = pt.load(featureNeuronsPath).to(deviceSparse)
		if(os.path.exists(featureConnectionsOutputPath)):
			featureConnectionsOutput = pt.load(featureConnectionsOutputPath).to(deviceSparse)
		else:
			featureConnectionsOutput = pt.load(legacyConnectionsPath).to(deviceSparse)
		if(os.path.exists(featureConnectionsInputPath)):
			featureConnectionsInput = pt.load(featureConnectionsInputPath).to(deviceSparse)
		else:
			featureConnectionsInput = GIAANNproto_sparseTensors.createEmptySparseTensor((arrayNumberOfProperties, arrayNumberOfSegments, databaseNetworkObject.f, databaseNetworkObject.c, databaseNetworkObject.f))
		column = InhibitoryObservedColumn(databaseNetworkObject, conceptIndex, lemma)
		column.featureNeurons = featureNeurons
		column.featureConnectionsOutput = featureConnectionsOutput
		column.featureConnectionsInput = featureConnectionsInput
		column.resizeConceptArrays(databaseNetworkObject.c)
		column.expandFeatureArrays(databaseNetworkObject.f)
	else:
		column = InhibitoryObservedColumn(databaseNetworkObject, conceptIndex, lemma)
	return column


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

	for cIdx, observedColumn in sequenceObservedColumnsDict.items():
		conceptIndex = observedColumn.conceptIndex
		lemma = observedColumn.conceptName
		inhibitoryColumn = getInhibitoryObservedColumn(sequenceObservedColumnsBase.databaseNetworkObject, conceptIndex, lemma)

		# feature neurons
		GIAANNproto_sparseTensors.insertSequenceObservedColumnIntoObservedColumnFeatures(None, cIdx, fIdxTensor, featureIndicesInObserved, featureNeuronsSparse, inhibitoryColumn, True)

		# feature connections output
		inhibitoryColumn.featureConnectionsOutput = GIAANNproto_sparseTensors.insertSequenceObservedColumnIntoObservedColumnConnections(sequenceObservedColumnsBase, cIdx, fIdxTensor, featureIndicesInObserved, featureConnectionsOutputSparse, inhibitoryColumn.featureConnectionsOutput, featureConnectionsOutput=True)

		# feature connections input
		inhibitoryColumn.featureConnectionsInput = GIAANNproto_sparseTensors.insertSequenceObservedColumnIntoObservedColumnConnections(sequenceObservedColumnsBase, cIdx, fIdxTensor, featureIndicesInObserved, featureConnectionsInputSparse, inhibitoryColumn.featureConnectionsInput, featureConnectionsOutput=False)

		inhibitoryColumn.save()
