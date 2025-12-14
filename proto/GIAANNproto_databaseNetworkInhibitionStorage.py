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
		indices = featureNeuronsSparse.indices()
		values = featureNeuronsSparse.values()
		if(indices.shape[1] > 0):
			mask = (indices[2] == cIdx) & pt.isin(indices[3], fIdxTensor)
			if pt.any(mask):
				filteredIndices = indices[:, mask]
				filteredValues = values[mask]
				filteredIndices[2] = filteredIndices[3]
				filteredIndices = filteredIndices[0:3]
				if(trainSequenceObservedColumnsUseSequenceFeaturesOnly):
					filteredIndices[2] = featureIndicesInObserved[filteredIndices[2]]
				inhibitoryColumn.featureNeurons = inhibitoryColumn.featureNeurons + pt.sparse_coo_tensor(filteredIndices, filteredValues, size=inhibitoryColumn.featureNeurons.size(), dtype=arrayType, device=deviceSparse)
				inhibitoryColumn.featureNeurons = inhibitoryColumn.featureNeurons.coalesce()
				inhibitoryColumn.featureNeurons.values().clamp_(min=0)

		# feature connections output
		indicesConnOut = featureConnectionsOutputSparse.indices()
		valuesConnOut = featureConnectionsOutputSparse.values()
		if(indicesConnOut.shape[1] > 0):
			maskConnOut = (indicesConnOut[2] == cIdx)
			if pt.any(maskConnOut):
				filteredIndices = indicesConnOut[:, maskConnOut]
				filteredValues = valuesConnOut[maskConnOut]
				filteredIndices[2] = filteredIndices[3]
				filteredIndices[3] = filteredIndices[4]
				filteredIndices[4] = filteredIndices[5]
				filteredIndices = filteredIndices[0:5]
				filteredIndices[3] = sequenceObservedColumnsBase.conceptIndicesInSequenceObservedTensor[filteredIndices[3]]
				if(trainSequenceObservedColumnsUseSequenceFeaturesOnly):
					filteredIndices[2] = featureIndicesInObserved[filteredIndices[2]]
					filteredIndices[4] = featureIndicesInObserved[filteredIndices[4]]
				inhibitoryColumn.featureConnectionsOutput = inhibitoryColumn.featureConnectionsOutput + pt.sparse_coo_tensor(filteredIndices, filteredValues, size=inhibitoryColumn.featureConnectionsOutput.size(), dtype=arrayType, device=deviceSparse)
				inhibitoryColumn.featureConnectionsOutput = inhibitoryColumn.featureConnectionsOutput.coalesce()
				inhibitoryColumn.featureConnectionsOutput.values().clamp_(min=0)

		# feature connections input
		indicesConnIn = featureConnectionsInputSparse.indices()
		valuesConnIn = featureConnectionsInputSparse.values()
		if(indicesConnIn.shape[1] > 0):
			maskConnIn = (indicesConnIn[4] == cIdx)
			if pt.any(maskConnIn):
				filteredIndices = indicesConnIn[:, maskConnIn]
				filteredValues = valuesConnIn[maskConnIn]
				reorderedIndices = pt.stack((
					filteredIndices[0],	# property index
					filteredIndices[1],	# segment index
					filteredIndices[5],	# inhibitory feature index (target)
					filteredIndices[2],	# source column index
					filteredIndices[3],	# source feature index
				), dim=0)
				reorderedIndices[3] = sequenceObservedColumnsBase.conceptIndicesInSequenceObservedTensor[reorderedIndices[3]]
				if(trainSequenceObservedColumnsUseSequenceFeaturesOnly):
					reorderedIndices[2] = featureIndicesInObserved[reorderedIndices[2]]
					reorderedIndices[4] = featureIndicesInObserved[reorderedIndices[4]]
				inhibitoryColumn.featureConnectionsInput = inhibitoryColumn.featureConnectionsInput + pt.sparse_coo_tensor(reorderedIndices, filteredValues, size=inhibitoryColumn.featureConnectionsInput.size(), dtype=arrayType, device=deviceSparse)
				inhibitoryColumn.featureConnectionsInput = inhibitoryColumn.featureConnectionsInput.coalesce()
				inhibitoryColumn.featureConnectionsInput.values().clamp_(min=0)

		inhibitoryColumn.save()
