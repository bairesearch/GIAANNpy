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

	for cIdx, observedColumn in sequenceObservedColumnsDict.items():
		conceptIndex = observedColumn.conceptIndex
		lemma = observedColumn.conceptName
		inhibitoryColumn = GIAANNproto_databaseNetworkFilesInhibition.getInhibitoryObservedColumn(sequenceObservedColumnsBase.databaseNetworkObject, conceptIndex, lemma)

		# feature neurons
		GIAANNproto_sparseTensors.insertSequenceObservedColumnIntoObservedColumnFeatures(None, cIdx, fIdxTensor, featureIndicesInObserved, featureNeuronsSparse, inhibitoryColumn, True)

		# feature connections output
		inhibitoryColumn.featureConnectionsOutput = GIAANNproto_sparseTensors.insertSequenceObservedColumnIntoObservedColumnConnections(sequenceObservedColumnsBase, cIdx, fIdxTensor, featureIndicesInObserved, featureConnectionsOutputSparse, inhibitoryColumn.featureConnectionsOutput, featureConnectionsOutput=True)

		# feature connections input
		inhibitoryColumn.featureConnectionsInput = GIAANNproto_sparseTensors.insertSequenceObservedColumnIntoObservedColumnConnections(sequenceObservedColumnsBase, cIdx, fIdxTensor, featureIndicesInObserved, featureConnectionsInputSparse, inhibitoryColumn.featureConnectionsInput, featureConnectionsOutput=False)

		GIAANNproto_databaseNetworkFilesInhibition.saveObservedColumnInhibition(inhibitoryColumn)
