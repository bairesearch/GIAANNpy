"""GIAANNproto_databaseNetworkInhibition.py

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

import os
import torch as pt

from GIAANNproto_globalDefs import *
import GIAANNproto_sparseTensors

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
