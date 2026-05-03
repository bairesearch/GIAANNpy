"""GIAANNor_sequenceObservedColumns.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN OR sequence Observed Columns

"""

import torch as pt

from GIAANNcmn_globalDefs import *
import GIAANNcmn_sequenceObservedColumns
import GIAANNor_sequenceAxis
import GIAANNor_sequenceAxes
import GIAANNor_sequenceDistance


class SequenceObservedColumns(GIAANNcmn_sequenceObservedColumns.SequenceObservedColumns):
	def __init__(self, databaseNetworkObject, observedColumnsDict, sequenceData, inferenceMode):
		self.databaseNetworkObject = databaseNetworkObject
		self.observedColumnsDict = observedColumnsDict
		self.observedColumnsSequenceWordIndexDict = {}
		self.sequenceData = sequenceData
		self.noDelimiterDetectedBetweenConceptTokens = False
		self.requiredSourceFeatureIndicesByObservedColumn = None
		self.trainConnectionsIncludeSameTimeIndex = False
		self.trainConnectionsUseSpatialDistance = False
		self.trainConnectionsUseSpatialAxis = False
		self.trainConnectionsUseSpatialAxes = False
		self.sequenceConceptFieldXTensor = None
		self.sequenceConceptFieldYTensor = None
		self.sequenceConceptAxisXTensor = None
		self.sequenceConceptAxisYTensor = None
		self.imageAxesFeatureFieldXTensor = None
		self.imageAxesFeatureFieldYTensor = None
		self.imageAxesFeatureAxisMaskTensor = None
		self.imageAxesFeatureCentralColumnMaskTensor = None
		self.imageAxesCentralFieldX = None
		self.imageAxesCentralFieldY = None
		self.imageAxesTargetColumnIndexTensor = None
		self.columnsIndexSequenceWordIndexDict = {}
		self.sequenceObservedColumnsDict = {}
		self.conceptIndicesInObservedList = []
		for sequenceConceptIndex, conceptName in enumerate(sequenceData["orderedConceptNameList"]):
			if(conceptName not in observedColumnsDict):
				raise RuntimeError("SequenceObservedColumns.__init__ error: conceptName not found in observedColumnsDict (" + conceptName + ")")
			observedColumn = observedColumnsDict[conceptName]
			self.sequenceObservedColumnsDict[sequenceConceptIndex] = observedColumn
			self.conceptIndicesInObservedList.append(int(observedColumn.conceptIndex))
		self.conceptIndicesInSequenceObservedTensor = pt.tensor(self.conceptIndicesInObservedList, dtype=pt.long)
		self.cs = len(self.sequenceObservedColumnsDict)
		self.tokens = sequenceData["activationList"]
		self.featureWords = list(sequenceData["featureWords"])
		self.featureIndicesInObservedTensor = pt.tensor(sequenceData["globalFeatureIndices"], dtype=pt.long)
		self.fIdxTensor = pt.arange(len(self.featureWords), dtype=pt.long)
		self.fs = len(self.featureWords)
		self.featureWordToIndex = {}
		self.indexToFeatureWord = {}
		for featureIndex, featureWord in enumerate(self.featureWords):
			self.featureWordToIndex[featureWord] = featureIndex
			self.indexToFeatureWord[featureIndex] = featureWord
		if(submodalityName=="image" and modalityORimageSequenceEncode=="distance"):
			GIAANNor_sequenceDistance.initialiseImageDistanceFieldCoordinates(self, sequenceData)
		elif(submodalityName=="image" and modalityORimageSequenceEncode=="axis"):
			GIAANNor_sequenceAxis.initialiseImageAxisCoordinates(self, sequenceData)
		elif(submodalityName=="image" and modalityORimageSequenceEncode=="axes"):
			GIAANNor_sequenceAxes.initialiseImageAxesCoordinates(self, sequenceData)
		self.columnStartIndicesTensor = None
		self.columnEndIndicesTensor = None
		self.columnFeatureLocalIndices = None
		self.featureNeuronChanges = [None]*self.cs
		if(trainSparseNeuronsTensor and not inferenceMode):
			self.featureNeurons = self.initialiseFeatureNeuronsSequenceSparse(self.cs, self.fs)
		else:
			self.featureNeurons = self.initialiseFeatureNeuronsSequence(self.cs, self.fs)
		if(trainSparseConnectionsTensor and not inferenceMode):
			self.featureConnections = self.initialiseFeatureConnectionsSequenceSparse(self.cs, self.fs)
		else:
			self.featureConnections = self.initialiseFeatureConnectionsSequence(self.cs, self.fs)

	def getTrainRequiredSourceFeatureIndicesByObservedColumn(self):
		result = {}
		for conceptName in self.sequenceData["orderedConceptNameList"]:
			observedColumn = self.observedColumnsDict[conceptName]
			conceptIndex = int(observedColumn.conceptIndex)
			result[conceptIndex] = sorted(set(self.sequenceData["requiredSourceFeatureIndicesByConceptName"][conceptName]))
			if(len(result[conceptIndex]) == 0):
				raise RuntimeError("getTrainRequiredSourceFeatureIndicesByObservedColumn error: no required source features for conceptName " + conceptName)
		self.requiredSourceFeatureIndicesByObservedColumn = result
		return result
