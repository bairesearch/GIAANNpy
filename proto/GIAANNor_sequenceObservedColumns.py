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
		self.sequenceConceptFieldXTensor = None
		self.sequenceConceptFieldYTensor = None
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
			self.initialiseImageDistanceFieldCoordinates(sequenceData)
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

	def initialiseImageDistanceFieldCoordinates(self, sequenceData):
		result = None
		fieldCoordinatesByConceptName = None
		fieldCoordinates = None
		fieldXList = []
		fieldYList = []
		if(submodalityName=="image" and modalityORimageSequenceEncode=="distance"):
			if("imageDistanceFieldCoordinatesByConceptName" not in sequenceData):
				raise RuntimeError("initialiseImageDistanceFieldCoordinates error: sequenceData missing imageDistanceFieldCoordinatesByConceptName")
			fieldCoordinatesByConceptName = sequenceData["imageDistanceFieldCoordinatesByConceptName"]
			for conceptName in sequenceData["orderedConceptNameList"]:
				if(conceptName not in fieldCoordinatesByConceptName):
					raise RuntimeError("initialiseImageDistanceFieldCoordinates error: conceptName missing image distance field coordinates (" + conceptName + ")")
				fieldCoordinates = fieldCoordinatesByConceptName[conceptName]
				if(not isinstance(fieldCoordinates, tuple) or len(fieldCoordinates) != 2):
					raise RuntimeError("initialiseImageDistanceFieldCoordinates error: fieldCoordinates must be a tuple of length 2")
				if(not isinstance(fieldCoordinates[0], int) or isinstance(fieldCoordinates[0], bool) or not isinstance(fieldCoordinates[1], int) or isinstance(fieldCoordinates[1], bool)):
					raise RuntimeError("initialiseImageDistanceFieldCoordinates error: fieldCoordinates values must be ints")
				fieldXList.append(int(fieldCoordinates[0]))
				fieldYList.append(int(fieldCoordinates[1]))
			self.trainConnectionsUseSpatialDistance = True
			self.sequenceConceptFieldXTensor = pt.tensor(fieldXList, dtype=pt.long)
			self.sequenceConceptFieldYTensor = pt.tensor(fieldYList, dtype=pt.long)
		else:
			raise RuntimeError("initialiseImageDistanceFieldCoordinates error: requires submodalityName=='image' and modalityORimageSequenceEncode=='distance'")
		return result

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
