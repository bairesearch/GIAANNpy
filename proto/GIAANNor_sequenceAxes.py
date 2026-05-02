"""GIAANNor_sequenceAxes.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN OR sequence Axes

"""

from GIAANNcmn_globalDefs import *
import GIAANNor_sequenceAxis


def buildImageAxesColumnCoordinatesByConceptName(columnMetadataList):
	result = None
	if(submodalityName=="image" and modalityORimageSequenceEncode=="axes"):
		result = GIAANNor_sequenceAxis.buildImageAxisColumnCoordinatesByConceptName(columnMetadataList)
	else:
		raise RuntimeError("buildImageAxesColumnCoordinatesByConceptName error: requires submodalityName=='image' and modalityORimageSequenceEncode=='axes'")
	return result


def expandImageAxesSequenceConcepts(orderedConceptNameList, requiredSourceFeatureIndicesByConceptName, activationList, columnMetadataList, axesColumnCoordinatesByConceptName):
	result = None
	if(submodalityName=="image" and modalityORimageSequenceEncode=="axes"):
		GIAANNor_sequenceAxis.expandImageAxisSequenceConcepts(orderedConceptNameList, requiredSourceFeatureIndicesByConceptName, activationList, columnMetadataList, axesColumnCoordinatesByConceptName)
		moveImageAxesCentralConceptToFirst(orderedConceptNameList, requiredSourceFeatureIndicesByConceptName, activationList, columnMetadataList)
	else:
		raise RuntimeError("expandImageAxesSequenceConcepts error: requires submodalityName=='image' and modalityORimageSequenceEncode=='axes'")
	return result


def moveImageAxesCentralConceptToFirst(orderedConceptNameList, requiredSourceFeatureIndicesByConceptName, activationList, columnMetadataList):
	result = None
	centralConceptName = None
	axesColumnIndex = None
	activationFeatureIndices = []
	if(submodalityName=="image" and modalityORimageSequenceEncode=="axes"):
		if(not isinstance(orderedConceptNameList, list)):
			raise RuntimeError("moveImageAxesCentralConceptToFirst error: orderedConceptNameList must be a list")
		if(not isinstance(requiredSourceFeatureIndicesByConceptName, dict)):
			raise RuntimeError("moveImageAxesCentralConceptToFirst error: requiredSourceFeatureIndicesByConceptName must be a dict")
		if(not isinstance(activationList, list)):
			raise RuntimeError("moveImageAxesCentralConceptToFirst error: activationList must be a list")
		if(len(activationList) > 0):
			validateImageSequenceEncodeAxesColumnIndex()
			axesColumnIndex = int(modalityORimageSequenceEncodeAxesColumnIndex)
			centralConceptName = findImageAxesCentralConceptName(columnMetadataList)
			if(centralConceptName not in requiredSourceFeatureIndicesByConceptName):
				raise RuntimeError("moveImageAxesCentralConceptToFirst error: centralConceptName missing required source feature list")
			for activation in activationList:
				if("globalFeatureIndex" not in activation):
					raise RuntimeError("moveImageAxesCentralConceptToFirst error: activation missing globalFeatureIndex")
				activationFeatureIndices.append(int(activation["globalFeatureIndex"]))
			requiredSourceFeatureIndicesByConceptName[centralConceptName].extend(activationFeatureIndices)
			if(centralConceptName in orderedConceptNameList):
				orderedConceptNameList.remove(centralConceptName)
			orderedConceptNameList.insert(axesColumnIndex, centralConceptName)
	else:
		raise RuntimeError("moveImageAxesCentralConceptToFirst error: requires submodalityName=='image' and modalityORimageSequenceEncode=='axes'")
	return result


def validateImageSequenceEncodeAxesColumnIndex():
	result = None
	if(submodalityName=="image" and modalityORimageSequenceEncode=="axes"):
		if(not isinstance(modalityORimageSequenceEncodeAxesColumnIndex, int) or isinstance(modalityORimageSequenceEncodeAxesColumnIndex, bool)):
			raise RuntimeError("validateImageSequenceEncodeAxesColumnIndex error: modalityORimageSequenceEncodeAxesColumnIndex must be an int")
		if(modalityORimageSequenceEncodeAxesColumnIndex != 0):
			raise RuntimeError("validateImageSequenceEncodeAxesColumnIndex error: modalityORimageSequenceEncodeAxesColumnIndex must equal 0")
	else:
		raise RuntimeError("validateImageSequenceEncodeAxesColumnIndex error: requires submodalityName=='image' and modalityORimageSequenceEncode=='axes'")
	return result


def findImageAxesCentralConceptName(columnMetadataList):
	result = None
	bestDistanceNumerator = None
	distanceNumerator = None
	xIndex = None
	yIndex = None
	maxXIndex = None
	maxYIndex = None
	if(submodalityName=="image" and modalityORimageSequenceEncode=="axes"):
		if(not isinstance(columnMetadataList, list)):
			raise RuntimeError("findImageAxesCentralConceptName error: columnMetadataList must be a list")
		if(len(columnMetadataList) == 0):
			raise RuntimeError("findImageAxesCentralConceptName error: columnMetadataList must not be empty")
		for columnMetadata in columnMetadataList:
			if(not isinstance(columnMetadata, dict)):
				raise RuntimeError("findImageAxesCentralConceptName error: columnMetadata must be a dict")
			if("conceptName" not in columnMetadata or "xIndex" not in columnMetadata or "yIndex" not in columnMetadata):
				raise RuntimeError("findImageAxesCentralConceptName error: columnMetadata missing conceptName, xIndex, or yIndex")
			xIndex = int(columnMetadata["xIndex"])
			yIndex = int(columnMetadata["yIndex"])
			if(maxXIndex is None or xIndex > maxXIndex):
				maxXIndex = xIndex
			if(maxYIndex is None or yIndex > maxYIndex):
				maxYIndex = yIndex
		for columnMetadata in columnMetadataList:
			xIndex = int(columnMetadata["xIndex"])
			yIndex = int(columnMetadata["yIndex"])
			distanceNumerator = ((2*xIndex - maxXIndex)*(2*xIndex - maxXIndex)) + ((2*yIndex - maxYIndex)*(2*yIndex - maxYIndex))
			if(bestDistanceNumerator is None or distanceNumerator < bestDistanceNumerator):
				bestDistanceNumerator = distanceNumerator
				result = columnMetadata["conceptName"]
		if(result is None):
			raise RuntimeError("findImageAxesCentralConceptName error: failed to identify central concept")
	else:
		raise RuntimeError("findImageAxesCentralConceptName error: requires submodalityName=='image' and modalityORimageSequenceEncode=='axes'")
	return result


def initialiseImageAxesCoordinates(sequenceObservedColumns, sequenceData):
	result = None
	if(submodalityName=="image" and modalityORimageSequenceEncode=="axes"):
		GIAANNor_sequenceAxis.initialiseImageAxisCoordinates(sequenceObservedColumns, sequenceData)
		sequenceObservedColumns.trainConnectionsUseSpatialAxes = True
	else:
		raise RuntimeError("initialiseImageAxesCoordinates error: requires submodalityName=='image' and modalityORimageSequenceEncode=='axes'")
	return result


def configureTrainConnectionsForImageAxesEncoding(sequenceObservedColumns):
	result = None
	if(submodalityName=="image" and modalityORimageSequenceEncode=="axes"):
		GIAANNor_sequenceAxis.configureTrainConnectionsForImageAxisEncoding(sequenceObservedColumns)
		sequenceObservedColumns.trainConnectionsUseSpatialAxes = True
	else:
		raise RuntimeError("configureTrainConnectionsForImageAxesEncoding error: requires submodalityName=='image' and modalityORimageSequenceEncode=='axes'")
	return result


def validateImageAxesSnapshotIndex(snapshotIndex):
	result = None
	if(submodalityName=="image" and modalityORimageSequenceEncode=="axes"):
		GIAANNor_sequenceAxis.validateImageAxisSnapshotIndex(snapshotIndex)
	else:
		raise RuntimeError("validateImageAxesSnapshotIndex error: requires submodalityName=='image' and modalityORimageSequenceEncode=='axes'")
	return result


def getActiveSegmentsForImageAxesEncoding(sequenceObservedColumns, sequenceConceptIndex, targetDevice):
	result = None
	if(submodalityName=="image" and modalityORimageSequenceEncode=="axes"):
		result = GIAANNor_sequenceAxis.getActiveSegmentsForImageAxisEncoding(sequenceObservedColumns, sequenceConceptIndex, targetDevice)
	else:
		raise RuntimeError("getActiveSegmentsForImageAxesEncoding error: requires submodalityName=='image' and modalityORimageSequenceEncode=='axes'")
	return result
