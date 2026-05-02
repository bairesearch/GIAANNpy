"""GIAANNor_sequenceAxis.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN OR sequence Axis

"""

import torch as pt

from GIAANNcmn_globalDefs import *
import GIAANNor_sequenceDistance


def isImageAxisLikeEncoding():
	result = False
	if(submodalityName=="image" and (modalityORimageSequenceEncode=="axis" or modalityORimageSequenceEncode=="axes")):
		result = True
	return result


def buildImageAxisColumnCoordinatesByConceptName(columnMetadataList):
	result = {}
	xIndex = None
	yIndex = None
	if(isImageAxisLikeEncoding()):
		GIAANNor_sequenceDistance.validateImageDistanceFieldCoordinateParameters(columnMetadataList)
		for columnMetadata in columnMetadataList:
			xIndex = int(columnMetadata["xIndex"])
			yIndex = int(columnMetadata["yIndex"])
			result[columnMetadata["conceptName"]] = (xIndex, yIndex)
	else:
		raise RuntimeError("buildImageAxisColumnCoordinatesByConceptName error: requires submodalityName=='image' and modalityORimageSequenceEncode=='axis' or 'axes'")
	return result


def expandImageAxisSequenceConcepts(orderedConceptNameList, requiredSourceFeatureIndicesByConceptName, activationList, columnMetadataList, axisColumnCoordinatesByConceptName):
	result = None
	sourceConceptName = None
	sourceFeatureIndex = None
	sourceCoordinates = None
	sourceXIndex = None
	sourceYIndex = None
	axisConceptName = None
	axisCoordinates = None
	expandedOrderedConceptNameList = []
	if(isImageAxisLikeEncoding()):
		validateImageAxisSequenceConceptExpansionInputs(orderedConceptNameList, requiredSourceFeatureIndicesByConceptName, activationList, columnMetadataList, axisColumnCoordinatesByConceptName)
		for activation in activationList:
			sourceConceptName = activation["conceptName"]
			sourceFeatureIndex = int(activation["globalFeatureIndex"])
			sourceCoordinates = axisColumnCoordinatesByConceptName[sourceConceptName]
			sourceXIndex = int(sourceCoordinates[0])
			sourceYIndex = int(sourceCoordinates[1])
			for columnMetadata in columnMetadataList:
				axisConceptName = columnMetadata["conceptName"]
				axisCoordinates = axisColumnCoordinatesByConceptName[axisConceptName]
				if(int(axisCoordinates[0]) == sourceXIndex or int(axisCoordinates[1]) == sourceYIndex):
					requiredSourceFeatureIndicesByConceptName[axisConceptName].append(sourceFeatureIndex)
		for columnMetadata in columnMetadataList:
			axisConceptName = columnMetadata["conceptName"]
			if(len(requiredSourceFeatureIndicesByConceptName[axisConceptName]) > 0):
				expandedOrderedConceptNameList.append(axisConceptName)
		orderedConceptNameList[:] = expandedOrderedConceptNameList
	else:
		raise RuntimeError("expandImageAxisSequenceConcepts error: requires submodalityName=='image' and modalityORimageSequenceEncode=='axis' or 'axes'")
	return result


def validateImageAxisSequenceConceptExpansionInputs(orderedConceptNameList, requiredSourceFeatureIndicesByConceptName, activationList, columnMetadataList, axisColumnCoordinatesByConceptName):
	result = None
	conceptName = None
	if(isImageAxisLikeEncoding()):
		if(not isinstance(orderedConceptNameList, list)):
			raise RuntimeError("validateImageAxisSequenceConceptExpansionInputs error: orderedConceptNameList must be a list")
		if(not isinstance(requiredSourceFeatureIndicesByConceptName, dict)):
			raise RuntimeError("validateImageAxisSequenceConceptExpansionInputs error: requiredSourceFeatureIndicesByConceptName must be a dict")
		if(not isinstance(activationList, list)):
			raise RuntimeError("validateImageAxisSequenceConceptExpansionInputs error: activationList must be a list")
		if(not isinstance(columnMetadataList, list)):
			raise RuntimeError("validateImageAxisSequenceConceptExpansionInputs error: columnMetadataList must be a list")
		if(not isinstance(axisColumnCoordinatesByConceptName, dict)):
			raise RuntimeError("validateImageAxisSequenceConceptExpansionInputs error: axisColumnCoordinatesByConceptName must be a dict")
		if(len(columnMetadataList) == 0):
			raise RuntimeError("validateImageAxisSequenceConceptExpansionInputs error: columnMetadataList must not be empty")
		for columnMetadata in columnMetadataList:
			if(not isinstance(columnMetadata, dict)):
				raise RuntimeError("validateImageAxisSequenceConceptExpansionInputs error: columnMetadata must be a dict")
			if("conceptName" not in columnMetadata):
				raise RuntimeError("validateImageAxisSequenceConceptExpansionInputs error: columnMetadata missing conceptName")
			conceptName = columnMetadata["conceptName"]
			if(conceptName not in requiredSourceFeatureIndicesByConceptName):
				raise RuntimeError("validateImageAxisSequenceConceptExpansionInputs error: conceptName missing required source feature list (" + conceptName + ")")
			if(conceptName not in axisColumnCoordinatesByConceptName):
				raise RuntimeError("validateImageAxisSequenceConceptExpansionInputs error: conceptName missing axis coordinates (" + conceptName + ")")
		for activation in activationList:
			if(not isinstance(activation, dict)):
				raise RuntimeError("validateImageAxisSequenceConceptExpansionInputs error: activation must be a dict")
			if("conceptName" not in activation or "globalFeatureIndex" not in activation):
				raise RuntimeError("validateImageAxisSequenceConceptExpansionInputs error: activation missing conceptName or globalFeatureIndex")
			if(activation["conceptName"] not in axisColumnCoordinatesByConceptName):
				raise RuntimeError("validateImageAxisSequenceConceptExpansionInputs error: activation conceptName missing axis coordinates (" + activation["conceptName"] + ")")
	else:
		raise RuntimeError("validateImageAxisSequenceConceptExpansionInputs error: requires submodalityName=='image' and modalityORimageSequenceEncode=='axis' or 'axes'")
	return result


def initialiseImageAxisCoordinates(sequenceObservedColumns, sequenceData):
	result = None
	if(isImageAxisLikeEncoding()):
		GIAANNor_sequenceDistance.initialiseImageDistanceFieldCoordinates(sequenceObservedColumns, sequenceData)
		initialiseImageAxisColumnCoordinates(sequenceObservedColumns, sequenceData)
		sequenceObservedColumns.trainConnectionsUseSpatialAxis = True
	else:
		raise RuntimeError("initialiseImageAxisCoordinates error: requires submodalityName=='image' and modalityORimageSequenceEncode=='axis' or 'axes'")
	return result


def initialiseImageAxisColumnCoordinates(sequenceObservedColumns, sequenceData):
	result = None
	axisColumnCoordinatesByConceptName = None
	axisCoordinates = None
	axisXList = []
	axisYList = []
	if(isImageAxisLikeEncoding()):
		if("imageAxisColumnCoordinatesByConceptName" not in sequenceData):
			raise RuntimeError("initialiseImageAxisColumnCoordinates error: sequenceData missing imageAxisColumnCoordinatesByConceptName")
		axisColumnCoordinatesByConceptName = sequenceData["imageAxisColumnCoordinatesByConceptName"]
		for conceptName in sequenceData["orderedConceptNameList"]:
			if(conceptName not in axisColumnCoordinatesByConceptName):
				raise RuntimeError("initialiseImageAxisColumnCoordinates error: conceptName missing image axis coordinates (" + conceptName + ")")
			axisCoordinates = axisColumnCoordinatesByConceptName[conceptName]
			if(not isinstance(axisCoordinates, tuple) or len(axisCoordinates) != 2):
				raise RuntimeError("initialiseImageAxisColumnCoordinates error: axisCoordinates must be a tuple of length 2")
			if(not isinstance(axisCoordinates[0], int) or isinstance(axisCoordinates[0], bool) or not isinstance(axisCoordinates[1], int) or isinstance(axisCoordinates[1], bool)):
				raise RuntimeError("initialiseImageAxisColumnCoordinates error: axisCoordinates values must be ints")
			axisXList.append(int(axisCoordinates[0]))
			axisYList.append(int(axisCoordinates[1]))
		sequenceObservedColumns.sequenceConceptAxisXTensor = pt.tensor(axisXList, dtype=pt.long)
		sequenceObservedColumns.sequenceConceptAxisYTensor = pt.tensor(axisYList, dtype=pt.long)
	else:
		raise RuntimeError("initialiseImageAxisColumnCoordinates error: requires submodalityName=='image' and modalityORimageSequenceEncode=='axis' or 'axes'")
	return result


def configureTrainConnectionsForImageAxisEncoding(sequenceObservedColumns):
	result = None
	if(isImageAxisLikeEncoding()):
		GIAANNor_sequenceDistance.configureTrainConnectionsForImageDistanceEncoding(sequenceObservedColumns)
		sequenceObservedColumns.trainConnectionsUseSpatialAxis = True
	else:
		raise RuntimeError("configureTrainConnectionsForImageAxisEncoding error: requires submodalityName=='image' and modalityORimageSequenceEncode=='axis' or 'axes'")
	return result


def validateImageAxisSnapshotIndex(snapshotIndex):
	result = None
	if(isImageAxisLikeEncoding()):
		GIAANNor_sequenceDistance.validateImageDistanceSnapshotIndex(snapshotIndex)
	else:
		raise RuntimeError("validateImageAxisSnapshotIndex error: requires submodalityName=='image' and modalityORimageSequenceEncode=='axis' or 'axes'")
	return result


def getActiveSegmentsForImageAxisEncoding(sequenceObservedColumns, sequenceConceptIndex, targetDevice):
	result = None
	if(isImageAxisLikeEncoding()):
		result = GIAANNor_sequenceDistance.getActiveSegmentsForImageDistanceEncoding(sequenceObservedColumns, sequenceConceptIndex, targetDevice)
	else:
		raise RuntimeError("getActiveSegmentsForImageAxisEncoding error: requires submodalityName=='image' and modalityORimageSequenceEncode=='axis' or 'axes'")
	return result
