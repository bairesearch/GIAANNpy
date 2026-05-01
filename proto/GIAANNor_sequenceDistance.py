"""GIAANNor_sequenceDistance.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN OR sequence Distance

"""

import math
import torch as pt

from GIAANNcmn_globalDefs import *


def isImageDistanceLikeEncoding():
	result = False
	if(submodalityName=="image" and (modalityORimageSequenceEncode=="distance" or modalityORimageSequenceEncode=="axis")):
		result = True
	return result


def buildImageDistanceFieldCoordinatesByConceptName(columnMetadataList):
	result = {}
	gridWidth = None
	gridHeight = None
	xIndex = None
	yIndex = None
	fieldXIndex = None
	fieldYIndex = None
	if(isImageDistanceLikeEncoding()):
		validateImageDistanceFieldCoordinateParameters(columnMetadataList)
		gridWidth, gridHeight = calculateImageDistanceFieldGridDimensions(columnMetadataList)
		for columnMetadata in columnMetadataList:
			xIndex = int(columnMetadata["xIndex"])
			yIndex = int(columnMetadata["yIndex"])
			fieldXIndex = calculateImageDistanceFieldCoordinate(xIndex, gridWidth)
			fieldYIndex = calculateImageDistanceFieldCoordinate(yIndex, gridHeight)
			result[columnMetadata["conceptName"]] = (fieldXIndex, fieldYIndex)
	else:
		raise RuntimeError("buildImageDistanceFieldCoordinatesByConceptName error: requires submodalityName=='image' and modalityORimageSequenceEncode=='distance' or 'axis'")
	return result


def validateImageDistanceFieldCoordinateParameters(columnMetadataList):
	result = None
	if(isImageDistanceLikeEncoding()):
		if(not isinstance(modalityORimageSequenceEncodeDistanceFieldSegments, int) or isinstance(modalityORimageSequenceEncodeDistanceFieldSegments, bool)):
			raise RuntimeError("validateImageDistanceFieldCoordinateParameters error: modalityORimageSequenceEncodeDistanceFieldSegments must be an int")
		if(modalityORimageSequenceEncodeDistanceFieldSegments <= 0):
			raise RuntimeError("validateImageDistanceFieldCoordinateParameters error: modalityORimageSequenceEncodeDistanceFieldSegments must be > 0")
		if(not isinstance(columnMetadataList, list)):
			raise RuntimeError("validateImageDistanceFieldCoordinateParameters error: columnMetadataList must be a list")
		if(len(columnMetadataList) == 0):
			raise RuntimeError("validateImageDistanceFieldCoordinateParameters error: columnMetadataList must not be empty")
		for columnMetadata in columnMetadataList:
			if(not isinstance(columnMetadata, dict)):
				raise RuntimeError("validateImageDistanceFieldCoordinateParameters error: columnMetadata must be a dict")
			if("conceptName" not in columnMetadata or "xIndex" not in columnMetadata or "yIndex" not in columnMetadata):
				raise RuntimeError("validateImageDistanceFieldCoordinateParameters error: columnMetadata missing conceptName, xIndex, or yIndex")
			if(not isinstance(columnMetadata["conceptName"], str) or columnMetadata["conceptName"] == ""):
				raise RuntimeError("validateImageDistanceFieldCoordinateParameters error: conceptName must be a non-empty string")
			if(not isinstance(columnMetadata["xIndex"], int) or isinstance(columnMetadata["xIndex"], bool) or not isinstance(columnMetadata["yIndex"], int) or isinstance(columnMetadata["yIndex"], bool)):
				raise RuntimeError("validateImageDistanceFieldCoordinateParameters error: xIndex/yIndex must be ints")
			if(columnMetadata["xIndex"] < 0 or columnMetadata["yIndex"] < 0):
				raise RuntimeError("validateImageDistanceFieldCoordinateParameters error: xIndex/yIndex must be >= 0")
	else:
		raise RuntimeError("validateImageDistanceFieldCoordinateParameters error: requires submodalityName=='image' and modalityORimageSequenceEncode=='distance' or 'axis'")
	return result


def calculateImageDistanceFieldGridDimensions(columnMetadataList):
	result = None
	maxXIndex = None
	maxYIndex = None
	if(isImageDistanceLikeEncoding()):
		validateImageDistanceFieldCoordinateParameters(columnMetadataList)
		maxXIndex = max(int(columnMetadata["xIndex"]) for columnMetadata in columnMetadataList)
		maxYIndex = max(int(columnMetadata["yIndex"]) for columnMetadata in columnMetadataList)
		result = (maxXIndex + 1, maxYIndex + 1)
	else:
		raise RuntimeError("calculateImageDistanceFieldGridDimensions error: requires submodalityName=='image' and modalityORimageSequenceEncode=='distance' or 'axis'")
	return result


def calculateImageDistanceFieldCoordinate(coordinateIndex, numberOfFieldColumns):
	result = None
	if(isImageDistanceLikeEncoding()):
		if(not isinstance(coordinateIndex, int) or isinstance(coordinateIndex, bool)):
			raise RuntimeError("calculateImageDistanceFieldCoordinate error: coordinateIndex must be an int")
		if(not isinstance(numberOfFieldColumns, int) or isinstance(numberOfFieldColumns, bool)):
			raise RuntimeError("calculateImageDistanceFieldCoordinate error: numberOfFieldColumns must be an int")
		if(numberOfFieldColumns <= 0):
			raise RuntimeError("calculateImageDistanceFieldCoordinate error: numberOfFieldColumns must be > 0")
		if(coordinateIndex < 0 or coordinateIndex >= numberOfFieldColumns):
			raise RuntimeError("calculateImageDistanceFieldCoordinate error: coordinateIndex out of range")
		result = int((int(coordinateIndex)*int(modalityORimageSequenceEncodeDistanceFieldSegments))//int(numberOfFieldColumns))
		if(result < 0 or result >= int(modalityORimageSequenceEncodeDistanceFieldSegments)):
			raise RuntimeError("calculateImageDistanceFieldCoordinate error: calculated field coordinate out of range")
	else:
		raise RuntimeError("calculateImageDistanceFieldCoordinate error: requires submodalityName=='image' and modalityORimageSequenceEncode=='distance' or 'axis'")
	return result


def initialiseImageDistanceFieldCoordinates(sequenceObservedColumns, sequenceData):
	result = None
	fieldCoordinatesByConceptName = None
	fieldCoordinates = None
	fieldXList = []
	fieldYList = []
	if(isImageDistanceLikeEncoding()):
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
		sequenceObservedColumns.trainConnectionsUseSpatialDistance = True
		sequenceObservedColumns.sequenceConceptFieldXTensor = pt.tensor(fieldXList, dtype=pt.long)
		sequenceObservedColumns.sequenceConceptFieldYTensor = pt.tensor(fieldYList, dtype=pt.long)
	else:
		raise RuntimeError("initialiseImageDistanceFieldCoordinates error: requires submodalityName=='image' and modalityORimageSequenceEncode=='distance' or 'axis'")
	return result


def configureTrainConnectionsForImageDistanceEncoding(sequenceObservedColumns):
	result = None
	if(isImageDistanceLikeEncoding()):
		sequenceObservedColumns.trainConnectionsIncludeSameTimeIndex = True
		sequenceObservedColumns.trainConnectionsUseSpatialDistance = True
	else:
		raise RuntimeError("configureTrainConnectionsForImageDistanceEncoding error: requires submodalityName=='image' and modalityORimageSequenceEncode=='distance' or 'axis'")
	return result


def validateImageDistanceSnapshotIndex(snapshotIndex):
	result = None
	if(isImageDistanceLikeEncoding()):
		if(not isinstance(snapshotIndex, int) or isinstance(snapshotIndex, bool)):
			raise RuntimeError("validateImageDistanceSnapshotIndex error: snapshotIndex must be an int")
		if(snapshotIndex != 0):
			raise RuntimeError("validateImageDistanceSnapshotIndex error: snapshotIndex must be 0 when modalityORimageSequenceEncode is 'distance' or 'axis'")
	else:
		raise RuntimeError("validateImageDistanceSnapshotIndex error: requires submodalityName=='image' and modalityORimageSequenceEncode=='distance' or 'axis'")
	return result


def getActiveSegmentsForImageDistanceEncoding(sequenceObservedColumns, sequenceConceptIndex, targetDevice):
	result = None
	fieldXIndex = None
	fieldYIndex = None
	maxSegmentIndex = None
	if(isImageDistanceLikeEncoding()):
		validateImageDistanceSequenceObservedColumnCoordinates(sequenceObservedColumns)
		if(not isinstance(sequenceConceptIndex, int) or isinstance(sequenceConceptIndex, bool)):
			raise RuntimeError("getActiveSegmentsForImageDistanceEncoding error: sequenceConceptIndex must be an int")
		if(sequenceConceptIndex < 0 or sequenceConceptIndex >= int(sequenceObservedColumns.sequenceConceptFieldXTensor.shape[0])):
			raise RuntimeError("getActiveSegmentsForImageDistanceEncoding error: sequenceConceptIndex out of range")
		fieldXIndex = int(sequenceObservedColumns.sequenceConceptFieldXTensor[sequenceConceptIndex].item())
		fieldYIndex = int(sequenceObservedColumns.sequenceConceptFieldYTensor[sequenceConceptIndex].item())
		maxSegmentIndex = calculateImageDistanceMaxSegmentIndexForTargetFieldCoordinate(fieldXIndex, fieldYIndex)
		result = pt.arange(0, maxSegmentIndex + 1, device=targetDevice, dtype=pt.long)
	else:
		raise RuntimeError("getActiveSegmentsForImageDistanceEncoding error: requires submodalityName=='image' and modalityORimageSequenceEncode=='distance' or 'axis'")
	return result


def validateImageDistanceSequenceObservedColumnCoordinates(sequenceObservedColumns):
	result = None
	if(isImageDistanceLikeEncoding()):
		if(not hasattr(sequenceObservedColumns, "sequenceConceptFieldXTensor") or not hasattr(sequenceObservedColumns, "sequenceConceptFieldYTensor")):
			raise RuntimeError("validateImageDistanceSequenceObservedColumnCoordinates error: sequenceObservedColumns missing sequence concept field coordinate tensors")
		if(sequenceObservedColumns.sequenceConceptFieldXTensor is None or sequenceObservedColumns.sequenceConceptFieldYTensor is None):
			raise RuntimeError("validateImageDistanceSequenceObservedColumnCoordinates error: sequence concept field coordinate tensors must not be None")
		if(not pt.is_tensor(sequenceObservedColumns.sequenceConceptFieldXTensor) or not pt.is_tensor(sequenceObservedColumns.sequenceConceptFieldYTensor)):
			raise RuntimeError("validateImageDistanceSequenceObservedColumnCoordinates error: sequence concept field coordinate tensors must be tensors")
		if(sequenceObservedColumns.sequenceConceptFieldXTensor.dim() != 1 or sequenceObservedColumns.sequenceConceptFieldYTensor.dim() != 1):
			raise RuntimeError("validateImageDistanceSequenceObservedColumnCoordinates error: sequence concept field coordinate tensors must be rank 1")
		if(int(sequenceObservedColumns.sequenceConceptFieldXTensor.shape[0]) != int(sequenceObservedColumns.cs) or int(sequenceObservedColumns.sequenceConceptFieldYTensor.shape[0]) != int(sequenceObservedColumns.cs)):
			raise RuntimeError("validateImageDistanceSequenceObservedColumnCoordinates error: sequence concept field coordinate tensor lengths must equal cs")
	else:
		raise RuntimeError("validateImageDistanceSequenceObservedColumnCoordinates error: requires submodalityName=='image' and modalityORimageSequenceEncode=='distance' or 'axis'")
	return result


def calculateImageDistanceMaxSegmentIndexForTargetFieldCoordinate(fieldXIndex, fieldYIndex):
	result = None
	maxDistanceX = None
	maxDistanceY = None
	if(isImageDistanceLikeEncoding()):
		validateImageDistanceFieldIndex(fieldXIndex, "fieldXIndex")
		validateImageDistanceFieldIndex(fieldYIndex, "fieldYIndex")
		maxDistanceX = max(int(fieldXIndex), int(modalityORimageSequenceEncodeDistanceFieldSegments) - 1 - int(fieldXIndex))
		maxDistanceY = max(int(fieldYIndex), int(modalityORimageSequenceEncodeDistanceFieldSegments) - 1 - int(fieldYIndex))
		result = int(math.ceil(math.sqrt(float((maxDistanceX*maxDistanceX) + (maxDistanceY*maxDistanceY)))))
		if(result < 0 or result >= int(arrayNumberOfSegments)):
			raise RuntimeError("calculateImageDistanceMaxSegmentIndexForTargetFieldCoordinate error: calculated segment index out of range")
	else:
		raise RuntimeError("calculateImageDistanceMaxSegmentIndexForTargetFieldCoordinate error: requires submodalityName=='image' and modalityORimageSequenceEncode=='distance' or 'axis'")
	return result


def validateImageDistanceFieldIndex(fieldIndex, fieldName):
	result = None
	if(isImageDistanceLikeEncoding()):
		if(not isinstance(fieldName, str) or fieldName == ""):
			raise RuntimeError("validateImageDistanceFieldIndex error: fieldName must be a non-empty string")
		if(not isinstance(fieldIndex, int) or isinstance(fieldIndex, bool)):
			raise RuntimeError("validateImageDistanceFieldIndex error: " + fieldName + " must be an int")
		if(fieldIndex < 0 or fieldIndex >= int(modalityORimageSequenceEncodeDistanceFieldSegments)):
			raise RuntimeError("validateImageDistanceFieldIndex error: " + fieldName + " out of range")
	else:
		raise RuntimeError("validateImageDistanceFieldIndex error: requires submodalityName=='image' and modalityORimageSequenceEncode=='distance' or 'axis'")
	return result
