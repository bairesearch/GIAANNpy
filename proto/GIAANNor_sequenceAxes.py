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

import torch as pt

from GIAANNcmn_globalDefs import *


def buildImageAxesColumnCoordinatesByConceptName(columnMetadataList):
	result = None
	if(submodalityName=="image" and modalityORimageSequenceEncode=="axes"):
		columnTensorData = buildImageAxesColumnTensorData(columnMetadataList)
		result = {columnTensorData["centralConceptName"]: (int(columnTensorData["axisXTensor"][columnTensorData["centralColumnIndex"]].item()), int(columnTensorData["axisYTensor"][columnTensorData["centralColumnIndex"]].item()))}
	else:
		raise RuntimeError("buildImageAxesColumnCoordinatesByConceptName error: requires submodalityName=='image' and modalityORimageSequenceEncode=='axes'")
	return result


def buildImageAxesSequenceTensorData(columnMetadataList, selectedFilterIndices):
	result = None
	columnTensorData = None
	activeCoordinateTensor = None
	featureAxisXTensor = None
	featureAxisYTensor = None
	featureAxisMaskTensor = None
	if(submodalityName=="image" and modalityORimageSequenceEncode=="axes"):
		if(not pt.is_tensor(selectedFilterIndices)):
			raise RuntimeError("buildImageAxesSequenceTensorData error: selectedFilterIndices must be a tensor")
		if(selectedFilterIndices.dim() != 2):
			raise RuntimeError("buildImageAxesSequenceTensorData error: selectedFilterIndices rank must be 2")
		if(int(selectedFilterIndices.shape[1]) != len(columnMetadataList)):
			raise RuntimeError("buildImageAxesSequenceTensorData error: selectedFilterIndices column count mismatch")
		columnTensorData = buildImageAxesColumnTensorData(columnMetadataList)
		activeCoordinateTensor = pt.nonzero(selectedFilterIndices >= 0, as_tuple=False)
		featureAxisXTensor = columnTensorData["axisXTensor"].index_select(0, activeCoordinateTensor[:, 1]).to(dtype=pt.long)
		featureAxisYTensor = columnTensorData["axisYTensor"].index_select(0, activeCoordinateTensor[:, 1]).to(dtype=pt.long)
		featureAxisMaskTensor = (featureAxisXTensor == int(columnTensorData["axisXTensor"][columnTensorData["centralColumnIndex"]].item())) | (featureAxisYTensor == int(columnTensorData["axisYTensor"][columnTensorData["centralColumnIndex"]].item()))
		result = {"centralConceptName": columnTensorData["centralConceptName"], "centralColumnIndex": columnTensorData["centralColumnIndex"], "centralFieldX": int(columnTensorData["fieldXTensor"][columnTensorData["centralColumnIndex"]].item()), "centralFieldY": int(columnTensorData["fieldYTensor"][columnTensorData["centralColumnIndex"]].item()), "centralAxisX": int(columnTensorData["axisXTensor"][columnTensorData["centralColumnIndex"]].item()), "centralAxisY": int(columnTensorData["axisYTensor"][columnTensorData["centralColumnIndex"]].item()), "featureFieldXTensor": columnTensorData["fieldXTensor"].index_select(0, activeCoordinateTensor[:, 1]).to(dtype=pt.long), "featureFieldYTensor": columnTensorData["fieldYTensor"].index_select(0, activeCoordinateTensor[:, 1]).to(dtype=pt.long), "featureAxisXTensor": featureAxisXTensor, "featureAxisYTensor": featureAxisYTensor, "featureAxisMaskTensor": featureAxisMaskTensor.to(dtype=pt.bool)}
	else:
		raise RuntimeError("buildImageAxesSequenceTensorData error: requires submodalityName=='image' and modalityORimageSequenceEncode=='axes'")
	return result


def configureImageAxesSequenceConcepts(orderedConceptNameList, requiredSourceFeatureIndicesByConceptName, globalFeatureIndices, imageAxesSequenceTensorData):
	result = None
	centralConceptName = None
	globalFeatureIndexTensor = None
	uniqueGlobalFeatureIndexTensor = None
	if(submodalityName=="image" and modalityORimageSequenceEncode=="axes"):
		if(not isinstance(orderedConceptNameList, list)):
			raise RuntimeError("configureImageAxesSequenceConcepts error: orderedConceptNameList must be a list")
		if(not isinstance(requiredSourceFeatureIndicesByConceptName, dict)):
			raise RuntimeError("configureImageAxesSequenceConcepts error: requiredSourceFeatureIndicesByConceptName must be a dict")
		if(not isinstance(globalFeatureIndices, list)):
			raise RuntimeError("configureImageAxesSequenceConcepts error: globalFeatureIndices must be a list")
		if(not isinstance(imageAxesSequenceTensorData, dict)):
			raise RuntimeError("configureImageAxesSequenceConcepts error: imageAxesSequenceTensorData must be a dict")
		validateImageSequenceEncodeAxesColumnIndex()
		if(len(globalFeatureIndices) > 0):
			if("centralConceptName" not in imageAxesSequenceTensorData):
				raise RuntimeError("configureImageAxesSequenceConcepts error: imageAxesSequenceTensorData missing centralConceptName")
			centralConceptName = imageAxesSequenceTensorData["centralConceptName"]
			if(centralConceptName not in requiredSourceFeatureIndicesByConceptName):
				raise RuntimeError("configureImageAxesSequenceConcepts error: centralConceptName missing required source feature list")
			globalFeatureIndexTensor = pt.tensor(globalFeatureIndices, dtype=pt.long)
			if(globalFeatureIndexTensor.dim() != 1):
				raise RuntimeError("configureImageAxesSequenceConcepts error: globalFeatureIndexTensor must be rank 1")
			uniqueGlobalFeatureIndexTensor = pt.unique(globalFeatureIndexTensor)
			requiredSourceFeatureIndicesByConceptName.clear()
			requiredSourceFeatureIndicesByConceptName[centralConceptName] = uniqueGlobalFeatureIndexTensor.tolist()
			orderedConceptNameList[:] = [centralConceptName]
	else:
		raise RuntimeError("configureImageAxesSequenceConcepts error: requires submodalityName=='image' and modalityORimageSequenceEncode=='axes'")
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
	columnTensorData = None
	if(submodalityName=="image" and modalityORimageSequenceEncode=="axes"):
		columnTensorData = buildImageAxesColumnTensorData(columnMetadataList)
		result = columnTensorData["centralConceptName"]
	else:
		raise RuntimeError("findImageAxesCentralConceptName error: requires submodalityName=='image' and modalityORimageSequenceEncode=='axes'")
	return result


def initialiseImageAxesCoordinates(sequenceObservedColumns, sequenceData):
	result = None
	if(submodalityName=="image" and modalityORimageSequenceEncode=="axes"):
		initialiseImageAxesFeatureFieldCoordinates(sequenceObservedColumns, sequenceData)
		sequenceObservedColumns.trainConnectionsUseSpatialAxes = True
	else:
		raise RuntimeError("initialiseImageAxesCoordinates error: requires submodalityName=='image' and modalityORimageSequenceEncode=='axes'")
	return result


def initialiseImageAxesFeatureFieldCoordinates(sequenceObservedColumns, sequenceData):
	result = None
	featureFieldXTensor = None
	featureFieldYTensor = None
	if(submodalityName=="image" and modalityORimageSequenceEncode=="axes"):
		if("imageAxesFeatureFieldXTensor" not in sequenceData or "imageAxesFeatureFieldYTensor" not in sequenceData or "imageAxesFeatureAxisMaskTensor" not in sequenceData or "imageAxesCentralFieldX" not in sequenceData or "imageAxesCentralFieldY" not in sequenceData):
			raise RuntimeError("initialiseImageAxesFeatureFieldCoordinates error: sequenceData missing image axes feature field coordinate tensors")
		featureFieldXTensor = sequenceData["imageAxesFeatureFieldXTensor"]
		featureFieldYTensor = sequenceData["imageAxesFeatureFieldYTensor"]
		if(not pt.is_tensor(featureFieldXTensor) or not pt.is_tensor(featureFieldYTensor)):
			raise RuntimeError("initialiseImageAxesFeatureFieldCoordinates error: feature field coordinates must be tensors")
		if(featureFieldXTensor.dim() != 1 or featureFieldYTensor.dim() != 1):
			raise RuntimeError("initialiseImageAxesFeatureFieldCoordinates error: feature field coordinate tensors must be rank 1")
		if(int(featureFieldXTensor.shape[0]) != int(sequenceObservedColumns.fs) or int(featureFieldYTensor.shape[0]) != int(sequenceObservedColumns.fs)):
			raise RuntimeError("initialiseImageAxesFeatureFieldCoordinates error: feature field coordinate tensor lengths must equal fs")
		if(not pt.is_tensor(sequenceData["imageAxesFeatureAxisMaskTensor"]) or sequenceData["imageAxesFeatureAxisMaskTensor"].dim() != 1 or int(sequenceData["imageAxesFeatureAxisMaskTensor"].shape[0]) != int(sequenceObservedColumns.fs)):
			raise RuntimeError("initialiseImageAxesFeatureFieldCoordinates error: imageAxesFeatureAxisMaskTensor must be a rank 1 tensor with length fs")
		sequenceObservedColumns.imageAxesFeatureFieldXTensor = featureFieldXTensor.to(dtype=pt.long)
		sequenceObservedColumns.imageAxesFeatureFieldYTensor = featureFieldYTensor.to(dtype=pt.long)
		sequenceObservedColumns.imageAxesFeatureAxisMaskTensor = sequenceData["imageAxesFeatureAxisMaskTensor"].to(dtype=pt.bool)
		sequenceObservedColumns.imageAxesCentralFieldX = int(sequenceData["imageAxesCentralFieldX"])
		sequenceObservedColumns.imageAxesCentralFieldY = int(sequenceData["imageAxesCentralFieldY"])
	else:
		raise RuntimeError("initialiseImageAxesFeatureFieldCoordinates error: requires submodalityName=='image' and modalityORimageSequenceEncode=='axes'")
	return result


def configureTrainConnectionsForImageAxesEncoding(sequenceObservedColumns):
	result = None
	if(submodalityName=="image" and modalityORimageSequenceEncode=="axes"):
		sequenceObservedColumns.trainConnectionsIncludeSameTimeIndex = True
		sequenceObservedColumns.trainConnectionsUseSpatialAxes = True
	else:
		raise RuntimeError("configureTrainConnectionsForImageAxesEncoding error: requires submodalityName=='image' and modalityORimageSequenceEncode=='axes'")
	return result


def validateImageAxesSnapshotIndex(snapshotIndex):
	result = None
	if(submodalityName=="image" and modalityORimageSequenceEncode=="axes"):
		if(not isinstance(snapshotIndex, int) or isinstance(snapshotIndex, bool)):
			raise RuntimeError("validateImageAxesSnapshotIndex error: snapshotIndex must be an int")
		if(snapshotIndex != 0):
			raise RuntimeError("validateImageAxesSnapshotIndex error: snapshotIndex must be 0 when modalityORimageSequenceEncode is 'axes'")
	else:
		raise RuntimeError("validateImageAxesSnapshotIndex error: requires submodalityName=='image' and modalityORimageSequenceEncode=='axes'")
	return result


def getActiveSegmentsForImageAxesEncoding(sequenceObservedColumns, sequenceConceptIndex, targetDevice):
	result = None
	activeSegmentMask = None
	if(submodalityName=="image" and modalityORimageSequenceEncode=="axes"):
		if(not isinstance(sequenceConceptIndex, int) or isinstance(sequenceConceptIndex, bool)):
			raise RuntimeError("getActiveSegmentsForImageAxesEncoding error: sequenceConceptIndex must be an int")
		if(sequenceConceptIndex != int(modalityORimageSequenceEncodeAxesColumnIndex)):
			raise RuntimeError("getActiveSegmentsForImageAxesEncoding error: sequenceConceptIndex must equal modalityORimageSequenceEncodeAxesColumnIndex")
		activeSegmentMask = calculateImageAxesFeatureActiveSegmentMask(sequenceObservedColumns, targetDevice)
		result = pt.nonzero(pt.any(activeSegmentMask, dim=1), as_tuple=False).flatten()
	else:
		raise RuntimeError("getActiveSegmentsForImageAxesEncoding error: requires submodalityName=='image' and modalityORimageSequenceEncode=='axes'")
	return result


def populateImageAxesTrainTensors(sequenceObservedColumns, featureNeuronsActive, featureNeuronsWordOrder, targetDevice):
	result = None
	activeSegmentMask = None
	axesColumnIndex = None
	if(submodalityName=="image" and modalityORimageSequenceEncode=="axes"):
		if(not pt.is_tensor(featureNeuronsActive) or not pt.is_tensor(featureNeuronsWordOrder)):
			raise RuntimeError("populateImageAxesTrainTensors error: featureNeuronsActive and featureNeuronsWordOrder must be tensors")
		if(featureNeuronsActive.dim() != 4):
			raise RuntimeError("populateImageAxesTrainTensors error: featureNeuronsActive rank must be 4")
		if(featureNeuronsWordOrder.dim() != 2):
			raise RuntimeError("populateImageAxesTrainTensors error: featureNeuronsWordOrder rank must be 2")
		axesColumnIndex = int(modalityORimageSequenceEncodeAxesColumnIndex)
		if(int(featureNeuronsActive.shape[2]) <= axesColumnIndex or int(featureNeuronsWordOrder.shape[0]) <= axesColumnIndex):
			raise RuntimeError("populateImageAxesTrainTensors error: modalityORimageSequenceEncodeAxesColumnIndex out of range")
		activeSegmentMask = calculateImageAxesFeatureActiveSegmentMask(sequenceObservedColumns, targetDevice)
		if(int(activeSegmentMask.shape[1]) != int(featureNeuronsActive.shape[3])):
			raise RuntimeError("populateImageAxesTrainTensors error: active segment mask feature dimension mismatch")
		featureNeuronsActive[0, :, axesColumnIndex, :] = activeSegmentMask.to(dtype=featureNeuronsActive.dtype)
		featureNeuronsWordOrder[axesColumnIndex, :] = 0
	else:
		raise RuntimeError("populateImageAxesTrainTensors error: requires submodalityName=='image' and modalityORimageSequenceEncode=='axes'")
	return result


def calculateImageAxesFeatureActiveSegmentMask(sequenceObservedColumns, targetDevice):
	result = None
	fieldXTensor = None
	fieldYTensor = None
	maxDistanceX = None
	maxDistanceY = None
	maxSegmentIndexTensor = None
	segmentIndexTensor = None
	if(submodalityName=="image" and modalityORimageSequenceEncode=="axes"):
		fieldXTensor, fieldYTensor = getImageAxesFeatureFieldCoordinates(sequenceObservedColumns, targetDevice)
		maxDistanceX = pt.maximum(fieldXTensor, int(modalityORimageSequenceEncodeDistanceFieldSegments) - 1 - fieldXTensor)
		maxDistanceY = pt.maximum(fieldYTensor, int(modalityORimageSequenceEncodeDistanceFieldSegments) - 1 - fieldYTensor)
		maxSegmentIndexTensor = pt.ceil(pt.sqrt(((maxDistanceX*maxDistanceX) + (maxDistanceY*maxDistanceY)).to(arrayType))).long()
		if(bool(pt.any(maxSegmentIndexTensor < 0).item()) or bool(pt.any(maxSegmentIndexTensor >= int(arrayNumberOfSegments)).item())):
			raise RuntimeError("calculateImageAxesFeatureActiveSegmentMask error: calculated segment index out of range")
		segmentIndexTensor = pt.arange(int(arrayNumberOfSegments), device=targetDevice, dtype=pt.long).view(int(arrayNumberOfSegments), 1)
		result = segmentIndexTensor <= maxSegmentIndexTensor.view(1, int(maxSegmentIndexTensor.shape[0]))
	else:
		raise RuntimeError("calculateImageAxesFeatureActiveSegmentMask error: requires submodalityName=='image' and modalityORimageSequenceEncode=='axes'")
	return result


def getImageAxesFeatureFieldCoordinates(sequenceObservedColumns, targetDevice):
	result = None
	fieldXTensor = None
	fieldYTensor = None
	if(submodalityName=="image" and modalityORimageSequenceEncode=="axes"):
		if(not hasattr(sequenceObservedColumns, "imageAxesFeatureFieldXTensor") or not hasattr(sequenceObservedColumns, "imageAxesFeatureFieldYTensor")):
			raise RuntimeError("getImageAxesFeatureFieldCoordinates error: sequenceObservedColumns missing image axes feature field coordinate tensors")
		fieldXTensor = sequenceObservedColumns.imageAxesFeatureFieldXTensor
		fieldYTensor = sequenceObservedColumns.imageAxesFeatureFieldYTensor
		if(fieldXTensor is None or fieldYTensor is None):
			raise RuntimeError("getImageAxesFeatureFieldCoordinates error: feature field coordinate tensors must not be None")
		if(not pt.is_tensor(fieldXTensor) or not pt.is_tensor(fieldYTensor)):
			raise RuntimeError("getImageAxesFeatureFieldCoordinates error: feature field coordinate tensors must be tensors")
		if(fieldXTensor.dim() != 1 or fieldYTensor.dim() != 1):
			raise RuntimeError("getImageAxesFeatureFieldCoordinates error: feature field coordinate tensors must be rank 1")
		if(int(fieldXTensor.shape[0]) != int(sequenceObservedColumns.fs) or int(fieldYTensor.shape[0]) != int(sequenceObservedColumns.fs)):
			raise RuntimeError("getImageAxesFeatureFieldCoordinates error: feature field coordinate tensor lengths must equal fs")
		result = (fieldXTensor.to(device=targetDevice), fieldYTensor.to(device=targetDevice))
	else:
		raise RuntimeError("getImageAxesFeatureFieldCoordinates error: requires submodalityName=='image' and modalityORimageSequenceEncode=='axes'")
	return result


def buildImageAxesColumnTensorData(columnMetadataList):
	result = None
	conceptNameList = None
	xIndexTensor = None
	yIndexTensor = None
	maxXIndex = None
	maxYIndex = None
	gridWidth = None
	gridHeight = None
	fieldXTensor = None
	fieldYTensor = None
	distanceNumeratorTensor = None
	centralColumnIndex = None
	if(submodalityName=="image" and modalityORimageSequenceEncode=="axes"):
		validateImageAxesColumnMetadataList(columnMetadataList)
		conceptNameList = list(map(readImageAxesColumnMetadataConceptName, columnMetadataList))
		xIndexTensor = pt.tensor(list(map(readImageAxesColumnMetadataXIndex, columnMetadataList)), dtype=pt.long)
		yIndexTensor = pt.tensor(list(map(readImageAxesColumnMetadataYIndex, columnMetadataList)), dtype=pt.long)
		if(xIndexTensor.numel() == 0 or yIndexTensor.numel() == 0):
			raise RuntimeError("buildImageAxesColumnTensorData error: column metadata coordinate tensors must not be empty")
		maxXIndex = int(pt.max(xIndexTensor).item())
		maxYIndex = int(pt.max(yIndexTensor).item())
		gridWidth = maxXIndex + 1
		gridHeight = maxYIndex + 1
		fieldXTensor = pt.div(xIndexTensor*int(modalityORimageSequenceEncodeDistanceFieldSegments), gridWidth, rounding_mode='floor')
		fieldYTensor = pt.div(yIndexTensor*int(modalityORimageSequenceEncodeDistanceFieldSegments), gridHeight, rounding_mode='floor')
		distanceNumeratorTensor = ((2*xIndexTensor - maxXIndex)*(2*xIndexTensor - maxXIndex)) + ((2*yIndexTensor - maxYIndex)*(2*yIndexTensor - maxYIndex))
		centralColumnIndex = int(pt.argmin(distanceNumeratorTensor).item())
		validateImageAxesFieldCoordinateTensor(fieldXTensor, "fieldXTensor")
		validateImageAxesFieldCoordinateTensor(fieldYTensor, "fieldYTensor")
		result = {"conceptNameList": conceptNameList, "axisXTensor": xIndexTensor, "axisYTensor": yIndexTensor, "fieldXTensor": fieldXTensor, "fieldYTensor": fieldYTensor, "centralColumnIndex": centralColumnIndex, "centralConceptName": conceptNameList[centralColumnIndex]}
	else:
		raise RuntimeError("buildImageAxesColumnTensorData error: requires submodalityName=='image' and modalityORimageSequenceEncode=='axes'")
	return result


def validateImageAxesColumnMetadataList(columnMetadataList):
	result = None
	if(submodalityName=="image" and modalityORimageSequenceEncode=="axes"):
		if(not isinstance(columnMetadataList, list)):
			raise RuntimeError("validateImageAxesColumnMetadataList error: columnMetadataList must be a list")
		if(len(columnMetadataList) == 0):
			raise RuntimeError("validateImageAxesColumnMetadataList error: columnMetadataList must not be empty")
		list(map(validateImageAxesColumnMetadata, columnMetadataList))
	else:
		raise RuntimeError("validateImageAxesColumnMetadataList error: requires submodalityName=='image' and modalityORimageSequenceEncode=='axes'")
	return result


def validateImageAxesColumnMetadata(columnMetadata):
	result = None
	if(submodalityName=="image" and modalityORimageSequenceEncode=="axes"):
		if(not isinstance(columnMetadata, dict)):
			raise RuntimeError("validateImageAxesColumnMetadata error: columnMetadata must be a dict")
		if("conceptName" not in columnMetadata or "xIndex" not in columnMetadata or "yIndex" not in columnMetadata):
			raise RuntimeError("validateImageAxesColumnMetadata error: columnMetadata missing conceptName, xIndex, or yIndex")
		if(not isinstance(columnMetadata["conceptName"], str) or columnMetadata["conceptName"] == ""):
			raise RuntimeError("validateImageAxesColumnMetadata error: conceptName must be a non-empty string")
		if(not isinstance(columnMetadata["xIndex"], int) or isinstance(columnMetadata["xIndex"], bool) or not isinstance(columnMetadata["yIndex"], int) or isinstance(columnMetadata["yIndex"], bool)):
			raise RuntimeError("validateImageAxesColumnMetadata error: xIndex/yIndex must be ints")
		if(columnMetadata["xIndex"] < 0 or columnMetadata["yIndex"] < 0):
			raise RuntimeError("validateImageAxesColumnMetadata error: xIndex/yIndex must be >= 0")
	else:
		raise RuntimeError("validateImageAxesColumnMetadata error: requires submodalityName=='image' and modalityORimageSequenceEncode=='axes'")
	return result


def readImageAxesColumnMetadataConceptName(columnMetadata):
	result = None
	validateImageAxesColumnMetadata(columnMetadata)
	result = columnMetadata["conceptName"]
	return result


def readImageAxesColumnMetadataXIndex(columnMetadata):
	result = None
	validateImageAxesColumnMetadata(columnMetadata)
	result = int(columnMetadata["xIndex"])
	return result


def readImageAxesColumnMetadataYIndex(columnMetadata):
	result = None
	validateImageAxesColumnMetadata(columnMetadata)
	result = int(columnMetadata["yIndex"])
	return result


def validateImageAxesFieldCoordinateTensor(fieldCoordinateTensor, fieldName):
	result = None
	if(submodalityName=="image" and modalityORimageSequenceEncode=="axes"):
		if(not pt.is_tensor(fieldCoordinateTensor)):
			raise RuntimeError("validateImageAxesFieldCoordinateTensor error: " + fieldName + " must be a tensor")
		if(fieldCoordinateTensor.dim() != 1):
			raise RuntimeError("validateImageAxesFieldCoordinateTensor error: " + fieldName + " must be rank 1")
		if(bool(pt.any(fieldCoordinateTensor < 0).item()) or bool(pt.any(fieldCoordinateTensor >= int(modalityORimageSequenceEncodeDistanceFieldSegments)).item())):
			raise RuntimeError("validateImageAxesFieldCoordinateTensor error: " + fieldName + " contains out of range coordinates")
	else:
		raise RuntimeError("validateImageAxesFieldCoordinateTensor error: requires submodalityName=='image' and modalityORimageSequenceEncode=='axes'")
	return result
