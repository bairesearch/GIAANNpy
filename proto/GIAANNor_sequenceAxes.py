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
	featureCentralColumnMaskTensor = None
	centralAxisX = None
	centralAxisY = None
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
		centralAxisX = int(columnTensorData["axisXTensor"][columnTensorData["centralColumnIndex"]].item())
		centralAxisY = int(columnTensorData["axisYTensor"][columnTensorData["centralColumnIndex"]].item())
		featureAxisMaskTensor = calculateImageAxesFeatureAxisMaskTensor(featureAxisXTensor, featureAxisYTensor, centralAxisX, centralAxisY)
		featureCentralColumnMaskTensor = calculateImageAxesFeatureCentralColumnMaskTensor(featureAxisXTensor, featureAxisYTensor, centralAxisX, centralAxisY)
		result = {"centralConceptName": columnTensorData["centralConceptName"], "centralColumnIndex": columnTensorData["centralColumnIndex"], "centralFieldX": int(columnTensorData["fieldXTensor"][columnTensorData["centralColumnIndex"]].item()), "centralFieldY": int(columnTensorData["fieldYTensor"][columnTensorData["centralColumnIndex"]].item()), "centralAxisX": centralAxisX, "centralAxisY": centralAxisY, "featureFieldXTensor": columnTensorData["fieldXTensor"].index_select(0, activeCoordinateTensor[:, 1]).to(dtype=pt.long), "featureFieldYTensor": columnTensorData["fieldYTensor"].index_select(0, activeCoordinateTensor[:, 1]).to(dtype=pt.long), "featureAxisXTensor": featureAxisXTensor, "featureAxisYTensor": featureAxisYTensor, "featureAxisMaskTensor": featureAxisMaskTensor.to(dtype=pt.bool), "featureCentralColumnMaskTensor": featureCentralColumnMaskTensor.to(dtype=pt.bool)}
	else:
		raise RuntimeError("buildImageAxesSequenceTensorData error: requires submodalityName=='image' and modalityORimageSequenceEncode=='axes'")
	return result


def calculateImageAxesFeatureCentralColumnMaskTensor(featureAxisXTensor, featureAxisYTensor, centralAxisX, centralAxisY):
	result = None
	if(submodalityName=="image" and modalityORimageSequenceEncode=="axes"):
		if(not pt.is_tensor(featureAxisXTensor) or not pt.is_tensor(featureAxisYTensor)):
			raise RuntimeError("calculateImageAxesFeatureCentralColumnMaskTensor error: feature axis coordinates must be tensors")
		if(featureAxisXTensor.dim() != 1 or featureAxisYTensor.dim() != 1):
			raise RuntimeError("calculateImageAxesFeatureCentralColumnMaskTensor error: feature axis coordinate tensors must be rank 1")
		if(int(featureAxisXTensor.shape[0]) != int(featureAxisYTensor.shape[0])):
			raise RuntimeError("calculateImageAxesFeatureCentralColumnMaskTensor error: feature axis coordinate tensor lengths must match")
		if(not isinstance(centralAxisX, int) or isinstance(centralAxisX, bool) or not isinstance(centralAxisY, int) or isinstance(centralAxisY, bool)):
			raise RuntimeError("calculateImageAxesFeatureCentralColumnMaskTensor error: central axis coordinates must be ints")
		result = (featureAxisXTensor == int(centralAxisX)) & (featureAxisYTensor == int(centralAxisY))
	else:
		raise RuntimeError("calculateImageAxesFeatureCentralColumnMaskTensor error: requires submodalityName=='image' and modalityORimageSequenceEncode=='axes'")
	return result


def calculateImageAxesFeatureAxisMaskTensor(featureAxisXTensor, featureAxisYTensor, centralAxisX, centralAxisY):
	result = None
	featureAxisTensorList = None
	centralAxisList = None
	axisIndex = None
	axisMaskTensor = None
	if(submodalityName=="image" and modalityORimageSequenceEncode=="axes"):
		if(not pt.is_tensor(featureAxisXTensor) or not pt.is_tensor(featureAxisYTensor)):
			raise RuntimeError("calculateImageAxesFeatureAxisMaskTensor error: feature axis coordinates must be tensors")
		if(featureAxisXTensor.dim() != 1 or featureAxisYTensor.dim() != 1):
			raise RuntimeError("calculateImageAxesFeatureAxisMaskTensor error: feature axis coordinate tensors must be rank 1")
		if(int(featureAxisXTensor.shape[0]) != int(featureAxisYTensor.shape[0])):
			raise RuntimeError("calculateImageAxesFeatureAxisMaskTensor error: feature axis coordinate tensor lengths must match")
		if(not isinstance(centralAxisX, int) or isinstance(centralAxisX, bool) or not isinstance(centralAxisY, int) or isinstance(centralAxisY, bool)):
			raise RuntimeError("calculateImageAxesFeatureAxisMaskTensor error: central axis coordinates must be ints")
		featureAxisTensorList = [featureAxisXTensor, featureAxisYTensor]
		centralAxisList = [centralAxisX, centralAxisY]
		result = pt.zeros_like(featureAxisXTensor, dtype=pt.bool)
		for axisIndex in range(2):
			axisMaskTensor = featureAxisTensorList[axisIndex] == int(centralAxisList[axisIndex])
			result = result | axisMaskTensor
	else:
		raise RuntimeError("calculateImageAxesFeatureAxisMaskTensor error: requires submodalityName=='image' and modalityORimageSequenceEncode=='axes'")
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
			validateImageSequenceEncodeAxesSourceColumnIndex()
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


def buildImageAxesSequenceConceptMetadataList(columnMetadataList, imageAxesSequenceTensorData):
	result = None
	centralColumnIndex = None
	orderedColumnMetadataList = None
	columnIndex = None
	columnMetadata = None
	if(submodalityName=="image" and modalityORimageSequenceEncode=="axes"):
		validateImageAxesColumnMetadataList(columnMetadataList)
		if(not isinstance(imageAxesSequenceTensorData, dict)):
			raise RuntimeError("buildImageAxesSequenceConceptMetadataList error: imageAxesSequenceTensorData must be a dict")
		if("centralColumnIndex" not in imageAxesSequenceTensorData):
			raise RuntimeError("buildImageAxesSequenceConceptMetadataList error: imageAxesSequenceTensorData missing centralColumnIndex")
		centralColumnIndex = int(imageAxesSequenceTensorData["centralColumnIndex"])
		if(centralColumnIndex < 0 or centralColumnIndex >= len(columnMetadataList)):
			raise RuntimeError("buildImageAxesSequenceConceptMetadataList error: centralColumnIndex out of range")
		if(modalityORimageSequenceEncodeAxesColumnRandom):
			validateImageSequenceEncodeAxesColumnRandomParameters(len(columnMetadataList))
			orderedColumnMetadataList = [columnMetadataList[centralColumnIndex]]
			for columnIndex, columnMetadata in enumerate(columnMetadataList):
				if(columnIndex != centralColumnIndex and len(orderedColumnMetadataList) < int(modalityORnumberOfColumnsV2)):
					orderedColumnMetadataList.append(columnMetadata)
			if(len(orderedColumnMetadataList) != int(modalityORnumberOfColumnsV2)):
				raise RuntimeError("buildImageAxesSequenceConceptMetadataList error: orderedColumnMetadataList length must equal modalityORnumberOfColumnsV2")
			result = orderedColumnMetadataList
		else:
			validateImageSequenceEncodeAxesTargetColumnIndex()
			result = [columnMetadataList[centralColumnIndex]]
	else:
		raise RuntimeError("buildImageAxesSequenceConceptMetadataList error: requires submodalityName=='image' and modalityORimageSequenceEncode=='axes'")
	return result


def buildImageAxesRequiredSourceFeatureIndicesByConceptName(orderedConceptNameList, uniqueGlobalFeatureIndices):
	result = None
	uniqueGlobalFeatureIndexTensor = None
	requiredSourceFeatureIndices = None
	conceptName = None
	if(submodalityName=="image" and modalityORimageSequenceEncode=="axes"):
		if(not isinstance(orderedConceptNameList, list)):
			raise RuntimeError("buildImageAxesRequiredSourceFeatureIndicesByConceptName error: orderedConceptNameList must be a list")
		if(not isinstance(uniqueGlobalFeatureIndices, list)):
			raise RuntimeError("buildImageAxesRequiredSourceFeatureIndicesByConceptName error: uniqueGlobalFeatureIndices must be a list")
		if(len(orderedConceptNameList) == 0):
			raise RuntimeError("buildImageAxesRequiredSourceFeatureIndicesByConceptName error: orderedConceptNameList must not be empty")
		if(len(uniqueGlobalFeatureIndices) == 0):
			raise RuntimeError("buildImageAxesRequiredSourceFeatureIndicesByConceptName error: uniqueGlobalFeatureIndices must not be empty")
		uniqueGlobalFeatureIndexTensor = pt.tensor(uniqueGlobalFeatureIndices, dtype=pt.long)
		if(uniqueGlobalFeatureIndexTensor.dim() != 1):
			raise RuntimeError("buildImageAxesRequiredSourceFeatureIndicesByConceptName error: uniqueGlobalFeatureIndexTensor must be rank 1")
		requiredSourceFeatureIndices = pt.unique(uniqueGlobalFeatureIndexTensor).tolist()
		result = {}
		for conceptName in orderedConceptNameList:
			if(not isinstance(conceptName, str) or conceptName == ""):
				raise RuntimeError("buildImageAxesRequiredSourceFeatureIndicesByConceptName error: conceptName must be a non-empty string")
			result[conceptName] = list(requiredSourceFeatureIndices)
	else:
		raise RuntimeError("buildImageAxesRequiredSourceFeatureIndicesByConceptName error: requires submodalityName=='image' and modalityORimageSequenceEncode=='axes'")
	return result


def validateImageSequenceEncodeAxesTargetColumnIndex():
	result = None
	if(submodalityName=="image" and modalityORimageSequenceEncode=="axes"):
		validateImageSequenceEncodeAxesSourceColumnIndex()
		if(not isinstance(modalityORimageSequenceEncodeAxesTargetColumnIndex, int) or isinstance(modalityORimageSequenceEncodeAxesTargetColumnIndex, bool)):
			raise RuntimeError("validateImageSequenceEncodeAxesTargetColumnIndex error: modalityORimageSequenceEncodeAxesTargetColumnIndex must be an int")
		if(modalityORimageSequenceEncodeAxesTargetColumnIndex != modalityORimageSequenceEncodeAxesSourceColumnIndex):
			raise RuntimeError("validateImageSequenceEncodeAxesTargetColumnIndex error: modalityORimageSequenceEncodeAxesTargetColumnIndex must equal modalityORimageSequenceEncodeAxesSourceColumnIndex")
	else:
		raise RuntimeError("validateImageSequenceEncodeAxesTargetColumnIndex error: requires submodalityName=='image' and modalityORimageSequenceEncode=='axes'")
	return result


def validateImageSequenceEncodeAxesSourceColumnIndex():
	result = None
	if(submodalityName=="image" and modalityORimageSequenceEncode=="axes"):
		if(not isinstance(modalityORimageSequenceEncodeAxesSourceColumnIndex, int) or isinstance(modalityORimageSequenceEncodeAxesSourceColumnIndex, bool)):
			raise RuntimeError("validateImageSequenceEncodeAxesSourceColumnIndex error: modalityORimageSequenceEncodeAxesSourceColumnIndex must be an int")
		if(modalityORimageSequenceEncodeAxesSourceColumnIndex != 0):
			raise RuntimeError("validateImageSequenceEncodeAxesSourceColumnIndex error: modalityORimageSequenceEncodeAxesSourceColumnIndex must equal 0")
	else:
		raise RuntimeError("validateImageSequenceEncodeAxesSourceColumnIndex error: requires submodalityName=='image' and modalityORimageSequenceEncode=='axes'")
	return result


def validateImageSequenceEncodeAxesColumnRandomParameters(numberOfAvailableColumns):
	result = None
	if(submodalityName=="image" and modalityORimageSequenceEncode=="axes"):
		if(not isinstance(modalityORimageSequenceEncodeAxesColumnRandom, bool)):
			raise RuntimeError("validateImageSequenceEncodeAxesColumnRandomParameters error: modalityORimageSequenceEncodeAxesColumnRandom must be a bool")
		if(not isinstance(modalityORnumberOfColumnsV2, int) or isinstance(modalityORnumberOfColumnsV2, bool)):
			raise RuntimeError("validateImageSequenceEncodeAxesColumnRandomParameters error: modalityORnumberOfColumnsV2 must be an int")
		if(not isinstance(numberOfAvailableColumns, int) or isinstance(numberOfAvailableColumns, bool)):
			raise RuntimeError("validateImageSequenceEncodeAxesColumnRandomParameters error: numberOfAvailableColumns must be an int")
		if(modalityORnumberOfColumnsV2 <= 0):
			raise RuntimeError("validateImageSequenceEncodeAxesColumnRandomParameters error: modalityORnumberOfColumnsV2 must be > 0")
		if(numberOfAvailableColumns < int(modalityORnumberOfColumnsV2)):
			raise RuntimeError("validateImageSequenceEncodeAxesColumnRandomParameters error: numberOfAvailableColumns must be >= modalityORnumberOfColumnsV2")
		validateImageSequenceEncodeAxesSourceColumnIndex()
	else:
		raise RuntimeError("validateImageSequenceEncodeAxesColumnRandomParameters error: requires submodalityName=='image' and modalityORimageSequenceEncode=='axes'")
	return result


def validateImageSequenceEncodeAxesColumnCount(cs):
	result = None
	if(submodalityName=="image" and modalityORimageSequenceEncode=="axes"):
		if(not isinstance(cs, int) or isinstance(cs, bool)):
			raise RuntimeError("validateImageSequenceEncodeAxesColumnCount error: cs must be an int")
		if(cs <= 0):
			raise RuntimeError("validateImageSequenceEncodeAxesColumnCount error: cs must be > 0")
		validateImageSequenceEncodeAxesSourceColumnIndex()
		if(modalityORimageSequenceEncodeAxesSourceColumnIndex < 0 or modalityORimageSequenceEncodeAxesSourceColumnIndex >= cs):
			raise RuntimeError("validateImageSequenceEncodeAxesColumnCount error: modalityORimageSequenceEncodeAxesSourceColumnIndex out of range")
		if(modalityORimageSequenceEncodeAxesColumnRandom):
			validateImageSequenceEncodeAxesColumnRandomParameters(cs)
			if(cs != int(modalityORnumberOfColumnsV2)):
				raise RuntimeError("validateImageSequenceEncodeAxesColumnCount error: cs must equal modalityORnumberOfColumnsV2")
		else:
			validateImageSequenceEncodeAxesTargetColumnIndex()
			if(modalityORimageSequenceEncodeAxesTargetColumnIndex < 0 or modalityORimageSequenceEncodeAxesTargetColumnIndex >= cs):
				raise RuntimeError("validateImageSequenceEncodeAxesColumnCount error: modalityORimageSequenceEncodeAxesTargetColumnIndex out of range")
	else:
		raise RuntimeError("validateImageSequenceEncodeAxesColumnCount error: requires submodalityName=='image' and modalityORimageSequenceEncode=='axes'")
	return result


def generateImageAxesRandomTargetColumnIndexTensor(numberOfFeatures, targetDevice):
	result = None
	if(submodalityName=="image" and modalityORimageSequenceEncode=="axes" and modalityORimageSequenceEncodeAxesColumnRandom):
		if(not isinstance(numberOfFeatures, int) or isinstance(numberOfFeatures, bool)):
			raise RuntimeError("generateImageAxesRandomTargetColumnIndexTensor error: numberOfFeatures must be an int")
		if(numberOfFeatures <= 0):
			raise RuntimeError("generateImageAxesRandomTargetColumnIndexTensor error: numberOfFeatures must be > 0")
		validateImageSequenceEncodeAxesColumnRandomParameters(int(modalityORnumberOfColumnsV2))
		result = pt.randint(int(modalityORimageSequenceEncodeAxesSourceColumnIndex), int(modalityORnumberOfColumnsV2), (numberOfFeatures,), dtype=pt.long, device=targetDevice)
	else:
		raise RuntimeError("generateImageAxesRandomTargetColumnIndexTensor error: requires random modalityORimageSequenceEncode=='axes'")
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
		if("imageAxesFeatureFieldXTensor" not in sequenceData or "imageAxesFeatureFieldYTensor" not in sequenceData or "imageAxesFeatureAxisMaskTensor" not in sequenceData or "imageAxesFeatureCentralColumnMaskTensor" not in sequenceData or "imageAxesCentralFieldX" not in sequenceData or "imageAxesCentralFieldY" not in sequenceData):
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
		if(not pt.is_tensor(sequenceData["imageAxesFeatureCentralColumnMaskTensor"]) or sequenceData["imageAxesFeatureCentralColumnMaskTensor"].dim() != 1 or int(sequenceData["imageAxesFeatureCentralColumnMaskTensor"].shape[0]) != int(sequenceObservedColumns.fs)):
			raise RuntimeError("initialiseImageAxesFeatureFieldCoordinates error: imageAxesFeatureCentralColumnMaskTensor must be a rank 1 tensor with length fs")
		sequenceObservedColumns.imageAxesFeatureFieldXTensor = featureFieldXTensor.to(dtype=pt.long)
		sequenceObservedColumns.imageAxesFeatureFieldYTensor = featureFieldYTensor.to(dtype=pt.long)
		sequenceObservedColumns.imageAxesFeatureAxisMaskTensor = sequenceData["imageAxesFeatureAxisMaskTensor"].to(dtype=pt.bool)
		sequenceObservedColumns.imageAxesFeatureCentralColumnMaskTensor = sequenceData["imageAxesFeatureCentralColumnMaskTensor"].to(dtype=pt.bool)
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
		validateImageSequenceEncodeAxesColumnCount(int(sequenceObservedColumns.cs))
		if(modalityORimageSequenceEncodeAxesColumnRandom):
			if(sequenceConceptIndex < int(modalityORimageSequenceEncodeAxesSourceColumnIndex) or sequenceConceptIndex >= int(modalityORnumberOfColumnsV2)):
				raise RuntimeError("getActiveSegmentsForImageAxesEncoding error: sequenceConceptIndex out of random axes column range")
		else:
			if(sequenceConceptIndex != int(modalityORimageSequenceEncodeAxesTargetColumnIndex)):
				raise RuntimeError("getActiveSegmentsForImageAxesEncoding error: sequenceConceptIndex must equal modalityORimageSequenceEncodeAxesTargetColumnIndex")
		activeSegmentMask = calculateImageAxesFeatureActiveSegmentMask(sequenceObservedColumns, targetDevice)
		result = pt.nonzero(pt.any(activeSegmentMask, dim=1), as_tuple=False).flatten()
	else:
		raise RuntimeError("getActiveSegmentsForImageAxesEncoding error: requires submodalityName=='image' and modalityORimageSequenceEncode=='axes'")
	return result


def populateImageAxesTrainTensors(sequenceObservedColumns, featureNeuronsActive, featureNeuronsWordOrder, targetDevice):
	result = None
	activeSegmentMask = None
	featureCentralColumnMaskTensor = None
	axesTargetColumnIndex = None
	targetColumnIndexTensor = None
	targetColumnMaskTensor = None
	featureIndexTensor = None
	if(submodalityName=="image" and modalityORimageSequenceEncode=="axes"):
		if(not pt.is_tensor(featureNeuronsActive) or not pt.is_tensor(featureNeuronsWordOrder)):
			raise RuntimeError("populateImageAxesTrainTensors error: featureNeuronsActive and featureNeuronsWordOrder must be tensors")
		if(featureNeuronsActive.dim() != 4):
			raise RuntimeError("populateImageAxesTrainTensors error: featureNeuronsActive rank must be 4")
		if(featureNeuronsWordOrder.dim() != 2):
			raise RuntimeError("populateImageAxesTrainTensors error: featureNeuronsWordOrder rank must be 2")
		validateImageSequenceEncodeAxesColumnCount(int(featureNeuronsActive.shape[2]))
		activeSegmentMask = calculateImageAxesFeatureActiveSegmentMask(sequenceObservedColumns, targetDevice)
		if(int(activeSegmentMask.shape[1]) != int(featureNeuronsActive.shape[3])):
			raise RuntimeError("populateImageAxesTrainTensors error: active segment mask feature dimension mismatch")
		featureCentralColumnMaskTensor = getImageAxesFeatureCentralColumnMask(sequenceObservedColumns, targetDevice)
		activeSegmentMask = activeSegmentMask & featureCentralColumnMaskTensor.view(1, int(featureCentralColumnMaskTensor.shape[0]))
		if(modalityORimageSequenceEncodeAxesColumnRandom):
			if(int(featureNeuronsActive.shape[2]) != int(modalityORnumberOfColumnsV2) or int(featureNeuronsWordOrder.shape[0]) != int(modalityORnumberOfColumnsV2)):
				raise RuntimeError("populateImageAxesTrainTensors error: axes random tensor column count must equal modalityORnumberOfColumnsV2")
			featureIndexTensor = pt.arange(int(featureNeuronsActive.shape[3]), dtype=pt.long, device=targetDevice)
			targetColumnIndexTensor = generateImageAxesRandomTargetColumnIndexTensor(int(featureNeuronsActive.shape[3]), targetDevice)
			targetColumnMaskTensor = pt.zeros((int(modalityORnumberOfColumnsV2), int(featureNeuronsActive.shape[3])), dtype=pt.bool, device=targetDevice)
			targetColumnMaskTensor[targetColumnIndexTensor, featureIndexTensor] = True
			featureNeuronsActive[0, :, :, :] = activeSegmentMask.view(int(activeSegmentMask.shape[0]), 1, int(activeSegmentMask.shape[1])).to(dtype=featureNeuronsActive.dtype) * targetColumnMaskTensor.view(1, int(targetColumnMaskTensor.shape[0]), int(targetColumnMaskTensor.shape[1])).to(dtype=featureNeuronsActive.dtype)
			featureNeuronsWordOrder[targetColumnMaskTensor] = 0
			sequenceObservedColumns.imageAxesTargetColumnIndexTensor = targetColumnIndexTensor
		else:
			axesTargetColumnIndex = int(modalityORimageSequenceEncodeAxesTargetColumnIndex)
			if(int(featureNeuronsActive.shape[2]) <= axesTargetColumnIndex or int(featureNeuronsWordOrder.shape[0]) <= axesTargetColumnIndex):
				raise RuntimeError("populateImageAxesTrainTensors error: modalityORimageSequenceEncodeAxesTargetColumnIndex out of range")
			featureNeuronsActive[0, :, axesTargetColumnIndex, :] = activeSegmentMask.to(dtype=featureNeuronsActive.dtype)
			featureNeuronsWordOrder[axesTargetColumnIndex, :] = 0
	else:
		raise RuntimeError("populateImageAxesTrainTensors error: requires submodalityName=='image' and modalityORimageSequenceEncode=='axes'")
	return result


def getImageAxesFeatureCentralColumnMask(sequenceObservedColumns, targetDevice):
	result = None
	featureCentralColumnMaskTensor = None
	if(submodalityName=="image" and modalityORimageSequenceEncode=="axes"):
		if(not hasattr(sequenceObservedColumns, "imageAxesFeatureCentralColumnMaskTensor")):
			raise RuntimeError("getImageAxesFeatureCentralColumnMask error: sequenceObservedColumns missing imageAxesFeatureCentralColumnMaskTensor")
		featureCentralColumnMaskTensor = sequenceObservedColumns.imageAxesFeatureCentralColumnMaskTensor
		if(featureCentralColumnMaskTensor is None):
			raise RuntimeError("getImageAxesFeatureCentralColumnMask error: imageAxesFeatureCentralColumnMaskTensor must not be None")
		if(not pt.is_tensor(featureCentralColumnMaskTensor)):
			raise RuntimeError("getImageAxesFeatureCentralColumnMask error: imageAxesFeatureCentralColumnMaskTensor must be a tensor")
		if(featureCentralColumnMaskTensor.dim() != 1):
			raise RuntimeError("getImageAxesFeatureCentralColumnMask error: imageAxesFeatureCentralColumnMaskTensor must be rank 1")
		if(int(featureCentralColumnMaskTensor.shape[0]) != int(sequenceObservedColumns.fs)):
			raise RuntimeError("getImageAxesFeatureCentralColumnMask error: imageAxesFeatureCentralColumnMaskTensor length must equal fs")
		result = featureCentralColumnMaskTensor.to(device=targetDevice, dtype=pt.bool)
	else:
		raise RuntimeError("getImageAxesFeatureCentralColumnMask error: requires submodalityName=='image' and modalityORimageSequenceEncode=='axes'")
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
