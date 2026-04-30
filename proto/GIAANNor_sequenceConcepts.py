"""GIAANNor_sequenceConcepts.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN OR sequence Concepts (and feature detection)

"""

import math
import torch as pt

from GIAANNcmn_globalDefs import *
import GIAANNcmn_databaseNetwork
import GIAANNcmn_databaseNetworkTrain
import GIAANNor_RFfilters
import GIAANNor_sequenceObservedColumns


def ensureConceptColumns(databaseNetworkObject, columnMetadataList, allowNewFeatures):
	result = False
	for columnMetadata in columnMetadataList:
		conceptName = columnMetadata["conceptName"]
		if(conceptName not in databaseNetworkObject.conceptColumnsDict):
			if(not allowNewFeatures):
				raise RuntimeError("ensureConceptColumns error: conceptName not found while allowNewFeatures is False (" + conceptName + ")")
			GIAANNcmn_databaseNetwork.addConceptToConceptColumnsDict(databaseNetworkObject, conceptName, False, False)
			result = True
	return result


def ensureFeatureIndex(databaseNetworkObject, featureWord, allowNewFeatures):
	result = None
	if(featureWord in databaseNetworkObject.conceptFeaturesDict):
		result = int(databaseNetworkObject.conceptFeaturesDict[featureWord])
	else:
		if(not allowNewFeatures):
			raise RuntimeError("ensureFeatureIndex error: featureWord not found while allowNewFeatures is False (" + featureWord + ")")
		featureIndex = int(databaseNetworkObject.f)
		databaseNetworkObject.conceptFeaturesDict[featureWord] = featureIndex
		databaseNetworkObject.conceptFeaturesList.append(featureWord)
		if(trainStoreFeatureMapsGlobally):
			databaseNetworkObject.conceptFeaturesIndexToWordDict[featureIndex] = featureWord
		databaseNetworkObject.f = databaseNetworkObject.f + 1
		result = featureIndex
	return result


def generateSequenceData(databaseNetworkObject, columnMetadataList, selectedFilterIndices, rfFilters, allowNewFeatures):
	# create a new function to generate the sequence data in the correct format expected by GIAANNcmn_databaseNetworkTrain.
	result = None
	orderedConceptNameList = []
	activationList = []
	featureWords = []
	globalFeatureIndices = []
	requiredSourceFeatureIndicesByConceptName = {}
	imageDistanceFieldCoordinatesByConceptName = None
	seenConceptNames = set()
	if(not pt.is_tensor(selectedFilterIndices)):
		raise RuntimeError("generateSequenceData error: selectedFilterIndices must be a tensor")
	if(selectedFilterIndices.dim() != 2):
		raise RuntimeError("generateSequenceData error: selectedFilterIndices rank must be 2")
	if(selectedFilterIndices.shape[1] != len(columnMetadataList)):
		raise RuntimeError("generateSequenceData error: selectedFilterIndices column count mismatch")
	if(submodalityName=="image" and modalityORimageSequenceEncode=="distance"):
		imageDistanceFieldCoordinatesByConceptName = buildImageDistanceFieldCoordinatesByConceptName(columnMetadataList)
	ensureConceptColumns(databaseNetworkObject, columnMetadataList, allowNewFeatures)
	for columnMetadata in columnMetadataList:
		requiredSourceFeatureIndicesByConceptName[columnMetadata["conceptName"]] = []
	for snapshotIndex in range(selectedFilterIndices.shape[0]):
		for columnIndex, columnMetadata in enumerate(columnMetadataList):
			rfFilterIndex = int(selectedFilterIndices[snapshotIndex, columnIndex].item())
			if(rfFilterIndex >= 0):
				conceptName = columnMetadata["conceptName"]
				featureWord = GIAANNor_RFfilters.convertRFfilterIndexToASCIItext(rfFilters, rfFilterIndex)
				featureWordVerbose = GIAANNor_RFfilters.convertRFfilterIndexToASCIItextVerbose(rfFilters, rfFilterIndex)
				globalFeatureIndex = ensureFeatureIndex(databaseNetworkObject, featureWord, allowNewFeatures)
				if(conceptName not in seenConceptNames):
					orderedConceptNameList.append(conceptName)
					seenConceptNames.add(conceptName)
				requiredSourceFeatureIndicesByConceptName[conceptName].append(globalFeatureIndex)
				activationList.append({"snapshotIndex": snapshotIndex, "columnIndex": columnIndex, "conceptName": conceptName, "featureWord": featureWord, "featureWordVerbose": featureWordVerbose, "globalFeatureIndex": globalFeatureIndex})
				featureWords.append(featureWord)
				globalFeatureIndices.append(globalFeatureIndex)
	for conceptName in list(requiredSourceFeatureIndicesByConceptName.keys()):
		requiredSourceFeatureIndicesByConceptName[conceptName] = sorted(set(requiredSourceFeatureIndicesByConceptName[conceptName]))
	if(len(activationList) > 0):
		columnConceptIndexMap = {}
		for sequenceConceptIndex, conceptName in enumerate(orderedConceptNameList):
			columnConceptIndexMap[conceptName] = sequenceConceptIndex
		for localFeatureIndex, activation in enumerate(activationList):
			activation["localFeatureIndex"] = localFeatureIndex
			activation["sequenceConceptIndex"] = int(columnConceptIndexMap[activation["conceptName"]])
		result = {"orderedConceptNameList": orderedConceptNameList, "activationList": activationList, "featureWords": featureWords, "globalFeatureIndices": globalFeatureIndices, "requiredSourceFeatureIndicesByConceptName": requiredSourceFeatureIndicesByConceptName, "numberOfSnapshots": int(selectedFilterIndices.shape[0]), "numberOfColumns": int(selectedFilterIndices.shape[1])}
		if(submodalityName=="image" and modalityORimageSequenceEncode=="distance"):
			result["imageDistanceFieldCoordinatesByConceptName"] = imageDistanceFieldCoordinatesByConceptName
	return result


def buildImageDistanceFieldCoordinatesByConceptName(columnMetadataList):
	result = {}
	gridWidth = None
	gridHeight = None
	xIndex = None
	yIndex = None
	fieldXIndex = None
	fieldYIndex = None
	if(submodalityName=="image" and modalityORimageSequenceEncode=="distance"):
		validateImageDistanceFieldCoordinateParameters(columnMetadataList)
		gridWidth, gridHeight = calculateImageDistanceFieldGridDimensions(columnMetadataList)
		for columnMetadata in columnMetadataList:
			xIndex = int(columnMetadata["xIndex"])
			yIndex = int(columnMetadata["yIndex"])
			fieldXIndex = calculateImageDistanceFieldCoordinate(xIndex, gridWidth)
			fieldYIndex = calculateImageDistanceFieldCoordinate(yIndex, gridHeight)
			result[columnMetadata["conceptName"]] = (fieldXIndex, fieldYIndex)
	else:
		raise RuntimeError("buildImageDistanceFieldCoordinatesByConceptName error: requires submodalityName=='image' and modalityORimageSequenceEncode=='distance'")
	return result


def validateImageDistanceFieldCoordinateParameters(columnMetadataList):
	result = None
	if(submodalityName=="image" and modalityORimageSequenceEncode=="distance"):
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
		raise RuntimeError("validateImageDistanceFieldCoordinateParameters error: requires submodalityName=='image' and modalityORimageSequenceEncode=='distance'")
	return result


def calculateImageDistanceFieldGridDimensions(columnMetadataList):
	result = None
	maxXIndex = None
	maxYIndex = None
	if(submodalityName=="image" and modalityORimageSequenceEncode=="distance"):
		validateImageDistanceFieldCoordinateParameters(columnMetadataList)
		maxXIndex = max(int(columnMetadata["xIndex"]) for columnMetadata in columnMetadataList)
		maxYIndex = max(int(columnMetadata["yIndex"]) for columnMetadata in columnMetadataList)
		result = (maxXIndex + 1, maxYIndex + 1)
	else:
		raise RuntimeError("calculateImageDistanceFieldGridDimensions error: requires submodalityName=='image' and modalityORimageSequenceEncode=='distance'")
	return result


def calculateImageDistanceFieldCoordinate(coordinateIndex, numberOfFieldColumns):
	result = None
	if(submodalityName=="image" and modalityORimageSequenceEncode=="distance"):
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
		raise RuntimeError("calculateImageDistanceFieldCoordinate error: requires submodalityName=='image' and modalityORimageSequenceEncode=='distance'")
	return result


def generateSequenceDataText(sequenceData):
	result = None
	snapshotTextList = []
	snapshotActivationDict = {}
	maxFeatureCount = None
	activationKeySet = set()
	if(sequenceData is None):
		raise RuntimeError("generateSequenceDataText error: sequenceData must not be None")
	if("activationList" not in sequenceData):
		raise RuntimeError("generateSequenceDataText error: sequenceData missing activationList")
	if("numberOfSnapshots" not in sequenceData):
		raise RuntimeError("generateSequenceDataText error: sequenceData missing numberOfSnapshots")
	if("numberOfColumns" not in sequenceData):
		raise RuntimeError("generateSequenceDataText error: sequenceData missing numberOfColumns")
	maxFeatureCount = int(sequenceData["numberOfSnapshots"])*int(sequenceData["numberOfColumns"])
	if(len(sequenceData["activationList"]) > maxFeatureCount):
		raise RuntimeError("generateSequenceDataText error: activationList length exceeds numberOfSnapshots*numberOfColumns")
	for activation in sequenceData["activationList"]:
		snapshotIndex = int(activation["snapshotIndex"])
		columnIndex = int(activation["columnIndex"])
		activationKey = (snapshotIndex, columnIndex)
		if(activationKey in activationKeySet):
			raise RuntimeError("generateSequenceDataText error: duplicate activation detected for snapshotIndex/columnIndex")
		activationKeySet.add(activationKey)
		if(snapshotIndex not in snapshotActivationDict):
			snapshotActivationDict[snapshotIndex] = []
		snapshotActivationDict[snapshotIndex].append(activation)
	for snapshotIndex in sorted(snapshotActivationDict.keys()):
		snapshotTextList.append(generateSnapshotDataText(snapshotIndex, snapshotActivationDict[snapshotIndex]))
	result = " | ".join(snapshotTextList)
	return result


def generateSnapshotDataText(snapshotIndex, snapshotActivationList):
	result = None
	activationTextList = []
	snapshotActivationListSorted = sorted(snapshotActivationList, key=lambda activation: int(activation["columnIndex"]))
	for activation in snapshotActivationListSorted:
		activationTextList.append(generateActivationText(activation))
	if(modalityORRFfilterNamesVerbose):
		result = "t" + str(snapshotIndex).zfill(3) + ": " + " ".join(activationTextList)
	else:
		result = " ".join(activationTextList)
	return result


def generateActivationText(activation):
	result = None
	if(modalityORRFfilterNamesVerbose):
		if("conceptName" not in activation):
			raise RuntimeError("generateActivationText error: activation missing conceptName")
		if("featureWordVerbose" not in activation):
			raise RuntimeError("generateActivationText error: activation missing featureWordVerbose")
		result = activation["conceptName"] + "=" + activation["featureWordVerbose"]
	else:
		if("featureWord" not in activation):
			raise RuntimeError("generateActivationText error: activation missing featureWord")
		result = activation["featureWord"]
	return result


def secondPass(databaseNetworkObject, sequenceData, inferenceMode):
	result = None
	observedColumnsDict = {}
	if(sequenceData is None):
		raise RuntimeError("secondPass error: sequenceData must not be None")
	for conceptName in sequenceData["orderedConceptNameList"]:
		conceptIndex = int(databaseNetworkObject.conceptColumnsDict[conceptName])
		if(inferenceMode):
			observedColumn = GIAANNcmn_databaseNetwork.loadOrCreateObservedColumn(databaseNetworkObject, conceptIndex, conceptName, conceptIndex, deviceLoadColumnInference, inferenceMode and deviceLoadColumnInferenceCopy)
		else:
			observedColumn = GIAANNcmn_databaseNetwork.loadOrCreateObservedColumn(databaseNetworkObject, conceptIndex, conceptName, conceptIndex)
		observedColumnsDict[conceptName] = observedColumn
	result = observedColumnsDict
	return result


def createSequenceObservedColumns(databaseNetworkObject, observedColumnsDict, sequenceData, inferenceMode):
	result = None
	result = GIAANNor_sequenceObservedColumns.SequenceObservedColumns(databaseNetworkObject, observedColumnsDict, sequenceData, inferenceMode)
	return result


def getActiveSegmentsForSnapshot(snapshotIndex, targetDevice):
	result = None
	if(useSANI):
		numberOfSegments = min(arrayNumberOfSegments, snapshotIndex + 1)
		result = pt.arange(0, numberOfSegments, device=targetDevice, dtype=pt.long)
	else:
		result = pt.tensor([arrayIndexSegmentFirst], dtype=pt.long, device=targetDevice)
	return result


def getActiveSegmentsForImageDistanceEncoding(sequenceObservedColumns, sequenceConceptIndex, targetDevice):
	result = None
	fieldXIndex = None
	fieldYIndex = None
	maxSegmentIndex = None
	if(submodalityName=="image" and modalityORimageSequenceEncode=="distance"):
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
		raise RuntimeError("getActiveSegmentsForImageDistanceEncoding error: requires submodalityName=='image' and modalityORimageSequenceEncode=='distance'")
	return result


def validateImageDistanceSequenceObservedColumnCoordinates(sequenceObservedColumns):
	result = None
	if(submodalityName=="image" and modalityORimageSequenceEncode=="distance"):
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
		raise RuntimeError("validateImageDistanceSequenceObservedColumnCoordinates error: requires submodalityName=='image' and modalityORimageSequenceEncode=='distance'")
	return result


def calculateImageDistanceMaxSegmentIndexForTargetFieldCoordinate(fieldXIndex, fieldYIndex):
	result = None
	maxDistanceX = None
	maxDistanceY = None
	if(submodalityName=="image" and modalityORimageSequenceEncode=="distance"):
		validateImageDistanceFieldIndex(fieldXIndex, "fieldXIndex")
		validateImageDistanceFieldIndex(fieldYIndex, "fieldYIndex")
		maxDistanceX = max(int(fieldXIndex), int(modalityORimageSequenceEncodeDistanceFieldSegments) - 1 - int(fieldXIndex))
		maxDistanceY = max(int(fieldYIndex), int(modalityORimageSequenceEncodeDistanceFieldSegments) - 1 - int(fieldYIndex))
		result = int(math.ceil(math.sqrt(float((maxDistanceX*maxDistanceX) + (maxDistanceY*maxDistanceY)))))
		if(result < 0 or result >= int(arrayNumberOfSegments)):
			raise RuntimeError("calculateImageDistanceMaxSegmentIndexForTargetFieldCoordinate error: calculated segment index out of range")
	else:
		raise RuntimeError("calculateImageDistanceMaxSegmentIndexForTargetFieldCoordinate error: requires submodalityName=='image' and modalityORimageSequenceEncode=='distance'")
	return result


def validateImageDistanceFieldIndex(fieldIndex, fieldName):
	result = None
	if(submodalityName=="image" and modalityORimageSequenceEncode=="distance"):
		if(not isinstance(fieldName, str) or fieldName == ""):
			raise RuntimeError("validateImageDistanceFieldIndex error: fieldName must be a non-empty string")
		if(not isinstance(fieldIndex, int) or isinstance(fieldIndex, bool)):
			raise RuntimeError("validateImageDistanceFieldIndex error: " + fieldName + " must be an int")
		if(fieldIndex < 0 or fieldIndex >= int(modalityORimageSequenceEncodeDistanceFieldSegments)):
			raise RuntimeError("validateImageDistanceFieldIndex error: " + fieldName + " out of range")
	else:
		raise RuntimeError("validateImageDistanceFieldIndex error: requires submodalityName=='image' and modalityORimageSequenceEncode=='distance'")
	return result


def configureTrainConnectionsForImageSaccadeEncoding(sequenceObservedColumns):
	result = None
	if(submodalityName=="image"):
		if(not isinstance(modalityORimageSequenceEncode, str)):
			raise RuntimeError("configureTrainConnectionsForImageSaccadeEncoding error: modalityORimageSequenceEncode must be a string")
		if(modalityORimageSequenceEncode=="distance" or modalityORimageSequenceEncode=="none"):
			sequenceObservedColumns.trainConnectionsIncludeSameTimeIndex = True
		if(modalityORimageSequenceEncode=="distance"):
			sequenceObservedColumns.trainConnectionsUseSpatialDistance = True
		elif(modalityORimageSequenceEncode!="saccades" and modalityORimageSequenceEncode!="none"):
			raise RuntimeError("configureTrainConnectionsForImageSaccadeEncoding error: modalityORimageSequenceEncode must be 'saccades', 'distance', or 'none'")
	return result


def buildTrainTensors(sequenceObservedColumns, sequenceData):
	result = None
	targetDevice = deviceDense
	featureNeuronsActive = pt.zeros((numberOfDendriticBranches, arrayNumberOfSegments, sequenceObservedColumns.cs, sequenceObservedColumns.fs), dtype=arrayType, device=targetDevice)
	sequenceConceptIndexMask = pt.ones((sequenceObservedColumns.cs, sequenceObservedColumns.fs), dtype=arrayType, device=targetDevice)
	columnsWordOrder = None
	featureNeuronsWordOrder = pt.zeros((sequenceObservedColumns.cs, sequenceObservedColumns.fs), dtype=pt.long, device=targetDevice)
	featureNeuronsPos = pt.zeros((sequenceObservedColumns.cs, sequenceObservedColumns.fs), dtype=arrayType, device=targetDevice)
	featureNeuronsSegmentMask = pt.ones((arrayNumberOfSegments, sequenceObservedColumns.cs), dtype=arrayType, device=targetDevice)
	configureTrainConnectionsForImageSaccadeEncoding(sequenceObservedColumns)
	for activation in sequenceData["activationList"]:
		# each layer column has a maximum of 1 feature trained for every iteration in a sequence.
		sequenceConceptIndex = int(activation["sequenceConceptIndex"])
		localFeatureIndex = int(activation["localFeatureIndex"])
		snapshotIndex = int(activation["snapshotIndex"])
		if(submodalityName=="image" and (modalityORimageSequenceEncode=="distance" or modalityORimageSequenceEncode=="none") and snapshotIndex != 0):
			raise RuntimeError("buildTrainTensors error: snapshotIndex must be 0 when modalityORimageSequenceEncode is 'distance' or 'none'")
		if(submodalityName=="image" and modalityORimageSequenceEncode=="distance"):
			activeSegments = getActiveSegmentsForImageDistanceEncoding(sequenceObservedColumns, sequenceConceptIndex, targetDevice)
			featureNeuronsWordOrder[sequenceConceptIndex, localFeatureIndex] = 0
		else:
			activeSegments = getActiveSegmentsForSnapshot(snapshotIndex, targetDevice)
			featureNeuronsWordOrder[sequenceConceptIndex, localFeatureIndex] = snapshotIndex
		featureNeuronsActive[0, activeSegments, sequenceConceptIndex, localFeatureIndex] = 1
	result = (featureNeuronsActive, sequenceObservedColumns.cs, sequenceObservedColumns.fs, sequenceConceptIndexMask, columnsWordOrder, featureNeuronsWordOrder, featureNeuronsPos, featureNeuronsSegmentMask)
	return result


def trainConceptWords(sequenceObservedColumns, sequenceIndex, sequenceData):
	result = False
	buildResult = None
	if(sequenceData is None):
		raise RuntimeError("trainConceptWords error: sequenceData must not be None")
	if(len(sequenceData["activationList"]) > 0):
		buildResult = buildTrainTensors(sequenceObservedColumns, sequenceData)
		GIAANNcmn_databaseNetworkTrain.processFeaturesActiveTrain(sequenceObservedColumns, buildResult[0], buildResult[1], buildResult[2], buildResult[3], buildResult[4], buildResult[5], buildResult[6], buildResult[7], sequenceIndex)
		result = True
	return result
