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

import torch as pt

from GIAANNcmn_globalDefs import *
import GIAANNcmn_databaseNetwork
import GIAANNcmn_databaseNetworkTrain
import GIAANNor_RFfilters
import GIAANNor_sequenceAxis
import GIAANNor_sequenceAxes
import GIAANNor_sequenceDistance
import GIAANNor_sequenceObservedColumns
import GIAANNor_sequenceSaccades


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
	imageAxisColumnCoordinatesByConceptName = None
	imageAxesSequenceTensorData = None
	seenConceptNames = set()
	if(not pt.is_tensor(selectedFilterIndices)):
		raise RuntimeError("generateSequenceData error: selectedFilterIndices must be a tensor")
	if(selectedFilterIndices.dim() != 2):
		raise RuntimeError("generateSequenceData error: selectedFilterIndices rank must be 2")
	if(selectedFilterIndices.shape[1] != len(columnMetadataList)):
		raise RuntimeError("generateSequenceData error: selectedFilterIndices column count mismatch")
	if(submodalityName=="image" and modalityORimageSequenceEncode=="axes"):
		result = generateSequenceDataImageAxes(databaseNetworkObject, columnMetadataList, selectedFilterIndices, rfFilters, allowNewFeatures)
	else:
		if(submodalityName=="image" and modalityORimageSequenceEncode=="distance"):
			imageDistanceFieldCoordinatesByConceptName = GIAANNor_sequenceDistance.buildImageDistanceFieldCoordinatesByConceptName(columnMetadataList)
		elif(submodalityName=="image" and modalityORimageSequenceEncode=="axis"):
			imageDistanceFieldCoordinatesByConceptName = GIAANNor_sequenceDistance.buildImageDistanceFieldCoordinatesByConceptName(columnMetadataList)
			imageAxisColumnCoordinatesByConceptName = GIAANNor_sequenceAxis.buildImageAxisColumnCoordinatesByConceptName(columnMetadataList)
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
		if(submodalityName=="image" and modalityORimageSequenceEncode=="axis"):
			GIAANNor_sequenceAxis.expandImageAxisSequenceConcepts(orderedConceptNameList, requiredSourceFeatureIndicesByConceptName, activationList, columnMetadataList, imageAxisColumnCoordinatesByConceptName)
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
			elif(submodalityName=="image" and modalityORimageSequenceEncode=="axis"):
				result["imageDistanceFieldCoordinatesByConceptName"] = imageDistanceFieldCoordinatesByConceptName
				result["imageAxisColumnCoordinatesByConceptName"] = imageAxisColumnCoordinatesByConceptName
	return result


def generateSequenceDataImageAxes(databaseNetworkObject, columnMetadataList, selectedFilterIndices, rfFilters, allowNewFeatures):
	result = None
	imageAxesSequenceTensorData = None
	activeCoordinateTensor = None
	activeFilterIndexTensor = None
	uniqueFilterIndexTensor = None
	inverseFilterIndexTensor = None
	uniqueFilterIndexList = None
	uniqueFeatureWords = None
	uniqueFeatureWordsVerbose = None
	uniqueGlobalFeatureIndices = None
	uniqueGlobalFeatureIndexTensor = None
	globalFeatureIndexTensor = None
	centralConceptName = None
	featureWords = None
	featureWordsVerbose = None
	globalFeatureIndices = None
	requiredSourceFeatureIndicesByConceptName = None
	activationList = None
	activationTupleIterable = None
	centralTargetActive = None
	if(submodalityName=="image" and modalityORimageSequenceEncode=="axes"):
		imageAxesSequenceTensorData = GIAANNor_sequenceAxes.buildImageAxesSequenceTensorData(columnMetadataList, selectedFilterIndices)
		activeCoordinateTensor = pt.nonzero(selectedFilterIndices >= 0, as_tuple=False)
		centralTargetActive = bool(pt.any(imageAxesSequenceTensorData["featureCentralColumnMaskTensor"]).item())
		if(activeCoordinateTensor.shape[0] > 0 and centralTargetActive):
			centralConceptName = imageAxesSequenceTensorData["centralConceptName"]
			ensureConceptColumns(databaseNetworkObject, [columnMetadataList[int(imageAxesSequenceTensorData["centralColumnIndex"])]], allowNewFeatures)
			activeFilterIndexTensor = selectedFilterIndices[activeCoordinateTensor[:, 0], activeCoordinateTensor[:, 1]].to(dtype=pt.long, device=selectedFilterIndices.device)
			uniqueFilterIndexTensor, inverseFilterIndexTensor = pt.unique(activeFilterIndexTensor, sorted=True, return_inverse=True)
			uniqueFilterIndexList = uniqueFilterIndexTensor.detach().cpu().tolist()
			uniqueFeatureWords = list(map(lambda rfFilterIndex: GIAANNor_RFfilters.convertRFfilterIndexToASCIItext(rfFilters, int(rfFilterIndex)), uniqueFilterIndexList))
			uniqueFeatureWordsVerbose = list(map(lambda rfFilterIndex: GIAANNor_RFfilters.convertRFfilterIndexToASCIItextVerbose(rfFilters, int(rfFilterIndex)), uniqueFilterIndexList))
			uniqueGlobalFeatureIndices = list(map(lambda featureWord: ensureFeatureIndex(databaseNetworkObject, featureWord, allowNewFeatures), uniqueFeatureWords))
			uniqueGlobalFeatureIndexTensor = pt.tensor(uniqueGlobalFeatureIndices, dtype=pt.long, device=inverseFilterIndexTensor.device)
			globalFeatureIndexTensor = uniqueGlobalFeatureIndexTensor.index_select(0, inverseFilterIndexTensor)
			featureWords = list(map(uniqueFeatureWords.__getitem__, inverseFilterIndexTensor.detach().cpu().tolist()))
			featureWordsVerbose = list(map(uniqueFeatureWordsVerbose.__getitem__, inverseFilterIndexTensor.detach().cpu().tolist()))
			globalFeatureIndices = globalFeatureIndexTensor.detach().cpu().tolist()
			requiredSourceFeatureIndicesByConceptName = {centralConceptName: sorted(uniqueGlobalFeatureIndices)}
			activationTupleIterable = zip(range(int(activeCoordinateTensor.shape[0])), activeCoordinateTensor[:, 0].detach().cpu().tolist(), activeCoordinateTensor[:, 1].detach().cpu().tolist(), featureWords, featureWordsVerbose, globalFeatureIndices)
			activationList = list(map(lambda activationTuple: buildImageAxesActivation(activationTuple, centralConceptName), activationTupleIterable))
			result = {"orderedConceptNameList": [centralConceptName], "activationList": activationList, "featureWords": featureWords, "globalFeatureIndices": globalFeatureIndices, "requiredSourceFeatureIndicesByConceptName": requiredSourceFeatureIndicesByConceptName, "numberOfSnapshots": int(selectedFilterIndices.shape[0]), "numberOfColumns": int(selectedFilterIndices.shape[1]), "imageAxesFeatureFieldXTensor": imageAxesSequenceTensorData["featureFieldXTensor"], "imageAxesFeatureFieldYTensor": imageAxesSequenceTensorData["featureFieldYTensor"], "imageAxesFeatureAxisMaskTensor": imageAxesSequenceTensorData["featureAxisMaskTensor"], "imageAxesFeatureCentralColumnMaskTensor": imageAxesSequenceTensorData["featureCentralColumnMaskTensor"], "imageAxesCentralFieldX": imageAxesSequenceTensorData["centralFieldX"], "imageAxesCentralFieldY": imageAxesSequenceTensorData["centralFieldY"]}
	else:
		raise RuntimeError("generateSequenceDataImageAxes error: requires submodalityName=='image' and modalityORimageSequenceEncode=='axes'")
	return result


def buildImageAxesActivation(activationTuple, centralConceptName):
	result = None
	localFeatureIndex = None
	snapshotIndex = None
	columnIndex = None
	featureWord = None
	featureWordVerbose = None
	globalFeatureIndex = None
	if(submodalityName=="image" and modalityORimageSequenceEncode=="axes"):
		if(not isinstance(activationTuple, tuple) or len(activationTuple) != 6):
			raise RuntimeError("buildImageAxesActivation error: activationTuple must be a tuple of length 6")
		if(not isinstance(centralConceptName, str) or centralConceptName == ""):
			raise RuntimeError("buildImageAxesActivation error: centralConceptName must be a non-empty string")
		localFeatureIndex, snapshotIndex, columnIndex, featureWord, featureWordVerbose, globalFeatureIndex = activationTuple
		result = {"snapshotIndex": int(snapshotIndex), "columnIndex": int(columnIndex), "conceptName": centralConceptName, "featureWord": featureWord, "featureWordVerbose": featureWordVerbose, "globalFeatureIndex": int(globalFeatureIndex), "localFeatureIndex": int(localFeatureIndex), "sequenceConceptIndex": int(modalityORimageSequenceEncodeAxesColumnIndex)}
	else:
		raise RuntimeError("buildImageAxesActivation error: requires submodalityName=='image' and modalityORimageSequenceEncode=='axes'")
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
	if(submodalityName=="image" and modalityORimageSequenceEncode=="axes"):
		result = generateImageAxesSequenceDataText(sequenceData)
	else:
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
		result = modalityORsequenceDataTextSnapshotDelimiter.join(snapshotTextList)
	return result


def generateImageAxesSequenceDataText(sequenceData):
	result = None
	fieldXTensor = None
	fieldYTensor = None
	axisMaskTensor = None
	centralFieldX = None
	centralFieldY = None
	deltaX = None
	deltaY = None
	segmentIndexTensor = None
	segmentActivationDict = {}
	segmentTextList = []
	localFeatureIndex = None
	segmentIndex = None
	if(submodalityName=="image" and modalityORimageSequenceEncode=="axes"):
		if("imageAxesFeatureFieldXTensor" not in sequenceData or "imageAxesFeatureFieldYTensor" not in sequenceData or "imageAxesFeatureAxisMaskTensor" not in sequenceData or "imageAxesCentralFieldX" not in sequenceData or "imageAxesCentralFieldY" not in sequenceData):
			raise RuntimeError("generateImageAxesSequenceDataText error: sequenceData missing image axes tensors")
		fieldXTensor = sequenceData["imageAxesFeatureFieldXTensor"]
		fieldYTensor = sequenceData["imageAxesFeatureFieldYTensor"]
		axisMaskTensor = sequenceData["imageAxesFeatureAxisMaskTensor"]
		if(not pt.is_tensor(fieldXTensor) or not pt.is_tensor(fieldYTensor) or not pt.is_tensor(axisMaskTensor)):
			raise RuntimeError("generateImageAxesSequenceDataText error: image axes fields and mask must be tensors")
		if(fieldXTensor.dim() != 1 or fieldYTensor.dim() != 1 or axisMaskTensor.dim() != 1):
			raise RuntimeError("generateImageAxesSequenceDataText error: image axes fields and mask must be rank 1")
		if(int(fieldXTensor.shape[0]) != len(sequenceData["activationList"]) or int(fieldYTensor.shape[0]) != len(sequenceData["activationList"]) or int(axisMaskTensor.shape[0]) != len(sequenceData["activationList"])):
			raise RuntimeError("generateImageAxesSequenceDataText error: image axes tensor lengths must match activationList")
		centralFieldX = int(sequenceData["imageAxesCentralFieldX"])
		centralFieldY = int(sequenceData["imageAxesCentralFieldY"])
		deltaX = pt.abs(centralFieldX - fieldXTensor)
		deltaY = pt.abs(centralFieldY - fieldYTensor)
		segmentIndexTensor = pt.ceil(pt.sqrt((deltaX*deltaX + deltaY*deltaY).to(arrayType))).long()
		if(bool(pt.any(segmentIndexTensor < 0).item()) or bool(pt.any(segmentIndexTensor >= int(arrayNumberOfSegments)).item())):
			raise RuntimeError("generateImageAxesSequenceDataText error: calculated segment index out of range")
		for activation in sequenceData["activationList"]:
			localFeatureIndex = int(activation["localFeatureIndex"])
			if(localFeatureIndex < 0 or localFeatureIndex >= int(axisMaskTensor.shape[0])):
				raise RuntimeError("generateImageAxesSequenceDataText error: activation localFeatureIndex out of range")
			if(bool(axisMaskTensor[localFeatureIndex].item())):
				segmentIndex = int(segmentIndexTensor[localFeatureIndex].item())
				if(segmentIndex not in segmentActivationDict):
					segmentActivationDict[segmentIndex] = []
				segmentActivationDict[segmentIndex].append(activation)
		for segmentIndex in sorted(segmentActivationDict.keys()):
			segmentTextList.append(generateImageAxesSegmentDataText(segmentIndex, segmentActivationDict[segmentIndex]))
		result = modalityORsequenceDataTextSegmentDelimiter.join(segmentTextList)
	else:
		raise RuntimeError("generateImageAxesSequenceDataText error: requires submodalityName=='image' and modalityORimageSequenceEncode=='axes'")
	return result


def generateImageAxesSegmentDataText(segmentIndex, segmentActivationList):
	result = None
	activationTextList = []
	segmentActivationListSorted = None
	if(submodalityName=="image" and modalityORimageSequenceEncode=="axes"):
		if(not isinstance(segmentIndex, int) or isinstance(segmentIndex, bool)):
			raise RuntimeError("generateImageAxesSegmentDataText error: segmentIndex must be an int")
		if(segmentIndex < 0 or segmentIndex >= int(arrayNumberOfSegments)):
			raise RuntimeError("generateImageAxesSegmentDataText error: segmentIndex out of range")
		if(not isinstance(segmentActivationList, list)):
			raise RuntimeError("generateImageAxesSegmentDataText error: segmentActivationList must be a list")
		segmentActivationListSorted = sorted(segmentActivationList, key=lambda activation: int(activation["columnIndex"]))
		for activation in segmentActivationListSorted:
			activationTextList.append(generateImageAxesActivationText(activation))
		result = modalityORsequenceDataTextSegmentPrefix + str(segmentIndex).zfill(int(modalityORsequenceDataTextIndexDigits)) + modalityORsequenceDataTextLabelSuffix + modalityORsequenceDataTextFeatureDelimiter.join(activationTextList)
	else:
		raise RuntimeError("generateImageAxesSegmentDataText error: requires submodalityName=='image' and modalityORimageSequenceEncode=='axes'")
	return result


def generateImageAxesActivationText(activation):
	result = None
	if(submodalityName=="image" and modalityORimageSequenceEncode=="axes"):
		if(modalityORRFfilterNamesVerbose):
			if("featureWordVerbose" not in activation):
				raise RuntimeError("generateImageAxesActivationText error: activation missing featureWordVerbose")
			result = activation["featureWordVerbose"]
		else:
			if("featureWord" not in activation):
				raise RuntimeError("generateImageAxesActivationText error: activation missing featureWord")
			result = activation["featureWord"]
	else:
		raise RuntimeError("generateImageAxesActivationText error: requires submodalityName=='image' and modalityORimageSequenceEncode=='axes'")
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


def configureTrainConnectionsForImageSequenceEncoding(sequenceObservedColumns):
	result = None
	if(submodalityName=="image"):
		if(not isinstance(modalityORimageSequenceEncode, str)):
			raise RuntimeError("configureTrainConnectionsForImageSequenceEncoding error: modalityORimageSequenceEncode must be a string")
		if(modalityORimageSequenceEncode=="saccades"):
			GIAANNor_sequenceSaccades.configureTrainConnectionsForImageSaccadeEncoding(sequenceObservedColumns)
		elif(modalityORimageSequenceEncode=="distance"):
			GIAANNor_sequenceDistance.configureTrainConnectionsForImageDistanceEncoding(sequenceObservedColumns)
		elif(modalityORimageSequenceEncode=="axis"):
			GIAANNor_sequenceAxis.configureTrainConnectionsForImageAxisEncoding(sequenceObservedColumns)
		elif(modalityORimageSequenceEncode=="axes"):
			GIAANNor_sequenceAxes.configureTrainConnectionsForImageAxesEncoding(sequenceObservedColumns)
		elif(modalityORimageSequenceEncode=="none"):
			sequenceObservedColumns.trainConnectionsIncludeSameTimeIndex = True
		else:
			raise RuntimeError("configureTrainConnectionsForImageSequenceEncoding error: modalityORimageSequenceEncode must be 'saccades', 'distance', 'axis', 'axes', or 'none'")
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
	configureTrainConnectionsForImageSequenceEncoding(sequenceObservedColumns)
	if(submodalityName=="image" and modalityORimageSequenceEncode=="axes"):
		GIAANNor_sequenceAxes.populateImageAxesTrainTensors(sequenceObservedColumns, featureNeuronsActive, featureNeuronsWordOrder, targetDevice)
	else:
		for activation in sequenceData["activationList"]:
			# each layer column has a maximum of 1 feature trained for every iteration in a sequence.
			sequenceConceptIndex = int(activation["sequenceConceptIndex"])
			localFeatureIndex = int(activation["localFeatureIndex"])
			snapshotIndex = int(activation["snapshotIndex"])
			if(submodalityName=="image" and modalityORimageSequenceEncode=="distance"):
				GIAANNor_sequenceDistance.validateImageDistanceSnapshotIndex(snapshotIndex)
				activeSegments = GIAANNor_sequenceDistance.getActiveSegmentsForImageDistanceEncoding(sequenceObservedColumns, sequenceConceptIndex, targetDevice)
				featureNeuronsWordOrder[sequenceConceptIndex, localFeatureIndex] = 0
			elif(submodalityName=="image" and modalityORimageSequenceEncode=="axis"):
				GIAANNor_sequenceAxis.validateImageAxisSnapshotIndex(snapshotIndex)
				activeSegments = GIAANNor_sequenceAxis.getActiveSegmentsForImageAxisEncoding(sequenceObservedColumns, sequenceConceptIndex, targetDevice)
				featureNeuronsWordOrder[sequenceConceptIndex, localFeatureIndex] = 0
			elif(submodalityName=="image" and modalityORimageSequenceEncode=="none"):
				if(snapshotIndex != 0):
					raise RuntimeError("buildTrainTensors error: snapshotIndex must be 0 when modalityORimageSequenceEncode is 'none'")
				activeSegments = getActiveSegmentsForSnapshot(snapshotIndex, targetDevice)
				featureNeuronsWordOrder[sequenceConceptIndex, localFeatureIndex] = snapshotIndex
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
