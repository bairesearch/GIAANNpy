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
	seenConceptNames = set()
	if(not pt.is_tensor(selectedFilterIndices)):
		raise RuntimeError("generateSequenceData error: selectedFilterIndices must be a tensor")
	if(selectedFilterIndices.dim() != 2):
		raise RuntimeError("generateSequenceData error: selectedFilterIndices rank must be 2")
	if(selectedFilterIndices.shape[1] != len(columnMetadataList)):
		raise RuntimeError("generateSequenceData error: selectedFilterIndices column count mismatch")
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


def configureTrainConnectionsForImageSaccadeEncoding(sequenceObservedColumns):
	result = None
	if(submodalityName=="image"):
		if(not isinstance(modalityORimageSaccadesEncode, bool)):
			raise RuntimeError("configureTrainConnectionsForImageSaccadeEncoding error: modalityORimageSaccadesEncode must be a bool")
		if(not modalityORimageSaccadesEncode):
			sequenceObservedColumns.trainConnectionsIncludeSameTimeIndex = True
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
		if(submodalityName=="image" and not modalityORimageSaccadesEncode and snapshotIndex != 0):
			raise RuntimeError("buildTrainTensors error: snapshotIndex must be 0 when modalityORimageSaccadesEncode is False")
		activeSegments = getActiveSegmentsForSnapshot(snapshotIndex, targetDevice)
		featureNeuronsActive[0, activeSegments, sequenceConceptIndex, localFeatureIndex] = 1
		featureNeuronsWordOrder[sequenceConceptIndex, localFeatureIndex] = snapshotIndex
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
