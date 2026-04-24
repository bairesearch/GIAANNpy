"""GIAANNor_main.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN OR main

"""

import torch as pt

pt.set_grad_enabled(False)

from GIAANNcmn_globalDefs import *
import GIAANNcmn_databaseNetwork
import GIAANNcmn_databaseNetworkFiles
import GIAANNcmn_databaseNetworkDraw
import GIAANNor_datasets
import GIAANNor_RFfilters
import GIAANNor_sequenceConcepts
import GIAANNor_sequenceTokens


def ensureORruntimeInitialised(databaseNetworkObject):
	if(not hasattr(databaseNetworkObject, "orRFfilters") or databaseNetworkObject.orRFfilters is None):
		databaseNetworkObject.orRFfilters = GIAANNor_RFfilters.initialiseRFfilters()
	return


def calculateSequenceColumnCounts(tokenisedSnapshots, selectedFilterIndices):
	result = None
	numberOfColumnsPerLayer = None
	totalColumnsInNetwork = None
	numberActiveColumnsSummedAcrossSnapshotsInSequence = None
	if("gridHeight" not in tokenisedSnapshots):
		raise RuntimeError("calculateSequenceColumnCounts error: tokenisedSnapshots missing gridHeight")
	if("gridWidth" not in tokenisedSnapshots):
		raise RuntimeError("calculateSequenceColumnCounts error: tokenisedSnapshots missing gridWidth")
	if(not pt.is_tensor(selectedFilterIndices)):
		raise RuntimeError("calculateSequenceColumnCounts error: selectedFilterIndices must be a tensor")
	if(selectedFilterIndices.dim() != 2):
		raise RuntimeError("calculateSequenceColumnCounts error: selectedFilterIndices rank must be 2")
	numberOfColumnsPerLayer = int(tokenisedSnapshots["gridHeight"])*int(tokenisedSnapshots["gridWidth"])
	totalColumnsInNetwork = numberOfColumnsPerLayer*int(modalityORnumberOfLayers)
	numberActiveColumnsSummedAcrossSnapshotsInSequence = int((selectedFilterIndices >= 0).sum().item())
	result = {"numberOfColumnsPerLayer": numberOfColumnsPerLayer, "totalColumnsInNetwork": totalColumnsInNetwork, "numberActiveColumnsSummedAcrossSnapshotsInSequence": numberActiveColumnsSummedAcrossSnapshotsInSequence}
	return result


def printSequenceColumnCounts(sequenceCount, sequenceColumnCounts):
	# printSequenceNumberColumns=True which prints the total number of columns per layer, total columns in network, and number of columns with active RF for every snapshot sequence (ie number active columns summed across all snapshots in sequence).
	print(f"Processing sequenceCount: {sequenceCount}, numberOfColumnsPerLayer={sequenceColumnCounts['numberOfColumnsPerLayer']}, totalColumnsInNetwork={sequenceColumnCounts['totalColumnsInNetwork']}, numberActiveColumnsSummedAcrossSnapshotsInSequence={sequenceColumnCounts['numberActiveColumnsSummedAcrossSnapshotsInSequence']}")
	return


def processPrompt(databaseNetworkObject, inferenceMode, sequenceCount):
	# GIAANNor_main.processPrompt() for modalityName=="OR" should use a testset part of the OR dataset as a prompt (rather than an independent prompt file).
	result = sequenceCount
	promptDataset = GIAANNor_datasets.loadPromptDataset()
	if(submodalityName=="image"):
		result = processImageDataset(databaseNetworkObject, False, result, promptDataset, modalityORdatasetPromptMaxSequences)
	elif(submodalityName=="video"):
		result = processVideoDataset(databaseNetworkObject, False, result, promptDataset)
	else:
		printe("submodalityName = ", submodalityName)
	return result


def processDataset(databaseNetworkObject, inferenceMode, sequenceCount, dataset):
	result = sequenceCount
	if(submodalityName=="video"):
		result = processVideoDataset(databaseNetworkObject, inferenceMode, result, dataset)
	elif(submodalityName=="image"):
		result = processImageDataset(databaseNetworkObject, inferenceMode, result, dataset, None)
	else:
		raise RuntimeError("processDataset error: unsupported OR submodalityName " + str(submodalityName))
	return result


def processVideoDataset(databaseNetworkObject, inferenceMode, sequenceCount, dataset):
	result = sequenceCount
	ensureORruntimeInitialised(databaseNetworkObject)
	if(submodalityName=="video"):
		for articleIndex, datasetEntry in enumerate(dataset):
			sequence = GIAANNor_datasets.sampleVideoSnapshots(datasetEntry)
			processSequence(databaseNetworkObject, False, result, articleIndex, 0, sequence, datasetEntry)
			result = result + 1
			if(result == trainMaxSequences and inferenceMode == False):
				break
	elif(submodalityName=="image"):
		result = processImageDataset(databaseNetworkObject, inferenceMode, result, dataset, None)
	else:
		raise RuntimeError("processVideoDataset error: unsupported OR submodalityName " + str(submodalityName))
	return result


def processImageDataset(databaseNetworkObject, inferenceMode, sequenceCount, dataset, sequenceLimit):
	result = sequenceCount
	processedSequenceCount = 0
	sequences = None
	if(sequenceLimit is not None):
		if(not isinstance(sequenceLimit, int)):
			raise RuntimeError("processImageDataset error: sequenceLimit must be an int or None")
		if(sequenceLimit <= 0):
			raise RuntimeError("processImageDataset error: sequenceLimit must be > 0")
	ensureORruntimeInitialised(databaseNetworkObject)
	for articleIndex, datasetEntry in enumerate(dataset):
		sequences = GIAANNor_datasets.sampleImageSaccadeSequences(datasetEntry)
		if(sequences is None):
			if(modalityORimageSaccadesSkipInsufficientUsableFeatures):
				if(debugPrintInsufficientUsableFeaturesWarning):
					printInsufficientUsableFeaturesWarning(articleIndex, result)
				continue
			else:
				raise RuntimeError("processImageDataset error: sampleImageSaccadeSequences returned no usable sequences for current image")
		for sequenceIndex, sequence in enumerate(sequences):
			processSequence(databaseNetworkObject, False, result, articleIndex, sequenceIndex, sequence, datasetEntry)
			result = result + 1
			processedSequenceCount = processedSequenceCount + 1
			if(sequenceLimit is not None and processedSequenceCount == sequenceLimit):
				break
			if(result == trainMaxSequences and inferenceMode == False):
				break
		if(sequenceLimit is not None and processedSequenceCount == sequenceLimit):
			break
		if(result == trainMaxSequences and inferenceMode == False):
			break
	return result


def printInsufficientUsableFeaturesWarning(articleIndex, sequenceCount):
	result = None
	print("Warning: skipping image due to insufficient usable features; articleIndex = ", articleIndex, ", sequenceCount = ", sequenceCount)
	return result


def processSequence(databaseNetworkObject, inferenceMode, sequenceCount, articleIndex, sequenceIndex, sequence, sequenceRaw):
	# processSequenceTemplate() for each OR sequence:
	ensureORruntimeInitialised(databaseNetworkObject)
	if(inferenceMode):
		raise RuntimeError("processSequenceTemplate error: OR inference prediction is not implemented in GIAANNproto2a1a")
	databaseNetworkObject.articleIndexDebug = articleIndex
	databaseNetworkObject.sequenceIndexDebug = sequenceIndex
	tokenisedSnapshots = GIAANNor_sequenceTokens.tokeniseSnapshotsToColumns(sequence)
	# apply the RF filters to the token (executed in parallel using pytorch):
	selectedFilterIndices, selectedFilterResponses = GIAANNor_RFfilters.applyRFfilters(databaseNetworkObject.orRFfilters, tokenisedSnapshots["columnTensor"])
	if(printSequenceNumberColumns):
		sequenceColumnCounts = calculateSequenceColumnCounts(tokenisedSnapshots, selectedFilterIndices)
		printSequenceColumnCounts(sequenceCount, sequenceColumnCounts)
	sequenceData = GIAANNor_sequenceConcepts.generateSequenceData(databaseNetworkObject, tokenisedSnapshots["columnMetadataList"], selectedFilterIndices, databaseNetworkObject.orRFfilters, True)
	
	if(sequenceData is not None):

		if(printTrainSequenceDefault):
			sequenceText = GIAANNor_sequenceConcepts.generateSequenceDataText(sequenceData)
			print(f"Processing sequenceCount: {sequenceCount}, {sequenceText}")
		if(printTrainSequenceCount):
			print(f"Processing sequenceCount: {sequenceCount}")	


		if(storeDatabaseGlobalFeatureNeuronsInRam):
			GIAANNcmn_databaseNetwork.ensureGlobalFeatureNeuronsSize(databaseNetworkObject, False)
		observedColumnsDict = GIAANNor_sequenceConcepts.secondPass(databaseNetworkObject, sequenceData, inferenceMode)
		sequenceObservedColumns = GIAANNor_sequenceConcepts.createSequenceObservedColumns(databaseNetworkObject, observedColumnsDict, sequenceData, inferenceMode)
		requiredSourceFeatureIndicesByObservedColumn = sequenceObservedColumns.getTrainRequiredSourceFeatureIndicesByObservedColumn()
		GIAANNcmn_databaseNetwork.prepareObservedColumnsForTrainSequence(observedColumnsDict, requiredSourceFeatureIndicesByObservedColumn)
		trained = GIAANNor_sequenceConcepts.trainConceptWords(sequenceObservedColumns, sequenceCount, sequenceData)
		if(trained):
			sequenceObservedColumns.updateObservedColumnsWrapper()
			if(useSaveData):
				if(not storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam):
					GIAANNcmn_databaseNetworkFiles.saveData(databaseNetworkObject, observedColumnsDict, sequenceCount)
			if(drawNetworkDuringTrain):
				GIAANNcmn_databaseNetworkDraw.visualizeGraph(sequenceObservedColumns, False, save=drawNetworkDuringTrainSave, fileName=drawNetworkDuringTrainSaveFilenamePrepend+generateDrawSequenceIndex(sequenceIndex))
		if(storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam):
			GIAANNcmn_databaseNetwork.moveObservedColumnsDictConnectionsToDatabaseAfterTrain(observedColumnsDict, inferenceMode)
	else:
		printe("sequenceData is None")

	return
