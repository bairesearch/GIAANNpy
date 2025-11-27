"""GIAANNproto_predictiveNetworkBeamSearch.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_predictiveNetworkBeamSearch.py

# Usage:
see GIAANNproto_predictiveNetworkBeamSearch.py

# Description: 
GIA ANN proto predictive Network Beam Search

"""

import torch as pt

from GIAANNproto_globalDefs import *
import GIAANNproto_databaseNetwork
import GIAANNproto_databaseNetworkTrain	   #low level processFeaturesActivePredict functions currently stored here
import GIAANNproto_sparseTensors


def beamSearchPredictNextFeature(sequenceObservedColumns, databaseNetworkObject, observedColumnsDict, globalFeatureNeuronsActivation, globalFeatureNeuronsStrength, globalFeatureConnectionsActivation, globalFeatureNeuronsTime, wordsSequence, lemmasSequence, wordPredictionIndex, sequenceWordIndex, conceptMask):
	#generate targets for debug/analysis output
	targetMultipleSources, targetPreviousColumnIndex, targetNextColumnIndex, targetFeatureIndex, targetConceptColumnsIndices, targetConceptColumnsFeatureIndices = GIAANNproto_databaseNetwork.getTokenConceptFeatureIndexTensor(sequenceObservedColumns, wordsSequence, lemmasSequence, conceptMask, sequenceWordIndex, kcNetwork)

	if(inferenceBeamDepth <= 0 or inferenceBeamWidth <= 0):
		return selectMostActiveFeature(sequenceObservedColumns, globalFeatureNeuronsActivation, globalFeatureNeuronsStrength, wordsSequence, lemmasSequence, wordPredictionIndex, sequenceWordIndex, conceptMask)

	strengthLookup = None
	if(globalFeatureNeuronsStrength is not None):
		strengthLookup = buildStrengthLookup(globalFeatureNeuronsStrength, databaseNetworkObject.f)

	initialState = initialiseBeamActivationState(globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, globalFeatureNeuronsTime)
	beams = [{"score": 0.0, "state": initialState, "sequence": []}]
	completedBeams = []
	beamDepth = max(1, inferenceBeamDepth)
	beamWidthLimit = max(1, inferenceBeamWidth)

	for depthIndex in range(beamDepth):
		newBeams = []
		for beam in beams:
			candidates = selectBeamCandidates(beam["state"]["features"], strengthLookup, beamWidthLimit, databaseNetworkObject)
			if(len(candidates) == 0):
				completedBeams.append(beam)
				continue
			for candidate in candidates:
				predictInfo = describeBeamCandidate(databaseNetworkObject, candidate)
				#if(printPredictionsDuringInferencePredict):
				#	 print("\t"*(depthIndex+2) + f"Predicting beam node(s): {predictInfo}")	   # Debug: print beam depth and the node(s)/column being predicted
				oldState = beam["state"]
				newState = cloneBeamActivationState(oldState)
				for nodeColumn, nodeFeature in candidate["nodes"]:
					executeBeamNodeActivation(databaseNetworkObject, observedColumnsDict, newState, nodeColumn, nodeFeature, sequenceWordIndex)
				newSequence = beam["sequence"] + [candidate]
				activationGain = computeCandidateActivationGain(newState["features"], oldState["features"], candidate["nodes"])
				if(inferenceBeamSearchConceptColumns and inferenceBeamScoreStrategy == "nodeActivation"):
					activationGain = candidate.get("activationValue", activationGain)
				candidateScore = computeBeamNodeScore(activationGain, candidate["connectionValue"])
				newScore = beam["score"] + candidateScore
				newBeams.append({"score": newScore, "state": newState, "sequence": newSequence})
		if(len(newBeams) == 0):
			break
		newBeams.sort(key=lambda item: item["score"], reverse=True)
		beams = newBeams[:beamWidthLimit]

	if(len(beams) == 0):
		beams = completedBeams
	if(len(beams) == 0 or len(beams[0]["sequence"]) == 0):
		return selectMostActiveFeature(sequenceObservedColumns, globalFeatureNeuronsActivation, globalFeatureNeuronsStrength, wordsSequence, lemmasSequence, wordPredictionIndex, sequenceWordIndex, conceptMask)

	allBeams = beams + completedBeams
	bestBeam = max(allBeams, key=lambda item: item["score"])
	bestAction = bestBeam["sequence"][0]
	conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext = convertNodesToPrediction(bestAction["nodes"])
	if(conceptColumnsIndicesNext.shape[0] == 0):
		return selectMostActiveFeature(sequenceObservedColumns, globalFeatureNeuronsActivation, globalFeatureNeuronsStrength, wordsSequence, lemmasSequence, wordPredictionIndex, sequenceWordIndex, conceptMask)
	multipleSourcesNext = conceptColumnsIndicesNext.shape[0] > 1
	kc = conceptColumnsIndicesNext.shape[0]
	if(printPredictionsDuringInferencePredict):
		printBestBeamPath(bestBeam, databaseNetworkObject)
	conceptColumnsIndicesPred = conceptColumnsIndicesNext.clone()
	conceptColumnsFeatureIndicesPred = conceptColumnsFeatureIndicesNext.clone()
	return conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext, multipleSourcesNext, kc, conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, targetMultipleSources, targetPreviousColumnIndex, targetNextColumnIndex


def initialiseBeamActivationState(globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, globalFeatureNeuronsTime):
	state = {"features": globalFeatureNeuronsActivation.clone()}
	if(transformerUseInputConnections and globalFeatureConnectionsActivation is not None):
		state["connections"] = globalFeatureConnectionsActivation.clone()
	else:
		state["connections"] = None
	if(inferenceUseNeuronFeaturePropertiesTime and globalFeatureNeuronsTime is not None):
		state["time"] = globalFeatureNeuronsTime.clone()
	else:
		state["time"] = None
	return state


def cloneBeamActivationState(state):
	clonedState = {"features": state["features"].clone()}
	if(transformerUseInputConnections and state.get("connections") is not None):
		clonedState["connections"] = state["connections"].clone()
	else:
		clonedState["connections"] = None
	if(inferenceUseNeuronFeaturePropertiesTime and state.get("time") is not None):
		clonedState["time"] = state["time"].clone()
	else:
		clonedState["time"] = None
	return clonedState


def executeBeamNodeActivation(databaseNetworkObject, observedColumnsDict, state, columnIndex, featureIndex, sequenceWordIndex):
	lemma = databaseNetworkObject.conceptColumnsList[columnIndex]
	if(lemma in observedColumnsDict):
		observedColumn = observedColumnsDict[lemma]
	else:
		observedColumn = GIAANNproto_databaseNetwork.loadOrCreateObservedColumn(databaseNetworkObject, columnIndex, lemma, sequenceWordIndex)
		observedColumnsDict[lemma] = observedColumn
	featureConnections = observedColumn.featureConnections
	conceptColumnsIndicesSource = pt.tensor([columnIndex], dtype=pt.long, device=deviceSparse)
	conceptColumnsFeatureIndicesSource = pt.tensor([[featureIndex]], dtype=pt.long, device=deviceSparse)
	state["features"], state["connections"] = GIAANNproto_databaseNetworkTrain.processFeaturesActivePredict(databaseNetworkObject, state["features"], state["connections"], featureConnections, conceptColumnsIndicesSource, conceptColumnsFeatureIndicesSource)
	applyBeamNodePredictionEffects(state, columnIndex, featureIndex)
	return state


def applyBeamNodePredictionEffects(state, columnIndex, featureIndex):
	modifyActivation = inferenceDeactivateNeuronsUponPrediction or inferenceInvertNeuronActivationUponPrediction
	modifyTime = inferenceUseNeuronFeaturePropertiesTime and state.get("time") is not None
	if(not modifyActivation and not modifyTime):
		return
	indicesToUpdate = buildBeamNodeIndices(state["features"].device, columnIndex, featureIndex)
	if(modifyActivation):
		if(inferenceDeactivateNeuronsUponPrediction):
			modifier = 0
		else:
			modifier = inferenceInvertNeuronActivationUponPredictionLevel
		state["features"] = GIAANNproto_sparseTensors.modifySparseTensor(state["features"], indicesToUpdate, modifier, multiply=inferenceInvertNeuronActivationUponPrediction)
	if(modifyTime):
		state["time"] = GIAANNproto_sparseTensors.modifySparseTensor(state["time"], indicesToUpdate, inferenceUseNeuronFeaturePropertiesTimeActivate)


def buildBeamNodeIndices(device, columnIndex, featureIndex):
	indicesToUpdateList = []
	columnTensor = pt.tensor(columnIndex, dtype=pt.long, device=device)
	featureTensor = pt.tensor(featureIndex, dtype=pt.long, device=device)
	if(useSANI):
		for segmentIndex in range(arrayNumberOfSegments):
			segmentTensor = pt.tensor(segmentIndex, dtype=pt.long, device=device)
			indicesToUpdateList.append(pt.stack([segmentTensor, columnTensor, featureTensor], dim=0))
	else:
		segmentTensor = pt.tensor(arrayIndexSegmentFirst, dtype=pt.long, device=device)
		indicesToUpdateList.append(pt.stack([segmentTensor, columnTensor, featureTensor], dim=0))
	return pt.stack(indicesToUpdateList, dim=0)


def describeBeamCandidate(databaseNetworkObject, candidate):
	if(inferenceBeamSearchConceptColumns):
		columnIndex = candidate["columnIndex"]
		columnName = databaseNetworkObject.conceptColumnsList[columnIndex]
		nodeIndices = [nodeFeature for _, nodeFeature in candidate["nodes"]]
		return f"column {columnIndex} ({columnName}), node indices {nodeIndices}"
	return describeBeamNodes(databaseNetworkObject, candidate["nodes"])


def describeBeamNodes(databaseNetworkObject, nodes):
	nodeDescriptions = []
	for nodeColumn, nodeFeature in nodes:
		columnName = databaseNetworkObject.conceptColumnsList[nodeColumn]
		if(nodeFeature == featureIndexConceptNeuron):
			nodeName = f"{columnName} (concept)"
		elif(nodeFeature < len(databaseNetworkObject.conceptFeaturesList)):
			nodeName = databaseNetworkObject.conceptFeaturesList[nodeFeature]
		else:
			nodeName = f"feature_{nodeFeature}"
		nodeDescriptions.append(f"column {nodeColumn} ({columnName}), node {nodeFeature} ({nodeName})")
	return "; ".join(nodeDescriptions)


def printBestBeamPath(bestBeam, databaseNetworkObject):
	sequence = bestBeam.get("sequence", [])
	if(len(sequence) == 0):
		return
	pathSegments = []
	for depthIndex, candidate in enumerate(sequence):
		description = describeBeamCandidate(databaseNetworkObject, candidate)
		pathSegments.append(f"Depth {depthIndex}: {description}")
	print("\t\tBest beam path:\n\t\t\t" + "\n\t\t\t".join(pathSegments))	# Debug: summary of the highest scoring beam path


def selectBeamCandidates(stateFeatures, strengthLookup, candidateLimit, databaseNetworkObject):
	candidateLimit = max(1, candidateLimit)
	columnIndices, featureIndices, activationValues = aggregateSparseColumnFeatureValues(stateFeatures, databaseNetworkObject.f)
	if(columnIndices is None):
		return []
	if(inferenceBeamSearchConceptColumns):
		return selectBeamCandidatesConceptColumns(columnIndices, featureIndices, activationValues, strengthLookup, candidateLimit, databaseNetworkObject.f)
	else:
		return selectBeamCandidatesInstanceNodes(columnIndices, featureIndices, activationValues, strengthLookup, candidateLimit, databaseNetworkObject.f)


def selectBeamCandidatesConceptColumns(columnIndices, featureIndices, activationValues, strengthLookup, candidateLimit, maxFeatures):
	if(activationValues.numel() == 0):
		return []
	uniqueColumns, inverseIndices = pt.unique(columnIndices, return_inverse=True)
	columnActivationTotals = pt.zeros(uniqueColumns.shape[0], dtype=activationValues.dtype, device=activationValues.device)
	columnActivationTotals.scatter_add_(0, inverseIndices, activationValues)
	selectionCount = min(candidateLimit, columnActivationTotals.shape[0])
	_, columnRanks = pt.topk(columnActivationTotals, selectionCount)
	candidates = []
	for rankTensor in columnRanks:
		columnTensorIndex = rankTensor.item()
		columnIndex = uniqueColumns[columnTensorIndex].item()
		mask = (inverseIndices == columnTensorIndex)
		columnFeatures = featureIndices[mask]
		columnFeatureActivations = activationValues[mask]
		if(columnFeatures.numel() == 0):
			continue
		nodeThreshold = inferenceBeamConceptColumnNodeActivationThreshold
		if(nodeThreshold > 0):
			activeMask = columnFeatureActivations >= nodeThreshold
		else:
			activeMask = columnFeatureActivations > 0
		if(activeMask.sum() == 0):
			#fallback to most active feature
			maxIdx = pt.argmax(columnFeatureActivations)
			activeMask = pt.zeros_like(columnFeatureActivations, dtype=pt.bool)
			activeMask[maxIdx] = True
		selectedFeatures = columnFeatures[activeMask]
		selectedActivations = columnFeatureActivations[activeMask]
		nodes = []
		connectionSum = 0.0
		for featureTensor in selectedFeatures:
			featureIndex = featureTensor.item()
			nodes.append((columnIndex, featureIndex))
			connectionSum += getConnectionValue(strengthLookup, columnIndex, featureIndex, maxFeatures)
		if(len(nodes) == 0):
			continue
		meanActivation = selectedActivations.mean().item()
		meanConnection = connectionSum/len(nodes)
		candidates.append({"columnIndex": columnIndex, "featureIndex": nodes[0][1], "nodes": nodes, "connectionValue": meanConnection, "activationValue": columnActivationTotals[columnTensorIndex].item()})
	return candidates


def selectBeamCandidatesInstanceNodes(columnIndices, featureIndices, activationValues, strengthLookup, candidateLimit, maxFeatures):
	if(activationValues.numel() == 0):
		return []
	useColumnPreferences = (inferenceBeamInstancePreferActiveNodeCounts or
		inferenceBeamInstancePreferInternalConnectivity or
		inferenceBeamInstancePreferAdjacentOverlap)
	if(not useColumnPreferences):
		return selectTopInstanceNodesByActivation(columnIndices, featureIndices, activationValues, strengthLookup, candidateLimit, maxFeatures)
	columnData = buildInstanceColumnData(columnIndices, featureIndices, activationValues)
	if(len(columnData) == 0):
		return selectTopInstanceNodesByActivation(columnIndices, featureIndices, activationValues, strengthLookup, candidateLimit, maxFeatures)
	columnScores = computeInstanceColumnScores(columnData, strengthLookup, maxFeatures, activationValues.device, activationValues.dtype)
	if(len(columnScores) == 0):
		return selectTopInstanceNodesByActivation(columnIndices, featureIndices, activationValues, strengthLookup, candidateLimit, maxFeatures)
	sortedColumns = sorted(columnScores.items(), key=lambda item: item[1], reverse=True)
	candidates = []
	threshold = inferenceBeamInstanceNodeActivationThreshold
	for columnIndex, _ in sortedColumns:
		columnEntry = columnData[columnIndex]
		activationsTensor = pt.tensor(columnEntry["activations"], device=activationValues.device, dtype=activationValues.dtype)
		featuresList = columnEntry["features"]
		if(activationsTensor.numel() == 0):
			continue
		order = pt.argsort(activationsTensor, descending=True)
		for idx in order.tolist():
			value = activationsTensor[idx].item()
			featureIndex = featuresList[idx]
			if(threshold > 0 and value < threshold and len(candidates) > 0):
				continue
			connectionValue = getConnectionValue(strengthLookup, columnIndex, featureIndex, maxFeatures)
			candidates.append({"columnIndex": columnIndex, "featureIndex": featureIndex, "nodes": [(columnIndex, featureIndex)], "connectionValue": connectionValue})
			break
		if(len(candidates) == candidateLimit):
			break
	if(len(candidates) == 0):
		return selectTopInstanceNodesByActivation(columnIndices, featureIndices, activationValues, strengthLookup, candidateLimit, maxFeatures)
	return candidates


def selectTopInstanceNodesByActivation(columnIndices, featureIndices, activationValues, strengthLookup, candidateLimit, maxFeatures):
	selectionCount = min(candidateLimit, activationValues.shape[0])
	values, indices = pt.topk(activationValues, selectionCount)
	candidates = []
	threshold = inferenceBeamInstanceNodeActivationThreshold
	selectedCount = 0
	for rankIndex, activationIndex in enumerate(indices.tolist()):
		value = values[rankIndex].item()
		if(threshold > 0 and value < threshold and selectedCount > 0):
			continue
		columnIndex = columnIndices[activationIndex].item()
		featureIndex = featureIndices[activationIndex].item()
		connectionValue = getConnectionValue(strengthLookup, columnIndex, featureIndex, maxFeatures)
		candidates.append({"columnIndex": columnIndex, "featureIndex": featureIndex, "nodes": [(columnIndex, featureIndex)], "connectionValue": connectionValue})
		selectedCount += 1
		if(selectedCount == selectionCount):
			break
	if(len(candidates) == 0 and indices.shape[0] > 0):
		columnIndex = columnIndices[indices[0]].item()
		featureIndex = featureIndices[indices[0]].item()
		connectionValue = getConnectionValue(strengthLookup, columnIndex, featureIndex, maxFeatures)
		candidates.append({"columnIndex": columnIndex, "featureIndex": featureIndex, "nodes": [(columnIndex, featureIndex)], "connectionValue": connectionValue})
	return candidates


def buildInstanceColumnData(columnIndices, featureIndices, activationValues):
	columnData = {}
	for idx in range(columnIndices.shape[0]):
		columnIndex = columnIndices[idx].item()
		featureIndex = featureIndices[idx].item()
		activationValue = activationValues[idx].item()
		if(columnIndex not in columnData):
			columnData[columnIndex] = {"features": [], "activations": []}
		columnData[columnIndex]["features"].append(featureIndex)
		columnData[columnIndex]["activations"].append(activationValue)
	return columnData


def computeInstanceColumnScores(columnData, strengthLookup, maxFeatures, device, dtype):
	columnScores = {}
	activeFeatureSets = {}
	for columnIndex, data in columnData.items():
		activeFeatures = set()
		for featureIndex, activationValue in zip(data["features"], data["activations"]):
			if(activationValue > 0):
				activeFeatures.add(featureIndex)
		activeFeatureSets[columnIndex] = activeFeatures
	for columnIndex, data in columnData.items():
		activationsTensor = pt.tensor(data["activations"], device=device, dtype=dtype)
		featuresList = data["features"]
		if(activationsTensor.numel() == 0):
			continue
		baseScore = activationsTensor.max().item()
		totalScore = baseScore
		if(inferenceBeamInstancePreferActiveNodeCounts):
			if(inferenceBeamInstanceNodeActivationThreshold > 0):
				activeCount = sum(activationValue >= inferenceBeamInstanceNodeActivationThreshold for activationValue in data["activations"])
			else:
				activeCount = sum(activationValue > 0 for activationValue in data["activations"])
			totalScore += float(activeCount)
		if(inferenceBeamInstancePreferInternalConnectivity):
			connectivityValues = []
			for featureIndex, activationValue in zip(featuresList, data["activations"]):
				if(activationValue > 0):
					connectivityValues.append(getConnectionValue(strengthLookup, columnIndex, featureIndex, maxFeatures))
			if(len(connectivityValues) > 0):
				totalScore += sum(connectivityValues)/len(connectivityValues)
		if(inferenceBeamInstancePreferAdjacentOverlap):
			overlapScore = computeAdjacentOverlapScore(columnIndex, activeFeatureSets)
			totalScore += overlapScore
		columnScores[columnIndex] = totalScore
	return columnScores


def computeAdjacentOverlapScore(columnIndex, activeFeatureSets):
	currentSet = activeFeatureSets.get(columnIndex, set())
	if(len(currentSet) == 0):
		return 0.0
	previousOverlap = len(currentSet.intersection(activeFeatureSets.get(columnIndex-1, set())))
	nextOverlap = len(currentSet.intersection(activeFeatureSets.get(columnIndex+1, set())))
	return float(max(previousOverlap, nextOverlap))


def computeBeamNodeScore(activationValue, connectionValue):
	strategy = inferenceBeamScoreStrategy
	if(strategy == "connection"):
		return connectionValue
	elif(strategy == "activation_connection"):
		return activationValue + connectionValue
	elif(strategy == "nodeActivation"):
		return activationValue


def convertNodesToPrediction(nodes):
        if(len(nodes) == 0):
                return pt.tensor([], dtype=pt.long), pt.tensor([], dtype=pt.long)
        conceptColumnsIndicesNextList = []
        conceptColumnsFeatureIndicesNextList = []
	for columnIndex, featureIndex in nodes:
		conceptColumnsIndicesNextList.append(columnIndex)
		conceptColumnsFeatureIndicesNextList.append(featureIndex)
	conceptColumnsIndicesNext = pt.tensor(conceptColumnsIndicesNextList, dtype=pt.long)
	conceptColumnsFeatureIndicesNext = pt.tensor(conceptColumnsFeatureIndicesNextList, dtype=pt.long).unsqueeze(1)
	return conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext


def aggregateSparseColumnFeatureValues(sparseTensor, maxFeatures):
        if(sparseTensor is None):
                return None, None, None
        sparseTensor = sparseTensor.coalesce()
        if(sparseTensor._nnz() == 0):
		return None, None, None
	indices = sparseTensor.indices()
	values = sparseTensor.values()
	columnIndices = indices[1]
	featureIndices = indices[2]
	keys = columnIndices * maxFeatures + featureIndices
	uniqueKeys, inverseIndices = pt.unique(keys, return_inverse=True)
	aggregatedValues = pt.zeros((uniqueKeys.shape[0],), dtype=values.dtype, device=values.device)
	aggregatedValues.scatter_add_(0, inverseIndices, values)
	aggregatedColumns = uniqueKeys // maxFeatures
	aggregatedFeatures = uniqueKeys % maxFeatures
	return aggregatedColumns, aggregatedFeatures, aggregatedValues


def buildStrengthLookup(globalFeatureNeuronsStrength, maxFeatures):
	columnIndices, featureIndices, values = aggregateSparseColumnFeatureValues(globalFeatureNeuronsStrength, maxFeatures)
	if(columnIndices is None):
		return None
	strengthLookup = {}
	for idx in range(columnIndices.shape[0]):
		columnIndex = columnIndices[idx].item()
		featureIndex = featureIndices[idx].item()
		key = columnIndex * maxFeatures + featureIndex
		strengthLookup[key] = values[idx].item()
	return strengthLookup


def getConnectionValue(strengthLookup, columnIndex, featureIndex, maxFeatures):
	if(strengthLookup is None):
		return 0.0
	key = columnIndex * maxFeatures + featureIndex
	return strengthLookup.get(key, 0.0)


def computeCandidateActivationGain(newStateFeatures, oldStateFeatures, candidateNodes):
	if(len(candidateNodes) == 0):
		return 0.0
	newStateFeatures = newStateFeatures.coalesce()
	oldStateFeatures = oldStateFeatures.coalesce()
	newValues = newStateFeatures.values()
	oldValues = oldStateFeatures.values()
	if(newValues.numel() == 0 and oldValues.numel() == 0):
		return 0.0
	device = newStateFeatures.device
	size = newStateFeatures.size()
	newIndices = newStateFeatures.indices()
	oldIndices = oldStateFeatures.indices()
	combinedIndices = pt.cat([newIndices, oldIndices], dim=1)
	combinedValues = pt.cat([newValues, -oldValues], dim=0)
	deltaTensor = pt.sparse_coo_tensor(combinedIndices, combinedValues, size=size, device=device).coalesce()
	if(deltaTensor._values().numel() == 0):
		return 0.0
	indices = deltaTensor.indices()
	values = deltaTensor.values()
	positiveMask = values > 0
	if(positiveMask.sum() == 0):
		return 0.0
	return values[positiveMask].sum().item()
