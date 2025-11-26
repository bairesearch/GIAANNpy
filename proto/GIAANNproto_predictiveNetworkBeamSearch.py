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
import GIAANNproto_databaseNetworkTrain	#low level processFeaturesActivePredict functions currently stored here
import GIAANNproto_sparseTensors


def beamSearchPredictNextFeature(sequenceObservedColumns, databaseNetworkObject, observedColumnsDict, globalFeatureNeuronsActivation, globalFeatureNeuronsStrength, globalFeatureConnectionsActivation, globalFeatureNeuronsTime, wordsSequence, lemmasSequence, wordPredictionIndex, sequenceWordIndex, conceptMask):
	#generate targets for debug/analysis output
	targetMultipleSources, targetPreviousColumnIndex, targetNextColumnIndex, targetFeatureIndex, targetConceptColumnsIndices, targetConceptColumnsFeatureIndices = GIAANNproto_databaseNetwork.getTokenConceptFeatureIndexTensor(sequenceObservedColumns, wordsSequence, lemmasSequence, conceptMask, sequenceWordIndex, kcNetwork)

	if(inferenceBeamDepth <= 0 or inferenceBeamWidth <= 0):
		return selectMostActiveFeature(sequenceObservedColumns, globalFeatureNeuronsActivation, globalFeatureNeuronsStrength, wordsSequence, lemmasSequence, wordPredictionIndex, sequenceWordIndex, conceptMask)

	strengthDense = None
	if(globalFeatureNeuronsStrength is not None):
		globalFeatureNeuronsStrengthAllSegments = pt.sum(globalFeatureNeuronsStrength, dim=0)
		strengthDense = globalFeatureNeuronsStrengthAllSegments.to_dense()

	initialState = initialiseBeamActivationState(globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, globalFeatureNeuronsTime)
	beams = [{"score": 0.0, "state": initialState, "sequence": []}]
	completedBeams = []
	beamDepth = max(1, inferenceBeamDepth)
	beamWidthLimit = max(1, inferenceBeamWidth)

	for depthIndex in range(beamDepth):
		newBeams = []
		for beam in beams:
			activationDense = pt.sum(beam["state"]["features"], dim=0).to_dense()
			candidates = selectBeamCandidates(activationDense, strengthDense, beamWidthLimit)
			if(len(candidates) == 0):
				completedBeams.append(beam)
				continue
			for candidate in candidates:
				predictInfo = describeBeamCandidate(databaseNetworkObject, candidate)
				if(printPredictionsDuringInferencePredict):
					print("\t"*(depthIndex+2) + f"Predicting beam node(s): {predictInfo}")	# Debug: print beam depth and the node(s)/column being predicted
				newState = cloneBeamActivationState(beam["state"])
				for nodeColumn, nodeFeature in candidate["nodes"]:
					executeBeamNodeActivation(databaseNetworkObject, observedColumnsDict, newState, nodeColumn, nodeFeature, sequenceWordIndex)
				newSequence = beam["sequence"] + [candidate]
				newScore = beam["score"] + candidate["score"]
				newBeams.append({"score": newScore, "state": newState, "sequence": newSequence})
		if(len(newBeams) == 0):
			break
		newBeams.sort(key=lambda item: item["score"], reverse=True)
		beams = newBeams[:beamWidthLimit]

	if(len(beams) == 0):
		beams = completedBeams
	if(len(beams) == 0 or len(beams[0]["sequence"]) == 0):
		return selectMostActiveFeature(sequenceObservedColumns, globalFeatureNeuronsActivation, globalFeatureNeuronsStrength, wordsSequence, lemmasSequence, wordPredictionIndex, sequenceWordIndex, conceptMask)

	bestBeam = max(beams, key=lambda item: item["score"])
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


def selectBeamCandidates(activationDense, strengthDense, candidateLimit):
	if(activationDense.numel() == 0):
		return []
	candidateLimit = max(1, candidateLimit)
	if(inferenceBeamSearchConceptColumns):
		return selectBeamCandidatesConceptColumns(activationDense, strengthDense, candidateLimit)
	else:
		return selectBeamCandidatesInstanceNodes(activationDense, strengthDense, candidateLimit)


def selectBeamCandidatesConceptColumns(activationDense, strengthDense, candidateLimit):
	conceptActivations = pt.sum(activationDense, dim=1)
	numColumns = conceptActivations.shape[0]
	if(numColumns == 0):
		return []
	selectionCount = min(candidateLimit, numColumns)
	columnValues, columnIndices = pt.topk(conceptActivations, selectionCount)
	candidates = []
	for idx in columnIndices.tolist():
		columnActivation = activationDense[idx]
		nodeThreshold = inferenceBeamConceptColumnNodeActivationThreshold
		if(nodeThreshold > 0):
			activeNodes = (columnActivation >= nodeThreshold).nonzero(as_tuple=True)[0].tolist()
		else:
			activeNodes = (columnActivation > 0).nonzero(as_tuple=True)[0].tolist()
		if(len(activeNodes) == 0):
			activeNodes = [pt.argmax(columnActivation).item()]
		nodes = [(idx, featureIndex) for featureIndex in activeNodes]
		score = 0.0
		for _, featureIndex in nodes:
			activationValue = columnActivation[featureIndex].item()
			connectionValue = 0.0
			if(strengthDense is not None):
				connectionValue = strengthDense[idx, featureIndex].item()
			score += computeBeamNodeScore(activationValue, connectionValue)
		candidateFeature = nodes[0][1]
		candidates.append({"columnIndex": idx, "featureIndex": candidateFeature, "nodes": nodes, "score": score})
	return candidates


def selectBeamCandidatesInstanceNodes(activationDense, strengthDense, candidateLimit):
	c, f = activationDense.shape
	if(c == 0 or f == 0):
		return []
	flatActivations = activationDense.view(-1)
	selectionCount = min(candidateLimit, flatActivations.shape[0])
	values, indices = pt.topk(flatActivations, selectionCount)
	candidates = []
	threshold = inferenceBeamInstanceNodeActivationThreshold
	selectedCount = 0
	for rankIndex, flatIndex in enumerate(indices.tolist()):
		value = values[rankIndex].item()
		if(threshold > 0 and value < threshold and selectedCount > 0):
			continue
		columnIndex = flatIndex // f
		featureIndex = flatIndex % f
		connectionValue = 0.0
		if(strengthDense is not None):
			connectionValue = strengthDense[columnIndex, featureIndex].item()
		score = computeBeamNodeScore(value, connectionValue)
		candidates.append({"columnIndex": columnIndex, "featureIndex": featureIndex, "nodes": [(columnIndex, featureIndex)], "score": score})
		selectedCount += 1
		if(selectedCount == selectionCount):
			break
	if(len(candidates) == 0 and indices.shape[0] > 0):
		flatIndex = indices[0].item()
		columnIndex = flatIndex // f
		featureIndex = flatIndex % f
		value = values[0].item()
		connectionValue = 0.0
		if(strengthDense is not None):
			connectionValue = strengthDense[columnIndex, featureIndex].item()
		score = computeBeamNodeScore(value, connectionValue)
		candidates.append({"columnIndex": columnIndex, "featureIndex": featureIndex, "nodes": [(columnIndex, featureIndex)], "score": score})
	return candidates


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
