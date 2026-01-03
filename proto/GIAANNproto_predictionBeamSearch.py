"""GIAANNproto_predictionBeamSearch.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_predictionBeamSearch.py

# Usage:
see GIAANNproto_predictionBeamSearch.py

# Description: 
GIA ANN proto prediction Beam Search

"""

import torch as pt

from GIAANNproto_globalDefs import *
import GIAANNproto_databaseNetworkExcitation
import GIAANNproto_sparseTensors
import GIAANNproto_predictionActivate

def beamSearchPredictNextFeature(sequenceObservedColumns, databaseNetworkObject, observedColumnsDict, globalFeatureNeuronsActivation, globalFeatureNeuronsStrength, globalFeatureConnectionsActivation, globalFeatureNeuronsTime, tokensSequence, wordPredictionIndex, sequenceWordIndex, conceptMask, allowedColumns=None, constraintMode=None, selectMostActiveFeatureFunc=None, conceptActivationState=None, connectedColumnsConstraint=None, connectedColumnsFeatures=None):
	#generate targets for debug/analysis output
	targetMultipleSources, targetPreviousColumnIndex, targetNextColumnIndex, targetFeatureIndex, targetConceptColumnsIndices, targetConceptColumnsFeatureIndices = GIAANNproto_databaseNetworkExcitation.getTokenConceptFeatureIndexTensor(sequenceObservedColumns, tokensSequence, conceptMask, sequenceWordIndex, kcNetwork)

	if(selectMostActiveFeatureFunc is None):
		raise ValueError("beamSearchPredictNextFeature requires selectMostActiveFeatureFunc to be provided to avoid circular imports")

	if(inferenceBeamDepth <= 0 or inferenceBeamWidth <= 0):
		return selectMostActiveFeatureFunc(sequenceObservedColumns, globalFeatureNeuronsActivation, globalFeatureNeuronsStrength, globalFeatureNeuronsTime, tokensSequence, wordPredictionIndex, sequenceWordIndex, conceptMask, allowedColumns, constraintMode, conceptActivationState, connectedColumnsConstraint, connectedColumnsFeatures)

	strengthLookup = None
	if(globalFeatureNeuronsStrength is not None):
		strengthLookup = buildStrengthLookup(globalFeatureNeuronsStrength, databaseNetworkObject.f)
	sequenceColumnIndex = None
	if(inferenceUseNeuronFeaturePropertiesTime):
		if(useSANIcolumns or useSANIfeaturesAndColumns):
			sequenceColumnIndex = GIAANNproto_predictionActivate.calculateSequenceColumnIndex(conceptMask, sequenceWordIndex)
	if(inferenceUseNeuronFeaturePropertiesTime and debugInferenceUseNeuronFeaturePropertiesTimeTargeted):
		if(debugInferenceUseNeuronFeaturePropertiesTimeTargetSequenceWordIndex is None or sequenceWordIndex == debugInferenceUseNeuronFeaturePropertiesTimeTargetSequenceWordIndex):
			if(debugInferenceUseNeuronFeaturePropertiesTimeTargetColumnName is None or debugInferenceUseNeuronFeaturePropertiesTimeTargetFeatureNames is None):
				raise RuntimeError("beamSearchPredictNextFeature: debug target column/feature names not configured")
			if(debugInferenceUseNeuronFeaturePropertiesTimeTargetColumnName not in databaseNetworkObject.conceptColumnsList):
				raise RuntimeError("beamSearchPredictNextFeature: debug target column not found")
			if(globalFeatureNeuronsActivation is None):
				raise RuntimeError("beamSearchPredictNextFeature: globalFeatureNeuronsActivation is None")
			if(globalFeatureNeuronsTime is None):
				raise RuntimeError("beamSearchPredictNextFeature: globalFeatureNeuronsTime is None")
			debugTargetColumnIndex = databaseNetworkObject.conceptColumnsList.index(debugInferenceUseNeuronFeaturePropertiesTimeTargetColumnName)
			debugTargetFeatureIndices = [databaseNetworkObject.conceptFeaturesList.index(featureName) if featureName in databaseNetworkObject.conceptFeaturesList else -1 for featureName in debugInferenceUseNeuronFeaturePropertiesTimeTargetFeatureNames]
			if(-1 in debugTargetFeatureIndices):
				raise RuntimeError("beamSearchPredictNextFeature: debug target feature not found")
			debugDevice = globalFeatureNeuronsActivation.device if globalFeatureNeuronsActivation is not None else deviceSparse
			debugBranchCount = numberOfDendriticBranches if multipleDendriticBranches else 1
			debugBranchRange = pt.arange(debugBranchCount, dtype=pt.long, device=debugDevice)
			debugSegmentRange = pt.arange(arrayNumberOfSegments, dtype=pt.long, device=debugDevice)
			debugBranchIndices = debugBranchRange.repeat_interleave(arrayNumberOfSegments)
			debugSegmentIndices = debugSegmentRange.repeat(debugBranchCount)
			debugValueDtype = globalFeatureNeuronsActivation.values().dtype if globalFeatureNeuronsActivation.is_sparse else globalFeatureNeuronsActivation.dtype
			for debugFeatureIndex, debugFeatureName in zip(debugTargetFeatureIndices, debugInferenceUseNeuronFeaturePropertiesTimeTargetFeatureNames):
				debugColumnIndices = pt.full_like(debugSegmentIndices, debugTargetColumnIndex)
				debugFeatureIndices = pt.full_like(debugSegmentIndices, debugFeatureIndex)
				debugIndices = pt.stack([debugBranchIndices, debugSegmentIndices, debugColumnIndices, debugFeatureIndices], dim=0)
				debugActivationValues = GIAANNproto_predictionActivate.gatherSparseTensorValuesAtIndices(globalFeatureNeuronsActivation, debugIndices, debugValueDtype)
				debugStoredTimes = GIAANNproto_predictionActivate.gatherSparseTensorValuesAtIndices(globalFeatureNeuronsTime, debugIndices, debugValueDtype)
				debugPenaltyValues = GIAANNproto_predictionActivate.computeTimePenaltyForSegments(debugSegmentIndices, debugStoredTimes, sequenceWordIndex, sequenceColumnIndex)
				debugModifiedValues = debugActivationValues - debugPenaltyValues
				debugActivationByBranch = debugActivationValues.view(debugBranchCount, arrayNumberOfSegments).sum(dim=1)
				debugPenaltyByBranch = debugPenaltyValues.view(debugBranchCount, arrayNumberOfSegments).sum(dim=1)
				debugModifiedByBranch = debugModifiedValues.view(debugBranchCount, arrayNumberOfSegments).sum(dim=1)
				debugActivationBySegment = debugActivationValues.view(debugBranchCount, arrayNumberOfSegments).max(dim=0).values
				debugPenaltyBySegment = debugPenaltyValues.view(debugBranchCount, arrayNumberOfSegments).max(dim=0).values
				debugModifiedBySegment = debugModifiedValues.view(debugBranchCount, arrayNumberOfSegments).max(dim=0).values
				debugStoredTimeBySegment = debugStoredTimes.view(debugBranchCount, arrayNumberOfSegments).max(dim=0).values
				print(f"debugInferenceUseNeuronFeaturePropertiesTimeTarget: sequenceWordIndex={sequenceWordIndex}, sequenceColumnIndex={sequenceColumnIndex}, column={debugInferenceUseNeuronFeaturePropertiesTimeTargetColumnName}({debugTargetColumnIndex}), feature={debugFeatureName}({debugFeatureIndex}), activationByBranch={debugActivationByBranch.tolist()}, penaltyByBranch={debugPenaltyByBranch.tolist()}, modifiedByBranch={debugModifiedByBranch.tolist()}, activationBySegmentMax={debugActivationBySegment.tolist()}, penaltyBySegmentMax={debugPenaltyBySegment.tolist()}, modifiedBySegmentMax={debugModifiedBySegment.tolist()}, storedTimeBySegmentMax={debugStoredTimeBySegment.tolist()}")
	initialConstraintState = createConstraintStateForBeam(allowedColumns, constraintMode)
	initialState = initialiseBeamActivationState(globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, globalFeatureNeuronsTime, conceptActivationState)
	beams = [{"score": 0.0, "state": initialState, "sequence": [], "constraintState": initialConstraintState, "connectedColumns": connectedColumnsConstraint, "connectedColumnsFeatures": connectedColumnsFeatures}]
	completedBeams = []
	beamDepth = max(1, inferenceBeamDepth)
	beamWidthLimit = max(1, inferenceBeamWidth)

	for depthIndex in range(beamDepth):
		newBeams = []
		for beam in beams:
			candidates = selectBeamCandidates(beam["state"]["features"], beam["state"].get("time"), strengthLookup, beamWidthLimit, databaseNetworkObject, beam.get("constraintState"), beam["state"].get("conceptActivations"), beam.get("connectedColumns"), beam.get("connectedColumnsFeatures"), sequenceWordIndex, sequenceColumnIndex)
			if(len(candidates) == 0):
				completedBeams.append(beam)
				continue
			for candidate in candidates:
				predictInfo = describeBeamCandidate(databaseNetworkObject, candidate)
				if(printPredictionsDuringInferencePredictBeamSearch):
					 print("\t"*(depthIndex+2) + f"Predicting beam node(s): {predictInfo}")	   # Debug: print beam depth and the node(s)/column being predicted
				oldState = beam["state"]
				newState = cloneBeamActivationState(oldState)
				for nodeColumn, nodeFeature in candidate["nodes"]:
					executeBeamNodeActivation(databaseNetworkObject, observedColumnsDict, newState, nodeColumn, nodeFeature, sequenceWordIndex, sequenceColumnIndex)
				newSequence = beam["sequence"] + [candidate]
				activationGain = computeCandidateActivationGain(newState["features"], oldState["features"], candidate["nodes"])
				if(inferenceBeamScoreStrategy == "nodeActivation"):
					if(inferenceUseNeuronFeaturePropertiesTime):
						# spec step (b): score beam nodes using time-modified activation.
						activationGain = candidate.get("activationValue", activationGain)
					elif(inferenceBeamSearchConceptColumns):
						activationGain = candidate.get("activationValue", activationGain)
				candidateScore = computeBeamNodeScore(activationGain, candidate["connectionValue"])
				newScore = beam["score"] + candidateScore
				newConstraintState = updateConstraintStateAfterNodes(databaseNetworkObject, beam.get("constraintState"), candidate["nodes"])
				nextConnectedColumns, nextConnectedFeatures = buildConnectedColumnsLookupForBeamNodes(databaseNetworkObject, observedColumnsDict, candidate["nodes"])
				newBeams.append({"score": newScore, "state": newState, "sequence": newSequence, "constraintState": newConstraintState, "connectedColumns": nextConnectedColumns, "connectedColumnsFeatures": nextConnectedFeatures})
		if(len(newBeams) == 0):
			break
		newBeams.sort(key=lambda item: item["score"], reverse=True)
		beams = newBeams[:beamWidthLimit]

	if(len(beams) == 0):
		beams = completedBeams
	if(len(beams) == 0 or len(beams[0]["sequence"]) == 0):
		return selectMostActiveFeatureFunc(sequenceObservedColumns, globalFeatureNeuronsActivation, globalFeatureNeuronsStrength, globalFeatureNeuronsTime, tokensSequence, wordPredictionIndex, sequenceWordIndex, conceptMask, allowedColumns, constraintMode, conceptActivationState, connectedColumnsConstraint, connectedColumnsFeatures)

	allBeams = beams + completedBeams
	bestBeam = max(allBeams, key=lambda item: item["score"])
	bestAction = bestBeam["sequence"][0]
	conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext = convertNodesToPrediction(bestAction["nodes"])
	if(conceptColumnsIndicesNext.shape[0] == 0):
		return selectMostActiveFeatureFunc(sequenceObservedColumns, globalFeatureNeuronsActivation, globalFeatureNeuronsStrength, globalFeatureNeuronsTime, tokensSequence, wordPredictionIndex, sequenceWordIndex, conceptMask, allowedColumns, constraintMode, conceptActivationState, connectedColumnsConstraint, connectedColumnsFeatures)
	multipleSourcesNext = conceptColumnsIndicesNext.shape[0] > 1
	kc = conceptColumnsIndicesNext.shape[0]
	if(printPredictionsDuringInferencePredict):
		printBestBeamPath(bestBeam, databaseNetworkObject)
	conceptColumnsIndicesPred = conceptColumnsIndicesNext.clone()
	conceptColumnsFeatureIndicesPred = conceptColumnsFeatureIndicesNext.clone()
	return conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext, multipleSourcesNext, kc, conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, targetMultipleSources, targetPreviousColumnIndex, targetNextColumnIndex


def initialiseBeamActivationState(globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, globalFeatureNeuronsTime, conceptActivationState):
	state = {"features": globalFeatureNeuronsActivation.clone()}
	if(transformerUseInputConnections and globalFeatureConnectionsActivation is not None):
		state["connections"] = globalFeatureConnectionsActivation.clone()
	else:
		state["connections"] = None
	if(inferenceUseNeuronFeaturePropertiesTime and globalFeatureNeuronsTime is not None):
		state["time"] = globalFeatureNeuronsTime.clone()
	else:
		state["time"] = None
	if(conceptActivationState is not None):
		state["conceptActivations"] = set(conceptActivationState)
	else:
		state["conceptActivations"] = None
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
	if("conceptActivations" in state and state["conceptActivations"] is not None):
		clonedState["conceptActivations"] = set(state["conceptActivations"])
	else:
		clonedState["conceptActivations"] = None
	return clonedState


def executeBeamNodeActivation(databaseNetworkObject, observedColumnsDict, state, columnIndex, featureIndex, sequenceWordIndex, sequenceColumnIndex):
	lemma = databaseNetworkObject.conceptColumnsList[columnIndex]
	if(lemma in observedColumnsDict):
		observedColumn = observedColumnsDict[lemma]
	else:
		observedColumn = GIAANNproto_databaseNetworkExcitation.loadOrCreateObservedColumn(databaseNetworkObject, columnIndex, lemma, sequenceWordIndex)
	observedColumnsDict[lemma] = observedColumn
	featureConnections = observedColumn.featureConnections
	conceptColumnsIndicesSource = pt.tensor([columnIndex], dtype=pt.long, device=deviceSparse)
	conceptColumnsFeatureIndicesSource = pt.tensor([[featureIndex]], dtype=pt.long, device=deviceSparse)
	state["features"], state["connections"], state["time"] = GIAANNproto_predictionActivate.processFeaturesActivePredict(databaseNetworkObject, state["features"], state["connections"], featureConnections, conceptColumnsIndicesSource, conceptColumnsFeatureIndicesSource, columnIndex, state.get("time"), sequenceWordIndex, sequenceColumnIndex, sequenceWordIndex, sequenceColumnIndex)
	applyBeamNodePredictionEffects(state, columnIndex, featureIndex, sequenceWordIndex)
	if(predictionColumnsMustActivateConceptFeature):
		conceptState = state.get("conceptActivations")
		if(conceptState is None):
			conceptState = set()
			state["conceptActivations"] = conceptState
		conceptState.add(columnIndex)
	return state


def applyBeamNodePredictionEffects(state, columnIndex, featureIndex, sequenceWordIndex):
	modifyActivation = inferenceDeactivateNeuronsUponPrediction or inferenceInvertNeuronActivationUponPrediction
	if(not modifyActivation):
		return
	branchIndex = 0
	if(multipleDendriticBranches):
		branchIndex = GIAANNproto_predictionActivate.selectActivatedBranchIndex(state["features"], columnIndex, featureIndex)
	indicesToUpdate = buildBeamNodeIndices(state["features"].device, columnIndex, featureIndex, branchIndex)
	if(modifyActivation):
		if(inferenceDeactivateNeuronsUponPrediction):
			modifier = 0
		else:
			modifier = inferenceInvertNeuronActivationUponPredictionLevel
		state["features"] = GIAANNproto_sparseTensors.modifySparseTensor(state["features"], indicesToUpdate, modifier, multiply=inferenceInvertNeuronActivationUponPrediction)


def buildBeamNodeIndices(device, columnIndex, featureIndex, branchIndex=0):
	indicesToUpdateList = []
	columnTensor = pt.tensor(columnIndex, dtype=pt.long, device=device)
	featureTensor = pt.tensor(featureIndex, dtype=pt.long, device=device)
	branchTensor = pt.tensor(branchIndex, dtype=pt.long, device=device)
	if(useSANI):
		for segmentIndex in range(arrayNumberOfSegments):
			segmentTensor = pt.tensor(segmentIndex, dtype=pt.long, device=device)
			indicesToUpdateList.append(pt.stack([branchTensor, segmentTensor, columnTensor, featureTensor], dim=0))
	else:
		segmentTensor = pt.tensor(arrayIndexSegmentFirst, dtype=pt.long, device=device)
		indicesToUpdateList.append(pt.stack([branchTensor, segmentTensor, columnTensor, featureTensor], dim=0))
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


def debugDescribeColumnFeatureName(databaseNetworkObject, columnIndex, featureIndex):
	if(0 <= columnIndex < len(databaseNetworkObject.conceptColumnsList)):
		columnName = databaseNetworkObject.conceptColumnsList[columnIndex]
	else:
		columnName = f"<invalid:{columnIndex}>"
	if(featureIndex == featureIndexConceptNeuron):
		featureName = "conceptNeuron"
	elif(0 <= featureIndex < len(databaseNetworkObject.conceptFeaturesList)):
		featureName = databaseNetworkObject.conceptFeaturesList[featureIndex]
	else:
		featureName = f"feature_{featureIndex}"
	return columnName, featureName


def debugDescribeAllowedBeamFeatures(databaseNetworkObject, connectedColumnsTensor, connectedColumnsFeatures):
	if(connectedColumnsTensor is None):
		return "<none>"
	if(connectedColumnsTensor.numel() == 0):
		return "[]"
	elements = []
	for columnValue in connectedColumnsTensor.cpu().tolist():
		columnName, _ = debugDescribeColumnFeatureName(databaseNetworkObject, columnValue, featureIndexConceptNeuron)
		if(connectedColumnsFeatures is not None and columnValue in connectedColumnsFeatures):
			featureList = []
			for featureIndex in sorted(connectedColumnsFeatures[columnValue]):
				_, featureName = debugDescribeColumnFeatureName(databaseNetworkObject, columnValue, featureIndex)
			featureList.append(f"{featureIndex}:{featureName}")
			elements.append(f"{columnName} -> [{', '.join(featureList)}]")
		else:
			elements.append(f"{columnName} -> <any feature>")
	return "[" + "; ".join(elements) + "]"


def filterCandidatesByActivationThreshold(columnIndices, featureIndices, activationValues):
	if(minimumPredictionActivationThreshold <= 0):
		return columnIndices, featureIndices, activationValues
	if(columnIndices is None or featureIndices is None or activationValues is None):
		return None, None, None
	if(columnIndices.numel() == 0 or activationValues.numel() == 0):
		return None, None, None
	activeMask = activationValues >= minimumPredictionActivationThreshold
	if(activeMask.sum().item() == 0):
		return None, None, None
	indexTensor = pt.nonzero(activeMask, as_tuple=False).view(-1)
	return columnIndices.index_select(0, indexTensor), featureIndices.index_select(0, indexTensor), activationValues.index_select(0, indexTensor)


def selectBeamCandidates(stateFeatures, stateTime, strengthLookup, candidateLimit, databaseNetworkObject, constraintState=None, conceptActivationState=None, connectedColumnsTensor=None, connectedColumnsFeatures=None, sequenceWordIndex=None, sequenceColumnIndex=None):
	candidateLimit = max(1, candidateLimit)
	if(inferenceUseNeuronFeaturePropertiesTime):
		# spec step (b): apply time-based activation modifier during beam candidate selection
		stateFeatures = GIAANNproto_predictionActivate.applyTimeBasedActivationModifier(stateFeatures, stateTime, sequenceWordIndex, sequenceColumnIndex)
	columnIndices, featureIndices, activationValues = aggregateSparseColumnFeatureValues(stateFeatures, databaseNetworkObject.f)
	if(columnIndices is None):
		return []
	columnIndices, featureIndices, activationValues = filterCandidatesByActivationThreshold(columnIndices, featureIndices, activationValues)
	columnIndices, featureIndices, activationValues = filterCandidatesByLastSegment(columnIndices, featureIndices, activationValues, stateFeatures, databaseNetworkObject.f)
	if(columnIndices is None):
		return []
	columnIndices, featureIndices, activationValues = filterCandidatesByConnectedColumns(columnIndices, featureIndices, activationValues, connectedColumnsTensor, connectedColumnsFeatures, databaseNetworkObject)
	if(columnIndices is None):
		return []
	if(inferenceBeamSearchConceptColumns):
		return selectBeamCandidatesConceptColumns(columnIndices, featureIndices, activationValues, strengthLookup, candidateLimit, databaseNetworkObject.f, databaseNetworkObject, constraintState, conceptActivationState)
	else:
		return selectBeamCandidatesInstanceNodes(columnIndices, featureIndices, activationValues, strengthLookup, candidateLimit, databaseNetworkObject.f, databaseNetworkObject, constraintState, conceptActivationState)


def filterCandidatesByConnectedColumns(columnIndices, featureIndices, activationValues, connectedColumnsTensor, connectedColumnsFeatures=None, databaseNetworkObject=None):
	if(connectedColumnsTensor is None):
		return columnIndices, featureIndices, activationValues
	if(columnIndices is None or featureIndices is None or activationValues is None):
		return None, None, None
	if(connectedColumnsTensor.numel() == 0):
		return None, None, None
	if(columnIndices.numel() == 0):
		return None, None, None
	device = columnIndices.device
	connectedColumnsDevice = connectedColumnsTensor.to(device)
	if(connectedColumnsDevice.numel() == 0):
		return None, None, None
	allowedColumnsSet = set(connectedColumnsDevice.tolist())
	selectedIndices = []
	if(connectedColumnsFeatures is not None and len(connectedColumnsFeatures) > 0):
		for idx in range(columnIndices.shape[0]):
			columnValue = int(columnIndices[idx].item())
			if(columnValue not in allowedColumnsSet):
				continue
			featureAllowed = connectedColumnsFeatures.get(columnValue)
			if(featureAllowed is None or len(featureAllowed) == 0):
				continue
			featureValue = int(featureIndices[idx].item())
			if(featureValue not in featureAllowed):
				continue
			selectedIndices.append(idx)
	else:
		comparison = columnIndices.unsqueeze(0) == connectedColumnsDevice.unsqueeze(1)
		mask = comparison.any(dim=0)
		selectedIndices = pt.nonzero(mask, as_tuple=False).view(-1).tolist()
	if(len(selectedIndices) == 0):
		return None, None, None
	indexTensor = pt.tensor(selectedIndices, dtype=pt.long, device=columnIndices.device)
	return columnIndices.index_select(0, indexTensor), featureIndices.index_select(0, indexTensor), activationValues.index_select(0, indexTensor)

def filterCandidatesByLastSegment(columnIndices, featureIndices, activationValues, stateFeatures, maxFeatures):
	filteredColumns = columnIndices
	filteredFeatures = featureIndices
	filteredActivations = activationValues
	if(useSANI and algorithmMatrixSANImethod=="enforceActivationAcrossSegments" and algorithmMatrixSANIenforceRequirement=="enforceLastSegmentMustBeActive"):
		if(filteredColumns is None or filteredFeatures is None or filteredActivations is None):
			filteredColumns = None
			filteredFeatures = None
			filteredActivations = None
		else:
			# Patch: beam candidates ignored last-segment gating, so filter by last-segment-active nodes.
			if(enforceActivationAcrossSegmentsIgnoreInternalColumn):
				lastSegmentConstraint = arrayIndexSegmentAdjacentColumn
			else:
				lastSegmentConstraint = arrayIndexSegmentLast
			if(stateFeatures is None):
				lastSegmentActivation = None
			else:
				hasBranchDim = (stateFeatures.dim() == 4)
				if(stateFeatures.is_sparse):
					if(hasBranchDim):
						lastSegmentActivation = GIAANNproto_sparseTensors.sliceSparseTensor(stateFeatures, 1, lastSegmentConstraint)
					else:
						lastSegmentActivation = GIAANNproto_sparseTensors.sliceSparseTensor(stateFeatures, 0, lastSegmentConstraint)
				else:
					if(hasBranchDim):
						lastSegmentActivation = stateFeatures[:, lastSegmentConstraint]
					else:
						lastSegmentActivation = stateFeatures[lastSegmentConstraint]
			if(lastSegmentActivation is None):
				filteredColumns = None
				filteredFeatures = None
				filteredActivations = None
			else:
				if(not lastSegmentActivation.is_sparse):
					lastSegmentActivation = lastSegmentActivation.to_sparse()
				if(hasBranchDim and multipleDendriticBranches and lastSegmentActivation.dim() == 3):
					lastSegmentActivation = GIAANNproto_sparseTensors.reduceSparseBranchMax(lastSegmentActivation)
				elif(hasBranchDim):
					lastSegmentActivation = GIAANNproto_sparseTensors.collapseSparseBranchDimension(lastSegmentActivation)
				lastSegmentActivation = lastSegmentActivation.coalesce()
				if(lastSegmentActivation._nnz() == 0):
					filteredColumns = None
					filteredFeatures = None
					filteredActivations = None
				else:
					lastIndices = lastSegmentActivation.indices()
					lastKeys = (lastIndices[0].cpu() * maxFeatures + lastIndices[1].cpu()).tolist()
					if(len(lastKeys) == 0):
						filteredColumns = None
						filteredFeatures = None
						filteredActivations = None
					else:
						lastKeySet = set(int(value) for value in lastKeys)
						selectedIndices = []
						for idx in range(filteredColumns.shape[0]):
							key = int(filteredColumns[idx].item()) * maxFeatures + int(filteredFeatures[idx].item())
							if(key in lastKeySet):
								selectedIndices.append(idx)
						if(len(selectedIndices) == 0):
							filteredColumns = None
							filteredFeatures = None
							filteredActivations = None
						else:
							indexTensor = pt.tensor(selectedIndices, dtype=pt.long, device=filteredColumns.device)
							filteredColumns = filteredColumns.index_select(0, indexTensor)
							filteredFeatures = filteredFeatures.index_select(0, indexTensor)
							filteredActivations = filteredActivations.index_select(0, indexTensor)

	return filteredColumns, filteredFeatures, filteredActivations


def selectBeamCandidatesConceptColumns(columnIndices, featureIndices, activationValues, strengthLookup, candidateLimit, maxFeatures, databaseNetworkObject, constraintState=None, conceptActivationState=None):
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
		if(not constraintAllowsColumn(columnIndex, constraintState)):
			continue
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
		nodes = [(columnIndex, featureTensor.item()) for featureTensor in selectedFeatures]
		nodes, connectionSum = prepareBeamNodes(databaseNetworkObject, nodes, conceptActivationState, constraintState, strengthLookup, maxFeatures)
		if(len(nodes) == 0):
			continue
		meanActivation = selectedActivations.mean().item()
		meanConnection = connectionSum/len(nodes)
		candidates.append({"columnIndex": columnIndex, "featureIndex": nodes[0][1], "nodes": nodes, "connectionValue": meanConnection, "activationValue": columnActivationTotals[columnTensorIndex].item()})
	return candidates


def selectBeamCandidatesInstanceNodes(columnIndices, featureIndices, activationValues, strengthLookup, candidateLimit, maxFeatures, databaseNetworkObject, constraintState=None, conceptActivationState=None):
	if(activationValues.numel() == 0):
		return []
	useColumnPreferences = (inferenceBeamInstancePreferActiveNodeCounts or
		inferenceBeamInstancePreferInternalConnectivity or
		inferenceBeamInstancePreferAdjacentOverlap)
	if(not useColumnPreferences):
		return selectTopInstanceNodesByActivation(columnIndices, featureIndices, activationValues, strengthLookup, candidateLimit, maxFeatures, databaseNetworkObject, constraintState, conceptActivationState)
	columnData = buildInstanceColumnData(columnIndices, featureIndices, activationValues, constraintState)
	if(len(columnData) == 0):
		return selectTopInstanceNodesByActivation(columnIndices, featureIndices, activationValues, strengthLookup, candidateLimit, maxFeatures, databaseNetworkObject, constraintState, conceptActivationState)
	columnScores = computeInstanceColumnScores(columnData, strengthLookup, maxFeatures, activationValues.device, activationValues.dtype)
	if(len(columnScores) == 0):
		return selectTopInstanceNodesByActivation(columnIndices, featureIndices, activationValues, strengthLookup, candidateLimit, maxFeatures, databaseNetworkObject, constraintState, conceptActivationState)
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
			nodes = [(columnIndex, featureIndex)]
			nodes, adjustedConnection = prepareBeamNodes(databaseNetworkObject, nodes, conceptActivationState, constraintState, strengthLookup, maxFeatures)
			if(len(nodes) == 0):
				continue
				candidates.append({"columnIndex": nodes[0][0], "featureIndex": nodes[0][1], "nodes": nodes, "connectionValue": adjustedConnection, "activationValue": value})
			break
		if(len(candidates) == candidateLimit):
			break
	if(len(candidates) == 0):
		return selectTopInstanceNodesByActivation(columnIndices, featureIndices, activationValues, strengthLookup, candidateLimit, maxFeatures, databaseNetworkObject, constraintState, conceptActivationState)
	return candidates


def selectTopInstanceNodesByActivation(columnIndices, featureIndices, activationValues, strengthLookup, candidateLimit, maxFeatures, databaseNetworkObject, constraintState=None, conceptActivationState=None):
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
		if(not constraintAllowsColumn(columnIndex, constraintState)):
			continue
		featureIndex = featureIndices[activationIndex].item()
		nodes = [(columnIndex, featureIndex)]
		nodes, connectionValue = prepareBeamNodes(databaseNetworkObject, nodes, conceptActivationState, constraintState, strengthLookup, maxFeatures)
		if(len(nodes) == 0):
			continue
		candidates.append({"columnIndex": nodes[0][0], "featureIndex": nodes[0][1], "nodes": nodes, "connectionValue": connectionValue, "activationValue": value})
		selectedCount += 1
		if(selectedCount == selectionCount):
			break
	if(len(candidates) == 0 and indices.shape[0] > 0):
		columnIndex = columnIndices[indices[0]].item()
		if(not constraintAllowsColumn(columnIndex, constraintState)):
			return candidates
		featureIndex = featureIndices[indices[0]].item()
		connectionValue = getConnectionValue(strengthLookup, columnIndex, featureIndex, maxFeatures)
		candidates.append({"columnIndex": columnIndex, "featureIndex": featureIndex, "nodes": [(columnIndex, featureIndex)], "connectionValue": connectionValue, "activationValue": values[0].item()})
	return candidates


def buildInstanceColumnData(columnIndices, featureIndices, activationValues, constraintState=None):
	columnData = {}
	for idx in range(columnIndices.shape[0]):
		columnIndex = columnIndices[idx].item()
		if(not constraintAllowsColumn(columnIndex, constraintState)):
			continue
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

def tensorToColumnSet(allowedColumnsTensor):
	if(allowedColumnsTensor is None or allowedColumnsTensor.numel() == 0):
		return None
	columnList = allowedColumnsTensor.cpu().tolist()
	if(len(columnList) == 0):
		return None
	return set(columnList)

def createConstraintStateForBeam(allowedColumnsTensor, constraintMode):
	if(constraintMode not in ["internal", "external", "delimiter"]):
		return None
	columnSet = tensorToColumnSet(allowedColumnsTensor)
	if(columnSet is None or len(columnSet) == 0):
		return None
	return {"columns": columnSet, "mode": constraintMode}

def constraintAllowsColumn(columnIndex, constraintState):
	if(constraintState is None):
		return True
	allowedSet = constraintState.get("columns")
	constraintMode = constraintState.get("mode")
	if(allowedSet is None or constraintMode is None or len(allowedSet) == 0):
		return True
	if(constraintMode == "internal"):
		return columnIndex in allowedSet
	elif(constraintMode == "external"):
		return columnIndex not in allowedSet
	elif(constraintMode == "delimiter"):
		return True
	return True


def prepareBeamNodes(databaseNetworkObject, nodes, conceptActivationState, constraintState, strengthLookup, maxFeatures):
	preparedNodes = []
	connectionSum = 0.0
	delimiterMode = False
	allowedSet = None
	if(constraintState is not None):
		delimiterMode = constraintState.get("mode") == "delimiter"
		allowedSet = constraintState.get("columns")
	seenNodes = set()
	for columnIndex, featureIndex in nodes:
		adjustedFeature = featureIndex
		if(delimiterMode and allowedSet is not None and columnIndex in allowedSet):
			isDeterministicDelimiter = GIAANNproto_databaseNetworkExcitation.isFeatureIndexReferenceSetDelimiterDeterministic(databaseNetworkObject, adjustedFeature)
			isProbabilisticDelimiter = GIAANNproto_databaseNetworkExcitation.isFeatureIndexReferenceSetDelimiterProbabilistic(databaseNetworkObject, adjustedFeature)
			if(not (isDeterministicDelimiter or isProbabilisticDelimiter)):
				continue
		nodeKey = (columnIndex, adjustedFeature)
		if(nodeKey in seenNodes):
			continue
		seenNodes.add(nodeKey)
		preparedNodes.append((columnIndex, adjustedFeature))
		connectionSum += getConnectionValue(strengthLookup, columnIndex, adjustedFeature, maxFeatures)
	return preparedNodes, connectionSum

def nodesContainReferenceSetDelimiter(databaseNetworkObject, nodes):
	if(not conceptColumnsDelimitByPOS):
		return False
	for nodeColumn, nodeFeature in nodes:
		if(GIAANNproto_databaseNetworkExcitation.isFeatureIndexReferenceSetDelimiterDeterministic(databaseNetworkObject, nodeFeature)):
			return True
	return False

def nodesContainProbabilisticReferenceSetDelimiter(databaseNetworkObject, nodes):
	if(not conceptColumnsDelimitByPOS or not detectReferenceSetDelimitersBetweenNouns):
		return False
	for nodeColumn, nodeFeature in nodes:
		if(GIAANNproto_databaseNetworkExcitation.isFeatureIndexReferenceSetDelimiterProbabilistic(databaseNetworkObject, nodeFeature)):
			return True
	return False

def updateConstraintStateAfterNodes(databaseNetworkObject, previousConstraintState, nodes):
	if(not conceptColumnsDelimitByPOS):
		return None
	if(len(nodes) == 0):
		return previousConstraintState
	newColumns = set(nodeColumn for nodeColumn, _ in nodes)
	if(len(newColumns) == 0):
		return previousConstraintState
	if(nodesContainReferenceSetDelimiter(databaseNetworkObject, nodes)):
		return {"columns": newColumns, "mode": "delimiter"}
	elif(nodesContainProbabilisticReferenceSetDelimiter(databaseNetworkObject, nodes)):
		return None
	else:
		return {"columns": newColumns, "mode": "internal"}


def buildConnectedColumnsLookupForBeamNodes(databaseNetworkObject, observedColumnsDict, nodes):
	if(nodes is None or len(nodes) == 0):
		return None, None
	connectedColumnsSet = set()
	if(debugConnectNodesToNextNodesInSequenceOnly or enforceDirectConnectionsMinWordDistance or enforceDirectConnectionsSANI):
		connectedColumnsFeatures = {}
	else:
		connectedColumnsFeatures = None
	for columnIndex, featureIndex in nodes:
		observedColumn = getObservedColumnForBeam(databaseNetworkObject, observedColumnsDict, columnIndex)
		if(observedColumn is None):
			continue
		targetColumns, targetFeatures = getConnectedColumnsForBeamFeature(observedColumn, featureIndex, includeFeatureDetails=(connectedColumnsFeatures is not None))
		connectedColumnsSet.update(targetColumns)
		if(connectedColumnsFeatures is not None and targetFeatures is not None):
			for targetColumnIndex, featureSet in targetFeatures.items():
				if(targetColumnIndex < 0 or targetColumnIndex >= databaseNetworkObject.c):
					continue
				columnFeatureSet = connectedColumnsFeatures.setdefault(targetColumnIndex, set())
				columnFeatureSet.update(featureSet)
	if(len(connectedColumnsSet) == 0):
		emptyTensor = pt.empty(0, dtype=pt.long, device=deviceSparse)
		return emptyTensor, ({} if connectedColumnsFeatures is not None else None)
	validColumns = [col for col in connectedColumnsSet if col >= 0 and col < databaseNetworkObject.c]
	if(len(validColumns) == 0):
		emptyTensor = pt.empty(0, dtype=pt.long, device=deviceSparse)
		return emptyTensor, ({} if connectedColumnsFeatures is not None else None)
	validColumns.sort()
	connectedTensor = pt.tensor(validColumns, dtype=pt.long, device=deviceSparse)
	if(connectedColumnsFeatures is not None):
		filteredFeatureMap = {}
		for columnIndex in validColumns:
			if(connectedColumnsFeatures is None):
				continue
			featureSet = connectedColumnsFeatures.get(columnIndex, set())
			if(len(featureSet) > 0):
				filteredFeatureMap[columnIndex] = set(featureSet)
		return connectedTensor, filteredFeatureMap
	else:
		return connectedTensor, None


def getObservedColumnForBeam(databaseNetworkObject, observedColumnsDict, columnIndex):
	if(columnIndex < 0 or columnIndex >= len(databaseNetworkObject.conceptColumnsList)):
		return None
	columnLemma = databaseNetworkObject.conceptColumnsList[columnIndex]
	if(columnLemma in observedColumnsDict):
		return observedColumnsDict[columnLemma]
	observedColumn = GIAANNproto_databaseNetworkExcitation.loadOrCreateObservedColumn(databaseNetworkObject, columnIndex, columnLemma, columnIndex)
	observedColumnsDict[columnLemma] = observedColumn
	return observedColumn


def getConnectedColumnsForBeamFeature(observedColumn, featureIndex, includeFeatureDetails=False):
	if(featureIndex is None or featureIndex < 0):
		return [], {} if includeFeatureDetails else None
	featureConnectionsStrength = observedColumn.featureConnections[arrayIndexPropertiesStrengthIndex]
	featureConnectionsStrength = GIAANNproto_sparseTensors.sliceSparseTensor(featureConnectionsStrength, 2, featureIndex)
	featureConnectionsStrength = featureConnectionsStrength.coalesce()
	if(featureConnectionsStrength._nnz() == 0):
		return [], {} if includeFeatureDetails else None
	targetColumnIndices = featureConnectionsStrength.indices()
	if(targetColumnIndices.shape[1] == 0):
		return [], {} if includeFeatureDetails else None
	minWordDistanceMask = GIAANNproto_predictionActivate.computeConnectionMinWordDistanceMask(observedColumn, featureIndex, targetColumnIndices)
	if(minWordDistanceMask is not None):
		if(minWordDistanceMask.sum().item() == 0):
			return [], {} if includeFeatureDetails else None
		targetColumnIndices = targetColumnIndices[:, minWordDistanceMask]
	elif(enforceDirectConnections and enforceDirectConnectionsSANI):
		lastSegmentMask = targetColumnIndices[1] == arrayIndexSegmentLast
		targetColumnIndices = targetColumnIndices[:, lastSegmentMask]
	targetColumns = targetColumnIndices[2].unique()
	targetColumnsList = targetColumns.cpu().tolist()
	if(includeFeatureDetails):
		targetFeatures = targetColumnIndices[3].cpu().tolist()
		columnFeatureMap = {}
		for columnValue, featureValue in zip(targetColumnIndices[2].tolist(), targetFeatures):
			columnFeatureMap.setdefault(columnValue, set()).add(featureValue)
		return targetColumnsList, columnFeatureMap
	else:
		return targetColumnsList, None


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
	if(sparseTensor.dim() == 4):
		if(multipleDendriticBranches):
			if(sparseTensor.is_sparse):
				sparseTensor = GIAANNproto_sparseTensors.reduceSparseBranchMax(sparseTensor)
			else:
				sparseTensor = sparseTensor.max(dim=0).values
		else:
			sparseTensor = sparseTensor.sum(dim=0)
		if(not sparseTensor.is_sparse):
			sparseTensor = sparseTensor.to_sparse()
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
