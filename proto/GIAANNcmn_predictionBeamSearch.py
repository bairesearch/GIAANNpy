"""GIAANNcmn_predictionBeamSearch.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 BAI Research Pty Ltd (bairesearch.com.au)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description: 
GIA ANN common prediction Beam Search

"""

import torch as pt
import time

from GIAANNcmn_globalDefs import *
import GIAANNcmn_debug
import GIAANNcmn_databaseNetwork
import GIAANNcmn_sparseTensors
import GIAANNcmn_predictionActivate
import GIAANNcmn_predictionConstraints
if(auxiliaryNeurons and auxiliaryNeuronsSimilar):
	import GIAANNnlp_auxiliaryNeuronsSimilarWords


def beamSearchPredictNextFeature(sequenceObservedColumns, databaseNetworkObject, observedColumnsDict, globalFeatureNeuronsActivation, globalFeatureNeuronsStrength, globalFeatureConnectionsActivation, globalFeatureNeuronsTime, tokensSequence, wordPredictionIndex, sequenceWordIndex, conceptMask, allowedColumns=None, constraintMode=None, conceptActivationState=None, connectedColumnsConstraint=None, connectedColumnsFeatures=None, somaActivationFromLastSegmentKeys=None, selectedColumnIndex=None, deactivatedNeuronState=None):
	#generate targets for debug/analysis output
	targetPreviousColumnIndex, targetNextColumnIndex, targetFeatureIndex = GIAANNcmn_databaseNetwork.getTokenConceptFeatureIndexTensor(sequenceObservedColumns, tokensSequence, conceptMask, sequenceWordIndex, kcNetwork)

	strengthLookup = None
	if(globalFeatureNeuronsStrength is not None):
		strengthLookup = buildStrengthLookup(databaseNetworkObject, globalFeatureNeuronsStrength, databaseNetworkObject.f)
	initialConstraintState = GIAANNcmn_predictionConstraints.createConstraintState(allowedColumns, constraintMode)
	if(inferenceLeakyIntegrateAndFire and (useSANIcolumns or useSANIfeaturesAndColumns)):
		initialState = initialiseBeamActivationState(globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, globalFeatureNeuronsTime, conceptActivationState, somaActivationFromLastSegmentKeys, selectedColumnIndex)
	elif(inferenceLeakyIntegrateAndFire and algorithmMatrixSANIenforceRequirement=="enforceLastSegmentMustBeActive"):
		initialState = initialiseBeamActivationState(globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, globalFeatureNeuronsTime, conceptActivationState, somaActivationFromLastSegmentKeys)
	else:
		initialState = initialiseBeamActivationState(globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, globalFeatureNeuronsTime, conceptActivationState)
	if(inferenceLeakyIntegrateAndFire):
		if(inferenceUseNextTokenPredictionsOrTargetsToActivateNextColumnFeatures and inferenceDeactivateSomaUponPrediction):
			if(deactivatedNeuronState is None):
				raise RuntimeError("beamSearchPredictNextFeature error: deactivatedNeuronState is required for prediction-driven LIF soma deactivation")
			initialState["deactivatedNeurons"] = set(deactivatedNeuronState)
	beams = [{"score": 0.0, "state": initialState, "sequence": [], "constraintState": initialConstraintState, "connectedColumns": connectedColumnsConstraint, "connectedColumnsFeatures": connectedColumnsFeatures}]
	completedBeams = []
	beamDepth = max(1, inferenceBeamDepth)
	beamWidthLimit = max(1, inferenceBeamWidth)
	tokensSequenceLength = len(tokensSequence)
	if(sequenceWordIndex < 0 or sequenceWordIndex >= tokensSequenceLength):
		raise RuntimeError("beamSearchPredictNextFeature error: sequenceWordIndex out of range")
	remainingTokens = tokensSequenceLength - sequenceWordIndex
	if(remainingTokens < beamDepth):
		beamDepth = remainingTokens
	result = None

	for depthIndex in range(beamDepth):
		depthSequenceWordIndex = sequenceWordIndex + depthIndex
		depthSequenceColumnIndex = None
		if(inferenceUseNeuronFeaturePropertiesTime):
			if(useSANIcolumns or useSANIfeaturesAndColumns):
				depthSequenceColumnIndex = GIAANNcmn_predictionActivate.calculateSequenceColumnIndex(conceptMask, depthSequenceWordIndex)
		newBeams = []
		for beam in beams:
			if(inferenceLeakyIntegrateAndFire and algorithmMatrixSANIenforceRequirement=="enforceLastSegmentMustBeActive"):
				candidates = selectBeamCandidates(beam["state"]["features"], beam["state"].get("time"), strengthLookup, beamWidthLimit, databaseNetworkObject, beam.get("constraintState"), beam["state"].get("conceptActivations"), beam.get("connectedColumns"), beam.get("connectedColumnsFeatures"), depthSequenceWordIndex, depthSequenceColumnIndex, beam["state"].get("somaActivationFromLastSegmentKeys"), beam["state"].get("deactivatedNeurons"))
			else:
				candidates = selectBeamCandidates(beam["state"]["features"], beam["state"].get("time"), strengthLookup, beamWidthLimit, databaseNetworkObject, beam.get("constraintState"), beam["state"].get("conceptActivations"), beam.get("connectedColumns"), beam.get("connectedColumnsFeatures"), depthSequenceWordIndex, depthSequenceColumnIndex, None, beam["state"].get("deactivatedNeurons"))
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
					executeBeamNodeActivation(databaseNetworkObject, observedColumnsDict, newState, nodeColumn, nodeFeature, depthSequenceWordIndex, depthSequenceColumnIndex)
				newSequence = beam["sequence"] + [candidate]
				activationGain = computeCandidateActivationGain(newState["features"], oldState["features"], candidate["nodes"])
				if(inferenceBeamScoreStrategy == "nodeActivation"):
					if(inferenceLeakyIntegrateAndFire):
						activationGain = candidate.get("activationValue", activationGain)
					elif(inferenceUseNeuronFeaturePropertiesTime):
						# spec step (b): score beam nodes using time-modified activation.
						activationGain = candidate.get("activationValue", activationGain)
				candidateScore = computeBeamNodeScore(activationGain, candidate["connectionValue"])
				newScore = beam["score"] + candidateScore
				newConstraintState = updateConstraintStateAfterNodes(databaseNetworkObject, beam.get("constraintState"), candidate["nodes"])
				nextConnectedColumns, nextConnectedFeatures = GIAANNcmn_predictionConstraints.buildConnectedColumnsLookupForBeamNodes(databaseNetworkObject, observedColumnsDict, candidate["nodes"])
				newBeams.append({"score": newScore, "state": newState, "sequence": newSequence, "constraintState": newConstraintState, "connectedColumns": nextConnectedColumns, "connectedColumnsFeatures": nextConnectedFeatures})
		if(len(newBeams) == 0):
			break
		newBeams.sort(key=lambda item: item["score"], reverse=True)
		beams = newBeams[:beamWidthLimit]

	if(len(beams) == 0):
		beams = completedBeams

	if(result is None):
		allBeams = beams + completedBeams
		if(len(allBeams) == 0):
			raise RuntimeError("beamSearchPredictNextFeature error: no beams available")
		availableBeams = [beam for beam in allBeams if len(beam["sequence"]) > 0]
		if(len(availableBeams) == 0):
			GIAANNcmn_predictionConstraints.raiseOrStopPredictionConnectivityError(sequenceWordIndex, wordPredictionIndex, tokensSequence, "beamSearchPredictNextFeature: no candidates available")
		bestBeam = max(availableBeams, key=lambda item: item["score"])
		bestAction = bestBeam["sequence"][0]
		conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext = convertNodesToPrediction(bestAction["nodes"])
		if(result is None):
			if(conceptColumnsIndicesNext is None or conceptColumnsFeatureIndicesNext is None or conceptColumnsIndicesNext.numel() == 0 or conceptColumnsFeatureIndicesNext.numel() == 0):
				raise RuntimeError("beamSearchPredictNextFeature error: no prediction candidates available")
			if(conceptColumnsIndicesNext.numel() != 1 or conceptColumnsFeatureIndicesNext.numel() != 1):
				raise RuntimeError("beamSearchPredictNextFeature error: multiple prediction candidates not supported")
			if(printPredictionsDuringInferencePredict):
				printBestBeamPath(bestBeam, databaseNetworkObject)
			conceptColumnIndexPred = int(conceptColumnsIndicesNext.squeeze().item())
			conceptColumnFeatureIndexPred = int(conceptColumnsFeatureIndicesNext.squeeze().item())
			result = (conceptColumnIndexPred, conceptColumnFeatureIndexPred, targetPreviousColumnIndex, targetNextColumnIndex)
	if(result is None):
		raise RuntimeError("beamSearchPredictNextFeature error: no prediction result available")
	return result

def beamSearchSelectSingleStepFeature(sequenceObservedColumns, databaseNetworkObject, observedColumnsDict, globalFeatureNeuronsActivation, globalFeatureNeuronsStrength, globalFeatureConnectionsActivation, globalFeatureNeuronsTime, tokensSequence, wordPredictionIndex, sequenceWordIndex, conceptMask, allowedColumns=None, constraintMode=None, conceptActivationState=None, connectedColumnsConstraint=None, connectedColumnsFeatures=None, somaActivationFromLastSegmentKeys=None, deactivatedNeuronState=None):
	#single-step beam candidate selection (no beam depth expansion)
	targetPreviousColumnIndex, targetNextColumnIndex, targetFeatureIndex = GIAANNcmn_databaseNetwork.getTokenConceptFeatureIndexTensor(sequenceObservedColumns, tokensSequence, conceptMask, sequenceWordIndex, kcNetwork)
	result = None

	strengthLookup = None
	if(globalFeatureNeuronsStrength is not None and not inferenceStrengthLookupBypass):
		strengthLookup = buildStrengthLookup(databaseNetworkObject, globalFeatureNeuronsStrength, databaseNetworkObject.f)
	sequenceColumnIndex = None
	if(inferenceUseNeuronFeaturePropertiesTime):
		if(useSANIcolumns or useSANIfeaturesAndColumns):
			sequenceColumnIndex = GIAANNcmn_predictionActivate.calculateSequenceColumnIndex(conceptMask, sequenceWordIndex)
	constraintState = GIAANNcmn_predictionConstraints.createConstraintState(allowedColumns, constraintMode)
	candidateLimit = 1	#inferenceBeamWidth
	if(inferenceLeakyIntegrateAndFire and algorithmMatrixSANIenforceRequirement=="enforceLastSegmentMustBeActive"):
		candidates = selectBeamCandidates(globalFeatureNeuronsActivation, globalFeatureNeuronsTime, strengthLookup, candidateLimit, databaseNetworkObject, constraintState, conceptActivationState, connectedColumnsConstraint, connectedColumnsFeatures, sequenceWordIndex, sequenceColumnIndex, somaActivationFromLastSegmentKeys, deactivatedNeuronState)
	else:
		candidates = selectBeamCandidates(globalFeatureNeuronsActivation, globalFeatureNeuronsTime, strengthLookup, candidateLimit, databaseNetworkObject, constraintState, conceptActivationState, connectedColumnsConstraint, connectedColumnsFeatures, sequenceWordIndex, sequenceColumnIndex, None, deactivatedNeuronState)
	if(len(candidates) == 0):
		GIAANNcmn_predictionConstraints.raiseOrStopPredictionConnectivityError(sequenceWordIndex, wordPredictionIndex, tokensSequence, "beamSearchSelectSingleStepFeature: no candidates available")
	else:
		bestCandidate = None
		bestScore = None
		for candidate in candidates:
			activationValue = candidate.get("activationValue", 0.0)
			connectionValue = candidate.get("connectionValue", 0.0)
			candidateScore = computeBeamNodeScore(activationValue, connectionValue)
			if(bestCandidate is None or candidateScore > bestScore):
				bestCandidate = candidate
				bestScore = candidateScore
		conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext = convertNodesToPrediction(bestCandidate["nodes"])
		if(conceptColumnsIndicesNext is None or conceptColumnsFeatureIndicesNext is None or conceptColumnsIndicesNext.shape[0] == 0 or conceptColumnsFeatureIndicesNext.shape[0] == 0):
			GIAANNcmn_predictionConstraints.raiseOrStopPredictionConnectivityError(sequenceWordIndex, wordPredictionIndex, tokensSequence, "beamSearchSelectSingleStepFeature: no prediction nodes available after selection")
		else:
			if(conceptColumnsIndicesNext.numel() != 1 or conceptColumnsFeatureIndicesNext.numel() != 1):
				raise RuntimeError("beamSearchSelectSingleStepFeature error: multiple prediction candidates not supported")
			conceptColumnIndexPred = int(conceptColumnsIndicesNext.squeeze().item())
			conceptColumnFeatureIndexPred = int(conceptColumnsFeatureIndicesNext.squeeze().item())
			result = (conceptColumnIndexPred, conceptColumnFeatureIndexPred, targetPreviousColumnIndex, targetNextColumnIndex)
	
	if(result is None):
		raise RuntimeError("beamSearchSelectSingleStepFeature error: no prediction result available")
	return result

def initialiseBeamActivationState(globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, globalFeatureNeuronsTime, conceptActivationState, somaActivationFromLastSegmentKeys=None, selectedColumnIndex=None):
	state = {"features": globalFeatureNeuronsActivation.clone()}
	state["connections"] = None
	if(inferenceLeakyIntegrateAndFire and algorithmMatrixSANIenforceRequirement=="enforceLastSegmentMustBeActive"):
		state["somaActivationFromLastSegmentKeys"] = somaActivationFromLastSegmentKeys
	if(inferenceLeakyIntegrateAndFire and (useSANIcolumns or useSANIfeaturesAndColumns)):
		if(selectedColumnIndex is None):
			raise RuntimeError("initialiseBeamActivationState error: selectedColumnIndex is required for LIF column propagation")
		state["selectedColumnIndex"] = int(selectedColumnIndex)
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
	clonedState["connections"] = None
	if(inferenceLeakyIntegrateAndFire and algorithmMatrixSANIenforceRequirement=="enforceLastSegmentMustBeActive"):
		clonedState["somaActivationFromLastSegmentKeys"] = state.get("somaActivationFromLastSegmentKeys")
	if(inferenceLeakyIntegrateAndFire and (useSANIcolumns or useSANIfeaturesAndColumns)):
		if(state.get("selectedColumnIndex") is None):
			raise RuntimeError("cloneBeamActivationState error: selectedColumnIndex is required for LIF column propagation")
		clonedState["selectedColumnIndex"] = state["selectedColumnIndex"]
	if(inferenceUseNeuronFeaturePropertiesTime and state.get("time") is not None):
		clonedState["time"] = state["time"].clone()
	else:
		clonedState["time"] = None
	if("conceptActivations" in state and state["conceptActivations"] is not None):
		clonedState["conceptActivations"] = set(state["conceptActivations"])
	else:
		clonedState["conceptActivations"] = None
	if(inferenceLeakyIntegrateAndFire):
		if(inferenceUseNextTokenPredictionsOrTargetsToActivateNextColumnFeatures and inferenceDeactivateSomaUponPrediction):
			if(state.get("deactivatedNeurons") is None):
				raise RuntimeError("cloneBeamActivationState error: deactivatedNeurons is required for prediction-driven LIF soma deactivation")
			clonedState["deactivatedNeurons"] = set(state["deactivatedNeurons"])
	return clonedState

def executeBeamNodeActivation(databaseNetworkObject, observedColumnsDict, state, columnIndex, featureIndex, sequenceWordIndex, sequenceColumnIndex):
	if(inferenceLeakyIntegrateAndFire):
		if(useSANIcolumns or useSANIfeaturesAndColumns):
			if(state.get("selectedColumnIndex") is None):
				raise RuntimeError("executeBeamNodeActivation error: selectedColumnIndex is required for LIF column propagation")
			if(int(columnIndex) != int(state["selectedColumnIndex"])):
				state["features"] = GIAANNcmn_predictionActivate.advanceLeakyIntegrateAndFireColumnActivations(state["features"])
			state["selectedColumnIndex"] = int(columnIndex)
		if(inferenceDecrementActivationsSoma):
			state["features"] = GIAANNcmn_predictionActivate.decrementLeakyIntegrateAndFireSomaActivation(state["features"])
		if(inferenceBurstAllPredictionsOrTargetsInSequence):
			state["features"] = activateBeamNodeLeakyIntegrateAndFireSoma(state["features"], columnIndex, featureIndex)
		if(algorithmMatrixSANIenforceRequirement=="enforceLastSegmentMustBeActive"):
			state["features"], somaActivationFromPropagatedLastSegmentKeys = GIAANNcmn_predictionActivate.propagateLeakyIntegrateAndFireActivationsEnforceLastSegment(state["features"])
			if(enforceDirectConnectionsSANI):
				state["somaActivationFromLastSegmentKeys"] = pt.empty((arrayIndexSegmentFirst,), dtype=pt.long, device=state["features"].device)
			else:
				state["somaActivationFromLastSegmentKeys"] = somaActivationFromPropagatedLastSegmentKeys
		else:
			state["features"] = GIAANNcmn_predictionActivate.propagateLeakyIntegrateAndFireActivations(state["features"])
	lemma = databaseNetworkObject.conceptColumnsList[columnIndex]
	if(lemma in observedColumnsDict):
		observedColumn = observedColumnsDict[lemma]
	else:
		observedColumn = GIAANNcmn_databaseNetwork.loadOrCreateObservedColumn(databaseNetworkObject, columnIndex, lemma, sequenceWordIndex)
	if(inferenceOnlyRetainPredictedTargetObservedColumn and inferenceOnlyRetainPredictedTargetObservedColumnBeamSearch):
		if(observedColumnsDict is None):
			raise RuntimeError("executeBeamNodeActivation error: observedColumnsDict is None")
		observedColumnsDict.clear()
	observedColumnsDict[lemma] = observedColumn
	featureConnections = observedColumn.prepareFeatureConnectionsForSourceFeature(featureIndex, targetDevice=state["features"].device, createMissing=False)
	if(inferenceLeakyIntegrateAndFire and algorithmMatrixSANIenforceRequirement=="enforceLastSegmentMustBeActive"):
		state["features"], state["connections"], state["time"], state["somaActivationFromLastSegmentKeys"] = GIAANNcmn_predictionActivate.processFeaturesActivePredictEnforceLastSegment(databaseNetworkObject, state["features"], state["connections"], featureConnections, columnIndex, featureIndex, state["somaActivationFromLastSegmentKeys"], state.get("time"), sequenceWordIndex, sequenceColumnIndex)
	else:
		state["features"], state["connections"], state["time"] = GIAANNcmn_predictionActivate.processFeaturesActivePredict(databaseNetworkObject, state["features"], state["connections"], featureConnections, columnIndex, featureIndex, state.get("time"), sequenceWordIndex, sequenceColumnIndex)
	if(auxiliaryNeurons and auxiliaryNeuronsSimilar):
		if(inferenceLeakyIntegrateAndFire and algorithmMatrixSANIenforceRequirement=="enforceLastSegmentMustBeActive"):
			state["features"], state["connections"], state["time"], state["somaActivationFromLastSegmentKeys"] = GIAANNnlp_auxiliaryNeuronsSimilarWords.processAuxiliaryFeaturePredictionActivationsEnforceLastSegment(databaseNetworkObject, observedColumn, state["features"], state["connections"], columnIndex, featureIndex, state["somaActivationFromLastSegmentKeys"], state.get("time"), sequenceWordIndex, sequenceColumnIndex)
		else:
			state["features"], state["connections"], state["time"] = GIAANNnlp_auxiliaryNeuronsSimilarWords.processAuxiliaryFeaturePredictionActivations(databaseNetworkObject, observedColumn, state["features"], state["connections"], columnIndex, featureIndex, state.get("time"), sequenceWordIndex, sequenceColumnIndex)
	applyBeamNodePredictionEffects(state, columnIndex, featureIndex, sequenceWordIndex)
	if(predictionColumnsMustActivateConceptFeature):
		conceptState = state.get("conceptActivations")
		if(conceptState is None):
			conceptState = set()
			state["conceptActivations"] = conceptState
		conceptState.add(columnIndex)
	return state

def activateBeamNodeLeakyIntegrateAndFireSoma(globalFeatureNeuronsActivation, columnIndex, featureIndex):
	result = None
	if(inferenceLeakyIntegrateAndFire and inferenceBurstAllPredictionsOrTargetsInSequence):
		if(globalFeatureNeuronsActivation is None or not globalFeatureNeuronsActivation.is_sparse):
			raise RuntimeError("activateBeamNodeLeakyIntegrateAndFireSoma error: globalFeatureNeuronsActivation must be sparse")
		if(globalFeatureNeuronsActivation.dim() != inferenceLeakyIntegrateAndFireNeuronTensorRank or globalFeatureNeuronsActivation.shape[inferenceLeakyIntegrateAndFireBranchDimension] != multipleDendriticBranchesNumber or globalFeatureNeuronsActivation.shape[inferenceLeakyIntegrateAndFireSegmentDimension] != arrayNumberOfSegments):
			raise RuntimeError("activateBeamNodeLeakyIntegrateAndFireSoma error: activation tensor shape is invalid")
		columnIndex = int(columnIndex)
		featureIndex = int(featureIndex)
		if(columnIndex < arrayIndexSegmentFirst or columnIndex >= globalFeatureNeuronsActivation.shape[inferenceLeakyIntegrateAndFireConceptDimension] or featureIndex < arrayIndexSegmentFirst or featureIndex >= globalFeatureNeuronsActivation.shape[inferenceLeakyIntegrateAndFireFeatureDimension]):
			raise RuntimeError("activateBeamNodeLeakyIntegrateAndFireSoma error: neuron index is out of range")
		updateIndices = pt.tensor([[inferenceLeakyIntegrateAndFireSomaBranchIndex], [arrayIndexSegmentSoma], [columnIndex], [featureIndex]], dtype=pt.long, device=globalFeatureNeuronsActivation.device)
		burstActivation = max(j1, inferenceLeakyIntegrateAndFireSomaActivationThreshold)
		updateValues = pt.full((updateIndices.shape[1],), burstActivation, dtype=globalFeatureNeuronsActivation.dtype, device=globalFeatureNeuronsActivation.device)
		updateTensor = pt.sparse_coo_tensor(updateIndices, updateValues, size=globalFeatureNeuronsActivation.size(), dtype=globalFeatureNeuronsActivation.dtype, device=globalFeatureNeuronsActivation.device)
		result = (globalFeatureNeuronsActivation.coalesce() + updateTensor).coalesce()
	else:
		raise RuntimeError("activateBeamNodeLeakyIntegrateAndFireSoma error: requires burst-enabled inferenceLeakyIntegrateAndFire")
	return result

def applyBeamNodePredictionEffects(state, columnIndex, featureIndex, sequenceWordIndex):
	if(inferenceLeakyIntegrateAndFire):
		modifyActivation = inferenceDeactivateSomaUponPrediction or inferenceDeactivateSegmentsUponPrediction or inferenceDeactivateLastColumnSegmentUponPrediction
	else:
		modifyActivation = inferenceDeactivateNeuronsUponPrediction
	if(modifyActivation):
		branchIndex = 0
		if(multipleDendriticBranches):
			branchIndex = GIAANNcmn_predictionActivate.selectActivatedBranchIndex(state["features"], columnIndex, featureIndex)
		indicesToUpdate = buildBeamNodeIndices(state["features"].device, columnIndex, featureIndex, branchIndex)
		modifier = 0
		state["features"] = GIAANNcmn_sparseTensors.modifySparseTensor(state["features"], indicesToUpdate, modifier, multiply=False)
	if(inferenceLeakyIntegrateAndFire):
		if(inferenceUseNextTokenPredictionsOrTargetsToActivateNextColumnFeatures and inferenceDeactivateSomaUponPrediction):
			if(state.get("deactivatedNeurons") is None):
				raise RuntimeError("applyBeamNodePredictionEffects error: deactivatedNeurons is required for prediction-driven LIF soma deactivation")
			state["deactivatedNeurons"].add((int(columnIndex), int(featureIndex)))

def buildBeamNodeIndices(device, columnIndex, featureIndex, branchIndex=0):
	columnTensor = pt.tensor(columnIndex, dtype=pt.long, device=device)
	featureTensor = pt.tensor(featureIndex, dtype=pt.long, device=device)
	branchTensor = pt.tensor(branchIndex, dtype=pt.long, device=device)
	if(inferenceLeakyIntegrateAndFire):
		indicesToUpdateList = []
		if(inferenceDeactivateSegmentsUponPrediction):
			branchIndices = pt.arange(multipleDendriticBranchesNumber, dtype=pt.long, device=device).repeat_interleave(arrayIndexSegmentSoma)
			segmentIndices = pt.arange(arrayIndexSegmentFirst, arrayIndexSegmentSoma, dtype=pt.long, device=device).repeat(multipleDendriticBranchesNumber)
			indicesToUpdateList.append(pt.stack([branchIndices, segmentIndices, columnTensor.expand(branchIndices.shape[0]), featureTensor.expand(branchIndices.shape[0])], dim=1))
		if(inferenceDeactivateSomaUponPrediction):
			branchIndices = pt.arange(multipleDendriticBranchesNumber, dtype=pt.long, device=device)
			segmentIndices = pt.full_like(branchIndices, arrayIndexSegmentSoma)
			indicesToUpdateList.append(pt.stack([branchIndices, segmentIndices, columnTensor.expand(branchIndices.shape[0]), featureTensor.expand(branchIndices.shape[0])], dim=1))
		if(inferenceDeactivateLastColumnSegmentUponPrediction and not inferenceDeactivateSegmentsUponPrediction):
			if(not (useSANIcolumns or useSANIfeaturesAndColumns)):
				raise RuntimeError("buildBeamNodeIndices error: inferenceDeactivateLastColumnSegmentUponPrediction requires LIF column segments")
			branchIndices = pt.arange(multipleDendriticBranchesNumber, dtype=pt.long, device=device)
			segmentIndices = pt.full_like(branchIndices, arrayIndexSegmentLastColumn)
			indicesToUpdateList.append(pt.stack([branchIndices, segmentIndices, columnTensor.expand(branchIndices.shape[0]), featureTensor.expand(branchIndices.shape[0])], dim=1))
		indicesToUpdate = pt.cat(indicesToUpdateList, dim=0)
	elif(useSANI):
		if(multipleDendriticBranchesBinaryTree):
			segmentIndices = pt.arange(arrayNumberOfSegments, dtype=pt.long, device=device)
			branchIndices = pt.full_like(segmentIndices, branchIndex)
			branchDivisors = pt.pow(pt.full_like(segmentIndices, multipleDendriticBranchesBinaryTreeBranchingFactor), segmentIndices)
			binaryTreeBranchIndices = pt.div(branchIndices, branchDivisors, rounding_mode="floor")
			indicesToUpdate = pt.stack([binaryTreeBranchIndices, segmentIndices, columnTensor.expand(arrayNumberOfSegments), featureTensor.expand(arrayNumberOfSegments)], dim=1)
		else:
			indicesToUpdateList = []
			for segmentIndex in range(arrayNumberOfSegments):
				segmentTensor = pt.tensor(segmentIndex, dtype=pt.long, device=device)
				indicesToUpdateList.append(pt.stack([branchTensor, segmentTensor, columnTensor, featureTensor], dim=0))
			indicesToUpdate = pt.stack(indicesToUpdateList, dim=0)
	else:
		indicesToUpdateList = []
		segmentTensor = pt.tensor(arrayIndexSegmentFirst, dtype=pt.long, device=device)
		indicesToUpdateList.append(pt.stack([branchTensor, segmentTensor, columnTensor, featureTensor], dim=0))
		indicesToUpdate = pt.stack(indicesToUpdateList, dim=0)
	return indicesToUpdate

def describeBeamCandidate(databaseNetworkObject, candidate):
	return describeBeamNodes(databaseNetworkObject, candidate["nodes"])

def describeBeamNodes(databaseNetworkObject, nodes):
	nodeDescriptions = []
	for nodeColumn, nodeFeature in nodes:
		columnName = databaseNetworkObject.conceptColumnsList[nodeColumn]
		if(nodeFeature == featureIndexPrimeConceptNeuron):
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
	if(inferenceBeamSearch):
		print("\t\tBest beam path:\n\t\t\t" + "\n\t\t\t".join(pathSegments))	# Debug: summary of the highest scoring beam path

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

def filterCandidatesByLeakyIntegrateAndFireSomaActivationThreshold(columnIndices, featureIndices, activationValues):
	filteredColumnIndices = columnIndices
	filteredFeatureIndices = featureIndices
	filteredActivationValues = activationValues
	if(inferenceLeakyIntegrateAndFire):
		if(columnIndices is None or featureIndices is None or activationValues is None or columnIndices.numel() == 0 or featureIndices.numel() == 0 or activationValues.numel() == 0):
			filteredColumnIndices = None
			filteredFeatureIndices = None
			filteredActivationValues = None
		else:
			activeMask = activationValues >= inferenceLeakyIntegrateAndFireSomaActivationThreshold
			if(activeMask.sum().item() == 0):
				filteredColumnIndices = None
				filteredFeatureIndices = None
				filteredActivationValues = None
			else:
				indexTensor = pt.nonzero(activeMask, as_tuple=False).view(-1)
				filteredColumnIndices = columnIndices.index_select(0, indexTensor)
				filteredFeatureIndices = featureIndices.index_select(0, indexTensor)
				filteredActivationValues = activationValues.index_select(0, indexTensor)
	else:
		raise RuntimeError("filterCandidatesByLeakyIntegrateAndFireSomaActivationThreshold error: requires inferenceLeakyIntegrateAndFire")
	return filteredColumnIndices, filteredFeatureIndices, filteredActivationValues

def selectBeamCandidates(stateFeatures, stateTime, strengthLookup, candidateLimit, databaseNetworkObject, constraintState=None, conceptActivationState=None, connectedColumnsTensor=None, connectedColumnsFeatures=None, sequenceWordIndex=None, sequenceColumnIndex=None, somaActivationFromLastSegmentKeys=None, deactivatedNeuronState=None):
	candidateLimit = max(1, candidateLimit)
	debugTimeStart = None
	debugTimeLast = None
	if(inferenceLeakyIntegrateAndFire and algorithmMatrixSANIenforceRequirement=="enforceLastSegmentMustBeActive"):
		columnIndices, featureIndices, activationValues = calculateSelectionActivationDistribution(databaseNetworkObject, stateFeatures, stateTime, constraintState, connectedColumnsTensor, connectedColumnsFeatures, sequenceWordIndex, sequenceColumnIndex, False, somaActivationFromLastSegmentKeys)
	else:
		columnIndices, featureIndices, activationValues = calculateSelectionActivationDistribution(databaseNetworkObject, stateFeatures, stateTime, constraintState, connectedColumnsTensor, connectedColumnsFeatures, sequenceWordIndex, sequenceColumnIndex, False)
	candidates = []
	if(columnIndices is not None):
		if(inferenceLeakyIntegrateAndFire):
			if(inferenceUseNextTokenPredictionsOrTargetsToActivateNextColumnFeatures and inferenceDeactivateSomaUponPrediction):
				columnIndices, featureIndices, activationValues = filterCandidatesByDeactivatedNeuronState(databaseNetworkObject, columnIndices, featureIndices, activationValues, deactivatedNeuronState)
		if(columnIndices is not None):
			candidates = selectBeamCandidatesInstanceNodes(columnIndices, featureIndices, activationValues, strengthLookup, candidateLimit, databaseNetworkObject.f, databaseNetworkObject, constraintState, conceptActivationState)
	return candidates

def filterCandidatesByDeactivatedNeuronState(databaseNetworkObject, columnIndices, featureIndices, activationValues, deactivatedNeuronState):
	filteredColumnIndices = columnIndices
	filteredFeatureIndices = featureIndices
	filteredActivationValues = activationValues
	if(inferenceLeakyIntegrateAndFire):
		if(inferenceUseNextTokenPredictionsOrTargetsToActivateNextColumnFeatures and inferenceDeactivateSomaUponPrediction):
			if(deactivatedNeuronState is None):
				raise RuntimeError("filterCandidatesByDeactivatedNeuronState error: deactivatedNeuronState is required for prediction-driven LIF soma deactivation")
			if(columnIndices is None or featureIndices is None or activationValues is None):
				raise RuntimeError("filterCandidatesByDeactivatedNeuronState error: candidate tensors must not be None")
			if(columnIndices.shape != featureIndices.shape or columnIndices.shape != activationValues.shape):
				raise RuntimeError("filterCandidatesByDeactivatedNeuronState error: candidate tensor shapes must match")
			if(len(deactivatedNeuronState) > arrayIndexSegmentFirst):
				deactivatedNeuronKeysList = []
				for columnIndex, featureIndex in deactivatedNeuronState:
					if(columnIndex < arrayIndexSegmentFirst or columnIndex >= databaseNetworkObject.c or featureIndex < arrayIndexSegmentFirst or featureIndex >= databaseNetworkObject.f):
						raise RuntimeError("filterCandidatesByDeactivatedNeuronState error: deactivated neuron index out of range")
					deactivatedNeuronKeysList.append(columnIndex*int(databaseNetworkObject.f)+featureIndex)
				deactivatedNeuronKeys = pt.tensor(deactivatedNeuronKeysList, dtype=pt.long, device=columnIndices.device)
				deactivatedNeuronKeys = pt.sort(pt.unique(deactivatedNeuronKeys)).values
				candidateKeys = columnIndices.long()*int(databaseNetworkObject.f)+featureIndices.long()
				deactivatedMask = GIAANNcmn_predictionConstraints.buildSortedKeyMembershipMask(candidateKeys, deactivatedNeuronKeys)
				activeMask = pt.logical_not(deactivatedMask)
				if(activeMask.sum().item() == arrayIndexSegmentFirst):
					filteredColumnIndices = None
					filteredFeatureIndices = None
					filteredActivationValues = None
				else:
					activeIndices = pt.nonzero(activeMask, as_tuple=False).view(-1)
					filteredColumnIndices = columnIndices.index_select(arrayIndexSegmentFirst, activeIndices)
					filteredFeatureIndices = featureIndices.index_select(arrayIndexSegmentFirst, activeIndices)
					filteredActivationValues = activationValues.index_select(arrayIndexSegmentFirst, activeIndices)
		else:
			raise RuntimeError("filterCandidatesByDeactivatedNeuronState error: requires prediction-driven LIF soma deactivation")
	else:
		raise RuntimeError("filterCandidatesByDeactivatedNeuronState error: requires inferenceLeakyIntegrateAndFire")
	return filteredColumnIndices, filteredFeatureIndices, filteredActivationValues

def calculateSelectionActivationDistribution(databaseNetworkObject, stateFeatures, stateTime, constraintState=None, connectedColumnsTensor=None, connectedColumnsFeatures=None, sequenceWordIndex=None, sequenceColumnIndex=None, applyConstraintFilter=False, somaActivationFromLastSegmentKeys=None):
	columnIndices = None
	featureIndices = None
	activationValues = None
	stateFeaturesSelection = stateFeatures
	requiredSegmentKeys = None
	if(inferenceLeakyIntegrateAndFire and algorithmMatrixSANIenforceRequirement=="enforceLastSegmentMustBeActive"):
		if(somaActivationFromLastSegmentKeys is None):
			raise RuntimeError("calculateSelectionActivationDistribution error: somaActivationFromLastSegmentKeys is None")
		if(somaActivationFromLastSegmentKeys.dim() != 1 or somaActivationFromLastSegmentKeys.dtype != pt.long):
			raise RuntimeError("calculateSelectionActivationDistribution error: somaActivationFromLastSegmentKeys is invalid")
		requiredSegmentKeys = somaActivationFromLastSegmentKeys
		columnIndices, featureIndices, activationValues = calculateLeakyIntegrateAndFireLastSegmentSelectionActivationDistribution(databaseNetworkObject, stateFeaturesSelection, requiredSegmentKeys)
	else:
		if(inferenceLeakyIntegrateAndFire):
			stateFeaturesSelection = GIAANNcmn_predictionActivate.calculateLeakyIntegrateAndFireSomaActivation(stateFeaturesSelection)
		elif(inferenceUseNeuronFeaturePropertiesTime):
			# spec step (b): apply time-based activation modifier during beam candidate selection
			stateFeaturesSelection = GIAANNcmn_predictionActivate.applyTimeBasedActivationModifier(stateFeaturesSelection, stateTime, sequenceWordIndex, sequenceColumnIndex)
		if(requiresCandidateRequiredSegmentFilter() and stateFeaturesSelection is not None):
			requiredSegmentKeys = calculateRequiredSegmentConstraintKeyTensor(stateFeaturesSelection, databaseNetworkObject.f, stateFeaturesSelection.device)
		columnIndices, featureIndices, activationValues = GIAANNcmn_predictionConstraints.aggregateSparseColumnFeatureValues(stateFeaturesSelection, databaseNetworkObject.f, requiredSegmentKeys)
	if(columnIndices is not None):
		if(inferenceLeakyIntegrateAndFire):
			columnIndices, featureIndices, activationValues = filterCandidatesByActivationThreshold(columnIndices, featureIndices, activationValues)
			if(columnIndices is not None):
				columnIndices, featureIndices, activationValues = filterCandidatesByLeakyIntegrateAndFireSomaActivationThreshold(columnIndices, featureIndices, activationValues)
		else:
			columnIndices, featureIndices, activationValues = filterCandidatesByActivationThreshold(columnIndices, featureIndices, activationValues)
		if(requiredSegmentKeys is None):
			columnIndices, featureIndices, activationValues = filterCandidatesByRequiredSegments(columnIndices, featureIndices, activationValues, stateFeaturesSelection, databaseNetworkObject.f)
		if(columnIndices is not None):
			columnIndices, featureIndices, activationValues = GIAANNcmn_predictionConstraints.filterColumnFeatureCandidatesByConnectedColumns(columnIndices, featureIndices, activationValues, connectedColumnsTensor, connectedColumnsFeatures)
		if(columnIndices is not None and applyConstraintFilter):
			columnIndices, featureIndices, activationValues = GIAANNcmn_predictionConstraints.filterColumnFeatureCandidatesByConstraint(databaseNetworkObject, columnIndices, featureIndices, activationValues, constraintState)
	return columnIndices, featureIndices, activationValues

def calculateLeakyIntegrateAndFireLastSegmentSelectionActivationDistribution(databaseNetworkObject, stateFeatures, somaActivationFromLastSegmentKeys):
	result = None, None, None
	if(inferenceLeakyIntegrateAndFire and algorithmMatrixSANIenforceRequirement=="enforceLastSegmentMustBeActive"):
		if(stateFeatures is None or not stateFeatures.is_sparse):
			raise RuntimeError("calculateLeakyIntegrateAndFireLastSegmentSelectionActivationDistribution error: stateFeatures must be sparse")
		if(stateFeatures.dim() != inferenceLeakyIntegrateAndFireNeuronTensorRank or stateFeatures.shape[inferenceLeakyIntegrateAndFireBranchDimension] != multipleDendriticBranchesNumber or stateFeatures.shape[inferenceLeakyIntegrateAndFireSegmentDimension] != arrayNumberOfSegments or stateFeatures.shape[inferenceLeakyIntegrateAndFireFeatureDimension] != databaseNetworkObject.f):
			raise RuntimeError("calculateLeakyIntegrateAndFireLastSegmentSelectionActivationDistribution error: stateFeatures shape is invalid")
		if(somaActivationFromLastSegmentKeys is None or somaActivationFromLastSegmentKeys.dim() != 1 or somaActivationFromLastSegmentKeys.dtype != pt.long):
			raise RuntimeError("calculateLeakyIntegrateAndFireLastSegmentSelectionActivationDistribution error: somaActivationFromLastSegmentKeys is invalid")
		if(stateFeatures.device != somaActivationFromLastSegmentKeys.device):
			raise RuntimeError("calculateLeakyIntegrateAndFireLastSegmentSelectionActivationDistribution error: activation tensors must use the same device")
		stateFeaturesSomaSignalByBranch = GIAANNcmn_predictionActivate.calculateLeakyIntegrateAndFireSomaActivationByBranch(stateFeatures)
		if(stateFeaturesSomaSignalByBranch._nnz() > 0 and somaActivationFromLastSegmentKeys.numel() > 0):
			stateIndices = stateFeaturesSomaSignalByBranch.indices()
			stateValues = stateFeaturesSomaSignalByBranch.values()
			branchIndices = stateIndices[inferenceLeakyIntegrateAndFireSomaActivationByBranchBranchDimension].long()
			somaColumnIndices = stateIndices[inferenceLeakyIntegrateAndFireSomaActivationByBranchConceptDimension].long()
			somaFeatureIndices = stateIndices[inferenceLeakyIntegrateAndFireSomaActivationByBranchFeatureDimension].long()
			candidateKeys = GIAANNcmn_predictionActivate.calculateLeakyIntegrateAndFireBranchColumnFeatureKeys(branchIndices, somaColumnIndices, somaFeatureIndices, int(stateFeatures.shape[inferenceLeakyIntegrateAndFireConceptDimension]), int(databaseNetworkObject.f))
			eligibleMask = GIAANNcmn_predictionConstraints.buildSortedKeyMembershipMask(candidateKeys, somaActivationFromLastSegmentKeys)
			if(bool(pt.any(eligibleMask).item())):
				eligibleSomaSignalByBranch = pt.sparse_coo_tensor(stateIndices[:, eligibleMask], stateValues[eligibleMask], size=stateFeaturesSomaSignalByBranch.size(), dtype=stateFeaturesSomaSignalByBranch.dtype, device=stateFeaturesSomaSignalByBranch.device).coalesce()
				eligibleSomaSignal = GIAANNcmn_sparseTensors.reduceSparseBranchMax(eligibleSomaSignalByBranch)
				eligibleIndices = eligibleSomaSignal.indices()
				result = eligibleIndices[inferenceLeakyIntegrateAndFireSomaActivationConceptDimension].long(), eligibleIndices[inferenceLeakyIntegrateAndFireSomaActivationFeatureDimension].long(), eligibleSomaSignal.values()
	else:
		raise RuntimeError("calculateLeakyIntegrateAndFireLastSegmentSelectionActivationDistribution error: requires inferenceLeakyIntegrateAndFire enforceLastSegmentMustBeActive")
	return result

def requiresCandidateRequiredSegmentFilter():
	result = False
	if(useSANI and algorithmMatrixSANImethod=="enforceActivationAcrossSegments" and algorithmMatrixSANIenforceRequirement!="enforceAnySegmentMustBeActive"):
		result = True
	return result

def filterCandidatesByRequiredSegments(columnIndices, featureIndices, activationValues, stateFeatures, maxFeatures):
	filteredColumns = columnIndices
	filteredFeatures = featureIndices
	filteredActivations = activationValues
	if(useSANI and algorithmMatrixSANImethod=="enforceActivationAcrossSegments" and algorithmMatrixSANIenforceRequirement!="enforceAnySegmentMustBeActive"):
		if(filteredColumns is None or filteredFeatures is None or filteredActivations is None):
			filteredColumns = None
			filteredFeatures = None
			filteredActivations = None
		else:
			constraintKeys = calculateRequiredSegmentConstraintKeyTensor(stateFeatures, maxFeatures, filteredColumns.device)
			if(constraintKeys is None or constraintKeys.numel() == 0):
				filteredColumns = None
				filteredFeatures = None
				filteredActivations = None
			else:
				candidateKeys = filteredColumns.long() * int(maxFeatures) + filteredFeatures.long()
				selectedMask = GIAANNcmn_predictionConstraints.buildSortedKeyMembershipMask(candidateKeys, constraintKeys)
				if(selectedMask is None or selectedMask.sum().item() == 0):
					filteredColumns = None
					filteredFeatures = None
					filteredActivations = None
				else:
					indexTensor = pt.nonzero(selectedMask, as_tuple=False).view(-1)
					filteredColumns = filteredColumns.index_select(0, indexTensor)
					filteredFeatures = filteredFeatures.index_select(0, indexTensor)
					filteredActivations = filteredActivations.index_select(0, indexTensor)

	return filteredColumns, filteredFeatures, filteredActivations

def buildConstraintActivationKeyTensor(constraintActivation, maxFeatures, device):
	result = None
	if(constraintActivation is not None):
		if(not constraintActivation.is_sparse):
			constraintActivation = constraintActivation.to_sparse()
		if(constraintActivation.dim() == 3):
			if(multipleDendriticBranches):
				constraintActivation = GIAANNcmn_sparseTensors.reduceSparseBranchMax(constraintActivation)
			else:
				constraintActivation = GIAANNcmn_sparseTensors.collapseSparseBranchDimension(constraintActivation)
		if(constraintActivation.dim() != 2):
			raise RuntimeError("buildConstraintActivationKeyTensor error: constraintActivation must collapse to column/feature dimensions")
		constraintActivation = constraintActivation.coalesce()
		if(constraintActivation._nnz() == 0):
			result = pt.empty((0,), dtype=pt.long, device=device)
		else:
			constraintIndices = constraintActivation.indices().to(device)
			constraintKeys = constraintIndices[0].long() * int(maxFeatures) + constraintIndices[1].long()
			result = pt.sort(pt.unique(constraintKeys)).values
	return result

def calculateRequiredSegmentConstraintKeyTensor(stateFeatures, maxFeatures, device):
	result = None
	if(stateFeatures is None):
		result = None
	else:
		if(algorithmMatrixSANIenforceRequirement=="enforceAnySegmentMustBeActive"):
			result = None
		elif(algorithmMatrixSANIenforceRequirement=="enforceLastSegmentMustBeActive"):
			result = calculateLastSegmentConstraintKeyTensor(stateFeatures, maxFeatures, device)
		elif(algorithmMatrixSANIenforceRequirement=="enforceAllSegmentsMustBeActive"):
			constraintActivation = calculateRequiredSegmentConstraintActivation(stateFeatures)
			result = buildConstraintActivationKeyTensor(constraintActivation, maxFeatures, device)
		else:
			raise RuntimeError("calculateRequiredSegmentConstraintKeyTensor error: algorithmMatrixSANIenforceRequirement is invalid")
	return result

def calculateLastSegmentConstraintKeyTensor(stateFeatures, maxFeatures, device):
	result = None
	if(stateFeatures is None):
		result = None
	else:
		if(stateFeatures.is_sparse):
			if(stateFeatures._nnz() == 0):
				result = pt.empty((0,), dtype=pt.long, device=device)
			else:
				lastSegmentConstraint = calculateLastSegmentConstraintIndex()
				indices = stateFeatures._indices()
				hasBranchDim = stateFeatures.dim() == 4
				if(hasBranchDim):
					segmentDim = 1
					columnDim = 2
					featureDim = 3
				else:
					if(stateFeatures.dim() != 3):
						raise RuntimeError("calculateLastSegmentConstraintKeyTensor error: sparse stateFeatures must have segment/column/feature dimensions")
					segmentDim = 0
					columnDim = 1
					featureDim = 2
				segmentMask = indices[segmentDim] == lastSegmentConstraint
				if(segmentMask.sum().item() == 0):
					result = pt.empty((0,), dtype=pt.long, device=device)
				else:
					constraintKeys = indices[columnDim, segmentMask].to(device).long() * int(maxFeatures) + indices[featureDim, segmentMask].to(device).long()
					result = pt.sort(pt.unique(constraintKeys)).values
		else:
			constraintActivation = calculateLastSegmentConstraintActivation(stateFeatures)
			result = buildConstraintActivationKeyTensor(constraintActivation, maxFeatures, device)
	return result

def calculateRequiredSegmentConstraintActivation(stateFeatures):
	result = None
	if(stateFeatures is None):
		result = None
	else:
		if(algorithmMatrixSANIenforceRequirement=="enforceAnySegmentMustBeActive"):
			result = None
		elif(algorithmMatrixSANIenforceRequirement=="enforceLastSegmentMustBeActive"):
			result = calculateLastSegmentConstraintActivation(stateFeatures)
		elif(algorithmMatrixSANIenforceRequirement=="enforceAllSegmentsMustBeActive"):
			stateFeaturesConstraint = stateFeatures
			if(not stateFeaturesConstraint.is_sparse):
				stateFeaturesConstraint = stateFeaturesConstraint.to_sparse()
			result = GIAANNcmn_sparseTensors.neuronActivationSparse(stateFeaturesConstraint, algorithmMatrixSANImethod)
		else:
			raise RuntimeError("calculateRequiredSegmentConstraintActivation error: algorithmMatrixSANIenforceRequirement is invalid")
	return result

def calculateLastSegmentConstraintIndex():
	result = None
	if(enforceActivationAcrossSegmentsIgnoreInternalColumn):
		result = arrayIndexSegmentAdjacentColumn
	else:
		result = arrayIndexSegmentLast
	return result

def calculateLastSegmentConstraintActivation(stateFeatures):
	lastSegmentActivation = None
	lastSegmentConstraint = calculateLastSegmentConstraintIndex()
	hasBranchDim = (stateFeatures.dim() == 4)
	if(stateFeatures.is_sparse):
		if(hasBranchDim):
			lastSegmentActivation = GIAANNcmn_sparseTensors.sliceSparseTensor(stateFeatures, 1, lastSegmentConstraint)
		else:
			lastSegmentActivation = GIAANNcmn_sparseTensors.sliceSparseTensor(stateFeatures, 0, lastSegmentConstraint)
	else:
		if(hasBranchDim):
			lastSegmentActivation = stateFeatures[:, lastSegmentConstraint]
		else:
			lastSegmentActivation = stateFeatures[lastSegmentConstraint]
	return lastSegmentActivation

def buildConstraintActivationKeySet(constraintActivation, maxFeatures):
	result = None
	if(constraintActivation is not None):
		if(not constraintActivation.is_sparse):
			constraintActivation = constraintActivation.to_sparse()
		if(constraintActivation.dim() == 3):
			if(multipleDendriticBranches):
				constraintActivation = GIAANNcmn_sparseTensors.reduceSparseBranchMax(constraintActivation)
			else:
				constraintActivation = GIAANNcmn_sparseTensors.collapseSparseBranchDimension(constraintActivation)
		if(constraintActivation.dim() != 2):
			raise RuntimeError("buildConstraintActivationKeySet error: constraintActivation must collapse to column/feature dimensions")
		constraintActivation = constraintActivation.coalesce()
		if(constraintActivation._nnz() == 0):
			result = set()
		else:
			constraintIndices = constraintActivation.indices()
			constraintKeys = (constraintIndices[0].cpu() * maxFeatures + constraintIndices[1].cpu()).tolist()
			result = set(int(value) for value in constraintKeys)
	return result

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
		if(not GIAANNcmn_predictionConstraints.constraintAllowsColumn(columnIndex, constraintState)):
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
	columnData = buildInstanceColumnData(columnIndices, featureIndices, activationValues, databaseNetworkObject, constraintState)
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
		featureIndex = featureIndices[activationIndex].item()
		if(not GIAANNcmn_predictionConstraints.constraintAllowsNode(databaseNetworkObject, columnIndex, featureIndex, constraintState)):
			continue
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
		featureIndex = featureIndices[indices[0]].item()
		if(not GIAANNcmn_predictionConstraints.constraintAllowsNode(databaseNetworkObject, columnIndex, featureIndex, constraintState)):
			return candidates
		connectionValue = getConnectionValue(strengthLookup, columnIndex, featureIndex, maxFeatures)
		candidates.append({"columnIndex": columnIndex, "featureIndex": featureIndex, "nodes": [(columnIndex, featureIndex)], "connectionValue": connectionValue, "activationValue": values[0].item()})
	return candidates

def buildInstanceColumnData(columnIndices, featureIndices, activationValues, databaseNetworkObject, constraintState=None):
	columnData = {}
	for idx in range(columnIndices.shape[0]):
		columnIndex = columnIndices[idx].item()
		featureIndex = featureIndices[idx].item()
		if(not GIAANNcmn_predictionConstraints.constraintAllowsNode(databaseNetworkObject, columnIndex, featureIndex, constraintState)):
			continue
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

def prepareBeamNodes(databaseNetworkObject, nodes, conceptActivationState, constraintState, strengthLookup, maxFeatures):
	preparedNodes = []
	connectionSum = 0.0
	seenNodes = set()
	for columnIndex, featureIndex in nodes:
		adjustedFeature = featureIndex
		if(constraintState is not None and not GIAANNcmn_predictionConstraints.constraintAllowsNode(databaseNetworkObject, columnIndex, adjustedFeature, constraintState)):
			continue
		nodeKey = (columnIndex, adjustedFeature)
		if(nodeKey in seenNodes):
			continue
		seenNodes.add(nodeKey)
		preparedNodes.append((columnIndex, adjustedFeature))
		connectionSum += getConnectionValue(strengthLookup, columnIndex, adjustedFeature, maxFeatures)
	return preparedNodes, connectionSum

def nodesContainReferenceSetDelimiter(databaseNetworkObject, nodes):
	for nodeColumn, nodeFeature in nodes:
		if(GIAANNcmn_databaseNetwork.isFeatureIndexReferenceSetDelimiterDeterministic(databaseNetworkObject, nodeFeature)):
			return True
	return False

def nodesContainProbabilisticReferenceSetDelimiter(databaseNetworkObject, nodes):
	for nodeColumn, nodeFeature in nodes:
		if(GIAANNcmn_databaseNetwork.isFeatureIndexReferenceSetDelimiterProbabilistic(databaseNetworkObject, nodeFeature)):
			return True
	return False

def updateConstraintStateAfterNodes(databaseNetworkObject, previousConstraintState, nodes):
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

def buildStrengthLookup(databaseNetworkObject, globalFeatureNeuronsStrength, maxFeatures):
	strengthLookup = None
	if(inferenceStrengthLookupCache):
		if(databaseNetworkObject is None):
			raise RuntimeError("buildStrengthLookup: databaseNetworkObject is None")
		cacheValid = False
		if(hasattr(databaseNetworkObject, "strengthLookupCache") and hasattr(databaseNetworkObject, "strengthLookupMaxFeatures") and hasattr(databaseNetworkObject, "strengthLookupMaxColumns")):
			if(databaseNetworkObject.strengthLookupCache is not None and databaseNetworkObject.strengthLookupMaxFeatures == maxFeatures and databaseNetworkObject.strengthLookupMaxColumns == databaseNetworkObject.c):
				cacheValid = True
		if(cacheValid):
			strengthLookup = databaseNetworkObject.strengthLookupCache
		else:
			strengthLookup = buildStrengthLookupInternal(globalFeatureNeuronsStrength, maxFeatures)
			databaseNetworkObject.strengthLookupCache = strengthLookup
			databaseNetworkObject.strengthLookupMaxFeatures = maxFeatures
			databaseNetworkObject.strengthLookupMaxColumns = databaseNetworkObject.c
	else:
		strengthLookup = buildStrengthLookupInternal(globalFeatureNeuronsStrength, maxFeatures)
	return strengthLookup

def buildStrengthLookupInternal(globalFeatureNeuronsStrength, maxFeatures):
	strengthLookup = None
	columnIndices = None
	featureIndices = None
	values = None
	columnIndices, featureIndices, values = GIAANNcmn_predictionConstraints.aggregateSparseColumnFeatureValues(globalFeatureNeuronsStrength, maxFeatures)
	if(columnIndices is not None):
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
