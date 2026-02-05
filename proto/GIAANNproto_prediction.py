"""GIAANNproto_prediction.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_main.py

# Usage:
see GIAANNproto_main.py

# Description:
GIA ANN proto prediction

"""

import torch as pt
import time

from GIAANNproto_globalDefs import *
import GIAANNproto_databaseNetworkExcitation
import GIAANNproto_databaseNetworkTrainExcitation
import GIAANNproto_databaseNetworkDrawExcitation
import GIAANNproto_sparseTensors
import GIAANNproto_sequenceTokens
import GIAANNproto_predictionBeamSearch
import GIAANNproto_sequenceConcepts
import GIAANNproto_predictionActivate
import GIAANNproto_predictionConstraints


if(debugPrintTotalInferenceTokens):
	totalInferenceTokensSeed = 0
	totalInferenceTokensPrediction = 0
	totalInferenceTokensAll = 0

totalInferenceTop1Matches = 0
totalInferenceTop1Tokens = 0
totalInferenceTop1PredictionMatches = 0
totalInferenceTop1PredictionTokens = 0

def resetTotalInferenceTokens():
	if(debugPrintTotalInferenceTokens):
		global totalInferenceTokensSeed
		global totalInferenceTokensPrediction
		global totalInferenceTokensAll
		totalInferenceTokensSeed = 0
		totalInferenceTokensPrediction = 0
		totalInferenceTokensAll = 0
	return

def addTotalInferenceTokens(seedTokensCount, predictionTokensCount):
	if(debugPrintTotalInferenceTokens):
		if(seedTokensCount is None or predictionTokensCount is None):
			raise RuntimeError("addTotalInferenceTokens error: token counts are None")
		if(seedTokensCount < 0 or predictionTokensCount < 0):
			raise RuntimeError("addTotalInferenceTokens error: token counts must be non-negative")
		global totalInferenceTokensSeed
		global totalInferenceTokensPrediction
		global totalInferenceTokensAll
		totalInferenceTokensSeed += int(seedTokensCount)
		totalInferenceTokensPrediction += int(predictionTokensCount)
		totalInferenceTokensAll += int(seedTokensCount) + int(predictionTokensCount)
	return

def printTotalInferenceTokens():
	if(debugPrintTotalInferenceTokens):
		print("debugPrintTotalInferenceTokens: seedPhaseTokens = ", totalInferenceTokensSeed, ", predictionPhaseTokens = ", totalInferenceTokensPrediction, ", totalInferenceTokens = ", totalInferenceTokensAll)
	return

def resetInferenceTop1AccuracyCounts():
	global totalInferenceTop1Matches
	global totalInferenceTop1Tokens
	global totalInferenceTop1PredictionMatches
	global totalInferenceTop1PredictionTokens
	totalInferenceTop1Matches = 0
	totalInferenceTop1Tokens = 0
	totalInferenceTop1PredictionMatches = 0
	totalInferenceTop1PredictionTokens = 0
	return

def addInferenceTop1AccuracyCount(featurePredictionTargetMatch, seedPhase):
	if(not inferenceUseNextTokenPredictionsOrTargetsToActivateNextColumnFeatures):
		if(featurePredictionTargetMatch):
			matchValue = 1
		else:
			matchValue = 0
		global totalInferenceTop1Matches
		global totalInferenceTop1Tokens
		global totalInferenceTop1PredictionMatches
		global totalInferenceTop1PredictionTokens
		totalInferenceTop1Matches += matchValue
		totalInferenceTop1Tokens += 1
		if(not seedPhase):
			totalInferenceTop1PredictionMatches += matchValue
			totalInferenceTop1PredictionTokens += 1
	return

def printInferenceTop1Accuracy():
	if(not inferenceUseNextTokenPredictionsOrTargetsToActivateNextColumnFeatures):
		if(totalInferenceTop1Tokens <= 0 or totalInferenceTop1PredictionTokens <= 0):
			print("printInferenceTop1Accuracy: no prediction tokens recorded; skipping accuracy")
		else:
			predictionAccuracy = totalInferenceTop1PredictionMatches / totalInferenceTop1PredictionTokens
			inferenceAccuracy = totalInferenceTop1Matches / totalInferenceTop1Tokens
			print("averageTop1Accuracy: predictionTokens = ", predictionAccuracy, ", inferenceTokens = ", inferenceAccuracy)
	return

if(inferenceOnlyRetainPredictedTargetObservedColumn):
	def loadObservedColumnInference(databaseNetworkObject, observedColumnsDict, conceptIndex, sequenceWordIndex):
		lemma = databaseNetworkObject.conceptColumnsList[conceptIndex]
		observedColumn = GIAANNproto_databaseNetworkExcitation.loadOrCreateObservedColumn(databaseNetworkObject, conceptIndex, lemma, sequenceWordIndex)
		if(inferenceOnlyRetainPredictedTargetObservedColumn):
			if(observedColumnsDict is None):
				raise RuntimeError("loadObservedColumnInference error: observedColumnsDict is None")
			observedColumnsDict.clear()
		observedColumnsDict[lemma] = observedColumn
		return observedColumn
	
# Define the SequenceObservedColumnsInferencePrediction class
class SequenceObservedColumnsInferencePrediction:
	def __init__(self, databaseNetworkObject, observedColumnsDict, observedColumnsSequenceWordIndexDict):
		#note cs may be slightly longer than number of unique columns in the sequence, if there are multiple instances of the same concept/noun lemma in the sequence
	
		self.databaseNetworkObject = databaseNetworkObject
		
		self.observedColumnsDict = observedColumnsDict	# key: lemma, value: ObservedColumn
		self.observedColumnsSequenceWordIndexDict = observedColumnsSequenceWordIndexDict	# key: sequence word index, value: ObservedColumn
		
		self.cs2 = len(databaseNetworkObject.conceptColumnsDict)
		self.fs2 = len(databaseNetworkObject.conceptFeaturesDict)
			
		featureConnectionsList = []
		for observedColumn in observedColumnsSequenceWordIndexDict.values():
			 featureConnectionsList.append(observedColumn.featureConnections)
		self.featureConnections = pt.stack(featureConnectionsList, dim=3)
		

def buildAllowedColumnsLookup(conceptColumnsIndices, totalColumns):
	if(conceptColumnsIndices is None or conceptColumnsIndices.numel() == 0):
		return None
	allowedColumnsList = []
	seenColumns = set()
	for columnValue in conceptColumnsIndices.cpu().tolist():
		if(columnValue < 0 or columnValue >= totalColumns):
			continue
		if(columnValue not in seenColumns):
			allowedColumnsList.append(columnValue)
			seenColumns.add(columnValue)
	if(len(allowedColumnsList) == 0):
		return None
	device = conceptColumnsIndices.device
	dtype = conceptColumnsIndices.dtype
	return pt.tensor(allowedColumnsList, dtype=dtype, device=device)

def activatedNodesAreReferenceSetDelimiters(databaseNetworkObject, conceptColumnsFeatureIndices):
	if(conceptColumnsFeatureIndices is None):
		return False
	if(conceptColumnsFeatureIndices.numel() == 0):
		return False
	flattenedFeatureIndices = conceptColumnsFeatureIndices.reshape(-1)
	if(flattenedFeatureIndices.numel() == 0):
		return False
	for featureIndexTensor in flattenedFeatureIndices:
		featureIndex = featureIndexTensor.item()
		if(not GIAANNproto_databaseNetworkExcitation.isFeatureIndexReferenceSetDelimiterDeterministic(databaseNetworkObject, featureIndex)):
			return False
	return True

def activatedNodesAreProbabilisticReferenceSetDelimiters(databaseNetworkObject, conceptColumnsFeatureIndices):
	if(conceptColumnsFeatureIndices is None):
		return False
	if(conceptColumnsFeatureIndices.numel() == 0):
		return False
	flattenedFeatureIndices = conceptColumnsFeatureIndices.reshape(-1)
	if(flattenedFeatureIndices.numel() == 0):
		return False
	for featureIndexTensor in flattenedFeatureIndices:
		featureIndex = featureIndexTensor.item()
		if(not GIAANNproto_databaseNetworkExcitation.isFeatureIndexReferenceSetDelimiterProbabilistic(databaseNetworkObject, featureIndex)):
			return False
	return True

def ensurePredictionStateAvailable(conceptColumnsIndices, conceptColumnsFeatureIndices, sequenceWordIndex, wordPredictionIndex, tokensSequence, reason):
	if(conceptColumnsIndices is None or conceptColumnsIndices.numel() == 0):
		GIAANNproto_predictionConstraints.raisePredictionConnectivityError(sequenceWordIndex, wordPredictionIndex, tokensSequence, reason + " (missing concept columns)")
	if(conceptColumnsFeatureIndices is None or conceptColumnsFeatureIndices.numel() == 0):
		GIAANNproto_predictionConstraints.raisePredictionConnectivityError(sequenceWordIndex, wordPredictionIndex, tokensSequence, reason + " (missing column features)")

if(predictionColumnsMustActivateConceptFeature):
	def initialiseConceptActivationState(conceptColumnIndex, conceptColumnFeatureIndex):
		printe("predictionColumnsMustActivateConceptFeature is incomplete")
		initialState = set()
		result = updateConceptActivationState(initialState, conceptColumnIndex, conceptColumnFeatureIndex)
		return result

	def updateConceptActivationState(conceptActivationState, conceptColumnIndex, conceptColumnFeatureIndex):
		printe("predictionColumnsMustActivateConceptFeature is incomplete")
		result = conceptActivationState
		if(result is None):
			result = set()
		if(conceptColumnIndex is not None and conceptColumnFeatureIndex is not None):
			result.add(int(conceptColumnIndex))
		return result

	def enforceConceptFeaturePredictionOrder(conceptColumnIndex, conceptColumnFeatureIndex, conceptActivationState):
		printe("predictionColumnsMustActivateConceptFeature is incomplete")
		conceptColumnIndexActivation = conceptColumnIndex
		conceptColumnFeatureIndexActivation = conceptColumnFeatureIndex
		if(resultColumnIndex is not None and resultFeatureIndex is not None):
			localState = conceptActivationState or set()
			if(int(resultColumnIndex) not in localState):
				localState.add(int(resultColumnIndex))
		#return conceptColumnIndexActivation, conceptColumnFeatureIndexActivation

if not drawSequenceObservedColumns:
	class SequenceObservedColumnsDraw:
		def __init__(self, databaseNetworkObject, observedColumnsDict):
			self.databaseNetworkObject = databaseNetworkObject
			self.observedColumnsDict = observedColumnsDict

def processConceptWordsInference(sequenceObservedColumns, sequenceIndex, sequence, sequenceSeed, sequencePredict, numSeedTokens):
	print("processConceptWordsInference:")

	sequenceWordIndex = 0

	tokensSequence = GIAANNproto_sequenceTokens.getTokens(sequence)
	conceptMask, conceptIndices, numberConcepts = GIAANNproto_sequenceConcepts.createConceptMask(sequenceObservedColumns, tokensSequence)

	numPredictionTokens = len(sequencePredict)	#set numPredictionTokens (dynamic)
				
	#identify first activated column(s) in seed phase:
	kcMax = 1	#not used
	initialContextWordIndex = 0		#use first seed token as the context for the first prediction
	initialContextWordIndex = max(0, min(initialContextWordIndex, len(tokensSequence)-1))
	targetPreviousColumnIndex, targetNextColumnIndex, targetFeatureIndex = GIAANNproto_databaseNetworkExcitation.getTokenConceptFeatureIndexTensor(sequenceObservedColumns, tokensSequence, conceptMask, initialContextWordIndex, kcMax)
	conceptColumnIndex = int(targetPreviousColumnIndex)
	conceptColumnFeatureIndex = int(targetFeatureIndex)
	conceptActivationState = None
	databaseNetworkObject = sequenceObservedColumns.databaseNetworkObject
	globalFeatureNeuronsActivation = databaseNetworkObject.globalFeatureNeurons[arrayIndexPropertiesActivationIndex]
	globalFeatureNeuronsTime = None
	if(inferenceUseNeuronFeaturePropertiesTime):
		globalFeatureNeuronsTime = databaseNetworkObject.globalFeatureNeurons[arrayIndexPropertiesTimeIndex]
	if(predictionColumnsMustActivateConceptFeature):
		conceptActivationState = initialiseConceptActivationState(conceptColumnIndex, conceptColumnFeatureIndex)
	observedColumnsDict = sequenceObservedColumns.observedColumnsDict  # key: lemma, value: ObservedColumn	#every observed column in inference (seed and prediction phases)
	if(inferenceOnlyRetainPredictedTargetObservedColumn):
		observedColumnsDict = {}
	seedTokensProcessed = 0
	predictionTokensProcessed = 0

	try:
		#seed first tokens;
		for wordSeedIndex in range(numSeedTokens):
			sequenceWordIndex = wordSeedIndex
			wordPredictionIndex = wordSeedIndex
			featurePredictionTargetMatch, conceptColumnIndexNext, conceptColumnFeatureIndexNext, conceptActivationState, globalFeatureNeuronsActivation, globalFeatureNeuronsTime = processColumnInferencePrediction(sequenceObservedColumns, sequenceIndex, observedColumnsDict, wordPredictionIndex, sequenceWordIndex, tokensSequence, conceptColumnIndex, conceptColumnFeatureIndex, conceptMask, conceptActivationState, globalFeatureNeuronsActivation, globalFeatureNeuronsTime, seedPhase=True)
			conceptColumnIndex = int(conceptColumnIndexNext)
			conceptColumnFeatureIndex = int(conceptColumnFeatureIndexNext)
			seedTokensProcessed += 1
			addInferenceTop1AccuracyCount(featurePredictionTargetMatch, True)

		#predict next tokens;
		for wordPredictionIndex in range(numPredictionTokens):
			sequenceWordIndex = numSeedTokens + wordPredictionIndex
			featurePredictionTargetMatch, conceptColumnIndexNext, conceptColumnFeatureIndexNext, conceptActivationState, globalFeatureNeuronsActivation, globalFeatureNeuronsTime = processColumnInferencePrediction(sequenceObservedColumns, sequenceIndex, observedColumnsDict, wordPredictionIndex, sequenceWordIndex, tokensSequence, conceptColumnIndex, conceptColumnFeatureIndex, conceptMask, conceptActivationState, globalFeatureNeuronsActivation, globalFeatureNeuronsTime)
			conceptColumnIndex = int(conceptColumnIndexNext)
			conceptColumnFeatureIndex = int(conceptColumnFeatureIndexNext)
			predictionTokensProcessed += 1
			addInferenceTop1AccuracyCount(featurePredictionTargetMatch, False)
			if(not featurePredictionTargetMatch):
				print("warning: featurePredictionTargetMatch=False")
				if(debugTerminateInferenceOnPredictionTargetMismatch):
					print("debugTerminateInferenceOnPredictionTargetMismatch: prematurely terminating inference")
					break
	except GIAANNproto_predictionConstraints.InferenceStopSequenceNoPredictionCandidatesAvailable:
		pass
	if(debugPrintTotalInferenceTokens):
		addTotalInferenceTokens(seedTokensProcessed, predictionTokensProcessed)
	if(drawNetworkDuringInference):
		databaseNetworkObject.globalFeatureNeurons = GIAANNproto_sparseTensors.replaceAllSparseTensorElementsAtFirstDimIndex(databaseNetworkObject.globalFeatureNeurons, globalFeatureNeuronsActivation, arrayIndexPropertiesActivationIndex)
		if(inferenceUseNeuronFeaturePropertiesTime):
			databaseNetworkObject.globalFeatureNeurons = GIAANNproto_sparseTensors.replaceAllSparseTensorElementsAtFirstDimIndex(databaseNetworkObject.globalFeatureNeurons, globalFeatureNeuronsTime, arrayIndexPropertiesTimeIndex)

def processColumnInferencePrediction(sequenceObservedColumns, sequenceIndex, observedColumnsDict, wordPredictionIndex, sequenceWordIndex, tokensSequence, conceptColumnIndex, conceptColumnFeatureIndex, conceptMask, conceptActivationState, globalFeatureNeuronsActivation, globalFeatureNeuronsTime, seedPhase=False):
	
	#intialise function variables;
	databaseNetworkObject = sequenceObservedColumns.databaseNetworkObject
	debugTimeStart = None
	debugTimeLast = None
	predictionCandidatesAvailable = True
	if(conceptColumnIndex is None or conceptColumnFeatureIndex is None):
		raise RuntimeError("processColumnInferencePrediction error: expected single concept/feature prediction pair")
	if(globalFeatureNeuronsActivation is None):
		raise RuntimeError("processColumnInferencePrediction error: globalFeatureNeuronsActivation is None")
	conceptColumnIndexTensor = pt.tensor([int(conceptColumnIndex)], dtype=pt.long)
	conceptColumnFeatureIndexTensor = pt.tensor([int(conceptColumnFeatureIndex)], dtype=pt.long)
	conceptColumnIndexActivation = int(conceptColumnIndex)
	conceptColumnFeatureIndexActivation = int(conceptColumnFeatureIndex)
	conceptColumnIndexTensorActivation = conceptColumnIndexTensor
	conceptColumnFeatureIndexTensorActivation = conceptColumnFeatureIndexTensor
	
	if(predictionEnsureConnectedToPreviousPrediction or enforceDirectConnectionsMinWordDistance):
		if(inferenceUseNextTokenPredictionsOrTargetsToActivateNextColumnFeatures):
			if(not (seedPhase and enforceDirectConnectionsIgnoreSeed)):
				ensurePredictionStateAvailable(conceptColumnIndexTensor, conceptColumnFeatureIndexTensor, sequenceWordIndex, wordPredictionIndex, tokensSequence, "no connected context available before prediction")

	#burst the initial seed in the sequence;
	if(sequenceWordIndex==0):
		#activate source token (incremental seed during train)
			#if(wordPredictionIndex == 1) will reactivate first seed token column feature (as it was not saved during wordPredictionIndex==0)
		branchIndex = 0
		indicesToUpdateList = [branchIndex, arrayIndexSegmentLast, int(conceptColumnIndex), int(conceptColumnFeatureIndex)]
		globalFeatureNeuronsActivation = GIAANNproto_sparseTensors.addElementValueToSparseTensor(globalFeatureNeuronsActivation, indicesToUpdateList, j1)
	
	globalFeatureNeuronsStrength = databaseNetworkObject.globalFeatureNeurons[arrayIndexPropertiesStrengthIndex]
	if(inferenceUseNeuronFeaturePropertiesTime and globalFeatureNeuronsTime is None):
		raise RuntimeError("processColumnInferencePrediction error: globalFeatureNeuronsTime is None while inferenceUseNeuronFeaturePropertiesTime")
	sequenceColumnIndex = None
	if(inferenceUseNeuronFeaturePropertiesTime):
		if(useSANIcolumns or useSANIfeaturesAndColumns):
			sequenceColumnIndex = GIAANNproto_predictionActivate.calculateSequenceColumnIndex(conceptMask, sequenceWordIndex)
	globalFeatureConnectionsActivation = None

	#set constraintModePrediction;
	allowedColumnsConstraint = None
	constraintModePrediction = None
	probabilisticDelimiterActive = False
	if(conceptColumnsDelimitByPOS):
		if(conceptColumnFeatureIndexTensor is not None and conceptColumnFeatureIndexTensor.numel() > 0):
			probabilisticDelimiterActive = activatedNodesAreProbabilisticReferenceSetDelimiters(databaseNetworkObject, conceptColumnFeatureIndexTensor)
		allowedColumnsConstraint = buildAllowedColumnsLookup(conceptColumnIndexTensor, databaseNetworkObject.c)
		if(allowedColumnsConstraint is not None and allowedColumnsConstraint.numel() > 0):
			if(conceptColumnFeatureIndexTensor is None or conceptColumnFeatureIndexTensor.numel() == 0):
				constraintModePrediction = "internal"
			else:
				isDelimiterNode = activatedNodesAreReferenceSetDelimiters(databaseNetworkObject, conceptColumnFeatureIndexTensor)
				if(isDelimiterNode):
					constraintModePrediction = "delimiter"
				else:
					if(probabilisticDelimiterActive):
						constraintModePrediction = None
					else:
						constraintModePrediction = "internal"
	else:
		printe("conceptColumnsDelimitByPOS is required")
	if(seedPhase and constraintModePrediction == "delimiter"):
		constraintModePrediction = None
	
	#set predictionEnsureConnectedToPreviousPrediction connectedColumnsConstraint/connectedColumnsFeatureMap;
	if((predictionEnsureConnectedToPreviousPrediction or enforceDirectConnectionsMinWordDistance) and (inferenceSeedNetwork and sequenceWordIndex > 0) and not (seedPhase and enforceDirectConnectionsIgnoreSeed)):
		#limit prediction candidates to columns directly connected to previously predicted nodes
		connectedColumnsConstraint, connectedColumnsFeatureMap = GIAANNproto_predictionConstraints.buildConnectedColumnsLookupFromPrediction(databaseNetworkObject, observedColumnsDict, conceptColumnIndexTensor, conceptColumnFeatureIndexTensor)
	else:
		connectedColumnsConstraint = None
		connectedColumnsFeatureMap = None
	if((predictionEnsureConnectedToPreviousPrediction or enforceDirectConnectionsMinWordDistance) and (inferenceSeedNetwork and sequenceWordIndex > 0) and connectedColumnsConstraint is not None and connectedColumnsConstraint.numel() == 0 and inferenceUseNextTokenPredictionsOrTargetsToActivateNextColumnFeatures):
		GIAANNproto_predictionConstraints.raiseOrStopPredictionConnectivityError(sequenceWordIndex, wordPredictionIndex, tokensSequence, "previous prediction has no outgoing connections")
		
	if(sequenceWordIndex > 0):
		#set conceptColumnIndexActivation/conceptColumnFeatureIndexActivation;
		conceptColumnIndexActivation = conceptColumnIndex
		conceptColumnFeatureIndexActivation = conceptColumnFeatureIndex
		if(predictionColumnsMustActivateConceptFeature):
			conceptColumnIndexActivation, conceptColumnFeatureIndexActivation = enforceConceptFeaturePredictionOrder(conceptColumnIndex, conceptColumnFeatureIndex, conceptActivationState)
		conceptColumnIndexTensorActivation = pt.tensor([int(conceptColumnIndexActivation)], dtype=pt.long)
		conceptColumnFeatureIndexTensorActivation = pt.tensor([int(conceptColumnFeatureIndexActivation)], dtype=pt.long)
		
		#populate sequence observed columns;
		if(inferenceOnlyRetainPredictedTargetObservedColumn):
			sequenceObservedColumnsPrediction = SequenceObservedColumnsDraw(databaseNetworkObject, observedColumnsDict)
		else:
			# Refresh the observed columns dictionary for each new sequence
			observedColumnsSequenceCandidateIndexDict = {}  # key: sequence candidate index, value: ObservedColumn	#used to populate sequence feature connection arrays based on observed columns (i does not correspond to sequence word index as assumed by observedColumnsSequenceWordIndexDict)
			words = []
			lemmas = []
			lemma = databaseNetworkObject.conceptColumnsList[int(conceptColumnIndex)]
			word = lemma	#same for concepts (not used)
			lemmas.append(lemma)
			words.append(word)
			# Load observed column from disk or create new one
			observedColumn = GIAANNproto_databaseNetworkExcitation.loadOrCreateObservedColumn(databaseNetworkObject, int(conceptColumnIndex), lemma, sequenceWordIndex)
			observedColumnsDict[lemma] = observedColumn
			observedColumnsSequenceCandidateIndexDict[0] = observedColumn
			sequenceObservedColumnsPrediction = SequenceObservedColumnsInferencePrediction(databaseNetworkObject, observedColumnsDict, observedColumnsSequenceCandidateIndexDict)
			
		#decrement global activations;
		if(inferenceDecrementActivations):
			#decrement activation after each prediction interval
			globalFeatureNeuronsActivation = GIAANNproto_predictionActivate.decrementActivation(globalFeatureNeuronsActivation, activationDecrementPerPredictedToken)
			#if(inferenceUseNeuronFeaturePropertiesTime):	#OLD
			#	globalFeatureNeuronsTime = GIAANNproto_predictionActivate.decrementActivation(globalFeatureNeuronsTime, inferenceUseNeuronFeaturePropertiesTimeDecrement)

		#set activationSequenceWordIndex/activationSequenceColumnIndex;
		activationSequenceWordIndex = sequenceWordIndex
		activationSequenceColumnIndex = sequenceColumnIndex
		if(inferenceUseNeuronFeaturePropertiesTime and sequenceWordIndex > 0):
			activationSequenceWordIndex = sequenceWordIndex - 1
			if(activationSequenceWordIndex < 0):
				raise RuntimeError("processColumnInferencePrediction: activationSequenceWordIndex out of range")
			if(useSANIcolumns or useSANIfeaturesAndColumns):
				activationSequenceColumnIndex = GIAANNproto_predictionActivate.calculateSequenceColumnIndex(conceptMask, activationSequenceWordIndex)
				
		#process features (activate global neurons based on connection targets);
		if(inferenceOnlyRetainPredictedTargetObservedColumn):
			sourceConceptIndexValue = int(conceptColumnIndexActivation)
			observedColumn = loadObservedColumnInference(databaseNetworkObject, observedColumnsDict, sourceConceptIndexValue, sequenceWordIndex)
			featureConnections = observedColumn.featureConnections
			globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, globalFeatureNeuronsTime = GIAANNproto_predictionActivate.processFeaturesActivePredict(databaseNetworkObject, globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, featureConnections, conceptColumnIndexActivation, conceptColumnFeatureIndexActivation, globalFeatureNeuronsTime, activationSequenceWordIndex, activationSequenceColumnIndex)
		else:
			globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, globalFeatureNeuronsTime = GIAANNproto_predictionActivate.processFeaturesActivePredictSingle(databaseNetworkObject, globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, sequenceObservedColumnsPrediction, conceptColumnIndexActivation, conceptColumnFeatureIndexActivation, globalFeatureNeuronsTime, activationSequenceWordIndex, activationSequenceColumnIndex)
	else:
		#activation targets have already been activated
		sequenceObservedColumnsPrediction = SequenceObservedColumnsDraw(databaseNetworkObject, observedColumnsDict)

	#deactivate previously predicted neurons;
	if(inferenceDeactivateNeuronsUponPrediction):
		branchIndex = 0
		if(multipleDendriticBranches):
			branchIndex = GIAANNproto_predictionActivate.selectActivatedBranchIndex(globalFeatureNeuronsActivation, int(conceptColumnIndex), int(conceptColumnFeatureIndex))
		branchTensor = pt.tensor(branchIndex, device=conceptColumnIndexTensor.device)
		if(useSANI):
			indicesToUpdateList = []
			for segmentIndex in range(arrayNumberOfSegments):
				indexToUpdate = pt.stack([branchTensor, pt.tensor(segmentIndex, device=conceptColumnIndexTensor.device), conceptColumnIndexTensor.squeeze(), conceptColumnFeatureIndexTensorActivation.squeeze()], dim=0)
				indicesToUpdateList.append(indexToUpdate)
			indicesToUpdate = pt.stack(indicesToUpdateList, dim=0)
		else:
			indicesToUpdate = pt.stack([branchTensor, pt.tensor(arrayIndexSegmentFirst, device=conceptColumnIndexTensor.device), conceptColumnIndexTensor.squeeze(), conceptColumnFeatureIndexTensorActivation.squeeze()], dim=0)
		if(inferenceDeactivateNeuronsUponPrediction):
			if(inferenceDeactivateNeuronsUponPrediction):
				modifier = 0
			globalFeatureNeuronsActivationOrig = globalFeatureNeuronsActivation
			globalFeatureNeuronsActivation = GIAANNproto_sparseTensors.modifySparseTensor(globalFeatureNeuronsActivation, indicesToUpdate, modifier, multiply=False)
			if(predictionColumnsMustActivateConceptFeature):
				conceptActivationState = updateConceptActivationState(conceptActivationState, conceptColumnIndexActivation, conceptColumnFeatureIndexActivation)
	
	#select next prediction column/feature;
	if(seedPhase):
		#seedPhase;
		targetPreviousColumnIndex, targetNextColumnIndex, targetFeatureIndex = GIAANNproto_databaseNetworkExcitation.getTokenConceptFeatureIndexTensor(sequenceObservedColumns, tokensSequence, conceptMask, sequenceWordIndex, kcNetwork)
		conceptColumnIndexNext = int(targetPreviousColumnIndex)
		conceptColumnFeatureIndexNext = int(targetFeatureIndex)
		conceptColumnIndexNextTensor = pt.tensor([conceptColumnIndexNext], dtype=pt.long)
		conceptColumnFeatureIndexNextTensor = pt.tensor([conceptColumnFeatureIndexNext], dtype=pt.long)
		conceptColumnIndexNextTensor, conceptColumnFeatureIndexNextTensor = GIAANNproto_predictionConstraints.applyColumnConstraintToPredictions(databaseNetworkObject, conceptColumnIndexNextTensor, conceptColumnFeatureIndexNextTensor, allowedColumnsConstraint, constraintModePrediction, connectedColumnsConstraint, connectedColumnsFeatureMap)
		if(conceptColumnIndexNextTensor is None or conceptColumnFeatureIndexNextTensor is None or conceptColumnIndexNextTensor.numel() == 0 or conceptColumnFeatureIndexNextTensor.numel() == 0):
			GIAANNproto_predictionConstraints.raiseOrStopPredictionConnectivityError(sequenceWordIndex, wordPredictionIndex, tokensSequence, "no prediction candidates available")
		if(conceptColumnIndexNextTensor.numel() != 1 or conceptColumnFeatureIndexNextTensor.numel() != 1):
			raise RuntimeError("processColumnInferencePrediction error: multiple prediction candidates not supported")
		conceptColumnIndexNext = int(conceptColumnIndexNextTensor.squeeze().item())
		conceptColumnFeatureIndexNext = int(conceptColumnFeatureIndexNextTensor.squeeze().item())
		if((predictionEnsureConnectedToPreviousPrediction or enforceDirectConnectionsMinWordDistance) and connectedColumnsConstraint is not None):
			if(conceptColumnIndexNext is None):
				GIAANNproto_predictionConstraints.raiseOrStopPredictionConnectivityError(sequenceWordIndex, wordPredictionIndex, tokensSequence, "no connected predictions available for current step")
			if(conceptColumnIndexNext is None):
				GIAANNproto_predictionConstraints.raiseOrStopPredictionConnectivityError(sequenceWordIndex, wordPredictionIndex, tokensSequence, "no connected activations available for next step")
		conceptColumnIndexSource = int(conceptColumnIndexNext)
		conceptColumnFeatureIndexSource = int(conceptColumnFeatureIndexNext)
		branchIndex = 0
		indicesToUpdateList = [branchIndex, arrayIndexSegmentLast, conceptColumnIndexSource, conceptColumnFeatureIndexSource]
		globalFeatureNeuronsActivation = GIAANNproto_sparseTensors.addElementValueToSparseTensor(globalFeatureNeuronsActivation, indicesToUpdateList, j1)
		conceptColumnIndexPred = conceptColumnIndexNext	#temporarily assign prediction from seed target for print only
		conceptColumnFeatureIndexPred = conceptColumnFeatureIndexNext	#temporarily assign prediction from seed target for print only
	else:	
		#predictionPhase;
		if(inferenceBeamSearch):
			try:
				conceptColumnIndexPred, conceptColumnFeatureIndexPred, targetPreviousColumnIndex, targetNextColumnIndex = GIAANNproto_predictionBeamSearch.beamSearchPredictNextFeature(sequenceObservedColumns, databaseNetworkObject, observedColumnsDict, globalFeatureNeuronsActivation, globalFeatureNeuronsStrength, globalFeatureConnectionsActivation, globalFeatureNeuronsTime, tokensSequence, wordPredictionIndex, sequenceWordIndex, conceptMask, allowedColumnsConstraint, constraintModePrediction, conceptActivationState, connectedColumnsConstraint, connectedColumnsFeatureMap)
				if(inferenceOnlyRetainPredictedTargetObservedColumn and not inferenceOnlyRetainPredictedTargetObservedColumnBeamSearch):
					if(observedColumnsDict is None):
						raise RuntimeError("processColumnInferencePrediction error: observedColumnsDict is None")
					observedColumnsDict.clear()
			except GIAANNproto_predictionConstraints.InferenceStopSequenceNoPredictionCandidatesAvailable:
				if(inferenceUseNextTokenPredictionsOrTargetsToActivateNextColumnFeatures):
					raise
				predictionCandidatesAvailable = False
				targetPreviousColumnIndex, targetNextColumnIndex, targetFeatureIndex = GIAANNproto_databaseNetworkExcitation.getTokenConceptFeatureIndexTensor(sequenceObservedColumns, tokensSequence, conceptMask, sequenceWordIndex, kcNetwork)
				conceptColumnIndexPred = int(targetPreviousColumnIndex)
				conceptColumnFeatureIndexPred = int(targetFeatureIndex)
		else:
			try:
				conceptColumnIndexPred, conceptColumnFeatureIndexPred, targetPreviousColumnIndex, targetNextColumnIndex = GIAANNproto_predictionBeamSearch.beamSearchSelectSingleStepFeature(sequenceObservedColumns, databaseNetworkObject, observedColumnsDict, globalFeatureNeuronsActivation, globalFeatureNeuronsStrength, globalFeatureConnectionsActivation, globalFeatureNeuronsTime, tokensSequence, wordPredictionIndex, sequenceWordIndex, conceptMask, allowedColumnsConstraint, constraintModePrediction, conceptActivationState, connectedColumnsConstraint, connectedColumnsFeatureMap)
			except GIAANNproto_predictionConstraints.InferenceStopSequenceNoPredictionCandidatesAvailable:
				if(inferenceUseNextTokenPredictionsOrTargetsToActivateNextColumnFeatures):
					raise
				predictionCandidatesAvailable = False
				targetPreviousColumnIndex, targetNextColumnIndex, targetFeatureIndex = GIAANNproto_databaseNetworkExcitation.getTokenConceptFeatureIndexTensor(sequenceObservedColumns, tokensSequence, conceptMask, sequenceWordIndex, kcNetwork)
				conceptColumnIndexPred = int(targetPreviousColumnIndex)
				conceptColumnFeatureIndexPred = int(targetFeatureIndex)
		if(inferenceUseNextTokenPredictionsOrTargetsToActivateNextColumnFeatures):
			conceptColumnIndexNext = conceptColumnIndexPred	#use prediction as next selected feature
			conceptColumnFeatureIndexNext = conceptColumnFeatureIndexPred	#use prediction as next selected feature
		else:
			targetPreviousColumnIndex, targetNextColumnIndex, targetFeatureIndex = GIAANNproto_databaseNetworkExcitation.getTokenConceptFeatureIndexTensor(sequenceObservedColumns, tokensSequence, conceptMask, sequenceWordIndex, kcNetwork)
			conceptColumnIndexNext = int(targetPreviousColumnIndex)
			conceptColumnFeatureIndexNext = int(targetFeatureIndex)
			conceptColumnIndexNextTensor = pt.tensor([conceptColumnIndexNext], dtype=pt.long)
			conceptColumnFeatureIndexNextTensor = pt.tensor([conceptColumnFeatureIndexNext], dtype=pt.long)
			constraintModeTarget = constraintModePrediction
			if(constraintModeTarget == "delimiter"):
				constraintModeTarget = None
			conceptColumnIndexNextTensor, conceptColumnFeatureIndexNextTensor = GIAANNproto_predictionConstraints.applyColumnConstraintToPredictions(databaseNetworkObject, conceptColumnIndexNextTensor, conceptColumnFeatureIndexNextTensor, allowedColumnsConstraint, constraintModeTarget, None, None)
			if(conceptColumnIndexNextTensor is None or conceptColumnFeatureIndexNextTensor is None or conceptColumnIndexNextTensor.numel() == 0 or conceptColumnFeatureIndexNextTensor.numel() == 0):
				GIAANNproto_predictionConstraints.raiseOrStopPredictionConnectivityError(sequenceWordIndex, wordPredictionIndex, tokensSequence, "no prediction candidates available")
			if(conceptColumnIndexNextTensor.numel() != 1 or conceptColumnFeatureIndexNextTensor.numel() != 1):
				raise RuntimeError("processColumnInferencePrediction error: multiple prediction candidates not supported")
			conceptColumnIndexNext = int(conceptColumnIndexNextTensor.squeeze().item())
			conceptColumnFeatureIndexNext = int(conceptColumnFeatureIndexNextTensor.squeeze().item())
			#connected predictions constraint is not applied in target-driven activation
	if(conceptColumnIndexPred is None or conceptColumnFeatureIndexPred is None):
		GIAANNproto_predictionConstraints.raiseOrStopPredictionConnectivityError(sequenceWordIndex, wordPredictionIndex, tokensSequence, "no prediction candidates available")

	#calculate featurePredictionTargetMatch; 
	featurePredictionTargetMatch = False
	targetToken = tokensSequence[sequenceWordIndex]
	targetWord = targetToken.word
	targetLemma = targetToken.lemma
	targetIsConceptFeature = bool(conceptMask[sequenceWordIndex].item())
	#compare topk column/feature predictions to sequencePredict (target words);
	#implementation limitation; only works with kf = 1;
	predictedWord = None
	predictedColumnName = None
	if(predictionCandidatesAvailable):
		columnName = databaseNetworkObject.conceptColumnsList[conceptColumnIndexPred]
		observedColumnFeatureIndex = conceptColumnFeatureIndexPred
		predictedIsConceptFeature = (observedColumnFeatureIndex == featureIndexPrimeConceptNeuron)
		if(predictedIsConceptFeature):
			predictedWord = columnName
		else:
			predictedWord = databaseNetworkObject.conceptFeaturesList[observedColumnFeatureIndex]
		predictedColumnName = columnName
		if(targetNextColumnIndex is None):
			targetColumnName = databaseNetworkObject.conceptColumnsList[targetPreviousColumnIndex]
		else:
			targetColumnName = databaseNetworkObject.conceptColumnsList[targetPreviousColumnIndex] + "/" + databaseNetworkObject.conceptColumnsList[targetNextColumnIndex]
		if(targetWord == predictedWord):
			featurePredictionTargetMatch = True
		elif(targetIsConceptFeature and predictedIsConceptFeature and targetLemma == predictedColumnName and targetColumnName == predictedColumnName):
			featurePredictionTargetMatch = True
	else:
		predictedWord = "<no prediction>"
		predictedColumnName = "<no prediction>"
		if(targetNextColumnIndex is None):
			targetColumnName = databaseNetworkObject.conceptColumnsList[targetPreviousColumnIndex]
		else:
			targetColumnName = databaseNetworkObject.conceptColumnsList[targetPreviousColumnIndex] + "/" + databaseNetworkObject.conceptColumnsList[targetNextColumnIndex]

	#print prediction; 
	if(printPredictionsDuringInferencePredict):
		print("\t sequenceWordIndex = ", sequenceWordIndex, ", wordPredictionIndex = ", wordPredictionIndex, ", targetWord = ", targetWord, ", predictedWord = ", predictedWord, ", targetColumn = ", targetColumnName, ", predictedColumn = ", predictedColumnName)

	#load predicted feature observed column; 
	if(inferenceOnlyRetainPredictedTargetObservedColumn):
		if(conceptColumnIndexPred is None):
			raise RuntimeError("processColumnInferencePrediction error: predicted columns required for inferenceOnlyRetainPredictedTargetObservedColumn")
		loadObservedColumnInference(databaseNetworkObject, observedColumnsDict, int(conceptColumnIndexPred), sequenceWordIndex)
	
	#draw network; 
	if(drawNetworkDuringInference):
		#FUTURE: convert globalFeatureNeuronsActivation back to globalFeatureNeurons for draw
		databaseNetworkObject.globalFeatureNeurons = GIAANNproto_sparseTensors.replaceAllSparseTensorElementsAtFirstDimIndex(databaseNetworkObject.globalFeatureNeurons, globalFeatureNeuronsActivation, arrayIndexPropertiesActivationIndex)
		if(inferenceUseNeuronFeaturePropertiesTime):
			databaseNetworkObject.globalFeatureNeurons = GIAANNproto_sparseTensors.replaceAllSparseTensorElementsAtFirstDimIndex(databaseNetworkObject.globalFeatureNeurons, globalFeatureNeuronsTime, arrayIndexPropertiesTimeIndex)
		GIAANNproto_databaseNetworkDrawExcitation.visualizeGraph(sequenceObservedColumnsPrediction, True, save=drawNetworkDuringInferenceSave, fileName=drawNetworkDuringInferenceSaveFilenamePrepend+generateDrawSequenceIndex(sequenceWordIndex))
	return featurePredictionTargetMatch, conceptColumnIndexNext, conceptColumnFeatureIndexNext, conceptActivationState, globalFeatureNeuronsActivation, globalFeatureNeuronsTime
