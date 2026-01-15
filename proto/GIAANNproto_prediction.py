"""GIAANNproto_prediction.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_prediction.py

# Usage:
see GIAANNproto_prediction.py

# Description:
GIA ANN proto prediction

"""

import torch as pt

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

	if(inferenceOnlyRetainPredictedTargetObservedColumn and transformerUseInputConnections):
		raise RuntimeError("processConceptWordsInference error: inferenceOnlyRetainPredictedTargetObservedColumn requires transformerUseInputConnections=False")
	
	if(transformerUseInputConnections):
		GIAANNproto_databaseNetworkExcitation.generateGlobalFeatureConnections(sequenceObservedColumns.databaseNetworkObject)
		
	numPredictionTokens = len(sequencePredict)	#set numPredictionTokens (dynamic)
				
	#identify first activated column(s) in seed phase:
	kcMax = 1	#not used
	initialContextWordIndex = 0		#use first seed token as the context for the first prediction
	initialContextWordIndex = max(0, min(initialContextWordIndex, len(tokensSequence)-1))
	targetPreviousColumnIndex, targetNextColumnIndex, targetFeatureIndex = GIAANNproto_databaseNetworkExcitation.getTokenConceptFeatureIndexTensor(sequenceObservedColumns, tokensSequence, conceptMask, initialContextWordIndex, kcMax)
	conceptColumnIndex = int(targetPreviousColumnIndex)
	conceptColumnFeatureIndex = int(targetFeatureIndex)
	conceptActivationState = None
	if(predictionColumnsMustActivateConceptFeature):
		conceptActivationState = initialiseConceptActivationState(conceptColumnIndex, conceptColumnFeatureIndex)
	observedColumnsDict = sequenceObservedColumns.observedColumnsDict  # key: lemma, value: ObservedColumn	#every observed column in inference (seed and prediction phases)
	if(inferenceOnlyRetainPredictedTargetObservedColumn):
		observedColumnsDict = {}

	try:
		#seed first tokens;
		for wordSeedIndex in range(numSeedTokens):
			sequenceWordIndex = wordSeedIndex
			wordPredictionIndex = wordSeedIndex
			featurePredictionTargetMatch, conceptColumnIndexNext, conceptColumnFeatureIndexNext, conceptActivationState = processColumnInferencePrediction(sequenceObservedColumns, sequenceIndex, observedColumnsDict, wordPredictionIndex, sequenceWordIndex, tokensSequence, conceptColumnIndex, conceptColumnFeatureIndex, conceptMask, conceptActivationState, seedPhase=True)
			conceptColumnIndex = int(conceptColumnIndexNext)
			conceptColumnFeatureIndex = int(conceptColumnFeatureIndexNext)

		#predict next tokens;
		for wordPredictionIndex in range(numPredictionTokens):
			sequenceWordIndex = numSeedTokens + wordPredictionIndex
			featurePredictionTargetMatch, conceptColumnIndexNext, conceptColumnFeatureIndexNext, conceptActivationState = processColumnInferencePrediction(sequenceObservedColumns, sequenceIndex, observedColumnsDict, wordPredictionIndex, sequenceWordIndex, tokensSequence, conceptColumnIndex, conceptColumnFeatureIndex, conceptMask, conceptActivationState)
			conceptColumnIndex = int(conceptColumnIndexNext)
			conceptColumnFeatureIndex = int(conceptColumnFeatureIndexNext)
			if(not featurePredictionTargetMatch):
				print("warning: featurePredictionTargetMatch=False")
				if(debugTerminateInferenceOnPredictionTargetMismatch):
					print("debugTerminateInferenceOnPredictionTargetMismatch: prematurely terminating inference")
					break
	except GIAANNproto_predictionConstraints.InferenceStopSequenceNoPredictionCandidatesAvailable:
		pass

def processColumnInferencePrediction(sequenceObservedColumns, sequenceIndex, observedColumnsDict, wordPredictionIndex, sequenceWordIndex, tokensSequence, conceptColumnIndex, conceptColumnFeatureIndex, conceptMask, conceptActivationState, seedPhase=False):
	
	#intialise function variables;
	databaseNetworkObject = sequenceObservedColumns.databaseNetworkObject
	if(conceptColumnIndex is None or conceptColumnFeatureIndex is None):
		raise RuntimeError("processColumnInferencePrediction error: expected single concept/feature prediction pair")
	conceptColumnIndexTensor = pt.tensor([int(conceptColumnIndex)], dtype=pt.long)
	conceptColumnFeatureIndexTensor = pt.tensor([int(conceptColumnFeatureIndex)], dtype=pt.long)
	conceptColumnIndexActivation = int(conceptColumnIndex)
	conceptColumnFeatureIndexActivation = int(conceptColumnFeatureIndex)
	conceptColumnIndexTensorActivation = conceptColumnIndexTensor
	conceptColumnFeatureIndexTensorActivation = conceptColumnFeatureIndexTensor
	
	if(predictionEnsureConnectedToPreviousPrediction or enforceDirectConnectionsMinWordDistance):
		ensurePredictionStateAvailable(conceptColumnIndexTensor, conceptColumnFeatureIndexTensor, sequenceWordIndex, wordPredictionIndex, tokensSequence, "no connected context available before prediction")

	#burst the initial seed in the sequence;
	if(sequenceWordIndex==0):
		#activate source token (incremental seed during train)
			#if(wordPredictionIndex == 1) will reactivate first seed token column feature (as it was not saved during wordPredictionIndex==0)
		branchIndex = 0
		indicesToUpdateList = [arrayIndexPropertiesActivationIndex, branchIndex, arrayIndexSegmentLast, int(conceptColumnIndex), int(conceptColumnFeatureIndex)]
		databaseNetworkObject.globalFeatureNeurons = GIAANNproto_sparseTensors.addElementValueToSparseTensor(databaseNetworkObject.globalFeatureNeurons, indicesToUpdateList, j1)
	
	#set globalFeatureNeuronsActivation;
	globalFeatureNeuronsActivation = databaseNetworkObject.globalFeatureNeurons[arrayIndexPropertiesActivationIndex]
	#print("1 globalFeatureNeuronsActivation = ", globalFeatureNeuronsActivation)
	globalFeatureNeuronsStrength = databaseNetworkObject.globalFeatureNeurons[arrayIndexPropertiesStrengthIndex]
	globalFeatureNeuronsTime = None
	if(inferenceUseNeuronFeaturePropertiesTime):
		globalFeatureNeuronsTime = databaseNetworkObject.globalFeatureNeurons[arrayIndexPropertiesTimeIndex]
	sequenceColumnIndex = None
	if(inferenceUseNeuronFeaturePropertiesTime):
		if(useSANIcolumns or useSANIfeaturesAndColumns):
			sequenceColumnIndex = GIAANNproto_predictionActivate.calculateSequenceColumnIndex(conceptMask, sequenceWordIndex)
	if(transformerUseInputConnections):
		globalFeatureConnectionsActivation = databaseNetworkObject.globalFeatureConnections[arrayIndexPropertiesActivationIndex]
	else:
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
	if((predictionEnsureConnectedToPreviousPrediction or enforceDirectConnectionsMinWordDistance) and (inferenceSeedNetwork and sequenceWordIndex > 0)):
		#limit prediction candidates to columns directly connected to previously predicted nodes
		connectedColumnsConstraint, connectedColumnsFeatureMap = GIAANNproto_predictionConstraints.buildConnectedColumnsLookupFromPrediction(databaseNetworkObject, observedColumnsDict, conceptColumnIndexTensor, conceptColumnFeatureIndexTensor)
	else:
		connectedColumnsConstraint = None
		connectedColumnsFeatureMap = None
	if((predictionEnsureConnectedToPreviousPrediction or enforceDirectConnectionsMinWordDistance) and (inferenceSeedNetwork and sequenceWordIndex > 0) and connectedColumnsConstraint is not None and connectedColumnsConstraint.numel() == 0):
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
			if(transformerUseInputConnections):
				globalFeatureConnectionsActivation = GIAANNproto_predictionActivate.decrementActivation(globalFeatureConnectionsActivation, activationDecrementPerPredictedToken)
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
			if(inferenceUseNeuronFeaturePropertiesTime):
				databaseNetworkObject.globalFeatureNeurons = GIAANNproto_sparseTensors.replaceAllSparseTensorElementsAtFirstDimIndex(databaseNetworkObject.globalFeatureNeurons, globalFeatureNeuronsTime, arrayIndexPropertiesTimeIndex)
		else:
			globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, globalFeatureNeuronsTime = GIAANNproto_predictionActivate.processFeaturesActivePredictSingle(databaseNetworkObject, globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, sequenceObservedColumnsPrediction, conceptColumnIndexActivation, conceptColumnFeatureIndexActivation, globalFeatureNeuronsTime, activationSequenceWordIndex, activationSequenceColumnIndex)
			if(inferenceUseNeuronFeaturePropertiesTime):
				databaseNetworkObject.globalFeatureNeurons = GIAANNproto_sparseTensors.replaceAllSparseTensorElementsAtFirstDimIndex(databaseNetworkObject.globalFeatureNeurons, globalFeatureNeuronsTime, arrayIndexPropertiesTimeIndex)
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
			databaseNetworkObject.globalFeatureNeurons = GIAANNproto_sparseTensors.replaceAllSparseTensorElementsAtFirstDimIndex(databaseNetworkObject.globalFeatureNeurons, globalFeatureNeuronsActivation, arrayIndexPropertiesActivationIndex)
			if(transformerUseInputConnections):
				databaseNetworkObject.globalFeatureConnections = GIAANNproto_sparseTensors.replaceAllSparseTensorElementsAtFirstDimIndex(databaseNetworkObject.globalFeatureConnections, globalFeatureConnectionsActivation, arrayIndexPropertiesActivationIndex)
			if(inferenceUseNeuronFeaturePropertiesTime):
				databaseNetworkObject.globalFeatureNeurons = GIAANNproto_sparseTensors.replaceAllSparseTensorElementsAtFirstDimIndex(databaseNetworkObject.globalFeatureNeurons, globalFeatureNeuronsTime, arrayIndexPropertiesTimeIndex)
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
		indicesToUpdateList = [arrayIndexPropertiesActivationIndex, branchIndex, arrayIndexSegmentLast, conceptColumnIndexSource, conceptColumnFeatureIndexSource]
		databaseNetworkObject.globalFeatureNeurons = GIAANNproto_sparseTensors.addElementValueToSparseTensor(databaseNetworkObject.globalFeatureNeurons, indicesToUpdateList, j1)
		globalFeatureNeuronsActivation = databaseNetworkObject.globalFeatureNeurons[arrayIndexPropertiesActivationIndex]
		conceptColumnIndexPred = conceptColumnIndexNext	#temporarily assign prediction from seed target for print only
		conceptColumnFeatureIndexPred = conceptColumnFeatureIndexNext	#temporarily assign prediction from seed target for print only
	else:	
		#predictionPhase;
		if(inferenceBeamSearch):
			conceptColumnIndexPred, conceptColumnFeatureIndexPred, targetPreviousColumnIndex, targetNextColumnIndex = GIAANNproto_predictionBeamSearch.beamSearchPredictNextFeature(sequenceObservedColumns, databaseNetworkObject, observedColumnsDict, globalFeatureNeuronsActivation, globalFeatureNeuronsStrength, globalFeatureConnectionsActivation, globalFeatureNeuronsTime, tokensSequence, wordPredictionIndex, sequenceWordIndex, conceptMask, allowedColumnsConstraint, constraintModePrediction, conceptActivationState, connectedColumnsConstraint, connectedColumnsFeatureMap)
			if(inferenceOnlyRetainPredictedTargetObservedColumn and not inferenceOnlyRetainPredictedTargetObservedColumnBeamSearch):
				if(observedColumnsDict is None):
					raise RuntimeError("processColumnInferencePrediction error: observedColumnsDict is None")
				observedColumnsDict.clear()
		else:
			conceptColumnIndexPred, conceptColumnFeatureIndexPred, targetPreviousColumnIndex, targetNextColumnIndex = GIAANNproto_predictionBeamSearch.beamSearchSelectSingleStepFeature(sequenceObservedColumns, databaseNetworkObject, observedColumnsDict, globalFeatureNeuronsActivation, globalFeatureNeuronsStrength, globalFeatureConnectionsActivation, globalFeatureNeuronsTime, tokensSequence, wordPredictionIndex, sequenceWordIndex, conceptMask, allowedColumnsConstraint, constraintModePrediction, conceptActivationState, connectedColumnsConstraint, connectedColumnsFeatureMap)
		conceptColumnIndexNext = conceptColumnIndexPred	#use prediction as next selected feature
		conceptColumnFeatureIndexNext = conceptColumnFeatureIndexPred	#use prediction as next selected feature
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

	#print prediction; 
	if(printPredictionsDuringInferencePredict):
		print("\t sequenceWordIndex = ", sequenceWordIndex, ", wordPredictionIndex = ", wordPredictionIndex, ", targetWord = ", targetWord, ", predictedWord = ", predictedWord, ", targetColumn = ", targetColumnName, ", predictedColumn = ", predictedColumnName)

	#load predicted feature observed column; 
	if(inferenceOnlyRetainPredictedTargetObservedColumn):
		if(conceptColumnIndexPred is None):
			raise RuntimeError("processColumnInferencePrediction error: predicted columns required for inferenceOnlyRetainPredictedTargetObservedColumn")
		loadObservedColumnInference(databaseNetworkObject, observedColumnsDict, int(conceptColumnIndexPred), sequenceWordIndex)
	
	#draw network; 
	if(drawNetworkDuringInferencePredict):
		#FUTURE: convert globalFeatureNeuronsActivation back to globalFeatureNeurons for draw
		GIAANNproto_databaseNetworkDrawExcitation.visualizeGraph(sequenceObservedColumnsPrediction, True, save=drawNetworkDuringInferenceSave, fileName=drawNetworkDuringInferenceSaveFilenamePrepend+generateDrawSequenceIndex(sequenceWordIndex))
	
	return featurePredictionTargetMatch, conceptColumnIndexNext, conceptColumnFeatureIndexNext, conceptActivationState


