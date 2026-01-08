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
if(inferencePredictiveNetwork):
	import GIAANNproto_predictionNetwork
import GIAANNproto_sequenceConcepts
import GIAANNproto_predictionActivate
import GIAANNproto_predictionConstraints
if(inferenceInhibitoryNeurons):
	import GIAANNproto_predictionInhibition

def inferenceSavePredictiveNetwork():
	GIAANNproto_predictionModel.saveModel(predictiveNetworkFolder, predictiveNetworkFileName)

def initialisePredictiveNetwork(databaseNetworkObject):
	GIAANNproto_predictionModel.nextWordPredictionModelCreate(databaseNetworkObject)
	
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
	if(conceptColumnsDelimitByPOS):
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
	else:
		return False

def activatedNodesAreProbabilisticReferenceSetDelimiters(databaseNetworkObject, conceptColumnsFeatureIndices):
	if(not conceptColumnsDelimitByPOS or not detectReferenceSetDelimitersBetweenNouns):
		return False
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

def getTargetWordForSequenceIndex(tokensSequence, sequenceWordIndex):
	if(tokensSequence is None):
		return "<unknown>"
	if(sequenceWordIndex < 0 or sequenceWordIndex >= len(tokensSequence)):
		return "<unknown>"
	return tokensSequence[sequenceWordIndex].word


def raisePredictionConnectivityError(sequenceWordIndex, wordPredictionIndex, tokensSequence, reason):
	targetWord = getTargetWordForSequenceIndex(tokensSequence, sequenceWordIndex)
	message = f"predictionEnsureConnectedToPreviousPrediction violation: {reason}. sequenceWordIndex={sequenceWordIndex}, wordPredictionIndex={wordPredictionIndex}, targetWord='{targetWord}'"
	raise RuntimeError(message)


def ensurePredictionStateAvailable(conceptColumnsIndices, conceptColumnsFeatureIndices, sequenceWordIndex, wordPredictionIndex, tokensSequence, reason):
	if(conceptColumnsIndices is None or conceptColumnsIndices.numel() == 0):
		raisePredictionConnectivityError(sequenceWordIndex, wordPredictionIndex, tokensSequence, reason + " (missing concept columns)")
	if(conceptColumnsFeatureIndices is None or conceptColumnsFeatureIndices.numel() == 0):
		raisePredictionConnectivityError(sequenceWordIndex, wordPredictionIndex, tokensSequence, reason + " (missing column features)")


def initialiseConceptActivationState(conceptColumnsIndices, conceptColumnsFeatureIndices):
	if(not predictionColumnsMustActivateConceptFeature):
		return None
	initialState = set()
	return updateConceptActivationState(initialState, conceptColumnsIndices, conceptColumnsFeatureIndices)


def cloneConceptActivationState(conceptActivationState):
	if(conceptActivationState is None):
		return None
	return set(conceptActivationState)


def updateConceptActivationState(conceptActivationState, conceptColumnsIndices, conceptColumnsFeatureIndices):
	if(not predictionColumnsMustActivateConceptFeature):
		return conceptActivationState
	if(conceptActivationState is None):
		conceptActivationState = set()
	if(conceptColumnsIndices is None or conceptColumnsFeatureIndices is None):
		return conceptActivationState
	if(conceptColumnsIndices.numel() == 0 or conceptColumnsFeatureIndices.numel() == 0):
		return conceptActivationState
	indicesCPU = conceptColumnsIndices.detach().cpu().tolist()
	featureIndices = conceptColumnsFeatureIndices
	originally1D = False
	if(featureIndices.dim() == 1):
		featureIndices = featureIndices.unsqueeze(1)
		originally1D = True
	for rowIndex, columnValue in enumerate(indicesCPU):
		rowTensor = featureIndices[rowIndex]
		if(rowTensor.numel() == 0):
			continue
		rowValues = rowTensor.detach().cpu().view(-1).tolist()
		if(len(rowValues) > 0):
			conceptActivationState.add(int(columnValue))
	return conceptActivationState


def enforceConceptFeaturePredictionOrder(conceptColumnsIndices, conceptColumnsFeatureIndices, conceptActivationState):
	if(not predictionColumnsMustActivateConceptFeature):
		return conceptColumnsIndices, conceptColumnsFeatureIndices
	if(conceptColumnsIndices is None or conceptColumnsFeatureIndices is None):
		return conceptColumnsIndices, conceptColumnsFeatureIndices
	if(conceptColumnsIndices.numel() == 0 or conceptColumnsFeatureIndices.numel() == 0):
		return conceptColumnsIndices, conceptColumnsFeatureIndices
	localState = conceptActivationState or set()
	featureIndices = conceptColumnsFeatureIndices
	originally1D = False
	if(featureIndices.dim() == 1):
		featureIndices = featureIndices.unsqueeze(1)
		originally1D = True
	for rowIndex in range(featureIndices.shape[0]):
		columnIndex = int(conceptColumnsIndices[rowIndex].item())
		rowTensor = featureIndices[rowIndex]
		if(columnIndex not in localState and rowTensor.numel() > 0):
			localState.add(columnIndex)
	return conceptColumnsIndices, conceptColumnsFeatureIndices

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
	
	if(transformerUseInputConnections):
		GIAANNproto_databaseNetworkExcitation.generateGlobalFeatureConnections(sequenceObservedColumns.databaseNetworkObject)
		
	numPredictionTokens = len(sequencePredict)	#set numPredictionTokens (dynamic)
	
	if(inferencePredictiveNetwork and not inferenceTrainPredictiveNetworkAllSequences):
		initialisePredictiveNetwork(sequenceObservedColumns.databaseNetworkObject)
			
	#identify first activated column(s) in seed phase:
	#TODO verify this;
	if(inferencePredictiveNetwork):
		kcMax = kcNetwork
	else:
		kcMax = 1	#not used
	initialContextWordIndex = 0		#use first seed token as the context for the first prediction
	initialContextWordIndex = max(0, min(initialContextWordIndex, len(tokensSequence)-1))
	multipleSources, targetPreviousColumnIndex, targetNextColumnIndex, targetFeatureIndex, conceptColumnsIndices, conceptColumnsFeatureIndices = GIAANNproto_databaseNetworkExcitation.getTokenConceptFeatureIndexTensor(sequenceObservedColumns, tokensSequence, conceptMask, initialContextWordIndex, kcMax)
	conceptActivationState = initialiseConceptActivationState(conceptColumnsIndices, conceptColumnsFeatureIndices)
	observedColumnsDict = sequenceObservedColumns.observedColumnsDict  # key: lemma, value: ObservedColumn	#every observed column in inference (seed and prediction phases)

	#seed first tokens;
	#TODO - execute processColumnInferencePrediction(seedPhase=True with inferenceUseNextTokenPredictionsOrTargetsToActivateNextColumnFeatures always overridden to False during the seed phase
	for wordSeedIndex in range(numSeedTokens):
		sequenceWordIndex = wordSeedIndex
		wordPredictionIndex = wordSeedIndex
		featurePredictionTargetMatch, conceptColumnsIndices, conceptColumnsFeatureIndices, multipleSources, conceptActivationState = processColumnInferencePrediction(sequenceObservedColumns, sequenceIndex, observedColumnsDict, wordPredictionIndex, sequenceWordIndex, tokensSequence, conceptColumnsIndices, conceptColumnsFeatureIndices, conceptMask, multipleSources, conceptActivationState, seedPhase=True)

	#predict next tokens;
	for wordPredictionIndex in range(numPredictionTokens):
		sequenceWordIndex = numSeedTokens + wordPredictionIndex
		featurePredictionTargetMatch, conceptColumnsIndices, conceptColumnsFeatureIndices, multipleSources, conceptActivationState = processColumnInferencePrediction(sequenceObservedColumns, sequenceIndex, observedColumnsDict, wordPredictionIndex, sequenceWordIndex, tokensSequence, conceptColumnsIndices, conceptColumnsFeatureIndices, conceptMask, multipleSources, conceptActivationState)
		if(not featurePredictionTargetMatch):
			print("warning: featurePredictionTargetMatch=False")
			if(debugTerminateInferenceOnPredictionTargetMismatch):
				print("debugTerminateInferenceOnPredictionTargetMismatch: prematurely terminating inference")
				break
		

def processColumnInferencePrediction(sequenceObservedColumns, sequenceIndex, observedColumnsDict, wordPredictionIndex, sequenceWordIndex, tokensSequence, conceptColumnsIndices, conceptColumnsFeatureIndices, conceptMask, multipleSources, conceptActivationState, seedPhase=False):
	
	databaseNetworkObject = sequenceObservedColumns.databaseNetworkObject
	conceptColumnsFeatureIndicesActivation = conceptColumnsFeatureIndices
	
	if(predictionEnsureConnectedToPreviousPrediction or enforceDirectConnectionsMinWordDistance):
		ensurePredictionStateAvailable(conceptColumnsIndices, conceptColumnsFeatureIndices, sequenceWordIndex, wordPredictionIndex, tokensSequence, "no connected context available before prediction")

	#burst the initial seed in the sequence
	if(sequenceWordIndex==0 or inferenceBurstAllPredictionsOrTargetsInSequence):
		#activate source token (incremental seed during train)
			#if(wordPredictionIndex == 1) will reactivate first seed token column feature (as it was not saved during wordPredictionIndex==0)
		for conceptIndex in range(conceptColumnsIndices.shape[0]):
			conceptColumnsIndicesSource = conceptColumnsIndices[conceptIndex].item()
			conceptColumnsFeatureIndicesSource = conceptColumnsFeatureIndices[conceptIndex].squeeze().item()
			branchIndex = 0
			indicesToUpdateList = [arrayIndexPropertiesActivationIndex, branchIndex, arrayIndexSegmentLast, conceptColumnsIndicesSource, conceptColumnsFeatureIndicesSource]
			databaseNetworkObject.globalFeatureNeurons = GIAANNproto_sparseTensors.addElementValueToSparseTensor(databaseNetworkObject.globalFeatureNeurons, indicesToUpdateList, j1)
				
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

	allowedColumnsConstraint = None
	constraintModePrediction = None
	probabilisticDelimiterActive = False
	if(conceptColumnsDelimitByPOS):
		if(conceptColumnsFeatureIndices is not None and conceptColumnsFeatureIndices.numel() > 0):
			probabilisticDelimiterActive = activatedNodesAreProbabilisticReferenceSetDelimiters(databaseNetworkObject, conceptColumnsFeatureIndices)
		allowedColumnsConstraint = buildAllowedColumnsLookup(conceptColumnsIndices, databaseNetworkObject.c)
		if(allowedColumnsConstraint is not None and allowedColumnsConstraint.numel() > 0):
			if(conceptColumnsFeatureIndices is None or conceptColumnsFeatureIndices.numel() == 0):
				constraintModePrediction = "internal"
			else:
				isDelimiterNode = activatedNodesAreReferenceSetDelimiters(databaseNetworkObject, conceptColumnsFeatureIndices)
				if(isDelimiterNode):
					constraintModePrediction = "delimiter"
				else:
					if(probabilisticDelimiterActive):
						constraintModePrediction = None
					else:
						constraintModePrediction = "internal"
	if(seedPhase and constraintModePrediction == "delimiter"):
		constraintModePrediction = None
	if((predictionEnsureConnectedToPreviousPrediction or enforceDirectConnectionsMinWordDistance) and (inferenceSeedNetwork and sequenceWordIndex > 0)):
		#limit prediction candidates to columns directly connected to previously predicted nodes
		connectedColumnsConstraint, connectedColumnsFeatureMap = GIAANNproto_predictionConstraints.buildConnectedColumnsLookupFromPrediction(databaseNetworkObject, observedColumnsDict, conceptColumnsIndices, conceptColumnsFeatureIndices)
	else:
		connectedColumnsConstraint = None
		connectedColumnsFeatureMap = None
	if((predictionEnsureConnectedToPreviousPrediction or enforceDirectConnectionsMinWordDistance) and (inferenceSeedNetwork and sequenceWordIndex > 0) and connectedColumnsConstraint is not None and connectedColumnsConstraint.numel() == 0):
		raisePredictionConnectivityError(sequenceWordIndex, wordPredictionIndex, tokensSequence, "previous prediction has no outgoing connections")
		
	if(sequenceWordIndex > 0):
		if(predictionColumnsMustActivateConceptFeature):
			_, conceptColumnsFeatureIndicesActivation = enforceConceptFeaturePredictionOrder(conceptColumnsIndices, conceptColumnsFeatureIndices, conceptActivationState)
		else:
			conceptColumnsFeatureIndicesActivation = conceptColumnsFeatureIndices
		# Refresh the observed columns dictionary for each new sequence
		observedColumnsSequenceCandidateIndexDict = {}  # key: sequence candidate index, value: ObservedColumn	#used to populate sequence feature connection arrays based on observed columns (i does not correspond to sequence word index as assumed by observedColumnsSequenceWordIndexDict)

		#populate sequence observed columns;
		words = []
		lemmas = []
		conceptColumnsIndicesList = conceptColumnsIndices.tolist()
		for i, conceptIndexVal in enumerate(conceptColumnsIndicesList):
			lemma = databaseNetworkObject.conceptColumnsList[conceptIndexVal]
			word = lemma	#same for concepts (not used)
			lemmas.append(lemma)
			words.append(word)
			# Load observed column from disk or create new one
			observedColumn = GIAANNproto_databaseNetworkExcitation.loadOrCreateObservedColumn(databaseNetworkObject, conceptIndexVal, lemma, sequenceWordIndex)
			observedColumnsDict[lemma] = observedColumn
			observedColumnsSequenceCandidateIndexDict[i] = observedColumn
		sequenceObservedColumnsPrediction = SequenceObservedColumnsInferencePrediction(databaseNetworkObject, observedColumnsDict, observedColumnsSequenceCandidateIndexDict)
		
		#decrement activations;
		if(inferenceDecrementActivations):
			#decrement activation after each prediction interval
			globalFeatureNeuronsActivation = GIAANNproto_predictionActivate.decrementActivation(globalFeatureNeuronsActivation, activationDecrementPerPredictedToken)
			if(transformerUseInputConnections):
				globalFeatureConnectionsActivation = GIAANNproto_predictionActivate.decrementActivation(globalFeatureConnectionsActivation, activationDecrementPerPredictedToken)
			#if(inferenceUseNeuronFeaturePropertiesTime):	#OLD
			#	globalFeatureNeuronsTime = GIAANNproto_predictionActivate.decrementActivation(globalFeatureNeuronsTime, inferenceUseNeuronFeaturePropertiesTimeDecrement)

		#process features (activate global target neurons);
		activationSequenceWordIndex = sequenceWordIndex
		activationSequenceColumnIndex = sequenceColumnIndex
		if(inferenceUseNeuronFeaturePropertiesTime and inferenceUseNextTokenPredictionsOrTargetsToActivateNextColumnFeatures and sequenceWordIndex > 0):
			activationSequenceWordIndex = sequenceWordIndex - 1
			if(activationSequenceWordIndex < 0):
				raise RuntimeError("processColumnInferencePrediction: activationSequenceWordIndex out of range")
			if(useSANIcolumns or useSANIfeaturesAndColumns):
				activationSequenceColumnIndex = GIAANNproto_predictionActivate.calculateSequenceColumnIndex(conceptMask, activationSequenceWordIndex)
		if(multipleSources):
			globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, globalFeatureNeuronsTime = GIAANNproto_predictionActivate.processFeaturesActivePredictMulti(databaseNetworkObject, globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, sequenceObservedColumnsPrediction, conceptColumnsIndices, conceptColumnsFeatureIndicesActivation, globalFeatureNeuronsTime, activationSequenceWordIndex, activationSequenceColumnIndex)
		else:
			globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, globalFeatureNeuronsTime = GIAANNproto_predictionActivate.processFeaturesActivePredictSingle(databaseNetworkObject, globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, sequenceObservedColumnsPrediction, conceptColumnsIndices, conceptColumnsFeatureIndicesActivation, globalFeatureNeuronsTime, activationSequenceWordIndex, activationSequenceColumnIndex)
		if(inferenceUseNeuronFeaturePropertiesTime):
			databaseNetworkObject.globalFeatureNeurons = GIAANNproto_sparseTensors.replaceAllSparseTensorElementsAtFirstDimIndex(databaseNetworkObject.globalFeatureNeurons, globalFeatureNeuronsTime, arrayIndexPropertiesTimeIndex)
		if(debugSANIfeaturesAndColumns and useSANIfeaturesAndColumns):
			if(globalFeatureNeuronsActivation.is_sparse):
				featureNeuronsActivationDense = globalFeatureNeuronsActivation.to_dense()
			else:
				featureNeuronsActivationDense = globalFeatureNeuronsActivation
			conceptIndexLookup = sequenceObservedColumns.conceptIndicesInSequenceObservedTensor.to(featureNeuronsActivationDense.device)
			featureIndexLookup = sequenceObservedColumns.featureIndicesInObservedTensor.to(featureNeuronsActivationDense.device)
			if(conceptIndexLookup.numel() == 0 or featureIndexLookup.numel() == 0):
				segmentFeatureActivations = [[] for _ in range(arrayNumberOfSegments)]
			else:
				featureNeuronsActivationDense = featureNeuronsActivationDense.index_select(2, conceptIndexLookup)
				featureNeuronsActivationDense = featureNeuronsActivationDense.index_select(3, featureIndexLookup)
				segmentFeatureActivations = featureNeuronsActivationDense.sum(dim=(0, 2)).to("cpu").tolist()
			print("\tdebugSANIfeaturesAndColumns: predict segmentFeatureActivations={0}".format(segmentFeatureActivations))

	else:
		#activation targets have already been activated
		sequenceObservedColumnsPrediction = SequenceObservedColumnsDraw(databaseNetworkObject, observedColumnsDict)

	if(inferenceDeactivateNeuronsUponPrediction or inferenceInvertNeuronActivationUponPrediction):
		indicesToUpdateList = []
		for conceptIndex in range(conceptColumnsIndices.shape[0]):
			conceptColumnsIndicesSource = conceptColumnsIndices[conceptIndex]
			conceptColumnsFeatureIndicesSource = conceptColumnsFeatureIndicesActivation[conceptIndex].squeeze(dim=0)
			branchIndex = 0
			if(multipleDendriticBranches):
				branchIndex = GIAANNproto_predictionActivate.selectActivatedBranchIndex(globalFeatureNeuronsActivation, int(conceptColumnsIndicesSource.item()), int(conceptColumnsFeatureIndicesSource.item()))
			branchTensor = pt.tensor(branchIndex, device=conceptColumnsIndicesSource.device)
			if(useSANI):
				for segmentIndex in range(arrayNumberOfSegments):
					indexToUpdate = pt.stack([branchTensor, pt.tensor(segmentIndex, device=conceptColumnsIndicesSource.device), conceptColumnsIndicesSource, conceptColumnsFeatureIndicesSource], dim=0)
					indicesToUpdateList.append(indexToUpdate)
			else:
				indicesToUpdate = pt.stack([branchTensor, pt.tensor(arrayIndexSegmentFirst, device=conceptColumnsIndicesSource.device), conceptColumnsIndicesSource, conceptColumnsFeatureIndicesSource], dim=0)
				indicesToUpdateList.append(indicesToUpdate)
		indicesToUpdate = pt.stack(indicesToUpdateList, dim=0)
		if(inferenceDeactivateNeuronsUponPrediction or inferenceInvertNeuronActivationUponPrediction):
			if(inferenceDeactivateNeuronsUponPrediction):
				modifier = 0
			elif(inferenceInvertNeuronActivationUponPrediction):
				modifier = inferenceInvertNeuronActivationUponPredictionLevel
			globalFeatureNeuronsActivationOrig = globalFeatureNeuronsActivation
			globalFeatureNeuronsActivation = GIAANNproto_sparseTensors.modifySparseTensor(globalFeatureNeuronsActivation, indicesToUpdate, modifier, multiply=inferenceInvertNeuronActivationUponPrediction)
			databaseNetworkObject.globalFeatureNeurons = GIAANNproto_sparseTensors.replaceAllSparseTensorElementsAtFirstDimIndex(databaseNetworkObject.globalFeatureNeurons, globalFeatureNeuronsActivation, arrayIndexPropertiesActivationIndex)
			if(transformerUseInputConnections):
				databaseNetworkObject.globalFeatureConnections = GIAANNproto_sparseTensors.replaceAllSparseTensorElementsAtFirstDimIndex(databaseNetworkObject.globalFeatureConnections, globalFeatureConnectionsActivation, arrayIndexPropertiesActivationIndex)
			if(inferenceUseNeuronFeaturePropertiesTime):
				databaseNetworkObject.globalFeatureNeurons = GIAANNproto_sparseTensors.replaceAllSparseTensorElementsAtFirstDimIndex(databaseNetworkObject.globalFeatureNeurons, globalFeatureNeuronsTime, arrayIndexPropertiesTimeIndex)
			if(predictionColumnsMustActivateConceptFeature):
				conceptActivationState = updateConceptActivationState(conceptActivationState, conceptColumnsIndices, conceptColumnsFeatureIndicesActivation)
	
	if(debugInferencePredictionActivationAccumulation):
		globalFeatureNeuronsTemp = databaseNetworkObject.globalFeatureNeurons.to_dense()
		print("globalFeatureNeuronsTemp = ", globalFeatureNeuronsTemp)

	if(seedPhase):
		targetMultipleSources, targetPreviousColumnIndex, targetNextColumnIndex, targetFeatureIndex, conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext = GIAANNproto_databaseNetworkExcitation.getTokenConceptFeatureIndexTensor(sequenceObservedColumns, tokensSequence, conceptMask, sequenceWordIndex, kcNetwork)
		multipleSourcesNext = targetMultipleSources
		if(multipleSourcesNext):
			kc = 2
		else:
			kc = 1
		assert kf==1
		conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext = GIAANNproto_predictionConstraints.applyColumnConstraintToPredictions(databaseNetworkObject, conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext, allowedColumnsConstraint, constraintModePrediction, connectedColumnsConstraint, connectedColumnsFeatureMap)
		conceptColumnsIndicesPred = conceptColumnsIndicesNext
		conceptColumnsFeatureIndicesPred = conceptColumnsFeatureIndicesNext
		if(conceptColumnsIndicesNext is None or conceptColumnsIndicesNext.numel() == 0):
			multipleSourcesNext = False
			kc = 0
		else:
			kc = conceptColumnsIndicesNext.shape[0]
		if((predictionEnsureConnectedToPreviousPrediction or enforceDirectConnectionsMinWordDistance) and connectedColumnsConstraint is not None):
			if(conceptColumnsIndicesPred is None or conceptColumnsIndicesPred.numel() == 0):
				raisePredictionConnectivityError(sequenceWordIndex, wordPredictionIndex, tokensSequence, "no connected predictions available for current step")
			if(conceptColumnsIndicesNext is None or conceptColumnsIndicesNext.numel() == 0):
				raisePredictionConnectivityError(sequenceWordIndex, wordPredictionIndex, tokensSequence, "no connected activations available for next step")
		if(conceptColumnsIndicesNext is not None and conceptColumnsFeatureIndicesNext is not None and conceptColumnsIndicesNext.numel() > 0 and conceptColumnsFeatureIndicesNext.numel() > 0):
			for conceptIndex in range(conceptColumnsIndicesNext.shape[0]):
				conceptColumnsIndicesSource = conceptColumnsIndicesNext[conceptIndex].item()
				conceptColumnsFeatureIndicesSource = conceptColumnsFeatureIndicesNext[conceptIndex].squeeze().item()
				branchIndex = 0
				indicesToUpdateList = [arrayIndexPropertiesActivationIndex, branchIndex, arrayIndexSegmentLast, conceptColumnsIndicesSource, conceptColumnsFeatureIndicesSource]
				databaseNetworkObject.globalFeatureNeurons = GIAANNproto_sparseTensors.addElementValueToSparseTensor(databaseNetworkObject.globalFeatureNeurons, indicesToUpdateList, j1)
			globalFeatureNeuronsActivation = databaseNetworkObject.globalFeatureNeurons[arrayIndexPropertiesActivationIndex]
	else:
		if(inferenceBeamSearch):
			conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext, multipleSourcesNext, kc, conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, targetMultipleSources, targetPreviousColumnIndex, targetNextColumnIndex = GIAANNproto_predictionBeamSearch.beamSearchPredictNextFeature(sequenceObservedColumns, databaseNetworkObject, observedColumnsDict, globalFeatureNeuronsActivation, globalFeatureNeuronsStrength, globalFeatureConnectionsActivation, globalFeatureNeuronsTime, tokensSequence, wordPredictionIndex, sequenceWordIndex, conceptMask, allowedColumnsConstraint, constraintModePrediction, conceptActivationState, connectedColumnsConstraint, connectedColumnsFeatureMap)
		else:
			if(inferencePredictiveNetwork):
				conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext, multipleSourcesNext, kc, conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, targetMultipleSources, targetPreviousColumnIndex, targetNextColumnIndex = GIAANNproto_predictionNetwork.predictMostActiveFeature(sequenceObservedColumns, databaseNetworkObject, tokensSequence, wordPredictionIndex, sequenceWordIndex, conceptMask, allowedColumnsConstraint, constraintModePrediction, conceptActivationState, connectedColumnsConstraint, connectedColumnsFeatureMap)	
			else:
				conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext, multipleSourcesNext, kc, conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, targetMultipleSources, targetPreviousColumnIndex, targetNextColumnIndex = GIAANNproto_predictionBeamSearch.beamSearchSelectSingleStepFeature(sequenceObservedColumns, databaseNetworkObject, observedColumnsDict, globalFeatureNeuronsActivation, globalFeatureNeuronsStrength, globalFeatureConnectionsActivation, globalFeatureNeuronsTime, tokensSequence, wordPredictionIndex, sequenceWordIndex, conceptMask, allowedColumnsConstraint, constraintModePrediction, conceptActivationState, connectedColumnsConstraint, connectedColumnsFeatureMap)

			if(inferencePredictiveNetwork):
				if((predictionEnsureConnectedToPreviousPrediction or enforceDirectConnectionsMinWordDistance) and connectedColumnsConstraint is not None):
					conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred = applyConnectedColumnsConstraint(conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, connectedColumnsConstraint, connectedColumnsFeatureMap)
					conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext = applyConnectedColumnsConstraint(conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext, connectedColumnsConstraint, connectedColumnsFeatureMap)
					if(conceptColumnsIndicesPred is None or conceptColumnsIndicesPred.numel() == 0):
						raisePredictionConnectivityError(sequenceWordIndex, wordPredictionIndex, tokensSequence, "no connected predictions available for current step")
					if(conceptColumnsIndicesNext is None or conceptColumnsIndicesNext.numel() == 0):
						raisePredictionConnectivityError(sequenceWordIndex, wordPredictionIndex, tokensSequence, "no connected activations available for next step")

	if(conceptColumnsIndicesPred is None or conceptColumnsIndicesPred.numel() == 0):
		raise RuntimeError("processColumnInferencePrediction: no prediction candidates available; sequenceWordIndex={0}, wordPredictionIndex={1}, targetWord='{2}'".format(sequenceWordIndex, wordPredictionIndex, getTargetWordForSequenceIndex(tokensSequence, sequenceWordIndex)))

	featurePredictionTargetMatch = False
	if(True):
		targetToken = tokensSequence[sequenceWordIndex]
		targetWord = targetToken.word
		targetLemma = targetToken.lemma
		targetIsConceptFeature = bool(conceptMask[sequenceWordIndex].item())
		#compare topk column/feature predictions to sequencePredict (target words);
		#implementation limitation; only works with kf = 1;
		for columnPredictionIndex in range(conceptColumnsIndicesPred.shape[0]):
			columnIndex = conceptColumnsIndicesPred[columnPredictionIndex]
			columnName = databaseNetworkObject.conceptColumnsList[columnIndex]
			observedColumnFeatureIndex = conceptColumnsFeatureIndicesPred[columnPredictionIndex, 0]
			predictedIsConceptFeature = (observedColumnFeatureIndex == featureIndexConceptNeuron)
			if(predictedIsConceptFeature):
				predictedWord = columnName
			else:
				predictedWord = databaseNetworkObject.conceptFeaturesList[observedColumnFeatureIndex]
			predictedColumnName = columnName
			if(targetMultipleSources):
				targetColumnName = databaseNetworkObject.conceptColumnsList[targetPreviousColumnIndex] + "/" + databaseNetworkObject.conceptColumnsList[targetNextColumnIndex]
			else:
				targetColumnName = databaseNetworkObject.conceptColumnsList[targetPreviousColumnIndex]
			if(printPredictionsDuringInferencePredict):
				print("\t sequenceWordIndex = ", sequenceWordIndex, ", wordPredictionIndex = ", wordPredictionIndex, ", targetWord = ", targetWord, ", predictedWord = ", predictedWord, ", targetColumn = ", targetColumnName, ", predictedColumn = ", predictedColumnName)
			if(targetWord == predictedWord):
				featurePredictionTargetMatch = True
			elif(targetIsConceptFeature and predictedIsConceptFeature and targetLemma == predictedColumnName and targetColumnName == predictedColumnName):
				featurePredictionTargetMatch = True
	if(inferenceInhibitoryNeurons):
		storeInhibitoryActivations(databaseNetworkObject, globalFeatureNeuronsActivation, conceptColumnsIndices, conceptColumnsFeatureIndicesActivation, conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext)
	if(drawNetworkDuringInferencePredict):
		#FUTURE: convert globalFeatureNeuronsActivation back to globalFeatureNeurons for draw
		GIAANNproto_databaseNetworkDrawExcitation.visualizeGraph(sequenceObservedColumnsPrediction, True, save=drawNetworkDuringInferenceSave, fileName=drawNetworkDuringInferenceSaveFilenamePrepend+generateDrawSequenceIndex(sequenceWordIndex))
	return featurePredictionTargetMatch, conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext, multipleSourcesNext, conceptActivationState
