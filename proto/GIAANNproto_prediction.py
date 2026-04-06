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
import math

from GIAANNproto_globalDefs import *
import GIAANNproto_debug
import GIAANNproto_databaseNetwork
import GIAANNproto_databaseNetworkTrain
import GIAANNproto_databaseNetworkDraw
import GIAANNproto_sparseTensors
import GIAANNproto_sequenceTokens
import GIAANNproto_predictionBeamSearch
import GIAANNproto_sequenceConcepts
import GIAANNproto_predictionActivate
import GIAANNproto_predictionConstraints

totalInferenceTop1Matches = 0
totalInferenceTop1Tokens = 0
totalInferenceTop1PredictionMatches = 0
totalInferenceTop1PredictionTokens = 0
totalInferenceTop1NegativeLogProbabilitySum = 0.0
totalInferenceTop1BitsPerByteTokens = 0
totalInferenceTop1BitsPerByteBytes = 0
totalInferenceTop1ProbabilitySumModified = 0.0

def resetInferenceTop1AccuracyCounts():
	global totalInferenceTop1Matches
	global totalInferenceTop1Tokens
	global totalInferenceTop1PredictionMatches
	global totalInferenceTop1PredictionTokens
	global totalInferenceTop1NegativeLogProbabilitySum
	global totalInferenceTop1BitsPerByteTokens
	global totalInferenceTop1BitsPerByteBytes
	global totalInferenceTop1ProbabilitySumModified
	totalInferenceTop1Matches = 0
	totalInferenceTop1Tokens = 0
	totalInferenceTop1PredictionMatches = 0
	totalInferenceTop1PredictionTokens = 0
	totalInferenceTop1NegativeLogProbabilitySum = 0.0
	totalInferenceTop1BitsPerByteTokens = 0
	totalInferenceTop1BitsPerByteBytes = 0
	totalInferenceTop1ProbabilitySumModified = 0.0
	return

def addInferenceTop1AccuracyCount(featurePredictionTargetMatch, seedPhase):
	if(printInferenceTop1Accuracy):
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

def addInferenceTop1AccuracyBitsPerByteProbability(targetProbability):
	if(printInferenceTop1AccuracyBitsPerByte):
		if(targetProbability is None):
			raise RuntimeError("addInferenceTop1AccuracyBitsPerByteProbability error: targetProbability is None")
		if(targetProbability < 0.0 or targetProbability > 1.0):
			raise RuntimeError("addInferenceTop1AccuracyBitsPerByteProbability error: targetProbability must be within [0, 1]")
		global totalInferenceTop1NegativeLogProbabilitySum
		global totalInferenceTop1BitsPerByteTokens
		global totalInferenceTop1ProbabilitySumModified
		totalInferenceTop1BitsPerByteTokens += 1
		totalInferenceTop1ProbabilitySumModified += targetProbability
		if(targetProbability == 0.0):
			totalInferenceTop1NegativeLogProbabilitySum = math.inf
		elif(not math.isinf(totalInferenceTop1NegativeLogProbabilitySum)):
			totalInferenceTop1NegativeLogProbabilitySum += -math.log(targetProbability)
	return

def addInferenceTop1AccuracyBitsPerByteBytes(sequenceRaw):
	if(printInferenceTop1AccuracyBitsPerByte):
		if(sequenceRaw is None):
			raise RuntimeError("addInferenceTop1AccuracyBitsPerByteBytes error: sequenceRaw is None")
		sequenceBytes = len(sequenceRaw.encode("utf-8"))
		if(sequenceBytes <= 0):
			raise RuntimeError("addInferenceTop1AccuracyBitsPerByteBytes error: sequenceBytes must be > 0")
		global totalInferenceTop1BitsPerByteBytes
		totalInferenceTop1BitsPerByteBytes += sequenceBytes
	return

def printInferenceTop1Accuracy(databaseNetworkObject):
	if(printInferenceTop1Accuracy):
		if(printInferenceTop1AccuracyBitsPerByte):
			if(totalInferenceTop1BitsPerByteTokens <= 0):
				if(printInferenceTop1AccuracyBitsPerByteModified):
					print("printInferenceTop1AccuracyBitsPerByteModified: no inference tokens recorded; skipping modified BPB")
				else:
					print("printInferenceTop1AccuracyBitsPerByte: no inference tokens recorded; skipping BPB")
			elif(totalInferenceTop1BitsPerByteBytes <= 0):
				if(printInferenceTop1AccuracyBitsPerByteModified):
					raise RuntimeError("printInferenceTop1AccuracyBitsPerByteModified error: totalInferenceTop1BitsPerByteBytes must be > 0")
				else:
					raise RuntimeError("printInferenceTop1AccuracyBitsPerByte error: totalInferenceTop1BitsPerByteBytes must be > 0")
			else:
				if(printInferenceTop1AccuracyBitsPerByteModified):
					averageProbabilityModified = totalInferenceTop1ProbabilitySumModified / totalInferenceTop1BitsPerByteTokens
					if(averageProbabilityModified < 0.0 or averageProbabilityModified > 1.0):
						raise RuntimeError("printInferenceTop1AccuracyBitsPerByteModified error: averageProbabilityModified must be within [0, 1]")
					if(averageProbabilityModified == 0.0):
						valLoss = math.inf
						bitsPerByte = math.inf
					else:
						valLoss = -math.log(averageProbabilityModified)
						bitsPerByte = (valLoss / math.log(2.0)) * (totalInferenceTop1BitsPerByteTokens / totalInferenceTop1BitsPerByteBytes)
				else:
					if(math.isinf(totalInferenceTop1NegativeLogProbabilitySum)):
						valLoss = math.inf
						bitsPerByte = math.inf
					else:
						valLoss = totalInferenceTop1NegativeLogProbabilitySum / totalInferenceTop1BitsPerByteTokens
						bitsPerByte = (valLoss / math.log(2.0)) * (totalInferenceTop1BitsPerByteTokens / totalInferenceTop1BitsPerByteBytes)
				if(useAutoresearch):
					print("---")
					if(printInferenceTop1AccuracyBitsPerByteModified):
						print("averageTop1BitsPerByteModified: ", bitsPerByte)
					else:
						print("averageTop1BitsPerByte: ", bitsPerByte)
					memory_gb = GIAANNproto_databaseNetwork.printCountTotalParametersRun(databaseNetworkObject)
					print("memory_gb: ", memory_gb)
				else:
					if(printInferenceTop1AccuracyBitsPerByteModified):
						print("averageTop1BitsPerByteModified: bitsPerByte = ", bitsPerByte, ", valLoss = ", valLoss, ", averageProbability = ", averageProbabilityModified, ", inferenceTokens = ", totalInferenceTop1BitsPerByteTokens, ", inferenceBytes = ", totalInferenceTop1BitsPerByteBytes)
					else:
						print("averageTop1BitsPerByte: bitsPerByte = ", bitsPerByte, ", valLoss = ", valLoss, ", inferenceTokens = ", totalInferenceTop1BitsPerByteTokens, ", inferenceBytes = ", totalInferenceTop1BitsPerByteBytes)
		elif(totalInferenceTop1Tokens <= 0 or totalInferenceTop1PredictionTokens <= 0):
			print("printInferenceTop1Accuracy: no prediction tokens recorded; skipping accuracy")
		else:
			predictionAccuracy = totalInferenceTop1PredictionMatches / totalInferenceTop1PredictionTokens
			inferenceAccuracy = totalInferenceTop1Matches / totalInferenceTop1Tokens
			if(useAutoresearch):
				print("---")
				print("averageTop1Accuracy: ", predictionAccuracy)
				memory_gb = GIAANNproto_databaseNetwork.printCountTotalParametersRun(databaseNetworkObject)
				print("memory_gb: ", memory_gb)
			else:
				if(inferenceReportTokenAccuracyConstrainByColumn):
					print("averageTop1Accuracy (col): predictionTokens = ", predictionAccuracy, ", inferenceTokens = ", inferenceAccuracy)
				else:
					print("averageTop1Accuracy: predictionTokens = ", predictionAccuracy, ", inferenceTokens = ", inferenceAccuracy)
	return

def addInferenceTop1AccuracyCountPadding(numSeedTokens, numPredictionTokens, seedTokensProcessed, predictionTokensProcessed):
	if(printInferenceTop1Accuracy):
		if(numSeedTokens is None or numPredictionTokens is None or seedTokensProcessed is None or predictionTokensProcessed is None):
			raise RuntimeError("addInferenceTop1AccuracyCountPadding error: token counts are None")
		if(numSeedTokens < 0 or numPredictionTokens < 0 or seedTokensProcessed < 0 or predictionTokensProcessed < 0):
			raise RuntimeError("addInferenceTop1AccuracyCountPadding error: token counts must be non-negative")
		remainingSeedTokens = int(numSeedTokens) - int(seedTokensProcessed)
		remainingPredictionTokens = int(numPredictionTokens) - int(predictionTokensProcessed)
		if(remainingSeedTokens < 0 or remainingPredictionTokens < 0):
			raise RuntimeError("addInferenceTop1AccuracyCountPadding error: processed token counts exceed expected token counts")
		for remainingSeedTokenIndex in range(remainingSeedTokens):
			addInferenceTop1AccuracyCount(False, True)
		for remainingPredictionTokenIndex in range(remainingPredictionTokens):
			addInferenceTop1AccuracyCount(False, False)
	return

def getInferenceTargetWord(tokensSequence, conceptMask, sequenceWordIndex):
	targetWord = None
	targetToken = tokensSequence[sequenceWordIndex]
	targetIsConceptFeature = bool(conceptMask[sequenceWordIndex].item())
	if(targetIsConceptFeature):
		if(targetToken.lemma is None):
			raise RuntimeError("getInferenceTargetWord error: concept token lemma is None")
		targetWord = targetToken.lemma
	else:
		targetWord = targetToken.word
	return targetWord

def getInferenceCandidateWord(databaseNetworkObject, columnIndex, featureIndex):
	candidateWord = None
	if(columnIndex < 0 or columnIndex >= len(databaseNetworkObject.conceptColumnsList)):
		raise RuntimeError("getInferenceCandidateWord error: columnIndex out of range")
	if(featureIndex == featureIndexPrimeConceptNeuron):
		candidateWord = databaseNetworkObject.conceptColumnsList[columnIndex]
	else:
		if(featureIndex < 0 or featureIndex >= len(databaseNetworkObject.conceptFeaturesList)):
			raise RuntimeError("getInferenceCandidateWord error: featureIndex out of range")
		candidateWord = databaseNetworkObject.conceptFeaturesList[featureIndex]
	return candidateWord

def calculateInferenceTargetProbability(databaseNetworkObject, globalFeatureNeuronsActivation, globalFeatureNeuronsTime, tokensSequence, sequenceWordIndex, conceptMask, allowedColumnsConstraint, constraintModePrediction, connectedColumnsConstraint, connectedColumnsFeatureMap):
	targetProbability = 0.0
	sequenceColumnIndex = None
	if(globalFeatureNeuronsActivation is None):
		raise RuntimeError("calculateInferenceTargetProbability error: globalFeatureNeuronsActivation is None")
	if(inferenceUseNeuronFeaturePropertiesTime):
		if(globalFeatureNeuronsTime is None):
			raise RuntimeError("calculateInferenceTargetProbability error: globalFeatureNeuronsTime is None while inferenceUseNeuronFeaturePropertiesTime")
		if(useSANIcolumns or useSANIfeaturesAndColumns):
			sequenceColumnIndex = GIAANNproto_predictionActivate.calculateSequenceColumnIndex(conceptMask, sequenceWordIndex)
	constraintState = GIAANNproto_predictionConstraints.createConstraintState(allowedColumnsConstraint, constraintModePrediction)
	columnIndices, featureIndices, activationValues = GIAANNproto_predictionBeamSearch.calculateSelectionActivationDistribution(databaseNetworkObject, globalFeatureNeuronsActivation, globalFeatureNeuronsTime, constraintState, connectedColumnsConstraint, connectedColumnsFeatureMap, sequenceWordIndex, sequenceColumnIndex, True)
	if(columnIndices is not None and featureIndices is not None and activationValues is not None and columnIndices.numel() > 0 and featureIndices.numel() > 0 and activationValues.numel() > 0):
		targetWord = getInferenceTargetWord(tokensSequence, conceptMask, sequenceWordIndex)
		totalActivation = 0.0
		targetActivation = 0.0
		for activationIndex in range(columnIndices.shape[0]):
			columnIndex = int(columnIndices[activationIndex].item())
			featureIndex = int(featureIndices[activationIndex].item())
			activationValue = float(activationValues[activationIndex].item())
			# Biased time penalties can drive activations negative; these contribute zero probability mass.
			if(activationValue < 0.0):
				activationValue = 0.0
			if(activationValue == 0.0):
				continue
			candidateWord = getInferenceCandidateWord(databaseNetworkObject, columnIndex, featureIndex)
			totalActivation += activationValue
			if(candidateWord == targetWord):
				targetActivation += activationValue
		if(totalActivation > 0.0):
			targetProbability = targetActivation / totalActivation
	return targetProbability

if(inferenceOnlyRetainPredictedTargetObservedColumn):
	def loadObservedColumnInference(databaseNetworkObject, observedColumnsDict, conceptIndex, sequenceWordIndex):
		lemma = databaseNetworkObject.conceptColumnsList[conceptIndex]
		observedColumn = GIAANNproto_databaseNetwork.loadOrCreateObservedColumn(databaseNetworkObject, conceptIndex, lemma, sequenceWordIndex, deviceLoadColumnInference, deviceLoadColumnInferenceCopy)
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
		return
		

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
		if(not GIAANNproto_databaseNetwork.isFeatureIndexReferenceSetDelimiterDeterministic(databaseNetworkObject, featureIndex)):
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
		if(not GIAANNproto_databaseNetwork.isFeatureIndexReferenceSetDelimiterProbabilistic(databaseNetworkObject, featureIndex)):
			return False
	return True


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

def processConceptWordsInference(sequenceObservedColumns, sequenceIndex, sequence, sequenceSeed, sequencePredict, numSeedTokens, sequenceRaw):
	if(printHeaderDuringInferencePredict):
		print("processConceptWordsInference:")

	sequenceWordIndex = 0

	tokensSequence = GIAANNproto_sequenceTokens.getTokens(sequence)
	addInferenceTop1AccuracyBitsPerByteBytes(sequenceRaw)
	conceptMask, conceptIndices, numberConcepts = GIAANNproto_sequenceConcepts.createConceptMask(sequenceObservedColumns, tokensSequence)

	numPredictionTokens = len(sequencePredict)	#set numPredictionTokens (dynamic)
				
	#identify first activated column(s) in seed phase:
	kcMax = 1	#not used
	initialContextWordIndex = 0		#use first seed token as the context for the first prediction
	initialContextWordIndex = max(0, min(initialContextWordIndex, len(tokensSequence)-1))
	targetPreviousColumnIndex, targetNextColumnIndex, targetFeatureIndex = GIAANNproto_databaseNetwork.getTokenConceptFeatureIndexTensor(sequenceObservedColumns, tokensSequence, conceptMask, initialContextWordIndex, kcMax)
	conceptColumnIndex = int(targetPreviousColumnIndex)
	conceptColumnFeatureIndex = int(targetFeatureIndex)
	conceptActivationState = None
	databaseNetworkObject = sequenceObservedColumns.databaseNetworkObject
	globalFeatureNeuronsActivation = databaseNetworkObject.globalFeatureNeurons[databaseNetworkObject.arrayIndexPropertiesActivationIndex]
	globalFeatureNeuronsTime = None
	if(inferenceUseNeuronFeaturePropertiesTime):
		globalFeatureNeuronsTime = databaseNetworkObject.globalFeatureNeurons[databaseNetworkObject.arrayIndexPropertiesTimeIndex]
	if(predictionColumnsMustActivateConceptFeature):
		conceptActivationState = initialiseConceptActivationState(conceptColumnIndex, conceptColumnFeatureIndex)
	observedColumnsDict = sequenceObservedColumns.observedColumnsDict  # key: lemma, value: ObservedColumn	#every observed column in inference (seed and prediction phases)
	if(inferenceOnlyRetainPredictedTargetObservedColumn):
		observedColumnsDict = {}
	seedTokensProcessed = 0
	predictionTokensProcessed = 0
	inferenceTerminatedPrematurely = False

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
				if(debugWarningInferenceOnPredictionTargetMismatch):
					print("warning: featurePredictionTargetMatch=False")
				if(debugTerminateInferenceOnPredictionTargetMismatch):
					print("debugTerminateInferenceOnPredictionTargetMismatch: prematurely terminating inference")
					inferenceTerminatedPrematurely = True
					break
	except GIAANNproto_predictionConstraints.InferenceStopSequenceNoPredictionCandidatesAvailable:
		inferenceTerminatedPrematurely = True
	if(inferenceTerminatedPrematurely and printInferenceTop1AccuracyBitsPerByte):
		raise RuntimeError("processConceptWordsInference error: BPB requires inference to evaluate every token in the sequence")
	if(inferenceTerminatedPrematurely):
		addInferenceTop1AccuracyCountPadding(numSeedTokens, numPredictionTokens, seedTokensProcessed, predictionTokensProcessed)
	if(debugPrintTotalInferenceTokens):
		GIAANNproto_debug.addTotalInferenceTokens(seedTokensProcessed, predictionTokensProcessed)
	if(drawNetworkDuringInference):
		databaseNetworkObject.globalFeatureNeurons = GIAANNproto_sparseTensors.replaceAllSparseTensorElementsAtFirstDimIndex(databaseNetworkObject.globalFeatureNeurons, globalFeatureNeuronsActivation, databaseNetworkObject.arrayIndexPropertiesActivationIndex)
		if(inferenceUseNeuronFeaturePropertiesTime):
			databaseNetworkObject.globalFeatureNeurons = GIAANNproto_sparseTensors.replaceAllSparseTensorElementsAtFirstDimIndex(databaseNetworkObject.globalFeatureNeurons, globalFeatureNeuronsTime, databaseNetworkObject.arrayIndexPropertiesTimeIndex)

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
	globalFeatureNeuronsActivation = activateInitialSeedPredictionIfRequired(sequenceWordIndex, conceptColumnIndex, conceptColumnFeatureIndex, globalFeatureNeuronsActivation)
	
	globalFeatureNeuronsStrength = databaseNetworkObject.globalFeatureNeurons[databaseNetworkObject.arrayIndexPropertiesStrengthIndex]
	if(inferenceUseNeuronFeaturePropertiesTime and globalFeatureNeuronsTime is None):
		raise RuntimeError("processColumnInferencePrediction error: globalFeatureNeuronsTime is None while inferenceUseNeuronFeaturePropertiesTime")
	sequenceColumnIndex = None
	if(inferenceUseNeuronFeaturePropertiesTime):
		if(useSANIcolumns or useSANIfeaturesAndColumns):
			sequenceColumnIndex = GIAANNproto_predictionActivate.calculateSequenceColumnIndex(conceptMask, sequenceWordIndex)
	globalFeatureConnectionsActivation = None

	#set constraintModePrediction;
	allowedColumnsConstraint, constraintModePrediction = calculatePredictionColumnConstraints(databaseNetworkObject, conceptColumnIndexTensor, conceptColumnFeatureIndexTensor, seedPhase)
	
	#set predictionEnsureConnectedToPreviousPrediction connectedColumnsConstraint/connectedColumnsFeatureMap;
	connectedColumnsConstraint, connectedColumnsFeatureMap = calculateConnectedColumnsConstraint(databaseNetworkObject, observedColumnsDict, conceptColumnIndexTensor, conceptColumnFeatureIndexTensor, sequenceWordIndex, wordPredictionIndex, tokensSequence, seedPhase)
		
	if(sequenceWordIndex > 0):
		#set conceptColumnIndexActivation/conceptColumnFeatureIndexActivation;
		conceptColumnIndexActivation, conceptColumnFeatureIndexActivation, conceptColumnIndexTensorActivation, conceptColumnFeatureIndexTensorActivation = calculateConceptActivationTarget(conceptColumnIndex, conceptColumnFeatureIndex, conceptActivationState)
		#populate sequence observed columns;
		sequenceObservedColumnsPrediction = createSequenceObservedColumnsPrediction(databaseNetworkObject, observedColumnsDict, conceptColumnIndex, sequenceWordIndex)
		#decrement global activations;
		globalFeatureNeuronsActivation = decrementGlobalFeatureActivationsForPrediction(globalFeatureNeuronsActivation)
		#set activationSequenceWordIndex/activationSequenceColumnIndex;
		activationSequenceWordIndex, activationSequenceColumnIndex = calculateActivationSequenceIndices(sequenceWordIndex, sequenceColumnIndex, conceptMask)
		#process features (activate global neurons based on connection targets);
		globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, globalFeatureNeuronsTime = processFeaturePredictionActivations(databaseNetworkObject, observedColumnsDict, sequenceObservedColumnsPrediction, globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, conceptColumnIndexActivation, conceptColumnFeatureIndexActivation, globalFeatureNeuronsTime, activationSequenceWordIndex, activationSequenceColumnIndex, sequenceWordIndex)
	else:
		#activation targets have already been activated
		sequenceObservedColumnsPrediction = SequenceObservedColumnsDraw(databaseNetworkObject, observedColumnsDict)

	targetProbability = None
	if(printInferenceTop1AccuracyBitsPerByte):
		if(sequenceWordIndex == 0):
			targetProbability = calculateInferenceTargetProbability(databaseNetworkObject, globalFeatureNeuronsActivation, globalFeatureNeuronsTime, tokensSequence, sequenceWordIndex, conceptMask, allowedColumnsConstraint, constraintModePrediction, connectedColumnsConstraint, connectedColumnsFeatureMap)

	#deactivate previously predicted neurons;
	globalFeatureNeuronsActivation, conceptActivationState = deactivatePredictedNeuronActivations(globalFeatureNeuronsActivation, conceptColumnIndexTensor, conceptColumnFeatureIndexTensor, conceptColumnFeatureIndexTensorActivation, conceptColumnIndex, conceptColumnFeatureIndex, conceptColumnIndexActivation, conceptColumnFeatureIndexActivation, conceptActivationState)

	if(printInferenceTop1AccuracyBitsPerByte):
		if(sequenceWordIndex > 0):
			targetProbability = calculateInferenceTargetProbability(databaseNetworkObject, globalFeatureNeuronsActivation, globalFeatureNeuronsTime, tokensSequence, sequenceWordIndex, conceptMask, allowedColumnsConstraint, constraintModePrediction, connectedColumnsConstraint, connectedColumnsFeatureMap)
		addInferenceTop1AccuracyBitsPerByteProbability(targetProbability)
	
	#select next prediction column/feature;
	if(seedPhase):
		#seedPhase;
		conceptColumnIndexPred, conceptColumnFeatureIndexPred, conceptColumnIndexNext, conceptColumnFeatureIndexNext, targetPreviousColumnIndex, targetNextColumnIndex, globalFeatureNeuronsActivation, predictionCandidatesAvailable = selectNextColumnFeatureSeedPhase(sequenceObservedColumns, databaseNetworkObject, globalFeatureNeuronsActivation, tokensSequence, conceptMask, sequenceWordIndex, wordPredictionIndex, allowedColumnsConstraint, constraintModePrediction, connectedColumnsConstraint, connectedColumnsFeatureMap)
	else:	
		#predictionPhase;
		conceptColumnIndexPred, conceptColumnFeatureIndexPred, conceptColumnIndexNext, conceptColumnFeatureIndexNext, targetPreviousColumnIndex, targetNextColumnIndex, predictionCandidatesAvailable = selectNextColumnFeaturePredictionPhase(sequenceObservedColumns, databaseNetworkObject, observedColumnsDict, globalFeatureNeuronsActivation, globalFeatureNeuronsStrength, globalFeatureConnectionsActivation, globalFeatureNeuronsTime, tokensSequence, wordPredictionIndex, sequenceWordIndex, conceptMask, allowedColumnsConstraint, constraintModePrediction, conceptActivationState, connectedColumnsConstraint, connectedColumnsFeatureMap)
	if(conceptColumnIndexPred is None or conceptColumnFeatureIndexPred is None):
		GIAANNproto_predictionConstraints.raiseOrStopPredictionConnectivityError(sequenceWordIndex, wordPredictionIndex, tokensSequence, "no prediction candidates available")

	#calculate featurePredictionTargetMatch; 
	featurePredictionTargetMatch, targetWord, predictedWord, targetColumnName, predictedColumnName = calculateInferencePredictionMatch(tokensSequence, sequenceWordIndex, conceptMask, databaseNetworkObject, conceptColumnIndexPred, conceptColumnFeatureIndexPred, targetPreviousColumnIndex, targetNextColumnIndex, predictionCandidatesAvailable)

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
		databaseNetworkObject.globalFeatureNeurons = GIAANNproto_sparseTensors.replaceAllSparseTensorElementsAtFirstDimIndex(databaseNetworkObject.globalFeatureNeurons, globalFeatureNeuronsActivation, databaseNetworkObject.arrayIndexPropertiesActivationIndex)
		if(inferenceUseNeuronFeaturePropertiesTime):
			databaseNetworkObject.globalFeatureNeurons = GIAANNproto_sparseTensors.replaceAllSparseTensorElementsAtFirstDimIndex(databaseNetworkObject.globalFeatureNeurons, globalFeatureNeuronsTime, databaseNetworkObject.arrayIndexPropertiesTimeIndex)
		GIAANNproto_databaseNetworkDraw.visualizeGraph(sequenceObservedColumnsPrediction, True, save=drawNetworkDuringInferenceSave, fileName=drawNetworkDuringInferenceSaveFilenamePrepend+generateDrawSequenceIndex(sequenceWordIndex))
	if(GIAANNproto_debug.debugPrintGPUramUsage):
		if(executionMode=="inference"):
			GIAANNproto_debug.debugPrintRamUsage("processColumnInferencePrediction", "sequenceIndex = " + str(sequenceIndex) + ", sequenceWordIndex = " + str(sequenceWordIndex) + ", wordPredictionIndex = " + str(wordPredictionIndex) + ", seedPhase = " + str(seedPhase))
	return featurePredictionTargetMatch, conceptColumnIndexNext, conceptColumnFeatureIndexNext, conceptActivationState, globalFeatureNeuronsActivation, globalFeatureNeuronsTime

def ensurePredictionStateAvailable(conceptColumnsIndices, conceptColumnsFeatureIndices, sequenceWordIndex, wordPredictionIndex, tokensSequence, reason):
	if(conceptColumnsIndices is None or conceptColumnsIndices.numel() == 0):
		GIAANNproto_predictionConstraints.raisePredictionConnectivityError(sequenceWordIndex, wordPredictionIndex, tokensSequence, reason + " (missing concept columns)")
	if(conceptColumnsFeatureIndices is None or conceptColumnsFeatureIndices.numel() == 0):
		GIAANNproto_predictionConstraints.raisePredictionConnectivityError(sequenceWordIndex, wordPredictionIndex, tokensSequence, reason + " (missing column features)")


def activateInitialSeedPredictionIfRequired(sequenceWordIndex, conceptColumnIndex, conceptColumnFeatureIndex, globalFeatureNeuronsActivation):
	globalFeatureNeuronsActivationResult = globalFeatureNeuronsActivation
	#burst the initial seed in the sequence;
	if(sequenceWordIndex == 0):
		#activate source token (incremental seed during train)
			#if(wordPredictionIndex == 1) will reactivate first seed token column feature (as it was not saved during wordPredictionIndex==0)
		branchIndex = 0
		indicesToUpdateList = [branchIndex, arrayIndexSegmentLast, int(conceptColumnIndex), int(conceptColumnFeatureIndex)]
		globalFeatureNeuronsActivationResult = GIAANNproto_sparseTensors.addElementValueToSparseTensor(globalFeatureNeuronsActivationResult, indicesToUpdateList, j1)
	return globalFeatureNeuronsActivationResult

def calculatePredictionColumnConstraints(databaseNetworkObject, conceptColumnIndexTensor, conceptColumnFeatureIndexTensor, seedPhase):
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
	return allowedColumnsConstraint, constraintModePrediction

def calculateConnectedColumnsConstraint(databaseNetworkObject, observedColumnsDict, conceptColumnIndexTensor, conceptColumnFeatureIndexTensor, sequenceWordIndex, wordPredictionIndex, tokensSequence, seedPhase):
	#set predictionEnsureConnectedToPreviousPrediction connectedColumnsConstraint/connectedColumnsFeatureMap;
	connectedColumnsConstraint = None
	connectedColumnsFeatureMap = None
	if((predictionEnsureConnectedToPreviousPrediction or enforceDirectConnectionsMinWordDistance) and (inferenceSeedNetwork and sequenceWordIndex > 0) and not (seedPhase and enforceDirectConnectionsIgnoreSeed)):
		#limit prediction candidates to columns directly connected to previously predicted nodes
		connectedColumnsConstraint, connectedColumnsFeatureMap = GIAANNproto_predictionConstraints.buildConnectedColumnsLookupFromPrediction(databaseNetworkObject, observedColumnsDict, conceptColumnIndexTensor, conceptColumnFeatureIndexTensor)
	if((predictionEnsureConnectedToPreviousPrediction or enforceDirectConnectionsMinWordDistance) and (inferenceSeedNetwork and sequenceWordIndex > 0) and connectedColumnsConstraint is not None and connectedColumnsConstraint.numel() == 0 and inferenceUseNextTokenPredictionsOrTargetsToActivateNextColumnFeatures):
		GIAANNproto_predictionConstraints.raiseOrStopPredictionConnectivityError(sequenceWordIndex, wordPredictionIndex, tokensSequence, "previous prediction has no outgoing connections")
	return connectedColumnsConstraint, connectedColumnsFeatureMap

def calculateConceptActivationTarget(conceptColumnIndex, conceptColumnFeatureIndex, conceptActivationState):
	#set conceptColumnIndexActivation/conceptColumnFeatureIndexActivation;
	conceptColumnIndexActivation = conceptColumnIndex
	conceptColumnFeatureIndexActivation = conceptColumnFeatureIndex
	if(predictionColumnsMustActivateConceptFeature):
		conceptColumnIndexActivation, conceptColumnFeatureIndexActivation = enforceConceptFeaturePredictionOrder(conceptColumnIndex, conceptColumnFeatureIndex, conceptActivationState)
	conceptColumnIndexTensorActivation = pt.tensor([int(conceptColumnIndexActivation)], dtype=pt.long)
	conceptColumnFeatureIndexTensorActivation = pt.tensor([int(conceptColumnFeatureIndexActivation)], dtype=pt.long)
	return conceptColumnIndexActivation, conceptColumnFeatureIndexActivation, conceptColumnIndexTensorActivation, conceptColumnFeatureIndexTensorActivation

def createSequenceObservedColumnsPrediction(databaseNetworkObject, observedColumnsDict, conceptColumnIndex, sequenceWordIndex):
	#populate sequence observed columns;
	if(inferenceOnlyRetainPredictedTargetObservedColumn):
		sequenceObservedColumnsPrediction = SequenceObservedColumnsDraw(databaseNetworkObject, observedColumnsDict)
	else:
		# Refresh the observed columns dictionary for each new sequence
		observedColumnsSequenceCandidateIndexDict = {}
		lemma = databaseNetworkObject.conceptColumnsList[int(conceptColumnIndex)]
		# Load observed column from disk or create new one
		observedColumn = GIAANNproto_databaseNetwork.loadOrCreateObservedColumn(databaseNetworkObject, int(conceptColumnIndex), lemma, sequenceWordIndex, deviceLoadColumnInference, deviceLoadColumnInferenceCopy)
		observedColumnsDict[lemma] = observedColumn
		observedColumnsSequenceCandidateIndexDict[0] = observedColumn
		sequenceObservedColumnsPrediction = SequenceObservedColumnsInferencePrediction(databaseNetworkObject, observedColumnsDict, observedColumnsSequenceCandidateIndexDict)
	return sequenceObservedColumnsPrediction

def decrementGlobalFeatureActivationsForPrediction(globalFeatureNeuronsActivation):
	#decrement global activations;
	globalFeatureNeuronsActivationResult = globalFeatureNeuronsActivation
	if(inferenceDecrementActivations):
		#decrement activation after each prediction interval
		globalFeatureNeuronsActivationResult = GIAANNproto_predictionActivate.decrementActivation(globalFeatureNeuronsActivationResult, activationDecrementPerPredictedToken)
		#if(inferenceUseNeuronFeaturePropertiesTime):	#OLD
		#	globalFeatureNeuronsTime = GIAANNproto_predictionActivate.decrementActivation(globalFeatureNeuronsTime, inferenceUseNeuronFeaturePropertiesTimeDecrement)
	return globalFeatureNeuronsActivationResult

def calculateActivationSequenceIndices(sequenceWordIndex, sequenceColumnIndex, conceptMask):
	#set activationSequenceWordIndex/activationSequenceColumnIndex;
	activationSequenceWordIndex = sequenceWordIndex
	activationSequenceColumnIndex = sequenceColumnIndex
	if(inferenceUseNeuronFeaturePropertiesTime and sequenceWordIndex > 0):
		activationSequenceWordIndex = sequenceWordIndex - 1
		if(activationSequenceWordIndex < 0):
			raise RuntimeError("processColumnInferencePrediction: activationSequenceWordIndex out of range")
		if(useSANIcolumns or useSANIfeaturesAndColumns):
			activationSequenceColumnIndex = GIAANNproto_predictionActivate.calculateSequenceColumnIndex(conceptMask, activationSequenceWordIndex)
	return activationSequenceWordIndex, activationSequenceColumnIndex

def processFeaturePredictionActivations(databaseNetworkObject, observedColumnsDict, sequenceObservedColumnsPrediction, globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, conceptColumnIndexActivation, conceptColumnFeatureIndexActivation, globalFeatureNeuronsTime, activationSequenceWordIndex, activationSequenceColumnIndex, sequenceWordIndex):
	#process features (activate global neurons based on connection targets);
	globalFeatureNeuronsActivationResult = globalFeatureNeuronsActivation
	globalFeatureConnectionsActivationResult = globalFeatureConnectionsActivation
	globalFeatureNeuronsTimeResult = globalFeatureNeuronsTime
	if(inferenceOnlyRetainPredictedTargetObservedColumn):
		sourceConceptIndexValue = int(conceptColumnIndexActivation)
		observedColumn = loadObservedColumnInference(databaseNetworkObject, observedColumnsDict, sourceConceptIndexValue, sequenceWordIndex)
		connectionDevice = globalFeatureNeuronsActivationResult.device
		featureConnections = observedColumn.prepareFeatureConnectionsForSourceFeature(int(conceptColumnFeatureIndexActivation), targetDevice=connectionDevice, createMissing=False)
		globalFeatureNeuronsActivationResult, globalFeatureConnectionsActivationResult, globalFeatureNeuronsTimeResult = GIAANNproto_predictionActivate.processFeaturesActivePredict(databaseNetworkObject, globalFeatureNeuronsActivationResult, globalFeatureConnectionsActivationResult, featureConnections, conceptColumnIndexActivation, conceptColumnFeatureIndexActivation, globalFeatureNeuronsTimeResult, activationSequenceWordIndex, activationSequenceColumnIndex)
	else:
		globalFeatureNeuronsActivationResult, globalFeatureConnectionsActivationResult, globalFeatureNeuronsTimeResult = GIAANNproto_predictionActivate.processFeaturesActivePredictSingle(databaseNetworkObject, globalFeatureNeuronsActivationResult, globalFeatureConnectionsActivationResult, sequenceObservedColumnsPrediction, conceptColumnIndexActivation, conceptColumnFeatureIndexActivation, globalFeatureNeuronsTimeResult, activationSequenceWordIndex, activationSequenceColumnIndex)
	return globalFeatureNeuronsActivationResult, globalFeatureConnectionsActivationResult, globalFeatureNeuronsTimeResult

def deactivatePredictedNeuronActivations(globalFeatureNeuronsActivation, conceptColumnIndexTensor, conceptColumnFeatureIndexTensor, conceptColumnFeatureIndexTensorActivation, conceptColumnIndex, conceptColumnFeatureIndex, conceptColumnIndexActivation, conceptColumnFeatureIndexActivation, conceptActivationState):
	#deactivate previously predicted neurons;
	globalFeatureNeuronsActivationResult = globalFeatureNeuronsActivation
	conceptActivationStateResult = conceptActivationState
	if(inferenceDeactivateNeuronsUponPrediction):
		branchIndex = 0
		if(multipleDendriticBranches):
			branchIndex = GIAANNproto_predictionActivate.selectActivatedBranchIndex(globalFeatureNeuronsActivationResult, int(conceptColumnIndex), int(conceptColumnFeatureIndex))
		branchTensor = pt.tensor(branchIndex, device=conceptColumnIndexTensor.device)
		if(useSANI):
			indicesToUpdateList = []
			for segmentIndex in range(arrayNumberOfSegments):
				indexToUpdate = pt.stack([branchTensor, pt.tensor(segmentIndex, device=conceptColumnIndexTensor.device), conceptColumnIndexTensor.squeeze(), conceptColumnFeatureIndexTensorActivation.squeeze()], dim=0)
				indicesToUpdateList.append(indexToUpdate)
			indicesToUpdate = pt.stack(indicesToUpdateList, dim=0)
		else:
			indicesToUpdate = pt.stack([branchTensor, pt.tensor(arrayIndexSegmentFirst, device=conceptColumnIndexTensor.device), conceptColumnIndexTensor.squeeze(), conceptColumnFeatureIndexTensorActivation.squeeze()], dim=0)
		modifier = 0
		globalFeatureNeuronsActivationResult = GIAANNproto_sparseTensors.modifySparseTensor(globalFeatureNeuronsActivationResult, indicesToUpdate, modifier, multiply=False)
		if(predictionColumnsMustActivateConceptFeature):
			conceptActivationStateResult = updateConceptActivationState(conceptActivationStateResult, conceptColumnIndexActivation, conceptColumnFeatureIndexActivation)
	return globalFeatureNeuronsActivationResult, conceptActivationStateResult

def selectNextColumnFeatureSeedPhase(sequenceObservedColumns, databaseNetworkObject, globalFeatureNeuronsActivation, tokensSequence, conceptMask, sequenceWordIndex, wordPredictionIndex, allowedColumnsConstraint, constraintModePrediction, connectedColumnsConstraint, connectedColumnsFeatureMap):
	#seedPhase;
	targetPreviousColumnIndex, targetNextColumnIndex, targetFeatureIndex = GIAANNproto_databaseNetwork.getTokenConceptFeatureIndexTensor(sequenceObservedColumns, tokensSequence, conceptMask, sequenceWordIndex, kcNetwork)
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
	branchIndex = 0
	indicesToUpdateList = [branchIndex, arrayIndexSegmentLast, int(conceptColumnIndexNext), int(conceptColumnFeatureIndexNext)]
	globalFeatureNeuronsActivationResult = GIAANNproto_sparseTensors.addElementValueToSparseTensor(globalFeatureNeuronsActivation, indicesToUpdateList, j1)
	conceptColumnIndexPred = conceptColumnIndexNext	#temporarily assign prediction from seed target for print only
	conceptColumnFeatureIndexPred = conceptColumnFeatureIndexNext	#temporarily assign prediction from seed target for print only
	return conceptColumnIndexPred, conceptColumnFeatureIndexPred, conceptColumnIndexNext, conceptColumnFeatureIndexNext, targetPreviousColumnIndex, targetNextColumnIndex, globalFeatureNeuronsActivationResult, True

def selectNextColumnFeaturePredictionPhase(sequenceObservedColumns, databaseNetworkObject, observedColumnsDict, globalFeatureNeuronsActivation, globalFeatureNeuronsStrength, globalFeatureConnectionsActivation, globalFeatureNeuronsTime, tokensSequence, wordPredictionIndex, sequenceWordIndex, conceptMask, allowedColumnsConstraint, constraintModePrediction, conceptActivationState, connectedColumnsConstraint, connectedColumnsFeatureMap):
	#predictionPhase;
	predictionCandidatesAvailable = True
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
			targetPreviousColumnIndex, targetNextColumnIndex, targetFeatureIndex = GIAANNproto_databaseNetwork.getTokenConceptFeatureIndexTensor(sequenceObservedColumns, tokensSequence, conceptMask, sequenceWordIndex, kcNetwork)
			conceptColumnIndexPred = int(targetPreviousColumnIndex)
			conceptColumnFeatureIndexPred = int(targetFeatureIndex)
	else:
		try:
			conceptColumnIndexPred, conceptColumnFeatureIndexPred, targetPreviousColumnIndex, targetNextColumnIndex = GIAANNproto_predictionBeamSearch.beamSearchSelectSingleStepFeature(sequenceObservedColumns, databaseNetworkObject, observedColumnsDict, globalFeatureNeuronsActivation, globalFeatureNeuronsStrength, globalFeatureConnectionsActivation, globalFeatureNeuronsTime, tokensSequence, wordPredictionIndex, sequenceWordIndex, conceptMask, allowedColumnsConstraint, constraintModePrediction, conceptActivationState, connectedColumnsConstraint, connectedColumnsFeatureMap)
		except GIAANNproto_predictionConstraints.InferenceStopSequenceNoPredictionCandidatesAvailable:
			if(inferenceUseNextTokenPredictionsOrTargetsToActivateNextColumnFeatures):
				raise
			predictionCandidatesAvailable = False
			targetPreviousColumnIndex, targetNextColumnIndex, targetFeatureIndex = GIAANNproto_databaseNetwork.getTokenConceptFeatureIndexTensor(sequenceObservedColumns, tokensSequence, conceptMask, sequenceWordIndex, kcNetwork)
			conceptColumnIndexPred = int(targetPreviousColumnIndex)
			conceptColumnFeatureIndexPred = int(targetFeatureIndex)
	if(inferenceUseNextTokenPredictionsOrTargetsToActivateNextColumnFeatures):
		conceptColumnIndexNext = conceptColumnIndexPred	#use prediction as next selected feature
		conceptColumnFeatureIndexNext = conceptColumnFeatureIndexPred	#use prediction as next selected feature
	else:
		targetPreviousColumnIndex, targetNextColumnIndex, targetFeatureIndex = GIAANNproto_databaseNetwork.getTokenConceptFeatureIndexTensor(sequenceObservedColumns, tokensSequence, conceptMask, sequenceWordIndex, kcNetwork)
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
	return conceptColumnIndexPred, conceptColumnFeatureIndexPred, conceptColumnIndexNext, conceptColumnFeatureIndexNext, targetPreviousColumnIndex, targetNextColumnIndex, predictionCandidatesAvailable

def calculateInferencePredictionMatch(tokensSequence, sequenceWordIndex, conceptMask, databaseNetworkObject, conceptColumnIndexPred, conceptColumnFeatureIndexPred, targetPreviousColumnIndex, targetNextColumnIndex, predictionCandidatesAvailable):
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
		if(inferenceReportTokenAccuracyConstrainByColumn and featurePredictionTargetMatch):
			if(predictedColumnName != targetColumnName):
				featurePredictionTargetMatch = False
				#printe("inferenceReportTokenAccuracyConstrainByColumn: featurePredictionTargetMatch=False")
	else:
		predictedWord = "<no prediction>"
		predictedColumnName = "<no prediction>"
		if(targetNextColumnIndex is None):
			targetColumnName = databaseNetworkObject.conceptColumnsList[targetPreviousColumnIndex]
		else:
			targetColumnName = databaseNetworkObject.conceptColumnsList[targetPreviousColumnIndex] + "/" + databaseNetworkObject.conceptColumnsList[targetNextColumnIndex]
	return featurePredictionTargetMatch, targetWord, predictedWord, targetColumnName, predictedColumnName
