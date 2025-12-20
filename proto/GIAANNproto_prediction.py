"""GIAANNproto_prediction.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

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
import GIAANNproto_predictionSeed
if(inferenceBeamSearch):
	import GIAANNproto_predictionBeamSearch
if(inferencePredictiveNetwork):
	import GIAANNproto_predictionNetwork
import GIAANNproto_sequenceConcepts
import GIAANNproto_predictionActivate
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
		self.featureConnections = pt.stack(featureConnectionsList, dim=2)
		

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

def applyColumnConstraintToPredictions(databaseNetworkObject, conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, allowedColumns, constraintMode, connectedColumns=None, connectedColumnsFeatures=None):
	if(conceptColumnsIndicesPred is None or constraintMode is None):
		return applyConnectedColumnsConstraint(conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, connectedColumns, connectedColumnsFeatures)
	if(allowedColumns is None or allowedColumns.numel() == 0):
		return applyConnectedColumnsConstraint(conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, connectedColumns, connectedColumnsFeatures)
	allowedSet = set(allowedColumns.cpu().tolist())
	if(len(allowedSet) == 0):
		return applyConnectedColumnsConstraint(conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, connectedColumns, connectedColumnsFeatures)
	predictedColumnsList = conceptColumnsIndicesPred.cpu().tolist()
	if(constraintMode == "internal"):
		indicesToKeep = [idx for idx, columnValue in enumerate(predictedColumnsList) if columnValue in allowedSet]
		if(len(indicesToKeep) == 0):
			repeatFactor = max(1, (conceptColumnsIndicesPred.shape[0] + allowedColumns.shape[0] - 1) // allowedColumns.shape[0])
			replacementColumns = allowedColumns.repeat(repeatFactor)[:conceptColumnsIndicesPred.shape[0]]
			conceptColumnsIndicesPred = replacementColumns
			return conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred
		indexTensor = pt.tensor(indicesToKeep, dtype=pt.long, device=conceptColumnsIndicesPred.device)
		conceptColumnsIndicesPred = conceptColumnsIndicesPred.index_select(0, indexTensor)
		if(conceptColumnsFeatureIndicesPred is not None and conceptColumnsFeatureIndicesPred.shape[0] >= indexTensor.shape[0]):
			conceptColumnsFeatureIndicesPred = conceptColumnsFeatureIndicesPred.index_select(0, indexTensor)
		return applyConnectedColumnsConstraint(conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, connectedColumns, connectedColumnsFeatures)
	elif(constraintMode == "external"):
		indicesToKeep = [idx for idx, columnValue in enumerate(predictedColumnsList) if columnValue not in allowedSet]
		if(len(indicesToKeep) > 0):
			indexTensor = pt.tensor(indicesToKeep, dtype=pt.long, device=conceptColumnsIndicesPred.device)
			conceptColumnsIndicesPred = conceptColumnsIndicesPred.index_select(0, indexTensor)
			if(conceptColumnsFeatureIndicesPred is not None and conceptColumnsFeatureIndicesPred.shape[0] >= indexTensor.shape[0]):
				conceptColumnsFeatureIndicesPred = conceptColumnsFeatureIndicesPred.index_select(0, indexTensor)
			return conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred
		if(len(allowedSet) >= databaseNetworkObject.c):
			raise RuntimeError("applyColumnConstraintToPredictions: external constraint requires columns outside allowed set, but no columns are available.")
		raise RuntimeError("applyColumnConstraintToPredictions: external constraint removed all predicted columns; no eligible external predictions remain.")
	elif(constraintMode == "delimiter"):
		if(allowedSet is None or len(allowedSet) == 0):
			return conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred
		def get_feature_value(rowIndex):
			if(conceptColumnsFeatureIndicesPred is None):
				return None
			if(conceptColumnsFeatureIndicesPred.dim() == 1):
				if(rowIndex >= conceptColumnsFeatureIndicesPred.shape[0]):
					return None
				return conceptColumnsFeatureIndicesPred[rowIndex].item()
			if(rowIndex >= conceptColumnsFeatureIndicesPred.shape[0]):
				return None
			rowTensor = conceptColumnsFeatureIndicesPred[rowIndex]
			if(rowTensor.numel() == 0):
				return None
			return rowTensor.view(-1)[0].item()
		indicesToKeep = []
		for idx, columnValue in enumerate(predictedColumnsList):
			if(columnValue not in allowedSet):
				indicesToKeep.append(idx)
			else:
				featureValue = get_feature_value(idx)
				if(featureValue is None):
					continue
				featureValueInt = int(featureValue)
				isDeterministicDelimiter = GIAANNproto_databaseNetworkExcitation.isFeatureIndexReferenceSetDelimiterDeterministic(databaseNetworkObject, featureValueInt)
				isProbabilisticDelimiter = GIAANNproto_databaseNetworkExcitation.isFeatureIndexReferenceSetDelimiterProbabilistic(databaseNetworkObject, featureValueInt)
				if(isDeterministicDelimiter or isProbabilisticDelimiter):
					indicesToKeep.append(idx)
		if(len(indicesToKeep) == 0):
			raise RuntimeError("applyColumnConstraintToPredictions: delimiter constraint removed all predictions; unable to find external/delimiter-compatible columns.")
		indexTensor = pt.tensor(indicesToKeep, dtype=pt.long, device=conceptColumnsIndicesPred.device)
		conceptColumnsIndicesPred = conceptColumnsIndicesPred.index_select(0, indexTensor)
		if(conceptColumnsFeatureIndicesPred is not None and conceptColumnsFeatureIndicesPred.shape[0] >= indexTensor.shape[0]):
			conceptColumnsFeatureIndicesPred = conceptColumnsFeatureIndicesPred.index_select(0, indexTensor)
		return applyConnectedColumnsConstraint(conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, connectedColumns, connectedColumnsFeatures)
	else:
		return applyConnectedColumnsConstraint(conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, connectedColumns)


def clearPredictionTensors(conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, connectedColumns=None):
	if(conceptColumnsIndicesPred is not None):
		conceptColumnsIndicesPred = conceptColumnsIndicesPred[:0]
	elif(connectedColumns is not None):
		conceptColumnsIndicesPred = connectedColumns[:0]
	else:
		conceptColumnsIndicesPred = pt.empty(0, dtype=pt.long)
	if(conceptColumnsFeatureIndicesPred is not None):
		conceptColumnsFeatureIndicesPred = conceptColumnsFeatureIndicesPred[:0]
	return conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred


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


def rowHasAllowedFeature(conceptColumnsFeatureIndicesPred, rowIndex, allowedFeaturesSet):
	if(conceptColumnsFeatureIndicesPred is None):
		return False
	if(conceptColumnsFeatureIndicesPred.dim() == 1):
		if(rowIndex >= conceptColumnsFeatureIndicesPred.shape[0]):
			return False
		featureValue = int(conceptColumnsFeatureIndicesPred[rowIndex].item())
		if(featureValue == featureIndexConceptNeuron):
			return True
		return featureValue in allowedFeaturesSet
	if(rowIndex >= conceptColumnsFeatureIndicesPred.shape[0]):
		return False
	rowTensor = conceptColumnsFeatureIndicesPred[rowIndex]
	if(rowTensor is None or rowTensor.numel() == 0):
		return False
	rowValues = rowTensor.view(-1)
	for value in rowValues:
		featureValue = int(value.item())
		if(featureValue == featureIndexConceptNeuron):
			return True
		if(featureValue in allowedFeaturesSet):
			return True
	return False


def applyConnectedColumnsConstraint(conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, connectedColumns, connectedColumnsFeatures=None):
	if(connectedColumns is None):
		return conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred
	if(connectedColumns.numel() == 0):
		return clearPredictionTensors(conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, connectedColumns)
	if(conceptColumnsIndicesPred is None or conceptColumnsIndicesPred.numel() == 0):
		return conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred
	connectedList = connectedColumns.view(-1).tolist()
	if(len(connectedList) == 0):
		return clearPredictionTensors(conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, connectedColumns)
	allowedSet = set(connectedList)
	if(len(allowedSet) == 0):
		return clearPredictionTensors(conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, connectedColumns)
	predictedColumnsList = conceptColumnsIndicesPred.cpu().tolist()
	indicesToKeep = []
	for idx, columnValue in enumerate(predictedColumnsList):
		if(columnValue not in allowedSet):
			continue
		if(connectedColumnsFeatures is not None):
			allowedFeatureSet = connectedColumnsFeatures.get(int(columnValue))
			if(allowedFeatureSet is None or len(allowedFeatureSet) == 0):
				continue
			if(not rowHasAllowedFeature(conceptColumnsFeatureIndicesPred, idx, allowedFeatureSet)):
				continue
		indicesToKeep.append(idx)
	if(len(indicesToKeep) == 0):
		return clearPredictionTensors(conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, connectedColumns)
	indexTensor = pt.tensor(indicesToKeep, dtype=pt.long, device=conceptColumnsIndicesPred.device)
	conceptColumnsIndicesPred = conceptColumnsIndicesPred.index_select(0, indexTensor)
	if(conceptColumnsFeatureIndicesPred is not None and conceptColumnsFeatureIndicesPred.shape[0] >= indexTensor.shape[0]):
		conceptColumnsFeatureIndicesPred = conceptColumnsFeatureIndicesPred.index_select(0, indexTensor)
	return conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred


def buildConnectedColumnsLookupFromPrediction(databaseNetworkObject, observedColumnsDict, conceptColumnsIndices, conceptColumnsFeatureIndices):
	if(not predictionEnsureConnectedToPreviousPrediction and not enforceDirectConnectionsMinWordDistance):
		return None, None
	if(conceptColumnsIndices is None or conceptColumnsFeatureIndices is None):
		return None, None
	if(conceptColumnsIndices.numel() == 0 or conceptColumnsFeatureIndices.numel() == 0):
		device = conceptColumnsIndices.device
		dtype = conceptColumnsIndices.dtype
		return pt.empty(0, dtype=dtype, device=device), None
	connectedColumnsSet = set()
	if(debugConnectNodesToNextNodesInSequenceOnly or enforceDirectConnectionsMinWordDistance):
		connectedColumnsFeatures = {}
	else:
		connectedColumnsFeatures = None
	for rowIndex in range(conceptColumnsIndices.shape[0]):
		columnIndex = int(conceptColumnsIndices[rowIndex].item())
		rowTensor = conceptColumnsFeatureIndices[rowIndex]
		if(rowTensor is None or rowTensor.numel() == 0):
			continue
		observedColumn = getObservedColumnForIndex(databaseNetworkObject, observedColumnsDict, columnIndex)
		if(observedColumn is None):
			continue
		featureValues = rowTensor.detach().view(-1).tolist()
		for featureValue in featureValues:
			targetColumns, targetColumnFeatures = getConnectedColumnsForFeature(observedColumn, int(featureValue), includeFeatureDetails=(connectedColumnsFeatures is not None))
			connectedColumnsSet.update(targetColumns)
			if(connectedColumnsFeatures is not None and targetColumnFeatures is not None):
				for targetColumnIndex, featureSet in targetColumnFeatures.items():
					if(targetColumnIndex < 0 or targetColumnIndex >= databaseNetworkObject.c):
						continue
					columnFeatureSet = connectedColumnsFeatures.setdefault(targetColumnIndex, set())
					columnFeatureSet.update(featureSet)
	if(len(connectedColumnsSet) == 0):
		device = conceptColumnsIndices.device
		dtype = conceptColumnsIndices.dtype
		return pt.empty(0, dtype=dtype, device=device), ({} if connectedColumnsFeatures is not None else None)
	validColumns = [col for col in connectedColumnsSet if col >= 0 and col < databaseNetworkObject.c]
	if(len(validColumns) == 0):
		device = conceptColumnsIndices.device
		dtype = conceptColumnsIndices.dtype
		return pt.empty(0, dtype=dtype, device=device), ({} if connectedColumnsFeatures is not None else None)
	validColumns.sort()
	device = conceptColumnsIndices.device
	dtype = conceptColumnsIndices.dtype
	connectedColumnsTensor = pt.tensor(validColumns, dtype=dtype, device=device)
	if(connectedColumnsFeatures is not None):
		filteredFeatureMap = {}
		for columnIndex in validColumns:
			if(connectedColumnsFeatures is None):
				continue
			featureSet = connectedColumnsFeatures.get(columnIndex, set())
			if(len(featureSet) > 0):
				filteredFeatureMap[columnIndex] = set(featureSet)
		return connectedColumnsTensor, filteredFeatureMap
	else:
		return connectedColumnsTensor, None


def getObservedColumnForIndex(databaseNetworkObject, observedColumnsDict, columnIndex):
	if(columnIndex < 0 or columnIndex >= len(databaseNetworkObject.conceptColumnsList)):
		return None
	columnLemma = databaseNetworkObject.conceptColumnsList[columnIndex]
	if(columnLemma in observedColumnsDict):
		return observedColumnsDict[columnLemma]
	observedColumn = GIAANNproto_databaseNetworkExcitation.loadOrCreateObservedColumn(databaseNetworkObject, columnIndex, columnLemma, columnIndex)
	observedColumnsDict[columnLemma] = observedColumn
	return observedColumn


def getConnectedColumnsForFeature(observedColumn, featureIndex, includeFeatureDetails=False):
	if(featureIndex is None or featureIndex < 0):
		return [], {} if includeFeatureDetails else None
	featureConnectionsStrength = observedColumn.featureConnections[arrayIndexPropertiesStrength]
	featureConnectionsStrength = GIAANNproto_sparseTensors.sliceSparseTensor(featureConnectionsStrength, 1, featureIndex)
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
	targetColumns = targetColumnIndices[1].unique()
	targetColumnsList = targetColumns.cpu().tolist()
	if(includeFeatureDetails):
		targetFeatures = targetColumnIndices[2].cpu().tolist()
		columnFeatureMap = {}
		for columnValue, featureValue in zip(targetColumnIndices[1].tolist(), targetFeatures):
			columnFeatureMap.setdefault(columnValue, set()).add(featureValue)
		return targetColumnsList, columnFeatureMap
	else:
		return targetColumnsList, None


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
	
	GIAANNproto_predictionSeed.seedNetwork(sequenceObservedColumns, sequenceIndex, sequence, numSeedTokens)
	
	numPredictionTokens = len(sequencePredict)	#set numPredictionTokens (dynamic)
	
	if(inferencePredictiveNetwork and not inferenceTrainPredictiveNetworkAllSequences):
		initialisePredictiveNetwork(sequenceObservedColumns.databaseNetworkObject)
			
	#identify first activated column(s) in prediction phase:
	if(inferencePredictiveNetwork):
		kcMax = kcNetwork
	else:
		kcMax = 1	#not used
	if(numSeedTokens > 0):
		initialContextWordIndex = numSeedTokens - 1	#use final seed token as the context for the first prediction
	else:
		initialContextWordIndex = 0
	initialContextWordIndex = max(0, min(initialContextWordIndex, len(tokensSequence)-1))
	multipleSources, targetPreviousColumnIndex, targetNextColumnIndex, targetFeatureIndex, conceptColumnsIndices, conceptColumnsFeatureIndices = GIAANNproto_databaseNetworkExcitation.getTokenConceptFeatureIndexTensor(sequenceObservedColumns, tokensSequence, conceptMask, initialContextWordIndex, kcMax)
	conceptActivationState = initialiseConceptActivationState(conceptColumnsIndices, conceptColumnsFeatureIndices)
	observedColumnsDict = sequenceObservedColumns.observedColumnsDict  # key: lemma, value: ObservedColumn	#every observed column in inference (seed and prediction phases)
	
	#predict next tokens;
	for wordPredictionIndex in range(numPredictionTokens):
		sequenceWordIndex = numSeedTokens + wordPredictionIndex
		featurePredictionTargetMatch, conceptColumnsIndices, conceptColumnsFeatureIndices, multipleSources, conceptActivationState = processColumnInferencePrediction(sequenceObservedColumns, sequenceIndex, observedColumnsDict, wordPredictionIndex, sequenceWordIndex, tokensSequence, conceptColumnsIndices, conceptColumnsFeatureIndices, conceptMask, multipleSources, conceptActivationState)
		

def processColumnInferencePrediction(sequenceObservedColumns, sequenceIndex, observedColumnsDict, wordPredictionIndex, sequenceWordIndex, tokensSequence, conceptColumnsIndices, conceptColumnsFeatureIndices, conceptMask, multipleSources, conceptActivationState):
	
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
			indicesToUpdateList = [arrayIndexPropertiesActivation, arrayIndexSegmentLast, conceptColumnsIndicesSource, conceptColumnsFeatureIndicesSource]
			databaseNetworkObject.globalFeatureNeurons = GIAANNproto_sparseTensors.addElementValueToSparseTensor(databaseNetworkObject.globalFeatureNeurons, indicesToUpdateList, j1)
				
	globalFeatureNeuronsActivation = databaseNetworkObject.globalFeatureNeurons[arrayIndexPropertiesActivation]
	#print("1 globalFeatureNeuronsActivation = ", globalFeatureNeuronsActivation)
	globalFeatureNeuronsStrength = databaseNetworkObject.globalFeatureNeurons[arrayIndexPropertiesStrength]
	globalFeatureNeuronsTime = databaseNetworkObject.globalFeatureNeurons[arrayIndexPropertiesTime]
	if(transformerUseInputConnections):
		globalFeatureConnectionsActivation = databaseNetworkObject.globalFeatureConnections[arrayIndexPropertiesActivation]
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
	if(predictionEnsureConnectedToPreviousPrediction or enforceDirectConnectionsMinWordDistance):
		#limit prediction candidates to columns directly connected to previously predicted nodes
		connectedColumnsConstraint, connectedColumnsFeatureMap = buildConnectedColumnsLookupFromPrediction(databaseNetworkObject, observedColumnsDict, conceptColumnsIndices, conceptColumnsFeatureIndices)
	else:
		connectedColumnsConstraint = None
		connectedColumnsFeatureMap = None
	if((predictionEnsureConnectedToPreviousPrediction or enforceDirectConnectionsMinWordDistance) and connectedColumnsConstraint is not None and connectedColumnsConstraint.numel() == 0):
		raisePredictionConnectivityError(sequenceWordIndex, wordPredictionIndex, tokensSequence, "previous prediction has no outgoing connections")
		
	if(wordPredictionIndex > 0):
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
			if(inferenceUseNeuronFeaturePropertiesTime):
				globalFeatureNeuronsTime = GIAANNproto_predictionActivate.decrementActivation(globalFeatureNeuronsTime, inferenceUseNeuronFeaturePropertiesTimeDecrement)

		#process features (activate global target neurons);
		if(multipleSources):
			globalFeatureNeuronsActivation, globalFeatureConnectionsActivation = GIAANNproto_predictionActivate.processFeaturesActivePredictMulti(databaseNetworkObject, globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, sequenceObservedColumnsPrediction, conceptColumnsIndices, conceptColumnsFeatureIndicesActivation)
		else:
			globalFeatureNeuronsActivation, globalFeatureConnectionsActivation = GIAANNproto_predictionActivate.processFeaturesActivePredictSingle(databaseNetworkObject, globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, sequenceObservedColumnsPrediction, conceptColumnsIndices, conceptColumnsFeatureIndicesActivation)

	else:
		#activation targets have already been activated
		sequenceObservedColumnsPrediction = SequenceObservedColumnsDraw(databaseNetworkObject, observedColumnsDict)

	if(inferenceDeactivateNeuronsUponPrediction or inferenceInvertNeuronActivationUponPrediction):
		indicesToUpdateList = []
		for conceptIndex in range(conceptColumnsIndices.shape[0]):
			conceptColumnsIndicesSource = conceptColumnsIndices[conceptIndex]
			conceptColumnsFeatureIndicesSource = conceptColumnsFeatureIndicesActivation[conceptIndex].squeeze(dim=0)
			if(useSANI):
				for segmentIndex in range(arrayNumberOfSegments):
					indexToUpdate = pt.stack([pt.tensor(segmentIndex, device=conceptColumnsIndicesSource.device), conceptColumnsIndicesSource, conceptColumnsFeatureIndicesSource], dim=0)
					indicesToUpdateList.append(indexToUpdate)
			else:
				indicesToUpdate = pt.stack([pt.tensor(arrayIndexSegmentFirst, device=conceptColumnsIndicesSource.device), conceptColumnsIndicesSource, conceptColumnsFeatureIndicesSource], dim=0)
				indicesToUpdateList.append(indicesToUpdate)
		indicesToUpdate = pt.stack(indicesToUpdateList, dim=0)
		if(inferenceDeactivateNeuronsUponPrediction or inferenceInvertNeuronActivationUponPrediction):
			if(inferenceDeactivateNeuronsUponPrediction):
				modifier = 0
			elif(inferenceInvertNeuronActivationUponPrediction):
				modifier = inferenceInvertNeuronActivationUponPredictionLevel
			globalFeatureNeuronsActivationOrig = globalFeatureNeuronsActivation
			globalFeatureNeuronsActivation = GIAANNproto_sparseTensors.modifySparseTensor(globalFeatureNeuronsActivation, indicesToUpdate, modifier, multiply=inferenceInvertNeuronActivationUponPrediction)
			if(inferenceUseNeuronFeaturePropertiesTime):
				globalFeatureNeuronsTime = GIAANNproto_sparseTensors.modifySparseTensor(globalFeatureNeuronsTime, indicesToUpdate, inferenceUseNeuronFeaturePropertiesTimeActivate)	#higher: neuron was more recently activated
			
			databaseNetworkObject.globalFeatureNeurons = GIAANNproto_sparseTensors.replaceAllSparseTensorElementsAtFirstDimIndex(databaseNetworkObject.globalFeatureNeurons, globalFeatureNeuronsActivation, arrayIndexPropertiesActivation)
			if(transformerUseInputConnections):
				databaseNetworkObject.globalFeatureConnections = GIAANNproto_sparseTensors.replaceAllSparseTensorElementsAtFirstDimIndex(databaseNetworkObject.globalFeatureConnections, globalFeatureConnectionsActivation, arrayIndexPropertiesActivation)
			if(inferenceUseNeuronFeaturePropertiesTime):
				databaseNetworkObject.globalFeatureNeurons = GIAANNproto_sparseTensors.replaceAllSparseTensorElementsAtFirstDimIndex(databaseNetworkObject.globalFeatureNeurons, globalFeatureNeuronsTime, arrayIndexPropertiesTime)
			if(predictionColumnsMustActivateConceptFeature):
				conceptActivationState = updateConceptActivationState(conceptActivationState, conceptColumnsIndices, conceptColumnsFeatureIndicesActivation)
	
	if(debugInferencePredictionActivationAccumulation):
		globalFeatureNeuronsTemp = databaseNetworkObject.globalFeatureNeurons.to_dense()
		print("globalFeatureNeuronsTemp = ", globalFeatureNeuronsTemp)

	if(inferenceBeamSearch):
		conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext, multipleSourcesNext, kc, conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, targetMultipleSources, targetPreviousColumnIndex, targetNextColumnIndex = GIAANNproto_predictionBeamSearch.beamSearchPredictNextFeature(sequenceObservedColumns, databaseNetworkObject, observedColumnsDict, globalFeatureNeuronsActivation, globalFeatureNeuronsStrength, globalFeatureConnectionsActivation, globalFeatureNeuronsTime, tokensSequence, wordPredictionIndex, sequenceWordIndex, conceptMask, allowedColumnsConstraint, constraintModePrediction, selectMostActiveFeature, conceptActivationState, connectedColumnsConstraint, connectedColumnsFeatureMap)
	else:
		if(inferencePredictiveNetwork):
			conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext, multipleSourcesNext, kc, conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, targetMultipleSources, targetPreviousColumnIndex, targetNextColumnIndex = GIAANNproto_predictionNetwork.predictMostActiveFeature(sequenceObservedColumns, databaseNetworkObject, tokensSequence, wordPredictionIndex, sequenceWordIndex, conceptMask, allowedColumnsConstraint, constraintModePrediction, conceptActivationState, connectedColumnsConstraint, connectedColumnsFeatureMap)	
		else:
			conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext, multipleSourcesNext, kc, conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, targetMultipleSources, targetPreviousColumnIndex, targetNextColumnIndex = selectMostActiveFeature(sequenceObservedColumns, globalFeatureNeuronsActivation, globalFeatureNeuronsStrength, tokensSequence, wordPredictionIndex, sequenceWordIndex, conceptMask, allowedColumnsConstraint, constraintModePrediction, conceptActivationState, connectedColumnsConstraint, connectedColumnsFeatureMap)

		if((predictionEnsureConnectedToPreviousPrediction or enforceDirectConnectionsMinWordDistance) and connectedColumnsConstraint is not None):
			conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred = applyConnectedColumnsConstraint(conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, connectedColumnsConstraint, connectedColumnsFeatureMap)
			conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext = applyConnectedColumnsConstraint(conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext, connectedColumnsConstraint, connectedColumnsFeatureMap)
			if(conceptColumnsIndicesPred is None or conceptColumnsIndicesPred.numel() == 0):
				raisePredictionConnectivityError(sequenceWordIndex, wordPredictionIndex, tokensSequence, "no connected predictions available for current step")
			if(conceptColumnsIndicesNext is None or conceptColumnsIndicesNext.numel() == 0):
				raisePredictionConnectivityError(sequenceWordIndex, wordPredictionIndex, tokensSequence, "no connected activations available for next step")

	featurePredictionTargetMatch = False
	if(printPredictionsDuringInferencePredict):
		#compare topk column/feature predictions to sequencePredict (target words);
		#implementation limitation; only works with kf = 1;
		for columnPredictionIndex in range(conceptColumnsIndicesPred.shape[0]):
			columnIndex = conceptColumnsIndicesPred[columnPredictionIndex]
			columnName = databaseNetworkObject.conceptColumnsList[columnIndex]
			observedColumnFeatureIndex = conceptColumnsFeatureIndicesPred[columnPredictionIndex, 0]
			if(observedColumnFeatureIndex == featureIndexConceptNeuron):
				predictedWord = columnName
			else:
				predictedWord = databaseNetworkObject.conceptFeaturesList[observedColumnFeatureIndex]
			predictedColumnName = columnName
			
			targetWord = tokensSequence[sequenceWordIndex].word
			if(targetMultipleSources):
				targetColumnName = databaseNetworkObject.conceptColumnsList[targetPreviousColumnIndex] + "/" + databaseNetworkObject.conceptColumnsList[targetNextColumnIndex]
			else:
				targetColumnName = databaseNetworkObject.conceptColumnsList[targetPreviousColumnIndex]
			
			print("\t sequenceWordIndex = ", sequenceWordIndex, ", wordPredictionIndex = ", wordPredictionIndex, ", targetWord = ", targetWord, ", predictedWord = ", predictedWord, ", targetColumn = ", targetColumnName, ", predictedColumn = ", predictedColumnName)
			if(targetWord == predictedWord):
				featurePredictionTargetMatch = True
	if(inferenceInhibitoryNeurons):
		globalFeatureNeuronsActivation = GIAANNproto_predictionInhibition.applyInferenceInhibition(databaseNetworkObject, globalFeatureNeuronsActivation, conceptColumnsIndices, conceptColumnsFeatureIndicesActivation, conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext)
		#persist the inhibited activations so the next prediction step observes the updated state
		databaseNetworkObject.globalFeatureNeurons = GIAANNproto_sparseTensors.replaceAllSparseTensorElementsAtFirstDimIndex(databaseNetworkObject.globalFeatureNeurons, globalFeatureNeuronsActivation, arrayIndexPropertiesActivation)
		if(inferenceDecrementActivationsInhibitory):
			globalInhibitoryActivation = getattr(databaseNetworkObject, "globalInhibitoryNeuronsActivation", None)
			if(globalInhibitoryActivation is not None):
				globalInhibitoryActivation = GIAANNproto_predictionActivate.decrementActivation(globalInhibitoryActivation, activationDecrementPerPredictedToken)
				databaseNetworkObject.globalInhibitoryNeuronsActivation = globalInhibitoryActivation
		if(inferenceDeactivateNeuronsUponPredictionInhibitory):
			deactivatePredictedInhibitoryNeurons(databaseNetworkObject, conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext)

	if(drawNetworkDuringInferencePredict):
		#FUTURE: convert globalFeatureNeuronsActivation back to globalFeatureNeurons for draw
		GIAANNproto_databaseNetworkDrawExcitation.visualizeGraph(sequenceObservedColumnsPrediction, True, save=drawNetworkDuringInferenceSave, fileName=drawNetworkDuringInferenceSaveFilenamePrepend+str(sequenceWordIndex))
	return featurePredictionTargetMatch, conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext, multipleSourcesNext, conceptActivationState



def enforceMinimumPredictionActivationThreshold(conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, activationValues):
	if(minimumPredictionActivationThreshold <= 0):
		return conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred
	if(conceptColumnsIndicesPred is None or conceptColumnsIndicesPred.numel() == 0):
		return conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred
	if(activationValues is None):
		return conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred
	if(activationValues.numel() == 0):
		return conceptColumnsIndicesPred[:0], (conceptColumnsFeatureIndicesPred[:0] if conceptColumnsFeatureIndicesPred is not None else None)
	if(activationValues.dim() == 1):
		activeMask = activationValues >= minimumPredictionActivationThreshold
	else:
		activeMask = (activationValues >= minimumPredictionActivationThreshold).all(dim=1)
	if(activeMask.sum().item() == 0):
		return conceptColumnsIndicesPred[:0], (conceptColumnsFeatureIndicesPred[:0] if conceptColumnsFeatureIndicesPred is not None else None)
	indexTensor = pt.nonzero(activeMask, as_tuple=False).view(-1)
	conceptColumnsIndicesPred = conceptColumnsIndicesPred.index_select(0, indexTensor)
	if(conceptColumnsFeatureIndicesPred is not None and conceptColumnsFeatureIndicesPred.shape[0] > 0):
		conceptColumnsFeatureIndicesPred = conceptColumnsFeatureIndicesPred.index_select(0, indexTensor)
	return conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred


def selectMostActiveFeature(sequenceObservedColumns, globalFeatureNeuronsActivation, globalFeatureNeuronsStrength, tokensSequence, wordPredictionIndex, sequenceWordIndex, conceptMask, allowedColumns=None, constraintMode=None, conceptActivationState=None, connectedColumns=None, connectedColumnsFeatures=None):
	databaseNetworkObject = sequenceObservedColumns.databaseNetworkObject
	#generate targets;
	targetMultipleSources, targetPreviousColumnIndex, targetNextColumnIndex, targetFeatureIndex, targetConceptColumnsIndices, targetConceptColumnsFeatureIndices = GIAANNproto_databaseNetworkExcitation.getTokenConceptFeatureIndexTensor(sequenceObservedColumns, tokensSequence, conceptMask, sequenceWordIndex, kcNetwork)

	globalFeatureNeuronsActivationAllSegments = pt.sum(globalFeatureNeuronsActivation, dim=0)	#sum across all segments 	#TODO: take into account SANI requirements (distal activation must precede proximal activation) 
	globalFeatureNeuronsStrengthAllSegments = pt.sum(globalFeatureNeuronsStrength, dim=0)	#sum across all segments 	#TODO: take into account SANI requirements (distal activation must precede proximal activation) 
	if(useSANI and algorithmMatrixSANImethod=="enforceActivationAcrossSegments" and algorithmMatrixSANIenforceRequirement=="enforceLastSegmentMustBeActive"):
		# Patch: selection ignored last-segment gating, allowing nodes without last-segment activation to fire.
		if(enforceActivationAcrossSegmentsIgnoreInternalColumn):
			lastSegmentConstraint = arrayIndexSegmentAdjacentColumn
		else:
			lastSegmentConstraint = arrayIndexSegmentLast
		lastSegmentActivation = globalFeatureNeuronsActivation[lastSegmentConstraint]
		if(globalFeatureNeuronsActivationAllSegments.is_sparse):
			globalFeatureNeuronsActivationAllSegments = GIAANNproto_sparseTensors.selectAindicesContainedInB(globalFeatureNeuronsActivationAllSegments, lastSegmentActivation)
			if(globalFeatureNeuronsActivationAllSegments._nnz() == 0):
				raise RuntimeError("selectMostActiveFeature error: enforceLastSegmentMustBeActive requires active last-segment nodes, but none are active.")
		else:
			lastSegmentMask = lastSegmentActivation.to_dense() > 0
			globalFeatureNeuronsActivationAllSegments = globalFeatureNeuronsActivationAllSegments * lastSegmentMask
			if(not (globalFeatureNeuronsActivationAllSegments > 0).any().item()):
				raise RuntimeError("selectMostActiveFeature error: enforceLastSegmentMustBeActive requires active last-segment nodes, but none are active.")

	#topk column selection;
	conceptColumnsActivation = pt.sum(globalFeatureNeuronsActivationAllSegments, dim=1)	#sum across all feature activations in columns
	conceptColumnsActivation = conceptColumnsActivation.to_dense()	#convert to dense tensor (required for topk)
	if(inferenceNormaliseColumnSelectionByFeatureConnections):
		conceptColumnsActivationTotalConnections = pt.sum(globalFeatureNeuronsStrengthAllSegments, dim=1)	#sum across all feature activations in columns
		conceptColumnsActivationTotalConnections = conceptColumnsActivationTotalConnections.to_dense()
		if(not inferenceNormaliseColumnSelectionByFeatureConnectionsStrength):
			conceptColumnsActivationTotalConnections = (conceptColumnsActivationTotalConnections > 0).float()
		conceptColumnsActivation = conceptColumnsActivation / conceptColumnsActivationTotalConnections

	if(conceptColumnsDelimitByPOS):
		columnIndexLookup = pt.arange(conceptColumnsActivation.shape[0], dtype=pt.long, device=conceptColumnsActivation.device)
		if(allowedColumns is not None and allowedColumns.numel() > 0):
			allowedColumnsDevice = allowedColumns.to(columnIndexLookup.device)
		else:
			allowedColumnsDevice = None
		if(allowedColumnsDevice is not None and constraintMode is not None):
			if(constraintMode == "internal"):
				columnIndexLookup = allowedColumnsDevice
				columnActivationValues = conceptColumnsActivation.index_select(0, columnIndexLookup)
			elif(constraintMode == "external"):
				mask = pt.ones(conceptColumnsActivation.shape[0], dtype=pt.bool, device=conceptColumnsActivation.device)
				mask.scatter_(0, allowedColumnsDevice, False)
				filteredIndices = columnIndexLookup[mask]
				filteredActivations = conceptColumnsActivation[mask]
				if(filteredIndices.numel() > 0):
					columnIndexLookup = filteredIndices
					columnActivationValues = filteredActivations
				else:
					columnActivationValues = conceptColumnsActivation
			else:
				columnActivationValues = conceptColumnsActivation
		else:
			columnActivationValues = conceptColumnsActivation
		if(columnIndexLookup.numel() == 0):
			columnIndexLookup = pt.arange(conceptColumnsActivation.shape[0], dtype=pt.long, device=conceptColumnsActivation.device)
			columnActivationValues = conceptColumnsActivation
		if(kcDynamic):
			activeMask = columnActivationValues > kcActivationThreshold
			columnActivationValues = columnActivationValues[activeMask]
			columnIndexLookup = columnIndexLookup[activeMask]
		kcAvailable = columnActivationValues.shape[0]
		if(kcDynamic and kcAvailable < 1):
			print("selectMostActiveFeature kcDynamic error: kc < 1; cannot continue to predict columns; consider disabling kcDynamic for debug")
			exit()
		if(kcAvailable == 0):
			columnIndexLookup = pt.arange(conceptColumnsActivation.shape[0], dtype=pt.long, device=conceptColumnsActivation.device)
			columnActivationValues = conceptColumnsActivation
			kcAvailable = columnActivationValues.shape[0]
		topkCount = min(kcMax, kcAvailable)
		if(topkCount < 1):
			topkCount = 1
		conceptColumnsActivationTopkConcepts = pt.topk(columnActivationValues, topkCount)
		kc = len(conceptColumnsActivationTopkConcepts.indices)
		selectedColumnIndices = columnIndexLookup.index_select(0, conceptColumnsActivationTopkConcepts.indices)
	else:
		if(kcDynamic):
			conceptColumnsActivation = conceptColumnsActivation[conceptColumnsActivation > kcActivationThreshold]	#select kcMax columns above threshold
		conceptColumnsActivationTopkConcepts = pt.topk(conceptColumnsActivation, kcMax)
		kc = len(conceptColumnsActivationTopkConcepts.indices)
		if(kcDynamic and kc < 1):
			print("selectMostActiveFeature kcDynamic error: kc < 1; cannot continue to predict columns; consider disabling kcDynamic for debug")
			exit()
		selectedColumnIndices = conceptColumnsActivationTopkConcepts.indices

	#top feature selection;
	if(kc==1):
		topkConceptColumnsActivation = globalFeatureNeuronsActivationAllSegments[selectedColumnIndices[0]].unsqueeze(0)	#select topk concept indices
	else:
		topkConceptColumnsActivation = GIAANNproto_sparseTensors.sliceSparseTensorMulti(globalFeatureNeuronsActivationAllSegments, 0, selectedColumnIndices)	#select topk concept indices
	topkConceptColumnsActivation = topkConceptColumnsActivation.to_dense()
	if(inferenceNormaliseFeatureSelectionByFeatureConnections):
		if(kc==1):
			topkConceptColumnsStrength = globalFeatureNeuronsStrengthAllSegments[selectedColumnIndices[0]].unsqueeze(0)	#select topk concept indices
		else:
			topkConceptColumnsStrength = GIAANNproto_sparseTensors.sliceSparseTensorMulti(globalFeatureNeuronsStrengthAllSegments, 0, selectedColumnIndices)	#select topk concept indices
		topkConceptColumnsStrength = topkConceptColumnsStrength.to_dense()
		if(not inferenceNormaliseFeatureSelectionByFeatureConnectionsStrength):
			topkConceptColumnsStrength = (topkConceptColumnsStrength > 0).float()
		topkConceptColumnsActivation = topkConceptColumnsActivation / topkConceptColumnsStrength
	topkConceptColumnsActivationTopkFeatures = pt.topk(topkConceptColumnsActivation, kf, dim=1)

	conceptColumnsIndicesPred = selectedColumnIndices
	conceptColumnsFeatureIndicesPred = topkConceptColumnsActivationTopkFeatures.indices
	conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred = enforceMinimumPredictionActivationThreshold(conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, topkConceptColumnsActivationTopkFeatures.values)
	conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred = applyColumnConstraintToPredictions(databaseNetworkObject, conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, allowedColumns, constraintMode, connectedColumns, connectedColumnsFeatures)
	
	if(inferenceUseNextTokenPredictionsOrTargetsToActivateNextColumnFeatures):
		conceptColumnsIndicesNext = conceptColumnsIndicesPred
		conceptColumnsFeatureIndicesNext = conceptColumnsFeatureIndicesPred
		if(kc > 1 or kf > 1):
			multipleSourcesNext = True
		else:
			multipleSourcesNext = False
	else:
		#while exclusively training predictive network; use targets rather than next token predictions when activating database network
		conceptColumnsIndicesNext = targetConceptColumnsIndices
		conceptColumnsFeatureIndicesNext = targetConceptColumnsFeatureIndices
		multipleSourcesNext = targetMultipleSources
		if(multipleSourcesNext):
			kc = 2
		else:
			kc = 1
		assert kf==1
	conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext = applyColumnConstraintToPredictions(databaseNetworkObject, conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext, allowedColumns, constraintMode, connectedColumns, connectedColumnsFeatures)
	if(conceptColumnsIndicesNext is None or conceptColumnsIndicesNext.numel() == 0):
		multipleSourcesNext = False
		kc = 0
	else:
		kc = conceptColumnsIndicesNext.shape[0]
		
	return conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext, multipleSourcesNext, kc, conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, targetMultipleSources, targetPreviousColumnIndex, targetNextColumnIndex

def deactivatePredictedInhibitoryNeurons(databaseNetworkObject, conceptColumnsIndices, conceptColumnsFeatureIndices):
	indicesToUpdate = buildInhibitoryIndices(conceptColumnsIndices, conceptColumnsFeatureIndices)
	if(indicesToUpdate is None):
		return
	globalInhibitoryActivation = getattr(databaseNetworkObject, "globalInhibitoryNeuronsActivation", None)
	if(globalInhibitoryActivation is None):
		return
	globalInhibitoryActivation = GIAANNproto_sparseTensors.modifySparseTensor(globalInhibitoryActivation, indicesToUpdate, 0, multiply=False)
	databaseNetworkObject.globalInhibitoryNeuronsActivation = globalInhibitoryActivation

def buildInhibitoryIndices(conceptColumnsIndices, conceptColumnsFeatureIndices):
	if(conceptColumnsIndices is None or conceptColumnsFeatureIndices is None):
		return None
	if(conceptColumnsIndices.numel() == 0 or conceptColumnsFeatureIndices.numel() == 0):
		return None
	indicesToUpdateList = []
	for rowIndex in range(conceptColumnsIndices.shape[0]):
		columnTensor = conceptColumnsIndices[rowIndex]
		rowFeatures = conceptColumnsFeatureIndices[rowIndex]
		if(rowFeatures is None or rowFeatures.numel() == 0):
			continue
		featureValues = rowFeatures.reshape(-1)
		for featureTensor in featureValues:
			if(useSANI):
				for segmentIndex in range(arrayNumberOfSegments):
					indexToUpdate = pt.stack([pt.tensor(segmentIndex, device=columnTensor.device), columnTensor, featureTensor], dim=0)
					indicesToUpdateList.append(indexToUpdate)
			else:
				indexToUpdate = pt.stack([pt.tensor(arrayIndexSegmentFirst, device=columnTensor.device), columnTensor, featureTensor], dim=0)
				indicesToUpdateList.append(indexToUpdate)
	if(len(indicesToUpdateList) == 0):
		return None
	return pt.stack(indicesToUpdateList, dim=0)


def debugGetColumnName(databaseNetworkObject, columnIndex):
	if(0 <= columnIndex < len(databaseNetworkObject.conceptColumnsList)):
		return databaseNetworkObject.conceptColumnsList[columnIndex]
	return f"<invalid:{columnIndex}>"


def debugGetFeatureName(databaseNetworkObject, featureIndex):
	if(featureIndex == featureIndexConceptNeuron):
		return "conceptNeuron"
	if(0 <= featureIndex < len(databaseNetworkObject.conceptFeaturesList)):
		return databaseNetworkObject.conceptFeaturesList[featureIndex]
	return f"<invalid:{featureIndex}>"


def debugDescribeColumnFeatures(databaseNetworkObject, columnIndices, featureIndices=None):
	if(columnIndices is None or columnIndices.numel() == 0):
		return []
	descriptions = []
	columnValues = columnIndices.detach().cpu().tolist()
	for rowIndex, columnValue in enumerate(columnValues):
		columnName = debugGetColumnName(databaseNetworkObject, int(columnValue))
		featureDescList = []
		if(featureIndices is not None and featureIndices.shape[0] > rowIndex):
			rowTensor = featureIndices[rowIndex]
			if(rowTensor is not None and rowTensor.numel() > 0):
				if(rowTensor.dim() == 0):
					rowValues = [int(rowTensor.item())]
				else:
					rowValues = [int(value.item()) for value in rowTensor.reshape(-1)]
				for featureValue in rowValues:
					featureDescList.append(f"{featureValue}:{debugGetFeatureName(databaseNetworkObject, featureValue)}")
		if(len(featureDescList) == 0):
			featureDesc = "[]"
		else:
			featureDesc = "[" + ", ".join(featureDescList) + "]"
		descriptions.append(f"{columnName}{featureDesc}")
	return descriptions


def debugDescribeFeatureSet(databaseNetworkObject, featureSet):
	if(featureSet is None or len(featureSet) == 0):
		return "[]"
	featureDescriptions = [f"{feature}:{debugGetFeatureName(databaseNetworkObject, feature)}" for feature in sorted(featureSet)]
	return "[" + ", ".join(featureDescriptions) + "]"
