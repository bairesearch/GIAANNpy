"""GIAANNproto_predictiveNetwork.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_predictiveNetwork.py

# Usage:
see GIAANNproto_predictiveNetwork.py

# Description:
GIA ANN proto predictive Network

"""

import torch as pt

from GIAANNproto_globalDefs import *
import GIAANNproto_databaseNetwork
import GIAANNproto_databaseNetworkTrain
if(inferencePredictiveNetwork):
	if(inferencePredictiveNetworkModel=="ColumnMLP"):
		import GIAANNproto_predictiveNetworkModelColumnMLP as GIAANNproto_predictiveNetworkModel
	elif(inferencePredictiveNetworkModel=="MLP"):
		import GIAANNproto_predictiveNetworkModelMLP as GIAANNproto_predictiveNetworkModel
	elif(inferencePredictiveNetworkModel=="Transformer"):
		import GIAANNproto_predictiveNetworkModelTransformer as GIAANNproto_predictiveNetworkModel
	import GIAANNproto_predictiveNetworkOperations
import GIAANNproto_databaseNetworkDraw
import GIAANNproto_sparseTensors
import GIAANNproto_predictiveNetworkBeamSearch

def inferenceSavePredictiveNetwork():
	GIAANNproto_predictiveNetworkModel.saveModel(predictiveNetworkFolder, predictiveNetworkFileName)

def initialisePredictiveNetwork(databaseNetworkObject):
	GIAANNproto_predictiveNetworkModel.nextWordPredictionModelCreate(databaseNetworkObject)

# Define the SequenceObservedColumnsInferencePrediction class
class SequenceObservedColumnsInferencePrediction:
	def __init__(self, databaseNetworkObject, words, lemmas, observedColumnsDict, observedColumnsSequenceWordIndexDict):
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
			if(not GIAANNproto_databaseNetwork.isFeatureIndexReferenceSetDelimiter(databaseNetworkObject, featureIndex)):
				return False
		return True
	else:
		return False

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
			return conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred
		fallbackColumns = []
		for columnIndex in range(databaseNetworkObject.c):
			if(columnIndex not in allowedSet):
				fallbackColumns.append(columnIndex)
			if(len(fallbackColumns) == conceptColumnsIndicesPred.shape[0]):
				break
		if(len(fallbackColumns) == 0):
			return conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred
		conceptColumnsIndicesPred = pt.tensor(fallbackColumns, dtype=conceptColumnsIndicesPred.dtype, device=conceptColumnsIndicesPred.device)
		if(conceptColumnsFeatureIndicesPred is not None):
			rowsAvailable = min(conceptColumnsFeatureIndicesPred.shape[0], len(fallbackColumns))
		conceptColumnsFeatureIndicesPred = conceptColumnsFeatureIndicesPred[:rowsAvailable]
		return applyConnectedColumnsConstraint(conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, connectedColumns, connectedColumnsFeatures)
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
				if(GIAANNproto_databaseNetwork.isFeatureIndexReferenceSetDelimiter(databaseNetworkObject, int(featureValue))):
					indicesToKeep.append(idx)
		if(len(indicesToKeep) == 0):
			fallbackColumns = []
			for columnIndex in range(databaseNetworkObject.c):
				if(columnIndex not in allowedSet):
					fallbackColumns.append(columnIndex)
				if(len(fallbackColumns) == conceptColumnsIndicesPred.shape[0]):
					break
			if(len(fallbackColumns) == 0):
				conceptColumnsIndicesPred = conceptColumnsIndicesPred[:0]
				if(conceptColumnsFeatureIndicesPred is not None):
					conceptColumnsFeatureIndicesPred = conceptColumnsFeatureIndicesPred[:0]
				return conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred
			conceptColumnsIndicesPred = pt.tensor(fallbackColumns, dtype=conceptColumnsIndicesPred.dtype, device=conceptColumnsIndicesPred.device)
			if(conceptColumnsFeatureIndicesPred is not None):
				rowsAvailable = min(conceptColumnsFeatureIndicesPred.shape[0], len(fallbackColumns))
				conceptColumnsFeatureIndicesPred = conceptColumnsFeatureIndicesPred[:rowsAvailable]
			return conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred
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


def getTargetWordForSequenceIndex(wordsSequence, sequenceWordIndex):
	if(wordsSequence is None):
		return "<unknown>"
	if(sequenceWordIndex < 0 or sequenceWordIndex >= len(wordsSequence)):
		return "<unknown>"
	return wordsSequence[sequenceWordIndex]


def raisePredictionConnectivityError(sequenceWordIndex, wordPredictionIndex, wordsSequence, reason):
	targetWord = getTargetWordForSequenceIndex(wordsSequence, sequenceWordIndex)
	message = f"predictionEnsureConnectedToPreviousPrediction violation: {reason}. sequenceWordIndex={sequenceWordIndex}, wordPredictionIndex={wordPredictionIndex}, targetWord='{targetWord}'"
	raise RuntimeError(message)


def ensurePredictionStateAvailable(conceptColumnsIndices, conceptColumnsFeatureIndices, sequenceWordIndex, wordPredictionIndex, wordsSequence, reason):
	if(conceptColumnsIndices is None or conceptColumnsIndices.numel() == 0):
		raisePredictionConnectivityError(sequenceWordIndex, wordPredictionIndex, wordsSequence, reason + " (missing concept columns)")
	if(conceptColumnsFeatureIndices is None or conceptColumnsFeatureIndices.numel() == 0):
		raisePredictionConnectivityError(sequenceWordIndex, wordPredictionIndex, wordsSequence, reason + " (missing column features)")


def rowHasAllowedFeature(conceptColumnsFeatureIndicesPred, rowIndex, allowedFeaturesSet):
	if(conceptColumnsFeatureIndicesPred is None):
		return False
	if(conceptColumnsFeatureIndicesPred.dim() == 1):
		if(rowIndex >= conceptColumnsFeatureIndicesPred.shape[0]):
			return False
		return int(conceptColumnsFeatureIndicesPred[rowIndex].item()) in allowedFeaturesSet
	if(rowIndex >= conceptColumnsFeatureIndicesPred.shape[0]):
		return False
	rowTensor = conceptColumnsFeatureIndicesPred[rowIndex]
	if(rowTensor is None or rowTensor.numel() == 0):
		return False
	rowValues = rowTensor.view(-1)
	for value in rowValues:
		if(int(value.item()) in allowedFeaturesSet):
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
	if(not predictionEnsureConnectedToPreviousPrediction):
		return None, None
	if(conceptColumnsIndices is None or conceptColumnsFeatureIndices is None):
		return None, None
	if(conceptColumnsIndices.numel() == 0 or conceptColumnsFeatureIndices.numel() == 0):
		device = conceptColumnsIndices.device
		dtype = conceptColumnsIndices.dtype
		return pt.empty(0, dtype=dtype, device=device), None
	connectedColumnsSet = set()
	if(debugConnectNodesToNextNodesInSequenceOnly):
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
			targetColumns, targetColumnFeatures = getConnectedColumnsForFeature(observedColumn, int(featureValue), includeFeatureDetails=debugConnectNodesToNextNodesInSequenceOnly)
			connectedColumnsSet.update(targetColumns)
			if(debugConnectNodesToNextNodesInSequenceOnly and targetColumnFeatures is not None):
				for targetColumnIndex, featureSet in targetColumnFeatures.items():
					if(targetColumnIndex < 0 or targetColumnIndex >= databaseNetworkObject.c):
						continue
					columnFeatureSet = connectedColumnsFeatures.setdefault(targetColumnIndex, set())
					columnFeatureSet.update(featureSet)
	if(len(connectedColumnsSet) == 0):
		device = conceptColumnsIndices.device
		dtype = conceptColumnsIndices.dtype
		return pt.empty(0, dtype=dtype, device=device), ({} if debugConnectNodesToNextNodesInSequenceOnly else None)
	validColumns = [col for col in connectedColumnsSet if col >= 0 and col < databaseNetworkObject.c]
	if(len(validColumns) == 0):
		device = conceptColumnsIndices.device
		dtype = conceptColumnsIndices.dtype
		return pt.empty(0, dtype=dtype, device=device), ({} if debugConnectNodesToNextNodesInSequenceOnly else None)
	validColumns.sort()
	device = conceptColumnsIndices.device
	dtype = conceptColumnsIndices.dtype
	connectedColumnsTensor = pt.tensor(validColumns, dtype=dtype, device=device)
	if(debugConnectNodesToNextNodesInSequenceOnly):
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
	observedColumn = GIAANNproto_databaseNetwork.loadOrCreateObservedColumn(databaseNetworkObject, columnIndex, columnLemma, columnIndex)
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
	minWordDistanceMask = GIAANNproto_databaseNetwork.computeConnectionMinWordDistanceMask(observedColumn, featureIndex, targetColumnIndices)
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
		for featureValue in rowValues:
			if(featureValue == featureIndexConceptNeuron):
				conceptActivationState.add(int(columnValue))
				break
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
	updatedRows = []
	modified = False
	for rowIndex in range(featureIndices.shape[0]):
		columnIndex = int(conceptColumnsIndices[rowIndex].item())
		rowTensor = featureIndices[rowIndex].clone()
		if(columnIndex not in localState and rowTensor.numel() > 0):
			rowTensor.fill_(featureIndexConceptNeuron)
			modified = True
		updatedRows.append(rowTensor)
	if(not modified):
		return conceptColumnsIndices, conceptColumnsFeatureIndices
	updatedTensor = pt.stack(updatedRows, dim=0)
	if(originally1D):
		updatedTensor = updatedTensor.squeeze(1)
	return conceptColumnsIndices, updatedTensor

if not drawSequenceObservedColumns:
	class SequenceObservedColumnsDraw:
		def __init__(self, databaseNetworkObject, observedColumnsDict):
			self.databaseNetworkObject = databaseNetworkObject
			self.observedColumnsDict = observedColumnsDict

def seedNetwork(sequenceObservedColumns, sequenceIndex, sequence, firstSeedTokenIndex, numSeedTokens):
	words, lemmas, posTags = GIAANNproto_databaseNetworkTrain.getLemmas(sequence)
	if(inferenceIncrementallySeedNetwork):
		print("\t seedNetwork: seedTokenIndex = ", firstSeedTokenIndex, ", word = ", words[firstSeedTokenIndex])
	else:
		print("\t seedNetwork: firstSeedTokenIndex = ", firstSeedTokenIndex, ", words = ", words[firstSeedTokenIndex:numSeedTokens])
	GIAANNproto_databaseNetworkTrain.processConceptWords(sequenceObservedColumns, sequenceIndex, sequence, words, lemmas, posTags, train=False, firstSeedTokenIndex=firstSeedTokenIndex, numSeedTokens=numSeedTokens)

	if(inferenceDecrementActivations):
		if(not inferenceSeedTargetActivationsGlobalFeatureArrays):
			globalFeatureNeuronsActivation = sequenceObservedColumns.databaseNetworkObject.globalFeatureNeurons[arrayIndexPropertiesActivation]
			globalFeatureNeuronsActivation = GIAANNproto_databaseNetworkTrain.decrementActivation(globalFeatureNeuronsActivation, activationDecrementSeed)
			sequenceObservedColumns.databaseNetworkObject.globalFeatureNeurons = GIAANNproto_sparseTensors.replaceAllSparseTensorElementsAtFirstDimIndex(sequenceObservedColumns.databaseNetworkObject.globalFeatureNeurons, globalFeatureNeuronsActivation, arrayIndexPropertiesActivation)
	
	if(drawNetworkDuringInferenceSeed):
		#FUTURE: convert globalFeatureNeuronsActivation back to globalFeatureNeurons for draw
		GIAANNproto_databaseNetworkDraw.visualize_graph(sequenceObservedColumns, save=drawNetworkDuringInferenceSave, fileName=drawNetworkDuringInferenceSaveFilenamePrepend+str(firstSeedTokenIndex))


def processConceptWordsInference(sequenceObservedColumns, sequenceIndex, sequence, sequenceSeed, sequencePredict, numSeedTokens):

	print("processConceptWordsInference:")

	sequenceWordIndex = 0
	
	wordsSequence, lemmasSequence, posTagsSequence = GIAANNproto_databaseNetworkTrain.getLemmas(sequence)
	conceptMask, conceptIndices, numberConcepts = GIAANNproto_databaseNetworkTrain.createConceptMask(sequenceObservedColumns, lemmasSequence)
	
	if(transformerUseInputConnections):
		GIAANNproto_databaseNetwork.generateGlobalFeatureConnections(sequenceObservedColumns.databaseNetworkObject)
	
	if(inferenceSeedNetwork):
		#seed network;
		if(inferenceIncrementallySeedNetwork):
			for seedTokenIndex in range(numSeedTokens):
				seedNetwork(sequenceObservedColumns, sequenceIndex, sequence, seedTokenIndex, 1)
		else:
			seedNetwork(sequenceObservedColumns, sequenceIndex, sequence, 0, numSeedTokens)

		if(not inferenceSeedTargetActivationsGlobalFeatureArrays):
			# Update observed columns from sequence observed columns
			sequenceObservedColumns.updateObservedColumnsWrapper()	#convert sequence observed columns feature neuron arrays back to global feature neuron arrays
	elif(inferenceBeamSearch and numSeedTokens > 0):
		#beam search requires prompt activations even when inferenceSeedNetwork is disabled
		if(inferenceIncrementallySeedNetwork):
			for seedTokenIndex in range(numSeedTokens):
				seedNetwork(sequenceObservedColumns, sequenceIndex, sequence, seedTokenIndex, 1)
		else:
			seedNetwork(sequenceObservedColumns, sequenceIndex, sequence, 0, numSeedTokens)
		if(not inferenceSeedTargetActivationsGlobalFeatureArrays):
			sequenceObservedColumns.updateObservedColumnsWrapper()
	
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
	initialContextWordIndex = max(0, min(initialContextWordIndex, len(wordsSequence)-1))
	multipleSources, targetPreviousColumnIndex, targetNextColumnIndex, targetFeatureIndex, conceptColumnsIndices, conceptColumnsFeatureIndices = GIAANNproto_databaseNetwork.getTokenConceptFeatureIndexTensor(sequenceObservedColumns, wordsSequence, lemmasSequence, conceptMask, initialContextWordIndex, kcMax)
	conceptActivationState = initialiseConceptActivationState(conceptColumnsIndices, conceptColumnsFeatureIndices)
	observedColumnsDict = sequenceObservedColumns.observedColumnsDict  # key: lemma, value: ObservedColumn	#every observed column in inference (seed and prediction phases)
	
	#predict next tokens;
	for wordPredictionIndex in range(numPredictionTokens):
		sequenceWordIndex = numSeedTokens + wordPredictionIndex
		featurePredictionTargetMatch, conceptColumnsIndices, conceptColumnsFeatureIndices, multipleSources, conceptActivationState = processColumnInferencePrediction(sequenceObservedColumns, sequenceIndex, observedColumnsDict, wordPredictionIndex, sequenceWordIndex, wordsSequence, lemmasSequence, conceptColumnsIndices, conceptColumnsFeatureIndices, conceptMask, multipleSources, conceptActivationState)
		

def processColumnInferencePrediction(sequenceObservedColumns, sequenceIndex, observedColumnsDict, wordPredictionIndex, sequenceWordIndex, wordsSequence, lemmasSequence, conceptColumnsIndices, conceptColumnsFeatureIndices, conceptMask, multipleSources, conceptActivationState):
	
	databaseNetworkObject = sequenceObservedColumns.databaseNetworkObject
	
	#print(f"processColumnInferencePrediction: {sequenceWordIndex}; conceptColumnsIndices = ", conceptColumnsIndices)

	if(predictionEnsureConnectedToPreviousPrediction):
		ensurePredictionStateAvailable(conceptColumnsIndices, conceptColumnsFeatureIndices, sequenceWordIndex, wordPredictionIndex, wordsSequence, "no connected context available before prediction")

	#burst the initial seed in the sequence
	if(sequenceWordIndex==0 or inferenceBurstAllPredictionsOrTargetsInSequence):
		#activate source token (incremental seed during train)
			#if(wordPredictionIndex == 1) will reactivate first seed token column feature (as it was not saved during wordPredictionIndex==0)
		for conceptIndex in range(conceptColumnsIndices.shape[0]):
			conceptColumnsIndicesSource = conceptColumnsIndices[conceptIndex].item()
			conceptColumnsFeatureIndicesSource = conceptColumnsFeatureIndices[conceptIndex].squeeze().item()
			indicesToUpdateList = [arrayIndexPropertiesActivation, arrayIndexSegmentInternalColumn, conceptColumnsIndicesSource, conceptColumnsFeatureIndicesSource]
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
	if(conceptColumnsDelimitByPOS):
		allowedColumnsConstraint = buildAllowedColumnsLookup(conceptColumnsIndices, databaseNetworkObject.c)
		if(allowedColumnsConstraint is not None and allowedColumnsConstraint.numel() > 0):
			if(conceptColumnsFeatureIndices is None or conceptColumnsFeatureIndices.numel() == 0):
				constraintModePrediction = "internal"
			else:
				isDelimiterNode = activatedNodesAreReferenceSetDelimiters(databaseNetworkObject, conceptColumnsFeatureIndices)
				if(isDelimiterNode):
					constraintModePrediction = "delimiter"
				else:
					constraintModePrediction = "internal"
	if(predictionEnsureConnectedToPreviousPrediction):
		#limit prediction candidates to columns directly connected to previously predicted nodes
		connectedColumnsConstraint, connectedColumnsFeatureMap = buildConnectedColumnsLookupFromPrediction(databaseNetworkObject, observedColumnsDict, conceptColumnsIndices, conceptColumnsFeatureIndices)
	else:
		connectedColumnsConstraint = None
		connectedColumnsFeatureMap = None
	if(predictionEnsureConnectedToPreviousPrediction and connectedColumnsConstraint is not None and connectedColumnsConstraint.numel() == 0):
		raisePredictionConnectivityError(sequenceWordIndex, wordPredictionIndex, wordsSequence, "previous prediction has no outgoing connections")
		
	if(wordPredictionIndex > 0):
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
			observedColumn = GIAANNproto_databaseNetwork.loadOrCreateObservedColumn(databaseNetworkObject, conceptIndexVal, lemma, sequenceWordIndex)
			observedColumnsDict[lemma] = observedColumn
			observedColumnsSequenceCandidateIndexDict[i] = observedColumn
		sequenceObservedColumnsPrediction = SequenceObservedColumnsInferencePrediction(databaseNetworkObject, words, lemmas, observedColumnsDict, observedColumnsSequenceCandidateIndexDict)
		
		#decrement activations;
		if(inferenceDecrementActivations):
			#decrement activation after each prediction interval
			globalFeatureNeuronsActivation = GIAANNproto_databaseNetworkTrain.decrementActivation(globalFeatureNeuronsActivation, activationDecrementPerPredictedToken)
			if(transformerUseInputConnections):
				globalFeatureConnectionsActivation = GIAANNproto_databaseNetworkTrain.decrementActivation(globalFeatureConnectionsActivation, activationDecrementPerPredictedToken)
			if(inferenceUseNeuronFeaturePropertiesTime):
				globalFeatureNeuronsTime = GIAANNproto_databaseNetworkTrain.decrementActivation(globalFeatureNeuronsTime, inferenceUseNeuronFeaturePropertiesTimeDecrement)
				
		#process features (activate global target neurons);
		if(multipleSources):
			globalFeatureNeuronsActivation, globalFeatureConnectionsActivation = GIAANNproto_databaseNetworkTrain.processFeaturesActivePredictMulti(databaseNetworkObject, globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, sequenceObservedColumnsPrediction, conceptColumnsIndices, conceptColumnsFeatureIndices)
		else:
			globalFeatureNeuronsActivation, globalFeatureConnectionsActivation = GIAANNproto_databaseNetworkTrain.processFeaturesActivePredictSingle(databaseNetworkObject, globalFeatureNeuronsActivation, globalFeatureConnectionsActivation, sequenceObservedColumnsPrediction, conceptColumnsIndices, conceptColumnsFeatureIndices)

		if(inferenceDeactivateNeuronsUponPrediction or inferenceInvertNeuronActivationUponPrediction):
			indicesToUpdateList = []
			for conceptIndex in range(conceptColumnsIndices.shape[0]):
				conceptColumnsIndicesSource = conceptColumnsIndices[conceptIndex]
				conceptColumnsFeatureIndicesSource = conceptColumnsFeatureIndices[conceptIndex].squeeze(dim=0)
				if(useSANI):
					for segmentIndex in range(arrayNumberOfSegments):
						indexToUpdate = pt.stack([pt.tensor(segmentIndex, device=conceptColumnsIndicesSource.device), conceptColumnsIndicesSource, conceptColumnsFeatureIndicesSource], dim=0)
						indicesToUpdateList.append(indexToUpdate)
				else:
					indicesToUpdate = pt.stack([pt.tensor(arrayIndexSegmentFirst, device=conceptColumnsIndicesSource.device), conceptColumnsIndicesSource, conceptColumnsFeatureIndicesSource], dim=0)
					indicesToUpdateList.append(indicesToUpdate)
			indicesToUpdate = pt.stack(indicesToUpdateList, dim=0)
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
	else:
		#activation targets have already been activated
		sequenceObservedColumnsPrediction = SequenceObservedColumnsDraw(databaseNetworkObject, observedColumnsDict)
	
	if(debugInferencePredictionActivationAccumulation):
		globalFeatureNeuronsTemp = databaseNetworkObject.globalFeatureNeurons.to_dense()
		print("globalFeatureNeuronsTemp = ", globalFeatureNeuronsTemp)

	if(inferenceBeamSearch):
		conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext, multipleSourcesNext, kc, conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, targetMultipleSources, targetPreviousColumnIndex, targetNextColumnIndex = GIAANNproto_predictiveNetworkBeamSearch.beamSearchPredictNextFeature(sequenceObservedColumns, databaseNetworkObject, observedColumnsDict, globalFeatureNeuronsActivation, globalFeatureNeuronsStrength, globalFeatureConnectionsActivation, globalFeatureNeuronsTime, wordsSequence, lemmasSequence, wordPredictionIndex, sequenceWordIndex, conceptMask, allowedColumnsConstraint, constraintModePrediction, selectMostActiveFeature, conceptActivationState, connectedColumnsConstraint, connectedColumnsFeatureMap)
	else:
		if(inferencePredictiveNetwork):
			conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext, multipleSourcesNext, kc, conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, targetMultipleSources, targetPreviousColumnIndex, targetNextColumnIndex = predictMostActiveFeature(sequenceObservedColumns, databaseNetworkObject, wordsSequence, lemmasSequence, wordPredictionIndex, sequenceWordIndex, conceptMask, allowedColumnsConstraint, constraintModePrediction, conceptActivationState, connectedColumnsConstraint, connectedColumnsFeatureMap)	
		else:
			conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext, multipleSourcesNext, kc, conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, targetMultipleSources, targetPreviousColumnIndex, targetNextColumnIndex = selectMostActiveFeature(sequenceObservedColumns, globalFeatureNeuronsActivation, globalFeatureNeuronsStrength, wordsSequence, lemmasSequence, wordPredictionIndex, sequenceWordIndex, conceptMask, allowedColumnsConstraint, constraintModePrediction, conceptActivationState, connectedColumnsConstraint, connectedColumnsFeatureMap)

	if(predictionEnsureConnectedToPreviousPrediction and connectedColumnsConstraint is not None):
		conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred = applyConnectedColumnsConstraint(conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, connectedColumnsConstraint, connectedColumnsFeatureMap)
		conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext = applyConnectedColumnsConstraint(conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext, connectedColumnsConstraint, connectedColumnsFeatureMap)
		if(conceptColumnsIndicesPred is None or conceptColumnsIndicesPred.numel() == 0):
			raisePredictionConnectivityError(sequenceWordIndex, wordPredictionIndex, wordsSequence, "no connected predictions available for current step")
		if(conceptColumnsIndicesNext is None or conceptColumnsIndicesNext.numel() == 0):
			raisePredictionConnectivityError(sequenceWordIndex, wordPredictionIndex, wordsSequence, "no connected activations available for next step")
	
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
			
			targetWord = wordsSequence[sequenceWordIndex]
			if(targetMultipleSources):
				targetColumnName = databaseNetworkObject.conceptColumnsList[targetPreviousColumnIndex] + "/" + databaseNetworkObject.conceptColumnsList[targetNextColumnIndex]
			else:
				targetColumnName = databaseNetworkObject.conceptColumnsList[targetPreviousColumnIndex]
			
			print("\t sequenceWordIndex = ", sequenceWordIndex, ", wordPredictionIndex = ", wordPredictionIndex, ", targetWord = ", targetWord, ", predictedWord = ", predictedWord, ", targetColumn = ", targetColumnName, ", predictedColumn = ", predictedColumnName)
			if(targetWord == predictedWord):
				featurePredictionTargetMatch = True
	
	if(drawNetworkDuringInferencePredict):
		#FUTURE: convert globalFeatureNeuronsActivation back to globalFeatureNeurons for draw
		GIAANNproto_databaseNetworkDraw.visualizeGraph(sequenceObservedColumnsPrediction, True, save=drawNetworkDuringInferenceSave, fileName=drawNetworkDuringInferenceSaveFilenamePrepend+str(sequenceWordIndex))
	if(predictionColumnsMustActivateConceptFeature):
		conceptActivationState = updateConceptActivationState(conceptActivationState, conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext)

	return featurePredictionTargetMatch, conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext, multipleSourcesNext, conceptActivationState

def predictMostActiveFeature(sequenceObservedColumns, databaseNetworkObject, wordsSequence, lemmasSequence, wordPredictionIndex, sequenceWordIndex, conceptMask, allowedColumns=None, constraintMode=None, conceptActivationState=None, connectedColumns=None, connectedColumnsFeatures=None):		
	#generate targets;
	targetMultipleSources, targetPreviousColumnIndex, targetNextColumnIndex, targetFeatureIndex, targetConceptColumnsIndices, targetConceptColumnsFeatureIndices = GIAANNproto_databaseNetwork.getTokenConceptFeatureIndexTensor(sequenceObservedColumns, wordsSequence, lemmasSequence, conceptMask, sequenceWordIndex, kcNetwork)
	
	if(inferencePredictiveNetworkIndependentFCpredictions):
		targets = None
		targetsC = pt.zeros(databaseNetworkObject.c)
		targetsF = pt.zeros(databaseNetworkObject.f)
		targetsC[targetPreviousColumnIndex] = 1
		targetsF[targetFeatureIndex] = 1
		if(targetMultipleSources):
			targetsC[targetNextColumnIndex] = 1	
	else:
		targets = pt.zeros(databaseNetworkObject.c, databaseNetworkObject.f)
		targets[targetPreviousColumnIndex, targetFeatureIndex] = 1
		if(targetMultipleSources):
			targets[targetNextColumnIndex, targetFeatureIndex] = 1
		targetsC = None
		targetsF = None
	
	globalFeatureConnections = None
	if(inferencePredictiveNetworkUseInputAllProperties):
		globalFeatureNeurons = databaseNetworkObject.globalFeatureNeurons
		if(transformerUseInputConnections):
			globalFeatureConnections = databaseNetworkObject.globalFeatureConnections
	else:
		globalFeatureNeurons = databaseNetworkObject.globalFeatureNeurons[arrayIndexPropertiesActivation]
		if(transformerUseInputConnections):
			globalFeatureConnections = databaseNetworkObject.globalFeatureConnections[arrayIndexPropertiesActivation]
	
	if(inferencePredictiveNetworkNormaliseInputs):
		#if(not useGPUpredictiveNetworkModel and inferencePredictiveNetworkNormaliseDim==0):
		#	globalFeatureNeurons = GIAANNproto_predictiveNetworkOperations.normaliseSparseTensor(globalFeatureNeurons, inferencePredictiveNetworkUseInputAllProperties)
		if(transformerUseInputConnections):	#globalFeatureConnections are currently retained on CPU
			globalFeatureConnections = GIAANNproto_predictiveNetworkOperations.normaliseSparseTensor(globalFeatureConnections, inferencePredictiveNetworkUseInputAllProperties)
			if(inferencePredictiveNetworkNormaliseDim != 0):
				print("predictMostActiveFeature warning: inferencePredictiveNetworkNormaliseDim>0 - can only normalise globalFeatureConnections along first dimension (properties)")
				
	if(inferencePredictiveNetworkModel=="ColumnMLP"):
		#GIAANNproto_predictiveNetworkModel.ensureModelMatchesDatabase(databaseNetworkObject)
		conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred = GIAANNproto_predictiveNetworkModel.nextWordPredictionColumnMLPtrainStep(globalFeatureNeurons, targets, targetsC, targetsF)
	elif(inferencePredictiveNetworkModel=="MLP"):
		conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred = GIAANNproto_predictiveNetworkModel.nextWordPredictionMLPtrainStep(globalFeatureNeurons, targets, targetsC, targetsF)
	elif(inferencePredictiveNetworkModel=="Transformer"):
		conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred = GIAANNproto_predictiveNetworkModel.nextWordPredictionTransformerTrainStep(globalFeatureNeurons, globalFeatureConnections, targets, targetsC, targetsF)

	conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred = enforceConceptFeaturePredictionOrder(conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, conceptActivationState)
	conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred = applyColumnConstraintToPredictions(databaseNetworkObject, conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, allowedColumns, constraintMode, connectedColumns, connectedColumnsFeatures)

	if(inferenceUseNextTokenPredictionsOrTargetsToActivateNextColumnFeatures):
		conceptColumnsIndicesNext = conceptColumnsIndicesPred
		conceptColumnsFeatureIndicesNext = conceptColumnsFeatureIndicesPred
		conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext = enforceConceptFeaturePredictionOrder(conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext, conceptActivationState)
		kc = kcNetwork
		if(kc == 1 and kf == 1):
			multipleSourcesNext = False
		else:
			multipleSourcesNext = True
	else:
		#while exclusively training predictive network; use targets rather than next token predictions when activating database network
		conceptColumnsIndicesNext = targetConceptColumnsIndices
		conceptColumnsFeatureIndicesNext = targetConceptColumnsFeatureIndices
		#print("targetConceptColumnsIndices = ", targetConceptColumnsIndices)
		#print("targetConceptColumnsFeatureIndices = ", targetConceptColumnsFeatureIndices)
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




def selectMostActiveFeature(sequenceObservedColumns, globalFeatureNeuronsActivation, globalFeatureNeuronsStrength, wordsSequence, lemmasSequence, wordPredictionIndex, sequenceWordIndex, conceptMask, allowedColumns=None, constraintMode=None, conceptActivationState=None, connectedColumns=None, connectedColumnsFeatures=None):
	databaseNetworkObject = sequenceObservedColumns.databaseNetworkObject
	#generate targets;
	targetMultipleSources, targetPreviousColumnIndex, targetNextColumnIndex, targetFeatureIndex, targetConceptColumnsIndices, targetConceptColumnsFeatureIndices = GIAANNproto_databaseNetwork.getTokenConceptFeatureIndexTensor(sequenceObservedColumns, wordsSequence, lemmasSequence, conceptMask, sequenceWordIndex, kcNetwork)

	globalFeatureNeuronsActivationAllSegments = pt.sum(globalFeatureNeuronsActivation, dim=0)	#sum across all segments 	#TODO: take into account SANI requirements (distal activation must precede proximal activation) 
	globalFeatureNeuronsStrengthAllSegments = pt.sum(globalFeatureNeuronsStrength, dim=0)	#sum across all segments 	#TODO: take into account SANI requirements (distal activation must precede proximal activation) 

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
	conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred = enforceConceptFeaturePredictionOrder(conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, conceptActivationState)
	conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred = applyColumnConstraintToPredictions(databaseNetworkObject, conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, allowedColumns, constraintMode, connectedColumns, connectedColumnsFeatures)
	
	if(inferenceUseNextTokenPredictionsOrTargetsToActivateNextColumnFeatures):
		conceptColumnsIndicesNext = conceptColumnsIndicesPred
		conceptColumnsFeatureIndicesNext = conceptColumnsFeatureIndicesPred
		conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext = enforceConceptFeaturePredictionOrder(conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext, conceptActivationState)
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
