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
	multipleSources, targetPreviousColumnIndex, targetNextColumnIndex, targetFeatureIndex, conceptColumnsIndices, conceptColumnsFeatureIndices = GIAANNproto_databaseNetwork.getTokenConceptFeatureIndexTensor(sequenceObservedColumns, wordsSequence, lemmasSequence, conceptMask, 0, kcMax)
	observedColumnsDict = sequenceObservedColumns.observedColumnsDict  # key: lemma, value: ObservedColumn	#every observed column in inference (seed and prediction phases)
	
	#predict next tokens;
	for wordPredictionIndex in range(numPredictionTokens):
		sequenceWordIndex = numSeedTokens + wordPredictionIndex
		featurePredictionTargetMatch, conceptColumnsIndices, conceptColumnsFeatureIndices, multipleSources = processColumnInferencePrediction(sequenceObservedColumns, sequenceIndex, observedColumnsDict, wordPredictionIndex, sequenceWordIndex, wordsSequence, lemmasSequence, conceptColumnsIndices, conceptColumnsFeatureIndices, conceptMask, multipleSources)
		

def processColumnInferencePrediction(sequenceObservedColumns, sequenceIndex, observedColumnsDict, wordPredictionIndex, sequenceWordIndex, wordsSequence, lemmasSequence, conceptColumnsIndices, conceptColumnsFeatureIndices, conceptMask, multipleSources):
	
	databaseNetworkObject = sequenceObservedColumns.databaseNetworkObject
	
	#print(f"processColumnInferencePrediction: {sequenceWordIndex}; conceptColumnsIndices = ", conceptColumnsIndices)

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
		conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext, multipleSourcesNext, kc, conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, targetMultipleSources, targetPreviousColumnIndex, targetNextColumnIndex = GIAANNproto_predictiveNetworkBeamSearch.beamSearchPredictNextFeature(sequenceObservedColumns, databaseNetworkObject, observedColumnsDict, globalFeatureNeuronsActivation, globalFeatureNeuronsStrength, globalFeatureConnectionsActivation, globalFeatureNeuronsTime, wordsSequence, lemmasSequence, wordPredictionIndex, sequenceWordIndex, conceptMask)
	else:
		if(inferencePredictiveNetwork):
			conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext, multipleSourcesNext, kc, conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, targetMultipleSources, targetPreviousColumnIndex, targetNextColumnIndex = predictMostActiveFeature(sequenceObservedColumns, databaseNetworkObject, wordsSequence, lemmasSequence, wordPredictionIndex, sequenceWordIndex, conceptMask)	
		else:
			conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext, multipleSourcesNext, kc, conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, targetMultipleSources, targetPreviousColumnIndex, targetNextColumnIndex = selectMostActiveFeature(sequenceObservedColumns, globalFeatureNeuronsActivation, globalFeatureNeuronsStrength, wordsSequence, lemmasSequence, wordPredictionIndex, sequenceWordIndex, conceptMask)
	
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

	return featurePredictionTargetMatch, conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext, multipleSourcesNext

def predictMostActiveFeature(sequenceObservedColumns, databaseNetworkObject, wordsSequence, lemmasSequence, wordPredictionIndex, sequenceWordIndex, conceptMask):		
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
		GIAANNproto_predictiveNetworkModel.ensureModelMatchesDatabase(databaseNetworkObject)
		conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred = GIAANNproto_predictiveNetworkModel.nextWordPredictionColumnMLPtrainStep(globalFeatureNeurons, targets, targetsC, targetsF)
	elif(inferencePredictiveNetworkModel=="MLP"):
		conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred = GIAANNproto_predictiveNetworkModel.nextWordPredictionMLPtrainStep(globalFeatureNeurons, targets, targetsC, targetsF)
	elif(inferencePredictiveNetworkModel=="Transformer"):
		conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred = GIAANNproto_predictiveNetworkModel.nextWordPredictionTransformerTrainStep(globalFeatureNeurons, globalFeatureConnections, targets, targetsC, targetsF)

	if(inferenceUseNextTokenPredictionsOrTargetsToActivateNextColumnFeatures):
		conceptColumnsIndicesNext = conceptColumnsIndicesPred
		conceptColumnsFeatureIndicesNext = conceptColumnsFeatureIndicesPred
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
		
	return conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext, multipleSourcesNext, kc, conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, targetMultipleSources, targetPreviousColumnIndex, targetNextColumnIndex




def selectMostActiveFeature(sequenceObservedColumns, globalFeatureNeuronsActivation, globalFeatureNeuronsStrength, wordsSequence, lemmasSequence, wordPredictionIndex, sequenceWordIndex, conceptMask):
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
	if(kcDynamic):
		conceptColumnsActivation = conceptColumnsActivation[conceptColumnsActivation > kcActivationThreshold]	#select kcMax columns above threshold
	conceptColumnsActivationTopkConcepts = pt.topk(conceptColumnsActivation, kcMax)
	kc = len(conceptColumnsActivationTopkConcepts.indices)
	if(kcDynamic and kc < 1):
		print("selectMostActiveFeature kcDynamic error: kc < 1; cannot continue to predict columns; consider disabling kcDynamic for debug")
		exit()

	#top feature selection;
	if(kc==1):
		topkConceptColumnsActivation = globalFeatureNeuronsActivationAllSegments[conceptColumnsActivationTopkConcepts.indices[0]].unsqueeze(0)	#select topk concept indices
	else:
		topkConceptColumnsActivation = GIAANNproto_sparseTensors.sliceSparseTensorMulti(globalFeatureNeuronsActivationAllSegments, 0, conceptColumnsActivationTopkConcepts.indices)	#select topk concept indices
	topkConceptColumnsActivation = topkConceptColumnsActivation.to_dense()
	if(inferenceNormaliseFeatureSelectionByFeatureConnections):
		if(kc==1):
			topkConceptColumnsStrength = globalFeatureNeuronsStrengthAllSegments[conceptColumnsActivationTopkConcepts.indices[0]].unsqueeze(0)	#select topk concept indices
		else:
			topkConceptColumnsStrength = GIAANNproto_sparseTensors.sliceSparseTensorMulti(globalFeatureNeuronsStrengthAllSegments, 0, conceptColumnsActivationTopkConcepts.indices)	#select topk concept indices
		topkConceptColumnsStrength = topkConceptColumnsStrength.to_dense()
		if(not inferenceNormaliseFeatureSelectionByFeatureConnectionsStrength):
			topkConceptColumnsStrength = (topkConceptColumnsStrength > 0).float()
		topkConceptColumnsActivation = topkConceptColumnsActivation / topkConceptColumnsStrength
	topkConceptColumnsActivationTopkFeatures = pt.topk(topkConceptColumnsActivation, kf, dim=1)

	conceptColumnsIndicesPred = conceptColumnsActivationTopkConcepts.indices
	conceptColumnsFeatureIndicesPred = topkConceptColumnsActivationTopkFeatures.indices
	
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
		
	return conceptColumnsIndicesNext, conceptColumnsFeatureIndicesNext, multipleSourcesNext, kc, conceptColumnsIndicesPred, conceptColumnsFeatureIndicesPred, targetMultipleSources, targetPreviousColumnIndex, targetNextColumnIndex
