"""GIAANNproto_predictionSeed.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_main.py

# Usage:
see GIAANNproto_main.py

# Description:
GIA ANN proto prediction Seed

"""

import torch as pt

from GIAANNproto_globalDefs import *
import GIAANNproto_sparseTensors
import GIAANNproto_sequenceTokens
import GIAANNproto_sequenceConcepts
import GIAANNproto_predictionActivate
import GIAANNproto_databaseNetworkTrain	#for createFeatureConnectionsActiveTrain only
import GIAANNproto_databaseNetworkDraw

def seedConceptWords(sequenceObservedColumns, sequenceIndex, sequence, tokens, firstSeedTokenIndex, numSeedTokens):
	conceptIndices, startIndices, endIndices = GIAANNproto_sequenceConcepts.processConceptWords(sequenceObservedColumns, sequenceIndex, sequence, tokens)
	featureNeuronsActive, cs, fs, sequenceConceptIndexMask, columnsWordOrder, featureNeuronsWordOrder, featureNeuronsPos, featureNeuronsSegmentMask = GIAANNproto_sequenceConcepts.processFeatures(sequenceObservedColumns, sequenceIndex, sequence, tokens, conceptIndices, startIndices, endIndices)

	firstSeedConceptIndex, numSeedConcepts, firstSeedFeatureIndex = identifySeedIndices(sequenceObservedColumns, sequenceIndex, startIndices, endIndices, sequence, tokens, conceptIndices, firstSeedTokenIndex, numSeedTokens)
	processFeaturesActiveSeed(sequenceObservedColumns, featureNeuronsActive, cs, fs, sequenceConceptIndexMask, columnsWordOrder, featureNeuronsWordOrder, featureNeuronsPos, firstSeedTokenIndex, numSeedTokens, firstSeedConceptIndex, numSeedConcepts, firstSeedFeatureIndex)
	
def seedNetwork(sequenceObservedColumns, sequenceIndex, sequence, numSeedTokens):
	if(inferenceSeedNetwork):
		if(inferenceIncrementallySeedNetwork):
			for seedTokenIndex in range(numSeedTokens):
				seedNetworkToken(sequenceObservedColumns, sequenceIndex, sequence, seedTokenIndex, 1)
		else:
			seedNetworkToken(sequenceObservedColumns, sequenceIndex, sequence, 0, numSeedTokens)
		if(not inferenceSeedTargetActivationsGlobalFeatureArrays):
			# Update observed columns from sequence observed columns
			sequenceObservedColumns.updateObservedColumnsWrapper()	#convert sequence observed columns feature neuron arrays back to global feature neuron arrays
	elif(inferenceBeamSearch and numSeedTokens > 0):
		#beam search requires prompt activations even when inferenceSeedNetwork is disabled
		if(inferenceIncrementallySeedNetwork):
			for seedTokenIndex in range(numSeedTokens):
				seedNetworkToken(sequenceObservedColumns, sequenceIndex, sequence, seedTokenIndex, 1)
		else:
			seedNetworkToken(sequenceObservedColumns, sequenceIndex, sequence, 0, numSeedTokens)
		if(not inferenceSeedTargetActivationsGlobalFeatureArrays):
			sequenceObservedColumns.updateObservedColumnsWrapper()
			
def seedNetworkToken(sequenceObservedColumns, sequenceIndex, sequence, firstSeedTokenIndex, numSeedTokens):
	tokens = GIAANNproto_sequenceTokens.getTokens(sequence)
	if(inferenceIncrementallySeedNetwork):
		print("\t seedNetwork: seedTokenIndex = ", firstSeedTokenIndex, ", word = ", tokens[firstSeedTokenIndex].word)
	else:
		print("\t seedNetwork: firstSeedTokenIndex = ", firstSeedTokenIndex, ", words = ", tokens[firstSeedTokenIndex:numSeedTokens].word)
	seedConceptWords(sequenceObservedColumns, sequenceIndex, sequence, tokens, firstSeedTokenIndex, numSeedTokens)

	if(inferenceDecrementActivations):
		if(not inferenceSeedTargetActivationsGlobalFeatureArrays):
			globalFeatureNeuronsActivation = sequenceObservedColumns.databaseNetworkObject.globalFeatureNeurons[arrayIndexPropertiesActivation]
			globalFeatureNeuronsActivation = GIAANNproto_predictionActivate.decrementActivation(globalFeatureNeuronsActivation, activationDecrementSeed)
			sequenceObservedColumns.databaseNetworkObject.globalFeatureNeurons = GIAANNproto_sparseTensors.replaceAllSparseTensorElementsAtFirstDimIndex(sequenceObservedColumns.databaseNetworkObject.globalFeatureNeurons, globalFeatureNeuronsActivation, arrayIndexPropertiesActivation)
	
	if(drawNetworkDuringInferenceSeed):
		#FUTURE: convert globalFeatureNeuronsActivation back to globalFeatureNeurons for draw
		GIAANNproto_databaseNetworkDraw.visualizeGraph(sequenceObservedColumns, True, save=drawNetworkDuringInferenceSave, fileName=drawNetworkDuringInferenceSaveFilenamePrepend+str(firstSeedTokenIndex))

def identifySeedIndices(sequenceObservedColumns, sequenceIndex, startIndices, endIndices, sequence, tokens, conceptIndices, firstSeedTokenIndex, numSeedTokens):
	firstSeedConceptIndex = None
	numSeedConcepts = None
	foundFirstSeedConcept = False
	if(inferenceSeedTargetActivationsGlobalFeatureArrays):
		featureWord = tokens[firstSeedTokenIndex].word
		if(useDedicatedConceptNames and firstSeedTokenIndex in sequenceObservedColumns.observedColumnsSequenceWordIndexDict):	
			firstSeedFeatureIndex = featureIndexConceptNeuron
		elif(featureWord in sequenceObservedColumns.featureWordToIndex):
			firstSeedFeatureWord = tokens[firstSeedTokenIndex].word
			firstSeedFeatureIndex = sequenceObservedColumns.databaseNetworkObject.conceptFeaturesDict[firstSeedFeatureWord]
	else:
		firstSeedFeatureIndex = None

	conceptIndicesList = conceptIndices.tolist()
	for i, sequenceConceptWordIndex in enumerate(conceptIndicesList):
		if(trainSequenceObservedColumnsMatchSequenceWords):
			sequenceConceptIndex = i
		else:
			conceptLemma = tokens[sequenceConceptWordIndex].lemma
			sequenceConceptIndex = sequenceObservedColumns.conceptNameToIndex[conceptLemma] 

		lastWordIndexSeedPhase = firstSeedTokenIndex+numSeedTokens-1
		if(not foundFirstSeedConcept):
			if(firstSeedTokenIndex >= startIndices[sequenceConceptIndex] and firstSeedTokenIndex < endIndices[sequenceConceptIndex]):
				foundFirstSeedConcept = True
				firstSeedConceptIndex = sequenceConceptIndex
				if(inferenceSeedTargetActivationsGlobalFeatureArrays):
					observedColumn = sequenceObservedColumns.observedColumnsSequenceWordIndexDict[sequenceConceptWordIndex]
					sequenceObservedColumns.featureConnections = observedColumn.featureConnections
		if(foundFirstSeedConcept):
			if(lastWordIndexSeedPhase >= startIndices[sequenceConceptIndex] and lastWordIndexSeedPhase < endIndices[sequenceConceptIndex]):
				lastSeedConceptIndex = sequenceConceptIndex
				numSeedConcepts = lastSeedConceptIndex-firstSeedConceptIndex+1
	
	print("firstSeedConceptIndex = ", firstSeedConceptIndex)
	print("numSeedConcepts = ", numSeedConcepts)
	print("firstSeedFeatureIndex = ", firstSeedFeatureIndex)
		
	return firstSeedConceptIndex, numSeedConcepts, firstSeedFeatureIndex
	
#first dim cs1 pertains to every concept node in sequence
def processFeaturesActiveSeed(sequenceObservedColumns, featureNeuronsActive, cs, fs, sequenceConceptIndexMask, columnsWordOrder, featureNeuronsWordOrder, featureNeuronsPos, firstSeedTokenIndex, numSeedTokens, firstSeedConceptIndex, numSeedConcepts, firstSeedFeatureIndex):
	featureNeuronsInactive = 1 - featureNeuronsActive
	
	fs2 = fs
	if(inferenceSeedTargetActivationsGlobalFeatureArrays):
		cs2 = sequenceObservedColumns.databaseNetworkObject.c
		featureConnectionsActive = pt.ones(cs, fs, cs2, fs2)
	else:
		cs2 = cs
		featureConnectionsActive, featureConnectionsSegmentMask = GIAANNproto_databaseNetworkTrain.createFeatureConnectionsActiveTrain(featureNeuronsActive[arrayIndexSegmentInternalColumn], cs, fs, columnsWordOrder, featureNeuronsWordOrder)

	firstWordIndexPredictPhase = firstSeedTokenIndex+numSeedTokens
	firstConceptIndexPredictPhase = firstSeedConceptIndex+numSeedConcepts
	featureConnectionsActive = createFeatureConnectionsActiveSeed(featureConnectionsActive, cs, fs, cs2, fs2, columnsWordOrder, featureNeuronsWordOrder, firstSeedTokenIndex, firstWordIndexPredictPhase, firstSeedConceptIndex, firstConceptIndexPredictPhase)

	if(inferenceSeedTargetActivationsGlobalFeatureArrays):
		featureConnectionsActivationUpdate = featureConnectionsActive[:, firstSeedConceptIndex] * sequenceObservedColumns.featureConnections[arrayIndexPropertiesStrength]
	else:
		featureConnectionsActivationUpdate = featureConnectionsActive * sequenceObservedColumns.featureConnections[arrayIndexPropertiesStrength]
	
	if(inferenceSeedTargetActivationsGlobalFeatureArrays):
		featureNeuronsTargetActivation = pt.sum(featureConnectionsActivationUpdate, dim=(1))
	else:
		featureNeuronsTargetActivation = pt.sum(featureConnectionsActivationUpdate, dim=(1, 2))
	if(inferenceActivationFunction):
		featureNeuronsTargetActivation = GIAANNproto_predictionActivate.activationFunction(featureNeuronsTargetActivation)
	else:
		featureNeuronsTargetActivation = featureNeuronsTargetActivation*j1
	if(inferenceSeedTargetActivationsGlobalFeatureArrays):
		globalFeatureNeuronsActivation = sequenceObservedColumns.databaseNetworkObject.globalFeatureNeurons[arrayIndexPropertiesActivation]
		globalFeatureNeuronsActivation = globalFeatureNeuronsActivation + featureNeuronsTargetActivation
	else:
		sequenceObservedColumns.featureNeurons[arrayIndexPropertiesActivation, :, :, :] += featureNeuronsTargetActivation
	
	if(inferenceDecrementActivations):
		if(inferenceSeedTargetActivationsGlobalFeatureArrays):
			globalFeatureNeuronsActivation = decrementActivation(globalFeatureNeuronsActivation, activationDecrementSeed)
		else:
			sequenceObservedColumns.featureNeurons[arrayIndexPropertiesActivation] = GIAANNproto_predictionActivate.decrementActivationDense(sequenceObservedColumns.featureNeurons[arrayIndexPropertiesActivation], activationDecrementSeed)
					
	if(inferenceDeactivateNeuronsUponPrediction):
		if(inferenceSeedTargetActivationsGlobalFeatureArrays):
			if(useSANI):
				printe("processFeaturesActiveSeed error: inferenceDeactivateNeuronsUponPrediction:inferenceSeedTargetActivationsGlobalFeatureArrays:useSANI is not yet implemented")
			else:
				indicesToUpdate = pt.tensor([0, firstSeedConceptIndex, firstSeedFeatureIndex]).unsqueeze(0)
				globalFeatureNeuronsActivation = globalFeatureNeuronsActivation.coalesce()
				globalFeatureNeuronsActivation = GIAANNproto_sparseTensors.modifySparseTensor(globalFeatureNeuronsActivation, indicesToUpdate, 0)
		else:
			wordOrderMask = pt.logical_and(featureNeuronsWordOrder >= firstSeedTokenIndex, featureNeuronsWordOrder < firstWordIndexPredictPhase)
			columnsWordOrderExpanded1 = columnsWordOrder.view(cs, 1).expand(cs, fs)
			columnsWordOrderMask = pt.logical_and(columnsWordOrderExpanded1 >= firstSeedConceptIndex, columnsWordOrderExpanded1 < firstConceptIndexPredictPhase)

			wordOrderMask = pt.logical_and(wordOrderMask, columnsWordOrderMask)
			wordOrderMask = wordOrderMask.unsqueeze(0).expand(arrayNumberOfSegments, cs, fs)
			featureNeuronsActiveSource = pt.logical_and(wordOrderMask, featureNeuronsActive > 0)
			featureNeuronsInactiveSource = pt.logical_not(featureNeuronsActiveSource).float()
			sequenceObservedColumns.featureNeurons[arrayIndexPropertiesActivation, :, :, :] *= featureNeuronsInactiveSource

	if(inferenceSeedTargetActivationsGlobalFeatureArrays):
		sequenceObservedColumns.databaseNetworkObject.globalFeatureNeurons = GIAANNproto_sparseTensors.replaceAllSparseTensorElementsAtFirstDimIndex(sequenceObservedColumns.databaseNetworkObject.globalFeatureNeurons, globalFeatureNeuronsActivation, arrayIndexPropertiesActivation)

def createFeatureConnectionsActiveSeed(featureConnectionsActive, cs, fs, cs2, fs2, columnsWordOrder, featureNeuronsWordOrder, firstSeedTokenIndex, firstWordIndexPredictPhase, firstSeedConceptIndex, firstConceptIndexPredictPhase):
	
	if(featureNeuronsWordOrder is not None):	
		featureNeuronsWordOrderExpanded1 = featureNeuronsWordOrder.view(cs, fs, 1, 1).expand(cs, fs, cs2, fs2)
		wordOrderMask = pt.logical_and(featureNeuronsWordOrderExpanded1 >= firstSeedTokenIndex, featureNeuronsWordOrderExpanded1 < firstWordIndexPredictPhase)
		featureConnectionsActive = featureConnectionsActive * wordOrderMask.unsqueeze(0)
	if(columnsWordOrder is not None):
		columnsWordOrderExpanded1 = columnsWordOrder.view(cs, 1, 1, 1).expand(cs, fs, cs2, fs2)
		columnsWordOrderMask = pt.logical_and(columnsWordOrderExpanded1 >= firstSeedConceptIndex, columnsWordOrderExpanded1 < firstConceptIndexPredictPhase)
		featureConnectionsActive = featureConnectionsActive * columnsWordOrderMask.unsqueeze(0)
	
	return featureConnectionsActive
