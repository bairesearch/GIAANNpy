"""GIAANNproto_predictionSeed.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 Baxter AI (baxterai.com)

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
import GIAANNproto_databaseNetworkExcitation
import GIAANNproto_databaseNetworkTrainExcitation	#for createFeatureConnectionsActiveTrain only
import GIAANNproto_databaseNetworkDrawExcitation

def seedConceptWords(sequenceObservedColumns, sequenceIndex, sequence, tokens, firstSeedTokenIndex, numSeedTokens):
	result  = GIAANNproto_sequenceConcepts.processConceptWords(sequenceObservedColumns, sequenceIndex, sequence, tokens)
	if(result is None):
		printe("error: inference requires properly formed sentences (delimiters between concepts)")
		return False
	conceptIndices, startIndices, endIndices = result
	featureNeuronsActive, cs, fs, sequenceConceptIndexMask, columnsWordOrder, featureNeuronsWordOrder, featureNeuronsPos, featureNeuronsSegmentMask = GIAANNproto_sequenceConcepts.processFeatures(sequenceObservedColumns, sequenceIndex, sequence, tokens, conceptIndices, startIndices, endIndices)

	firstSeedConceptIndex, numSeedConcepts, firstSeedFeatureIndex = identifySeedIndices(sequenceObservedColumns, sequenceIndex, startIndices, endIndices, sequence, tokens, conceptIndices, firstSeedTokenIndex, numSeedTokens)
	printSeedNodeDetails(sequenceObservedColumns, tokens, conceptIndices, startIndices, endIndices, firstSeedTokenIndex, numSeedTokens)
	sequenceWordIndex = None
	sequenceColumnIndex = None
	if(inferenceUseNeuronFeaturePropertiesTime):
		sequenceWordIndex = firstSeedTokenIndex + numSeedTokens - 1
		if(sequenceWordIndex < 0 or sequenceWordIndex >= len(tokens)):
			raise RuntimeError("seedConceptWords: sequenceWordIndex out of range")
		if(useSANIcolumns or useSANIfeaturesAndColumns):
			conceptMask, _, _ = GIAANNproto_sequenceConcepts.createConceptMask(sequenceObservedColumns, tokens)
			sequenceColumnIndex = GIAANNproto_predictionActivate.calculateSequenceColumnIndex(conceptMask, sequenceWordIndex)
	processFeaturesActiveSeed(sequenceObservedColumns, featureNeuronsActive, cs, fs, sequenceConceptIndexMask, columnsWordOrder, featureNeuronsWordOrder, featureNeuronsPos, firstSeedTokenIndex, numSeedTokens, firstSeedConceptIndex, numSeedConcepts, firstSeedFeatureIndex, sequenceWordIndex, sequenceColumnIndex)

def printSeedNodeDetails(sequenceObservedColumns, tokens, conceptIndices, startIndices, endIndices, firstSeedTokenIndex, numSeedTokens):
	conceptIndicesList = conceptIndices.tolist()
	startIndicesList = startIndices.tolist()
	endIndicesList = endIndices.tolist()
	for offset in range(numSeedTokens):
		tokenIndex = firstSeedTokenIndex + offset
		if(tokenIndex >= len(tokens)):
			continue
		token = tokens[tokenIndex]
		columnSequenceIndex = None
		columnName = None
		columnIndex = None
		for seqConceptIndex, (startIdx, endIdx) in enumerate(zip(startIndicesList, endIndicesList)):
			if(tokenIndex >= startIdx and tokenIndex < endIdx):
				columnSequenceIndex = seqConceptIndex
				if(seqConceptIndex < len(conceptIndicesList)):
					columnConceptWordIndex = conceptIndicesList[seqConceptIndex]
					columnName = tokens[columnConceptWordIndex].lemma
					columnIndex = sequenceObservedColumns.columnsIndexSequenceWordIndexDict.get(columnConceptWordIndex)
				break
		if(columnIndex is None):
			columnIndex = sequenceObservedColumns.columnsIndexSequenceWordIndexDict.get(tokenIndex)
		featureName = token.word
		featureIndex = sequenceObservedColumns.featureWordToIndex.get(featureName)
		if(featureIndex is None and useDedicatedConceptNames and tokenIndex in sequenceObservedColumns.columnsIndexSequenceWordIndexDict):
			featureName = variableConceptNeuronFeatureName
			featureIndex = featureIndexConceptNeuron

def getSequenceColumnName(sequenceObservedColumns, sequenceConceptIndex):
	columnName = None
	if(hasattr(sequenceObservedColumns, "sequenceObservedColumnsDict") and sequenceConceptIndex in sequenceObservedColumns.sequenceObservedColumnsDict):
		columnName = sequenceObservedColumns.sequenceObservedColumnsDict[sequenceConceptIndex].conceptName
	elif(hasattr(sequenceObservedColumns, "observedColumnsDict2") and sequenceConceptIndex in sequenceObservedColumns.observedColumnsDict2):
		columnName = sequenceObservedColumns.observedColumnsDict2[sequenceConceptIndex].conceptName
	return columnName

def debugPrintActiveConnectionSamples(sequenceObservedColumns, featureConnectionsActive, maxSamples=5):
	nonzero = pt.nonzero(featureConnectionsActive)
	if(nonzero.shape[0] == 0):
		print("\tseed debug: no active connections after mask")
		return
	nonzero = nonzero[:maxSamples].cpu()
	for idx in nonzero:
		if(idx.shape[0] == 5):
			segmentIndex, sourceColumnIndex, sourceFeatureIndex, targetColumnIndex, targetFeatureIndex = idx.tolist()
		else:
			segmentIndex, sourceColumnIndex, sourceFeatureIndex, targetColumnIndex, targetFeatureIndex = idx[0], idx[1], idx[2], idx[3], idx[4]
		sourceColumnName = getSequenceColumnName(sequenceObservedColumns, sourceColumnIndex)
		targetColumnName = getSequenceColumnName(sequenceObservedColumns, targetColumnIndex)
		sourceFeatureName = sequenceObservedColumns.indexToFeatureWord.get(sourceFeatureIndex, "NA") if hasattr(sequenceObservedColumns, "indexToFeatureWord") else "NA"
		targetFeatureName = sequenceObservedColumns.indexToFeatureWord.get(targetFeatureIndex, "NA") if hasattr(sequenceObservedColumns, "indexToFeatureWord") else "NA"
		connectionStrength = sequenceObservedColumns.featureConnections[arrayIndexPropertiesStrengthIndex, 0, segmentIndex, sourceColumnIndex, sourceFeatureIndex, targetColumnIndex, targetFeatureIndex]
		indexTuple = (int(segmentIndex), int(sourceColumnIndex), int(sourceFeatureIndex), int(targetColumnIndex), int(targetFeatureIndex))
		print("\tseed debug: idx={0}, source=({1}:{2}), target=({3}:{4}), strength={5}".format(indexTuple, sourceColumnName, sourceFeatureName, targetColumnName, targetFeatureName, float(connectionStrength)))

def debugPrintDenseConnectionSnapshot(sequenceObservedColumns, label="featureConnections snapshot", maxSamples=10):
	denseStrength = sequenceObservedColumns.featureConnections[arrayIndexPropertiesStrengthIndex]
	nonzero = pt.nonzero(denseStrength)
	numEntries = nonzero.shape[0]
	print("\tseed debug: {0} non-zero entries = {1}".format(label, numEntries))
	if(numEntries == 0):
		return
	sample = nonzero[:maxSamples].cpu()
	for idx in sample:
		indexTuple = tuple(int(value) for value in idx.tolist())
		value = float(denseStrength[indexTuple])
		print("\tseed debug: tensor idx={0}, strength={1}".format(indexTuple, value))

def seedNetwork(sequenceObservedColumns, sequenceIndex, sequence, numSeedTokens):
	if(inferenceSeedNetwork):
		if(inferenceIncrementallySeedNetwork):
			for seedTokenIndex in range(numSeedTokens):
				seedNetworkToken(sequenceObservedColumns, sequenceIndex, sequence, seedTokenIndex, 1)
		else:
			seedNetworkToken(sequenceObservedColumns, sequenceIndex, sequence, 0, numSeedTokens)
		if(not inferenceSeedTargetActivationsGlobalFeatureArrays):
			# Update observed columns from sequence observed columns
			sequenceObservedColumns.updateObservedColumnsWrapper(inference=True)	#convert sequence observed columns feature neuron arrays back to global feature neuron arrays
	elif(inferenceBeamSearch and numSeedTokens > 0):
		#beam search requires prompt activations even when inferenceSeedNetwork is disabled
		if(inferenceIncrementallySeedNetwork):
			for seedTokenIndex in range(numSeedTokens):
				seedNetworkToken(sequenceObservedColumns, sequenceIndex, sequence, seedTokenIndex, 1)
		else:
			seedNetworkToken(sequenceObservedColumns, sequenceIndex, sequence, 0, numSeedTokens)
		if(not inferenceSeedTargetActivationsGlobalFeatureArrays):
			sequenceObservedColumns.updateObservedColumnsWrapper(inference=True)
			
def seedNetworkToken(sequenceObservedColumns, sequenceIndex, sequence, firstSeedTokenIndex, numSeedTokens):
	tokens = GIAANNproto_sequenceTokens.getTokens(sequence)
	if(inferenceIncrementallySeedNetwork):
		print("\t seedNetwork: seedTokenIndex = ", firstSeedTokenIndex, ", word = ", tokens[firstSeedTokenIndex].word)
	else:
		print("\t seedNetwork: firstSeedTokenIndex = ", firstSeedTokenIndex, ", words = ", tokens[firstSeedTokenIndex:numSeedTokens].word)
	
	seedConceptWords(sequenceObservedColumns, sequenceIndex, sequence, tokens, firstSeedTokenIndex, numSeedTokens)

	if(inferenceDecrementActivations):
		if(not inferenceSeedTargetActivationsGlobalFeatureArrays):
			globalFeatureNeuronsActivation = sequenceObservedColumns.databaseNetworkObject.globalFeatureNeurons[arrayIndexPropertiesActivationIndex]
			globalFeatureNeuronsActivation = GIAANNproto_predictionActivate.decrementActivation(globalFeatureNeuronsActivation, activationDecrementSeed)
			sequenceObservedColumns.databaseNetworkObject.globalFeatureNeurons = GIAANNproto_sparseTensors.replaceAllSparseTensorElementsAtFirstDimIndex(sequenceObservedColumns.databaseNetworkObject.globalFeatureNeurons, globalFeatureNeuronsActivation, arrayIndexPropertiesActivationIndex)
	
	if(drawNetworkDuringInferenceSeed):
		sequenceObservedColumns.updateObservedColumnsWrapper(inference=True)
		GIAANNproto_databaseNetworkDrawExcitation.visualizeGraph(sequenceObservedColumns, True, save=drawNetworkDuringInferenceSave, fileName=drawNetworkDuringInferenceSaveFilenamePrepend+generateDrawSequenceIndex(firstSeedTokenIndex))

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
			
	return firstSeedConceptIndex, numSeedConcepts, firstSeedFeatureIndex
	
#first dim cs1 pertains to every concept node in sequence
def processFeaturesActiveSeed(sequenceObservedColumns, featureNeuronsActive, cs, fs, sequenceConceptIndexMask, columnsWordOrder, featureNeuronsWordOrder, featureNeuronsPos, firstSeedTokenIndex, numSeedTokens, firstSeedConceptIndex, numSeedConcepts, firstSeedFeatureIndex, sequenceWordIndex, sequenceColumnIndex):
	featureNeuronsInactive = 1 - featureNeuronsActive
	
	fs2 = fs
	if(inferenceSeedTargetActivationsGlobalFeatureArrays):
		cs2 = sequenceObservedColumns.databaseNetworkObject.c
		featureConnectionsActive = pt.ones(cs, fs, cs2, fs2)
	else:
		cs2 = cs
		if(multipleDendriticBranches):
			featureNeuronsActiveSeed = (featureNeuronsActive > 0).any(dim=1).to(arrayType)
		else:
			featureNeuronsActiveSeed = (featureNeuronsActive > 0).any(dim=(0, 1)).to(arrayType)
		featureConnectionsActive, featureConnectionsSegmentMask = GIAANNproto_databaseNetworkTrainExcitation.createFeatureConnectionsActiveTrain(featureNeuronsActiveSeed, cs, fs, columnsWordOrder, featureNeuronsWordOrder)
	
	firstWordIndexPredictPhase = firstSeedTokenIndex+numSeedTokens
	firstConceptIndexPredictPhase = firstSeedConceptIndex+numSeedConcepts
	featureConnectionsActive = createFeatureConnectionsActiveSeed(featureConnectionsActive, cs, fs, cs2, fs2, columnsWordOrder, featureNeuronsWordOrder, firstSeedTokenIndex, firstWordIndexPredictPhase, firstSeedConceptIndex, firstConceptIndexPredictPhase)

	featureConnectionsStrength = sequenceObservedColumns.featureConnections[arrayIndexPropertiesStrengthIndex]
	if(featureConnectionsStrength.dim() == 6 and (not multipleDendriticBranches or inferenceSeedTargetActivationsGlobalFeatureArrays)):
		featureConnectionsStrength = featureConnectionsStrength.sum(dim=0)
	if(inferenceSeedTargetActivationsGlobalFeatureArrays):
		featureConnectionsActivationUpdate = featureConnectionsActive[:, firstSeedConceptIndex] * featureConnectionsStrength
	else:
		featureConnectionsActivationUpdate = featureConnectionsActive * featureConnectionsStrength

	if(inferenceSeedTargetActivationsGlobalFeatureArrays):
		featureNeuronsTargetActivation = pt.sum(featureConnectionsActivationUpdate, dim=(1))
	else:
		if(featureConnectionsActivationUpdate.dim() == 6):
			featureNeuronsTargetActivation = pt.sum(featureConnectionsActivationUpdate, dim=(2, 3))
		else:
			featureNeuronsTargetActivation = pt.sum(featureConnectionsActivationUpdate, dim=(1, 2))
	
	if(inferenceActivationFunction):
		featureNeuronsTargetActivation = GIAANNproto_predictionActivate.activationFunction(featureNeuronsTargetActivation)
	else:
		featureNeuronsTargetActivation = featureNeuronsTargetActivation*j1
	seedFeatureNeuronsTime = None
	if(inferenceUseNeuronFeaturePropertiesTime):
		globalFeatureNeuronsTime = sequenceObservedColumns.databaseNetworkObject.globalFeatureNeurons[arrayIndexPropertiesTimeIndex]
		seedFeatureNeuronsTime = globalFeatureNeuronsTime if inferenceSeedTargetActivationsGlobalFeatureArrays else sequenceObservedColumns.featureNeurons[arrayIndexPropertiesTimeIndex]
		if(featureNeuronsTargetActivation.dim() == 3):
			featureNeuronsTargetActivation = featureNeuronsTargetActivation.unsqueeze(0)
	if(inferenceUseNeuronFeaturePropertiesTimeExact):
		if(featureNeuronsTargetActivation.dim() == 3):
			featureNeuronsTargetActivation = featureNeuronsTargetActivation.unsqueeze(0)
		# spec step (a): only allow segment activation when the time difference to the previous segment is exactly 1.
		featureNeuronsTargetActivation = GIAANNproto_predictionActivate.applyExactTimeActivationConstraint(featureNeuronsTargetActivation, seedFeatureNeuronsTime, sequenceWordIndex, sequenceColumnIndex)
	if(inferenceSeedTargetActivationsGlobalFeatureArrays):
		globalFeatureNeuronsActivation = sequenceObservedColumns.databaseNetworkObject.globalFeatureNeurons[arrayIndexPropertiesActivationIndex]
	featureNeuronsTargetActivationApplied = featureNeuronsTargetActivation
	if(useSANI and algorithmMatrixSANImethod=="enforceActivationAcrossSegments" and enforceSequentialActivation):
		# Patch: seed activation skipped SANI sequential gating, so later segments could activate without prior segments.
		if(inferenceSeedTargetActivationsGlobalFeatureArrays):
			globalFeatureNeuronsActivationDense = globalFeatureNeuronsActivation.to_dense()
			featureNeuronsTargetActivationDense = featureNeuronsTargetActivation
			if(featureNeuronsTargetActivationDense.is_sparse):
				featureNeuronsTargetActivationDense = featureNeuronsTargetActivationDense.to_dense()
			featureNeuronsTargetActivationAppliedDense = pt.zeros_like(featureNeuronsTargetActivationDense)
			for branchIndex in range(globalFeatureNeuronsActivationDense.shape[0]):
				branchActivation = globalFeatureNeuronsActivationDense[branchIndex]
				if(featureNeuronsTargetActivationDense.dim() == 4):
					branchTargetActivation = featureNeuronsTargetActivationDense[branchIndex]
					branchTargetActivationApplied = featureNeuronsTargetActivationAppliedDense[branchIndex]
				else:
					branchTargetActivation = featureNeuronsTargetActivationDense
					branchTargetActivationApplied = featureNeuronsTargetActivationAppliedDense
				if(useSANIfeaturesAndColumns):
					featureSegmentsOffset = arrayNumberOfSegmentsColumnDistance
					assert featureSegmentsOffset >= 0 and featureSegmentsOffset < arrayNumberOfSegments
					previousConceptChannelActivation = branchActivation[:featureSegmentsOffset-1] > 0 if featureSegmentsOffset > 1 else None
					previousFeatureChannelActivation = branchActivation[featureSegmentsOffset:arrayNumberOfSegments-1] > 0 if featureSegmentsOffset+1 < arrayNumberOfSegments else None
					if(previousConceptChannelActivation is not None):
						branchActivation[1:featureSegmentsOffset] += branchTargetActivation[1:featureSegmentsOffset] * previousConceptChannelActivation
						branchTargetActivationApplied[1:featureSegmentsOffset] = branchTargetActivation[1:featureSegmentsOffset] * previousConceptChannelActivation
					if(previousFeatureChannelActivation is not None):
						branchActivation[featureSegmentsOffset+1:] += branchTargetActivation[featureSegmentsOffset+1:] * previousFeatureChannelActivation
						branchTargetActivationApplied[featureSegmentsOffset+1:] = branchTargetActivation[featureSegmentsOffset+1:] * previousFeatureChannelActivation
					branchActivation[0] += branchTargetActivation[0]
					branchActivation[featureSegmentsOffset] += branchTargetActivation[featureSegmentsOffset]
					branchTargetActivationApplied[0] = branchTargetActivation[0]
					branchTargetActivationApplied[featureSegmentsOffset] = branchTargetActivation[featureSegmentsOffset]
				else:
					previousChannelActivation = branchActivation[:-1] > 0
					branchActivation[1:] += branchTargetActivation[1:] * previousChannelActivation
					branchActivation[0] += branchTargetActivation[0]
					branchTargetActivationApplied[1:] = branchTargetActivation[1:] * previousChannelActivation
					branchTargetActivationApplied[0] = branchTargetActivation[0]
				globalFeatureNeuronsActivationDense[branchIndex] = branchActivation
				if(featureNeuronsTargetActivationDense.dim() == 4):
					featureNeuronsTargetActivationAppliedDense[branchIndex] = branchTargetActivationApplied
			globalFeatureNeuronsActivation = globalFeatureNeuronsActivationDense.to_sparse_coo()
			if(featureNeuronsTargetActivation.is_sparse):
				featureNeuronsTargetActivationApplied = featureNeuronsTargetActivationAppliedDense.to_sparse_coo()
			else:
				featureNeuronsTargetActivationApplied = featureNeuronsTargetActivationAppliedDense
		else:
			featureNeuronsActivationDense = sequenceObservedColumns.featureNeurons[arrayIndexPropertiesActivationIndex]
			featureNeuronsTargetActivationDense = featureNeuronsTargetActivation
			if(featureNeuronsTargetActivationDense.is_sparse):
				featureNeuronsTargetActivationDense = featureNeuronsTargetActivationDense.to_dense()
			featureNeuronsTargetActivationAppliedDense = pt.zeros_like(featureNeuronsTargetActivationDense)
			for branchIndex in range(featureNeuronsActivationDense.shape[0]):
				branchActivation = featureNeuronsActivationDense[branchIndex]
				if(featureNeuronsTargetActivationDense.dim() == 4):
					branchTargetActivation = featureNeuronsTargetActivationDense[branchIndex]
					branchTargetActivationApplied = featureNeuronsTargetActivationAppliedDense[branchIndex]
				else:
					branchTargetActivation = featureNeuronsTargetActivationDense
					branchTargetActivationApplied = featureNeuronsTargetActivationAppliedDense
				if(useSANIfeaturesAndColumns):
					featureSegmentsOffset = arrayNumberOfSegmentsColumnDistance
					assert featureSegmentsOffset >= 0 and featureSegmentsOffset < arrayNumberOfSegments
					previousConceptChannelActivation = branchActivation[:featureSegmentsOffset-1] > 0 if featureSegmentsOffset > 1 else None
					previousFeatureChannelActivation = branchActivation[featureSegmentsOffset:arrayNumberOfSegments-1] > 0 if featureSegmentsOffset+1 < arrayNumberOfSegments else None
					if(previousConceptChannelActivation is not None):
						branchActivation[1:featureSegmentsOffset] += branchTargetActivation[1:featureSegmentsOffset] * previousConceptChannelActivation
						branchTargetActivationApplied[1:featureSegmentsOffset] = branchTargetActivation[1:featureSegmentsOffset] * previousConceptChannelActivation
					if(previousFeatureChannelActivation is not None):
						branchActivation[featureSegmentsOffset+1:] += branchTargetActivation[featureSegmentsOffset+1:] * previousFeatureChannelActivation
						branchTargetActivationApplied[featureSegmentsOffset+1:] = branchTargetActivation[featureSegmentsOffset+1:] * previousFeatureChannelActivation
					branchActivation[0] += branchTargetActivation[0]
					branchActivation[featureSegmentsOffset] += branchTargetActivation[featureSegmentsOffset]
					branchTargetActivationApplied[0] = branchTargetActivation[0]
					branchTargetActivationApplied[featureSegmentsOffset] = branchTargetActivation[featureSegmentsOffset]
				else:
					previousChannelActivation = branchActivation[:-1] > 0
					branchActivation[1:] += branchTargetActivation[1:] * previousChannelActivation
					branchActivation[0] += branchTargetActivation[0]
					branchTargetActivationApplied[1:] = branchTargetActivation[1:] * previousChannelActivation
					branchTargetActivationApplied[0] = branchTargetActivation[0]
				featureNeuronsActivationDense[branchIndex] = branchActivation
				if(featureNeuronsTargetActivationDense.dim() == 4):
					featureNeuronsTargetActivationAppliedDense[branchIndex] = branchTargetActivationApplied
			sequenceObservedColumns.featureNeurons[arrayIndexPropertiesActivationIndex] = featureNeuronsActivationDense
			if(featureNeuronsTargetActivation.is_sparse):
				featureNeuronsTargetActivationApplied = featureNeuronsTargetActivationAppliedDense.to_sparse_coo()
			else:
				featureNeuronsTargetActivationApplied = featureNeuronsTargetActivationAppliedDense
	else:
		if(inferenceSeedTargetActivationsGlobalFeatureArrays):
			globalFeatureNeuronsActivation = globalFeatureNeuronsActivation + featureNeuronsTargetActivation
		else:
			sequenceObservedColumns.featureNeurons[arrayIndexPropertiesActivationIndex] += featureNeuronsTargetActivation
	
	if(debugSANIfeaturesAndColumns and useSANIfeaturesAndColumns):
		if(inferenceSeedTargetActivationsGlobalFeatureArrays):
			featureNeuronsActivation = globalFeatureNeuronsActivation
		else:
			featureNeuronsActivation = sequenceObservedColumns.featureNeurons[arrayIndexPropertiesActivationIndex]
		if(featureNeuronsActivation.is_sparse):
			featureNeuronsActivationDense = featureNeuronsActivation.to_dense()
		else:
			featureNeuronsActivationDense = featureNeuronsActivation
		segmentFeatureActivations = featureNeuronsActivationDense.sum(dim=(0, 2)).to("cpu").tolist()
		print("\tdebugSANIfeaturesAndColumns: seed segmentFeatureActivations={0}".format(segmentFeatureActivations))

	if(inferenceDecrementActivations):
		if(inferenceSeedTargetActivationsGlobalFeatureArrays):
			globalFeatureNeuronsActivation = decrementActivation(globalFeatureNeuronsActivation, activationDecrementSeed)
		else:
			sequenceObservedColumns.featureNeurons[arrayIndexPropertiesActivationIndex] = GIAANNproto_predictionActivate.decrementActivationDense(sequenceObservedColumns.featureNeurons[arrayIndexPropertiesActivationIndex], activationDecrementSeed)
	if(inferenceUseNeuronFeaturePropertiesTime):
		if(inferenceSeedTargetActivationsGlobalFeatureArrays):
			# spec step (a): store last timeValue for activated segments during each seed step.
			globalFeatureNeuronsTime = GIAANNproto_predictionActivate.updateTimeValuesFromActivation(globalFeatureNeuronsTime, featureNeuronsTargetActivationApplied, sequenceWordIndex, sequenceColumnIndex)
		else:
			sequenceObservedColumns.featureNeurons[arrayIndexPropertiesTimeIndex] = updateTimeValuesFromActivationDense(sequenceObservedColumns.featureNeurons[arrayIndexPropertiesTimeIndex], featureNeuronsTargetActivationApplied, sequenceWordIndex, sequenceColumnIndex)
			activationGlobalSparse = None
			activationSparse = featureNeuronsTargetActivationApplied
			if(activationSparse is not None):
				if(not activationSparse.is_sparse):
					activationSparse = activationSparse.to_sparse_coo()
				activationSparse = activationSparse.coalesce()
				if(activationSparse._nnz() > 0):
					updateIndices = activationSparse.indices()
					if(updateIndices.shape[0] != 4):
						raise RuntimeError("processFeaturesActiveSeed: activation indices must be 4D")
					conceptIndexLookup = sequenceObservedColumns.conceptIndicesInSequenceObservedTensor.to(updateIndices.device)
					featureIndexLookup = sequenceObservedColumns.featureIndicesInObservedTensor.to(updateIndices.device)
					globalConceptIndices = conceptIndexLookup.index_select(0, updateIndices[2])
					globalFeatureIndices = featureIndexLookup.index_select(0, updateIndices[3])
					globalIndices = pt.stack([updateIndices[0], updateIndices[1], globalConceptIndices, globalFeatureIndices], dim=0).to(globalFeatureNeuronsTime.device)
					activationGlobalSparse = pt.sparse_coo_tensor(globalIndices, pt.ones(globalIndices.shape[1], dtype=globalFeatureNeuronsTime.dtype, device=globalIndices.device), size=globalFeatureNeuronsTime.size(), device=globalFeatureNeuronsTime.device).coalesce()
			if(activationGlobalSparse is not None):
				# spec step (a): store last timeValue for activated segments during each seed step.
				globalFeatureNeuronsTime = GIAANNproto_predictionActivate.updateTimeValuesFromActivation(globalFeatureNeuronsTime, activationGlobalSparse, sequenceWordIndex, sequenceColumnIndex)
		sequenceObservedColumns.databaseNetworkObject.globalFeatureNeurons = GIAANNproto_sparseTensors.replaceAllSparseTensorElementsAtFirstDimIndex(sequenceObservedColumns.databaseNetworkObject.globalFeatureNeurons, globalFeatureNeuronsTime, arrayIndexPropertiesTimeIndex)
	if(debugInferenceUseNeuronFeaturePropertiesTime2 and inferenceUseNeuronFeaturePropertiesTime):
		debugTokens = sequenceObservedColumns.tokens
		debugTargetSequenceWordIndex = numSeedTokensInference
		if(debugTargetSequenceWordIndex >= 0 and debugTargetSequenceWordIndex < len(debugTokens)):
			debugConceptMask, _, _ = GIAANNproto_sequenceConcepts.createConceptMask(sequenceObservedColumns, debugTokens)
			debugTargetFoundNextColumnIndex, debugTargetPreviousColumnIndex, debugTargetNextColumnIndex, debugTargetFeatureIndex = GIAANNproto_databaseNetworkExcitation.getTokenConceptFeatureIndex(sequenceObservedColumns, debugTokens, debugConceptMask, debugTargetSequenceWordIndex)
			debugTargetColumnIndex = debugTargetPreviousColumnIndex
			debugConceptIndexLookup = sequenceObservedColumns.conceptIndicesInSequenceObservedTensor
			debugFeatureIndexLookup = sequenceObservedColumns.featureIndicesInObservedTensor
			debugLocalColumnMatch = pt.nonzero(debugConceptIndexLookup == debugTargetColumnIndex, as_tuple=False)
			debugLocalFeatureMatch = pt.nonzero(debugFeatureIndexLookup == debugTargetFeatureIndex, as_tuple=False)
			if(debugLocalColumnMatch.numel() > 0 and debugLocalFeatureMatch.numel() > 0):
				debugLocalColumnIndex = int(debugLocalColumnMatch[0].item())
				debugLocalFeatureIndex = int(debugLocalFeatureMatch[0].item())
				debugBranchCount = numberOfDendriticBranches if multipleDendriticBranches else 1
				debugDevice = featureNeuronsTargetActivationApplied.device
				debugBranchRange = pt.arange(debugBranchCount, dtype=pt.long, device=debugDevice)
				debugSegmentRange = pt.arange(arrayNumberOfSegments, dtype=pt.long, device=debugDevice)
				debugBranchIndices = debugBranchRange.repeat_interleave(arrayNumberOfSegments)
				debugSegmentIndices = debugSegmentRange.repeat(debugBranchCount)
				debugColumnIndices = pt.full_like(debugSegmentIndices, debugLocalColumnIndex)
				debugFeatureIndices = pt.full_like(debugSegmentIndices, debugLocalFeatureIndex)
				debugIndices = pt.stack([debugBranchIndices, debugSegmentIndices, debugColumnIndices, debugFeatureIndices], dim=0)
				debugActivationDtype = featureNeuronsTargetActivationApplied.values().dtype if featureNeuronsTargetActivationApplied.is_sparse else featureNeuronsTargetActivationApplied.dtype
				debugActivationValues = GIAANNproto_predictionActivate.gatherSparseTensorValuesAtIndices(featureNeuronsTargetActivationApplied, debugIndices, debugActivationDtype)
				debugActivationBySegment = debugActivationValues.view(debugBranchCount, arrayNumberOfSegments).to("cpu").tolist()
				debugLocalTimeValues = sequenceObservedColumns.featureNeurons[arrayIndexPropertiesTimeIndex][:, :, debugLocalColumnIndex, debugLocalFeatureIndex].to("cpu").tolist()
				debugGlobalTimeValues = None
				if(globalFeatureNeuronsTime is not None):
					debugGlobalDevice = globalFeatureNeuronsTime.device
					debugGlobalBranchRange = pt.arange(debugBranchCount, dtype=pt.long, device=debugGlobalDevice)
					debugGlobalSegmentRange = pt.arange(arrayNumberOfSegments, dtype=pt.long, device=debugGlobalDevice)
					debugGlobalBranchIndices = debugGlobalBranchRange.repeat_interleave(arrayNumberOfSegments)
					debugGlobalSegmentIndices = debugGlobalSegmentRange.repeat(debugBranchCount)
					debugGlobalColumnIndices = pt.full_like(debugGlobalSegmentIndices, debugTargetColumnIndex)
					debugGlobalFeatureIndices = pt.full_like(debugGlobalSegmentIndices, debugTargetFeatureIndex)
					debugGlobalIndices = pt.stack([debugGlobalBranchIndices, debugGlobalSegmentIndices, debugGlobalColumnIndices, debugGlobalFeatureIndices], dim=0)
					debugGlobalValues = GIAANNproto_predictionActivate.gatherSparseTensorValuesAtIndices(globalFeatureNeuronsTime, debugGlobalIndices, globalFeatureNeuronsTime.dtype)
					debugGlobalTimeValues = debugGlobalValues.view(debugBranchCount, arrayNumberOfSegments).to("cpu").tolist()
				debugSeedWord = debugTokens[sequenceWordIndex].word if sequenceWordIndex is not None and sequenceWordIndex < len(debugTokens) else "NA"
				print(f"debugInferenceUseNeuronFeaturePropertiesTime2: seedWordIndex={sequenceWordIndex}, seedWord={debugSeedWord}, targetWordIndex={debugTargetSequenceWordIndex}, targetWord={debugTokens[debugTargetSequenceWordIndex].word}, targetColumnIndex={debugTargetColumnIndex}, targetFeatureIndex={debugTargetFeatureIndex}, activationBySegment={debugActivationBySegment}, localTimeBySegment={debugLocalTimeValues}, globalTimeBySegment={debugGlobalTimeValues}")
					
	if(inferenceDeactivateNeuronsUponPrediction):
		wordOrderMask = pt.logical_and(featureNeuronsWordOrder >= firstSeedTokenIndex, featureNeuronsWordOrder < firstWordIndexPredictPhase)
		columnsWordOrderExpanded1 = columnsWordOrder.view(cs, 1).expand(cs, fs)
		columnsWordOrderMask = pt.logical_and(columnsWordOrderExpanded1 >= firstSeedConceptIndex, columnsWordOrderExpanded1 < firstConceptIndexPredictPhase)
		wordOrderMask = pt.logical_and(wordOrderMask, columnsWordOrderMask)
		seedFeatureMask = wordOrderMask.unsqueeze(0).unsqueeze(0).expand(numberOfDendriticBranches, arrayNumberOfSegments, cs, fs)
		if(multipleDendriticBranches):
			seedFeatureMask = seedFeatureMask & (featureNeuronsActive > 0)
		indicesToUpdateLocal = pt.nonzero(seedFeatureMask, as_tuple=False)
		indicesToUpdateGlobal = None
		if(indicesToUpdateLocal.numel() > 0):
			conceptIndexLookup = sequenceObservedColumns.conceptIndicesInSequenceObservedTensor.to(indicesToUpdateLocal.device)
			featureIndexLookup = sequenceObservedColumns.featureIndicesInObservedTensor.to(indicesToUpdateLocal.device)
			branchIndices = indicesToUpdateLocal[:, 0]
			sequenceConceptIndices = indicesToUpdateLocal[:, 2]
			sequenceFeatureIndices = indicesToUpdateLocal[:, 3]
			globalConceptIndices = conceptIndexLookup.index_select(0, sequenceConceptIndices)
			globalFeatureIndices = featureIndexLookup.index_select(0, sequenceFeatureIndices)
			segmentIndices = indicesToUpdateLocal[:, 1]
			indicesToUpdateGlobal = pt.stack([branchIndices, segmentIndices, globalConceptIndices, globalFeatureIndices], dim=1)
		if(inferenceSeedTargetActivationsGlobalFeatureArrays):
			# Patch: seed deactivation must also clear matching global activations so fired features cannot persist into prediction.
			if(indicesToUpdateGlobal is not None and indicesToUpdateGlobal.numel() > 0):
				indicesToUpdateGlobal = indicesToUpdateGlobal.to(globalFeatureNeuronsActivation.device)
				globalFeatureNeuronsActivation = globalFeatureNeuronsActivation.coalesce()
				globalFeatureNeuronsActivation = GIAANNproto_sparseTensors.modifySparseTensor(globalFeatureNeuronsActivation, indicesToUpdateGlobal, 0)
		else:
			featureNeuronsInactiveSource = pt.logical_not(seedFeatureMask).float()
			sequenceObservedColumns.featureNeurons[arrayIndexPropertiesActivationIndex] *= featureNeuronsInactiveSource
			if(indicesToUpdateGlobal is not None and indicesToUpdateGlobal.numel() > 0):
				globalFeatureNeuronsActivation = sequenceObservedColumns.databaseNetworkObject.globalFeatureNeurons[arrayIndexPropertiesActivationIndex]
				indicesToUpdateGlobal = indicesToUpdateGlobal.to(globalFeatureNeuronsActivation.device)
				globalFeatureNeuronsActivation = globalFeatureNeuronsActivation.coalesce()
				globalFeatureNeuronsActivation = GIAANNproto_sparseTensors.modifySparseTensor(globalFeatureNeuronsActivation, indicesToUpdateGlobal, 0)
				sequenceObservedColumns.databaseNetworkObject.globalFeatureNeurons = GIAANNproto_sparseTensors.replaceAllSparseTensorElementsAtFirstDimIndex(sequenceObservedColumns.databaseNetworkObject.globalFeatureNeurons, globalFeatureNeuronsActivation, arrayIndexPropertiesActivationIndex)

	if(inferenceSeedTargetActivationsGlobalFeatureArrays):
		sequenceObservedColumns.databaseNetworkObject.globalFeatureNeurons = GIAANNproto_sparseTensors.replaceAllSparseTensorElementsAtFirstDimIndex(sequenceObservedColumns.databaseNetworkObject.globalFeatureNeurons, globalFeatureNeuronsActivation, arrayIndexPropertiesActivationIndex)

def updateTimeValuesFromActivationDense(featureNeuronsTime, featureNeuronsTargetActivation, sequenceWordIndex, sequenceColumnIndex):
	result = featureNeuronsTime
	if(inferenceUseNeuronFeaturePropertiesTime):
		if(not useSANI):
			raise RuntimeError("updateTimeValuesFromActivationDense: useSANI is required")
		if(featureNeuronsTime is None):
			raise RuntimeError("updateTimeValuesFromActivationDense: featureNeuronsTime is None")
		if(featureNeuronsTargetActivation is None):
			result = featureNeuronsTime
		else:
			activationSparse = featureNeuronsTargetActivation
			if(not activationSparse.is_sparse):
				activationSparse = activationSparse.to_sparse_coo()
			activationSparse = activationSparse.coalesce()
			if(activationSparse._nnz() == 0):
				result = featureNeuronsTime
			else:
				updateIndices = activationSparse.indices()
				if(updateIndices.shape[0] != 4):
					raise RuntimeError("updateTimeValuesFromActivationDense: activation indices must be 4D")
				segmentIndices = updateIndices[1]
				updateValues = GIAANNproto_predictionActivate.buildSegmentTimeValues(segmentIndices, sequenceWordIndex, sequenceColumnIndex, featureNeuronsTime.dtype, activationSparse.device)
				updateIndices = updateIndices.to(featureNeuronsTime.device)
				updateValues = updateValues.to(featureNeuronsTime.device)
				featureNeuronsTime[updateIndices[0], updateIndices[1], updateIndices[2], updateIndices[3]] = updateValues
				result = featureNeuronsTime
	return result

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
