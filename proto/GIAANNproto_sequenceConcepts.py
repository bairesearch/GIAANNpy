"""GIAANNproto_sequenceConcepts.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_main.py

# Usage:
see GIAANNproto_main.py

# Description:
GIA ANN proto sequence Concepts (and feature detection)

"""

import torch as pt

from GIAANNproto_globalDefs import *
import GIAANNproto_databaseNetwork
import GIAANNproto_sequenceTokens


def firstPass(databaseNetworkObject, sequence):
	newConceptsAdded = False
	conceptsFound = False
	conceptMask = []
	
	for preprocessedToken in sequence:
		token = GIAANNproto_sequenceTokens.convertPreprocessedTokenToSequenceToken(preprocessedToken)

		conceptFound = False
		if usePOS:
			if GIAANNproto_sequenceTokens.isConcept(token):
				# Only assign unique concept columns for nouns
				conceptFound = True
		else:
			# When usePOS is disabled, assign concept columns for every new lemma encountered
			conceptFound = True
		
		if(conceptFound):
			conceptsFound, newConceptsAdded = GIAANNproto_databaseNetwork.addConceptToConceptColumnsDict(databaseNetworkObject, token.lemma, conceptsFound, newConceptsAdded)
			conceptMask.append(True)
		else:
			conceptMask.append(False)

	# If new concept columns have been added, expand arrays as needed
	if newConceptsAdded:
		if not lowMem:
			# Expand global feature neuron arrays
			if databaseNetworkObject.globalFeatureNeurons.shape[2] < databaseNetworkObject.c:
				newShape = (databaseNetworkObject.globalFeatureNeurons.shape[0], databaseNetworkObject.globalFeatureNeurons.shape[1], databaseNetworkObject.c, databaseNetworkObject.globalFeatureNeurons.shape[3])
				if(performRedundantCoalesce):
					databaseNetworkObject.globalFeatureNeurons = databaseNetworkObject.globalFeatureNeurons.coalesce()
				databaseNetworkObject.globalFeatureNeurons = pt.sparse_coo_tensor(databaseNetworkObject.globalFeatureNeurons._indices(), databaseNetworkObject.globalFeatureNeurons._values(), size=newShape, dtype=arrayType, device=deviceSparse)
				
	return conceptsFound, conceptMask

				
def secondPass(databaseNetworkObject, tokens):
	observedColumnsDict = {}
	observedColumnsSequenceWordIndexDict = {}
	for i, token in enumerate(tokens):
		lemma = token.lemma
		pos = token.pos
		if usePOS:
			if GIAANNproto_sequenceTokens.isConcept(token):
				conceptIndex = databaseNetworkObject.conceptColumnsDict[lemma]
				# Load observed column from disk or create new one
				observedColumn = GIAANNproto_databaseNetwork.loadOrCreateObservedColumn(databaseNetworkObject, conceptIndex, lemma, i)
				observedColumnsDict[lemma] = observedColumn
				observedColumnsSequenceWordIndexDict[i] = observedColumn
		else:
			conceptIndex = databaseNetworkObject.conceptColumnsDict[lemma]
			# Load observed column from disk or create new one
			observedColumn = GIAANNproto_databaseNetwork.loadOrCreateObservedColumn(databaseNetworkObject, conceptIndex, lemma, i)
			observedColumnsDict[lemma] = observedColumn
			observedColumnsSequenceWordIndexDict[i] = observedColumn
	return observedColumnsDict, observedColumnsSequenceWordIndexDict


def detectNewFeatures(databaseNetworkObject, tokens):
	"""
	When usePOS mode is enabled, detect all possible new features in the sequence
	by searching for all new non-nouns in the sequence.
	"""

	if(conceptColumnsDelimitByPOS):
		databaseNetworkObject.sequenceReferenceSetDelimiterList = [None]*len(tokens)

	numNewFeatures = 0
	for tokenIndex, token in enumerate(tokens):
		if(processFeatureDetection(databaseNetworkObject, tokenIndex, token, tokens)):
			numNewFeatures += 1

	# After processing all features, update f
	databaseNetworkObject.f += numNewFeatures

	# Now, expand arrays accordingly
	if not lowMem:
		if databaseNetworkObject.f > databaseNetworkObject.globalFeatureNeurons.shape[3]:
			extraCols = databaseNetworkObject.f - databaseNetworkObject.globalFeatureNeurons.shape[3]
			newShape = (databaseNetworkObject.globalFeatureNeurons.shape[0], databaseNetworkObject.globalFeatureNeurons.shape[1], databaseNetworkObject.globalFeatureNeurons.shape[2], databaseNetworkObject.f)
			databaseNetworkObject.globalFeatureNeurons = databaseNetworkObject.globalFeatureNeurons.coalesce()
			databaseNetworkObject.globalFeatureNeurons = pt.sparse_coo_tensor(databaseNetworkObject.globalFeatureNeurons.indices(), databaseNetworkObject.globalFeatureNeurons.values(), size=newShape, dtype=arrayType, device=deviceSparse)

def processFeatureDetection(databaseNetworkObject, tokenIndex, token, tokens):
	"""
	Helper function to detect new features prior to processing concept words.
	"""
	
	featureWord = token.word

	if usePOS and (GIAANNproto_sequenceTokens.isConcept(token)):
		return False  # Skip nouns as features
	else:
		if featureWord not in databaseNetworkObject.conceptFeaturesDict:
			featureIndex = len(databaseNetworkObject.conceptFeaturesDict)
			databaseNetworkObject.conceptFeaturesDict[featureWord] = featureIndex
			databaseNetworkObject.conceptFeaturesList.append(featureWord)
			isDelimiter, isDelimiterDeterministic, isDelimiterProbabilistic = isFeaturePOSreferenceSetDelimiterType(featureWord, token, tokens, tokenIndex)
			if(conceptColumnsDelimitByPOS):
				databaseNetworkObject.sequenceReferenceSetDelimiterList[tokenIndex] = isDelimiter
				if(detectReferenceSetDelimitersBetweenNouns):
					databaseNetworkObject.conceptFeaturesReferenceSetDelimiterDeterministicList.append(isDelimiterDeterministic)
					databaseNetworkObject.conceptFeaturesReferenceSetDelimiterProbabilisticList.append(isDelimiterProbabilistic)
				else:
					databaseNetworkObject.conceptFeaturesReferenceSetDelimiterList.append(isDelimiter)
			return True
		else:
			if(conceptColumnsDelimitByPOS):
				isDelimiter, isDelimiterDeterministic, isDelimiterProbabilistic = isFeaturePOSreferenceSetDelimiterType(featureWord, token, tokens, tokenIndex)
				databaseNetworkObject.sequenceReferenceSetDelimiterList[tokenIndex] = isDelimiter	#deterministic or incontext probabilistic reference set delimiter detected (train only)
				if(detectReferenceSetDelimitersBetweenNouns):
					featureIndex = databaseNetworkObject.conceptFeaturesDict[featureWord]
					databaseNetworkObject.conceptFeaturesReferenceSetDelimiterProbabilisticList[featureIndex] = databaseNetworkObject.conceptFeaturesReferenceSetDelimiterProbabilisticList[featureIndex] or isDelimiterProbabilistic	#reassign probabilistic if ever probabilistic in past (inference only)
			return False
	
def isFeaturePOSreferenceSetDelimiterType(nodeNameString, token, tokens, tokenIndex):
	isDelimiterDeterministic = False
	isDelimiterProbabilistic = False
	if(conceptColumnsDelimitByPOS):
		nodeWordLower = nodeNameString.lower()
		if(GIAANNproto_sequenceTokens.isTokenReferenceSetDelimiterDeterministic(token)):
			isDelimiterDeterministic = True
		if(detectReferenceSetDelimitersBetweenNouns and not isDelimiterDeterministic):
			isDelimiterProbabilistic = detectProbabilisticReferenceSetDelimiterBetweenNouns(nodeWordLower, token, tokens, tokenIndex)
		isDelimiter = isDelimiterDeterministic or isDelimiterProbabilistic
	else:
		isDelimiter = False
	return isDelimiter, isDelimiterDeterministic, isDelimiterProbabilistic


def detectProbabilisticReferenceSetDelimiterBetweenNouns(nodeWordLower, token, tokens, tokenIndex):
	if(tokenIndex < 0 or tokenIndex >= len(tokens)):
		return False
	if(not GIAANNproto_sequenceTokens.isTokenReferenceSetDelimiterProbabilistic(token)):
		return False
	leftNounIndex = findNearestNounIndex(tokens, tokenIndex-1, -1)
	rightNounIndex = findNearestNounIndex(tokens, tokenIndex+1, 1)
	if(leftNounIndex is None or rightNounIndex is None):
		return False
	if(leftNounIndex >= rightNounIndex):
		return False
	if(hasDeterministicDelimiterBetween(tokens, leftNounIndex, rightNounIndex)):
		return False
	candidateIndices = collectProbabilisticDelimiterIndices(tokens, leftNounIndex, rightNounIndex)
	if(len(candidateIndices) == 0):
		return False
	#NOTE: Only assign the first probabilistic delimiter per noun span for now (upgradeable in future).
	return candidateIndices[0] == tokenIndex


def findNearestNounIndex(tokens, startIndex, step):
	index = startIndex
	while(0 <= index < len(tokens)):
		if GIAANNproto_sequenceTokens.isConcept(tokens[index]):
			return index
		index += step
	return None

def hasDeterministicDelimiterBetween(tokens, startIndex, endIndex):
	for idx in range(startIndex+1, endIndex):
		token = tokens[idx]
		if(GIAANNproto_sequenceTokens.isTokenReferenceSetDelimiterDeterministic(token)):
			return True
	return False

def collectProbabilisticDelimiterIndices(tokens, startIndex, endIndex):
	candidateIndices = []
	for idx in range(startIndex+1, endIndex):
		token = tokens[idx]
		if(GIAANNproto_sequenceTokens.isTokenReferenceSetDelimiterProbabilistic(token)):
			if(not GIAANNproto_sequenceTokens.isTokenReferenceSetDelimiterDeterministic(token)):
				candidateIndices.append(idx)
	return candidateIndices
	

def createConceptMask(sequenceObservedColumns, tokens):
	conceptMask = pt.tensor([i in sequenceObservedColumns.columnsIndexSequenceWordIndexDict for i in range(len(tokens))], dtype=pt.bool)
	conceptIndices = pt.nonzero(conceptMask).squeeze(1)
	numberConcepts = conceptIndices.shape[0]
	return conceptMask, conceptIndices, numberConcepts
	
def processConceptWords(sequenceObservedColumns, sequenceIndex, sequence, tokens):
	"""
	For every concept word (lemma) in the sequence, identify every feature neuron in that column that occurs q words before or after the concept word in the sequence, including the concept neuron. This function has been parallelized using PyTorch array operations.
	"""

	if not usePOS:
		q = 5  # Fixed window size when not using POS tags

	# Identify all concept word indices
	conceptMask, conceptIndices, numberConceptsInSequence = createConceptMask(sequenceObservedColumns, tokens)
	#conceptIndices may be slightly longer than number of unique columns in sequence, if there are multiple instances of the same concept/noun lemma in the sequence
	
	if numberConceptsInSequence == 0:
		return  # No concept words to process

	if usePOS:
		# Sort conceptIndices
		conceptIndicesSorted = conceptIndices.sort().values
		
		# Find previous concept indices for each concept index
		prevConceptPositions = pt.searchsorted(conceptIndicesSorted, conceptIndices, right=False) - 1
		prevConceptExists = prevConceptPositions >= 0
		prevConceptPositions = prevConceptPositions.clamp(min=0)
		prevConceptIndices = pt.where(prevConceptExists, conceptIndicesSorted[prevConceptPositions], pt.zeros_like(conceptIndices))
		distToPrevConcept = pt.where(prevConceptExists, conceptIndices - prevConceptIndices, conceptIndices+1) #If no previous concept, distance is the index itself
		
		# Find next concept indices for each concept index
		nextConceptPositions = pt.searchsorted(conceptIndicesSorted, conceptIndices, right=True)
		nextConceptExists = nextConceptPositions < len(conceptIndices)
		nextConceptPositions = nextConceptPositions.clamp(max=len(nextConceptPositions)-1)
		nextConceptIndices = pt.where(nextConceptExists, conceptIndicesSorted[nextConceptPositions], pt.full_like(conceptIndices, len(sequence)))
		distToNextConcept = pt.where(nextConceptExists, nextConceptIndices - conceptIndices, len(sequence) - conceptIndices)
	else:
		q = 5
		distToPrevConcept = pt.full((conceptIndices.size(0),), q, dtype=pt.long)
		distToNextConcept = pt.full((conceptIndices.size(0),), q, dtype=pt.long)

	# Calculate start and end indices for each concept word
	if(conceptColumnsDelimitByPOS):
		databaseNetworkObject = sequenceObservedColumns.databaseNetworkObject
		conceptFeaturesDict = databaseNetworkObject.conceptFeaturesDict
		sequenceReferenceSetDelimiterList = databaseNetworkObject.sequenceReferenceSetDelimiterList
		sequenceLength = len(tokens)
		if(sequenceLength == 0):
			startIndices = pt.empty_like(conceptIndices)
			endIndices = pt.empty_like(conceptIndices)

		def token_is_reference_set_delimiter(token_index):
			isDelimiter = sequenceReferenceSetDelimiterList[token_index]
			return isDelimiter
		def has_next_reference_delimiter(token_index):
			next_index = token_index + 1
			if(next_index >= sequenceLength):
				return False
			return token_is_reference_set_delimiter(next_index)

		delimiterMaskList = []
		for posIndex in range(sequenceLength):
			if(token_is_reference_set_delimiter(posIndex)):
				#print("token_is_reference_set_delimiter: posIndex ", posIndex)
				if(has_next_reference_delimiter(posIndex)):
					delimiterMaskList.append(False)
				else:
					delimiterMaskList.append(True)
			else:
				delimiterMaskList.append(False)
		if(len(delimiterMaskList) == 0):
			delimiterIndices = pt.tensor([], dtype=conceptIndices.dtype)
		else:
			delimiterMask = pt.tensor(delimiterMaskList, dtype=pt.bool)
			delimiterIndices = pt.nonzero(delimiterMask).squeeze(1)
		if(delimiterIndices.numel() == 0):
			startIndices = pt.zeros_like(conceptIndices)
			endIndices = pt.full_like(conceptIndices, sequenceLength)
		else:
			delimiterIndicesSorted = delimiterIndices.sort().values
			prevDelimiterPositions = pt.searchsorted(delimiterIndicesSorted, conceptIndices, right=False) - 1
			prevDelimiterExists = prevDelimiterPositions >= 0
			prevDelimiterPositions = prevDelimiterPositions.clamp(min=0)
			prevDelimiterIndices = pt.where(prevDelimiterExists, delimiterIndicesSorted[prevDelimiterPositions], pt.full_like(conceptIndices, -1))
			startIndices = pt.where(prevDelimiterExists, prevDelimiterIndices + 1, pt.zeros_like(conceptIndices))
			nextDelimiterPositions = pt.searchsorted(delimiterIndicesSorted, conceptIndices, right=True)
			nextDelimiterExists = nextDelimiterPositions < delimiterIndicesSorted.shape[0]
			if(delimiterIndicesSorted.shape[0] > 0):
				nextDelimiterPositions = nextDelimiterPositions.clamp(max=delimiterIndicesSorted.shape[0]-1)
			nextDelimiterIndices = pt.where(nextDelimiterExists, delimiterIndicesSorted[nextDelimiterPositions], pt.full_like(conceptIndices, sequenceLength))
			endIndices = pt.where(nextDelimiterExists, nextDelimiterIndices + 1, pt.full_like(conceptIndices, sequenceLength))	#Include delimiter token in current column before advancing to next column
		startIndices = startIndices.clamp(min=0, max=sequenceLength)
		endIndices = endIndices.clamp(min=0, max=sequenceLength)
	elif(conceptColumnsDelimitByConceptFeaturesStart):
		if usePOS:
			startIndices = (conceptIndices).clamp(min=0)
			endIndices = (conceptIndices + distToNextConcept).clamp(max=len(sequence))
		else:
			startIndices = (conceptIndices).clamp(min=0)
			endIndices = (conceptIndices + q + 1).clamp(max=len(sequence))	
	elif(conceptColumnsDelimitByConceptFeaturesMid):
		if usePOS:
			startIndices = (conceptIndices - distToPrevConcept + 1).clamp(min=0)
			endIndices = (conceptIndices + distToNextConcept).clamp(max=len(sequence))
		else:
			startIndices = (conceptIndices - q).clamp(min=0)
			endIndices = (conceptIndices + q + 1).clamp(max=len(sequence))
	
	return conceptIndices, startIndices, endIndices

def processFeatures(sequenceObservedColumns, sequenceIndex, sequence, tokens, conceptIndices, startIndices, endIndices):
	numberConceptsInSequence = conceptIndices.shape[0]
	
	cs = sequenceObservedColumns.cs
	fs = sequenceObservedColumns.fs
	featureNeuronsActive = pt.zeros((arrayNumberOfSegments, cs, fs), dtype=arrayType)
	featureNeuronsWordOrder = pt.arange(fs).unsqueeze(0).repeat(cs, 1)
	pt.zeros((cs, fs), dtype=pt.long)
	columnsWordOrder = pt.zeros((cs), dtype=pt.long)
	featureNeuronsPos = pt.zeros((cs, fs), dtype=arrayType)
	if(trainSequenceObservedColumnsMatchSequenceWords):
		sequenceConceptIndexMask = pt.ones((cs, fs), dtype=arrayType)
	else:
		sequenceConceptIndexMask = None
	if(useSANI):
		featureNeuronsSegmentMask = pt.zeros((cs, arrayNumberOfSegments), dtype=arrayType)	#note this mask is for permanence updates (it assumes that the network has been constructed with forward column connections only)
	else:
		featureNeuronsSegmentMask = pt.ones((cs, arrayNumberOfSegments), dtype=arrayType)
	
	conceptIndicesList = conceptIndices.tolist()
	for i, sequenceConceptWordIndex in enumerate(conceptIndicesList):
		if(trainSequenceObservedColumnsMatchSequenceWords):
			sequenceConceptIndex = i
		else:
			conceptLemma = tokens[sequenceConceptWordIndex].lemma
			sequenceConceptIndex = sequenceObservedColumns.conceptNameToIndex[conceptLemma] 
				
		if(useSANI):
			numberOfSegments = min(arrayNumberOfSegments, i+1)
			featureNeuronsSegmentMask[sequenceConceptIndex, :] = pt.cat([pt.zeros(arrayNumberOfSegments-numberOfSegments), pt.ones(numberOfSegments)], dim=0)
			minSequentialSegmentIndex = max(0, arrayNumberOfSegments-sequenceConceptIndex-1)
			activeSequentialSegments = pt.arange(minSequentialSegmentIndex, arrayNumberOfSegments, 1)
		if(trainSequenceObservedColumnsUseSequenceFeaturesOnly and trainSequenceObservedColumnsMatchSequenceWords):
			if(useSANI):
				featureNeuronsActive[activeSequentialSegments, sequenceConceptIndex, startIndices[sequenceConceptIndex]:endIndices[sequenceConceptIndex]] = 1
			else:
				featureNeuronsActive[arrayIndexSegmentFirst, sequenceConceptIndex, startIndices[sequenceConceptIndex]:endIndices[sequenceConceptIndex]] = 1
			columnsWordOrder[sequenceConceptIndex] = sequenceConceptIndex
			sequenceConceptIndexMask[:, sequenceConceptWordIndex] = 0
			sequenceConceptIndexMask[sequenceConceptIndex, sequenceConceptWordIndex] = 1
			for j in range(startIndices[sequenceConceptIndex], endIndices[sequenceConceptIndex]):
				featurePos = posStringToPosInt(sequenceObservedColumns.databaseNetworkObject.nlp, tokens[j].pos)
				featureNeuronsPos[sequenceConceptIndex, j] = featurePos
				featureNeuronsWordOrder[sequenceConceptIndex, j] = j
		else:
			for j in range(startIndices[i], endIndices[i]):
				featureWord = tokens[j].word	#redundant: .lower()
				featureLemma = tokens[j].lemma
				featurePos = posStringToPosInt(sequenceObservedColumns.databaseNetworkObject.nlp, tokens[j].pos)
				if(j in sequenceObservedColumns.columnsIndexSequenceWordIndexDict):
					sequenceConceptWordIndex = j
					columnsWordOrder[sequenceConceptIndex] = sequenceConceptIndex
					if(useDedicatedConceptNames2):
						sequenceFeatureIndex = sequenceObservedColumns.featureWordToIndex[variableConceptNeuronFeatureName]
					else:
						sequenceFeatureIndex = sequenceObservedColumns.featureWordToIndex[featureLemma]
					if(useSANI):
						featureNeuronsActive[activeSequentialSegments, sequenceConceptIndex, sequenceFeatureIndex] = 1
					else:
						featureNeuronsActive[arrayIndexSegmentFirst, sequenceConceptIndex, sequenceFeatureIndex] = 1
				elif(featureWord in sequenceObservedColumns.featureWordToIndex):
					sequenceFeatureIndex = sequenceObservedColumns.featureWordToIndex[featureWord]
					if(useSANI):
						featureNeuronsActive[activeSequentialSegments, sequenceConceptIndex, sequenceFeatureIndex] = 1
					else:
						featureNeuronsActive[arrayIndexSegmentFirst, sequenceConceptIndex, sequenceFeatureIndex] = 1
				featureNeuronsWordOrder[sequenceConceptIndex, sequenceFeatureIndex] = j
				featureNeuronsPos[sequenceConceptIndex, sequenceFeatureIndex] = featurePos
	
	featureNeuronsSegmentMask = featureNeuronsSegmentMask.swapdims(0, 1)	#swap from dims [c, s] to [s, c] (in line with featureNeuronsActive)
	#print("featureNeuronsSegmentMask = ", featureNeuronsSegmentMask)	
	if(debugDrawNeuronActivations):
		startList = startIndices.tolist()
		endList = endIndices.tolist()
		for seqConceptIndex, (startIdx, endIdx) in enumerate(zip(startList, endList)):
			columnName = None
			if(hasattr(sequenceObservedColumns, "sequenceObservedColumnsDict") and seqConceptIndex in sequenceObservedColumns.sequenceObservedColumnsDict):
				columnName = sequenceObservedColumns.sequenceObservedColumnsDict[seqConceptIndex].conceptName
			elif(hasattr(sequenceObservedColumns, "observedColumnsDict2") and seqConceptIndex in sequenceObservedColumns.observedColumnsDict2):
				columnName = sequenceObservedColumns.observedColumnsDict2[seqConceptIndex].conceptName
			tokenSpan = ""
			if(endIdx > startIdx and len(tokens) > 0):
				tokenSlice = []
				for tokenIndex in range(startIdx, min(endIdx, len(tokens))):
					tokenSlice.append(tokens[tokenIndex].word)
				tokenSpan = " ".join(tokenSlice)
			
			if(debugPrintNeuronActivations):
				print(f"\tprocessFeatures debug: columnIndex={seqConceptIndex}, columnName={columnName}, localRange=[{startIdx}, {endIdx}), tokens={tokenSpan}")
	
	return featureNeuronsActive, cs, fs, sequenceConceptIndexMask, columnsWordOrder, featureNeuronsWordOrder, featureNeuronsPos, featureNeuronsSegmentMask
