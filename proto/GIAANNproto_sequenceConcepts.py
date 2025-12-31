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
import GIAANNproto_databaseNetworkExcitation
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
			conceptsFound, newConceptsAdded = GIAANNproto_databaseNetworkExcitation.addConceptToConceptColumnsDict(databaseNetworkObject, token.lemma, conceptsFound, newConceptsAdded)
			conceptMask.append(True)
		else:
			conceptMask.append(False)

	# If new concept columns have been added, expand arrays as needed
	if newConceptsAdded:
		if not lowMem:
			# Expand global feature neuron arrays
			if databaseNetworkObject.globalFeatureNeurons.shape[3] < databaseNetworkObject.c:
				newShape = (databaseNetworkObject.globalFeatureNeurons.shape[0], databaseNetworkObject.globalFeatureNeurons.shape[1], databaseNetworkObject.globalFeatureNeurons.shape[2], databaseNetworkObject.c, databaseNetworkObject.globalFeatureNeurons.shape[4])
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
				observedColumn = GIAANNproto_databaseNetworkExcitation.loadOrCreateObservedColumn(databaseNetworkObject, conceptIndex, lemma, i)
				observedColumnsDict[lemma] = observedColumn
				observedColumnsSequenceWordIndexDict[i] = observedColumn
		else:
			conceptIndex = databaseNetworkObject.conceptColumnsDict[lemma]
			# Load observed column from disk or create new one
			observedColumn = GIAANNproto_databaseNetworkExcitation.loadOrCreateObservedColumn(databaseNetworkObject, conceptIndex, lemma, i)
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
		if databaseNetworkObject.f > databaseNetworkObject.globalFeatureNeurons.shape[4]:
			extraCols = databaseNetworkObject.f - databaseNetworkObject.globalFeatureNeurons.shape[4]
			newShape = (databaseNetworkObject.globalFeatureNeurons.shape[0], databaseNetworkObject.globalFeatureNeurons.shape[1], databaseNetworkObject.globalFeatureNeurons.shape[2], databaseNetworkObject.globalFeatureNeurons.shape[3], databaseNetworkObject.f)
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

def token_is_reference_set_delimiter(token_index, sequenceReferenceSetDelimiterList):
	isDelimiter = sequenceReferenceSetDelimiterList[token_index]
	return bool(isDelimiter)

def has_next_reference_delimiter(token_index, sequenceLength, sequenceReferenceSetDelimiterList):
	next_index = token_index + 1
	if(next_index >= sequenceLength):
		return False
	return token_is_reference_set_delimiter(next_index, sequenceReferenceSetDelimiterList)

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
		if(detectReferenceSetDelimitersBetweenNouns):
			conceptFeaturesReferenceSetDelimiterDeterministicList = databaseNetworkObject.conceptFeaturesReferenceSetDelimiterDeterministicList
			conceptFeaturesReferenceSetDelimiterProbabilisticList = databaseNetworkObject.conceptFeaturesReferenceSetDelimiterProbabilisticList
			conceptFeaturesReferenceSetDelimiterList = None
		else:
			conceptFeaturesReferenceSetDelimiterList = databaseNetworkObject.conceptFeaturesReferenceSetDelimiterList
			conceptFeaturesReferenceSetDelimiterDeterministicList = None
			conceptFeaturesReferenceSetDelimiterProbabilisticList = None
		sequenceLength = len(tokens)
		if(sequenceLength == 0):
			startIndices = pt.empty_like(conceptIndices)
			endIndices = pt.empty_like(conceptIndices)
		else:
			conceptIndicesSorted = conceptIndices.sort().values
			numConcepts = conceptIndicesSorted.shape[0]
			if(numConcepts <= 1):
				startIndices = pt.zeros_like(conceptIndicesSorted)
				endIndices = pt.full_like(conceptIndicesSorted, sequenceLength)
			else:
				delimiterIndices = []
				# Choose the rightmost delimiter between adjacent concept tokens to define boundaries.
				for conceptPosition in range(numConcepts - 1):
					leftIndex = int(conceptIndicesSorted[conceptPosition].item())
					rightIndex = int(conceptIndicesSorted[conceptPosition + 1].item())
					rightmostDeterministic = None
					rightmostIndeterministic = None
					for tokenIndex in range(leftIndex + 1, rightIndex):
						token = tokens[tokenIndex]
						featureIndex = conceptFeaturesDict.get(token.word)
						if(featureIndex is None):
							continue
						if(detectReferenceSetDelimitersBetweenNouns):
							if(featureIndex < len(conceptFeaturesReferenceSetDelimiterDeterministicList) and conceptFeaturesReferenceSetDelimiterDeterministicList[featureIndex]):
								rightmostDeterministic = tokenIndex
							elif(featureIndex < len(conceptFeaturesReferenceSetDelimiterProbabilisticList) and conceptFeaturesReferenceSetDelimiterProbabilisticList[featureIndex]):
								rightmostIndeterministic = tokenIndex
						else:
							if(featureIndex < len(conceptFeaturesReferenceSetDelimiterList) and conceptFeaturesReferenceSetDelimiterList[featureIndex]):
								rightmostDeterministic = tokenIndex
					if(rightmostDeterministic is not None):
						delimiterIndices.append(rightmostDeterministic)
					elif(rightmostIndeterministic is not None):
						delimiterIndices.append(rightmostIndeterministic)
					else:
						print("warning: no delimiter detected between concept tokens")
						return None
				startIndices = pt.zeros_like(conceptIndicesSorted)
				endIndices = pt.full_like(conceptIndicesSorted, sequenceLength)
				for conceptPosition, delimiterIndex in enumerate(delimiterIndices):
					endIndices[conceptPosition] = delimiterIndex + 1
					startIndices[conceptPosition + 1] = delimiterIndex + 1
			conceptIndices = conceptIndicesSorted
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
	
	sentenceWithConceptAssignment = buildSequenceConceptAssignment(sequenceObservedColumns, sequence, tokens, conceptIndices, startIndices, endIndices)
	if(debugPrintTrainSequenceConceptAssignment):
		sequenceObservedColumns.sentenceWithConceptAssignment = sentenceWithConceptAssignment
		#print(f"Processing sequenceCount: {sequenceIndex}, {sequenceObservedColumns.sentenceWithConceptAssignment}")

	return conceptIndices, startIndices, endIndices

def buildSequenceConceptAssignment(sequenceObservedColumns, sequence, tokens, conceptIndices, startIndices, endIndices):
	tokenConceptColumnIndexList = [None] * len(tokens)
	conceptIndicesList = conceptIndices.tolist()
	startList = startIndices.tolist()
	endList = endIndices.tolist()
	for conceptPosition, conceptWordIndex in enumerate(conceptIndicesList):
		columnIndex = sequenceObservedColumns.columnsIndexSequenceWordIndexDict.get(conceptWordIndex)
		if(columnIndex is None):
			continue
		startIndexValue = max(0, int(startList[conceptPosition]))
		endIndexValue = min(len(tokens), int(endList[conceptPosition]))
		for tokenIndex in range(startIndexValue, endIndexValue):
			tokenConceptColumnIndexList[tokenIndex] = columnIndex
	sequenceObservedColumns.tokenConceptColumnIndexList = tokenConceptColumnIndexList
	conceptColumnsList = sequenceObservedColumns.databaseNetworkObject.conceptColumnsList
	def getTokenDisplayText(token):
		if hasattr(token, "text"):
			return token.text
		if hasattr(token, "word"):
			return token.word
		return str(token)
	sentenceWithConceptAssignment = " ".join(
		f"{getTokenDisplayText(token)} ({tokenIndex}:{conceptColumnsList[tokenConceptColumnIndexList[tokenIndex]] if tokenConceptColumnIndexList[tokenIndex] is not None else 'none'})"
		for tokenIndex, token in enumerate(sequence)
	)
	return sentenceWithConceptAssignment

def processFeatures(sequenceObservedColumns, sequenceIndex, sequence, tokens, conceptIndices, startIndices, endIndices):
	numberConceptsInSequence = conceptIndices.shape[0]
	
	cs = sequenceObservedColumns.cs
	fs = sequenceObservedColumns.fs
	featureNeuronsActive = pt.zeros((numberOfDendriticBranches, arrayNumberOfSegments, cs, fs), dtype=arrayType)
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
	branchCounters = None
	if(multipleDendriticBranches):
		branchCounters = [{} for _ in range(cs)]
	
	conceptIndicesList = conceptIndices.tolist()
	for i, sequenceConceptWordIndex in enumerate(conceptIndicesList):
		if(trainSequenceObservedColumnsMatchSequenceWords):
			sequenceConceptIndex = i
		else:
			conceptLemma = tokens[sequenceConceptWordIndex].lemma
			sequenceConceptIndex = sequenceObservedColumns.conceptNameToIndex[conceptLemma] 
				
		if(useSANI):
			# When useSANIcolumns is True (original behaviour), assign segment indices based on
			# the concept/column position in the sequence (sequenceConceptIndex).
			#
			# When useSANIfeatures is True (legacy !useSANIcolumns behaviour), assign segment
			# indices based on the underlying feature/word position in the sentence
			# (sequenceConceptWordIndex), so that feature proximity is captured by the
			# sequential segments.
			#
			# When useSANIfeaturesAndColumns is enabled, apply column-distance segments first,
			# then add feature-distance segments based on sequenceConceptWordIndex.
			segmentMask = pt.zeros(arrayNumberOfSegments, dtype=arrayType)
			if(useSANIcolumns):
				positionIndex = sequenceConceptIndex
				numberOfSegments = min(arrayNumberOfSegments, positionIndex+1)
				segmentMask[:numberOfSegments] = 1
				activeSequentialSegments = pt.arange(0, numberOfSegments, 1)
			elif(useSANIfeatures):
				# sequenceConceptWordIndex is the absolute token index of this concept's word
				# in the original sequence; this gives "feature-position-based" segments.
				positionIndex = sequenceConceptWordIndex
				numberOfSegments = min(arrayNumberOfSegments, positionIndex+1)
				segmentMask[:numberOfSegments] = 1
				activeSequentialSegments = pt.arange(0, numberOfSegments, 1)
			elif(useSANIfeaturesAndColumns):
				# Assign concept/column-distance segments first, then feature-distance segments.
				# Note: when useSANIfeaturesAndColumnsInternal is enabled, include the internal
				# column segment (sequenceConceptIndex==0) in the concept segment budget.
				if(useSANIfeaturesAndColumnsInternal):
					columnSegments = min(arrayNumberOfSegmentsColumnDistance, sequenceConceptIndex+1)
				else:
					# External columns only: exclude the internal column from column-distance segments.
					columnSegments = min(arrayNumberOfSegmentsColumnDistance, max(sequenceConceptIndex, 0))
				featureSegments = min(arrayNumberOfSegmentsFeatureDistance, sequenceConceptWordIndex+1)
				if(columnSegments > 0):
					segmentMask[:columnSegments] = 1
				featureSegmentStart = arrayNumberOfSegmentsColumnDistance
				featureSegmentEnd = min(arrayNumberOfSegments, featureSegmentStart + featureSegments)
				if(featureSegmentEnd > featureSegmentStart):
					segmentMask[featureSegmentStart:featureSegmentEnd] = 1
				activeSequentialSegments = pt.nonzero(segmentMask > 0, as_tuple=False).view(-1)
			featureNeuronsSegmentMask[sequenceConceptIndex, :] = segmentMask
		if(trainSequenceObservedColumnsUseSequenceFeaturesOnly and trainSequenceObservedColumnsMatchSequenceWords):
			branchIndex = 0
			if(multipleDendriticBranches):
				featureBranchCounts = branchCounters[sequenceConceptIndex]
				startIndexValue = int(startIndices[sequenceConceptIndex].item())
				endIndexValue = int(endIndices[sequenceConceptIndex].item())
				featureIndicesInObservedTensor = sequenceObservedColumns.featureIndicesInObservedTensor
				for j in range(startIndexValue, endIndexValue):
					if(j >= featureIndicesInObservedTensor.shape[0]):
						continue
					globalFeatureIndex = int(featureIndicesInObservedTensor[j].item())
					branchIndex = featureBranchCounts.get(globalFeatureIndex, 0)
					featureBranchCounts[globalFeatureIndex] = branchIndex + 1
					if(branchIndex >= numberOfDendriticBranches):
						branchIndex = numberOfDendriticBranches - 1
					if(useSANI):
						featureNeuronsActive[branchIndex, activeSequentialSegments, sequenceConceptIndex, j] = 1
					else:
						featureNeuronsActive[branchIndex, arrayIndexSegmentFirst, sequenceConceptIndex, j] = 1
					featurePos = posStringToPosInt(sequenceObservedColumns.databaseNetworkObject.nlp, tokens[j].pos)
					featureNeuronsPos[sequenceConceptIndex, j] = featurePos
					featureNeuronsWordOrder[sequenceConceptIndex, j] = j
			else:
				if(useSANI):
					featureNeuronsActive[branchIndex, activeSequentialSegments, sequenceConceptIndex, startIndices[sequenceConceptIndex]:endIndices[sequenceConceptIndex]] = 1
				else:
					featureNeuronsActive[branchIndex, arrayIndexSegmentFirst, sequenceConceptIndex, startIndices[sequenceConceptIndex]:endIndices[sequenceConceptIndex]] = 1
			columnsWordOrder[sequenceConceptIndex] = sequenceConceptIndex
			sequenceConceptIndexMask[:, sequenceConceptWordIndex] = 0
			sequenceConceptIndexMask[sequenceConceptIndex, sequenceConceptWordIndex] = 1
			if(not multipleDendriticBranches):
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
					branchIndex = 0
					if(multipleDendriticBranches):
						featureBranchCounts = branchCounters[sequenceConceptIndex]
						branchIndex = featureBranchCounts.get(sequenceFeatureIndex, 0)
						featureBranchCounts[sequenceFeatureIndex] = branchIndex + 1
						if(branchIndex >= numberOfDendriticBranches):
							branchIndex = numberOfDendriticBranches - 1
					if(useSANI):
						featureNeuronsActive[branchIndex, activeSequentialSegments, sequenceConceptIndex, sequenceFeatureIndex] = 1
					else:
						featureNeuronsActive[branchIndex, arrayIndexSegmentFirst, sequenceConceptIndex, sequenceFeatureIndex] = 1
				elif(featureWord in sequenceObservedColumns.featureWordToIndex):
					sequenceFeatureIndex = sequenceObservedColumns.featureWordToIndex[featureWord]
					branchIndex = 0
					if(multipleDendriticBranches):
						featureBranchCounts = branchCounters[sequenceConceptIndex]
						branchIndex = featureBranchCounts.get(sequenceFeatureIndex, 0)
						featureBranchCounts[sequenceFeatureIndex] = branchIndex + 1
						if(branchIndex >= numberOfDendriticBranches):
							branchIndex = numberOfDendriticBranches - 1
					if(useSANI):
						featureNeuronsActive[branchIndex, activeSequentialSegments, sequenceConceptIndex, sequenceFeatureIndex] = 1
					else:
						featureNeuronsActive[branchIndex, arrayIndexSegmentFirst, sequenceConceptIndex, sequenceFeatureIndex] = 1
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
			
	return featureNeuronsActive, cs, fs, sequenceConceptIndexMask, columnsWordOrder, featureNeuronsWordOrder, featureNeuronsPos, featureNeuronsSegmentMask
