"""GIAANNnlp_sequenceConcepts.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 BAI Research Pty Ltd (bairesearch.com.au)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN NLP sequence Concepts (and feature detection)

"""

import random
import torch as pt

from GIAANNcmn_globalDefs import *
import GIAANNcmn_databaseNetwork
import GIAANNnlp_sequenceTokens


def isTokenConceptColumnCandidate(token, tokens, tokenIndex):
	result = False
	if(tokeniserSubword):
		if(tokens is None):
			raise RuntimeError("isTokenConceptColumnCandidate error: tokens must not be None when tokeniserSubword is enabled")
		if(not isinstance(tokenIndex, int) or isinstance(tokenIndex, bool)):
			raise RuntimeError("isTokenConceptColumnCandidate error: tokenIndex must be an int")
		if(tokenIndex < 0 or tokenIndex >= len(tokens)):
			raise RuntimeError("isTokenConceptColumnCandidate error: tokenIndex out of range")
	if usePOS:
		if(GIAANNnlp_sequenceTokens.isConcept(token)):
			result = True
			if(tokeniserSubword):
				if(tokenIndex > 0 and GIAANNnlp_sequenceTokens.isConcept(tokens[tokenIndex - 1])):
					result = False
	else:
		result = True
	return result

def getTokenConceptName(databaseNetworkObject, token):
	result = None
	if(useDedicatedConceptsLists):
		if(tokeniserSubword and useDedicatedConceptListsSubword):
			conceptIndex = getTokenConceptIndex(databaseNetworkObject, token)
			result = databaseNetworkObject.conceptColumnsList[conceptIndex]
		else:
			raise RuntimeError("getTokenConceptName error: useDedicatedConceptsLists is only implemented for tokeniserSubword/useDedicatedConceptListsSubword")
	else:
		result = token.lemma
	return result

def getTokenConceptIndex(databaseNetworkObject, token):
	result = None
	if(useDedicatedConceptsLists):
		if(tokeniserSubword and useDedicatedConceptListsSubword):
			result = GIAANNnlp_sequenceTokens.getTokeniserSubwordConceptIndex(token)
			if(result < 0 or result >= databaseNetworkObject.c):
				raise RuntimeError("getTokenConceptIndex error: tokeniserSubword concept index out of range")
		else:
			raise RuntimeError("getTokenConceptIndex error: useDedicatedConceptsLists is only implemented for tokeniserSubword/useDedicatedConceptListsSubword")
	else:
		if(token.lemma not in databaseNetworkObject.conceptColumnsDict):
			raise RuntimeError("getTokenConceptIndex error: concept lemma not found (" + token.lemma + ")")
		result = databaseNetworkObject.conceptColumnsDict[token.lemma]
	return result

def firstPass(databaseNetworkObject, sequence, allowNewFeatures):
	newConceptsAdded = False
	conceptsFound = False
	conceptMask = []
	tokens = None
	if(tokeniserSubword):
		tokens = GIAANNnlp_sequenceTokens.getTokens(sequence)
	
	for tokenIndex, preprocessedToken in enumerate(sequence):
		if(tokeniserSubword):
			token = tokens[tokenIndex]
		else:
			token = GIAANNnlp_sequenceTokens.convertPreprocessedTokenToSequenceToken(preprocessedToken)

		conceptFound = False
		if usePOS:
			if isTokenConceptColumnCandidate(token, tokens, tokenIndex):
				# Only assign unique concept columns for nouns
				conceptFound = True
		else:
			# When usePOS is disabled, assign concept columns for every new lemma encountered
			conceptFound = True
		
		if(conceptFound):
			if(useDedicatedConceptsLists):
				getTokenConceptIndex(databaseNetworkObject, token)
				conceptsFound = True
			else:
				if(allowNewFeatures):
					conceptsFound, newConceptsAdded = GIAANNcmn_databaseNetwork.addConceptToConceptColumnsDict(databaseNetworkObject, token.lemma, conceptsFound, newConceptsAdded)
				else:
					if(token.lemma not in databaseNetworkObject.conceptColumnsDict):
						raise RuntimeError("firstPass error: concept lemma not found while allowNewFeatures is False (" + token.lemma + ")")
					conceptsFound = True
			conceptMask.append(True)
		else:
			conceptMask.append(False)

	# If new concept columns have been added, expand arrays as needed
	if newConceptsAdded:
		if storeDatabaseGlobalFeatureNeuronsInRam:
			# Expand global feature neuron arrays
			if databaseNetworkObject.globalFeatureNeurons.shape[3] < databaseNetworkObject.c:
				newShape = (databaseNetworkObject.globalFeatureNeurons.shape[0], databaseNetworkObject.globalFeatureNeurons.shape[1], databaseNetworkObject.globalFeatureNeurons.shape[2], databaseNetworkObject.c, databaseNetworkObject.globalFeatureNeurons.shape[4])
				databaseNetworkObject.globalFeatureNeurons = pt.sparse_coo_tensor(databaseNetworkObject.globalFeatureNeurons._indices(), databaseNetworkObject.globalFeatureNeurons._values(), size=newShape, dtype=arrayType, device=deviceSparse)
				
	return conceptsFound, conceptMask

				
def secondPass(databaseNetworkObject, tokens, inferenceMode):
	observedColumnsDict = {}
	observedColumnsSequenceWordIndexDict = {}
	for i, token in enumerate(tokens):
		conceptFound = False
		if usePOS:
			if isTokenConceptColumnCandidate(token, tokens, i):
				conceptFound = True
		else:
			conceptFound = True
		if(conceptFound):
			lemma = getTokenConceptName(databaseNetworkObject, token)
			conceptIndex = getTokenConceptIndex(databaseNetworkObject, token)
			if(inferenceMode and inferenceOnlyRetainPredictedTargetObservedColumn):
				observedColumn = GIAANNcmn_databaseNetwork.ObservedColumnStub(databaseNetworkObject, conceptIndex, lemma, i)
				observedColumnsSequenceWordIndexDict[i] = observedColumn
			else:
				# Load observed column from disk or create new one (reuse per-lemma instance for multi-occurrence concepts)
				if(lemma in observedColumnsDict):
					observedColumn = observedColumnsDict[lemma]
				else:
					if(inferenceMode):
						observedColumn = GIAANNcmn_databaseNetwork.loadOrCreateObservedColumn(databaseNetworkObject, conceptIndex, lemma, i, deviceLoadColumnInference, inferenceMode and deviceLoadColumnInferenceCopy)
					else:
						observedColumn = GIAANNcmn_databaseNetwork.loadOrCreateObservedColumn(databaseNetworkObject, conceptIndex, lemma, i)
					observedColumnsDict[lemma] = observedColumn
				observedColumnsSequenceWordIndexDict[i] = observedColumn
	return observedColumnsDict, observedColumnsSequenceWordIndexDict


def detectNewFeatures(databaseNetworkObject, tokens, allowNewFeatures):
	"""
	When usePOS mode is enabled, detect all possible new features in the sequence
	by searching for all new non-nouns in the sequence.
	"""

	if(conceptColumnsDelimitByPOS):
		databaseNetworkObject.sequenceReferenceSetDelimiterList = [None]*len(tokens)
		if(detectReferenceSetDelimitersBetweenNouns):
			databaseNetworkObject.sequenceReferenceSetDelimiterDeterministicList = [False]*len(tokens)
			databaseNetworkObject.sequenceReferenceSetDelimiterProbabilisticList = [False]*len(tokens)

	numNewFeatures = 0
	for tokenIndex, token in enumerate(tokens):
		if(processFeatureDetection(databaseNetworkObject, tokenIndex, token, tokens, allowNewFeatures)):
			numNewFeatures += 1
	
	# After processing all features, update f
	if(allowNewFeatures):
		databaseNetworkObject.f += numNewFeatures
	
	# Now, expand arrays accordingly
	if(allowNewFeatures and storeDatabaseGlobalFeatureNeuronsInRam):
		if databaseNetworkObject.f > databaseNetworkObject.globalFeatureNeurons.shape[4]:
			extraCols = databaseNetworkObject.f - databaseNetworkObject.globalFeatureNeurons.shape[4]
			newShape = (databaseNetworkObject.globalFeatureNeurons.shape[0], databaseNetworkObject.globalFeatureNeurons.shape[1], databaseNetworkObject.globalFeatureNeurons.shape[2], databaseNetworkObject.globalFeatureNeurons.shape[3], databaseNetworkObject.f)
			databaseNetworkObject.globalFeatureNeurons = databaseNetworkObject.globalFeatureNeurons.coalesce()
			databaseNetworkObject.globalFeatureNeurons = pt.sparse_coo_tensor(databaseNetworkObject.globalFeatureNeurons.indices(), databaseNetworkObject.globalFeatureNeurons.values(), size=newShape, dtype=arrayType, device=deviceSparse)

def processFeatureDetection(databaseNetworkObject, tokenIndex, token, tokens, allowNewFeatures):
	"""
	Helper function to detect new features prior to processing concept words.
	"""
	
	result = False
	featureWord = token.word

	if usePOS and isTokenConceptColumnCandidate(token, tokens, tokenIndex):
		result = False  # Skip nouns as features
	else:
		if(tokeniserSubword and useDedicatedFeatureListsSubword):
			featureIndex = getTokenFeatureIndex(databaseNetworkObject, token)
			isDelimiter, isDelimiterDeterministic, isDelimiterProbabilistic = isFeaturePOSreferenceSetDelimiterType(featureWord, token, tokens, tokenIndex)
			if(conceptColumnsDelimitByPOS):
				databaseNetworkObject.sequenceReferenceSetDelimiterList[tokenIndex] = isDelimiter
				if(detectReferenceSetDelimitersBetweenNouns):
					databaseNetworkObject.sequenceReferenceSetDelimiterDeterministicList[tokenIndex] = isDelimiterDeterministic
					databaseNetworkObject.sequenceReferenceSetDelimiterProbabilisticList[tokenIndex] = isDelimiterProbabilistic
					databaseNetworkObject.conceptFeaturesReferenceSetDelimiterDeterministicList[featureIndex] = databaseNetworkObject.conceptFeaturesReferenceSetDelimiterDeterministicList[featureIndex] or isDelimiterDeterministic
					databaseNetworkObject.conceptFeaturesReferenceSetDelimiterProbabilisticList[featureIndex] = databaseNetworkObject.conceptFeaturesReferenceSetDelimiterProbabilisticList[featureIndex] or isDelimiterProbabilistic
				else:
					databaseNetworkObject.conceptFeaturesReferenceSetDelimiterList[featureIndex] = databaseNetworkObject.conceptFeaturesReferenceSetDelimiterList[featureIndex] or isDelimiter
			result = False
		else:
			if featureWord not in databaseNetworkObject.conceptFeaturesDict:
				if(not allowNewFeatures):
					raise RuntimeError("processFeatureDetection error: feature word not found while allowNewFeatures is False (" + featureWord + ")")
				featureIndex = len(databaseNetworkObject.conceptFeaturesDict)
				databaseNetworkObject.conceptFeaturesDict[featureWord] = featureIndex
				databaseNetworkObject.conceptFeaturesList.append(featureWord)
				if(trainStoreFeatureMapsGlobally):
					databaseNetworkObject.conceptFeaturesIndexToWordDict[featureIndex] = featureWord
				isDelimiter, isDelimiterDeterministic, isDelimiterProbabilistic = isFeaturePOSreferenceSetDelimiterType(featureWord, token, tokens, tokenIndex)
				if(conceptColumnsDelimitByPOS):
					databaseNetworkObject.sequenceReferenceSetDelimiterList[tokenIndex] = isDelimiter
					if(detectReferenceSetDelimitersBetweenNouns):
						databaseNetworkObject.sequenceReferenceSetDelimiterDeterministicList[tokenIndex] = isDelimiterDeterministic
						databaseNetworkObject.sequenceReferenceSetDelimiterProbabilisticList[tokenIndex] = isDelimiterProbabilistic
						databaseNetworkObject.conceptFeaturesReferenceSetDelimiterDeterministicList.append(isDelimiterDeterministic)
						databaseNetworkObject.conceptFeaturesReferenceSetDelimiterProbabilisticList.append(isDelimiterProbabilistic)
					else:
						databaseNetworkObject.conceptFeaturesReferenceSetDelimiterList.append(isDelimiter)
				result = True
			else:
				if(conceptColumnsDelimitByPOS):
					isDelimiter, isDelimiterDeterministic, isDelimiterProbabilistic = isFeaturePOSreferenceSetDelimiterType(featureWord, token, tokens, tokenIndex)
					databaseNetworkObject.sequenceReferenceSetDelimiterList[tokenIndex] = isDelimiter	#deterministic or incontext probabilistic reference set delimiter detected (train only)
					if(detectReferenceSetDelimitersBetweenNouns):
						databaseNetworkObject.sequenceReferenceSetDelimiterDeterministicList[tokenIndex] = isDelimiterDeterministic
						databaseNetworkObject.sequenceReferenceSetDelimiterProbabilisticList[tokenIndex] = isDelimiterProbabilistic
						featureIndex = databaseNetworkObject.conceptFeaturesDict[featureWord]
						databaseNetworkObject.conceptFeaturesReferenceSetDelimiterProbabilisticList[featureIndex] = databaseNetworkObject.conceptFeaturesReferenceSetDelimiterProbabilisticList[featureIndex] or isDelimiterProbabilistic	#reassign probabilistic if ever probabilistic in past (inference only)
				result = False
	return result

def getTokenFeatureIndex(databaseNetworkObject, token):
	result = None
	if(tokeniserSubword and useDedicatedFeatureListsSubword):
		result = GIAANNnlp_sequenceTokens.getTokeniserSubwordFeatureIndex(token)
		if(result < 0 or result >= databaseNetworkObject.f):
			raise RuntimeError("getTokenFeatureIndex error: tokeniserSubword feature index out of range")
	else:
		if(token.word not in databaseNetworkObject.conceptFeaturesDict):
			raise RuntimeError("getTokenFeatureIndex error: feature word not found (" + token.word + ")")
		result = databaseNetworkObject.conceptFeaturesDict[token.word]
	return result

def getTokenFeatureIndexIfKnown(databaseNetworkObject, token):
	result = None
	if(tokeniserSubword and useDedicatedFeatureListsSubword):
		result = getTokenFeatureIndex(databaseNetworkObject, token)
	else:
		result = databaseNetworkObject.conceptFeaturesDict.get(token.word)
	return result
	
def isFeaturePOSreferenceSetDelimiterType(nodeNameString, token, tokens, tokenIndex):
	isDelimiterDeterministic = False
	isDelimiterProbabilistic = False
	if(conceptColumnsDelimitByPOS):
		nodeWordLower = nodeNameString.lower()
		if(GIAANNnlp_sequenceTokens.isTokenReferenceSetDelimiterDeterministic(token)):
			isDelimiterDeterministic = True
		if(detectReferenceSetDelimitersBetweenNouns and not isDelimiterDeterministic):
			isDelimiterProbabilistic = detectProbabilisticReferenceSetDelimiterBetweenNouns(nodeWordLower, token, tokens, tokenIndex)
		isDelimiter = isDelimiterDeterministic or isDelimiterProbabilistic
	else:
		printe("conceptColumnsDelimitByPOS is required")
	return isDelimiter, isDelimiterDeterministic, isDelimiterProbabilistic


def detectProbabilisticReferenceSetDelimiterBetweenNouns(nodeWordLower, token, tokens, tokenIndex):
	if(tokenIndex < 0 or tokenIndex >= len(tokens)):
		return False
	if(not GIAANNnlp_sequenceTokens.isTokenReferenceSetDelimiterProbabilistic(token)):
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
		if GIAANNnlp_sequenceTokens.isConcept(tokens[index]):
			return index
		index += step
	return None

def hasDeterministicDelimiterBetween(tokens, startIndex, endIndex):
	for idx in range(startIndex+1, endIndex):
		token = tokens[idx]
		if(GIAANNnlp_sequenceTokens.isTokenReferenceSetDelimiterDeterministic(token)):
			return True
	return False

def collectProbabilisticDelimiterIndices(tokens, startIndex, endIndex):
	candidateIndices = []
	for idx in range(startIndex+1, endIndex):
		token = tokens[idx]
		if(GIAANNnlp_sequenceTokens.isTokenReferenceSetDelimiterProbabilistic(token)):
			if(not GIAANNnlp_sequenceTokens.isTokenReferenceSetDelimiterDeterministic(token)):
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
	For every concept word (lemma) in the sequence, identify every feature neuron in that column that occurs q words before or after the concept word in the sequence, including the prime concept neuron. This function has been parallelized using PyTorch array operations.
	"""

	noDelimiterDetectedBetweenConceptTokens = False

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
		useSequenceReferenceSetDelimiterLists = False
		if(tokeniserSubword):
			if(useDedicatedFeatureListsSubword):
				useSequenceReferenceSetDelimiterLists = True
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
						if(useSequenceReferenceSetDelimiterLists):
							if(detectReferenceSetDelimitersBetweenNouns):
								if(tokenIndex < len(databaseNetworkObject.sequenceReferenceSetDelimiterDeterministicList) and databaseNetworkObject.sequenceReferenceSetDelimiterDeterministicList[tokenIndex]):
									rightmostDeterministic = tokenIndex
								elif(tokenIndex < len(databaseNetworkObject.sequenceReferenceSetDelimiterProbabilisticList) and databaseNetworkObject.sequenceReferenceSetDelimiterProbabilisticList[tokenIndex]):
									rightmostIndeterministic = tokenIndex
							else:
								if(tokenIndex < len(databaseNetworkObject.sequenceReferenceSetDelimiterList) and databaseNetworkObject.sequenceReferenceSetDelimiterList[tokenIndex]):
									rightmostDeterministic = tokenIndex
						else:
							token = tokens[tokenIndex]
							featureIndex = getTokenFeatureIndexIfKnown(databaseNetworkObject, token)
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
						noDelimiterDetectedBetweenConceptTokens = True
						if(debugTerminateOnConceptColumnsDelimitByPOSwarning):
							print("warning: no delimiter detected between concept tokens: concept #1: Position = ", conceptPosition, ", Index = ", leftIndex, ", Name = ", getTokenDisplayText(tokens[leftIndex]), ". concept #2: Position = ", conceptPosition+1, ", Index = ", rightIndex, ", Name = ", getTokenDisplayText(tokens[rightIndex]), ".")						
						if(debugTerminateOnConceptColumnsDelimitByPOSerror):
							exitWithError()
						#sequenceObservedColumns.noDelimiterDetectedBetweenConceptTokens = True
						#return None
						delimiterIndices.append(leftIndex)	#or rightIndex	#append dummy so that sequence can still be printed
				startIndices = pt.zeros_like(conceptIndicesSorted)
				endIndices = pt.full_like(conceptIndicesSorted, sequenceLength)
				for conceptPosition, delimiterIndex in enumerate(delimiterIndices):
					endIndices[conceptPosition] = delimiterIndex + 1
					startIndices[conceptPosition + 1] = delimiterIndex + 1
			conceptIndices = conceptIndicesSorted
	else:
		printe("conceptColumnsDelimitByPOS is required")
	
	sentenceWithConceptAssignment = buildSequenceConceptAssignment(sequenceObservedColumns, sequence, tokens, conceptIndices, startIndices, endIndices)
	if(printSequenceConceptAssignment):
		sequenceObservedColumns.sentenceWithConceptAssignment = sentenceWithConceptAssignment
		#print(f"Processing sequenceCount: {sequenceIndex}, {sequenceObservedColumns.sentenceWithConceptAssignment}")

	if(noDelimiterDetectedBetweenConceptTokens):
		sequenceObservedColumns.noDelimiterDetectedBetweenConceptTokens = True
		return None
	
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
	
	if(printSequenceConceptAssignmentByLine):
		sentenceWithConceptAssignment = ""
		currentColumnIndex = tokenConceptColumnIndexList[tokenIndex]
		for tokenIndex, token in enumerate(sequence):
			sentenceWithConceptAssignmentToken = ""
			if(tokenConceptColumnIndexList[tokenIndex] != currentColumnIndex):
				#newColumnDetected
				sentenceWithConceptAssignmentToken += "\nConcept Column " + conceptColumnsList[tokenConceptColumnIndexList[tokenIndex]] + ": "
				currentColumnIndex = tokenConceptColumnIndexList[tokenIndex]
			sentenceWithConceptAssignmentToken += f"{getTokenDisplayText(token)} ({tokenIndex}:{conceptColumnsList[tokenConceptColumnIndexList[tokenIndex]]}) "
			sentenceWithConceptAssignment += sentenceWithConceptAssignmentToken
	else:
		sentenceWithConceptAssignment = " ".join(
			f"{getTokenDisplayText(token)} ({tokenIndex}:{conceptColumnsList[tokenConceptColumnIndexList[tokenIndex]]})"	# if tokenConceptColumnIndexList[tokenIndex] is not None else 'none'
			for tokenIndex, token in enumerate(sequence)
		)

	return sentenceWithConceptAssignment

def getTokenDisplayText(token):
	if hasattr(token, "text"):
		return token.text
	if hasattr(token, "word"):
		return token.word
	return str(token)

def buildDeterministicBranchOrder(conceptIndexKey, featureIndex):
	seedValue = ((conceptIndexKey + 1) * 2654435761) ^ ((featureIndex + 1) * 1013904223)
	seedValue = seedValue & 0xFFFFFFFF
	rng = random.Random(seedValue)
	branchOrder = list(range(multipleDendriticBranchesNumber))
	rng.shuffle(branchOrder)
	return branchOrder

def selectFeatureBranchIndex(featureBranchCounts, branchAssignments, conceptIndexKey, featureIndex):
	branchIndex = featureBranchCounts.get(featureIndex, 0)
	featureBranchCounts[featureIndex] = branchIndex + 1
	if(multipleDendriticBranchesRandom):
		if(branchAssignments is None):
			raise RuntimeError("selectFeatureBranchIndex error: branchAssignments is None while multipleDendriticBranchesRandom enabled")
		featureBranchOrders = branchAssignments.get(conceptIndexKey)
		if(featureBranchOrders is None):
			featureBranchOrders = {}
			branchAssignments[conceptIndexKey] = featureBranchOrders
		branchOrder = featureBranchOrders.get(featureIndex)
		if(branchOrder is None):
			branchOrder = buildDeterministicBranchOrder(conceptIndexKey, featureIndex)
			featureBranchOrders[featureIndex] = branchOrder
		if(branchIndex < multipleDendriticBranchesNumber):
			branchIndex = branchOrder[branchIndex]
		else:
			branchIndex = multipleDendriticBranchesNumber - 1
	if(branchIndex >= multipleDendriticBranchesNumber):
		branchIndex = multipleDendriticBranchesNumber - 1
	return branchIndex
	
def processFeatures(sequenceObservedColumns, sequenceIndex, sequence, tokens, conceptIndices, startIndices, endIndices):
	numberConceptsInSequence = conceptIndices.shape[0]
	
	cs = sequenceObservedColumns.cs
	fs = sequenceObservedColumns.fs
	featureNeuronsActive = pt.zeros((multipleDendriticBranchesNumber, arrayNumberOfSegments, cs, fs), dtype=arrayType)
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
	branchAssignments = None
	if(multipleDendriticBranches):
		branchCounters = {}
		if(multipleDendriticBranchesRandom):
			branchAssignments = {}
	
	conceptIndicesList = conceptIndices.tolist()
	for i, sequenceConceptWordIndex in enumerate(conceptIndicesList):
		if(trainSequenceObservedColumnsMatchSequenceWords):
			sequenceConceptIndex = i
		else:
			if(useDedicatedConceptsLists):
				conceptLemma = getTokenConceptName(sequenceObservedColumns.databaseNetworkObject, tokens[sequenceConceptWordIndex])
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
				observedColumn = sequenceObservedColumns.observedColumnsSequenceWordIndexDict.get(sequenceConceptWordIndex)
				if(observedColumn is None):
					raise RuntimeError("processFeatures error: missing observedColumn for sequence concept word index")
				conceptIndexKey = observedColumn.conceptIndex
				featureBranchCounts = branchCounters.get(conceptIndexKey)
				if(featureBranchCounts is None):
					featureBranchCounts = {}
					branchCounters[conceptIndexKey] = featureBranchCounts
				startIndexValue = int(startIndices[sequenceConceptIndex].item())
				endIndexValue = int(endIndices[sequenceConceptIndex].item())
				featureIndicesInObservedTensor = sequenceObservedColumns.featureIndicesInObservedTensor
				for j in range(startIndexValue, endIndexValue):
					if(j >= featureIndicesInObservedTensor.shape[0]):
						continue
					globalFeatureIndex = int(featureIndicesInObservedTensor[j].item())
					branchIndex = selectFeatureBranchIndex(featureBranchCounts, branchAssignments, conceptIndexKey, globalFeatureIndex)
					if(multipleDendriticBranchesBinaryTree):
						if(useTrainDuringInference):
							if(multipleDendriticBranchesBinaryTreeDepthSelectMostActivatedRootBranches):
								branchIndex = selectFeatureBinaryTreeBranchIndexFromInference(sequenceObservedColumns.databaseNetworkObject, conceptIndexKey, globalFeatureIndex, branchIndex)
						if(trainVerifyConnectionNonexistentAcrossBranches):
							if(multipleDendriticBranchesBinaryTreeDepthSelectMostConnectedRootBranches):
								branchIndex = selectFeatureBinaryTreeBranchIndexFromConnections(sequenceObservedColumns, conceptIndexKey, globalFeatureIndex, branchIndex)
					if(useSANI):
						if(multipleDendriticBranchesBinaryTree):
							binaryTreeBranchIndices = calculateFeatureBinaryTreeBranchIndices(branchIndex, activeSequentialSegments)
							featureNeuronsActive[binaryTreeBranchIndices, activeSequentialSegments, sequenceConceptIndex, j] = 1
						else:
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
						sequenceFeatureIndex = sequenceObservedColumns.featureWordToIndex[variablePrimeConceptFeatureNeuronName]
					else:
						if(tokeniserSubword and useDedicatedFeatureListsSubword):
							sequenceFeatureIndex = getTokenFeatureIndex(sequenceObservedColumns.databaseNetworkObject, tokens[j])
						else:
							sequenceFeatureIndex = sequenceObservedColumns.featureWordToIndex[featureLemma]
					branchIndex = 0
					if(multipleDendriticBranches):
						observedColumn = sequenceObservedColumns.observedColumnsSequenceWordIndexDict.get(sequenceConceptWordIndex)
						if(observedColumn is None):
							raise RuntimeError("processFeatures error: missing observedColumn for sequence concept word index")
						conceptIndexKey = observedColumn.conceptIndex
						featureBranchCounts = branchCounters.get(conceptIndexKey)
						if(featureBranchCounts is None):
							featureBranchCounts = {}
							branchCounters[conceptIndexKey] = featureBranchCounts
						branchIndex = selectFeatureBranchIndex(featureBranchCounts, branchAssignments, conceptIndexKey, sequenceFeatureIndex)
						if(multipleDendriticBranchesBinaryTree):
							if(useTrainDuringInference):
								if(multipleDendriticBranchesBinaryTreeDepthSelectMostActivatedRootBranches):
									branchIndex = selectFeatureBinaryTreeBranchIndexFromInference(sequenceObservedColumns.databaseNetworkObject, conceptIndexKey, sequenceFeatureIndex, branchIndex)
							if(trainVerifyConnectionNonexistentAcrossBranches):
								if(multipleDendriticBranchesBinaryTreeDepthSelectMostConnectedRootBranches):
									branchIndex = selectFeatureBinaryTreeBranchIndexFromConnections(sequenceObservedColumns, conceptIndexKey, sequenceFeatureIndex, branchIndex)
					if(useSANI):
						if(multipleDendriticBranchesBinaryTree):
							binaryTreeBranchIndices = calculateFeatureBinaryTreeBranchIndices(branchIndex, activeSequentialSegments)
							featureNeuronsActive[binaryTreeBranchIndices, activeSequentialSegments, sequenceConceptIndex, sequenceFeatureIndex] = 1
						else:
							featureNeuronsActive[branchIndex, activeSequentialSegments, sequenceConceptIndex, sequenceFeatureIndex] = 1
					else:
						featureNeuronsActive[branchIndex, arrayIndexSegmentFirst, sequenceConceptIndex, sequenceFeatureIndex] = 1
				else:
					sequenceFeatureIndex = getTokenFeatureIndexIfKnown(sequenceObservedColumns.databaseNetworkObject, tokens[j])
					if(sequenceFeatureIndex is None):
						continue
					branchIndex = 0
					if(multipleDendriticBranches):
						observedColumn = sequenceObservedColumns.observedColumnsSequenceWordIndexDict.get(sequenceConceptWordIndex)
						if(observedColumn is None):
							raise RuntimeError("processFeatures error: missing observedColumn for sequence concept word index")
						conceptIndexKey = observedColumn.conceptIndex
						featureBranchCounts = branchCounters.get(conceptIndexKey)
						if(featureBranchCounts is None):
							featureBranchCounts = {}
							branchCounters[conceptIndexKey] = featureBranchCounts
						branchIndex = selectFeatureBranchIndex(featureBranchCounts, branchAssignments, conceptIndexKey, sequenceFeatureIndex)
						if(multipleDendriticBranchesBinaryTree):
							if(useTrainDuringInference):
								if(multipleDendriticBranchesBinaryTreeDepthSelectMostActivatedRootBranches):
									branchIndex = selectFeatureBinaryTreeBranchIndexFromInference(sequenceObservedColumns.databaseNetworkObject, conceptIndexKey, sequenceFeatureIndex, branchIndex)
							if(trainVerifyConnectionNonexistentAcrossBranches):
								if(multipleDendriticBranchesBinaryTreeDepthSelectMostConnectedRootBranches):
									branchIndex = selectFeatureBinaryTreeBranchIndexFromConnections(sequenceObservedColumns, conceptIndexKey, sequenceFeatureIndex, branchIndex)
					if(useSANI):
						if(multipleDendriticBranchesBinaryTree):
							binaryTreeBranchIndices = calculateFeatureBinaryTreeBranchIndices(branchIndex, activeSequentialSegments)
							featureNeuronsActive[binaryTreeBranchIndices, activeSequentialSegments, sequenceConceptIndex, sequenceFeatureIndex] = 1
						else:
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

def calculateFeatureBinaryTreeBranchIndices(branchIndex, segmentIndices):
	result = None
	if(multipleDendriticBranchesBinaryTree):
		if(not multipleDendriticBranchesRandom):
			raise RuntimeError("calculateFeatureBinaryTreeBranchIndices error: multipleDendriticBranchesRandom is required")
		if(not isinstance(branchIndex, int) or isinstance(branchIndex, bool)):
			raise RuntimeError("calculateFeatureBinaryTreeBranchIndices error: branchIndex must be an int")
		if(branchIndex < arrayIndexSegmentFirst or branchIndex >= multipleDendriticBranchesNumber):
			raise RuntimeError("calculateFeatureBinaryTreeBranchIndices error: branchIndex out of range")
		if(not pt.is_tensor(segmentIndices) or segmentIndices.dim() != 1):
			raise RuntimeError("calculateFeatureBinaryTreeBranchIndices error: segmentIndices must be a rank 1 tensor")
		if(segmentIndices.numel() > 0 and (bool(pt.any(segmentIndices < arrayIndexSegmentFirst).item()) or bool(pt.any(segmentIndices >= arrayNumberOfSegments).item()))):
			raise RuntimeError("calculateFeatureBinaryTreeBranchIndices error: segmentIndices out of range")
		if(multipleDendriticBranchesBinaryTreeDepth != arrayNumberOfSegments):
			raise RuntimeError("calculateFeatureBinaryTreeBranchIndices error: binary tree depth must equal arrayNumberOfSegments")
		branchIndices = pt.full_like(segmentIndices, branchIndex)
		branchingFactors = pt.full_like(segmentIndices, multipleDendriticBranchesBinaryTreeBranchingFactor)
		branchDivisors = pt.pow(branchingFactors, segmentIndices)
		result = pt.div(branchIndices, branchDivisors, rounding_mode="floor")
	else:
		raise RuntimeError("calculateFeatureBinaryTreeBranchIndices error: requires multipleDendriticBranchesBinaryTree")
	return result

def selectFeatureBinaryTreeBranchIndexFromInference(databaseNetworkObject, conceptIndexKey, featureIndex, defaultBranchIndex):
	result = defaultBranchIndex
	if(multipleDendriticBranchesBinaryTree):
		if(not useTrainDuringInference or not multipleDendriticBranchesBinaryTreeDepthSelectMostActivatedRootBranches):
			raise RuntimeError("selectFeatureBinaryTreeBranchIndexFromInference error: requires useTrainDuringInference root-branch selection")
		if(databaseNetworkObject is None):
			raise RuntimeError("selectFeatureBinaryTreeBranchIndexFromInference error: databaseNetworkObject is None")
		if(not isinstance(conceptIndexKey, int) or isinstance(conceptIndexKey, bool) or not isinstance(featureIndex, int) or isinstance(featureIndex, bool)):
			raise RuntimeError("selectFeatureBinaryTreeBranchIndexFromInference error: conceptIndexKey and featureIndex must be ints")
		if(not isinstance(defaultBranchIndex, int) or isinstance(defaultBranchIndex, bool) or defaultBranchIndex < arrayIndexSegmentFirst or defaultBranchIndex >= multipleDendriticBranchesNumber):
			raise RuntimeError("selectFeatureBinaryTreeBranchIndexFromInference error: defaultBranchIndex out of range")
		if(multipleDendriticBranchesBinaryTreeDepth != arrayNumberOfSegments):
			raise RuntimeError("selectFeatureBinaryTreeBranchIndexFromInference error: binary tree depth must equal arrayNumberOfSegments")
		inferenceActivation = getattr(databaseNetworkObject, "multipleDendriticBranchesBinaryTreeInferenceActivation", None)
		if(inferenceActivation is None):
			raise RuntimeError("selectFeatureBinaryTreeBranchIndexFromInference error: inference activation is unavailable")
		if(inferenceActivation.dim() != 4 or inferenceActivation.shape[0] != multipleDendriticBranchesNumber or inferenceActivation.shape[1] != arrayNumberOfSegments):
			raise RuntimeError("selectFeatureBinaryTreeBranchIndexFromInference error: inference activation shape is invalid")
		if(conceptIndexKey < arrayIndexSegmentFirst or conceptIndexKey >= inferenceActivation.shape[2] or featureIndex < arrayIndexSegmentFirst or featureIndex >= inferenceActivation.shape[3]):
			raise RuntimeError("selectFeatureBinaryTreeBranchIndexFromInference error: target neuron index out of range")
		if(inferenceActivation.is_sparse):
			inferenceActivationSparse = inferenceActivation.coalesce()
			inferenceActivationIndices = inferenceActivationSparse.indices()
			inferenceActivationValues = inferenceActivationSparse.values()
			inferenceActivationMask = (inferenceActivationIndices[2] == conceptIndexKey) & (inferenceActivationIndices[3] == featureIndex)
			branchActivation = pt.zeros((multipleDendriticBranchesNumber, arrayNumberOfSegments), dtype=inferenceActivationValues.dtype, device=inferenceActivationValues.device)
			if(inferenceActivationMask.any()):
				branchActivation.index_put_((inferenceActivationIndices[0, inferenceActivationMask], inferenceActivationIndices[1, inferenceActivationMask]), inferenceActivationValues[inferenceActivationMask], accumulate=True)
		else:
			branchActivation = inferenceActivation[:, :, conceptIndexKey, featureIndex]
		rootBranchIndices = pt.arange(multipleDendriticBranchesNumber, dtype=pt.long, device=branchActivation.device).unsqueeze(1)
		segmentIndices = pt.arange(arrayNumberOfSegments, dtype=pt.long, device=branchActivation.device).unsqueeze(0)
		branchingFactors = pt.full_like(segmentIndices, multipleDendriticBranchesBinaryTreeBranchingFactor)
		branchDivisors = pt.pow(branchingFactors, segmentIndices)
		pathBranchIndices = pt.div(rootBranchIndices, branchDivisors, rounding_mode="floor")
		pathActivations = branchActivation[pathBranchIndices, segmentIndices]
		pathActivationMask = (pathActivations > 0).to(pt.long)
		contiguouslyActiveFinalPath = pt.flip(pt.cumprod(pt.flip(pathActivationMask, dims=(1,)), dim=1), dims=(1,))
		numberOfContiguouslyActiveFinalSegments = contiguouslyActiveFinalPath.sum(dim=1)
		maximumNumberOfContiguouslyActiveFinalSegments = int(numberOfContiguouslyActiveFinalSegments.max().item())
		if(maximumNumberOfContiguouslyActiveFinalSegments > arrayIndexSegmentFirst):
			firstContiguouslyActiveFinalSegmentIndex = arrayNumberOfSegments - maximumNumberOfContiguouslyActiveFinalSegments
			mostCompleteRootBranches = pt.nonzero(numberOfContiguouslyActiveFinalSegments == maximumNumberOfContiguouslyActiveFinalSegments, as_tuple=False).view(-1)
			firstContiguouslyActiveFinalSegmentBranchDivisor = multipleDendriticBranchesBinaryTreeBranchingFactor**firstContiguouslyActiveFinalSegmentIndex
			mostCompleteBranchIndices = pt.div(mostCompleteRootBranches, firstContiguouslyActiveFinalSegmentBranchDivisor, rounding_mode="floor")
			mostCompleteBranchCounts = pt.bincount(mostCompleteBranchIndices, minlength=multipleDendriticBranchesNumber)
			maximumBranchCount = mostCompleteBranchCounts.max()
			mostCompleteBranchCandidates = pt.nonzero(mostCompleteBranchCounts == maximumBranchCount, as_tuple=False).view(-1)
			selectedCandidateIndex = pt.randint(mostCompleteBranchCandidates.numel(), (1,), device=branchActivation.device)
			selectedBranchIndex = int(mostCompleteBranchCandidates[selectedCandidateIndex].item())
			rootBranchesPerSelectedBranch = multipleDendriticBranchesNumber//(multipleDendriticBranchesBinaryTreeBranchingFactor**(maximumNumberOfContiguouslyActiveFinalSegments-1))
			selectedRootBranchStart = rootBranchesPerSelectedBranch*selectedBranchIndex
			selectedRootBranchEnd = selectedRootBranchStart + rootBranchesPerSelectedBranch
			result = int(pt.randint(selectedRootBranchStart, selectedRootBranchEnd, (1,), device=branchActivation.device).item())
	else:
		raise RuntimeError("selectFeatureBinaryTreeBranchIndexFromInference error: requires multipleDendriticBranchesBinaryTree")
	return result

def selectFeatureBinaryTreeBranchIndexFromConnections(sequenceObservedColumns, conceptIndexKey, featureIndex, defaultBranchIndex):
	result = defaultBranchIndex
	if(trainVerifyConnectionNonexistentAcrossBranches):
		if(not multipleDendriticBranchesBinaryTree or not multipleDendriticBranchesBinaryTreeDepthSelectMostConnectedRootBranches):
			raise RuntimeError("selectFeatureBinaryTreeBranchIndexFromConnections error: requires binary-tree most-connected-root-branch selection")
		if(sequenceObservedColumns is None):
			raise RuntimeError("selectFeatureBinaryTreeBranchIndexFromConnections error: sequenceObservedColumns is None")
		if(not isinstance(conceptIndexKey, int) or isinstance(conceptIndexKey, bool) or not isinstance(featureIndex, int) or isinstance(featureIndex, bool)):
			raise RuntimeError("selectFeatureBinaryTreeBranchIndexFromConnections error: conceptIndexKey and featureIndex must be ints")
		if(not isinstance(defaultBranchIndex, int) or isinstance(defaultBranchIndex, bool) or defaultBranchIndex < arrayIndexSegmentFirst or defaultBranchIndex >= multipleDendriticBranchesNumber):
			raise RuntimeError("selectFeatureBinaryTreeBranchIndexFromConnections error: defaultBranchIndex out of range")
		if(conceptIndexKey < arrayIndexSegmentFirst or conceptIndexKey >= sequenceObservedColumns.databaseNetworkObject.c or featureIndex < arrayIndexSegmentFirst or featureIndex >= sequenceObservedColumns.databaseNetworkObject.f):
			raise RuntimeError("selectFeatureBinaryTreeBranchIndexFromConnections error: target neuron index out of range")
		targetCombinedKeys = getattr(sequenceObservedColumns, "trainVerifyConnectionNonexistentAcrossBranchesTargetCombinedKeys", None)
		selectedRootBranches = getattr(sequenceObservedColumns, "trainVerifyConnectionNonexistentAcrossBranchesSelectedRootBranches", None)
		connectedTargetMask = getattr(sequenceObservedColumns, "trainVerifyConnectionNonexistentAcrossBranchesConnectedTargetMask", None)
		if(targetCombinedKeys is None or selectedRootBranches is None or connectedTargetMask is None):
			raise RuntimeError("selectFeatureBinaryTreeBranchIndexFromConnections error: preloaded sequence connection selection is unavailable")
		if(not pt.is_tensor(targetCombinedKeys) or not pt.is_tensor(selectedRootBranches) or not pt.is_tensor(connectedTargetMask)):
			raise RuntimeError("selectFeatureBinaryTreeBranchIndexFromConnections error: preloaded sequence connection selection tensors are invalid")
		if(targetCombinedKeys.dim() != 1 or selectedRootBranches.dim() != 1 or connectedTargetMask.dim() != 1 or targetCombinedKeys.shape[0] != selectedRootBranches.shape[0] or targetCombinedKeys.shape[0] != connectedTargetMask.shape[0]):
			raise RuntimeError("selectFeatureBinaryTreeBranchIndexFromConnections error: preloaded sequence connection selection tensor dimensions are invalid")
		targetCombinedKey = conceptIndexKey*sequenceObservedColumns.databaseNetworkObject.f + featureIndex
		targetCombinedKeyTensor = pt.tensor([targetCombinedKey], dtype=targetCombinedKeys.dtype, device=targetCombinedKeys.device)
		targetPosition = pt.searchsorted(targetCombinedKeys, targetCombinedKeyTensor)
		if(targetPosition.item() < targetCombinedKeys.numel() and targetCombinedKeys[targetPosition.item()].item() == targetCombinedKey and connectedTargetMask[targetPosition.item()].item()):
			result = int(selectedRootBranches[targetPosition.item()].item())
			if(result < arrayIndexSegmentFirst or result >= multipleDendriticBranchesNumber):
				raise RuntimeError("selectFeatureBinaryTreeBranchIndexFromConnections error: selected root branch index out of range")
	return result
