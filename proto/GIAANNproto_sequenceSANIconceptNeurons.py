"""GIAANNproto_sequenceSANIconceptNeurons.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_main.py

# Usage:
see GIAANNproto_main.py

# Description:
GIA ANN proto sequence SANI concept neurons

"""

import torch as pt

from GIAANNproto_globalDefs import *
import GIAANNproto_databaseNetworkTrainExcitation
import GIAANNproto_databaseNetworkFilesExcitation
#from collections import OrderedDict
from sortedcontainers import SortedDict	#keeps keys in a sorted order.

class ArtificialSpacyToken:	#required for SANIconceptNeurons
    def __init__(self, text, lemma, pos):
        self.text = text
        self.lemma_ = lemma
        self.pos_ = pos

SANIconceptNeuronsDict = {}  # key: lemma, value: index
SANIconceptNeuronWeightsList = []  # list of concept neuron weights (float)
#SANIconceptNeuronsList = []  # list of concept neuron names (lemmas)
#sc = 0  # current number of SANI concept neurons
	
def initialiseSANIconceptNeurons():
	# Initialize the concept columns dictionary
	if(GIAANNproto_databaseNetworkFilesExcitation.pathExists(SANIconceptNeuronsDictFile)):
		SANIconceptNeuronsDict = GIAANNproto_databaseNetworkFilesExcitation.loadDictFile(SANIconceptNeuronsDictFile)	#must be ordered dictionary
		SANIconceptNeuronsWeightList = GIAANNproto_databaseNetworkFilesExcitation.loadListFile(SANIconceptNeuronsDictFile)
		#sc = len(conceptColumnsDict)
		#SANIconceptNeuronsList = list(conceptColumnsDict.keys())
		
def finaliseSANIconceptNeurons():
	GIAANNproto_databaseNetworkFilesExcitation.saveDictFile(SANIconceptNeuronsDictFile, SANIconceptNeuronsDict)
	GIAANNproto_databaseNetworkFilesExcitation.saveListFile(SANIconceptNeuronWeightsListFile, SANIconceptNeuronWeightsList)


def generateSANIsequenceList(inputSequence, conceptMask, nlp):
	sequenceList = []
	
	SANIsequence = []
	currentSubsequence = []
	if(not SANIconceptNeuronsAllocateConceptFeatureWordNeuron):
		firstConceptFound = False
		previousSubsequence = []
		previousConceptToken = None
		
	if(SANIconceptNeuronsAllocateConceptFeatureWordNeuron):
		#each SANI concept neuron is similar to a phrasal verb
		for tokenIndex, token in enumerate(inputSequence):
			if(conceptMask[tokenIndex]):
				subsequenceConceptMask = [False] * len(currentSubsequence)
				SANItokens = createArtificialConceptNeuronTokens(currentSubsequence, subsequenceConceptMask, None)
				SANIsequence += SANItokens
				SANIsequence.append(token)	#append noun (concept feature) token
				currentSubsequence = []	#reset current subsequence
			else:
				currentSubsequence.append(token)
		subsequenceConceptMask = [False] * len(currentSubsequence)
		SANItokens = createArtificialConceptNeuronTokens(currentSubsequence, subsequenceConceptMask, None)	#generate last token for sequence
		SANIsequence += SANItokens
	else:
		for tokenIndex, token in enumerate(inputSequence):
			if(conceptMask[tokenIndex]):
				if(firstConceptFound):
					subsequence = previousSubsequence+currentSubsequence
					subsequenceConceptMask = [False] * len(subsequence)
					previousConceptTokenIndex = len(previousSubsequence)-1
					subsequenceConceptMask[previousConceptTokenIndex] = True
					conceptName = previousConceptToken.lemma_.lower()
					SANItokens = createArtificialConceptNeuronTokens(subsequence, subsequenceConceptMask, conceptName)
					SANIsequence += SANItokens
					previousSubsequence = currentSubsequence
					previousSubsequence.append(token)	#append concept token
					currentSubsequence = []
					previousConceptToken = token
				else:
					firstConceptFound = True
					previousSubsequence = currentSubsequence
					previousSubsequence.append(token)	#append concept token
					currentSubsequence = []
					previousConceptToken = token
			else:
				currentSubsequence.append(token)	
		subsequence = previousSubsequence+currentSubsequence
		subsequenceConceptMask = [False] * len(subsequence)
		previousConceptTokenIndex = len(previousSubsequence)-1
		subsequenceConceptMask[previousConceptTokenIndex] = True
		conceptName = previousConceptToken.lemma_.lower()
		SANItokens = createArtificialConceptNeuronTokens(subsequence, subsequenceConceptMask, conceptName)
		SANIsequence += SANItokens

	sequenceList.append(SANIsequence)
	
	if(SANIconceptNeuronsAllocateWordNeurons):	#also add the original word discretised sequence for database network redundancy
		sequenceList.append(inputSequence)
	
	return sequenceList


def createArtificialConceptNeuronTokens(subsequence, subsequenceConceptMask, conceptName):
	if(SANIconceptNeuronsAllocateForPartialSubsequences):
		SANItokens = createArtificialConceptNeuronTokensDynamic(subsequence, subsequenceConceptMask)
	else:
		SANItoken = createArtificialConceptNeuronToken(subsequence, subsequenceConceptMask, conceptName)
		SANItokens = [SANItoken]

		if(debugSANIconceptNeurons):
			for token in SANItokens:
				print("word = ", token.text)
				print("lemma = ", token.lemma_)
				print("pos = ", token.pos_)
	
	return SANItokens

def createArtificialConceptNeuronTokensDynamic(subsequence, subsequenceConceptMask):
	SANItokens = []
	
	#populate SANInodesBestCandidates; 
	subsequenceLen = len(subsequence)
	minTupleSize = min(subsequenceLen, SANIconceptNeuronsAllocateForPartialSubsequencesMinTupleSize)
	maxTupleSize = min(subsequenceLen, SANIconceptNeuronsAllocateForPartialSubsequencesMaxTupleSize)
	SANInodesBestCandidates = SortedDict()	#best tuple candidates	#multi SortedDict sorted by key in ascending order (highest/best sized tuple candidate is always last in dictionary)	#key: tupleSize, value:[(tupleText, tokenIndex, tupleSize)] a list of all candidates at that particular tuple size
	for tupleSize in range(minTupleSize, maxTupleSize+1):
		for tokenIndex, token in enumerate(subsequence):
			if(tokenIndex+tupleSize-1 < subsequenceLen):
				tupleText = createArtificialConceptNeuronTupleText(subsequence, subsequenceConceptMask, tokenIndex, tupleSize)
				if(tupleText in SANIconceptNeuronsDict):
					SANIconceptNeuronIndex = SANIconceptNeuronsDict[tupleText]
					SANIconceptNeuronWeight = SANIconceptNeuronWeightsList[SANIconceptNeuronIndex]
					if(SANIconceptNeuronWeight >= SANIconceptNeuronsAllocateForPartialSubsequencesMinWeight):	#only consider common tuples in corpus for assignment of SANIconceptNeurons to database network concept columns
						SANInodesBestCandidateTupleProperties = (tupleText, tokenIndex, tupleSize)
						addToMultiDict(SANInodesBestCandidates, tupleSize, SANInodesBestCandidateTupleProperties)
					SANIconceptNeuronWeightsList[SANIconceptNeuronIndex] += SANIconceptNeuronsAllocateForPartialSubsequencesWeightIncrement
				else:
					sc = len(SANIconceptNeuronsDict)+1	#calculate new dictionary size
					SANIconceptNeuronsDict[tupleText] = sc
					SANIconceptNeuronWeightsList.append(SANIconceptNeuronsAllocateForPartialSubsequencesWeightIncrement)
				
	#populate subsequenceTuplePropertiesList;
	subsequenceTuplePropertiesList = []
	subsequenceSANInodesFoundMask = [False]*subsequenceLen
	stillFindingCandidates = len(SANInodesBestCandidates) > 0	
	while stillFindingCandidates:
		candidateBestIndex = -1	#len(SANInodesBestCandidates)-1	#always select the best (largest sized) candidate tuple in SANInodesBestCandidates
		SANInodesBestCandidateScore, SANInodesBestCandidateTuplePropertiesList = SANInodesBestCandidates.peekitem(candidateBestIndex)
		SANInodesBestCandidateTupleProperties = SANInodesBestCandidateTuplePropertiesList[0]	#get first candidate in list (of same tuple size)
		(tupleText, tokenIndex, tupleSize) = SANInodesBestCandidateTupleProperties
		tupleConflictFound = False	#conflict with existing assigned SANIconceptNeuron tuple
		for i in range(tokenIndex, tokenIndex+tupleSize):
			if(subsequenceSANInodesFoundMask[i]):
				tupleConflictFound = True
		if(not tupleConflictFound):
			print("not tupleConflictFound")
			subsequenceTuplePropertiesList.append(SANInodesBestCandidateTupleProperties)
			for i in range(tokenIndex, tokenIndex+tupleSize):
				subsequenceSANInodesFoundMask[i] = True
		removeKeyFromMultiDict(SANInodesBestCandidates, SANInodesBestCandidateTuplePropertiesList, candidateBestIndex) #remove last tuple from SANInodesBestCandidates
		stillFindingCandidates = updateStillFindingCandidates(SANInodesBestCandidates, subsequenceSANInodesFoundMask, subsequenceLen)

	#populate SANItokens (SANI concept neuron tokens);
	currentTokenIndex = 0
	while currentTokenIndex < subsequenceLen:
		if(subsequenceSANInodesFoundMask[currentTokenIndex]):
			#find the SANI concept token in subsequenceTuplePropertiesList
			for subsequenceTupleProperties in subsequenceTuplePropertiesList:
				(tupleText, tokenIndex, tupleSize) = subsequenceTupleProperties
				if(tokenIndex == currentTokenIndex):
					tupleSubsequence = subsequence[tokenIndex:tokenIndex+tupleSize]
					tupleSubsequenceConceptMask = subsequenceConceptMask[tokenIndex:tokenIndex+tupleSize]
					SANItoken = createArtificialConceptNeuronToken(tupleSubsequence, tupleSubsequenceConceptMask, None)
					SANItokens.append(SANItoken)
					currentTokenIndex += tupleSize
		else:
			#add a single word token to list (no suitable SANI concept neuron found);
			token = subsequence[currentTokenIndex]
			SANItokens.append(token)
			currentTokenIndex += 1
	
	if(debugSANIconceptNeurons):
		#print SANItokens;
		for SANItoken in SANItokens:
			print("SANItoken assigned: SANItoken.text = ", SANItoken.text)
		
	return SANItokens
		
def updateStillFindingCandidates(SANInodesBestCandidates, subsequenceSANInodesFoundMask, subsequenceLen):
	stillFindingCandidates = True
	
	sequenceHasMissingTuples = False
	for i in range(subsequenceLen):
		if(subsequenceSANInodesFoundMask[i] == False):
			sequenceHasMissingTuples = True
	if(sequenceHasMissingTuples):
		stillFindingCandidates = False
		
	if(len(SANInodesBestCandidates) == 0):
		stillFindingCandidates = False
		
	return stillFindingCandidates
							
def addToMultiDict(dictionary, key, value):
	if(key in dictionary):
		dictionary[key].append(value)
	else:
		dictionary[key] = [value] 

def removeKeyFromMultiDict(dictionary, firstValueList, dictIndexToRemove):
	firstValueList.pop(0)
	if(len(firstValueList) == 0):
		dictionary.popitem(dictIndexToRemove)	#remove last element

def createArtificialConceptNeuronToken(subsequence, subsequenceConceptMask, conceptName):
	tupleText = ""
	lemma = ""
	for tokenIndex, token in enumerate(subsequence):
		tupleText += getTupleTextElement(token, tokenIndex, subsequenceConceptMask)
		if(SANIconceptNeuronsAllocateConceptFeatureWordNeuron):
			lemma += token.lemma_.lower()

	if(SANIconceptNeuronsAllocateConceptFeatureWordNeuron):
		pos = "VERB"	#any non-NOUN spacy tag
	else:
		pos = "NOUN"	#SANI tokens will be interpreted as concept feature neurons
		lemma = conceptName

	if(not SANIconceptNeuronsAllocateConceptFeatureWordNeuron):
		printe("createArtificialConceptNeuronToken:!SANIconceptNeuronsAllocateConceptFeatureWordNeuron not yet coded; need to update entire codebase to ensure only token.lemma or token.pos=NOUN is used to detect concept features and only token.word is used to generate a feature neuron name")
	
	SANItoken = ArtificialSpacyToken(tupleText, lemma, pos)	#TODO: verify no additional spacy tag properties are required to be generated
	return SANItoken

def createArtificialConceptNeuronTupleText(subsequence, subsequenceConceptMask, startIndex, length):
	tupleText = ""
	for tokenIndex in range(startIndex, startIndex+length):
		token = subsequence[tokenIndex]
		tupleText += getTupleTextElement(token, tokenIndex, subsequenceConceptMask)
	return tupleText

def getTupleTextElement(token, tokenIndex, subsequenceConceptMask):
	if(subsequenceConceptMask and subsequenceConceptMask[tokenIndex]):
		tupleTextElement = variableConceptNeuronFeatureNameAbbreviation
	else:
		tupleTextElement = token.text.lower()
	return tupleTextElement
			
