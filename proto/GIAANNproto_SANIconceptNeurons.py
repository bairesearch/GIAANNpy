"""GIAANNproto_SANIconceptNeurons.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_main.py

# Usage:
see GIAANNproto_main.py

# Description:
GIA ANN proto SANI concept neurons

"""

import torch as pt

from GIAANNproto_globalDefs import *
import GIAANNproto_databaseNetworkTrain

class ArtificialSpacyToken:	#required for SANIconceptNeurons
    def __init__(self, text, lemma, pos):
        self.text = text
        self.lemma_ = lemma
        self.pos_ = pos
			
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
				SANItoken = createArtificialConceptNeuronToken(currentSubsequence, None, None, nlp)
				SANIsequence.append(SANItoken)
				SANIsequence.append(token)	#append noun (concept feature) token
				currentSubsequence = []	#reset current subsequence
			else:
				currentSubsequence.append(token)
		SANItoken = createArtificialConceptNeuronToken(currentSubsequence, None, None, nlp)	#generate last token for sequence
		SANIsequence.append(SANItoken)
	else:
		for tokenIndex, token in enumerate(inputSequence):
			if(conceptMask[tokenIndex]):
				if(firstConceptFound):
					subsequence = previousSubsequence+currentSubsequence
					subsequenceConceptMask = [False] * len(subsequence)
					previousConceptTokenIndex = len(previousSubsequence)-1
					subsequenceConceptMask[previousConceptTokenIndex] = True
					conceptName = previousConceptToken.lemma_.lower()
					SANItoken = createArtificialConceptNeuronToken(subsequence, conceptName, subsequenceConceptMask, nlp)
					SANIsequence.append(SANItoken)
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
		SANItoken = createArtificialConceptNeuronToken(subsequence, conceptName, subsequenceConceptMask, nlp)

	sequenceList.append(SANIsequence)
	
	if(debugSANIconceptNeurons):
		for token in SANIsequence:
			print("word = ", token.text)
			print("lemma = ", token.lemma_)
			print("pos = ", token.pos_)
	
	if(SANIconceptNeuronsAllocateWordNeurons):	#also add the original word discretised sequence for database network redundancy
		sequenceList.append(inputSequence)
	
	return sequenceList


def createArtificialConceptNeuronToken(subsequence, conceptName, subsequenceConceptMask, nlp):
	word = ""
	lemma = ""
	for tokenIndex, token in enumerate(subsequence):
		if(subsequenceConceptMask and subsequenceConceptMask[tokenIndex]):
			word += variableConceptNeuronFeatureNameAbbreviation
		else:
			word += token.text.lower()
		if(SANIconceptNeuronsAllocateConceptFeatureWordNeuron):
			lemma += token.lemma_.lower()

	if(SANIconceptNeuronsAllocateConceptFeatureWordNeuron):
		pos = "VERB"	#any non-NOUN spacy tag
	else:
		pos = "NOUN"	#SANI tokens will be interpreted as concept feature neurons
		lemma = conceptName

	if(not SANIconceptNeuronsAllocateConceptFeatureWordNeuron):
		printe("createArtificialConceptNeuronToken:!SANIconceptNeuronsAllocateConceptFeatureWordNeuron not yet coded; need to update entire codebase to ensure only token.lemma or token.pos=NOUN is used to detect concept features and only token.word is used to generate a feature neuron name")
	
	SANItoken = ArtificialSpacyToken(word, lemma, pos)	#TODO: verify no additional spacy tag properties are required to be generated
	return SANItoken
	
