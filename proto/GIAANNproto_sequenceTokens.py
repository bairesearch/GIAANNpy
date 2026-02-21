"""GIAANNproto_sequenceTokens.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_main.py

# Usage:
see GIAANNproto_main.py

# Description:
GIA ANN proto sequence Tokens

"""

import torch as pt

from GIAANNproto_globalDefs import *

if(usePOS):
	import GIAANNproto_sequencePOS

def loadPOSdatabase():
	GIAANNproto_sequencePOS.loadPOSdatabase()

def isTokenReferenceSetDelimiterDeterministic(token):
	result = False
	if(token.word in conceptColumnsDelimiterWordTypes or token.tag in conceptColumnsDelimiterTagTypes):
		result = True
	elif(GIAANNproto_sequencePOS.isWordEverInPOStypeList(token.word, conceptColumnsDelimiterPOStypes)):
		result = True
	elif(token.lemma is not None and GIAANNproto_sequencePOS.isWordEverInPOStypeList(token.lemma, conceptColumnsDelimiterPOStypes)):
		result = True
	return result
	
def isTokenReferenceSetDelimiterProbabilistic(token):
	result = False
	if(token.word in detectReferenceSetDelimitersBetweenNounsWordTypes or token.tag in detectReferenceSetDelimitersBetweenNounsTagTypes):
		result = True
	elif(GIAANNproto_sequencePOS.isWordEverInPOStypeList(token.word, detectReferenceSetDelimitersBetweenNounsPOStypes)):
		result = True
	elif(token.lemma is not None and GIAANNproto_sequencePOS.isWordEverInPOStypeList(token.lemma, detectReferenceSetDelimitersBetweenNounsPOStypes)):
		result = True
	return result
	

class SequenceToken:
	def __init__(self, word, lemma, pos, tag):
		self.word = word
		self.lemma = lemma
		self.pos = pos
		self.tag = tag

# Preprocessing helpers
class PreprocessedToken:
	__slots__ = ("text", "lemma_", "pos_", "tag_")
	def __init__(self, text, lemma, pos, tag):
		self.text = text
		self.lemma_ = lemma
		self.pos_ = pos
		self.tag_ = tag

def convertPreprocessedTokenToSequenceToken(preprocessedToken):
	word = preprocessedToken.text.lower()
	lemma = preprocessedToken.lemma_.lower()
	pos = preprocessedToken.pos_  #coarse Part-of-speech (e.g. PRON) 
	tag = preprocessedToken.tag_	#fine-grained POS (e.g., PRP, PRP$, WP, WP$, etc.)
	token = SequenceToken(word, lemma, pos, tag)
	return token

def getTokens(sequence):
	tokens = []
	for preprocessedToken in sequence:
		token = convertPreprocessedTokenToSequenceToken(preprocessedToken)
		tokens.append(token)
	return tokens

class PreprocessedSequence:
	__slots__ = ("tokens",)

	def __init__(self, tokens):
		self.tokens = tokens

	def __len__(self):
		return len(self.tokens)

	def __iter__(self):
		return iter(self.tokens)

	def __getitem__(self, item):
		if isinstance(item, slice):
			return PreprocessedSequence(self.tokens[item])
		return self.tokens[item]

	@property
	def text(self):
		return " ".join(token.text for token in self.tokens)


def preprocessSequence(sequence):
	return pretrain(sequence)

def pretrain(sequence):
	if(pretrainCombineHyphenatedNouns):
		sequence = pretrainCombineConsecutiveNounHyphenated(sequence)
	if(pretrainCombineConsecutiveNouns):
		sequence = pretrainCombineConsecutiveNoun(sequence)
	if(pretrainConceptColumnsDelimitByPOSenforce):
		sequence = pretrainConceptColumnsDelimitByPOSenforce(sequence)
	return sequence

if(pretrainConceptColumnsDelimitByPOSenforce):
	
	def pretrainConceptColumnsDelimitByPOSenforce(sequence):
		result = None
		sequenceLocal = sequence if isinstance(sequence, PreprocessedSequence) else PreprocessedSequence([PreprocessedToken(token.text, token.lemma_, token.pos_, token.tag_) for token in sequence])
		if(not usePOS):
			printe("pretrainConceptColumnsDelimitByPOSenforce requires usePOS")
		if(not useSpacyForConceptNounPOSdetection):
			printe("pretrainConceptColumnsDelimitByPOSenforce requires useSpacyForConceptNounPOSdetection")
		conceptIndices = [tokenIndex for tokenIndex, token in enumerate(sequenceLocal.tokens) if isConcept(token, pretrain=True)]
		numConcepts = len(conceptIndices)
		demoteIndices = set()
		if(numConcepts > 1):
			sequenceTokens = [convertPreprocessedTokenToSequenceToken(token) for token in sequenceLocal.tokens]
			conceptIndicesSorted = sorted(conceptIndices)
			for conceptPosition in range(numConcepts - 1):
				leftIndex = conceptIndicesSorted[conceptPosition]
				rightIndex = conceptIndicesSorted[conceptPosition + 1]
				rightmostDeterministic = None
				rightmostIndeterministic = None
				for tokenIndex in range(leftIndex + 1, rightIndex):
					token = sequenceTokens[tokenIndex]
					if(isTokenReferenceSetDelimiterDeterministic(token)):
						rightmostDeterministic = tokenIndex
					elif(detectReferenceSetDelimitersBetweenNouns and isTokenReferenceSetDelimiterProbabilistic(token) and not isTokenReferenceSetDelimiterDeterministic(token)):
						rightmostIndeterministic = tokenIndex
				if(rightmostDeterministic is None and rightmostIndeterministic is None):
					for tokenIndex in range(leftIndex, rightIndex):
						demoteIndices.add(tokenIndex)
		if(len(demoteIndices) > 0):
			nonConceptPos = "X"
			for tokenIndex in demoteIndices:
				sequenceLocal.tokens[tokenIndex].pos_ = nonConceptPos
		result = sequenceLocal
		return result

if(pretrainCombineConsecutiveNouns):

	def pretrainCombineConsecutiveNoun(sequence):
		sequence = ensure_preprocessed_sequence(sequence)
		preprocessedTokens = []
		buffer = []
		def flush_buffer():
			nonlocal buffer, preprocessedTokens
			if(len(buffer) == 0):
				return
			preprocessedTokens.append(createCombinedToken(buffer))
			buffer = []
		for token in sequence.tokens:
			if(isConcept(token, pretrain=True)):
				buffer.append(token)
			else:
				flush_buffer()
				preprocessedTokens.append(token)
		flush_buffer()
		return PreprocessedSequence(preprocessedTokens)

	def ensure_preprocessed_sequence(sequence):
		if isinstance(sequence, PreprocessedSequence):
			return sequence
		preprocessed = [createSinglePreprocessedToken(token) for token in sequence]
		return PreprocessedSequence(preprocessed)

	def createSinglePreprocessedToken(token):
		text = token.text
		lemma = token.lemma_
		pos = token.pos_
		tag = token.tag_
		return PreprocessedToken(text, lemma, pos, tag)

	def createCombinedToken(tokens):
		if(len(tokens) == 1):
			return createSinglePreprocessedToken(tokens[0])
		combinedText = "_".join(token.text for token in tokens)
		combinedLemma = "_".join(token.lemma_ for token in tokens)
		combinedPos = tokens[0].pos_
		combinedTag = tokens[0].tag_
		return PreprocessedToken(combinedText, combinedLemma, combinedPos, combinedTag)

if(pretrainCombineHyphenatedNouns):

	def pretrainCombineConsecutiveNounHyphenated(sequence):
		result = None
		sequence = ensure_preprocessed_sequence(sequence)
		preprocessedTokens = []
		buffer = []
		bufferJoiners = []
		pendingJoiner = None
		def flush_buffer():
			nonlocal buffer, preprocessedTokens, bufferJoiners, pendingJoiner
			if(len(buffer) > 0):
				preprocessedTokens.append(createCombinedTokenWithJoiners(buffer, bufferJoiners))
			buffer = []
			bufferJoiners = []
			pendingJoiner = None
		sequenceTokens = sequence.tokens
		sequenceTokenCount = len(sequenceTokens)
		for tokenIndex, token in enumerate(sequenceTokens):
			if(isConcept(token, pretrain=True)):
				if(len(buffer) > 0):
					if(pendingJoiner is None):
						bufferJoiners.append("_")
					else:
						bufferJoiners.append(pendingJoiner)
					pendingJoiner = None
				buffer.append(token)
			elif(isHyphenToken(token) and len(buffer) > 0 and tokenIndex + 1 < sequenceTokenCount and isConcept(sequenceTokens[tokenIndex + 1], pretrain=True)):
				pendingJoiner = "-"
			else:
				flush_buffer()
				preprocessedTokens.append(token)
		flush_buffer()
		result = PreprocessedSequence(preprocessedTokens)
		return result

	def createCombinedTokenWithJoiners(tokens, joiners):
		result = None
		if(len(tokens) == 1):
			result = createSinglePreprocessedToken(tokens[0])
		else:
			joinerCount = len(tokens) - 1
			joinersLocal = joiners if (joiners is not None and len(joiners) == joinerCount) else ["_"] * joinerCount
			combinedText = buildCombinedTokenString(tokens, joinersLocal, False)
			combinedLemma = buildCombinedTokenString(tokens, joinersLocal, True)
			combinedPos = tokens[0].pos_
			combinedTag = tokens[0].tag_
			result = PreprocessedToken(combinedText, combinedLemma, combinedPos, combinedTag)
		return result

	def buildCombinedTokenString(tokens, joiners, useLemma):
		combined = None
		parts = []
		for tokenIndex, token in enumerate(tokens):
			tokenText = token.lemma_ if useLemma else token.text
			if(tokenIndex == 0):
				parts.append(tokenText)
			else:
				parts.append(joiners[tokenIndex - 1])
				parts.append(tokenText)
		combined = "".join(parts)
		return combined

	def isHyphenToken(token):
		result = False
		hyphenChars = ("-", "–")
		if(token.text in hyphenChars or token.lemma_ in hyphenChars):
			result = True
		return result

def isConcept(token, pretrain=False):
	result = False
	if(pretrain):
		tokenPos = token.pos_
		tokenWord = token.text.lower()
		tokenLemma = token.lemma_
	else:
		tokenPos = token.pos
		tokenWord = token.word
		tokenLemma = token.lemma
	if(useSpacyForConceptNounPOSdetection):
		if tokenPos in nounPos:
			result = True
		#if tokenPos in nounTags:
		#	result = True
	else:
		nounNounCandidateDetected = False
		if(GIAANNproto_sequencePOS.isWordEverInPOStypeList(tokenWord, nonNounPos)):
			nounNounCandidateDetected = True
		elif(tokenLemma is not None and GIAANNproto_sequencePOS.isWordEverInPOStypeList(tokenLemma, nonNounPos)):
			nounNounCandidateDetected = True
		if(not nounNounCandidateDetected):
			result = True
	return result
