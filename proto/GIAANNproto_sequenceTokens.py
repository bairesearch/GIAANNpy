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
	return sequence

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
			if(token.pos_ in nounPos):
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
			if(token.pos_ in nounPos):
				if(len(buffer) > 0):
					if(pendingJoiner is None):
						bufferJoiners.append("_")
					else:
						bufferJoiners.append(pendingJoiner)
					pendingJoiner = None
				buffer.append(token)
			elif(isHyphenToken(token) and len(buffer) > 0 and tokenIndex + 1 < sequenceTokenCount and sequenceTokens[tokenIndex + 1].pos_ in nounPos):
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
		if(token.text == "-" or token.lemma_ == "-"):
			result = True
		return result

def isConcept(token):
	result = False
	if token.pos in nounPos:
		result = True
	if token.tag in nounTags:
		result = True
	return result
