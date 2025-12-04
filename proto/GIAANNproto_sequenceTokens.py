"""GIAANNproto_sequenceTokens.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

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

def isTokenReferenceSetDelimiterDeterministic(token):
	if(token.pos in conceptColumnsDelimiterPOStypes or token.word in conceptColumnsDelimiterWordTypes or token.tag in conceptColumnsDelimiterTagTypes):
		return True
	else:
		return False
	
def isTokenReferenceSetDelimiterProbabilistic(token):
	if(token.pos in detectReferenceSetDelimitersBetweenNounsPOStypes or token.word in detectReferenceSetDelimitersBetweenNounsWordTypes or token.tag in detectReferenceSetDelimitersBetweenNounsTagTypes):
		return True
	else:
		return False
	


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

def isConcept(token):
	result = False
	if token.pos in nounPos:
		result = True
	if token.tag in nounTags:
		result = True
	return result

