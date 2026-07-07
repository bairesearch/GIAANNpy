"""GIAANNnlp_sequenceTokens.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 BAI Research Pty Ltd (bairesearch.com.au)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN NLP sequence Tokens

"""

import torch as pt

from GIAANNcmn_globalDefs import *

if(usePOS):
	import GIAANNnlp_sequencePOS
if(tokeniserSubword):
	try:
		import tiktoken
	except ImportError as exception:
		raise RuntimeError("GIAANNnlp_sequenceTokens error: tokeniserSubword requires the tiktoken package") from exception
	_tokeniserSubwordEncoding = None

def loadPOSdatabase():
	GIAANNnlp_sequencePOS.loadPOSdatabase()

def isTokenReferenceSetDelimiterDeterministic(token):
	result = False
	if(tokeniserSubword and tokeniserSubwordPOS):
		if(token.word in conceptColumnsDelimiterWordTypes or token.tag in conceptColumnsDelimiterTagTypes):
			result = True
		elif(token.pos in conceptColumnsDelimiterPOStypes):
			result = True
	else:
		if(token.word in conceptColumnsDelimiterWordTypes or token.tag in conceptColumnsDelimiterTagTypes):
			result = True
		elif(GIAANNnlp_sequencePOS.isWordEverInPOStypeList(token.word, conceptColumnsDelimiterPOStypes)):
			result = True
		elif(token.lemma is not None and GIAANNnlp_sequencePOS.isWordEverInPOStypeList(token.lemma, conceptColumnsDelimiterPOStypes)):
			result = True
	return result
	
def isTokenReferenceSetDelimiterProbabilistic(token):
	result = False
	if(tokeniserSubword and tokeniserSubwordPOS):
		if(token.word in detectReferenceSetDelimitersBetweenNounsWordTypes or token.tag in detectReferenceSetDelimitersBetweenNounsTagTypes):
			result = True
		elif(token.pos in detectReferenceSetDelimitersBetweenNounsPOStypes):
			result = True
	else:
		if(token.word in detectReferenceSetDelimitersBetweenNounsWordTypes or token.tag in detectReferenceSetDelimitersBetweenNounsTagTypes):
			result = True
		elif(GIAANNnlp_sequencePOS.isWordEverInPOStypeList(token.word, detectReferenceSetDelimitersBetweenNounsPOStypes)):
			result = True
		elif(token.lemma is not None and GIAANNnlp_sequencePOS.isWordEverInPOStypeList(token.lemma, detectReferenceSetDelimitersBetweenNounsPOStypes)):
			result = True
	return result
	

class SequenceToken:
	def __init__(self, word, lemma, pos, tag, tokenId=None):
		self.word = word
		self.lemma = lemma
		self.pos = pos
		self.tag = tag
		self.tokenId = tokenId

# Preprocessing helpers
class PreprocessedToken:
	__slots__ = ("text", "lemma_", "pos_", "tag_", "tokenId")
	def __init__(self, text, lemma, pos, tag, tokenId=None):
		self.text = text
		self.lemma_ = lemma
		self.pos_ = pos
		self.tag_ = tag
		self.tokenId = tokenId

def convertPreprocessedTokenToSequenceToken(preprocessedToken):
	if(tokeniserSubword and useDedicatedFeatureListsSubword):
		word = preprocessedToken.text
		lemma = preprocessedToken.lemma_
		if(preprocessedToken.tokenId is None):
			raise RuntimeError("convertPreprocessedTokenToSequenceToken error: tokeniserSubword dedicated feature lists require tokenId")
	else:
		word = preprocessedToken.text.lower()
		lemma = preprocessedToken.lemma_.lower()
	pos = preprocessedToken.pos_  #coarse Part-of-speech (e.g. PRON) 
	tag = preprocessedToken.tag_	#fine-grained POS (e.g., PRP, PRP$, WP, WP$, etc.)
	token = SequenceToken(word, lemma, pos, tag, preprocessedToken.tokenId)
	return token

def getTokens(sequence):
	tokens = []
	for preprocessedToken in sequence:
		token = convertPreprocessedTokenToSequenceToken(preprocessedToken)
		tokens.append(token)
	return tokens

class PreprocessedSequence:
	__slots__ = ("tokens", "originalText")

	def __init__(self, tokens, originalText=None):
		self.tokens = tokens
		self.originalText = originalText

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
		result = None
		if(self.originalText is not None):
			result = self.originalText
		else:
			result = " ".join(token.text for token in self.tokens)
		return result


def preprocessSequence(sequence):
	return pretrain(sequence)

def createSinglePreprocessedToken(token):
	text = token.text
	lemma = token.lemma_
	pos = token.pos_
	tag = token.tag_
	return PreprocessedToken(text, lemma, pos, tag)

def pretrain(sequence):
	if(tokeniserSubword):
		sequence = pretrainTokeniserSubword(sequence)
	else:
		if(pretrainCombineHyphenatedNouns):
			sequence = pretrainCombineConsecutiveNounHyphenated(sequence)
		if(pretrainCombineConsecutiveNouns):
			sequence = pretrainCombineConsecutiveNoun(sequence)
		if(pretrainConceptColumnsDelimitByPOSenforce):
			sequence = pretrainConceptColumnsDelimitByPOSenforce(sequence)
	return sequence

if(tokeniserSubword):

	def pretrainTokeniserSubword(sequence):
		result = None
		if(not usePOS):
			raise RuntimeError("pretrainTokeniserSubword error: tokeniserSubword requires usePOS")
		if(isinstance(sequence, PreprocessedSequence)):
			result = sequence
		else:
			sequenceText = getTokeniserSubwordSequenceText(sequence)
			encoding = getTokeniserSubwordEncoding()
			preprocessedTokens = createTokeniserSubwordPreprocessedTokens(sequence, sequenceText, encoding)
			result = PreprocessedSequence(preprocessedTokens, sequenceText)
		return result

	def createTokeniserSubwordPreprocessedTokens(sequence, sequenceText, encoding):
		result = []
		parentTokenSpans = createTokeniserSubwordParentTokenSpans(sequence, sequenceText)
		sequenceBytes = encodeTokeniserSubwordText(sequenceText)
		tokenIds = encoding.encode_ordinary(sequenceText)
		if(len(tokenIds) == 0):
			raise RuntimeError("createTokeniserSubwordPreprocessedTokens error: no subword token ids generated")
		byteIndex = 0
		for tokenId in tokenIds:
			tokenBytes = getTokeniserSubwordTokenBytes(encoding, tokenId)
			subwordStartByte = byteIndex
			subwordEndByte = subwordStartByte + len(tokenBytes)
			if(subwordEndByte > len(sequenceBytes)):
				raise RuntimeError("createTokeniserSubwordPreprocessedTokens error: subword byte span exceeds sequence byte length")
			if(tokenBytes != sequenceBytes[subwordStartByte:subwordEndByte]):
				raise RuntimeError("createTokeniserSubwordPreprocessedTokens error: subword token bytes do not match sequence bytes")
			parentToken = getTokeniserSubwordParentToken(parentTokenSpans, subwordStartByte, subwordEndByte)
			subwordText = decodeTokeniserSubwordTokenBytes(tokenBytes)
			subwordPos, subwordTag = detectTokeniserSubwordPOS(parentToken, subwordText)
			result.append(PreprocessedToken(subwordText, subwordText, subwordPos, subwordTag, tokenId))
			byteIndex = subwordEndByte
		if(byteIndex != len(sequenceBytes)):
			raise RuntimeError("createTokeniserSubwordPreprocessedTokens error: subword token bytes do not cover sequence bytes")
		return result

	def getTokeniserSubwordSequenceText(sequence):
		result = None
		if(not hasattr(sequence, "text")):
			raise RuntimeError("getTokeniserSubwordSequenceText error: sequence has no text attribute")
		result = sequence.text
		if(not isinstance(result, str)):
			raise RuntimeError("getTokeniserSubwordSequenceText error: sequence text must be a str")
		if(result == ""):
			raise RuntimeError("getTokeniserSubwordSequenceText error: sequence text must not be empty")
		return result

	def createTokeniserSubwordParentTokenSpans(sequence, sequenceText):
		result = []
		charByteOffsets = createTokeniserSubwordCharacterByteOffsets(sequenceText)
		currentCharIndex = 0
		for token in sequence:
			tokenText = token.text
			if(not isinstance(tokenText, str)):
				raise RuntimeError("createTokeniserSubwordParentTokenSpans error: parent token text must be a str")
			if(tokenText == ""):
				raise RuntimeError("createTokeniserSubwordParentTokenSpans error: parent token text must not be empty")
			tokenStartChar = sequenceText.find(tokenText, currentCharIndex)
			if(tokenStartChar < 0):
				raise RuntimeError("createTokeniserSubwordParentTokenSpans error: parent token text not found in sequence text")
			tokenEndChar = tokenStartChar + len(tokenText)
			tokenStartByte = charByteOffsets[tokenStartChar]
			tokenEndByte = charByteOffsets[tokenEndChar]
			result.append((tokenStartByte, tokenEndByte, token))
			currentCharIndex = tokenEndChar
		if(len(result) == 0):
			raise RuntimeError("createTokeniserSubwordParentTokenSpans error: no parent token spans generated")
		return result

	def createTokeniserSubwordCharacterByteOffsets(sequenceText):
		result = [0]
		byteOffset = 0
		for character in sequenceText:
			byteOffset += len(encodeTokeniserSubwordText(character))
			result.append(byteOffset)
		return result

	def getTokeniserSubwordParentToken(parentTokenSpans, subwordStartByte, subwordEndByte):
		result = None
		bestOverlap = 0
		nextToken = None
		if(not isinstance(subwordStartByte, int) or isinstance(subwordStartByte, bool)):
			raise RuntimeError("getTokeniserSubwordParentToken error: subwordStartByte must be an int")
		if(not isinstance(subwordEndByte, int) or isinstance(subwordEndByte, bool)):
			raise RuntimeError("getTokeniserSubwordParentToken error: subwordEndByte must be an int")
		if(subwordStartByte < 0 or subwordEndByte <= subwordStartByte):
			raise RuntimeError("getTokeniserSubwordParentToken error: invalid subword byte span")
		if(len(parentTokenSpans) == 0):
			raise RuntimeError("getTokeniserSubwordParentToken error: parentTokenSpans must not be empty")
		for parentStartByte, parentEndByte, parentToken in parentTokenSpans:
			overlapStartByte = max(parentStartByte, subwordStartByte)
			overlapEndByte = min(parentEndByte, subwordEndByte)
			overlap = max(0, overlapEndByte - overlapStartByte)
			if(overlap > bestOverlap):
				bestOverlap = overlap
				result = parentToken
			if(nextToken is None and parentStartByte >= subwordEndByte):
				nextToken = parentToken
		if(result is None):
			if(nextToken is not None):
				result = nextToken
			else:
				result = parentTokenSpans[-1][2]
		return result

	def encodeTokeniserSubwordText(text):
		result = None
		if(not isinstance(text, str)):
			raise RuntimeError("encodeTokeniserSubwordText error: text must be a str")
		result = text.encode(tokeniserSubwordTextEncoding, errors=tokeniserSubwordTextEncodingErrorMode)
		return result

	def getTokeniserSubwordTokenBytes(encoding, tokenId):
		result = None
		if(not isinstance(tokenId, int) or isinstance(tokenId, bool) or tokenId < 0):
			raise RuntimeError("getTokeniserSubwordTokenBytes error: tokenId must be a non-negative int")
		result = encoding.decode_single_token_bytes(tokenId)
		if(not isinstance(result, bytes)):
			raise RuntimeError("getTokeniserSubwordTokenBytes error: token bytes must be bytes")
		if(len(result) == 0):
			raise RuntimeError("getTokeniserSubwordTokenBytes error: token bytes must not be empty")
		return result

	def getTokeniserSubwordEncoding():
		global _tokeniserSubwordEncoding
		result = None
		if(_tokeniserSubwordEncoding is None):
			try:
				_tokeniserSubwordEncoding = tiktoken.get_encoding(tokeniserSubwordTiktokenEncodingName)
			except Exception as exception:
				raise RuntimeError("getTokeniserSubwordEncoding error: failed to load tiktoken encoding " + tokeniserSubwordTiktokenEncodingName) from exception
		result = _tokeniserSubwordEncoding
		return result

	def getTokeniserSubwordFeatureCount():
		encoding = getTokeniserSubwordEncoding()
		result = int(encoding.max_token_value) + tokeniserSubwordFeatureIndexOffset + 1
		return result

	def getTokeniserSubwordFeatureIndex(token):
		if(not hasattr(token, "tokenId")):
			raise RuntimeError("getTokeniserSubwordFeatureIndex error: token has no tokenId")
		result = getTokeniserSubwordFeatureIndexFromTokenId(token.tokenId)
		return result

	def getTokeniserSubwordFeatureIndexFromTokenId(tokenId):
		result = None
		encoding = getTokeniserSubwordEncoding()
		if(not isinstance(tokenId, int) or isinstance(tokenId, bool)):
			raise RuntimeError("getTokeniserSubwordFeatureIndexFromTokenId error: tokenId must be an int")
		if(tokenId < 0 or tokenId > int(encoding.max_token_value)):
			raise RuntimeError("getTokeniserSubwordFeatureIndexFromTokenId error: tokenId out of range")
		result = tokeniserSubwordFeatureIndexOffset + tokenId
		return result

	def getTokeniserSubwordFeatureNameForTokenId(encoding, tokenId):
		result = None
		if(not isinstance(tokenId, int) or isinstance(tokenId, bool)):
			raise RuntimeError("getTokeniserSubwordFeatureNameForTokenId error: tokenId must be an int")
		if(tokenId < 0 or tokenId > int(encoding.max_token_value)):
			raise RuntimeError("getTokeniserSubwordFeatureNameForTokenId error: tokenId out of range")
		try:
			tokenBytes = getTokeniserSubwordTokenBytes(encoding, tokenId)
			result = decodeTokeniserSubwordTokenBytes(tokenBytes)
		except KeyError:
			result = tokeniserSubwordInvalidTokenFeatureNamePrefix + str(tokenId) + tokeniserSubwordInvalidTokenFeatureNameSuffix
		return result

	def decodeTokeniserSubwordTokenText(encoding, tokenId):
		tokenBytes = getTokeniserSubwordTokenBytes(encoding, tokenId)
		result = decodeTokeniserSubwordTokenBytes(tokenBytes)
		return result

	def decodeTokeniserSubwordTokenBytes(tokenBytes):
		result = None
		byteTokenGenerated = False
		if(not isinstance(tokenBytes, bytes)):
			raise RuntimeError("decodeTokeniserSubwordTokenBytes error: tokenBytes must be bytes")
		if(len(tokenBytes) == 0):
			raise RuntimeError("decodeTokeniserSubwordTokenBytes error: tokenBytes must not be empty")
		try:
			result = tokenBytes.decode(tokeniserSubwordTextEncoding, errors=tokeniserSubwordTextEncodingErrorMode)
		except UnicodeDecodeError:
			result = tokeniserSubwordByteTokenPrefix + tokenBytes.hex() + tokeniserSubwordByteTokenSuffix
			byteTokenGenerated = True
		if(result == ""):
			raise RuntimeError("decodeTokeniserSubwordTokenBytes error: decoded subword text must not be empty")
		if(not byteTokenGenerated and result.startswith(tokeniserSubwordByteTokenPrefix)):
			raise RuntimeError("decodeTokeniserSubwordTokenBytes error: decoded subword text uses reserved byte-token prefix")
		return result

	def detectTokeniserSubwordPOS(parentToken, subwordText):
		resultPos = parentToken.pos_
		resultTag = parentToken.tag_
		if(tokeniserSubwordPOS):
			if(isTokeniserSubwordByteTokenText(subwordText)):
				resultPos = parentToken.pos_
				resultTag = parentToken.tag_
			elif(isTokeniserSubwordWhitespaceText(subwordText)):
				resultPos = tokeniserSubwordPOSspace
				resultTag = tokeniserSubwordTagSpace
			elif(GIAANNnlp_sequencePOS.isPunctWord(subwordText)):
				resultPos = tokeniserSubwordPOSpunct
				resultTag = tokeniserSubwordTagPunct
			elif(GIAANNnlp_sequencePOS.isNumericWord(subwordText)):
				resultPos = tokeniserSubwordPOSnum
				resultTag = tokeniserSubwordTagNum
			elif(GIAANNnlp_sequencePOS.isSymbolWord(subwordText)):
				resultPos = tokeniserSubwordPOSsym
				resultTag = tokeniserSubwordTagSym
		return resultPos, resultTag

	def isTokeniserSubwordByteTokenText(subwordText):
		result = False
		if(subwordText.startswith(tokeniserSubwordByteTokenPrefix) and subwordText.endswith(tokeniserSubwordByteTokenSuffix)):
			result = True
		return result

	def isTokeniserSubwordWhitespaceText(subwordText):
		result = False
		if(not isinstance(subwordText, str)):
			raise RuntimeError("isTokeniserSubwordWhitespaceText error: subwordText must be a str")
		if(subwordText.isspace()):
			result = True
		return result

if(pretrainConceptColumnsDelimitByPOSenforce):
	
	def pretrainConceptColumnsDelimitByPOSenforce(sequence):
		result = None
		sequenceLocal = sequence if isinstance(sequence, PreprocessedSequence) else PreprocessedSequence([createSinglePreprocessedToken(token) for token in sequence])
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
	if(tokeniserSubword):
		if tokenPos in nounPos:
			result = True
	elif(useSpacyForConceptNounPOSdetection):
		if tokenPos in nounPos:
			result = True
		#if tokenPos in nounTags:
		#	result = True
	else:
		nounNounCandidateDetected = False
		if(GIAANNnlp_sequencePOS.isWordEverInPOStypeList(tokenWord, nonNounPos)):
			nounNounCandidateDetected = True
		elif(tokenLemma is not None and GIAANNnlp_sequencePOS.isWordEverInPOStypeList(tokenLemma, nonNounPos)):
			nounNounCandidateDetected = True
		if(not nounNounCandidateDetected):
			result = True
	return result
