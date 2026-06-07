"""GIAANNcmn_executionProgress.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 BAI Research Pty Ltd (bairesearch.com.au)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN common execution progress printing

"""

from GIAANNcmn_globalDefs import *
if(modalityName=="NLP"):
	import GIAANNnlp_sequenceConcepts
	import GIAANNnlp_sequenceTokens

if(printTrainSequenceBar or printEvalSequenceBar):
	from tqdm import tqdm

trainSequenceBar = None

def printTrainSequenceText(sequenceCount, sequence, tokens, sequenceRaw):
	if(printSequencePOS):
		sentenceWithPOS = " ".join(f"{convertTokenTextToTerminalSafeText(token.text)} ({tokenIndex}:{token.pos_})" for tokenIndex, token in enumerate(sequence))
		print(f"Processing sequenceCount: {sequenceCount}, {sentenceWithPOS}")	#article: {articleIndex}, sequence: {sequenceIndex}
	if(printSequenceDelimiters):
		sentenceWithDelimiters = buildSequenceWithDelimiters(sequence, tokens)
		print(f"Processing sequenceCount: {sequenceCount}, {sentenceWithDelimiters}")	#article: {articleIndex}, sequence: {sequenceIndex}
	if(printSequenceRaw):
		print(sequenceRaw)
	if(printSequenceDefault):
		print(f"Processing sequenceCount: {sequenceCount}, {convertTokenTextToTerminalSafeText(sequence.text)}")	#"{sequence.text}"	#"Processing sequenceCount: {sequenceCount}, {sequence.text}"	#article: {articleIndex}, sequence: {sequenceIndex}
	if(printSequenceCount):
		print(f"Processing sequenceCount: {sequenceCount}")
	return

def buildSequenceWithDelimiters(sequence, tokens):
	result = None
	if(conceptColumnsDelimitByPOS):
		delimiterTypes = []
		for tokenIndex, token in enumerate(tokens):
			_, isDelimiterDeterministic, isDelimiterProbabilistic = GIAANNnlp_sequenceConcepts.isFeaturePOSreferenceSetDelimiterType(token.word, token, tokens, tokenIndex)
			if(isDelimiterDeterministic):
				delimiterTypes.append("Dd")	#deterministic
			elif(isDelimiterProbabilistic):
				delimiterTypes.append("Di")	#indeterministic
			elif(GIAANNnlp_sequenceTokens.isConcept(token)):
				delimiterTypes.append("C")	#concept
			else:
				delimiterTypes.append("")	#non
	else:
		printe("conceptColumnsDelimitByPOS is required")
	result = " ".join(f"{convertTokenTextToTerminalSafeText(token.text)} ({tokenIndex}:{delimiterTypes[tokenIndex]})" for tokenIndex, token in enumerate(sequence))
	return result


def convertTokenTextToTerminalSafeText(tokenText):
	result = None
	if(not isinstance(tokenText, str)):
		raise RuntimeError("convertTokenTextToTerminalSafeText error: tokenText must be a str")
	result = printSequenceTerminalSafeTextEmpty.join(convertTokenCharacterToTerminalSafeText(character) for character in tokenText)
	return result


def convertTokenCharacterToTerminalSafeText(character):
	result = None
	if(not isinstance(character, str)):
		raise RuntimeError("convertTokenCharacterToTerminalSafeText error: character must be a str")
	if(len(character) != 1):
		raise RuntimeError("convertTokenCharacterToTerminalSafeText error: character length must be 1")
	if(character.isprintable()):
		result = character
	else:
		result = convertTokenCharacterToTerminalSafeEscape(character)
	return result


def convertTokenCharacterToTerminalSafeEscape(character):
	result = None
	codepoint = None
	escapePrefix = None
	escapeCodepointWidth = None
	if(not isinstance(character, str)):
		raise RuntimeError("convertTokenCharacterToTerminalSafeEscape error: character must be a str")
	if(len(character) != 1):
		raise RuntimeError("convertTokenCharacterToTerminalSafeEscape error: character length must be 1")
	codepoint = ord(character)
	if(codepoint <= printSequenceTerminalSafeEscapeCodepointMax16Bit):
		escapePrefix = printSequenceTerminalSafeEscapePrefix16Bit
		escapeCodepointWidth = printSequenceTerminalSafeEscapeCodepointWidth16Bit
	else:
		escapePrefix = printSequenceTerminalSafeEscapePrefix32Bit
		escapeCodepointWidth = printSequenceTerminalSafeEscapeCodepointWidth32Bit
	result = escapePrefix + format(codepoint, printSequenceTerminalSafeEscapeFormatPadPrefix + str(escapeCodepointWidth) + printSequenceTerminalSafeEscapeFormatType)
	return result


def initialiseTrainSequenceBar(sequenceCount):
	result = None
	initialiseSequenceBar(sequenceCount, trainMaxSequences, printTrainSequenceBarDescription, printTrainSequenceBarUnit, printTrainSequenceBar)
	return result


def initialiseEvalSequenceBar(sequenceCount, evalSequenceTotal):
	result = None
	initialiseSequenceBar(sequenceCount, evalSequenceTotal, printEvalSequenceBarDescription, printEvalSequenceBarUnit, printEvalSequenceBar)
	return result


def initialisePromptSequenceBar(sequenceCount, promptSequenceTotal):
	result = None
	initialiseSequenceBar(sequenceCount, promptSequenceTotal, printPromptSequenceBarDescription, printPromptSequenceBarUnit, printTrainSequenceBar or printEvalSequenceBar)
	return result


def initialiseSequenceBar(sequenceCount, sequenceTotal, sequenceBarDescription, sequenceBarUnit, sequenceBarEnabled):
	result = None
	global trainSequenceBar
	if(sequenceBarEnabled):
		if(not isinstance(sequenceCount, int)):
			raise RuntimeError("initialiseSequenceBar error: sequenceCount must be an int")
		if(sequenceCount < 0):
			raise RuntimeError("initialiseSequenceBar error: sequenceCount must be >= 0")
		if(not isinstance(sequenceTotal, int)):
			raise RuntimeError("initialiseSequenceBar error: sequenceTotal must be an int")
		if(sequenceTotal <= 0):
			raise RuntimeError("initialiseSequenceBar error: sequenceTotal must be > 0")
		if(sequenceCount > sequenceTotal):
			raise RuntimeError("initialiseSequenceBar error: sequenceCount must be <= sequenceTotal")
		if(trainSequenceBar is None):
			trainSequenceBar = tqdm(total=sequenceTotal, initial=sequenceCount, desc=sequenceBarDescription, unit=sequenceBarUnit)
		else:
			raise RuntimeError("initialiseSequenceBar error: trainSequenceBar is already initialised")
	return result


def updateTrainSequenceBar(sequenceCount):
	result = None
	updateSequenceBar(sequenceCount, printTrainSequenceBar, printTrainSequenceBarUpdateStep)
	return result


def updateEvalSequenceBar(sequenceCount):
	result = None
	updateSequenceBar(sequenceCount, printEvalSequenceBar, printEvalSequenceBarUpdateStep)
	return result


def updatePromptSequenceBar(sequenceCount):
	result = None
	updateSequenceBar(sequenceCount, printTrainSequenceBar or printEvalSequenceBar, printPromptSequenceBarUpdateStep)
	return result


def updateSequenceBar(sequenceCount, sequenceBarEnabled, sequenceBarUpdateStep):
	result = None
	if(sequenceBarEnabled):
		if(not isinstance(sequenceCount, int)):
			raise RuntimeError("updateSequenceBar error: sequenceCount must be an int")
		if(sequenceCount < 0):
			raise RuntimeError("updateSequenceBar error: sequenceCount must be >= 0")
		if(not isinstance(sequenceBarUpdateStep, int)):
			raise RuntimeError("updateSequenceBar error: sequenceBarUpdateStep must be an int")
		if(sequenceBarUpdateStep <= 0):
			raise RuntimeError("updateSequenceBar error: sequenceBarUpdateStep must be > 0")
		if(trainSequenceBar is None):
			raise RuntimeError("trainSequenceBar is None; requires initialisation")
		if(trainSequenceBar.n + sequenceBarUpdateStep > trainSequenceBar.total):
			if(useTrainDuringInference):
				sequenceBarUpdateStep = trainSequenceBar.total - trainSequenceBar.n
			else:
				raise RuntimeError("updateSequenceBar error: sequence bar update exceeds total")
		if(sequenceBarUpdateStep > 0):
			trainSequenceBar.update(sequenceBarUpdateStep)
	return result


def closeTrainSequenceBar():
	result = None
	global trainSequenceBar
	if(printTrainSequenceBar or printEvalSequenceBar):
		if(trainSequenceBar is not None):
			trainSequenceBar.close()
			trainSequenceBar = None
	return result
