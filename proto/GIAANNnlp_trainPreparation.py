"""Shared preparation for lightweight training workers and the enabled parent path."""

from GIAANNcmn_globalDefs import *

if(optimiseTrainLightweightWorkers):
	import GIAANNnlp_sequenceTokens
	nlpArticle = None


def initialiseWorker():
	global nlpArticle
	if(optimiseTrainLightweightWorkers):
		import spacy
		nlpArticle = spacy.load(spacyModelName, disable=trainLightweightWorkerDisabledComponents)
	return


def loadPOSdatabase():
	if(optimiseTrainLightweightWorkers):
		if(usePOS):
			GIAANNnlp_sequenceTokens.loadPOSdatabase()
	return


def generateSeqencesBatchOrSerial(textParsed, skipMode):
	result = None
	if(optimiseTrainLightweightWorkers):
		minSequenceLength = numSeedTokensInference + trainSequenceBatchOne
		sequences = []
		sequencesRaw = []
		startIndex = trainSequenceBatchZero
		endIndex = len(textParsed)
		sequenceEndIndex = None
		sequenceStartIndex = None
		sequenceParsed = None
		sequenceText = None
		if(maxSequenceLength < minSequenceLength):
			raise RuntimeError(trainLightweightWorkerInvalidInput)
		while(startIndex < endIndex and textParsed[startIndex].is_space):
			startIndex += trainSequenceBatchOne
		while(endIndex > startIndex and textParsed[endIndex-trainSequenceBatchOne].is_space):
			endIndex -= trainSequenceBatchOne
		while(startIndex < endIndex):
			sequenceStartIndex = startIndex
			sequenceEndIndex = min(startIndex+maxSequenceLength, endIndex)
			while(sequenceStartIndex < sequenceEndIndex and textParsed[sequenceStartIndex].is_space):
				sequenceStartIndex += trainSequenceBatchOne
			while(sequenceEndIndex > sequenceStartIndex and textParsed[sequenceEndIndex-trainSequenceBatchOne].is_space):
				sequenceEndIndex -= trainSequenceBatchOne
			if(sequenceStartIndex < sequenceEndIndex):
				sequenceParsed = textParsed[sequenceStartIndex:sequenceEndIndex]
				sequenceText = sequenceParsed.text
			if(sequenceStartIndex < sequenceEndIndex and len(sequenceParsed) >= minSequenceLength):
				sequences.append(sequenceParsed)
				sequencesRaw.append(sequenceText)
			startIndex += maxSequenceLength
		result = (sequences, sequencesRaw)
	return result


def enforceSequenceLengthTokenLimits(sequence, sequenceRaw):
	result = None
	if(optimiseTrainLightweightWorkers):
		resultSequence = sequence
		resultSequenceRaw = sequenceRaw
		resultSequenceAccepted = True
		if(tokeniserSubword):
			if(not isinstance(maxSequenceLengthTokens, int) or isinstance(maxSequenceLengthTokens, bool) or maxSequenceLengthTokens <= trainSequenceBatchZero):
				raise RuntimeError(trainLightweightWorkerInvalidInput)
			if(tokenisationEnforcePrecharacterByteLimit):
				resultSequenceAccepted = GIAANNnlp_sequenceTokens.isTokeniserSubwordSequenceWithinPrecharacterByteLimit(resultSequence)
			if(resultSequenceAccepted):
				resultSequence = GIAANNnlp_sequenceTokens.preprocessSequence(resultSequence)
				if(len(resultSequence) > maxSequenceLengthTokens):
					if(sequencesCropToMaxLength):
						resultSequence = GIAANNnlp_sequenceTokens.cropTokeniserSubwordSequence(resultSequence, maxSequenceLengthTokens)
						resultSequenceRaw = resultSequence.text
					else:
						resultSequenceAccepted = False
		result = (resultSequence, resultSequenceRaw, resultSequenceAccepted)
	return result
