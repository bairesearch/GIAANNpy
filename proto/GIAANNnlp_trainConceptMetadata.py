"""List-based concept boundaries for the supported fixed-vocabulary training path."""

from GIAANNcmn_globalDefs import *


def processConceptWords(sequenceObservedColumns, sequenceIndex, sequence, tokens):
	result = None
	if(optimiseTrainConceptMetadata):
		import GIAANNnlp_sequenceConcepts
		conceptMask, conceptIndices, numberConcepts = GIAANNnlp_sequenceConcepts.createConceptMask(sequenceObservedColumns, tokens)
		if(numberConcepts > trainSequenceBatchZero):
			positions = conceptIndices.tolist()
			if(positions != sorted(set(positions)) or positions[trainSequenceBatchZero] < trainSequenceBatchZero or positions[-trainSequenceBatchOne] >= len(tokens)):
				raise RuntimeError(trainConceptMetadataInvalidInput)
			database = sequenceObservedColumns.databaseNetworkObject
			deterministic = database.sequenceReferenceSetDelimiterDeterministicList
			probabilistic = database.sequenceReferenceSetDelimiterProbabilisticList
			starts = [trainSequenceBatchZero] * numberConcepts
			ends = [len(tokens)] * numberConcepts
			missing = False
			for position, (left, right) in enumerate(zip(positions, positions[trainSequenceBatchOne:])):
				deterministicBoundary = None
				probabilisticBoundary = None
				for tokenIndex in range(left + trainSequenceBatchOne, right):
					if(tokenIndex < len(deterministic) and deterministic[tokenIndex]):
						deterministicBoundary = tokenIndex
					elif(tokenIndex < len(probabilistic) and probabilistic[tokenIndex]):
						probabilisticBoundary = tokenIndex
				if(deterministicBoundary is not None):
					boundary = deterministicBoundary
				elif(probabilisticBoundary is not None):
					boundary = probabilisticBoundary
				else:
					missing = missing or skipSequenceNoDelimiterDetectedBetweenConceptTokens
					boundary = left if skipSequenceNoDelimiterDetectedBetweenConceptTokens else right - trainSequenceBatchOne
				ends[position] = boundary + trainSequenceBatchOne
				starts[position + trainSequenceBatchOne] = boundary + trainSequenceBatchOne
			startIndices = pt.tensor(starts, dtype=pt.long)
			endIndices = pt.tensor(ends, dtype=pt.long)
			# This also populates tokenConceptColumnIndexList, even when printing is disabled.
			assignment = GIAANNnlp_sequenceConcepts.buildSequenceConceptAssignment(sequenceObservedColumns, sequence, tokens, conceptIndices, startIndices, endIndices)
			if(printSequenceConceptAssignment):
				sequenceObservedColumns.sentenceWithConceptAssignment = assignment
			if(missing):
				sequenceObservedColumns.noDelimiterDetectedBetweenConceptTokens = True
			else:
				result = (conceptIndices, startIndices, endIndices)
	return result
