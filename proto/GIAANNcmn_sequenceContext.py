"""GIAANNcmn_sequenceContext.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 BAI Research Pty Ltd (bairesearch.com.au)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN common sequence Context

Exact training-sequence context index for the optional GIAANN LIF readout.

"""


from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import pickle
import numpy as np

from GIAANNcmn_globalDefs import *
import GIAANNcmn_databaseNetworkFiles


def initialiseSequenceContextTraining(databaseNetworkObject):
	if(useSequenceContext):
		if(databaseNetworkObject.inferenceMode or (trainLoadExistingDatabase and GIAANNcmn_databaseNetworkFiles.pathExists(conceptColumnsDictFile)) or hasattr(databaseNetworkObject, sequenceContextTrainingAttribute)):
			raise RuntimeError(sequenceContextInvalidTraining)
		folder = Path(sequenceContextFolder)
		folder.mkdir(parents=True, exist_ok=False)
		with (folder / sequenceContextTokensFile).open(sequenceContextWriteMode):
			pass
		databaseNetworkObject.sequenceContextTrainingBoundaries = [sequenceContextZero]
		databaseNetworkObject.sequenceContextLastTrainingSequence = -sequenceContextOne
	return


def recordSequenceContextTraining(databaseNetworkObject, tokens, sequenceCount):
	if(useSequenceContext):
		if(databaseNetworkObject.inferenceMode or not hasattr(databaseNetworkObject, sequenceContextTrainingAttribute) or type(sequenceCount) is not int or sequenceCount <= databaseNetworkObject.sequenceContextLastTrainingSequence or not tokens):
			raise RuntimeError(sequenceContextInvalidTraining)
		indices = [databaseNetworkObject.conceptFeaturesDict[token.word] for token in tokens]
		if(any(type(index) is not int or not sequenceContextZero < index < databaseNetworkObject.f for index in indices)):
			raise ValueError(sequenceContextInvalidIndex)
		total = databaseNetworkObject.sequenceContextTrainingBoundaries[-sequenceContextOne] + len(indices)
		if(total >= sequenceContextOne << sequenceContextPositionBits):
			raise ValueError(sequenceContextInvalidTraining)
		with (Path(sequenceContextFolder) / sequenceContextTokensFile).open(sequenceContextAppendMode) as handle:
			np.asarray(indices, dtype=sequenceContextTokenDtype).tofile(handle)
		databaseNetworkObject.sequenceContextTrainingBoundaries.append(total)
		databaseNetworkObject.sequenceContextLastTrainingSequence = sequenceCount
	return


def finaliseSequenceContextTraining(databaseNetworkObject, sequenceCount):
	if(useSequenceContext):
		if(databaseNetworkObject.inferenceMode or not hasattr(databaseNetworkObject, sequenceContextTrainingAttribute) or type(sequenceCount) is not int or sequenceCount <= databaseNetworkObject.sequenceContextLastTrainingSequence):
			raise RuntimeError(sequenceContextInvalidTraining)
		boundaries = np.asarray(databaseNetworkObject.sequenceContextTrainingBoundaries, dtype=sequenceContextKeyDtype)
		np.save(Path(sequenceContextFolder) / sequenceContextBoundariesFile, boundaries)
		buildSequenceContextIndex(sequenceContextFolder, databaseNetworkObject.conceptFeaturesList, sequenceCount)
	return


def buildSequenceContextIndex(folderName, featureList, processedSequences):
	if(useSequenceContext):
		folder = Path(folderName)
		if(type(processedSequences) is not int or processedSequences <= sequenceContextZero or (folder / sequenceContextMetadataFile).exists()):
			raise ValueError(sequenceContextInvalidTraining)
		tokens = np.memmap(folder / sequenceContextTokensFile, mode=sequenceContextReadMode, dtype=sequenceContextTokenDtype)
		boundaries = np.load(folder / sequenceContextBoundariesFile, allow_pickle=False)
		validateSequenceContextCorpus(tokens, boundaries, len(featureList))
		if(len(boundaries) - sequenceContextOne > processedSequences):
			raise ValueError(sequenceContextInvalidIndex)
		length = len(tokens) - sequenceContextMinimumLength
		keys = np.zeros(length, dtype=sequenceContextKeyDtype)
		for offset in range(sequenceContextMinimumLength):
			keys <<= sequenceContextBits
			keys |= tokens[offset:offset + length]
		# Every indexed occurrence includes its continuation token and stays within one training sequence.
		for offset in range(sequenceContextOne, sequenceContextMinimumLength + sequenceContextOne):
			positions = boundaries[sequenceContextOne:-sequenceContextOne].astype(np.int64) - offset
			positions = positions[(positions >= sequenceContextZero) & (positions < length)]
			keys[positions] = sequenceContextZero
		permutation = np.argsort(keys, kind=sequenceContextStableSort).astype(sequenceContextTokenDtype)
		keys = keys[permutation]
		first = int(np.searchsorted(keys, np.uint64(sequenceContextOne)))
		keys = keys[first:]
		permutation = permutation[first:] + sequenceContextMinimumLength
		if(not len(keys)):
			raise ValueError(sequenceContextInvalidIndex)
		starts = np.concatenate((np.asarray([sequenceContextZero], dtype=sequenceContextTokenDtype), (np.flatnonzero(keys[sequenceContextOne:] != keys[:-sequenceContextOne]) + sequenceContextOne).astype(sequenceContextTokenDtype)))
		np.save(folder / sequenceContextKeysFile, keys[starts])
		np.save(folder / sequenceContextOffsetsFile, np.concatenate((starts, np.asarray([len(keys)], dtype=sequenceContextTokenDtype))))
		np.save(folder / sequenceContextPositionsFile, permutation)
		metadata = {sequenceContextSchemaField: sequenceContextSchema, sequenceContextVocabularyField: getSequenceContextVocabularyFingerprint(featureList), sequenceContextFeaturesField: len(featureList), sequenceContextProcessedField: processedSequences, sequenceContextTrainedField: len(boundaries) - sequenceContextOne, sequenceContextTokensField: len(tokens), sequenceContextMinimumField: sequenceContextMinimumLength, sequenceContextBitsField: sequenceContextBits, sequenceContextFilesField: {name: hashSequenceContextFile(folder / name) for name in getSequenceContextFileNames()}}
		# The manifest is the completion marker; partially built models cannot be loaded.
		with (folder / sequenceContextMetadataFile).open(sequenceContextTextWriteMode, encoding=sequenceContextEncoding) as handle:
			json.dump(metadata, handle)
	return


def loadSequenceContextModel(databaseNetworkObject):
	result = None
	if(useSequenceContext):
		folder = Path(sequenceContextFolder)
		with (folder / sequenceContextMetadataFile).open(encoding=sequenceContextEncoding) as handle:
			metadata = json.load(handle)
		expected = {sequenceContextSchemaField: sequenceContextSchema, sequenceContextVocabularyField: getSequenceContextVocabularyFingerprint(databaseNetworkObject.conceptFeaturesList), sequenceContextFeaturesField: databaseNetworkObject.f, sequenceContextProcessedField: trainMaxSequences, sequenceContextMinimumField: sequenceContextMinimumLength, sequenceContextBitsField: sequenceContextBits}
		if(any(metadata.get(key) != value for key, value in expected.items()) or set(metadata[sequenceContextFilesField]) != set(getSequenceContextFileNames())):
			raise RuntimeError(sequenceContextInvalidModel)
		for name in getSequenceContextFileNames():
			if(hashSequenceContextFile(folder / name) != metadata[sequenceContextFilesField][name]):
				raise RuntimeError(sequenceContextInvalidModel)
		tokens = np.memmap(folder / sequenceContextTokensFile, mode=sequenceContextReadMode, dtype=sequenceContextTokenDtype)
		boundaries = np.load(folder / sequenceContextBoundariesFile, mmap_mode=sequenceContextReadMode, allow_pickle=False)
		keys = np.load(folder / sequenceContextKeysFile, mmap_mode=sequenceContextReadMode, allow_pickle=False)
		offsets = np.load(folder / sequenceContextOffsetsFile, mmap_mode=sequenceContextReadMode, allow_pickle=False)
		positions = np.load(folder / sequenceContextPositionsFile, mmap_mode=sequenceContextReadMode, allow_pickle=False)
		validateSequenceContextCorpus(tokens, boundaries, databaseNetworkObject.f)
		if(keys.ndim != sequenceContextOne or keys.dtype != np.dtype(sequenceContextKeyDtype) or not len(keys) or offsets.ndim != sequenceContextOne or offsets.dtype != np.dtype(sequenceContextTokenDtype) or positions.ndim != sequenceContextOne or positions.dtype != np.dtype(sequenceContextTokenDtype) or len(offsets) != len(keys) + sequenceContextOne or offsets[sequenceContextZero] != sequenceContextZero or offsets[-sequenceContextOne] != len(positions) or len(tokens) != metadata[sequenceContextTokensField] or len(boundaries) - sequenceContextOne != metadata[sequenceContextTrainedField]):
			raise RuntimeError(sequenceContextInvalidModel)
		if((keys[sequenceContextOne:] <= keys[:-sequenceContextOne]).any() or (offsets[sequenceContextOne:] <= offsets[:-sequenceContextOne]).any() or positions.min() < sequenceContextMinimumLength or positions.max() >= len(tokens)):
			raise RuntimeError(sequenceContextInvalidModel)
		result = SimpleNamespace(tokens=tokens, boundaries=boundaries, keys=keys, offsets=offsets, positions=positions)
	return result


def validateSequenceContextCorpus(tokens, boundaries, featureCount):
	if(useSequenceContext):
		if(type(featureCount) is not int or not sequenceContextOne < featureCount <= sequenceContextOne << sequenceContextBits or tokens.ndim != sequenceContextOne or tokens.dtype != np.dtype(sequenceContextTokenDtype) or not sequenceContextMinimumLength < len(tokens) < sequenceContextOne << sequenceContextPositionBits or boundaries.ndim != sequenceContextOne or boundaries.dtype != np.dtype(sequenceContextKeyDtype) or len(boundaries) <= sequenceContextOne or boundaries[sequenceContextZero] != sequenceContextZero or boundaries[-sequenceContextOne] != len(tokens)):
			raise ValueError(sequenceContextInvalidIndex)
		if((boundaries[sequenceContextOne:] <= boundaries[:-sequenceContextOne]).any() or tokens.min() <= sequenceContextZero or tokens.max() >= featureCount):
			raise ValueError(sequenceContextInvalidIndex)
	return


def getSequenceContextVocabularyFingerprint(featureList):
	result = None
	if(useSequenceContext):
		result = hashlib.sha256(pickle.dumps(featureList, protocol=sequenceContextPickleProtocol)).hexdigest()
	return result


def getSequenceContextFileNames():
	result = None
	if(useSequenceContext):
		result = (sequenceContextTokensFile, sequenceContextBoundariesFile, sequenceContextKeysFile, sequenceContextOffsetsFile, sequenceContextPositionsFile)
	return result


def hashSequenceContextFile(path):
	result = None
	if(useSequenceContext):
		digest = hashlib.sha256()
		with Path(path).open(sequenceContextReadBinaryMode) as handle:
			while(chunk := handle.read(sequenceContextHashChunkBytes)):
				digest.update(chunk)
		result = digest.hexdigest()
	return result
