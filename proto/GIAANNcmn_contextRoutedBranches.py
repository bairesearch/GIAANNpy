"""GIAANNcmn_contextRoutedBranches.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 BAI Research Pty Ltd (bairesearch.com.au)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN common context Routed Branches

Experimental ordered-prefix routing of native dendritic branch connections.

"""


import hashlib
import json
import random
from pathlib import Path
import torch as pt
from GIAANNcmn_globalDefs import *


def initialiseModel(database, inferenceMode):
	if(useContextRoutedBranches):
		path = Path(databaseFolder)/contextRoutedBranchesModelFile
		if(inferenceMode):
			if(not path.is_file() or json.loads(path.read_text(encoding=contextRoutedBranchesEncoding)) != modelMetadata(database)):
				raise RuntimeError(contextRoutedBranchesInvalidModel)
		elif(path.exists() or Path(conceptColumnsDictFile).exists() or any(Path(observedColumnsDir).iterdir())):
			raise RuntimeError(contextRoutedBranchesInvalidModel)
	return


def saveModel(database, sequenceCount):
	if(useContextRoutedBranches):
		path = Path(databaseFolder)/contextRoutedBranchesModelFile
		if(type(sequenceCount) is not int or sequenceCount != trainMaxSequences or debugDisableDatabaseFileWrite or path.exists()):
			raise RuntimeError(contextRoutedBranchesInvalidModel)
		path.write_text(json.dumps(modelMetadata(database), sort_keys=True), encoding=contextRoutedBranchesEncoding)
	return


def prepareTraining(sequence, tokens, starts, ends):
	if(useContextRoutedBranches):
		if(len(tokens) != sequence.fs or starts.shape != ends.shape or starts.dim() != contextRoutedBranchesOne or starts.numel() != sequence.cs):
			raise RuntimeError(contextRoutedBranchesInvalidInput)
		if(len(sequence.conceptIndicesInSequenceObservedTensor) != sequence.cs or len(sequence.featureIndicesInObservedTensor) != sequence.fs):
			raise RuntimeError(contextRoutedBranchesInvalidInput)
		identities = tokenIdentities(tokens)
		branches = pt.full((len(tokens),), -contextRoutedBranchesOne, dtype=pt.long)
		for start,end in zip(starts.tolist(),ends.tolist()):
			if(not contextRoutedBranchesZero <= start < end <= len(tokens)):
				raise RuntimeError(contextRoutedBranchesInvalidInput)
			for position in range(start,end):
				prefix = tuple(identities[max(contextRoutedBranchesZero,position-contextRoutedBranchesLength):position])
				if(contextRoutedBranchesRandom):
					branch = random.randrange(contextRoutedBranchesNumber) + int(contextRoutedBranchesShared)
				else:
					branch = route(prefix)
				branches[position] = branch
		sequence.contextRoutedBranchesTraining = branches
	return


def activateTraining(sequence, tokens, active, positions, wordOrder, concept, start, end, segments):
	if(useContextRoutedBranches):
		import GIAANNnlp_sequenceConcepts
		if(not hasattr(sequence, contextRoutedBranchesTrainingAttribute) or not contextRoutedBranchesZero <= start < end <= len(tokens) or segments.dim() != contextRoutedBranchesOne):
			raise RuntimeError(contextRoutedBranchesInvalidInput)
		branches = sequence.contextRoutedBranchesTraining[start:end].to(active.device)
		if(bool(((branches<contextRoutedBranchesZero)|(branches>=multipleDendriticBranchesNumber)).any().item())):
			raise RuntimeError(contextRoutedBranchesInvalidInput)
		words = pt.arange(start,end,device=active.device)
		segments = segments.to(active.device)
		active[branches[:,None],segments[None,:],concept,words[:,None]] = contextRoutedBranchesOne
		if(contextRoutedBranchesShared):
			active[contextRoutedBranchesZero,segments[None,:],concept,words[:,None]] = contextRoutedBranchesOne
		positions[concept,start:end] = pt.tensor([GIAANNnlp_sequenceConcepts.posStringToPosInt(sequence.databaseNetworkObject.nlp,tokens[position].pos) for position in range(start,end)], dtype=positions.dtype, device=positions.device)
		wordOrder[concept,start:end] = words.to(wordOrder.device)
	return


def filterTrainingConnections(connections):
	result = connections
	if(useContextRoutedBranches):
		if(not connections.is_sparse or connections.dim() != contextRoutedBranchesTrainingConnectionRank or not connections.is_coalesced() or connections.shape[contextRoutedBranchesZero] != multipleDendriticBranchesNumber):
			raise RuntimeError(contextRoutedBranchesInvalidInput)
		if(contextRoutedBranchesShared and not trainConnectionsAllowSelfTransitions):
			indices = connections.indices()
			# Local feature positions identify occurrences, so repeated words at different positions remain connected.
			mask = (indices[contextRoutedBranchesSourceConceptAxis] != indices[contextRoutedBranchesTargetConceptAxis]) | (indices[contextRoutedBranchesSourceFeatureAxis] != indices[contextRoutedBranchesTargetFeatureAxis])
			result = pt.sparse_coo_tensor(indices[:,mask], connections.values()[mask], size=connections.shape, dtype=connections.dtype, device=connections.device).coalesce()
	return result


def prepareInference(database, prefix):
	if(useContextRoutedBranches):
		database.contextRoutedBranchesInference = route(tokenIdentities(prefix[-contextRoutedBranchesLength:]))
		if(inferenceContextRoutedBranchesCounts):
			import GIAANNcmn_contextRoutedBranchesCounts
			GIAANNcmn_contextRoutedBranchesCounts.prepareInference(database)
	return


def reduceActivations(database, signals):
	result = None
	if(useContextRoutedBranches):
		if(not signals.is_sparse or not signals.is_coalesced() or signals.dim() != inferenceLeakyIntegrateAndFireNeuronTensorRank-contextRoutedBranchesOne or signals.shape[contextRoutedBranchesZero] != multipleDendriticBranchesNumber or not hasattr(database, contextRoutedBranchesInferenceAttribute)):
			raise RuntimeError(contextRoutedBranchesInvalidInput)
		if(inferenceContextRoutedBranchesGating):
			indices,values = signals.indices(),signals.values()
			branches = indices[contextRoutedBranchesZero]
			selected = selectBranches(database, indices)
			if(contextRoutedBranchesShared):
				weights = (branches == contextRoutedBranchesZero).to(values.dtype) + selected.to(values.dtype)*contextRoutedBranchesGain
			else:
				weights = selected.to(values.dtype)
			mask = weights > contextRoutedBranchesZero
			result = pt.sparse_coo_tensor(indices[contextRoutedBranchesOne:,mask], values[mask]*weights[mask], size=signals.shape[contextRoutedBranchesOne:], dtype=signals.dtype, device=signals.device).coalesce()
		else:
			import GIAANNcmn_sparseTensors
			result = GIAANNcmn_sparseTensors.reduceSparseBranchMax(signals)
	return result


def selectBranches(database, indices):
	result = None
	if(useContextRoutedBranches):
		branches = indices[contextRoutedBranchesZero]
		result = branches == database.contextRoutedBranchesInference
	return result


def modelMetadata(database):
	result = None
	if(useContextRoutedBranches):
		vocabulary = hashlib.sha256(json.dumps(database.conceptFeaturesList).encode(contextRoutedBranchesEncoding)).hexdigest()
		result = {contextRoutedBranchesSchemaKey:contextRoutedBranchesSchema, contextRoutedBranchesLengthKey:contextRoutedBranchesLength, contextRoutedBranchesCountKey:multipleDendriticBranchesNumber, contextRoutedBranchesSharedKey:contextRoutedBranchesShared, contextRoutedBranchesRandomKey:contextRoutedBranchesRandom, contextRoutedBranchesVocabularyKey:vocabulary, contextRoutedBranchesSequencesKey:trainMaxSequences}
		if(contextRoutedBranchesShared):
			result[contextRoutedBranchesSelfFilterKey] = True
	return result


def sharedConnections(connections):
	result = None
	if(useContextRoutedBranches):
		if(not contextRoutedBranchesShared or not connections.is_sparse or connections.dim() != inferenceLeakyIntegrateAndFireNeuronTensorRank or connections.shape[contextRoutedBranchesZero] != multipleDendriticBranchesNumber):
			raise RuntimeError(contextRoutedBranchesInvalidInput)
		connections = connections.coalesce()
		mask = connections.indices()[contextRoutedBranchesZero] == contextRoutedBranchesZero
		result = pt.sparse_coo_tensor(connections.indices()[:,mask], connections.values()[mask], size=(contextRoutedBranchesOne,*connections.shape[contextRoutedBranchesOne:]), dtype=connections.dtype, device=connections.device).coalesce()
	return result


def tokenIdentities(tokens):
	result = None
	if(useContextRoutedBranches):
		result = [token.tokenId for token in tokens]
		if(any(type(token) is not int or token < contextRoutedBranchesZero or token >= contextRoutedBranchesOne << (contextRoutedBranchesTokenBytes*contextRoutedBranchesBitsPerByte) for token in result)):
			raise ValueError(contextRoutedBranchesInvalidInput)
	return result


def route(prefix):
	result = None
	if(useContextRoutedBranches):
		if(len(prefix)>contextRoutedBranchesLength or any(type(token) is not int or token<contextRoutedBranchesZero or token>=contextRoutedBranchesOne<<(contextRoutedBranchesTokenBytes*contextRoutedBranchesBitsPerByte) for token in prefix)):
			raise ValueError(contextRoutedBranchesInvalidInput)
		payload = contextRoutedBranchesEmptyPayload.join(token.to_bytes(contextRoutedBranchesTokenBytes, contextRoutedBranchesByteOrder) for token in prefix)
		digest = hashlib.blake2b(payload, digest_size=contextRoutedBranchesDigestBytes).digest()
		result = int.from_bytes(digest, contextRoutedBranchesByteOrder)%contextRoutedBranchesNumber + int(contextRoutedBranchesShared)
	return result
