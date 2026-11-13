"""GIAANNcmn_contextRoutedBranchesSupport.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 BAI Research Pty Ltd (bairesearch.com.au)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN common context Routed Branches Support

Training-only context support, fixed branch allocation and confidence weighting.

"""


import hashlib
import json
import pickle
import struct
from pathlib import Path
import torch as pt
from GIAANNcmn_globalDefs import *


def initialise(database, inferenceMode):
	if(useContextRoutedBranchesSupport):
		if(inferenceMode):
			with (Path(databaseFolder)/contextRoutedBranchesSupportFile).open(contextRoutedBranchesReadBinary) as source:
				payload = pickle.load(source)
			if(payload[contextRoutedBranchesSupportMetadataKey] != planMetadata(database) or payload[contextRoutedBranchesSupportFrequencyKey] != contextRoutedBranchesSupportFrequency):
				raise RuntimeError(contextRoutedBranchesSupportInvalid)
			database.contextRoutedBranchesSupportExpected = payload[contextRoutedBranchesSupportCountsKey]
			validateAllocatedSupport(database)
		else:
			database.contextRoutedBranchesSupportDigest = hashlib.sha256()
			database.contextRoutedBranchesSupportSequences = contextRoutedBranchesZero
			database.contextRoutedBranchesSupportTokens = contextRoutedBranchesZero
			if(trainContextRoutedBranchesSupportCollect):
				if(Path(contextRoutedBranchesSupportPlanFile).exists()):
					raise RuntimeError(contextRoutedBranchesSupportInvalid)
				database.contextRoutedBranchesSupportCounts = {}
			else:
				with Path(contextRoutedBranchesSupportPlanFile).open(contextRoutedBranchesReadBinary) as source:
					plan = pickle.load(source)
				allocate(database, plan)
	return


def allocate(database, plan):
	if(useContextRoutedBranchesSupport):
		if(plan[contextRoutedBranchesSupportMetadataKey] != planMetadata(database) or not plan[contextRoutedBranchesSupportCountsKey] or type(plan[contextRoutedBranchesSupportSequencesKey]) is not int or not contextRoutedBranchesZero < plan[contextRoutedBranchesSupportSequencesKey] <= trainMaxSequences or type(plan[contextRoutedBranchesSupportTokensKey]) is not int or plan[contextRoutedBranchesSupportTokensKey] <= contextRoutedBranchesZero):
			raise RuntimeError(contextRoutedBranchesSupportInvalid)
		database.contextRoutedBranchesPrototypes = {}
		database.contextRoutedBranchesSupportExpected = {}
		total = contextRoutedBranchesZero
		for key,counts in plan[contextRoutedBranchesSupportCountsKey].items():
			if(type(key) is not int or not contextRoutedBranchesZero <= key < database.c*database.f or type(counts) is not dict or not counts):
				raise RuntimeError(contextRoutedBranchesSupportInvalid)
			for prefix,count in counts.items():
				validatePrefixCount(prefix, count)
				total += count
			# Stable ties retain training encounter order, isolating the effect of support ranking.
			prefixes = sorted(counts, key=lambda prefix:-counts[prefix]) if contextRoutedBranchesSupportFrequency else list(counts)
			prefixes = prefixes[:contextRoutedBranchesNumber]
			database.contextRoutedBranchesPrototypes[key] = {prefix:index+contextRoutedBranchesOne for index,prefix in enumerate(prefixes)}
			database.contextRoutedBranchesSupportExpected[key] = tuple(counts[prefix] for prefix in prefixes)
		if(total != plan[contextRoutedBranchesSupportTokensKey]):
			raise RuntimeError(contextRoutedBranchesSupportInvalid)
		database.contextRoutedBranchesSupportPlanState = {key:plan[key] for key in (contextRoutedBranchesSupportStreamKey,contextRoutedBranchesSupportSequencesKey,contextRoutedBranchesSupportTokensKey)}
		database.contextRoutedBranchesSupportSeen = {key:[contextRoutedBranchesZero]*len(counts) for key,counts in database.contextRoutedBranchesSupportExpected.items()}
		validateAllocatedSupport(database)
	return


def recordTraining(sequence, tokens, starts, ends, sequenceIndex):
	if(useContextRoutedBranchesSupport):
		import GIAANNcmn_contextRoutedBranches
		if(type(sequenceIndex) is not int or not contextRoutedBranchesZero <= sequenceIndex < trainMaxSequences or len(tokens) != sequence.fs or starts.shape != ends.shape or starts.dim() != contextRoutedBranchesOne or starts.numel() != sequence.cs):
			raise RuntimeError(contextRoutedBranchesSupportInvalid)
		database = sequence.databaseNetworkObject
		identities = GIAANNcmn_contextRoutedBranches.tokenIdentities(tokens)
		concepts,features = sequence.conceptIndicesInSequenceObservedTensor.tolist(),sequence.featureIndicesInObservedTensor.tolist()
		if(len(concepts) != sequence.cs or len(features) != sequence.fs):
			raise RuntimeError(contextRoutedBranchesSupportInvalid)
		payload = pickle.dumps((sequenceIndex,identities,concepts,features,starts.tolist(),ends.tolist()),protocol=contextRoutedBranchesSupportPickleProtocol)
		database.contextRoutedBranchesSupportDigest.update(struct.pack(contextRoutedBranchesSupportFrameFormat,len(payload)))
		database.contextRoutedBranchesSupportDigest.update(payload)
		database.contextRoutedBranchesSupportSequences += contextRoutedBranchesOne
		for concept,start,end in zip(concepts,starts.tolist(),ends.tolist()):
			if(not contextRoutedBranchesZero <= concept < database.c or not contextRoutedBranchesZero <= start < end <= len(tokens)):
				raise RuntimeError(contextRoutedBranchesSupportInvalid)
			database.contextRoutedBranchesSupportTokens += end-start
			if(trainContextRoutedBranchesSupportCollect):
				for position in range(start,end):
					if(not contextRoutedBranchesZero <= features[position] < database.f):
						raise RuntimeError(contextRoutedBranchesSupportInvalid)
					key = concept*database.f+features[position]
					prefix = tuple(identities[max(contextRoutedBranchesZero,position-contextRoutedBranchesLength):position])
					counts = database.contextRoutedBranchesSupportCounts.setdefault(key,{})
					counts[prefix] = counts.get(prefix,contextRoutedBranchesZero)+contextRoutedBranchesOne
	return


def trainingBranch(database, key, prefix):
	result = None
	if(useContextRoutedBranchesSupport):
		if(trainContextRoutedBranchesSupportCollect or key not in database.contextRoutedBranchesPrototypes):
			raise RuntimeError(contextRoutedBranchesSupportInvalid)
		result = database.contextRoutedBranchesPrototypes[key].get(prefix,contextRoutedBranchesZero)
		if(result > contextRoutedBranchesZero):
			database.contextRoutedBranchesSupportSeen[key][result-contextRoutedBranchesOne] += contextRoutedBranchesOne
			if(database.contextRoutedBranchesSupportSeen[key][result-contextRoutedBranchesOne] > database.contextRoutedBranchesSupportExpected[key][result-contextRoutedBranchesOne]):
				raise RuntimeError(contextRoutedBranchesSupportInvalid)
	return result


def confidence(database, indices, selected):
	result = None
	if(useContextRoutedBranchesSupport and inferenceContextRoutedBranchesSupportConfidence):
		if(indices.dim() != contextRoutedBranchesOne+contextRoutedBranchesOne or indices.shape[contextRoutedBranchesZero] != contextRoutedBranchesOne+contextRoutedBranchesOne+contextRoutedBranchesOne or selected.shape != indices.shape[contextRoutedBranchesOne:] or selected.dtype != pt.bool):
			raise RuntimeError(contextRoutedBranchesSupportInvalid)
		chosen = indices[:,selected]
		if(indices.dtype != pt.long or (chosen[contextRoutedBranchesZero] <= contextRoutedBranchesZero).any() or (chosen[contextRoutedBranchesZero] >= multipleDendriticBranchesNumber).any() or (chosen[contextRoutedBranchesOne] < contextRoutedBranchesZero).any() or (chosen[contextRoutedBranchesOne] >= database.c).any() or (chosen[-contextRoutedBranchesOne] < contextRoutedBranchesZero).any() or (chosen[-contextRoutedBranchesOne] >= database.f).any()):
			raise RuntimeError(contextRoutedBranchesSupportInvalid)
		keys = (chosen[contextRoutedBranchesOne]*database.f+chosen[-contextRoutedBranchesOne]).tolist()
		counts = pt.tensor([database.contextRoutedBranchesSupportExpected[key][branch-contextRoutedBranchesOne] for key,branch in zip(keys,chosen[contextRoutedBranchesZero].tolist())],dtype=pt.float64,device=indices.device)
		if(not pt.isfinite(counts).all() or (counts < contextRoutedBranchesOne).any()):
			raise RuntimeError(contextRoutedBranchesSupportInvalid)
		result = pt.zeros(selected.shape,dtype=pt.float64,device=indices.device)
		result[selected] = (counts >= contextRoutedBranchesSupportMinimum).to(counts.dtype)*counts/(counts+contextRoutedBranchesSupportTau)
	return result


def saveCollection(database, sequenceCount):
	if(useContextRoutedBranchesSupport):
		if(not trainContextRoutedBranchesSupportCollect):
			raise RuntimeError(contextRoutedBranchesSupportInvalid)
		verifyTraining(database,sequenceCount)
		payload = {contextRoutedBranchesSupportMetadataKey:planMetadata(database),contextRoutedBranchesSupportCountsKey:database.contextRoutedBranchesSupportCounts,**streamState(database)}
		with Path(contextRoutedBranchesSupportPlanFile).open(contextRoutedBranchesCreateBinary) as output:
			pickle.dump(payload,output,protocol=contextRoutedBranchesSupportPickleProtocol)
	return


def saveTraining(database, sequenceCount):
	if(useContextRoutedBranchesSupport):
		if(trainContextRoutedBranchesSupportCollect):
			raise RuntimeError(contextRoutedBranchesSupportInvalid)
		verifyTraining(database,sequenceCount)
		payload = {contextRoutedBranchesSupportMetadataKey:planMetadata(database),contextRoutedBranchesSupportCountsKey:database.contextRoutedBranchesSupportExpected,contextRoutedBranchesSupportFrequencyKey:contextRoutedBranchesSupportFrequency,**streamState(database)}
		with (Path(databaseFolder)/contextRoutedBranchesSupportFile).open(contextRoutedBranchesCreateBinary) as output:
			pickle.dump(payload,output,protocol=contextRoutedBranchesSupportPickleProtocol)
	return


def verifyTraining(database, sequenceCount):
	if(useContextRoutedBranchesSupport):
		if(type(sequenceCount) is not int or sequenceCount != trainMaxSequences or database.contextRoutedBranchesSupportSequences <= contextRoutedBranchesZero or database.contextRoutedBranchesSupportTokens <= contextRoutedBranchesZero):
			raise RuntimeError(contextRoutedBranchesSupportInvalid)
		if(not trainContextRoutedBranchesSupportCollect):
			if(streamState(database) != database.contextRoutedBranchesSupportPlanState or any(tuple(database.contextRoutedBranchesSupportSeen[key]) != counts for key,counts in database.contextRoutedBranchesSupportExpected.items())):
				raise RuntimeError(contextRoutedBranchesSupportInvalid)
	return


def modelMetadata():
	result = None
	if(useContextRoutedBranchesSupport):
		digest = hashlib.sha256()
		with (Path(databaseFolder)/contextRoutedBranchesSupportFile).open(contextRoutedBranchesReadBinary) as source:
			while(block := source.read(contextRoutedBranchesHashBlockBytes)):
				digest.update(block)
		result = {contextRoutedBranchesSupportEnabledKey:True,contextRoutedBranchesSupportFrequencyKey:contextRoutedBranchesSupportFrequency,contextRoutedBranchesSupportHashKey:digest.hexdigest()}
	return result


def planMetadata(database):
	result = None
	if(useContextRoutedBranchesSupport):
		vocabulary = hashlib.sha256(json.dumps(database.conceptFeaturesList).encode(contextRoutedBranchesEncoding)).hexdigest()
		result = {contextRoutedBranchesSupportSchemaKey:contextRoutedBranchesSupportSchema,contextRoutedBranchesLengthKey:contextRoutedBranchesLength,contextRoutedBranchesCountKey:multipleDendriticBranchesNumber,contextRoutedBranchesVocabularyKey:vocabulary,contextRoutedBranchesSequencesKey:trainMaxSequences}
	return result


def streamState(database):
	result = None
	if(useContextRoutedBranchesSupport):
		result = {contextRoutedBranchesSupportStreamKey:database.contextRoutedBranchesSupportDigest.hexdigest(),contextRoutedBranchesSupportSequencesKey:database.contextRoutedBranchesSupportSequences,contextRoutedBranchesSupportTokensKey:database.contextRoutedBranchesSupportTokens}
	return result


def validateAllocatedSupport(database):
	if(useContextRoutedBranchesSupport):
		if(database.contextRoutedBranchesPrototypes.keys() != database.contextRoutedBranchesSupportExpected.keys()):
			raise RuntimeError(contextRoutedBranchesSupportInvalid)
		for key,patterns in database.contextRoutedBranchesPrototypes.items():
			counts = database.contextRoutedBranchesSupportExpected[key]
			if(type(key) is not int or not contextRoutedBranchesZero <= key < database.c*database.f or not contextRoutedBranchesZero < len(patterns) <= contextRoutedBranchesNumber or len(patterns) != len(counts) or sorted(patterns.values()) != list(range(contextRoutedBranchesOne,len(patterns)+contextRoutedBranchesOne))):
				raise RuntimeError(contextRoutedBranchesSupportInvalid)
			for prefix,branch in patterns.items():
				validatePrefixCount(prefix,counts[branch-contextRoutedBranchesOne])
	return


def validatePrefixCount(prefix, count):
	if(useContextRoutedBranchesSupport):
		if(type(prefix) is not tuple or len(prefix) > contextRoutedBranchesLength or any(type(token) is not int or not contextRoutedBranchesZero <= token < contextRoutedBranchesOne << (contextRoutedBranchesTokenBytes*contextRoutedBranchesBitsPerByte) for token in prefix) or type(count) is not int or count < contextRoutedBranchesOne):
			raise RuntimeError(contextRoutedBranchesSupportInvalid)
	return
