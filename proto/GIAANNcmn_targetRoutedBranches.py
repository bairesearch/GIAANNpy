"""GIAANNcmn_targetRoutedBranches.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 BAI Research Pty Ltd (bairesearch.com.au)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN common target Routed Branches

Fixed target-cell branch routing without context selection or extra evidence.

"""


import hashlib
import json
from pathlib import Path
import torch as pt
from GIAANNcmn_globalDefs import *


def initialiseModel(database, inferenceMode):
	if(useTargetRoutedBranches):
		path = Path(databaseFolder)/targetRoutedBranchesModelFile
		if(inferenceMode):
			if(not path.is_file() or json.loads(path.read_text(encoding=targetRoutedBranchesEncoding)) != modelMetadata(database)):
				raise RuntimeError(targetRoutedBranchesInvalidModel)
		elif(path.exists() or Path(conceptColumnsDictFile).exists() or any(Path(observedColumnsDir).iterdir())):
			raise RuntimeError(targetRoutedBranchesInvalidModel)
	return


def saveModel(database, sequenceCount):
	if(useTargetRoutedBranches):
		if(type(sequenceCount) is not int or sequenceCount != trainMaxSequences or debugDisableDatabaseFileWrite):
			raise RuntimeError(targetRoutedBranchesInvalidModel)
		with (Path(databaseFolder)/targetRoutedBranchesModelFile).open(targetRoutedBranchesCreateText,encoding=targetRoutedBranchesEncoding) as output:
			json.dump(modelMetadata(database),output,sort_keys=True)
	return


def activateTraining(sequence, tokens, active, positions, wordOrder, concept, start, end, segments):
	if(useTargetRoutedBranches):
		import GIAANNnlp_sequenceConcepts
		if(type(concept) is not int or not targetRoutedBranchesZero <= concept < sequence.cs or type(start) is not int or type(end) is not int or not targetRoutedBranchesZero <= start < end <= len(tokens) or len(tokens) != sequence.fs or tuple(active.shape) != (multipleDendriticBranchesNumber,arrayNumberOfSegments,sequence.cs,sequence.fs) or positions.shape != wordOrder.shape or tuple(wordOrder.shape) != (sequence.cs,sequence.fs) or segments.dim() != targetRoutedBranchesOne or segments.dtype != pt.long or (segments < targetRoutedBranchesZero).any() or (segments >= arrayNumberOfSegments).any()):
			raise RuntimeError(targetRoutedBranchesInvalidInput)
		features = sequence.featureIndicesInObservedTensor[start:end].to(device=active.device,dtype=pt.long)
		concepts = pt.full_like(features,int(sequence.conceptIndicesInSequenceObservedTensor[concept].item()))
		branches = route(sequence.databaseNetworkObject,concepts,features)
		words = pt.arange(start,end,device=active.device)
		segments = segments.to(active.device)
		active[branches[:,None],segments[None,:],concept,words[:,None]] = targetRoutedBranchesOne
		positions[concept,start:end] = pt.tensor([GIAANNnlp_sequenceConcepts.posStringToPosInt(sequence.databaseNetworkObject.nlp,tokens[position].pos) for position in range(start,end)],dtype=positions.dtype,device=positions.device)
		wordOrder[concept,start:end] = words.to(wordOrder.device)
	return


def collapseConnections(database, connections):
	result = None
	if(useTargetRoutedBranches):
		if(not connections.is_sparse or not connections.is_coalesced() or connections.dim() != targetRoutedBranchesConnectionRank or tuple(connections.shape) != (multipleDendriticBranchesNumber,arrayNumberOfSegments,database.c,database.f)):
			raise RuntimeError(targetRoutedBranchesInvalidInput)
		indices = connections.indices().clone()
		expected = route(database,indices[targetRoutedBranchesColumnAxis],indices[targetRoutedBranchesFeatureAxis])
		if(not pt.equal(indices[targetRoutedBranchesZero],expected)):
			raise RuntimeError(targetRoutedBranchesInvalidInput)
		indices[targetRoutedBranchesZero] = targetRoutedBranchesZero
		result = pt.sparse_coo_tensor(indices,connections.values(),size=(targetRoutedBranchesOne,*connections.shape[targetRoutedBranchesOne:]),dtype=connections.dtype,device=connections.device).coalesce()
	return result


def selectBranch(activation, concept, feature):
	result = None
	if(useTargetRoutedBranches):
		if(not isinstance(activation,pt.Tensor) or activation.dim() != targetRoutedBranchesConnectionRank or activation.shape[targetRoutedBranchesZero] != multipleDendriticBranchesNumber or type(concept) is not int or type(feature) is not int or not targetRoutedBranchesZero <= concept < activation.shape[targetRoutedBranchesColumnAxis] or not targetRoutedBranchesZero <= feature < activation.shape[targetRoutedBranchesFeatureAxis]):
			raise ValueError(targetRoutedBranchesInvalidInput)
		result = (concept*activation.shape[targetRoutedBranchesFeatureAxis]+feature)%multipleDendriticBranchesNumber
	return result


def route(database, concepts, features):
	result = None
	if(useTargetRoutedBranches):
		if(type(database.c) is not int or type(database.f) is not int or database.c <= targetRoutedBranchesZero or database.f <= targetRoutedBranchesZero or database.c*database.f > pt.iinfo(pt.long).max or concepts.dtype != pt.long or features.dtype != pt.long or concepts.dim() != targetRoutedBranchesOne or concepts.shape != features.shape or concepts.device != features.device or (concepts < targetRoutedBranchesZero).any() or (concepts >= database.c).any() or (features < targetRoutedBranchesZero).any() or (features >= database.f).any()):
			raise ValueError(targetRoutedBranchesInvalidInput)
		result = (concepts*database.f+features).remainder(multipleDendriticBranchesNumber)
	return result


def modelMetadata(database):
	result = None
	if(useTargetRoutedBranches):
		vocabulary = hashlib.sha256(json.dumps((database.conceptColumnsList,database.conceptFeaturesList)).encode(targetRoutedBranchesEncoding)).hexdigest()
		result = {targetRoutedBranchesSchemaKey:targetRoutedBranchesSchema,targetRoutedBranchesCountKey:multipleDendriticBranchesNumber,targetRoutedBranchesVocabularyKey:vocabulary,targetRoutedBranchesSequencesKey:trainMaxSequences}
	return result
