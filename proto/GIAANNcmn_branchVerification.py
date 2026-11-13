"""GIAANNcmn_branchVerification.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 BAI Research Pty Ltd (bairesearch.com.au)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN common branch Verification

Cumulative connection ownership and exact branch-verification indexes.

"""


import os
import torch as pt
from GIAANNcmn_globalDefs import *


def initialise(database, loaded):
	if(trainVerifyConnectionNonexistentAcrossBranchesCumulative):
		database.branchVerifyTargets = pt.empty((branchVerifyZero,), dtype=pt.long)
		database.branchVerifyParents = pt.empty((branchVerifyZero, multipleDendriticBranchesNumber), dtype=pt.long)
		database.branchVerifyRevision = branchVerifyZero
		if(loaded):
			path = os.path.join(databaseFolder, branchVerifyStateFileName)
			if(not os.path.isfile(path)):
				raise RuntimeError(branchVerifyInvalidModel)
			state = pt.load(path, map_location=branchVerifyCPU, weights_only=True)
			if(state[branchVerifySchemaKey] != branchVerifySchema or tuple(state[branchVerifyShapeKey]) != (database.c, database.f, multipleDendriticBranchesNumber, arrayNumberOfSegments)):
				raise RuntimeError(branchVerifyInvalidModel)
			database.branchVerifyTargets = state[branchVerifyTargetsKey]
			database.branchVerifyParents = state[branchVerifyParentsKey]
			validateParents(database.branchVerifyTargets, database.branchVerifyParents, database.c*database.f)
	return


def prepareTraining(sequence, active, cs, fs, columnsOrder, wordsOrder):
	result = None
	if(trainVerifyConnectionNonexistentAcrossBranchesCumulative):
		import GIAANNcmn_databaseNetworkTrain as training
		database = sequence.databaseNetworkObject
		includeSameTime = training.getTrainConnectionsIncludeSameTimeIndex(sequence)
		if(training.getTrainConnectionsUseSpatialAxes(sequence)):
			raise RuntimeError(branchVerifyInvalidConfiguration)
		prospective = training.createFeatureConnectionsActiveTrainSparse(active, cs, fs, columnsOrder, wordsOrder, includeSameTime, sequence).coalesce()
		indices = prospective.indices()
		concepts = sequence.conceptIndicesInSequenceObservedTensor
		features = sequence.featureIndicesInObservedTensor
		localTargets = pt.nonzero(active.amax(dim=(branchVerifyZero, branchVerifyOne)).reshape(-branchVerifyOne) > branchVerifyZero, as_tuple=False).flatten()
		globalTargets = concepts[pt.div(localTargets, fs, rounding_mode=branchVerifyRounding)]*database.f + features[localTargets % fs]
		originalBranches = active.amax(dim=branchVerifyOne).reshape(multipleDendriticBranchesNumber, -branchVerifyOne)[:, localTargets].argmax(dim=branchVerifyZero)
		targetLookup = pt.full((cs*fs,), branchVerifyInvalid, dtype=pt.long)
		targetLookup[localTargets] = pt.arange(localTargets.numel(), dtype=pt.long)
		targetPositions = targetLookup[indices[branchVerifyProspectiveTargetConceptAxis]*fs + indices[branchVerifyProspectiveTargetFeatureAxis]]
		sourceKeys = concepts[indices[branchVerifyProspectiveSourceConceptAxis]]*database.f + features[indices[branchVerifyProspectiveSourceFeatureAxis]]
		connectionKeys = indices[branchVerifyProspectiveSegmentAxis]*(database.c*database.f) + globalTargets[targetPositions]
		existingBranches = lookupConnections(sequence, sourceKeys, connectionKeys)
		selected = selectBranches(database, globalTargets, originalBranches, sourceKeys, connectionKeys, targetPositions, existingBranches)
		activation = active.sum(dim=branchVerifyZero).reshape(arrayNumberOfSegments, -branchVerifyOne)[:, localTargets].clone()
		if(bool(pt.any(activation > branchVerifyOne))):
			raise RuntimeError(branchVerifyInvalidInput)
		active.zero_()
		active.view(multipleDendriticBranchesNumber, arrayNumberOfSegments, -branchVerifyOne)[selected, :, localTargets] = activation.transpose(branchVerifyZero, branchVerifyOne)
		mapped = indices.clone()
		mapped[branchVerifyProspectiveBranchAxis] = selected[targetPositions]
		result = pt.sparse_coo_tensor(mapped, prospective.values(), prospective.size(), dtype=prospective.dtype).coalesce()
		# Existing neuron evidence must follow the same merge as incoming connections.
		for column in sequence.getObservedColumnsByConceptIndex(sequence.observedColumnsDict).values():
			canonicaliseColumnNeurons(column)
	return result


def selectBranches(database, targets, originalBranches, sourceKeys, connectionKeys, positions, existingBranches):
	result = None
	if(trainVerifyConnectionNonexistentAcrossBranchesCumulative):
		count = targets.numel()
		if(count == branchVerifyZero or positions.numel() != connectionKeys.numel() or existingBranches.numel() != positions.numel()):
			raise RuntimeError(branchVerifyInvalidInput)
		if(any(tensor.dtype != pt.long or tensor.dim() != branchVerifyOne or tensor.device.type != branchVerifyCPU for tensor in (targets, originalBranches, sourceKeys, connectionKeys, positions, existingBranches)) or originalBranches.numel() != count or sourceKeys.numel() != positions.numel()):
			raise RuntimeError(branchVerifyInvalidInput)
		if(bool(pt.any(targets < branchVerifyZero)) or bool(pt.any(targets >= database.c*database.f)) or bool(pt.any(originalBranches < branchVerifyZero)) or bool(pt.any(originalBranches >= multipleDendriticBranchesNumber)) or bool(pt.any(sourceKeys < branchVerifyZero)) or bool(pt.any(sourceKeys >= database.c*database.f)) or bool(pt.any(connectionKeys < branchVerifyZero)) or bool(pt.any(connectionKeys >= arrayNumberOfSegments*database.c*database.f)) or bool(pt.any(positions < branchVerifyZero)) or bool(pt.any(positions >= count)) or bool(pt.any(existingBranches < branchVerifyInvalid)) or bool(pt.any(existingBranches >= multipleDendriticBranchesNumber))):
			raise RuntimeError(branchVerifyInvalidInput)
		if(not pt.equal(connectionKeys % (database.c*database.f), targets[positions])):
			raise RuntimeError(branchVerifyInvalidInput)
		uniqueTargets, targetGroups = pt.unique(targets, sorted=True, return_inverse=True)
		parents = getParents(database, uniqueTargets)
		originalBranches = parents[targetGroups, originalBranches]
		valid = existingBranches >= branchVerifyZero
		existingBranches = existingBranches.clone()
		existingBranches[valid] = parents[targetGroups[positions[valid]], existingBranches[valid]]
		# Occurrences sharing a new connection must also agree before this update commits.
		_, sourceGroups = pt.unique(sourceKeys, sorted=True, return_inverse=True)
		stride = arrayNumberOfSegments*database.c*database.f
		if(stride*max(branchVerifyOne, int(sourceGroups.max().item())+branchVerifyOne if sourceGroups.numel() else branchVerifyOne) > branchVerifyInt64Maximum):
			raise RuntimeError(branchVerifyInvalidInput)
		_, keyGroups = pt.unique(sourceGroups*stride + connectionKeys, sorted=True, return_inverse=True)
		labels = pt.arange(count, dtype=pt.long)
		changed = True
		iterations = branchVerifyZero
		while(changed):
			keyLabels = pt.full((int(keyGroups.max().item())+branchVerifyOne if keyGroups.numel() else branchVerifyZero,), count, dtype=pt.long)
			keyLabels.scatter_reduce_(branchVerifyZero, keyGroups, labels[positions], reduce=branchVerifyReduction, include_self=True)
			nextLabels = labels.clone()
			nextLabels.scatter_reduce_(branchVerifyZero, positions, keyLabels[keyGroups], reduce=branchVerifyReduction, include_self=True)
			nextLabels = nextLabels[nextLabels]
			changed = not pt.equal(labels, nextLabels)
			labels = nextLabels
			iterations += branchVerifyOne
			if(iterations > count+branchVerifyOne):
				raise RuntimeError(branchVerifyInvalidInput)
		owners = pt.full((count,), multipleDendriticBranchesNumber, dtype=pt.long)
		owners.scatter_reduce_(branchVerifyZero, labels[positions[valid]], existingBranches[valid], reduce=branchVerifyReduction, include_self=True)
		chosen = pt.where(owners[labels] < multipleDendriticBranchesNumber, owners[labels], originalBranches[labels])
		# Union only already-trained branches; unused random proposals do not force merges.
		left = targetGroups[positions[valid]]*multipleDendriticBranchesNumber + existingBranches[valid]
		right = targetGroups[positions[valid]]*multipleDendriticBranchesNumber + chosen[positions[valid]]
		branchLabels = (pt.arange(uniqueTargets.numel(), dtype=pt.long).unsqueeze(branchVerifyOne)*multipleDendriticBranchesNumber + parents).reshape(-branchVerifyOne)
		changed = True
		iterations = branchVerifyZero
		while(changed):
			nextLabels = branchLabels.clone()
			edgeLabels = pt.minimum(branchLabels[left], branchLabels[right])
			nextLabels.scatter_reduce_(branchVerifyZero, left, edgeLabels, reduce=branchVerifyReduction, include_self=True)
			nextLabels.scatter_reduce_(branchVerifyZero, right, edgeLabels, reduce=branchVerifyReduction, include_self=True)
			nextLabels = nextLabels[nextLabels]
			changed = not pt.equal(branchLabels, nextLabels)
			branchLabels = nextLabels
			iterations += branchVerifyOne
			if(iterations > multipleDendriticBranchesNumber+branchVerifyOne):
				raise RuntimeError(branchVerifyInvalidInput)
		merged = branchLabels.reshape(-branchVerifyOne, multipleDendriticBranchesNumber) % multipleDendriticBranchesNumber
		if(not pt.equal(parents, merged)):
			changedTargets = (parents != merged).any(dim=branchVerifyOne)
			storeParents(database, uniqueTargets[changedTargets], merged[changedTargets])
		result = merged[targetGroups, chosen]
	return result


def lookupConnections(sequence, sourceKeys, connectionKeys):
	result = None
	if(trainVerifyConnectionNonexistentAcrossBranchesCumulative):
		database = sequence.databaseNetworkObject
		result = pt.full_like(connectionKeys, branchVerifyInvalid)
		columns = sequence.getObservedColumnsByConceptIndex(sequence.observedColumnsDict)
		uniqueSources, counts = pt.unique(sourceKeys, sorted=True, return_counts=True)
		order = pt.argsort(sourceKeys, stable=True)
		start = branchVerifyZero
		for source, count in zip(uniqueSources.tolist(), counts.tolist()):
			concept, feature = divmod(source, database.f)
			column = columns[concept]
			tensor = column.getFeatureConnectionsForSourceFeature(feature, targetDevice=branchVerifyCPU, createMissing=False).coalesce()
			keys, branches = ownershipIndex(column, feature, tensor, database.arrayIndexPropertiesStrengthIndex)
			selection = order[start:start+count]
			locations, found = lookup(keys, connectionKeys[selection])
			result[selection[found]] = branches[locations[found]]
			start += count
	return result


def filterOptimisedUpdates(references, sources, updates, strengthIndex):
	result = updates
	if(trainVerifyConnectionNonexistentAcrossBranchesOptimisation1):
		if(not trainVerifyConnectionNonexistentAcrossBranchesCumulative):
			indices, values = updates.indices(), updates.values()
			keep = pt.ones_like(values, dtype=pt.bool)
			for bucket, ((column, feature), tensor) in enumerate(zip(references, sources)):
				keys, branches = ownershipIndex(column, feature, tensor, strengthIndex)
				selection = pt.nonzero(indices[branchVerifyTargetConceptAxis] == bucket, as_tuple=False).flatten()
				query = (indices[branchVerifySegmentAxis, selection]*tensor.size(branchVerifyTargetConceptAxis) + indices[branchVerifyTargetFeatureAxis, selection])*tensor.size(branchVerifyTargetFeatureAxis) + indices[branchVerifyProspectiveTargetFeatureAxis, selection]
				locations, found = lookup(keys, query)
				keep[selection] = ~found
			result = pt.sparse_coo_tensor(indices[:, keep], values[keep], updates.size(), dtype=updates.dtype).coalesce()
	return result


def ownershipIndex(column, feature, tensor, strengthIndex):
	result = None
	if(trainVerifyConnectionNonexistentAcrossBranchesCumulative or trainVerifyConnectionNonexistentAcrossBranchesOptimisation1):
		if(not pt.is_tensor(tensor) or tensor.layout != pt.sparse_coo or tensor.dim() != branchVerifySourceRank or not tensor.is_coalesced() or tensor.device.type != branchVerifyCPU or type(strengthIndex) is not int or strengthIndex < branchVerifyZero or strengthIndex >= tensor.size(branchVerifyPropertyAxis) or type(feature) is not int or feature < branchVerifyZero or feature >= tensor.size(branchVerifyTargetFeatureAxis)):
			raise RuntimeError(branchVerifyInvalidInput)
		indices, values = tensor.indices(), tensor.values()
		mask = (indices[branchVerifyPropertyAxis] == strengthIndex) & (values > branchVerifyZero)
		cached = None
		if(trainVerifyConnectionNonexistentAcrossBranchesOptimisation1):
			if(not hasattr(column, branchVerifyIndexAttribute)):
				column.branchVerifyIndexes = {}
			cached = column.branchVerifyIndexes.get(feature)
			if(cached is not None and (cached[branchVerifyZero].data_ptr() != indices.data_ptr() or cached[branchVerifyCacheVersionIndex] != indices._version or tuple(cached[branchVerifyZero].shape) != tuple(indices.shape) or cached[branchVerifyOne] != tuple(tensor.size()) or not pt.equal(cached[branchVerifyTwo], mask))):
				cached = None
		if(cached is None):
			keys = (indices[branchVerifySegmentAxis, mask]*tensor.size(branchVerifyTargetConceptAxis) + indices[branchVerifyTargetConceptAxis, mask])*tensor.size(branchVerifyTargetFeatureAxis) + indices[branchVerifyTargetFeatureAxis, mask]
			keys, order = pt.sort(keys, stable=True)
			branches = indices[branchVerifyBranchAxis, mask][order]
			if(trainVerifyConnectionNonexistentAcrossBranchesCumulative and bool(pt.any(keys[branchVerifyOne:] == keys[:-branchVerifyOne]))):
				raise RuntimeError(branchVerifyInvalidInput)
			if(trainVerifyConnectionNonexistentAcrossBranchesOptimisation1):
				column.branchVerifyIndexes[feature] = (indices, tuple(tensor.size()), mask, indices._version, keys, branches)
			result = keys, branches
		else:
			result = cached[-branchVerifyTwo], cached[-branchVerifyOne]
	return result


def invalidateIndex(column, feature, tensor=None):
	if(trainVerifyConnectionNonexistentAcrossBranchesOptimisation1):
		if(hasattr(column, branchVerifyIndexAttribute)):
			cached = column.branchVerifyIndexes.get(feature)
			if(cached is not None and (tensor is None or cached[branchVerifyZero].data_ptr() != tensor.indices().data_ptr() or cached[branchVerifyCacheVersionIndex] != tensor.indices()._version or cached[branchVerifyOne] != tuple(tensor.size()))):
				del column.branchVerifyIndexes[feature]
	return


def canonicaliseSource(column, tensor):
	result = tensor
	if(trainVerifyConnectionNonexistentAcrossBranchesCumulative):
		tensor = tensor.coalesce()
		result = tensor
		indices = tensor.indices()
		targets = indices[branchVerifyTargetConceptAxis]*column.databaseNetworkObject.f + indices[branchVerifyTargetFeatureAxis]
		branches = canonicalBranches(column.databaseNetworkObject, targets, indices[branchVerifyBranchAxis])
		if(not pt.equal(branches, indices[branchVerifyBranchAxis])):
			mapped = indices.clone()
			mapped[branchVerifyBranchAxis] = branches
			result = pt.sparse_coo_tensor(mapped, tensor.values(), tensor.size(), dtype=tensor.dtype).coalesce()
	return result


def canonicaliseColumnNeurons(column):
	if(trainVerifyConnectionNonexistentAcrossBranchesCumulative):
		if(hasattr(column, branchVerifyNeuronAttribute)):
			tensor = column.featureNeurons.coalesce()
			indices = tensor.indices()
			targets = column.conceptIndex*column.databaseNetworkObject.f + indices[branchVerifyTargetConceptAxis]
			branches = canonicalBranches(column.databaseNetworkObject, targets, indices[branchVerifyBranchAxis])
			if(not pt.equal(branches, indices[branchVerifyBranchAxis])):
				mapped = indices.clone()
				mapped[branchVerifyBranchAxis] = branches
				column.featureNeurons = pt.sparse_coo_tensor(mapped, tensor.values(), tensor.size(), dtype=tensor.dtype).coalesce()
	return


def saveState(database):
	if(trainVerifyConnectionNonexistentAcrossBranchesCumulative):
		state = {branchVerifySchemaKey:branchVerifySchema, branchVerifyShapeKey:(database.c, database.f, multipleDendriticBranchesNumber, arrayNumberOfSegments), branchVerifyTargetsKey:database.branchVerifyTargets, branchVerifyParentsKey:database.branchVerifyParents}
		pt.save(state, os.path.join(databaseFolder, branchVerifyStateFileName))
	return


def canonicalBranches(database, targets, branches):
	result = branches
	if(trainVerifyConnectionNonexistentAcrossBranchesCumulative):
		positions, found = lookup(database.branchVerifyTargets, targets)
		if(bool(found.any())):
			result = branches.clone()
			result[found] = database.branchVerifyParents[positions[found], branches[found]]
	return result


def getParents(database, targets):
	result = None
	if(trainVerifyConnectionNonexistentAcrossBranchesCumulative):
		result = pt.arange(multipleDendriticBranchesNumber, dtype=pt.long).expand(targets.numel(), -branchVerifyOne).clone()
		positions, found = lookup(database.branchVerifyTargets, targets)
		result[found] = database.branchVerifyParents[positions[found]]
	return result


def storeParents(database, targets, parents):
	if(trainVerifyConnectionNonexistentAcrossBranchesCumulative):
		positions, found = lookup(database.branchVerifyTargets, targets)
		database.branchVerifyParents[positions[found]] = parents[found]
		newTargets, newParents = targets[~found], parents[~found]
		if(newTargets.numel()):
			combined, order = pt.sort(pt.cat((database.branchVerifyTargets, newTargets)))
			database.branchVerifyTargets = combined
			database.branchVerifyParents = pt.cat((database.branchVerifyParents, newParents))[order]
		database.branchVerifyRevision += branchVerifyOne
	return


def lookup(keys, query):
	result = None
	if(trainVerifyConnectionNonexistentAcrossBranchesCumulative or trainVerifyConnectionNonexistentAcrossBranchesOptimisation1):
		positions = pt.searchsorted(keys, query)
		found = positions < keys.numel()
		found[found.clone()] = keys[positions[found]] == query[found]
		result = positions, found
	return result


def validateParents(targets, parents, targetCount):
	if(trainVerifyConnectionNonexistentAcrossBranchesCumulative):
		if(targets.dtype != pt.long or parents.dtype != pt.long or targets.dim() != branchVerifyOne or tuple(parents.shape) != (targets.numel(), multipleDendriticBranchesNumber) or bool(pt.any(targets < branchVerifyZero)) or bool(pt.any(targets >= targetCount)) or bool(pt.any(targets[branchVerifyOne:] <= targets[:-branchVerifyOne])) or bool(pt.any(parents < branchVerifyZero)) or bool(pt.any(parents >= multipleDendriticBranchesNumber))):
			raise RuntimeError(branchVerifyInvalidModel)
		if(not pt.equal(parents.gather(branchVerifyOne, parents), parents)):
			raise RuntimeError(branchVerifyInvalidModel)
	return
