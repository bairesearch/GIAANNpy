"""GIAANNcmn_databaseNetworkTrainSelectBranch.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 BAI Research Pty Ltd (bairesearch.com.au)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN common database Network Train Select Branch

"""

import torch as pt

from GIAANNcmn_globalDefs import *
import GIAANNcmn_databaseNetworkTrain


def createTrainSelectMostSimilarBranchProspectiveConnectionsDense(sequenceObservedColumns, featureNeuronsActive, cs, fs, columnsWordOrder, featureNeuronsWordOrder, featureNeuronsTargetMask=None):
	result = None
	if(trainSelectMostSimilarBranch):
		trainConnectionsIncludeSameTimeIndex = GIAANNcmn_databaseNetworkTrain.getTrainConnectionsIncludeSameTimeIndex(sequenceObservedColumns)
		if(GIAANNcmn_databaseNetworkTrain.getTrainConnectionsUseSpatialAxes(sequenceObservedColumns)):
			featureConnectionsActive, featureConnectionsSegmentMask = GIAANNcmn_databaseNetworkTrain.createFeatureConnectionsActiveTrainSpatialAxes(featureNeuronsActive, cs, fs, featureNeuronsWordOrder, trainConnectionsIncludeSameTimeIndex, sequenceObservedColumns)
		else:
			featureConnectionsActiveBase = calculateTrainSelectMostSimilarBranchProspectiveConnectionsDenseBase(featureNeuronsActive, cs, fs, columnsWordOrder, featureNeuronsWordOrder, trainConnectionsIncludeSameTimeIndex)
			if(useSANI):
				featureConnectionsActive, featureConnectionsSegmentMask = GIAANNcmn_databaseNetworkTrain.assignFeatureConnectionsToTargetSegments(featureConnectionsActiveBase, cs, fs, featureNeuronsWordOrder, sequenceObservedColumns)
			else:
				featureConnectionsActive = featureConnectionsActiveBase.unsqueeze(1)
				featureConnectionsSegmentMask = pt.ones_like(featureConnectionsActive, dtype=pt.bool)
		if(useTrainDuringInference):
			featureConnectionsActive, featureConnectionsSegmentMask = GIAANNcmn_databaseNetworkTrain.applyTrainDuringInferenceFeatureConnectionsTargetMask(featureConnectionsActive, featureConnectionsSegmentMask, featureNeuronsTargetMask)
		result = (featureConnectionsActive, featureConnectionsSegmentMask)
	return result

def createTrainSelectMostSimilarBranchProspectiveConnectionsSparse(sequenceObservedColumns, featureNeuronsActive, cs, fs, columnsWordOrder, featureNeuronsWordOrder, featureNeuronsTargetMask=None):
	result = None
	if(trainSelectMostSimilarBranch):
		trainConnectionsIncludeSameTimeIndex = GIAANNcmn_databaseNetworkTrain.getTrainConnectionsIncludeSameTimeIndex(sequenceObservedColumns)
		if(GIAANNcmn_databaseNetworkTrain.getTrainConnectionsUseSpatialAxes(sequenceObservedColumns)):
			connectionActiveSparse = GIAANNcmn_databaseNetworkTrain.createFeatureConnectionsActiveTrainSpatialAxesSparse(featureNeuronsActive, cs, fs, featureNeuronsWordOrder, trainConnectionsIncludeSameTimeIndex, sequenceObservedColumns)
		else:
			connectionTargetSize = (multipleDendriticBranchesNumber, arrayNumberOfSegments, cs, fs, cs, fs)
			connectionIndices = calculateTrainSelectMostSimilarBranchProspectiveConnectionsSparseIndices(featureNeuronsActive, cs, fs, columnsWordOrder, featureNeuronsWordOrder, trainConnectionsIncludeSameTimeIndex, sequenceObservedColumns)
			connectionValues = pt.ones((connectionIndices.shape[1],), dtype=arrayType, device=featureNeuronsActive.device)
			connectionActiveSparse = pt.sparse_coo_tensor(connectionIndices, connectionValues, size=connectionTargetSize, dtype=arrayType, device=featureNeuronsActive.device).coalesce()
			if(connectionActiveSparse._nnz() > 0):
				connectionActiveSparse.values().clamp_(max=trainSelectMostSimilarBranchMaximumConnectionActivation)
		if(useTrainDuringInference):
			connectionActiveSparse = GIAANNcmn_databaseNetworkTrain.applyTrainDuringInferenceFeatureConnectionsSparseTargetMask(connectionActiveSparse, featureNeuronsTargetMask)
		result = connectionActiveSparse
	return result

def calculateTrainSelectMostSimilarBranchProspectiveConnectionsDenseBase(featureNeuronsActive, cs, fs, columnsWordOrder, featureNeuronsWordOrder, trainConnectionsIncludeSameTimeIndex):
	result = None
	if(trainSelectMostSimilarBranch):
		if(not pt.is_tensor(featureNeuronsActive) or featureNeuronsActive.dim() != 4):
			raise RuntimeError("calculateTrainSelectMostSimilarBranchProspectiveConnectionsDenseBase error: featureNeuronsActive must be a rank 4 tensor")
		featureNeuronsActiveBySegment = featureNeuronsActive
		if(not useSANI):
			featureNeuronsActiveBySegment = featureNeuronsActive[:, arrayIndexSegmentLast].unsqueeze(1)
		branchCount = int(featureNeuronsActiveBySegment.shape[0])
		activationSegmentCount = int(featureNeuronsActiveBySegment.shape[1])
		# Collapse source branches, form every same-segment source/target pair, then collapse the activation-segment dimension.
		sourceActive = featureNeuronsActiveBySegment.amax(dim=0).reshape(activationSegmentCount, cs*fs)
		targetActive = featureNeuronsActiveBySegment.reshape(branchCount, activationSegmentCount, cs*fs)
		featureConnectionsActive = (sourceActive.view(1, activationSegmentCount, cs*fs, 1)*targetActive.view(branchCount, activationSegmentCount, 1, cs*fs)).amax(dim=1).view(branchCount, cs, fs, cs, fs)
		if(featureNeuronsWordOrder is not None):
			featureNeuronsWordOrderExpanded1 = featureNeuronsWordOrder.view(cs, fs, 1, 1).expand(cs, fs, cs, fs)
			featureNeuronsWordOrderExpanded2 = featureNeuronsWordOrder.view(1, 1, cs, fs).expand(cs, fs, cs, fs)
			wordOrderMask = GIAANNcmn_databaseNetworkTrain.createFeatureWordOrderConnectionMask(featureNeuronsWordOrderExpanded1, featureNeuronsWordOrderExpanded2, trainConnectionsIncludeSameTimeIndex)
			if(trainConnectionsAllowSelfTransitions):
				selfWordOrderMask = pt.eye(cs*fs, dtype=pt.bool, device=featureConnectionsActive.device).view(cs, fs, cs, fs)
				wordOrderMask = wordOrderMask | selfWordOrderMask
			featureConnectionsActive = featureConnectionsActive*wordOrderMask
		if(columnsWordOrder is not None):
			columnsWordOrderExpanded1 = columnsWordOrder.view(cs, 1, 1, 1).expand(cs, fs, cs, fs)
			columnsWordOrderExpanded2 = columnsWordOrder.view(1, 1, cs, 1).expand(cs, fs, cs, fs)
			if(debugConnectColumnsToNextColumnsInSequenceOnly):
				columnsWordOrderMask = pt.logical_and(columnsWordOrderExpanded2 >= columnsWordOrderExpanded1, columnsWordOrderExpanded2 <= columnsWordOrderExpanded1+1)
			else:
				columnsWordOrderMask = columnsWordOrderExpanded2 >= columnsWordOrderExpanded1
			featureConnectionsActive = featureConnectionsActive*columnsWordOrderMask
		csIndices1 = pt.arange(cs, device=featureConnectionsActive.device).view(cs, 1, 1, 1).expand(cs, fs, cs, fs)
		csIndices2 = pt.arange(cs, device=featureConnectionsActive.device).view(1, 1, cs, 1).expand(cs, fs, cs, fs)
		fsIndices1 = pt.arange(fs, device=featureConnectionsActive.device).view(1, fs, 1, 1).expand(cs, fs, cs, fs)
		fsIndices2 = pt.arange(fs, device=featureConnectionsActive.device).view(1, 1, 1, fs).expand(cs, fs, cs, fs)
		if(trainConnectionsAllowSelfTransitions):
			identityMask = pt.ones_like(csIndices1, dtype=pt.bool)
			featureConnectionsActive = featureConnectionsActive*identityMask
		else:
			identityMask = ((csIndices1 != csIndices2) | (fsIndices1 != fsIndices2)).unsqueeze(0)
			repeatedFeatureMaskBySegment = (featureNeuronsActiveBySegment > 0).sum(dim=0) > 1
			repeatedFeatureActiveByBranch = ((featureNeuronsActiveBySegment > 0) & repeatedFeatureMaskBySegment.unsqueeze(0)).any(dim=1)
			selfMask = (csIndices1 == csIndices2) & (fsIndices1 == fsIndices2)
			identityMask = identityMask | (selfMask.unsqueeze(0) & repeatedFeatureActiveByBranch.view(branchCount, cs, fs, 1, 1))
			featureConnectionsActive = featureConnectionsActive*identityMask
		result = featureConnectionsActive
	return result

def calculateTrainSelectMostSimilarBranchProspectiveConnectionsSparseIndices(featureNeuronsActive, cs, fs, columnsWordOrder, featureNeuronsWordOrder, trainConnectionsIncludeSameTimeIndex, sequenceObservedColumns):
	result = None
	if(trainSelectMostSimilarBranch):
		if(not pt.is_tensor(featureNeuronsActive) or featureNeuronsActive.dim() != 4):
			raise RuntimeError("calculateTrainSelectMostSimilarBranchProspectiveConnectionsSparseIndices error: featureNeuronsActive must be a rank 4 tensor")
		if(not pt.is_tensor(featureNeuronsWordOrder)):
			raise RuntimeError("calculateTrainSelectMostSimilarBranchProspectiveConnectionsSparseIndices error: featureNeuronsWordOrder must be a tensor")
		featureNeuronsActiveBySegment = featureNeuronsActive
		if(not useSANI):
			featureNeuronsActiveBySegment = featureNeuronsActive[:, arrayIndexSegmentLast].unsqueeze(1)
		sourceActivationIndices = pt.nonzero(featureNeuronsActiveBySegment.amax(dim=0) > 0, as_tuple=False)
		targetActivationIndices = pt.nonzero(featureNeuronsActiveBySegment > 0, as_tuple=False)
		result = pt.empty((trainSelectMostSimilarBranchProspectiveConnectionTensorRank, 0), dtype=pt.long, device=featureNeuronsActive.device)
		if(sourceActivationIndices.shape[0] > 0 and targetActivationIndices.shape[0] > 0):
			activationSegmentCount = int(featureNeuronsActiveBySegment.shape[1])
			targetActivationSortOrder = pt.argsort(targetActivationIndices[:, 1])
			targetActivationIndices = targetActivationIndices[targetActivationSortOrder]
			sourceCountsBySegment = pt.bincount(sourceActivationIndices[:, 0], minlength=activationSegmentCount)
			targetCountsBySegment = pt.bincount(targetActivationIndices[:, 1], minlength=activationSegmentCount)
			pairCountsBySegment = sourceCountsBySegment*targetCountsBySegment
			pairSegmentIndices = pt.repeat_interleave(pt.arange(activationSegmentCount, dtype=pt.long, device=featureNeuronsActive.device), pairCountsBySegment)
			pairStartsBySegment = pt.cumsum(pairCountsBySegment, dim=0) - pairCountsBySegment
			pairOffsets = pt.arange(pairSegmentIndices.shape[0], dtype=pt.long, device=featureNeuronsActive.device) - pairStartsBySegment[pairSegmentIndices]
			sourceStartsBySegment = pt.cumsum(sourceCountsBySegment, dim=0) - sourceCountsBySegment
			targetStartsBySegment = pt.cumsum(targetCountsBySegment, dim=0) - targetCountsBySegment
			sourcePairIndices = sourceStartsBySegment[pairSegmentIndices] + pt.div(pairOffsets, targetCountsBySegment[pairSegmentIndices], rounding_mode="floor")
			targetPairIndices = targetStartsBySegment[pairSegmentIndices] + pt.remainder(pairOffsets, targetCountsBySegment[pairSegmentIndices])
			sourceActivationSegmentIndices = sourceActivationIndices[sourcePairIndices, 0]
			sourceConceptIndices = sourceActivationIndices[sourcePairIndices, 1]
			sourceFeatureIndices = sourceActivationIndices[sourcePairIndices, 2]
			targetBranchIndices = targetActivationIndices[targetPairIndices, 0]
			targetConceptIndices = targetActivationIndices[targetPairIndices, 2]
			targetFeatureIndices = targetActivationIndices[targetPairIndices, 3]
			connectionMask = pt.ones((pairSegmentIndices.shape[0],), dtype=pt.bool, device=featureNeuronsActive.device)
			sourceWordOrder = featureNeuronsWordOrder[sourceConceptIndices, sourceFeatureIndices]
			targetWordOrder = featureNeuronsWordOrder[targetConceptIndices, targetFeatureIndices]
			connectionMask = connectionMask & GIAANNcmn_databaseNetworkTrain.createFeatureWordOrderConnectionMask(sourceWordOrder, targetWordOrder, trainConnectionsIncludeSameTimeIndex)
			if(columnsWordOrder is not None):
				sourceColumnWordOrder = columnsWordOrder[sourceConceptIndices]
				targetColumnWordOrder = columnsWordOrder[targetConceptIndices]
				if(debugConnectColumnsToNextColumnsInSequenceOnly):
					connectionMask = connectionMask & pt.logical_and(targetColumnWordOrder >= sourceColumnWordOrder, targetColumnWordOrder <= sourceColumnWordOrder+1)
				else:
					connectionMask = connectionMask & (targetColumnWordOrder >= sourceColumnWordOrder)
			selfMask = (sourceConceptIndices == targetConceptIndices) & (sourceFeatureIndices == targetFeatureIndices)
			if(trainConnectionsAllowSelfTransitions):
				connectionMask = connectionMask | selfMask
			else:
				repeatedFeatureMask = (featureNeuronsActiveBySegment > 0).sum(dim=0) > 1
				repeatedSourceMask = repeatedFeatureMask[sourceActivationSegmentIndices, sourceConceptIndices, sourceFeatureIndices]
				connectionMask = (connectionMask & pt.logical_not(selfMask)) | (selfMask & repeatedSourceMask)
			if(connectionMask.any()):
				result = GIAANNcmn_databaseNetworkTrain.assignFeatureConnectionsToTargetSegmentsSparse(targetBranchIndices[connectionMask], sourceConceptIndices[connectionMask], sourceFeatureIndices[connectionMask], targetConceptIndices[connectionMask], targetFeatureIndices[connectionMask], sourceWordOrder[connectionMask], targetWordOrder[connectionMask], sequenceObservedColumns)
	return result

def selectTrainMostSimilarBranches(sequenceObservedColumns, featureNeuronsActive, prospectiveConnections, featureNeuronsTargetMask=None):
	result = prospectiveConnections
	if(trainSelectMostSimilarBranch):
		if(not pt.is_tensor(featureNeuronsActive) or featureNeuronsActive.dim() != 4):
			raise RuntimeError("selectTrainMostSimilarBranches error: featureNeuronsActive must be a rank 4 tensor")
		if(int(featureNeuronsActive.shape[0]) != multipleDendriticBranchesNumber or int(featureNeuronsActive.shape[1]) != arrayNumberOfSegments):
			raise RuntimeError("selectTrainMostSimilarBranches error: featureNeuronsActive branch or segment dimensions mismatch")
		if(not pt.is_tensor(prospectiveConnections)):
			raise RuntimeError("selectTrainMostSimilarBranches error: prospectiveConnections must be a tensor")
		if(prospectiveConnections.layout != pt.sparse_coo and prospectiveConnections.layout != pt.strided):
			raise RuntimeError("selectTrainMostSimilarBranches error: prospectiveConnections must be sparse COO or dense")
		if(prospectiveConnections.dim() != trainSelectMostSimilarBranchProspectiveConnectionTensorRank):
			raise RuntimeError("selectTrainMostSimilarBranches error: prospectiveConnections rank mismatch")
		if(tuple(prospectiveConnections.shape) != (multipleDendriticBranchesNumber, arrayNumberOfSegments, int(featureNeuronsActive.shape[2]), int(featureNeuronsActive.shape[3]), int(featureNeuronsActive.shape[2]), int(featureNeuronsActive.shape[3]))):
			raise RuntimeError("selectTrainMostSimilarBranches error: prospectiveConnections dimensions mismatch")
		if(prospectiveConnections.device != featureNeuronsActive.device):
			raise RuntimeError("selectTrainMostSimilarBranches error: prospectiveConnections and featureNeuronsActive device mismatch")
		if(useTrainDuringInference):
			if(featureNeuronsTargetMask is None or not pt.is_tensor(featureNeuronsTargetMask)):
				raise RuntimeError("selectTrainMostSimilarBranches error: featureNeuronsTargetMask must be a tensor when useTrainDuringInference")
			if(tuple(featureNeuronsTargetMask.shape) != (trainSelectMostSimilarBranchMinimumCount, trainSelectMostSimilarBranchMinimumCount, int(featureNeuronsActive.shape[2]), int(featureNeuronsActive.shape[3]))):
				raise RuntimeError("selectTrainMostSimilarBranches error: featureNeuronsTargetMask dimensions mismatch")
		prospectiveConnectionsSparse = prospectiveConnections.coalesce() if prospectiveConnections.layout == pt.sparse_coo else prospectiveConnections.to_sparse_coo().coalesce()
		targetActiveMask = featureNeuronsActive.amax(dim=(0, 1)) > 0
		if(useTrainDuringInference):
			targetActiveMask = targetActiveMask & (featureNeuronsTargetMask.squeeze(0).squeeze(0).to(device=targetActiveMask.device) > 0)
		targetLinearIndices = pt.nonzero(targetActiveMask.reshape(-1), as_tuple=False).flatten()
		if(targetLinearIndices.numel() == 0):
			raise RuntimeError("selectTrainMostSimilarBranches error: no active target features")
		prospectiveConnectionsComparisonSparse = prospectiveConnectionsSparse
		if(trainSelectMostSimilarBranchCompareFeatureSegmentsOnly):
			prospectiveConnectionsComparisonSparse = filterTrainSelectMostSimilarBranchComparisonConnections(prospectiveConnectionsSparse)
		existingConnectionsSparse, sourceCombinedKeysUnique, featureIndicesInObserved, conceptIndicesTensor = extractTrainSelectMostSimilarBranchExistingConnections(sequenceObservedColumns, prospectiveConnectionsComparisonSparse)
		prospectiveConnectionIndices = prospectiveConnectionsComparisonSparse.indices()
		prospectiveConnectionValues = prospectiveConnectionsComparisonSparse.values()
		prospectiveConnectionMask = prospectiveConnectionValues > 0
		prospectiveConnectionIndices = prospectiveConnectionIndices[:, prospectiveConnectionMask]
		targetCount = int(targetLinearIndices.shape[0])
		branchCount = int(featureNeuronsActive.shape[0])
		segmentCount = int(featureNeuronsActive.shape[1])
		cs = int(featureNeuronsActive.shape[2])
		fs = int(featureNeuronsActive.shape[3])
		targetPositionLookup = pt.full((cs*fs,), trainSelectMostSimilarBranchInvalidIndex, dtype=pt.long, device=featureNeuronsActive.device)
		targetPositionLookup[targetLinearIndices] = pt.arange(targetCount, dtype=pt.long, device=featureNeuronsActive.device)
		branchSimilarities = pt.zeros((targetCount, branchCount), dtype=arrayType, device=featureNeuronsActive.device)
		plannedSegmentCounts = pt.zeros((targetCount,), dtype=pt.long, device=featureNeuronsActive.device)
		if(prospectiveConnectionIndices.numel() > 0):
			sourceCombinedKeys = sequenceObservedColumns.buildConnectionSourceCombinedKeys(prospectiveConnectionIndices, featureIndicesInObserved, conceptIndicesTensor)
			sourceBucketIndices = pt.searchsorted(sourceCombinedKeysUnique, sourceCombinedKeys)
			if(bool(pt.any(sourceBucketIndices >= sourceCombinedKeysUnique.numel()).item()) or bool(pt.any(sourceCombinedKeysUnique[sourceBucketIndices] != sourceCombinedKeys).item())):
				raise RuntimeError("selectTrainMostSimilarBranches error: prospective source bucket lookup failed")
			prospectiveTargetLinearIndices = prospectiveConnectionIndices[trainSelectMostSimilarBranchProspectiveConnectionTargetConceptDimension]*fs + prospectiveConnectionIndices[trainSelectMostSimilarBranchProspectiveConnectionTargetFeatureDimension]
			prospectiveTargetPositions = targetPositionLookup[prospectiveTargetLinearIndices]
			if(bool(pt.any(prospectiveTargetPositions == trainSelectMostSimilarBranchInvalidIndex).item())):
				raise RuntimeError("selectTrainMostSimilarBranches error: prospective connection targets an inactive feature")
			prospectiveSegmentIndices = prospectiveConnectionIndices[trainSelectMostSimilarBranchProspectiveConnectionSegmentDimension]
			sourceBucketCount = int(sourceCombinedKeysUnique.shape[0])
			# The packed key removes duplicate local occurrences of the same prospective global source connection within a target segment.
			plannedConnectionKeys = (prospectiveTargetPositions*segmentCount + prospectiveSegmentIndices)*sourceBucketCount + sourceBucketIndices
			plannedConnectionKeys = pt.unique(plannedConnectionKeys, sorted=True)
			plannedSourceBucketIndices = pt.remainder(plannedConnectionKeys, sourceBucketCount)
			plannedTargetSegmentIndices = pt.div(plannedConnectionKeys, sourceBucketCount, rounding_mode="floor")
			plannedSegmentIndices = pt.remainder(plannedTargetSegmentIndices, segmentCount)
			plannedTargetPositions = pt.div(plannedTargetSegmentIndices, segmentCount, rounding_mode="floor")
			plannedSourceCounts = pt.zeros((targetCount*segmentCount,), dtype=pt.long, device=featureNeuronsActive.device)
			plannedSourceCounts.scatter_add_(0, plannedTargetSegmentIndices, pt.ones_like(plannedTargetSegmentIndices))
			plannedSourceCounts = plannedSourceCounts.view(targetCount, segmentCount)
			plannedSegmentMask = plannedSourceCounts > 0
			plannedSegmentCounts = plannedSegmentMask.sum(dim=1)
			plannedTargetLinearIndices = targetLinearIndices[plannedTargetPositions]
			plannedTargetSequenceConceptIndices = pt.div(plannedTargetLinearIndices, fs, rounding_mode="floor")
			plannedTargetSequenceFeatureIndices = pt.remainder(plannedTargetLinearIndices, fs)
			plannedTargetConceptIndices = conceptIndicesTensor[plannedTargetSequenceConceptIndices]
			plannedTargetFeatureIndices = plannedTargetSequenceFeatureIndices
			if(trainSequenceObservedColumnsUseSequenceFeaturesOnly):
				plannedTargetFeatureIndices = featureIndicesInObserved[plannedTargetSequenceFeatureIndices]
			maximumPackedKeyExclusive = branchCount*segmentCount*sourceBucketCount*int(sequenceObservedColumns.databaseNetworkObject.c)*int(sequenceObservedColumns.databaseNetworkObject.f)
			if(maximumPackedKeyExclusive > pt.iinfo(pt.long).max):
				raise RuntimeError("selectTrainMostSimilarBranches error: connection comparison key exceeds int64 capacity")
			candidateBranchIndices = pt.arange(branchCount, dtype=pt.long, device=featureNeuronsActive.device).view(1, branchCount).expand(plannedConnectionKeys.shape[0], branchCount)
			candidateConnectionKeys = (((candidateBranchIndices*segmentCount + plannedSegmentIndices.unsqueeze(1))*sourceBucketCount + plannedSourceBucketIndices.unsqueeze(1))*int(sequenceObservedColumns.databaseNetworkObject.c) + plannedTargetConceptIndices.unsqueeze(1))*int(sequenceObservedColumns.databaseNetworkObject.f) + plannedTargetFeatureIndices.unsqueeze(1)
			existingConnectionIndices = existingConnectionsSparse.indices()
			existingConnectionValues = existingConnectionsSparse.values()
			existingConnectionMask = (existingConnectionIndices[trainSelectMostSimilarBranchExistingConnectionPropertyDimension] == sequenceObservedColumns.databaseNetworkObject.arrayIndexPropertiesStrengthIndex) & (existingConnectionValues > trainSelectMostSimilarBranchConnectionExistsStrengthThreshold)
			existingConnectionKeys = (((existingConnectionIndices[trainSelectMostSimilarBranchExistingConnectionBranchDimension, existingConnectionMask]*segmentCount + existingConnectionIndices[trainSelectMostSimilarBranchExistingConnectionSegmentDimension, existingConnectionMask])*sourceBucketCount + existingConnectionIndices[trainSelectMostSimilarBranchExistingConnectionSourceBucketDimension, existingConnectionMask])*int(sequenceObservedColumns.databaseNetworkObject.c) + existingConnectionIndices[trainSelectMostSimilarBranchExistingConnectionTargetConceptDimension, existingConnectionMask])*int(sequenceObservedColumns.databaseNetworkObject.f) + existingConnectionIndices[trainSelectMostSimilarBranchExistingConnectionTargetFeatureDimension, existingConnectionMask]
			existingConnectionKeys = pt.unique(existingConnectionKeys, sorted=True)
			candidateConnectionExists = pt.zeros(candidateConnectionKeys.shape, dtype=pt.bool, device=featureNeuronsActive.device)
			if(existingConnectionKeys.numel() > 0):
				candidateConnectionPositions = pt.searchsorted(existingConnectionKeys, candidateConnectionKeys.reshape(-1))
				candidateConnectionPositionMask = candidateConnectionPositions < existingConnectionKeys.numel()
				candidateConnectionExistsFlat = candidateConnectionExists.reshape(-1)
				candidateConnectionExistsFlat[candidateConnectionPositionMask] = existingConnectionKeys[candidateConnectionPositions[candidateConnectionPositionMask]] == candidateConnectionKeys.reshape(-1)[candidateConnectionPositionMask]
				candidateConnectionExists = candidateConnectionExistsFlat.view(candidateConnectionKeys.shape)
			matchCountIndices = ((plannedTargetPositions.unsqueeze(1)*branchCount + candidateBranchIndices)*segmentCount + plannedSegmentIndices.unsqueeze(1)).reshape(-1)
			matchedSourceCounts = pt.zeros((targetCount*branchCount*segmentCount,), dtype=pt.long, device=featureNeuronsActive.device)
			matchedSourceCounts.scatter_add_(0, matchCountIndices, candidateConnectionExists.reshape(-1).to(pt.long))
			matchedSourceCounts = matchedSourceCounts.view(targetCount, branchCount, segmentCount)
			# Each nonempty segment contributes equally, regardless of how many prospective sources it contains.
			segmentSimilarities = matchedSourceCounts.to(arrayType)/plannedSourceCounts.clamp(min=trainSelectMostSimilarBranchMinimumCount).unsqueeze(1).to(arrayType)
			segmentSimilarities = segmentSimilarities*plannedSegmentMask.unsqueeze(1).to(arrayType)
			branchSimilarities = segmentSimilarities.sum(dim=2)/plannedSegmentCounts.clamp(min=trainSelectMostSimilarBranchMinimumCount).unsqueeze(1).to(arrayType)
		maximumBranchSimilarities = branchSimilarities.max(dim=1).values
		qualifiedTargetMask = (plannedSegmentCounts > 0) & (maximumBranchSimilarities >= trainSelectMostSimilarBranchThreshold)
		mostSimilarBranchMask = branchSimilarities == maximumBranchSimilarities.unsqueeze(1)
		mostSimilarBranchScores = pt.rand((targetCount, branchCount), device=featureNeuronsActive.device)
		mostSimilarBranchScores = mostSimilarBranchScores.masked_fill(pt.logical_not(mostSimilarBranchMask), trainSelectMostSimilarBranchNoCandidateScore)
		mostSimilarBranchIndices = pt.argmax(mostSimilarBranchScores, dim=1)
		randomBranchIndices = pt.randint(branchCount, (targetCount,), dtype=pt.long, device=featureNeuronsActive.device)
		selectedBranchIndices = pt.where(qualifiedTargetMask, mostSimilarBranchIndices, randomBranchIndices)
		selectedBranchLookup = pt.full((cs*fs,), trainSelectMostSimilarBranchInvalidIndex, dtype=pt.long, device=featureNeuronsActive.device)
		selectedBranchLookup[targetLinearIndices] = selectedBranchIndices
		featureNeuronsActiveFlat = featureNeuronsActive.view(branchCount, segmentCount, cs*fs)
		featureNeuronsActiveCollapsed = featureNeuronsActiveFlat.amax(dim=0)
		selectedBranchesByFeature = pt.zeros((cs*fs,), dtype=pt.long, device=featureNeuronsActive.device)
		selectedBranchesByFeature[targetLinearIndices] = selectedBranchIndices
		selectedBranchMask = pt.nn.functional.one_hot(selectedBranchesByFeature, num_classes=branchCount).transpose(0, 1).unsqueeze(1).to(featureNeuronsActive.dtype)
		featureNeuronsActiveRemapped = selectedBranchMask*featureNeuronsActiveCollapsed.unsqueeze(0)
		featureNeuronsActiveFlat.copy_(pt.where(targetActiveMask.reshape(1, 1, cs*fs), featureNeuronsActiveRemapped, featureNeuronsActiveFlat))
		prospectiveConnectionIndicesRemapped = prospectiveConnectionsSparse.indices().clone()
		if(prospectiveConnectionIndicesRemapped.numel() > 0):
			prospectiveConnectionTargetLinearIndices = prospectiveConnectionIndicesRemapped[trainSelectMostSimilarBranchProspectiveConnectionTargetConceptDimension]*fs + prospectiveConnectionIndicesRemapped[trainSelectMostSimilarBranchProspectiveConnectionTargetFeatureDimension]
			prospectiveConnectionSelectedBranches = selectedBranchLookup[prospectiveConnectionTargetLinearIndices]
			if(bool(pt.any(prospectiveConnectionSelectedBranches == trainSelectMostSimilarBranchInvalidIndex).item())):
				raise RuntimeError("selectTrainMostSimilarBranches error: prospective connection branch remap target is inactive")
			prospectiveConnectionIndicesRemapped[trainSelectMostSimilarBranchProspectiveConnectionBranchDimension] = prospectiveConnectionSelectedBranches
		prospectiveConnectionsSparseRemapped = pt.sparse_coo_tensor(prospectiveConnectionIndicesRemapped, prospectiveConnectionsSparse.values(), size=prospectiveConnectionsSparse.size(), dtype=prospectiveConnectionsSparse.dtype, device=prospectiveConnectionsSparse.device).coalesce()
		if(prospectiveConnectionsSparseRemapped._nnz() > 0):
			prospectiveConnectionsSparseRemapped.values().clamp_(max=trainSelectMostSimilarBranchMaximumConnectionActivation)
		result = prospectiveConnectionsSparseRemapped if prospectiveConnections.layout == pt.sparse_coo else prospectiveConnectionsSparseRemapped.to_dense()
	return result

def filterTrainSelectMostSimilarBranchComparisonConnections(prospectiveConnectionsSparse):
	result = None
	if(trainSelectMostSimilarBranch):
		if(not pt.is_tensor(prospectiveConnectionsSparse) or prospectiveConnectionsSparse.layout != pt.sparse_coo):
			raise RuntimeError("filterTrainSelectMostSimilarBranchComparisonConnections error: prospectiveConnectionsSparse must be sparse COO")
		prospectiveConnectionsSparse = prospectiveConnectionsSparse.coalesce()
		if(prospectiveConnectionsSparse.dim() != trainSelectMostSimilarBranchProspectiveConnectionTensorRank):
			raise RuntimeError("filterTrainSelectMostSimilarBranchComparisonConnections error: prospective connection rank mismatch")
		result = prospectiveConnectionsSparse
		if(trainSelectMostSimilarBranchCompareFeatureSegmentsOnly):
			if(not (useSANIfeatures or useSANIfeaturesAndColumns)):
				raise RuntimeError("filterTrainSelectMostSimilarBranchComparisonConnections error: trainSelectMostSimilarBranchCompareFeatureSegmentsOnly requires feature segments")
			featureSegmentStart = arrayIndexSegmentFirst
			if(useSANIfeaturesAndColumns):
				featureSegmentStart = arrayNumberOfSegmentsColumnDistance
			if(featureSegmentStart < arrayIndexSegmentFirst or featureSegmentStart >= int(prospectiveConnectionsSparse.shape[trainSelectMostSimilarBranchProspectiveConnectionSegmentDimension])):
				raise RuntimeError("filterTrainSelectMostSimilarBranchComparisonConnections error: feature segment start index out of range")
			prospectiveConnectionIndices = prospectiveConnectionsSparse.indices()
			prospectiveConnectionValues = prospectiveConnectionsSparse.values()
			featureSegmentMask = prospectiveConnectionIndices[trainSelectMostSimilarBranchProspectiveConnectionSegmentDimension] >= featureSegmentStart
			result = pt.sparse_coo_tensor(prospectiveConnectionIndices[:, featureSegmentMask], prospectiveConnectionValues[featureSegmentMask], size=prospectiveConnectionsSparse.size(), dtype=prospectiveConnectionsSparse.dtype, device=prospectiveConnectionsSparse.device).coalesce()
	return result

def extractTrainSelectMostSimilarBranchExistingConnections(sequenceObservedColumns, prospectiveConnectionsSparse):
	result = None
	if(trainSelectMostSimilarBranch):
		if(not pt.is_tensor(prospectiveConnectionsSparse) or prospectiveConnectionsSparse.layout != pt.sparse_coo):
			raise RuntimeError("extractTrainSelectMostSimilarBranchExistingConnections error: prospectiveConnectionsSparse must be sparse COO")
		prospectiveConnectionsSparse = prospectiveConnectionsSparse.coalesce()
		if(prospectiveConnectionsSparse.dim() != trainSelectMostSimilarBranchProspectiveConnectionTensorRank):
			raise RuntimeError("extractTrainSelectMostSimilarBranchExistingConnections error: prospective connection rank mismatch")
		prospectiveConnectionIndices = prospectiveConnectionsSparse.indices()
		prospectiveConnectionValues = prospectiveConnectionsSparse.values()
		prospectiveConnectionIndices = prospectiveConnectionIndices[:, prospectiveConnectionValues > 0]
		featureIndicesInObserved, _ = sequenceObservedColumns.getObservedColumnFeatureIndices()
		featureIndicesInObserved = featureIndicesInObserved.to(device=prospectiveConnectionsSparse.device, dtype=pt.long)
		conceptIndicesTensor = sequenceObservedColumns.conceptIndicesInSequenceObservedTensor.to(device=prospectiveConnectionsSparse.device, dtype=pt.long)
		if(conceptIndicesTensor.dim() != 1 or int(conceptIndicesTensor.shape[0]) != int(prospectiveConnectionsSparse.shape[trainSelectMostSimilarBranchProspectiveConnectionSourceConceptDimension])):
			raise RuntimeError("extractTrainSelectMostSimilarBranchExistingConnections error: sequence concept index tensor dimensions mismatch")
		if(trainSequenceObservedColumnsUseSequenceFeaturesOnly):
			if(featureIndicesInObserved.dim() != 1 or int(featureIndicesInObserved.shape[0]) != int(prospectiveConnectionsSparse.shape[trainSelectMostSimilarBranchProspectiveConnectionSourceFeatureDimension])):
				raise RuntimeError("extractTrainSelectMostSimilarBranchExistingConnections error: observed feature index tensor dimensions mismatch")
		sourceCombinedKeys = sequenceObservedColumns.buildConnectionSourceCombinedKeys(prospectiveConnectionIndices, featureIndicesInObserved, conceptIndicesTensor)
		sourceCombinedKeysUnique = pt.unique(sourceCombinedKeys, sorted=True)
		if(sourceCombinedKeysUnique.numel() == 0):
			existingConnectionTargetSize = (sequenceObservedColumns.databaseNetworkObject.arrayNumberOfProperties, multipleDendriticBranchesNumber, arrayNumberOfSegments, 0, sequenceObservedColumns.databaseNetworkObject.c, sequenceObservedColumns.databaseNetworkObject.f)
			existingConnectionIndices = pt.empty((trainSelectMostSimilarBranchExistingConnectionTensorRank, 0), dtype=pt.long, device=prospectiveConnectionsSparse.device)
			existingConnectionValues = pt.empty((0,), dtype=arrayType, device=prospectiveConnectionsSparse.device)
			existingConnectionsSparse = pt.sparse_coo_tensor(existingConnectionIndices, existingConnectionValues, size=existingConnectionTargetSize, dtype=arrayType, device=prospectiveConnectionsSparse.device).coalesce()
		else:
			# Reuse the efficient source-bucket extractor so only prospective source shards are materialised.
			if(trainSequenceObservedColumnsMatchSequenceWords):
				sequenceObservedColumnsDict = sequenceObservedColumns.sequenceObservedColumnsDict
			else:
				sequenceObservedColumnsDict = sequenceObservedColumns.observedColumnsDict2
			observedColumnsByConceptIndex = sequenceObservedColumns.getObservedColumnsByConceptIndex(sequenceObservedColumnsDict)
			existingConnectionsSparse = sequenceObservedColumns.gatherConnectionSourceBucketTensor(observedColumnsByConceptIndex, sourceCombinedKeysUnique, prospectiveConnectionsSparse.device).coalesce()
		if(existingConnectionsSparse.dim() != trainSelectMostSimilarBranchExistingConnectionTensorRank):
			raise RuntimeError("extractTrainSelectMostSimilarBranchExistingConnections error: existing connection rank mismatch")
		if(int(existingConnectionsSparse.shape[trainSelectMostSimilarBranchExistingConnectionBranchDimension]) != multipleDendriticBranchesNumber or int(existingConnectionsSparse.shape[trainSelectMostSimilarBranchExistingConnectionSegmentDimension]) != arrayNumberOfSegments):
			raise RuntimeError("extractTrainSelectMostSimilarBranchExistingConnections error: existing connection branch or segment dimensions mismatch")
		if(int(existingConnectionsSparse.shape[trainSelectMostSimilarBranchExistingConnectionSourceBucketDimension]) != int(sourceCombinedKeysUnique.shape[0])):
			raise RuntimeError("extractTrainSelectMostSimilarBranchExistingConnections error: existing connection source bucket dimension mismatch")
		if(int(existingConnectionsSparse.shape[trainSelectMostSimilarBranchExistingConnectionTargetConceptDimension]) != int(sequenceObservedColumns.databaseNetworkObject.c) or int(existingConnectionsSparse.shape[trainSelectMostSimilarBranchExistingConnectionTargetFeatureDimension]) != int(sequenceObservedColumns.databaseNetworkObject.f)):
			raise RuntimeError("extractTrainSelectMostSimilarBranchExistingConnections error: existing connection target dimensions mismatch")
		strengthPropertyIndex = sequenceObservedColumns.databaseNetworkObject.arrayIndexPropertiesStrengthIndex
		if(not isinstance(strengthPropertyIndex, int) or isinstance(strengthPropertyIndex, bool) or strengthPropertyIndex < 0 or strengthPropertyIndex >= int(existingConnectionsSparse.shape[trainSelectMostSimilarBranchExistingConnectionPropertyDimension])):
			raise RuntimeError("extractTrainSelectMostSimilarBranchExistingConnections error: strength property index out of range")
		result = (existingConnectionsSparse, sourceCombinedKeysUnique, featureIndicesInObserved, conceptIndicesTensor)
	return result
