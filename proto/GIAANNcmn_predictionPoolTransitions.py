"""GIAANNcmn_predictionPoolTransitions.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 BAI Research Pty Ltd (bairesearch.com.au)

# License:
MIT License

# Description:
Pool immediate trained transitions across representations of the same token.

"""

from array import array
from numbers import Integral
import math
import torch as pt

from GIAANNcmn_globalDefs import *
import GIAANNcmn_databaseNetworkFiles
import GIAANNcmn_predictionConstraints


def selectPooledTransition(databaseNetworkObject, sourceColumnIndex, sourceFeatureIndex, connectedColumnsConstraint):
	result = None
	if(inferenceReviewPatch13poolTransitionsFromSimilarFeatures):
		validatePooledTransitionConfiguration(databaseNetworkObject)
		sourceToken = getPooledTransitionToken(databaseNetworkObject, sourceColumnIndex, sourceFeatureIndex)
		if(connectedColumnsConstraint is not None):
			if(not pt.is_tensor(connectedColumnsConstraint) or connectedColumnsConstraint.dim() != inferenceReviewPatch13ConnectivityRank):
				raise RuntimeError(inferenceReviewPatch13InvalidConnectivity)
			if(connectedColumnsConstraint.numel() == arrayIndexSegmentFirst):
				if(databaseNetworkObject.inferenceReviewPatch13SourceIndex is None):
					databaseNetworkObject.inferenceReviewPatch13SourceIndex = buildPooledTransitionSourceIndex(databaseNetworkObject)
				if(sourceToken not in databaseNetworkObject.inferenceReviewPatch13Predictions):
					databaseNetworkObject.inferenceReviewPatch13Predictions[sourceToken] = calculatePooledTransition(databaseNetworkObject, sourceToken)
				result = databaseNetworkObject.inferenceReviewPatch13Predictions[sourceToken]
	return result


def calculatePooledTransitionConnectedColumnsConstraint(databaseNetworkObject, observedColumnsDict, sourceColumnIndex, sourceFeatureIndex, connectedColumnsConstraint, sequenceWordIndex, seedPhase):
	result = None
	if(inferenceReviewPatch13poolTransitionsFromSimilarFeatures):
		result = connectedColumnsConstraint
		if(not predictionEnsureConnectedToPreviousPrediction):
			if(not isinstance(sequenceWordIndex, Integral) or isinstance(sequenceWordIndex, bool) or sequenceWordIndex < arrayIndexSegmentFirst):
				raise RuntimeError(inferenceReviewPatch13InvalidSequenceIndex)
			getPooledTransitionToken(databaseNetworkObject, sourceColumnIndex, sourceFeatureIndex)
			result = None
			if(inferenceSeedNetwork and sequenceWordIndex > arrayIndexSegmentFirst and not (seedPhase and enforceDirectConnectionsIgnoreSeed)):
				# Use the original lookup semantics before propagation so patches 12 and 13 observe the same source connectivity.
				result, _ = GIAANNcmn_predictionConstraints.buildConnectedColumnsLookup(databaseNetworkObject, observedColumnsDict, [(sourceColumnIndex, sourceFeatureIndex)], inferenceReviewPatch13PoolDevice, pt.long)
	return result


def validatePooledTransitionConfiguration(databaseNetworkObject):
	if(inferenceReviewPatch13poolTransitionsFromSimilarFeatures):
		#The cache represents a static trained graph; beam search and BPB need their own pooled distributions.
		if(not inferenceLeakyIntegrateAndFire or inferenceBeamSearch or printInferenceTop1AccuracyBitsPerByte or useTrainDuringInference or inferenceInferMissingFeatures):
			raise RuntimeError(inferenceReviewPatch13InvalidConfiguration)
		if(databaseNetworkObject is None or not databaseNetworkObject.inferenceMode or len(databaseNetworkObject.conceptColumnsList) != databaseNetworkObject.c or len(databaseNetworkObject.conceptFeaturesList) != databaseNetworkObject.f or databaseNetworkObject.arrayIndexPropertiesStrengthIndex is None):
			raise RuntimeError(inferenceReviewPatch13InvalidDatabase)
		if(storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam):
			if(not databaseNetworkObject.observedColumnsRAMLoaded or databaseNetworkObject.observedColumnsDictRAM is None):
				raise RuntimeError(inferenceReviewPatch13InvalidDatabase)
	return


def buildPooledTransitionSourceIndex(databaseNetworkObject):
	result = None
	if(inferenceReviewPatch13poolTransitionsFromSimilarFeatures):
		#Index filenames once, then read only the source tensors needed for each queried token.
		#Use a fixed stride so inference vocabulary growth cannot change stored source coordinates.
		sourceFeatureStride = databaseNetworkObject.f
		sourcesByToken = {}
		if(storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam):
			observedColumns = sorted(databaseNetworkObject.observedColumnsDictRAM.values(), key=lambda observedColumn: observedColumn.conceptIndex)
			for observedColumn in observedColumns:
				indexPooledTransitionSources(databaseNetworkObject, sourcesByToken, sourceFeatureStride, observedColumn.conceptIndex, observedColumn.listStoredSourceFeatureIndices())
		else:
			for sourceColumnIndex in GIAANNcmn_databaseNetworkFiles.listPersistedObservedColumnConceptIndices():
				indexPooledTransitionSources(databaseNetworkObject, sourcesByToken, sourceFeatureStride, sourceColumnIndex, GIAANNcmn_databaseNetworkFiles.listObservedColumnSourceFeatureIndices(sourceColumnIndex))
		result = sourceFeatureStride, sourcesByToken
	return result


def indexPooledTransitionSources(databaseNetworkObject, sourcesByToken, sourceFeatureStride, sourceColumnIndex, sourceFeatureIndices):
	if(inferenceReviewPatch13poolTransitionsFromSimilarFeatures):
		if(not isinstance(sourceFeatureStride, Integral) or sourceFeatureStride != databaseNetworkObject.f or sourceFeatureStride <= arrayIndexSegmentFirst):
			raise RuntimeError(inferenceReviewPatch13InvalidDatabase)
		for sourceFeatureIndex in sourceFeatureIndices:
			sourceToken = getPooledTransitionToken(databaseNetworkObject, sourceColumnIndex, sourceFeatureIndex)
			if(sourceToken not in sourcesByToken):
				sourcesByToken[sourceToken] = array(inferenceReviewPatch13SourceIndexTypeCode)
			sourcesByToken[sourceToken].append(sourceColumnIndex*sourceFeatureStride + sourceFeatureIndex)
	return


def calculatePooledTransition(databaseNetworkObject, sourceToken):
	result = None
	if(inferenceReviewPatch13poolTransitionsFromSimilarFeatures):
		sourceFeatureStride, sourcesByToken = databaseNetworkObject.inferenceReviewPatch13SourceIndex
		targetStrengths = {}
		targetExemplars = {}
		targetExemplarStrengths = {}
		if(sourceToken in sourcesByToken):
			for sourceKey in sourcesByToken[sourceToken]:
				sourceColumnIndex, sourceFeatureIndex = divmod(sourceKey, sourceFeatureStride)
				if(storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam):
					observedColumn = databaseNetworkObject.observedColumnsDictRAM[databaseNetworkObject.conceptColumnsList[sourceColumnIndex]]
					sourceConnections = observedColumn.getFeatureConnectionsForSourceFeature(sourceFeatureIndex, targetDevice=inferenceReviewPatch13PoolDevice, createMissing=False)
				else:
					sourceConnections = GIAANNcmn_databaseNetworkFiles.loadObservedColumnSourceFeatureConnectionsTensor(databaseNetworkObject, sourceColumnIndex, sourceFeatureIndex, inferenceReviewPatch13PoolDevice)
				accumulatePooledTransitionStrengths(databaseNetworkObject, sourceConnections, targetStrengths, targetExemplars, targetExemplarStrengths)
		if(targetStrengths):
			#Ties retain the first target encountered in sorted source-column/source-feature and COO order.
			winningToken = max(targetStrengths, key=targetStrengths.get)
			result = targetExemplars[winningToken]
	return result


def accumulatePooledTransitionStrengths(databaseNetworkObject, sourceConnections, targetStrengths, targetExemplars, targetExemplarStrengths):
	if(inferenceReviewPatch13poolTransitionsFromSimilarFeatures):
		if(not pt.is_tensor(sourceConnections) or sourceConnections.layout != pt.sparse_coo or sourceConnections.dim() != inferenceReviewPatch13ConnectionTensorRank):
			raise RuntimeError(inferenceReviewPatch13InvalidConnections)
		if(sourceConnections.shape[inferenceReviewPatch13PropertyDimension] != databaseNetworkObject.arrayNumberOfProperties or sourceConnections.shape[inferenceReviewPatch13BranchDimension] != multipleDendriticBranchesNumber or sourceConnections.shape[inferenceReviewPatch13SegmentDimension] != arrayNumberOfSegments or sourceConnections.shape[inferenceReviewPatch13ColumnDimension] > databaseNetworkObject.c or sourceConnections.shape[inferenceReviewPatch13FeatureDimension] > databaseNetworkObject.f):
			raise RuntimeError(inferenceReviewPatch13InvalidConnections)
		sourceConnections = sourceConnections.coalesce()
		indices = sourceConnections.indices()
		values = sourceConnections.values()
		#In LIF storage, distance-one transitions are at the soma coordinate; do not pool other segments.
		directMask = (indices[inferenceReviewPatch13PropertyDimension] == databaseNetworkObject.arrayIndexPropertiesStrengthIndex) & (indices[inferenceReviewPatch13SegmentDimension] == arrayIndexSegmentSoma)
		directStrengths = values[directMask]
		if(not bool(pt.all(pt.isfinite(directStrengths)).item()) or bool(pt.any(directStrengths < inferenceReviewPatch13MinimumStrength).item())):
			raise RuntimeError(inferenceReviewPatch13InvalidConnections)
		positiveMask = directMask & (values > inferenceReviewPatch13MinimumStrength)
		targetColumns = indices[inferenceReviewPatch13ColumnDimension, positiveMask].tolist()
		targetFeatures = indices[inferenceReviewPatch13FeatureDimension, positiveMask].tolist()
		strengths = values[positiveMask].tolist()
		for targetColumnIndex, targetFeatureIndex, strength in zip(targetColumns, targetFeatures, strengths):
			targetToken = getPooledTransitionToken(databaseNetworkObject, targetColumnIndex, targetFeatureIndex)
			pooledStrength = targetStrengths.get(targetToken, inferenceReviewPatch13MinimumStrength) + strength
			if(not math.isfinite(pooledStrength)):
				raise RuntimeError(inferenceReviewPatch13InvalidConnections)
			targetStrengths[targetToken] = pooledStrength
			if(targetToken not in targetExemplarStrengths or strength > targetExemplarStrengths[targetToken]):
				targetExemplarStrengths[targetToken] = strength
				targetExemplars[targetToken] = targetColumnIndex, targetFeatureIndex
	return


def getPooledTransitionToken(databaseNetworkObject, columnIndex, featureIndex):
	result = None
	if(inferenceReviewPatch13poolTransitionsFromSimilarFeatures):
		if(not isinstance(columnIndex, Integral) or isinstance(columnIndex, bool) or not isinstance(featureIndex, Integral) or isinstance(featureIndex, bool) or columnIndex < arrayIndexSegmentFirst or columnIndex >= databaseNetworkObject.c or featureIndex < arrayIndexSegmentFirst or featureIndex >= databaseNetworkObject.f):
			raise RuntimeError(inferenceReviewPatch13InvalidNeuron)
		#Ordinary features share f across columns; each f=0 prime denotes its own column's token.
		if(featureIndex == featureIndexPrimeConceptNeuron):
			result = databaseNetworkObject.conceptColumnsList[columnIndex]
		else:
			result = databaseNetworkObject.conceptFeaturesList[featureIndex]
	return result
