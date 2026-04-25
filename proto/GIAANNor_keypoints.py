"""GIAANNor_keypoints.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN or keypoints

"""

import torch as pt

from GIAANNcmn_globalDefs import *
import GIAANNor_features


def sampleAdjacentSalientImageSaccadeOffsetPairs(preparedImageTensor, cropMarginX, cropMarginY):
	# upgrade submodalityName=="image" to perform saccades augomentations between nearby (i.e. adjacent) salient regions of the image: a) segment centres and b) corner features.
	result = None
	salientFeatureCoordinates = None
	reachableFeatureCoordinates = None
	workHeight = None
	workWidth = None
	if(not pt.is_tensor(preparedImageTensor)):
		raise RuntimeError("sampleAdjacentSalientImageSaccadeOffsetPairs error: preparedImageTensor must be a tensor")
	if(preparedImageTensor.dim() != 3):
		raise RuntimeError("sampleAdjacentSalientImageSaccadeOffsetPairs error: preparedImageTensor rank must be 3")
	if(cropMarginX < 0 or cropMarginY < 0):
		raise RuntimeError("sampleAdjacentSalientImageSaccadeOffsetPairs error: cropMarginX/cropMarginY must be >= 0")
	workHeight = int(preparedImageTensor.shape[1])
	workWidth = int(preparedImageTensor.shape[2])
	salientFeatureCoordinates = GIAANNor_features.detectSalientFeatureCoordinatesFromImageTensor(preparedImageTensor)
	if(salientFeatureCoordinates.shape[0] >= 2):
		reachableFeatureCoordinates = filterReachableSalientImageFeatureCoordinates(salientFeatureCoordinates, workWidth, workHeight)
		if(reachableFeatureCoordinates.shape[0] >= 2):
			result = calculateAdjacentSalientImageSaccadeOffsetPairs(reachableFeatureCoordinates, workWidth, workHeight)
	return result


def filterReachableSalientImageFeatureCoordinates(featureCoordinates, workWidth, workHeight):
	result = None
	reachableWindowBounds = None
	minFeatureCoordinateX = None
	maxFeatureCoordinateX = None
	minFeatureCoordinateY = None
	maxFeatureCoordinateY = None
	projectedFeatureCoordinates = None
	if(not pt.is_tensor(featureCoordinates)):
		raise RuntimeError("filterReachableSalientImageFeatureCoordinates error: featureCoordinates must be a tensor")
	if(featureCoordinates.dim() != 2):
		raise RuntimeError("filterReachableSalientImageFeatureCoordinates error: featureCoordinates rank must be 2")
	if(int(featureCoordinates.shape[1]) != 2):
		raise RuntimeError("filterReachableSalientImageFeatureCoordinates error: featureCoordinates last dimension must equal 2")
	if(workWidth <= 0 or workHeight <= 0):
		raise RuntimeError("filterReachableSalientImageFeatureCoordinates error: workWidth/workHeight must be > 0")
	reachableWindowBounds = calculateReachableFeatureCoordinateBounds(workWidth, workHeight)
	minFeatureCoordinateX = reachableWindowBounds[0]
	maxFeatureCoordinateX = reachableWindowBounds[1]
	minFeatureCoordinateY = reachableWindowBounds[2]
	maxFeatureCoordinateY = reachableWindowBounds[3]
	projectedFeatureCoordinates = featureCoordinates.clone()
	projectedFeatureCoordinates[:, 0] = projectedFeatureCoordinates[:, 0].clamp(min=minFeatureCoordinateX, max=maxFeatureCoordinateX)
	projectedFeatureCoordinates[:, 1] = projectedFeatureCoordinates[:, 1].clamp(min=minFeatureCoordinateY, max=maxFeatureCoordinateY)
	result = pt.unique(projectedFeatureCoordinates, dim=0)
	if(debugPrintNumberFeatures):
		printReachableFeatureDetectionCount(result)
	return result


def calculateReachableFeatureCoordinateBounds(workWidth, workHeight):
	result = None
	minFeatureCoordinateX = None
	maxFeatureCoordinateX = None
	minFeatureCoordinateY = None
	maxFeatureCoordinateY = None
	if(workWidth <= 0 or workHeight <= 0):
		raise RuntimeError("calculateReachableFeatureCoordinateBounds error: workWidth/workHeight must be > 0")
	minFeatureCoordinateX = float(modalityORsnapshotWidth)/2.0
	maxFeatureCoordinateX = float(workWidth) - (float(modalityORsnapshotWidth)/2.0)
	minFeatureCoordinateY = float(modalityORsnapshotHeight)/2.0
	maxFeatureCoordinateY = float(workHeight) - (float(modalityORsnapshotHeight)/2.0)
	if(minFeatureCoordinateX > maxFeatureCoordinateX or minFeatureCoordinateY > maxFeatureCoordinateY):
		raise RuntimeError("calculateReachableFeatureCoordinateBounds error: reachable feature coordinate window is invalid for current snapshot/work dimensions")
	result = (minFeatureCoordinateX, maxFeatureCoordinateX, minFeatureCoordinateY, maxFeatureCoordinateY)
	return result


def printReachableFeatureDetectionCount(reachableFeatureCoordinates):
	result = None
	numReachableFeatureCoordinates = None
	if(not pt.is_tensor(reachableFeatureCoordinates)):
		raise RuntimeError("printReachableFeatureDetectionCount error: reachableFeatureCoordinates must be a tensor")
	if(reachableFeatureCoordinates.dim() != 2):
		raise RuntimeError("printReachableFeatureDetectionCount error: reachableFeatureCoordinates rank must be 2")
	numReachableFeatureCoordinates = int(reachableFeatureCoordinates.shape[0])
	print("featureDetection: numReachableFeaturesDetected = ", numReachableFeatureCoordinates)
	return result


def printAdjacentFeaturePairCount(pairIndices):
	result = None
	numAdjacentFeaturePairs = None
	if(not pt.is_tensor(pairIndices)):
		raise RuntimeError("printAdjacentFeaturePairCount error: pairIndices must be a tensor")
	if(pairIndices.dim() != 2):
		raise RuntimeError("printAdjacentFeaturePairCount error: pairIndices rank must be 2")
	numAdjacentFeaturePairs = int(pairIndices.shape[0])
	print("featureDetection: numAdjacentFeaturePairsDetected = ", numAdjacentFeaturePairs)
	return result


def calculateAdjacentSalientImageSaccadeOffsetPairs(featureCoordinates, workWidth, workHeight):
	result = None
	pairIndices = None
	startFeatureCoordinates = None
	endFeatureCoordinates = None
	pairDistances = None
	sortOrder = None
	selectedPairIndices = None
	requiredPairIndices = None
	repeatCount = None
	imageCentreCoordinate = None
	startDistanceToCentre = None
	endDistanceToCentre = None
	swapMask = None
	tempFeatureCoordinates = None
	startOffsets = None
	endOffsets = None
	if(not pt.is_tensor(featureCoordinates)):
		raise RuntimeError("calculateAdjacentSalientImageSaccadeOffsetPairs error: featureCoordinates must be a tensor")
	if(featureCoordinates.dim() != 2):
		raise RuntimeError("calculateAdjacentSalientImageSaccadeOffsetPairs error: featureCoordinates rank must be 2")
	if(int(featureCoordinates.shape[1]) != 2):
		raise RuntimeError("calculateAdjacentSalientImageSaccadeOffsetPairs error: featureCoordinates last dimension must equal 2")
	if(workWidth <= 0 or workHeight <= 0):
		raise RuntimeError("calculateAdjacentSalientImageSaccadeOffsetPairs error: workWidth/workHeight must be > 0")
	if(modalityORimageSaccadesPerImage <= 0):
		raise RuntimeError("calculateAdjacentSalientImageSaccadeOffsetPairs error: modalityORimageSaccadesPerImage must be > 0")
	pairIndices = calculateAdjacentSalientImageFeaturePairIndices(featureCoordinates)
	if(debugPrintNumberFeatures):
		printAdjacentFeaturePairCount(pairIndices)
	if(int(pairIndices.shape[0]) > 0):
		startFeatureCoordinates = featureCoordinates.index_select(0, pairIndices[:, 0])
		endFeatureCoordinates = featureCoordinates.index_select(0, pairIndices[:, 1])
		pairDistances = pt.linalg.vector_norm(endFeatureCoordinates - startFeatureCoordinates, dim=1)
		sortOrder = pt.argsort(pairDistances, descending=False)
		selectedPairIndices = pairIndices.index_select(0, sortOrder)
		if(int(selectedPairIndices.shape[0]) >= int(modalityORimageSaccadesPerImage)):
			requiredPairIndices = pt.arange(int(modalityORimageSaccadesPerImage), device=featureCoordinates.device)
			selectedPairIndices = selectedPairIndices.index_select(0, requiredPairIndices)
		else:
			repeatCount = int((int(modalityORimageSaccadesPerImage) + int(selectedPairIndices.shape[0]) - 1)/int(selectedPairIndices.shape[0]))
			selectedPairIndices = selectedPairIndices.repeat((repeatCount, 1))
			requiredPairIndices = pt.arange(int(modalityORimageSaccadesPerImage), device=featureCoordinates.device)
			selectedPairIndices = selectedPairIndices.index_select(0, requiredPairIndices)
		imageCentreCoordinate = pt.tensor((float(workWidth)/2.0, float(workHeight)/2.0), dtype=featureCoordinates.dtype, device=featureCoordinates.device)
		startFeatureCoordinates = featureCoordinates.index_select(0, selectedPairIndices[:, 0])
		endFeatureCoordinates = featureCoordinates.index_select(0, selectedPairIndices[:, 1])
		startDistanceToCentre = pt.linalg.vector_norm(startFeatureCoordinates - imageCentreCoordinate, dim=1)
		endDistanceToCentre = pt.linalg.vector_norm(endFeatureCoordinates - imageCentreCoordinate, dim=1)
		swapMask = startDistanceToCentre > endDistanceToCentre
		if(pt.any(swapMask)):
			tempFeatureCoordinates = startFeatureCoordinates[swapMask].clone()
			startFeatureCoordinates[swapMask] = endFeatureCoordinates[swapMask]
			endFeatureCoordinates[swapMask] = tempFeatureCoordinates
		startOffsets = startFeatureCoordinates - imageCentreCoordinate
		endOffsets = endFeatureCoordinates - imageCentreCoordinate
		result = pt.cat((startOffsets, endOffsets), dim=1)
	return result


def calculateAdjacentSalientImageFeaturePairIndices(featureCoordinates):
	result = None
	numFeatures = None
	nearbyPairsPerFeature = None
	coordinateDeltas = None
	squaredDistanceMatrix = None
	featureIndices = None
	nearestFeatureIndices = None
	sourceFeatureIndices = None
	targetFeatureIndices = None
	pairIndices = None
	unorderedPairIndices = None
	if(not pt.is_tensor(featureCoordinates)):
		raise RuntimeError("calculateAdjacentSalientImageFeaturePairIndices error: featureCoordinates must be a tensor")
	if(featureCoordinates.dim() != 2):
		raise RuntimeError("calculateAdjacentSalientImageFeaturePairIndices error: featureCoordinates rank must be 2")
	if(int(featureCoordinates.shape[1]) != 2):
		raise RuntimeError("calculateAdjacentSalientImageFeaturePairIndices error: featureCoordinates last dimension must equal 2")
	if(not isinstance(modalityORimageSaccadesNumberOfNearbyPairs, int)):
		raise RuntimeError("calculateAdjacentSalientImageFeaturePairIndices error: modalityORimageSaccadesNumberOfNearbyPairs must be an int")
	if(modalityORimageSaccadesNumberOfNearbyPairs <= 0):
		raise RuntimeError("calculateAdjacentSalientImageFeaturePairIndices error: modalityORimageSaccadesNumberOfNearbyPairs must be > 0")
	numFeatures = int(featureCoordinates.shape[0])
	if(numFeatures >= 2):
		nearbyPairsPerFeature = min(int(modalityORimageSaccadesNumberOfNearbyPairs), numFeatures - 1)
		coordinateDeltas = featureCoordinates.unsqueeze(1) - featureCoordinates.unsqueeze(0)
		squaredDistanceMatrix = pt.sum(coordinateDeltas*coordinateDeltas, dim=2)
		featureIndices = pt.arange(numFeatures, device=featureCoordinates.device)
		squaredDistanceMatrix[featureIndices, featureIndices] = float("inf")
		nearestFeatureIndices = pt.argsort(squaredDistanceMatrix, dim=1)[:, :nearbyPairsPerFeature]
		sourceFeatureIndices = featureIndices.unsqueeze(1).repeat((1, nearbyPairsPerFeature)).reshape(-1)
		targetFeatureIndices = nearestFeatureIndices.reshape(-1)
		pairIndices = pt.stack((sourceFeatureIndices, targetFeatureIndices), dim=1)
		unorderedPairIndices = pt.sort(pairIndices, dim=1).values
		result = pt.unique(unorderedPairIndices, dim=0)
	else:
		result = pt.empty((0, 2), dtype=pt.long, device=featureCoordinates.device)
	return result
