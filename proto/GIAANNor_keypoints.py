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
import GIAANNor_features as GIAANNor_featureDetection


def sampleAdjacentSalientImageSaccadeOffsetPairs(preparedImageTensor, cropMarginX, cropMarginY, snapshotWidth, snapshotHeight):
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
	if(snapshotWidth <= 0 or snapshotHeight <= 0):
		raise RuntimeError("sampleAdjacentSalientImageSaccadeOffsetPairs error: snapshotWidth/snapshotHeight must be > 0")
	workHeight = int(preparedImageTensor.shape[1])
	workWidth = int(preparedImageTensor.shape[2])
	salientFeatureCoordinates = GIAANNor_featureDetection.detectSalientFeatureCoordinatesFromImageTensor(preparedImageTensor)
	if(salientFeatureCoordinates.shape[0] >= 2):
		reachableFeatureCoordinates = filterReachableSalientImageFeatureCoordinates(salientFeatureCoordinates, workWidth, workHeight, snapshotWidth, snapshotHeight)
		if(reachableFeatureCoordinates.shape[0] >= 2):
			result = calculateAdjacentSalientImageSaccadeOffsetPairs(reachableFeatureCoordinates, workWidth, workHeight)
	return result


def detectReachableSalientImageFeatureCoordinates(preparedImageTensor, snapshotWidth, snapshotHeight):
	result = None
	salientFeatureCoordinates = None
	workHeight = None
	workWidth = None
	if(not pt.is_tensor(preparedImageTensor)):
		raise RuntimeError("detectReachableSalientImageFeatureCoordinates error: preparedImageTensor must be a tensor")
	if(preparedImageTensor.dim() != 3):
		raise RuntimeError("detectReachableSalientImageFeatureCoordinates error: preparedImageTensor rank must be 3")
	if(int(preparedImageTensor.shape[0]) != 3):
		raise RuntimeError("detectReachableSalientImageFeatureCoordinates error: preparedImageTensor channel count must equal 3")
	if(snapshotWidth <= 0 or snapshotHeight <= 0):
		raise RuntimeError("detectReachableSalientImageFeatureCoordinates error: snapshotWidth/snapshotHeight must be > 0")
	workHeight = int(preparedImageTensor.shape[1])
	workWidth = int(preparedImageTensor.shape[2])
	salientFeatureCoordinates = GIAANNor_featureDetection.detectSalientFeatureCoordinatesFromImageTensor(preparedImageTensor)
	if(salientFeatureCoordinates.shape[0] > 0):
		result = filterReachableSalientImageFeatureCoordinates(salientFeatureCoordinates, workWidth, workHeight, snapshotWidth, snapshotHeight)
	else:
		result = salientFeatureCoordinates
		if(debugPrintNumberFeatures):
			printReachableFeatureDetectionCount(result)
	return result


def alignVideoFrameKeypointsToSubsequences(videoFrameFeatureCoordinateList):
	result = None
	alignedFeatureCoordinateList = None
	numberOfSubsequences = None
	frameFeatureCoordinates = None
	selectedFeatureIndices = None
	if(submodalityName=="video" and modalityORvideoGenerateMultipleSnapshotsPerFrame):
		if(not isinstance(videoFrameFeatureCoordinateList, list)):
			raise RuntimeError("alignVideoFrameKeypointsToSubsequences error: videoFrameFeatureCoordinateList must be a list")
		if(len(videoFrameFeatureCoordinateList) <= 0):
			raise RuntimeError("alignVideoFrameKeypointsToSubsequences error: videoFrameFeatureCoordinateList must not be empty")
		if(not pt.is_tensor(videoFrameFeatureCoordinateList[0])):
			raise RuntimeError("alignVideoFrameKeypointsToSubsequences error: first frame feature coordinates must be a tensor")
		if(videoFrameFeatureCoordinateList[0].dim() != 2):
			raise RuntimeError("alignVideoFrameKeypointsToSubsequences error: first frame feature coordinates rank must be 2")
		if(int(videoFrameFeatureCoordinateList[0].shape[1]) != 2):
			raise RuntimeError("alignVideoFrameKeypointsToSubsequences error: first frame feature coordinates last dimension must equal 2")
		numberOfSubsequences = int(videoFrameFeatureCoordinateList[0].shape[0])
		if(numberOfSubsequences <= 0):
			raise RuntimeError("alignVideoFrameKeypointsToSubsequences error: first sequence iteration must contain at least one reachable keypoint")
		alignedFeatureCoordinateList = []
		alignedFeatureCoordinateList.append(videoFrameFeatureCoordinateList[0])
		for frameFeatureCoordinates in videoFrameFeatureCoordinateList[1:]:
			if(not pt.is_tensor(frameFeatureCoordinates)):
				raise RuntimeError("alignVideoFrameKeypointsToSubsequences error: frame feature coordinates must be a tensor")
			if(frameFeatureCoordinates.dim() != 2):
				raise RuntimeError("alignVideoFrameKeypointsToSubsequences error: frame feature coordinates rank must be 2")
			if(int(frameFeatureCoordinates.shape[1]) != 2):
				raise RuntimeError("alignVideoFrameKeypointsToSubsequences error: frame feature coordinates last dimension must equal 2")
			if(int(frameFeatureCoordinates.shape[0]) < numberOfSubsequences):
				raise RuntimeError("alignVideoFrameKeypointsToSubsequences error: frame contains fewer reachable keypoints than the first sequence iteration")
			selectedFeatureIndices = calculateNearestUniqueFeatureIndices(alignedFeatureCoordinateList[-1], frameFeatureCoordinates)
			alignedFeatureCoordinateList.append(frameFeatureCoordinates.index_select(0, selectedFeatureIndices))
		result = pt.stack(alignedFeatureCoordinateList, dim=0)
	else:
		raise RuntimeError("alignVideoFrameKeypointsToSubsequences error: requires submodalityName=='video' and modalityORvideoGenerateMultipleSnapshotsPerFrame")
	return result


def calculateNearestUniqueFeatureIndices(sourceFeatureCoordinates, targetFeatureCoordinates):
	result = None
	distanceMatrix = None
	flatDistanceSortIndices = None
	numberOfSourceFeatures = None
	numberOfTargetFeatures = None
	selectedSourceMask = None
	selectedTargetMask = None
	selectedTargetIndices = None
	numberOfSelectedFeatures = None
	flatDistanceSortIndex = None
	sourceFeatureIndex = None
	targetFeatureIndex = None
	if(submodalityName=="video" and modalityORvideoGenerateMultipleSnapshotsPerFrame):
		if(not pt.is_tensor(sourceFeatureCoordinates)):
			raise RuntimeError("calculateNearestUniqueFeatureIndices error: sourceFeatureCoordinates must be a tensor")
		if(not pt.is_tensor(targetFeatureCoordinates)):
			raise RuntimeError("calculateNearestUniqueFeatureIndices error: targetFeatureCoordinates must be a tensor")
		if(sourceFeatureCoordinates.dim() != 2 or targetFeatureCoordinates.dim() != 2):
			raise RuntimeError("calculateNearestUniqueFeatureIndices error: feature coordinate tensors must be rank 2")
		if(int(sourceFeatureCoordinates.shape[1]) != 2 or int(targetFeatureCoordinates.shape[1]) != 2):
			raise RuntimeError("calculateNearestUniqueFeatureIndices error: feature coordinate tensors last dimension must equal 2")
		if(sourceFeatureCoordinates.device != targetFeatureCoordinates.device):
			raise RuntimeError("calculateNearestUniqueFeatureIndices error: sourceFeatureCoordinates and targetFeatureCoordinates must be on the same device")
		numberOfSourceFeatures = int(sourceFeatureCoordinates.shape[0])
		numberOfTargetFeatures = int(targetFeatureCoordinates.shape[0])
		if(numberOfSourceFeatures <= 0):
			raise RuntimeError("calculateNearestUniqueFeatureIndices error: numberOfSourceFeatures must be > 0")
		if(numberOfTargetFeatures < numberOfSourceFeatures):
			raise RuntimeError("calculateNearestUniqueFeatureIndices error: numberOfTargetFeatures must be >= numberOfSourceFeatures")
		distanceMatrix = pt.cdist(sourceFeatureCoordinates.to(dtype=pt.float32), targetFeatureCoordinates.to(dtype=pt.float32))
		flatDistanceSortIndices = pt.argsort(distanceMatrix.reshape(-1), descending=False)
		selectedSourceMask = pt.zeros(numberOfSourceFeatures, dtype=pt.bool, device=sourceFeatureCoordinates.device)
		selectedTargetMask = pt.zeros(numberOfTargetFeatures, dtype=pt.bool, device=sourceFeatureCoordinates.device)
		selectedTargetIndices = pt.full((numberOfSourceFeatures,), -1, dtype=pt.long, device=sourceFeatureCoordinates.device)
		numberOfSelectedFeatures = 0
		for flatDistanceSortIndex in flatDistanceSortIndices:
			sourceFeatureIndex = int(flatDistanceSortIndex.item())//numberOfTargetFeatures
			targetFeatureIndex = int(flatDistanceSortIndex.item())%numberOfTargetFeatures
			if(not bool(selectedSourceMask[sourceFeatureIndex].item()) and not bool(selectedTargetMask[targetFeatureIndex].item())):
				selectedSourceMask[sourceFeatureIndex] = True
				selectedTargetMask[targetFeatureIndex] = True
				selectedTargetIndices[sourceFeatureIndex] = targetFeatureIndex
				numberOfSelectedFeatures = numberOfSelectedFeatures + 1
				if(numberOfSelectedFeatures == numberOfSourceFeatures):
					break
		if(numberOfSelectedFeatures != numberOfSourceFeatures):
			raise RuntimeError("calculateNearestUniqueFeatureIndices error: failed to align all source features")
		result = selectedTargetIndices
	else:
		raise RuntimeError("calculateNearestUniqueFeatureIndices error: requires submodalityName=='video' and modalityORvideoGenerateMultipleSnapshotsPerFrame")
	return result


def filterReachableSalientImageFeatureCoordinates(featureCoordinates, workWidth, workHeight, snapshotWidth, snapshotHeight):
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
	if(snapshotWidth <= 0 or snapshotHeight <= 0):
		raise RuntimeError("filterReachableSalientImageFeatureCoordinates error: snapshotWidth/snapshotHeight must be > 0")
	reachableWindowBounds = calculateReachableFeatureCoordinateBounds(workWidth, workHeight, snapshotWidth, snapshotHeight)
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


def calculateReachableFeatureCoordinateBounds(workWidth, workHeight, snapshotWidth, snapshotHeight):
	result = None
	minFeatureCoordinateX = None
	maxFeatureCoordinateX = None
	minFeatureCoordinateY = None
	maxFeatureCoordinateY = None
	if(workWidth <= 0 or workHeight <= 0):
		raise RuntimeError("calculateReachableFeatureCoordinateBounds error: workWidth/workHeight must be > 0")
	if(snapshotWidth <= 0 or snapshotHeight <= 0):
		raise RuntimeError("calculateReachableFeatureCoordinateBounds error: snapshotWidth/snapshotHeight must be > 0")
	minFeatureCoordinateX = float(snapshotWidth)/2.0
	maxFeatureCoordinateX = float(workWidth) - (float(snapshotWidth)/2.0)
	minFeatureCoordinateY = float(snapshotHeight)/2.0
	maxFeatureCoordinateY = float(workHeight) - (float(snapshotHeight)/2.0)
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
