"""GIAANNor_snapshots.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN OR snapshots

"""

import math
import torch as pt

from GIAANNcmn_globalDefs import *
import GIAANNor_keypoints
import GIAANNor_snapshotDimensions


def sampleVideoSnapshotSubsequences(frameTensor, articleIndex, sequenceCount):
	result = None
	videoFrameFeatureCoordinateList = None
	alignedFeatureCoordinates = None
	if(submodalityName=="video" and modalityORvideoGenerateMultipleSnapshotsPerFrame):
		videoFrameFeatureCoordinateList = detectVideoFrameFeatureCoordinateList(frameTensor, articleIndex, sequenceCount)
		alignedFeatureCoordinates = GIAANNor_keypoints.alignVideoFrameKeypointsToSubsequences(videoFrameFeatureCoordinateList)
		if(modalityORvideoGenerateMultipleSnapshotsPerFrameParallel):
			result = generateVideoSnapshotSubsequencesParallel(frameTensor, alignedFeatureCoordinates)
		else:
			result = generateVideoSnapshotSubsequencesSequential(frameTensor, alignedFeatureCoordinates)
	else:
		raise RuntimeError("sampleVideoSnapshotSubsequences error: requires submodalityName=='video' and modalityORvideoGenerateMultipleSnapshotsPerFrame")
	return result


def sampleImageSaccadeSequences(imageTensor):
	# generate sequence data by augmenting each image:
	# for each image, generate up to modalityORimageSaccadesPerImage sequences by performing saccade augmentations:
	result = []
	preparedImageTensor = None
	cropMarginX = None
	cropMarginY = None
	snapshotWidth = None
	snapshotHeight = None
	preparedImageWorkWidth = None
	preparedImageWorkHeight = None
	saccadeOffsetPairs = None
	saccadeOffsetPair = None
	targetOffsetX = None
	targetOffsetY = None
	sequence = None
	if(modalityORimageSaccadesPerImage <= 0):
		raise RuntimeError("sampleImageSaccadeSequences error: modalityORimageSaccadesPerImage must be > 0")
	if(modalityORimageSnapshotsPerSaccade <= 0):
		raise RuntimeError("sampleImageSaccadeSequences error: modalityORimageSnapshotsPerSaccade must be > 0")
	validateImageSaccadeEncodingParameters()
	if(not pt.is_tensor(imageTensor)):
		raise RuntimeError("sampleImageSaccadeSequences error: imageTensor must be a tensor")
	if(imageTensor.dim() != 3):
		raise RuntimeError("sampleImageSaccadeSequences error: imageTensor rank must be 3")
	if(int(imageTensor.shape[0]) != 3):
		raise RuntimeError("sampleImageSaccadeSequences error: imageTensor channel count must equal 3")
	snapshotWidth, snapshotHeight = GIAANNor_snapshotDimensions.calculateSnapshotDimensionsFromImageDimensions(int(imageTensor.shape[2]), int(imageTensor.shape[1]), "sampleImageSaccadeSequences")
	if(modalityORimageSaccadesCrop):
		cropMarginX, cropMarginY = calculateImageSaccadeCropMargins(snapshotWidth, snapshotHeight)
		preparedImageWorkWidth = int(snapshotWidth) + (2*cropMarginX)
		preparedImageWorkHeight = int(snapshotHeight) + (2*cropMarginY)
		preparedImageTensor = prepareImageTensorForSaccades(imageTensor, preparedImageWorkWidth, preparedImageWorkHeight)
	else:
		preparedImageTensor = prepareImageTensorForSaccades(imageTensor, int(snapshotWidth), int(snapshotHeight))
		cropMarginX, cropMarginY = calculateImageSaccadeCropMarginsFromPreparedImage(preparedImageTensor, snapshotWidth, snapshotHeight)
	if(submodalityName=="image"):
		if(modalityORimageSaccadesUseAdjacentSalientRegions):
			saccadeOffsetPairs = GIAANNor_keypoints.sampleAdjacentSalientImageSaccadeOffsetPairs(preparedImageTensor, cropMarginX, cropMarginY, snapshotWidth, snapshotHeight)
			if(saccadeOffsetPairs is None):
				result = None
			else:
				for saccadeOffsetPair in saccadeOffsetPairs:
					sequence = generateImageSaccadeSequenceBetweenOffsets(preparedImageTensor, float(saccadeOffsetPair[0].item()), float(saccadeOffsetPair[1].item()), float(saccadeOffsetPair[2].item()), float(saccadeOffsetPair[3].item()), cropMarginX, cropMarginY, snapshotWidth, snapshotHeight)
					result.append(sequence)
		else:
			for _ in range(modalityORimageSaccadesPerImage):
				targetOffsetX, targetOffsetY = sampleRandomImageSaccadeOffset(cropMarginX, cropMarginY)
				sequence = generateImageSaccadeSequence(preparedImageTensor, targetOffsetX, targetOffsetY, cropMarginX, cropMarginY, snapshotWidth, snapshotHeight)
				result.append(sequence)
	else:
		for _ in range(modalityORimageSaccadesPerImage):
			targetOffsetX, targetOffsetY = sampleRandomImageSaccadeOffset(cropMarginX, cropMarginY)
			sequence = generateImageSaccadeSequence(preparedImageTensor, targetOffsetX, targetOffsetY, cropMarginX, cropMarginY, snapshotWidth, snapshotHeight)
			result.append(sequence)
	return result


def validateImageSaccadeEncodingParameters():
	result = None
	if(not isinstance(modalityORimageSaccadesEncode, bool)):
		raise RuntimeError("validateImageSaccadeEncodingParameters error: modalityORimageSaccadesEncode must be a bool")
	if(modalityORimageSnapshotsPerSaccade <= 0):
		raise RuntimeError("validateImageSaccadeEncodingParameters error: modalityORimageSnapshotsPerSaccade must be > 0")
	if(not modalityORimageSaccadesEncode and modalityORimageSnapshotsPerSaccade != 1):
		raise RuntimeError("validateImageSaccadeEncodingParameters error: modalityORimageSnapshotsPerSaccade must equal 1 when modalityORimageSaccadesEncode is False")
	if(modalityORimageSaccadesEncode and modalityORimageSnapshotsPerSaccade < 2):
		raise RuntimeError("validateImageSaccadeEncodingParameters error: modalityORimageSnapshotsPerSaccade must be >= 2 when modalityORimageSaccadesEncode is True")
	return result


def detectVideoFrameFeatureCoordinateList(frameTensor, articleIndex, sequenceCount):
	result = None
	frameIndex = None
	frameFeatureCoordinates = None
	snapshotWidth = None
	snapshotHeight = None
	if(submodalityName=="video" and modalityORvideoGenerateMultipleSnapshotsPerFrame):
		validateVideoFrameTensor(frameTensor, "detectVideoFrameFeatureCoordinateList")
		validateVideoFeatureDetectionProgressInputs(articleIndex, sequenceCount)
		snapshotWidth, snapshotHeight = GIAANNor_snapshotDimensions.calculateSnapshotDimensionsFromImageDimensions(int(frameTensor.shape[3]), int(frameTensor.shape[2]), "detectVideoFrameFeatureCoordinateList")
		result = []
		for frameIndex in range(int(frameTensor.shape[0])):
			printVideoFeatureDetectionSequenceIterationProgress(articleIndex, sequenceCount, frameIndex, int(frameTensor.shape[0]))
			frameFeatureCoordinates = GIAANNor_keypoints.detectReachableSalientImageFeatureCoordinates(frameTensor[frameIndex], snapshotWidth, snapshotHeight)
			result.append(frameFeatureCoordinates)
	else:
		raise RuntimeError("detectVideoFrameFeatureCoordinateList error: requires submodalityName=='video' and modalityORvideoGenerateMultipleSnapshotsPerFrame")
	return result


def generateVideoSnapshotSubsequencesParallel(frameTensor, alignedFeatureCoordinates):
	result = None
	startXTensor = None
	startYTensor = None
	numberOfFrames = None
	numberOfSubsequences = None
	snapshotHeight = None
	snapshotWidth = None
	frameIndexTensor = None
	yIndexTensor = None
	xIndexTensor = None
	rowIndexTensor = None
	columnIndexTensor = None
	snapshotTensor = None
	if(submodalityName=="video" and modalityORvideoGenerateMultipleSnapshotsPerFrame and modalityORvideoGenerateMultipleSnapshotsPerFrameParallel):
		validateVideoSnapshotSubsequenceInputs(frameTensor, alignedFeatureCoordinates, "generateVideoSnapshotSubsequencesParallel")
		startXTensor, startYTensor = calculateVideoSnapshotStartCoordinateTensors(alignedFeatureCoordinates, int(frameTensor.shape[3]), int(frameTensor.shape[2]))
		numberOfFrames = int(frameTensor.shape[0])
		numberOfSubsequences = int(alignedFeatureCoordinates.shape[1])
		snapshotWidth, snapshotHeight = GIAANNor_snapshotDimensions.calculateSnapshotDimensionsFromImageDimensions(int(frameTensor.shape[3]), int(frameTensor.shape[2]), "generateVideoSnapshotSubsequencesParallel")
		frameIndexTensor = pt.arange(numberOfFrames, device=frameTensor.device).view(numberOfFrames, 1, 1, 1).expand(numberOfFrames, numberOfSubsequences, snapshotHeight, snapshotWidth)
		rowIndexTensor = pt.arange(snapshotHeight, device=frameTensor.device).view(1, 1, snapshotHeight, 1)
		columnIndexTensor = pt.arange(snapshotWidth, device=frameTensor.device).view(1, 1, 1, snapshotWidth)
		yIndexTensor = startYTensor.view(numberOfFrames, numberOfSubsequences, 1, 1) + rowIndexTensor
		xIndexTensor = startXTensor.view(numberOfFrames, numberOfSubsequences, 1, 1) + columnIndexTensor
		snapshotTensor = frameTensor[frameIndexTensor, :, yIndexTensor, xIndexTensor]
		result = snapshotTensor.permute(1, 0, 4, 2, 3).contiguous()
	else:
		raise RuntimeError("generateVideoSnapshotSubsequencesParallel error: requires submodalityName=='video', modalityORvideoGenerateMultipleSnapshotsPerFrame, and modalityORvideoGenerateMultipleSnapshotsPerFrameParallel")
	return result


def generateVideoSnapshotSubsequencesSequential(frameTensor, alignedFeatureCoordinates):
	result = None
	startXTensor = None
	startYTensor = None
	numberOfFrames = None
	numberOfSubsequences = None
	snapshotHeight = None
	snapshotWidth = None
	subsequenceIndex = None
	if(submodalityName=="video" and modalityORvideoGenerateMultipleSnapshotsPerFrame and not modalityORvideoGenerateMultipleSnapshotsPerFrameParallel):
		validateVideoSnapshotSubsequenceInputs(frameTensor, alignedFeatureCoordinates, "generateVideoSnapshotSubsequencesSequential")
		startXTensor, startYTensor = calculateVideoSnapshotStartCoordinateTensors(alignedFeatureCoordinates, int(frameTensor.shape[3]), int(frameTensor.shape[2]))
		numberOfFrames = int(frameTensor.shape[0])
		numberOfSubsequences = int(alignedFeatureCoordinates.shape[1])
		snapshotWidth, snapshotHeight = GIAANNor_snapshotDimensions.calculateSnapshotDimensionsFromImageDimensions(int(frameTensor.shape[3]), int(frameTensor.shape[2]), "generateVideoSnapshotSubsequencesSequential")
		result = pt.empty((numberOfSubsequences, numberOfFrames, 3, snapshotHeight, snapshotWidth), dtype=frameTensor.dtype, device=frameTensor.device)
		for subsequenceIndex in range(numberOfSubsequences):
			result[subsequenceIndex] = generateVideoSnapshotSubsequence(frameTensor, startXTensor[:, subsequenceIndex], startYTensor[:, subsequenceIndex])
	else:
		raise RuntimeError("generateVideoSnapshotSubsequencesSequential error: requires submodalityName=='video', modalityORvideoGenerateMultipleSnapshotsPerFrame, and not modalityORvideoGenerateMultipleSnapshotsPerFrameParallel")
	return result


def generateVideoSnapshotSubsequence(frameTensor, startXTensor, startYTensor):
	result = None
	numberOfFrames = None
	snapshotHeight = None
	snapshotWidth = None
	frameIndex = None
	startX = None
	startY = None
	endX = None
	endY = None
	if(submodalityName=="video" and modalityORvideoGenerateMultipleSnapshotsPerFrame):
		validateVideoFrameTensor(frameTensor, "generateVideoSnapshotSubsequence")
		if(not pt.is_tensor(startXTensor) or not pt.is_tensor(startYTensor)):
			raise RuntimeError("generateVideoSnapshotSubsequence error: startXTensor/startYTensor must be tensors")
		if(startXTensor.dim() != 1 or startYTensor.dim() != 1):
			raise RuntimeError("generateVideoSnapshotSubsequence error: startXTensor/startYTensor rank must be 1")
		numberOfFrames = int(frameTensor.shape[0])
		if(int(startXTensor.shape[0]) != numberOfFrames or int(startYTensor.shape[0]) != numberOfFrames):
			raise RuntimeError("generateVideoSnapshotSubsequence error: startXTensor/startYTensor length mismatch")
		snapshotWidth, snapshotHeight = GIAANNor_snapshotDimensions.calculateSnapshotDimensionsFromImageDimensions(int(frameTensor.shape[3]), int(frameTensor.shape[2]), "generateVideoSnapshotSubsequence")
		result = pt.empty((numberOfFrames, 3, snapshotHeight, snapshotWidth), dtype=frameTensor.dtype, device=frameTensor.device)
		for frameIndex in range(numberOfFrames):
			startX = int(startXTensor[frameIndex].item())
			startY = int(startYTensor[frameIndex].item())
			endX = startX + snapshotWidth
			endY = startY + snapshotHeight
			if(startX < 0 or startY < 0 or endX > int(frameTensor.shape[3]) or endY > int(frameTensor.shape[2])):
				raise RuntimeError("generateVideoSnapshotSubsequence error: computed snapshot window exceeds frameTensor bounds")
			result[frameIndex] = frameTensor[frameIndex, :, startY:endY, startX:endX]
	else:
		raise RuntimeError("generateVideoSnapshotSubsequence error: requires submodalityName=='video' and modalityORvideoGenerateMultipleSnapshotsPerFrame")
	return result


def generateImageSaccadeSequenceBetweenOffsets(preparedImageTensor, startOffsetX, startOffsetY, endOffsetX, endOffsetY, cropMarginX, cropMarginY, snapshotWidth, snapshotHeight):
	result = None
	snapshotSequenceTensor = None
	workHeight = None
	workWidth = None
	snapshotIndexFraction = None
	snapshotOffsetX = None
	snapshotOffsetY = None
	startX = None
	startY = None
	endX = None
	endY = None
	snapshotTensor = None
	if(not pt.is_tensor(preparedImageTensor)):
		raise RuntimeError("generateImageSaccadeSequenceBetweenOffsets error: preparedImageTensor must be a tensor")
	if(preparedImageTensor.dim() != 3):
		raise RuntimeError("generateImageSaccadeSequenceBetweenOffsets error: preparedImageTensor rank must be 3")
	if(preparedImageTensor.shape[0] != 3):
		raise RuntimeError("generateImageSaccadeSequenceBetweenOffsets error: preparedImageTensor channel count must be 3")
	if(snapshotWidth <= 0 or snapshotHeight <= 0):
		raise RuntimeError("generateImageSaccadeSequenceBetweenOffsets error: snapshotWidth/snapshotHeight must be > 0")
	workHeight = int(preparedImageTensor.shape[1])
	workWidth = int(preparedImageTensor.shape[2])
	snapshotSequenceTensor = pt.zeros((modalityORimageSnapshotsPerSaccade, 3, snapshotHeight, snapshotWidth), dtype=arrayType, device=deviceDense)
	for snapshotIndex in range(modalityORimageSnapshotsPerSaccade):
		snapshotIndexFraction = calculateImageSaccadeSnapshotIndexFraction(snapshotIndex)
		snapshotOffsetX = float(startOffsetX) + (float(endOffsetX - startOffsetX)*snapshotIndexFraction)
		snapshotOffsetY = float(startOffsetY) + (float(endOffsetY - startOffsetY)*snapshotIndexFraction)
		startX = int(cropMarginX) + int(round(snapshotOffsetX))
		startY = int(cropMarginY) + int(round(snapshotOffsetY))
		endX = startX + int(snapshotWidth)
		endY = startY + int(snapshotHeight)
		if(startX < 0 or startY < 0 or endX > workWidth or endY > workHeight):
			raise RuntimeError("generateImageSaccadeSequenceBetweenOffsets error: computed crop window exceeds preparedImageTensor bounds")
		snapshotTensor = preparedImageTensor[:, startY:endY, startX:endX]
		if(snapshotTensor.shape[1] != snapshotHeight or snapshotTensor.shape[2] != snapshotWidth):
			raise RuntimeError("generateImageSaccadeSequenceBetweenOffsets error: snapshotTensor shape mismatch")
		snapshotSequenceTensor[snapshotIndex] = snapshotTensor
	result = snapshotSequenceTensor
	return result


def generateImageSaccadeSequence(preparedImageTensor, targetOffsetX, targetOffsetY, cropMarginX, cropMarginY, snapshotWidth, snapshotHeight):
	# for each saccade (sequence) generate modalityORimageSnapshotsPerSaccade by taking snapshots along a linear pathway of the saccade offset:
	# crop each augmented snapshot by a predefined amount (dependent on modalityORimageSaccadesMaxAngularOffsetDegrees) so that every snapshot contains only image data (no pixels outside the original image data).
	result = None
	snapshotSequenceTensor = None
	workHeight = None
	workWidth = None
	snapshotIndexFraction = None
	snapshotOffsetX = None
	snapshotOffsetY = None
	startX = None
	startY = None
	endX = None
	endY = None
	snapshotTensor = None
	if(not pt.is_tensor(preparedImageTensor)):
		raise RuntimeError("generateImageSaccadeSequence error: preparedImageTensor must be a tensor")
	if(preparedImageTensor.dim() != 3):
		raise RuntimeError("generateImageSaccadeSequence error: preparedImageTensor rank must be 3")
	if(preparedImageTensor.shape[0] != 3):
		raise RuntimeError("generateImageSaccadeSequence error: preparedImageTensor channel count must be 3")
	if(snapshotWidth <= 0 or snapshotHeight <= 0):
		raise RuntimeError("generateImageSaccadeSequence error: snapshotWidth/snapshotHeight must be > 0")
	workHeight = int(preparedImageTensor.shape[1])
	workWidth = int(preparedImageTensor.shape[2])
	snapshotSequenceTensor = pt.zeros((modalityORimageSnapshotsPerSaccade, 3, snapshotHeight, snapshotWidth), dtype=arrayType, device=deviceDense)
	for snapshotIndex in range(modalityORimageSnapshotsPerSaccade):
		snapshotIndexFraction = calculateImageSaccadeSnapshotIndexFraction(snapshotIndex)
		snapshotOffsetX = int(round(float(targetOffsetX)*snapshotIndexFraction))
		snapshotOffsetY = int(round(float(targetOffsetY)*snapshotIndexFraction))
		startX = int(cropMarginX) + snapshotOffsetX
		startY = int(cropMarginY) + snapshotOffsetY
		endX = startX + int(snapshotWidth)
		endY = startY + int(snapshotHeight)
		if(startX < 0 or startY < 0 or endX > workWidth or endY > workHeight):
			raise RuntimeError("generateImageSaccadeSequence error: computed crop window exceeds preparedImageTensor bounds")
		snapshotTensor = preparedImageTensor[:, startY:endY, startX:endX]
		if(snapshotTensor.shape[1] != snapshotHeight or snapshotTensor.shape[2] != snapshotWidth):
			raise RuntimeError("generateImageSaccadeSequence error: snapshotTensor shape mismatch")
		snapshotSequenceTensor[snapshotIndex] = snapshotTensor
	result = snapshotSequenceTensor
	return result


def calculateImageSaccadeSnapshotIndexFraction(snapshotIndex):
	result = None
	if(not isinstance(snapshotIndex, int)):
		raise RuntimeError("calculateImageSaccadeSnapshotIndexFraction error: snapshotIndex must be an int")
	if(snapshotIndex < 0 or snapshotIndex >= modalityORimageSnapshotsPerSaccade):
		raise RuntimeError("calculateImageSaccadeSnapshotIndexFraction error: snapshotIndex out of range")
	if(modalityORimageSnapshotsPerSaccade == 1):
		result = 1.0
	elif(modalityORimageSnapshotsPerSaccade == 2):
		result = float(snapshotIndex)
	else:
		result = float(snapshotIndex)/float(modalityORimageSnapshotsPerSaccade - 1)
	return result


def sampleRandomImageSaccadeOffset(cropMarginX, cropMarginY):
	# saccade augmentations are calculated by translating the image to a random polar coordinates offset from the centre (using modalityORimageSaccadesMaxAngularOffsetDegrees)
	result = None
	angleRadians = None
	radiusScale = None
	offsetX = None
	offsetY = None
	if(cropMarginX < 0 or cropMarginY < 0):
		raise RuntimeError("sampleRandomImageSaccadeOffset error: cropMarginX/cropMarginY must be >= 0")
	angleRadians = float(pt.rand(1).item())*(2.0*math.pi)
	radiusScale = float(pt.rand(1).item())
	offsetX = math.cos(angleRadians)*float(cropMarginX)*radiusScale
	offsetY = math.sin(angleRadians)*float(cropMarginY)*radiusScale
	result = (offsetX, offsetY)
	return result


def calculateImageSaccadeCropMargins(snapshotWidth, snapshotHeight):
	result = None
	angleRadians = None
	cropMarginX = None
	cropMarginY = None
	if(snapshotWidth <= 0 or snapshotHeight <= 0):
		raise RuntimeError("calculateImageSaccadeCropMargins error: snapshotWidth/snapshotHeight must be > 0")
	if(modalityORimageSaccadesMaxAngularOffsetDegrees < 0 or modalityORimageSaccadesMaxAngularOffsetDegrees >= 90):
		raise RuntimeError("calculateImageSaccadeCropMargins error: modalityORimageSaccadesMaxAngularOffsetDegrees must be >= 0 and < 90")
	angleRadians = math.radians(float(modalityORimageSaccadesMaxAngularOffsetDegrees))
	cropMarginX = int(math.ceil((float(snapshotWidth)/2.0)*math.tan(angleRadians)))
	cropMarginY = int(math.ceil((float(snapshotHeight)/2.0)*math.tan(angleRadians)))
	result = (cropMarginX, cropMarginY)
	return result


def calculateImageSaccadeCropMarginsFromPreparedImage(preparedImageTensor, snapshotWidth, snapshotHeight):
	result = None
	workHeight = None
	workWidth = None
	cropMarginX = None
	cropMarginY = None
	if(not pt.is_tensor(preparedImageTensor)):
		raise RuntimeError("calculateImageSaccadeCropMarginsFromPreparedImage error: preparedImageTensor must be a tensor")
	if(preparedImageTensor.dim() != 3):
		raise RuntimeError("calculateImageSaccadeCropMarginsFromPreparedImage error: preparedImageTensor rank must be 3")
	if(preparedImageTensor.shape[0] != 3):
		raise RuntimeError("calculateImageSaccadeCropMarginsFromPreparedImage error: preparedImageTensor channel count must be 3")
	if(snapshotWidth <= 0 or snapshotHeight <= 0):
		raise RuntimeError("calculateImageSaccadeCropMarginsFromPreparedImage error: snapshotWidth/snapshotHeight must be > 0")
	workHeight = int(preparedImageTensor.shape[1])
	workWidth = int(preparedImageTensor.shape[2])
	if(workWidth < int(snapshotWidth) or workHeight < int(snapshotHeight)):
		raise RuntimeError("calculateImageSaccadeCropMarginsFromPreparedImage error: prepared image must be at least as large as snapshotWidth/snapshotHeight")
	cropMarginX = int((float(workWidth) - float(snapshotWidth))/2.0)
	cropMarginY = int((float(workHeight) - float(snapshotHeight))/2.0)
	result = (cropMarginX, cropMarginY)
	return result


def prepareImageTensorForSaccades(imageTensor, workWidth, workHeight):
	result = None
	imageHeight = None
	imageWidth = None
	cropStartX = None
	cropStartY = None
	if(workWidth <= 0 or workHeight <= 0):
		raise RuntimeError("prepareImageTensorForSaccades error: workWidth/workHeight must be > 0")
	if(not pt.is_tensor(imageTensor)):
		raise RuntimeError("prepareImageTensorForSaccades error: imageTensor must be a tensor")
	if(imageTensor.dim() != 3):
		raise RuntimeError("prepareImageTensorForSaccades error: imageTensor rank must be 3")
	if(int(imageTensor.shape[0]) != 3):
		raise RuntimeError("prepareImageTensorForSaccades error: imageTensor channel count must equal 3")
	imageHeight = int(imageTensor.shape[1])
	imageWidth = int(imageTensor.shape[2])
	if(imageWidth < workWidth or imageHeight < workHeight):
		raise RuntimeError("prepareImageTensorForSaccades error: input image must cover workWidth/workHeight without resizing")
	if(modalityORimageSaccadesCrop):
		cropStartX = int((imageWidth - workWidth)//2)
		cropStartY = int((imageHeight - workHeight)//2)
		result = imageTensor[:, cropStartY:cropStartY + workHeight, cropStartX:cropStartX + workWidth].contiguous()
		if(result.shape[1] != workHeight or result.shape[2] != workWidth):
			raise RuntimeError("prepareImageTensorForSaccades error: prepared image crop shape mismatch")
	else:
		result = imageTensor.contiguous()
		if(result.shape[1] < workHeight or result.shape[2] < workWidth):
			raise RuntimeError("prepareImageTensorForSaccades error: prepared image shape must cover workWidth/workHeight")
	return result


def validateVideoFeatureDetectionProgressInputs(articleIndex, sequenceCount):
	result = None
	if(submodalityName=="video" and modalityORvideoGenerateMultipleSnapshotsPerFrame):
		if(not isinstance(articleIndex, int)):
			raise RuntimeError("validateVideoFeatureDetectionProgressInputs error: articleIndex must be an int")
		if(articleIndex < 0):
			raise RuntimeError("validateVideoFeatureDetectionProgressInputs error: articleIndex must be >= 0")
		if(not isinstance(sequenceCount, int)):
			raise RuntimeError("validateVideoFeatureDetectionProgressInputs error: sequenceCount must be an int")
		if(sequenceCount < 0):
			raise RuntimeError("validateVideoFeatureDetectionProgressInputs error: sequenceCount must be >= 0")
	else:
		raise RuntimeError("validateVideoFeatureDetectionProgressInputs error: requires submodalityName=='video' and modalityORvideoGenerateMultipleSnapshotsPerFrame")
	return result


def printVideoFeatureDetectionSequenceIterationProgress(articleIndex, sequenceCount, frameIndex, numberOfFrames):
	result = None
	if(submodalityName=="video" and modalityORvideoGenerateMultipleSnapshotsPerFrame):
		if(printTrainSequenceDefault):
			if(frameIndex < 0 or numberOfFrames <= 0 or frameIndex >= numberOfFrames):
				raise RuntimeError("printVideoFeatureDetectionSequenceIterationProgress error: frameIndex/numberOfFrames out of range")
			print(f"Processing sequenceCount: {sequenceCount}, articleIndex={articleIndex}, videoSequenceIteration={frameIndex}/{numberOfFrames}")
	else:
		raise RuntimeError("printVideoFeatureDetectionSequenceIterationProgress error: requires submodalityName=='video' and modalityORvideoGenerateMultipleSnapshotsPerFrame")
	return result


def calculateVideoSnapshotStartCoordinateTensors(alignedFeatureCoordinates, frameWidth, frameHeight):
	result = None
	startXTensor = None
	startYTensor = None
	snapshotWidth = None
	snapshotHeight = None
	if(submodalityName=="video" and modalityORvideoGenerateMultipleSnapshotsPerFrame):
		if(not pt.is_tensor(alignedFeatureCoordinates)):
			raise RuntimeError("calculateVideoSnapshotStartCoordinateTensors error: alignedFeatureCoordinates must be a tensor")
		if(alignedFeatureCoordinates.dim() != 3):
			raise RuntimeError("calculateVideoSnapshotStartCoordinateTensors error: alignedFeatureCoordinates rank must be 3")
		if(int(alignedFeatureCoordinates.shape[2]) != 2):
			raise RuntimeError("calculateVideoSnapshotStartCoordinateTensors error: alignedFeatureCoordinates last dimension must equal 2")
		if(frameWidth <= 0 or frameHeight <= 0):
			raise RuntimeError("calculateVideoSnapshotStartCoordinateTensors error: frameWidth/frameHeight must be > 0")
		snapshotWidth, snapshotHeight = GIAANNor_snapshotDimensions.calculateSnapshotDimensionsFromImageDimensions(frameWidth, frameHeight, "calculateVideoSnapshotStartCoordinateTensors")
		if(snapshotWidth <= 0 or snapshotHeight <= 0):
			raise RuntimeError("calculateVideoSnapshotStartCoordinateTensors error: snapshotWidth/snapshotHeight must be > 0")
		startXTensor = pt.round(alignedFeatureCoordinates[:, :, 0] - (float(snapshotWidth)/2.0)).to(dtype=pt.long)
		startYTensor = pt.round(alignedFeatureCoordinates[:, :, 1] - (float(snapshotHeight)/2.0)).to(dtype=pt.long)
		if(bool(pt.any(startXTensor < 0).item()) or bool(pt.any(startYTensor < 0).item())):
			raise RuntimeError("calculateVideoSnapshotStartCoordinateTensors error: computed snapshot start is outside frame bounds")
		if(bool(pt.any(startXTensor + snapshotWidth > frameWidth).item()) or bool(pt.any(startYTensor + snapshotHeight > frameHeight).item())):
			raise RuntimeError("calculateVideoSnapshotStartCoordinateTensors error: computed snapshot end is outside frame bounds")
		result = startXTensor, startYTensor
	else:
		raise RuntimeError("calculateVideoSnapshotStartCoordinateTensors error: requires submodalityName=='video' and modalityORvideoGenerateMultipleSnapshotsPerFrame")
	return result


def validateVideoSnapshotSubsequenceInputs(frameTensor, alignedFeatureCoordinates, functionName):
	result = None
	if(submodalityName=="video" and modalityORvideoGenerateMultipleSnapshotsPerFrame):
		if(functionName == ""):
			raise RuntimeError("validateVideoSnapshotSubsequenceInputs error: functionName must not be empty")
		validateVideoFrameTensor(frameTensor, functionName)
		if(not pt.is_tensor(alignedFeatureCoordinates)):
			raise RuntimeError(functionName + " error: alignedFeatureCoordinates must be a tensor")
		if(alignedFeatureCoordinates.dim() != 3):
			raise RuntimeError(functionName + " error: alignedFeatureCoordinates rank must be 3")
		if(int(alignedFeatureCoordinates.shape[0]) != int(frameTensor.shape[0])):
			raise RuntimeError(functionName + " error: alignedFeatureCoordinates frame count must equal frameTensor frame count")
		if(int(alignedFeatureCoordinates.shape[1]) <= 0):
			raise RuntimeError(functionName + " error: alignedFeatureCoordinates subsequence count must be > 0")
		if(int(alignedFeatureCoordinates.shape[2]) != 2):
			raise RuntimeError(functionName + " error: alignedFeatureCoordinates last dimension must equal 2")
	else:
		raise RuntimeError("validateVideoSnapshotSubsequenceInputs error: requires submodalityName=='video' and modalityORvideoGenerateMultipleSnapshotsPerFrame")
	return result


def validateVideoFrameTensor(frameTensor, functionName):
	result = None
	if(submodalityName=="video" and modalityORvideoGenerateMultipleSnapshotsPerFrame):
		if(functionName == ""):
			raise RuntimeError("validateVideoFrameTensor error: functionName must not be empty")
		if(not pt.is_tensor(frameTensor)):
			raise RuntimeError(functionName + " error: frameTensor must be a tensor")
		if(frameTensor.dim() != 4):
			raise RuntimeError(functionName + " error: frameTensor rank must be 4")
		if(int(frameTensor.shape[0]) <= 0):
			raise RuntimeError(functionName + " error: frameTensor frame count must be > 0")
		if(int(frameTensor.shape[1]) != 3):
			raise RuntimeError(functionName + " error: frameTensor channel count must equal 3")
		GIAANNor_snapshotDimensions.calculateSnapshotDimensionsFromImageDimensions(int(frameTensor.shape[3]), int(frameTensor.shape[2]), functionName)
	else:
		raise RuntimeError("validateVideoFrameTensor error: requires submodalityName=='video' and modalityORvideoGenerateMultipleSnapshotsPerFrame")
	return result
