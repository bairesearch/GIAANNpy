"""GIAANNor_sequenceSaccades.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 BAI Research Pty Ltd (bairesearch.com.au)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN OR sequence Saccades

"""

import math
import torch as pt

from GIAANNcmn_globalDefs import *
import GIAANNor_keypoints
import GIAANNor_snapshotDimensions


def sampleImageSaccadeSequences(imageTensor):
	# generate sequence data by augmenting each image:
	# for each image, generate up to modalityORimageMaxSequencesPerImage sequences by performing saccade augmentations:
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
	if(modalityORimageMaxSequencesPerImage <= 0):
		raise RuntimeError("sampleImageSaccadeSequences error: modalityORimageMaxSequencesPerImage must be > 0")
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
			for _ in range(modalityORimageMaxSequencesPerImage):
				targetOffsetX, targetOffsetY = sampleRandomImageSaccadeOffset(cropMarginX, cropMarginY)
				sequence = generateImageSaccadeSequence(preparedImageTensor, targetOffsetX, targetOffsetY, cropMarginX, cropMarginY, snapshotWidth, snapshotHeight)
				result.append(sequence)
	else:
		for _ in range(modalityORimageMaxSequencesPerImage):
			targetOffsetX, targetOffsetY = sampleRandomImageSaccadeOffset(cropMarginX, cropMarginY)
			sequence = generateImageSaccadeSequence(preparedImageTensor, targetOffsetX, targetOffsetY, cropMarginX, cropMarginY, snapshotWidth, snapshotHeight)
			result.append(sequence)
	return result


def validateImageSaccadeEncodingParameters():
	result = None
	if(not isinstance(modalityORimageSequenceEncode, str)):
		raise RuntimeError("validateImageSaccadeEncodingParameters error: modalityORimageSequenceEncode must be a string")
	if(modalityORimageSequenceEncode!="saccades" and modalityORimageSequenceEncode!="distance" and modalityORimageSequenceEncode!="axis" and modalityORimageSequenceEncode!="axes" and modalityORimageSequenceEncode!="none"):
		raise RuntimeError("validateImageSaccadeEncodingParameters error: modalityORimageSequenceEncode must be 'saccades', 'distance', 'axis', 'axes', or 'none'")
	if(not isinstance(modalityORimageSaccadeKeypointsPerEncoding, int) or isinstance(modalityORimageSaccadeKeypointsPerEncoding, bool)):
		raise RuntimeError("validateImageSaccadeEncodingParameters error: modalityORimageSaccadeKeypointsPerEncoding must be an int")
	if(modalityORimageSaccadeKeypointsPerEncoding <= 0):
		raise RuntimeError("validateImageSaccadeEncodingParameters error: modalityORimageSaccadeKeypointsPerEncoding must be > 0")
	if(not isinstance(modalityORimageSnapshotsPerSaccade, int) or isinstance(modalityORimageSnapshotsPerSaccade, bool)):
		raise RuntimeError("validateImageSaccadeEncodingParameters error: modalityORimageSnapshotsPerSaccade must be an int")
	if(modalityORimageSnapshotsPerSaccade < 0):
		raise RuntimeError("validateImageSaccadeEncodingParameters error: modalityORimageSnapshotsPerSaccade must be >= 0")
	if((modalityORimageSequenceEncode=="distance" or modalityORimageSequenceEncode=="axis" or modalityORimageSequenceEncode=="axes" or modalityORimageSequenceEncode=="none") and modalityORimageSaccadeKeypointsPerEncoding != 1):
		raise RuntimeError("validateImageSaccadeEncodingParameters error: modalityORimageSaccadeKeypointsPerEncoding must equal 1 when modalityORimageSequenceEncode is 'distance', 'axis', 'axes', or 'none'")
	if((modalityORimageSequenceEncode=="distance" or modalityORimageSequenceEncode=="axis" or modalityORimageSequenceEncode=="axes" or modalityORimageSequenceEncode=="none") and modalityORimageSnapshotsPerSaccade != 0):
		raise RuntimeError("validateImageSaccadeEncodingParameters error: modalityORimageSnapshotsPerSaccade must equal 0 when modalityORimageSequenceEncode is 'distance', 'axis', 'axes', or 'none'")
	if(modalityORimageSequenceEncode=="saccades" and modalityORimageSnapshotsPerSaccade < 1):
		raise RuntimeError("validateImageSaccadeEncodingParameters error: modalityORimageSnapshotsPerSaccade must be >= 1 when modalityORimageSequenceEncode is 'saccades'")
	return result


def configureTrainConnectionsForImageSaccadeEncoding(sequenceObservedColumns):
	result = None
	if(submodalityName=="image" and modalityORimageSequenceEncode=="saccades"):
		pass
	else:
		raise RuntimeError("configureTrainConnectionsForImageSaccadeEncoding error: requires submodalityName=='image' and modalityORimageSequenceEncode=='saccades'")
	return result


def generateImageSaccadeSequenceBetweenOffsets(preparedImageTensor, startOffsetX, startOffsetY, endOffsetX, endOffsetY, cropMarginX, cropMarginY, snapshotWidth, snapshotHeight):
	result = None
	snapshotSequenceTensor = None
	snapshotsPerEncoding = None
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
	snapshotsPerEncoding = calculateImageSaccadeSnapshotsPerEncoding()
	snapshotSequenceTensor = pt.zeros((snapshotsPerEncoding, 3, snapshotHeight, snapshotWidth), dtype=arrayType, device=deviceDense)
	for snapshotIndex in range(snapshotsPerEncoding):
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
	# for each saccade (sequence) generate snapshots by taking snapshots along a linear pathway of the saccade offset:
	# crop each augmented snapshot by a predefined amount (dependent on modalityORimageSaccadesMaxAngularOffsetDegrees) so that every snapshot contains only image data (no pixels outside the original image data).
	result = None
	snapshotSequenceTensor = None
	snapshotsPerEncoding = None
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
	snapshotsPerEncoding = calculateImageSaccadeSnapshotsPerEncoding()
	snapshotSequenceTensor = pt.zeros((snapshotsPerEncoding, 3, snapshotHeight, snapshotWidth), dtype=arrayType, device=deviceDense)
	for snapshotIndex in range(snapshotsPerEncoding):
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


def calculateImageSaccadeSnapshotsPerEncoding():
	# total snaphots per column feature neuron encoding = modalityORimageSaccadeKeypointsPerEncoding*modalityORimageSnapshotsPerSaccade+1.
	result = None
	validateImageSaccadeEncodingParameters()
	result = int(modalityORimageSaccadeKeypointsPerEncoding)*int(modalityORimageSnapshotsPerSaccade) + 1
	return result


def calculateImageSaccadeSnapshotIndexFraction(snapshotIndex):
	result = None
	snapshotsPerEncoding = None
	if(not isinstance(snapshotIndex, int)):
		raise RuntimeError("calculateImageSaccadeSnapshotIndexFraction error: snapshotIndex must be an int")
	snapshotsPerEncoding = calculateImageSaccadeSnapshotsPerEncoding()
	if(snapshotIndex < 0 or snapshotIndex >= snapshotsPerEncoding):
		raise RuntimeError("calculateImageSaccadeSnapshotIndexFraction error: snapshotIndex out of range")
	if(snapshotsPerEncoding == 1):
		result = 1.0
	else:
		result = float(snapshotIndex)/float(snapshotsPerEncoding - 1)
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
