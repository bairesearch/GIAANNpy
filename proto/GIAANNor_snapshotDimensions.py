"""GIAANNor_snapshotDimensions.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN OR snapshot dimensions

"""

from GIAANNcmn_globalDefs import *


def calculateSnapshotDimensionsFromImageDimensions(imageWidth, imageHeight, functionName):
	result = None
	snapshotWidth = None
	snapshotHeight = None
	snapshotDimension = None
	if(functionName == ""):
		raise RuntimeError("calculateSnapshotDimensionsFromImageDimensions error: functionName must not be empty")
	validateSnapshotImageDimensions(imageWidth, imageHeight, functionName)
	validateSnapshotFraction(functionName)
	snapshotDimension = calculateSquareSnapshotDimension(imageWidth, imageHeight, functionName)
	snapshotWidth = snapshotDimension
	snapshotHeight = snapshotDimension
	if(snapshotWidth > imageWidth or snapshotHeight > imageHeight):
		raise RuntimeError(functionName + " error: calculated snapshot dimensions exceed image dimensions")
	result = snapshotWidth, snapshotHeight
	return result


def validateSnapshotImageDimensions(imageWidth, imageHeight, functionName):
	result = None
	if(not isinstance(imageWidth, int) or not isinstance(imageHeight, int)):
		raise RuntimeError(functionName + " error: imageWidth/imageHeight must be ints")
	if(imageWidth <= 0 or imageHeight <= 0):
		raise RuntimeError(functionName + " error: imageWidth/imageHeight must be > 0")
	return result


def validateSnapshotFraction(functionName):
	result = None
	if(not isinstance(modalityORsnapshotFractionOfImage, int) and not isinstance(modalityORsnapshotFractionOfImage, float)):
		raise RuntimeError(functionName + " error: modalityORsnapshotFractionOfImage must be an int or float")
	if(modalityORsnapshotFractionOfImage <= 0.0 or modalityORsnapshotFractionOfImage > 1.0):
		raise RuntimeError(functionName + " error: modalityORsnapshotFractionOfImage must be > 0.0 and <= 1.0")
	return result


def calculateSquareSnapshotDimension(imageWidth, imageHeight, functionName):
	result = None
	imageDimension = None
	snapshotDimensionFloat = None
	snapshotDimension = None
	imageDimension = min(imageWidth, imageHeight)
	snapshotDimensionFloat = float(imageDimension)*float(modalityORsnapshotFractionOfImage)
	if(snapshotDimensionFloat < 1.0):
		raise RuntimeError(functionName + " error: modalityORsnapshotFractionOfImage produces a snapshot dimension < 1 pixel")
	snapshotDimension = int(round(snapshotDimensionFloat))
	if(snapshotDimension <= 0):
		raise RuntimeError(functionName + " error: calculated snapshot dimension must be > 0")
	result = snapshotDimension
	return result
