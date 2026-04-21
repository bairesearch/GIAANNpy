"""GIAANNor_RFfilters.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN OR receptive field filters

"""

import importlib
import math
import torch as pt

from GIAANNcmn_globalDefs import *


class ORRFfilters:
	def __init__(self, filterTensor, filterCodeList, filterWordList):
		self.filterTensor = filterTensor
		self.filterCodeList = filterCodeList
		self.filterWordList = filterWordList
		self.filterIndexToCodeDict = dict(enumerate(filterCodeList))


def initialiseRFfilters():
	# initialise a set of receptive field (RF) filters:
	result = None
	if(modalityORuseExternalRFfilterLibrary):
		result = initialiseRFfiltersExternal()
	else:
		result = initialiseRFfiltersInternal()
	return result


def initialiseRFfiltersExternal():
	result = None
	module = None
	module = importlib.import_module(modalityORexternalRFfilterLibraryModuleName)
	if(hasattr(module, "initialiseRFfilters")):
		externalResult = module.initialiseRFfilters(modalityORpixelsPerColumn)
	elif(hasattr(module, "createRFfilters")):
		externalResult = module.createRFfilters(modalityORpixelsPerColumn)
	else:
		raise RuntimeError("initialiseRFfiltersExternal error: external RF filter library must expose initialiseRFfilters(filterSize) or createRFfilters(filterSize)")
	result = adaptExternalRFfilters(externalResult)
	return result


def adaptExternalRFfilters(externalResult):
	result = None
	if(isinstance(externalResult, ORRFfilters)):
		result = externalResult
	elif(isinstance(externalResult, dict)):
		if("filterTensor" not in externalResult):
			raise RuntimeError("adaptExternalRFfilters error: externalResult dict missing filterTensor")
		filterTensor = externalResult["filterTensor"]
		filterCodeList = externalResult.get("filterCodeList")
		filterWordList = externalResult.get("filterWordList")
		if(filterCodeList is None):
			filterCodeList = buildDefaultRFfilterCodeList(filterTensor.shape[0])
		result = createRFfilterContainer(filterTensor, filterCodeList, filterWordList)
	elif(pt.is_tensor(externalResult)):
		filterCodeList = buildDefaultRFfilterCodeList(externalResult.shape[0])
		result = createRFfilterContainer(externalResult, filterCodeList, None)
	else:
		raise RuntimeError("adaptExternalRFfilters error: unsupported externalResult type")
	return result


def initialiseRFfiltersInternal():
	# the natural RF filters can be a combination of ellipsoidal and gabor filters.
	result = None
	filterTensor = None
	filterCodeList = []
	filterTensor, filterCodeList = buildInternalRFfilterBank()
	result = createRFfilterContainer(filterTensor, filterCodeList)
	return result


def buildInternalRFfilterBank():
	resultFilterTensor = None
	resultFilterCodeList = []
	ellipsoidalFilters, ellipsoidalCodes = buildEllipsoidalRFfilters()
	gaborFilters, gaborCodes = buildGaborRFfilters()
	resultFilterTensor = pt.cat((ellipsoidalFilters, gaborFilters), dim=0)
	resultFilterCodeList = ellipsoidalCodes + gaborCodes
	return resultFilterTensor, resultFilterCodeList


def buildEllipsoidalRFfilters():
	filterTensor = None
	filterCodeList = []
	filterList = []
	size = modalityORpixelsPerColumn
	yGrid, xGrid = createRotationGrid(size)
	orientationList = [0.0, math.pi/4.0, math.pi/2.0, 3.0*math.pi/4.0]
	colourWeightsList = buildColourWeights()
	for orientationIndex, orientation in enumerate(orientationList):
		xRot, yRot = rotateGrid(xGrid, yGrid, orientation)
		for polarityIndex, polarity in enumerate([1.0, -1.0]):
			baseKernel = createEllipsoidalKernel(xRot, yRot, polarity)
			for colourIndex, colourWeights in enumerate(colourWeightsList):
				filterList.append(applyColourWeights(baseKernel, colourWeights))
				filterCodeList.append(buildRFfilterCode("ELL", orientationIndex, polarityIndex, colourIndex, 0))
	filterTensor = stackRFfilters(filterList)
	return filterTensor, filterCodeList


def buildGaborRFfilters():
	filterTensor = None
	filterCodeList = []
	filterList = []
	size = modalityORpixelsPerColumn
	yGrid, xGrid = createRotationGrid(size)
	orientationList = [0.0, math.pi/4.0, math.pi/2.0, 3.0*math.pi/4.0]
	frequencyList = [0.15, 0.25]
	phaseList = [0.0, math.pi/2.0]
	colourWeightsList = buildColourWeights()
	for orientationIndex, orientation in enumerate(orientationList):
		xRot, yRot = rotateGrid(xGrid, yGrid, orientation)
		for frequencyIndex, frequency in enumerate(frequencyList):
			for phaseIndex, phase in enumerate(phaseList):
				baseKernel = createGaborKernel(xRot, yRot, frequency, phase)
				for colourIndex, colourWeights in enumerate(colourWeightsList):
					filterList.append(applyColourWeights(baseKernel, colourWeights))
					filterCodeList.append(buildRFfilterCode("GAB", orientationIndex, frequencyIndex, colourIndex, phaseIndex))
	filterTensor = stackRFfilters(filterList)
	return filterTensor, filterCodeList


def createRotationGrid(size):
	yGrid = None
	xGrid = None
	if(not isinstance(size, int)):
		raise RuntimeError("createRotationGrid error: size must be an int")
	if(size <= 0):
		raise RuntimeError("createRotationGrid error: size must be > 0")
	axis = pt.linspace(-1.0, 1.0, size, dtype=arrayType, device=deviceDense)
	yGrid, xGrid = pt.meshgrid(axis, axis, indexing="ij")
	return yGrid, xGrid


def rotateGrid(xGrid, yGrid, orientation):
	xRot = None
	yRot = None
	cosTheta = pt.cos(pt.tensor(orientation, dtype=arrayType, device=xGrid.device))
	sinTheta = pt.sin(pt.tensor(orientation, dtype=arrayType, device=xGrid.device))
	xRot = xGrid*cosTheta + yGrid*sinTheta
	yRot = -xGrid*sinTheta + yGrid*cosTheta
	return xRot, yRot


def createEllipsoidalKernel(xRot, yRot, polarity):
	result = None
	sigmaX = 0.45
	sigmaY = 0.2
	innerMask = pt.exp(-0.5*((xRot/sigmaX)**2 + (yRot/sigmaY)**2))
	leftMask = pt.exp(-0.5*(((xRot + 0.4)/(sigmaX*1.2))**2 + (yRot/(sigmaY*1.4))**2))
	rightMask = pt.exp(-0.5*(((xRot - 0.4)/(sigmaX*1.2))**2 + (yRot/(sigmaY*1.4))**2))
	result = polarity*(innerMask - 0.5*(leftMask + rightMask))
	result = normaliseRFfilter2d(result)
	return result


def createGaborKernel(xRot, yRot, frequency, phase):
	result = None
	sigmaX = 0.45
	sigmaY = 0.25
	gaussianEnvelope = pt.exp(-0.5*((xRot/sigmaX)**2 + (yRot/sigmaY)**2))
	carrier = pt.cos((2.0*math.pi*frequency*xRot) + phase)
	result = gaussianEnvelope*carrier
	result = normaliseRFfilter2d(result)
	return result


def buildColourWeights():
	result = []
	result.append(pt.tensor([1.0, 1.0, 1.0], dtype=arrayType, device=deviceDense))
	result.append(pt.tensor([1.0, -1.0, 0.0], dtype=arrayType, device=deviceDense))
	result.append(pt.tensor([0.0, 1.0, -1.0], dtype=arrayType, device=deviceDense))
	result.append(pt.tensor([1.0, 0.0, -1.0], dtype=arrayType, device=deviceDense))
	return result


def applyColourWeights(kernel2d, colourWeights):
	result = None
	result = colourWeights.view(3, 1, 1)*kernel2d.unsqueeze(0)
	result = normaliseRFfilterTensor(result)
	return result


def normaliseRFfilter2d(kernel2d):
	result = None
	kernelCentered = kernel2d - kernel2d.mean()
	kernelNorm = pt.linalg.vector_norm(kernelCentered.reshape(-1))
	if(kernelNorm <= 0):
		raise RuntimeError("normaliseRFfilter2d error: kernelNorm must be > 0")
	result = kernelCentered/kernelNorm
	return result


def normaliseRFfilterTensor(filterTensor):
	result = None
	filterCentered = filterTensor - filterTensor.mean()
	filterNorm = pt.linalg.vector_norm(filterCentered.reshape(-1))
	if(filterNorm <= 0):
		raise RuntimeError("normaliseRFfilterTensor error: filterNorm must be > 0")
	result = filterCentered/filterNorm
	return result


def stackRFfilters(filterList):
	result = None
	if(len(filterList) == 0):
		raise RuntimeError("stackRFfilters error: filterList must not be empty")
	result = pt.stack(filterList, dim=0).to(deviceDense)
	return result


def buildRFfilterCode(prefix, orientationIndex, variantIndex, colourIndex, phaseIndex):
	result = None
	result = prefix + "_O" + str(orientationIndex) + "_V" + str(variantIndex) + "_C" + str(colourIndex) + "_P" + str(phaseIndex)
	return result


def buildDefaultRFfilterCodeList(numberOfFilters):
	result = []
	if(not isinstance(numberOfFilters, int)):
		raise RuntimeError("buildDefaultRFfilterCodeList error: numberOfFilters must be an int")
	if(numberOfFilters <= 0):
		raise RuntimeError("buildDefaultRFfilterCodeList error: numberOfFilters must be > 0")
	for filterIndex in range(numberOfFilters):
		result.append("RF_" + str(filterIndex).zfill(3))
	return result


def buildDefaultRFfilterWordList(numberOfFilters):
	result = []
	if(not isinstance(numberOfFilters, int)):
		raise RuntimeError("buildDefaultRFfilterWordList error: numberOfFilters must be an int")
	if(numberOfFilters <= 0):
		raise RuntimeError("buildDefaultRFfilterWordList error: numberOfFilters must be > 0")
	for filterIndex in range(numberOfFilters):
		result.append(buildCompactRFfilterWord(filterIndex))
	return result


def buildCompactRFfilterWord(filterIndex):
	result = None
	letters = "abcdefghijklmnopqrstuvwxyz"
	base = len(letters)
	if(not isinstance(filterIndex, int)):
		raise RuntimeError("buildCompactRFfilterWord error: filterIndex must be an int")
	if(filterIndex < 0):
		raise RuntimeError("buildCompactRFfilterWord error: filterIndex must be >= 0")
	value = filterIndex
	characters = []
	while(True):
		characters.append(letters[value % base])
		value = value // base
		if(value == 0):
			break
	while(len(characters) < 2):
		characters.append("a")
	characters.reverse()
	result = "".join(characters)
	return result


def createRFfilterContainer(filterTensor, filterCodeList, filterWordList=None):
	result = None
	if(not pt.is_tensor(filterTensor)):
		raise RuntimeError("createRFfilterContainer error: filterTensor must be a tensor")
	if(filterTensor.dim() != 4):
		raise RuntimeError("createRFfilterContainer error: filterTensor rank must be 4")
	if(filterTensor.shape[1] != 3):
		raise RuntimeError("createRFfilterContainer error: filterTensor channel count must be 3")
	if(filterTensor.shape[2] != modalityORpixelsPerColumn or filterTensor.shape[3] != modalityORpixelsPerColumn):
		raise RuntimeError("createRFfilterContainer error: filterTensor spatial size must equal modalityORpixelsPerColumn")
	if(len(filterCodeList) != filterTensor.shape[0]):
		raise RuntimeError("createRFfilterContainer error: filterCodeList length mismatch")
	if(filterWordList is None):
		filterWordList = buildDefaultRFfilterWordList(filterTensor.shape[0])
	if(len(filterWordList) != filterTensor.shape[0]):
		raise RuntimeError("createRFfilterContainer error: filterWordList length mismatch")
	result = ORRFfilters(filterTensor.to(deviceDense, dtype=arrayType), list(filterCodeList), list(filterWordList))
	return result


def normaliseColumnPatches(columnPatches):
	result = None
	patchMean = columnPatches.mean(dim=(1, 2, 3), keepdim=True)
	patchCentered = columnPatches - patchMean
	patchNorm = pt.linalg.vector_norm(patchCentered.reshape(columnPatches.shape[0], -1), dim=1, keepdim=True)
	patchNorm = pt.clamp(patchNorm, min=1.0e-6).view(-1, 1, 1, 1)
	result = patchCentered/patchNorm
	return result


def applyRFfilters(rfFilters, columnPatches):
	# apply a threshold and take the a) most activated filter in the column, and b) only if it is above modalityORRFfilterThreshold.
	selectedFilterIndices = None
	selectedFilterResponses = None
	flatColumnPatches = None
	leadingShape = None
	if(not isinstance(rfFilters, ORRFfilters)):
		raise RuntimeError("applyRFfilters error: rfFilters must be an ORRFfilters instance")
	if(not pt.is_tensor(columnPatches)):
		raise RuntimeError("applyRFfilters error: columnPatches must be a tensor")
	if(columnPatches.dim() == 5):
		leadingShape = columnPatches.shape[0:2]
		flatColumnPatches = columnPatches.reshape(columnPatches.shape[0]*columnPatches.shape[1], columnPatches.shape[2], columnPatches.shape[3], columnPatches.shape[4])
	elif(columnPatches.dim() == 4):
		leadingShape = columnPatches.shape[0:1]
		flatColumnPatches = columnPatches
	else:
		raise RuntimeError("applyRFfilters error: columnPatches rank must be 4 or 5")
	if(flatColumnPatches.shape[1] != 3):
		raise RuntimeError("applyRFfilters error: columnPatches channel count must be 3")
	if(flatColumnPatches.shape[2] != modalityORpixelsPerColumn or flatColumnPatches.shape[3] != modalityORpixelsPerColumn):
		raise RuntimeError("applyRFfilters error: columnPatches spatial size must equal modalityORpixelsPerColumn")
	if(flatColumnPatches.shape[0] == 0):
		selectedFilterIndices = pt.empty(leadingShape, dtype=pt.long, device=columnPatches.device)
		selectedFilterResponses = pt.empty(leadingShape, dtype=arrayType, device=columnPatches.device)
	else:
		flatColumnPatches = normaliseColumnPatches(flatColumnPatches.to(deviceDense, dtype=arrayType))
		filterResponses = pt.einsum("nchw,fchw->nf", flatColumnPatches, rfFilters.filterTensor)
		maxResponses, maxIndices = pt.max(filterResponses, dim=1)
		inactiveIndices = pt.full_like(maxIndices, -1)
		selectedFilterIndices = pt.where(maxResponses >= modalityORRFfilterThreshold, maxIndices, inactiveIndices)
		selectedFilterResponses = maxResponses
		if(columnPatches.device != selectedFilterIndices.device):
			selectedFilterIndices = selectedFilterIndices.to(columnPatches.device)
			selectedFilterResponses = selectedFilterResponses.to(columnPatches.device)
		selectedFilterIndices = selectedFilterIndices.reshape(leadingShape)
		selectedFilterResponses = selectedFilterResponses.reshape(leadingShape)
	return selectedFilterIndices, selectedFilterResponses


def convertRFfilterIndexToASCIItext(rfFilters, rfFilterIndex):
	# use a predefined function to convert the RFfilter index selected into an ASCII textual code (this becomes the feature "word" and "lemma" used by GIAANN).
	result = None
	if(not isinstance(rfFilters, ORRFfilters)):
		raise RuntimeError("convertRFfilterIndexToASCIItext error: rfFilters must be an ORRFfilters instance")
	if(not isinstance(rfFilterIndex, int)):
		raise RuntimeError("convertRFfilterIndexToASCIItext error: rfFilterIndex must be an int")
	if(rfFilterIndex < 0 or rfFilterIndex >= len(rfFilters.filterWordList)):
		raise RuntimeError("convertRFfilterIndexToASCIItext error: rfFilterIndex out of range")
	result = rfFilters.filterWordList[rfFilterIndex]
	return result


def convertRFfilterIndexToASCIItextVerbose(rfFilters, rfFilterIndex):
	result = None
	if(not isinstance(rfFilters, ORRFfilters)):
		raise RuntimeError("convertRFfilterIndexToASCIItextVerbose error: rfFilters must be an ORRFfilters instance")
	if(not isinstance(rfFilterIndex, int)):
		raise RuntimeError("convertRFfilterIndexToASCIItextVerbose error: rfFilterIndex must be an int")
	if(rfFilterIndex < 0 or rfFilterIndex >= len(rfFilters.filterCodeList)):
		raise RuntimeError("convertRFfilterIndexToASCIItextVerbose error: rfFilterIndex out of range")
	result = rfFilters.filterCodeList[rfFilterIndex]
	return result
