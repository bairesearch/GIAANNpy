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
	def __init__(self, filterTensor, filterCodeList, filterWordList, filterOutputIndexTensor, filterInputCodeList):
		self.filterTensor = filterTensor
		self.filterCodeList = filterCodeList
		self.filterWordList = filterWordList
		self.filterOutputIndexTensor = filterOutputIndexTensor
		self.filterInputCodeList = filterInputCodeList
		self.filterIndexToCodeDict = dict(enumerate(filterCodeList))


def initialiseRFfilters():
	# initialise a set of receptive field (RF) filters:
	result = None
	if(modalityORfilterUseExternalRFLibrary):
		result = initialiseRFfiltersExternal()
	else:
		result = initialiseRFfiltersInternal()
	return result


def initialiseRFfiltersExternal():
	result = None
	module = None
	module = importlib.import_module(modalityORfilterExternalRFLibraryModuleName)
	if(hasattr(module, "initialiseRFfilters")):
		if(tokensiationMethodOneColumnPerSnapshotPixel):
			externalResult = module.initialiseRFfilters(modalityORfilterWidth)
		else:
			externalResult = module.initialiseRFfilters(modalityORpixelsPerColumn)
	elif(hasattr(module, "createRFfilters")):
		if(tokensiationMethodOneColumnPerSnapshotPixel):
			externalResult = module.createRFfilters(modalityORfilterWidth)
		else:
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
		filterOutputIndexList = externalResult.get("filterOutputIndexList")
		filterOutputCodeList = externalResult.get("filterOutputCodeList")
		filterOutputWordList = externalResult.get("filterOutputWordList")
		if(filterCodeList is None):
			filterCodeList = buildDefaultRFfilterCodeList(filterTensor.shape[0])
		result = createRFfilterContainer(filterTensor, filterCodeList, filterWordList, filterOutputIndexList, filterOutputCodeList, filterOutputWordList)
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
	filterOutputIndexList = None
	filterTensor, filterCodeList, filterOutputIndexList = buildInternalRFfilterBank()
	result = createRFfilterContainer(filterTensor, filterCodeList, None, filterOutputIndexList, None, None)
	return result


def buildInternalRFfilterBank():
	resultFilterTensor = None
	resultFilterCodeList = []
	resultFilterOutputIndexList = []
	filterList = []
	size = None
	yGrid = None
	xGrid = None
	orientation = None
	xRot = None
	yRot = None
	filterPrototype = None
	baseKernel = None
	colourWeightsList = None
	colourIndex = None
	filterPrototypeList = None
	validateRFfilterBankParameters()
	size = calculateRFfilterSpatialSize()
	yGrid, xGrid = createRotationGrid(size)
	colourWeightsList = buildColourWeights()
	filterPrototypeList = buildGeneratedRFfilterPrototypeList()
	for rotationIndex in range(int(modalityORfiltersRotations)):
		orientation = calculateRFfilterOrientation(rotationIndex)
		xRot, yRot = rotateGrid(xGrid, yGrid, orientation)
		for filterIndex in range(int(modalityORfiltersPerRotation)):
			filterPrototype = filterPrototypeList[filterIndex]
			colourIndex = int(filterPrototype[modalityORfilterPrototypeColourIndex])
			if(colourIndex < 0 or colourIndex >= len(colourWeightsList)):
				raise RuntimeError("buildInternalRFfilterBank error: prototype colourIndex out of range")
			baseKernel = createRFfilterKernelFromPrototype(xRot, yRot, filterPrototype)
			filterList.append(applyColourWeights(baseKernel, colourWeightsList[colourIndex]))
			resultFilterCodeList.append(buildRFfilterCode(str(filterPrototype[modalityORfilterPrototypeTypeIndex]), rotationIndex, filterIndex, colourIndex, filterIndex))
			resultFilterOutputIndexList.append(calculateRFfilterOutputIndex(rotationIndex, filterIndex))
	if(len(filterList) != int(modalityORfilterChannels)):
		raise RuntimeError("buildInternalRFfilterBank error: filter count must equal modalityORfilterChannels")
	resultFilterTensor = stackRFfilters(filterList)
	return resultFilterTensor, resultFilterCodeList, resultFilterOutputIndexList


def validateRFfilterBankParameters():
	result = None
	expectedFilterChannels = None
	expectedFilterChannelsOutput = None
	if(not isinstance(modalityORfiltersRotations, int) or isinstance(modalityORfiltersRotations, bool)):
		raise RuntimeError("validateRFfilterBankParameters error: modalityORfiltersRotations must be an int")
	if(not isinstance(modalityORfiltersPerRotation, int) or isinstance(modalityORfiltersPerRotation, bool)):
		raise RuntimeError("validateRFfilterBankParameters error: modalityORfiltersPerRotation must be an int")
	if(modalityORfiltersRotations <= 0):
		raise RuntimeError("validateRFfilterBankParameters error: modalityORfiltersRotations must be > 0")
	if(modalityORfiltersPerRotation <= 0):
		raise RuntimeError("validateRFfilterBankParameters error: modalityORfiltersPerRotation must be > 0")
	if(not isinstance(modalityORfilterChannels, int) or isinstance(modalityORfilterChannels, bool)):
		raise RuntimeError("validateRFfilterBankParameters error: modalityORfilterChannels must be an int")
	if(not isinstance(modalityORfilterChannelsOutput, int) or isinstance(modalityORfilterChannelsOutput, bool)):
		raise RuntimeError("validateRFfilterBankParameters error: modalityORfilterChannelsOutput must be an int")
	expectedFilterChannels = int(modalityORfiltersPerRotation)*int(modalityORfiltersRotations)
	if(modalityORfilterChannels != expectedFilterChannels):
		raise RuntimeError("validateRFfilterBankParameters error: modalityORfilterChannels must equal modalityORfiltersPerRotation*modalityORfiltersRotations")
	if(modalityORRFfilterRotationallyInvariant):
		expectedFilterChannelsOutput = int(modalityORfiltersPerRotation)
	else:
		expectedFilterChannelsOutput = int(modalityORfilterChannels)
	if(modalityORfilterChannelsOutput != expectedFilterChannelsOutput):
		raise RuntimeError("validateRFfilterBankParameters error: modalityORfilterChannelsOutput does not match rotational invariance setting")
	if(not isinstance(modalityORfilterThreshold, int) and not isinstance(modalityORfilterThreshold, float)):
		raise RuntimeError("validateRFfilterBankParameters error: modalityORfilterThreshold must be an int or float")
	return result


def calculateRFfilterSpatialSize():
	result = None
	if(tokensiationMethodOneColumnPerSnapshotPixel):
		result = int(modalityORfilterWidth)
	else:
		if(not isinstance(modalityORpixelsPerColumn, int) or isinstance(modalityORpixelsPerColumn, bool)):
			raise RuntimeError("calculateRFfilterSpatialSize error: modalityORpixelsPerColumn must be an int")
		if(modalityORpixelsPerColumn <= 0):
			raise RuntimeError("calculateRFfilterSpatialSize error: modalityORpixelsPerColumn must be > 0")
		result = int(modalityORpixelsPerColumn)
	return result


def calculateRFfilterOrientation(rotationIndex):
	result = None
	if(not isinstance(rotationIndex, int) or isinstance(rotationIndex, bool)):
		raise RuntimeError("calculateRFfilterOrientation error: rotationIndex must be an int")
	if(rotationIndex < 0 or rotationIndex >= int(modalityORfiltersRotations)):
		raise RuntimeError("calculateRFfilterOrientation error: rotationIndex out of range")
	result = (float(modalityORfilterRadiansPerCircle)*float(rotationIndex))/float(modalityORfiltersRotations)
	return result


def calculateRFfilterOutputIndex(rotationIndex, filterIndex):
	result = None
	if(rotationIndex < 0 or rotationIndex >= int(modalityORfiltersRotations)):
		raise RuntimeError("calculateRFfilterOutputIndex error: rotationIndex out of range")
	if(filterIndex < 0 or filterIndex >= int(modalityORfiltersPerRotation)):
		raise RuntimeError("calculateRFfilterOutputIndex error: filterIndex out of range")
	if(modalityORRFfilterRotationallyInvariant):
		result = int(filterIndex)
	else:
		result = int(rotationIndex)*int(modalityORfiltersPerRotation) + int(filterIndex)
	if(result < 0 or result >= int(modalityORfilterChannelsOutput)):
		raise RuntimeError("calculateRFfilterOutputIndex error: calculated output index out of range")
	return result


def buildGeneratedRFfilterPrototypeList():
	result = []
	filterPrototype = None
	filterPrototypeKey = None
	filterPrototypeKeySet = set()
	validateGeneratedRFfilterPrototypeLists()
	for filterIndex in range(int(modalityORfiltersPerRotation)):
		filterPrototype = getRFfilterPrototype(filterIndex)
		filterPrototypeKey = buildGeneratedRFfilterPrototypeKey(filterPrototype)
		if(filterPrototypeKey in filterPrototypeKeySet):
			raise RuntimeError("buildGeneratedRFfilterPrototypeList error: duplicate generated RF filter prototype")
		filterPrototypeKeySet.add(filterPrototypeKey)
		result.append(filterPrototype)
	if(len(result) != int(modalityORfiltersPerRotation)):
		raise RuntimeError("buildGeneratedRFfilterPrototypeList error: result length must equal modalityORfiltersPerRotation")
	return result


def buildGeneratedRFfilterPrototypeKey(filterPrototype):
	result = None
	validateRFfilterPrototype(filterPrototype, "buildGeneratedRFfilterPrototypeKey")
	result = tuple(filterPrototype)
	return result


def getRFfilterPrototype(filterIndex):
	result = None
	if(not isinstance(filterIndex, int) or isinstance(filterIndex, bool)):
		raise RuntimeError("getRFfilterPrototype error: filterIndex must be an int")
	if(filterIndex < 0 or filterIndex >= int(modalityORfiltersPerRotation)):
		raise RuntimeError("getRFfilterPrototype error: filterIndex out of range")
	result = generateRFfilterPrototype(filterIndex)
	validateRFfilterPrototype(result, "getRFfilterPrototype")
	return result


def generateRFfilterPrototype(generatedFilterIndex):
	result = None
	typeListIndex = None
	typeOrdinal = None
	typeFilterCount = None
	polarityColourListIndex = None
	polarityColourOrdinal = None
	polarityColourFilterCount = None
	polarityColourListCount = None
	polarityListIndex = None
	colourListIndex = None
	generatedFilterCount = None
	frequencyFraction = None
	phaseFraction = None
	sigmaXFraction = None
	sigmaYFraction = None
	lobeOffsetFraction = None
	surroundScaleFraction = None
	frequency = None
	phase = None
	sigmaX = None
	sigmaY = None
	lobeOffset = None
	surroundScale = None
	validateGeneratedRFfilterPrototypeLists()
	if(not isinstance(generatedFilterIndex, int) or isinstance(generatedFilterIndex, bool)):
		raise RuntimeError("generateRFfilterPrototype error: generatedFilterIndex must be an int")
	if(generatedFilterIndex < 0):
		raise RuntimeError("generateRFfilterPrototype error: generatedFilterIndex must be >= 0")
	generatedFilterCount = int(modalityORfiltersPerRotation)
	if(generatedFilterCount <= 0):
		raise RuntimeError("generateRFfilterPrototype error: generatedFilterCount must be > 0")
	if(generatedFilterIndex >= generatedFilterCount):
		raise RuntimeError("generateRFfilterPrototype error: generatedFilterIndex must be < generatedFilterCount")
	typeListIndex, typeOrdinal, typeFilterCount = calculateGeneratedRFfilterCategoryIndexAndOrdinal(generatedFilterIndex, generatedFilterCount, len(modalityORfilterGeneratedTypeList), modalityORfilterGeneratedTypePermutationStrideSeed, modalityORfilterGeneratedTypePermutationOffsetSeed, "generateRFfilterPrototypeType")
	polarityColourListCount = int(len(modalityORfilterGeneratedPolarityList)*len(modalityORfilterGeneratedColourIndexList))
	polarityColourListIndex, polarityColourOrdinal, polarityColourFilterCount = calculateGeneratedRFfilterCategoryIndexAndOrdinal(typeOrdinal, typeFilterCount, polarityColourListCount, modalityORfilterGeneratedPolarityColourPermutationStrideSeed, modalityORfilterGeneratedPolarityColourPermutationOffsetSeed, "generateRFfilterPrototypePolarityColour")
	polarityListIndex = int(polarityColourListIndex)//int(len(modalityORfilterGeneratedColourIndexList))
	colourListIndex = int(polarityColourListIndex)%int(len(modalityORfilterGeneratedColourIndexList))
	frequencyFraction = calculateGeneratedRFfilterStratifiedFraction(polarityColourOrdinal, polarityColourFilterCount, modalityORfilterGeneratedFrequencyPermutationStrideSeed, modalityORfilterGeneratedFrequencyPermutationOffsetSeed, "generateRFfilterPrototypeFrequency")
	phaseFraction = calculateGeneratedRFfilterStratifiedFraction(polarityColourOrdinal, polarityColourFilterCount, modalityORfilterGeneratedPhasePermutationStrideSeed, modalityORfilterGeneratedPhasePermutationOffsetSeed, "generateRFfilterPrototypePhase")
	sigmaXFraction = calculateGeneratedRFfilterStratifiedFraction(polarityColourOrdinal, polarityColourFilterCount, modalityORfilterGeneratedSigmaXPermutationStrideSeed, modalityORfilterGeneratedSigmaXPermutationOffsetSeed, "generateRFfilterPrototypeSigmaX")
	sigmaYFraction = calculateGeneratedRFfilterStratifiedFraction(polarityColourOrdinal, polarityColourFilterCount, modalityORfilterGeneratedSigmaYPermutationStrideSeed, modalityORfilterGeneratedSigmaYPermutationOffsetSeed, "generateRFfilterPrototypeSigmaY")
	lobeOffsetFraction = calculateGeneratedRFfilterStratifiedFraction(polarityColourOrdinal, polarityColourFilterCount, modalityORfilterGeneratedLobeOffsetPermutationStrideSeed, modalityORfilterGeneratedLobeOffsetPermutationOffsetSeed, "generateRFfilterPrototypeLobeOffset")
	surroundScaleFraction = calculateGeneratedRFfilterStratifiedFraction(polarityColourOrdinal, polarityColourFilterCount, modalityORfilterGeneratedSurroundScalePermutationStrideSeed, modalityORfilterGeneratedSurroundScalePermutationOffsetSeed, "generateRFfilterPrototypeSurroundScale")
	frequency = calculateGeneratedRFfilterParameter(modalityORfilterGeneratedFrequencyMin, modalityORfilterGeneratedFrequencyMax, frequencyFraction, "generateRFfilterPrototypeFrequency")
	phase = calculateGeneratedRFfilterParameter(modalityORfilterGeneratedPhaseMin, modalityORfilterGeneratedPhaseMax, phaseFraction, "generateRFfilterPrototypePhase")
	sigmaX = calculateGeneratedRFfilterParameter(modalityORfilterGeneratedSigmaXMin, modalityORfilterGeneratedSigmaXMax, sigmaXFraction, "generateRFfilterPrototypeSigmaX")
	sigmaY = calculateGeneratedRFfilterParameter(modalityORfilterGeneratedSigmaYMin, modalityORfilterGeneratedSigmaYMax, sigmaYFraction, "generateRFfilterPrototypeSigmaY")
	lobeOffset = calculateGeneratedRFfilterParameter(modalityORfilterGeneratedLobeOffsetMin, modalityORfilterGeneratedLobeOffsetMax, lobeOffsetFraction, "generateRFfilterPrototypeLobeOffset")
	surroundScale = calculateGeneratedRFfilterParameter(modalityORfilterGeneratedSurroundScaleMin, modalityORfilterGeneratedSurroundScaleMax, surroundScaleFraction, "generateRFfilterPrototypeSurroundScale")
	if(modalityORfilterGeneratedTypeList[typeListIndex] == modalityORfilterTypeEllipsoidal):
		frequency = modalityORfilterNoFrequency
		phase = modalityORfilterPhaseCosine
	elif(modalityORfilterGeneratedTypeList[typeListIndex] == modalityORfilterTypeGabor):
		lobeOffset = modalityORfilterNoLobeOffset
		surroundScale = modalityORfilterNoSurroundScale
	else:
		raise RuntimeError("generateRFfilterPrototype error: unsupported generated filter type")
	result = (modalityORfilterGeneratedTypeList[typeListIndex], modalityORfilterGeneratedPolarityList[polarityListIndex], frequency, phase, modalityORfilterGeneratedColourIndexList[colourListIndex], sigmaX, sigmaY, lobeOffset, surroundScale)
	return result


def calculateGeneratedRFfilterCategoryIndex(generatedFilterIndex, generatedFilterCount, categoryCount, permutationStrideSeed, permutationOffsetSeed, functionName):
	result = None
	categoryOrdinal = None
	categoryFilterCount = None
	result, categoryOrdinal, categoryFilterCount = calculateGeneratedRFfilterCategoryIndexAndOrdinal(generatedFilterIndex, generatedFilterCount, categoryCount, permutationStrideSeed, permutationOffsetSeed, functionName)
	return result


def calculateGeneratedRFfilterCategoryIndexAndOrdinal(generatedFilterIndex, generatedFilterCount, categoryCount, permutationStrideSeed, permutationOffsetSeed, functionName):
	result = None
	categoryIndex = None
	categoryOrdinal = None
	categoryFilterCount = None
	permutedFilterIndex = None
	if(functionName == ""):
		raise RuntimeError("calculateGeneratedRFfilterCategoryIndexAndOrdinal error: functionName must not be empty")
	if(not isinstance(categoryCount, int) or isinstance(categoryCount, bool)):
		raise RuntimeError(functionName + " error: categoryCount must be an int")
	if(categoryCount < int(modalityORfilterGeneratedMinimumCategoryCount)):
		raise RuntimeError(functionName + " error: categoryCount must be >= modalityORfilterGeneratedMinimumCategoryCount")
	if(int(generatedFilterCount) < int(categoryCount)):
		raise RuntimeError(functionName + " error: generatedFilterCount must be >= categoryCount to guarantee categorical coverage")
	permutedFilterIndex = calculateGeneratedRFfilterPermutationIndex(generatedFilterIndex, generatedFilterCount, permutationStrideSeed, permutationOffsetSeed, functionName)
	categoryIndex = int(permutedFilterIndex)%int(categoryCount)
	categoryOrdinal = int(permutedFilterIndex)//int(categoryCount)
	categoryFilterCount = calculateGeneratedRFfilterCategoryFilterCount(generatedFilterCount, categoryCount, categoryIndex, functionName)
	if(categoryIndex < 0 or categoryIndex >= int(categoryCount)):
		raise RuntimeError(functionName + " error: generated category index out of range")
	if(categoryOrdinal < 0 or categoryOrdinal >= int(categoryFilterCount)):
		raise RuntimeError(functionName + " error: generated category ordinal out of range")
	result = (categoryIndex, categoryOrdinal, categoryFilterCount)
	return result


def calculateGeneratedRFfilterCategoryFilterCount(generatedFilterCount, categoryCount, categoryIndex, functionName):
	result = None
	if(functionName == ""):
		raise RuntimeError("calculateGeneratedRFfilterCategoryFilterCount error: functionName must not be empty")
	if(not isinstance(generatedFilterCount, int) or isinstance(generatedFilterCount, bool)):
		raise RuntimeError(functionName + " error: generatedFilterCount must be an int")
	if(not isinstance(categoryCount, int) or isinstance(categoryCount, bool)):
		raise RuntimeError(functionName + " error: categoryCount must be an int")
	if(not isinstance(categoryIndex, int) or isinstance(categoryIndex, bool)):
		raise RuntimeError(functionName + " error: categoryIndex must be an int")
	if(generatedFilterCount < int(modalityORfilterGeneratedMinimumFilterCount)):
		raise RuntimeError(functionName + " error: generatedFilterCount must be >= modalityORfilterGeneratedMinimumFilterCount")
	if(categoryCount < int(modalityORfilterGeneratedMinimumCategoryCount)):
		raise RuntimeError(functionName + " error: categoryCount must be >= modalityORfilterGeneratedMinimumCategoryCount")
	if(int(generatedFilterCount) < int(categoryCount)):
		raise RuntimeError(functionName + " error: generatedFilterCount must be >= categoryCount to guarantee categorical coverage")
	if(categoryIndex < 0 or categoryIndex >= int(categoryCount)):
		raise RuntimeError(functionName + " error: categoryIndex out of range")
	result = int(((int(generatedFilterCount) - int(modalityORfilterGeneratedMinimumFilterCount) - int(categoryIndex))//int(categoryCount)) + int(modalityORfilterGeneratedMinimumFilterCount))
	if(result < int(modalityORfilterGeneratedMinimumFilterCount)):
		raise RuntimeError(functionName + " error: category filter count must be >= modalityORfilterGeneratedMinimumFilterCount")
	return result


def calculateGeneratedRFfilterStratifiedFraction(generatedFilterIndex, generatedFilterCount, permutationStrideSeed, permutationOffsetSeed, functionName):
	result = None
	permutedFilterIndex = None
	if(functionName == ""):
		raise RuntimeError("calculateGeneratedRFfilterStratifiedFraction error: functionName must not be empty")
	permutedFilterIndex = calculateGeneratedRFfilterPermutationIndex(generatedFilterIndex, generatedFilterCount, permutationStrideSeed, permutationOffsetSeed, functionName)
	result = calculateGeneratedRFfilterFraction(permutedFilterIndex, generatedFilterCount)
	return result


def calculateGeneratedRFfilterPermutationIndex(generatedFilterIndex, generatedFilterCount, permutationStrideSeed, permutationOffsetSeed, functionName):
	result = None
	permutationStride = None
	permutationOffset = None
	if(functionName == ""):
		raise RuntimeError("calculateGeneratedRFfilterPermutationIndex error: functionName must not be empty")
	validateGeneratedRFfilterIndexAndCount(generatedFilterIndex, generatedFilterCount, functionName)
	if(not isinstance(permutationOffsetSeed, int) or isinstance(permutationOffsetSeed, bool)):
		raise RuntimeError(functionName + " error: permutationOffsetSeed must be an int")
	permutationStride = calculateGeneratedRFfilterCoprimeStride(generatedFilterCount, permutationStrideSeed, functionName)
	permutationOffset = int(permutationOffsetSeed)%int(generatedFilterCount)
	result = ((int(generatedFilterIndex)*int(permutationStride)) + int(permutationOffset))%int(generatedFilterCount)
	if(result < 0 or result >= int(generatedFilterCount)):
		raise RuntimeError(functionName + " error: generated permutation index out of range")
	return result


def calculateGeneratedRFfilterCoprimeStride(generatedFilterCount, permutationStrideSeed, functionName):
	result = None
	candidateStride = None
	if(functionName == ""):
		raise RuntimeError("calculateGeneratedRFfilterCoprimeStride error: functionName must not be empty")
	if(not isinstance(generatedFilterCount, int) or isinstance(generatedFilterCount, bool)):
		raise RuntimeError(functionName + " error: generatedFilterCount must be an int")
	if(generatedFilterCount < int(modalityORfilterGeneratedMinimumFilterCount)):
		raise RuntimeError(functionName + " error: generatedFilterCount must be >= modalityORfilterGeneratedMinimumFilterCount")
	if(not isinstance(permutationStrideSeed, int) or isinstance(permutationStrideSeed, bool)):
		raise RuntimeError(functionName + " error: permutationStrideSeed must be an int")
	if(permutationStrideSeed < int(modalityORfilterGeneratedCoprimeStrideMinimum)):
		raise RuntimeError(functionName + " error: permutationStrideSeed must be >= modalityORfilterGeneratedCoprimeStrideMinimum")
	if(int(modalityORfilterGeneratedCoprimeSearchStep) != int(modalityORfilterGeneratedCoprimeStrideMinimum)):
		raise RuntimeError(functionName + " error: modalityORfilterGeneratedCoprimeSearchStep must equal modalityORfilterGeneratedCoprimeStrideMinimum")
	candidateStride = int(permutationStrideSeed)%int(generatedFilterCount)
	if(candidateStride < int(modalityORfilterGeneratedCoprimeStrideMinimum)):
		candidateStride = int(modalityORfilterGeneratedCoprimeStrideMinimum)
	for searchIndex in range(int(generatedFilterCount)):
		if(result is None):
			if(math.gcd(int(candidateStride), int(generatedFilterCount)) == int(modalityORfilterGeneratedCoprimeGcdRequired)):
				result = int(candidateStride)
			else:
				candidateStride = int(candidateStride) + int(modalityORfilterGeneratedCoprimeSearchStep)
				if(candidateStride >= int(generatedFilterCount)):
					candidateStride = int(modalityORfilterGeneratedCoprimeStrideMinimum)
	if(result is None):
		raise RuntimeError(functionName + " error: failed to find coprime permutation stride")
	return result


def validateGeneratedRFfilterIndexAndCount(generatedFilterIndex, generatedFilterCount, functionName):
	result = None
	if(functionName == ""):
		raise RuntimeError("validateGeneratedRFfilterIndexAndCount error: functionName must not be empty")
	if(not isinstance(generatedFilterIndex, int) or isinstance(generatedFilterIndex, bool)):
		raise RuntimeError(functionName + " error: generatedFilterIndex must be an int")
	if(not isinstance(generatedFilterCount, int) or isinstance(generatedFilterCount, bool)):
		raise RuntimeError(functionName + " error: generatedFilterCount must be an int")
	if(generatedFilterCount < int(modalityORfilterGeneratedMinimumFilterCount)):
		raise RuntimeError(functionName + " error: generatedFilterCount must be >= modalityORfilterGeneratedMinimumFilterCount")
	if(generatedFilterIndex < 0 or generatedFilterIndex >= generatedFilterCount):
		raise RuntimeError(functionName + " error: generatedFilterIndex out of range")
	return result


def calculateGeneratedRFfilterFraction(generatedFilterIndex, generatedFilterCount):
	result = None
	if(not isinstance(generatedFilterIndex, int) or isinstance(generatedFilterIndex, bool)):
		raise RuntimeError("calculateGeneratedRFfilterFraction error: generatedFilterIndex must be an int")
	if(not isinstance(generatedFilterCount, int) or isinstance(generatedFilterCount, bool)):
		raise RuntimeError("calculateGeneratedRFfilterFraction error: generatedFilterCount must be an int")
	if(generatedFilterIndex < 0 or generatedFilterIndex >= generatedFilterCount):
		raise RuntimeError("calculateGeneratedRFfilterFraction error: generatedFilterIndex out of range")
	if(generatedFilterCount <= 0):
		raise RuntimeError("calculateGeneratedRFfilterFraction error: generatedFilterCount must be > 0")
	result = (float(generatedFilterIndex) + float(modalityORfilterGeneratedFractionOffset))/float(generatedFilterCount)
	if(result <= 0.0 or result >= 1.0):
		raise RuntimeError("calculateGeneratedRFfilterFraction error: generated fraction must be > 0.0 and < 1.0")
	return result


def calculateGeneratedRFfilterParameter(parameterMin, parameterMax, generatedFilterFraction, functionName):
	result = None
	if(functionName == ""):
		raise RuntimeError("calculateGeneratedRFfilterParameter error: functionName must not be empty")
	if(not isinstance(parameterMin, int) and not isinstance(parameterMin, float)):
		raise RuntimeError(functionName + " error: parameterMin must be an int or float")
	if(not isinstance(parameterMax, int) and not isinstance(parameterMax, float)):
		raise RuntimeError(functionName + " error: parameterMax must be an int or float")
	if(parameterMax <= parameterMin):
		raise RuntimeError(functionName + " error: parameterMax must be > parameterMin")
	if(generatedFilterFraction <= 0.0 or generatedFilterFraction >= 1.0):
		raise RuntimeError(functionName + " error: generatedFilterFraction must be > 0.0 and < 1.0")
	result = float(parameterMin) + ((float(parameterMax) - float(parameterMin))*float(generatedFilterFraction))
	return result


def validateGeneratedRFfilterPrototypeLists():
	result = None
	if(len(modalityORfilterGeneratedTypeList) <= 0):
		raise RuntimeError("validateGeneratedRFfilterPrototypeLists error: modalityORfilterGeneratedTypeList must not be empty")
	if(len(modalityORfilterGeneratedPolarityList) <= 0):
		raise RuntimeError("validateGeneratedRFfilterPrototypeLists error: modalityORfilterGeneratedPolarityList must not be empty")
	if(len(modalityORfilterGeneratedColourIndexList) <= 0):
		raise RuntimeError("validateGeneratedRFfilterPrototypeLists error: modalityORfilterGeneratedColourIndexList must not be empty")
	return result


def validateRFfilterPrototype(filterPrototype, functionName):
	result = None
	colourIndex = None
	frequency = None
	polarity = None
	sigmaX = None
	sigmaY = None
	lobeOffset = None
	surroundScale = None
	if(functionName == ""):
		raise RuntimeError("validateRFfilterPrototype error: functionName must not be empty")
	if(not isinstance(filterPrototype, tuple) or len(filterPrototype) != int(modalityORfilterPrototypeLength)):
		raise RuntimeError(functionName + " error: filterPrototype length mismatch")
	if(filterPrototype[modalityORfilterPrototypeTypeIndex] != modalityORfilterTypeEllipsoidal and filterPrototype[modalityORfilterPrototypeTypeIndex] != modalityORfilterTypeGabor):
		raise RuntimeError(functionName + " error: unsupported RF filter prototype type")
	polarity = float(filterPrototype[modalityORfilterPrototypePolarityIndex])
	if(polarity != modalityORfilterPolarityPositive and polarity != modalityORfilterPolarityNegative):
		raise RuntimeError(functionName + " error: filterPrototype polarity is unsupported")
	frequency = float(filterPrototype[modalityORfilterPrototypeFrequencyIndex])
	if(filterPrototype[modalityORfilterPrototypeTypeIndex] == modalityORfilterTypeGabor and frequency <= 0.0):
		raise RuntimeError(functionName + " error: gabor filterPrototype frequency must be > 0.0")
	colourIndex = int(filterPrototype[modalityORfilterPrototypeColourIndex])
	if(colourIndex < 0 or colourIndex >= len(modalityORfilterColourWeightsList)):
		raise RuntimeError(functionName + " error: filterPrototype colourIndex out of range")
	sigmaX = float(filterPrototype[modalityORfilterPrototypeSigmaXIndex])
	sigmaY = float(filterPrototype[modalityORfilterPrototypeSigmaYIndex])
	lobeOffset = float(filterPrototype[modalityORfilterPrototypeLobeOffsetIndex])
	surroundScale = float(filterPrototype[modalityORfilterPrototypeSurroundScaleIndex])
	if(sigmaX <= 0.0 or sigmaY <= 0.0):
		raise RuntimeError(functionName + " error: filterPrototype sigma values must be > 0.0")
	if(lobeOffset < 0.0):
		raise RuntimeError(functionName + " error: filterPrototype lobeOffset must be >= 0.0")
	if(surroundScale <= 0.0):
		raise RuntimeError(functionName + " error: filterPrototype surroundScale must be > 0.0")
	return result


def createRFfilterKernelFromPrototype(xRot, yRot, filterPrototype):
	result = None
	filterType = None
	polarity = None
	frequency = None
	phase = None
	sigmaX = None
	sigmaY = None
	lobeOffset = None
	surroundScale = None
	validateRFfilterPrototype(filterPrototype, "createRFfilterKernelFromPrototype")
	filterType = filterPrototype[modalityORfilterPrototypeTypeIndex]
	polarity = float(filterPrototype[modalityORfilterPrototypePolarityIndex])
	frequency = float(filterPrototype[modalityORfilterPrototypeFrequencyIndex])
	phase = float(filterPrototype[modalityORfilterPrototypePhaseIndex])
	sigmaX = float(filterPrototype[modalityORfilterPrototypeSigmaXIndex])
	sigmaY = float(filterPrototype[modalityORfilterPrototypeSigmaYIndex])
	lobeOffset = float(filterPrototype[modalityORfilterPrototypeLobeOffsetIndex])
	surroundScale = float(filterPrototype[modalityORfilterPrototypeSurroundScaleIndex])
	if(filterType == modalityORfilterTypeEllipsoidal):
		result = createEllipsoidalKernel(xRot, yRot, polarity, sigmaX, sigmaY, lobeOffset, surroundScale)
	elif(filterType == modalityORfilterTypeGabor):
		result = createGaborKernel(xRot, yRot, frequency, phase, sigmaX, sigmaY)
		if(polarity == modalityORfilterPolarityNegative):
			result = normaliseRFfilter2d(-result)
		elif(polarity != modalityORfilterPolarityPositive):
			raise RuntimeError("createRFfilterKernelFromPrototype error: gabor polarity must be modalityORfilterPolarityPositive or modalityORfilterPolarityNegative")
	else:
		raise RuntimeError("createRFfilterKernelFromPrototype error: unsupported RF filter prototype type")
	return result


if(tokensiationMethodOneColumnPerSnapshotPixel):
	def validatePixelColumnRFfilterParameters():
		result = None
		if(tokensiationMethodOneColumnPerSnapshotPixel):
			validateRFfilterBankParameters()
			if(not isinstance(modalityORfilterWidth, int)):
				raise RuntimeError("validatePixelColumnRFfilterParameters error: modalityORfilterWidth must be an int")
			if(modalityORfilterWidth < 3):
				raise RuntimeError("validatePixelColumnRFfilterParameters error: modalityORfilterWidth must be >= 3")
			if(modalityORfilterWidth%2 != 1):
				raise RuntimeError("validatePixelColumnRFfilterParameters error: modalityORfilterWidth must be odd")
			if(not isinstance(modalityORfilterChannels, int)):
				raise RuntimeError("validatePixelColumnRFfilterParameters error: modalityORfilterChannels must be an int")
			if(modalityORfilterChannels <= 0):
				raise RuntimeError("validatePixelColumnRFfilterParameters error: modalityORfilterChannels must be > 0")
			if(not isinstance(modalityORnumberOfFeaturesPerColumn, int)):
				raise RuntimeError("validatePixelColumnRFfilterParameters error: modalityORnumberOfFeaturesPerColumn must be an int")
			if(modalityORnumberOfFeaturesPerColumn != modalityORfilterWidth*modalityORfilterWidth*modalityORfilterChannelsOutput):
				raise RuntimeError("validatePixelColumnRFfilterParameters error: modalityORnumberOfFeaturesPerColumn must equal modalityORfilterWidth*modalityORfilterWidth*modalityORfilterChannelsOutput")
		else:
			raise RuntimeError("validatePixelColumnRFfilterParameters error: requires tokensiationMethodOneColumnPerSnapshotPixel")
		return result


	def adjustInternalRFfilterBankToPixelColumnFilterChannels(filterTensor, filterCodeList):
		resultFilterTensor = None
		resultFilterCodeList = None
		numberOfSupplementaryFilters = None
		supplementaryFilterTensor = None
		supplementaryFilterCodeList = None
		if(tokensiationMethodOneColumnPerSnapshotPixel):
			validatePixelColumnRFfilterParameters()
			if(not pt.is_tensor(filterTensor)):
				raise RuntimeError("adjustInternalRFfilterBankToPixelColumnFilterChannels error: filterTensor must be a tensor")
			if(filterTensor.dim() != 4):
				raise RuntimeError("adjustInternalRFfilterBankToPixelColumnFilterChannels error: filterTensor rank must be 4")
			if(len(filterCodeList) != int(filterTensor.shape[0])):
				raise RuntimeError("adjustInternalRFfilterBankToPixelColumnFilterChannels error: filterCodeList length mismatch")
			if(int(filterTensor.shape[0]) > int(modalityORfilterChannels)):
				resultFilterTensor = filterTensor[:int(modalityORfilterChannels)]
				resultFilterCodeList = list(filterCodeList[:int(modalityORfilterChannels)])
			elif(int(filterTensor.shape[0]) < int(modalityORfilterChannels)):
				numberOfSupplementaryFilters = int(modalityORfilterChannels) - int(filterTensor.shape[0])
				supplementaryFilterTensor, supplementaryFilterCodeList = buildSupplementaryPixelColumnRFfilters(numberOfSupplementaryFilters, int(filterTensor.shape[2]))
				resultFilterTensor = pt.cat((filterTensor, supplementaryFilterTensor), dim=0)
				resultFilterCodeList = list(filterCodeList) + supplementaryFilterCodeList
			else:
				resultFilterTensor = filterTensor
				resultFilterCodeList = list(filterCodeList)
		else:
			raise RuntimeError("adjustInternalRFfilterBankToPixelColumnFilterChannels error: requires tokensiationMethodOneColumnPerSnapshotPixel")
		return resultFilterTensor, resultFilterCodeList


	def buildSupplementaryPixelColumnRFfilters(numberOfFilters, size):
		resultFilterTensor = None
		resultFilterCodeList = []
		filterList = []
		yGrid = None
		xGrid = None
		colourWeightsList = None
		colourIndex = None
		polarityIndex = None
		baseKernel = None
		if(tokensiationMethodOneColumnPerSnapshotPixel):
			if(numberOfFilters <= 0):
				raise RuntimeError("buildSupplementaryPixelColumnRFfilters error: numberOfFilters must be > 0")
			if(size != int(modalityORfilterWidth)):
				raise RuntimeError("buildSupplementaryPixelColumnRFfilters error: size must equal modalityORfilterWidth")
			yGrid, xGrid = createRotationGrid(size)
			colourWeightsList = buildColourWeights()
			for filterIndex in range(numberOfFilters):
				colourIndex = filterIndex%len(colourWeightsList)
				polarityIndex = int(filterIndex//len(colourWeightsList))%2
				baseKernel = createSupplementaryPixelColumnRFfilterKernel(xGrid, yGrid, filterIndex, polarityIndex)
				filterList.append(applyColourWeights(baseKernel, colourWeightsList[colourIndex]))
				resultFilterCodeList.append(buildRFfilterCode("PIX", filterIndex, 0, colourIndex, polarityIndex))
			resultFilterTensor = stackRFfilters(filterList)
		else:
			raise RuntimeError("buildSupplementaryPixelColumnRFfilters error: requires tokensiationMethodOneColumnPerSnapshotPixel")
		return resultFilterTensor, resultFilterCodeList


	def createSupplementaryPixelColumnRFfilterKernel(xGrid, yGrid, filterIndex, polarityIndex):
		result = None
		orientationIndex = None
		radiusIndex = None
		angleRadians = None
		radius = None
		centreX = None
		centreY = None
		sigma = None
		innerMask = None
		outerMask = None
		if(tokensiationMethodOneColumnPerSnapshotPixel):
			orientationIndex = filterIndex%8
			radiusIndex = int(filterIndex//8)%2
			angleRadians = (2.0*math.pi*float(orientationIndex))/8.0
			radius = 0.0
			if(radiusIndex == 1):
				radius = 0.35
			centreX = math.cos(angleRadians)*radius
			centreY = math.sin(angleRadians)*radius
			sigma = 0.2 + (0.05*float(int(filterIndex//16)%3))
			innerMask = pt.exp(-0.5*(((xGrid - centreX)/sigma)**2 + ((yGrid - centreY)/sigma)**2))
			outerMask = pt.exp(-0.5*((xGrid/(sigma*2.0))**2 + (yGrid/(sigma*2.0))**2))
			result = innerMask - (0.5*outerMask)
			if(polarityIndex == 1):
				result = -result
			result = normaliseRFfilter2d(result)
		else:
			raise RuntimeError("createSupplementaryPixelColumnRFfilterKernel error: requires tokensiationMethodOneColumnPerSnapshotPixel")
		return result


def buildEllipsoidalRFfilters():
	filterTensor = None
	filterCodeList = []
	filterList = []
	size = None
	if(tokensiationMethodOneColumnPerSnapshotPixel):
		size = modalityORfilterWidth
	else:
		size = modalityORpixelsPerColumn
	yGrid, xGrid = createRotationGrid(size)
	orientationList = [0.0, math.pi/4.0, math.pi/2.0, 3.0*math.pi/4.0]
	colourWeightsList = buildColourWeights()
	for orientationIndex, orientation in enumerate(orientationList):
		xRot, yRot = rotateGrid(xGrid, yGrid, orientation)
		for polarityIndex, polarity in enumerate([1.0, -1.0]):
			baseKernel = createEllipsoidalKernel(xRot, yRot, polarity, modalityORfilterEllipsoidalSigmaX, modalityORfilterEllipsoidalSigmaY, modalityORfilterEllipsoidalLobeOffset, modalityORfilterEllipsoidalSurroundScale)
			for colourIndex, colourWeights in enumerate(colourWeightsList):
				filterList.append(applyColourWeights(baseKernel, colourWeights))
				filterCodeList.append(buildRFfilterCode("ELL", orientationIndex, polarityIndex, colourIndex, 0))
	filterTensor = stackRFfilters(filterList)
	return filterTensor, filterCodeList


def buildGaborRFfilters():
	filterTensor = None
	filterCodeList = []
	filterList = []
	size = None
	if(tokensiationMethodOneColumnPerSnapshotPixel):
		size = modalityORfilterWidth
	else:
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
				baseKernel = createGaborKernel(xRot, yRot, frequency, phase, modalityORfilterGaborSigmaX, modalityORfilterGaborSigmaY)
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


def createEllipsoidalKernel(xRot, yRot, polarity, sigmaX, sigmaY, lobeOffset, surroundScale):
	result = None
	innerMask = pt.exp(-0.5*((xRot/sigmaX)**2 + (yRot/sigmaY)**2))
	leftMask = pt.exp(-0.5*(((xRot + lobeOffset)/(sigmaX*modalityORfilterEllipsoidalLobeSigmaXScale))**2 + (yRot/(sigmaY*modalityORfilterEllipsoidalLobeSigmaYScale))**2))
	rightMask = pt.exp(-0.5*(((xRot - lobeOffset)/(sigmaX*modalityORfilterEllipsoidalLobeSigmaXScale))**2 + (yRot/(sigmaY*modalityORfilterEllipsoidalLobeSigmaYScale))**2))
	result = polarity*(innerMask - surroundScale*(leftMask + rightMask))
	result = normaliseRFfilter2d(result)
	return result


def createGaborKernel(xRot, yRot, frequency, phase, sigmaX, sigmaY):
	result = None
	gaussianEnvelope = pt.exp(-0.5*((xRot/sigmaX)**2 + (yRot/sigmaY)**2))
	carrier = pt.cos((modalityORfilterRadiansPerCircle*frequency*xRot) + phase)
	result = gaussianEnvelope*carrier
	result = normaliseRFfilter2d(result)
	return result


def buildColourWeights():
	result = []
	for colourWeights in modalityORfilterColourWeightsList:
		if(not isinstance(colourWeights, tuple) or len(colourWeights) != int(modalityORfilterColourChannelCount)):
			raise RuntimeError("buildColourWeights error: colourWeights length mismatch")
		result.append(pt.tensor(colourWeights, dtype=arrayType, device=deviceDense))
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
	result = prefix + modalityORfilterCodeOrientationPrefix + str(orientationIndex) + modalityORfilterCodeVariantPrefix + str(variantIndex) + modalityORfilterCodeColourPrefix + str(colourIndex) + modalityORfilterCodePhasePrefix + str(phaseIndex)
	return result


def buildDefaultRFfilterCodeList(numberOfFilters):
	result = []
	if(not isinstance(numberOfFilters, int)):
		raise RuntimeError("buildDefaultRFfilterCodeList error: numberOfFilters must be an int")
	if(numberOfFilters <= 0):
		raise RuntimeError("buildDefaultRFfilterCodeList error: numberOfFilters must be > 0")
	for filterIndex in range(numberOfFilters):
		result.append(modalityORfilterCodeDefaultPrefix + str(filterIndex).zfill(int(modalityORfilterCodeIndexDigits)))
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
	letters = modalityORfilterWordAlphabet
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
	while(len(characters) < int(modalityORfilterWordMinLetters)):
		characters.append(letters[0])
	characters.reverse()
	result = "".join(characters)
	return result


def createRFfilterContainer(filterTensor, filterCodeList, filterWordList=None, filterOutputIndexList=None, filterOutputCodeList=None, filterOutputWordList=None):
	result = None
	filterOutputIndexTensor = None
	outputCodeList = None
	outputWordList = None
	validateRFfilterBankParameters()
	if(not pt.is_tensor(filterTensor)):
		raise RuntimeError("createRFfilterContainer error: filterTensor must be a tensor")
	if(filterTensor.dim() != 4):
		raise RuntimeError("createRFfilterContainer error: filterTensor rank must be 4")
	if(filterTensor.shape[1] != int(modalityORfilterColourChannelCount)):
		raise RuntimeError("createRFfilterContainer error: filterTensor channel count must be 3")
	if(int(filterTensor.shape[0]) != int(modalityORfilterChannels)):
		raise RuntimeError("createRFfilterContainer error: filterTensor filter count must equal modalityORfilterChannels")
	if(tokensiationMethodOneColumnPerSnapshotPixel):
		validatePixelColumnRFfilterParameters()
		if(filterTensor.shape[2] != modalityORfilterWidth or filterTensor.shape[3] != modalityORfilterWidth):
			raise RuntimeError("createRFfilterContainer error: filterTensor spatial size must equal modalityORfilterWidth")
	else:
		if(filterTensor.shape[2] != modalityORpixelsPerColumn or filterTensor.shape[3] != modalityORpixelsPerColumn):
			raise RuntimeError("createRFfilterContainer error: filterTensor spatial size must equal modalityORpixelsPerColumn")
	if(len(filterCodeList) != filterTensor.shape[0]):
		raise RuntimeError("createRFfilterContainer error: filterCodeList length mismatch")
	filterOutputIndexTensor = buildRFfilterOutputIndexTensor(int(filterTensor.shape[0]), filterOutputIndexList)
	if(filterOutputCodeList is None):
		outputCodeList = buildRFfilterOutputCodeList(filterCodeList, filterOutputIndexTensor)
	else:
		outputCodeList = list(filterOutputCodeList)
	if(len(outputCodeList) != int(modalityORfilterChannelsOutput)):
		raise RuntimeError("createRFfilterContainer error: filterOutputCodeList length must equal modalityORfilterChannelsOutput")
	if(filterOutputWordList is None):
		if(filterWordList is None):
			outputWordList = buildDefaultRFfilterWordList(int(modalityORfilterChannelsOutput))
		else:
			if(len(filterWordList) != int(modalityORfilterChannelsOutput)):
				raise RuntimeError("createRFfilterContainer error: filterWordList length must equal modalityORfilterChannelsOutput")
			outputWordList = list(filterWordList)
	else:
		outputWordList = list(filterOutputWordList)
	if(len(outputWordList) != int(modalityORfilterChannelsOutput)):
		raise RuntimeError("createRFfilterContainer error: filterOutputWordList length must equal modalityORfilterChannelsOutput")
	result = ORRFfilters(filterTensor.to(deviceDense, dtype=arrayType), outputCodeList, outputWordList, filterOutputIndexTensor.to(deviceDense, dtype=pt.long), list(filterCodeList))
	return result


def buildRFfilterOutputIndexTensor(numberOfInputFilters, filterOutputIndexList=None):
	result = None
	outputIndexList = []
	uniqueOutputIndexTensor = None
	if(not isinstance(numberOfInputFilters, int) or isinstance(numberOfInputFilters, bool)):
		raise RuntimeError("buildRFfilterOutputIndexTensor error: numberOfInputFilters must be an int")
	if(numberOfInputFilters != int(modalityORfilterChannels)):
		raise RuntimeError("buildRFfilterOutputIndexTensor error: numberOfInputFilters must equal modalityORfilterChannels")
	if(filterOutputIndexList is None):
		for filterIndex in range(numberOfInputFilters):
			if(modalityORRFfilterRotationallyInvariant):
				outputIndexList.append(int(filterIndex)%int(modalityORfiltersPerRotation))
			else:
				outputIndexList.append(int(filterIndex))
		result = pt.tensor(outputIndexList, dtype=pt.long, device=deviceDense)
	elif(pt.is_tensor(filterOutputIndexList)):
		result = filterOutputIndexList.to(dtype=pt.long, device=deviceDense).reshape(-1)
	elif(isinstance(filterOutputIndexList, list)):
		result = pt.tensor(filterOutputIndexList, dtype=pt.long, device=deviceDense).reshape(-1)
	else:
		raise RuntimeError("buildRFfilterOutputIndexTensor error: filterOutputIndexList must be None, a list, or a tensor")
	if(int(result.shape[0]) != numberOfInputFilters):
		raise RuntimeError("buildRFfilterOutputIndexTensor error: output index count must equal numberOfInputFilters")
	if(bool(pt.any(result < 0).item()) or bool(pt.any(result >= int(modalityORfilterChannelsOutput)).item())):
		raise RuntimeError("buildRFfilterOutputIndexTensor error: output index out of range")
	uniqueOutputIndexTensor = pt.unique(result)
	if(int(uniqueOutputIndexTensor.shape[0]) != int(modalityORfilterChannelsOutput)):
		raise RuntimeError("buildRFfilterOutputIndexTensor error: output index map must include every output channel")
	return result


def buildRFfilterOutputCodeList(filterCodeList, filterOutputIndexTensor):
	result = []
	outputIndex = None
	if(not isinstance(filterCodeList, list)):
		raise RuntimeError("buildRFfilterOutputCodeList error: filterCodeList must be a list")
	if(not pt.is_tensor(filterOutputIndexTensor)):
		raise RuntimeError("buildRFfilterOutputCodeList error: filterOutputIndexTensor must be a tensor")
	if(len(filterCodeList) != int(modalityORfilterChannels)):
		raise RuntimeError("buildRFfilterOutputCodeList error: filterCodeList length must equal modalityORfilterChannels")
	if(int(filterOutputIndexTensor.shape[0]) != int(modalityORfilterChannels)):
		raise RuntimeError("buildRFfilterOutputCodeList error: filterOutputIndexTensor length must equal modalityORfilterChannels")
	result = [None]*int(modalityORfilterChannelsOutput)
	for filterIndex in range(int(modalityORfilterChannels)):
		outputIndex = int(filterOutputIndexTensor[filterIndex].item())
		if(result[outputIndex] is None):
			if(modalityORRFfilterRotationallyInvariant):
				result[outputIndex] = buildRFfilterCode(modalityORfilterCodeOutputPrefix, modalityORfilterCodeDefaultIndex, outputIndex, modalityORfilterCodeDefaultIndex, modalityORfilterCodeDefaultIndex)
			else:
				result[outputIndex] = filterCodeList[filterIndex]
	for outputIndex in range(int(modalityORfilterChannelsOutput)):
		if(result[outputIndex] is None):
			raise RuntimeError("buildRFfilterOutputCodeList error: failed to assign output code")
	return result


def mapRFfilterInputIndicesToOutputIndices(rfFilters, filterInputIndexTensor):
	result = None
	filterOutputIndexTensor = None
	if(not isinstance(rfFilters, ORRFfilters)):
		raise RuntimeError("mapRFfilterInputIndicesToOutputIndices error: rfFilters must be an ORRFfilters instance")
	if(not pt.is_tensor(filterInputIndexTensor)):
		raise RuntimeError("mapRFfilterInputIndicesToOutputIndices error: filterInputIndexTensor must be a tensor")
	if(not hasattr(rfFilters, "filterOutputIndexTensor")):
		raise RuntimeError("mapRFfilterInputIndicesToOutputIndices error: rfFilters missing filterOutputIndexTensor")
	if(int(rfFilters.filterOutputIndexTensor.shape[0]) != int(modalityORfilterChannels)):
		raise RuntimeError("mapRFfilterInputIndicesToOutputIndices error: filterOutputIndexTensor length must equal modalityORfilterChannels")
	if(bool(pt.any(filterInputIndexTensor < 0).item()) or bool(pt.any(filterInputIndexTensor >= int(modalityORfilterChannels)).item())):
		raise RuntimeError("mapRFfilterInputIndicesToOutputIndices error: filter input index out of range")
	filterOutputIndexTensor = rfFilters.filterOutputIndexTensor.to(device=filterInputIndexTensor.device, dtype=pt.long)
	result = filterOutputIndexTensor.index_select(0, filterInputIndexTensor.reshape(-1)).reshape(filterInputIndexTensor.shape)
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
	# apply a threshold and take the a) most activated filter in the column, and b) only if it is above modalityORfilterThreshold.
	selectedFilterIndices = None
	selectedFilterResponses = None
	flatColumnPatches = None
	leadingShape = None
	maxOutputIndices = None
	if(not isinstance(rfFilters, ORRFfilters)):
		raise RuntimeError("applyRFfilters error: rfFilters must be an ORRFfilters instance")
	if(not pt.is_tensor(columnPatches)):
		raise RuntimeError("applyRFfilters error: columnPatches must be a tensor")
	if(tokensiationMethodOneColumnPerSnapshotPixel):
		selectedFilterIndices, selectedFilterResponses = applyRFfiltersToPixelColumns(rfFilters, columnPatches)
	else:
		if(columnPatches.dim() == 5):
			leadingShape = columnPatches.shape[0:2]
			flatColumnPatches = columnPatches.reshape(columnPatches.shape[0]*columnPatches.shape[1], columnPatches.shape[2], columnPatches.shape[3], columnPatches.shape[4])
		elif(columnPatches.dim() == 4):
			leadingShape = columnPatches.shape[0:1]
			flatColumnPatches = columnPatches
		else:
			raise RuntimeError("applyRFfilters error: columnPatches rank must be 4 or 5")
		if(flatColumnPatches.shape[1] != int(modalityORfilterColourChannelCount)):
			raise RuntimeError("applyRFfilters error: columnPatches channel count must equal modalityORfilterColourChannelCount")
		if(flatColumnPatches.shape[2] != modalityORpixelsPerColumn or flatColumnPatches.shape[3] != modalityORpixelsPerColumn):
			raise RuntimeError("applyRFfilters error: columnPatches spatial size must equal modalityORpixelsPerColumn")
		if(int(rfFilters.filterTensor.shape[0]) != int(modalityORfilterChannels)):
			raise RuntimeError("applyRFfilters error: rfFilters filter count must equal modalityORfilterChannels")
		if(flatColumnPatches.shape[0] == 0):
			selectedFilterIndices = pt.empty(leadingShape, dtype=pt.long, device=columnPatches.device)
			selectedFilterResponses = pt.empty(leadingShape, dtype=arrayType, device=columnPatches.device)
		else:
			flatColumnPatches = normaliseColumnPatches(flatColumnPatches.to(deviceDense, dtype=arrayType))
			filterResponses = pt.einsum("nchw,fchw->nf", flatColumnPatches, rfFilters.filterTensor)
			maxResponses, maxIndices = pt.max(filterResponses, dim=1)
			maxOutputIndices = mapRFfilterInputIndicesToOutputIndices(rfFilters, maxIndices)
			inactiveIndices = pt.full_like(maxOutputIndices, -1)
			selectedFilterIndices = pt.where(maxResponses >= modalityORfilterThreshold, maxOutputIndices, inactiveIndices)
			selectedFilterResponses = maxResponses
			if(columnPatches.device != selectedFilterIndices.device):
				selectedFilterIndices = selectedFilterIndices.to(columnPatches.device)
				selectedFilterResponses = selectedFilterResponses.to(columnPatches.device)
		selectedFilterIndices = selectedFilterIndices.reshape(leadingShape)
		selectedFilterResponses = selectedFilterResponses.reshape(leadingShape)
	return selectedFilterIndices, selectedFilterResponses


if(tokensiationMethodOneColumnPerSnapshotPixel):
	def applyRFfiltersToPixelColumns(rfFilters, transformedSnapshotTensor):
		resultSelectedFilterIndices = None
		resultSelectedFilterResponses = None
		pixelColumnPatches = None
		flatColumnPatches = None
		leadingShape = None
		filterResponses = None
		maxResponses = None
		maxIndices = None
		maxOutputIndices = None
		inactiveIndices = None
		if(tokensiationMethodOneColumnPerSnapshotPixel):
			validatePixelColumnRFfilterParameters()
			if(not isinstance(rfFilters, ORRFfilters)):
				raise RuntimeError("applyRFfiltersToPixelColumns error: rfFilters must be an ORRFfilters instance")
			if(not pt.is_tensor(transformedSnapshotTensor)):
				raise RuntimeError("applyRFfiltersToPixelColumns error: transformedSnapshotTensor must be a tensor")
			if(transformedSnapshotTensor.dim() != 4):
				raise RuntimeError("applyRFfiltersToPixelColumns error: transformedSnapshotTensor rank must be 4")
			if(int(transformedSnapshotTensor.shape[1]) != int(modalityORfilterColourChannelCount)):
				raise RuntimeError("applyRFfiltersToPixelColumns error: transformedSnapshotTensor channel count must equal modalityORfilterColourChannelCount")
			if(int(transformedSnapshotTensor.shape[2])*int(transformedSnapshotTensor.shape[3]) != int(modalityORnumberOfColumnsVX)):
				raise RuntimeError("applyRFfiltersToPixelColumns error: transformed snapshot pixel count must equal modalityORnumberOfColumnsVX")
			if(int(rfFilters.filterTensor.shape[0]) != int(modalityORfilterChannels)):
				raise RuntimeError("applyRFfiltersToPixelColumns error: rfFilters filter count must equal modalityORfilterChannels")
			if(int(rfFilters.filterTensor.shape[2]) != int(modalityORfilterWidth) or int(rfFilters.filterTensor.shape[3]) != int(modalityORfilterWidth)):
				raise RuntimeError("applyRFfiltersToPixelColumns error: rfFilters spatial size must equal modalityORfilterWidth")
			leadingShape = (int(transformedSnapshotTensor.shape[0]), int(transformedSnapshotTensor.shape[2])*int(transformedSnapshotTensor.shape[3]))
			if(int(transformedSnapshotTensor.shape[0]) == 0):
				resultSelectedFilterIndices = pt.empty(leadingShape, dtype=pt.long, device=transformedSnapshotTensor.device)
				resultSelectedFilterResponses = pt.empty(leadingShape, dtype=arrayType, device=transformedSnapshotTensor.device)
			else:
				pixelColumnPatches = extractPixelColumnFilterPatches(transformedSnapshotTensor.to(dtype=arrayType))
				flatColumnPatches = pixelColumnPatches.reshape(pixelColumnPatches.shape[0]*pixelColumnPatches.shape[1], pixelColumnPatches.shape[2], pixelColumnPatches.shape[3], pixelColumnPatches.shape[4])
				flatColumnPatches = normaliseColumnPatches(flatColumnPatches.to(deviceDense, dtype=arrayType))
				filterResponses = pt.einsum("nchw,fchw->nf", flatColumnPatches, rfFilters.filterTensor)
				maxResponses, maxIndices = pt.max(filterResponses, dim=1)
				maxOutputIndices = mapRFfilterInputIndicesToOutputIndices(rfFilters, maxIndices)
				inactiveIndices = pt.full_like(maxOutputIndices, -1)
				resultSelectedFilterIndices = pt.where(maxResponses >= modalityORfilterThreshold, maxOutputIndices, inactiveIndices)
				resultSelectedFilterResponses = maxResponses
				if(transformedSnapshotTensor.device != resultSelectedFilterIndices.device):
					resultSelectedFilterIndices = resultSelectedFilterIndices.to(transformedSnapshotTensor.device)
					resultSelectedFilterResponses = resultSelectedFilterResponses.to(transformedSnapshotTensor.device)
				resultSelectedFilterIndices = resultSelectedFilterIndices.reshape(leadingShape)
				resultSelectedFilterResponses = resultSelectedFilterResponses.reshape(leadingShape)
		else:
			raise RuntimeError("applyRFfiltersToPixelColumns error: requires tokensiationMethodOneColumnPerSnapshotPixel")
		return resultSelectedFilterIndices, resultSelectedFilterResponses


	def extractPixelColumnFilterPatches(transformedSnapshotTensor):
		result = None
		padding = None
		paddedSnapshotTensor = None
		patchTensor = None
		if(tokensiationMethodOneColumnPerSnapshotPixel):
			validatePixelColumnRFfilterParameters()
			if(not pt.is_tensor(transformedSnapshotTensor)):
				raise RuntimeError("extractPixelColumnFilterPatches error: transformedSnapshotTensor must be a tensor")
			if(transformedSnapshotTensor.dim() != 4):
				raise RuntimeError("extractPixelColumnFilterPatches error: transformedSnapshotTensor rank must be 4")
			if(int(transformedSnapshotTensor.shape[1]) != 3):
				raise RuntimeError("extractPixelColumnFilterPatches error: transformedSnapshotTensor channel count must be 3")
			if(int(transformedSnapshotTensor.shape[2])*int(transformedSnapshotTensor.shape[3]) != int(modalityORnumberOfColumnsVX)):
				raise RuntimeError("extractPixelColumnFilterPatches error: transformed snapshot pixel count must equal modalityORnumberOfColumnsVX")
			padding = int(modalityORfilterWidth//2)
			paddedSnapshotTensor = pt.nn.functional.pad(transformedSnapshotTensor, (padding, padding, padding, padding), mode="replicate")
			patchTensor = paddedSnapshotTensor.unfold(2, int(modalityORfilterWidth), 1).unfold(3, int(modalityORfilterWidth), 1)
			patchTensor = patchTensor.permute(0, 2, 3, 1, 4, 5).contiguous()
			result = patchTensor.view(int(transformedSnapshotTensor.shape[0]), int(transformedSnapshotTensor.shape[2])*int(transformedSnapshotTensor.shape[3]), 3, int(modalityORfilterWidth), int(modalityORfilterWidth))
		else:
			raise RuntimeError("extractPixelColumnFilterPatches error: requires tokensiationMethodOneColumnPerSnapshotPixel")
		return result


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
