"""GIAANNnlp_groundedDataset.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN NLP closed-world grounded dataset

"""

import os
import hashlib

from GIAANNcmn_globalDefs import *

closedWorldGroundedEvalItemsCache = None


def validateClosedWorldGroundedDatasetEnabled(functionName):
	if(functionName is None or functionName == ""):
		raise RuntimeError("validateClosedWorldGroundedDatasetEnabled error: functionName must not be empty")
	if(datasetType not in closedWorldGroundedDatasetTypes):
		raise RuntimeError(functionName + " error: requires datasetType in closedWorldGroundedDatasetTypes")
	return

def validateClosedWorldGroundedToken(tokenValue, tokenName):
	if(tokenName is None or tokenName == ""):
		raise RuntimeError("validateClosedWorldGroundedToken error: tokenName must not be empty")
	if(tokenValue is None or not isinstance(tokenValue, str)):
		raise RuntimeError("validateClosedWorldGroundedToken error: " + tokenName + " must be a string")
	if(tokenValue == ""):
		raise RuntimeError("validateClosedWorldGroundedToken error: " + tokenName + " must not be empty")
	if(closedWorldGroundedPromptTokenSeparator in tokenValue):
		raise RuntimeError("validateClosedWorldGroundedToken error: " + tokenName + " must be one token")
	return

def normaliseClosedWorldGroundedToken(tokenValue, tokenName):
	result = None
	validateClosedWorldGroundedToken(tokenValue, tokenName)
	result = tokenValue.lower()
	return result

def validateClosedWorldGroundedTuple(tupleValue, expectedLength, functionName):
	if(functionName is None or functionName == ""):
		raise RuntimeError("validateClosedWorldGroundedTuple error: functionName must not be empty")
	if(tupleValue is None or not isinstance(tupleValue, tuple)):
		raise RuntimeError(functionName + " error: tupleValue must be a tuple")
	if(not isinstance(expectedLength, int)):
		raise RuntimeError(functionName + " error: expectedLength must be an int")
	if(expectedLength <= 0):
		raise RuntimeError(functionName + " error: expectedLength must be > 0")
	if(len(tupleValue) != expectedLength):
		raise RuntimeError(functionName + " error: tupleValue length must be " + str(expectedLength))
	return

def normaliseClosedWorldGroundedAnswerTuple(answerTuple, tupleName):
	result = None
	normalisedAnswers = []
	if(tupleName is None or tupleName == ""):
		raise RuntimeError("normaliseClosedWorldGroundedAnswerTuple error: tupleName must not be empty")
	if(answerTuple is None or not isinstance(answerTuple, tuple)):
		raise RuntimeError("normaliseClosedWorldGroundedAnswerTuple error: " + tupleName + " must be a tuple")
	for answerIndex, answerValue in enumerate(answerTuple):
		normalisedAnswers.append(normaliseClosedWorldGroundedToken(answerValue, tupleName + "[" + str(answerIndex) + "]"))
	result = tuple(normalisedAnswers)
	return result

def buildClosedWorldGroundedSentence(entityName, propertyName, answerName):
	result = None
	entityName = normaliseClosedWorldGroundedToken(entityName, closedWorldGroundedEvalItemFieldEntity)
	propertyName = normaliseClosedWorldGroundedToken(propertyName, closedWorldGroundedEvalItemFieldProperty)
	answerName = normaliseClosedWorldGroundedToken(answerName, closedWorldGroundedEvalItemFieldTargetAnswer)
	sentenceTokens = [closedWorldGroundedPromptSubjectDeterminer, entityName, closedWorldGroundedPromptPredicate, closedWorldGroundedPromptPropertyLabel, propertyName, closedWorldGroundedPromptConjunction, closedWorldGroundedPromptAnswerLabel, closedWorldGroundedPromptCopula, closedWorldGroundedPromptAnswerQualifier, answerName, closedWorldGroundedPromptSentenceTerminator]
	if(sentenceTokens[closedWorldGroundedPromptRawAnswerTokenIndex] != answerName):
		raise RuntimeError("buildClosedWorldGroundedSentence error: raw answer token index mismatch")
	result = closedWorldGroundedPromptTokenSeparator.join(sentenceTokens)
	return result

def buildClosedWorldGroundedTrainEntry(factTuple):
	result = None
	validateClosedWorldGroundedTuple(factTuple, closedWorldGroundedFactTupleLength, "buildClosedWorldGroundedTrainEntry")
	entityName = factTuple[closedWorldGroundedFactTupleEntityIndex]
	propertyName = factTuple[closedWorldGroundedFactTuplePropertyIndex]
	answerName = factTuple[closedWorldGroundedFactTupleAnswerIndex]
	result = {closedWorldGroundedDatasetTextFieldName: buildClosedWorldGroundedSentence(entityName, propertyName, answerName)}
	return result

def buildClosedWorldGroundedEvalItem(evalItemTuple, sequenceIndex):
	result = None
	if(not isinstance(sequenceIndex, int)):
		raise RuntimeError("buildClosedWorldGroundedEvalItem error: sequenceIndex must be an int")
	if(sequenceIndex < 0):
		raise RuntimeError("buildClosedWorldGroundedEvalItem error: sequenceIndex must be >= 0")
	validateClosedWorldGroundedTuple(evalItemTuple, closedWorldGroundedEvalItemTupleLength, "buildClosedWorldGroundedEvalItem")
	categoryName = normaliseClosedWorldGroundedToken(evalItemTuple[closedWorldGroundedEvalItemTupleCategoryIndex], closedWorldGroundedEvalItemFieldCategory)
	entityName = normaliseClosedWorldGroundedToken(evalItemTuple[closedWorldGroundedEvalItemTupleEntityIndex], closedWorldGroundedEvalItemFieldEntity)
	propertyName = normaliseClosedWorldGroundedToken(evalItemTuple[closedWorldGroundedEvalItemTuplePropertyIndex], closedWorldGroundedEvalItemFieldProperty)
	targetAnswer = normaliseClosedWorldGroundedToken(evalItemTuple[closedWorldGroundedEvalItemTupleTargetAnswerIndex], closedWorldGroundedEvalItemFieldTargetAnswer)
	trueAnswers = normaliseClosedWorldGroundedAnswerTuple(evalItemTuple[closedWorldGroundedEvalItemTupleTrueAnswersIndex], closedWorldGroundedEvalItemFieldTrueAnswers)
	supportedAnswers = normaliseClosedWorldGroundedAnswerTuple(evalItemTuple[closedWorldGroundedEvalItemTupleSupportedAnswersIndex], closedWorldGroundedEvalItemFieldSupportedAnswers)
	result = {closedWorldGroundedEvalItemFieldSequenceIndex: sequenceIndex, closedWorldGroundedEvalItemFieldCategory: categoryName, closedWorldGroundedEvalItemFieldEntity: entityName, closedWorldGroundedEvalItemFieldProperty: propertyName, closedWorldGroundedEvalItemFieldTargetAnswer: targetAnswer, closedWorldGroundedEvalItemFieldTrueAnswers: trueAnswers, closedWorldGroundedEvalItemFieldSupportedAnswers: supportedAnswers, closedWorldGroundedEvalItemFieldText: buildClosedWorldGroundedSentence(entityName, propertyName, targetAnswer), closedWorldGroundedEvalItemFieldAnswerTokenIndex: closedWorldGroundedPromptAnswerTokenIndex}
	return result

def buildClosedWorldGroundedEvalItemWithText(evalItemTuple, sequenceIndex, evalItemText):
	result = None
	result = buildClosedWorldGroundedEvalItemWithTextAndAnswerTokenIndex(evalItemTuple, sequenceIndex, evalItemText, closedWorldGroundedPromptAnswerTokenIndex)
	return result

def buildClosedWorldGroundedEvalItemWithTextAndAnswerTokenIndex(evalItemTuple, sequenceIndex, evalItemText, answerTokenIndex):
	result = None
	if(evalItemText is None or not isinstance(evalItemText, str) or evalItemText == ""):
		raise RuntimeError("buildClosedWorldGroundedEvalItemWithTextAndAnswerTokenIndex error: evalItemText must be a non-empty string")
	if(not isinstance(answerTokenIndex, int)):
		raise RuntimeError("buildClosedWorldGroundedEvalItemWithTextAndAnswerTokenIndex error: answerTokenIndex must be an int")
	if(answerTokenIndex < closedWorldGroundedEvalItemAnswerTokenIndexDynamic):
		raise RuntimeError("buildClosedWorldGroundedEvalItemWithTextAndAnswerTokenIndex error: answerTokenIndex is below dynamic marker")
	result = buildClosedWorldGroundedEvalItem(evalItemTuple, sequenceIndex)
	result[closedWorldGroundedEvalItemFieldText] = evalItemText
	result[closedWorldGroundedEvalItemFieldAnswerTokenIndex] = answerTokenIndex
	return result

def loadClosedWorldGroundedDataset(inferenceMode):
	result = None
	datasetEntries = []
	validateClosedWorldGroundedDatasetEnabled("loadClosedWorldGroundedDataset")
	if(not isinstance(inferenceMode, bool)):
		raise RuntimeError("loadClosedWorldGroundedDataset error: inferenceMode must be a bool")
	if(inferenceMode):
		for evalItem in getClosedWorldGroundedEvalItems():
			datasetEntries.append({closedWorldGroundedDatasetTextFieldName: evalItem[closedWorldGroundedEvalItemFieldText]})
	else:
		if(closedWorldGroundedDatasetGenerated):
			for factTuple in closedWorldGroundedTrainFactTuples:
				datasetEntries.append(buildClosedWorldGroundedTrainEntry(factTuple))
		else:
			datasetEntries = loadClosedWorldGroundedHfDatasetEntries(False)
	result = datasetEntries
	return result

def getClosedWorldGroundedEvalItems():
	result = None
	evalItems = []
	global closedWorldGroundedEvalItemsCache
	validateClosedWorldGroundedDatasetEnabled("getClosedWorldGroundedEvalItems")
	if(closedWorldGroundedEvalItemsCache is None):
		if(closedWorldGroundedDatasetGenerated):
			for sequenceIndex, evalItemTuple in enumerate(closedWorldGroundedEvalItemTuples):
				evalItems.append(buildClosedWorldGroundedEvalItem(evalItemTuple, sequenceIndex))
		else:
			evalItems = loadClosedWorldGroundedHfEvalItems()
		closedWorldGroundedEvalItemsCache = evalItems
	result = closedWorldGroundedEvalItemsCache
	return result

def loadClosedWorldGroundedHfDatasetEntries(inferenceMode):
	result = None
	datasetEntries = []
	if(not isinstance(inferenceMode, bool)):
		raise RuntimeError("loadClosedWorldGroundedHfDatasetEntries error: inferenceMode must be a bool")
	if(closedWorldGroundedDatasetGenerated):
		raise RuntimeError("loadClosedWorldGroundedHfDatasetEntries error: requires closedWorldGroundedDatasetGenerated==False")
	if(inferenceMode):
		evalItems = getClosedWorldGroundedEvalItems()
	else:
		evalItems = loadClosedWorldGroundedHfTrainItems()
	for evalItem in evalItems:
		datasetEntries.append({closedWorldGroundedDatasetTextFieldName: evalItem[closedWorldGroundedEvalItemFieldText]})
	result = datasetEntries
	return result

def loadClosedWorldGroundedHfTrainItems():
	result = None
	if(closedWorldGroundedDatasetGenerated):
		raise RuntimeError("loadClosedWorldGroundedHfTrainItems error: requires closedWorldGroundedDatasetGenerated==False")
	trainBaseItems = loadClosedWorldGroundedHfBaseItems(closedWorldGroundedDatasetTypeToTrainSplitDict[datasetType], trainMaxSequences)
	result = buildClosedWorldGroundedHfProbeItems(trainBaseItems, True, False)
	return result

def loadClosedWorldGroundedHfEvalItems():
	result = None
	if(closedWorldGroundedDatasetGenerated):
		raise RuntimeError("loadClosedWorldGroundedHfEvalItems error: requires closedWorldGroundedDatasetGenerated==False")
	result = loadClosedWorldGroundedHfEvalProbeItems()
	return result

def loadClosedWorldGroundedHfEvalProbeItems():
	result = None
	evalItemPools = []
	if(closedWorldGroundedDatasetGenerated):
		raise RuntimeError("loadClosedWorldGroundedHfEvalProbeItems error: requires closedWorldGroundedDatasetGenerated==False")
	trainBaseItems = loadClosedWorldGroundedHfBaseItems(closedWorldGroundedDatasetTypeToTrainSplitDict[datasetType], closedWorldGroundedHfEvalMaxItems)
	heldoutBaseItems = loadClosedWorldGroundedHfBaseItems(closedWorldGroundedDatasetTypeToEvalSplitDict[datasetType], closedWorldGroundedHfEvalMaxItems)
	evalItemPools.append(buildClosedWorldGroundedHfProbeItems(trainBaseItems, True, False))
	evalItemPools.append(buildClosedWorldGroundedHfProbeItems(trainBaseItems, True, True))
	evalItemPools.append(buildClosedWorldGroundedHfProbeItems(heldoutBaseItems, False, False))
	evalItemPools.append(buildClosedWorldGroundedHfProbeItems(heldoutBaseItems, False, True))
	result = selectClosedWorldGroundedHfEvalItems(evalItemPools)
	return result

def loadClosedWorldGroundedHfBaseItems(splitName, targetDatasetItemCount):
	result = None
	baseItems = []
	baseItemsByAnswer = {}
	seenAnswersByEntityName = {}
	seenClaimTextByEntityName = {}
	seenClaimSignatureTokensByEntityName = {}
	if(splitName is None or not isinstance(splitName, str) or splitName == ""):
		raise RuntimeError("loadClosedWorldGroundedHfBaseItems error: splitName must be a non-empty string")
	if(not isinstance(targetDatasetItemCount, int)):
		raise RuntimeError("loadClosedWorldGroundedHfBaseItems error: targetDatasetItemCount must be an int")
	if(targetDatasetItemCount < closedWorldGroundedHfDatasetItemCountMinimum):
		raise RuntimeError("loadClosedWorldGroundedHfBaseItems error: targetDatasetItemCount is below closedWorldGroundedHfDatasetItemCountMinimum")
	if(inferenceReportGroundedAccuracyMod1_labelBalancedDataset):
		baseItemsByAnswer = initialiseClosedWorldGroundedHfBaseItemsByAnswer()
	rawDatasetSplit = loadClosedWorldGroundedHfRawDatasetSplit(splitName)
	for rawEntryIndex, rawDatasetEntry in enumerate(rawDatasetSplit):
		validateClosedWorldGroundedHfRawEntry(rawDatasetEntry, rawEntryIndex)
		entityName = buildClosedWorldGroundedHfEntityName(rawDatasetEntry)
		answerName = getClosedWorldGroundedHfAnswerName(rawDatasetEntry)
		if(inferenceReportGroundedRealisticNLPmetric):
			claimText = getClosedWorldGroundedHfClaimText(rawDatasetEntry)
		elif(inferenceReportGroundedStrongerGroundedNLPmetric):
			claimSignatureTokens = buildClosedWorldGroundedHfClaimSignatureTokens(rawDatasetEntry)
		if(entityName in seenAnswersByEntityName):
			if(seenAnswersByEntityName[entityName] != answerName):
				raise RuntimeError("loadClosedWorldGroundedHfBaseItems error: duplicate entityName has conflicting labels: " + entityName)
			if(inferenceReportGroundedRealisticNLPmetric):
				if(seenClaimTextByEntityName[entityName] != claimText):
					raise RuntimeError("loadClosedWorldGroundedHfBaseItems error: duplicate entityName has conflicting claim text: " + entityName)
			elif(inferenceReportGroundedStrongerGroundedNLPmetric):
				if(seenClaimSignatureTokensByEntityName[entityName] != claimSignatureTokens):
					raise RuntimeError("loadClosedWorldGroundedHfBaseItems error: duplicate entityName has conflicting claim signatures: " + entityName)
		else:
			seenAnswersByEntityName[entityName] = answerName
			if(inferenceReportGroundedRealisticNLPmetric):
				seenClaimTextByEntityName[entityName] = claimText
			elif(inferenceReportGroundedStrongerGroundedNLPmetric):
				seenClaimSignatureTokensByEntityName[entityName] = claimSignatureTokens
			baseItem = buildClosedWorldGroundedHfBaseItem(rawDatasetEntry)
			if(inferenceReportGroundedAccuracyMod1_labelBalancedDataset):
				addClosedWorldGroundedHfBaseItemByAnswer(baseItemsByAnswer, baseItem)
				if(hasClosedWorldGroundedHfBalancedBaseItemsTarget(baseItemsByAnswer, targetDatasetItemCount)):
					break
			else:
				baseItems.append(baseItem)
				if(len(baseItems) >= targetDatasetItemCount):
					break
	if(inferenceReportGroundedAccuracyMod1_labelBalancedDataset):
		baseItems = buildClosedWorldGroundedHfBalancedBaseItems(baseItemsByAnswer, targetDatasetItemCount)
	if(len(baseItems) < closedWorldGroundedHfDatasetItemCountMinimum):
		raise RuntimeError("loadClosedWorldGroundedHfBaseItems error: no valid entries loaded from " + datasetName + " split " + splitName)
	result = baseItems
	return result

def initialiseClosedWorldGroundedHfBaseItemsByAnswer():
	result = None
	baseItemsByAnswer = {}
	if(closedWorldGroundedHfAnswerOptions is None or not isinstance(closedWorldGroundedHfAnswerOptions, list)):
		raise RuntimeError("initialiseClosedWorldGroundedHfBaseItemsByAnswer error: closedWorldGroundedHfAnswerOptions must be a list")
	if(len(closedWorldGroundedHfAnswerOptions) < closedWorldGroundedHfDatasetItemCountMinimum):
		raise RuntimeError("initialiseClosedWorldGroundedHfBaseItemsByAnswer error: closedWorldGroundedHfAnswerOptions must not be empty")
	for answerName in closedWorldGroundedHfAnswerOptions:
		baseItemsByAnswer[answerName] = []
	result = baseItemsByAnswer
	return result

def addClosedWorldGroundedHfBaseItemByAnswer(baseItemsByAnswer, baseItem):
	if(baseItemsByAnswer is None or not isinstance(baseItemsByAnswer, dict)):
		raise RuntimeError("addClosedWorldGroundedHfBaseItemByAnswer error: baseItemsByAnswer must be a dict")
	if(baseItem is None or not isinstance(baseItem, dict)):
		raise RuntimeError("addClosedWorldGroundedHfBaseItemByAnswer error: baseItem must be a dict")
	if(closedWorldGroundedEvalItemFieldTargetAnswer not in baseItem):
		raise RuntimeError("addClosedWorldGroundedHfBaseItemByAnswer error: baseItem missing target answer")
	answerName = baseItem[closedWorldGroundedEvalItemFieldTargetAnswer]
	if(answerName not in baseItemsByAnswer):
		raise RuntimeError("addClosedWorldGroundedHfBaseItemByAnswer error: unsupported answerName " + str(answerName))
	baseItemsByAnswer[answerName].append(baseItem)
	return

def hasClosedWorldGroundedHfBalancedBaseItemsTarget(baseItemsByAnswer, targetDatasetItemCount):
	result = None
	if(baseItemsByAnswer is None or not isinstance(baseItemsByAnswer, dict)):
		raise RuntimeError("hasClosedWorldGroundedHfBalancedBaseItemsTarget error: baseItemsByAnswer must be a dict")
	targetAnswerCounts = calculateClosedWorldGroundedHfBalancedTargetAnswerCounts(targetDatasetItemCount)
	result = True
	for answerName in targetAnswerCounts:
		if(answerName not in baseItemsByAnswer):
			raise RuntimeError("hasClosedWorldGroundedHfBalancedBaseItemsTarget error: missing answerName " + answerName)
		if(len(baseItemsByAnswer[answerName]) < targetAnswerCounts[answerName]):
			result = False
	return result

def calculateClosedWorldGroundedHfBalancedTargetAnswerCounts(targetDatasetItemCount):
	result = None
	targetAnswerCounts = {}
	if(not isinstance(targetDatasetItemCount, int)):
		raise RuntimeError("calculateClosedWorldGroundedHfBalancedTargetAnswerCounts error: targetDatasetItemCount must be an int")
	if(targetDatasetItemCount < closedWorldGroundedHfDatasetItemCountMinimum):
		raise RuntimeError("calculateClosedWorldGroundedHfBalancedTargetAnswerCounts error: targetDatasetItemCount is below closedWorldGroundedHfDatasetItemCountMinimum")
	if(closedWorldGroundedHfAnswerOptions is None or not isinstance(closedWorldGroundedHfAnswerOptions, list)):
		raise RuntimeError("calculateClosedWorldGroundedHfBalancedTargetAnswerCounts error: closedWorldGroundedHfAnswerOptions must be a list")
	if(len(closedWorldGroundedHfAnswerOptions) < closedWorldGroundedHfDatasetItemCountMinimum):
		raise RuntimeError("calculateClosedWorldGroundedHfBalancedTargetAnswerCounts error: closedWorldGroundedHfAnswerOptions must not be empty")
	answerCountBase = targetDatasetItemCount // len(closedWorldGroundedHfAnswerOptions)
	if(answerCountBase < closedWorldGroundedHfDatasetItemCountMinimum):
		raise RuntimeError("calculateClosedWorldGroundedHfBalancedTargetAnswerCounts error: targetDatasetItemCount must support at least one item per answer")
	for answerName in closedWorldGroundedHfAnswerOptions:
		targetAnswerCounts[answerName] = answerCountBase
	result = targetAnswerCounts
	return result

def buildClosedWorldGroundedHfBalancedBaseItems(baseItemsByAnswer, targetDatasetItemCount):
	result = None
	balancedBaseItems = []
	balancedAnswerCount = None
	if(baseItemsByAnswer is None or not isinstance(baseItemsByAnswer, dict)):
		raise RuntimeError("buildClosedWorldGroundedHfBalancedBaseItems error: baseItemsByAnswer must be a dict")
	balancedAnswerCount = calculateClosedWorldGroundedHfBalancedAvailableAnswerCount(baseItemsByAnswer, targetDatasetItemCount)
	for answerItemIndex in range(balancedAnswerCount):
		for answerName in closedWorldGroundedHfAnswerOptions:
			balancedBaseItems.append(baseItemsByAnswer[answerName][answerItemIndex])
	if(len(balancedBaseItems) < closedWorldGroundedHfDatasetItemCountMinimum):
		raise RuntimeError("buildClosedWorldGroundedHfBalancedBaseItems error: no balanced base items selected")
	result = balancedBaseItems
	return result

def calculateClosedWorldGroundedHfBalancedAvailableAnswerCount(baseItemsByAnswer, targetDatasetItemCount):
	result = None
	balancedAnswerCount = None
	targetAnswerCounts = calculateClosedWorldGroundedHfBalancedTargetAnswerCounts(targetDatasetItemCount)
	if(baseItemsByAnswer is None or not isinstance(baseItemsByAnswer, dict)):
		raise RuntimeError("calculateClosedWorldGroundedHfBalancedAvailableAnswerCount error: baseItemsByAnswer must be a dict")
	for answerName in closedWorldGroundedHfAnswerOptions:
		if(answerName not in baseItemsByAnswer):
			raise RuntimeError("calculateClosedWorldGroundedHfBalancedAvailableAnswerCount error: missing answerName " + answerName)
		if(len(baseItemsByAnswer[answerName]) < closedWorldGroundedHfDatasetItemCountMinimum):
			raise RuntimeError("calculateClosedWorldGroundedHfBalancedAvailableAnswerCount error: no examples available for answerName " + answerName)
		answerCount = min(targetAnswerCounts[answerName], len(baseItemsByAnswer[answerName]))
		if(balancedAnswerCount is None or answerCount < balancedAnswerCount):
			balancedAnswerCount = answerCount
	if(balancedAnswerCount is None or balancedAnswerCount < closedWorldGroundedHfDatasetItemCountMinimum):
		raise RuntimeError("calculateClosedWorldGroundedHfBalancedAvailableAnswerCount error: no balanced answer count available")
	result = balancedAnswerCount
	return result

def buildClosedWorldGroundedHfProbeItems(baseItems, trainingSupported, falseTarget):
	result = None
	evalItems = []
	if(baseItems is None or not isinstance(baseItems, list)):
		raise RuntimeError("buildClosedWorldGroundedHfProbeItems error: baseItems must be a list")
	if(not isinstance(trainingSupported, bool)):
		raise RuntimeError("buildClosedWorldGroundedHfProbeItems error: trainingSupported must be a bool")
	if(not isinstance(falseTarget, bool)):
		raise RuntimeError("buildClosedWorldGroundedHfProbeItems error: falseTarget must be a bool")
	for baseItem in baseItems:
		evalItems.append(buildClosedWorldGroundedHfProbeItem(baseItem, len(evalItems), trainingSupported, falseTarget))
	result = evalItems
	return result

def selectClosedWorldGroundedHfEvalItems(evalItemPools):
	result = None
	selectedEvalItems = []
	if(evalItemPools is None or not isinstance(evalItemPools, list)):
		raise RuntimeError("selectClosedWorldGroundedHfEvalItems error: evalItemPools must be a list")
	if(len(evalItemPools) < closedWorldGroundedHfDatasetItemCountMinimum):
		raise RuntimeError("selectClosedWorldGroundedHfEvalItems error: evalItemPools must not be empty")
	if(inferenceReportGroundedAccuracyMod1_labelBalancedDataset):
		selectedEvalItems = selectClosedWorldGroundedHfBalancedEvalItems(evalItemPools)
	else:
		poolItemIndices = [closedWorldGroundedHfInitialPoolItemIndex for evalItemPool in evalItemPools]
		selectedItemAdded = True
		while(len(selectedEvalItems) < closedWorldGroundedHfEvalMaxItems and selectedItemAdded):
			selectedItemAdded = False
			for poolIndex, evalItemPool in enumerate(evalItemPools):
				if(not isinstance(evalItemPool, list)):
					raise RuntimeError("selectClosedWorldGroundedHfEvalItems error: evalItemPool must be a list")
				if(len(selectedEvalItems) < closedWorldGroundedHfEvalMaxItems and poolItemIndices[poolIndex] < len(evalItemPool)):
					evalItem = evalItemPool[poolItemIndices[poolIndex]]
					evalItem[closedWorldGroundedEvalItemFieldSequenceIndex] = len(selectedEvalItems)
					selectedEvalItems.append(evalItem)
					poolItemIndices[poolIndex] += closedWorldGroundedHfPoolItemIndexIncrement
					selectedItemAdded = True
	if(len(selectedEvalItems) < closedWorldGroundedHfDatasetItemCountMinimum):
		raise RuntimeError("selectClosedWorldGroundedHfEvalItems error: no eval items selected")
	result = selectedEvalItems
	return result

def selectClosedWorldGroundedHfBalancedEvalItems(evalItemPools):
	result = None
	selectedEvalItems = []
	balancedPoolItemCount = calculateClosedWorldGroundedHfBalancedEvalPoolItemCount(evalItemPools)
	for poolItemIndex in range(balancedPoolItemCount):
		for evalItemPool in evalItemPools:
			evalItem = evalItemPool[poolItemIndex]
			evalItem[closedWorldGroundedEvalItemFieldSequenceIndex] = len(selectedEvalItems)
			selectedEvalItems.append(evalItem)
	result = selectedEvalItems
	return result

def calculateClosedWorldGroundedHfBalancedEvalPoolItemCount(evalItemPools):
	result = None
	balancedPoolItemCount = None
	if(evalItemPools is None or not isinstance(evalItemPools, list)):
		raise RuntimeError("calculateClosedWorldGroundedHfBalancedEvalPoolItemCount error: evalItemPools must be a list")
	if(len(evalItemPools) < closedWorldGroundedHfDatasetItemCountMinimum):
		raise RuntimeError("calculateClosedWorldGroundedHfBalancedEvalPoolItemCount error: evalItemPools must not be empty")
	balancedPoolItemCount = closedWorldGroundedHfEvalMaxItems // len(evalItemPools)
	if(balancedPoolItemCount < closedWorldGroundedHfDatasetItemCountMinimum):
		raise RuntimeError("calculateClosedWorldGroundedHfBalancedEvalPoolItemCount error: closedWorldGroundedHfEvalMaxItems must support at least one item per eval pool")
	for evalItemPool in evalItemPools:
		if(not isinstance(evalItemPool, list)):
			raise RuntimeError("calculateClosedWorldGroundedHfBalancedEvalPoolItemCount error: evalItemPool must be a list")
		if(len(evalItemPool) < closedWorldGroundedHfDatasetItemCountMinimum):
			raise RuntimeError("calculateClosedWorldGroundedHfBalancedEvalPoolItemCount error: evalItemPool must not be empty")
		if(len(evalItemPool) < balancedPoolItemCount):
			balancedPoolItemCount = len(evalItemPool)
	result = balancedPoolItemCount
	return result

def loadClosedWorldGroundedHfRawDatasetSplit(splitName):
	result = None
	if(splitName is None or not isinstance(splitName, str) or splitName == ""):
		raise RuntimeError("loadClosedWorldGroundedHfRawDatasetSplit error: splitName must be a non-empty string")
	from datasets import load_dataset
	if(datasetCfg == ""):
		result = load_dataset(datasetName, split=splitName, streaming=closedWorldGroundedHfStreaming, trust_remote_code=True)
	else:
		result = load_dataset(datasetName, datasetCfg, split=splitName, streaming=closedWorldGroundedHfStreaming, trust_remote_code=True)
	return result

def buildClosedWorldGroundedHfBaseItem(rawDatasetEntry):
	result = None
	entityName = buildClosedWorldGroundedHfEntityName(rawDatasetEntry)
	answerName = getClosedWorldGroundedHfAnswerName(rawDatasetEntry)
	result = {closedWorldGroundedEvalItemFieldEntity: entityName, closedWorldGroundedEvalItemFieldTargetAnswer: answerName}
	if(inferenceReportGroundedRealisticNLPmetric):
		result[closedWorldGroundedEvalItemFieldClaimText] = getClosedWorldGroundedHfClaimText(rawDatasetEntry)
	elif(inferenceReportGroundedStrongerGroundedNLPmetric):
		result[closedWorldGroundedEvalItemFieldClaimSignatureTokens] = buildClosedWorldGroundedHfClaimSignatureTokens(rawDatasetEntry)
	return result

def buildClosedWorldGroundedHfProbeItem(baseItem, sequenceIndex, trainingSupported, falseTarget):
	result = None
	if(not isinstance(baseItem, dict)):
		raise RuntimeError("buildClosedWorldGroundedHfProbeItem error: baseItem must be a dict")
	if(not isinstance(trainingSupported, bool)):
		raise RuntimeError("buildClosedWorldGroundedHfProbeItem error: trainingSupported must be a bool")
	if(not isinstance(falseTarget, bool)):
		raise RuntimeError("buildClosedWorldGroundedHfProbeItem error: falseTarget must be a bool")
	if(closedWorldGroundedEvalItemFieldEntity not in baseItem or closedWorldGroundedEvalItemFieldTargetAnswer not in baseItem):
		raise RuntimeError("buildClosedWorldGroundedHfProbeItem error: baseItem missing required fields")
	entityName = baseItem[closedWorldGroundedEvalItemFieldEntity]
	answerName = baseItem[closedWorldGroundedEvalItemFieldTargetAnswer]
	if(falseTarget):
		targetAnswer = getClosedWorldGroundedHfAlternativeAnswer(answerName)
	else:
		targetAnswer = answerName
	if(trainingSupported):
		categoryName = closedWorldGroundedLabelDirectSupport
		supportedAnswers = (answerName,)
	else:
		supportedAnswers = ()
		if(falseTarget):
			categoryName = closedWorldGroundedLabelUnsupportedFalse
		else:
			categoryName = closedWorldGroundedLabelUnsupportedWorldTrue
	evalItemTuple = (categoryName, entityName, closedWorldGroundedHfPropertyVerdict, targetAnswer, (answerName,), supportedAnswers)
	if(inferenceReportGroundedRealisticNLPmetric):
		result = buildClosedWorldGroundedHfRealisticNLPmetricEvalItem(evalItemTuple, sequenceIndex, baseItem, targetAnswer)
	elif(inferenceReportGroundedStrongerGroundedNLPmetric):
		result = buildClosedWorldGroundedHfStrongerGroundedNLPmetricEvalItem(evalItemTuple, sequenceIndex, baseItem, targetAnswer)
	else:
		result = buildClosedWorldGroundedEvalItem(evalItemTuple, sequenceIndex)
	return result

def buildClosedWorldGroundedHfRealisticNLPmetricEvalItem(evalItemTuple, sequenceIndex, baseItem, targetAnswer):
	result = None
	if(baseItem is None or not isinstance(baseItem, dict)):
		raise RuntimeError("buildClosedWorldGroundedHfRealisticNLPmetricEvalItem error: baseItem must be a dict")
	if(closedWorldGroundedEvalItemFieldEntity not in baseItem):
		raise RuntimeError("buildClosedWorldGroundedHfRealisticNLPmetricEvalItem error: baseItem missing entity")
	if(closedWorldGroundedEvalItemFieldClaimText not in baseItem):
		raise RuntimeError("buildClosedWorldGroundedHfRealisticNLPmetricEvalItem error: baseItem missing claim text")
	evalItemText = buildClosedWorldGroundedHfRealisticNLPmetricSentence(baseItem[closedWorldGroundedEvalItemFieldEntity], baseItem[closedWorldGroundedEvalItemFieldClaimText], targetAnswer)
	result = buildClosedWorldGroundedEvalItemWithTextAndAnswerTokenIndex(evalItemTuple, sequenceIndex, evalItemText, closedWorldGroundedEvalItemAnswerTokenIndexDynamic)
	return result

def buildClosedWorldGroundedHfRealisticNLPmetricSentence(entityName, claimText, answerName):
	result = None
	entityName = normaliseClosedWorldGroundedToken(entityName, closedWorldGroundedEvalItemFieldEntity)
	claimText = getClosedWorldGroundedHfBoundedClaimText(claimText)
	answerName = normaliseClosedWorldGroundedToken(answerName, closedWorldGroundedEvalItemFieldTargetAnswer)
	sentencePrefix = closedWorldGroundedRealisticNLPmetricPromptClaimLabel + closedWorldGroundedPromptTokenSeparator + entityName + closedWorldGroundedPromptTokenSeparator + closedWorldGroundedRealisticNLPmetricPromptStates + closedWorldGroundedPromptTokenSeparator + closedWorldGroundedRealisticNLPmetricPromptThat
	sentenceSuffix = closedWorldGroundedPromptConjunction + closedWorldGroundedPromptTokenSeparator + closedWorldGroundedHfPropertyVerdict + closedWorldGroundedPromptTokenSeparator + closedWorldGroundedPromptPredicate + closedWorldGroundedPromptTokenSeparator + closedWorldGroundedPromptAnswerQualifier + closedWorldGroundedPromptTokenSeparator + answerName + closedWorldGroundedPromptTokenSeparator + closedWorldGroundedPromptSentenceTerminator
	result = sentencePrefix + closedWorldGroundedPromptTokenSeparator + claimText + closedWorldGroundedPromptTokenSeparator + sentenceSuffix
	return result

def getClosedWorldGroundedHfClaimText(rawDatasetEntry):
	result = None
	claimText = None
	if(rawDatasetEntry is None or not isinstance(rawDatasetEntry, dict)):
		raise RuntimeError("getClosedWorldGroundedHfClaimText error: rawDatasetEntry must be a dict")
	if(closedWorldGroundedHfFieldClaim not in rawDatasetEntry):
		raise RuntimeError("getClosedWorldGroundedHfClaimText error: rawDatasetEntry missing claim field")
	claimText = rawDatasetEntry[closedWorldGroundedHfFieldClaim]
	result = getClosedWorldGroundedHfBoundedClaimText(claimText)
	return result

def getClosedWorldGroundedHfBoundedClaimText(claimText):
	result = None
	claimWords = []
	if(claimText is None or not isinstance(claimText, str) or claimText == ""):
		raise RuntimeError("getClosedWorldGroundedHfBoundedClaimText error: claimText must be a non-empty string")
	claimTextNormalised = closedWorldGroundedPromptTokenSeparator.join(claimText.split())
	claimTextNormalised = sanitiseClosedWorldGroundedHfClaimTextForSingleSequence(claimTextNormalised)
	while(len(claimTextNormalised) > 0 and claimTextNormalised[-1] in closedWorldGroundedRealisticNLPmetricClaimTerminalCharacters):
		claimTextNormalised = claimTextNormalised[:-1]
	if(claimTextNormalised == ""):
		raise RuntimeError("getClosedWorldGroundedHfBoundedClaimText error: claimTextNormalised must not be empty")
	claimWords = claimTextNormalised.split(closedWorldGroundedPromptTokenSeparator)
	if(len(claimWords) > closedWorldGroundedRealisticNLPmetricMaxClaimWords):
		claimWords = claimWords[:closedWorldGroundedRealisticNLPmetricMaxClaimWords]
	result = closedWorldGroundedPromptTokenSeparator.join(claimWords)
	return result

def sanitiseClosedWorldGroundedHfClaimTextForSingleSequence(claimText):
	result = None
	claimTextCharacters = []
	if(claimText is None or not isinstance(claimText, str) or claimText == ""):
		raise RuntimeError("sanitiseClosedWorldGroundedHfClaimTextForSingleSequence error: claimText must be a non-empty string")
	for claimCharacter in claimText:
		if(claimCharacter in closedWorldGroundedRealisticNLPmetricClaimTerminalCharacters):
			claimTextCharacters.append(closedWorldGroundedRealisticNLPmetricClaimTerminalCharacterReplacement)
		else:
			claimTextCharacters.append(claimCharacter)
	result = str().join(claimTextCharacters)
	result = closedWorldGroundedPromptTokenSeparator.join(result.split())
	if(result == ""):
		raise RuntimeError("sanitiseClosedWorldGroundedHfClaimTextForSingleSequence error: result must be a non-empty string")
	return result

def buildClosedWorldGroundedHfStrongerGroundedNLPmetricEvalItem(evalItemTuple, sequenceIndex, baseItem, targetAnswer):
	result = None
	if(baseItem is None or not isinstance(baseItem, dict)):
		raise RuntimeError("buildClosedWorldGroundedHfStrongerGroundedNLPmetricEvalItem error: baseItem must be a dict")
	if(closedWorldGroundedEvalItemFieldEntity not in baseItem):
		raise RuntimeError("buildClosedWorldGroundedHfStrongerGroundedNLPmetricEvalItem error: baseItem missing entity")
	if(closedWorldGroundedEvalItemFieldClaimSignatureTokens not in baseItem):
		raise RuntimeError("buildClosedWorldGroundedHfStrongerGroundedNLPmetricEvalItem error: baseItem missing claim signature tokens")
	evalItemText = buildClosedWorldGroundedHfStrongerGroundedNLPmetricSentence(baseItem[closedWorldGroundedEvalItemFieldEntity], baseItem[closedWorldGroundedEvalItemFieldClaimSignatureTokens], targetAnswer)
	result = buildClosedWorldGroundedEvalItemWithText(evalItemTuple, sequenceIndex, evalItemText)
	return result

def buildClosedWorldGroundedHfStrongerGroundedNLPmetricSentence(entityName, claimSignatureTokens, answerName):
	result = None
	entityName = normaliseClosedWorldGroundedToken(entityName, closedWorldGroundedEvalItemFieldEntity)
	answerName = normaliseClosedWorldGroundedToken(answerName, closedWorldGroundedEvalItemFieldTargetAnswer)
	validateClosedWorldGroundedHfClaimSignatureTokens(claimSignatureTokens)
	sentenceTokens = [closedWorldGroundedStrongerGroundedNLPmetricPromptClaimLabel, entityName]
	for claimSignatureToken in claimSignatureTokens:
		sentenceTokens.append(claimSignatureToken)
	sentenceTokens.extend([closedWorldGroundedPromptPropertyLabel, closedWorldGroundedHfPropertyVerdict, closedWorldGroundedPromptPredicate, answerName, closedWorldGroundedPromptSentenceTerminator])
	if(sentenceTokens[closedWorldGroundedPromptRawAnswerTokenIndex] != answerName):
		raise RuntimeError("buildClosedWorldGroundedHfStrongerGroundedNLPmetricSentence error: raw answer token index mismatch")
	result = closedWorldGroundedPromptTokenSeparator.join(sentenceTokens)
	return result

def buildClosedWorldGroundedHfClaimSignatureTokens(rawDatasetEntry):
	result = None
	claimText = None
	if(rawDatasetEntry is None or not isinstance(rawDatasetEntry, dict)):
		raise RuntimeError("buildClosedWorldGroundedHfClaimSignatureTokens error: rawDatasetEntry must be a dict")
	if(closedWorldGroundedHfFieldClaim not in rawDatasetEntry):
		raise RuntimeError("buildClosedWorldGroundedHfClaimSignatureTokens error: rawDatasetEntry missing claim field")
	claimText = rawDatasetEntry[closedWorldGroundedHfFieldClaim]
	result = buildClosedWorldGroundedHfClaimSignatureTokensFromText(claimText)
	return result

def buildClosedWorldGroundedHfClaimSignatureTokensFromText(claimText):
	result = None
	claimSignatureTokens = []
	if(claimText is None or not isinstance(claimText, str) or claimText == ""):
		raise RuntimeError("buildClosedWorldGroundedHfClaimSignatureTokensFromText error: claimText must be a non-empty string")
	claimDigest = hashlib.sha256(claimText.encode(closedWorldGroundedInferencePromptFileEncoding)).digest()
	if(len(claimDigest) < closedWorldGroundedHfClaimDigestByteOffset + closedWorldGroundedHfClaimDigestTokenCount):
		raise RuntimeError("buildClosedWorldGroundedHfClaimSignatureTokensFromText error: claimDigest is shorter than required claim digest span")
	for claimDigestByteIndex in range(closedWorldGroundedHfClaimDigestTokenCount):
		claimDigestByte = claimDigest[closedWorldGroundedHfClaimDigestByteOffset + claimDigestByteIndex]
		claimSignatureTokens.append(str(claimDigestByte).zfill(closedWorldGroundedHfClaimDigestTokenWidth))
	result = tuple(claimSignatureTokens)
	return result

def validateClosedWorldGroundedHfClaimSignatureTokens(claimSignatureTokens):
	if(claimSignatureTokens is None or not isinstance(claimSignatureTokens, tuple)):
		raise RuntimeError("validateClosedWorldGroundedHfClaimSignatureTokens error: claimSignatureTokens must be a tuple")
	if(len(claimSignatureTokens) != closedWorldGroundedHfClaimDigestTokenCount):
		raise RuntimeError("validateClosedWorldGroundedHfClaimSignatureTokens error: claimSignatureTokens length mismatch")
	for claimSignatureTokenIndex, claimSignatureToken in enumerate(claimSignatureTokens):
		normaliseClosedWorldGroundedToken(claimSignatureToken, "claimSignatureToken" + str(claimSignatureTokenIndex))
	return

def validateClosedWorldGroundedHfRawEntry(rawDatasetEntry, rawEntryIndex):
	if(not isinstance(rawEntryIndex, int)):
		raise RuntimeError("validateClosedWorldGroundedHfRawEntry error: rawEntryIndex must be an int")
	if(rawEntryIndex < 0):
		raise RuntimeError("validateClosedWorldGroundedHfRawEntry error: rawEntryIndex must be >= 0")
	if(not isinstance(rawDatasetEntry, dict)):
		raise RuntimeError("validateClosedWorldGroundedHfRawEntry error: rawDatasetEntry must be a dict")
	validateClosedWorldGroundedHfField(rawDatasetEntry, closedWorldGroundedHfFieldId, rawEntryIndex)
	validateClosedWorldGroundedHfField(rawDatasetEntry, closedWorldGroundedHfFieldClaim, rawEntryIndex)
	if(not isinstance(rawDatasetEntry[closedWorldGroundedHfFieldClaim], str) or rawDatasetEntry[closedWorldGroundedHfFieldClaim] == ""):
		raise RuntimeError("validateClosedWorldGroundedHfRawEntry error: claim must be a non-empty string at rawEntryIndex=" + str(rawEntryIndex))
	if(datasetType == datasetTypeClosedWorldGrounded2):
		validateClosedWorldGroundedHfField(rawDatasetEntry, closedWorldGroundedHfFieldLabel, rawEntryIndex)
	elif(datasetType == datasetTypeClosedWorldGrounded3):
		validateClosedWorldGroundedHfField(rawDatasetEntry, closedWorldGroundedHfFieldEvidenceLabel, rawEntryIndex)
	else:
		raise RuntimeError("validateClosedWorldGroundedHfRawEntry error: unsupported datasetType " + str(datasetType))
	return

def validateClosedWorldGroundedHfField(rawDatasetEntry, fieldName, rawEntryIndex):
	if(fieldName not in rawDatasetEntry):
		raise RuntimeError("validateClosedWorldGroundedHfField error: missing field " + fieldName + " at rawEntryIndex=" + str(rawEntryIndex))
	return

def buildClosedWorldGroundedHfEntityName(rawDatasetEntry):
	result = None
	entityId = rawDatasetEntry[closedWorldGroundedHfFieldId]
	if(datasetType == datasetTypeClosedWorldGrounded2):
		entityPrefix = closedWorldGroundedHfEntityPrefixFever
	elif(datasetType == datasetTypeClosedWorldGrounded3):
		entityPrefix = closedWorldGroundedHfEntityPrefixSciFact
	else:
		raise RuntimeError("buildClosedWorldGroundedHfEntityName error: unsupported datasetType " + str(datasetType))
	result = normaliseClosedWorldGroundedHfToken(entityPrefix + closedWorldGroundedHfTokenSeparatorReplacement + str(entityId), closedWorldGroundedEvalItemFieldEntity)
	return result

def getClosedWorldGroundedHfAnswerName(rawDatasetEntry):
	result = None
	if(datasetType == datasetTypeClosedWorldGrounded2):
		labelValue = rawDatasetEntry[closedWorldGroundedHfFieldLabel]
		if(labelValue == closedWorldGroundedFeverLabelSupportsIndex or labelValue == closedWorldGroundedFeverLabelSupportsName):
			result = closedWorldGroundedHfSupportedAnswer
		elif(labelValue == closedWorldGroundedFeverLabelRefutesIndex or labelValue == closedWorldGroundedFeverLabelRefutesName):
			result = closedWorldGroundedHfRefutedAnswer
		else:
			raise RuntimeError("getClosedWorldGroundedHfAnswerName error: unsupported FEVER label " + str(labelValue))
	elif(datasetType == datasetTypeClosedWorldGrounded3):
		labelValue = rawDatasetEntry[closedWorldGroundedHfFieldEvidenceLabel]
		if(labelValue == closedWorldGroundedSciFactLabelSupport):
			result = closedWorldGroundedHfSupportedAnswer
		elif(labelValue == closedWorldGroundedSciFactLabelContradict):
			result = closedWorldGroundedHfRefutedAnswer
		elif(labelValue == closedWorldGroundedSciFactLabelUnknown):
			result = closedWorldGroundedHfUnknownAnswer
		else:
			raise RuntimeError("getClosedWorldGroundedHfAnswerName error: unsupported SciFact evidence_label " + str(labelValue))
	else:
		raise RuntimeError("getClosedWorldGroundedHfAnswerName error: unsupported datasetType " + str(datasetType))
	return result

def getClosedWorldGroundedHfAlternativeAnswer(answerName):
	result = None
	if(answerName is None or not isinstance(answerName, str) or answerName == ""):
		raise RuntimeError("getClosedWorldGroundedHfAlternativeAnswer error: answerName must be a non-empty string")
	if(closedWorldGroundedHfAnswerOptions is None or not isinstance(closedWorldGroundedHfAnswerOptions, list)):
		raise RuntimeError("getClosedWorldGroundedHfAlternativeAnswer error: closedWorldGroundedHfAnswerOptions must be a list")
	if(len(closedWorldGroundedHfAnswerOptions) <= closedWorldGroundedHfAlternativeAnswerIndexOffset):
		raise RuntimeError("getClosedWorldGroundedHfAlternativeAnswer error: closedWorldGroundedHfAnswerOptions has insufficient labels")
	if(answerName not in closedWorldGroundedHfAnswerOptions):
		raise RuntimeError("getClosedWorldGroundedHfAlternativeAnswer error: answerName not found in closedWorldGroundedHfAnswerOptions: " + answerName)
	answerIndex = closedWorldGroundedHfAnswerOptions.index(answerName)
	alternativeAnswerIndex = (answerIndex + closedWorldGroundedHfAlternativeAnswerIndexOffset) % len(closedWorldGroundedHfAnswerOptions)
	result = closedWorldGroundedHfAnswerOptions[alternativeAnswerIndex]
	return result

def getClosedWorldGroundedHfCategoryName(answerName):
	result = None
	if(answerName == closedWorldGroundedHfSupportedAnswer or answerName == closedWorldGroundedHfRefutedAnswer):
		result = closedWorldGroundedLabelDirectSupport
	elif(answerName == closedWorldGroundedHfUnknownAnswer):
		result = closedWorldGroundedLabelUnsupportedWorldTrue
	else:
		raise RuntimeError("getClosedWorldGroundedHfCategoryName error: unsupported answerName " + str(answerName))
	return result

def normaliseClosedWorldGroundedHfToken(tokenValue, tokenName):
	result = None
	normalisedCharacters = []
	if(tokenName is None or tokenName == ""):
		raise RuntimeError("normaliseClosedWorldGroundedHfToken error: tokenName must not be empty")
	if(tokenValue is None):
		raise RuntimeError("normaliseClosedWorldGroundedHfToken error: tokenValue must not be None")
	tokenString = str(tokenValue).lower()
	if(tokenString == ""):
		raise RuntimeError("normaliseClosedWorldGroundedHfToken error: tokenString must not be empty")
	for tokenCharacter in tokenString:
		if(tokenCharacter.isalnum()):
			normalisedCharacters.append(tokenCharacter)
		else:
			normalisedCharacters.append(closedWorldGroundedHfTokenSeparatorReplacement)
	result = normaliseClosedWorldGroundedToken("".join(normalisedCharacters), tokenName)
	return result

def getClosedWorldGroundedEvalItem(sequenceIndex):
	result = None
	if(not isinstance(sequenceIndex, int)):
		raise RuntimeError("getClosedWorldGroundedEvalItem error: sequenceIndex must be an int")
	if(sequenceIndex < 0):
		raise RuntimeError("getClosedWorldGroundedEvalItem error: sequenceIndex must be >= 0")
	evalItems = getClosedWorldGroundedEvalItems()
	if(sequenceIndex >= len(evalItems)):
		raise RuntimeError("getClosedWorldGroundedEvalItem error: sequenceIndex out of range")
	result = evalItems[sequenceIndex]
	return result

def buildClosedWorldGroundedInferencePromptText():
	result = None
	promptLines = []
	evalDatasetEntries = loadClosedWorldGroundedDataset(True)
	for datasetEntry in evalDatasetEntries:
		promptLines.append(datasetEntry[closedWorldGroundedDatasetTextFieldName])
	result = closedWorldGroundedPromptArticleSeparator.join(promptLines)
	return result

def ensureClosedWorldGroundedInferencePromptFile():
	result = None
	validateClosedWorldGroundedDatasetEnabled("ensureClosedWorldGroundedInferencePromptFile")
	expectedPromptText = buildClosedWorldGroundedInferencePromptText()
	inferencePromptDirectory = os.path.dirname(inferencePromptFile)
	if(inferencePromptDirectory == ""):
		raise RuntimeError("ensureClosedWorldGroundedInferencePromptFile error: inferencePromptFile directory is empty")
	if(os.path.exists(inferencePromptDirectory) and not os.path.isdir(inferencePromptDirectory)):
		raise RuntimeError("ensureClosedWorldGroundedInferencePromptFile error: inferencePromptFile directory exists but is not a directory: " + inferencePromptDirectory)
	os.makedirs(inferencePromptDirectory, exist_ok=True)
	if(os.path.isfile(inferencePromptFile)):
		with open(inferencePromptFile, "r", encoding=closedWorldGroundedInferencePromptFileEncoding) as file:
			existingPromptText = file.read()
		if(existingPromptText != expectedPromptText):
			raise RuntimeError("ensureClosedWorldGroundedInferencePromptFile error: existing inferencePromptFile does not match " + datasetType + " eval split: " + inferencePromptFile)
	elif(os.path.exists(inferencePromptFile)):
		raise RuntimeError("ensureClosedWorldGroundedInferencePromptFile error: inferencePromptFile exists but is not a file: " + inferencePromptFile)
	else:
		with open(inferencePromptFile, "w", encoding=closedWorldGroundedInferencePromptFileEncoding) as file:
			file.write(expectedPromptText)
	result = expectedPromptText
	return result
