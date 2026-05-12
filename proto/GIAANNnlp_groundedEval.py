"""GIAANNnlp_groundedEval.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 BAI Research Pty Ltd (bairesearch.com.au)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN NLP grounded prediction evaluation

"""

from GIAANNcmn_globalDefs import *
import GIAANNnlp_groundedDataset

totalInferenceGroundedItems = 0
totalInferenceGroundedAnswered = 0
totalInferenceGroundedJustified = 0
totalInferenceGroundedCorrectUngrounded = 0
totalInferenceGroundedGroundedFalsehood = 0
totalInferenceGroundedUngroundedHallucination = 0
totalInferenceGroundedAbstained = 0
totalInferenceGroundedDirectSupportItems = 0
totalInferenceGroundedDirectSupportJustified = 0
totalInferenceGroundedCompositionalSupportItems = 0
totalInferenceGroundedCompositionalSupportJustified = 0
totalInferenceGroundedUnsupportedWorldTrueItems = 0
totalInferenceGroundedUnsupportedWorldTrueCorrectUngrounded = 0
totalInferenceGroundedUnsupportedFalseItems = 0
totalInferenceGroundedUnsupportedFalseUngroundedHallucination = 0
totalInferenceGroundedNoisySupportItems = 0
totalInferenceGroundedNoisySupportGroundedFalsehood = 0
totalInferenceGroundedTargetLabelCounts = {}
totalInferenceGroundedPredictedLabelCounts = {}
totalInferenceGroundedTrueLabelOutcomeCounts = {}
inferenceGroundedRecordedSequenceIndices = set()

def resetInferenceGroundedAccuracyCounts():
	global totalInferenceGroundedItems
	global totalInferenceGroundedAnswered
	global totalInferenceGroundedJustified
	global totalInferenceGroundedCorrectUngrounded
	global totalInferenceGroundedGroundedFalsehood
	global totalInferenceGroundedUngroundedHallucination
	global totalInferenceGroundedAbstained
	global totalInferenceGroundedDirectSupportItems
	global totalInferenceGroundedDirectSupportJustified
	global totalInferenceGroundedCompositionalSupportItems
	global totalInferenceGroundedCompositionalSupportJustified
	global totalInferenceGroundedUnsupportedWorldTrueItems
	global totalInferenceGroundedUnsupportedWorldTrueCorrectUngrounded
	global totalInferenceGroundedUnsupportedFalseItems
	global totalInferenceGroundedUnsupportedFalseUngroundedHallucination
	global totalInferenceGroundedNoisySupportItems
	global totalInferenceGroundedNoisySupportGroundedFalsehood
	global totalInferenceGroundedTargetLabelCounts
	global totalInferenceGroundedPredictedLabelCounts
	global totalInferenceGroundedTrueLabelOutcomeCounts
	global inferenceGroundedRecordedSequenceIndices
	totalInferenceGroundedItems = 0
	totalInferenceGroundedAnswered = 0
	totalInferenceGroundedJustified = 0
	totalInferenceGroundedCorrectUngrounded = 0
	totalInferenceGroundedGroundedFalsehood = 0
	totalInferenceGroundedUngroundedHallucination = 0
	totalInferenceGroundedAbstained = 0
	totalInferenceGroundedDirectSupportItems = 0
	totalInferenceGroundedDirectSupportJustified = 0
	totalInferenceGroundedCompositionalSupportItems = 0
	totalInferenceGroundedCompositionalSupportJustified = 0
	totalInferenceGroundedUnsupportedWorldTrueItems = 0
	totalInferenceGroundedUnsupportedWorldTrueCorrectUngrounded = 0
	totalInferenceGroundedUnsupportedFalseItems = 0
	totalInferenceGroundedUnsupportedFalseUngroundedHallucination = 0
	totalInferenceGroundedNoisySupportItems = 0
	totalInferenceGroundedNoisySupportGroundedFalsehood = 0
	if(inferenceReportGroundedAccuracyMod3_perLabelMetrics):
		totalInferenceGroundedTargetLabelCounts = {}
		totalInferenceGroundedPredictedLabelCounts = {}
		totalInferenceGroundedTrueLabelOutcomeCounts = {}
	inferenceGroundedRecordedSequenceIndices = set()
	return

def validateInferenceGroundedAccuracyEnabled(functionName):
	if(functionName is None or functionName == ""):
		raise RuntimeError("validateInferenceGroundedAccuracyEnabled error: functionName must not be empty")
	if(not inferenceReportGroundedAccuracy):
		raise RuntimeError(functionName + " error: requires inferenceReportGroundedAccuracy")
	if(datasetType not in closedWorldGroundedDatasetTypes):
		raise RuntimeError(functionName + " error: requires datasetType in closedWorldGroundedDatasetTypes")
	return

def validateInferenceGroundedPredictionArguments(sequenceIndex, sequenceWordIndex, targetWord, predictedWord, predictionCandidatesAvailable, groundedAnswerTokenIndex):
	if(not isinstance(sequenceIndex, int)):
		raise RuntimeError("validateInferenceGroundedPredictionArguments error: sequenceIndex must be an int")
	if(sequenceIndex < 0):
		raise RuntimeError("validateInferenceGroundedPredictionArguments error: sequenceIndex must be >= 0")
	if(not isinstance(sequenceWordIndex, int)):
		raise RuntimeError("validateInferenceGroundedPredictionArguments error: sequenceWordIndex must be an int")
	if(sequenceWordIndex < 0):
		raise RuntimeError("validateInferenceGroundedPredictionArguments error: sequenceWordIndex must be >= 0")
	if(targetWord is None or not isinstance(targetWord, str) or targetWord == ""):
		raise RuntimeError("validateInferenceGroundedPredictionArguments error: targetWord must be a non-empty string")
	if(predictedWord is None or not isinstance(predictedWord, str) or predictedWord == ""):
		raise RuntimeError("validateInferenceGroundedPredictionArguments error: predictedWord must be a non-empty string")
	if(not isinstance(predictionCandidatesAvailable, bool)):
		raise RuntimeError("validateInferenceGroundedPredictionArguments error: predictionCandidatesAvailable must be a bool")
	if(groundedAnswerTokenIndex is not None):
		if(not isinstance(groundedAnswerTokenIndex, int)):
			raise RuntimeError("validateInferenceGroundedPredictionArguments error: groundedAnswerTokenIndex must be an int")
		if(groundedAnswerTokenIndex < 0):
			raise RuntimeError("validateInferenceGroundedPredictionArguments error: groundedAnswerTokenIndex must be >= 0")
	return

def classifyInferenceGroundedPrediction(predictedWord, predictionCandidatesAvailable, trueAnswers, supportedAnswers):
	result = None
	if(trueAnswers is None or not isinstance(trueAnswers, tuple)):
		raise RuntimeError("classifyInferenceGroundedPrediction error: trueAnswers must be a tuple")
	if(supportedAnswers is None or not isinstance(supportedAnswers, tuple)):
		raise RuntimeError("classifyInferenceGroundedPrediction error: supportedAnswers must be a tuple")
	if(not predictionCandidatesAvailable or predictedWord == closedWorldGroundedNoPredictionWord):
		result = closedWorldGroundedOutcomeAbstained
	else:
		predictedWordNormalised = GIAANNnlp_groundedDataset.normaliseClosedWorldGroundedToken(predictedWord, "predictedWord")
		predictionCorrect = predictedWordNormalised in trueAnswers
		predictionGrounded = predictedWordNormalised in supportedAnswers
		if(predictionCorrect and predictionGrounded):
			result = closedWorldGroundedOutcomeJustified
		elif(predictionCorrect and not predictionGrounded):
			result = closedWorldGroundedOutcomeCorrectUngrounded
		elif(not predictionCorrect and predictionGrounded):
			result = closedWorldGroundedOutcomeGroundedFalsehood
		else:
			result = closedWorldGroundedOutcomeUngroundedHallucination
	return result

def addInferenceGroundedAccuracyCategoryCount(categoryName, outcomeName):
	global totalInferenceGroundedDirectSupportItems
	global totalInferenceGroundedDirectSupportJustified
	global totalInferenceGroundedCompositionalSupportItems
	global totalInferenceGroundedCompositionalSupportJustified
	global totalInferenceGroundedUnsupportedWorldTrueItems
	global totalInferenceGroundedUnsupportedWorldTrueCorrectUngrounded
	global totalInferenceGroundedUnsupportedFalseItems
	global totalInferenceGroundedUnsupportedFalseUngroundedHallucination
	global totalInferenceGroundedNoisySupportItems
	global totalInferenceGroundedNoisySupportGroundedFalsehood
	if(categoryName == closedWorldGroundedLabelDirectSupport):
		totalInferenceGroundedDirectSupportItems += 1
		if(outcomeName == closedWorldGroundedOutcomeJustified):
			totalInferenceGroundedDirectSupportJustified += 1
	elif(categoryName == closedWorldGroundedLabelCompositionalSupport):
		totalInferenceGroundedCompositionalSupportItems += 1
		if(outcomeName == closedWorldGroundedOutcomeJustified):
			totalInferenceGroundedCompositionalSupportJustified += 1
	elif(categoryName == closedWorldGroundedLabelUnsupportedWorldTrue):
		totalInferenceGroundedUnsupportedWorldTrueItems += 1
		if(outcomeName == closedWorldGroundedOutcomeCorrectUngrounded):
			totalInferenceGroundedUnsupportedWorldTrueCorrectUngrounded += 1
	elif(categoryName == closedWorldGroundedLabelUnsupportedFalse):
		totalInferenceGroundedUnsupportedFalseItems += 1
		if(outcomeName == closedWorldGroundedOutcomeUngroundedHallucination):
			totalInferenceGroundedUnsupportedFalseUngroundedHallucination += 1
	elif(categoryName == closedWorldGroundedLabelNoisySupport):
		totalInferenceGroundedNoisySupportItems += 1
		if(outcomeName == closedWorldGroundedOutcomeGroundedFalsehood):
			totalInferenceGroundedNoisySupportGroundedFalsehood += 1
	return

def addInferenceGroundedAccuracyOutcomeCount(outcomeName):
	global totalInferenceGroundedItems
	global totalInferenceGroundedAnswered
	global totalInferenceGroundedJustified
	global totalInferenceGroundedCorrectUngrounded
	global totalInferenceGroundedGroundedFalsehood
	global totalInferenceGroundedUngroundedHallucination
	global totalInferenceGroundedAbstained
	totalInferenceGroundedItems += 1
	if(outcomeName == closedWorldGroundedOutcomeJustified):
		totalInferenceGroundedAnswered += 1
		totalInferenceGroundedJustified += 1
	elif(outcomeName == closedWorldGroundedOutcomeCorrectUngrounded):
		totalInferenceGroundedAnswered += 1
		totalInferenceGroundedCorrectUngrounded += 1
	elif(outcomeName == closedWorldGroundedOutcomeGroundedFalsehood):
		totalInferenceGroundedAnswered += 1
		totalInferenceGroundedGroundedFalsehood += 1
	elif(outcomeName == closedWorldGroundedOutcomeUngroundedHallucination):
		totalInferenceGroundedAnswered += 1
		totalInferenceGroundedUngroundedHallucination += 1
	elif(outcomeName == closedWorldGroundedOutcomeAbstained):
		totalInferenceGroundedAbstained += 1
	else:
		raise RuntimeError("addInferenceGroundedAccuracyOutcomeCount error: unsupported outcomeName " + str(outcomeName))
	return

def recordInferenceGroundedPrediction(sequenceIndex, sequenceWordIndex, targetWord, predictedWord, predictionCandidatesAvailable, groundedAnswerTokenIndex=None):
	if(inferenceReportGroundedAccuracy):
		validateInferenceGroundedAccuracyEnabled("recordInferenceGroundedPrediction")
		validateInferenceGroundedPredictionArguments(sequenceIndex, sequenceWordIndex, targetWord, predictedWord, predictionCandidatesAvailable, groundedAnswerTokenIndex)
		if(groundedAnswerTokenIndex is None):
			groundedAnswerTokenIndex = getInferenceGroundedEvalItemAnswerTokenIndex(GIAANNnlp_groundedDataset.getClosedWorldGroundedEvalItem(sequenceIndex))
		if(sequenceWordIndex == groundedAnswerTokenIndex):
			global inferenceGroundedRecordedSequenceIndices
			if(sequenceIndex in inferenceGroundedRecordedSequenceIndices):
				raise RuntimeError("recordInferenceGroundedPrediction error: duplicate grounded prediction for sequenceIndex " + str(sequenceIndex))
			evalItem = GIAANNnlp_groundedDataset.getClosedWorldGroundedEvalItem(sequenceIndex)
			targetWordNormalised = GIAANNnlp_groundedDataset.normaliseClosedWorldGroundedToken(targetWord, "targetWord")
			expectedTargetAnswer = evalItem[closedWorldGroundedEvalItemFieldTargetAnswer]
			if(targetWordNormalised != expectedTargetAnswer):
				raise RuntimeError("recordInferenceGroundedPrediction error: targetWord does not match closed-world eval target answer at sequenceIndex " + str(sequenceIndex) + "; targetWord=" + targetWordNormalised + ", expectedTargetAnswer=" + expectedTargetAnswer)
			outcomeName = classifyInferenceGroundedPrediction(predictedWord, predictionCandidatesAvailable, evalItem[closedWorldGroundedEvalItemFieldTrueAnswers], evalItem[closedWorldGroundedEvalItemFieldSupportedAnswers])
			addInferenceGroundedAccuracyOutcomeCount(outcomeName)
			addInferenceGroundedAccuracyCategoryCount(evalItem[closedWorldGroundedEvalItemFieldCategory], outcomeName)
			if(inferenceReportGroundedAccuracyMod3_perLabelMetrics):
				addInferenceGroundedAccuracyPerLabelCount(evalItem, predictedWord, predictionCandidatesAvailable, outcomeName)
			inferenceGroundedRecordedSequenceIndices.add(sequenceIndex)
	return

def hasInferenceGroundedPredictionBeenRecorded(sequenceIndex):
	result = None
	if(not isinstance(sequenceIndex, int)):
		raise RuntimeError("hasInferenceGroundedPredictionBeenRecorded error: sequenceIndex must be an int")
	if(sequenceIndex < 0):
		raise RuntimeError("hasInferenceGroundedPredictionBeenRecorded error: sequenceIndex must be >= 0")
	result = sequenceIndex in inferenceGroundedRecordedSequenceIndices
	return result

def getInferenceGroundedAnswerTokenIndexForSequence(sequenceIndex, tokensSequence, conceptMask, numSeedTokens):
	result = None
	evalItem = None
	if(not isinstance(sequenceIndex, int)):
		raise RuntimeError("getInferenceGroundedAnswerTokenIndexForSequence error: sequenceIndex must be an int")
	if(sequenceIndex < 0):
		raise RuntimeError("getInferenceGroundedAnswerTokenIndexForSequence error: sequenceIndex must be >= 0")
	if(tokensSequence is None or not isinstance(tokensSequence, list)):
		raise RuntimeError("getInferenceGroundedAnswerTokenIndexForSequence error: tokensSequence must be a list")
	if(conceptMask is None):
		raise RuntimeError("getInferenceGroundedAnswerTokenIndexForSequence error: conceptMask must not be None")
	if(not isinstance(numSeedTokens, int)):
		raise RuntimeError("getInferenceGroundedAnswerTokenIndexForSequence error: numSeedTokens must be an int")
	if(numSeedTokens < 0):
		raise RuntimeError("getInferenceGroundedAnswerTokenIndexForSequence error: numSeedTokens must be >= 0")
	evalItem = GIAANNnlp_groundedDataset.getClosedWorldGroundedEvalItem(sequenceIndex)
	if(closedWorldGroundedEvalItemFieldAnswerTokenIndex in evalItem and evalItem[closedWorldGroundedEvalItemFieldAnswerTokenIndex] != closedWorldGroundedEvalItemAnswerTokenIndexDynamic):
		result = getInferenceGroundedEvalItemAnswerTokenIndex(evalItem)
	else:
		result = findInferenceGroundedAnswerTokenIndex(tokensSequence, evalItem)
		evalItem[closedWorldGroundedEvalItemFieldAnswerTokenIndex] = result
	if(result < numSeedTokens):
		raise RuntimeError("getInferenceGroundedAnswerTokenIndexForSequence error: grounded answer token occurs before prediction phase; answerTokenIndex=" + str(result) + ", numSeedTokens=" + str(numSeedTokens))
	if(result >= len(tokensSequence)):
		raise RuntimeError("getInferenceGroundedAnswerTokenIndexForSequence error: grounded answer token index out of range")
	return result

def getInferenceGroundedEvalItemAnswerTokenIndex(evalItem):
	result = None
	if(evalItem is None or not isinstance(evalItem, dict)):
		raise RuntimeError("getInferenceGroundedEvalItemAnswerTokenIndex error: evalItem must be a dict")
	if(closedWorldGroundedEvalItemFieldAnswerTokenIndex not in evalItem):
		raise RuntimeError("getInferenceGroundedEvalItemAnswerTokenIndex error: evalItem missing answer token index")
	result = evalItem[closedWorldGroundedEvalItemFieldAnswerTokenIndex]
	if(not isinstance(result, int)):
		raise RuntimeError("getInferenceGroundedEvalItemAnswerTokenIndex error: answer token index must be an int")
	if(result < 0):
		raise RuntimeError("getInferenceGroundedEvalItemAnswerTokenIndex error: answer token index must be >= 0")
	return result

def findInferenceGroundedAnswerTokenIndex(tokensSequence, evalItem):
	result = None
	expectedTargetAnswer = None
	if(tokensSequence is None or not isinstance(tokensSequence, list)):
		raise RuntimeError("findInferenceGroundedAnswerTokenIndex error: tokensSequence must be a list")
	if(evalItem is None or not isinstance(evalItem, dict)):
		raise RuntimeError("findInferenceGroundedAnswerTokenIndex error: evalItem must be a dict")
	expectedTargetAnswer = evalItem[closedWorldGroundedEvalItemFieldTargetAnswer]
	for tokenIndex, token in enumerate(tokensSequence):
		tokenWord = token.word
		previousTokenWord = None
		if(tokenIndex > 0):
			previousTokenWord = tokensSequence[tokenIndex - closedWorldGroundedHfPoolItemIndexIncrement].word
		if(tokenWord == expectedTargetAnswer and previousTokenWord == closedWorldGroundedPromptAnswerQualifier):
			result = tokenIndex
	if(result is None):
		raise RuntimeError("findInferenceGroundedAnswerTokenIndex error: unable to find answer token for sequenceIndex " + str(evalItem[closedWorldGroundedEvalItemFieldSequenceIndex]))
	return result

def addInferenceGroundedAccuracyPerLabelCount(evalItem, predictedWord, predictionCandidatesAvailable, outcomeName):
	global totalInferenceGroundedTargetLabelCounts
	global totalInferenceGroundedPredictedLabelCounts
	global totalInferenceGroundedTrueLabelOutcomeCounts
	if(evalItem is None or not isinstance(evalItem, dict)):
		raise RuntimeError("addInferenceGroundedAccuracyPerLabelCount error: evalItem must be a dict")
	if(outcomeName is None or not isinstance(outcomeName, str) or outcomeName == ""):
		raise RuntimeError("addInferenceGroundedAccuracyPerLabelCount error: outcomeName must be a non-empty string")
	targetLabel = evalItem[closedWorldGroundedEvalItemFieldTargetAnswer]
	trueLabel = getInferenceGroundedAccuracyPrimaryTrueLabel(evalItem)
	if(not predictionCandidatesAvailable or predictedWord == closedWorldGroundedNoPredictionWord):
		predictedLabel = closedWorldGroundedNoPredictionWord
	else:
		predictedLabel = GIAANNnlp_groundedDataset.normaliseClosedWorldGroundedToken(predictedWord, "predictedWord")
	incrementInferenceGroundedAccuracyLabelCount(totalInferenceGroundedTargetLabelCounts, targetLabel)
	incrementInferenceGroundedAccuracyLabelCount(totalInferenceGroundedPredictedLabelCounts, predictedLabel)
	if(trueLabel not in totalInferenceGroundedTrueLabelOutcomeCounts):
		totalInferenceGroundedTrueLabelOutcomeCounts[trueLabel] = {}
	incrementInferenceGroundedAccuracyLabelCount(totalInferenceGroundedTrueLabelOutcomeCounts[trueLabel], outcomeName)
	return

def getInferenceGroundedAccuracyPrimaryTrueLabel(evalItem):
	result = None
	if(evalItem is None or not isinstance(evalItem, dict)):
		raise RuntimeError("getInferenceGroundedAccuracyPrimaryTrueLabel error: evalItem must be a dict")
	if(closedWorldGroundedEvalItemFieldTrueAnswers not in evalItem):
		raise RuntimeError("getInferenceGroundedAccuracyPrimaryTrueLabel error: evalItem missing true answers")
	trueAnswers = evalItem[closedWorldGroundedEvalItemFieldTrueAnswers]
	if(trueAnswers is None or not isinstance(trueAnswers, tuple)):
		raise RuntimeError("getInferenceGroundedAccuracyPrimaryTrueLabel error: trueAnswers must be a tuple")
	if(len(trueAnswers) != closedWorldGroundedHfPoolItemIndexIncrement):
		raise RuntimeError("getInferenceGroundedAccuracyPrimaryTrueLabel error: trueAnswers must contain exactly one answer")
	result = trueAnswers[closedWorldGroundedHfInitialAnswerCount]
	return result

def incrementInferenceGroundedAccuracyLabelCount(labelCounts, labelName):
	if(labelCounts is None or not isinstance(labelCounts, dict)):
		raise RuntimeError("incrementInferenceGroundedAccuracyLabelCount error: labelCounts must be a dict")
	if(labelName is None or not isinstance(labelName, str) or labelName == ""):
		raise RuntimeError("incrementInferenceGroundedAccuracyLabelCount error: labelName must be a non-empty string")
	if(labelName not in labelCounts):
		labelCounts[labelName] = closedWorldGroundedHfInitialAnswerCount
	labelCounts[labelName] += closedWorldGroundedHfPoolItemIndexIncrement
	return

def calculateInferenceGroundedAccuracyRate(numerator, denominator, rateName):
	result = None
	if(rateName is None or rateName == ""):
		raise RuntimeError("calculateInferenceGroundedAccuracyRate error: rateName must not be empty")
	if(not isinstance(numerator, int)):
		raise RuntimeError("calculateInferenceGroundedAccuracyRate error: numerator must be an int")
	if(not isinstance(denominator, int)):
		raise RuntimeError("calculateInferenceGroundedAccuracyRate error: denominator must be an int")
	if(numerator < 0):
		raise RuntimeError("calculateInferenceGroundedAccuracyRate error: numerator must be >= 0")
	if(denominator < 0):
		raise RuntimeError("calculateInferenceGroundedAccuracyRate error: denominator must be >= 0")
	if(numerator > denominator):
		raise RuntimeError("calculateInferenceGroundedAccuracyRate error: numerator must be <= denominator")
	if(denominator == 0):
		result = 0.0
	else:
		result = numerator / denominator
	return result

def calculateInferenceGroundedAccuracyMajorityClassBaseline(evalItems):
	result = None
	trueLabelCounts = {}
	baselineOutcomeCounts = initialiseInferenceGroundedAccuracyOutcomeCounts()
	if(evalItems is None or not isinstance(evalItems, list)):
		raise RuntimeError("calculateInferenceGroundedAccuracyMajorityClassBaseline error: evalItems must be a list")
	if(len(evalItems) < closedWorldGroundedHfDatasetItemCountMinimum):
		raise RuntimeError("calculateInferenceGroundedAccuracyMajorityClassBaseline error: evalItems must not be empty")
	for evalItem in evalItems:
		trueLabel = getInferenceGroundedAccuracyPrimaryTrueLabel(evalItem)
		incrementInferenceGroundedAccuracyLabelCount(trueLabelCounts, trueLabel)
	majorityLabel = getInferenceGroundedAccuracyMajorityLabel(trueLabelCounts)
	for evalItem in evalItems:
		outcomeName = classifyInferenceGroundedPrediction(majorityLabel, True, evalItem[closedWorldGroundedEvalItemFieldTrueAnswers], evalItem[closedWorldGroundedEvalItemFieldSupportedAnswers])
		incrementInferenceGroundedAccuracyLabelCount(baselineOutcomeCounts, outcomeName)
	groundedAccuracy = calculateInferenceGroundedAccuracyRate(baselineOutcomeCounts[closedWorldGroundedOutcomeJustified], len(evalItems), "majorityClassBaselineGroundedAccuracy")
	result = majorityLabel, groundedAccuracy, baselineOutcomeCounts
	return result

def initialiseInferenceGroundedAccuracyOutcomeCounts():
	result = None
	outcomeCounts = {}
	outcomeCounts[closedWorldGroundedOutcomeJustified] = closedWorldGroundedHfInitialAnswerCount
	outcomeCounts[closedWorldGroundedOutcomeCorrectUngrounded] = closedWorldGroundedHfInitialAnswerCount
	outcomeCounts[closedWorldGroundedOutcomeGroundedFalsehood] = closedWorldGroundedHfInitialAnswerCount
	outcomeCounts[closedWorldGroundedOutcomeUngroundedHallucination] = closedWorldGroundedHfInitialAnswerCount
	outcomeCounts[closedWorldGroundedOutcomeAbstained] = closedWorldGroundedHfInitialAnswerCount
	result = outcomeCounts
	return result

def getInferenceGroundedAccuracyMajorityLabel(labelCounts):
	result = None
	if(labelCounts is None or not isinstance(labelCounts, dict)):
		raise RuntimeError("getInferenceGroundedAccuracyMajorityLabel error: labelCounts must be a dict")
	if(len(labelCounts) < closedWorldGroundedHfDatasetItemCountMinimum):
		raise RuntimeError("getInferenceGroundedAccuracyMajorityLabel error: labelCounts must not be empty")
	labelOrder = getInferenceGroundedAccuracyLabelOrder(labelCounts)
	majorityLabel = None
	majorityLabelCount = None
	for labelName in labelOrder:
		if(labelName not in labelCounts):
			raise RuntimeError("getInferenceGroundedAccuracyMajorityLabel error: labelOrder includes unknown label " + labelName)
		if(majorityLabel is None or labelCounts[labelName] > majorityLabelCount):
			majorityLabel = labelName
			majorityLabelCount = labelCounts[labelName]
	result = majorityLabel
	return result

def getInferenceGroundedAccuracyLabelOrder(labelCounts):
	result = None
	labelOrder = []
	if(labelCounts is None or not isinstance(labelCounts, dict)):
		raise RuntimeError("getInferenceGroundedAccuracyLabelOrder error: labelCounts must be a dict")
	if(closedWorldGroundedHfAnswerOptions is not None and isinstance(closedWorldGroundedHfAnswerOptions, list)):
		for labelName in closedWorldGroundedHfAnswerOptions:
			if(labelName in labelCounts):
				labelOrder.append(labelName)
	for labelName in sorted(labelCounts.keys()):
		if(labelName not in labelOrder):
			labelOrder.append(labelName)
	result = labelOrder
	return result

def buildInferenceGroundedAccuracySortedLabelCountDict(labelCounts):
	result = None
	sortedLabelCounts = {}
	if(labelCounts is None or not isinstance(labelCounts, dict)):
		raise RuntimeError("buildInferenceGroundedAccuracySortedLabelCountDict error: labelCounts must be a dict")
	for labelName in getInferenceGroundedAccuracyLabelOrder(labelCounts):
		sortedLabelCounts[labelName] = labelCounts[labelName]
	result = sortedLabelCounts
	return result

def buildInferenceGroundedAccuracySortedNestedLabelCountDict(nestedLabelCounts):
	result = None
	sortedNestedLabelCounts = {}
	if(nestedLabelCounts is None or not isinstance(nestedLabelCounts, dict)):
		raise RuntimeError("buildInferenceGroundedAccuracySortedNestedLabelCountDict error: nestedLabelCounts must be a dict")
	for labelName in getInferenceGroundedAccuracyLabelOrder(nestedLabelCounts):
		sortedNestedLabelCounts[labelName] = buildInferenceGroundedAccuracySortedLabelCountDict(nestedLabelCounts[labelName])
	result = sortedNestedLabelCounts
	return result

def finaliseInferenceGroundedAccuracyMissingItems(evalItems):
	if(evalItems is None or not isinstance(evalItems, list)):
		raise RuntimeError("finaliseInferenceGroundedAccuracyMissingItems error: evalItems must be a list")
	for evalItem in evalItems:
		if(evalItem is None or not isinstance(evalItem, dict)):
			raise RuntimeError("finaliseInferenceGroundedAccuracyMissingItems error: evalItem must be a dict")
		if(closedWorldGroundedEvalItemFieldSequenceIndex not in evalItem):
			raise RuntimeError("finaliseInferenceGroundedAccuracyMissingItems error: evalItem missing sequence index")
		sequenceIndex = evalItem[closedWorldGroundedEvalItemFieldSequenceIndex]
		if(not isinstance(sequenceIndex, int)):
			raise RuntimeError("finaliseInferenceGroundedAccuracyMissingItems error: sequenceIndex must be an int")
		if(sequenceIndex < 0):
			raise RuntimeError("finaliseInferenceGroundedAccuracyMissingItems error: sequenceIndex must be >= 0")
		if(sequenceIndex not in inferenceGroundedRecordedSequenceIndices):
			outcomeName = closedWorldGroundedOutcomeAbstained
			addInferenceGroundedAccuracyOutcomeCount(outcomeName)
			addInferenceGroundedAccuracyCategoryCount(evalItem[closedWorldGroundedEvalItemFieldCategory], outcomeName)
			if(inferenceReportGroundedAccuracyMod3_perLabelMetrics):
				addInferenceGroundedAccuracyPerLabelCount(evalItem, closedWorldGroundedNoPredictionWord, False, outcomeName)
			inferenceGroundedRecordedSequenceIndices.add(sequenceIndex)
	return

def printInferenceGroundedAccuracy(databaseNetworkObject, autoresearchExecutionTimeInference=None, autoresearchExecutionTimeTrain=None):
	if(inferenceReportGroundedAccuracy):
		validateInferenceGroundedAccuracyEnabled("printInferenceGroundedAccuracy")
		evalItems = GIAANNnlp_groundedDataset.getClosedWorldGroundedEvalItems()
		expectedGroundedItems = len(evalItems)
		finaliseInferenceGroundedAccuracyMissingItems(evalItems)
		if(totalInferenceGroundedItems != expectedGroundedItems):
			raise RuntimeError("printInferenceGroundedAccuracy error: evaluated grounded item count does not match expected count; evaluated=" + str(totalInferenceGroundedItems) + ", expected=" + str(expectedGroundedItems))
		groundedAccuracy = calculateInferenceGroundedAccuracyRate(totalInferenceGroundedJustified, totalInferenceGroundedItems, "groundedAccuracy")
		answeredRate = calculateInferenceGroundedAccuracyRate(totalInferenceGroundedAnswered, totalInferenceGroundedItems, "answeredRate")
		abstainedRate = calculateInferenceGroundedAccuracyRate(totalInferenceGroundedAbstained, totalInferenceGroundedItems, "abstainedRate")
		directSupportGroundedAccuracy = calculateInferenceGroundedAccuracyRate(totalInferenceGroundedDirectSupportJustified, totalInferenceGroundedDirectSupportItems, "directSupportGroundedAccuracy")
		compositionalSupportGroundedAccuracy = calculateInferenceGroundedAccuracyRate(totalInferenceGroundedCompositionalSupportJustified, totalInferenceGroundedCompositionalSupportItems, "compositionalSupportGroundedAccuracy")
		unsupportedWorldTrueCorrectUngroundedRate = calculateInferenceGroundedAccuracyRate(totalInferenceGroundedUnsupportedWorldTrueCorrectUngrounded, totalInferenceGroundedUnsupportedWorldTrueItems, "unsupportedWorldTrueCorrectUngroundedRate")
		unsupportedFalseUngroundedHallucinationRate = calculateInferenceGroundedAccuracyRate(totalInferenceGroundedUnsupportedFalseUngroundedHallucination, totalInferenceGroundedUnsupportedFalseItems, "unsupportedFalseUngroundedHallucinationRate")
		noisySupportGroundedFalsehoodRate = calculateInferenceGroundedAccuracyRate(totalInferenceGroundedNoisySupportGroundedFalsehood, totalInferenceGroundedNoisySupportItems, "noisySupportGroundedFalsehoodRate")
		print("averageGroundedAccuracy: groundedCorrectItems = ", groundedAccuracy, ", totalItems = ", totalInferenceGroundedItems)
		print("groundedAccuracyDetails: answeredRate = ", answeredRate, ", abstainedRate = ", abstainedRate, ", justified = ", totalInferenceGroundedJustified, ", correctUngrounded = ", totalInferenceGroundedCorrectUngrounded, ", groundedFalsehood = ", totalInferenceGroundedGroundedFalsehood, ", ungroundedHallucination = ", totalInferenceGroundedUngroundedHallucination, ", abstained = ", totalInferenceGroundedAbstained)
		print("groundedAccuracySupportBreakdown: directSupportGroundedAccuracy = ", directSupportGroundedAccuracy, ", compositionalSupportGroundedAccuracy = ", compositionalSupportGroundedAccuracy, ", directSupportItems = ", totalInferenceGroundedDirectSupportItems, ", compositionalSupportItems = ", totalInferenceGroundedCompositionalSupportItems)
		print("groundedAccuracyUnsupportedBreakdown: unsupportedWorldTrueCorrectUngroundedRate = ", unsupportedWorldTrueCorrectUngroundedRate, ", unsupportedFalseUngroundedHallucinationRate = ", unsupportedFalseUngroundedHallucinationRate, ", noisySupportGroundedFalsehoodRate = ", noisySupportGroundedFalsehoodRate, ", unsupportedWorldTrueItems = ", totalInferenceGroundedUnsupportedWorldTrueItems, ", unsupportedFalseItems = ", totalInferenceGroundedUnsupportedFalseItems, ", noisySupportItems = ", totalInferenceGroundedNoisySupportItems)
		if(inferenceReportGroundedAccuracyMod2_majorityClassBaseline):
			majorityLabel, majorityGroundedAccuracy, majorityOutcomeCounts = calculateInferenceGroundedAccuracyMajorityClassBaseline(evalItems)
			print("groundedAccuracyMajorityClassBaseline: majorityLabel = ", majorityLabel, ", groundedAccuracy = ", majorityGroundedAccuracy, ", outcomeCounts = ", buildInferenceGroundedAccuracySortedLabelCountDict(majorityOutcomeCounts))
		if(inferenceReportGroundedAccuracyMod3_perLabelMetrics):
			print("groundedAccuracyPerLabelCounts: targetLabels = ", buildInferenceGroundedAccuracySortedLabelCountDict(totalInferenceGroundedTargetLabelCounts), ", predictedLabels = ", buildInferenceGroundedAccuracySortedLabelCountDict(totalInferenceGroundedPredictedLabelCounts), ", trueLabelOutcomes = ", buildInferenceGroundedAccuracySortedNestedLabelCountDict(totalInferenceGroundedTrueLabelOutcomeCounts))
	return
