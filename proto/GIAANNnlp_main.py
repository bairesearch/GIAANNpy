"""GIAANNnlp_main.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 BAI Research Pty Ltd (bairesearch.com.au)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN NLP main

"""

# Import necessary libraries
import gc
import time
import torch as pt
import spacy

pt.set_grad_enabled(False)

from GIAANNcmn_globalDefs import *
import GIAANNcmn_debug
import GIAANNcmn_sparseTensors
import GIAANNcmn_databaseNetwork
import GIAANNcmn_databaseNetworkFiles
import GIAANNcmn_databaseNetworkDraw
import GIAANNcmn_databaseNetworkDrawLarge
import GIAANNcmn_executionProgress
import GIAANNnlp_sequenceTokens
import GIAANNnlp_sequenceConcepts
if(auxiliaryNeurons and auxiliaryNeuronsSimilar):
	import GIAANNnlp_auxiliaryNeuronsSimilarWords
if(auxiliaryNeurons and auxiliaryNeuronsSimilarWordsAuto):
	import GIAANNnlp_auxiliaryNeuronsSimilarWordsAuto
if(auxiliaryNeurons and auxiliaryNeuronsSimilarSubwordAuto):
	import GIAANNnlp_auxiliaryNeuronsSimilarSubwordAuto
import GIAANNcmn_sequenceObservedColumns
import GIAANNcmn_databaseNetworkTrain
if(executionMode=="inference" or executionMode=="trainAndInference" or executionMode=="trainDuringInference"):
	import GIAANNcmn_prediction
if(datasetType != "textfile" and executionMode != "inference" and not useDrawNetworkIndependently):
	import GIAANNnlp_datasets
if(datasetType in closedWorldGroundedDatasetTypes):
	import GIAANNnlp_groundedDataset

def loadPOSdatabase():
	if(usePOS):
		GIAANNnlp_sequenceTokens.loadPOSdatabase()
	return

def trainAutoAuxiliaryNeuronsEnd(databaseNetworkObject):
	if(auxiliaryNeurons and auxiliaryNeuronsAuto):
		if(auxiliaryNeuronsSimilarWordsAuto):
			GIAANNnlp_auxiliaryNeuronsSimilarWordsAuto.updateAutoAuxiliaryConnections(databaseNetworkObject)
		if(auxiliaryNeuronsSimilarSubwordAuto):
			GIAANNnlp_auxiliaryNeuronsSimilarSubwordAuto.updateAutoAuxiliaryConnections(databaseNetworkObject)
	return

# Initialize spaCy model
if(useDrawNetworkIndependently):
	nlpArticle = None
	nlpSequence = None
elif(spacyPipelineSingleParse):
	if(spacyPipelineMinimalComponents or spacyPipelineLightweightSentenceSegmentation):
		spacyArticleDisableComponents = []
		if(spacyPipelineMinimalComponents):
			spacyArticleDisableComponents.append("ner")
		nlpArticle = spacy.load(spacyModelName, disable=spacyArticleDisableComponents)
	else:
		nlpArticle = spacy.load(spacyModelName)
	nlpSequence = nlpArticle
else:
	if(spacyPipelineLightweightSentenceSegmentation):
		nlpArticle = spacy.blank("en")
		nlpArticle.add_pipe("sentencizer")
		if(spacyPipelineMinimalComponents):
			nlpSequence = spacy.load(spacyModelName, disable=["ner", "parser"])
		else:
			nlpSequence = spacy.load(spacyModelName)
	else:
		if(spacyPipelineMinimalComponents):
			nlpArticle = spacy.load(spacyModelName, disable=["ner"])
			nlpSequence = spacy.load(spacyModelName, disable=["ner", "parser"])
		else:
			nlpArticle = spacy.load(spacyModelName)
			nlpSequence = nlpArticle


def processPrompt(databaseNetworkObject, inferenceMode, sequenceCount):
	text = None
	if(datasetType in closedWorldGroundedDatasetTypes):
		GIAANNnlp_groundedDataset.ensureClosedWorldGroundedInferencePromptFile()
		with open(inferencePromptFile, 'r', encoding=closedWorldGroundedInferencePromptFileEncoding) as file:
			text = file.read()
	else:
		with open(inferencePromptFile, 'r', encoding='utf-8') as file:
			text = file.read()
	articleIndex = 0
	sequenceCount = processArticle(databaseNetworkObject, inferenceMode, sequenceCount, text, articleIndex)
	return sequenceCount

def expandSequenceForInference(databaseNetworkObject, sequence):
	conceptsFound = False
	conceptMask = None
	tokens = None
	observedColumnsDict = None
	observedColumnsSequenceWordIndexDict = None
	conceptsFound, conceptMask = GIAANNnlp_sequenceConcepts.firstPass(databaseNetworkObject, sequence, True)
	if(conceptsFound):
		tokens = GIAANNnlp_sequenceTokens.getTokens(sequence)
		if not (useDedicatedFeatureLists):
			GIAANNnlp_sequenceConcepts.detectNewFeatures(databaseNetworkObject, tokens, True)
		observedColumnsDict, observedColumnsSequenceWordIndexDict = GIAANNnlp_sequenceConcepts.secondPass(databaseNetworkObject, tokens, False)
	GIAANNcmn_databaseNetwork.ensureGlobalFeatureNeuronsSize(databaseNetworkObject, True)
	return
	
def processDataset(databaseNetworkObject, inferenceMode, sequenceCount, dataset):
	trainMode = not inferenceMode
	if(printTrainSequenceBar and trainMode):
		GIAANNcmn_executionProgress.initialiseTrainSequenceBar(sequenceCount)

	for articleIndex, datasetEntry in enumerate(dataset):
		if(debugPrintSpacySectionTimes):
			getDatasetEntryTextStartTime = None
			getDatasetEntryTextDuration = 0.0
			getDatasetEntryTextStartTime = time.perf_counter()
		text = GIAANNnlp_datasets.getDatasetEntryText(datasetEntry, articleIndex)
		if(debugPrintSpacySectionTimes):
			getDatasetEntryTextDuration = time.perf_counter() - getDatasetEntryTextStartTime
			print(f"debugPrintSpacySectionTimes: articleIndex={articleIndex} sequenceCount={sequenceCount} sequenceCount={sequenceCount} datasetEntryTextSeconds={getDatasetEntryTextDuration:.6f}")
		sequenceCount = processArticle(databaseNetworkObject, inferenceMode, sequenceCount, text, articleIndex)
		if(sequenceCount >= trainMaxSequences and inferenceMode==False):
			break
	return sequenceCount

def processArticle(databaseNetworkObject, inferenceMode, sequenceCount, text, articleIndex):
	#sequences = sent_tokenize(text)
	if(debugPrintSpacySectionTimes):
		processArticlePart1StartTime = None
		processArticlePart1Duration = 0.0
		processArticlePart1StartTime = time.perf_counter()

	if(ignoreNewlineCharacters):
		text = text.replace('\n', ' ')
	textParsed = nlpArticle(text)

	if(executionMode=="inference"):
		skipMode = False	
	else:
		if(datasetType=="textfile"):	#executionMode=="trainAndInference":
			skipMode = False
		else:	#executionMode=="train": 
			if(trainTestSet):
				skipMode = False
			else:
				skipMode = (sequenceCount < (trainSetStartOffsetSequences-maxSentencesPerArticle))
	sequences, sequencesRaw = generateSeqencesBatchOrSerial(textParsed, skipMode)
	if(inferenceMode and inferenceTrainFirstSequences and (printTrainSequenceBar or printEvalSequenceBar)):
		promptSequenceTotal = calculateProcessArticlePromptSequenceTotal(sequenceCount, sequences)
		GIAANNcmn_executionProgress.initialisePromptSequenceBar(printPromptSequenceBarInitialSequenceCount, promptSequenceTotal)
	elif(printEvalSequenceBar and inferenceMode):
		evalSequenceTotal = calculateProcessArticleEvalSequenceTotal(sequenceCount, sequences)
		GIAANNcmn_executionProgress.initialiseEvalSequenceBar(printEvalSequenceBarInitialSequenceCount, evalSequenceTotal)

	if(debugPrintSpacySectionTimes):
		processArticlePart1Duration = time.perf_counter() - processArticlePart1StartTime
		processArticlePart2StartTime = None
		processArticlePart2Duration = 0.0
		processArticlePart2StartTime = time.perf_counter()

	if(inferenceMode and not inferenceTrainFirstSequences):
		if(printHeaderDuringInferencePredict):
			print("executing inference:")

	numberOfSequences = len(sequences)
	for sequenceIndex, sequence in enumerate(sequences):
		if(useTrainDuringInference and inferenceMode==False and sequenceCount >= trainMaxSequences):
			break
		sequenceRaw = sequencesRaw[sequenceIndex]
		inferenceSequenceInPrompt = False
		if(inferenceMode):
			if(inferenceTrainFirstSequences):	#inferenceTrainFirstSequences assumes processArticle() is executed with inferenceMode==True
				if(sequenceIndex == numberOfSequences-1):
					if(printHeaderDuringInferencePredict):
						print("\ninferenceTrainFirstSequences: executing inference:")
					inferenceSequenceInPrompt = True
				else:
					if(sequenceIndex==0):
						if(printHeaderDuringInferencePredict):
							print("\ninferenceTrainFirstSequences: executing train:")
			else:
				inferenceSequenceInPrompt = True
		if(len(sequence) <= maxSequenceLength):
			if(sequenceCount >= trainSetStartOffsetSequences):
				if(executionMode=="trainDuringInference"):
					inferenceSuccessfulPredictionMask = processSequence(databaseNetworkObject, True, sequenceCount, articleIndex, sequenceIndex, sequence, sequenceRaw)
					processSequence(databaseNetworkObject, False, sequenceCount, articleIndex, sequenceIndex, sequence, sequenceRaw, inferenceSuccessfulPredictionMask)
				else:
					processSequence(databaseNetworkObject, inferenceSequenceInPrompt, sequenceCount, articleIndex, sequenceIndex, sequence, sequenceRaw)
			else:
				#if(printSequenceCount):
				print(f"(sequenceCount < trainSetStartOffsetSequences: Processing sequenceCount: {sequenceCount}")	
			sequenceCount += 1
		if(sequenceCount >= trainMaxSequences and inferenceMode==False):
			break

	if(debugPrintSpacySectionTimes):
		processArticlePart2Duration = time.perf_counter() - processArticlePart2StartTime
		print(f"debugPrintSpacySectionTimes: articleIndex={articleIndex} sequenceCount={sequenceCount} processArticlePart1Seconds={processArticlePart1Duration:.6f} processArticlePart2Seconds={processArticlePart2Duration:.6f}")
		global processArticlePart1totalTime
		global processArticlePart2totalTime
		global processArticlePart1count
		global processArticlePart2count
		processArticlePart1totalTime += processArticlePart1Duration
		processArticlePart2totalTime += processArticlePart2Duration
		processArticlePart1count += 1
		processArticlePart2count += 1

	return sequenceCount

def calculateProcessArticlePromptSequenceTotal(sequenceCount, sequences):
	result = 0
	sequenceCountTemp = None
	if(not isinstance(sequenceCount, int)):
		raise RuntimeError("calculateProcessArticlePromptSequenceTotal error: sequenceCount must be an int")
	if(sequenceCount < 0):
		raise RuntimeError("calculateProcessArticlePromptSequenceTotal error: sequenceCount must be >= 0")
	if(sequences is None or not isinstance(sequences, list)):
		raise RuntimeError("calculateProcessArticlePromptSequenceTotal error: sequences must be a list")
	sequenceCountTemp = sequenceCount
	for sequence in sequences:
		if(len(sequence) <= maxSequenceLength):
			if(sequenceCountTemp >= trainSetStartOffsetSequences):
				result += 1
			sequenceCountTemp += 1
	if(result <= 0):
		raise RuntimeError("calculateProcessArticlePromptSequenceTotal error: promptSequenceTotal must be > 0")
	return result

def calculateProcessArticleEvalSequenceTotal(sequenceCount, sequences):
	result = 0
	numberOfSequences = None
	sequenceCountTemp = None
	inferenceSequenceInPrompt = None
	if(not isinstance(sequenceCount, int)):
		raise RuntimeError("calculateProcessArticleEvalSequenceTotal error: sequenceCount must be an int")
	if(sequenceCount < 0):
		raise RuntimeError("calculateProcessArticleEvalSequenceTotal error: sequenceCount must be >= 0")
	if(sequences is None or not isinstance(sequences, list)):
		raise RuntimeError("calculateProcessArticleEvalSequenceTotal error: sequences must be a list")
	numberOfSequences = len(sequences)
	sequenceCountTemp = sequenceCount
	for sequenceIndex, sequence in enumerate(sequences):
		inferenceSequenceInPrompt = False
		if(inferenceTrainFirstSequences):
			if(sequenceIndex == numberOfSequences-1):
				inferenceSequenceInPrompt = True
		else:
			inferenceSequenceInPrompt = True
		if(len(sequence) <= maxSequenceLength):
			if(sequenceCountTemp >= trainSetStartOffsetSequences):
				if(inferenceSequenceInPrompt):
					result += 1
			sequenceCountTemp += 1
	if(result <= 0):
		raise RuntimeError("calculateProcessArticleEvalSequenceTotal error: evalSequenceTotal must be > 0")
	return result

def generateSeqencesBatchOrSerial(textParsed, skipMode):
	sentences = list(textParsed.sents)
	minSequenceLength = numSeedTokensInference + 1
	sequences = []
	sequencesRaw = []
	sequencesText = []
	if(multisentencePredictions):
		for i in range(0, len(sentences), numSentencesPerSequence):
			startIndex = sentences[i].start
			endIndex = sentences[min(i + numSentencesPerSequence, len(sentences)) - 1].end
			span = textParsed[startIndex:endIndex]
			sequenceText = span.text
			if(not sequenceText.strip()):
				continue	#avoid whitespace-only sequences (spaCy transformer shape mismatch)
			sequenceText = sequenceText.lstrip()
			if(not spacyPipelineBatchSequences):
				if(spacyPipelineSingleParse or skipMode):
					sequenceParsed = span.as_doc()
				else:
					sequenceParsed = nlpSequence(sequenceText)
				if(len(sequenceParsed) == 0):
					continue
				if(len(sequenceParsed) < minSequenceLength):
					continue
				sequences.append(sequenceParsed)
				sequencesRaw.append(sequenceText)
			else:
				sequencesText.append(sequenceText)
	else:
		for sentence in sentences:
			sequenceText = sentence.text
			if(not sequenceText.strip()):
				continue	#avoid whitespace-only sequences (spaCy transformer shape mismatch)
			sequenceText = sequenceText.lstrip()
			if(not spacyPipelineBatchSequences):
				if(spacyPipelineSingleParse or skipMode):
					sequenceParsed = sentence.as_doc()
				else:
					sequenceParsed = nlpSequence(sequenceText)
				if(len(sequenceParsed) == 0):
					continue
				if(len(sequenceParsed) < minSequenceLength):
					continue
				sequences.append(sequenceParsed)
				sequencesRaw.append(sequenceText)
			else:
				sequencesText.append(sequenceText)
	if(spacyPipelineBatchSequences):
		for sequenceIndex, sequenceParsed in enumerate(nlpSequence.pipe(sequencesText)):
			if(len(sequenceParsed) == 0):
				continue
			if(len(sequenceParsed) < minSequenceLength):
				continue
			sequences.append(sequenceParsed)
			sequencesRaw.append(sequencesText[sequenceIndex])
	return sequences, sequencesRaw


def processSequence(databaseNetworkObject, inferenceMode, sequenceCount, articleIndex, sequenceIndex, sequence, sequenceRaw, inferenceSuccessfulPredictionMask=None):
	trainMode = not inferenceMode
	if(inferenceMode):
		if(useTrainDuringInference):
			if(multipleDendriticBranchesBinaryTree):
				databaseNetworkObject.multipleDendriticBranchesBinaryTreeInferenceActivation = None
	
	sequenceTrainTotalStartTime = None
	if(debugPrintTrainSectionTimes and trainMode):
		GIAANNcmn_debug.debugTrainSectionTimesReset(databaseNetworkObject, sequenceCount)
		sequenceTrainTotalStartTime = time.perf_counter()
	preprocessSequenceStartTime = None
	if(debugPrintTrainSectionTimes and trainMode):
		preprocessSequenceStartTime = time.perf_counter()

	sequence = GIAANNnlp_sequenceTokens.preprocessSequence(sequence)
	
	if(debugPrintTrainSectionTimes and trainMode):
		GIAANNcmn_debug.debugTrainSectionTimesAdd(databaseNetworkObject, "preprocessSequence", time.perf_counter() - preprocessSequenceStartTime)
	if(debugReloadGlobalFeatureNeuronsEverySequence):
		GIAANNcmn_databaseNetwork.initialiseDatabaseNetwork(inferenceMode)
		if(storeDatabaseGlobalFeatureNeuronsInRam):
			databaseNetworkObject.globalFeatureNeurons = GIAANNcmn_databaseNetwork.initialiseFeatureNeuronsGlobal(inferenceMode, databaseNetworkObject.c, databaseNetworkObject.f)
	
	if(inferenceMode):
		if(not inferenceTrainFirstSequences):
			if(not useTrainDuringInference):
				GIAANNcmn_databaseNetwork.restoreGlobalArrays(databaseNetworkObject)	#reset activations so each prompt sequence is independent
	
	databaseNetworkObject.articleIndexDebug = articleIndex
	databaseNetworkObject.sequenceIndexDebug = sequenceIndex
	
	# Refresh the observed columns dictionary for each new sequence
	observedColumnsDict = {}  # key: lemma, value: ObservedColumn
	observedColumnsSequenceWordIndexDict = {}  # key: sequence word index, value: ObservedColumn
	
	if(inferenceMode):
		if(numSeedTokens >= len(sequence)):
			if(useTrainDuringInference):
				tokens = GIAANNnlp_sequenceTokens.getTokens(sequence)
				inferenceSuccessfulPredictionMask = GIAANNcmn_prediction.createInferenceSuccessfulPredictionMask(tokens)
				return inferenceSuccessfulPredictionMask
			return
		sequenceSeed = sequence[0:numSeedTokens]	#prompt
		sequencePredict = sequence[numSeedTokens:]
	
	allowNewFeatures = True
	if(storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam):
		if(not databaseNetworkObject.observedColumnsRAMLoaded):
			raise RuntimeError("processSequence error: storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam requires observedColumnsRAMLoaded after startup")
	if(inferenceMode and inferenceAddNewFeatures):
		expandSequenceForInference(databaseNetworkObject, sequence)
		allowNewFeatures = False

	# First pass: Extract words, lemmas, pos, tags, and update concept_columns_dict and c
	firstPassStartTime = None
	if(debugPrintTrainSectionTimes and trainMode):
		firstPassStartTime = time.perf_counter()
	if(debugPrintRamMaxUsagePhaseLocal and not inferenceMode):
		GIAANNcmn_debug.debugResetGpuRamMaxUsagePhaseLocal("firstPass")
	
	conceptsFound, conceptMask = GIAANNnlp_sequenceConcepts.firstPass(databaseNetworkObject, sequence, allowNewFeatures)
	
	if(debugPrintRamMaxUsagePhaseLocal and not inferenceMode):
		GIAANNcmn_debug.debugRecordGpuRamMaxUsagePhaseLocal("firstPass")
	if(debugPrintTrainSectionTimes and trainMode):
		GIAANNcmn_debug.debugTrainSectionTimesAdd(databaseNetworkObject, "firstPass", time.perf_counter() - firstPassStartTime)
	
	if(conceptsFound):
		getTokensStartTime = None
		if(debugPrintTrainSectionTimes and trainMode):
			getTokensStartTime = time.perf_counter()
		
		tokens = GIAANNnlp_sequenceTokens.getTokens(sequence)
		if(inferenceMode and useTrainDuringInference and inferenceSuccessfulPredictionMask is None):
			inferenceSuccessfulPredictionMask = GIAANNcmn_prediction.createInferenceSuccessfulPredictionMask(tokens)
		
		if(debugPrintTrainSectionTimes and trainMode):
			GIAANNcmn_debug.debugTrainSectionTimesAdd(databaseNetworkObject, "getTokens", time.perf_counter() - getTokensStartTime)

		# When usePOS is enabled, detect all possible new features in the sequence
		detectNewFeaturesStartTime = None
		if not (useDedicatedFeatureLists):
			if(debugPrintTrainSectionTimes and trainMode):
				detectNewFeaturesStartTime = time.perf_counter()
			if(debugPrintRamMaxUsagePhaseLocal and not inferenceMode):
				GIAANNcmn_debug.debugResetGpuRamMaxUsagePhaseLocal("detectNewFeatures")
			
			GIAANNnlp_sequenceConcepts.detectNewFeatures(databaseNetworkObject, tokens, allowNewFeatures)
			
			if(debugPrintRamMaxUsagePhaseLocal and not inferenceMode):
				GIAANNcmn_debug.debugRecordGpuRamMaxUsagePhaseLocal("detectNewFeatures")
			if(debugPrintTrainSectionTimes and trainMode):
				GIAANNcmn_debug.debugTrainSectionTimesAdd(databaseNetworkObject, "detectNewFeatures", time.perf_counter() - detectNewFeaturesStartTime)

		if((not printTrainSequenceBar and trainMode) or (not printEvalSequenceBar and not trainMode)):
			GIAANNcmn_executionProgress.printTrainSequenceText(sequenceCount, sequence, tokens, sequenceRaw)
			
		# Second pass: Create observed_columns_dict
		secondPassStartTime = None
		if(debugPrintTrainSectionTimes and trainMode):
			secondPassStartTime = time.perf_counter()
		if(debugPrintRamMaxUsagePhaseLocal and not inferenceMode):
			GIAANNcmn_debug.debugResetGpuRamMaxUsagePhaseLocal("secondPass")
		
		observedColumnsDict, observedColumnsSequenceWordIndexDict = GIAANNnlp_sequenceConcepts.secondPass(databaseNetworkObject, tokens, inferenceMode)
		
		if(debugPrintRamMaxUsagePhaseLocal and not inferenceMode):
			GIAANNcmn_debug.debugRecordGpuRamMaxUsagePhaseLocal("secondPass")
		if(debugPrintTrainSectionTimes and trainMode):
			GIAANNcmn_debug.debugTrainSectionTimesAdd(databaseNetworkObject, "secondPass", time.perf_counter() - secondPassStartTime)

		# Create the sequence observed columns object
		sequenceObservedColumnsInitStartTime = None
		if(debugPrintTrainSectionTimes and trainMode):
			sequenceObservedColumnsInitStartTime = time.perf_counter()
		if(debugPrintRamMaxUsagePhaseLocal and not inferenceMode):
			GIAANNcmn_debug.debugResetGpuRamMaxUsagePhaseLocal("SequenceObservedColumns.__init__")
		
		sequenceObservedColumns = GIAANNcmn_sequenceObservedColumns.SequenceObservedColumns(databaseNetworkObject, tokens, observedColumnsDict, observedColumnsSequenceWordIndexDict, inferenceMode)
		
		if(debugPrintRamMaxUsagePhaseLocal and not inferenceMode):
			GIAANNcmn_debug.debugRecordGpuRamMaxUsagePhaseLocal("SequenceObservedColumns.__init__")
		if(debugPrintTrainSectionTimes and trainMode):
			GIAANNcmn_debug.debugTrainSectionTimesAdd(databaseNetworkObject, "SequenceObservedColumns.__init__", time.perf_counter() - sequenceObservedColumnsInitStartTime)

		if(inferenceMode):
			if(conceptColumnsDelimitByPOS and sequenceObservedColumns.noDelimiterDetectedBetweenConceptTokens):
				if(debugWarningInferenceNoDelimiterDetectedBetweenConceptTokens):
					print("warning: inference skipped due to missing concept column delimiter detection in sequence")
			else:
				# Process each concept word in the sequence (predict)
				inferenceSuccessfulPredictionMask = GIAANNcmn_prediction.processConceptWordsInference(sequenceObservedColumns, sequenceCount, sequence, sequenceSeed, sequencePredict, numSeedTokens, sequenceRaw)
		else:
			# Process each concept word in the sequence (train)
			requiredSourceFeatureIndicesByObservedColumn = sequenceObservedColumns.getTrainRequiredSourceFeatureIndicesByObservedColumn()
			
			if(debugPrintRamMaxUsagePhaseLocal):
				GIAANNcmn_debug.debugResetGpuRamMaxUsagePhaseLocal("prepareObservedColumnsForTrainSequence")
				
			GIAANNcmn_databaseNetwork.prepareObservedColumnsForTrainSequence(observedColumnsDict, requiredSourceFeatureIndicesByObservedColumn)
			if(trainVerifyConnectionNonexistentAcrossBranches):
				if(multipleDendriticBranchesBinaryTree):
					if(multipleDendriticBranchesBinaryTreeDepthSelectMostConnectedRootBranches):
						sequenceObservedColumns.prepareTrainVerifyConnectionNonexistentAcrossBranches()
			if(auxiliaryNeurons and auxiliaryNeuronsSimilar):
				GIAANNnlp_auxiliaryNeuronsSimilarWords.prepareObservedColumnsForTrainSequenceAuxiliary(sequenceObservedColumns, observedColumnsDict, allowNewFeatures)
			
			if(debugPrintRamMaxUsagePhaseLocal):
				GIAANNcmn_debug.debugRecordGpuRamMaxUsagePhaseLocal("prepareObservedColumnsForTrainSequence")
			if(debugPrintRamMaxUsagePhaseLocal):
				GIAANNcmn_debug.debugResetGpuRamMaxUsagePhaseLocal("trainConceptWords")
			
			trained = GIAANNcmn_databaseNetworkTrain.trainConceptWords(sequenceObservedColumns, sequenceCount, sequence, tokens, inferenceSuccessfulPredictionMask)
			
			if(debugPrintRamMaxUsagePhaseLocal):
				GIAANNcmn_debug.debugRecordGpuRamMaxUsagePhaseLocal("trainConceptWords")
			
			if(trained):
				# Update observed columns from sequence observed columns
				updateObservedColumnsWrapperStartTime = None
				
				if(debugPrintTrainSectionTimes and trainMode):
					updateObservedColumnsWrapperStartTime = time.perf_counter()
				if(debugPrintRamMaxUsagePhaseLocal):
					GIAANNcmn_debug.debugResetGpuRamMaxUsagePhaseLocal("updateObservedColumnsWrapper")
				
				sequenceObservedColumns.updateObservedColumnsWrapper()
				
				if(debugPrintRamMaxUsagePhaseLocal):
					GIAANNcmn_debug.debugRecordGpuRamMaxUsagePhaseLocal("updateObservedColumnsWrapper")
				if(debugPrintTrainSectionTimes and trainMode):
					GIAANNcmn_debug.debugTrainSectionTimesAdd(databaseNetworkObject, "updateObservedColumnsWrapper", time.perf_counter() - updateObservedColumnsWrapperStartTime)

				# Save observed columns to disk
				if(useSaveData):
					if(not storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam):
						if(debugPrintRamMaxUsagePhaseLocal):
							GIAANNcmn_debug.debugResetGpuRamMaxUsagePhaseLocal("saveData(sequence)")
						GIAANNcmn_databaseNetworkFiles.saveData(databaseNetworkObject, observedColumnsDict, sequenceCount)
						if(debugPrintRamMaxUsagePhaseLocal):
							GIAANNcmn_debug.debugRecordGpuRamMaxUsagePhaseLocal("saveData(sequence)")
				if(drawNetworkDuringTrain):
					# Visualize the complete graph every time a new sequence is parsed by the application.
					GIAANNcmn_databaseNetworkDraw.visualizeGraph(sequenceObservedColumns, False, save=drawNetworkDuringTrainSave, fileName=drawNetworkDuringTrainSaveFilenamePrepend+generateDrawSequenceIndex(sequenceIndex))

		if(storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam):
			GIAANNcmn_databaseNetwork.moveObservedColumnsDictConnectionsToDatabaseAfterTrain(observedColumnsDict, inferenceMode)

		releaseRuntimeGpuMemoryStartTime = None
		if(debugPrintTrainSectionTimes and trainMode):
			releaseRuntimeGpuMemoryStartTime = time.perf_counter()
		releaseRuntimeGpuMemory(sequenceCount)
		if(debugPrintTrainSectionTimes and trainMode):
			GIAANNcmn_debug.debugTrainSectionTimesAdd(databaseNetworkObject, "releaseRuntimeGpuMemory", time.perf_counter() - releaseRuntimeGpuMemoryStartTime)

	if(debugPrintTrainSectionTimes and trainMode):
		GIAANNcmn_debug.debugTrainSectionTimesAdd(databaseNetworkObject, "totalSequenceTrain", time.perf_counter() - sequenceTrainTotalStartTime)
		GIAANNcmn_debug.debugTrainSectionTimesPrint(databaseNetworkObject)
	if(inferenceTrainFirstSequences and (printTrainSequenceBar or printEvalSequenceBar)):
		GIAANNcmn_executionProgress.updatePromptSequenceBar(sequenceCount)
	elif(printTrainSequenceBar and trainMode):
		GIAANNcmn_executionProgress.updateTrainSequenceBar(sequenceCount)
	elif(printEvalSequenceBar and not trainMode and executionMode!="trainDuringInference"):
		GIAANNcmn_executionProgress.updateEvalSequenceBar(sequenceCount)

	#note sequenceCount can be used as sequenceIndex (independent of index in sequenceList) because sequenceIndex is only used to index sequence time (same for all sequences in sequenceList)
	return inferenceSuccessfulPredictionMask

		
def releaseRuntimeGpuMemory(sequenceCount):
	if(sequenceCount < 0):
		raise RuntimeError("releaseRuntimeGpuMemory error: sequenceCount must be >= 0")
	releaseGpuMemory = False
	if(debugRuntimeReleaseGPUMemory):
		if(debugRuntimeReleaseGPUMemoryEverySequenceCount <= 0):
			raise RuntimeError("releaseRuntimeGpuMemory error: debugRuntimeReleaseGPUMemoryEverySequenceCount must be > 0")
		if((sequenceCount % debugRuntimeReleaseGPUMemoryEverySequenceCount) == 0):
			releaseGpuMemory = True
	if(debugDeleteGPUcache):
		releaseGpuMemory = True
	if(releaseGpuMemory):
		if(pt.cuda.is_available()):
			gc.collect()
			pt.cuda.empty_cache()
			pt.cuda.ipc_collect()
	return
