"""GIAANNproto_main.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:

conda create -n pytorchsenv
source activate pytorchsenv
conda install python=3.12
python -m pip install --upgrade pip
pip install networkx
pip install matplotlib
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install spacy
datasetsLibrary4plus=False: pip install "datasets<4" "fsspec==2024.6.1" "gcsfs==2024.6.1"
python -m spacy download en_core_web_sm [spacyModelName]
pip install nltk

# Usage:
source activate pytorchsenv
python GIAANNproto_main.py

# Description:
GIA ANN proto main

"""

# Import necessary libraries
import gc
import time
import torch as pt
import spacy

pt.set_grad_enabled(False)

from GIAANNproto_globalDefs import *
import GIAANNproto_debug
import GIAANNproto_count
import GIAANNproto_sparseTensors
import GIAANNproto_databaseNetwork
import GIAANNproto_databaseNetworkFiles
import GIAANNproto_databaseNetworkDraw
import GIAANNproto_sequenceTokens
import GIAANNproto_sequenceConcepts
import GIAANNproto_sequenceObservedColumns
import GIAANNproto_databaseNetworkTrain
if(executionMode=="inference" or executionMode=="trainAndInference"):
	import GIAANNproto_prediction

if(printTimeDatabaseLoadSaveTimes):
	debugPrintTimeDatabaseLoadSaveTimesExecutionModeCount = GIAANNproto_debug.getDebugPrintTimeDatabaseLoadSaveTimesExecutionModeCount()
	debugPrintTimeDatabaseLoadSaveTimesCompletedExecutionModeCount = 0
	debugPrintTimeDatabaseLoadSaveTimesProgramExecutionStartTime = 0.0
	debugPrintTimeDatabaseLoadSaveTimesHuggingFaceDatasetLoadExecutionTime = 0.0
	debugPrintTimeDatabaseLoadSaveTimesLoadAllObservedColumnsToRamExecutionTime = 0.0
	debugPrintTimeDatabaseLoadSaveTimesSaveAllObservedColumnsToDiskExecutionTime = 0.0
	debugPrintTimeDatabaseLoadSaveTimesExecuteModeStartTime = 0.0
	debugPrintTimeDatabaseLoadSaveTimesLoadAllObservedColumnsToRamStartTime = 0.0
	debugPrintTimeDatabaseLoadSaveTimesSaveAllObservedColumnsToDiskStartTime = 0.0

if(printTimeDatabaseLoadSaveTimes):
	debugPrintTimeDatabaseLoadSaveTimesProgramExecutionStartTime = time.perf_counter()

# Load the selected dataset using Hugging Face datasets
if(datasetType != "textfile" and executionMode != "inference"):
	import GIAANNproto_datasets
	if(printTimeDatabaseLoadSaveTimes):
		debugPrintTimeDatabaseLoadSaveTimesHuggingFaceDatasetLoadStartTime = time.perf_counter()
	dataset = GIAANNproto_datasets.loadDataset()
	if(printTimeDatabaseLoadSaveTimes):
		debugPrintTimeDatabaseLoadSaveTimesHuggingFaceDatasetLoadExecutionTime = debugPrintTimeDatabaseLoadSaveTimesHuggingFaceDatasetLoadExecutionTime + (time.perf_counter() - debugPrintTimeDatabaseLoadSaveTimesHuggingFaceDatasetLoadStartTime)

if(debugPrintSpacySectionTimes):
	processArticlePart1totalTime = 0
	processArticlePart2totalTime = 0
	processArticlePart1count = 0
	processArticlePart2count = 0

# Initialize spaCy model
if(spacyPipelineSingleParse):
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

def main():
	GIAANNproto_databaseNetworkFiles.prepareDatabaseFilesStartup()
	if(printRamMaxUsage or debugPrintRamMaxUsagePhaseLocal):
		GIAANNproto_debug.debugResetGpuRamMaxUsage()
	if(executionMode=="inference"):
		executeMode(True)
	elif(executionMode=="trainAndInference"):
		executeMode(False) 
		executeMode(True)
	elif(executionMode=="train"):
		executeMode(False)
	if(debugPrintRamAverageUsage and not debugPrintRamCurrentUsage):
		GIAANNproto_debug.debugPrintRamUsageSummary()
	if(debugPrintRamMaxUsagePhaseLocal):
		GIAANNproto_debug.debugPrintGpuRamMaxUsagePhaseLocalSummary()
	if(printRamMaxUsage):
		GIAANNproto_debug.debugPrintGpuRamMaxUsageSummary()
	
def executeMode(inferenceMode):
	if(printTimeDatabaseLoadSaveTimes):
		global debugPrintTimeDatabaseLoadSaveTimesExecuteModeStartTime
		global debugPrintTimeDatabaseLoadSaveTimesLoadAllObservedColumnsToRamStartTime
		global debugPrintTimeDatabaseLoadSaveTimesLoadAllObservedColumnsToRamExecutionTime
		global debugPrintTimeDatabaseLoadSaveTimesSaveAllObservedColumnsToDiskStartTime
		global debugPrintTimeDatabaseLoadSaveTimesSaveAllObservedColumnsToDiskExecutionTime
		global debugPrintTimeDatabaseLoadSaveTimesCompletedExecutionModeCount
		debugPrintTimeDatabaseLoadSaveTimesExecuteModeStartTime = time.perf_counter()
	
	databaseNetworkObject = GIAANNproto_databaseNetwork.initialiseDatabaseNetwork(inferenceMode)
	databaseNetworkObject.nlp = nlpSequence	#used by posStringToPosInt
	if(inferenceMode):
		if(inferenceStartGenerateGlobalFeatureNeuronsTensor):
			GIAANNproto_databaseNetwork.generateHighMemGlobalFeatureNeuronsForInferenceStartup(databaseNetworkObject)

	if(printCountTotalParameters):
		if(len(databaseNetworkObject.conceptColumnsList) > 0):
			GIAANNproto_count.printCountTotalParametersRun(databaseNetworkObject)
		else:
			print("printCountTotalParameters totalColumns = 0 (empty database)")
	if(usePOS):
		GIAANNproto_sequenceTokens.loadPOSdatabase()
	if(inferenceMode and not inferenceTrainFirstSequences):
		GIAANNproto_databaseNetwork.backupGlobalArrays(databaseNetworkObject)
	if(storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam):
		if(printTimeDatabaseLoadSaveTimes):
			debugPrintTimeDatabaseLoadSaveTimesLoadAllObservedColumnsToRamStartTime = time.perf_counter()
		GIAANNproto_databaseNetwork.loadAllObservedColumnsToRam(databaseNetworkObject)
		if(printTimeDatabaseLoadSaveTimes):
			debugPrintTimeDatabaseLoadSaveTimesLoadAllObservedColumnsToRamExecutionTime = debugPrintTimeDatabaseLoadSaveTimesLoadAllObservedColumnsToRamExecutionTime + (time.perf_counter() - debugPrintTimeDatabaseLoadSaveTimesLoadAllObservedColumnsToRamStartTime)
		
	for epochIndex in range(numberEpochs):
		#print("\nepochIndex = ", epochIndex)
		# Start processing the dataset
		sequenceCount = 0
		if(inferenceMode and debugPrintTotalInferenceTokens):
			GIAANNproto_debug.resetTotalInferenceTokens()
		if(inferenceMode):
			GIAANNproto_prediction.resetInferenceTop1AccuracyCounts()
		if(inferenceMode):
			sequenceCount = processPrompt(databaseNetworkObject, inferenceMode, sequenceCount)
		else:
			sequenceCount = processDataset(databaseNetworkObject, inferenceMode, sequenceCount, dataset)
		if(inferenceMode and debugPrintTotalInferenceTokens):
			GIAANNproto_debug.printTotalInferenceTokens()
		if(inferenceMode and printInferenceTop1Accuracy):
			GIAANNproto_prediction.printInferenceTop1Accuracy(databaseNetworkObject)

	if(debugPrintSpacySectionTimes):
		processArticlePart1averageTime = processArticlePart1totalTime/processArticlePart1count
		processArticlePart2averageTime = processArticlePart2totalTime/processArticlePart2count
		print(f"debugPrintSpacySectionTimes: processArticlePart1averageTime={processArticlePart1averageTime:.6f} processArticlePart2averageTime={processArticlePart2averageTime:.6f}")

	if(not inferenceMode or inferenceTrainFirstSequences):
		if(useSaveData):
			if(storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam):
				if(debugPrintRamMaxUsagePhaseLocal):
					GIAANNproto_debug.debugResetGpuRamMaxUsagePhaseLocal("saveAllObservedColumnsToDisk")
				if(printTimeDatabaseLoadSaveTimes):
					debugPrintTimeDatabaseLoadSaveTimesSaveAllObservedColumnsToDiskStartTime = time.perf_counter()
				GIAANNproto_databaseNetwork.saveAllObservedColumnsToDisk(databaseNetworkObject)
				if(printTimeDatabaseLoadSaveTimes):
					debugPrintTimeDatabaseLoadSaveTimesSaveAllObservedColumnsToDiskExecutionTime = debugPrintTimeDatabaseLoadSaveTimesSaveAllObservedColumnsToDiskExecutionTime + (time.perf_counter() - debugPrintTimeDatabaseLoadSaveTimesSaveAllObservedColumnsToDiskStartTime)
				if(debugPrintRamMaxUsagePhaseLocal):
					GIAANNproto_debug.debugRecordGpuRamMaxUsagePhaseLocal("saveAllObservedColumnsToDisk")
			if(debugPrintRamMaxUsagePhaseLocal):
				GIAANNproto_debug.debugResetGpuRamMaxUsagePhaseLocal("saveData(final)")
			GIAANNproto_databaseNetworkFiles.saveData(databaseNetworkObject, {}, sequenceCount, forceSaveGlobalState=True)
			if(debugPrintRamMaxUsagePhaseLocal):
				GIAANNproto_debug.debugRecordGpuRamMaxUsagePhaseLocal("saveData(final)")
			#only required if trainMaxSequences%saveGlobalFeatureNeuronsRate != 0

	if(printTimeDatabaseLoadSaveTimes):
		debugPrintTimeDatabaseLoadSaveTimesProgramExecutionEndTime = time.perf_counter()
		debugPrintTimeDatabaseLoadSaveTimesTotalExecutionTime = debugPrintTimeDatabaseLoadSaveTimesProgramExecutionEndTime - debugPrintTimeDatabaseLoadSaveTimesProgramExecutionStartTime
		debugPrintTimeDatabaseLoadSaveTimesCompletedExecutionModeCount = debugPrintTimeDatabaseLoadSaveTimesCompletedExecutionModeCount + 1
		if(debugPrintTimeDatabaseLoadSaveTimesCompletedExecutionModeCount == debugPrintTimeDatabaseLoadSaveTimesExecutionModeCount):
			debugPrintTimeDatabaseLoadSaveTimesHuggingFaceAdjustedTotalExecutionTime = debugPrintTimeDatabaseLoadSaveTimesTotalExecutionTime - debugPrintTimeDatabaseLoadSaveTimesHuggingFaceDatasetLoadExecutionTime
			if(debugPrintTimeDatabaseLoadSaveTimesHuggingFaceAdjustedTotalExecutionTime < 0):
				raise RuntimeError("executeMode error: debugPrintTimeDatabaseLoadSaveTimesHuggingFaceAdjustedTotalExecutionTime must be >= 0")
			GIAANNproto_debug.printDebugPrintTimeDatabaseLoadSaveTimesEntry("Hugging Face dataset load execution time", debugPrintTimeDatabaseLoadSaveTimesHuggingFaceDatasetLoadExecutionTime)
			GIAANNproto_debug.printDebugPrintTimeDatabaseLoadSaveTimesSummary("printTimeDatabaseLoadSaveTimes execution times:", debugPrintTimeDatabaseLoadSaveTimesTotalExecutionTime, debugPrintTimeDatabaseLoadSaveTimesLoadAllObservedColumnsToRamExecutionTime, debugPrintTimeDatabaseLoadSaveTimesSaveAllObservedColumnsToDiskExecutionTime)
			GIAANNproto_debug.printDebugPrintTimeDatabaseLoadSaveTimesSummary("printTimeDatabaseLoadSaveTimes execution times with Hugging Face dataset load time subtracted from total execution time:", debugPrintTimeDatabaseLoadSaveTimesHuggingFaceAdjustedTotalExecutionTime, debugPrintTimeDatabaseLoadSaveTimesLoadAllObservedColumnsToRamExecutionTime, debugPrintTimeDatabaseLoadSaveTimesSaveAllObservedColumnsToDiskExecutionTime)
	
def releaseRuntimeGpuMemory(sequenceCount):
	if(sequenceCount < 0):
		raise RuntimeError("releaseRuntimeGpuMemory error: sequenceCount must be >= 0")
	releaseGpuMemory = False
	if(runtimeReleaseGPUMemory):
		if(runtimeReleaseGPUMemoryEverySequenceCount <= 0):
			raise RuntimeError("releaseRuntimeGpuMemory error: runtimeReleaseGPUMemoryEverySequenceCount must be > 0")
		if((sequenceCount % runtimeReleaseGPUMemoryEverySequenceCount) == 0):
			releaseGpuMemory = True
	if(debugDeleteGPUcache):
		releaseGpuMemory = True
	if(releaseGpuMemory):
		if(pt.cuda.is_available()):
			gc.collect()
			pt.cuda.empty_cache()
			pt.cuda.ipc_collect()
	return

def buildSequenceWithDelimiters(sequence, tokens):
	if(conceptColumnsDelimitByPOS):
		delimiterTypes = []
		for tokenIndex, token in enumerate(tokens):
			_, isDelimiterDeterministic, isDelimiterProbabilistic = GIAANNproto_sequenceConcepts.isFeaturePOSreferenceSetDelimiterType(token.word, token, tokens, tokenIndex)
			if(isDelimiterDeterministic):
				delimiterTypes.append("Dd")	#deterministic
			elif(isDelimiterProbabilistic):
				delimiterTypes.append("Di")	#indeterministic
			elif(GIAANNproto_sequenceTokens.isConcept(token)):
				delimiterTypes.append("C")	#concept
			else:
				delimiterTypes.append("")	#non
	else:
		printe("conceptColumnsDelimitByPOS is required")
	sentenceWithDelimiters = " ".join(
		f"{token.text} ({tokenIndex}:{delimiterTypes[tokenIndex]})"
		for tokenIndex, token in enumerate(sequence)
	)
	return sentenceWithDelimiters

			
def processPrompt(databaseNetworkObject, inferenceMode, sequenceCount):
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
	conceptsFound, conceptMask = GIAANNproto_sequenceConcepts.firstPass(databaseNetworkObject, sequence, True)
	if(conceptsFound):
		tokens = GIAANNproto_sequenceTokens.getTokens(sequence)
		if not (useDedicatedFeatureLists):
			GIAANNproto_sequenceConcepts.detectNewFeatures(databaseNetworkObject, tokens, True)
		observedColumnsDict, observedColumnsSequenceWordIndexDict = GIAANNproto_sequenceConcepts.secondPass(databaseNetworkObject, tokens, False)
	GIAANNproto_databaseNetwork.ensureGlobalFeatureNeuronsSize(databaseNetworkObject, True)
	return
	
def processDataset(databaseNetworkObject, inferenceMode, sequenceCount, dataset):
	for articleIndex, datasetEntry in enumerate(dataset):
		if(debugPrintSpacySectionTimes):
			getDatasetEntryTextStartTime = None
			getDatasetEntryTextDuration = 0.0
			getDatasetEntryTextStartTime = time.perf_counter()
		text = GIAANNproto_datasets.getDatasetEntryText(datasetEntry, articleIndex)
		if(debugPrintSpacySectionTimes):
			getDatasetEntryTextDuration = time.perf_counter() - getDatasetEntryTextStartTime
			print(f"debugPrintSpacySectionTimes: articleIndex={articleIndex} sequenceCount={sequenceCount} sequenceCount={sequenceCount} datasetEntryTextSeconds={getDatasetEntryTextDuration:.6f}")
		sequenceCount = processArticle(databaseNetworkObject, inferenceMode, sequenceCount, text, articleIndex)
		if(sequenceCount == trainMaxSequences and inferenceMode==False):
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

	if(debugPrintSpacySectionTimes):
		processArticlePart1Duration = time.perf_counter() - processArticlePart1StartTime
		processArticlePart2StartTime = None
		processArticlePart2Duration = 0.0
		processArticlePart2StartTime = time.perf_counter()

	numberOfSequences = len(sequences)
	for sequenceIndex, sequence in enumerate(sequences):
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
				if(sequenceIndex==0):
					if(printHeaderDuringInferencePredict):
						print("\n!inferenceTrainFirstSequences: executing inference:")
		if(len(sequence) <= maxSequenceLength):
			if(sequenceCount >= trainSetStartOffsetSequences):
				processSequence(databaseNetworkObject, inferenceSequenceInPrompt, sequenceCount, articleIndex, sequenceIndex, sequence, sequenceRaw)
			else:
				#if(printTrainSequenceCount):
				print(f"(sequenceCount < trainSetStartOffsetSequences: Processing sequenceCount: {sequenceCount}")	
			sequenceCount += 1
		if(sequenceCount == trainMaxSequences and inferenceMode==False):
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


def processSequence(databaseNetworkObject, inferenceMode, sequenceCount, articleIndex, sequenceIndex, sequence, sequenceRaw):
	trainMode = not inferenceMode
	sequenceTrainTotalStartTime = None
	if(debugPrintTrainSectionTimes and trainMode):
		GIAANNproto_debug.debugTrainSectionTimesReset(databaseNetworkObject, sequenceCount)
		sequenceTrainTotalStartTime = time.perf_counter()
	preprocessSequenceStartTime = None
	if(debugPrintTrainSectionTimes and trainMode):
		preprocessSequenceStartTime = time.perf_counter()

	sequence = GIAANNproto_sequenceTokens.preprocessSequence(sequence)
	if(debugPrintTrainSectionTimes and trainMode):
		GIAANNproto_debug.debugTrainSectionTimesAdd(databaseNetworkObject, "preprocessSequence", time.perf_counter() - preprocessSequenceStartTime)
	
	if(debugReloadGlobalFeatureNeuronsEverySequence):
		GIAANNproto_databaseNetwork.initialiseDatabaseNetwork(inferenceMode)
		if(storeDatabaseGlobalFeatureNeuronsInRam):
			databaseNetworkObject.globalFeatureNeurons = GIAANNproto_databaseNetwork.initialiseFeatureNeuronsGlobal(inferenceMode, databaseNetworkObject.c, databaseNetworkObject.f)
	if(inferenceMode):
		if(not inferenceTrainFirstSequences):
			GIAANNproto_databaseNetwork.restoreGlobalArrays(databaseNetworkObject)	#reset activations so each prompt sequence is independent
	
	databaseNetworkObject.articleIndexDebug = articleIndex
	databaseNetworkObject.sequenceIndexDebug = sequenceIndex
	
	# Refresh the observed columns dictionary for each new sequence
	observedColumnsDict = {}  # key: lemma, value: ObservedColumn
	observedColumnsSequenceWordIndexDict = {}  # key: sequence word index, value: ObservedColumn
	
	if(inferenceMode):
		if(numSeedTokens >= len(sequence)):
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
		GIAANNproto_debug.debugResetGpuRamMaxUsagePhaseLocal("firstPass")
	conceptsFound, conceptMask = GIAANNproto_sequenceConcepts.firstPass(databaseNetworkObject, sequence, allowNewFeatures)
	if(debugPrintRamMaxUsagePhaseLocal and not inferenceMode):
		GIAANNproto_debug.debugRecordGpuRamMaxUsagePhaseLocal("firstPass")
	if(debugPrintTrainSectionTimes and trainMode):
		GIAANNproto_debug.debugTrainSectionTimesAdd(databaseNetworkObject, "firstPass", time.perf_counter() - firstPassStartTime)
	
	if(conceptsFound):
		getTokensStartTime = None
		if(debugPrintTrainSectionTimes and trainMode):
			getTokensStartTime = time.perf_counter()
		tokens = GIAANNproto_sequenceTokens.getTokens(sequence)
		if(debugPrintTrainSectionTimes and trainMode):
			GIAANNproto_debug.debugTrainSectionTimesAdd(databaseNetworkObject, "getTokens", time.perf_counter() - getTokensStartTime)

		# When usePOS is enabled, detect all possible new features in the sequence
		detectNewFeaturesStartTime = None
		if not (useDedicatedFeatureLists):
			if(debugPrintTrainSectionTimes and trainMode):
				detectNewFeaturesStartTime = time.perf_counter()
			if(debugPrintRamMaxUsagePhaseLocal and not inferenceMode):
				GIAANNproto_debug.debugResetGpuRamMaxUsagePhaseLocal("detectNewFeatures")
			GIAANNproto_sequenceConcepts.detectNewFeatures(databaseNetworkObject, tokens, allowNewFeatures)
			if(debugPrintRamMaxUsagePhaseLocal and not inferenceMode):
				GIAANNproto_debug.debugRecordGpuRamMaxUsagePhaseLocal("detectNewFeatures")
			if(debugPrintTrainSectionTimes and trainMode):
				GIAANNproto_debug.debugTrainSectionTimesAdd(databaseNetworkObject, "detectNewFeatures", time.perf_counter() - detectNewFeaturesStartTime)

		if(printTrainSequencePOS):
			sentenceWithPOS = " ".join(f"{token.text} ({tokenIndex}:{token.pos_})" for tokenIndex, token in enumerate(sequence))
			print(f"Processing sequenceCount: {sequenceCount}, {sentenceWithPOS}")	#article: {articleIndex}, sequence: {sequenceIndex}
		if(printTrainSequenceDelimiters):
			sentenceWithDelimiters = buildSequenceWithDelimiters(sequence, tokens)
			print(f"Processing sequenceCount: {sequenceCount}, {sentenceWithDelimiters}")	#article: {articleIndex}, sequence: {sequenceIndex}
		if(printTrainSequenceRaw):
			print(sequenceRaw)
		if(printTrainSequenceDefault):
			print(f"Processing sequenceCount: {sequenceCount}, {sequence.text}")	#"{sequence.text}"	#"Processing sequenceCount: {sequenceCount}, {sequence.text}"	#article: {articleIndex}, sequence: {sequenceIndex}
		if(printTrainSequenceCount):
			print(f"Processing sequenceCount: {sequenceCount}")	
			
		# Second pass: Create observed_columns_dict
		secondPassStartTime = None
		if(debugPrintTrainSectionTimes and trainMode):
			secondPassStartTime = time.perf_counter()
		if(debugPrintRamMaxUsagePhaseLocal and not inferenceMode):
			GIAANNproto_debug.debugResetGpuRamMaxUsagePhaseLocal("secondPass")
		observedColumnsDict, observedColumnsSequenceWordIndexDict = GIAANNproto_sequenceConcepts.secondPass(databaseNetworkObject, tokens, inferenceMode)
		if(debugPrintRamMaxUsagePhaseLocal and not inferenceMode):
			GIAANNproto_debug.debugRecordGpuRamMaxUsagePhaseLocal("secondPass")
		if(debugPrintTrainSectionTimes and trainMode):
			GIAANNproto_debug.debugTrainSectionTimesAdd(databaseNetworkObject, "secondPass", time.perf_counter() - secondPassStartTime)

		# Create the sequence observed columns object
		sequenceObservedColumnsInitStartTime = None
		if(debugPrintTrainSectionTimes and trainMode):
			sequenceObservedColumnsInitStartTime = time.perf_counter()
		if(debugPrintRamMaxUsagePhaseLocal and not inferenceMode):
			GIAANNproto_debug.debugResetGpuRamMaxUsagePhaseLocal("SequenceObservedColumns.__init__")
		sequenceObservedColumns = GIAANNproto_sequenceObservedColumns.SequenceObservedColumns(databaseNetworkObject, tokens, observedColumnsDict, observedColumnsSequenceWordIndexDict, inferenceMode)
		if(debugPrintRamMaxUsagePhaseLocal and not inferenceMode):
			GIAANNproto_debug.debugRecordGpuRamMaxUsagePhaseLocal("SequenceObservedColumns.__init__")
		if(debugPrintTrainSectionTimes and trainMode):
			GIAANNproto_debug.debugTrainSectionTimesAdd(databaseNetworkObject, "SequenceObservedColumns.__init__", time.perf_counter() - sequenceObservedColumnsInitStartTime)

		if(inferenceMode):
			if(conceptColumnsDelimitByPOS and sequenceObservedColumns.noDelimiterDetectedBetweenConceptTokens):
				if(debugWarningInferenceNoDelimiterDetectedBetweenConceptTokens):
					print("warning: inference skipped due to missing concept column delimiter detection in sequence")
			else:
				# Process each concept word in the sequence (predict)
				GIAANNproto_prediction.processConceptWordsInference(sequenceObservedColumns, sequenceCount, sequence, sequenceSeed, sequencePredict, numSeedTokens, sequenceRaw)
		else:
			# Process each concept word in the sequence (train)
			requiredSourceFeatureIndicesByObservedColumn = sequenceObservedColumns.getTrainRequiredSourceFeatureIndicesByObservedColumn()
			if(debugPrintRamMaxUsagePhaseLocal):
				GIAANNproto_debug.debugResetGpuRamMaxUsagePhaseLocal("prepareObservedColumnsForTrainSequence")
			GIAANNproto_databaseNetwork.prepareObservedColumnsForTrainSequence(observedColumnsDict, requiredSourceFeatureIndicesByObservedColumn)
			if(debugPrintRamMaxUsagePhaseLocal):
				GIAANNproto_debug.debugRecordGpuRamMaxUsagePhaseLocal("prepareObservedColumnsForTrainSequence")
			if(debugPrintRamMaxUsagePhaseLocal):
				GIAANNproto_debug.debugResetGpuRamMaxUsagePhaseLocal("trainConceptWords")
			trained = GIAANNproto_databaseNetworkTrain.trainConceptWords(sequenceObservedColumns, sequenceCount, sequence, tokens)
			if(debugPrintRamMaxUsagePhaseLocal):
				GIAANNproto_debug.debugRecordGpuRamMaxUsagePhaseLocal("trainConceptWords")
			if(trained):
				# Update observed columns from sequence observed columns
				updateObservedColumnsWrapperStartTime = None
				if(debugPrintTrainSectionTimes and trainMode):
					updateObservedColumnsWrapperStartTime = time.perf_counter()
				if(debugPrintRamMaxUsagePhaseLocal):
					GIAANNproto_debug.debugResetGpuRamMaxUsagePhaseLocal("updateObservedColumnsWrapper")
				sequenceObservedColumns.updateObservedColumnsWrapper()
				if(debugPrintRamMaxUsagePhaseLocal):
					GIAANNproto_debug.debugRecordGpuRamMaxUsagePhaseLocal("updateObservedColumnsWrapper")
				if(debugPrintTrainSectionTimes and trainMode):
					GIAANNproto_debug.debugTrainSectionTimesAdd(databaseNetworkObject, "updateObservedColumnsWrapper", time.perf_counter() - updateObservedColumnsWrapperStartTime)

				# Save observed columns to disk
				if(useSaveData):
					if(not storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam):
						if(debugPrintRamMaxUsagePhaseLocal):
							GIAANNproto_debug.debugResetGpuRamMaxUsagePhaseLocal("saveData(sequence)")
						GIAANNproto_databaseNetworkFiles.saveData(databaseNetworkObject, observedColumnsDict, sequenceCount)
						if(debugPrintRamMaxUsagePhaseLocal):
							GIAANNproto_debug.debugRecordGpuRamMaxUsagePhaseLocal("saveData(sequence)")

				if(drawNetworkDuringTrain):
					# Visualize the complete graph every time a new sequence is parsed by the application.
					GIAANNproto_databaseNetworkDraw.visualizeGraph(sequenceObservedColumns, False, save=drawNetworkDuringTrainSave, fileName=drawNetworkDuringTrainSaveFilenamePrepend+generateDrawSequenceIndex(sequenceIndex))

		if(storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam):
			GIAANNproto_databaseNetwork.moveObservedColumnsDictConnectionsToDatabaseAfterTrain(observedColumnsDict, inferenceMode)

		releaseRuntimeGpuMemoryStartTime = None
		if(debugPrintTrainSectionTimes and trainMode):
			releaseRuntimeGpuMemoryStartTime = time.perf_counter()
		releaseRuntimeGpuMemory(sequenceCount)
		if(debugPrintTrainSectionTimes and trainMode):
			GIAANNproto_debug.debugTrainSectionTimesAdd(databaseNetworkObject, "releaseRuntimeGpuMemory", time.perf_counter() - releaseRuntimeGpuMemoryStartTime)

	if(debugPrintTrainSectionTimes and trainMode):
		GIAANNproto_debug.debugTrainSectionTimesAdd(databaseNetworkObject, "totalSequenceTrain", time.perf_counter() - sequenceTrainTotalStartTime)
		GIAANNproto_debug.debugTrainSectionTimesPrint(databaseNetworkObject)

	#note sequenceCount can be used as sequenceIndex (independent of index in sequenceList) because sequenceIndex is only used to index sequence time (same for all sequences in sequenceList)

if __name__ == "__main__":
	with pt.no_grad():
		main()
