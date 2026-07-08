"""GIAANNnlp_globalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 BAI Research Pty Ltd (bairesearch.com.au)

# License:
MIT License

# Installation:
see GIAANNcmn_globalDefs.py

# Usage:
see GIAANNcmn_globalDefs.py

# Description:
GIA ANN NLP global Defs

"""

import torch as pt
import math
import sys


#Recent debug vars;


#Execution mode selection;
from GIAANNcmn_globalDefs import useQuickExecution
from GIAANNcmn_globalDefs import useBenchmark
from GIAANNcmn_globalDefs import useAutoresearch
from GIAANNcmn_globalDefs import useDrawNetworkIndependently
from GIAANNcmn_globalDefs import useDefault
from GIAANNcmn_globalDefs import useTrainDuringInference
#from GIAANNcmn_globalDefs import multipleDendriticBranches
#from GIAANNcmn_globalDefs import multipleDendriticBranchesNumber
from GIAANNcmn_globalDefs import useSANI
from GIAANNcmn_globalDefs import useInference
from GIAANNcmn_globalDefs import inferenceEvaluateTestSet
from GIAANNcmn_globalDefs import inferenceAddNewFeatures
#useBenchmark dependencies:
from GIAANNcmn_globalDefs import multipleDendriticBranches
from GIAANNcmn_globalDefs import multipleDendriticBranchesNumber
from GIAANNcmn_globalDefs import multipleDendriticBranchesRandom
from GIAANNcmn_globalDefs import trainMaxSequences
from GIAANNcmn_globalDefs import numSeedTokensInference
from GIAANNcmn_globalDefs import inferenceEvaluateTestSetTrainMaxSequences10M
#useBenchmark dependencies [v2]:
#inferenceReportGroundedAccuracy
#auxiliaryNeurons
from GIAANNcmn_globalDefs import useTrainDuringInference
from GIAANNcmn_globalDefs import multipleDendriticBranchesBinaryTree
from GIAANNcmn_globalDefs import trainVerifyConnectionNonexistentAcrossBranches

#Dataset Type;
if(useQuickExecution):
	datasetType = "textfile"
	useBenchmarkEvalDataSet = False	#not used
elif(useDefault):
	datasetType = "oscar"	#"oscar" / "wikipedia" / "textfile" [experimental: "closedWorldGrounded1" / "closedWorldGrounded2" / "closedWorldGrounded3"]
	useBenchmarkEvalDataSet = True	#default: True	#optional	#use an official eval dataset (prompt) - else user must provide a custom inference_prompt.txt
elif(useBenchmark):
	datasetType = "oscar"	#"oscar"/"wikipedia"
	useBenchmarkEvalDataSet = True	#mandatory: True	#use benchmark file naming schemes and evals
elif(useAutoresearch):
	datasetType = "oscar"
	useBenchmarkEvalDataSet = True	#default: True	#optional
elif(useDrawNetworkIndependently):
	datasetType = "oscar"
	useBenchmarkEvalDataSet = True	#default: True	#optional
elif(useTrainDuringInference):
	datasetType = "oscar"	#"oscar" / "wikipedia" / "textfile" [experimental: "closedWorldGrounded1" / "closedWorldGrounded2" / "closedWorldGrounded3"]
	useBenchmarkEvalDataSet = True	#default: True	#optional	#use an official eval dataset (prompt) - else user must provide a custom inference_prompt.txt

#Multisentence predictions;
sentencePredictions = True 	#default: True	orig: True
if(sentencePredictions):
	skipSequenceNoDelimiterDetectedBetweenConceptTokens = True	#default: True #orig: True
	multisentencePredictions = False	#default: False	#each sequence comprises multiple sentences	#requires higher GPU RAM for train
	if(multisentencePredictions):
		maxSequenceLength = 80	#512
		numSentencesPerSequence = 3	#default: 3	#int(maxSequenceLength/25) ~= 20 for maxSequenceLength=512
		#print("numSentencesPerSequence = ", numSentencesPerSequence)
	else:
		maxSequenceLength = 80	#default:80	#orig:100		#in words	#depends on CPU/GPU RAM availability during train 	#measured in spacy word tokens, not subword tiktokens (even if tokeniserSubword is enabled).
		numSentencesPerSequence = 1
	sequencesCropToMaxLength = False	#default: True	#orig: False
else:
	skipSequenceNoDelimiterDetectedBetweenConceptTokens = False
	multisentencePredictions = False	#mandatory: False
	numSentencesPerSequence = 1	#mandatory: 1
	sequencesCropToMaxLength = True	#mandatory: True
	tokensPerWord = 1.25 	#note approx avg 1.25 tiktokens per word, so 80*1.25 = 100 tokens (assuming o200k_base tokeniser with OSCAR-2201 en dataset)
	targetContextLengthInTokens = 512	#emulate
	maxSequenceLength = int(targetContextLengthInTokens/tokensPerWord)	#measured in spacy word tokens, not subword tiktokens (even if tokeniserSubword is enabled)
	
#Closed world grounded dataset constants;
datasetTypeClosedWorldGrounded1 = "closedWorldGrounded1"
datasetTypeClosedWorldGrounded2 = "closedWorldGrounded2"
datasetTypeClosedWorldGrounded3 = "closedWorldGrounded3"
closedWorldGroundedDatasetTypes = [datasetTypeClosedWorldGrounded1, datasetTypeClosedWorldGrounded2, datasetTypeClosedWorldGrounded3]
closedWorldGroundedHfDatasetTypes = [datasetTypeClosedWorldGrounded2, datasetTypeClosedWorldGrounded3]
closedWorldGroundedDatasetTypeToDatasetNameDict = {datasetTypeClosedWorldGrounded1:"closedWorldGrounded1", datasetTypeClosedWorldGrounded2:"EleutherAI/fever", datasetTypeClosedWorldGrounded3:"allenai/scifact"}
closedWorldGroundedDatasetTypeToDatasetCfgDict = {datasetTypeClosedWorldGrounded1:"", datasetTypeClosedWorldGrounded2:"v1.0", datasetTypeClosedWorldGrounded3:"claims"}
closedWorldGroundedDatasetTypeToTrainSplitDict = {datasetTypeClosedWorldGrounded1:"train", datasetTypeClosedWorldGrounded2:"train", datasetTypeClosedWorldGrounded3:"train"}
closedWorldGroundedDatasetTypeToEvalSplitDict = {datasetTypeClosedWorldGrounded1:"test", datasetTypeClosedWorldGrounded2:"dev", datasetTypeClosedWorldGrounded3:"validation"}
closedWorldGroundedDatasetTypeToDatabaseTypeTextDict = {datasetTypeClosedWorldGrounded1:"ClosedWorldGrounded1", datasetTypeClosedWorldGrounded2:"ClosedWorldGrounded2", datasetTypeClosedWorldGrounded3:"ClosedWorldGrounded3"}
closedWorldGroundedDatasetTypeToInferencePromptFileNameDict = {datasetTypeClosedWorldGrounded1:"inference_prompt.txt.closedWorldGrounded1", datasetTypeClosedWorldGrounded2:"inference_prompt.txt.closedWorldGrounded2", datasetTypeClosedWorldGrounded3:"inference_prompt.txt.closedWorldGrounded3"}
closedWorldGroundedRealisticNLPmetricName = ""	#RealisticNLPNoSentenceSplits
closedWorldGroundedRealisticNLPmetricInferencePromptFileNameSuffix = ""	#.realisticNLPNoSentenceSplits
closedWorldGroundedStrongerGroundedNLPmetricName = "StrongerGroundedNLP"
closedWorldGroundedStrongerGroundedNLPmetricInferencePromptFileNameSuffix = ".strongerGroundedNLP"
closedWorldGroundedDatasetTypeToInferencePromptFileNameRealisticNLPmetricDict = {datasetTypeClosedWorldGrounded1:"inference_prompt.txt.closedWorldGrounded1", datasetTypeClosedWorldGrounded2:closedWorldGroundedDatasetTypeToInferencePromptFileNameDict[datasetTypeClosedWorldGrounded2] + closedWorldGroundedRealisticNLPmetricInferencePromptFileNameSuffix, datasetTypeClosedWorldGrounded3:closedWorldGroundedDatasetTypeToInferencePromptFileNameDict[datasetTypeClosedWorldGrounded3] + closedWorldGroundedRealisticNLPmetricInferencePromptFileNameSuffix}
closedWorldGroundedDatasetTypeToInferencePromptFileNameStrongerGroundedNLPmetricDict = {datasetTypeClosedWorldGrounded1:"inference_prompt.txt.closedWorldGrounded1", datasetTypeClosedWorldGrounded2:closedWorldGroundedDatasetTypeToInferencePromptFileNameDict[datasetTypeClosedWorldGrounded2] + closedWorldGroundedStrongerGroundedNLPmetricInferencePromptFileNameSuffix, datasetTypeClosedWorldGrounded3:closedWorldGroundedDatasetTypeToInferencePromptFileNameDict[datasetTypeClosedWorldGrounded3] + closedWorldGroundedStrongerGroundedNLPmetricInferencePromptFileNameSuffix}
inferenceReportGroundedAccuracy = datasetType in closedWorldGroundedDatasetTypes	#report closed-world grounded prediction accuracy
inferenceReportGroundedRealisticNLPmetric = datasetType in closedWorldGroundedHfDatasetTypes	#use real claim text for HF-backed grounded evals
inferenceReportGroundedStrongerGroundedNLPmetric = False	#use claim-derived seed tokens for HF-backed grounded evals
closedWorldGroundedMaxSentencesPerArticle = 1


#Dataset;
datasetsLibrary4plus = False	#default: False	#orig: False	#set False during dev to maintain benchmark consistency
trainTestSet = False	#default: False	#only set True to generate an inference test set (with printSequenceRaw=True)
if(trainTestSet):
	generateEvalText = True	#mandatory: True
else:
	generateEvalText = False	#optional
if(useQuickExecution):
	trainLoadExistingDatabase = True	#default: True	#set true for safety only (users must manually delete their databases)
elif(useAutoresearch):
	trainLoadExistingDatabase = False	#wipe database on new experiment start
else:
	trainLoadExistingDatabase = True	#default: True	#orig: True	#loads existing database if existant upon startup	#requires user to manually wipe database
if(datasetType=="textfile"):
	datasetName = "train_prompt.txt"
elif(datasetType=="oscar"):
	datasetName = "oscar-corpus/OSCAR-2201"
	datasetCfg = "en"
	datasetsLibrary4plus = True
	useLocalDataset = False	#not supported
elif(datasetType=="wikipedia"):
	if(datasetsLibrary4plus):
		datasetName = "wikimedia/wikipedia"
		datasetCfg = "20231101.en"
	else:
		datasetName = "wikipedia"
		datasetCfg = "20220301.en"
	useLocalDataset = True	#default: True	#orig: False (stream)	#use local dataset	#automatic huggingface access to dataset is unreliable
elif(datasetType in closedWorldGroundedDatasetTypes):
	datasetName = closedWorldGroundedDatasetTypeToDatasetNameDict[datasetType]
	datasetCfg = closedWorldGroundedDatasetTypeToDatasetCfgDict[datasetType]
	useLocalDataset = False
else:
	printe("Dataset selection error: enable either datasetType==textfile or datasetType==oscar or datasetType==wikipedia or datasetType in closedWorldGroundedDatasetTypes")
if(not datasetType=="textfile"):
	if(useLocalDataset):
		datasetFolder = "../../dataset/nlp/"
		if(datasetType=="wikipedia"):
			useLocalDatasetDownloadManual = True	#default: True	#manual download dataset files into datasetFolder	#automatic huggingface access to dataset is unreliable
		elif(datasetType=="oscar"):
			useLocalDatasetDownloadManual = False	#OSCAR2201 uses custom HF dataset code and non-parquet source files; do not use manual parquet downloader
		else:
			printe("Dataset selection error: unsupported dataset for useLocalDatasetDownloadManual configuration")
		datasetProcessedCacheFolderName = "processed_dataset_cache"	#manual name for processed dataset cache
		datasetProcessedCacheFolder = datasetFolder + datasetProcessedCacheFolderName + "/"
	else:
		useLocalDatasetDownloadManual = False
	if(trainTestSet):
		if(datasetType=="wikipedia"):
			testSetRatio = 0.1	#ratio of articles in dataset to be used for test (vs train) set - taken from end of dataset
			assert useLocalDataset	#required for efficiency
		elif(datasetType=="oscar"):
			trainMaxSequencesEver = 10000000	#orig: 1000000	#highest value of trainMaxSequences expected during current dev (using this instead of a much high value closer to 1-testSetRatio because testSetStartOffset takes time to load)
			if(sentencePredictions):
				numSentencesPerSequenceEver = 20	#orig: 3
				datasetOscarAverageEligibleSentencesPerArticle = 32	#measured across 1m raw sentences (therefore appropriate for trainMaxSequencesEver=1m)
				testSetStartOffset = int(trainMaxSequencesEver / datasetOscarAverageEligibleSentencesPerArticle)*numSentencesPerSequenceEver
			else:
				testSetStartOffset = trainMaxSequencesEver
			testSetSize = 1000	#number of entries to include in test set
		else:
			printe("trainTestSet configuration error: unsupported dataset selection")
		trainSetStartOffsetSequences = 0
	else:
		trainSetStartOffsetSequences = 0	#2000000	#1000000	#default: 0	#orig: 0	
		if(datasetType=="oscar"):
			maxSentencesPerArticle = 100	#CHECKTHIS
		elif(datasetType=="wikipedia"):
			maxSentencesPerArticle = 1000	#CHECKTHIS
		elif(datasetType in closedWorldGroundedDatasetTypes):
			maxSentencesPerArticle = closedWorldGroundedMaxSentencesPerArticle
else:
	trainSetStartOffsetSequences = 0


#Benchmarking defaults;
if(useBenchmark):
	useBenchmarkDefaults = True	#default: True
else:
	useBenchmarkDefaults = False	#default: False
if(useBenchmarkDefaults):
	spacyPipelineOptimisations = True	#default: True	#orig: False	#spacyPipelineOptimisations do not significantly affect test-set accuracies (~-0.002)
else:
	spacyPipelineOptimisations = True	#default: True


#Tokensier:
tokeniserSubword = False	#default: False #orig: False
if(tokeniserSubword):
	tokeniserSubwordPOS = True	#default: ?	#orig: False
	tokeniserSubwordTiktokenEncodingName = "o200k_base"
	tokeniserSubwordTextEncoding = "utf-8"
	tokeniserSubwordTextEncodingErrorMode = "strict"
	tokeniserSubwordByteTokenPrefix = "<tokeniserSubwordByte:"
	tokeniserSubwordByteTokenSuffix = ">"
	tokeniserSubwordInvalidTokenFeatureNamePrefix = "<tokeniserSubwordInvalidToken:"
	tokeniserSubwordInvalidTokenFeatureNameSuffix = ">"
	tokeniserSubwordFeatureIndexOffset = 1
	tokeniserSubwordPOSpunct = "PUNCT"
	tokeniserSubwordPOSnum = "NUM"
	tokeniserSubwordPOSsym = "SYM"
	tokeniserSubwordPOSspace = "SPACE"
	tokeniserSubwordTagPunct = "."
	tokeniserSubwordTagNum = "CD"
	tokeniserSubwordTagSym = "SYM"
	tokeniserSubwordTagSpace = "_SP"
else:
	tokeniserSubwordPOS = False
	

#Concept column delimiters:
conceptColumnsDelimitByPOS = True	#mandatory: True	#orig: False	#closer to original GIA specification	#FUTURE: still requires working for edge cases
if(conceptColumnsDelimitByPOS):
	conceptColumnsDelimiterPOStypes = ['VERB', 'ADP']	#deterministic reference set delimiters (GIA actions/conditions)
	conceptColumnsDelimiterWordTypes = [';', ':', '.', '?', '!', '.']	#deterministic reference set delimiters (GIA logical conditions)
	conceptColumnsDelimiterTagTypes = ['POS']	#eg possessive apostrophe "'s" (singular) or "'" (plural) -> pos: PART, tag: POS.
	attachTrailingTokensToLastConcept = True	#default: False	#attach tokens after the final concept to that last column
	detectReferenceSetDelimitersBetweenNouns = True	#default: assign reference set delimiters if they appear between two nouns (without designated reference set delimiter types)
	if(detectReferenceSetDelimitersBetweenNouns):
		detectReferenceSetDelimitersBetweenNounsPOStypes = ['CCONJ', 'SCONJ']	#probabilistic reference set delimiters (GIA logical conditions) - only assign if they are detected inbetween nouns (without intermediate deterministic delimiters)
		detectReferenceSetDelimitersBetweenNounsWordTypes = ['is', 'are', ',', '(']	#eg a dog is an animal / dogs are animals	#'-'
		detectReferenceSetDelimitersBetweenNounsTagTypes = []
	predictionColumnsMustActivateConceptFeature = False	#default: False	#orig: False
	if(tokeniserSubword):
		pretrainCombineConsecutiveNouns = False
		pretrainCombineHyphenatedNouns = False
		pretrainConceptColumnsDelimitByPOSenforce = False
	else:
		pretrainCombineConsecutiveNouns = True #default: True	#orig: False
		pretrainCombineHyphenatedNouns = True	#default: True	#orig: False
		if(useBenchmarkDefaults):
			pretrainConceptColumnsDelimitByPOSenforce = False	
		else:
			pretrainConceptColumnsDelimitByPOSenforce = True	#default: True	#orig: False	#disable when debugging debugTerminateOnConceptColumnsDelimitByPOSwarning	#when consecutive concepts are detected without a delimiter between them, it modifies all tokens to the left of the right most concept token (noun) as ordinary non-concept (non-noun) tokens.


#Connection strength modifiers;
trainConnectionStrengthPOSdependence = False	#default: False	#orig: False
trainConnectionStrengthLimitTanh = False	#default: False	#orig: False	#TODO: review this - reduce algorithmic dependency on high frequency tokens
trainConnectionStrengthLimitMax = False	#default: False	#orig: False	#TODO: review this - reduce algorithmic dependency on high frequency tokens
if(useSANI):
	trainConnectionStrengthLimitMax = False	#planned new default: True	#apply normalisation to with SANI to emphasise the combination of relevant precedents rather than overweight to specific precedents
inferenceConnectionStrengthPOSdependence = False	#default: False	#orig: False
if(trainConnectionStrengthPOSdependence or inferenceConnectionStrengthPOSdependence):
	connectionStrengthPOSdependenceTypes = ['NOUN', 'PROPN', 'ADJ', 'ADV', 'VERB', 'ADP', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NUM', 'PART', 'PRON', 'SCONJ', 'SYM', 'X']	
	connectionStrengthPOSdependenceValues = [10, 10, 3, 3, 10, 5, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1]
	connectionStrengthPOSdependenceExternal = True	#default: True	#orig: True	#False: apply modifiers to both internal/external connections, True: external connections only


#Mandatory vars;
usePOS = True		 # usePOS mode	#mandatory


#Database save paths;
if(useInference):
	if(useQuickExecution):
		if(datasetType=="textfile"):
			inferencePromptFileName = "inference_prompt.txt.trainAndInference"
		else:
			printe("useQuickExecution requires datasetType==textfile")
	else:
		if(datasetType=="wikipedia"):	
			if(useBenchmarkEvalDataSet):
				if(sentencePredictions):
					if(inferenceEvaluateTestSet):
						inferencePromptFileName = 'inference_prompt.txt.longTestWikipedia'
					else:
						inferencePromptFileName = 'inference_prompt.txt.longTrainWikipedia'	
				else:
					printe("datasetType==wikipedia sentencePredictions=False eval sets not available")
			else:
				inferencePromptFileName = 'inference_prompt.txt'
		elif(datasetType=="oscar"):
			if(useBenchmarkEvalDataSet):
				if(sentencePredictions):				
					if(multisentencePredictions):
						if(inferenceEvaluateTestSet):
							if(inferenceEvaluateTestSetTrainMaxSequences10M):
								inferencePromptFileName = 'inference_prompt.txt.longTestOscarMultiSentence2'
							else:
								inferencePromptFileName = 'inference_prompt.txt.longTestOscarMultiSentence'
						else:
							#ensure within distribution trainset;
							if(not useBenchmarkDefaults):
								inferencePromptFileName = 'inference_prompt.txt.longTrainOscarMultiSentence'
							elif(spacyPipelineOptimisations):
								inferencePromptFileName = 'inference_prompt.txt.longTrainOscarMultiSentence10Optim'
							else:
								printe("datasetType==oscar multisentencePredictions dataset incompatibility")
					else:
						if(useBenchmarkDefaults):
							if(inferenceEvaluateTestSet):
								if(inferenceEvaluateTestSetTrainMaxSequences10M):
									inferencePromptFileName = 'inference_prompt.txt.longTestOscar2'
								else:
									inferencePromptFileName = 'inference_prompt.txt.longTestOscar'
							else:
								#ensure within distribution trainset ;
								if(spacyPipelineOptimisations):
									inferencePromptFileName = 'inference_prompt.txt.longTrainOscarOptim'
								else:
									inferencePromptFileName = 'inference_prompt.txt.longTrainOscar'
						else:
							if(inferenceEvaluateTestSet):
								inferencePromptFileName = 'inference_prompt.txt.longTestOscar-useBenchmarkDefaultsFalse'
								#printe("inference_prompt.txt.longTestOscar-useBenchmarkDefaultsFalse not yet created")
							else:
								inferencePromptFileName = 'inference_prompt.txt.longTrainOscar-useBenchmarkDefaultsFalse'
								#printe("inference_prompt.txt.longTrainOscar-useBenchmarkDefaultsFalse.txt not yet created")
				else:
					if(inferenceEvaluateTestSet):
						inferencePromptFileName = 'inference_prompt.txt.longTestOscar-SentencePredictionsFalse-maxSequenceLength409'
					else:
						inferencePromptFileName = 'inference_prompt.txt.longTrainOscar-SentencePredictionsFalse-maxSequenceLength409'
			else:
				inferencePromptFileName = 'inference_prompt.txt'
		elif(datasetType=="textfile"):
			#experimental (untested)
			trainPromptFileName = datasetName	#"train_prompt.txt"
			inferencePromptFileName = "inference_prompt.txt"
		elif(datasetType in closedWorldGroundedDatasetTypes):
			if(inferenceReportGroundedRealisticNLPmetric):
				inferencePromptFileName = closedWorldGroundedDatasetTypeToInferencePromptFileNameRealisticNLPmetricDict[datasetType]
			elif(inferenceReportGroundedStrongerGroundedNLPmetric):
				inferencePromptFileName = closedWorldGroundedDatasetTypeToInferencePromptFileNameStrongerGroundedNLPmetricDict[datasetType]
			else:
				inferencePromptFileName = closedWorldGroundedDatasetTypeToInferencePromptFileNameDict[datasetType]
		else:
			printe("invalid datasetType")

posFolder = "POS/"
posDictFile = "everPos.wordnet.pkl.gz"


#POS;
useSpacyForConceptNounPOSdetection = True	#orig: True	#False: use GIAANNnlp_sequencePOS predetermined word-POS dictionaries for all pos detection (never use spacy dynamically assigned pos tags)
if(sentencePredictions):
	if(spacyPipelineOptimisations):
		spacyModelName = 'en_core_web_sm'	#default: en_core_web_sm
		spacyPipelineSingleParse = False	#default: False	#Avoid re-parsing each sentence: reuse the original Doc and create sequence docs with Span.as_doc() (or operate directly on spans) instead of nlp(sequenceText).	#parsing sequences individually helps alignment of train/test parsing for dev
		if(spacyPipelineSingleParse):
			spacyPipelineBatchSequences = False
			spacyPipelineLightweightSentenceSegmentation = False
		else:
			spacyPipelineBatchSequences = True	#default: True		#batch second pass: collect sequenceText and run nlp.pipe(...) with batch_size (and n_process if CPU) to amortize overhead.
			spacyPipelineLightweightSentenceSegmentation = True	#default: True	#Use sentence segmentation only on a lightweight pipeline (sentencizer), then run full nlp.pipe only for sequences that pass quick length/whitespace filters.
		spacyPipelineMinimalComponents = True	#default: True		#Disable unused pipeline components at spacy.load(...) (e.g., ner) if you don't use them downstream.
	else:
		spacyModelName = 'en_core_web_trf'	#default: en_core_web_trf
		spacyPipelineSingleParse = False	#default: False #orig: True	#parsing sequences individually helps alignment of train/test parsing for dev
		spacyPipelineBatchSequences = False
		spacyPipelineLightweightSentenceSegmentation = False
		spacyPipelineMinimalComponents = False
else:
	if(not spacyPipelineOptimisations):
		raise RuntimeError("sentencePredictions=False requires spacyPipelineOptimisations=True")
	spacyModelName = 'en_core_web_sm'	#default: en_core_web_sm
	spacyPipelineSingleParse = True	#mandatory: True
	spacyPipelineBatchSequences = False	#mandatory: False
	spacyPipelineLightweightSentenceSegmentation = False	#mandatory: False
	spacyPipelineMinimalComponents = True	#mandatory: True

# Define POS tag sets for nouns and non-nouns
nounPos = {'NOUN', 'PROPN'}
nonNounPos = {'ADJ', 'ADV', 'VERB', 'ADP', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NUM', 'PART', 'PRON', 'PUNCT', 'SCONJ', 'SYM', 'X'}	#incomplete as nounTags can be a subset of these (e.g. PRON: 'PRP', 'WP')
#nounTags = {}	#{'PRP', 'WP'}


def posIntToPosString(nlp, posInt):
	if posInt in nlp.vocab.strings:
		return nlp.vocab[posInt].text
	else:
		return ''
def posStringToPosInt(nlp, posString):
	return nlp.vocab.strings[posString]


#Dedicated concept/feature lists (non-dynamic);
useDedicatedFeatureLists = False	#derived var
useDedicatedFeatureListsSubword = False	#derived var
useDedicatedConceptsLists = False	#derived var
useDedicatedConceptListsSubword = False	#derived var
if(tokeniserSubword):
	useDedicatedFeatureListsSubword = True	#default: True	#orig: False
	useDedicatedConceptListsSubword = True	#default: True	#orig: False
	if(useDedicatedFeatureListsSubword):
		useDedicatedFeatureLists = True
		useDedicatedFeatureListsSubwordBenchmarkAblationSuffix = "-useDedicatedFeatureListsSubword"
	if(useDedicatedConceptListsSubword):
		useDedicatedConceptsLists = True
		useDedicatedConceptListsSubwordBenchmarkAblationSuffix = "-useDedicatedConceptListsSubword"
else:
	useDedicatedFeatureLists = False	#default: False - dynamically learn concept features	#True: use static feature lists (depreciated)
	#if usePOS and storeDatabaseGlobalFeatureNeuronsInRam:
	#	useDedicatedFeatureLists = True
	if useDedicatedFeatureLists:
		nltk.download('punkt')
		nltk.download('wordnet')
		nltk.download('omw-1.4')
		from nltk.corpus import wordnet as wn
		from nltk.tokenize import sent_tokenize

		# Obtain lists of nouns and non-nouns using the NLTK wordnet library
		nouns = set()
		for synset in wn.all_synsets('n'):
			for lemma in synset.lemma_names():
				nouns.add(lemma.lower())

		allWords = set()
		for synset in wn.all_synsets():
			for lemma in synset.lemma_names():
				allWords.add(lemma.lower())

		nonNouns = allWords - nouns
		maxNumNonNouns = len(nonNouns)


#Closed world grounded dataset constants;
closedWorldGroundedDatasetGenerated = False
if(inferenceReportGroundedAccuracy):
	inferenceReportGroundedAccuracyMod1_labelBalancedDataset = True
	inferenceReportGroundedAccuracyMod2_majorityClassBaseline = True
	inferenceReportGroundedAccuracyMod3_perLabelMetrics = True
	closedWorldGroundedDatasetGenerated = datasetType==datasetTypeClosedWorldGrounded1
	closedWorldGroundedDatasetTextFieldName = "text"
	closedWorldGroundedInferencePromptFileEncoding = "utf-8"
	closedWorldGroundedPromptArticleSeparator = "\n"
	closedWorldGroundedPromptTokenSeparator = " "
	closedWorldGroundedPromptAnswerTokenIndex = 8
	closedWorldGroundedPromptRawAnswerTokenIndex = 9
	closedWorldGroundedPromptSubjectDeterminer = "the"
	closedWorldGroundedPromptPredicate = "is"
	closedWorldGroundedPromptPropertyLabel = "property"
	closedWorldGroundedPromptConjunction = "and"
	closedWorldGroundedPromptAnswerLabel = "answer"
	closedWorldGroundedPromptCopula = "is"
	closedWorldGroundedPromptAnswerQualifier = "exactly"
	closedWorldGroundedPromptSentenceTerminator = "."
	closedWorldGroundedLabelDirectSupport = "direct_support"
	closedWorldGroundedLabelCompositionalSupport = "compositional_support"
	closedWorldGroundedLabelUnsupportedWorldTrue = "unsupported_world_true"
	closedWorldGroundedLabelUnsupportedFalse = "unsupported_false"
	closedWorldGroundedLabelNoisySupport = "noisy_support"
	closedWorldGroundedOutcomeJustified = "justified"
	closedWorldGroundedOutcomeCorrectUngrounded = "correct_ungrounded"
	closedWorldGroundedOutcomeGroundedFalsehood = "grounded_falsehood"
	closedWorldGroundedOutcomeUngroundedHallucination = "ungrounded_hallucination"
	closedWorldGroundedOutcomeAbstained = "abstained"
	closedWorldGroundedNoPredictionWord = "<no prediction>"
	closedWorldGroundedFactTupleLength = 3
	closedWorldGroundedFactTupleEntityIndex = 0
	closedWorldGroundedFactTuplePropertyIndex = 1
	closedWorldGroundedFactTupleAnswerIndex = 2
	closedWorldGroundedEvalItemTupleLength = 6
	closedWorldGroundedEvalItemTupleCategoryIndex = 0
	closedWorldGroundedEvalItemTupleEntityIndex = 1
	closedWorldGroundedEvalItemTuplePropertyIndex = 2
	closedWorldGroundedEvalItemTupleTargetAnswerIndex = 3
	closedWorldGroundedEvalItemTupleTrueAnswersIndex = 4
	closedWorldGroundedEvalItemTupleSupportedAnswersIndex = 5
	closedWorldGroundedEvalItemFieldSequenceIndex = "sequenceIndex"
	closedWorldGroundedEvalItemFieldCategory = "category"
	closedWorldGroundedEvalItemFieldEntity = "entity"
	closedWorldGroundedEvalItemFieldProperty = "property"
	closedWorldGroundedEvalItemFieldTargetAnswer = "targetAnswer"
	closedWorldGroundedEvalItemFieldTrueAnswers = "trueAnswers"
	closedWorldGroundedEvalItemFieldSupportedAnswers = "supportedAnswers"
	closedWorldGroundedEvalItemFieldText = "text"
	closedWorldGroundedEvalItemFieldAnswerTokenIndex = "answerTokenIndex"
	closedWorldGroundedEvalItemFieldClaimText = "claimText"
	closedWorldGroundedEvalItemFieldClaimSignatureTokens = "claimSignatureTokens"
	closedWorldGroundedEvalItemAnswerTokenIndexDynamic = -1
	closedWorldGroundedHfDatasetItemCountMinimum = 1
	closedWorldGroundedHfEvalMaxItems = 1000
	closedWorldGroundedHfInitialPoolItemIndex = 0
	closedWorldGroundedHfPoolItemIndexIncrement = 1
	closedWorldGroundedHfInitialAnswerCount = 0
	closedWorldGroundedHfStreaming = True
	closedWorldGroundedHfTokenSeparatorReplacement = "_"
	closedWorldGroundedHfClaimDigestTokenCount = 4
	closedWorldGroundedHfClaimDigestTokenWidth = 3
	closedWorldGroundedHfClaimDigestByteOffset = 0
	closedWorldGroundedRealisticNLPmetricPromptClaimLabel = "claim"
	closedWorldGroundedRealisticNLPmetricPromptStates = "states"
	closedWorldGroundedRealisticNLPmetricPromptThat = "that"
	closedWorldGroundedRealisticNLPmetricMaxClaimWords = 60
	closedWorldGroundedRealisticNLPmetricClaimTerminalCharacters = (".","?","!")
	closedWorldGroundedRealisticNLPmetricClaimTerminalCharacterReplacement = closedWorldGroundedPromptTokenSeparator
	closedWorldGroundedStrongerGroundedNLPmetricPromptClaimLabel = "claim"
	closedWorldGroundedHfEntityPrefixFever = "feverclaim"
	closedWorldGroundedHfEntityPrefixSciFact = "scifactclaim"
	closedWorldGroundedHfPropertyVerdict = "verdict"
	closedWorldGroundedHfFieldId = "id"
	closedWorldGroundedHfFieldClaim = "claim"
	closedWorldGroundedHfFieldLabel = "label"
	closedWorldGroundedHfFieldEvidenceLabel = "evidence_label"
	closedWorldGroundedHfSupportedAnswer = "supported"
	closedWorldGroundedHfRefutedAnswer = "refuted"
	closedWorldGroundedHfUnknownAnswer = "unknown"
	closedWorldGroundedHfAlternativeAnswerIndexOffset = 1
	closedWorldGroundedHfAnswerOptions = []
	if(datasetType==datasetTypeClosedWorldGrounded2):
		closedWorldGroundedHfAnswerOptions = [closedWorldGroundedHfSupportedAnswer, closedWorldGroundedHfRefutedAnswer]
	elif(datasetType==datasetTypeClosedWorldGrounded3):
		closedWorldGroundedHfAnswerOptions = [closedWorldGroundedHfSupportedAnswer, closedWorldGroundedHfRefutedAnswer, closedWorldGroundedHfUnknownAnswer]
	closedWorldGroundedFeverLabelRefutesIndex = 0
	closedWorldGroundedFeverLabelSupportsIndex = 1
	closedWorldGroundedFeverLabelRefutesName = "REFUTES"
	closedWorldGroundedFeverLabelSupportsName = "SUPPORTS"
	closedWorldGroundedSciFactLabelSupport = "SUPPORT"
	closedWorldGroundedSciFactLabelContradict = "CONTRADICT"
	closedWorldGroundedSciFactLabelUnknown = ""
	closedWorldGroundedTrainFactTuples = [("aurorakey","color","blue"),("aurorakey","shape","triangle"),("emberkey","color","red"),("meadowkey","color","green"),("spareitem","shape","circle"),("basaltkey","color","black"),("yellowseed","color","yellow"),("noisedrift","color","purple"),("aerolith","region","northland"),("northland","color","silver"),("solstone","material","copper"),("copper","color","orange")]
	closedWorldGroundedEvalItemTuples = [(closedWorldGroundedLabelDirectSupport,"aurorakey","color","blue",("blue",),("blue",)),(closedWorldGroundedLabelDirectSupport,"emberkey","color","red",("red",),("red",)),(closedWorldGroundedLabelCompositionalSupport,"aerolith","color","silver",("silver",),("silver",)),(closedWorldGroundedLabelCompositionalSupport,"solstone","color","orange",("orange",),("orange",)),(closedWorldGroundedLabelUnsupportedWorldTrue,"hiddenkey","color","red",("red",),()),(closedWorldGroundedLabelUnsupportedWorldTrue,"meadowkey","shape","circle",("circle",),()),(closedWorldGroundedLabelUnsupportedFalse,"falconkey","color","black",("green",),()),(closedWorldGroundedLabelNoisySupport,"noisedrift","color","purple",("yellow",),("purple",))]
	if(numSeedTokensInference != closedWorldGroundedPromptAnswerTokenIndex):
		raise RuntimeError("inferenceReportGroundedAccuracy requires numSeedTokensInference==" + str(closedWorldGroundedPromptAnswerTokenIndex))


#Auxiliary neurons;
auxiliaryNeurons=False	#default: False	#orig: False
if(auxiliaryNeurons):
	auxiliaryNeuronsAuto = True	#default: True	#orig: False
	trainReverseConnections = True
	if(auxiliaryNeuronsAuto):
		'''
		note current auxiliaryNeuronsSimilarWords implementation is computationally efficient but not biologically feasible as it relies on the existence of reverse connections
			CONSIDER adding option auxiliaryNeuronsSimilarWordsCofire:
				for each pair of feature neurons in the network:
					fire both their forward and reverse connections:
						measure the co-occurance of activations across the global feature activation matrix.
						if high co-occurance, then the two neurons are related.
		'''
		auxiliaryNeuronsSimilarWordsAuto = True	#default: True
		if(auxiliaryNeurons and auxiliaryNeuronsSimilarWordsAuto):
			auxiliaryNeuronsSimilarWordsPrimeConceptFeatures = True	#find similar noun words
			auxiliaryNeuronsSimilarWordsSecondaryConceptFeatures = True	#find similar non-noun words
			auxiliaryNeuronsSimilarWordsAutoThreshold = 0.5
			auxiliaryNeuronsSimilarWordsAutoSecondaryConceptFeaturesTrainIdentifyColumn = True
			auxiliaryNeuronsSimilarWordsSecondaryConceptFeaturesIdentifySameColumn = auxiliaryNeuronsSimilarWordsAutoSecondaryConceptFeaturesTrainIdentifyColumn
			if(auxiliaryNeuronsSimilarWordsPrimeConceptFeatures):
				auxiliaryNeuronsSimilarWordsPrimeConceptFeaturesDatasetFileName = "auxiliaryNeuronsSimilarWordsPrimeConceptFeaturesDataset.txt"
			if(auxiliaryNeuronsSimilarWordsSecondaryConceptFeatures):
				auxiliaryNeuronsSimilarWordsSecondaryConceptFeaturesDatasetFileName = "auxiliaryNeuronsSimilarWordsSecondaryConceptFeaturesDataset.txt"
			auxiliaryNeuronsSimilarWordsSecondaryConceptFeaturesLimit = True
			if(auxiliaryNeuronsSimilarWordsSecondaryConceptFeaturesLimit):
				auxiliaryNeuronsSimilarWordsSecondaryConceptFeaturesMaximumSharedSourceFeatureIndexFraction = 0.1	#sparse: 0.05	#maximum fraction of concept columns that can have the secondary feature index for it to have similar word detection applied
				auxiliaryNeuronsSimilarWordsSecondaryConceptFeaturesMinimumSharedSourceFeatureIndex = 1	#sparse: 3	#minimum number of concept columns that can have the secondary feature index for it to have similar word detection applied
		auxiliaryNeuronsSimilarSubwordAuto = True	#default: True
		if(auxiliaryNeurons and auxiliaryNeuronsSimilarSubwordAuto):
			auxiliaryNeuronsSimilarSubwordPrimeConceptFeatures = True	#find similar noun subwords
			auxiliaryNeuronsSimilarSubwordSecondaryConceptFeatures = True	#find similar non-noun subwords
			auxiliaryNeuronsSimilarSubwordAutoThreshold = 0.7
			auxiliaryNeuronsSimilarSubwordPrefixThreshold = 3	#in number of prefix characters that must be shared
			auxiliaryNeuronsSimilarSubwordSimilarityBatchSize = 256	#orig: 256
			auxiliaryNeuronsSimilarSubwordSecondaryConceptFeaturesIdentifySameColumn = True
			if(auxiliaryNeuronsSimilarSubwordPrimeConceptFeatures):
				auxiliaryNeuronsSimilarSubwordPrimeConceptFeaturesDatasetFileName = "auxiliaryNeuronsSimilarSubwordPrimeConceptFeaturesDataset.txt"
			if(auxiliaryNeuronsSimilarSubwordSecondaryConceptFeatures):
				auxiliaryNeuronsSimilarSubwordSecondaryConceptFeaturesDatasetFileName = "auxiliaryNeuronsSimilarSubwordSecondaryConceptFeaturesDataset.txt"
		auxiliaryNeuronsAutoFeatureDatasetFileWriteMode = "w"
		auxiliaryNeuronsAutoFeatureDatasetFileReadMode = "r"
		auxiliaryNeuronsAutoFeatureDatasetFileEncoding = "utf-8"
		auxiliaryNeuronsAutoFeatureDatasetLineTerminator = "\n"
		auxiliaryNeuronsSimilar = auxiliaryNeuronsSimilarWordsAuto or auxiliaryNeuronsSimilarSubwordAuto
		if(inferenceAddNewFeatures):
			auxiliaryNeuronsAutoInference = True	#default: True	#orig: False
		else:
			auxiliaryNeuronsAutoInference = False
	if(auxiliaryNeurons and auxiliaryNeuronsSimilar):
		auxiliaryNeuronsSimilarWordsFeatureNamePrefixPrimeConcept = "SIMC"
		auxiliaryNeuronsSimilarWordsFeatureNamePrefixSecondary = "SIMF"
		auxiliaryNeuronsSimilarWordsFeatureNamePrefixSubwordPrimeConcept = "SUBC"
		auxiliaryNeuronsSimilarWordsFeatureNamePrefixSubwordSecondary = "SUBF"
		auxiliaryNeuronsSimilarWordsFeatureNameDelimiter = ":"
		auxiliaryNeuronsSimilarWordsFeatureValueEmpty = ""
		auxiliaryNeuronsSimilarWordsScopedFeatureNameParts = 3
		auxiliaryNeuronsSimilarWordsConnectionProximityMultiplier = 10
		auxiliaryNeuronsSimilarWordsMinimumSimilarity = 0.0
		auxiliaryNeuronsSimilarWordsMaximumSimilarity = 1.0
		auxiliaryNeuronsSimilarWordsIdentitySimilarity = 1.0
		auxiliaryNeuronsAutoFeatureDatasetDelimiter = "\t"
		auxiliaryNeuronsAutoFeatureDatasetMinimumFields = 1
		auxiliaryNeuronsAutoFeatureDatasetSourceWordFieldIndex = 0
		auxiliaryNeuronsAutoFeatureDatasetSimilarWordStartFieldIndex = 1
		auxiliaryNeuronsAutoFeatureDatasetSimilarWordPairFields = 2
		auxiliaryNeuronsAutoFeatureDatasetSimilarWordOffset = 0
		auxiliaryNeuronsAutoFeatureDatasetSimilarityOffset = 1
		auxiliaryNeuronsAutoFeatureDatasetTempFileSuffix = ".tmp"
		auxiliaryNeuronsAutoFeatureDatasetSimilarityFormat = "{:.6f}"
		if(auxiliaryNeurons and auxiliaryNeuronsSimilarWordsAuto):
			auxiliaryNeuronsSimilarWordsThreshold = auxiliaryNeuronsSimilarWordsAutoThreshold
		auxiliaryNeuronsSimilarWordsFeaturesDictFileName = "auxiliarySimilarFeaturesDict.pkl"
		auxiliaryNeuronsSimilarWordsFeatureWordWeightsByParentWordFileName = "auxiliarySimilarFeatureWordWeightsByParentWord.pkl"
		auxiliaryNeuronsSimilarWordsConnectionsFolderName = "auxiliarySimilarFeatureConnections"
		auxiliaryNeuronsSimilarWordsSourceFeatureConnectionsFileNamePrefix = "simIndex"
	if(auxiliaryNeuronsAuto):
		auxiliaryNeuronsAutoReverseConnectionsFolderName = "reverseFeatureConnections"
		auxiliaryNeuronsAutoReverseTargetFeatureConnectionsFileNamePrefix = "revIndex"
else:
	trainReverseConnections = False


#Benchmarking filenames;
if(useBenchmark):
	#generate benchmark filename:
	#v2 benchmarks;
	if(inferenceReportGroundedAccuracy):
		benchmarkAblationText = "-inferenceReportGroundedAccuracy"
	elif(tokeniserSubword):
		if(tokeniserSubwordPOS):
			benchmarkAblationText = "-tokeniserSubwordPOS"
		else:
			benchmarkAblationText = "-tokeniserSubword"
		if(useDedicatedFeatureListsSubword):
			benchmarkAblationText += useDedicatedFeatureListsSubwordBenchmarkAblationSuffix
		if(useDedicatedConceptListsSubword):
			benchmarkAblationText += useDedicatedConceptListsSubwordBenchmarkAblationSuffix
	elif(auxiliaryNeurons):
		benchmarkAblationText = "-auxiliaryNeurons"
	elif(useTrainDuringInference):
		benchmarkAblationText = "-useTrainDuringInference"
	elif(trainVerifyConnectionNonexistentAcrossBranches):
		if(multipleDendriticBranchesBinaryTree):
			benchmarkAblationText = "-trainVerifyConnectionNonexistentAcrossBranches-multipleDendriticBranchesBinaryTreeDepthSelectMostConnectedRootBranches"
		else:
			benchmarkAblationText = "-trainVerifyConnectionNonexistentAcrossBranches"
	elif(multipleDendriticBranchesBinaryTree):
		benchmarkAblationText = "-multipleDendriticBranchesBinaryTree"	
	#v1 benchmarks;
	elif(multipleDendriticBranches and multipleDendriticBranchesRandom):
		if(spacyPipelineOptimisations):
			benchmarkAblationText = "-multipleDendriticBranchesRandom" + str(multipleDendriticBranchesNumber)
		else:
			printe("multipleDendriticBranchesRandom currently assumes spacyPipelineOptimisations")
	elif(multisentencePredictions):
		if(not useBenchmarkDefaults):
			benchmarkAblationText = "-multisentencePredictions"
		else:
			printe("multisentencePredictions currently assumes not useBenchmarkDefaults")
	elif(not useBenchmarkDefaults):
		benchmarkAblationText = "-useBenchmarkDefaultsFalse"
	elif(spacyPipelineOptimisations):
		benchmarkAblationText = "-spacyPipelineOptimisations"
	else:
		benchmarkAblationText = ""
	if(not sentencePredictions):
		sentencePredictionsFalseBenchmarkAblationSuffix = "-sentencePredictionsFalse"
		benchmarkAblationText = sentencePredictionsFalseBenchmarkAblationSuffix + benchmarkAblationText
	if(datasetType=="wikipedia"):
		databaseTypeText = ""	#or Wikipedia
	elif(datasetType=="oscar"):
		databaseTypeText = "Oscar"
	elif(datasetType in closedWorldGroundedDatasetTypes):
		databaseTypeText = closedWorldGroundedDatasetTypeToDatabaseTypeTextDict[datasetType]
		if(inferenceReportGroundedRealisticNLPmetric):
			databaseTypeText += closedWorldGroundedRealisticNLPmetricName
		elif(inferenceReportGroundedStrongerGroundedNLPmetric):
			databaseTypeText += closedWorldGroundedStrongerGroundedNLPmetricName
	databaseFolderExtension = databaseTypeText + str(trainMaxSequences) + "-numSeedTokensInference" + str(numSeedTokensInference) + benchmarkAblationText		#useSANIfeaturesAndColumns
elif(useAutoresearch):
	databaseFolderExtension = "Autoresearch"
elif(datasetType in closedWorldGroundedDatasetTypes):
	databaseFolderExtension = closedWorldGroundedDatasetTypeToDatabaseTypeTextDict[datasetType]
	if(inferenceReportGroundedRealisticNLPmetric):
		databaseFolderExtension += closedWorldGroundedRealisticNLPmetricName
	elif(inferenceReportGroundedStrongerGroundedNLPmetric):
		databaseFolderExtension += closedWorldGroundedStrongerGroundedNLPmetricName
else:
	databaseFolderExtension = ""
	

	
