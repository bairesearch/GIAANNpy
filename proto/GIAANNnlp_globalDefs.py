"""GIAANNnlp_globalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 Baxter AI (baxterai.com)

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
#from GIAANNcmn_globalDefs import multipleDendriticBranches
#from GIAANNcmn_globalDefs import numberOfDendriticBranches
from GIAANNcmn_globalDefs import useSANI
from GIAANNcmn_globalDefs import useInference
from GIAANNcmn_globalDefs import inferenceEvaluateTestSet
#useBenchmark dependencies:
from GIAANNcmn_globalDefs import multipleDendriticBranches
from GIAANNcmn_globalDefs import numberOfDendriticBranches
from GIAANNcmn_globalDefs import randomlyAssignBranches
from GIAANNcmn_globalDefs import trainMaxSequences
from GIAANNcmn_globalDefs import numSeedTokensInference


#Dataset Type;
if(useQuickExecution):
	datasetType = "textfile"
elif(useBenchmark):
	datasetType = "oscar"	#"oscar"/"wikipedia"
elif(useAutoresearch):
	datasetType = "oscar"
else:
	datasetType = "oscar"	#"oscar" / "wikipedia" / "textfile" [experimental]


#Multisentence predictions;
multisentencePredictions = False	#default: False	#each sequence comprises multiple sentences	#requires higher GPU RAM for train
if(multisentencePredictions):
	numSentencesPerSequence = 3 #default: 3
else:
	numSentencesPerSequence = 1


#Dataset;
datasetsLibrary4plus = False	#default: False	#orig: False	#set False during dev to maintain benchmark consistency
trainTestSet = False	#default: False	#only set True to generate an inference test set (with printTrainSequenceRaw=True)
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
else:
	printe("Dataset selection error: enable either datasetType==textfile or datasetType==oscar or datasetType==wikipedia")
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
			trainMaxSequencesEver = 10000000	#highest value of trainMaxSequences expected during current dev (using this instead of a much high value closer to 1-testSetRatio because testSetStartOffset takes time to load)
			numSentencesPerSequenceEver = 3
			datasetOscarAverageEligibleSentencesPerArticle = 32	#measured across 1m raw sentences (therefore appropriate for trainMaxSequencesEver=1m)
			testSetStartOffset = int(trainMaxSequencesEver / datasetOscarAverageEligibleSentencesPerArticle)*numSentencesPerSequenceEver
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
else:
	trainSetStartOffsetSequences = 0


#Benchmarking;
if(useBenchmark):
	useBenchmarkDefaults = True	#default: True
else:
	useBenchmarkDefaults = False	#default: False
if(useBenchmarkDefaults):
	spacyPipelineOptimisations = True	#default: True	#orig: False	#spacyPipelineOptimisations do not significantly affect test-set accuracies (~-0.002)
else:
	spacyPipelineOptimisations = True	#default: True
if(useBenchmark):
	#generate benchmark filename:
	if(multipleDendriticBranches and randomlyAssignBranches):
		if(spacyPipelineOptimisations):
			benchmarkAblationText = "-randomlyAssignBranches" + str(numberOfDendriticBranches)
		else:
			printe("randomlyAssignBranches currently assumes spacyPipelineOptimisations")
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
	if(datasetType=="wikipedia"):
		databaseTypeText = ""	#or Wikipedia
	elif(datasetType=="oscar"):
		databaseTypeText = "Oscar"
	databaseFolderExtension = databaseTypeText + str(trainMaxSequences) + "-numSeedTokensInference" + str(numSeedTokensInference) + benchmarkAblationText		#useSANIfeaturesAndColumns
else:
	databaseFolderExtension = ""


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
			if(useBenchmark):
				if(inferenceEvaluateTestSet):
					inferencePromptFileName = 'inference_prompt.txt.longTestWikipedia'
				else:
					inferencePromptFileName = 'inference_prompt.txt.longTrainWikipedia'	
			else:
				inferencePromptFileName = 'inference_prompt.txt'
		elif(datasetType=="oscar"):
			if(useBenchmark):
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
							printe("datasetType==oscar multisentencePredictions was trained with useBenchmarkDefaults=False")
						else:
							printe("datasetType==oscar multisentencePredictions was trained with useBenchmarkDefaults=False")
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
				inferencePromptFileName = 'inference_prompt.txt'
		elif(datasetType=="textfile"):
			#experimental (untested)
			trainPromptFileName = datasetName	#"train_prompt.txt"
			inferencePromptFileName = "inference_prompt.txt"
		else:
			printe("invalid datasetType")

posFolder = "POS/"
posDictFile = "everPos.wordnet.pkl.gz"


#POS;
useSpacyForConceptNounPOSdetection = True	#orig: True	#False: use GIAANNnlp_sequencePOS predetermined word-POS dictionaries for all pos detection (never use spacy dynamically assigned pos tags)
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


#Dedicated feature lists (non-dynamic);
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


	
