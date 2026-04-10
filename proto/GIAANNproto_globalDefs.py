"""GIAANNproto_globalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_globalDefs.py

# Usage:
see GIAANNproto_globalDefs.py

# Description:
GIA ANN proto global Defs

"""

import torch as pt
import math
import sys


#Recent debug vars;


#Execution mode selection;
useQuickExecution = True	#default: False	#orig: True
useBenchmark = False		#use benchmark file naming schemes and evals
useAutoresearch = False
#useDefault = True	#default: True
inferenceTrainFirstSequences = False	#dependent var
if(useQuickExecution):
	executionMode = "inference" 	#mandatory: "inference" (effective trainAndInference but uses a text datafile)
	inferenceTrainFirstSequences = True	#trains first sequences in inference_prompt.txt, performs inference only on last sequence
elif(useBenchmark):
	executionMode = "train"	#optional: "train/"inference"/"trainAndInference" 
elif(useAutoresearch):
	executionMode = "trainAndInference"
else:
	executionMode = "train"	#optional: "train/"inference"/"trainAndInference" 
	

#Primary Draw settings:
drawNetworkDuringTrain = False	#default: False  	#network drawing for prototype (not suitable for fast training)
drawNetworkDuringInference = False	#default: False


#Inference settings:
numSeedTokensInference = 8	#default: 5, 8, 12, 16	#this is also set during train phase only so that the derived numberOfSegments always matches inference phase
useInference = True  #mandatory: True	#enable options that support inference mode
if(useInference):
	inferenceUseNextTokenPredictionsOrTargetsToActivateNextColumnFeatures = False	#default: False	#orig: True	#True: activate next column features using current prediction; False: use current target (default top-1 accuracy measurement)
inferenceAddNewFeatures = True	#default: True	#orig: False	#run a controlled expansion pass during inference to add missing columns/features without training updates
if(useQuickExecution):
	useBenchmarkDefaultsEvalTestSet = False	#default: False: eval training-set
elif(useBenchmark):
	useBenchmarkDefaultsEvalTestSet = False	#default: False: eval training-set
elif(useAutoresearch):
	useBenchmarkDefaultsEvalTestSet = True	#default: True: eval test-set
else:
	useBenchmarkDefaultsEvalTestSet = True	#default: True: eval test-set
if(useBenchmarkDefaultsEvalTestSet):
	inferenceEvaluateTestSet = True
	#inferenceSegmentTiming = "none"	#~optimum
	inferenceSegmentTiming = "biased"	#default
	#inferenceSegmentTiming = "exact"
	#inferenceSegmentTiming = "seq"
	inferenceActivationsType = "boolf"	#default
	#inferenceActivationsType = "boolf+c"
	#inferenceActivationsType = "intf+c" 	#~optimum
else:
	inferenceEvaluateTestSet = False
	#inferenceSegmentTiming = "none"
	#inferenceSegmentTiming = "biased"
	inferenceSegmentTiming = "exact"		#default
	#inferenceSegmentTiming = "seq"
	inferenceActivationsType = "boolf"	#default
	#inferenceActivationsType = "boolf+c"
	#inferenceActivationsType = "intf+c"
inferenceReportTokenAccuracyConstrainByColumn = False	#default: False	#orig: False


#Dataset Type;
if(useQuickExecution):
	datasetType = "textfile"
elif(useBenchmark):
	datasetType = "oscar"	#"oscar"/"wikipedia"
elif(useAutoresearch):
	datasetType = "oscar"
else:
	datasetType = "wikipedia"	#"oscar" / "wikipedia" / "textfile" [experimental]


#Database;
if(useQuickExecution):
	trainMaxSequences = 10	#N/A: auto generated from inference_prompt.txt.trainAndInference
	databaseFolderBase = "../database"	#default: "../database/"
elif(useBenchmark):
	trainMaxSequences = 200000	#5000, 200000, 1000000
	databaseFolderBase = "/media/user/ssdpro/GIAANN/database"
elif(useAutoresearch):
	trainMaxSequences = 5000
	databaseFolderBase = "/media/user/ssdpro/GIAANN/database"
else:
	trainMaxSequences = 1000000	#dev: 10, 500, 5000, 10000, 200000 	#default: 1000000	  #adjust as needed	#max sequences for train
	databaseFolderBase = "../database"
maxSequenceLength = 80	#default:80	#orig:100		#in words	#depends on CPU/GPU RAM availability during train 
numberEpochs = 1	#default: 1


#Multisentence predictions;
multisentencePredictions = False	#default: False	#each sequence comprises multiple sentences	#requires higher GPU RAM for train
if(multisentencePredictions):
	numSentencesPerSequence = 3 #default: 3
else:
	numSentencesPerSequence = 1


#Dendritic branches;
multipleDendriticBranches = True	#default: True	#orig: False
if(multipleDendriticBranches):
	randomlyAssignBranches = False	#optional	#orig: False
	if(randomlyAssignBranches):
		numberOfDendriticBranches = 5
	else:
		numberOfDendriticBranches = 2	#default: 5, 2	#affects train+inference RAM
else:
	numberOfDendriticBranches = 1
	randomlyAssignBranches = False


#Error report;
def printe(str):
	raise RuntimeError(str)


#Dataset;
datasetsLibrary4plus = False	#default: False	#orig: False	#set False during dev to maintain benchmark consistency
trainTestSet = False	#default: False	#only set True to generate an inference test set (with printTrainSequenceRaw=True)
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
		datasetFolder = "../../dataset/"
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
			trainMaxSequencesEver = 1000000	#highest value of trainMaxSequences expected during current dev (using this instead of a much high value closer to 1-testSetRatio because testSetStartOffset takes time to load)
			numSentencesPerSequenceEver = 3
			datasetOscarAverageEligibleSentencesPerArticle = 32	#measured across 1m raw sentences (therefore appropriate for trainMaxSequencesEver=1m)
			testSetStartOffset = int(trainMaxSequencesEver / datasetOscarAverageEligibleSentencesPerArticle)*numSentencesPerSequenceEver
			testSetSize = 1000	#number of entries to include in test set
		else:
			printe("trainTestSet configuration error: unsupported dataset selection")
		trainSetStartOffsetSequences = 0
	else:
		trainSetStartOffsetSequences = 0	#200000	#1000000	#default: 0	#orig: 0	
		if(datasetType=="oscar"):
			maxSentencesPerArticle = 100	#CHECKTHIS
		elif(datasetType=="wikipedia"):
			maxSentencesPerArticle = 1000	#CHECKTHIS
else:
	trainSetStartOffsetSequences = 0


#RAM;
useGPUdense = True	#default: True
if(executionMode=="inference" or executionMode=="trainAndInference"):
	useGPUsparse = False	#default: False	#orig: True	#inference requires high RAM to store sparse tensors
elif(executionMode=="train"):
	useGPUsparse = True	#default: True		#slight performance increase during train (does not use significant additional GPU ram during train)
useGPUsparseStrict = True	#default: True	#orig: False	#enforce strict sparse device during transfer to/from dense tensors
runtimeReleaseGPUMemory = False	#default: True	#aggressively release cached CUDA memory after sequence processing
runtimeReleaseGPUMemoryEverySequenceCount = 1	#default: 1	#only apply release every N processed sequences
if(runtimeReleaseGPUMemory):
	assert runtimeReleaseGPUMemoryEverySequenceCount > 0, "runtimeReleaseGPUMemoryEverySequenceCount must be > 0"
useGPUfileio = False	#default: useGPUsparse	#orig: useGPUsparse
storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam = True	#default: True	#orig: False
if(storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam):
	useGPUdatabase = False	#default: False	#default: False
	resizeTensorsOnRAMdatabaseSave = False	#default: False #orig: True	#resize all feature neuron and connections tensors during final RAM database save
	resizeTensorsOnRAMdatabaseLoad = False	#default: False #orig: True	#resize all feature neuron and connections tensors during initial RAM database load
trainEndGenerateGlobalFeatureNeuronsTensor = False	#derived var
inferenceStartGenerateGlobalFeatureNeuronsTensor = False	#derived var
if(executionMode=="train"):
	storeDatabaseGlobalFeatureNeuronsInRam = True		 #default: True	#orig: False	 #currently required to be True for inference compatibility	#optional
	if(not storeDatabaseGlobalFeatureNeuronsInRam):
		trainEndGenerateGlobalFeatureNeuronsTensor = False	#default: False	#orig: False	#generates and saves a globalFeatureNeurons at the end of training (not just individual column featureNeurons tensors)
	trainSparseConnectionsTensor = False	#default: False	#orig: False
	trainSparseNeuronsTensor = False	#default: False	#orig: False
	if(trainSparseNeuronsTensor):
		assert trainSparseConnectionsTensor, "trainSparseNeuronsTensor requires trainSparseConnectionsTensor=True"
else:
	storeDatabaseGlobalFeatureNeuronsInRam = True		#mandatory: True	#if storeDatabaseGlobalFeatureNeuronsInRam=True use global feature neuron tensors, else use feature neuron tensors in observed columns (note feature connection tensors are always in observed columns)
	if(executionMode=="inference"):
		inferenceStartGenerateGlobalFeatureNeuronsTensor = False	#default: False	#orig: False	#generates and saves a globalFeatureNeurons at the start of inference after being trained with storeDatabaseGlobalFeatureNeuronsInRam=False (with individual column featureNeurons tensors)
	trainSparseConnectionsTensor = False	#default for executionMode=="trainAndInference"
	trainSparseNeuronsTensor = False	#default for executionMode=="trainAndInference"


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
	databaseFolder = databaseFolderBase + databaseTypeText + str(trainMaxSequences) + "-numSeedTokensInference" + str(numSeedTokensInference) + benchmarkAblationText		#useSANIfeaturesAndColumns
else:
	databaseFolder = databaseFolderBase
databaseFolder = databaseFolder + "/"


#Optimisations;
inferenceOnlyRetainPredictedTargetObservedColumn = False	#default: False	#orig: False	#load/evict one observed column per prediction step	#the majority of inference memory is the sparse global activation tensors (not the observed column connections)
inferenceOnlyRetainPredictedTargetObservedColumnBeamSearch = False	#default: False	#orig: False	#True: retain only current beam-search target(s); False: retain all beam-search targets	#the majority of inference memory is the sparse global activation tensors (not the observed column connections)
trainStoreFeatureMapsGlobally = True	#default: True	#orig: False	#True: avoid per-column persistence of global feature index maps; False: preserve legacy per-column map persistence
if not trainStoreFeatureMapsGlobally:
	assert not storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam


#Inference segment activation times;
if(useInference):
	if(inferenceSegmentTiming=="none"):
		inferenceUseNeuronFeaturePropertiesTime = False
		inferenceUseNeuronFeaturePropertiesTimeExact = False
	elif(inferenceSegmentTiming=="biased"):
		inferenceUseNeuronFeaturePropertiesTime = True
		inferenceUseNeuronFeaturePropertiesTimeExact = False	
	elif(inferenceSegmentTiming=="exact"):
		inferenceUseNeuronFeaturePropertiesTime = True
		inferenceUseNeuronFeaturePropertiesTimeExact = True
	elif(inferenceSegmentTiming=="seq"):
		inferenceUseNeuronFeaturePropertiesTime = True	#default: True	#orig: True
		inferenceUseNeuronFeaturePropertiesTimeExact = False
	else:
		printe("inferenceSegmentTiming error")


#Array properties (disable to optimise train speed/RAM during train);
arrayIndexPropertiesEfficient = True	#default: True	#orig: False (required for drawing pos types)
if(arrayIndexPropertiesEfficient):
	arrayIndexPropertiesStrength = True
	arrayIndexPropertiesPermanence = False
	arrayIndexPropertiesActivation = False	#inference only (see arrayIndexPropertiesActivationCreate)
	arrayIndexPropertiesTime = False	#inference only (see arrayIndexPropertiesTimeCreate)
	arrayIndexPropertiesPos = False
else:
	arrayIndexPropertiesStrength = True	#default: True
	arrayIndexPropertiesPermanence = True	#default: True
	arrayIndexPropertiesActivation = False	#default: False	#inference only (see arrayIndexPropertiesActivationCreate)
	arrayIndexPropertiesTime = False 	#default: False	#inference only (see arrayIndexPropertiesTimeCreate)
	arrayIndexPropertiesPos = True	#default: True
arrayIndexPropertiesActivationCreateInference = arrayIndexPropertiesActivation or True
arrayIndexPropertiesTimeCreateInference = arrayIndexPropertiesTime or inferenceUseNeuronFeaturePropertiesTime

#SANI;
useSANI = True	#default: True	#orig: False	#sequentially activated neuronal input


#Immediate (direct) connections;
enforceDirectConnections = True	#default: True	#orig: False	#prediction requires a direct connection from previous prediction as observed during training (ie adjacent tokens)
if(enforceDirectConnections):
	enforceDirectConnectionsSANI = True	#default: True #orig: False	#enforce activation of first segment (direct feature connection)
	enforceDirectConnectionsIgnoreSeed = True	#default: True #orig: False
else:
	enforceDirectConnectionsSANI = False
	enforceDirectConnectionsIgnoreSeed = False
if(enforceDirectConnectionsSANI):
	useSANI = True	#sequentially activated neuronal input (divide dendrites into segments)	#override
	enforceDirectConnectionsSANIminimal = False	#default: False	#orig: True	#deprecated
else:
	enforceDirectConnectionsSANIminimal = False
minimumPredictionActivationThreshold = 0.0	#explicit threshold application not required (for verification only)
predictionEnsureConnectedToPreviousPrediction = True	#default: True	#ensure every new prediction connects to previous node (requirement is independent of enforceDirectConnections)


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


#Beam search;
if(useInference):
	inferenceBeamSearch = False	#default: False	#orig: False
	inferenceBeamScoreStrategy = "nodeActivation"	#options: "nodeActivation", "activation_connection", "connection"
	inferenceBeamConceptColumnNodeActivationThreshold = 0.0
	inferenceBeamInstanceNodeActivationThreshold = 0.0
	inferenceBeamInstancePreferActiveNodeCounts = False		  #optional: prioritise columns with more active nodes (count-based)
	inferenceBeamInstancePreferInternalConnectivity = False      #optional: prioritise columns with stronger internal connectivity between active nodes
	inferenceBeamInstancePreferAdjacentOverlap = False           #optional: prioritise columns sharing active features with adjacent columns
	inferenceBeamWidth = 3	#default: 3	#orig: 3
	inferenceBeamDepth = 6	#default: 3	#orig: 6
	#optimisations;
	inferenceStrengthLookupCache = True	#default: True	#orig: False	#cache strength lookup during inference
	if(inferenceBeamSearch):
		inferenceStrengthLookupBypass = False	#mandatory: False
	else:
		inferenceStrengthLookupBypass = True	#default: True	#bypass cache strength lookup during inference
	if(inferenceStrengthLookupBypass):
		assert not inferenceBeamSearch, "beamSearchSelectSingleStepFeature: inferenceStrengthLookupBypass requires inferenceBeamSearch=False"
		assert inferenceBeamScoreStrategy == "nodeActivation", "beamSearchSelectSingleStepFeature: inferenceStrengthLookupBypass requires inferenceBeamScoreStrategy='nodeActivation'"
		assert not inferenceBeamInstancePreferInternalConnectivity, "beamSearchSelectSingleStepFeature: inferenceStrengthLookupBypass requires connectivity inferenceBeamInstancePreferInternalConnectivity disabled"
		assert not inferenceBeamInstancePreferAdjacentOverlap, "beamSearchSelectSingleStepFeature: inferenceStrengthLookupBypass requires connectivity inferenceBeamInstancePreferAdjacentOverlap disabled"


#Inference activations;
if(useInference):
	inferenceActivationFunction = True	#default:True	#orig:False	#required to prevent exponential runaway of activations (that negatively affects predictionNetwork loss optimisation)
	if(useSANI):
		inferenceApplySequentialActivationSparse = True	#default: True	#orig: False
		inferenceConnectionsStrengthBoolean = True	#default: True	#do not overweight by common features (e.g. determiners)
		if(inferenceActivationsType == "intf+c"):
			inferenceSegmentActivationsBoolean = False
		elif(inferenceActivationsType == "boolf"):
			inferenceSegmentActivationsBoolean = True
			inferenceSegmentActivationsBooleanFeatureSegmentsOnly = True
		elif(inferenceActivationsType == "boolf+c"):
			inferenceSegmentActivationsBoolean = True
			inferenceSegmentActivationsBooleanFeatureSegmentsOnly = False
		else:
			printe("inferenceActivationsType error")
		inferenceSourceActivationsBoolean = True	#default: True (do not sum SANI segments)	#orig: False
	else:
		inferenceConnectionsStrengthBoolean = False	#default: False
		inferenceSegmentActivationsBoolean = False	#default: False	
		inferenceSourceActivationsBoolean = True	#default: True	#orig: False (theoretically effectively True)
	inferenceSeedNetwork = True	#default: True
	

#Train optimisations;
trainSequenceObservedColumnsUseSequenceFeaturesOnly = True	#default:True	#optional	#sequence observed columns arrays only store sequence features.	#will affect which network changes can be visualised
#trainSequenceObservedColumnsUseSequenceFeaturesOnly can be upgraded so only a limited amount of data is ever loaded to GPU during train (it currently temporarily masks entire feature arrays in GPU during transfer phase)
trainSequenceObservedColumnsMatchSequenceWords = True	#mantatory		#introduced GIAANNproto1b12a; more robust method for training (independently train each instance of a concept in a sequence)	#False: not robust as there may be less concept columns than concepts referenced in sequence (if multiple references to the same column)	
optimisationCombineSparseUpdatesPerSequence = True	#default: True	#orig: False	#updateObservedColumnsEfficient combines sparse updates per sequence instead of per column (reduces calls to coalesce) 
optimisationUseCUDAObservedColumnUpdateKernel = False	#default: False	#use custom CUDA sparse accumulator for updateObservedColumnsEfficient strength updates
if(optimisationUseCUDAObservedColumnUpdateKernel):
	assert useGPUsparse, "optimisationUseCUDAObservedColumnUpdateKernel requires useGPUsparse=True"
	assert useGPUsparseStrict, "optimisationUseCUDAObservedColumnUpdateKernel requires useGPUsparseStrict=True"
optimisationGetTrainRequiredSourceFeatureIndicesByObservedColumnVectorize = True	#default: True	#orig: False	#vectorise exact per-column source-feature detection for trainSequenceObservedColumnsUseSequenceFeaturesOnly/trainSequenceObservedColumnsMatchSequenceWords
optimisationGetFeatureConnectionsForSourceFeatureCache = False 	#default: not storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam	#orig: False	#cache stored source-feature file indices per observed column to avoid repeated directory scans when storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam=False
optimisationNormaliseSourceFeatureIndicesDisabled = False	#default: False	#orig: False
optimisationObservedColumnsWriteMetadataCheck = False	#default: False, orig: False
optimisationArrayIndexPropertiesEfficientSerialConnections = False	#default: False #orig: True	#uses less GPU RAM
optimisationArrayIndexPropertiesEfficientSerialNeurons = False	#default: False #orig: False


#Draw;
#select a single draw method (colouring scheme);
drawSegments = False and useSANI	#optional
drawBranches = False and multipleDendriticBranches	#optional
drawRelationTypes = False and not arrayIndexPropertiesEfficient	#optional
drawDelimiters = False	#optional
drawDefault = True	#optional
drawNetworkDuringTrainSave = False	#default: False	#save drawn network during train
drawNetworkDuringInferenceSave = False	#True is only for debug
if(executionMode=="inference" or executionMode=="trainAndInference"):
	drawSequenceObservedColumns = False	#mandatory
	drawAllColumns = False	#mandatory
else:
	drawSequenceObservedColumns = False	#default: False	#optional	#draw sequence observed columns (instead of complete observed columns)	#note if !drawSequenceObservedColumns and !trainSequenceObservedColumnsUseSequenceFeaturesOnly, then will still draw complete columns	#optional (will affect which network changes can be visualised)
	drawAllColumns = False	#default: False	#optional	#draw all columns in network (only used for automated visualisation; drawNetworkDuringTrainSave)	#requires !drawSequenceObservedColumns
	if(drawAllColumns):
		assert not trainSequenceObservedColumnsUseSequenceFeaturesOnly
drawNetworkSaveFormatVector = True	#default: False	#orig: False	#True: save matplotlib network images as svg instead of png
drawSegmentsTrain = False	#derived
drawSegmentsInference = False	#derived
drawBranchesTrain = False	#derived
drawBranchesInference = False	#derived
drawRelationTypesTrain = False	#derived
drawRelationTypesInference = False	#derived
drawDelimitersTrain = False	#derived
drawDelimitersInference = False	#derived
drawDefaultTrain = False	#derived
drawDefaultInference = False	#derived
if(drawSegments):
	drawSegmentsTrain = True 	#draws connection colours based on their target node incoming segment index
	drawSegmentsInference = False	#False: draw activation status
elif(drawBranches):
	drawBranchesTrain = True 	#draws connection colours based on their target node incoming dendritic branch index
	drawBranchesInference = False 	#False: draw activation status
elif(drawRelationTypes):
	drawRelationTypesTrain = True	#draws feature neuron and connection relation types in different colours
	drawRelationTypesInference = False	#False: draw activation status
elif(drawDelimiters):
	drawDelimitersTrain = True	#draws feature neuron column delimiters (and their external connections) in different colours
	drawDelimitersInference = False		#False: draw activation status
elif(drawDefault):
	drawDefaultTrain = True	#standard colours (prime concept feature neurons in blue and instance feature neurons in cyan)
	drawDefaultInference = True	#False: draw activation status
else:
	printe("warning: draw scheme not defined")
drawNetworkDuringTrainSaveFilenamePrepend = "GIAANNproto1cAllColumnsTrainSequenceIndex"
drawNetworkDuringInferenceSaveFilenamePrepend = "GIAANNproto1cSequenceObservedColumnsInferenceTokenIndex"
drawHighResolutionFigure = True	#required for inference debug
ignoreNewlineCharacters = True
drawSparseArrays = False	#default: False	#orig: False	#can draw sequences contained within much larger databases without running out of memory (due to densifying arrays)
if(trainSequenceObservedColumnsMatchSequenceWords):
	#sumChangesToConceptNeuronSequenceInstances = True	#mandatory	#for multiple instances of concept in sequence, need to take the sum of the changes between the existing and modified arrays for each instance of a same concept in the sequence
	assert not drawSequenceObservedColumns, "trainSequenceObservedColumnsMatchSequenceWords does not currently support drawSequenceObservedColumns; requires concept_name_to_index (i.e. one-to-one mapping between name and feature index in SequenceObservedColumns arrays) etc"
drawRelationTypesConnectionsFromSource = True	#orig: False	#required for useSANIfeaturesAndColumns only


#Algorithm preferences (normalisation, permanence etc);
inferenceNormaliseColumnSelectionByFeatureConnections = False  	#default: False		#cannot select one column over another if column activations are perfectly normalised with respect to each other	#see HFconnectionMatrixAlgorithmNormalise
if(inferenceNormaliseColumnSelectionByFeatureConnections):
	inferenceNormaliseColumnSelectionByFeatureConnectionsStrength = False	#else normalise column selection by number connections
inferenceNormaliseFeatureSelectionByFeatureConnections = False	#default: False
if(inferenceNormaliseFeatureSelectionByFeatureConnections):
	inferenceNormaliseFeatureSelectionByFeatureConnectionsStrength = True	#mandatory
trainConnectionStrengthNormaliseWrtContextLength = True	#default: True
trainDecreasePermanenceOfInactiveFeatureNeuronsAndConnections = False	#default: True
trainConnectionStrengthIncreaseColumnInternal = False	#not required as internal column nodes will be predicted unless current node is a reference set delimiter

if(trainConnectionStrengthIncreaseColumnInternal):
 	trainIncreaseColumnInternalConnectionsStrengthModifier = 10.0
# Define constants for permanence and activation trace	#TODO: train hyperparameters
z1 = 3  # Initial permanence value	
z2 = 1  # Decrement value when not activated
if(inferenceActivationFunction):
	j1 = 1 #default: 1
else:
	j1 = 5   # Activation trace duration


#Mandatory vars;
useSaveData = True	#save data is required to allow consecutive sequence training and inference (because connection data are stored in observed columns, which are refreshed every sequence)
usePOS = True		 # usePOS mode	#mandatory
useParallelProcessing = True	#mandatory (else restore original code pre-GIAANNproto1b3a)
randomiseColumnFeatureXposition = True	#shuffle x position of column internal features such that their connections can be better visualised


#Information vars;
printTimeDatabaseLoadSaveTimes = False
printRamMaxUsage = False

printCountTotalParameters = False	#count number of connections in network
printInferenceTop1Accuracy = True	#default: True	#print inference top-1 accuracy
printInferenceTop1AccuracyBitsPerByte = False	#dependent var
printInferenceTop1AccuracyBitsPerByteModified = False	#dependent var
if(printInferenceTop1Accuracy):
	printInferenceTop1AccuracyBitsPerByte = False	#default: False	#print inference top-1 accuracy in BPB
	if(printInferenceTop1AccuracyBitsPerByte):
		assert not inferenceUseNextTokenPredictionsOrTargetsToActivateNextColumnFeatures, "printInferenceTop1AccuracyBitsPerByte requires inferenceUseNextTokenPredictionsOrTargetsToActivateNextColumnFeatures=False"
		assert not inferenceBeamSearch, "printInferenceTop1AccuracyBitsPerByte requires inferenceBeamSearch=False"
		if(printInferenceTop1AccuracyBitsPerByte):
			printInferenceTop1AccuracyBitsPerByteModified = True	#default: True	#print inference top-1 accuracy in modified BPB

if(useAutoresearch):
	printTotalFeatures = False
	printConfiguration = True
	printHeaderDuringInferencePredict = False
	printPredictionsDuringInferencePredict = False
else:
	printTotalFeatures = True	#print c+f upon load
	printConfiguration = True	#print common global defs configuration
	printHeaderDuringInferencePredict = True
	printPredictionsDuringInferencePredict = True
printPredictionsDuringInferencePredictBeamSearch = False
printTrainSequenceDefault = False	#default: True	#orig: True
printTrainSequenceRaw = False	#print each training sequence raw text (suitable for inference_prompt.txt generation)
printTrainSequenceConceptAssignment = False	#print each training sequence split by column assignment
printTrainSequenceConceptAssignmentByLine = False	#display each column on a new line
printTrainSequenceDelimiters = False	#print each training sequence with delimiters
printTrainSequencePOS = False	#print each training sequence with POS tags
printTrainSequenceCount = False	#print each training sequence count
if(not useAutoresearch):
	if(datasetType=="oscar"):
		printTrainSequenceCount = True	#non-visible characters affect terminal print consistency
		#printTrainSequenceRaw = True
	elif(datasetType=="wikipedia"):
		printTrainSequenceDefault = True
		

#Debug vars;
debugPrintTrainSectionTimes = False	#print per-sequence timing breakdown for key train sections
debugPrintTrainSectionTimesSourceFeatureConnections = False	#print granular source-feature-connection timings within updateObservedColumnsEfficient
debugPrintRamCurrentUsage = False
debugPrintRamAverageUsage = False
debugPrintRamMaxUsagePhaseLocal = False

debugPrintTotalInferenceTokens = False	#print total number of inference tokens in seed phase, prediction phase, and both phases (summed across all sequences) 
debugPrintSpacySectionTimes = False	#print spacy preprocessing times

if(useAutoresearch):
	debugWarningInferenceOnConnectivityError = False
	debugWarningInferenceOnPredictionTargetMismatch = False
	debugWarningInferenceNoDelimiterDetectedBetweenConceptTokens = False
else:
	debugWarningInferenceOnConnectivityError = True
	debugWarningInferenceOnPredictionTargetMismatch = True
	debugWarningInferenceNoDelimiterDetectedBetweenConceptTokens = True
debugTerminateInferenceOnPredictionTargetMismatch = False
debugTerminateInferenceOnNoPredictionCandidatesAvailable = False
debugTerminateOnConceptColumnsDelimitByPOSwarning = False
if(not useAutoresearch):
	if(not printTrainSequenceRaw):
		debugTerminateOnConceptColumnsDelimitByPOSwarning = False	#default: True
if(pretrainConceptColumnsDelimitByPOSenforce):
	debugTerminateOnConceptColumnsDelimitByPOSerror = False
else:
	debugTerminateOnConceptColumnsDelimitByPOSerror = False

debugDeleteGPUcache = False

debugLimitFeatures = False	#can be used to recover database for inference if run out of ram during training
if(debugLimitFeatures):
	debugLimitFeaturesCMax = 207910
	debugLimitFeaturesFMax = 50089

debugOnlyDrawBranchIndexConnections = False
debugOnlyDrawBranchIndexX = 0

debugConnectNodesToNextNodesInSequenceOnly = False
debugConnectColumnsToNextColumnsInSequenceOnly = False
debugDrawNeuronActivations = True
debugReloadGlobalFeatureNeuronsEverySequence = False


#Concept/feature names;
useDedicatedFeatureLists = False	#default: False - dynamically learn concept features	#True: use static feature lists (depreciated)
#if usePOS and storeDatabaseGlobalFeatureNeuronsInRam:
#	useDedicatedFeatureLists = True
useDedicatedConceptNames = False
useDedicatedConceptNames2 = False
if usePOS:
	useDedicatedConceptNames = True
	if(useDedicatedConceptNames):
		#same word can have different pos making it classed as an instance feature or prime concept feature
		useDedicatedConceptNames2 = True	#mandatory
#if usePOS: same word can have different pos making it classed as an instance feature or prime concept feature
variablePrimeConceptFeatureNeuronName = "variablePrimeConceptFeatureNeuron"
variablePrimeConceptFeatureNeuronNameAbbreviation = "VPCFN"
featureIndexPrimeConceptNeuron = 0


#Inference prediction selection;
if(useInference):
	inferenceDeactivateNeuronsUponPrediction = True	#default: True
	inferenceDecrementActivations = False	#default: False - CHECKTHIS #orig: False
	if(inferenceDecrementActivations):
		inferenceDecrementActivationsNonlinear = True
		activationDecrementPerPredictedToken = 0.1	#0.05	#CHECKTHIS
		activationDecrementPerPredictedSequence = 0.5
	if(inferenceSeedNetwork):
		numSeedTokens = numSeedTokensInference	#default: 5	#number of seed tokens in last sequence of inference prompt (remaining tokens will be prediction tokens)
	else:
		numSeedTokens = 0
	kcNetwork = 1	#number of topk columns to target
	kcMax = 1 	#topk next concept column prediction
	kf = 1
	assert kcNetwork==1 and kf==1 and kcMax==1, "multiple prediction column/feature pairs not supported"
	if(executionMode == "inference"):
		assert storeDatabaseGlobalFeatureNeuronsInRam, "useInference: global feature neuron lists are required" 
	assert useSaveData,  "useInference: useSaveData is required" 
		

#Database save paths;
conceptColumnsDictFile = databaseFolder + 'conceptColumnsDict.pkl'
conceptFeaturesDictFile = databaseFolder + 'conceptFeaturesDict.pkl'
observedColumnsFolderName = 'observedColumns'
observedColumnsDir = databaseFolder + observedColumnsFolderName
observedColumnFolderNamePrefix = 'cIndex'
observedColumnMetadataFileName = 'data.pkl'
observedColumnFeatureConnectionsFolderName = 'featureConnections'
observedColumnSourceFeatureConnectionsFileNamePrefix = 'fIndex'
observedColumnFeatureNeuronsTensorName = 'featureNeurons'
observedColumnLegacyMetadataFileSuffix = '_data.pkl'
observedColumnLegacyFeatureConnectionsTensorNameSuffix = '_featureConnections'
observedColumnLegacyFeatureNeuronsTensorNameSuffix = '_featureNeurons'
observedColumnFeatureConnectionsFormat = 'bySourceFeature.v1'
pytorchTensorFileExtension = ".pt"
if(useInference):
	if(useQuickExecution):
		if(datasetType=="textfile"):
			inferencePromptFileName = "inference_prompt.txt.trainAndInference"
		else:
			printe("useQuickExecution requires datasetType==textfile")
	else:
		if(datasetType=="wikipedia"):
			if(inferenceEvaluateTestSet):
				inferencePromptFileName = 'inference_prompt.txt.longTestWikipedia'
			else:
				inferencePromptFileName = 'inference_prompt.txt.longTrainWikipedia'	
		elif(datasetType=="oscar"):
			if(multisentencePredictions):
				if(inferenceEvaluateTestSet):
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
						
		elif(datasetType=="textfile"):
			#experimental (untested)
			trainPromptFileName = datasetName	#"train_prompt.txt"
			trainPromptFileName = databaseFolder + trainPromptFileName
			inferencePromptFileName = "inference_prompt.txt"
		else:
			printe("invalid datasetType")
	inferencePromptFile = databaseFolder + inferencePromptFileName
if(conceptColumnsDelimitByPOS):
	if(detectReferenceSetDelimitersBetweenNouns):
		conceptFeaturesReferenceSetDelimiterDeterministicListFile = databaseFolder + 'conceptFeaturesReferenceSetDelimiterDeterministicList.pkl'
		conceptFeaturesReferenceSetDelimiterProbabilisticListFile = databaseFolder + 'conceptFeaturesReferenceSetDelimiterProbabilisticList.pkl'
	else:
		conceptFeaturesReferenceSetDelimiterListFile = databaseFolder + 'conceptFeaturesReferenceSetDelimiterList.pkl'
globalFeatureNeuronsFile = 'globalFeatureNeurons'
globalFeatureNeuronsFileFull = databaseFolder + globalFeatureNeuronsFile + pytorchTensorFileExtension
posFolder = databaseFolder + "POS/"
posDictFile = "everPos.wordnet.pkl.gz"
saveGlobalFeatureNeuronsRate = 1000


#Common array indices;
arrayIndexPropertiesStrengthIndexTrain = None
arrayIndexPropertiesPermanenceIndexTrain = None
arrayIndexPropertiesActivationIndexTrain = None
arrayIndexPropertiesTimeIndexTrain = None
arrayIndexPropertiesPosIndexTrain = None
arrayIndexPropertiesStrengthIndexInference = None
arrayIndexPropertiesPermanenceIndexInference = None
arrayIndexPropertiesActivationIndexInference = None
arrayIndexPropertiesTimeIndexInference = None
arrayIndexPropertiesPosIndexInference = None
arrayPropertiesListTrain = []
if(arrayIndexPropertiesStrength):
	arrayIndexPropertiesStrengthIndexTrain = len(arrayPropertiesListTrain)
	arrayPropertiesListTrain.append(arrayIndexPropertiesStrengthIndexTrain)
if(arrayIndexPropertiesPermanence):
	arrayIndexPropertiesPermanenceIndexTrain = len(arrayPropertiesListTrain)
	arrayPropertiesListTrain.append(arrayIndexPropertiesPermanenceIndexTrain)
if(arrayIndexPropertiesActivation):
	arrayIndexPropertiesActivationIndexTrain = len(arrayPropertiesListTrain)
	arrayPropertiesListTrain.append(arrayIndexPropertiesActivationIndexTrain)
if(arrayIndexPropertiesTime):
	arrayIndexPropertiesTimeIndexTrain = len(arrayPropertiesListTrain)
	arrayPropertiesListTrain.append(arrayIndexPropertiesTimeIndexTrain)
if(arrayIndexPropertiesPos):
	arrayIndexPropertiesPosIndexTrain = len(arrayPropertiesListTrain)
	arrayPropertiesListTrain.append(arrayIndexPropertiesPosIndexTrain)
arrayNumberOfPropertiesTrain = len(arrayPropertiesListTrain)
arrayPropertiesListInference = []
if(arrayIndexPropertiesStrength):
	arrayIndexPropertiesStrengthIndexInference = len(arrayPropertiesListInference)
	arrayPropertiesListInference.append(arrayIndexPropertiesStrengthIndexInference)
if(arrayIndexPropertiesPermanence):
	arrayIndexPropertiesPermanenceIndexInference = len(arrayPropertiesListInference)
	arrayPropertiesListInference.append(arrayIndexPropertiesPermanenceIndexInference)
if(arrayIndexPropertiesActivationCreateInference):
	arrayIndexPropertiesActivationIndexInference = len(arrayPropertiesListInference)
	arrayPropertiesListInference.append(arrayIndexPropertiesActivationIndexInference)
if(arrayIndexPropertiesTimeCreateInference):
	arrayIndexPropertiesTimeIndexInference = len(arrayPropertiesListInference)
	arrayPropertiesListInference.append(arrayIndexPropertiesTimeIndexInference)
if(arrayIndexPropertiesPos):
	arrayIndexPropertiesPosIndexInference = len(arrayPropertiesListInference)
	arrayPropertiesListInference.append(arrayIndexPropertiesPosIndexInference)
arrayNumberOfPropertiesInference = len(arrayPropertiesListInference)


#SANI settings;
arrayIndexSegmentFirst = 0
if(useSANI):
	SANIfeaturesLinkFirstSegmentToAllPriorTrainSeqTokens = True	#default: True	#orig: True	#first feature segment captures all prior train sequence tokens
	if(enforceDirectConnectionsSANIminimal):
		useSANIcolumns = False
		useSANIfeatures = True
		useSANIfeaturesAndColumns = False
	else:
		useSANIcolumns = False	#assign segments by concept column proximity to connection target during train (includes internal concept column)
		useSANIfeatures = False #assign segments by feature proximity to connection target during train
		useSANIfeaturesAndColumns = True	#assign segments by column proximity first then feature proximity

	if(useSANIfeaturesAndColumns):
		useSANIfeaturesAndColumnsInternal = True	#default: True	#orig: False	#also include internal columns in column segments (not just external columns)
		#these are highly dependent on numSeedTokensInference and the specific seed text (ie number of features per column);
		arrayNumberOfSegmentsColumnDistance = math.floor(numSeedTokensInference / 4) + 1	#orig: + 1	#min number of concept/column segments (if useSANIfeaturesAndColumnsInternal, includes internal column segment)
		arrayNumberOfSegmentsFeatureDistance = math.ceil(numSeedTokensInference / 2) + 1 	#number of nearest features to target node
		arrayNumberOfSegments = arrayNumberOfSegmentsColumnDistance + arrayNumberOfSegmentsFeatureDistance
	elif(useSANIcolumns):
		if(multisentencePredictions):
			arrayNumberOfSegments = arrayNumberOfSegmentsColumnDistance = math.floor(numSeedTokensInference / 4) + 1	#orig: + 1	#* numSentencesPerSequence 	#default: 5	#min number of external and internal column connections to target node (note first segment captures all other external columns)
		else:
			arrayNumberOfSegments = arrayNumberOfSegmentsColumnDistance = math.floor(numSeedTokensInference / 4) + 1	#orig: + 1	#default: 2	#orig:3		#min number of external and internal column connections to target node (note first segment captures all other external columns)
				#max number of SANI segments per sequence (= max number of concept columns per sequence - 1)
				#note if arrayNumberOfSegments=3 then;	sIndex=2: sequential segment connections for current column, sIndex=1: adjacent column connections, sIndex=0: all other column connections
				#must be less than the (total number of concepts in a sequence - total number of concepts in effective predictive seed sequence)
	elif(useSANIfeatures):
		if(enforceDirectConnectionsSANIminimal):
			arrayNumberOfSegments = 2
		else:
			arrayNumberOfSegments = numSeedTokensInference	#default: 5	#min number of nearest features to target node (note first segment captures all other features)
			#arrayNumberOfSegments = math.ceil(numSeedTokensInference / 2) + 1	#temp for benchmarking compared to useSANIfeaturesAndColumns/useSANIcolumns [remove this]
	
	algorithmMatrixSANImethod="enforceActivationAcrossSegments"	#default	#only activate a segment under conditions		
	#algorithmMatrixSANImethod="doNotEnforceActivationAcrossSegments"	#orig	#activate segments without any sequentiality requirement	simply addActivationAcrossSegments	#equivalent to !useSANI
	if(algorithmMatrixSANImethod=="enforceActivationAcrossSegments"):
		#algorithmMatrixSANIenforceRequirement="enforceAnySegmentMustBeActive"	#activate neuron if any segment is active
		algorithmMatrixSANIenforceRequirement="enforceLastSegmentMustBeActive"	#default	#only activate neuron if last segment active
		#algorithmMatrixSANIenforceRequirement="enforceAllSegmentsMustBeActive" #only activate neuron if all segments are active	#if(enforceSequentialActivation) then redundant; use enforceLastSegmentMustBeActive instead
		if(enforceDirectConnectionsSANIminimal):
			enforceSequentialActivation = False
		else:
			if(inferenceSegmentTiming == "none" or inferenceSegmentTiming == "biased"):
				enforceSequentialActivation = False #default: False
			elif(inferenceSegmentTiming == "exact" or inferenceSegmentTiming == "seq"):
				enforceSequentialActivation = True	#optional	#default: True #orig: True	#only activation next segment if previous segment activated
			else:
				printe("inferenceSegmentTiming error")
	else:
		enforceSequentialActivation = False

	enforceActivationAcrossSegmentsIgnoreInternalColumn = False
	if(useSANIcolumns):	
		enforceActivationAcrossSegmentsIgnoreInternalColumn = True	#ignore internal column as this column features do not necessarily have an input from the current column
	assert (int(useSANIcolumns) + int(useSANIfeatures) + int(useSANIfeaturesAndColumns)) == 1

	if(enforceDirectConnectionsSANI):	#min requirements for enforceDirectConnectionsSANI
		assert not useSANIcolumns	#enforceDirectConnectionsSANI requires last segment to be adjacent feature segment
		if(SANIfeaturesLinkFirstSegmentToAllPriorTrainSeqTokens):
			assert arrayNumberOfSegments >= 2		#note if arrayNumberOfSegments=2 then; sIndex=1: sequential segment connections for adjacent feature, sIndex=0: sequential segment connections for all other feature
		assert algorithmMatrixSANImethod=="enforceActivationAcrossSegments", "enforceDirectConnectionsSANI requires enforceActivationAcrossSegments"
		assert algorithmMatrixSANIenforceRequirement=="enforceLastSegmentMustBeActive", "enforceDirectConnectionsSANI requires enforceLastSegmentMustBeActive"
		
	if(useSANIfeaturesAndColumns):
		arrayIndexSegmentLast = arrayNumberOfSegments-1	#last feature index
		#arrayIndexSegmentAdjacentColumn = arrayNumberOfSegmentsColumnDistance-1
	elif(useSANIcolumns):
		arrayIndexSegmentLast = arrayNumberOfSegments-1
		arrayIndexSegmentAdjacentColumn = arrayNumberOfSegments-2
	elif(useSANIfeatures):
		arrayIndexSegmentLast = arrayNumberOfSegments-1

	'''
	if(useInference):	#no restrictions with useBenchmarkDefaultsEvalTestSet;
		#arrayNumberOfSegments must be <= numSeedTokens (eg with numSeedTokens = 5, segment budget = 5)
		#absolute minimum, for useSANIcolumns (and useSANIfeaturesAndColumns with arrayNumberOfSegmentsColumnDistance>1), arrayNumberOfSegments must be significantly less than numSeedTokens
		assert arrayNumberOfSegments <= numSeedTokensInference	
	'''
else:
	arrayNumberOfSegments = 1
	algorithmMatrixSANImethod = "NA"
	arrayIndexSegmentLast = 0

arrayType = pt.float32	#pt.long	#pt.float32


#POS;
useSpacyForConceptNounPOSdetection = True	#orig: True	#False: use GIAANNproto_sequencePOS predetermined word-POS dictionaries for all pos detection (never use spacy dynamically assigned pos tags)
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


#Devices;
if(useGPUdense):
	if pt.cuda.is_available():
		deviceDense = pt.device("cuda")
		pt.set_default_device(deviceDense)
	else:
		printe("useGPUdense and !pt.cuda.is_available")
else:
	deviceDense = pt.device("cpu")
if(useGPUsparse):
	if(pt.cuda.is_available()):
		deviceSparse = pt.device("cuda")
	else:
		printe("useGPUsparse and !pt.cuda.is_available")
else:
	deviceSparse = pt.device("cpu")
if(useGPUfileio):
	if(pt.cuda.is_available()):
		deviceFileIO = pt.device("cuda")
	else:
		printe("useGPUfileio and !pt.cuda.is_available")
else:
	deviceFileIO = pt.device("cpu")
if(storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam):
	if(useGPUdatabase):
		if(not pt.cuda.is_available()):
			printe("useGPUdatabase and !pt.cuda.is_available")
		deviceDatabase = pt.device("cuda")
	else:
		deviceDatabase = pt.device("cpu")
else:
	deviceDatabase = deviceSparse
deviceLoadColumnInference = deviceSparse if (storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam and useGPUdatabase != useGPUsparse) else None
deviceLoadColumnInferenceCopy = storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam and useGPUdatabase != useGPUsparse


#Dedicated feature lists (non-dynamic);
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


#Tensor print helpers;
useLovelyTensors = False
if(useLovelyTensors):
	import lovely_tensors as lt
	lt.monkey_patch()
else:
	#pass
	pt.set_printoptions(profile="full")	#pt.set_printoptions(threshold=float('inf'))

def compareSparseArrayDiff(array1, array2):
	return compareDenseArrayDiff(array1.coalesce().values(), array2.coalesce().values())

def compareDenseArrayDiff(array1, array2):
	difference = array1 != array2
	indices = pt.nonzero(difference, as_tuple=False)
	print("Indices of differences:")
	print(indices)

def getTensorSizeInMB(tensor):
	return tensor.element_size() * tensor.nelement() / (1024 ** 2)

def generateDrawSequenceIndex(sequenceWordIndex):
	return str(sequenceWordIndex).zfill(3)


#printConfiguration;
if(printConfiguration): 
	print("***** printConfiguration: ***** ")
	print("")
	print("#Execution mode selection;")
	print("useQuickExecution:", useQuickExecution)
	print("useBenchmark:", useBenchmark)
	print("useAutoresearch:", useAutoresearch)
	#print("useDefault:", useDefault)
	print("executionMode:", executionMode)
	print("inferenceTrainFirstSequences:", inferenceTrainFirstSequences)
	print("")
	print("#Primary Draw settings;")
	print("drawNetworkDuringTrain:", drawNetworkDuringTrain)
	print("drawNetworkDuringInference:", drawNetworkDuringInference)
	print("")
	print("#Inference settings;")
	print("numSeedTokensInference:", numSeedTokensInference)
	if(useInference):
		print("inferenceUseNextTokenPredictionsOrTargetsToActivateNextColumnFeatures:", inferenceUseNextTokenPredictionsOrTargetsToActivateNextColumnFeatures)
	print("useBenchmarkDefaultsEvalTestSet:", useBenchmarkDefaultsEvalTestSet)
	print("inferenceEvaluateTestSet:", inferenceEvaluateTestSet)
	print("inferenceSegmentTiming:", inferenceSegmentTiming)
	print("inferenceActivationsType:", inferenceActivationsType)
	print("inferenceReportTokenAccuracyConstrainByColumn:", inferenceReportTokenAccuracyConstrainByColumn)
	print("")
	print("#Dataset Type;")
	print("datasetType:", datasetType)
	print("")
	print("#Database;")
	print("databaseFolder:", databaseFolder)
	print("trainLoadExistingDatabase:", trainLoadExistingDatabase)
	print("trainMaxSequences:", trainMaxSequences)
	print("maxSequenceLength:", maxSequenceLength)
	print("numberEpochs:", numberEpochs)
	print("")
	print("#Multisentence predictions;")
	print("multisentencePredictions:", multisentencePredictions)
	print("numSentencesPerSequence:", numSentencesPerSequence)
	print("")
	print("#Dendritic branches;")
	print("multipleDendriticBranches:", multipleDendriticBranches)
	print("numberOfDendriticBranches:", numberOfDendriticBranches)
	print("randomlyAssignBranches:", randomlyAssignBranches)
	print("")
	print("#Dataset;")
	print("datasetName:", datasetName)
	if(not datasetType=="textfile"):
		print("datasetsLibrary4plus:", datasetsLibrary4plus)
		print("datasetCfg:", datasetCfg)
		print("useLocalDataset:", useLocalDataset)
		if(useLocalDataset):
			print("datasetFolder:", datasetFolder)
	print("")
	print("#RAM;")
	print("useGPUdense:", useGPUdense)
	print("useGPUsparse:", useGPUsparse)
	print("useGPUfileio:", useGPUfileio)
	print("storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam:", storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam)
	print("storeDatabaseGlobalFeatureNeuronsInRam:", storeDatabaseGlobalFeatureNeuronsInRam)
	#print("trainEndGenerateGlobalFeatureNeuronsTensor:", trainEndGenerateGlobalFeatureNeuronsTensor)
	#print("inferenceStartGenerateGlobalFeatureNeuronsTensor:", inferenceStartGenerateGlobalFeatureNeuronsTensor)
	if(executionMode=="train"):
		print("trainSparseConnectionsTensor:", trainSparseConnectionsTensor)
		print("trainSparseNeuronsTensor:", trainSparseNeuronsTensor)
	if(storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam):
		print("useGPUdatabase:", useGPUdatabase)
	print("")
	print("#Benchmarking;")
	print("useBenchmarkDefaults:", useBenchmarkDefaults)
	print("spacyPipelineOptimisations:", spacyPipelineOptimisations)
	print("")
	print("#Optimisations;")
	print("inferenceOnlyRetainPredictedTargetObservedColumn:", inferenceOnlyRetainPredictedTargetObservedColumn)
	print("inferenceOnlyRetainPredictedTargetObservedColumnBeamSearch:", inferenceOnlyRetainPredictedTargetObservedColumnBeamSearch)
	print("trainStoreFeatureMapsGlobally:", trainStoreFeatureMapsGlobally)
	print("")
	print("#Segment activation time;")
	print("inferenceUseNeuronFeaturePropertiesTime:", inferenceUseNeuronFeaturePropertiesTime)
	print("inferenceUseNeuronFeaturePropertiesTimeExact:", inferenceUseNeuronFeaturePropertiesTimeExact)
	print("")
	print("#Array properties;")
	print("arrayIndexPropertiesEfficient:", arrayIndexPropertiesEfficient)
	print("")
	print("#SANI;")
	print("useSANI:", useSANI)
	print("")
	print("#Immediate (direct) connections;")
	print("enforceDirectConnections:", enforceDirectConnections)
	print("enforceDirectConnectionsSANI:", enforceDirectConnectionsSANI)
	print("")
	print("#Concept column delimiters:")
	print("pretrainConceptColumnsDelimitByPOSenforce:", pretrainConceptColumnsDelimitByPOSenforce)
	print("")
	print("#Beam search;")
	if(useInference):
		print("inferenceBeamSearch:", inferenceBeamSearch)
		print("inferenceBeamWidth:", inferenceBeamWidth)
		print("inferenceBeamDepth:", inferenceBeamDepth)
	print("")
	print("#Inference activations;")
	if(useInference):
		print("inferenceConnectionsStrengthBoolean:", inferenceConnectionsStrengthBoolean)
		print("inferenceSegmentActivationsBoolean:", inferenceSegmentActivationsBoolean)
		if(inferenceSegmentActivationsBoolean):
			print("inferenceSegmentActivationsBooleanFeatureSegmentsOnly:", inferenceSegmentActivationsBooleanFeatureSegmentsOnly)
		print("inferenceSourceActivationsBoolean:", inferenceSourceActivationsBoolean)
	print("")
	print("#Train optimisations;")
	print("trainSequenceObservedColumnsUseSequenceFeaturesOnly:", trainSequenceObservedColumnsUseSequenceFeaturesOnly)
	print("trainSequenceObservedColumnsMatchSequenceWords:", trainSequenceObservedColumnsMatchSequenceWords)
	print("optimisationCombineSparseUpdatesPerSequence:", optimisationCombineSparseUpdatesPerSequence)
	print("optimisationUseCUDAObservedColumnUpdateKernel:", optimisationUseCUDAObservedColumnUpdateKernel)
	print("optimisationGetTrainRequiredSourceFeatureIndicesByObservedColumnVectorize:", optimisationGetTrainRequiredSourceFeatureIndicesByObservedColumnVectorize)
	print("optimisationGetFeatureConnectionsForSourceFeatureCache:", optimisationGetFeatureConnectionsForSourceFeatureCache)
	print("optimisationNormaliseSourceFeatureIndicesDisabled:", optimisationNormaliseSourceFeatureIndicesDisabled)
	print("optimisationObservedColumnsWriteMetadataCheck:", optimisationObservedColumnsWriteMetadataCheck)
	print("")
	print("#Draw;")
	print("drawSegments:", drawSegments)
	print("drawBranches:", drawBranches)
	print("drawRelationTypes:", drawRelationTypes)
	print("drawDelimiters:", drawDelimiters)
	print("drawDefault:", drawDefault)
	print("drawNetworkDuringTrainSave:", drawNetworkDuringTrainSave)
	if(useInference):
		print("drawNetworkDuringInferenceSave:", drawNetworkDuringInferenceSave)
	print("drawNetworkSaveFormatVector:", drawNetworkSaveFormatVector)
	print("")
	print("#Database save paths;")
	if(executionMode=="inference" or executionMode=="trainAndInference"):
		print("inferencePromptFileName:", inferencePromptFileName)
	print("")
	print("#SANI settings;")
	if(useSANI):
		print("useSANIcolumns:", useSANIcolumns)
		print("useSANIfeatures:", useSANIfeatures)
		print("useSANIfeaturesAndColumns:", useSANIfeaturesAndColumns)
		if(useSANIfeaturesAndColumns):
			print("arrayNumberOfSegmentsColumnDistance: ", arrayNumberOfSegmentsColumnDistance)
			print("arrayNumberOfSegmentsFeatureDistance: ", arrayNumberOfSegmentsFeatureDistance)
		print("arrayNumberOfSegments: ", arrayNumberOfSegments)
		print("algorithmMatrixSANImethod: ", algorithmMatrixSANImethod)
		if(algorithmMatrixSANImethod=="enforceActivationAcrossSegments"):
			print("algorithmMatrixSANIenforceRequirement: ", algorithmMatrixSANIenforceRequirement)
		print("enforceSequentialActivation: ", enforceSequentialActivation)
	print("")
	print("************************************ ")
	
