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
debugPrintConfiguration = True	#print common global defs configuration
debugPrintTotalInferenceTokens = False	#print total number of inference tokens in seed phase, prediction phase, and both phases (summed across all sequences) 
debugPrintTotalFeatures = False	#print c+f upon load


#Train/inference mode selection;
useInference = True  #default: True	#support inference mode else train (inferenceTrainFirstSequences: only) mode
drawNetworkDuringTrain = False	#default: False  	#network drawing for prototype (not suitable for fast training)
if(useInference):
	drawNetworkDuringInference = False	#default: False
	inferenceTrainFirstSequences = True	#default: True	#orig: True	#True: trains first sequences in inference_prompt.txt, performs inference only on last sequence; False: run inference on every sequence as independent seed/target prompts
	inferenceUseNextTokenPredictionsOrTargetsToActivateNextColumnFeatures = True	#default: True	#orig: True	#True: activate next column features using current prediction; False: use current target (top-1 accuracy measurement)
numSeedTokensInference = 12	#default: 5, 8, 12	#this is also set during train phase only so that the derived numberOfSegments always matches inference phase
inferenceAddNewFeatures = True	#default: True	#orig: False	#run a controlled expansion pass during inference to add missing columns/features without training updates


#Database;
databaseFolder = "../database/"	#default: "../database/"	#performance: "/media/user/ssdpro/GIAANN/database/"	#orig: ""
trainMaxSequences = 100000		#dev: 10, 500, 5000, 10000, 100000 	#default: 1000000	  #adjust as needed	#max sequences for train
maxSequenceLength = 80	#default:80	#orig:100		#in words	#depends on CPU/GPU RAM availability during train 
numberEpochs = 1	#default: 1


#Dataset;
datasetsLibrary4plus = False	#default: False	#orig: False	#set False during dev to maintain benchmark consistency
if(datasetsLibrary4plus):
	datasetName = "wikimedia/wikipedia"
	datasetCfg = "20231101.en"
else:
	datasetName = "wikipedia"
	datasetCfg = "20220301.en"
useLocalDataset = True	#default: True	#orig: False (stream)	#use local dataset	#automatic huggingface access to dataset is unreliable
if(useLocalDataset):
	datasetFolder = "../../dataset/"
	useLocalDatasetDownloadManual = True	#default: True	#manual download dataset files into datasetFolder	#automatic huggingface access to dataset is unreliable
	datasetProcessedCacheFolderName = "processed_dataset_cache"	#manual name for processed dataset cache
	datasetProcessedCacheFolder = datasetFolder + datasetProcessedCacheFolderName + "/"
else:
	useLocalDatasetDownloadManual = False
trainTestSet = False	#default: False	#only set True to generate an inference test set (with debugPrintTrainSequenceRaw=True)
if(trainTestSet):
	testSetRatio = 0.1	#ratio of articles in dataset to be used for test (vs train) set - taken from end of dataset
	assert useLocalDataset	#required for efficiency


#Multisentence predictions;
multisentencePredictions = False	#default: False	#each sequence comprises multiple sentences	#requires higher GPU RAM for train
if(multisentencePredictions):
	numSentencesPerSequence = 3	#default: 3
else:
	numSentencesPerSequence = 1


#RAM;
useGPUdense = True	#default: True
if(useInference):
	useGPUsparse = True	#default: False	#orig: True	#inference requires high RAM to store sparse tensors
else:
	useGPUsparse = True		#default: True	#orig: False	#slight performance increase during train (does not use significant additional GPU ram during train)
useGPUsparseStrict = True	#orig: False	#enforce strict sparse device during transfer to/from dense tensors
runtimeReleaseGPUMemory = False	#default: True	#aggressively release cached CUDA memory after sequence processing
runtimeReleaseGPUMemoryEverySequenceCount = 1	#default: 1	#only apply release every N processed sequences
if(runtimeReleaseGPUMemory):
	if(runtimeReleaseGPUMemoryEverySequenceCount <= 0):
		raise RuntimeError("runtimeReleaseGPUMemoryEverySequenceCount must be > 0")


#Optimisations;
inferenceOnlyRetainPredictedTargetObservedColumn = False	#default: False	#orig: False	#load/evict one observed column per prediction step	#the majority of inference memory is the sparse global activation tensors (not the observed column connections)
inferenceOnlyRetainPredictedTargetObservedColumnBeamSearch = False	#default: False	#orig: False	#True: retain only current beam-search target(s); False: retain all beam-search targets	#the majority of inference memory is the sparse global activation tensors (not the observed column connections)


#Segment activation time;
if(useInference):
	inferenceUseNeuronFeaturePropertiesTime = True	#optional	#orig:False
	inferenceUseNeuronFeaturePropertiesTimeExact = True	#optional	#orig:False
else:
	inferenceUseNeuronFeaturePropertiesTime = False
	inferenceUseNeuronFeaturePropertiesTimeExact = False


#Dendritic branches;
multipleDendriticBranches = True	#default: True	#orig: False
if(multipleDendriticBranches):
	numberOfDendriticBranches = 2	#default: 5, 2	#affects train+inference RAM
	randomlyAssignBranches = False	#optional	#orig: False
else:
	numberOfDendriticBranches = 1
	randomlyAssignBranches = False


#Array properties (disable to optimise train speed/RAM during train);
arrayIndexPropertiesEfficient = True	#default: True	#orig: False (required for drawing pos types)
if(arrayIndexPropertiesEfficient):
	arrayIndexPropertiesStrength = True
	arrayIndexPropertiesPermanence = False
	arrayIndexPropertiesActivation = False	#inference only (see arrayIndexPropertiesActivationCreate)
	arrayIndexPropertiesTime = False	#inference only (see arrayIndexPropertiesTimeCreate)
	arrayIndexPropertiesPos = False
else:
	arrayIndexPropertiesStrength = True
	arrayIndexPropertiesPermanence = True
	arrayIndexPropertiesActivation = False	#inference only (see arrayIndexPropertiesActivationCreate)
	arrayIndexPropertiesTime = False 	#inference only (see arrayIndexPropertiesTimeCreate)
	arrayIndexPropertiesPos = True
arrayIndexPropertiesActivationCreate = arrayIndexPropertiesActivation or useInference
arrayIndexPropertiesTimeCreate = arrayIndexPropertiesTime or inferenceUseNeuronFeaturePropertiesTime


#SANI;
useSANI = True	#default: True	#orig: False	#sequentially activated neuronal input


#Immediate (direct) connections;
enforceDirectConnections = True	#default: True	#orig: False	#prediction requires a direct connection from previous prediction as observed during training (ie adjacent tokens)
if(enforceDirectConnections):
	enforceDirectConnectionsSANI = True	#default: True #orig: False	#enforce activation of first segment (direct feature connection)
	enforceDirectConnectionsMinWordDistance = False	#default: False #orig: True	#enforce min word distance (=1) during inference
	enforceDirectConnectionsIgnoreSeed = True	#default: True #orig: False
else:
	enforceDirectConnectionsSANI = False
	enforceDirectConnectionsMinWordDistance = False
if(enforceDirectConnectionsSANI):
	useSANI = True	#sequentially activated neuronal input (divide dendrites into segments)	#override
	enforceDirectConnectionsSANIminimal = False	#default: False	#orig: True
else:
	enforceDirectConnectionsSANIminimal = False
if(enforceDirectConnectionsMinWordDistance):
	arrayIndexPropertiesMinWordDistance = True	#store min word distance per connection
else:
	arrayIndexPropertiesMinWordDistance = False	#optional	#default: False
minimumPredictionActivationThreshold = 0.0	#explicit threshold application not required (for verification only)
predictionEnsureConnectedToPreviousPrediction = True	#default: True	#ensure every new prediction connects to previous node (requirement is independent of enforceDirectConnections)


#Concept column delimiters:
conceptColumnsDelimitByPOS = True	#mandatory: True	#orig: False	#closer to original GIA specification	#FUTURE: still requires working for edge cases
if(conceptColumnsDelimitByPOS):
	conceptColumnsDelimiterPOStypes = ['VERB', 'ADP']	#deterministic reference set delimiters (GIA actions/conditions)
	conceptColumnsDelimiterWordTypes = [';', ':', '.', '?', '!']	#deterministic reference set delimiters (GIA logical conditions)
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
		inferenceSegmentActivationsBoolean = True	#default: True	#do not overweight by common features (e.g. determiners)
		if(inferenceSegmentActivationsBoolean):
			inferenceSegmentActivationsBooleanFeatureSegmentsOnly = True	#orig: False
		inferenceSourceActivationsBoolean = True	#default: True (do not sum SANI segments)	#orig: False
	else:
		inferenceConnectionsStrengthBoolean = False	#default: False
		inferenceSegmentActivationsBoolean = False	#default: False	
		inferenceSourceActivationsBoolean = True	#default: True	#orig: False (theoretically effectively True)

	inferenceSeedNetwork = True	#default: True
else:
	inferenceActivationFunction = False
	inferenceStrengthLookupCache = False
if(useInference):
	useMaxSequences = False	#False: use all sequences from inference_prompt.txt
else:
	useMaxSequences = True	#True: use all sequences from dataset


#Train optimisations;
#trainSequenceObservedColumnsUseSequenceFeaturesOnly can be upgraded so only a limited amount of data is ever loaded to GPU during train (it currently temporarily masks entire feature arrays in GPU during transfer phase)
if(useInference):
	lowMem = False		#mandatory: False	#if lowMem=False use global feature neuron tensors, else use feature neuron tensors in observed columns (note feature connection tensors are always in observed columns)
else:
	lowMem = False		 #default: False	#orig: True	#currently required to be False for inference compatibility	#optional
trainSequenceObservedColumnsUseSequenceFeaturesOnly = True	#default:True	#optional	#sequence observed columns arrays only store sequence features.	#will affect which network changes can be visualised
trainSequenceObservedColumnsMatchSequenceWords = True	#mantatory		#introduced GIAANNproto1b12a; more robust method for training (independently train each instance of a concept in a sequence)	#False: not robust as there may be less concept columns than concepts referenced in sequence (if multiple references to the same column)	
combineSparseUpdatesPerSequence = True	#default: True	#orig: False	#updateObservedColumnsEfficient combines sparse updates per sequence instead of per column (reduces calls to coalesce) 


#Draw;
#select a single draw method (colouring scheme);
drawSegments = False and useSANI	#optional
drawBranches = False and multipleDendriticBranches	#optional
drawRelationTypes = False and not arrayIndexPropertiesEfficient	#optional
drawDelimiters = False	#optional
drawDefault = True	#optional
if(useInference):
	drawSequenceObservedColumns = False	#mandatory
	drawAllColumns = False	#mandatory
	drawNetworkDuringTrainSave = False	#default: False
	drawNetworkDuringInferenceSave = False	#True is only for debug
else:
	drawSequenceObservedColumns = False	#default: False	#optional	#draw sequence observed columns (instead of complete observed columns)	#note if !drawSequenceObservedColumns and !trainSequenceObservedColumnsUseSequenceFeaturesOnly, then will still draw complete columns	#optional (will affect which network changes can be visualised)
	drawAllColumns = False	#default: False	#optional	#draw all columns in network (only used for automated visualisation; drawNetworkDuringTrainSave)	#requires !drawSequenceObservedColumns
	if(drawAllColumns):
		assert not trainSequenceObservedColumnsUseSequenceFeaturesOnly
	drawNetworkDuringTrainSave = True	#default: False
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
	drawDefaultInference = False	#False: draw activation status
else:
	print("warning: draw scheme not defined")
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


#Debug vars;
debugPrintTrainSequenceDefault = True	#default: True	#orig: True
debugPrintTrainSequenceRaw = False	#print each training sequence raw text (suitable for inference_prompt.txt generation)
debugPrintTrainSequenceConceptAssignment = False	#print each training sequence split by column assignment
debugPrintTrainSequenceConceptAssignmentByLine = False	#display each column on a new line
debugPrintTrainSequenceDelimiters = False	#print each training sequence with delimiters
debugPrintTrainSequencePOS = False	#print each training sequence with POS tags

debugTerminateInferenceOnPredictionTargetMismatch = False
debugTerminateInferenceOnNoPredictionCandidatesAvailable = False
debugTerminateOnConceptColumnsDelimitByPOSerror = False
if(debugPrintTrainSequenceRaw):
	debugTerminateOnConceptColumnsDelimitByPOSwarning = False
else:
	debugTerminateOnConceptColumnsDelimitByPOSwarning = True
debugDeleteGPUcache = False

debugLimitFeatures = False
if(debugLimitFeatures):
	debugLimitFeaturesCMax = 520437
	debugLimitFeaturesFMax = 80955

printPredictionsDuringInferencePredict = True
printPredictionsDuringInferencePredictBeamSearch = False
debugPrintMinWordDistanceDetails = False
debugOnlyDrawBranchIndexConnections = False
debugOnlyDrawBranchIndexX = 0

debugConnectNodesToNextNodesInSequenceOnly = False
debugConnectColumnsToNextColumnsInSequenceOnly = False
debugSmallDataset = False	#required if huggingface Wikipedia dataset is offline
debugDrawNeuronActivations = False
if(useInference):
	debugDrawNeuronActivations = True
debugReloadGlobalFeatureNeuronsEverySequence = False


#Concept/feature names;
useDedicatedFeatureLists = False	#default: False - dynamically learn concept features	#True: use static feature lists (depreciated)
#if usePOS and not lowMem:
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
	assert not lowMem, "useInference: global feature neuron lists are required" 
	assert useSaveData,  "useInference: useSaveData is required" 


#Database save paths;
inferencePromptFile = databaseFolder + 'inference_prompt.txt'	#inference_prompt.txt
conceptColumnsDictFile = databaseFolder + 'conceptColumnsDict.pkl'
conceptFeaturesDictFile = databaseFolder + 'conceptFeaturesDict.pkl'
observedColumnsDir = databaseFolder + 'observedColumns'
pytorchTensorFileExtension = ".pt"
if(conceptColumnsDelimitByPOS):
	if(detectReferenceSetDelimitersBetweenNouns):
		conceptFeaturesReferenceSetDelimiterDeterministicListFile = databaseFolder + 'conceptFeaturesReferenceSetDelimiterDeterministicList.pkl'
		conceptFeaturesReferenceSetDelimiterProbabilisticListFile = databaseFolder + 'conceptFeaturesReferenceSetDelimiterProbabilisticList.pkl'
	else:
		conceptFeaturesReferenceSetDelimiterListFile = databaseFolder + 'conceptFeaturesReferenceSetDelimiterList.pkl'
if not lowMem:
	globalFeatureNeuronsFile = 'globalFeatureNeurons'
	globalFeatureNeuronsFileFull = databaseFolder + globalFeatureNeuronsFile + pytorchTensorFileExtension
posFolder = databaseFolder + "POS/"
posDictFile = "everPos.wordnet.pkl.gz"
saveGlobalFeatureNeuronsRate = 1000


#Common array indices;
arrayIndexPropertiesStrengthIndex = None
arrayIndexPropertiesPermanenceIndex = None
arrayIndexPropertiesActivationIndex = None
arrayIndexPropertiesTimeIndex = None
arrayIndexPropertiesPosIndex = None
arrayIndexPropertiesMinWordDistanceIndex = None	#optional property storing min distance between connected words
arrayPropertiesList = []
if(arrayIndexPropertiesStrength):
	arrayIndexPropertiesStrengthIndex = len(arrayPropertiesList)
	arrayPropertiesList.append(arrayIndexPropertiesStrengthIndex)
if(arrayIndexPropertiesPermanence):
	arrayIndexPropertiesPermanenceIndex = len(arrayPropertiesList)
	arrayPropertiesList.append(arrayIndexPropertiesPermanenceIndex)
if(arrayIndexPropertiesActivationCreate):
	arrayIndexPropertiesActivationIndex = len(arrayPropertiesList)
	arrayPropertiesList.append(arrayIndexPropertiesActivationIndex)
if(arrayIndexPropertiesTimeCreate):
	arrayIndexPropertiesTimeIndex = len(arrayPropertiesList)
	arrayPropertiesList.append(arrayIndexPropertiesTimeIndex)
if(arrayIndexPropertiesPos):
	arrayIndexPropertiesPosIndex = len(arrayPropertiesList)
	arrayPropertiesList.append(arrayIndexPropertiesPosIndex)
if(arrayIndexPropertiesMinWordDistance):
	arrayIndexPropertiesMinWordDistanceIndex = len(arrayPropertiesList)
	arrayPropertiesList.append(arrayIndexPropertiesMinWordDistanceIndex)
arrayNumberOfProperties = len(arrayPropertiesList)


#SANI settings;
arrayIndexSegmentFirst = 0
if(useSANI):
	if(enforceDirectConnectionsSANIminimal):
		useSANIcolumns = False
		useSANIfeatures = True
		useSANIfeaturesAndColumns = False
	else:
		useSANIcolumns = False	#assign segments by concept column proximity to connection target during train (includes internal concept column)
		useSANIfeatures = False	#assign segments by feature proximity to connection target during train
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

	algorithmMatrixSANImethod="enforceActivationAcrossSegments"	#default	#only activate a segment under conditions
	#algorithmMatrixSANImethod="doNotEnforceActivationAcrossSegments"	#orig	#activate segments without any sequentiality requirement	simply addActivationAcrossSegments	#equivalent to !useSANI
	if(algorithmMatrixSANImethod=="enforceActivationAcrossSegments"):
		#algorithmMatrixSANIenforceRequirement="enforceAnySegmentMustBeActive"	#activate neuron if any segment is active
		algorithmMatrixSANIenforceRequirement="enforceLastSegmentMustBeActive"	#default	#only activate neuron if last segment active
		#algorithmMatrixSANIenforceRequirement="enforceAllSegmentsMustBeActive" #only activate neuron if all segments are active	#if(enforceSequentialActivation) then redundant; use enforceLastSegmentMustBeActive instead
		if(enforceDirectConnectionsSANIminimal):
			enforceSequentialActivation = False
		else:
			enforceSequentialActivation = True	#optional	#default: True #orig: True	#only activation next segment if previous segment activated
	
	enforceActivationAcrossSegmentsIgnoreInternalColumn = False
	if(useSANIcolumns):	
		enforceActivationAcrossSegmentsIgnoreInternalColumn = True	#ignore internal column as this column features do not necessarily have an input from the current column
	assert (int(useSANIcolumns) + int(useSANIfeatures) + int(useSANIfeaturesAndColumns)) == 1

	if(enforceDirectConnectionsSANI):	#min requirements for enforceDirectConnectionsSANI
		assert not useSANIcolumns	#enforceDirectConnectionsSANI requires last segment to be adjacent feature segment
		assert arrayNumberOfSegments >= 2		#note if arrayNumberOfSegments=2 then; sIndex=1: sequential segment connections for adjacent feature, sIndex=0: sequential segment connections for all other feature
		assert enforceSequentialActivation or not enforceSequentialActivation
		
	if(useSANIfeaturesAndColumns):
		arrayIndexSegmentLast = arrayNumberOfSegments-1	#last feature index
		#arrayIndexSegmentAdjacentColumn = arrayNumberOfSegmentsColumnDistance-1
	elif(useSANIcolumns):
		arrayIndexSegmentLast = arrayNumberOfSegments-1
		arrayIndexSegmentAdjacentColumn = arrayNumberOfSegments-2
	elif(useSANIfeatures):
		arrayIndexSegmentLast = arrayNumberOfSegments-1

	if(useInference):
		#arrayNumberOfSegments must be <= numSeedTokens (eg with numSeedTokens = 5, segment budget = 5)
		#absolute minimum, for useSANIcolumns (and useSANIfeaturesAndColumns with arrayNumberOfSegmentsColumnDistance>1), arrayNumberOfSegments must be significantly less than numSeedTokens
		assert arrayNumberOfSegments <= numSeedTokensInference	
else:
	arrayNumberOfSegments = 1
	algorithmMatrixSANImethod = "NA"
	arrayIndexSegmentLast = 0

arrayType = pt.float32	#pt.long	#pt.float32


#POS;
useSpacyForConceptNounPOSdetection = True	#orig: True	#False: use GIAANNproto_sequencePOS predetermined word-POS dictionaries for all pos detection (never use spacy dynamically assigned pos tags)
spacyModelName = 'en_core_web_trf'	#orig: 'en_core_web_sm'
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


#Error report;
ERROR_SIGNAL = 1
def exitWithError():
	sys.exit(ERROR_SIGNAL)
def printe(str):
	print(str)
	exitWithError()

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


#debugPrintConfiguration;
if(debugPrintConfiguration): 
	print("***** debugPrintConfiguration: ***** ")
	print("")
	print("#Train/inference mode selection;")
	print("useInference:", useInference)
	print("drawNetworkDuringTrain:", drawNetworkDuringTrain)
	if(useInference):
		print("drawNetworkDuringInference:", drawNetworkDuringInference)
		print("inferenceTrainFirstSequences:", inferenceTrainFirstSequences)
		print("inferenceUseNextTokenPredictionsOrTargetsToActivateNextColumnFeatures:", inferenceUseNextTokenPredictionsOrTargetsToActivateNextColumnFeatures)
	print("numSeedTokensInference:", numSeedTokensInference)
	print("")
	print("#Database;")
	print("databaseFolder:", databaseFolder)
	print("trainMaxSequences:", trainMaxSequences)
	print("maxSequenceLength:", maxSequenceLength)
	print("numberEpochs:", numberEpochs)
	print("")
	print("#Dataset;")
	print("datasetsLibrary4plus:", datasetsLibrary4plus)
	print("datasetName:", datasetName)
	print("datasetCfg:", datasetCfg)
	print("useLocalDataset:", useLocalDataset)
	if(useLocalDataset):
		print("datasetFolder:", datasetFolder)
	print("")
	print("#Multisentence predictions;")
	print("multisentencePredictions:", multisentencePredictions)
	print("numSentencesPerSequence:", numSentencesPerSequence)
	print("")
	print("#RAM;")
	print("useGPUdense:", useGPUdense)
	print("useGPUsparse:", useGPUsparse)
	print("useGPUsparseStrict:", useGPUsparseStrict)
	print("runtimeReleaseGPUMemory:", runtimeReleaseGPUMemory)
	print("runtimeReleaseGPUMemoryEverySequenceCount:", runtimeReleaseGPUMemoryEverySequenceCount)
	print("inferenceOnlyRetainPredictedTargetObservedColumn:", inferenceOnlyRetainPredictedTargetObservedColumn)
	print("inferenceOnlyRetainPredictedTargetObservedColumnBeamSearch:", inferenceOnlyRetainPredictedTargetObservedColumnBeamSearch)
	print("")
	print("#Segment activation time;")
	print("inferenceUseNeuronFeaturePropertiesTime:", inferenceUseNeuronFeaturePropertiesTime)
	print("inferenceUseNeuronFeaturePropertiesTimeExact:", inferenceUseNeuronFeaturePropertiesTimeExact)
	print("")
	print("#Dendritic branches;")
	print("multipleDendriticBranches:", multipleDendriticBranches)
	print("numberOfDendriticBranches:", numberOfDendriticBranches)
	print("randomlyAssignBranches:", randomlyAssignBranches)
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
	print("enforceDirectConnectionsMinWordDistance:", enforceDirectConnectionsMinWordDistance)
	print("")
	print("#Beam search;")
	if(useInference):
		print("inferenceBeamSearch:", inferenceBeamSearch)
		print("inferenceBeamWidth:", inferenceBeamWidth)
		print("inferenceBeamDepth:", inferenceBeamDepth)
	print("")
	print("#Draw;")
	print("drawSegments:", drawSegments)
	print("drawBranches:", drawBranches)
	print("drawRelationTypes:", drawRelationTypes)
	print("drawDelimiters:", drawDelimiters)
	print("drawDefault:", drawDefault)
	if(not useInference or inferenceTrainFirstSequences):
		print("drawNetworkDuringTrainSave:", drawNetworkDuringTrainSave)
	if(useInference):
		print("drawNetworkDuringInferenceSave:", drawNetworkDuringInferenceSave)
	print("drawNetworkSaveFormatVector:", drawNetworkSaveFormatVector)
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
	print("")
	print("************************************ ")
	
