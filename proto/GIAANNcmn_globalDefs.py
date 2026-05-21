"""GIAANNcmn_globalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 BAI Research Pty Ltd (bairesearch.com.au)

# License:
MIT License

# Installation:
see GIAANNcmn_globalDefs.py

# Usage:
see GIAANNcmn_globalDefs.py

# Description:
GIA ANN common global Defs

"""

import torch as pt
import math
import sys

if(__name__ == "__main__"):
	sys.modules["GIAANNcmn_globalDefs"] = sys.modules[__name__]


#modality selection;
useModalityNLP = True	#default: True	#orig: True
if(useModalityNLP):
	modalityName = "NLP"
	useModalityOR = False
else:
	#dev only;
	modalityName = "OR"
	useModalityOR = True


#Execution mode selection;
useQuickExecution = True	#intro: True	#default: False
useDefault = False	#default: True
useBenchmark = False		#use benchmark file naming schemes and evals
useAutoresearch = False
useDrawNetworkIndependently = False	#default: False	#default: True
inferenceTrainFirstSequences = False	#dependent var
if(useQuickExecution):
	executionMode = "inference" 	#mandatory: "inference" (effective trainAndInference but uses a text datafile)
	inferenceTrainFirstSequences = True	#trains first sequences in inference_prompt.txt, performs inference only on last sequence
elif(useDefault):
	executionMode = "train"	#optional: "train/"inference"/"trainAndInference"
elif(useBenchmark):
	executionMode = "inference"	#optional: "train/"inference"/"trainAndInference" 
elif(useAutoresearch):
	executionMode = "trainAndInference"
elif(useDrawNetworkIndependently):
	executionMode = "train"	#default: "train" or "trainAndInference" #set to the execution mode the network was trained on
else:
	raise RuntimeError("execution mode undefined")


#Primary Draw settings:
drawNetworkDuringTrain = False	#default: False  	#network drawing for prototype (not suitable for fast training)
drawNetworkDuringInference = False	#default: False


#Inference settings:
numSeedTokensInference = 8	#default: 5, 8, 12, 16	#this is also set during train phase only so that the derived numberOfSegments always matches inference phase
useInference = True  #mandatory: True	#enable options that support inference mode
if(useInference):
	if(useBenchmark or useAutoresearch):
		inferenceUseNextTokenPredictionsOrTargetsToActivateNextColumnFeatures = False	#default: False	#orig: False	#False: use current target (default top-1 accuracy measurement)
	else:
		inferenceUseNextTokenPredictionsOrTargetsToActivateNextColumnFeatures = True	#default: True	#orig: True		#True: activate next column features using current prediction
inferenceAddNewFeatures = True	#default: True	#orig: False	#run a controlled expansion pass during inference to add missing columns/features without training updates

if(useQuickExecution):
	useBenchmarkDefaultsEvalTestSet = False	#default: False: eval training-set
elif(useDefault):
	useBenchmarkDefaultsEvalTestSet = True	#default: True: eval test-set
elif(useBenchmark):
	useBenchmarkDefaultsEvalTestSet = False	#default: False: eval training-set
elif(useAutoresearch):
	useBenchmarkDefaultsEvalTestSet = True	#default: True: eval test-set
elif(useDrawNetworkIndependently):
	useBenchmarkDefaultsEvalTestSet = True	#N/A

if(useBenchmarkDefaultsEvalTestSet):
	inferenceEvaluateTestSet = True
	inferenceEvaluateTestSetTrainMaxSequences10M = False	#default: False	#orig: False	#required if performing test-set eval on database trained with > 3M sequences (based on how the original test-set was generated)
	inferenceSegmentTiming = "none"	#~optimum
	#inferenceSegmentTiming = "biased"	#default
	#inferenceSegmentTiming = "exact"
	#inferenceSegmentTiming = "seq"
	inferenceActivationsType = "boolf"	#default
	#inferenceActivationsType = "boolf+c"
	#inferenceActivationsType = "intf+c" 	#~optimum
else:
	inferenceEvaluateTestSet = False
	inferenceEvaluateTestSetTrainMaxSequences10M = False
	#inferenceSegmentTiming = "none"
	#inferenceSegmentTiming = "biased"
	inferenceSegmentTiming = "exact"		#default
	#inferenceSegmentTiming = "seq"
	inferenceActivationsType = "boolf"	#default
	#inferenceActivationsType = "boolf+c"
	#inferenceActivationsType = "intf+c"
inferenceReportTokenAccuracyConstrainByColumn = False	#default: False	#orig: False


#Database;
databaseFolderBaseLocal = "../database"	#default: "../database"
databaseFolderBaseSSD = "/media/user/ssdpro/GIAANN/database"	#default: "/media/user/ssdpro/GIAANN/database"
if(useQuickExecution):
	trainMaxSequences = 10	#N/A: auto generated from inference_prompt.txt.trainAndInference
	databaseFolderBase = databaseFolderBaseLocal
elif(useDefault):
	trainMaxSequences = 5000	#dev: 5000, 200000, 1000000 	#default: 5000	  #adjust as needed	#max sequences for train
	databaseFolderBase = databaseFolderBaseSSD
elif(useBenchmark):
	trainMaxSequences = 5000	#5000, 200000, 1000000
	databaseFolderBase = databaseFolderBaseSSD
elif(useAutoresearch):
	trainMaxSequences = 5000	#5000
	#databaseFolderBase = "../database"
	databaseFolderBase = databaseFolderBaseSSD
elif(useDrawNetworkIndependently):
	trainMaxSequences = 0	#not used
	databaseFolderBase = databaseFolderBaseLocal
	#databaseFolderBase = "/media/user/ssdpro/GIAANN/databaseOscar1000-numSeedTokensInference8-spacyPipelineOptimisations"
databaseFolderTemplate = databaseFolderBase + "Template/"
if(databaseFolderBase==databaseFolderBaseSSD):
	inferenceCopyTemplateDatasets = True	#default: True	#copy template dataset files into databaseFolder at inference startup
else:
	inferenceCopyTemplateDatasets = False
databaseFolderTemplateDatasetFileNamePattern = "*.*"
maxSequenceLength = 80	#default:80	#orig:100		#in words	#depends on CPU/GPU RAM availability during train 
numberEpochs = 1	#default: 1


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


#SANI;
useSANI = True	#default: True	#orig: False	#sequentially activated neuronal input


#modality global defs;
if(modalityName=="NLP"):
	from GIAANNnlp_globalDefs import *
elif(modalityName=="OR"):
	from GIAANNor_globalDefs import *
		

#database folder finalise;
databaseFolder = databaseFolderBase + databaseFolderExtension + "/"
databaseClearScriptName = "clear.sh"
inferencePromptFile = databaseFolder + inferencePromptFileName
if(modalityName=="NLP"):
	if(useInference):
		if(not useQuickExecution):
			if(datasetType=="textfile"):
				trainPromptFileName = databaseFolder + trainPromptFileName
	posFolder = databaseFolder + posFolder


#Error report;
def printe(str):
	raise RuntimeError(str)


#RAM;
if(useModalityOR):
	useGPUdense = False
	useGPUsparse = False
elif(useModalityNLP):
	if(useAutoresearch):
		useGPUdense = False
		useGPUsparse = False
	else:
		useGPUdense = True	#default: True
		if(executionMode=="inference" or executionMode=="trainAndInference"):
			useGPUsparse = False	#default: False	#orig: True	#inference requires high RAM to store sparse tensors	#inference can be slightly faster CPU sparse tensor operations
		elif(executionMode=="train"):
			useGPUsparse = True	#default: True		#slight performance increase during train (does not use significant additional GPU ram during train)
useGPUsparseStrict = True	#default: True	#orig: False	#optional	#enforce strict sparse device during transfer to/from dense tensors (make conversion process always use sparse device)	 #no significant difference in speed; can theoretically affect peak CPU or GPU RAM
useGPUfileio = False	#default: False	#orig: useGPUsparse

if(useGPUsparse):
	trainSparseConnectionsTensor = False	#default: False	#orig: False
	trainSparseNeuronsTensor = False	#default: False	#orig: False
else:
	trainSparseConnectionsTensor = True	#default: True	#orig: True	#use sparse connections tensor during training of sequence
	trainSparseNeuronsTensor = True		#default: True	#orig: True	#use sparse neurons tensor during training of sequence
if(trainSparseNeuronsTensor):
	assert trainSparseConnectionsTensor, "trainSparseNeuronsTensor requires trainSparseConnectionsTensor=True"

storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam = False	#default: False	#orig2: True	#orig1: False	#optional	#store database feature connections and column separated feature neuron data in RAM, else dynamically load these from filesystem per sequence
if(storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam):
	useGPUdatabase = False	#default: False	#default: False
	resizeTensorsOnRAMdatabaseSave = False	#default: False #orig: True	#resize all feature neuron and connections tensors during final RAM database save
	resizeTensorsOnRAMdatabaseLoad = False	#default: False #orig: True	#resize all feature neuron and connections tensors during initial RAM database load

if(executionMode=="train"):
	storeDatabaseGlobalFeatureNeuronsInRam = False		 #default: False	#orig: True	 #not required to be True for inference compatibility	#optional
else:
	storeDatabaseGlobalFeatureNeuronsInRam = True		#mandatory: True	#if storeDatabaseGlobalFeatureNeuronsInRam=True use global feature neuron tensors, else use feature neuron tensors in observed columns (note feature connection tensors are always in observed columns)
trainEndGenerateGlobalFeatureNeuronsTensor = False	#derived var
inferenceStartGenerateGlobalFeatureNeuronsTensor = False	#derived var
if(not storeDatabaseGlobalFeatureNeuronsInRam):
	if(executionMode=="train"):
		trainEndGenerateGlobalFeatureNeuronsTensor = False	#default: False	#orig: False	#generates and saves a globalFeatureNeurons at the end of training (not just individual column featureNeurons tensors)
	elif(executionMode=="inference"):
		inferenceStartGenerateGlobalFeatureNeuronsTensor = False	#default: False	#orig: False	#generates and saves a globalFeatureNeurons at the start of inference after being trained with storeDatabaseGlobalFeatureNeuronsInRam=False (with individual column featureNeurons tensors)


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
	if(useDrawNetworkIndependently):
		drawAllColumns = True	#mandatory
	else:
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

drawNetworkDuringTrainSaveFilenamePrepend = "GIAANNproto1xAllColumnsTrainSequenceIndex"
drawNetworkDuringInferenceSaveFilenamePrepend = "GIAANNproto1xSequenceObservedColumnsInferenceTokenIndex"
drawNetworkIndependentSaveFilename = "GIAANNproto1xAllColumnsDraw"
drawHighResolutionFigure = True	#required for inference debug
ignoreNewlineCharacters = True
drawSparseArrays = True	#default: True	#orig: False	#can draw sequences contained within larger databases without running out of memory (due to densifying arrays). drawEfficient=True is required to draw even large sized databases (>10000 neurons)
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
useParallelProcessing = True	#mandatory (else restore original code pre-GIAANNproto1b3a)
randomiseColumnFeatureXposition = True	#shuffle x position of column internal features such that their connections can be better visualised


#print vars (Information);
if(useAutoresearch):
	printEvalSequenceBar = False
	printTrainSequenceBar = False
else:
	printEvalSequenceBar = True	#default: True	#orig: False	#print each eval sequence iteration using standard tqdm bar
	printTrainSequenceBar = True	#default: True	#orig: False	#print each training sequence iteration using standard tqdm bar
if(useBenchmark):
	printTimeDatabaseLoadSaveTimes = True
	printRamMaxUsage = True
else:
	printTimeDatabaseLoadSaveTimes = False
	printRamMaxUsage = False
if(useDrawNetworkIndependently):
	printCountTotalParameters = True	#count number of connections in network
else:
	printCountTotalParameters = False	
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
	if(printEvalSequenceBar):
		printHeaderDuringInferencePredict = False
		printPredictionsDuringInferencePredict = False
	else:
		printHeaderDuringInferencePredict = True
		printPredictionsDuringInferencePredict = True
printPredictionsDuringInferencePredictBeamSearch = False
printSequenceConceptAssignment = False	#print each training sequence split by column assignment
printSequenceConceptAssignmentByLine = False	#display each column on a new line

printSequenceDefault = False	#default: True	#orig: True
printSequenceRaw = False	#print each training sequence raw text (suitable for inference_prompt.txt generation)
printSequenceDelimiters = False	#print each training sequence with delimiters
printSequencePOS = False	#print each training sequence with POS tags
printSequenceCount = False	#print each training sequence count
if(useModalityOR):
	printSequenceDefault = True
	printTrainSequenceBar = False
	printEvalSequenceBar = False
elif(useModalityNLP):
	if(generateEvalText):
		printSequenceRaw = True
		printTrainSequenceBar = False
	else:
		#these settings are ignored if printTrainSequenceBar/printEvalSequenceBar:
		printSequenceDefault = True
		#printSequenceCount = True

printTrainSequenceBarDescription = "Training sequences"
printTrainSequenceBarUnit = "sequence"
printTrainSequenceBarUpdateStep = 1
printEvalSequenceBarDescription = "Eval sequences"
printEvalSequenceBarUnit = "sequence"
printEvalSequenceBarUpdateStep = 1
printEvalSequenceBarInitialSequenceCount = 0
printPromptSequenceBarDescription = "Prompt sequences"
printPromptSequenceBarUnit = "sequence"
printPromptSequenceBarUpdateStep = 1
printPromptSequenceBarInitialSequenceCount = 0
printSequenceTerminalSafeEscapePrefix16Bit = "\\u"
printSequenceTerminalSafeEscapePrefix32Bit = "\\U"
printSequenceTerminalSafeEscapeCodepointWidth16Bit = 4
printSequenceTerminalSafeEscapeCodepointWidth32Bit = 8
printSequenceTerminalSafeEscapeCodepointMax16Bit = 0xFFFF
printSequenceTerminalSafeEscapeFormatPadPrefix = "0"
printSequenceTerminalSafeEscapeFormatType = "X"
printSequenceTerminalSafeTextEmpty = ""


#Debug vars;
debugPrintTrainSectionTimes = False	#print per-sequence timing breakdown for key train sections
debugPrintTrainSectionTimesSourceFeatureConnections = False	#print granular source-feature-connection timings within updateObservedColumnsEfficient
debugPrintRamCurrentUsage = False
debugPrintRamAverageUsage = False
debugPrintRamMaxUsagePhaseLocal = False
if(debugPrintRamMaxUsagePhaseLocal):
	assert not printRamMaxUsage

debugPrintTotalInferenceTokens = False	#print total number of inference tokens in seed phase, prediction phase, and both phases (summed across all sequences) 
debugPrintSpacySectionTimes = False	#print spacy preprocessing times

if(useAutoresearch or printEvalSequenceBar):
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
	if(not printSequenceRaw):
		debugTerminateOnConceptColumnsDelimitByPOSwarning = False	#default: True
if(pretrainConceptColumnsDelimitByPOSenforce):
	debugTerminateOnConceptColumnsDelimitByPOSerror = False
else:
	debugTerminateOnConceptColumnsDelimitByPOSerror = False

debugDeleteGPUcache = False
debugRuntimeReleaseGPUMemory = False	#default: False	#aggressively release cached CUDA memory after sequence processing	#results in significant performance drop
debugRuntimeReleaseGPUMemoryEverySequenceCount = 1	#default: 1	#only apply release every N processed sequences
if(debugRuntimeReleaseGPUMemory):
	assert debugRuntimeReleaseGPUMemoryEverySequenceCount > 0, "debugRuntimeReleaseGPUMemoryEverySequenceCount must be > 0"

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
if(auxiliaryNeurons and auxiliaryNeuronsTokenisationSubword):
	auxiliaryNeuronsTokenisationSubwordFeaturesDictFile = databaseFolder + auxiliaryNeuronsTokenisationSubwordFeaturesDictFileName
	auxiliaryNeuronsTokenisationSubwordFeatureWordsByParentWordFile = databaseFolder + auxiliaryNeuronsTokenisationSubwordFeatureWordsByParentWordFileName
if(auxiliaryNeurons and auxiliaryNeuronsSimilar):
	auxiliaryNeuronsSimilarWordsFeaturesDictFile = databaseFolder + auxiliaryNeuronsSimilarWordsFeaturesDictFileName
	auxiliaryNeuronsSimilarWordsFeatureWordWeightsByParentWordFile = databaseFolder + auxiliaryNeuronsSimilarWordsFeatureWordWeightsByParentWordFileName
	if(auxiliaryNeuronsSimilarWordsStatic):
		auxiliaryNeuronsSimilarWordsDatasetFolder = databaseFolder + auxiliaryNeuronsSimilarWordsDatasetFolderName + "/"
		auxiliaryNeuronsSimilarWordsDataset2File = auxiliaryNeuronsSimilarWordsDatasetFolder + auxiliaryNeuronsSimilarWordsDataset2FileName
		auxiliaryNeuronsSimilarWordsDataset3File = auxiliaryNeuronsSimilarWordsDatasetFolder + auxiliaryNeuronsSimilarWordsDataset3FileName
		auxiliaryNeuronsSimilarWordsDataset3SourceFile = auxiliaryNeuronsSimilarWordsDatasetFolder + auxiliaryNeuronsSimilarWordsDataset3SourceFileName
		auxiliaryNeuronsSimilarWordsDataset3SourceDownloadArchiveFile = auxiliaryNeuronsSimilarWordsDatasetFolder + auxiliaryNeuronsSimilarWordsDataset3SourceDownloadArchiveFileName
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

if(conceptColumnsDelimitByPOS):
	if(detectReferenceSetDelimitersBetweenNouns):
		conceptFeaturesReferenceSetDelimiterDeterministicListFile = databaseFolder + 'conceptFeaturesReferenceSetDelimiterDeterministicList.pkl'
		conceptFeaturesReferenceSetDelimiterProbabilisticListFile = databaseFolder + 'conceptFeaturesReferenceSetDelimiterProbabilisticList.pkl'
	else:
		conceptFeaturesReferenceSetDelimiterListFile = databaseFolder + 'conceptFeaturesReferenceSetDelimiterList.pkl'
globalFeatureNeuronsFile = 'globalFeatureNeurons'
globalFeatureNeuronsFileFull = databaseFolder + globalFeatureNeuronsFile + pytorchTensorFileExtension

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
	if(modalityName=="OR"):
		useSANIcolumns = False
		useSANIfeatures = True
		useSANIfeaturesAndColumns = False
		if(submodalityName=="image"):
			if(modalityORimageSequenceEncode=="saccades"):
				arrayNumberOfSegments = modalityORimageSnapshotsPerSequence
			elif(modalityORimageSequenceEncode=="distance" or modalityORimageSequenceEncode=="axis" or modalityORimageSequenceEncode=="axes"):
				arrayNumberOfSegments = int(math.ceil(math.sqrt(float((modalityORimageSequenceEncodeDistanceFieldSegments - 1)*(modalityORimageSequenceEncodeDistanceFieldSegments - 1)*2)))) + 1
			elif(modalityORimageSequenceEncode=="none"):
				arrayNumberOfSegments = 1
			else:
				raise RuntimeError("GIAANNcmn_globalDefs error: modalityORimageSequenceEncode must be 'saccades', 'distance', 'axis', 'axes', or 'none'")
		elif(submodalityName=="video"):
			arrayNumberOfSegments = modalityORvideoMaxEncodedSnapshotsPerSequence
	else:
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


#Draw Network Independently;
if(useDrawNetworkIndependently):
	drawEfficient = True
else:
	drawEfficient = False
if(drawEfficient):
	drawEfficientFormat3D = False	#default: False	#optional	#True: save standalone drawEfficient large-network output in LDraw .ldr format #False: save in 2D matplotlib .svg format
	if(drawEfficientFormat3D):
		drawEfficientFormat3Dprism = True	#default: False	#True: position standalone drawEfficient 3D columns on a square 2D grid and draw each column as a rectangular prism
	drawEfficientIntracolumnHorizontalOffset = True #default: True	#feature neurons within columns have a horizontal x (or xy) offset applied
	if(drawEfficientIntracolumnHorizontalOffset):
		drawEfficientIntracolumnHorizontalOffsetWidth = 5	#default: 5
	drawEfficientGrid = False	#default: False #draws column feature neuron y positions at their real featureIndex
	drawEfficientCompact = True	#default: True	#better emulates the original draw visualisation of drawEfficient=False (but still not the same as no randomised horizontal position of nodes within columns)
	if(drawEfficientGrid == drawEfficientCompact):
		printe("drawEfficient configuration error: exactly one of drawEfficientGrid or drawEfficientCompact must be True")
	drawEfficientDrawDeadNeurons = True	#default: True	#draw empty columns with no connected neurons
	

#printConfiguration;
if(printConfiguration): 
	print("***** printConfiguration: ***** ")
	print("")
	print("#Modality selection;")
	print("modalityName:", modalityName)
	print("")
	print("#Execution mode selection;")
	print("useQuickExecution:", useQuickExecution)
	print("useBenchmark:", useBenchmark)
	print("useAutoresearch:", useAutoresearch)
	print("useDrawNetworkIndependently:", useDrawNetworkIndependently)
	#print("useDefault:", useDefault)
	print("executionMode:", executionMode)
	print("inferenceTrainFirstSequences:", inferenceTrainFirstSequences)
	print("")
	print("#Primary Draw settings;")
	print("drawNetworkDuringTrain:", drawNetworkDuringTrain)
	print("drawNetworkDuringInference:", drawNetworkDuringInference)
	if(drawEfficient):
		print("")
		print("#Draw Network Independently;")
		print("drawEfficient:", drawEfficient)
		print("drawEfficientFormat3D:", drawEfficientFormat3D)
		if(drawEfficientFormat3D):
			print("drawEfficientFormat3Dprism:", drawEfficientFormat3Dprism)
		print("drawEfficientIntracolumnHorizontalOffset:", drawEfficientIntracolumnHorizontalOffset)
		if(drawEfficientIntracolumnHorizontalOffset):
			print("drawEfficientIntracolumnHorizontalOffsetWidth:", drawEfficientIntracolumnHorizontalOffsetWidth)
		print("drawEfficientGrid:", drawEfficientGrid)
		print("drawEfficientCompact:", drawEfficientCompact)
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
	print("inferenceReportGroundedAccuracy:", inferenceReportGroundedAccuracy)
	if(inferenceReportGroundedAccuracy):
		print("closedWorldGroundedDatasetGenerated:", closedWorldGroundedDatasetGenerated)
		print("inferenceReportGroundedRealisticNLPmetric:", inferenceReportGroundedRealisticNLPmetric)
		print("inferenceReportGroundedStrongerGroundedNLPmetric:", inferenceReportGroundedStrongerGroundedNLPmetric)
		print("inferenceReportGroundedAccuracyMod1_labelBalancedDataset:", inferenceReportGroundedAccuracyMod1_labelBalancedDataset)
		print("inferenceReportGroundedAccuracyMod2_majorityClassBaseline:", inferenceReportGroundedAccuracyMod2_majorityClassBaseline)
		print("inferenceReportGroundedAccuracyMod3_perLabelMetrics:", inferenceReportGroundedAccuracyMod3_perLabelMetrics)
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
	if(executionMode=="train" or executionMode=="trainAndInference"):
		print("\ttrainSparseConnectionsTensor:", trainSparseConnectionsTensor)
		print("\ttrainSparseNeuronsTensor:", trainSparseNeuronsTensor)
	print("storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam:", storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam)
	if(storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam):
		print("\tuseGPUdatabase:", useGPUdatabase)
	print("storeDatabaseGlobalFeatureNeuronsInRam:", storeDatabaseGlobalFeatureNeuronsInRam)
	#print("trainEndGenerateGlobalFeatureNeuronsTensor:", trainEndGenerateGlobalFeatureNeuronsTensor)
	#print("inferenceStartGenerateGlobalFeatureNeuronsTensor:", inferenceStartGenerateGlobalFeatureNeuronsTensor)
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
	
