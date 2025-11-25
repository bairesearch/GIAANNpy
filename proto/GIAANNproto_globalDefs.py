"""GIAANNproto_globalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

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

#train/inference mode selection:
useInference = False  # useInference mode	#else train mode
trainNetworkForInference = False	#disables network drawing for fast training

#RAM availability vars;
useGPUdense = True	#default: True
useGPUsparse = False	#default: False	#orig: True
useGPUpredictiveNetworkModel = True	#orig: True	#use GPU to train transformer/MLP predictive network model
maxSequenceLength = 100	#orig:10000	#default:100	#in words	#depends on CPU RAM availability during train (with trainSequenceObservedColumnsUseSequenceFeaturesOnly only limited amount of data is ever loaded to GPU during train)
databaseFolder = "../../../database"	#orig: ""
maxSequences = 10	#debug: 10, 500 	#default: 100000000	  #adjust as needed (eg lower max_sequences during train before independent inferenceTrainPredictiveNetworkAllSequences execution)	#max sequences for train or inference
numberEpochs = 1	#default: 1
multisentencePredictions = False	#default: False	#requires higher GPU RAM for train
if(multisentencePredictions):
	numSentencesPerSequence = 3	#default: 3

#Beam Search parameters
inferenceBeamSearch = False	#orig: False
if(inferenceBeamSearch):
	inferenceBeamSearchConceptColumns = False
	inferenceBeamScoreStrategy = "nodeActivation"	#options: "nodeActivation", "activation_connection", "connection"
	inferenceBeamConceptColumnNodeActivationThreshold = 0.0
	inferenceBeamInstanceNodeActivationThreshold = 0.0
	if(inferenceBeamSearchConceptColumns):
		inferenceBeamWidth = 3
		inferenceBeamDepth = 3
	else:
		inferenceBeamWidth = 3
		inferenceBeamDepth = 6

#SANI concept neuron vars;
SANIconceptNeurons = False	#execute preprocessor to allocate neurons to non-noun tuples for each concept	#similar to SANIHFNLP algorithmMatrixSANI - emulate DendriticSANIbiologicalSimulationSimple	#these are effectively concept neurons but not specific to a particular concept
if(SANIconceptNeurons):
	SANIconceptNeuronsAllocateConceptFeatureWordNeuron = True	 #allocate a separate neuron for the concept feature neuron	#currently required for implementation
	SANIconceptNeuronsAllocateWordNeurons = False	#still allocate original individual word neurons (create word connnections along with concept connections)
	SANIconceptNeuronsAllocateForPartialSubsequences = True	#assign SANI concept neurons for partial subsequences (2, 3, 4, etc word tuples; not just x word tupes where x is the length of the non-noun word tuple)
	if(SANIconceptNeuronsAllocateForPartialSubsequences):
		SANIconceptNeuronsAllocateForPartialSubsequencesMinTupleSize = 2
		SANIconceptNeuronsAllocateForPartialSubsequencesMaxTupleSize = 5
		SANIconceptNeuronsAllocateForPartialSubsequencesMinWeight = 1	#number of times a tuple instance must occur in corpus before a SANIconceptNeuron is assigned to the database network concept column
		SANIconceptNeuronsAllocateForPartialSubsequencesWeightIncrement = 1
	assert SANIconceptNeuronsAllocateConceptFeatureWordNeuron, "!SANIconceptNeuronsAllocateConceptFeatureWordNeuron not yet coded; need to update entire codebase to ensure only token.lemma or token.pos=NOUN is used to detect concept features and only token.word is used to generate a feature neuron name"
	debugSANIconceptNeurons = True
	
# Set boolean variables as per specification
useSANI = False	#sequentially activated neuronal input (divide dendrites into segments)
if(useInference):
	if(inferenceBeamSearch):
		inferencePredictiveNetwork = False
	else:
		inferencePredictiveNetwork = True	#use MLP to predict next token	#orig:False
	inferenceTrainPredictiveNetworkAllSequences = True	#support predictive network training on every sequence in corpus.	#precondition: expects database network to have been completely trained (with !useInference on all sequences)
	inferenceIncrementallySeedNetwork = True	#default:True	#orig:False	#incremental seeding is used to match the inference prediction phase algorithm (for consistency in activation method)	#requires inferenceSeedNetwork
	inferenceActivationFunction = True	#default:True	#orig:False	#required to prevent exponential runaway of activations (that negatively affects predictionNetwork loss optimisation)
	transformerUseInputConnections = False	#initialise (dependent var)
	printPredictionsDuringInferencePredict = True	#default: True
	if(useSANI):
		inferenceConnectionsStrengthBoolean = False	#default: False
		inferenceActivationStrengthBoolean = False	#default: False
	else:
		inferenceConnectionsStrengthBoolean = False	#default: False
		inferenceActivationStrengthBoolean = False	#default: False	
	if(inferenceTrainPredictiveNetworkAllSequences):
		inferenceRetainActivationsAcrossMultipleSequences = False	#default: False	#retain activations across sequences such that these can be used during training/inference
	if(inferencePredictiveNetwork):
		inferencePredictiveNetworkModel = "ColumnMLP"
		#inferencePredictiveNetworkModel = "MLP"
		#inferencePredictiveNetworkModel = "Transformer"
		inferenceSavePredictiveNetwork = False
		inferencePredictiveNetworkIndependentFCpredictions = True	#required for large database network (else may require output MLP of shape c*f * c*f)
		inferencePredictiveNetworkNormaliseInputs = True
		if(inferencePredictiveNetworkNormaliseInputs):
			inferencePredictiveNetworkNormaliseDim = 1	#orig: 0 #default: 1 -  normalise across SANI segments independently
		inferenceUseNeuronFeaturePropertiesTime = True	#default:True	#orig:False		#FUTURE; else can use during train	#requires inferencePredictiveNetworkUseInputAllProperties
		if(inferenceTrainPredictiveNetworkAllSequences):
			inferenceSavePredictiveNetwork = True
			inferenceUseNextTokenPredictionsOrTargetsToActivateNextColumnFeatures = False #default: False	#next token predictions are used to activate the next column features (rather than prediction targets)
			numberEpochs = 1000	#default: 1	#10	#debug: 1000	#number of epochs to train predictive network
		else:
			inferenceUseNextTokenPredictionsOrTargetsToActivateNextColumnFeatures = False	#default: False	#orig: True
		if(inferencePredictiveNetworkModel=="ColumnMLP"):
			inferencePredictiveNetworkLearningRate = 0.0005	#default: 0.0005
			inferencePredictiveNetworkModelFilterColumnsK = max(5, maxSequences//10)	#max(5, maxSequences//10)	#heuristic: int(c/10)	#5	#10	#50		#only consider top k columns for prediction (prefilter)
			print("inferencePredictiveNetworkModelFilterColumnsK = ", inferencePredictiveNetworkModelFilterColumnsK)
			inferencePredictiveNetworkModelFilterColumnsKmax = True	#filter columns by max column activation rather than sum column activation.
			inferencePredictiveNetworkUseInputAllProperties = True	#default: True
			inferencePredictiveNetworkIndependentFCpredictions = True	#currently required
			numberOfHiddenLayers = 1	#default: 2	#orig: 1
		elif(inferencePredictiveNetworkModel=="MLP"):
			inferencePredictiveNetworkLearningRate = 0.0005	#default: 0.0005
			inferencePredictiveNetworkUseInputAllProperties = False	#default: False
			numberOfHiddenLayers = 1
		elif(inferencePredictiveNetworkModel=="Transformer"):
			inferencePredictiveNetworkLearningRate = 0.0005	#default: 0.0005	0.005
			inferencePredictiveNetworkUseInputAllProperties = True	#default: True
			transformerUseInputConnections = False	#incomplete	#optional
			inferencePredictiveNetworkInitialiseWeightsNearZero = True	#help predictive model to learn faster (rely exclusively on input activation levels at start of training)
			transformerOutputLayerUseEveryColumn = True	#default: True	#orig: False	#whether the output layer uses features from every column (or just the final column in the sequence)
	else:
		inferenceUseNeuronFeaturePropertiesTime = False
		inferenceUseNextTokenPredictionsOrTargetsToActivateNextColumnFeatures = False	#default:True	#set to False to compare predictive performance with inferencePredictiveNetwork
	if(inferenceUseNextTokenPredictionsOrTargetsToActivateNextColumnFeatures):
		inferenceSeedNetwork = True	#default: True
		inferenceBurstAllPredictionsOrTargetsInSequence = False	#default: False	#orig: False
	else:
		inferenceSeedNetwork = False	#default: False	#True is only for debug
		inferenceBurstAllPredictionsOrTargetsInSequence = True	#default: True	#orig: True
	trainSequenceObservedColumnsUseSequenceFeaturesOnly = True	#optional	#sequence observed columns arrays only store sequence features.	#will affect which network changes can be visualised	#if used during seed phase will bias prediction towards target sequence words
	if(inferenceSeedNetwork):
		if(inferenceIncrementallySeedNetwork):
			inferenceSeedTargetActivationsGlobalFeatureArrays = False	#optional	#default:True	#orig:False	
		else:
			inferenceSeedTargetActivationsGlobalFeatureArrays = False	#not supported
		if(inferenceSeedTargetActivationsGlobalFeatureArrays):
			trainSequenceObservedColumnsUseSequenceFeaturesOnly = False	#mandatory	#global feature arrays are directly written to during inference seed phase
	lowMem = False		#mandatory
	trainSequenceObservedColumnsMatchSequenceWords = True	#optional	#introduced GIAANNproto1b12a; more robust method for training (independently train each instance of a concept in a sequence)	#False: not robust as there may be less concept columns than concepts referenced in sequence (if multiple references to the same column)	
	drawSequenceObservedColumns = False	#mandatory
	drawAllColumns = False	#mandatory
	drawRelationTypes = False	#False: draw activation status
	drawNetworkDuringTrain = False
	drawNetworkDuringTrainSave = False
	drawNetworkDuringInferenceSeed = False
	drawNetworkDuringInferencePredict = False	#True is only for debug
	drawNetworkDuringInferenceSave = False	#True is only for debug
	if(SANIconceptNeurons):
		print("SANIconceptNeurons:useInference warning: there are too many SANI concept neuron (ie non-noun tuple) features per column to perform production level GIAANN inference on a conventional system; eg 100m phrases")
else:
	inferenceUseNeuronFeaturePropertiesTime = True	#required to initialise time part of inferencePredictiveNetworkUseInputAllProperties to 0
	lowMem = False		 #default: False	#orig: True	#currently required to be False for inference compatibility	#optional
	trainSequenceObservedColumnsUseSequenceFeaturesOnly = True	#optional	#sequence observed columns arrays only store sequence features.	#will affect which network changes can be visualised
	trainSequenceObservedColumnsMatchSequenceWords = True	#optional	#introduced GIAANNproto1b12a; more robust method for training (independently train each instance of a concept in a sequence)	#False: not robust as there may be less concept columns than concepts referenced in sequence (if multiple references to the same column)	
	drawSequenceObservedColumns = False	#optional	#draw sequence observed columns (instead of complete observed columns)	#note if !drawSequenceObservedColumns and !trainSequenceObservedColumnsUseSequenceFeaturesOnly, then will still draw complete columns	#optional (will affect which network changes can be visualised)
	drawAllColumns = False	#optional	#draw all columns in network (only used for automated visualisation; drawNetworkDuringTrainSave)	#requires !drawSequenceObservedColumns
	if(drawAllColumns):
		assert not trainSequenceObservedColumnsUseSequenceFeaturesOnly
	drawRelationTypes = True	#draw feature neuron and connection relation types in different colours
	if(trainNetworkForInference):
		drawNetworkDuringTrain = False	#disabled as intend to use useInference:inferenceTrainPredictiveNetworkAllSequences after train
	else:
		drawNetworkDuringTrain = True
	drawNetworkDuringTrainSave = False
	inferenceActivationFunction = False
	if(SANIconceptNeurons):
		assert trainSequenceObservedColumnsUseSequenceFeaturesOnly	#required to significantly decrease GPU RAM during training
	
drawNetworkDuringTrainSaveFilenamePrepend = "GIAANNproto1cAllColumnsTrainSequenceIndex"
drawNetworkDuringInferenceSaveFilenamePrepend = "GIAANNproto1cSequenceObservedColumnsInferenceTokenIndex"
drawHighResolutionFigure = True	#required for inference debug
ignoreNewlineCharacters = True

#algorithm preferences;
inferenceNormaliseColumnSelectionByFeatureConnections = False  	#default: False		#cannot select one column over another if column activations are perfectly normalised with respect to each other	#see HFconnectionMatrixAlgorithmNormalise
if(inferenceNormaliseColumnSelectionByFeatureConnections):
	inferenceNormaliseColumnSelectionByFeatureConnectionsStrength = False	#else normalise column selection by number connections
inferenceNormaliseFeatureSelectionByFeatureConnections = False	#default: False
if(inferenceNormaliseFeatureSelectionByFeatureConnections):
	inferenceNormaliseFeatureSelectionByFeatureConnectionsStrength = True	#mandatory
trainNormaliseConnectionStrengthWrtContextLength = True	#default: True
trainDecreasePermanenceOfInactiveFeatureNeuronsAndConnections = False	#default: True

performRedundantCoalesce = False	#additional redundant coalesce operations

if(trainSequenceObservedColumnsMatchSequenceWords):
	#sumChangesToConceptNeuronSequenceInstances = True	#mandatory	#for multiple instances of concept in sequence, need to take the sum of the changes between the existing and modified arrays for each instance of a same concept in the sequence
	assert not drawSequenceObservedColumns, "trainSequenceObservedColumnsMatchSequenceWords does not currently support drawSequenceObservedColumns; requires concept_name_to_index (i.e. one-to-one mapping between name and feature index in SequenceObservedColumns arrays) etc"

useSaveData = True	#save data is required to allow consecutive sequence training and inference (because connection data are stored in observed columns, which are refreshed every sequence)
usePOS = True		 # usePOS mode	#mandatory
useParallelProcessing = True	#mandatory (else restore original code pre-GIAANNproto1b3a)
randomiseColumnFeatureXposition = True	#shuffle x position of column internal features such that their connections can be better visualised

trainIncreaseColumnInternalConnectionsStrength = True #Increase column internal connections strength
if(trainIncreaseColumnInternalConnectionsStrength):
 	trainIncreaseColumnInternalConnectionsStrengthModifier = 10.0
	
#debug vars;
debugSmallDataset = False	#required if huggingface Wikipedia dataset is offline
debugConceptFeaturesOccurFirstInSubsequence = False #Constrain column feature detection to be after concept feature detection
debugConnectColumnsToNextColumnsInSequenceOnly = False
debugDrawNeuronActivations = False
if(useInference):
	debugConceptFeaturesOccurFirstInSubsequence = False #default: False	#orig: True	#enables higher performance prediction without training (ie before learning appropriate column feature associations by forgetting features belonging to external columns)
	debugDrawNeuronActivations = True
debugReloadGlobalFeatureNeuronsEverySequence = False
debugInferencePredictionActivationAccumulation = False	#monitor exponential runaway of activations negatively affects predictionNetwork loss optimisation (go to nan)	 #see inferenceUseNextTokenPredictionsOrTargetsToActivateNextColumnFeatures for comparison	#solved by inferenceActivationFunction

useDedicatedFeatureLists = False
#if usePOS and not lowMem:
#	useDedicatedFeatureLists = True

useDedicatedConceptNames = False
useDedicatedConceptNames2 = False
if usePOS:
	useDedicatedConceptNames = True
	if(useDedicatedConceptNames):
		#same word can have different pos making it classed as an instance feature or concept feature
		useDedicatedConceptNames2 = True	#mandatory

#if usePOS: same word can have different pos making it classed as an instance feature or concept feature

inferencePromptFile = databaseFolder + 'inference_prompt.txt'
if(useInference):
	if(inferencePredictiveNetwork):
		inferenceInvertNeuronActivationUponPrediction = False	#default: True	#orig: False	#set activations of previously activated neurons to negative - refractory period preventing consecutive feature reactivation and facilitating prediction based on past predictions
		#if(not inferenceActivationFunction):	#do not decrement activations if they are decremented every time the activation function is applied
		inferenceDecrementActivations = True	#default: True	#False is only for debug
		if(inferenceDecrementActivations):
			inferenceDecrementActivationsNonlinear = True
		if(inferenceInvertNeuronActivationUponPrediction):
			inferenceInvertNeuronActivationUponPredictionLevel = -0.1	#orig: -1.0	#	#default: -0.1; inferenceInvertNeuronActivationUponPrediction inversion also significantly decreases activation level by a factor of approx 10x (ie 0.1), enabling reactivation after only a few feature predictions (by positive addition). inferenceDecrementActivations decrement will continue to be applied non-linearly (inferenceDecrementActivationsNonlinear) to negative activations thereby retaining their negative activation until they are predicted again (making them positive).
			inferenceDeactivateNeuronsUponPrediction = False #do not use inferenceDeactivateNeuronsUponPrediction as the predictive network needs a temporarily consistent trace of the activations in the network		
		else:
			inferenceDeactivateNeuronsUponPrediction = True
		if(inferenceUseNeuronFeaturePropertiesTime):
			inferenceUseNeuronFeaturePropertiesTimeActivate = 100	#default: 100 #max tokens remembered in sequence	#time is not reinitialised upon feature selection (deactivation)
			inferenceUseNeuronFeaturePropertiesTimeDecrement = -1
	else:
		inferenceInvertNeuronActivationUponPrediction = False
		inferenceDeactivateNeuronsUponPrediction = True	#default: True
		inferenceDecrementActivations = False
	activationDecrementPerPredictedToken = 0.1	#0.05	#CHECKTHIS
	activationDecrementPerPredictedSequence = 0.5
	if(inferenceSeedNetwork):
		if(inferenceIncrementallySeedNetwork):
			activationDecrementSeed = activationDecrementPerPredictedToken
		else:
			activationDecrementSeed = activationDecrementPerPredictedSequence
	
	if(inferenceSeedNetwork):
		numSeedTokens = 5	#number of seed tokens in last sequence of inference prompt (remaining tokens will be prediction tokens)
	else:
		numSeedTokens = 0
	
	if(debugConceptFeaturesOccurFirstInSubsequence):
		kcNetwork = 1	#number of topk columns to target
	else:
		kcNetwork = 2	#number of topk columns to target	#it is unknown which exact column a token belongs to (unless it corresponds to a concept feature/noun)
			
	if(inferencePredictiveNetwork):
		if(debugConceptFeaturesOccurFirstInSubsequence):
			kcPred = 1 	#number of topk columns to predict	#mandatory: 1
			#inferenceTrainPredictiveNetworkAllSequences currently requires debugConceptFeaturesOccurFirstInSubsequence:!multipleTargets if kcNetwork == 1"
			multipleTargets = False
		else:
			kcPred = 1 	#number of topk columns to predict
			multipleTargets = True
		kf = 1	#number of topk features to predict
		if inferenceTrainPredictiveNetworkAllSequences:
			assert kf==1
		if kf>1:
			multipleTargets = True
	else:
		#TODO: train hyperparameters
		kcMax = 1 	#(if kcDynamic: max) topk next concept column prediction
		kcDynamic = False
		if(kcDynamic):
			kcActivationThreshold = 3.0	#total column activation threshold	#minimum required to select topk
		kf = 1
	
	assert not lowMem, "useInference: global feature neuron lists are required" 
	assert useSaveData,  "useInference: useSaveData is required" 




# Paths for saving data
conceptColumnsDictFile = databaseFolder + 'conceptColumnsDict.pkl'
conceptFeaturesDictFile = databaseFolder + 'conceptFeaturesDict.pkl'
observedColumnsDir = databaseFolder + 'observedColumns'
pytorchTensorFileExtension = ".pt"
predictiveNetworkFolder = "."
predictiveNetworkFileName = "predictiveNetworkModel.pt"
SANIconceptNeuronsDictFile = databaseFolder + 'SANIconceptNeuronsDict.pkl'
SANIconceptNeuronWeightsListFile = databaseFolder + 'SANIconceptNeuronWeightsList.pkl'

#common array indices
arrayIndexPropertiesStrength = 0
arrayIndexPropertiesPermanence = 1
arrayIndexPropertiesActivation = 2
arrayIndexPropertiesTime = 3
arrayIndexPropertiesPos = 4
arrayNumberOfProperties = 5
arrayPropertiesList = [arrayIndexPropertiesStrength, arrayIndexPropertiesPermanence, arrayIndexPropertiesActivation, arrayIndexPropertiesTime, arrayIndexPropertiesPos]
arrayIndexSegmentFirst = 0
if(useSANI):
	if(multisentencePredictions):
		arrayNumberOfSegments = 10	#default: 5
	else:
		arrayNumberOfSegments = 3	#default:3	#orig: 10	#max number of SANI segments per sequence (= max number of concept columns per sequence)	
		#note if arrayNumberOfSegments=3 then; sIndex=2: sequential segment connections for current column, sIndex=1: adjacent column connections, sIndex=0: all other column connections
		#must be less than the (total number of concepts in a sequence - total number of concepts in effective predictive seed sequence)
	#algorithmMatrixSANImethod="doNotEnforceSequentialityAcrossSegments"	#orig	#activate segments without any sequentiality requirement	#simply addActivationAcrossSegments
	algorithmMatrixSANImethod="enforceSequentialActivationAcrossSegments"	#default	#only activate a segment if previous segment active
	if(algorithmMatrixSANImethod=="enforceSequentialActivationAcrossSegments"):
		#algorithmMatrixSANIenforceRequirement="enforceAnySegmentMustBeActive"	#activate neuron if any segment is active
		algorithmMatrixSANIenforceRequirement="enforceLastSegmentMustBeActive"	#default	#only activate neuron if last (ie adjacent or internal column) segment active
		#algorithmMatrixSANIenforceRequirement="enforceAllSegmentsMustBeActive"	#only activate neuron if all segments are active	#redundant; use enforceLastSegmentMustBeActive instead
else:
	arrayNumberOfSegments = 1
	algorithmMatrixSANImethod = "NA"
	
arrayIndexSegmentInternalColumn = arrayNumberOfSegments-1
arrayIndexSegmentAdjacentColumn = arrayNumberOfSegments-2
arrayType = pt.float32	#pt.long	#pt.float32

# Define POS tag sets for nouns and non-nouns
nounPosTags = {'NOUN', 'PROPN'}
nonNounPosTags = {'ADJ', 'ADV', 'VERB', 'ADP', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NUM', 'PART', 'PRON', 'SCONJ', 'SYM', 'X'}

def posIntToPosString(nlp, posInt):
	if posInt in nlp.vocab.strings:
		return nlp.vocab[posInt].text
	else:
		return ''
		
def posStringToPosInt(nlp, posString):
	return nlp.vocab.strings[posString]
		

	
# Define constants for permanence and activation trace	#TODO: train hyperparameters
z1 = 3  # Initial permanence value	
z2 = 1  # Decrement value when not activated
if(inferenceActivationFunction):
	j1 = 1 #default: 1
else:
	j1 = 5   # Activation trace duration

	

if not lowMem:
	globalFeatureNeuronsFile = 'globalFeatureNeurons'

variableConceptNeuronFeatureName = "variableConceptNeuronFeature"
variableConceptNeuronFeatureNameAbbreviation = "VCNF"
featureIndexConceptNeuron = 0

def printe(str):
	print(str)
	exite

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
if(useGPUpredictiveNetworkModel):
	if pt.cuda.is_available():
		devicePredictiveNetworkModel = pt.device("cuda")
	else:
		printe("useGPUmodel and !pt.cuda.is_available")
else:
	devicePredictiveNetworkModel = pt.device("cpu")
	
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

def getTensorSizeInMB(tensor):
	return tensor.element_size() * tensor.nelement() / (1024 ** 2)

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
