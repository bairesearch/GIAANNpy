"""GIAANNproto_globalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024 Baxter AI (baxterai.com)

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

#RAM availability vars;
useGPUdense = True	#default: True
useGPUsparse = False	#default: False	#orig: True
useGPUpredictiveNetworkModel = True	#orig: True	#use GPU to train transformer/MLP predictive network model
maxSentenceLength = 100	#orig:10000	#default:100	#in words	#depends on CPU RAM availability during train (with trainSequenceObservedColumnsUseSequenceFeaturesOnly only limited amount of data is ever loaded to GPU during train)
databaseFolder = "" #default: ""
max_sentences_train = 100 #default: 100000000	#orig: 1000  # Adjust as needed (eg lower max_sentences_train before independent useInference execution)

# Set boolean variables as per specification
useSANI = False
useInference = False  # useInference mode
if(useInference):
	inferencePredictiveNetwork = False	#use MLP to predict next token	#orig:False
	inferenceTrainPredictionNetworkAllSentences = False	#support predictive network training on every sentence in corpus.
	inferenceIncrementallySeedNetwork = True	#default:True	#orig:False
	inferenceUseNeuronFeaturePropertiesTime = False	#default:False	#orig:False	#not yet implemented	#FUTURE; else use during train
	inferenceActivationFunction = True	#default:False	#orig:False	#required to prevent exponential runaway of activations (that negatively affects predictionNetwork loss optimisation)
	transformerUseInputConnections = False	#initialise (dependent var)
	transformerUseInputAllProperties = False	#initialise (dependent var)
	if(inferencePredictiveNetwork):
		inferenceSavePredictiveNetwork = False
		if(inferenceTrainPredictionNetworkAllSentences):
			#inferenceIncrementallySeedNetwork = True	#not supported (uses custom incremental seeding implementation)
			inferenceSavePredictiveNetwork = True
			inferenceUseNextTokenPredictionsOrTargetsToActivateNextColumnFeatures = False #default: False	#next token predictions are used to activate the next column features (rather than prediction targets)
		else:
			inferenceUseNextTokenPredictionsOrTargetsToActivateNextColumnFeatures = False	#default: False	#orig: True
		inferencePredictiveNetworkModelMLP = False
		inferencePredictiveNetworkModelTransformer = True
		if(inferencePredictiveNetworkModelTransformer):
			transformerUseInputConnections = False	#incomplete	#optional
			transformerUseInputAllProperties = True
	else:
		inferenceUseNextTokenPredictionsOrTargetsToActivateNextColumnFeatures = True #currently mandatory (only implementation available)
	if(inferenceIncrementallySeedNetwork):
		inferenceSeedTargetActivationsGlobalFeatureArrays = False	#optional	#orig:False
	else:
		inferenceSeedTargetActivationsGlobalFeatureArrays = False	#not supported
	lowMem = False		#mandatory
	if(inferenceSeedTargetActivationsGlobalFeatureArrays):
		trainSequenceObservedColumnsUseSequenceFeaturesOnly = False	#mandatory	#global feature arrays are directly written to during inference seed phase
	else:
		trainSequenceObservedColumnsUseSequenceFeaturesOnly = True	#optional	#sequence observed columns arrays only store sequence features.	#will affect which network changes can be visualised	#during seed phase only (will bias prediction towards target sentence words)
	trainSequenceObservedColumnsMatchSequenceWords = True	#optional	#introduced GIAANNproto1b12a; more robust method for training (independently train each instance of a concept in a sentence)	#False: not robust as there may be less concept columns than concepts referenced in sequence (if multiple references to the same column)	
	drawSequenceObservedColumns = False	#mandatory
	drawAllColumns = False	#mandatory
	drawRelationTypes = False	#False: draw activation status
	drawNetworkDuringTrain = False
	drawNetworkDuringTrainSave = False
	drawNetworkDuringInferenceSeed = False
	drawNetworkDuringInferencePredict = False
	drawNetworkDuringInferenceSave = False
else:
	lowMem = False		 #default: False	#orig: True	#currently required to be False for inference compatibility	#optional
	trainSequenceObservedColumnsUseSequenceFeaturesOnly = True	#optional	#sequence observed columns arrays only store sequence features.	#will affect which network changes can be visualised
	trainSequenceObservedColumnsMatchSequenceWords = True	#optional	#introduced GIAANNproto1b12a; more robust method for training (independently train each instance of a concept in a sentence)	#False: not robust as there may be less concept columns than concepts referenced in sequence (if multiple references to the same column)	
	drawSequenceObservedColumns = False	#optional	#draw sequence observed columns (instead of complete observed columns)	#note if !drawSequenceObservedColumns and !trainSequenceObservedColumnsUseSequenceFeaturesOnly, then will still draw complete columns	#optional (will affect which network changes can be visualised)
	drawAllColumns = False	#optional	#draw all columns in network (only used for automated visualisation; drawNetworkDuringTrainSave)	#requires !drawSequenceObservedColumns
	if(drawAllColumns):
		assert not trainSequenceObservedColumnsUseSequenceFeaturesOnly
	drawRelationTypes = True	#draw feature neuron and connection relation types in different colours
	drawNetworkDuringTrain = True	#default: True
	drawNetworkDuringTrainSave = False
	inferenceActivationFunction = False
	
drawNetworkDuringTrainSaveFilenamePrepend = "GIAANNproto1cAllColumnsTrainSentenceIndex"
drawNetworkDuringInferenceSaveFilenamePrepend = "GIAANNproto1cSequenceObservedColumnsInferenceTokenIndex"

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

useSaveData = True	#save data is required to allow consecutive sentence training and inference (because connection data are stored in observed columns, which are refreshed every sentence)
usePOS = True		 # usePOS mode	#mandatory
useParallelProcessing = True	#mandatory (else restore original code pre-GIAANNproto1b3a)
randomiseColumnFeatureXposition = True	#shuffle x position of column internal features such that their connections can be better visualised

trainIncreaseColumnInternalConnectionsStrength = True #Increase column internal connections strength
if(trainIncreaseColumnInternalConnectionsStrength):
 	trainIncreaseColumnInternalConnectionsStrengthModifier = 10.0
	
#debug vars;
debugSmallDataset = False
debugConceptFeaturesOccurFirstInSubsequence = False #Constrain column feature detection to be after concept feature detection
debugConnectColumnsToNextColumnsInSequenceOnly = False
debugDrawNeuronActivations = False
if(useInference):
	debugConceptFeaturesOccurFirstInSubsequence = False #default: False	#orig: True	#enables higher performance prediction without training (ie before learning appropriate column feature associations by forgetting features belonging to external columns)
	debugDrawNeuronActivations = True
debugReloadGlobalFeatureNeuronsEverySentence = False
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

inference_prompt_file = databaseFolder + 'inference_prompt.txt'
if(useInference):
	if(inferencePredictiveNetwork):
		inferenceInvertNeuronActivationUponPrediction = True	#default: True	#orig: False	#set activations of previously activated neurons to negative - refractory period preventing consecutive feature reactivation and facilitating prediction based on past predictions
		inferenceDeactivateNeuronsUponPrediction = False #do not use inferenceDeactivateNeuronsUponPrediction as the predictive network needs a temporarily consistent trace of the activations in the network
		#if(not inferenceActivationFunction):	#do not decrement activations if they are decremented every time the activation function is applied
		inferenceDecrementActivations = True
		if(inferenceDecrementActivations):
			inferenceDecrementActivationsNonlinear = True
		if(inferenceInvertNeuronActivationUponPrediction):
			inferenceInvertNeuronActivationUponPredictionLevel = -0.1	#orig: -1.0	#	#default: -0.1; inferenceInvertNeuronActivationUponPrediction inversion also significantly decreases activation level by a factor of approx 10x (ie 0.1), enabling reactivation after only a few feature predictions (by positive addition). inferenceDecrementActivations decrement will continue to be applied non-linearly (inferenceDecrementActivationsNonlinear) to negative activations thereby retaining their negative activation until they are predicted again (making them positive).
	else:
		inferenceInvertNeuronActivationUponPrediction = False
		inferenceDeactivateNeuronsUponPrediction = True
		inferenceDecrementActivations = False
	activationDecrementPerPredictedToken = 0.1	#0.05	#CHECKTHIS
	if(inferenceIncrementallySeedNetwork):
		activationDecrementSeed = activationDecrementPerPredictedToken
	else:
		activationDecrementPerPredictedSentence = 0.5
		activationDecrementSeed = activationDecrementPerPredictedSentence
	
	if(inferenceTrainPredictionNetworkAllSentences):
		num_seed_tokens = 0
		num_prediction_tokens = None	#sentence length
	else:
		num_seed_tokens = 5	#number of seed tokens in last sentence of inference prompt (remaining tokens will be prediction tokens)
		num_prediction_tokens = 10	#number of words to predict after network seed

	if(inferencePredictiveNetwork):
		if(debugConceptFeaturesOccurFirstInSubsequence):
			kcNetwork = 1	#number of topk columns to predict
			#inferenceTrainPredictionNetworkAllSentences currently requires debugConceptFeaturesOccurFirstInSubsequence:!multipleTargets if kcNetwork == 1"
			multipleTargets = False
		else:
			kcNetwork = 2	#number of topk columns to predict	#it is unknown which exact column a token belongs to (unless it corresponds to a concept feature/noun)
			multipleTargets = True
		kf = 1	#number of topk features to predict
		if inferenceTrainPredictionNetworkAllSentences:
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
concept_columns_dict_file = databaseFolder + 'concept_columns_dict.pkl'
concept_features_dict_file = databaseFolder + 'concept_features_dict.pkl'
observed_columns_dir = databaseFolder + 'observed_columns'
pytorch_tensor_file_extension = ".pt"
predictive_network_folder = "."
predictive_network_file_name = "model.pt"

#common array indices
array_index_properties_strength = 0
array_index_properties_permanence = 1
array_index_properties_activation = 2
array_index_properties_time = 3
array_index_properties_pos = 4
array_number_of_properties = 5
array_properties_list = [array_index_properties_strength, array_index_properties_permanence, array_index_properties_activation, array_index_properties_time, array_index_properties_pos]
array_index_segment_first = 0
if(useSANI):
	array_number_of_segments = 10	#max number of SANI segments per sequence (= max number of concept columns per sequence)
else:
	array_number_of_segments = 1
array_index_segment_internal_column = array_number_of_segments-1
array_type = pt.float32	#pt.long	#pt.float32

# Define POS tag sets for nouns and non-nouns
noun_pos_tags = {'NOUN', 'PROPN'}
non_noun_pos_tags = {'ADJ', 'ADV', 'VERB', 'ADP', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NUM', 'PART', 'PRON', 'SCONJ', 'SYM', 'X'}

def pos_int_to_pos_string(nlp, pos_int):
	if pos_int in nlp.vocab.strings:
		return nlp.vocab[pos_int].text
	else:
		return ''
		
def pos_string_to_pos_int(nlp, pos_string):
	return nlp.vocab.strings[pos_string]
		

	
# Define constants for permanence and activation trace	#TODO: train hyperparameters
z1 = 3  # Initial permanence value	
z2 = 1  # Decrement value when not activated
if(inferenceActivationFunction):
	j1 = 1
else:
	j1 = 5   # Activation trace duration


# For the purpose of the example, process a limited number of sentences
sentence_count = 0

	

if not lowMem:
	global_feature_neurons_file = 'global_feature_neurons'

variableConceptNeuronFeatureName = "variableConceptNeuronFeature"
feature_index_concept_neuron = 0

def printe(str):
	print(str)
	exite

if(useGPUdense):
	if pt.cuda.is_available():
		deviceDense = pt.device("cuda")
		pt.set_default_device(deviceDense)
	else:
		printe("useGPUdense and !pt.cuda.is_available")
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
	
	all_words = set()
	for synset in wn.all_synsets():
		for lemma in synset.lemma_names():
			all_words.add(lemma.lower())
	
	non_nouns = all_words - nouns
	max_num_non_nouns = len(non_nouns)

def get_tensor_size_in_mb(tensor):
	return tensor.element_size() * tensor.nelement() / (1024 ** 2)

useLovelyTensors = False
if(useLovelyTensors):
	import lovely_tensors as lt
	lt.monkey_patch()
else:
	pass
	#pt.set_printoptions(profile="full")	#pt.set_printoptions(threshold=float('inf'))
	
