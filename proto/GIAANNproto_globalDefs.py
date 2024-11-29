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

useGPU = True
if(useGPU):
	if pt.cuda.is_available():
		pt.set_default_device(pt.device("cuda"))
	
# Set boolean variables as per specification
useSANI = False
useInference = False  # useInference mode
if(useInference):
	inferencePredictiveNetwork = False	#use MLP to predict next token	#orig:False
	incrementallySeedNetwork = True	#default:True	#orig:False
	useNeuronFeaturePropertiesTimeDuringInference = False	#default:False	#orig:False	#not yet implemented
	transformerUseInputConnections = False	#initialise (dependent var)
	if(inferencePredictiveNetwork):
		inferencePredictiveNetworkModelMLP = False
		inferencePredictiveNetworkModelTransformer = True
		if(inferencePredictiveNetworkModelTransformer):
			transformerUseInputConnections = False	#incomplete	#optional
	if(incrementallySeedNetwork):
		inferenceSeedTargetActivationsGlobalFeatureArrays = False	#optional	#orig:False
	else:
		inferenceSeedTargetActivationsGlobalFeatureArrays = False	#not supported
	lowMem = False		#mandatory
	if(inferenceSeedTargetActivationsGlobalFeatureArrays):
		sequenceObservedColumnsUseSequenceFeaturesOnly = False	#mandatory	#global feature arrays are directly written to during inference seed phase
	else:
		sequenceObservedColumnsUseSequenceFeaturesOnly = True	#optional	#sequence observed columns arrays only store sequence features.	#will affect which network changes can be visualised	#during seed phase only (will bias prediction towards target sentence words)
	sequenceObservedColumnsMatchSequenceWords = True	#optional	#introduced GIAANNproto1b12a; more robust method for training (independently train each instance of a concept in a sentence)	#False: not robust as there may be less concept columns than concepts referenced in sequence (if multiple references to the same column)	
	drawSequenceObservedColumns = False	#mandatory
	drawRelationTypes = False	#False: draw activation status
	drawNetworkDuringTrain = False
else:
	lowMem = True		 #optional
	sequenceObservedColumnsUseSequenceFeaturesOnly = True	#optional	#sequence observed columns arrays only store sequence features.	#will affect which network changes can be visualised
	sequenceObservedColumnsMatchSequenceWords = True	#optional	#introduced GIAANNproto1b12a; more robust method for training (independently train each instance of a concept in a sentence)	#False: not robust as there may be less concept columns than concepts referenced in sequence (if multiple references to the same column)	
	drawSequenceObservedColumns = False	#optional	#draw sequence observed columns (instead of complete observed columns)	#note if !drawSequenceObservedColumns and !sequenceObservedColumnsUseSequenceFeaturesOnly, then will still draw complete columns	#optional (will affect which network changes can be visualised)
	drawRelationTypes = True	#draw feature neuron and connection relation types in different colours
	drawNetworkDuringTrain = True	#default: True

	 
decreasePermanenceOfInactiveFeatureNeuronsAndConnections = True	#default: True
performRedundantCoalesce = False	#additional redundant coalesce operations

if(sequenceObservedColumnsMatchSequenceWords):
	#sumChangesToConceptNeuronSequenceInstances = True	#mandatory	#for multiple instances of concept in sequence, need to take the sum of the changes between the existing and modified arrays for each instance of a same concept in the sequence
	assert not drawSequenceObservedColumns, "sequenceObservedColumnsMatchSequenceWords does not currently support drawSequenceObservedColumns; requires concept_name_to_index (i.e. one-to-one mapping between name and feature index in SequenceObservedColumns arrays) etc"

useSaveData = True	#save data is required to allow consecutive sentence training and inference (because connection data are stored in observed columns, which are refreshed every sentence)
usePOS = True		 # usePOS mode	#mandatory
useParallelProcessing = True	#mandatory (else restore original code pre-GIAANNproto1b3a)
randomiseColumnFeatureXposition = True	#shuffle x position of column internal features such that their connections can be better visualised
databaseFolder = "" #default: ""

increaseColumnInternalConnectionsStrength = True #Increase column internal connections strength
if(increaseColumnInternalConnectionsStrength):
 	increaseColumnInternalConnectionsStrengthModifier = 10.0
	
#debug vars;
debugSmallDataset = False
debugConceptFeaturesOccurFirstInSubsequence = False #Constrain column feature detection to be after concept feature detection
debugConnectColumnsToNextColumnsInSequenceOnly = False
debugDrawNeuronStrengths = False
if(useInference):
	if(not inferencePredictiveNetwork):
		debugConceptFeaturesOccurFirstInSubsequence = True	#enables higher performance prediction without training (ie before learning appropriate column feature associations by forgetting features belonging to external columns)
	debugDrawNeuronStrengths = True
debugReloadGlobalFeatureNeuronsEverySentence = False

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
	deactivateNeuronsUponPrediction = False
	useActivationDecrement = False
	if(inferencePredictiveNetwork):
		useActivationDecrement = True
		if(useActivationDecrement):
			useActivationDecrementNonlinear = True
	else:
		#pass
		deactivateNeuronsUponPrediction = True
	activationDecrementPerPredictedToken = 0.1	#0.05	#CHECKTHIS
	if(incrementallySeedNetwork):
		activationDecrementSeed = activationDecrementPerPredictedToken
	else:
		activationDecrementPerPredictedSentence = 0.5
		activationDecrementSeed = activationDecrementPerPredictedSentence
				
	num_seed_tokens = 5	#number of seed tokens in last sentence of inference prompt (remaining tokens will be prediction tokens)
	num_prediction_tokens = 10	#number of words to predict after network seed

	if(inferencePredictiveNetwork):
		kc = 1	#number of topk columns to predict
		kf = 1	#number of topk features to predict
		
		#if kc>1 or kf>1:
		if debugConceptFeaturesOccurFirstInSubsequence:
			multipleTargets = False
		else:
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

#common array indices
array_index_properties_strength = 0
array_index_properties_permanence = 1
array_index_properties_activation = 2
array_index_properties_time = 3
array_index_properties_pos = 4
array_number_of_properties = 5
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
j1 = 5   # Activation trace duration


# For the purpose of the example, process a limited number of sentences
sentence_count = 0
max_sentences_train = 1000  # Adjust as needed

	

if not lowMem:
	global_feature_neurons_file = databaseFolder + 'global_feature_neurons.pt'


variableConceptNeuronFeatureName = "variableConceptNeuronFeature"
feature_index_concept_neuron = 0



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

def printe(str):
	print(str)
	exite
