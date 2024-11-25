"""GIAANNproto_predictiveNetwork.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_predictiveNetwork.py

# Usage:
see GIAANNproto_predictiveNetwork.py

# Description:
GIA ANN proto predictive Network

"""

import torch as pt

from GIAANNproto_globalDefs import *
import GIAANNproto_databaseNetwork
import GIAANNproto_databaseNetworkTrain
if(inferencePredictiveNetwork):
	if(inferencePredictiveNetworkModelMLP):
		import GIAANNproto_predictiveNetworkMLP
	elif(inferencePredictiveNetworkModelTransformer):
		import GIAANNproto_predictiveNetworkTransformer
import GIAANNproto_databaseNetworkDraw
import GIAANNproto_sparseTensors

# Define the SequenceObservedColumnsInferencePrediction class
class SequenceObservedColumnsInferencePrediction:
	def __init__(self, databaseNetworkObject, words, lemmas, observed_columns_dict, observed_columns_sequence_word_index_dict):
		#note cs may be slightly longer than number of unique columns in the sequence, if there are multiple instances of the same concept/noun lemma in the sequence
	
		self.databaseNetworkObject = databaseNetworkObject
		
		self.observed_columns_dict = observed_columns_dict	# key: lemma, value: ObservedColumn
		self.observed_columns_sequence_word_index_dict = observed_columns_sequence_word_index_dict	# key: sequence word index, value: ObservedColumn
		
		self.cs2 = len(databaseNetworkObject.concept_columns_dict)
		self.fs2 = len(databaseNetworkObject.concept_features_dict)
			
		feature_connections_list = []
		for observed_column in observed_columns_sequence_word_index_dict.values():
			 feature_connections_list.append(observed_column.feature_connections)
		self.feature_connections = pt.stack(feature_connections_list, dim=2)
		

if not drawSequenceObservedColumns:
	class SequenceObservedColumnsDraw:
		def __init__(self, databaseNetworkObject, observed_columns_dict):
			self.databaseNetworkObject = databaseNetworkObject
			self.observed_columns_dict = observed_columns_dict
			
def process_concept_words_inference(sequence_observed_columns, sentenceIndex, doc, doc_seed, doc_predict, num_seed_tokens, num_prediction_tokens):

	print("process_concept_words_inference:")
		
	sequenceWordIndex = 0
	
	words_doc, lemmas_doc, pos_tags_doc = getLemmas(doc)
	concept_mask = pt.tensor([i in sequence_observed_columns.columns_index_sequence_word_index_dict for i in range(len(lemmas_doc))], dtype=pt.bool)
	concept_indices = pt.nonzero(concept_mask).squeeze(1)
	numberConcepts = concept_indices.shape[0]

	words_seed, lemmas_seed, pos_tags_seed = getLemmas(doc_seed)
	concept_mask_seed = pt.tensor([i in sequence_observed_columns.columns_index_sequence_word_index_dict for i in range(len(lemmas_seed))], dtype=pt.bool)
	concept_indices_seed = pt.nonzero(concept_mask_seed).squeeze(1)
	numberConceptsInSeed = concept_indices_seed.shape[0]
	
	if(inferencePredictiveNetwork and inferencePredictiveNetworkModelTransformer):
		GIAANNproto_databaseNetwork.generate_global_feature_connections(sequence_observed_columns.databaseNetworkObject)
		
	#seed network;
	words, lemmas, pos_tags= getLemmas(doc)
	GIAANNproto_databaseNetworkTrain.process_concept_words(sequence_observed_columns, sentenceIndex, doc, words, lemmas, pos_tags, train=False, num_seed_tokens=num_seed_tokens, numberConceptsInSeed=numberConceptsInSeed)
	
	if(not inferenceSeedTargetActivationsGlobalFeatureArrays):
		# Update observed columns from sequence observed columns
		sequence_observed_columns.update_observed_columns_wrapper()	#convert sequence observed columns feature neuron arrays back to global feature neuron arrays

	GIAANNproto_databaseNetworkDraw.visualize_graph(sequence_observed_columns)
	
	if(inferencePredictiveNetwork):
		if(inferencePredictiveNetworkModelMLP):
			GIAANNproto_predictiveNetworkMLP.nextWordPredictionMLPcreate(sequence_observed_columns.databaseNetworkObject)
		elif(inferencePredictiveNetworkModelTransformer):
			GIAANNproto_predictiveNetworkTransformer.nextWordPredictionTransformerCreate(sequence_observed_columns.databaseNetworkObject)

	'''
	global_feature_neurons_dense = sequence_observed_columns.databaseNetworkObject.global_feature_neurons.to_dense()
	print("global_feature_neurons_dense = ", global_feature_neurons_dense)
	'''
	
	#identify first activated column(s) in prediction phase:
	concept_columns_indices = None #not currently used (target connection neurons have already been activated)
	concept_columns_feature_indices = None	#not currently used (target connection neurons have already been activated)
	
	observed_columns_dict = sequence_observed_columns.observed_columns_dict  # key: lemma, value: ObservedColumn	#every observed column in inference (seed and prediction phases)
			
	#predict next tokens;
	for wordPredictionIndex in range(num_prediction_tokens):
		sequenceWordIndex = num_seed_tokens + wordPredictionIndex
		featurePredictionTargetMatch, concept_columns_indices, concept_columns_feature_indices = process_column_inference_prediction(sequence_observed_columns.databaseNetworkObject, observed_columns_dict, wordPredictionIndex, sequenceWordIndex, words_doc, lemmas_doc, concept_columns_indices, concept_columns_feature_indices, concept_mask, sequence_observed_columns.columns_index_sequence_word_index_dict)
		


def process_column_inference_prediction(databaseNetworkObject, observed_columns_dict, wordPredictionIndex, sequenceWordIndex, words_doc, lemmas_doc, concept_columns_indices, concept_columns_feature_indices, concept_mask, columns_index_sequence_word_index_dict):
	
	#print(f"process_column_inference_prediction: {wordPredictionIndex}; concept_columns_indices = ", concept_columns_indices)

	global_feature_neurons_activation = databaseNetworkObject.global_feature_neurons[array_index_properties_activation]

	if(wordPredictionIndex > 0):
		# Refresh the observed columns dictionary for each new sequence
		observed_columns_sequence_candidate_index_dict = {}  # key: sequence candidate index, value: ObservedColumn	#used to populate sequence feature connection arrays based on observed columns (i does not correspond to sequence word index as assumed by observed_columns_sequence_word_index_dict)

		#populate sequence observed columns;
		words = []
		lemmas = []
		concept_columns_indices_list = concept_columns_indices.tolist()
		for i, concept_index in enumerate(concept_columns_indices_list):
			lemma = databaseNetworkObject.concept_columns_list[concept_index]
			word = lemma	#same for concepts (not used)
			lemmas.append(lemma)
			words.append(word)
			# Load observed column from disk or create new one
			observed_column = GIAANNproto_databaseNetwork.load_or_create_observed_column(databaseNetworkObject, concept_index, lemma, sequenceWordIndex)
			observed_columns_dict[lemma] = observed_column
			observed_columns_sequence_candidate_index_dict[i] = observed_column
		sequence_observed_columns_prediction = SequenceObservedColumnsInferencePrediction(databaseNetworkObject, words, lemmas, observed_columns_dict, observed_columns_sequence_candidate_index_dict)
		
		#process features (activate global target neurons);
		#process_features_active_predict(global_feature_neurons_activation, sequence_observed_columns_prediction, concept_columns_indices, concept_columns_feature_indices)
		process_features_active_predict_single(global_feature_neurons_activation, sequence_observed_columns_prediction, concept_columns_indices, concept_columns_feature_indices)

		#decrement activations;
		if(useActivationDecrement):
			#decrement activation after each prediction interval
			global_feature_neurons_activation = GIAANNproto_sparseTensors.subtract_value_from_sparse_tensor_values(global_feature_neurons_activation, activationDecrementPerPredictedColumn)
		if(deactivateNeuronsUponPrediction):
			if(useSANI):
				indices_to_update_list = []
				for segment_index in range(array_number_of_segments):
					number_features_predicted = concept_columns_indices.shape[0]
					index_to_update = pt.stack([pt.tensor([segment_index]*number_features_predicted), concept_columns_indices, concept_columns_feature_indices.squeeze(dim=0)], dim=0)
					indices_to_update_list.append(index_to_update)
				indices_to_update = pt.stack(indices_to_update_list, dim=0).squeeze(dim=2)
				global_feature_neurons_activation = GIAANNproto_sparseTensors.modify_sparse_tensor(global_feature_neurons_activation, indices_to_update, 0)
				#global_feature_neurons_activation[concept_columns_indices, concept_columns_feature_indices] = 0	
			else:
				segment_indices = pt.zeros_like(concept_columns_indices)
				indices_to_update = pt.stack([segment_indices, concept_columns_indices, concept_columns_feature_indices.squeeze(dim=0)], dim=0)
				global_feature_neurons_activation = GIAANNproto_sparseTensors.modify_sparse_tensor(global_feature_neurons_activation, indices_to_update, 0)
				#global_feature_neurons_activation[segment_indices, concept_columns_indices, concept_columns_feature_indices] = 0

		databaseNetworkObject.global_feature_neurons = GIAANNproto_sparseTensors.replaceAllSparseTensorElementsAtFirstDimIndex(databaseNetworkObject.global_feature_neurons, global_feature_neurons_activation, array_index_properties_activation)
	else:
		#activation targets have already been activated
		sequence_observed_columns_prediction = SequenceObservedColumnsDraw(databaseNetworkObject, observed_columns_dict)
		
	if(inferencePredictiveNetwork):
		concept_columns_indices_next, concept_columns_feature_indices_next, kc = predictMostActiveFeature(global_feature_neurons_activation, databaseNetworkObject, words_doc, lemmas_doc, wordPredictionIndex, sequenceWordIndex, concept_mask, columns_index_sequence_word_index_dict)	
	else:
		concept_columns_indices_next, concept_columns_feature_indices_next, kc = selectMostActiveFeature(global_feature_neurons_activation)
	
	#print("concept_columns_indices_next = ", concept_columns_indices_next)
	#print("concept_columns_feature_indices_next = ", concept_columns_feature_indices_next)
			
	#compare topk column/feature predictions to doc_predict (target words);
	featurePredictionTargetMatch = False
	#implementation limitation; only works with kf = 1;
	for columnPredictionIndex in range(kc):
		columnIndex = concept_columns_indices_next[columnPredictionIndex]
		columnName = databaseNetworkObject.concept_columns_list[columnIndex]
		observedColumnFeatureIndex = concept_columns_feature_indices_next[columnPredictionIndex, 0]
		if(observedColumnFeatureIndex == feature_index_concept_neuron):
			predictedWord = columnName
		else:
			predictedWord = databaseNetworkObject.concept_features_list[observedColumnFeatureIndex]
		targetWord = words_doc[sequenceWordIndex]
		print("\t columnName = ", columnName, ", sequenceWordIndex = ", sequenceWordIndex, ", wordPredictionIndex = ", wordPredictionIndex, ", targetWord = ", targetWord, ", predictedWord = ", predictedWord)
		if(targetWord == predictedWord):
			featurePredictionTargetMatch = True
			
	#FUTURE: convert global_feature_neurons_activation back to global_feature_neurons for draw
	GIAANNproto_databaseNetworkDraw.visualize_graph(sequence_observed_columns_prediction)
	
	return featurePredictionTargetMatch, concept_columns_indices_next, concept_columns_feature_indices_next
	
def predictMostActiveFeature(global_feature_neurons_activation, databaseNetworkObject, words_doc, lemmas_doc, wordPredictionIndex, sequenceWordIndex, concept_mask, columns_index_sequence_word_index_dict):	

	global kc
	
	#generate targets;
	if(concept_mask[sequenceWordIndex]):
		lemma = lemmas_doc[sequenceWordIndex]
		targetFeatureIndex = databaseNetworkObject.concept_columns_dict[lemma]
	else:
		word = words_doc[sequenceWordIndex]
		targetFeatureIndex = databaseNetworkObject.concept_features_dict[word]
	doc_len = concept_mask.shape[0]
	foundFeature = False
	foundNextColumnIndex = False
	previousColumnIndex = 0
	nextColumnIndex = 0
	for i in range(doc_len):
		if(foundFeature):
			if(not foundNextColumnIndex):
				if(concept_mask[i] != 0):
					nextColumnIndex = columns_index_sequence_word_index_dict[i]
					foundNextColumnIndex = True
		else:
			if(concept_mask[i] != 0):
				previousColumnIndex = columns_index_sequence_word_index_dict[i]
		if(i == sequenceWordIndex):
			foundFeature = True
	targets = pt.zeros(databaseNetworkObject.c, databaseNetworkObject.f)
	targets[previousColumnIndex, targetFeatureIndex] = 1
	if(not debugConceptFeaturesOccurFirstInSubsequence):	#or if multipleTargets
		targets[nextColumnIndex, targetFeatureIndex] = 1
	
	#print("previousColumnIndex = ", previousColumnIndex)
	#print("nextColumnIndex = ", nextColumnIndex)
	#print("targetFeatureIndex = ", targetFeatureIndex)
	#print("targets = ", targets)
	
	if(inferencePredictiveNetworkModelMLP):
		concept_columns_indices_next, concept_columns_feature_indices_next = GIAANNproto_predictiveNetworkMLP.nextWordPredictionMLPtrainStep(global_feature_neurons_activation, targets)
	elif(inferencePredictiveNetworkModelTransformer):
		concept_columns_indices_next, concept_columns_feature_indices_next = GIAANNproto_predictiveNetworkTransformer.nextWordPredictionTransformerTrainStep(databaseNetworkObject.global_feature_neurons, databaseNetworkObject.global_feature_connections, targets)

	return concept_columns_indices_next, concept_columns_feature_indices_next, kc


def selectMostActiveFeature(global_feature_neurons_activation):

	global_feature_neurons_activation_all_segments = pt.sum(global_feature_neurons_activation, dim=0)	#sum across all segments 	#TODO: take into account SANI requirements (distal activation must precede proximal activation) 

	#topk column selection;
	concept_columns_activation = pt.sum(global_feature_neurons_activation_all_segments, dim=1)	#sum across all feature activations in columns
	concept_columns_activation = concept_columns_activation.to_dense()	#convert to dense tensor (required for topk)
	if(kcDynamic):
		concept_columns_activation = concept_columns_activation[concept_columns_activation > kcActivationThreshold]	#select kcMax columns above threshold
	concept_columns_activation_topk_concepts = pt.topk(concept_columns_activation, kcMax)
	kc = len(concept_columns_activation_topk_concepts.indices)
	if(kcDynamic and kc < 1):
		print("process_column_prediction kcDynamic error: kc < 1; cannot continue to predict columns; consider disabling kcDynamic for debug")
		exit()

	#top feature selection;
	if(kc==1):
		topk_concept_columns_activation = global_feature_neurons_activation_all_segments[concept_columns_activation_topk_concepts.indices[0]].unsqueeze(0)	#select topk concept indices
	else:
		topk_concept_columns_activation = GIAANNproto_sparseTensors.slice_sparse_tensor_multi(global_feature_neurons_activation_all_segments, 0, concept_columns_activation_topk_concepts.indices)	#select topk concept indices
	topk_concept_columns_activation = topk_concept_columns_activation.to_dense()
	topk_concept_columns_activation_topk_features = pt.topk(topk_concept_columns_activation, kf, dim=1)

	#print("concept_columns_activation_topk_concepts.values = ", concept_columns_activation_topk_concepts.values)	
	#print("concept_columns_activation_topk_concepts.indices = ", concept_columns_activation_topk_concepts.indices)	
	
	#print("topk_concept_columns_activation_topk_features.values = ", topk_concept_columns_activation_topk_features.values)	
	#print("topk_concept_columns_activation_topk_features.indices = ", topk_concept_columns_activation_topk_features.indices)	

	concept_columns_indices_next = concept_columns_activation_topk_concepts.indices
	concept_columns_feature_indices_next = topk_concept_columns_activation_topk_features.indices
	
	#print("concept_columns_indices_next = ", concept_columns_indices_next)
	#print("concept_columns_feature_indices_next = ", concept_columns_feature_indices_next)
			
	return concept_columns_indices_next, concept_columns_feature_indices_next, kc


#first dim cs1 restricted to a single token
def process_features_active_predict_single(global_feature_neurons_activation, sequence_observed_columns_prediction, concept_columns_indices, concept_columns_feature_indices):
	
	feature_neurons_active = global_feature_neurons_activation[array_index_segment_internal_column] 		#select last (most proximal) segment activation	#TODO: checkthis
	feature_neurons_active = feature_neurons_active[concept_columns_indices.squeeze().item()]	#select columns
	feature_neurons_active = feature_neurons_active[concept_columns_feature_indices.squeeze().squeeze().item()]	#select features
	#print("feature_neurons_active = ", feature_neurons_active)
	
	#target neuron activation dependence on connection strength;
	#print("feature_neurons_active.shape = ", feature_neurons_active.shape)
	feature_connections = sequence_observed_columns_prediction.feature_connections[array_index_properties_strength]
	feature_connections = GIAANNproto_sparseTensors.slice_sparse_tensor(feature_connections, 1, 0)	#sequence concept index dimension (not used)
	feature_connections = GIAANNproto_sparseTensors.slice_sparse_tensor(feature_connections, 1, concept_columns_feature_indices.squeeze())
	#print("feature_connections.shape = ", feature_connections.shape)
	feature_neurons_target_activation = feature_neurons_active * feature_connections
	
	#update the activations of the target nodes;
	global_feature_neurons_activation += feature_neurons_target_activation*j1
	return global_feature_neurons_activation
		
#first dim cs1 restricted to a single token (or candiate set of tokens).
def process_features_active_predict(global_feature_neurons_activation, sequence_observed_columns_prediction, concept_columns_indices, concept_columns_feature_indices):
	
	feature_neurons_active = global_feature_neurons_activation[array_index_segment_internal_column]		#select last (most proximal) segment activation	#TODO: checkthis
	feature_neurons_active = feature_neurons_active[concept_columns_indices]	#select columns
	feature_neurons_active = feature_neurons_active[concept_columns_feature_indices]	#select features
	
	#target neuron activation dependence on connection strength;
	feature_connections = sequence_observed_columns_prediction.feature_connections[array_index_properties_strength]
	feature_connections = GIAANNproto_sparseTensors.slice_sparse_tensor(feature_connections, 1, 0)	#sequence concept index dimension (not used)
	feature_connections = GIAANNproto_sparseTensors.slice_sparse_tensor_multi(feature_connections, 1, concept_columns_feature_indices)
	feature_neurons_target_activation = feature_neurons_active * feature_connections
	
	#update the activations of the target nodes;
	global_feature_neurons_activation += feature_neurons_target_activation*j1
	return global_feature_neurons_activation
	

def getLemmas(doc):
	words = []
	lemmas = []
	pos_tags = []
	
	for token in doc:
		word = token.text.lower()
		lemma = token.lemma_.lower()
		pos = token.pos_  # Part-of-speech tag
		words.append(word)
		lemmas.append(lemma)
		pos_tags.append(pos)
	
	return words, lemmas, pos_tags
	
	


