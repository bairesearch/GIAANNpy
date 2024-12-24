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

def inferenceSavePredictiveNetwork():
	if(inferencePredictiveNetworkModelMLP):
		GIAANNproto_predictiveNetworkMLP.save_model(predictive_network_folder, predictive_network_file_name)
	elif(inferencePredictiveNetworkModelTransformer):
		GIAANNproto_predictiveNetworkTransformer.save_model(predictive_network_folder, predictive_network_file_name)

def initialisePredictiveNetwork(databaseNetworkObject):
	if(inferencePredictiveNetworkModelMLP):
		GIAANNproto_predictiveNetworkMLP.nextWordPredictionMLPcreate(databaseNetworkObject)
	elif(inferencePredictiveNetworkModelTransformer):
		GIAANNproto_predictiveNetworkTransformer.nextWordPredictionTransformerCreate(databaseNetworkObject)


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

def seed_network(sequence_observed_columns, sentenceIndex, doc, first_seed_token_index, num_seed_tokens):
	words, lemmas, pos_tags = getLemmas(doc)
	if(inferenceIncrementallySeedNetwork):
		print("\t seed_network: seed_token_index = ", first_seed_token_index, ", word = ", words[first_seed_token_index])
	else:
		print("\t seed_network: first_seed_token_index = ", first_seed_token_index, ", words = ", words[first_seed_token_index:num_seed_tokens])
	GIAANNproto_databaseNetworkTrain.process_concept_words(sequence_observed_columns, sentenceIndex, doc, words, lemmas, pos_tags, train=False, first_seed_token_index=first_seed_token_index, num_seed_tokens=num_seed_tokens)

	if(inferenceDecrementActivations):
		if(not inferenceSeedTargetActivationsGlobalFeatureArrays):
			global_feature_neurons_activation = sequence_observed_columns.databaseNetworkObject.global_feature_neurons[array_index_properties_activation]
			global_feature_neurons_activation = GIAANNproto_databaseNetworkTrain.decrementActivation(global_feature_neurons_activation, activationDecrementSeed)
			sequence_observed_columns.databaseNetworkObject.global_feature_neurons = GIAANNproto_sparseTensors.replaceAllSparseTensorElementsAtFirstDimIndex(sequence_observed_columns.databaseNetworkObject.global_feature_neurons, global_feature_neurons_activation, array_index_properties_activation)
	
	if(drawNetworkDuringInferenceSeed):
		#FUTURE: convert global_feature_neurons_activation back to global_feature_neurons for draw
		GIAANNproto_databaseNetworkDraw.visualize_graph(sequence_observed_columns, save=drawNetworkDuringInferenceSave, fileName=drawNetworkDuringInferenceSaveFilenamePrepend+str(first_seed_token_index))


def process_concept_words_inference(sequence_observed_columns, sentenceIndex, doc, doc_seed, doc_predict, num_seed_tokens, num_prediction_tokens):

	print("process_concept_words_inference:")

	sequenceWordIndex = 0
	
	words_doc, lemmas_doc, pos_tags_doc = getLemmas(doc)
	concept_mask, concept_indices, numberConcepts = GIAANNproto_databaseNetworkTrain.createConceptMask(sequence_observed_columns, lemmas_doc)
	
	if(transformerUseInputConnections):
		GIAANNproto_databaseNetwork.generate_global_feature_connections(sequence_observed_columns.databaseNetworkObject)
	
	if(inferenceTrainPredictionNetworkAllSentences):
		num_prediction_tokens = len(doc_predict)
	else:
		#seed network;
		if(inferenceIncrementallySeedNetwork):
			for seed_token_index in range(num_seed_tokens):
				seed_network(sequence_observed_columns, sentenceIndex, doc, seed_token_index, 1)
		else:
			seed_network(sequence_observed_columns, sentenceIndex, doc, 0, num_seed_tokens)

		if(not inferenceSeedTargetActivationsGlobalFeatureArrays):
			# Update observed columns from sequence observed columns
			sequence_observed_columns.update_observed_columns_wrapper()	#convert sequence observed columns feature neuron arrays back to global feature neuron arrays
		
		if(inferencePredictiveNetwork):
			initialisePredictiveNetwork(sequence_observed_columns.databaseNetworkObject)
		
	#identify first activated column(s) in prediction phase:
	if(inferencePredictiveNetwork):
		kcMax = kcNetwork
	else:
		kcMax = 1	#not used
	multiple_sources, previousColumnIndex, nextColumnIndex, targetFeatureIndex, concept_columns_indices, concept_columns_feature_indices = GIAANNproto_databaseNetwork.getTokenConceptFeatureIndexTensor(sequence_observed_columns, words_doc, lemmas_doc, concept_mask, 0, kcMax)
	observed_columns_dict = sequence_observed_columns.observed_columns_dict  # key: lemma, value: ObservedColumn	#every observed column in inference (seed and prediction phases)
	
	#predict next tokens;
	for wordPredictionIndex in range(num_prediction_tokens):
		sequenceWordIndex = num_seed_tokens + wordPredictionIndex
		featurePredictionTargetMatch, concept_columns_indices, concept_columns_feature_indices, multiple_sources = process_column_inference_prediction(sequence_observed_columns, observed_columns_dict, wordPredictionIndex, sequenceWordIndex, words_doc, lemmas_doc, concept_columns_indices, concept_columns_feature_indices, concept_mask, multiple_sources)
		

def process_column_inference_prediction(sequence_observed_columns, observed_columns_dict, wordPredictionIndex, sequenceWordIndex, words_doc, lemmas_doc, concept_columns_indices, concept_columns_feature_indices, concept_mask, multiple_sources):
	
	databaseNetworkObject = sequence_observed_columns.databaseNetworkObject
	
	#print(f"process_column_inference_prediction: {sequenceWordIndex}; concept_columns_indices = ", concept_columns_indices)

	if(inferenceTrainPredictionNetworkAllSentences):
		if(wordPredictionIndex==0 or not inferenceUseNextTokenPredictionsOrTargetsToActivateNextColumnFeatures):
			#activate source token (incremental seed during train)
				#if(wordPredictionIndex == 1) will reactivate first seed token column feature (as it was not saved during wordPredictionIndex==0)
			for concept_index in range(concept_columns_indices.shape[0]):
				seedTokenConceptIndex = concept_columns_indices[concept_index].item()
				seedTokenFeatureIndex = concept_columns_feature_indices[concept_index].squeeze().item()
				dimensions = [array_index_properties_activation, array_index_segment_first, seedTokenConceptIndex, seedTokenFeatureIndex]
				sequence_observed_columns.databaseNetworkObject.global_feature_neurons = GIAANNproto_sparseTensors.addElementValueToSparseTensor(sequence_observed_columns.databaseNetworkObject.global_feature_neurons, dimensions, j1)
			
	global_feature_neurons_activation = databaseNetworkObject.global_feature_neurons[array_index_properties_activation]
	global_feature_neurons_strength = databaseNetworkObject.global_feature_neurons[array_index_properties_strength]
	if(transformerUseInputConnections):
		global_feature_connections_activation = databaseNetworkObject.global_feature_connections[array_index_properties_activation]
	else:
		global_feature_connections_activation = None
		
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
		
		#decrement activations;
		if(inferenceDecrementActivations):
			#decrement activation after each prediction interval
			global_feature_neurons_activation = GIAANNproto_databaseNetworkTrain.decrementActivation(global_feature_neurons_activation, activationDecrementPerPredictedToken)
			if(transformerUseInputConnections):
				global_feature_connections_activation = GIAANNproto_databaseNetworkTrain.decrementActivation(global_feature_connections_activation, activationDecrementPerPredictedToken)
				
		#process features (activate global target neurons);
		if(multiple_sources):
			global_feature_neurons_activation, global_feature_connections_activation = process_features_active_predict_multi(databaseNetworkObject, global_feature_neurons_activation, global_feature_connections_activation, sequence_observed_columns_prediction, concept_columns_indices, concept_columns_feature_indices)
		else:
			global_feature_neurons_activation, global_feature_connections_activation = process_features_active_predict_single(databaseNetworkObject, global_feature_neurons_activation, global_feature_connections_activation, sequence_observed_columns_prediction, concept_columns_indices, concept_columns_feature_indices)

		if(inferenceDeactivateNeuronsUponPrediction or inferenceInvertNeuronActivationUponPrediction):
			indices_to_update_list = []
			for conceptIndex in range(concept_columns_indices.shape[0]):
				concept_columns_indices_source = concept_columns_indices[conceptIndex]
				concept_columns_feature_indices_source = concept_columns_feature_indices[conceptIndex].squeeze(dim=0)
				if(useSANI):
					for segment_index in range(array_number_of_segments):
						index_to_update = pt.stack([pt.tensor(segment_index, device=concept_columns_indices_source.device), concept_columns_indices_source, concept_columns_feature_indices_source], dim=0)
						indices_to_update_list.append(index_to_update)
				else:
					indices_to_update = pt.stack([pt.tensor(array_index_segment_first, device=concept_columns_indices_source.device), concept_columns_indices_source, concept_columns_feature_indices_source], dim=0)
					indices_to_update_list.append(indices_to_update)
			indices_to_update = pt.stack(indices_to_update_list, dim=0)
			if(inferenceDeactivateNeuronsUponPrediction):
				modifier = 0
			elif(inferenceInvertNeuronActivationUponPrediction):
				modifier = inferenceInvertNeuronActivationUponPredictionLevel
			global_feature_neurons_activation = GIAANNproto_sparseTensors.modify_sparse_tensor(global_feature_neurons_activation, indices_to_update, modifier, multiply=inferenceInvertNeuronActivationUponPrediction)
			#global_feature_neurons_activation[concept_columns_indices, concept_columns_feature_indices] = 0	

		databaseNetworkObject.global_feature_neurons = GIAANNproto_sparseTensors.replaceAllSparseTensorElementsAtFirstDimIndex(databaseNetworkObject.global_feature_neurons, global_feature_neurons_activation, array_index_properties_activation)
		if(transformerUseInputConnections):
			databaseNetworkObject.global_feature_connections = GIAANNproto_sparseTensors.replaceAllSparseTensorElementsAtFirstDimIndex(databaseNetworkObject.global_feature_connections, global_feature_connections_activation, array_index_properties_activation)

	else:
		#activation targets have already been activated
		sequence_observed_columns_prediction = SequenceObservedColumnsDraw(databaseNetworkObject, observed_columns_dict)
	
	if(debugInferencePredictionActivationAccumulation):
		global_feature_neurons_temp = databaseNetworkObject.global_feature_neurons.to_dense()
		print("global_feature_neurons_temp = ", global_feature_neurons_temp)

	if(inferencePredictiveNetwork):
		concept_columns_indices_next, concept_columns_feature_indices_next, multiple_sources, kc, concept_columns_indices_pred, concept_columns_feature_indices_pred = predictMostActiveFeature(sequence_observed_columns, global_feature_neurons_activation, databaseNetworkObject, words_doc, lemmas_doc, wordPredictionIndex, sequenceWordIndex, concept_mask)	
	else:
		concept_columns_indices_next, concept_columns_feature_indices_next, multiple_sources, kc, concept_columns_indices_pred, concept_columns_feature_indices_pred = selectMostActiveFeature(global_feature_neurons_activation, global_feature_neurons_strength)
	
	featurePredictionTargetMatch = False
	if(printPredictionsDuringInferencePredict):
		#compare topk column/feature predictions to doc_predict (target words);
		#implementation limitation; only works with kf = 1;
		for columnPredictionIndex in range(concept_columns_indices_pred.shape[0]):
			columnIndex = concept_columns_indices_pred[columnPredictionIndex]
			columnName = databaseNetworkObject.concept_columns_list[columnIndex]
			observedColumnFeatureIndex = concept_columns_feature_indices_pred[columnPredictionIndex, 0]
			if(observedColumnFeatureIndex == feature_index_concept_neuron):
				predictedWord = columnName
			else:
				predictedWord = databaseNetworkObject.concept_features_list[observedColumnFeatureIndex]
			targetWord = words_doc[sequenceWordIndex]
			print("\t columnName = ", columnName, ", sequenceWordIndex = ", sequenceWordIndex, ", wordPredictionIndex = ", wordPredictionIndex, ", targetWord = ", targetWord, ", predictedWord = ", predictedWord)
			if(targetWord == predictedWord):
				featurePredictionTargetMatch = True
	
	if(drawNetworkDuringInferencePredict):
		#FUTURE: convert global_feature_neurons_activation back to global_feature_neurons for draw
		GIAANNproto_databaseNetworkDraw.visualize_graph(sequence_observed_columns_prediction, save=drawNetworkDuringInferenceSave, fileName=drawNetworkDuringInferenceSaveFilenamePrepend+str(sequenceWordIndex))

	return featurePredictionTargetMatch, concept_columns_indices_next, concept_columns_feature_indices_next, multiple_sources

def predictMostActiveFeature(sequence_observed_columns, global_feature_neurons_activation, databaseNetworkObject, words_doc, lemmas_doc, wordPredictionIndex, sequenceWordIndex, concept_mask):		
	#generate targets;
	multiple_sources, previousColumnIndex, nextColumnIndex, targetFeatureIndex, concept_columns_indices_prev, concept_columns_feature_indices_prev = GIAANNproto_databaseNetwork.getTokenConceptFeatureIndexTensor(sequence_observed_columns, words_doc, lemmas_doc, concept_mask, sequenceWordIndex, kcNetwork)
	
	if(inferencePredictiveNetworkModelTransformer):
		targets_c = pt.zeros(databaseNetworkObject.c)
		targets_f = pt.zeros(databaseNetworkObject.f)
		targets_c[previousColumnIndex] = 1
		targets_f[targetFeatureIndex] = 1
		if(multiple_sources):
			targets_c[nextColumnIndex] = 1	
	elif(inferencePredictiveNetworkModelMLP):
		targets = pt.zeros(databaseNetworkObject.c, databaseNetworkObject.f)
		targets[previousColumnIndex, targetFeatureIndex] = 1
		if(multiple_sources):
			targets[nextColumnIndex, targetFeatureIndex] = 1

	if(inferencePredictiveNetworkModelMLP):
		concept_columns_indices_pred, concept_columns_feature_indices_pred = GIAANNproto_predictiveNetworkMLP.nextWordPredictionMLPtrainStep(global_feature_neurons_activation, targets)
	elif(inferencePredictiveNetworkModelTransformer):
		concept_columns_indices_pred, concept_columns_feature_indices_pred = GIAANNproto_predictiveNetworkTransformer.nextWordPredictionTransformerTrainStep(databaseNetworkObject.global_feature_neurons, databaseNetworkObject.global_feature_connections, targets_c, targets_f)

	if(inferenceUseNextTokenPredictionsOrTargetsToActivateNextColumnFeatures):
		concept_columns_indices_next = concept_columns_indices_pred
		concept_columns_feature_indices_next = concept_columns_feature_indices_pred
		kc = kcNetwork
		if(kc == 1 and kf == 1):
			multiple_sources = False
		else:
			multiple_sources = True
	else:
		#while exclusively training predictive network; use targets rather than next token predictions when activating database network
		concept_columns_indices_next = concept_columns_indices_prev
		concept_columns_feature_indices_next = concept_columns_feature_indices_prev
		if(multiple_sources):
			kc = 2
		else:
			kc = 1
		assert kf==1
		
	return concept_columns_indices_next, concept_columns_feature_indices_next, multiple_sources, kc, concept_columns_indices_pred, concept_columns_feature_indices_pred


def selectMostActiveFeature(global_feature_neurons_activation, global_feature_neurons_strength):

	global_feature_neurons_activation_all_segments = pt.sum(global_feature_neurons_activation, dim=0)	#sum across all segments 	#TODO: take into account SANI requirements (distal activation must precede proximal activation) 
	global_feature_neurons_strength_all_segments = pt.sum(global_feature_neurons_strength, dim=0)	#sum across all segments 	#TODO: take into account SANI requirements (distal activation must precede proximal activation) 

	#topk column selection;
	concept_columns_activation = pt.sum(global_feature_neurons_activation_all_segments, dim=1)	#sum across all feature activations in columns
	concept_columns_activation = concept_columns_activation.to_dense()	#convert to dense tensor (required for topk)
	if(inferenceNormaliseColumnSelectionByFeatureConnections):
		concept_columns_activation_total_connections = pt.sum(global_feature_neurons_strength_all_segments, dim=1)	#sum across all feature activations in columns
		concept_columns_activation_total_connections = concept_columns_activation_total_connections.to_dense()
		if(not inferenceNormaliseColumnSelectionByFeatureConnectionsStrength):
			concept_columns_activation_total_connections = (concept_columns_activation_total_connections > 0).float()
		concept_columns_activation = concept_columns_activation / concept_columns_activation_total_connections
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
	if(inferenceNormaliseFeatureSelectionByFeatureConnections):
		if(kc==1):
			topk_concept_columns_strength = global_feature_neurons_strength_all_segments[concept_columns_activation_topk_concepts.indices[0]].unsqueeze(0)	#select topk concept indices
		else:
			topk_concept_columns_strength = GIAANNproto_sparseTensors.slice_sparse_tensor_multi(global_feature_neurons_strength_all_segments, 0, concept_columns_activation_topk_concepts.indices)	#select topk concept indices
		topk_concept_columns_strength = topk_concept_columns_strength.to_dense()
		if(not inferenceNormaliseFeatureSelectionByFeatureConnectionsStrength):
			topk_concept_columns_strength = (topk_concept_columns_strength > 0).float()
		topk_concept_columns_activation = topk_concept_columns_activation / topk_concept_columns_strength
	topk_concept_columns_activation_topk_features = pt.topk(topk_concept_columns_activation, kf, dim=1)

	concept_columns_indices_next = concept_columns_activation_topk_concepts.indices
	concept_columns_feature_indices_next = topk_concept_columns_activation_topk_features.indices
	
	if(kc > 1 or kf > 1):
		multiple_sources = True
	else:
		multiple_sources = False
		
	return concept_columns_indices_next, concept_columns_feature_indices_next, multiple_sources, kc, concept_columns_indices_next, concept_columns_feature_indices_next


#first dim cs1 restricted to a single token
def process_features_active_predict_single(databaseNetworkObject, global_feature_neurons_activation, global_feature_connections_activation, sequence_observed_columns_prediction, concept_columns_indices, concept_columns_feature_indices):
	
	feature_neurons_active = global_feature_neurons_activation[array_index_segment_internal_column] 		#select last (most proximal) segment activation	#TODO: checkthis
	feature_neurons_active = feature_neurons_active[concept_columns_indices.squeeze().item()]	#select columns
	feature_neurons_active = feature_neurons_active[concept_columns_feature_indices.squeeze().squeeze().item()]	#select features
	#print("feature_neurons_active = ", feature_neurons_active)
	
	#target neuron activation dependence on connection strength;
	#print("feature_neurons_active.shape = ", feature_neurons_active.shape)
	feature_connections = sequence_observed_columns_prediction.feature_connections[array_index_properties_strength]
	feature_connections = GIAANNproto_sparseTensors.slice_sparse_tensor(feature_connections, 1, 0)	#sequence concept index dimension (not used)
	if(inferencePredictiveNetwork and not useGPUsparse):
		concept_columns_feature_indices = concept_columns_feature_indices.to(deviceSparse)
	feature_connections = GIAANNproto_sparseTensors.slice_sparse_tensor(feature_connections, 1, concept_columns_feature_indices.squeeze())
	#print("feature_connections.shape = ", feature_connections.shape)
	feature_neurons_target_activation = feature_neurons_active * feature_connections

	if(inferenceActivationFunction):
		feature_neurons_target_activation = GIAANNproto_databaseNetworkTrain.activation_function(feature_neurons_target_activation)
	else:
		feature_neurons_target_activation = feature_neurons_target_activation*j1
		
	#update the activations of the target nodes;
	global_feature_neurons_activation += feature_neurons_target_activation
	
	if(transformerUseInputConnections):
		feature_neurons_target_activation = GIAANNproto_sparseTensors.expand_sparse_tensor(feature_neurons_target_activation, 1, concept_columns_indices.squeeze(), new_dim_size=databaseNetworkObject.c)
		feature_neurons_target_activation = GIAANNproto_sparseTensors.expand_sparse_tensor(feature_neurons_target_activation, 2, concept_columns_feature_indices.squeeze(), new_dim_size=databaseNetworkObject.f)
		global_feature_connections_activation = global_feature_connections_activation + feature_neurons_target_activation

	return global_feature_neurons_activation, global_feature_connections_activation
		
#first dim cs1 restricted to a candiate set of tokens.
def process_features_active_predict_multi(databaseNetworkObject, global_feature_neurons_activation, global_feature_connections_activation, sequence_observed_columns_prediction, concept_columns_indices, concept_columns_feature_indices):
	#print("process_features_active_predict_multi:")
	for conceptIndex in range(concept_columns_indices.shape[0]):
		concept_columns_indices_source = concept_columns_indices[conceptIndex].unsqueeze(dim=0)
		concept_columns_feature_indices_source = concept_columns_feature_indices[conceptIndex].unsqueeze(dim=0)
		#print("concept_columns_indices_source = ", concept_columns_indices_source)
		#print("concept_columns_feature_indices_source = ", concept_columns_feature_indices_source)
		global_feature_neurons_activation, global_feature_connections_activation = process_features_active_predict_single(databaseNetworkObject, global_feature_neurons_activation, global_feature_connections_activation, sequence_observed_columns_prediction, concept_columns_indices_source, concept_columns_feature_indices_source)
	
	return global_feature_neurons_activation, global_feature_connections_activation
	
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

