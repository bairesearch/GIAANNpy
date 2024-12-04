"""GIAANNproto_databaseNetworkTrain.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_main.py

# Usage:
see GIAANNproto_main.py

# Description:
GIA ANN proto database Network Train

"""

import torch as pt

from GIAANNproto_globalDefs import *
import GIAANNproto_sparseTensors

# Define the SequenceObservedColumns class
class SequenceObservedColumns:
	"""
	Contains sequence observed columns object arrays which stack a feature subset of the observed columns object arrays for the current sequence.
	"""
	def __init__(self, databaseNetworkObject, words, lemmas, observed_columns_dict, observed_columns_sequence_word_index_dict):
		#note cs may be slightly longer than number of unique columns in the sequence, if there are multiple instances of the same concept/noun lemma in the sequence
	
		self.databaseNetworkObject = databaseNetworkObject
		self.observed_columns_dict = observed_columns_dict	# key: lemma, value: ObservedColumn
		self.observed_columns_sequence_word_index_dict = observed_columns_sequence_word_index_dict	# key: sequence word index, value: ObservedColumn

		if(sequenceObservedColumnsMatchSequenceWords):
			self.cs = len(observed_columns_sequence_word_index_dict)

			self.columns_index_sequence_word_index_dict = {}	#key: sequence word index, value: concept_index
			self.sequence_observed_columns_dict = {}	#key: sequence_concept_index, value: observed_column 
			self.concept_indices_in_observed_list = []	#value: concept index
			for sequence_concept_index, (sequence_word_index, observed_column) in enumerate(self.observed_columns_sequence_word_index_dict.items()):
				self.columns_index_sequence_word_index_dict[sequence_word_index] = observed_column.concept_index
				self.sequence_observed_columns_dict[sequence_concept_index] = observed_column
				self.concept_indices_in_observed_list.append(observed_column.concept_index)
			self.concept_indices_in_sequence_observed_tensor = pt.tensor(self.concept_indices_in_observed_list, dtype=pt.long)
		else:
			self.cs = len(observed_columns_dict) 

			self.columns_index_sequence_word_index_dict = {}	#key: sequence word index, value: concept_index
			for idx, (sequence_word_index, observed_column) in enumerate(observed_columns_sequence_word_index_dict.items()):
				self.columns_index_sequence_word_index_dict[sequence_word_index] = observed_column.concept_index

			# Map from concept names to indices in sequence arrays
			self.concept_indices_in_observed_list = []
			self.concept_name_to_index = {}	# key: lemma, value: sequence_concept_index
			self.index_to_concept_name = {}	# key: sequence_concept_index, value: lemma
			self.observed_columns_dict2 = {}	# key: sequence_concept_index, value: ObservedColumn
			for idx, (lemma, observed_column) in enumerate(observed_columns_dict.items()):
				self.concept_indices_in_observed_list.append(observed_column.concept_index)
				self.concept_name_to_index[lemma] = idx
				self.index_to_concept_name[idx] = lemma
				self.observed_columns_dict2[idx] = observed_column
			self.concept_indices_in_sequence_observed_tensor = pt.tensor(self.concept_indices_in_observed_list, dtype=pt.long)
				
		self.feature_neuron_changes = [None]*self.cs
			
		# Collect all feature words from observed columns
		self.words = words
		self.lemmas = lemmas
		#identify feature indices from complete ObservedColumns.featureNeurons or globalFeatureNeurons feature lists currently stored in SequenceObservedColumns.feature_neurons	#required for useInference
		observed_column = list(observed_columns_dict.values())[0]	#all features (including words) are identical per observed column
		self.feature_words, self.feature_indices_in_observed_tensor, self.f_idx_tensor = self.identifyObservedColumnFeatureWords(words, lemmas, observed_column)

		if(sequenceObservedColumnsUseSequenceFeaturesOnly):
			self.fs = self.feature_indices_in_observed_tensor.shape[0]
		else:
			self.fs = len(self.feature_words)
		self.feature_word_to_index = {}
		self.index_to_feature_word = {}
		for idx, feature_word in enumerate(self.feature_words):
			self.feature_word_to_index[feature_word] = idx
			self.index_to_feature_word[idx] = feature_word

		# Initialize arrays
		self.feature_neurons = self.initialiseFeatureNeuronsSequence(self.cs, self.fs)
		self.feature_connections = self.initialiseFeatureConnectionsSequence(self.cs, self.fs)

		# Populate arrays with data from observed_columns_dict
		if(sequenceObservedColumnsMatchSequenceWords):
			self.populate_arrays(words, lemmas, self.sequence_observed_columns_dict)
		else:
			self.populate_arrays(words, lemmas, self.observed_columns_dict2)

			
	def identifyObservedColumnFeatureWords(self, words, lemmas, observed_column):
		if(sequenceObservedColumnsUseSequenceFeaturesOnly):
			feature_words = []
			feature_indices_in_observed = []
			#print("\nidentifyObservedColumnFeatureWords: words = ", len(words))
			for wordIndex, (word, lemma) in enumerate(zip(words, lemmas)):
				feature_word = word
				feature_lemma = lemma
				if(useDedicatedConceptNames and wordIndex in self.observed_columns_sequence_word_index_dict):	
					if(useDedicatedConceptNames2):
						#only provide 1 observed_column to identifyObservedColumnFeatureWords (therefore this condition will only be triggered once when when feature_lemma == observed_column.concept_name of some arbitrary concept column. Once triggered a singular artificial variableConceptNeuronFeatureName will be added)
						feature_words.append(variableConceptNeuronFeatureName)
					feature_indices_in_observed.append(feature_index_concept_neuron)
					#print("concept node found = ", feature_lemma)
				elif(feature_word in observed_column.feature_word_to_index):
					feature_words.append(feature_word)
					feature_indices_in_observed.append(observed_column.feature_word_to_index[feature_word])
			if(not sequenceObservedColumnsMatchSequenceWords):
				feature_indices_in_observed = self.removeDuplicates(feature_indices_in_observed)
				feature_words = self.removeDuplicates(feature_words)
			feature_indices_in_observed_tensor = pt.tensor(feature_indices_in_observed, dtype=pt.long)
		else:
			feature_words = observed_column.feature_word_to_index.keys()
			feature_indices_in_observed_tensor = pt.tensor(list(observed_column.feature_word_to_index.values()), dtype=pt.long)
		
		if(sequenceObservedColumnsUseSequenceFeaturesOnly and sequenceObservedColumnsMatchSequenceWords):
			f_idx_tensor = pt.arange(len(feature_words), dtype=pt.long)
		else:
			feature_word_to_index = {}
			for idx, feature_word in enumerate(feature_words):
				feature_word_to_index[feature_word] = idx
			f_idx_tensor = pt.tensor([feature_word_to_index[fw] for fw in feature_words], dtype=pt.long)
		
		return feature_words, feature_indices_in_observed_tensor, f_idx_tensor
		
	def getObservedColumnFeatureIndices(self):
		return self.feature_indices_in_observed_tensor, self.f_idx_tensor
	
	def removeDuplicates(self, lst):
		#python requires ordered sets
		lst = list(dict.fromkeys(lst))
		return lst
				
	@staticmethod
	def initialiseFeatureNeuronsSequence(cs, fs):
		feature_neurons = pt.zeros(array_number_of_properties, array_number_of_segments, cs, fs, dtype=array_type)
		return feature_neurons

	@staticmethod
	def initialiseFeatureConnectionsSequence(cs, fs):
		feature_connections = pt.zeros(array_number_of_properties, array_number_of_segments, cs, fs, cs, fs, dtype=array_type)
		return feature_connections
	
	def populate_arrays(self, words, lemmas, sequence_observed_columns_dict):
		#print("\n\n\n\n\npopulate_arrays:")
		
		# Collect indices and data for feature neurons
		c_idx_list = []
		f_idx_list = []
		feature_list_indices = []
		feature_list_values = []
					
		for c_idx, observed_column in sequence_observed_columns_dict.items():
			feature_indices_in_observed, f_idx_tensor = self.getObservedColumnFeatureIndices()

			num_features = len(f_idx_tensor)

			c_idx_list.append(pt.full((num_features,), c_idx, dtype=pt.long))
			f_idx_list.append(f_idx_tensor)

			if lowMem:
				feature_neurons = observed_column.feature_neurons.coalesce()
			else:
				feature_neurons = GIAANNproto_sparseTensors.slice_sparse_tensor(self.databaseNetworkObject.global_feature_neurons, 2, observed_column.concept_index)	
			
			# Get indices and values from sparse tensor
			indices = feature_neurons.indices()
			values = feature_neurons.values()
			
			filter_feature_indices = pt.nonzero(indices[2].unsqueeze(1) == feature_indices_in_observed, as_tuple=True)
			filtered_indices = indices[:, filter_feature_indices[0]]
			filtered_f_idx_tensor = f_idx_tensor[filter_feature_indices[1]]
			filtered_values = values[filter_feature_indices[0]]
			# Adjust indices
			filtered_indices[0] = filtered_indices[0]  # properties
			filtered_indices[1] = filtered_indices[1]  # types
			filtered_indices[2] = filtered_f_idx_tensor
			filtered_indices = pt.cat([filtered_indices[0:2], pt.full_like(filtered_indices[2:3], c_idx), filtered_indices[2:3]], dim=0)	#insert dim3 for c_idx
			feature_list_indices.append(filtered_indices)
			feature_list_values.append(filtered_values)
	   
		# Combine indices and values
		if feature_list_indices:
			combined_indices = pt.cat(feature_list_indices, dim=1)
			combined_values = pt.cat(feature_list_values, dim=0)
			# Create sparse tensor
			self.feature_neurons = pt.sparse_coo_tensor(combined_indices, combined_values, size=self.feature_neurons.size(), dtype=array_type).to_dense()
			self.feature_neurons_original = self.feature_neurons.clone()

		# Now handle connections
		connection_indices_list = []
		connection_values_list = []
		
		for c_idx, observed_column in sequence_observed_columns_dict.items():
			feature_indices_in_observed, f_idx_tensor = self.getObservedColumnFeatureIndices()

			# Get indices and values from sparse tensor
			feature_connections = observed_column.feature_connections.coalesce()
			indices = feature_connections.indices()
			values = feature_connections.values()

			for other_c_idx, other_observed_column in sequence_observed_columns_dict.items():
				other_feature_indices_in_observed, other_f_idx_tensor = self.getObservedColumnFeatureIndices()
				other_concept_index = other_observed_column.concept_index
				#print("\tother_concept_index = ", other_concept_index)

				# Create meshgrid of feature indices
				feature_idx_obs_mesh, other_feature_idx_obs_mesh = pt.meshgrid(feature_indices_in_observed, other_feature_indices_in_observed, indexing='ij')
				f_idx_mesh, other_f_idx_mesh = pt.meshgrid(f_idx_tensor, other_f_idx_tensor, indexing='ij')

				# Flatten the meshgrid indices
				feature_idx_obs_flat = feature_idx_obs_mesh.reshape(-1)
				other_feature_idx_obs_flat = other_feature_idx_obs_mesh.reshape(-1)
				f_idx_flat = f_idx_mesh.reshape(-1)
				other_f_idx_flat = other_f_idx_mesh.reshape(-1)
				
				# Filter indices for the desired features and concepts
				other_concept_index_expanded = pt.full(feature_idx_obs_flat.size(), fill_value=other_concept_index, dtype=pt.long)
				#print("feature_idx_obs_flat.shape = ", feature_idx_obs_flat.shape)
				filter_feature_indices2 = indices[2].unsqueeze(1) == feature_idx_obs_flat		
				filter_feature_indices3 = indices[3].unsqueeze(1) == other_concept_index_expanded
				filter_feature_indices4 = indices[4].unsqueeze(1) == other_feature_idx_obs_flat
				combined_condition = filter_feature_indices2 & filter_feature_indices3 & filter_feature_indices4
				filter_feature_indices = pt.nonzero(combined_condition, as_tuple=True)
				filtered_indices = indices[:, filter_feature_indices[0]]
				filtered_values = values[filter_feature_indices[0]]
				filtered_f_idx_tensor = f_idx_flat[filter_feature_indices[1]]
				filtered_other_f_idx_tensor = other_f_idx_flat[filter_feature_indices[1]]
						
				# Create tensors for concept indices
				c_idx_flat = pt.full_like(f_idx_flat, c_idx, dtype=pt.long)
				other_c_idx_flat = pt.full_like(other_f_idx_flat, other_c_idx, dtype=pt.long)
				filtered_other_c_idx_flat = other_c_idx_flat[filter_feature_indices[1]]
				
				# Adjust indices
				filtered_indices[0] = filtered_indices[0]  # properties
				filtered_indices[1] = filtered_indices[1]  # types
				filtered_indices[2] = filtered_f_idx_tensor
				filtered_indices[3] = filtered_other_c_idx_flat
				filtered_indices[4] = filtered_other_f_idx_tensor
				filtered_indices = pt.cat([filtered_indices[0:2], pt.full_like(filtered_indices[2:3], c_idx), filtered_indices[2:]], dim=0)	#insert dim3 for c_idx
				connection_indices_list.append(filtered_indices)
				connection_values_list.append(filtered_values)

		# Combine indices and values
		if connection_indices_list:
			combined_indices = pt.cat(connection_indices_list, dim=1)
			combined_values = pt.cat(connection_values_list, dim=0)
			# Create sparse tensor
			self.feature_connections = pt.sparse_coo_tensor(combined_indices, combined_values, size=self.feature_connections.size(), dtype=array_type).to_dense()
			self.feature_connections_original = self.feature_connections.clone()
			
	def update_observed_columns_wrapper(self):
		if(sequenceObservedColumnsMatchSequenceWords):
			#for multiple instances of concept in sequence, need to take the sum of the changes between the existing and modified arrays for each instance of a same concept in the sequence
			self.update_observed_columns(self.sequence_observed_columns_dict, mode="default")
		else:
			self.update_observed_columns(self.observed_columns_dict2, mode="default")
			
	def update_observed_columns(self, sequence_observed_columns_dict, mode):
		# Update observed columns with data from sequence arrays
			
		feature_neurons = self.feature_neurons - self.feature_neurons_original	#convert to changes
		feature_connections = self.feature_connections - self.feature_connections_original	#convert to changes
		
		feature_neurons = feature_neurons.to_sparse()
		feature_connections = feature_connections.to_sparse()
		if(performRedundantCoalesce):
			feature_neurons = feature_neurons.coalesce()
			feature_connections = feature_connections.coalesce()		

		for c_idx, observed_column in sequence_observed_columns_dict.items():
			feature_indices_in_observed, f_idx_tensor = self.getObservedColumnFeatureIndices()
			concept_index = observed_column.concept_index

			if lowMem:
				observed_column.feature_neurons = observed_column.feature_neurons.coalesce()
			else:
				#temporarily store slices of the global_feature_neurons array in the observed_columns (used by update_observed_columns only)
				observed_column.feature_neurons = GIAANNproto_sparseTensors.slice_sparse_tensor(self.databaseNetworkObject.global_feature_neurons, 2, concept_index)	
			
			# feature neurons;
			# Use advanced indexing to get values from feature_neurons
			indices = feature_neurons.indices()
			values = feature_neurons.values()
			#convert indices from SequenceObservedColumns neuronFeatures array indices to ObservedColumns neuronFeatures array indices			
			# Filter indices
			mask = (indices[2] == c_idx) & pt.isin(indices[3], f_idx_tensor)
			filtered_indices = indices[:, mask]
			filtered_values = values[mask]
			# Adjust indices for observed_column
			filtered_indices[2] = filtered_indices[3]  # feature indices	#concept dim is removed from filtered_indices (as it has already been selected out)
			filtered_indices = filtered_indices[0:3]
			#convert indices from sequence observed columns format back to observed columns format
			if(sequenceObservedColumnsUseSequenceFeaturesOnly):
				filtered_indices[2] = feature_indices_in_observed[filtered_indices[2]]	#convert feature indices from sequence observed columns format back to observed columns format
			# Update observed_column's feature_neurons
			if lowMem:
				observed_column.feature_neurons = observed_column.feature_neurons + pt.sparse_coo_tensor(filtered_indices, filtered_values, size=observed_column.feature_neurons.size(), dtype=array_type)
				observed_column.feature_neurons = observed_column.feature_neurons.coalesce()
				observed_column.feature_neurons.values().clamp_(min=0)
			else:
				self.feature_neuron_changes[c_idx] = pt.sparse_coo_tensor(filtered_indices, filtered_values, size=observed_column.feature_neurons.size(), dtype=array_type)
			
			# feature connections;
			indices = feature_connections.indices()
			values = feature_connections.values()
			# Filter indices
			mask = (indices[2] == c_idx)
			filtered_indices = indices[:, mask]
			filtered_values = values[mask]
			# Adjust indices for observed_column
			filtered_indices[2] = filtered_indices[3]  # feature indices	#concept dim is removed from filtered_indices (as it has already been selected out)
			filtered_indices[3] = filtered_indices[4]  # concept indices
			filtered_indices[4] = filtered_indices[5]  # feature indices
			filtered_indices = filtered_indices[0:5]
			#convert indices from sequence observed columns format back to observed columns format
			filtered_indices[3] = self.concept_indices_in_sequence_observed_tensor[filtered_indices[3]]	#convert concept indices from sequence observed columns format back to observed columns format
			if(sequenceObservedColumnsUseSequenceFeaturesOnly):
				filtered_indices[2] = feature_indices_in_observed[filtered_indices[2]]	#convert feature indices from sequence observed columns format back to observed columns format
				filtered_indices[4] = feature_indices_in_observed[filtered_indices[4]]	#convert feature indices from sequence observed columns format back to observed columns format
			# Update observed_column's feature_connections
			observed_column.feature_connections = observed_column.feature_connections + pt.sparse_coo_tensor(filtered_indices, filtered_values, size=observed_column.feature_connections.size(), dtype=array_type)
			observed_column.feature_connections = observed_column.feature_connections.coalesce()
			observed_column.feature_connections.values().clamp_(min=0)
	
		if not lowMem:
			observed_column_feature_neurons_dict = {}
			for c_idx, observed_column in sequence_observed_columns_dict.items():
				concept_index = observed_column.concept_index
				observed_column_feature_neurons_dict[concept_index] = self.feature_neuron_changes[c_idx]
			self.databaseNetworkObject.global_feature_neurons = GIAANNproto_sparseTensors.merge_tensor_slices_sum(self.databaseNetworkObject.global_feature_neurons, observed_column_feature_neurons_dict, 2)



def process_concept_words(sequence_observed_columns, sentenceIndex, doc, words, lemmas, pos_tags, train=True, first_seed_token_index=None, num_seed_tokens=None):
	"""
	For every concept word (lemma) in the sequence, identify every feature neuron in that column that occurs q words before or after the concept word in the sequence, including the concept neuron. This function has been parallelized using PyTorch array operations.
	"""

	if not usePOS:
		q = 5  # Fixed window size when not using POS tags

	# Identify all concept word indices
	#print("\n\nsequence_observed_columns.columns_index_sequence_word_index_dict = ", sequence_observed_columns.columns_index_sequence_word_index_dict)
	concept_mask = pt.tensor([i in sequence_observed_columns.columns_index_sequence_word_index_dict for i in range(len(lemmas))], dtype=pt.bool)
	concept_indices = pt.nonzero(concept_mask).squeeze(1)
	
	#concept_indices may be slightly longer than number of unique columns in sequence, if there are multiple instances of the same concept/noun lemma in the sequence
	
	numberConceptsInSequence = concept_indices.shape[0]	#concept_indices.numel()	
	if numberConceptsInSequence == 0:
		return  # No concept words to process

	if usePOS:
		# Sort concept_indices
		concept_indices_sorted = concept_indices.sort().values
		
		# Find previous concept indices for each concept index
		prev_concept_positions = pt.searchsorted(concept_indices_sorted, concept_indices, right=False) - 1
		prev_concept_exists = prev_concept_positions >= 0
		prev_concept_positions = prev_concept_positions.clamp(min=0)
		prev_concept_indices = pt.where(prev_concept_exists, concept_indices_sorted[prev_concept_positions], pt.zeros_like(concept_indices))
		dist_to_prev_concept = pt.where(prev_concept_exists, concept_indices - prev_concept_indices, concept_indices+1) #If no previous concept, distance is the index itself
		
		# Find next concept indices for each concept index
		next_concept_positions = pt.searchsorted(concept_indices_sorted, concept_indices, right=True)
		next_concept_exists = next_concept_positions < len(concept_indices)
		next_concept_positions = next_concept_positions.clamp(max=len(next_concept_positions)-1)
		next_concept_indices = pt.where(next_concept_exists, concept_indices_sorted[next_concept_positions], pt.full_like(concept_indices, len(doc)))	# If no next concept, set to len(doc)
		dist_to_next_concept = pt.where(next_concept_exists, next_concept_indices - concept_indices, len(doc) - concept_indices)	# Distance to end if no next concept
	else:
		q = 5
		dist_to_prev_concept = pt.full((concept_indices.size(0),), q, dtype=pt.long)
		dist_to_next_concept = pt.full((concept_indices.size(0),), q, dtype=pt.long)

	# Calculate start and end indices for each concept word
	if(debugConceptFeaturesOccurFirstInSubsequence):
		if usePOS:
			start_indices = (concept_indices).clamp(min=0)
			end_indices = (concept_indices + dist_to_next_concept).clamp(max=len(doc))
		else:
			start_indices = (concept_indices).clamp(min=0)
			end_indices = (concept_indices + q + 1).clamp(max=len(doc))	
	else:
		if usePOS:
			start_indices = (concept_indices - dist_to_prev_concept + 1).clamp(min=0)
			end_indices = (concept_indices + dist_to_next_concept).clamp(max=len(doc))
		else:
			start_indices = (concept_indices - q).clamp(min=0)
			end_indices = (concept_indices + q + 1).clamp(max=len(doc))

	process_features(sequence_observed_columns, sentenceIndex, start_indices, end_indices, doc, words, lemmas, pos_tags, concept_indices, train, first_seed_token_index, num_seed_tokens)
	
	return concept_indices, start_indices, end_indices

def process_features(sequence_observed_columns, sentenceIndex, start_indices, end_indices, doc, words, lemmas, pos_tags, concept_indices, train, first_seed_token_index=None, num_seed_tokens=None):
	numberConceptsInSequence = concept_indices.shape[0]
	
	cs = sequence_observed_columns.cs #!sequenceObservedColumnsMatchSequenceWords: will be less than len(concept_indices) if there are multiple instances of a concept in a sequence
	fs = sequence_observed_columns.fs  #sequenceObservedColumnsUseSequenceFeaturesOnly+sequenceObservedColumnsMatchSequenceWords: len(doc), sequenceObservedColumnsUseSequenceFeaturesOnly+!sequenceObservedColumnsMatchSequenceWords: number of feature neurons in sentence, !sequenceObservedColumnsUseSequenceFeaturesOnly: number of feature neurons in column
	feature_neurons_active = pt.zeros((array_number_of_segments, cs, fs), dtype=array_type)
	feature_neurons_word_order = pt.arange(fs).unsqueeze(0).repeat(cs, 1)
	pt.zeros((cs, fs), dtype=pt.long)
	columns_word_order = pt.zeros((cs), dtype=pt.long)
	feature_neurons_pos = pt.zeros((cs, fs), dtype=array_type)
	if(sequenceObservedColumnsMatchSequenceWords):
		sequence_concept_index_mask = pt.ones((cs, fs), dtype=array_type)	#ensure to ignore concept feature neurons from other columns
	else:
		sequence_concept_index_mask = None
	if(useSANI):
		feature_neurons_segment_mask = pt.zeros((cs, array_number_of_segments), dtype=array_type)
	else:
		feature_neurons_segment_mask = pt.ones((cs, array_number_of_segments), dtype=array_type)
	
	concept_indices_list = concept_indices.tolist()
	#convert start/end indices to active features arrays
	for i, sequence_concept_word_index in enumerate(concept_indices_list):
		if(sequenceObservedColumnsMatchSequenceWords):
			sequence_concept_index = i
		else:
			concept_lemma = lemmas[sequence_concept_word_index]	# lemmas[concept_indices[i]]
			sequence_concept_index = sequence_observed_columns.concept_name_to_index[concept_lemma] 
				
		if(useSANI):
			number_of_segments = min(array_number_of_segments-1, i)
			feature_neurons_segment_mask[sequence_concept_index, :] = pt.cat([pt.zeros(array_number_of_segments-number_of_segments), pt.ones(number_of_segments)], dim=0)
			minSequentialSegmentIndex = min(0, array_number_of_segments-sequence_concept_index-1)
			activeSequentialSegments = pt.arange(minSequentialSegmentIndex, array_number_of_segments, 1)
		
		if(sequenceObservedColumnsUseSequenceFeaturesOnly and sequenceObservedColumnsMatchSequenceWords):
			if(useSANI):
				feature_neurons_active[activeSequentialSegments, sequence_concept_index, start_indices[sequence_concept_index]:end_indices[sequence_concept_index]] = 1
			else:
				feature_neurons_active[0, sequence_concept_index, start_indices[sequence_concept_index]:end_indices[sequence_concept_index]] = 1
			columns_word_order[sequence_concept_index] = sequence_concept_index
			sequence_concept_index_mask[:, sequence_concept_word_index] = 0	#ignore concept feature neurons from other columns
			sequence_concept_index_mask[sequence_concept_index, sequence_concept_word_index] = 1
			for j in range(start_indices[sequence_concept_index], end_indices[sequence_concept_index]):
				feature_pos = pos_string_to_pos_int(sequence_observed_columns.databaseNetworkObject.nlp, pos_tags[j])
				feature_neurons_pos[sequence_concept_index, j] = feature_pos
				feature_neurons_word_order[sequence_concept_index, j] = j
		else:
			for j in range(start_indices[i], end_indices[i]):	#sequence word index
				feature_word = words[j].lower()
				feature_lemma = lemmas[j]
				feature_pos = pos_string_to_pos_int(sequence_observed_columns.databaseNetworkObject.nlp, pos_tags[j])
				if(j in sequence_observed_columns.columns_index_sequence_word_index_dict):	#test is required for concept neurons
					sequence_concept_word_index = j
					columns_word_order[sequence_concept_index] = sequence_concept_index	#alternatively use sequence_concept_word_index; not robust in either case - there may be less concept columns than concepts referenced in sequence (if multiple references to the same column). sequenceObservedColumnsMatchSequenceWords overcomes this limitation.
					if(useDedicatedConceptNames2):
						sequence_feature_index = sequence_observed_columns.feature_word_to_index[variableConceptNeuronFeatureName]
					else:
						sequence_feature_index = sequence_observed_columns.feature_word_to_index[feature_lemma]
					if(useSANI):
						feature_neurons_active[activeSequentialSegments, sequence_concept_index, sequence_feature_index] = 1
					else:
						feature_neurons_active[0, sequence_concept_index, sequence_feature_index] = 1
				elif(feature_word in sequence_observed_columns.feature_word_to_index):
					sequence_feature_index = sequence_observed_columns.feature_word_to_index[feature_word]
					if(useSANI):
						feature_neurons_active[activeSequentialSegments, sequence_concept_index, sequence_feature_index] = 1
					else:
						feature_neurons_active[0, sequence_concept_index, sequence_feature_index] = 1
				feature_neurons_word_order[sequence_concept_index, sequence_feature_index] = j
				feature_neurons_pos[sequence_concept_index, sequence_feature_index] = feature_pos
	
	feature_neurons_segment_mask = feature_neurons_segment_mask.swapdims(0, 1)
	
	if(train):
		process_features_active_train(sequence_observed_columns, feature_neurons_active, cs, fs, sequence_concept_index_mask, columns_word_order, feature_neurons_word_order, feature_neurons_pos, feature_neurons_segment_mask, sentenceIndex)
	else:
		first_seed_concept_index, num_seed_concepts, first_seed_feature_index = identify_seed_indices(sequence_observed_columns, sentenceIndex, start_indices, end_indices, doc, words, lemmas, pos_tags, concept_indices, first_seed_token_index, num_seed_tokens)
		process_features_active_seed(sequence_observed_columns, feature_neurons_active, cs, fs, sequence_concept_index_mask, columns_word_order, feature_neurons_word_order, feature_neurons_pos, first_seed_token_index, num_seed_tokens, first_seed_concept_index, num_seed_concepts, first_seed_feature_index)

def identify_seed_indices(sequence_observed_columns, sentenceIndex, start_indices, end_indices, doc, words, lemmas, pos_tags, concept_indices, first_seed_token_index, num_seed_tokens):
	first_seed_concept_index = None
	num_seed_concepts = None
	found_first_seed_concept = False
	if(inferenceSeedTargetActivationsGlobalFeatureArrays):
		feature_word = words[first_seed_token_index]
		if(useDedicatedConceptNames and first_seed_token_index in sequence_observed_columns.observed_columns_sequence_word_index_dict):	
			first_seed_feature_index = feature_index_concept_neuron
		elif(feature_word in sequence_observed_columns.feature_word_to_index):
			first_seed_feature_word = words[first_seed_token_index]
			first_seed_feature_index = sequence_observed_columns.databaseNetworkObject.concept_features_dict[first_seed_feature_word]
	else:
		first_seed_feature_index = None

	concept_indices_list = concept_indices.tolist()
	for i, sequence_concept_word_index in enumerate(concept_indices_list):
		if(sequenceObservedColumnsMatchSequenceWords):
			sequence_concept_index = i
		else:
			concept_lemma = lemmas[sequence_concept_word_index]	# lemmas[concept_indices[i]]
			sequence_concept_index = sequence_observed_columns.concept_name_to_index[concept_lemma] 

		lastWordIndexSeedPhase = first_seed_token_index+num_seed_tokens-1
		if(not found_first_seed_concept):
			if(first_seed_token_index >= start_indices[sequence_concept_index] and first_seed_token_index < end_indices[sequence_concept_index]):
				found_first_seed_concept = True
				first_seed_concept_index = sequence_concept_index
				if(inferenceSeedTargetActivationsGlobalFeatureArrays):
					observed_column = sequence_observed_columns.observed_columns_sequence_word_index_dict[sequence_concept_word_index]
					sequence_observed_columns.feature_connections = observed_column.feature_connections	#uses global arrays only	#shape: array_number_of_properties, array_number_of_segments, f, c, f
		if(found_first_seed_concept):
			if(lastWordIndexSeedPhase >= start_indices[sequence_concept_index] and lastWordIndexSeedPhase < end_indices[sequence_concept_index]):
				last_seed_concept_index = sequence_concept_index
				num_seed_concepts = last_seed_concept_index-first_seed_concept_index+1
					
	return first_seed_concept_index, num_seed_concepts, first_seed_feature_index
	
#first dim cs1 pertains to every concept node in sequence
def process_features_active_seed(sequence_observed_columns, feature_neurons_active, cs, fs, sequence_concept_index_mask, columns_word_order, feature_neurons_word_order, feature_neurons_pos, first_seed_token_index, num_seed_tokens, first_seed_concept_index, num_seed_concepts, first_seed_feature_index):
	feature_neurons_inactive = 1 - feature_neurons_active
	
	fs2 = fs
	if(inferenceSeedTargetActivationsGlobalFeatureArrays):
		cs2 = sequence_observed_columns.databaseNetworkObject.c
		feature_connections_active = pt.ones(cs, fs, cs2, fs2)
		#print("feature_connections_active.shape = ", feature_connections_active.shape)
	else:
		cs2 = cs
		feature_connections_active, feature_connections_segment_mask = createFeatureConnectionsActiveTrain(feature_neurons_active[array_index_segment_internal_column], cs, fs, columns_word_order, feature_neurons_word_order)

	firstWordIndexPredictPhase = first_seed_token_index+num_seed_tokens
	firstConceptIndexPredictPhase = first_seed_concept_index+num_seed_concepts
	feature_connections_active = createFeatureConnectionsActiveSeed(feature_connections_active, cs, fs, cs2, fs2, columns_word_order, feature_neurons_word_order, first_seed_token_index, firstWordIndexPredictPhase, first_seed_concept_index, firstConceptIndexPredictPhase)

	if(inferenceSeedTargetActivationsGlobalFeatureArrays):
		feature_connections_active = feature_connections_active[:, first_seed_concept_index]
		
	#target neuron activation dependence on connection strength;
	feature_connections_activation_update = feature_connections_active * sequence_observed_columns.feature_connections[array_index_properties_strength]
	
	#update the activations of the target nodes;
	#feature_connections_activation_update = pt.sum(feature_connections_activation_update, dim=(0))	#sum over segment dim	#TODO: take into account SANI requirements (distal activation must precede proximal activation) 
	if(inferenceSeedTargetActivationsGlobalFeatureArrays):
		feature_neurons_target_activation = pt.sum(feature_connections_activation_update, dim=(1))		#sum over f dimensions
	else:
		feature_neurons_target_activation = pt.sum(feature_connections_activation_update, dim=(1, 2))		#sum over source c and f dimensions
	if(inferenceSeedTargetActivationsGlobalFeatureArrays):
		global_feature_neurons_activation = sequence_observed_columns.databaseNetworkObject.global_feature_neurons[array_index_properties_activation]
		global_feature_neurons_activation = global_feature_neurons_activation + feature_neurons_target_activation*j1
		#print("global_feature_neurons_activation = ", global_feature_neurons_activation)
	else:
		sequence_observed_columns.feature_neurons[array_index_properties_activation, :, :, :] += feature_neurons_target_activation*j1
		#will only activate target neurons in sequence_observed_columns (not suitable for inference seed/prediction phase)
	
	if(useActivationDecrement):
		if(inferenceSeedTargetActivationsGlobalFeatureArrays):
			global_feature_neurons_activation = decrementActivation(global_feature_neurons_activation, activationDecrementSeed)
		else:
			sequence_observed_columns.feature_neurons[array_index_properties_activation] = decrementActivationDense(sequence_observed_columns.feature_neurons[array_index_properties_activation], activationDecrementSeed)
					
	if(deactivateNeuronsUponPrediction):
		if(inferenceSeedTargetActivationsGlobalFeatureArrays):
			if(useSANI):
				printe("process_features_active_seed error: deactivateNeuronsUponPrediction:inferenceSeedTargetActivationsGlobalFeatureArrays:useSANI is not yet implemented")
			else:
				indices_to_update = pt.tensor([0, first_seed_concept_index, first_seed_feature_index]).unsqueeze(0)	#first SANI dim, source concept dim, source feature dim 
				global_feature_neurons_activation = global_feature_neurons_activation.coalesce()
				global_feature_neurons_activation = GIAANNproto_sparseTensors.modify_sparse_tensor(global_feature_neurons_activation, indices_to_update, 0)
		else:
			word_order_mask = pt.logical_and(feature_neurons_word_order >= first_seed_token_index, feature_neurons_word_order < firstWordIndexPredictPhase)
			columns_word_order_expanded_1 = columns_word_order.view(cs, 1).expand(cs, fs)
			columns_word_order_mask = pt.logical_and(columns_word_order_expanded_1 >= first_seed_concept_index, columns_word_order_expanded_1 < firstConceptIndexPredictPhase)

			word_order_mask = pt.logical_and(word_order_mask, columns_word_order_mask)
			word_order_mask = word_order_mask.unsqueeze(0).expand(array_number_of_segments, cs, fs)
			feature_neurons_active_source = pt.logical_and(word_order_mask, feature_neurons_active > 0)
			feature_neurons_inactive_source = pt.logical_not(feature_neurons_active_source).float()
			sequence_observed_columns.feature_neurons[array_index_properties_activation, :, :, :] *= feature_neurons_inactive_source

	if(inferenceSeedTargetActivationsGlobalFeatureArrays):
		sequence_observed_columns.databaseNetworkObject.global_feature_neurons = GIAANNproto_sparseTensors.replaceAllSparseTensorElementsAtFirstDimIndex(sequence_observed_columns.databaseNetworkObject.global_feature_neurons, global_feature_neurons_activation, array_index_properties_activation)

def createFeatureConnectionsActiveSeed(feature_connections_active, cs, fs, cs2, fs2, columns_word_order, feature_neurons_word_order, first_seed_token_index, firstWordIndexPredictPhase, first_seed_concept_index, firstConceptIndexPredictPhase):
	
	if(feature_neurons_word_order is not None):	
		feature_neurons_word_order_expanded_1 = feature_neurons_word_order.view(cs, fs, 1, 1).expand(cs, fs, cs2, fs2)  # For the first node
		word_order_mask = pt.logical_and(feature_neurons_word_order_expanded_1 >= first_seed_token_index, feature_neurons_word_order_expanded_1 < firstWordIndexPredictPhase)
		feature_connections_active = feature_connections_active * word_order_mask.unsqueeze(0)
	if(columns_word_order is not None):
		columns_word_order_expanded_1 = columns_word_order.view(cs, 1, 1, 1).expand(cs, fs, cs2, fs2)  # For the first node's cs index
		columns_word_order_mask = pt.logical_and(columns_word_order_expanded_1 >= first_seed_concept_index, columns_word_order_expanded_1 < firstConceptIndexPredictPhase)
		feature_connections_active = feature_connections_active * columns_word_order_mask.unsqueeze(0)
	
	#feature_connections_active = feature_connections_active.unsqueeze(0).expand(array_number_of_segments, cs, fs, cs2, fs2)

	return feature_connections_active
	
	
#first dim cs1 pertains to every concept node in sequence
def process_features_active_train(sequence_observed_columns, feature_neurons_active, cs, fs, sequence_concept_index_mask, columns_word_order, feature_neurons_word_order, feature_neurons_pos, feature_neurons_segment_mask, sentenceIndex):
	feature_neurons_inactive = 1 - feature_neurons_active
		
	# Update feature neurons in sequence_observed_columns
	sequence_observed_columns.feature_neurons[array_index_properties_strength, :, :, :] += feature_neurons_active
	sequence_observed_columns.feature_neurons[array_index_properties_permanence, :, :, :] += feature_neurons_active*z1	#orig = feature_neurons_active*(sequence_observed_columns.feature_neurons[array_index_properties_permanence] ** 2) + feature_neurons_inactive*sequence_observed_columns.feature_neurons[array_index_properties_permanence]
	#sequence_observed_columns.feature_neurons[array_index_properties_activation, :, :, :] += feature_neurons_active*j1	#update the activations of the target not source nodes
	if(useInference and not useNeuronFeaturePropertiesTimeDuringInference):
		sequence_observed_columns.feature_neurons[array_index_properties_time, :, :, :] = feature_neurons_inactive*sequence_observed_columns.feature_neurons[array_index_properties_time] + feature_neurons_active*sentenceIndex
	sequence_observed_columns.feature_neurons[array_index_properties_pos, :, :, :] = feature_neurons_inactive*sequence_observed_columns.feature_neurons[array_index_properties_pos] + feature_neurons_active*feature_neurons_pos

	feature_connections_active, feature_connections_segment_mask = createFeatureConnectionsActiveTrain(feature_neurons_active[array_index_segment_internal_column], cs, fs, columns_word_order, feature_neurons_word_order)
	
	feature_connections_pos = feature_neurons_pos.view(1, cs, fs, 1, 1).expand(array_number_of_segments, cs, fs, cs, fs)

	feature_connections_inactive = 1 - feature_connections_active

	#prefer closer than further target neurons when strengthening connections (and activating target neurons) in sentence;
	feature_neurons_word_order_1d = feature_neurons_word_order.flatten()
	feature_connections_distances = pt.abs(feature_neurons_word_order_1d.unsqueeze(1) - feature_neurons_word_order_1d).reshape(cs, fs, cs, fs)
	feature_connections_proximity = 1/(feature_connections_distances + 1) * 10
	feature_connections_proximity.unsqueeze(0)	#add SANI segment dimension
	feature_connections_strength_update = feature_connections_active*feature_connections_proximity
	#print("feature_connections_strength_update = ", feature_connections_strength_update)

	if(increaseColumnInternalConnectionsStrength):
		cs_indices_1 = pt.arange(cs).view(1, cs, 1, 1, 1).expand(array_number_of_segments, cs, fs, cs, fs)  # First cs dimension
		cs_indices_2 = pt.arange(cs).view(1, 1, 1, cs, 1).expand(array_number_of_segments, cs, fs, cs, fs)  # Second cs dimension
		column_internal_connections_mask = (cs_indices_1 == cs_indices_2)
		column_internal_connections_mask_off = pt.logical_not(column_internal_connections_mask)
		feature_connections_strength_update = column_internal_connections_mask.float()*feature_connections_strength_update*increaseColumnInternalConnectionsStrengthModifier + column_internal_connections_mask_off.float()*feature_connections_strength_update

	#print("feature_connections_active[array_index_segment_first] = ", feature_connections_active[array_index_segment_first])
	#print("feature_connections_active[array_index_segment_internal_column] = ", feature_connections_active[array_index_segment_internal_column])
	
	sequence_observed_columns.feature_connections[array_index_properties_strength, :, :, :, :, :] += feature_connections_strength_update
	sequence_observed_columns.feature_connections[array_index_properties_permanence, :, :, :, :, :] += feature_connections_active*z1	#orig = feature_connections_active*(sequence_observed_columns.feature_connections[array_index_properties_permanence] ** 2) + feature_connections_inactive*sequence_observed_columns.feature_connections[array_index_properties_permanence]
	#sequence_observed_columns.feature_connections[array_index_properties_activation, :, :, :, :, :] += feature_connections_active*j1	#connection activations are not currently used
	if(useInference and not useNeuronFeaturePropertiesTimeDuringInference):
		sequence_observed_columns.feature_connections[array_index_properties_time, :, :, :, :, :] = feature_connections_inactive*sequence_observed_columns.feature_connections[array_index_properties_time] + feature_connections_active*sentenceIndex
	sequence_observed_columns.feature_connections[array_index_properties_pos, :, :, :, :, :] = feature_connections_inactive*sequence_observed_columns.feature_connections[array_index_properties_pos] + feature_connections_active*feature_connections_pos

	#decrease permanence;
	if(decreasePermanenceOfInactiveFeatureNeuronsAndConnections):
		decrease_permanence_active(sequence_observed_columns, feature_neurons_active[array_index_segment_internal_column], feature_neurons_inactive[array_index_segment_internal_column], sequence_concept_index_mask, feature_neurons_segment_mask, feature_connections_segment_mask)
	

def createFeatureConnectionsActiveTrain(feature_neurons_active, cs, fs, columns_word_order, feature_neurons_word_order):

	feature_neurons_active_1d = feature_neurons_active.view(cs*fs)
	feature_connections_active = pt.matmul(feature_neurons_active_1d.unsqueeze(1), feature_neurons_active_1d.unsqueeze(0)).view(cs, fs, cs, fs)

	if(feature_neurons_word_order is not None):
		#ensure word order is maintained (between connection source/target) for internal and external feature connections;
		feature_neurons_word_order_expanded_1 = feature_neurons_word_order.view(cs, fs, 1, 1).expand(cs, fs, cs, fs)  # For the first node
		feature_neurons_word_order_expanded_2 = feature_neurons_word_order.view(1, 1, cs, fs).expand(cs, fs, cs, fs)  # For the second node
		word_order_mask = feature_neurons_word_order_expanded_2 > feature_neurons_word_order_expanded_1
		feature_connections_active = feature_connections_active * word_order_mask
	if(columns_word_order is not None):
		#ensure word order is maintained for connections between columns (does not support multiple same concepts in same sentence);
		columns_word_order_expanded_1 = columns_word_order.view(cs, 1, 1, 1).expand(cs, fs, cs, fs)  # For the first node's cs index
		columns_word_order_expanded_2 = columns_word_order.view(1, 1, cs, 1).expand(cs, fs, cs, fs)  # For the second node's cs index
		if(debugConnectColumnsToNextColumnsInSequenceOnly):
			columns_word_order_mask = pt.logical_and(columns_word_order_expanded_2 >= columns_word_order_expanded_1, columns_word_order_expanded_2 <= columns_word_order_expanded_1+1)
		else:
			columns_word_order_mask = columns_word_order_expanded_2 >= columns_word_order_expanded_1
		feature_connections_active = feature_connections_active * columns_word_order_mask
	
	#ensure identical feature nodes are not connected together;
	cs_indices_1 = pt.arange(cs).view(cs, 1, 1, 1).expand(cs, fs, cs, fs)  # First cs dimension
	cs_indices_2 = pt.arange(cs).view(1, 1, cs, 1).expand(cs, fs, cs, fs)  # Second cs dimension
	fs_indices_1 = pt.arange(fs).view(1, fs, 1, 1).expand(cs, fs, cs, fs)  # First fs dimension
	fs_indices_2 = pt.arange(fs).view(1, 1, 1, fs).expand(cs, fs, cs, fs)  # Second fs dimension
	identity_mask = (cs_indices_1 != cs_indices_2) | (fs_indices_1 != fs_indices_2)
	feature_connections_active = feature_connections_active * identity_mask

	if(useSANI):
		feature_connections_active, feature_connections_segment_mask = assign_feature_connections_to_target_segments(feature_connections_active, cs, fs)
	else:
		feature_connections_active = feature_connections_active.unsqueeze(0)
		feature_connections_segment_mask = pt.ones_like(feature_connections_active)
	
	return feature_connections_active, feature_connections_segment_mask

def assign_feature_connections_to_target_segments(feature_connections_active, cs, fs):

	#arrange active connections according to target neuron sequential segment index
	concept_neurons_concept_order_1d = pt.arange(cs)
	concept_neurons_distances = pt.abs(concept_neurons_concept_order_1d.unsqueeze(1) - concept_neurons_concept_order_1d).reshape(cs, cs)
	connections_segment_index = array_number_of_segments-concept_neurons_distances-1
	connections_segment_index = pt.clamp(connections_segment_index, min=0)
	
	feature_connections_segment_mask = pt.zeros((array_number_of_segments, cs, cs), dtype=pt.bool)
	feature_connections_segment_mask = feature_connections_segment_mask.scatter_(0, connections_segment_index.unsqueeze(0), True)
	feature_connections_segment_mask = feature_connections_segment_mask.view(array_number_of_segments, cs, 1, cs, 1).expand(array_number_of_segments, cs, fs, cs, fs)
	feature_connections_active = feature_connections_segment_mask * feature_connections_active.unsqueeze(0)
	
	return feature_connections_active, feature_connections_segment_mask
		
def decrease_permanence_active(sequence_observed_columns, feature_neurons_active, feature_neurons_inactive, sequence_concept_index_mask, feature_neurons_segment_mask, feature_connections_segment_mask):

	if(sequenceObservedColumnsMatchSequenceWords):
		feature_neurons_inactive = feature_neurons_inactive*sequence_concept_index_mask	#when decreasing a value based on inactivation, ignore duplicate feature column neurons in the sequence
	
	cs = sequence_observed_columns.cs
	fs = sequence_observed_columns.fs 
	
	# Decrease permanence for feature neurons not activated
	feature_neurons_decrease = feature_neurons_inactive.unsqueeze(0)*z2 * feature_neurons_segment_mask.unsqueeze(2)
	sequence_observed_columns.feature_neurons[array_index_properties_permanence, :, :, :] -= feature_neurons_decrease
	sequence_observed_columns.feature_neurons[array_index_properties_permanence] = pt.clamp(sequence_observed_columns.feature_neurons[array_index_properties_permanence], min=0)

	feature_neurons_all = pt.ones((cs, fs), dtype=array_type)
	feature_neurons_all_1d = feature_neurons_all.view(cs*fs)
	feature_neurons_active_1d = feature_neurons_active.view(cs*fs)
	feature_neurons_inactive_1d = feature_neurons_inactive.view(cs*fs)
	 
	# Decrease permanence of connections from inactive feature neurons in column
	feature_connections_decrease1 = pt.matmul(feature_neurons_inactive_1d.unsqueeze(1), feature_neurons_all_1d.unsqueeze(0)).view(cs, fs, cs, fs)
	feature_connections_decrease1 = feature_connections_decrease1.unsqueeze(0)*feature_connections_segment_mask
	sequence_observed_columns.feature_connections[array_index_properties_permanence, :, :, :, :, :] -= feature_connections_decrease1
	sequence_observed_columns.feature_connections[array_index_properties_permanence] = pt.clamp(sequence_observed_columns.feature_connections[array_index_properties_permanence], min=0)
	
	# Decrease permanence of inactive connections for activated features in column 
	feature_connections_decrease2 = pt.matmul(feature_neurons_active_1d.unsqueeze(1), feature_neurons_inactive_1d.unsqueeze(0)).view(cs, fs, cs, fs)
	feature_connections_decrease2 = feature_connections_decrease2.unsqueeze(0)*feature_connections_segment_mask
	sequence_observed_columns.feature_connections[array_index_properties_permanence, :, :, :, :, :] -= feature_connections_decrease2
	sequence_observed_columns.feature_connections[array_index_properties_permanence] = pt.clamp(sequence_observed_columns.feature_connections[array_index_properties_permanence], min=0)
 
	#current limitation; will not deactivate neurons or remove their strength if their permanence goes to zero


def decrementActivationDense(feature_neurons_activation, activationDecrement):
	if(useActivationDecrementNonlinear):
		feature_neurons_activation = feature_neurons_activation * (1-activationDecrement)
	else:
		feature_neurons_activation = feature_neurons_activation - activationDecrementPerPredictedSentence
	return feature_neurons_activation


def decrementActivation(feature_neurons_activation, activationDecrement):
	if(useActivationDecrementNonlinear):
		feature_neurons_activation = feature_neurons_activation * (1-activationDecrement)
	else:
		feature_neurons_activation = GIAANNproto_sparseTensors.subtract_value_from_sparse_tensor_values(feature_neurons_activation, activationDecrementPerPredictedSentence)
	return feature_neurons_activation


