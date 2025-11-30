# Import necessary libraries
import torch
import networkx as nx
import matplotlib.pyplot as plt
import nltk
import spacy
from datasets import load_dataset
import os
import pickle
import numpy as np
import random
torch.set_printoptions(threshold=float('inf'))

useGPU = True
if(useGPU):
	if torch.cuda.is_available():
		torch.set_default_device(torch.device("cuda"))
	
# Set boolean variables as per specification
useInference = False  # useInference mode
if(useInference):
	inferenceSeedTargetActivationsGlobalFeatureArrays = False
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
conceptColumnsDelimitByConceptFeaturesStart = False #Constrain column feature detection to be after concept feature detection
debugConnectColumnsToNextColumnsInSequenceOnly = False
debugDrawNeuronStrengths = False
if(useInference):
	conceptColumnsDelimitByConceptFeaturesStart = True	#enables higher performance prediction without training (ie before learning appropriate column feature associations by forgetting features belonging to external columns)
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
	deactivateNeuronsUponPrediction = True

	num_seed_tokens = 5	#number of seed tokens in last sentence of inference prompt (remaining tokens will be prediction tokens)
	num_prediction_tokens = 10	#number of words to predict after network seed

	#TODO: train hyperparameters
	kcMax = 1 	#(if kcDynamic: max) topk next concept column prediction
	kcDynamic = False
	if(kcDynamic):
		kcActivationThreshold = 3.0	#total column activation threshold	#minimum required to select topk
	kf = 1
	activationDecrementPerPredictedColumn = 0.01
	
	assert not lowMem, "useInference: global feature neuron lists are required" 
	assert useSaveData,  "useInference: useSaveData is required" 

useActivationDecrement = False
if not lowMem:
	useActivationDecrement = False
activationDecrementPerPredictedSentence = 0.1
	
if(useDedicatedFeatureLists):
	nltk.download('punkt')
	nltk.download('wordnet')
	nltk.download('omw-1.4')
	from nltk.corpus import wordnet as wn
	from nltk.tokenize import sent_tokenize

# Initialize spaCy model
nlp = spacy.load('en_core_web_sm')

# Paths for saving data
concept_columns_dict_file = databaseFolder + 'concept_columns_dict.pkl'
concept_features_dict_file = databaseFolder + 'concept_features_dict.pkl'
observed_columns_dir = databaseFolder + 'observed_columns'
os.makedirs(observed_columns_dir, exist_ok=True)

#common array indices
array_index_properties_strength = 0
array_index_properties_permanence = 1
array_index_properties_activation = 2
array_index_properties_time = 3
array_index_properties_pos = 4
array_number_of_properties = 5
array_index_segment_first = 0
array_index_segment_internal_column = 9
array_number_of_segments = 10	#max number of SANI segments per sequence (= max number of concept columns per sequence)
array_type = torch.float32	#torch.long	#torch.float32

# Define POS tag sets for nouns and non-nouns
noun_pos_tags = {'NOUN', 'PROPN'}
non_noun_pos_tags = {'ADJ', 'ADV', 'VERB', 'ADP', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NUM', 'PART', 'PRON', 'SCONJ', 'SYM', 'X'}

def pos_int_to_pos_string(pos_int):
	if pos_int in nlp.vocab.strings:
		return nlp.vocab[pos_int].text
	else:
		return ''
		
def pos_string_to_pos_int(pos_string):
	return nlp.vocab.strings[pos_string]
		
if(drawRelationTypes):
	relation_type_concept_pos1 = 'NOUN'
	relation_type_concept_pos2 = 'PROPN'
	relation_type_action_pos = 'VERB'
	relation_type_condition_pos = 'ADP'
	relation_type_quality_pos = 'ADJ'
	relation_type_modifier_pos = 'ADV'
	
	relation_type_determiner_pos = 'DET'
	relation_type_conjunction_pos1 = 'CCONJ'
	relation_type_conjunction_pos2 = 'SCONJ'
	relation_type_quantity_pos1 = 'SYM'
	relation_type_quantity_pos2 = 'NUM'
	relation_type_aux_pos = 'AUX'

	neuron_pos_to_relation_type_dict = {}
	neuron_pos_to_relation_type_dict[relation_type_concept_pos1] = 'blue'
	neuron_pos_to_relation_type_dict[relation_type_concept_pos2] = 'blue'
	neuron_pos_to_relation_type_dict[relation_type_action_pos] = 'green'
	neuron_pos_to_relation_type_dict[relation_type_condition_pos] = 'red'
	neuron_pos_to_relation_type_dict[relation_type_quality_pos] = 'turquoise'
	neuron_pos_to_relation_type_dict[relation_type_modifier_pos] = 'lightskyblue'
	
	neuron_pos_to_relation_type_dict[relation_type_determiner_pos] = 'magenta'
	neuron_pos_to_relation_type_dict[relation_type_conjunction_pos1] = 'black'
	neuron_pos_to_relation_type_dict[relation_type_conjunction_pos2] = 'black'
	neuron_pos_to_relation_type_dict[relation_type_quantity_pos1] = 'purple'
	neuron_pos_to_relation_type_dict[relation_type_quantity_pos2] = 'purple'
	neuron_pos_to_relation_type_dict[relation_type_aux_pos] = 'lightskyblue'

	relation_type_part_property_col = 'cyan'
	relation_type_aux_definition_col = 'blue'
	relation_type_aux_quality_col = 'turquoise'
	relation_type_aux_action_col = 'green'
	relation_type_aux_property_col = 'cyan'
	
	relation_type_other_col = 'gray'	#INTJ, X, other AUX
	
	be_auxiliaries = ["am", "is", "are", "was", "were", "being", "been"]
	have_auxiliaries = ["have", "has", "had", "having"]
	do_auxiliaries = ["do", "does", "did", "doing"]

	def generateFeatureNeuronColour(pos_float_torch, word, internal_connection=False):
		#print("pos_float_torch = ", pos_float_torch)
		pos_int = pos_float_torch.int().item()
		pos_string = pos_int_to_pos_string(pos_int)
		if(pos_string):
			if(pos_string in neuron_pos_to_relation_type_dict):
				colour = neuron_pos_to_relation_type_dict[pos_string]
			else:
				colour = relation_type_other_col
				
			#special cases;
			if(pos_string == 'AUX'):
				if(word in have_auxiliaries):
					colour = relation_type_aux_property_col
				elif(word in be_auxiliaries):
					if(internal_connection):
						colour = relation_type_aux_quality_col
					else:
						colour = relation_type_aux_definition_col
				elif(word in do_auxiliaries):
					colour = relation_type_aux_action_col
			if(pos_string == 'PART'):
				if(word == "'s"):
					colour = relation_type_part_property_col
		else:
			colour = relation_type_other_col
			#print("generateFeatureNeuronColour error; pos int = 0")
			
		return colour
	

# Define constants for permanence and activation trace	#TODO: train hyperparameters
z1 = 3  # Initial permanence value	
z2 = 1  # Decrement value when not activated
j1 = 5   # Activation trace duration

# Initialize NetworkX graph for visualization
G = nx.DiGraph()

# For the purpose of the example, process a limited number of sentences
sentence_count = 0
max_sentences_train = 1000  # Adjust as needed


def modify_sparse_tensor(sparse_tensor, indices_to_update, new_value):
	sparse_tensor = sparse_tensor.coalesce()
	
	# Transpose indices_to_update to match dimensions
	indices_to_update = indices_to_update.t()  # Shape: (3, N)
	
	# Get sparse tensor indices
	sparse_indices = sparse_tensor.indices()   # Shape: (3, nnz)
	
	# Expand dimensions to enable broadcasting
	sparse_indices_expanded = sparse_indices.unsqueeze(2)	   # Shape: (3, nnz, 1)
	indices_to_update_expanded = indices_to_update.unsqueeze(1) # Shape: (3, 1, N)
	
	# Compare indices
	matches = (sparse_indices_expanded == indices_to_update_expanded).all(dim=0)  # Shape: (nnz, N)
	
	# Identify matches
	match_mask = matches.any(dim=1)  # Shape: (nnz,)
	
	# Update the values at the matched indices
	sparse_tensor.values()[match_mask] = new_value
	
	return sparse_tensor

	
def merge_tensor_slices_sum(original_sparse_tensor, sparse_slices, d):
	# Extract indices and values from the original tensor
	original_indices = original_sparse_tensor._indices()
	original_values = original_sparse_tensor._values()

	# Prepare lists for new indices and values
	all_indices = [original_indices]
	all_values = [original_values]

	# Process each slice and adjust for the d dimension
	for index, tensor_slice in sparse_slices.items():
		# Create the index tensor for dimension 'd'
		num_nonzero = tensor_slice._indices().size(1)
		d_indices = torch.full((1, num_nonzero), index, dtype=tensor_slice._indices().dtype)

		# Build the new indices by inserting d_indices at position 'd'
		slice_indices = tensor_slice._indices()
		before = slice_indices[:d, :]
		after = slice_indices[d:, :]
		new_indices = torch.cat([before, d_indices, after], dim=0)

		# Collect the adjusted indices and values
		all_indices.append(new_indices)
		all_values.append(tensor_slice._values())

	# Concatenate all indices and values, including the original tensor's
	final_indices = torch.cat(all_indices, dim=1)
	final_values = torch.cat(all_values)

	# Define the final size of the merged tensor, matching the original
	final_size = original_sparse_tensor.size()

	# Create the updated sparse tensor and coalesce to handle duplicates
	merged_sparse_tensor = torch.sparse_coo_tensor(final_indices, final_values, size=final_size)

	merged_sparse_tensor = merged_sparse_tensor.coalesce()
	merged_sparse_tensor.values().clamp_(min=0)

	return merged_sparse_tensor


def slice_sparse_tensor_multi(sparse_tensor, slice_dim, slice_indices):
	"""
	Slices a PyTorch sparse tensor along a specified dimension at given indices,
	without reducing the number of dimensions.

	Args:
		sparse_tensor (torch.sparse.FloatTensor): The input sparse tensor.
		slice_dim (int): The dimension along which to slice.
		slice_indices (torch.Tensor): A 1D tensor of indices to slice.

	Returns:
		torch.sparse.FloatTensor: The sliced sparse tensor with the same number of dimensions.
	"""
	import torch

	# Ensure slice_indices is a 1D tensor and sorted
	slice_indices = slice_indices.view(-1).long()
	slice_indices_sorted, _ = torch.sort(slice_indices)

	# Get the indices and values from the sparse tensor
	indices = sparse_tensor.indices()  # Shape: (ndim, nnz)
	values = sparse_tensor.values()	# Shape: (nnz, ...)

	# Get indices along the slicing dimension
	indices_along_dim = indices[slice_dim]  # Shape: (nnz,)

	# Use searchsorted to find positions in slice_indices
	positions = torch.searchsorted(slice_indices_sorted, indices_along_dim)

	# Check if indices_along_dim are actually in slice_indices
	in_bounds = positions < len(slice_indices_sorted)
	matched = in_bounds & (slice_indices_sorted[positions.clamp(max=len(slice_indices_sorted)-1)] == indices_along_dim)

	# Mask to select relevant indices and values
	mask = matched

	# Select the indices and values where mask is True
	selected_indices = indices[:, mask]
	selected_values = values[mask]

	# Adjust indices along slice_dim
	new_indices_along_dim = positions[mask]

	# Update the indices along slice_dim
	selected_indices[slice_dim] = new_indices_along_dim

	# Adjust the size of the tensor
	new_size = list(sparse_tensor.size())
	new_size[slice_dim] = len(slice_indices)

	# Create the new sparse tensor
	new_sparse_tensor = torch.sparse_coo_tensor(selected_indices, selected_values, size=new_size)

	return new_sparse_tensor
	
	

def slice_sparse_tensor(sparse_tensor, slice_dim, slice_index):
	"""
	Slices a PyTorch sparse tensor along a specified dimension at a given index.

	Args:
		sparse_tensor (torch.sparse.FloatTensor): The input sparse tensor.
		slice_dim (int): The dimension along which to slice.
		slice_index (int): The index at which to slice.

	Returns:
		torch.sparse.FloatTensor: The sliced sparse tensor.
	"""
	sparse_tensor = sparse_tensor.coalesce()	
	
	# Step 1: Extract indices and values
	indices = sparse_tensor._indices()  # Shape: (ndim, nnz)
	values = sparse_tensor._values()	# Shape: (nnz, ...)

	# Step 2: Create a mask for entries where indices match slice_index at slice_dim
	mask = (indices[slice_dim, :] == slice_index)

	# Step 3: Filter indices and values using the mask
	filtered_indices = indices[:, mask]
	filtered_values = values[mask]

	# Step 4: Remove the slice_dim from indices
	new_indices = torch.cat((filtered_indices[:slice_dim, :], filtered_indices[slice_dim+1:, :]), dim=0)

	# Step 5: Adjust the size of the new sparse tensor
	original_size = sparse_tensor.size()
	new_size = original_size[:slice_dim] + original_size[slice_dim+1:]

	# Step 6: Create the new sparse tensor
	new_sparse_tensor = torch.sparse_coo_tensor(new_indices, filtered_values, size=new_size)
	new_sparse_tensor = new_sparse_tensor.coalesce()  # Ensure the tensor is in canonical form

	return new_sparse_tensor

'''
def slice_sparse_tensor(sparse_tensor, slice_dim, slice_index):
	"""
	Slices a PyTorch sparse tensor along a specified dimension at a given index.

	Args:
		sparse_tensor (torch.sparse.FloatTensor): The input sparse tensor.
		slice_dim (int): The dimension along which to slice.
		slice_index (int): The index at which to slice.

	Returns:
		torch.sparse.FloatTensor: The sliced sparse tensor.
	"""
	sparse_tensor = sparse_tensor.coalesce()

	# Step 1: Extract indices and values
	indices = sparse_tensor._indices()  # Shape: (ndim, nnz)
	values = sparse_tensor._values()	# Shape: (nnz, ...)

	# Step 2: Create a mask for entries where indices match slice_index at slice_dim
	mask = (indices[slice_dim, :] == slice_index)

	# Step 3: Filter indices and values using the mask
	filtered_indices = indices[:, mask]
	filtered_values = values[mask]

	# Step 4: Remove the slice_dim from indices using boolean indexing
	keep_dims = torch.arange(indices.size(0)) != slice_dim
	new_indices = filtered_indices[keep_dims, :]

	# Step 5: Adjust the size of the new sparse tensor
	new_size = list(sparse_tensor.size())
	del new_size[slice_dim]
	new_size = tuple(new_size)

	# Step 6: Create the new sparse tensor
	new_sparse_tensor = torch.sparse_coo_tensor(new_indices, filtered_values, size=new_size)
	# Remove the following line if coalesce is unnecessary
	new_sparse_tensor = new_sparse_tensor.coalesce()

	return new_sparse_tensor
'''

if not lowMem:
	global_feature_neurons_file = databaseFolder + 'global_feature_neurons.pt'

if useDedicatedFeatureLists:
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

variableConceptNeuronFeatureName = "variableConceptNeuronFeature"
feature_index_concept_neuron = 0


concept_columns_dict = {}  # key: lemma, value: index
concept_columns_list = []  # list of concept column names (lemmas)
c = 0  # current number of concept columns
concept_features_dict = {}  # key: lemma, value: index
concept_features_list = []  # list of concept column names (lemmas)
f = 0  # current number of concept features

def initialiseConceptColumnsDictionary():
	global c, f, concept_columns_dict, concept_columns_list, concept_features_dict, concept_features_list

	concept_columns_dict.clear() 
	concept_columns_list.clear()
	c = 0  # current number of concept columns
	concept_features_dict.clear()
	concept_features_list.clear()
	f = 0  # current number of concept features
	
	# Initialize the concept columns dictionary
	if(os.path.exists(concept_columns_dict_file)):
		with open(concept_columns_dict_file, 'rb') as f_in:
			concept_columns_dict = pickle.load(f_in)
		c = len(concept_columns_dict)
		concept_columns_list = list(concept_columns_dict.keys())
		with open(concept_features_dict_file, 'rb') as f_in:
			concept_features_dict = pickle.load(f_in)
		f = len(concept_features_dict)
		concept_features_list = list(concept_features_dict.keys())
	else:
		if(useDedicatedConceptNames):
			# Add dummy feature for concept neuron (different per concept column)
			concept_features_list.append(variableConceptNeuronFeatureName)
			concept_features_dict[variableConceptNeuronFeatureName] = len(concept_features_dict)
			f += 1  # Will be updated dynamically based on c

		if useDedicatedFeatureLists:
			print("error: useDedicatedFeatureLists case not yet coded - need to set f and populate concept_features_list/concept_features_dict etc")
			exit()
			# f = max_num_non_nouns + 1  # Maximum number of non-nouns in an English dictionary, plus the concept neuron of each column

initialiseConceptColumnsDictionary()

def createEmptySparseTensor(shape):
	sparse_zero_tensor = torch.sparse_coo_tensor(indices=torch.empty((len(shape), 0), dtype=torch.long), values=torch.empty(0), size=shape)
	return sparse_zero_tensor

# Define the ObservedColumn class
class ObservedColumn:
	"""
	Create a class defining observed columns. The observed column class contains an index to the dataset concept column dictionary. The observed column class contains a list of feature connection arrays. The observed column class also contains a list of feature neuron arrays when lowMem mode is enabled.
	"""
	def __init__(self, concept_index, lemma, i):
		self.concept_index = concept_index  # Index to the concept columns dictionary
		self.concept_name = lemma
		self.concept_sequence_word_index = i	#not currently used (use SequenceObservedColumns observed_columns_sequence_word_index_dict instead)

		if lowMem:
			# If lowMem is enabled, the observed columns contain a list of arrays (pytorch) of f feature neurons, where f is the maximum number of feature neurons per column.
			self.feature_neurons = self.initialiseFeatureNeurons(f)

		# Map from feature words to indices in feature neurons
		self.feature_word_to_index = {}  # Maps feature words to indices
		self.feature_index_to_word = {}  # Maps indices to feature words
		if(useDedicatedConceptNames):
			self.next_feature_index = 1  # Start from 1 since index 0 is reserved for concept neuron
			if(useDedicatedConceptNames2):
				self.feature_word_to_index[variableConceptNeuronFeatureName] = feature_index_concept_neuron
				self.feature_index_to_word[feature_index_concept_neuron] = variableConceptNeuronFeatureName
			
		# Store all connections for each source column in a list of integer feature connection arrays, each of size f * c * f, where c is the length of the dictionary of columns, and f is the maximum number of feature neurons.
		self.feature_connections = self.initialiseFeatureConnections(c, f) 

		self.next_feature_index = 0
		for feature_index in range(1, f, 1):
			feature_word = concept_features_list[feature_index]
			self.feature_word_to_index[feature_word] = feature_index
			self.feature_index_to_word[feature_index] = feature_word
			self.next_feature_index += 1
					
	@staticmethod
	def initialiseFeatureNeurons(f):
		feature_neurons = createEmptySparseTensor((array_number_of_properties, array_number_of_segments, f))
		return feature_neurons

	@staticmethod
	def initialiseFeatureConnections(c, f):
		feature_connections = createEmptySparseTensor((array_number_of_properties, array_number_of_segments, f, c, f))
		return feature_connections
	
	def resize_concept_arrays(self, new_c):
		load_c = self.feature_connections.shape[3]
		if new_c > load_c:
			expanded_size = (self.feature_connections.shape[0], self.feature_connections.shape[1], self.feature_connections.shape[2], new_c, self.feature_connections.shape[4])
			self.feature_connections = torch.sparse_coo_tensor(self.feature_connections.indices(), self.feature_connections.values(), size=expanded_size, dtype=array_type)
		
	def expand_feature_arrays(self, new_f):
		load_f = self.feature_connections.shape[2]  # or self.feature_connections.shape[4]
		if new_f > load_f:
			# Expand feature_connections along dimensions 2 and 4
			self.feature_connections = self.feature_connections.coalesce()
			expanded_size_connections = (self.feature_connections.shape[0], self.feature_connections.shape[1], new_f, self.feature_connections.shape[3], new_f)
			self.feature_connections = torch.sparse_coo_tensor(self.feature_connections.indices(), self.feature_connections.values(), size=expanded_size_connections, dtype=array_type)
	
			if lowMem:
				expanded_size_neurons = (self.feature_neurons.shape[0], self.feature_neurons.shape[1], new_f)
				self.feature_neurons = self.feature_neurons.coalesce()
				self.feature_neurons = torch.sparse_coo_tensor(self.feature_neurons.indices(), self.feature_neurons.values(), size=expanded_size_neurons, dtype=array_type)

			for feature_index in range(load_f, new_f):
				feature_word = concept_features_list[feature_index]
				self.feature_word_to_index[feature_word] = feature_index
				self.feature_index_to_word[feature_index] = feature_word
				self.next_feature_index += 1

	def save_to_disk(self):
		"""
		Save the observed column data to disk.
		"""
		data = {
			'concept_index': self.concept_index,
			'feature_word_to_index': self.feature_word_to_index,
			'feature_index_to_word': self.feature_index_to_word,
			'next_feature_index': self.next_feature_index
		}
		# Save the data dictionary using pickle
		with open(os.path.join(observed_columns_dir, f"{self.concept_index}_data.pkl"), 'wb') as f:
			pickle.dump(data, f)
		# Save the tensors using torch.save
		if(performRedundantCoalesce):
			self.feature_connections = self.feature_connections.coalesce()
			print("self.feature_connections = ", self.feature_connections)
		torch.save(self.feature_connections, os.path.join(observed_columns_dir, f"{self.concept_index}_feature_connections.pt"))
		if lowMem:
			if(performRedundantCoalesce):
				self.feature_neurons = self.feature_neurons.coalesce()
				print("self.feature_neurons = ", self.feature_neurons)
			torch.save(self.feature_neurons, os.path.join(observed_columns_dir, f"{self.concept_index}_feature_neurons.pt"))

	@classmethod
	def load_from_disk(cls, concept_index, lemma, i):
		"""
		Load the observed column data from disk.
		"""
		# Load the data dictionary
		with open(os.path.join(observed_columns_dir, f"{concept_index}_data.pkl"), 'rb') as f:
			data = pickle.load(f)
		instance = cls(concept_index, lemma, i)
		instance.feature_word_to_index = data['feature_word_to_index']
		instance.feature_index_to_word = data['feature_index_to_word']
		instance.next_feature_index = data['next_feature_index']
		# Load the tensors
		instance.feature_connections = torch.load(os.path.join(observed_columns_dir, f"{concept_index}_feature_connections.pt"))
		if lowMem:
			instance.feature_neurons = torch.load(os.path.join(observed_columns_dir, f"{concept_index}_feature_neurons.pt"))
		return instance
		
# Define the SequenceObservedColumnsInferencePrediction class
class SequenceObservedColumnsInferencePrediction:
	def __init__(self, words, lemmas, observed_columns_dict, observed_columns_sequence_word_index_dict):
		#note cs may be slightly longer than number of unique columns in the sequence, if there are multiple instances of the same concept/noun lemma in the sequence
	
		self.observed_columns_dict = observed_columns_dict	# key: lemma, value: ObservedColumn
		self.observed_columns_sequence_word_index_dict = observed_columns_sequence_word_index_dict	# key: sequence word index, value: ObservedColumn
		
		self.cs2 = len(concept_columns_dict)
		self.fs2 = len(concept_features_dict)
			
		feature_connections_list = []
		for observed_column in observed_columns_sequence_word_index_dict.values():
			 feature_connections_list.append(observed_column.feature_connections)
		self.feature_connections = torch.stack(feature_connections_list, dim=2)

# Define the SequenceObservedColumns class
class SequenceObservedColumns:
	"""
	Contains sequence observed columns object arrays which stack a feature subset of the observed columns object arrays for the current sequence.
	"""
	def __init__(self, words, lemmas, observed_columns_dict, observed_columns_sequence_word_index_dict):
		#note cs may be slightly longer than number of unique columns in the sequence, if there are multiple instances of the same concept/noun lemma in the sequence
	
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
			self.concept_indices_in_sequence_observed_tensor = torch.tensor(self.concept_indices_in_observed_list, dtype=torch.long)
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
			self.concept_indices_in_sequence_observed_tensor = torch.tensor(self.concept_indices_in_observed_list, dtype=torch.long)
				
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
			feature_indices_in_observed_tensor = torch.tensor(feature_indices_in_observed, dtype=torch.long)
		else:
			feature_words = observed_column.feature_word_to_index.keys()
			feature_indices_in_observed_tensor = torch.tensor(list(observed_column.feature_word_to_index.values()), dtype=torch.long)
		
		if(sequenceObservedColumnsUseSequenceFeaturesOnly and sequenceObservedColumnsMatchSequenceWords):
			f_idx_tensor = torch.arange(len(feature_words), dtype=torch.long)
		else:
			feature_word_to_index = {}
			for idx, feature_word in enumerate(feature_words):
				feature_word_to_index[feature_word] = idx
			f_idx_tensor = torch.tensor([feature_word_to_index[fw] for fw in feature_words], dtype=torch.long)
		
		return feature_words, feature_indices_in_observed_tensor, f_idx_tensor
		
	def getObservedColumnFeatureIndices(self):
		return self.feature_indices_in_observed_tensor, self.f_idx_tensor
	
	def removeDuplicates(self, lst):
		#python requires ordered sets
		lst = list(dict.fromkeys(lst))
		return lst
				
	@staticmethod
	def initialiseFeatureNeuronsSequence(cs, fs):
		feature_neurons = torch.zeros(array_number_of_properties, array_number_of_segments, cs, fs, dtype=array_type)
		return feature_neurons

	@staticmethod
	def initialiseFeatureConnectionsSequence(cs, fs):
		feature_connections = torch.zeros(array_number_of_properties, array_number_of_segments, cs, fs, cs, fs, dtype=array_type)
		return feature_connections
	
	def populate_arrays(self, words, lemmas, sequence_observed_columns_dict):
		#print("\n\n\n\n\npopulate_arrays:")
		
		# Collect indices and data for feature neurons
		c_idx_list = []
		f_idx_list = []
		feature_list_indices = []
		feature_list_values = []

		if(not lowMem):
			global global_feature_neurons
					
		for c_idx, observed_column in sequence_observed_columns_dict.items():
			feature_indices_in_observed, f_idx_tensor = self.getObservedColumnFeatureIndices()

			num_features = len(f_idx_tensor)

			c_idx_list.append(torch.full((num_features,), c_idx, dtype=torch.long))
			f_idx_list.append(f_idx_tensor)

			if lowMem:
				feature_neurons = observed_column.feature_neurons.coalesce()
			else:
				feature_neurons = slice_sparse_tensor(global_feature_neurons, 2, observed_column.concept_index)	
			
			# Get indices and values from sparse tensor
			indices = feature_neurons.indices()
			values = feature_neurons.values()
			
			filter_feature_indices = torch.nonzero(indices[2].unsqueeze(1) == feature_indices_in_observed, as_tuple=True)
			filtered_indices = indices[:, filter_feature_indices[0]]
			filtered_f_idx_tensor = f_idx_tensor[filter_feature_indices[1]]
			filtered_values = values[filter_feature_indices[0]]
			# Adjust indices
			filtered_indices[0] = filtered_indices[0]  # properties
			filtered_indices[1] = filtered_indices[1]  # types
			filtered_indices[2] = filtered_f_idx_tensor
			filtered_indices = torch.cat([filtered_indices[0:2], torch.full_like(filtered_indices[2:3], c_idx), filtered_indices[2:3]], dim=0)	#insert dim3 for c_idx
			feature_list_indices.append(filtered_indices)
			feature_list_values.append(filtered_values)
	   
		# Combine indices and values
		if feature_list_indices:
			combined_indices = torch.cat(feature_list_indices, dim=1)
			combined_values = torch.cat(feature_list_values, dim=0)
			# Create sparse tensor
			self.feature_neurons = torch.sparse_coo_tensor(combined_indices, combined_values, size=self.feature_neurons.size(), dtype=array_type).to_dense()
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
				feature_idx_obs_mesh, other_feature_idx_obs_mesh = torch.meshgrid(feature_indices_in_observed, other_feature_indices_in_observed, indexing='ij')
				f_idx_mesh, other_f_idx_mesh = torch.meshgrid(f_idx_tensor, other_f_idx_tensor, indexing='ij')

				# Flatten the meshgrid indices
				feature_idx_obs_flat = feature_idx_obs_mesh.reshape(-1)
				other_feature_idx_obs_flat = other_feature_idx_obs_mesh.reshape(-1)
				f_idx_flat = f_idx_mesh.reshape(-1)
				other_f_idx_flat = other_f_idx_mesh.reshape(-1)
				
				# Filter indices for the desired features and concepts
				other_concept_index_expanded = torch.full(feature_idx_obs_flat.size(), fill_value=other_concept_index, dtype=torch.long)
				#print("feature_idx_obs_flat.shape = ", feature_idx_obs_flat.shape)
				filter_feature_indices2 = indices[2].unsqueeze(1) == feature_idx_obs_flat		
				filter_feature_indices3 = indices[3].unsqueeze(1) == other_concept_index_expanded
				filter_feature_indices4 = indices[4].unsqueeze(1) == other_feature_idx_obs_flat
				combined_condition = filter_feature_indices2 & filter_feature_indices3 & filter_feature_indices4
				filter_feature_indices = torch.nonzero(combined_condition, as_tuple=True)
				filtered_indices = indices[:, filter_feature_indices[0]]
				filtered_values = values[filter_feature_indices[0]]
				filtered_f_idx_tensor = f_idx_flat[filter_feature_indices[1]]
				filtered_other_f_idx_tensor = other_f_idx_flat[filter_feature_indices[1]]
						
				# Create tensors for concept indices
				c_idx_flat = torch.full_like(f_idx_flat, c_idx, dtype=torch.long)
				other_c_idx_flat = torch.full_like(other_f_idx_flat, other_c_idx, dtype=torch.long)
				filtered_other_c_idx_flat = other_c_idx_flat[filter_feature_indices[1]]
				
				# Adjust indices
				filtered_indices[0] = filtered_indices[0]  # properties
				filtered_indices[1] = filtered_indices[1]  # types
				filtered_indices[2] = filtered_f_idx_tensor
				filtered_indices[3] = filtered_other_c_idx_flat
				filtered_indices[4] = filtered_other_f_idx_tensor
				filtered_indices = torch.cat([filtered_indices[0:2], torch.full_like(filtered_indices[2:3], c_idx), filtered_indices[2:]], dim=0)	#insert dim3 for c_idx
				connection_indices_list.append(filtered_indices)
				connection_values_list.append(filtered_values)

		# Combine indices and values
		if connection_indices_list:
			combined_indices = torch.cat(connection_indices_list, dim=1)
			combined_values = torch.cat(connection_values_list, dim=0)
			# Create sparse tensor
			self.feature_connections = torch.sparse_coo_tensor(combined_indices, combined_values, size=self.feature_connections.size(), dtype=array_type).to_dense()
			self.feature_connections_original = self.feature_connections.clone()
			
	def update_observed_columns_wrapper(self):
		if(sequenceObservedColumnsMatchSequenceWords):
			#for multiple instances of concept in sequence, need to take the sum of the changes between the existing and modified arrays for each instance of a same concept in the sequence
			self.update_observed_columns(self.sequence_observed_columns_dict, mode="default")
		else:
			self.update_observed_columns(self.observed_columns_dict2, mode="default")
			
	def update_observed_columns(self, sequence_observed_columns_dict, mode):
		# Update observed columns with data from sequence arrays
		
		if(not lowMem):
			global global_feature_neurons
			
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
				observed_column.feature_neurons = slice_sparse_tensor(global_feature_neurons, 2, concept_index)	
			
			# feature neurons;
			# Use advanced indexing to get values from feature_neurons
			indices = feature_neurons.indices()
			values = feature_neurons.values()
			#convert indices from SequenceObservedColumns neuronFeatures array indices to ObservedColumns neuronFeatures array indices			
			# Filter indices
			mask = (indices[2] == c_idx) & torch.isin(indices[3], f_idx_tensor)
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
				observed_column.feature_neurons = observed_column.feature_neurons + torch.sparse_coo_tensor(filtered_indices, filtered_values, size=observed_column.feature_neurons.size(), dtype=array_type)
				observed_column.feature_neurons = observed_column.feature_neurons.coalesce()
				observed_column.feature_neurons.values().clamp_(min=0)
			else:
				self.feature_neuron_changes[c_idx] = torch.sparse_coo_tensor(filtered_indices, filtered_values, size=observed_column.feature_neurons.size(), dtype=array_type)
			
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
			observed_column.feature_connections = observed_column.feature_connections + torch.sparse_coo_tensor(filtered_indices, filtered_values, size=observed_column.feature_connections.size(), dtype=array_type)
			observed_column.feature_connections = observed_column.feature_connections.coalesce()
			observed_column.feature_connections.values().clamp_(min=0)
	
		if not lowMem:
			observed_column_feature_neurons_dict = {}
			for c_idx, observed_column in sequence_observed_columns_dict.items():
				concept_index = observed_column.concept_index
				observed_column_feature_neurons_dict[concept_index] = self.feature_neuron_changes[c_idx]
			global_feature_neurons = merge_tensor_slices_sum(global_feature_neurons, observed_column_feature_neurons_dict, 2)
		
	
# Initialize global feature neuron arrays if lowMem is disabled
if not lowMem:
	def initialiseFeatureNeuronsGlobal(c, f):
		feature_neurons = createEmptySparseTensor((array_number_of_properties, array_number_of_segments, c, f))
		return feature_neurons
		
	if os.path.exists(global_feature_neurons_file):
		global_feature_neurons = torch.load(global_feature_neurons_file)
	else:
		global_feature_neurons = initialiseFeatureNeuronsGlobal(c, f)
		#print("initialiseFeatureNeuronsGlobal: global_feature_neurons = ", global_feature_neurons)

def process_prompt():
	global sentence_count
	with open(inference_prompt_file, 'r', encoding='utf-8') as file:
		text = file.read()
	process_article(text)
	
def process_dataset(dataset):
	global sentence_count
	for article in dataset:
		process_article(article['text'])

def process_article(text):
	global sentence_count
	#sentences = sent_tokenize(text)
	sentences = nlp(text)
	numberOfSentences = len(list(sentences.sents))
	for sentenceIndex, sentence in enumerate(sentences.sents):
		lastSentenceInPrompt = False
		if(useInference and sentenceIndex == numberOfSentences-1):
			lastSentenceInPrompt = True
		process_sentence(sentenceIndex, sentence, lastSentenceInPrompt)
		if sentence_count == max_sentences_train:
			exit(0)
			
def process_sentence(sentenceIndex, doc, lastSentenceInPrompt):
	global sentence_count, c, f, concept_columns_dict, concept_columns_list, concept_features_dict, concept_features_list
	if(not lowMem):
		global global_feature_neurons
	
	if(debugReloadGlobalFeatureNeuronsEverySentence):
		#global_feature_neurons = torch.load(global_feature_neurons_file)
		initialiseConceptColumnsDictionary()
		if(not lowMem):
			global_feature_neurons = initialiseFeatureNeuronsGlobal(c, f)

	print(f"Processing sentence: {sentenceIndex} {doc.text}")

	# Refresh the observed columns dictionary for each new sequence
	observed_columns_dict = {}  # key: lemma, value: ObservedColumn
	observed_columns_sequence_word_index_dict = {}  # key: sequence word index, value: ObservedColumn
	
	if(lastSentenceInPrompt):
		doc_seed = doc[0:num_seed_tokens]	#prompt
		doc_predict = doc[num_seed_tokens:]

	# First pass: Extract words, lemmas, POS tags, and update concept_columns_dict and c
	concepts_found, words, lemmas, pos_tags = first_pass(doc)
	
	if(concepts_found):
		# When usePOS is enabled, detect all possible new features in the sequence
		if not (useDedicatedFeatureLists):
			detect_new_features(words, lemmas, pos_tags)

		# Second pass: Create observed_columns_dict
		observed_columns_dict, observed_columns_sequence_word_index_dict = second_pass(lemmas, pos_tags)

		# Create the sequence observed columns object
		sequence_observed_columns = SequenceObservedColumns(words, lemmas, observed_columns_dict, observed_columns_sequence_word_index_dict)

		if(lastSentenceInPrompt):
			# Process each concept word in the sequence (predict)
			process_concept_words_inference(sequence_observed_columns, doc, doc_seed, doc_predict, num_seed_tokens, num_prediction_tokens)
		else:
			# Process each concept word in the sequence (train)
			process_concept_words(doc, words, lemmas, pos_tags, sequence_observed_columns)

			# Update observed columns from sequence observed columns
			sequence_observed_columns.update_observed_columns_wrapper()

			if(drawNetworkDuringTrain):
				# Visualize the complete graph every time a new sentence is parsed by the application.
				visualize_graph(sequence_observed_columns)

			# Save observed columns to disk
			if(useSaveData):
				save_data(observed_columns_dict, concept_features_dict)
			
			'''
			if(useActivationDecrement):
				#decrement activation after each train interval; not currently used
				global_feature_neurons[array_index_properties_activation, array_index_segment_first] -= activationDecrementPerPredictedSentence
				global_feature_neurons[array_index_properties_activation, array_index_segment_first] = torch.clamp(global_feature_neurons[array_index_properties_activation, array_index_segment_first], min=0)
			'''
			
	# Break if we've reached the maximum number of sentences
	global sentence_count
	sentence_count += 1
		
def first_pass(doc):
	words = []
	lemmas = []
	pos_tags = []
	new_concepts_added = False
	concepts_found = False
	
	for token in doc:
		word = token.text.lower()
		lemma = token.lemma_.lower()
		pos = token.pos_  # Part-of-speech tag

		if usePOS:
			if pos in noun_pos_tags:
				# Only assign unique concept columns for nouns
				concepts_found, new_concepts_added = addConceptToConceptColumnsDict(lemma, concepts_found, new_concepts_added)
		else:
			# When usePOS is disabled, assign concept columns for every new lemma encountered
			concepts_found, new_concepts_added = addConceptToConceptColumnsDict(lemma, concepts_found, new_concepts_added)

		words.append(word)
		lemmas.append(lemma)
		pos_tags.append(pos)

	# If new concept columns have been added, expand arrays as needed
	if new_concepts_added:
		if not lowMem:
			# Expand global feature neuron arrays
			global global_feature_neurons
			if global_feature_neurons.shape[2] < c:
				new_shape = (global_feature_neurons.shape[0], global_feature_neurons.shape[1], c, global_feature_neurons.shape[3])
				if(performRedundantCoalesce):
					global_feature_neurons = global_feature_neurons.coalesce()
				global_feature_neurons = torch.sparse_coo_tensor(global_feature_neurons._indices(), global_feature_neurons._values(), size=new_shape, dtype=array_type)
				
	return concepts_found, words, lemmas, pos_tags

def addConceptToConceptColumnsDict(lemma, concepts_found, new_concepts_added):
	global c, concept_columns_dict, concept_columns_list
	concepts_found = True
	if lemma not in concept_columns_dict:
		# Add to concept columns dictionary
		#print("adding concept = ", lemma)
		concept_columns_dict[lemma] = c
		concept_columns_list.append(lemma)
		c += 1
		new_concepts_added = True
	return concepts_found, new_concepts_added
					
def second_pass(lemmas, pos_tags):
	observed_columns_dict = {}
	observed_columns_sequence_word_index_dict = {}
	for i, lemma in enumerate(lemmas):
		pos = pos_tags[i]
		if usePOS:
			if pos in noun_pos_tags:
				concept_index = concept_columns_dict[lemma]
				# Load observed column from disk or create new one
				observed_column = load_or_create_observed_column(concept_index, lemma, i)
				observed_columns_dict[lemma] = observed_column
				observed_columns_sequence_word_index_dict[i] = observed_column
		else:
			concept_index = concept_columns_dict[lemma]
			# Load observed column from disk or create new one
			observed_column = load_or_create_observed_column(concept_index, lemma, i)
			observed_columns_dict[lemma] = observed_column
			observed_columns_sequence_word_index_dict[i] = observed_column
	return observed_columns_dict, observed_columns_sequence_word_index_dict

def load_or_create_observed_column(concept_index, lemma, i):
	observed_column_file = os.path.join(observed_columns_dir, f"{concept_index}_data.pkl")
	if os.path.exists(observed_column_file):
		observed_column = ObservedColumn.load_from_disk(concept_index, lemma, i)
		# Resize connection arrays if c has increased
		observed_column.resize_concept_arrays(c)
		# Also expand feature arrays if f has increased
		observed_column.expand_feature_arrays(f)
	else:
		observed_column = ObservedColumn(concept_index, lemma, i)
		# Initialize connection arrays with correct size
		observed_column.resize_concept_arrays(c)
		observed_column.expand_feature_arrays(f)
	return observed_column

def detect_new_features(words, lemmas, pos_tags):
	"""
	When usePOS mode is enabled, detect all possible new features in the sequence
	by searching for all new non-nouns in the sequence.
	"""
	global f

	num_new_features = 0
	for j, (word_j, pos_j) in enumerate(zip(words, pos_tags)):
		if(process_feature_detection(j, word_j, pos_tags)):
			num_new_features += 1

	# After processing all features, update f
	f += num_new_features

	# Now, expand arrays accordingly
	if not lowMem:
		global global_feature_neurons
		if f > global_feature_neurons.shape[3]:
			extra_cols = f - global_feature_neurons.shape[3]
			new_shape = (global_feature_neurons.shape[0], global_feature_neurons.shape[1], global_feature_neurons.shape[2], f)
			global_feature_neurons = global_feature_neurons.coalesce()
			global_feature_neurons = torch.sparse_coo_tensor(global_feature_neurons.indices(), global_feature_neurons.values(), size=new_shape, dtype=array_type)

def process_feature_detection(j, word_j, pos_tags):
	"""
	Helper function to detect new features prior to processing concept words.
	"""
	global concept_features_dict, concept_features_list
	
	pos_j = pos_tags[j]
	feature_word = word_j.lower()
	
	if usePOS:
		if pos_j in noun_pos_tags:
			return False  # Skip nouns as features

	if feature_word not in concept_features_dict:
		concept_features_dict[feature_word] = len(concept_features_dict)
		concept_features_list.append(feature_word)
		return True
	else:
		return False
	
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
	
def process_concept_words_inference(sequence_observed_columns, doc, doc_seed, doc_predict, num_seed_tokens, num_prediction_tokens):

	print("process_concept_words_inference:")
	
	global global_feature_neurons, global_feature_neurons_activation
	
	sequenceWordIndex = 0
	
	words_seed, lemmas_seed, pos_tags_seed = getLemmas(doc_seed)
	concept_mask_seed = torch.tensor([i in sequence_observed_columns.columns_index_sequence_word_index_dict for i in range(len(lemmas_seed))], dtype=torch.bool)
	concept_indices_seed = torch.nonzero(concept_mask_seed).squeeze(1)
	numberConceptsInSeed = concept_indices_seed.shape[0]
	
	#seed network;
	words, lemmas, pos_tags= getLemmas(doc)
	process_concept_words(doc, words, lemmas, pos_tags, sequence_observed_columns, train=False, num_seed_tokens=num_seed_tokens, numberConceptsInSeed=numberConceptsInSeed)
	
	if(not inferenceSeedTargetActivationsGlobalFeatureArrays):
		# Update observed columns from sequence observed columns
		sequence_observed_columns.update_observed_columns_wrapper()	#convert sequence observed columns feature neuron arrays back to global feature neuron arrays

	visualize_graph(sequence_observed_columns)
	
	'''
	global_feature_neurons_dense = global_feature_neurons.to_dense()
	print("global_feature_neurons_dense = ", global_feature_neurons_dense)
	'''
	
	global_feature_neurons_activation = slice_sparse_tensor(global_feature_neurons, 0, array_index_properties_activation)

	#identify first activated column(s) in prediction phase:
	concept_columns_indices = None #not currently used (target connection neurons have already been activated)
	concept_columns_feature_indices = None	#not currently used (target connection neurons have already been activated)
	
	observed_columns_dict = sequence_observed_columns.observed_columns_dict  # key: lemma, value: ObservedColumn	#every observed column in inference (seed and prediction phases)
			
	#predict next tokens;
	for wordPredictionIndex in range(num_prediction_tokens):
		sequenceWordIndex = num_seed_tokens + wordPredictionIndex
		featurePredictionTargetMatch, concept_columns_indices, concept_columns_feature_indices = process_column_inference_prediction(observed_columns_dict, wordPredictionIndex, sequenceWordIndex, doc_predict, concept_columns_indices, concept_columns_feature_indices)
		

if not drawSequenceObservedColumns:
	class SequenceObservedColumnsDraw:
		def __init__(self, observed_columns_dict):
			self.observed_columns_dict = observed_columns_dict
							
def process_column_inference_prediction(observed_columns_dict, wordPredictionIndex, sequenceWordIndex, doc_predict, concept_columns_indices, concept_columns_feature_indices):
	global global_feature_neurons_activation
	
	print(f"process_column_inference_prediction: {wordPredictionIndex}; concept_columns_indices = ", concept_columns_indices)

	if(wordPredictionIndex > 0):
		# Refresh the observed columns dictionary for each new sequence
		observed_columns_sequence_candidate_index_dict = {}  # key: sequence candidate index, value: ObservedColumn	#used to populate sequence feature connection arrays based on observed columns (i does not correspond to sequence word index as assumed by observed_columns_sequence_word_index_dict)

		#populate sequence observed columns;
		words = []
		lemmas = []
		concept_columns_indices_list = concept_columns_indices.tolist()
		for i, concept_index in enumerate(concept_columns_indices_list):
			lemma = concept_columns_list[concept_index]
			word = lemma	#same for concepts (not used)
			lemmas.append(lemma)
			words.append(word)
			# Load observed column from disk or create new one
			observed_column = load_or_create_observed_column(concept_index, lemma, sequenceWordIndex)
			observed_columns_dict[lemma] = observed_column
			observed_columns_sequence_candidate_index_dict[i] = observed_column
		sequence_observed_columns_prediction = SequenceObservedColumnsInferencePrediction(words, lemmas, observed_columns_dict, observed_columns_sequence_candidate_index_dict)
	
		#process features (activate global target neurons);
		#process_features_active_predict(sequence_observed_columns_prediction, concept_columns_indices, concept_columns_feature_indices)
		process_features_active_predict_single(sequence_observed_columns_prediction, concept_columns_indices, concept_columns_feature_indices)

		#decrement activations;
		if(useActivationDecrement):
			#decrement activation after each prediction interval
			global_feature_neurons_activation -= activationDecrementPerPredictedColumn
			global_feature_neurons_activation.values().clamp_(min=0)
		if(deactivateNeuronsUponPrediction):
			indices_to_update_list = []
			for segment_index in range(array_number_of_segments):
				number_features_predicted = concept_columns_indices.shape[0]
				index_to_update = torch.stack([torch.tensor([segment_index]*number_features_predicted), concept_columns_indices, concept_columns_feature_indices.squeeze(dim=0)], dim=0)
				indices_to_update_list.append(index_to_update)
			indices_to_update = torch.stack(indices_to_update_list, dim=0).squeeze(dim=2)
			global_feature_neurons_activation = modify_sparse_tensor(global_feature_neurons_activation, indices_to_update, 0)
			#global_feature_neurons_activation[concept_columns_indices, concept_columns_feature_indices] = 0	
	else:
		#activation targets have already been activated
		sequence_observed_columns_prediction = SequenceObservedColumnsDraw(observed_columns_dict)
		
	global_feature_neurons_activation_all_segments = torch.sum(global_feature_neurons_activation, dim=0)	#sum across all segments 	#TODO: take into account SANI requirements (distal activation must precede proximal activation) 

	#topk column selection;
	concept_columns_activation = torch.sum(global_feature_neurons_activation_all_segments, dim=1)	#sum across all feature activations in columns
	concept_columns_activation = concept_columns_activation.to_dense()	#convert to dense tensor (required for topk)
	if(kcDynamic):
		concept_columns_activation = concept_columns_activation[concept_columns_activation > kcActivationThreshold]	#select kcMax columns above threshold
	concept_columns_activation_topk_concepts = torch.topk(concept_columns_activation, kcMax)
	kc = len(concept_columns_activation_topk_concepts.indices)
	if(kcDynamic and kc < 1):
		print("process_column_prediction kcDynamic error: kc < 1; cannot continue to predict columns; consider disabling kcDynamic for debug")
		exit()

	#top feature selection;
	if(kc==1):
		topk_concept_columns_activation = slice_sparse_tensor(global_feature_neurons_activation_all_segments, 0, concept_columns_activation_topk_concepts.indices[0]).unsqueeze(0)	#select topk concept indices
	else:
		topk_concept_columns_activation = slice_sparse_tensor_multi(global_feature_neurons_activation_all_segments, 0, concept_columns_activation_topk_concepts.indices)	#select topk concept indices
	topk_concept_columns_activation = topk_concept_columns_activation.to_dense()
	topk_concept_columns_activation_topk_features = torch.topk(topk_concept_columns_activation, kf, dim=1)

	#print("concept_columns_activation_topk_concepts.values = ", concept_columns_activation_topk_concepts.values)	
	#print("concept_columns_activation_topk_concepts.indices = ", concept_columns_activation_topk_concepts.indices)	
	
	#print("topk_concept_columns_activation_topk_features.values = ", topk_concept_columns_activation_topk_features.values)	
	#print("topk_concept_columns_activation_topk_features.indices = ", topk_concept_columns_activation_topk_features.indices)	
	
	#compare topk column/feature predictions to doc_predict (target words);
	featurePredictionTargetMatch = False
	for columnPredictionIndex in range(kc):
		columnIndex = concept_columns_activation_topk_concepts.indices[columnPredictionIndex]
		columnName = concept_columns_list[columnIndex]
		observedColumnFeatureIndex = topk_concept_columns_activation_topk_features.indices[columnPredictionIndex, 0]
		if(observedColumnFeatureIndex == feature_index_concept_neuron):
			predictedWord = columnName
		else:
			predictedWord = concept_features_list[observedColumnFeatureIndex]
		targetWord = doc_predict[wordPredictionIndex].text
		print("\t columnName = ", columnName, ", sequenceWordIndex = ", sequenceWordIndex, ", wordPredictionIndex = ", wordPredictionIndex, ", targetWord = ", targetWord, ", predictedWord = ", predictedWord)
		if(targetWord == predictedWord):
			featurePredictionTargetMatch = True
			
	concept_columns_indices_next = concept_columns_activation_topk_concepts.indices
	concept_columns_feature_indices_next = topk_concept_columns_activation_topk_features.indices
	print("concept_columns_indices_next = ", concept_columns_indices_next)
	print("concept_columns_feature_indices_next = ", concept_columns_feature_indices_next)

	#FUTURE: convert global_feature_neurons_activation back to global_feature_neurons for draw
	visualize_graph(sequence_observed_columns_prediction)
	
	return featurePredictionTargetMatch, concept_columns_indices_next, concept_columns_feature_indices_next
	 
#first dim cs1 restricted to a single token
def process_features_active_predict_single(sequence_observed_columns, concept_columns_indices, concept_columns_feature_indices):
	global global_feature_neurons_activation
	
	feature_neurons_active = slice_sparse_tensor(global_feature_neurons_activation, 0, array_index_segment_internal_column)	 		#select last (most proximal) segment activation	#TODO: checkthis
	feature_neurons_active = slice_sparse_tensor(feature_neurons_active, 0, concept_columns_indices.squeeze().item())	#select columns
	feature_neurons_active = slice_sparse_tensor(feature_neurons_active, 0, concept_columns_feature_indices.squeeze().squeeze().item())	#select features
	#print("feature_neurons_active = ", feature_neurons_active)
	
	#target neuron activation dependence on connection strength;
	#print("feature_neurons_active.shape = ", feature_neurons_active.shape)
	feature_connections = slice_sparse_tensor(sequence_observed_columns.feature_connections, 0, array_index_properties_strength)
	feature_connections = slice_sparse_tensor(feature_connections, 1, 0)	#sequence concept index dimension (not used)
	feature_connections = slice_sparse_tensor(feature_connections, 1, concept_columns_feature_indices.squeeze())
	#print("feature_connections.shape = ", feature_connections.shape)
	feature_neurons_target_activation = feature_neurons_active * feature_connections
	
	#update the activations of the target nodes;
	global_feature_neurons_activation += feature_neurons_target_activation*j1
		
#first dim cs1 restricted to a single token (or candiate set of tokens).
def process_features_active_predict(sequence_observed_columns, concept_columns_indices, concept_columns_feature_indices):
	global global_feature_neurons_activation
	
	feature_neurons_active = slice_sparse_tensor(global_feature_neurons_activation, 0, array_index_segment_internal_column)	 		#select last (most proximal) segment activation	#TODO: checkthis
	feature_neurons_active = slice_sparse_tensor_multi(global_feature_neurons_activation, 0, concept_columns_indices)	#select columns
	feature_neurons_active = slice_sparse_tensor_multi(feature_neurons_active, 0, concept_columns_feature_indices)	#select features
	
	#target neuron activation dependence on connection strength;
	feature_connections = slice_sparse_tensor(sequence_observed_columns.feature_connections, 0, array_index_properties_strength)
	feature_connections = slice_sparse_tensor(feature_connections, 1, 0)	#sequence concept index dimension (not used)
	feature_connections = slice_sparse_tensor_multi(feature_connections, 1, concept_columns_feature_indices)
	feature_neurons_target_activation = feature_neurons_active * feature_connections
	
	#update the activations of the target nodes;
	global_feature_neurons_activation += feature_neurons_target_activation*j1


				
def process_concept_words(doc, words, lemmas, pos_tags, sequence_observed_columns, train=True, num_seed_tokens=None, numberConceptsInSeed=None):
	"""
	For every concept word (lemma) in the sequence, identify every feature neuron in that column that occurs q words before or after the concept word in the sequence, including the concept neuron. This function has been parallelized using PyTorch array operations.
	"""
	global c, f, global_feature_neurons

	if not usePOS:
		q = 5  # Fixed window size when not using POS tags

	# Identify all concept word indices
	#print("\n\nsequence_observed_columns.columns_index_sequence_word_index_dict = ", sequence_observed_columns.columns_index_sequence_word_index_dict)
	concept_mask = torch.tensor([i in sequence_observed_columns.columns_index_sequence_word_index_dict for i in range(len(lemmas))], dtype=torch.bool)
	concept_indices = torch.nonzero(concept_mask).squeeze(1)
	
	#concept_indices may be slightly longer than number of unique columns in sequence, if there are multiple instances of the same concept/noun lemma in the sequence
	
	numberConceptsInSequence = concept_indices.shape[0]	#concept_indices.numel()	
	if numberConceptsInSequence == 0:
		return  # No concept words to process

	if usePOS:
		# Sort concept_indices
		concept_indices_sorted = concept_indices.sort().values
		
		# Find previous concept indices for each concept index
		prev_concept_positions = torch.searchsorted(concept_indices_sorted, concept_indices, right=False) - 1
		prev_concept_exists = prev_concept_positions >= 0
		prev_concept_positions = prev_concept_positions.clamp(min=0)
		prev_concept_indices = torch.where(prev_concept_exists, concept_indices_sorted[prev_concept_positions], torch.zeros_like(concept_indices))
		dist_to_prev_concept = torch.where(prev_concept_exists, concept_indices - prev_concept_indices, concept_indices+1) #If no previous concept, distance is the index itself
		
		# Find next concept indices for each concept index
		next_concept_positions = torch.searchsorted(concept_indices_sorted, concept_indices, right=True)
		next_concept_exists = next_concept_positions < len(concept_indices)
		next_concept_positions = next_concept_positions.clamp(max=len(next_concept_positions)-1)
		next_concept_indices = torch.where(next_concept_exists, concept_indices_sorted[next_concept_positions], torch.full_like(concept_indices, len(doc)))	# If no next concept, set to len(doc)
		dist_to_next_concept = torch.where(next_concept_exists, next_concept_indices - concept_indices, len(doc) - concept_indices)	# Distance to end if no next concept
	else:
		q = 5
		dist_to_prev_concept = torch.full((concept_indices.size(0),), q, dtype=torch.long)
		dist_to_next_concept = torch.full((concept_indices.size(0),), q, dtype=torch.long)

	# Calculate start and end indices for each concept word
	if(conceptColumnsDelimitByConceptFeaturesStart):
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

	process_features(start_indices, end_indices, doc, words, lemmas, pos_tags, sequence_observed_columns, concept_indices, train, num_seed_tokens, numberConceptsInSeed)
	
	return concept_indices, start_indices, end_indices
	
def process_features(start_indices, end_indices, doc, words, lemmas, pos_tags, sequence_observed_columns, concept_indices, train, num_seed_tokens=None, numberConceptsInSeed=None):
	numberConceptsInSequence = concept_indices.shape[0]
	
	cs = sequence_observed_columns.cs #!sequenceObservedColumnsMatchSequenceWords: will be less than len(concept_indices) if there are multiple instances of a concept in a sequence
	fs = sequence_observed_columns.fs  #sequenceObservedColumnsUseSequenceFeaturesOnly+sequenceObservedColumnsMatchSequenceWords: len(doc), sequenceObservedColumnsUseSequenceFeaturesOnly+!sequenceObservedColumnsMatchSequenceWords: number of feature neurons in sentence, !sequenceObservedColumnsUseSequenceFeaturesOnly: number of feature neurons in column
	feature_neurons_active = torch.zeros((array_number_of_segments, cs, fs), dtype=array_type)
	feature_neurons_word_order = torch.arange(fs).unsqueeze(0).repeat(cs, 1)
	torch.zeros((cs, fs), dtype=torch.long)
	columns_word_order = torch.zeros((cs), dtype=torch.long)
	feature_neurons_pos = torch.zeros((cs, fs), dtype=array_type)
	if(sequenceObservedColumnsMatchSequenceWords):
		sequence_concept_index_mask = torch.ones((cs, fs), dtype=array_type)	#ensure to ignore concept feature neurons from other columns
	else:
		sequence_concept_index_mask = None
	feature_neurons_segment_mask = torch.zeros((cs, array_number_of_segments), dtype=array_type)
	
	concept_indices_list = concept_indices.tolist()
	#convert start/end indices to active features arrays
	for i, sequence_concept_word_index in enumerate(concept_indices_list):
		if(sequenceObservedColumnsMatchSequenceWords):
			sequence_concept_index = i
		else:
			concept_lemma = lemmas[concept_indices[i]]
			sequence_concept_index = sequence_observed_columns.concept_name_to_index[concept_lemma] 
		
		number_of_segments = min(array_number_of_segments-1, i)
		feature_neurons_segment_mask[sequence_concept_index, :] = torch.cat([torch.zeros(array_number_of_segments-number_of_segments), torch.ones(number_of_segments)], dim=0)
		minSequentialSegmentIndex = min(0, array_number_of_segments-sequence_concept_index-1)
		activeSequentialSegments = torch.arange(minSequentialSegmentIndex, array_number_of_segments, 1)
		
		if(sequenceObservedColumnsUseSequenceFeaturesOnly and sequenceObservedColumnsMatchSequenceWords):
			feature_neurons_active[activeSequentialSegments, sequence_concept_index, start_indices[sequence_concept_index]:end_indices[sequence_concept_index]] = 1
			columns_word_order[sequence_concept_index] = sequence_concept_index
			sequence_concept_index_mask[:, sequence_concept_word_index] = 0	#ignore concept feature neurons from other columns
			sequence_concept_index_mask[sequence_concept_index, sequence_concept_word_index] = 1
			for j in range(start_indices[sequence_concept_index], end_indices[sequence_concept_index]):
				feature_pos = pos_string_to_pos_int(pos_tags[j])
				feature_neurons_pos[sequence_concept_index, j] = feature_pos
				feature_neurons_word_order[sequence_concept_index, j] = j
		else:
			for j in range(start_indices[i], end_indices[i]):	#sequence word index
				feature_word = words[j].lower()
				feature_lemma = lemmas[j]
				feature_pos = pos_string_to_pos_int(pos_tags[j])
				if(j in sequence_observed_columns.columns_index_sequence_word_index_dict):	#test is required for concept neurons
					sequence_concept_word_index = j
					columns_word_order[sequence_concept_index] = sequence_concept_index	#alternatively use sequence_concept_word_index; not robust in either case - there may be less concept columns than concepts referenced in sequence (if multiple references to the same column). sequenceObservedColumnsMatchSequenceWords overcomes this limitation.
					if(useDedicatedConceptNames2):
						sequence_feature_index = sequence_observed_columns.feature_word_to_index[variableConceptNeuronFeatureName]
					else:
						sequence_feature_index = sequence_observed_columns.feature_word_to_index[feature_lemma]
					feature_neurons_active[activeSequentialSegments, sequence_concept_index, sequence_feature_index] = 1
					feature_neurons_word_order[sequence_concept_index, sequence_feature_index] = j
				elif(feature_word in sequence_observed_columns.feature_word_to_index):
					sequence_feature_index = sequence_observed_columns.feature_word_to_index[feature_word]
					feature_neurons_active[activeSequentialSegments, sequence_concept_index, sequence_feature_index] = 1
					feature_neurons_word_order[sequence_concept_index, sequence_feature_index] = j
				feature_neurons_pos[sequence_concept_index, sequence_feature_index] = feature_pos
	
	feature_neurons_segment_mask = feature_neurons_segment_mask.swapdims(0, 1)
	if(train):
		process_features_active_train(sequence_observed_columns, feature_neurons_active, cs, fs, sequence_concept_index_mask, columns_word_order, feature_neurons_word_order, feature_neurons_pos, feature_neurons_segment_mask)
	else:
		process_features_active_seed(sequence_observed_columns, feature_neurons_active, cs, fs, sequence_concept_index_mask, columns_word_order, feature_neurons_word_order, feature_neurons_pos, num_seed_tokens, numberConceptsInSeed)

#first dim cs1 pertains to every concept node in sequence
def process_features_active_train(sequence_observed_columns, feature_neurons_active, cs, fs, sequence_concept_index_mask, columns_word_order, feature_neurons_word_order, feature_neurons_pos, feature_neurons_segment_mask):
	feature_neurons_inactive = 1 - feature_neurons_active
		
	# Update feature neurons in sequence_observed_columns
	sequence_observed_columns.feature_neurons[array_index_properties_strength, :, :, :] += feature_neurons_active
	sequence_observed_columns.feature_neurons[array_index_properties_permanence, :, :, :] += feature_neurons_active*z1	#orig = feature_neurons_active*(sequence_observed_columns.feature_neurons[array_index_properties_permanence] ** 2) + feature_neurons_inactive*sequence_observed_columns.feature_neurons[array_index_properties_permanence]
	#sequence_observed_columns.feature_neurons[array_index_properties_activation, :, :, :] += feature_neurons_active*j1	#update the activations of the target not source nodes
	sequence_observed_columns.feature_neurons[array_index_properties_time, :, :, :] = feature_neurons_inactive*sequence_observed_columns.feature_neurons[array_index_properties_time] + feature_neurons_active*sentence_count
	sequence_observed_columns.feature_neurons[array_index_properties_pos, :, :, :] = feature_neurons_inactive*sequence_observed_columns.feature_neurons[array_index_properties_pos] + feature_neurons_active*feature_neurons_pos

	feature_connections_active, feature_connections_segment_mask = createFeatureConnectionsActive(feature_neurons_active[array_index_segment_internal_column], cs, fs, columns_word_order, feature_neurons_word_order)
	
	feature_connections_pos = feature_neurons_pos.view(1, cs, fs, 1, 1).expand(array_number_of_segments, cs, fs, cs, fs)

	feature_connections_inactive = 1 - feature_connections_active

	#prefer closer than further target neurons when strengthening connections (and activating target neurons) in sentence;
	feature_neurons_word_order_1d = feature_neurons_word_order.flatten()
	feature_connections_distances = torch.abs(feature_neurons_word_order_1d.unsqueeze(1) - feature_neurons_word_order_1d).reshape(cs, fs, cs, fs)
	feature_connections_proximity = 1/(feature_connections_distances + 1) * 10
	feature_connections_proximity.unsqueeze(0)	#add SANI segment dimension
	feature_connections_strength_update = feature_connections_active*feature_connections_proximity
	#print("feature_connections_strength_update = ", feature_connections_strength_update)

	if(increaseColumnInternalConnectionsStrength):
		cs_indices_1 = torch.arange(cs).view(1, cs, 1, 1, 1).expand(array_number_of_segments, cs, fs, cs, fs)  # First cs dimension
		cs_indices_2 = torch.arange(cs).view(1, 1, 1, cs, 1).expand(array_number_of_segments, cs, fs, cs, fs)  # Second cs dimension
		column_internal_connections_mask = (cs_indices_1 == cs_indices_2)
		column_internal_connections_mask_off = torch.logical_not(column_internal_connections_mask)
		feature_connections_strength_update = column_internal_connections_mask.float()*feature_connections_strength_update*increaseColumnInternalConnectionsStrengthModifier + column_internal_connections_mask_off.float()*feature_connections_strength_update

	#print("feature_connections_active[array_index_segment_first] = ", feature_connections_active[array_index_segment_first])
	#print("feature_connections_active[array_index_segment_internal_column] = ", feature_connections_active[array_index_segment_internal_column])
	
	sequence_observed_columns.feature_connections[array_index_properties_strength, :, :, :, :, :] += feature_connections_strength_update
	sequence_observed_columns.feature_connections[array_index_properties_permanence, :, :, :, :, :] += feature_connections_active*z1	#orig = feature_connections_active*(sequence_observed_columns.feature_connections[array_index_properties_permanence] ** 2) + feature_connections_inactive*sequence_observed_columns.feature_connections[array_index_properties_permanence]
	#sequence_observed_columns.feature_connections[array_index_properties_activation, :, :, :, :, :] += feature_connections_active*j1	#connection activations are not currently used
	sequence_observed_columns.feature_connections[array_index_properties_time, :, :, :, :, :] = feature_connections_inactive*sequence_observed_columns.feature_connections[array_index_properties_time] + feature_connections_active*sentence_count
	sequence_observed_columns.feature_connections[array_index_properties_pos, :, :, :, :, :] = feature_connections_inactive*sequence_observed_columns.feature_connections[array_index_properties_pos] + feature_connections_active*feature_connections_pos

	#decrease permanence;
	if(decreasePermanenceOfInactiveFeatureNeuronsAndConnections):
		decrease_permanence_active(sequence_observed_columns, feature_neurons_active[array_index_segment_internal_column], feature_neurons_inactive[array_index_segment_internal_column], sequence_concept_index_mask, feature_neurons_segment_mask, feature_connections_segment_mask)
	
#first dim cs1 pertains to every concept node in sequence
def process_features_active_seed(sequence_observed_columns, feature_neurons_active, cs, fs, sequence_concept_index_mask, columns_word_order, feature_neurons_word_order, feature_neurons_pos, num_seed_tokens, numberConceptsInSeed):
	feature_neurons_inactive = 1 - feature_neurons_active
	
	fs2 = fs
	if(inferenceSeedTargetActivationsGlobalFeatureArrays):
		global global_feature_neurons
		cs2 = c
		feature_connections_active = torch.ones(array_number_of_segments, cs, fs, cs2, fs2)	#TODO: assign segment dimension
		print("feature_connections_active.shape = ", feature_connections_active.shape)
	else:
		cs2 = cs
		feature_connections_active, feature_connections_segment_mask = createFeatureConnectionsActive(feature_neurons_active[array_index_segment_internal_column], cs, fs, columns_word_order, feature_neurons_word_order)
	
	firstWordIndexPredictPhase = num_seed_tokens
	feature_neurons_word_order_expanded_1 = feature_neurons_word_order.view(cs, fs, 1, 1).expand(cs, fs, cs2, fs2)  # For the first node
	word_order_mask = feature_neurons_word_order_expanded_1 < firstWordIndexPredictPhase
	word_order_mask = word_order_mask.unsqueeze(0).expand(array_number_of_segments, cs, fs, cs2, fs2)
	feature_connections_active = feature_connections_active * word_order_mask
	
	#print("sequence_observed_columns.feature_connections[array_index_properties_strength] = ", sequence_observed_columns.feature_connections[array_index_properties_strength])

	#target neuron activation dependence on connection strength;
	feature_connections_activation_update = feature_connections_active * sequence_observed_columns.feature_connections[array_index_properties_strength]
	#print("feature_connections_activation_update = ", feature_connections_activation_update)
	
	#update the activations of the target nodes;
	print("feature_connections_activation_update.shape = ", feature_connections_activation_update.shape)
	feature_connections_activation_update = torch.sum(feature_connections_activation_update, dim=(0))	#TODO: take into account SANI requirements (distal activation must precede proximal activation) 
	feature_neurons_target_activation = torch.sum(feature_connections_activation_update, dim=(0, 1))	
	if(inferenceSeedTargetActivationsGlobalFeatureArrays):
		global_feature_neurons[array_index_properties_activation, :, :, :] += feature_neurons_target_activation*j1
	else:
		print("sequence_observed_columns.feature_neurons[array_index_properties_activation].shape = ", sequence_observed_columns.feature_neurons[array_index_properties_activation].shape)
		print("feature_neurons_target_activation.shape = ", feature_neurons_target_activation.shape)
		sequence_observed_columns.feature_neurons[array_index_properties_activation, :, :, :] += feature_neurons_target_activation*j1
		#will only activate target neurons in sequence_observed_columns (not suitable for inference seed/prediction phase)

	if(deactivateNeuronsUponPrediction):
		if(inferenceSeedTargetActivationsGlobalFeatureArrays):
			for sequence_concept_index, concept_index in enumerate(sequence_observed_columns.concept_indices_in_observed_list):
				global_feature_neurons[array_index_properties_activation, :, concept_index, :] *= feature_neurons_inactive[:, sequence_concept_index]	#TODO: assign segment dimension
		else:
			feature_neurons_source_mask = feature_neurons_word_order < num_seed_tokens
			feature_neurons_source_mask.unsqueeze(0).expand(array_number_of_segments, cs, fs)
			feature_neurons_active_source = torch.logical_and(feature_neurons_source_mask, feature_neurons_active > 0)
			feature_neurons_inactive_source = torch.logical_not(feature_neurons_active_source).float()
			sequence_observed_columns.feature_neurons[array_index_properties_activation, :, :, :] *= feature_neurons_inactive_source


def createFeatureConnectionsActive(feature_neurons_active, cs, fs, columns_word_order, feature_neurons_word_order):

	feature_neurons_active_1d = feature_neurons_active.view(cs*fs)
	feature_connections_active = torch.matmul(feature_neurons_active_1d.unsqueeze(1), feature_neurons_active_1d.unsqueeze(0)).view(cs, fs, cs, fs)

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
			columns_word_order_mask = torch.logical_and(columns_word_order_expanded_2 >= columns_word_order_expanded_1, columns_word_order_expanded_2 <= columns_word_order_expanded_1+1)
		else:
			columns_word_order_mask = columns_word_order_expanded_2 >= columns_word_order_expanded_1
		feature_connections_active = feature_connections_active * columns_word_order_mask
	
	#ensure identical feature nodes are not connected together;
	cs_indices_1 = torch.arange(cs).view(cs, 1, 1, 1).expand(cs, fs, cs, fs)  # First cs dimension
	cs_indices_2 = torch.arange(cs).view(1, 1, cs, 1).expand(cs, fs, cs, fs)  # Second cs dimension
	fs_indices_1 = torch.arange(fs).view(1, fs, 1, 1).expand(cs, fs, cs, fs)  # First fs dimension
	fs_indices_2 = torch.arange(fs).view(1, 1, 1, fs).expand(cs, fs, cs, fs)  # Second fs dimension
	identity_mask = (cs_indices_1 != cs_indices_2) | (fs_indices_1 != fs_indices_2)
	feature_connections_active = feature_connections_active * identity_mask

	feature_connections_active, feature_connections_segment_mask = assign_feature_connections_to_target_segments(feature_connections_active, cs, fs)
	
	return feature_connections_active, feature_connections_segment_mask

def assign_feature_connections_to_target_segments(feature_connections_active, cs, fs):

	#arrange active connections according to target neuron sequential segment index
	concept_neurons_concept_order_1d = torch.arange(cs)
	concept_neurons_distances = torch.abs(concept_neurons_concept_order_1d.unsqueeze(1) - concept_neurons_concept_order_1d).reshape(cs, cs)
	connections_segment_index = array_number_of_segments-concept_neurons_distances-1
	connections_segment_index = torch.clamp(connections_segment_index, min=0)
	
	feature_connections_segment_mask = torch.zeros((array_number_of_segments, cs, cs), dtype=torch.bool)
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
	sequence_observed_columns.feature_neurons[array_index_properties_permanence] = torch.clamp(sequence_observed_columns.feature_neurons[array_index_properties_permanence], min=0)

	feature_neurons_all = torch.ones((cs, fs), dtype=array_type)
	feature_neurons_all_1d = feature_neurons_all.view(cs*fs)
	feature_neurons_active_1d = feature_neurons_active.view(cs*fs)
	feature_neurons_inactive_1d = feature_neurons_inactive.view(cs*fs)
	 
	# Decrease permanence of connections from inactive feature neurons in column
	feature_connections_decrease1 = torch.matmul(feature_neurons_inactive_1d.unsqueeze(1), feature_neurons_all_1d.unsqueeze(0)).view(cs, fs, cs, fs)
	feature_connections_decrease1 = feature_connections_decrease1.unsqueeze(0)*feature_connections_segment_mask
	sequence_observed_columns.feature_connections[array_index_properties_permanence, :, :, :, :, :] -= feature_connections_decrease1
	sequence_observed_columns.feature_connections[array_index_properties_permanence] = torch.clamp(sequence_observed_columns.feature_connections[array_index_properties_permanence], min=0)
	
	# Decrease permanence of inactive connections for activated features in column 
	feature_connections_decrease2 = torch.matmul(feature_neurons_active_1d.unsqueeze(1), feature_neurons_inactive_1d.unsqueeze(0)).view(cs, fs, cs, fs)
	feature_connections_decrease2 = feature_connections_decrease2.unsqueeze(0)*feature_connections_segment_mask
	sequence_observed_columns.feature_connections[array_index_properties_permanence, :, :, :, :, :] -= feature_connections_decrease2
	sequence_observed_columns.feature_connections[array_index_properties_permanence] = torch.clamp(sequence_observed_columns.feature_connections[array_index_properties_permanence], min=0)
 
	#current limitation; will not deactivate neurons or remove their strength if their permanence goes to zero

def createNeuronLabelWithStrength(name, strength):
	label = name + "\n" + floatToString(strength)
	return label
	
def floatToString(value):
	result = str(round(value.item(), 2))
	return result
		
def visualize_graph(sequence_observed_columns):
	G.clear()

	if not lowMem:
		global global_feature_neurons
		if(performRedundantCoalesce):
			global_feature_neurons = global_feature_neurons.coalesce()

	# Draw concept columns
	pos_dict = {}
	x_offset = 0
	for lemma, observed_column in sequence_observed_columns.observed_columns_dict.items():
		concept_index = observed_column.concept_index
		
		if(performRedundantCoalesce):
			if lowMem:
				observed_column.feature_neurons = observed_column.feature_neurons.coalesce()
		
		if(drawSequenceObservedColumns):
			feature_word_to_index = sequence_observed_columns.feature_word_to_index
			y_offset = 1 + 1	#reserve space at bottom of column for feature concept neuron (as it will not appear first in sequence_observed_columns.feature_word_to_index, only observed_column.feature_word_to_index)
			c_idx = sequence_observed_columns.concept_name_to_index[lemma]
			feature_neurons = sequence_observed_columns.feature_neurons[:, :, c_idx]
		else:
			feature_word_to_index = observed_column.feature_word_to_index
			y_offset = 1
			if lowMem:
				feature_neurons = observed_column.feature_neurons
			else:
				feature_neurons = slice_sparse_tensor(global_feature_neurons, 2, concept_index)
				#feature_neurons = global_feature_neurons[:, :, concept_index]	#operation not supported for sparse tensors
					
		# Draw feature neurons
		for feature_word, feature_index_in_observed_column in feature_word_to_index.items():
			conceptNeuronFeature = False
			if(useDedicatedConceptNames and useDedicatedConceptNames2):
				if feature_word==variableConceptNeuronFeatureName:
					neuron_color = 'blue'
					neuron_name = observed_column.concept_name
					conceptNeuronFeature = True
					#print("\nvisualize_graph: conceptNeuronFeature neuron_name = ", neuron_name)
				else:
					neuron_color = 'turquoise'
					neuron_name = feature_word
			else:
				neuron_color = 'turqoise'
				neuron_name = feature_word

			f_idx = feature_index_in_observed_column	#not used
		
			featurePresent = False
			featureActive = False
			if(feature_neurons[array_index_properties_strength, array_index_segment_internal_column, feature_index_in_observed_column] > 0 and feature_neurons[array_index_properties_permanence, array_index_segment_internal_column, feature_index_in_observed_column] > 0):
				featurePresent = True
			if(feature_neurons[array_index_properties_activation, array_index_segment_internal_column, feature_index_in_observed_column] > 0):
				featureActive = True
				
			if(featurePresent):
				if(drawRelationTypes):
					if not conceptNeuronFeature:
						neuron_color = generateFeatureNeuronColour(feature_neurons[array_index_properties_pos, array_index_segment_internal_column, feature_index_in_observed_column], feature_word)
				elif(featureActive):
					if(conceptNeuronFeature):
						neuron_color = 'lightskyblue'
					else:
						neuron_color = 'cyan'
						
				if(debugDrawNeuronStrengths):
					neuron_name = createNeuronLabelWithStrength(neuron_name, feature_neurons[array_index_properties_activation, array_index_segment_internal_column, feature_index_in_observed_column])

				feature_node = f"{lemma}_{feature_word}_{f_idx}"
				if(randomiseColumnFeatureXposition and not conceptNeuronFeature):
					x_offset_shuffled = x_offset + random.uniform(-0.5, 0.5)
				else:
					x_offset_shuffled = x_offset
				if(drawSequenceObservedColumns and conceptNeuronFeature):
					y_offset_prev = y_offset
					y_offset = 1
				G.add_node(feature_node, pos=(x_offset_shuffled, y_offset), color=neuron_color, label=neuron_name)
				if(drawSequenceObservedColumns and conceptNeuronFeature):
					y_offset = y_offset_prev
				else:
					y_offset += 1

		# Draw rectangle around the column
		plt.gca().add_patch(plt.Rectangle((x_offset - 0.5, -0.5), 1, max(y_offset, 1) + 0.5, fill=False, edgecolor='black'))
		x_offset += 2  # Adjust x_offset for the next column

	# Draw connections
	for lemma, observed_column in sequence_observed_columns.observed_columns_dict.items():
	
		if(performRedundantCoalesce):
			observed_column.feature_connections = observed_column.feature_connections.coalesce()
					
		concept_index = observed_column.concept_index
		if(drawSequenceObservedColumns):
			feature_word_to_index = sequence_observed_columns.feature_word_to_index
			other_feature_word_to_index = sequence_observed_columns.feature_word_to_index
			c_idx = sequence_observed_columns.concept_name_to_index[lemma]
			feature_connections = sequence_observed_columns.feature_connections[:, :, c_idx]
		else:
			feature_word_to_index = observed_column.feature_word_to_index
			other_feature_word_to_index = observed_column.feature_word_to_index
			c_idx = concept_columns_dict[lemma]
			feature_connections = observed_column.feature_connections
		feature_connections = torch.sum(feature_connections, dim=1)	#sum along sequential segment index (draw connections to all segments)
	
		# Internal connections (yellow)
		for feature_word, feature_index_in_observed_column in feature_word_to_index.items():
			source_node = f"{lemma}_{feature_word}_{feature_index_in_observed_column}"
			if G.has_node(source_node):
				for other_feature_word, other_feature_index_in_observed_column in feature_word_to_index.items():
					target_node = f"{lemma}_{other_feature_word}_{other_feature_index_in_observed_column}"
					if G.has_node(target_node):
						if feature_word != other_feature_word:
							f_idx = feature_word_to_index[feature_word]
							other_f_idx = feature_word_to_index[other_feature_word]
							
							featurePresent = False
							if(feature_connections[array_index_properties_strength, f_idx, c_idx, other_f_idx] > 0 and feature_connections[array_index_properties_permanence, f_idx, c_idx, other_f_idx] > 0):
								featurePresent = True
								
							if(drawRelationTypes):
								connection_color = generateFeatureNeuronColour(feature_connections[array_index_properties_pos, f_idx, c_idx, other_f_idx], feature_word, internal_connection=True)
							else:
								connection_color = 'yellow'
								
							if(featurePresent):
								G.add_edge(source_node, target_node, color=connection_color)
		
		# External connections (orange)
		for feature_word, feature_index_in_observed_column in feature_word_to_index.items():
			source_node = f"{lemma}_{feature_word}_{feature_index_in_observed_column}"
			if G.has_node(source_node):
				for other_lemma, other_observed_column in sequence_observed_columns.observed_columns_dict.items():
					if(drawSequenceObservedColumns):
						other_feature_word_to_index = sequence_observed_columns.feature_word_to_index
					else:
						other_feature_word_to_index = other_observed_column.feature_word_to_index
					for other_feature_word, other_feature_index_in_observed_column in other_feature_word_to_index.items():
						target_node = f"{other_lemma}_{other_feature_word}_{other_feature_index_in_observed_column}"
						if G.has_node(target_node):
							f_idx = feature_word_to_index[feature_word]
							other_f_idx = other_feature_word_to_index[other_feature_word]
							
							externalConnection = False
							if(drawSequenceObservedColumns):
								other_c_idx = sequence_observed_columns.concept_name_to_index[other_lemma]
								if other_c_idx != c_idx:
									externalConnection = True
							else:
								other_c_idx = concept_columns_dict[other_lemma]
								if lemma != other_lemma:	#if observed_column != other_observed_column:
									externalConnection = True
					
							featurePresent = False
							if(externalConnection):
								#print("feature_connections[array_index_properties_strength, f_idx, other_c_idx, other_f_idx] = ", feature_connections[array_index_properties_strength, f_idx, other_c_idx, other_f_idx])
								#print("feature_connections[array_index_properties_permanence, f_idx, other_c_idx, other_f_idx] = ", feature_connections[array_index_properties_permanence, f_idx, other_c_idx, other_f_idx])
								if(feature_connections[array_index_properties_strength, f_idx, other_c_idx, other_f_idx] > 0 and feature_connections[array_index_properties_permanence, f_idx, other_c_idx, other_f_idx] > 0):
									featurePresent = True
									#print("\tfeaturePresent")

							if(drawRelationTypes):
								connection_color = generateFeatureNeuronColour(feature_connections[array_index_properties_pos, f_idx, other_c_idx, other_f_idx], feature_word, internal_connection=False)
							else:
								connection_color = 'orange'
								
							if(featurePresent):
								G.add_edge(source_node, target_node, color=connection_color)
								
	# Get positions and colors for drawing
	pos = nx.get_node_attributes(G, 'pos')
	colors = [data['color'] for node, data in G.nodes(data=True)]
	edge_colors = [data['color'] for u, v, data in G.edges(data=True)]
	labels = nx.get_node_attributes(G, 'label')

	# Draw the graph
	nx.draw(G, pos, with_labels=True, labels=labels, arrows=True, node_color=colors, edge_color=edge_colors, node_size=500, font_size=8)
	plt.axis('off')  # Hide the axes
	plt.show()

def save_data(observed_columns_dict, concept_features_dict):
	# Save observed columns to disk
	for observed_column in observed_columns_dict.values():
		observed_column.save_to_disk()

	# Save global feature neuron arrays if not lowMem
	if not lowMem:
		global global_feature_neurons
		if(performRedundantCoalesce):
			global_feature_neurons = global_feature_neurons.coalesce()
		torch.save(global_feature_neurons, global_feature_neurons_file)

	# Save concept columns dictionary to disk
	with open(concept_columns_dict_file, 'wb') as f_out:
		pickle.dump(concept_columns_dict, f_out)

	# Save concept features dictionary to disk
	with open(concept_features_dict_file, 'wb') as f_out:
		pickle.dump(concept_features_dict, f_out)

# Load the Wikipedia dataset using Hugging Face datasets
dataset = load_dataset('wikipedia', '20220301.en', split='train', streaming=True)

# Start processing the dataset
if(useInference or debugSmallDataset):
	process_prompt()
else:
	process_dataset(dataset)


