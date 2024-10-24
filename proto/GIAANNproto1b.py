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

# Set boolean variables as per specification
useInference = False  # useInference mode
if(useInference):
	lowMem = False		 # lowMem mode (can only be used when useInference is disabled)
	sequenceObservedColumnsUseSequenceFeaturesOnly = False	#must be set to false as global_feature_neurons are updated (which have complete feature lists, not sequence limited feature lists)
else:
	lowMem = True		 # lowMem mode (can only be used when useInference is disabled)
	sequenceObservedColumnsUseSequenceFeaturesOnly = True	#sequence observed columns arrays only store sequence features.	#optional (will affect which network changes can be visualised)

drawSequenceObservedColumns = False	#draw sequence observed columns (instead of complete observed columns)	#note if !drawSequenceObservedColumns and !sequenceObservedColumnsUseSequenceFeaturesOnly, then will still draw complete columns	#optional (will affect which network changes can be visualised)
useSaveData = True	#save data is required to allow consecutive sentence training and inference (because connection data are stored in observed columns, which are refreshed every sentence)

sequenceObservedColumnsMatchSequenceWords = False
if(sequenceObservedColumnsUseSequenceFeaturesOnly):
	sequenceObservedColumnsMatchSequenceWords = True	#optional	#introduced GIAANNproto1b12a; more robust method for training (independently train each instance of a concept in a sentence)

if(sequenceObservedColumnsMatchSequenceWords):
	#sumChangesToConceptNeuronSequenceInstances = True	#mandatory	#for multiple instances of concept in sequence, need to take the sum of the changes between the existing and modified arrays for each instance of a same concept in the sequence
	assert not drawSequenceObservedColumns, "sequenceObservedColumnsMatchSequenceWords does not currently support drawSequenceObservedColumns"

usePOS = True		 # usePOS mode	#mandatory
useParallelProcessing = True	#mandatory (else restore original code pre-GIAANNproto1b3a)
randomiseColumnFeatureXposition = True	#shuffle x position of column internal features such that their connections can be better visualised

#debug vars;
debugSmallDataset = False

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

inference_prompt_file = 'inference_prompt.txt'
if(useInference):
	num_seed_tokens = 5	#number of seed tokens in last sentence of inference prompt (remaining tokens will be prediction tokens)
	
	deactivateNeuronsUponPrediction = True

	#TODO: train hyperparameters
	num_prediction_tokens = 10	#number of words to predict after network seed
	
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
concept_columns_dict_file = 'concept_columns_dict.pkl'
concept_features_dict_file = 'concept_features_dict.pkl'
observed_columns_dir = 'observed_columns'
os.makedirs(observed_columns_dir, exist_ok=True)

#common array indices
array_index_properties_strength = 0
array_index_properties_permanence = 1
array_index_properties_activation = 2
array_index_type_all = 0
array_type = torch.float32	#torch.long	#torch.float32

# Define POS tag sets for nouns and non-nouns
noun_pos_tags = {'NOUN', 'PROPN'}
non_noun_pos_tags = {'ADJ', 'ADV', 'VERB', 'ADP', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NUM', 'PART', 'PRON', 'SCONJ', 'SYM', 'X'}

# Define constants for permanence and activation trace	#TODO: train hyperparameters
z1 = 3  # Initial permanence value	
z2 = 1  # Decrement value when not activated
j1 = 5   # Activation trace duration

# Initialize NetworkX graph for visualization
G = nx.DiGraph()

# For the purpose of the example, process a limited number of sentences
sentence_count = 0
max_sentences_train = 1000  # Adjust as needed


if not lowMem:
	global_feature_neurons_file = 'global_feature_neurons.pt'

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
	concept_columns_dict = {}  # key: lemma, value: index
	concept_columns_list = []  # list of concept column names (lemmas)
	c = 0  # current number of concept columns
	concept_features_dict = {}  # key: lemma, value: index
	concept_features_list = []  # list of concept column names (lemmas)
	f = 0  # current number of concept features

	if(useDedicatedConceptNames):
		# Add dummy feature for concept neuron (different per concept column)
		concept_features_list.append(variableConceptNeuronFeatureName)
		concept_features_dict[variableConceptNeuronFeatureName] = len(concept_features_dict)
		f += 1  # Will be updated dynamically based on c
	
	if useDedicatedFeatureLists:
		print("error: useDedicatedFeatureLists case not yet coded - need to set f and populate concept_features_list/concept_features_dict etc")
		exit()
		# f = max_num_non_nouns + 1  # Maximum number of non-nouns in an English dictionary, plus the concept neuron of each column
  
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
		feature_neurons_strength = torch.zeros(f, dtype=array_type)
		feature_neurons_permanence = torch.full((f,), z1, dtype=array_type)  # Initialize permanence to z1=3
		feature_neurons_activation = torch.zeros(f, dtype=array_type)  # Activation trace counters
		feature_neurons = torch.stack([feature_neurons_strength, feature_neurons_permanence, feature_neurons_activation])
		feature_neurons = feature_neurons.unsqueeze(1)	#add type dimension (action, condition, quality, modifier etc) #not currently used
		return feature_neurons

	@staticmethod
	def initialiseFeatureConnections(c, f):
		connection_strength = torch.zeros(f, c, f, dtype=array_type)
		connection_permanence = torch.full((f, c, f), z1, dtype=array_type)  # Initialize permanence to z1=3
		connection_activation = torch.zeros(f, c, f, dtype=array_type)  # Activation trace counters
		feature_connections = torch.stack([connection_strength, connection_permanence, connection_activation])
		feature_connections = feature_connections.unsqueeze(1)	#add type dimension (action, condition, quality, modifier etc) #not currently used
		return feature_connections
	
	def resize_concept_arrays(self, new_c):
		load_c = self.feature_connections.shape[3]
		if new_c > load_c:
			extra_cols = new_c - load_c
			# Expand along dimension 3
			self.feature_connections = torch.cat([self.feature_connections, torch.zeros(self.feature_connections.shape[0], self.feature_connections.shape[1], self.feature_connections.shape[2], extra_cols, self.feature_connections.shape[4], dtype=array_type)], dim=3)

	def expand_feature_arrays(self, new_f):
		load_f = self.feature_connections.shape[2]	# or self.feature_connections.shape[4]	   
		if new_f > load_f:
			extra_features = new_f - load_f
			
			# Expand along dimension 2
			self.feature_connections = torch.cat([self.feature_connections, torch.zeros(self.feature_connections.shape[0], self.feature_connections.shape[1], extra_features, self.feature_connections.shape[3], self.feature_connections.shape[4], dtype=array_type)], dim=2)

			# Also expand along dimension 4
			self.feature_connections = torch.cat([self.feature_connections, torch.zeros(self.feature_connections.shape[0], self.feature_connections.shape[1], self.feature_connections.shape[2], self.feature_connections.shape[3], extra_features, dtype=array_type)], dim=4)

			if lowMem:
				extra_feature_neurons = self.initialiseFeatureNeurons(extra_features)
				self.feature_neurons = torch.cat([self.feature_neurons, extra_feature_neurons], dim=2)

			for feature_index in range(load_f, new_f, 1):
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
		torch.save(self.feature_connections, os.path.join(observed_columns_dir, f"{self.concept_index}_feature_connections.pt"))
		if lowMem:
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

# Define the SequenceObservedColumns class
class SequenceObservedColumns:
	"""
	Contains sequence observed columns object arrays which stack a feature subset of the observed columns object arrays for the current sequence.
	"""
	def __init__(self, words, lemmas, observed_columns_dict, observed_columns_sequence_word_index_dict, train=True):
		#note cs may be slightly longer than number of unique columns in the sequence, if there are multiple instances of the same concept/noun lemma in the sequence
	
		self.observed_columns_dict = observed_columns_dict	# key: lemma, value: ObservedColumn
		self.observed_columns_sequence_word_index_dict = observed_columns_sequence_word_index_dict	# key: sequence word index, value: ObservedColumn

		if(sequenceObservedColumnsMatchSequenceWords):
			self.cs = len(observed_columns_sequence_word_index_dict)

			self.columns_index_sequence_word_index_dict = {}	#key: sequence word index, value: concept_index
			self.sequence_observed_columns_dict = {}	#key: sequence_concept_index, value: observed_column 
			concept_indices_in_observed_list = []	#value: concept index
			for sequence_concept_index, (sequence_word_index, observed_column) in enumerate(self.observed_columns_sequence_word_index_dict.items()):
				self.columns_index_sequence_word_index_dict[sequence_word_index] = observed_column.concept_index
				self.sequence_observed_columns_dict[sequence_concept_index] = observed_column
				concept_indices_in_observed_list.append(observed_column.concept_index)
			self.concept_indices_in_sequence_observed_tensor = torch.tensor(concept_indices_in_observed_list, dtype=torch.long)
		else:
			self.cs = len(observed_columns_dict) 

			self.columns_index_sequence_word_index_dict = {}	#key: sequence word index, value: concept_index
			for idx, (sequence_word_index, observed_column) in enumerate(observed_columns_sequence_word_index_dict.items()):
				self.columns_index_sequence_word_index_dict[sequence_word_index] = observed_column.concept_index

			# Map from concept names to indices in sequence arrays
			concept_indices_in_observed_list = []
			self.concept_name_to_index = {}	# key: lemma, value: sequence_concept_index
			self.index_to_concept_name = {}	# key: sequence_concept_index, value: lemma
			self.observed_columns_dict2 = {}	# key: sequence_concept_index, value: ObservedColumn
			for idx, (lemma, observed_column) in enumerate(observed_columns_dict.items()):
				concept_indices_in_observed_list.append(observed_column.concept_index)
				self.concept_name_to_index[lemma] = idx
				self.index_to_concept_name[idx] = lemma
				self.observed_columns_dict2[idx] = observed_column
			self.concept_indices_in_sequence_observed_tensor = torch.tensor(concept_indices_in_observed_list, dtype=torch.long)
				
		if(sequenceObservedColumnsMatchSequenceWords):
			self.feature_neuron_changes = [None]*self.cs
			self.feature_connection_changes = [None]*self.cs
			
		# Collect all feature words from observed columns
		self.words = words
		self.lemmas = lemmas
		#identify feature indices from complete ObservedColumns.featureNeurons or globalFeatureNeurons feature lists currently stored in SequenceObservedColumns.feature_neurons	#required for useInference
		observed_column = list(observed_columns_dict.values())[0]	#all features (including words) are identical per observed column
		self.feature_words, self.feature_indices_in_observed_tensor, self.f_idx_tensor = self.identifyObservedColumnFeatureWords(words, lemmas, observed_column)

		if(sequenceObservedColumnsMatchSequenceWords):
			self.fs = self.feature_indices_in_observed_tensor.shape[0]
			self.index_to_feature_word = {}
			for idx, feature_word in enumerate(self.feature_words):
				self.index_to_feature_word[idx] = feature_word
		else:
			self.fs = len(self.feature_words)
			self.feature_word_to_index = {}
			self.index_to_feature_word = {}
			for idx, feature_word in enumerate(self.feature_words):
				self.feature_word_to_index[feature_word] = idx
				self.index_to_feature_word[idx] = feature_word

		if(train):
			# Initialize arrays
			self.feature_neurons = self.initialiseFeatureNeuronsSequence(self.cs, self.fs)
			self.feature_connections = self.initialiseFeatureConnectionsSequence(self.cs, self.fs)

			# Populate arrays with data from observed_columns_dict
			if(sequenceObservedColumnsMatchSequenceWords):
				self.populate_arrays(words, lemmas, self.sequence_observed_columns_dict)
			else:
				self.populate_arrays(words, lemmas, self.observed_columns_dict2)
		else:
			self.cs2 = len(concept_columns_dict)
			self.fs2 = self.fs
			
			feature_connections_list = []
			for observed_column in observed_columns_sequence_word_index_dict.values():
				 feature_connections_list.append(observed_column.feature_connections)
			self.feature_connections = torch.stack(feature_connections_list, dim=2)
			
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
			feature_indices_in_observed_tensor = torch.tensor(feature_indices_in_observed, dtype=torch.long)
		else:
			if(sequenceObservedColumnsMatchSequenceWords):
				print("identifyObservedColumnFeatureWords+!sequenceObservedColumnsUseSequenceFeaturesOnly+sequenceObservedColumnsMatchSequenceWords requires coding:")
				exit()
			else:
				feature_words = observed_column.feature_word_to_index.keys()
				feature_indices_in_observed_tensor = torch.tensor(list(observed_column.feature_word_to_index.values()), dtype=torch.long)
		if(not sequenceObservedColumnsMatchSequenceWords):
			feature_words = self.removeDuplicates(feature_words)
		
		if(sequenceObservedColumnsMatchSequenceWords):
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
		feature_neurons_strength = torch.zeros(cs, fs, dtype=array_type)
		feature_neurons_permanence = torch.full((cs, fs), z1, dtype=array_type)
		feature_neurons_activation = torch.zeros(cs, fs, dtype=array_type)
		feature_neurons = torch.stack([feature_neurons_strength, feature_neurons_permanence, feature_neurons_activation])
		feature_neurons = feature_neurons.unsqueeze(1)	#add type dimension (action, condition, quality, modifier etc) #not currently used
		return feature_neurons

	@staticmethod
	def initialiseFeatureConnectionsSequence(cs, fs):
		connection_strength = torch.zeros(cs, fs, cs, fs, dtype=array_type)
		connection_permanence = torch.full((cs, fs, cs, fs), z1, dtype=array_type)
		connection_activation = torch.zeros(cs, fs, cs, fs, dtype=array_type)
		feature_connections = torch.stack([connection_strength, connection_permanence, connection_activation])
		feature_connections = feature_connections.unsqueeze(1)	#add type dimension (action, condition, quality, modifier etc) #not currently used
		return feature_connections
	
	def populate_arrays(self, words, lemmas, sequence_observed_columns_dict):
		# Collect indices and data for feature neurons
		c_idx_list = []
		f_idx_list = []
		feature_list = []
		
		for c_idx, observed_column in sequence_observed_columns_dict.items():
			feature_indices_in_observed, f_idx_tensor = self.getObservedColumnFeatureIndices()

			num_features = len(f_idx_tensor)

			c_idx_list.append(torch.full((num_features,), c_idx, dtype=torch.long))
			f_idx_list.append(f_idx_tensor)

			if lowMem:
				feature_list.append(observed_column.feature_neurons[:, :, feature_indices_in_observed])
			else:
				feature_list.append(global_feature_neurons[:, :, observed_column.concept_index, feature_indices_in_observed])
			
		# Concatenate lists to tensors
		c_idx_tensor = torch.cat(c_idx_list)
		f_idx_tensor = torch.cat(f_idx_list)
		feature_tensor = torch.cat(feature_list, dim=2)
		
		# Use advanced indexing to assign values
		self.feature_neurons[:, :, c_idx_tensor, f_idx_tensor] = feature_tensor

		# Now handle connections
		connection_indices = []
		connection_values = []

		for c_idx, observed_column in sequence_observed_columns_dict.items():
			feature_indices_in_observed, f_idx_tensor = self.getObservedColumnFeatureIndices()

			for other_c_idx, other_observed_column in sequence_observed_columns_dict.items():
				other_feature_indices_in_observed, other_f_idx_tensor = self.getObservedColumnFeatureIndices()
				other_concept_index = other_observed_column.concept_index

				# Create meshgrid of indices
				feature_idx_obs_mesh, other_feature_idx_obs_mesh = torch.meshgrid(feature_indices_in_observed, other_feature_indices_in_observed, indexing='ij')
				f_idx_mesh, other_f_idx_mesh = torch.meshgrid(f_idx_tensor, other_f_idx_tensor, indexing='ij')

				# Flatten the meshgrid indices
				feature_idx_obs_flat = feature_idx_obs_mesh.reshape(-1)
				other_feature_idx_obs_flat = other_feature_idx_obs_mesh.reshape(-1)
				f_idx_flat = f_idx_mesh.reshape(-1)
				other_f_idx_flat = other_f_idx_mesh.reshape(-1)

				# Create tensors for concept indices
				c_idx_flat = torch.full_like(f_idx_flat, c_idx, dtype=torch.long)
				other_c_idx_flat = torch.full_like(other_f_idx_flat, other_c_idx, dtype=torch.long)

				# Get the corresponding values from observed_column arrays
				values = observed_column.feature_connections[:, :, feature_idx_obs_flat, other_concept_index, other_feature_idx_obs_flat]

				# Append to lists
				connection_indices.append((c_idx_flat, f_idx_flat, other_c_idx_flat, other_f_idx_flat))
				connection_values.append(values)

		# Concatenate tensors
		c_idx_conn_tensor = torch.cat([idx[0] for idx in connection_indices])
		f_idx_conn_tensor = torch.cat([idx[1] for idx in connection_indices])
		other_c_idx_conn_tensor = torch.cat([idx[2] for idx in connection_indices])
		other_f_idx_conn_tensor = torch.cat([idx[3] for idx in connection_indices])

		connection_tensor = torch.cat(connection_values, dim=2)

		# Use advanced indexing to assign connection values
		self.feature_connections[:, :, c_idx_conn_tensor, f_idx_conn_tensor, other_c_idx_conn_tensor, other_f_idx_conn_tensor] = connection_tensor

	def update_observed_columns_wrapper(self):
		if(sequenceObservedColumnsMatchSequenceWords):
			#for multiple instances of concept in sequence, need to take the sum of the changes between the existing and modified arrays for each instance of a same concept in the sequence
			self.update_observed_columns(self.sequence_observed_columns_dict, mode="recordChangesToConceptNeuronSequenceInstances")
			self.update_observed_columns(self.sequence_observed_columns_dict, mode="applySummedChangesToConceptNeuronSequenceInstances")
		else:
			self.update_observed_columns(self.observed_columns_dict2, mode="default")
			
	def update_observed_columns(self, sequence_observed_columns_dict, mode):
		# Update observed columns with data from sequence arrays
		
		#print("\n\n\n update_observed_columns:")
		
		self.index_to_feature_word = {}
		for c_idx, observed_column in sequence_observed_columns_dict.items():
			feature_indices_in_observed, f_idx_tensor = self.getObservedColumnFeatureIndices()

			# Use advanced indexing to get values from self.feature_neurons_*
			values = self.feature_neurons[:, :, c_idx, f_idx_tensor]

			# Assign values to observed_column's feature_neurons arrays
			if(mode=="default"):
				if lowMem:
					observed_column.feature_neurons[:, :, feature_indices_in_observed] = values
				else:
					global_feature_neurons[:, :, observed_column.concept_index, feature_indices_in_observed] = values
			elif(mode=="recordChangesToConceptNeuronSequenceInstances"):
				if lowMem:
					self.feature_neuron_changes[c_idx] = values - observed_column.feature_neurons[:, :, feature_indices_in_observed]
				else:
					self.feature_neuron_changes[c_idx] = values - global_feature_neurons[:, :, observed_column.concept_index, feature_indices_in_observed]
				#print("self.feature_neuron_changes[c_idx] = ", self.feature_neuron_changes[c_idx])
			elif(mode=="applySummedChangesToConceptNeuronSequenceInstances"):
				if lowMem:
					batch_size, channels, feature_dim = observed_column.feature_neurons.shape
					expanded_feature_indices = feature_indices_in_observed.unsqueeze(0).expand(batch_size, -1)  # Shape: (batch_size, num_indices)
					observed_column.feature_neurons.scatter_add_(2, expanded_feature_indices.unsqueeze(1), self.feature_neuron_changes[c_idx])
					#observed_column.feature_neurons[:, :, feature_indices_in_observed] += observed_column.feature_neuron_changes_summed
				else:
					batch_size, channels, num_concepts, feature_dim = global_feature_neurons.shape
					expanded_feature_indices = feature_indices_in_observed.unsqueeze(0).expand(batch_size, -1)
					global_feature_neurons[:, :, observed_column.concept_index].scatter_add_(2, expanded_feature_indices.unsqueeze(1), self.feature_neuron_changes[c_idx])	
					#global_feature_neurons[:, :, observed_column.concept_index, feature_indices_in_observed] += observed_column.feature_neuron_changes_summed

			# Now handle connections
			conn_feature_indices_obs = []
			conn_other_concept_indices = []
			conn_other_feature_indices_obs = []
			conn_values = []
			
			for other_c_idx, other_observed_column in sequence_observed_columns_dict.items():
				other_feature_indices_in_observed, other_f_idx_tensor = self.getObservedColumnFeatureIndices()
				other_concept_index = other_observed_column.concept_index

				# Create meshgrid of indices
				feature_idx_obs_mesh, other_feature_idx_obs_mesh = torch.meshgrid(feature_indices_in_observed, other_feature_indices_in_observed, indexing='ij')
				f_idx_mesh, other_f_idx_mesh = torch.meshgrid(f_idx_tensor, other_f_idx_tensor, indexing='ij')

				# Flatten the meshgrid indices
				feature_idx_obs_flat = feature_idx_obs_mesh.reshape(-1)
				other_feature_idx_obs_flat = other_feature_idx_obs_mesh.reshape(-1)
				f_idx_flat = f_idx_mesh.reshape(-1)
				other_f_idx_flat = other_f_idx_mesh.reshape(-1)

				# Get the corresponding values from self.connection_* arrays
				values = self.feature_connections[:, :, c_idx, f_idx_flat, other_c_idx, other_f_idx_flat]

				# Append data to lists
				conn_feature_indices_obs.append(feature_idx_obs_flat)
				conn_other_concept_indices.append(torch.full_like(feature_idx_obs_flat, other_concept_index, dtype=torch.long))
				conn_other_feature_indices_obs.append(other_feature_idx_obs_flat)
				conn_values.append(values)

			# Concatenate lists to form tensors
			conn_feature_indices_obs = torch.cat(conn_feature_indices_obs)
			conn_other_concept_indices = torch.cat(conn_other_concept_indices)
			conn_other_feature_indices_obs = torch.cat(conn_other_feature_indices_obs)
			conn_values = torch.cat(conn_values, dim=2)

			# Assign values to observed_column's feature connection arrays using advanced indexing
			if(mode=="default"):
				observed_column.feature_connections[:, :, conn_feature_indices_obs, conn_other_concept_indices, conn_other_feature_indices_obs] = conn_values
			elif(mode=="recordChangesToConceptNeuronSequenceInstances"):
				self.feature_connection_changes[c_idx] = conn_values - observed_column.feature_connections[:, :, conn_feature_indices_obs, conn_other_concept_indices, conn_other_feature_indices_obs]
			elif(mode=="applySummedChangesToConceptNeuronSequenceInstances"):
				batch_size, channels, dim1, dim2, dim3 = observed_column.feature_connections.shape
				flat_feature_connections = observed_column.feature_connections.view(batch_size, channels, -1)
				flattened_indices = (conn_feature_indices_obs * (dim2 * dim3) + conn_other_concept_indices * dim3 + conn_other_feature_indices_obs)
				expanded_flattened_indices = flattened_indices.unsqueeze(0).expand(batch_size, -1)
				flat_feature_connections.scatter_add_(2, expanded_flattened_indices.unsqueeze(1), self.feature_connection_changes[c_idx])
				observed_column.feature_connections = flat_feature_connections.view(batch_size, channels, dim1, dim2, dim3)
				#observed_column.feature_connections[:, :, conn_feature_indices_obs, conn_other_concept_indices, conn_other_feature_indices_obs] += observed_column.feature_connection_changes_summed

# Initialize global feature neuron arrays if lowMem is disabled
if not lowMem:
	if os.path.exists(global_feature_neurons_file):
		global_feature_neurons = torch.load(global_feature_neurons_file)
	else:
		global_feature_neurons = SequenceObservedColumns.initialiseFeatureNeuronsSequence(c, f)

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
	print(f"Processing sentence: {sentenceIndex} {doc.text}")

	# Refresh the observed columns dictionary for each new sequence
	observed_columns_dict = {}  # key: lemma, value: ObservedColumn
	observed_columns_sequence_word_index_dict = {}  # key: sequence word index, value: ObservedColumn
	
	if(lastSentenceInPrompt):
		doc_seed = doc[0:num_seed_tokens]	#prompt
		doc_predict = doc[num_seed_tokens:]
		doc = doc_seed

	# First pass: Extract words, lemmas, POS tags, and update concept_columns_dict and c
	concepts_found, words, lemmas, pos_tags = first_pass(doc)

	if(concepts_found):
		# When usePOS is enabled, detect all possible new features in the sequence
		if not (useDedicatedFeatureLists):
			detect_new_features(words, lemmas, pos_tags)

		# Second pass: Create observed_columns_dict
		observed_columns_dict, observed_columns_sequence_word_index_dict = second_pass(lemmas, pos_tags)

		# Create the sequence observed columns object
		sequence_observed_columns = SequenceObservedColumns(words, lemmas, observed_columns_dict, observed_columns_sequence_word_index_dict, train=(not lastSentenceInPrompt))

		if(lastSentenceInPrompt):
			# Process each concept word in the sequence (predict)
			process_concept_words_inference(doc_seed, words, lemmas, pos_tags, doc_predict, sequence_observed_columns, num_seed_tokens, num_prediction_tokens)
		else:
			# Process each concept word in the sequence (train)
			process_concept_words(doc, words, lemmas, pos_tags, sequence_observed_columns, train=True)

			# Update observed columns from sequence observed columns
			sequence_observed_columns.update_observed_columns_wrapper()

			# Visualize the complete graph every time a new sentence is parsed by the application.
			visualize_graph(sequence_observed_columns)

			# Save observed columns to disk
			if(useSaveData):
				save_data(observed_columns_dict, concept_features_dict)

			if(useActivationDecrement):
				#decrement activation after each train interval; not currently used
				global global_feature_neurons
				global_feature_neurons[array_index_properties_activation, array_index_type_all] -= activationDecrementPerPredictedSentence
				global_feature_neurons[array_index_properties_activation, array_index_type_all] = torch.clamp(global_feature_neurons[array_index_properties_activation, array_index_type_all], min=0)
	
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

		if usePOS and pos in noun_pos_tags:
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
			global c, global_feature_neurons
			if global_feature_neurons.shape[2] < c:
				extra_rows = c - global_feature_neurons.shape[2]
				extra_global_feature_neurons = SequenceObservedColumns.initialiseFeatureNeuronsSequence(extra_rows, f)
				global_feature_neurons = torch.cat([global_feature_neurons, extra_global_feature_neurons], dim=2)
				
	return concepts_found, words, lemmas, pos_tags

def addConceptToConceptColumnsDict(lemma, concepts_found, new_concepts_added):
	global c, concept_columns_dict, concept_columns_list
	concepts_found = True
	if lemma not in concept_columns_dict:
		# Add to concept columns dictionary
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
	global f, lowMem, global_feature_neurons

	num_new_features = 0
	for j, (word_j, pos_j) in enumerate(zip(words, pos_tags)):
		if(process_feature_detection(j, word_j, pos_tags)):
			num_new_features += 1

	# After processing all features, update f
	f += num_new_features

	# Now, expand arrays accordingly
	if not lowMem:
		if f > global_feature_neurons.shape[3]:
			extra_cols = f - global_feature_neurons.shape[3]
			extra_global_feature_neurons = SequenceObservedColumns.initialiseFeatureNeuronsSequence(global_feature_neurons.shape[2], extra_cols)
			global_feature_neurons = torch.cat([global_feature_neurons, extra_global_feature_neurons], dim=3)

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
	
def process_concept_words_inference(doc_seed, words_seed, lemmas_seed, pos_tags_seed, doc_predict, sequence_observed_columns, num_seed_tokens, num_prediction_tokens):

	sequenceWordIndex = 0
	conceptColumnNewlyActivated = False
	
	#seed next tokens;
	conceptColumnNewlyActivated = process_column_inference_seed(doc_seed, words_seed, lemmas_seed, pos_tags_seed, sequence_observed_columns, conceptColumnNewlyActivated)

	#identify first activated column(s) in prediction phase:
	concept_columns_indices = sequence_observed_columns.concept_indices_in_sequence_observed_tensor[-1]
	concept_columns_indices_list = [concept_columns_indices.item()]
	#the first number of prediction candidates from seed will always match the first list of prediction candidates (1) from the seed sequence, even if kcMax > 1
	sequence_observed_columns = None	#will be reset every token prediction

	#predict next tokens;
	for wordPredictionIndex in range(num_prediction_tokens):
		sequenceWordIndex = num_seed_tokens + wordPredictionIndex
		concept_columns_indices_list, featurePredictionTargetMatch, conceptColumnNewlyActivated = process_column_inference_prediction(wordPredictionIndex, sequenceWordIndex, concept_columns_indices_list, doc_predict, conceptColumnNewlyActivated)
		
def process_column_inference_seed(doc_seed, words_seed, lemmas_seed, pos_tags_seed, sequence_observed_columns, conceptColumnNewlyActivated):
	global global_feature_neurons
	
	concept_indices, start_indices, end_indices = process_concept_words(doc_seed, words_seed, lemmas_seed, pos_tags_seed, sequence_observed_columns, train=False)

	fs = len(concept_features_list)
	sequence_observed_columns.feature_connections_all = sequence_observed_columns.feature_connections	#for every concept column in seed sequence
	
	conceptColumnNewlyActivated = True
	concept_indices_list = concept_indices.tolist()
	for sequence_concept_index, sequence_concept_word_index in enumerate(concept_indices_list):
		
		feature_neurons_active = torch.zeros((1, fs), dtype=torch.long)
		feature_neurons_active[0, start_indices[sequence_concept_index]:end_indices[sequence_concept_index]] = 1
		sequence_observed_columns.feature_connections = sequence_observed_columns.feature_connections_all[:, :, sequence_concept_index].unsqueeze(2)	#readd concept column dimension cs2
		
		#process features (activate global target neurons);
		global_feature_neurons[array_index_properties_activation, array_index_type_all, sequence_concept_index, :] = feature_neurons_active[0]
		#print("global_feature_neurons active = ", global_feature_neurons[array_index_properties_activation, array_index_type_all])
		#print("feature_neurons_active = ", feature_neurons_active)
		process_features_active_predict(sequence_observed_columns, feature_neurons_active)
		
		if(deactivateNeuronsUponPrediction):
			feature_neurons_active[0, start_indices[sequence_concept_index]:end_indices[sequence_concept_index]] = 0

				
def process_column_inference_prediction(wordPredictionIndex, sequenceWordIndex, concept_columns_indices_list, doc_predict, conceptColumnNewlyActivated):
	global global_feature_neurons
	
	print(f"process_column_inference_prediction: {wordPredictionIndex}; concept_columns_indices_list = ", concept_columns_indices_list)

	# Refresh the observed columns dictionary for each new sequence
	observed_columns_dict = {}  # key: lemma, value: ObservedColumn
	observed_columns_sequence_candidate_index_dict = {}  # key: sequence candidate index, value: ObservedColumn	#used to populate sequence feature connection arrays based on observed columns (i does not correspond to sequence word index as assumed by observed_columns_sequence_word_index_dict)
	
	#populate sequence observed columns;
	words = []
	lemmas = []
	for i, concept_index in enumerate(concept_columns_indices_list):
		lemma = concept_columns_list[concept_index]
		word = lemma	#same for concepts (not used)
		lemmas.append(lemma)
		words.append(word)
		# Load observed column from disk or create new one
		observed_column = load_or_create_observed_column(concept_index, lemma, sequenceWordIndex)
		observed_columns_dict[lemma] = observed_column
		observed_columns_sequence_candidate_index_dict[i] = observed_column
	sequence_observed_columns = SequenceObservedColumns(words, lemmas, observed_columns_dict, observed_columns_sequence_candidate_index_dict, train=False)
	
	if(conceptColumnNewlyActivated):
		#process features (activate global target neurons);
		feature_neurons_active = global_feature_neurons[array_index_properties_activation, array_index_type_all, sequence_observed_columns.concept_indices_in_sequence_observed_tensor, :]
		#print("global_feature_neurons active = ", global_feature_neurons[array_index_properties_activation, array_index_type_all])
		#print("feature_neurons_active = ", feature_neurons_active)
		process_features_active_predict(sequence_observed_columns, feature_neurons_active)

	#topk column selection;
	concept_columns = global_feature_neurons[array_index_properties_activation, array_index_type_all, :, :]
	concept_columns_activation = torch.sum(concept_columns, dim=1)	#sum across all feature activations in columns
	if(kcDynamic):
		concept_columns_activation = concept_columns_activation[concept_columns_activation > kcActivationThreshold]	#select kcMax columns above threshold
	concept_columns_activation_topk_concepts = torch.topk(concept_columns_activation, kcMax)
	kc = len(concept_columns_activation_topk_concepts.indices)
	if(kcDynamic and kc < 1):
		print("process_column_prediction kcDynamic error: kc < 1; cannot continue to predict columns; consider disabling kcDynamic for debug")
		exit()

	#top feature selection;
	topk_concept_columns_activation = global_feature_neurons[array_index_properties_activation, array_index_type_all, concept_columns_activation_topk_concepts.indices, :]
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
			
	concept_columns_indices_next_list = concept_columns_activation_topk_concepts.indices.tolist()
	
	#decrement activations;
	if(useActivationDecrement):
		#decrement activation after each prediction interval
		global_feature_neurons[array_index_properties_activation, array_index_type_all] -= activationDecrementPerPredictedColumn
		global_feature_neurons[array_index_properties_activation, array_index_type_all] = torch.clamp(global_feature_neurons[array_index_properties_activation, array_index_type_all], min=0)
	if(deactivateNeuronsUponPrediction):
		global_feature_neurons[array_index_properties_activation, array_index_type_all, concept_columns_activation_topk_concepts.indices, topk_concept_columns_activation_topk_features.indices] = 0
		
	#assert global_feature_neurons != global_feature_neuronsOrig
	visualize_graph(sequence_observed_columns)
	
	if(kcMax == 1):
		if(concept_columns_indices_next_list[0] == concept_columns_indices_list[0]):
			conceptColumnNewlyActivated = False
		else:
			conceptColumnNewlyActivated = True
	else:
		print("process_column_inference_prediction implementation limitation: conceptColumnNewlyActivated detection currently requires kcMax == 1")
		exit()
	
	return concept_columns_indices_next_list, featurePredictionTargetMatch, conceptColumnNewlyActivated
	 
#first dim cs1 restricted to a single token (or candiate set of tokens).
def process_features_active_predict(sequence_observed_columns, feature_neurons_active):
 	
	feature_neurons_active_expanded = feature_neurons_active.unsqueeze(2).unsqueeze(3)
	feature_connections_active = feature_neurons_active_expanded.expand(-1, -1, sequence_observed_columns.cs2, sequence_observed_columns.fs2)
	
	#not required as predicted nodes are deactivated upon firing; ensure identical feature nodes are not connected together
	
	#target neuron activation dependence on connection strength;
	feature_connections_activation_update = feature_connections_active * sequence_observed_columns.feature_connections[array_index_properties_strength, array_index_type_all]
	#update the activations of the target nodes;
	feature_neurons_target_activation = torch.sum(feature_connections_activation_update, dim=(0, 1))

	global global_feature_neurons
	global_feature_neurons[array_index_properties_activation, array_index_type_all] += feature_neurons_target_activation*j1
		
				
def process_concept_words(doc, words, lemmas, pos_tags, sequence_observed_columns, train=True):
	"""
	For every concept word (lemma) in the sequence, identify every feature neuron in that column that occurs q words before or after the concept word in the sequence, including the concept neuron. This function has been parallelized using PyTorch array operations.
	"""
	global c, f, lowMem, global_feature_neurons

	if not usePOS:
		q = 5  # Fixed window size when not using POS tags

	# Identify all concept word indices
	#print("\n\nsequence_observed_columns.columns_index_sequence_word_index_dict = ", sequence_observed_columns.columns_index_sequence_word_index_dict)
	concept_mask = torch.tensor([i in sequence_observed_columns.columns_index_sequence_word_index_dict for i in range(len(lemmas))], dtype=torch.bool)
	concept_indices = torch.nonzero(concept_mask).squeeze(1)
	#concept_indices may be slightly longer than number of unique columns in sequence, if there are multiple instances of the same concept/noun lemma in the sequence
	
	if concept_indices.numel() == 0:
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
	if usePOS:
		start_indices = (concept_indices - dist_to_prev_concept + 1).clamp(min=0)
		end_indices = (concept_indices + dist_to_next_concept).clamp(max=len(doc))
	else:
		start_indices = (concept_indices - q).clamp(min=0)
		end_indices = (concept_indices + q + 1).clamp(max=len(doc))

	if(train):
		process_features(start_indices, end_indices, doc, words, lemmas, pos_tags, sequence_observed_columns, concept_indices)
	
	return concept_indices, start_indices, end_indices
	
def process_features(start_indices, end_indices, doc, words, lemmas, pos_tags, sequence_observed_columns, concept_indices):

	cs = sequence_observed_columns.cs #will be different than len(concept_indices) if there are multiple instances of a concept in a sequence
	fs = sequence_observed_columns.fs  #len(doc)
	feature_neurons_active = torch.zeros((cs, fs), dtype=array_type)
	feature_neurons_word_order = torch.arange(fs).unsqueeze(0).repeat(cs, 1)
	torch.zeros((cs, fs), dtype=torch.long)
	columns_word_order = torch.zeros((cs), dtype=torch.long)
	
	concept_indices_list = concept_indices.tolist()
	#convert start/end indices to active features arrays
	if(sequenceObservedColumnsMatchSequenceWords):
		sequence_concept_index_mask = torch.ones((cs, fs), dtype=array_type)	#ensure to ignore concept feature neurons from other columns
		for sequence_concept_index, sequence_concept_word_index in enumerate(concept_indices_list):
			feature_neurons_active[sequence_concept_index, start_indices[sequence_concept_index]:end_indices[sequence_concept_index]] = 1
			columns_word_order[sequence_concept_index] = sequence_concept_index	#CHECKTHIS; or sequence_concept_word_index
			sequence_concept_index_mask[:, sequence_concept_word_index] = 0	#ignore concept feature neurons from other columns
			sequence_concept_index_mask[sequence_concept_index, sequence_concept_word_index] = 1
	else:
		sequence_concept_index_mask = None
		for i in range(concept_indices.shape[0]):
			concept_lemma = lemmas[concept_indices[i]]
			sequence_concept_index = sequence_observed_columns.concept_name_to_index[concept_lemma]
			for j in range(start_indices[i], end_indices[i]):	#sequence word index
				feature_word = words[j].lower()
				feature_lemma = lemmas[j]
				if(j in sequence_observed_columns.columns_index_sequence_word_index_dict):	#test is required for concept neurons
					sequence_concept_word_index = j
					columns_word_order[sequence_concept_index] = sequence_concept_word_index	#CHECKTHIS; or sequence_concept_index
					if(useDedicatedConceptNames2):
						sequence_feature_index = sequence_observed_columns.feature_word_to_index[variableConceptNeuronFeatureName]
					else:
						sequence_feature_index = sequence_observed_columns.feature_word_to_index[feature_lemma]
					feature_neurons_active[sequence_concept_index, sequence_feature_index] = 1
					feature_neurons_word_order[sequence_concept_index, sequence_feature_index] = j
				elif(feature_word in sequence_observed_columns.feature_word_to_index):
					sequence_feature_index = sequence_observed_columns.feature_word_to_index[feature_word]
					feature_neurons_active[sequence_concept_index, sequence_feature_index] = 1
					feature_neurons_word_order[sequence_concept_index, sequence_feature_index] = j
	
	process_features_active_train(sequence_observed_columns, feature_neurons_active, cs, fs, sequence_concept_index_mask, columns_word_order, feature_neurons_word_order)

#first dim cs1 pertains to every concept node in sequence
def process_features_active_train(sequence_observed_columns, feature_neurons_active, cs, fs, sequence_concept_index_mask, columns_word_order, feature_neurons_word_order):
	feature_neurons_inactive = 1 - feature_neurons_active
 
	# Update feature neurons in sequence_observed_columns
	sequence_observed_columns.feature_neurons[array_index_properties_strength, array_index_type_all, :, :] += feature_neurons_active
	sequence_observed_columns.feature_neurons[array_index_properties_permanence, array_index_type_all, :, :] += feature_neurons_active*z1	#orig = feature_neurons_active*(sequence_observed_columns.feature_neurons[array_index_properties_permanence, array_index_type_all] ** 2) + feature_neurons_inactive*sequence_observed_columns.feature_neurons[array_index_properties_permanence, array_index_type_all]
	#sequence_observed_columns.feature_neurons[array_index_properties_activation, array_index_type_all, :, :] += feature_neurons_active*j1	#update the activations of the target not source nodes

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
		columns_word_order_mask = columns_word_order_expanded_2 >= columns_word_order_expanded_1
		feature_connections_active = feature_connections_active * columns_word_order_mask
	
	#ensure identical feature nodes are not connected together;
	cs_indices_1 = torch.arange(cs).view(cs, 1, 1, 1).expand(cs, fs, cs, fs)  # First cs dimension
	cs_indices_2 = torch.arange(cs).view(1, 1, cs, 1).expand(cs, fs, cs, fs)  # Second cs dimension
	fs_indices_1 = torch.arange(fs).view(1, fs, 1, 1).expand(cs, fs, cs, fs)  # First fs dimension
	fs_indices_2 = torch.arange(fs).view(1, 1, 1, fs).expand(cs, fs, cs, fs)  # Second fs dimension
	identity_mask = (cs_indices_1 != cs_indices_2) | (fs_indices_1 != fs_indices_2)
	feature_connections_active = feature_connections_active * identity_mask
	
	feature_connections_inactive = 1 - feature_connections_active
	
	#prefer closer than further target neurons when strengthening connections or activating target neurons in sentence;
	feature_neurons_word_order_expanded_1 = feature_neurons_word_order.view(cs, fs, 1, 1).expand(cs, fs, cs, fs)  # For the first node
	feature_connections_strength_update = feature_connections_active*feature_neurons_word_order_expanded_1	#orig: feature_connections_active

	sequence_observed_columns.feature_connections[array_index_properties_strength, array_index_type_all, :, :, :, :] += feature_connections_strength_update
	sequence_observed_columns.feature_connections[array_index_properties_permanence, array_index_type_all, :, :, :, :] += feature_connections_active*z1	#orig = feature_connections_active*(sequence_observed_columns.feature_connections[array_index_properties_permanence, array_index_type_all] ** 2) + feature_connections_inactive*sequence_observed_columns.feature_connections[array_index_properties_permanence, array_index_type_all]
	#sequence_observed_columns.feature_connections[array_index_properties_activation, array_index_type_all, :, :, :, :] += feature_connections_active*j1	#connection activations are not currently used
	
	#target neuron activation dependence on connection strength;
	feature_connections_activation_update = feature_connections_active * sequence_observed_columns.feature_connections[array_index_properties_strength, array_index_type_all]
	#update the activations of the target nodes;
	feature_neurons_target_activation = torch.sum(feature_connections_activation_update, dim=(0, 1))
	sequence_observed_columns.feature_neurons[array_index_properties_activation, array_index_type_all, :, :] += feature_neurons_target_activation*j1
		#will only activate target neurons in sequence_observed_columns (not suitable for inference seed/prediction phase)
		
	#decrease permanence;
	decrease_permanence_active(sequence_observed_columns, feature_neurons_active, feature_neurons_inactive, sequence_concept_index_mask)
		
def decrease_permanence_active(sequence_observed_columns, feature_neurons_active, feature_neurons_inactive, sequence_concept_index_mask):

	if(sequenceObservedColumnsMatchSequenceWords):
		feature_neurons_inactive = feature_neurons_inactive*sequence_concept_index_mask	#when decreasing a value based on inactivation, ignore duplicate feature column neurons in the sequence
	
	cs = sequence_observed_columns.cs
	fs = sequence_observed_columns.fs 
	
	# Decrease permanence for feature neurons not activated
	sequence_observed_columns.feature_neurons[array_index_properties_permanence, array_index_type_all, :, :] -= feature_neurons_inactive*z2
	sequence_observed_columns.feature_neurons[array_index_properties_permanence, array_index_type_all] = torch.clamp(sequence_observed_columns.feature_neurons[array_index_properties_permanence, array_index_type_all], min=0)

	feature_neurons_all = torch.ones((cs, fs), dtype=array_type)
	feature_neurons_all_1d = feature_neurons_all.view(cs*fs)
	feature_neurons_active_1d = feature_neurons_active.view(cs*fs)
	feature_neurons_inactive_1d = feature_neurons_inactive.view(cs*fs)
	 
	# Decrease permanence of connections from inactive feature neurons in column
	feature_connections_decrease1 = torch.matmul(feature_neurons_inactive_1d.unsqueeze(1), feature_neurons_all_1d.unsqueeze(0)).view(cs, fs, cs, fs)
	sequence_observed_columns.feature_connections[array_index_properties_permanence, array_index_type_all, :, :, :, :] -= feature_connections_decrease1
	sequence_observed_columns.feature_connections[array_index_properties_permanence, array_index_type_all] = torch.clamp(sequence_observed_columns.feature_connections[array_index_properties_permanence, array_index_type_all], min=0)
	
	# Decrease permanence of inactive connections for activated features in column 
	feature_connections_decrease2 = torch.matmul(feature_neurons_active_1d.unsqueeze(1), feature_neurons_inactive_1d.unsqueeze(0)).view(cs, fs, cs, fs)
	sequence_observed_columns.feature_connections[array_index_properties_permanence, array_index_type_all, :, :, :, :] -= feature_connections_decrease2
	sequence_observed_columns.feature_connections[array_index_properties_permanence, array_index_type_all] = torch.clamp(sequence_observed_columns.feature_connections[array_index_properties_permanence, array_index_type_all], min=0)
 
	#current limitation; will not deactivate neurons or remove their strength if their permanence goes to zero
	
def visualize_graph(sequence_observed_columns):
	G.clear()

	# Draw concept columns
	pos_dict = {}
	x_offset = 0
	for lemma, observed_column in sequence_observed_columns.observed_columns_dict.items():
		concept_index = observed_column.concept_index
		
		if(drawSequenceObservedColumns):
			feature_word_to_index = sequence_observed_columns.feature_word_to_index
			y_offset = 1 + 1	#reserve space at bottom of column for feature concept neuron (as it will not appear first in sequence_observed_columns.feature_word_to_index, only observed_column.feature_word_to_index)
		else:
			feature_word_to_index = observed_column.feature_word_to_index
			y_offset = 1

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
					neuron_color = 'cyan'
					neuron_name = feature_word
			else:
				neuron_color = 'cyan'
				neuron_name = feature_word

			featureActive = False
			if(drawSequenceObservedColumns):
				c_idx = sequence_observed_columns.concept_name_to_index[lemma]
				if(feature_word in sequence_observed_columns.feature_word_to_index):
					f_idx = sequence_observed_columns.feature_word_to_index[feature_word]
					if(sequence_observed_columns.feature_neurons[array_index_properties_strength, array_index_type_all, c_idx, f_idx] > 0 and sequence_observed_columns.feature_neurons[array_index_properties_permanence, array_index_type_all, c_idx, f_idx] > 0):
						featureActive = True
			else:
				c_idx = concept_columns_dict[lemma]
				f_idx = feature_index_in_observed_column
				if lowMem:
					#print("observed_column.feature_neurons[array_index_properties_strength, array_index_type_all, feature_index_in_observed_column] = ", observed_column.feature_neurons[array_index_properties_strength, array_index_type_all, feature_index_in_observed_column])
					#print("observed_column.feature_neurons[array_index_properties_permanence, array_index_type_all, feature_index_in_observed_column] = ", observed_column.feature_neurons[array_index_properties_permanence, array_index_type_all, feature_index_in_observed_column])
					if(observed_column.feature_neurons[array_index_properties_strength, array_index_type_all, feature_index_in_observed_column] > 0 and observed_column.feature_neurons[array_index_properties_permanence, array_index_type_all, feature_index_in_observed_column] > 0):
						featureActive = True
				else:
					if(global_feature_neurons[array_index_properties_strength, array_index_type_all, concept_index, feature_index_in_observed_column] > 0 and global_feature_neurons[array_index_properties_permanence, array_index_type_all, concept_index, feature_index_in_observed_column] > 0):
						featureActive = True
			if(featureActive):	
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
	
		concept_index = observed_column.concept_index
		if(drawSequenceObservedColumns):
			feature_word_to_index = sequence_observed_columns.feature_word_to_index
			other_feature_word_to_index = sequence_observed_columns.feature_word_to_index
			c_idx = sequence_observed_columns.concept_name_to_index[lemma]
		else:
			feature_word_to_index = observed_column.feature_word_to_index
			other_feature_word_to_index = observed_column.feature_word_to_index
			c_idx = concept_columns_dict[lemma]

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
							featureActive = False
							if(drawSequenceObservedColumns):
								if(sequence_observed_columns.feature_connections[array_index_properties_strength, array_index_type_all, c_idx, f_idx, c_idx, other_f_idx] > 0 and sequence_observed_columns.feature_connections[array_index_properties_permanence, array_index_type_all, c_idx, f_idx, c_idx, other_f_idx] > 0):
									featureActive = True
							else:
								if(observed_column.feature_connections[array_index_properties_strength, array_index_type_all, f_idx, c_idx, other_f_idx] > 0 and observed_column.feature_connections[array_index_properties_permanence, array_index_type_all, f_idx, c_idx, other_f_idx] > 0):
									featureActive = True
							if(featureActive):
								G.add_edge(source_node, target_node, color='yellow')
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
							featureActive = False
							if(drawSequenceObservedColumns):
								other_c_idx = sequence_observed_columns.concept_name_to_index[other_lemma]
								if other_c_idx != c_idx:
									if(sequence_observed_columns.feature_connections[array_index_properties_strength, array_index_type_all, c_idx, f_idx, other_c_idx, other_f_idx] > 0 and sequence_observed_columns.feature_connections[array_index_properties_permanence, array_index_type_all, c_idx, f_idx, other_c_idx, other_f_idx] > 0):
										featureActive = True
							else:
								other_c_idx = concept_columns_dict[other_lemma]
								if lemma != other_lemma:	#if observed_column != other_observed_column:
									if(observed_column.feature_connections[array_index_properties_strength, array_index_type_all, f_idx, other_c_idx, other_f_idx] > 0 and observed_column.feature_connections[array_index_properties_permanence, array_index_type_all, f_idx, other_c_idx, other_f_idx] > 0):
										featureActive = True
							if(featureActive):
								G.add_edge(source_node, target_node, color='orange')
								
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
