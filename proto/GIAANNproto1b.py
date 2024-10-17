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
torch.set_printoptions(threshold=float('inf'))

# Set boolean variables as per specification
useInference = False  # Disable useInference mode
lowMem = True		 # Enable lowMem mode (can only be used when useInference is disabled)
usePOS = True		 # Enable usePOS mode
useParallelProcessing = True	#mandatory (else restore original code pre-GIAANNproto1b3a)
useSaveData = True
sequenceObservedColumnsUseSequenceFeaturesOnly = True	#sequence observed columns arrays only store sequence features.
drawSequenceObservedColumns = False	#draw sequence observed columns (instead of complete observed columns)	#note if !drawSequenceObservedColumns and !sequenceObservedColumnsUseSequenceFeaturesOnly, then will still draw complete columns
	
useDedicatedFeatureLists = False
if usePOS and not lowMem:
	useDedicatedFeatureLists = True
	
useDedicatedConceptNames1 = False
useDedicatedConceptNames2 = False
if usePOS:
	#same word can have different pos making it classed as an instance feature or concept feature
	useDedicatedConceptNames2 = True
	#useDedicatedConceptNames1 = True
	if(useDedicatedConceptNames1):
		concept_prefix = "C_"
	
	
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

# Define constants for permanence and activation trace
z1 = 3  # Initial permanence value
z2 = 1  # Decrement value when not activated
j1 = 5   # Activation trace duration

# Initialize NetworkX graph for visualization
G = nx.Graph()

# For the purpose of the example, process a limited number of sentences
sentence_count = 0
max_sentences = 1000  # Adjust as needed


if not lowMem:
	feature_neurons_file = 'global_feature_neurons.pt'

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

# Initialize the concept columns dictionary
if useInference and os.path.exists(concept_columns_dict_file):
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
	
	# Add dummy feature for concept neuron (different per concept column)
	feature_index_concept_neuron = 0
	variableConceptNeuronFeatureName = "variableConceptNeuronFeature"
	concept_features_list.append(variableConceptNeuronFeatureName)
	concept_features_dict[variableConceptNeuronFeatureName] = len(concept_features_dict)
	f = 1  # Will be updated dynamically based on c

	if useDedicatedFeatureLists:
		print("error: useDedicatedFeatureLists case not yet coded - need to set f and populate concept_features_list/concept_features_dict etc")
		exit()
		# f = max_num_non_nouns + 1  # Maximum number of non-nouns in an English dictionary, plus the concept neuron of each column
  
# Initialize global feature neuron arrays if lowMem is disabled
if not lowMem:
	if os.path.exists(feature_neurons_file):
		global_feature_neurons = torch.load(feature_neurons_file)
	else:
		global_feature_neurons = SequenceObservedColumns.initialiseFeatureNeuronsSequence(c, f)

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
		self.next_feature_index = 1  # Start from 1 since index 0 is reserved for concept neuron
		if(useDedicatedConceptNames2):
			self.feature_word_to_index[variableConceptNeuronFeatureName] = feature_index_concept_neuron
			self.feature_index_to_word[feature_index_concept_neuron] = variableConceptNeuronFeatureName
		else:
			self.feature_word_to_index[lemma] = feature_index_concept_neuron
			self.feature_index_to_word[feature_index_concept_neuron] = lemma

		# Store all connections for each source column in a list of integer feature connection arrays, each of size f * c * f, where c is the length of the dictionary of columns, and f is the maximum number of feature neurons.
		self.feature_connections = self.initialiseFeatureConnections(c, f) 

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
	def __init__(self, words, lemmas, observed_columns_dict, observed_columns_sequence_word_index_dict):
		self.observed_columns_dict = observed_columns_dict
		self.observed_columns_sequence_word_index_dict = observed_columns_sequence_word_index_dict
		
		# Map from concept names to indices in sequence arrays
		self.concept_name_to_index = {}
		self.index_to_concept_name = {}
		self.cs = len(observed_columns_dict)
		for idx, (lemma, observed_column) in enumerate(observed_columns_dict.items()):
			self.concept_name_to_index[lemma] = idx
			self.index_to_concept_name[idx] = lemma

		# Collect all feature words from observed columns
		self.words = words
		self.lemmas = lemmas
		feature_words = []
		for observed_column in observed_columns_dict.values():
			column_feature_words, _ = self.getObservedColumnFeatureWords(words, lemmas, observed_column)
			feature_words.extend(column_feature_words)
		feature_words = self.removeDuplicates(feature_words)
			
		self.fs = len(feature_words)
		self.feature_word_to_index = {}
		self.index_to_feature_word = {}
		for idx, feature_word in enumerate(feature_words):
			self.feature_word_to_index[feature_word] = idx
			self.index_to_feature_word[idx] = feature_word
		
		# Initialize arrays
		self.feature_neurons = self.initialiseFeatureNeuronsSequence(self.cs, self.fs)
		self.feature_connections = self.initialiseFeatureConnectionsSequence(self.cs, self.fs)

		# Populate arrays with data from observed_columns_dict
		self.populate_arrays(words, lemmas, observed_columns_dict)

	def getObservedColumnFeatureWords(self, words, lemmas, observed_column):
		if(sequenceObservedColumnsUseSequenceFeaturesOnly):
			conceptNeuronFound = False
			feature_words = []
			feature_indices_in_observed = []
			for word, lemma in zip(words, lemmas):
				feature_word = word.lower()
				feature_lemma = lemma
				if(feature_lemma == observed_column.concept_name):
					if(useDedicatedConceptNames2):
						feature_words.append(variableConceptNeuronFeatureName)
					else:
						feature_words.append(feature_lemma)
					feature_indices_in_observed.append(feature_index_concept_neuron)	#or observed_column.feature_word_to_index[feature_lemma]
					conceptNeuronFound = True
					#print("concept node found = ", feature_lemma)
				elif(feature_word in observed_column.feature_word_to_index):
					feature_words.append(feature_word)
					feature_indices_in_observed.append(observed_column.feature_word_to_index[feature_word])
			feature_indices_in_observed = self.removeDuplicates(feature_indices_in_observed)
			feature_indices_in_observed_tensor = torch.tensor(list(feature_indices_in_observed), dtype=torch.long)
			if(not conceptNeuronFound):
				print("getObservedColumnFeatureWords error: !conceptNeuronFound")
				exit()
		else:
			feature_words = observed_column.feature_word_to_index.keys()
			feature_indices_in_observed_tensor = torch.tensor(list(observed_column.feature_word_to_index.values()), dtype=torch.long)
		feature_words = self.removeDuplicates(feature_words)
		return feature_words, feature_indices_in_observed_tensor
	
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
	
	def populate_arrays(self, words, lemmas, observed_columns_dict):
		# Collect indices and data for feature neurons
		c_idx_list = []
		f_idx_list = []
		feature_list = []

		for lemma, observed_column in observed_columns_dict.items():
			feature_words, feature_indices_in_observed = self.getObservedColumnFeatureWords(self.words, self.lemmas, observed_column)
			c_idx = self.concept_name_to_index[lemma]
			f_idx_tensor = torch.tensor([self.feature_word_to_index[fw] for fw in feature_words], dtype=torch.long)

			num_features = len(f_idx_tensor)

			c_idx_list.append(torch.full((num_features,), c_idx, dtype=torch.long))
			f_idx_list.append(f_idx_tensor)

			feature_list.append(observed_column.feature_neurons[:, :, feature_indices_in_observed])

		# Concatenate lists to tensors
		c_idx_tensor = torch.cat(c_idx_list)
		f_idx_tensor = torch.cat(f_idx_list)
		feature_tensor = torch.cat(feature_list, dim=2)

		if lowMem:
			# Use advanced indexing to assign values
			self.feature_neurons[:, :, c_idx_tensor, f_idx_tensor] = feature_tensor

		# Now handle connections
		connection_indices = []
		connection_values = []

		for lemma, observed_column in observed_columns_dict.items():
			feature_words, feature_indices_in_observed = self.getObservedColumnFeatureWords(self.words, self.lemmas, observed_column)
			c_idx = self.concept_name_to_index[lemma]
			f_idx_tensor = torch.tensor([self.feature_word_to_index[fw] for fw in feature_words], dtype=torch.long)

			for other_lemma, other_observed_column in observed_columns_dict.items():
				other_feature_words, other_feature_indices_in_observed = self.getObservedColumnFeatureWords(self.words, self.lemmas, other_observed_column)
				other_c_idx = self.concept_name_to_index[other_lemma]
				other_concept_index = other_observed_column.concept_index
				other_f_idx_tensor = torch.tensor([self.feature_word_to_index[fw] for fw in other_feature_words], dtype=torch.long)

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

	def update_observed_columns(self):
		# Update observed columns with data from sequence arrays
		
		self.index_to_feature_word = {}
		for lemma, observed_column in self.observed_columns_dict.items():
			c_idx = self.concept_name_to_index[lemma]
			feature_words, feature_indices_in_observed = self.getObservedColumnFeatureWords(self.words, self.lemmas, observed_column)
			f_idx_tensor = torch.tensor([self.feature_word_to_index[fw] for fw in feature_words], dtype=torch.long)

			if lowMem:
				# Use advanced indexing to get values from self.feature_neurons_*
				values = self.feature_neurons[:, :, c_idx, f_idx_tensor]
				
				# Assign values to observed_column's feature_neurons_* arrays
				observed_column.feature_neurons[:, :, feature_indices_in_observed] = values

			# Now handle connections
			conn_feature_indices_obs = []
			conn_other_concept_indices = []
			conn_other_feature_indices_obs = []
			conn_values = []
			
			for other_lemma, other_observed_column in self.observed_columns_dict.items():
				other_c_idx = self.concept_name_to_index[other_lemma]
				other_feature_words_set, other_feature_indices_in_observed = self.getObservedColumnFeatureWords(self.words, self.lemmas, other_observed_column)
				other_concept_index = other_observed_column.concept_index
				other_feature_words = list(other_feature_words_set)
				other_f_idx_tensor = torch.tensor([self.feature_word_to_index[fw] for fw in other_feature_words], dtype=torch.long)

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

			# Assign values to observed_column's connection_* arrays using advanced indexing
			observed_column.feature_connections[:, :, conn_feature_indices_obs, conn_other_concept_indices, conn_other_feature_indices_obs] = conn_values

def process_dataset(dataset):
	global sentence_count
	for article in dataset:
		process_article(article)
		if sentence_count >= max_sentences:
			break

def process_article(article):
	global sentence_count
	#sentences = sent_tokenize(article['text'])
	sentences = nlp(article['text'])
	for sentence in sentences.sents:
		process_sentence(sentence)
		if sentence_count >= max_sentences:
			break

def process_sentence(doc):
	global sentence_count, c, f, concept_columns_dict, concept_columns_list, concept_features_dict, concept_features_list
	print(f"Processing sentence: {doc.text}")

	# Refresh the observed columns dictionary for each new sequence
	observed_columns_dict = {}  # key: lemma, value: ObservedColumn
	observed_columns_sequence_word_index_dict = {}  # key: sequence word index, value: concept_index

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

		# Process each concept word in the sequence
		process_concept_words(doc, words, lemmas, pos_tags, sequence_observed_columns)

		# Update activation traces for feature neurons and connections
		#update_activation(sequence_observed_columns)

		# Update observed columns from sequence observed columns
		sequence_observed_columns.update_observed_columns()

		# Visualize the complete graph every time a new sentence is parsed by the application.
		visualize_graph(sequence_observed_columns)
		
		# Save observed columns to disk
		if(useSaveData):
			save_data(observed_columns_dict, concept_features_dict)

	# Break if we've reached the maximum number of sentences
	global sentence_count
	sentence_count += 1

def first_pass(doc):
	global c, f, concept_columns_dict, concept_columns_list
	if not lowMem:
		global global_feature_neurons
	words = []
	lemmas = []
	pos_tags = []
	new_concepts_added = False
	concepts_found = False
	
	for token in doc:
		word = token.text
		lemma = token.lemma_.lower()
		pos = token.pos_  # Part-of-speech tag

		if usePOS:
			if pos in noun_pos_tags:
				# Only assign unique concept columns for nouns
				concepts_found = True
				if(useDedicatedConceptNames1):
					lemma = concept_prefix + lemma
				if lemma not in concept_columns_dict:
					# Add to concept columns dictionary
					concept_columns_dict[lemma] = c
					concept_columns_list.append(lemma)
					c += 1
					new_concepts_added = True
		else:
			# When usePOS is disabled, assign concept columns for every new lemma encountered
			concepts_found = True
			if lemma not in concept_columns_dict:
				concept_columns_dict[lemma] = c
				concept_columns_list.append(lemma)
				c += 1
				new_concepts_added = True

		words.append(word)
		lemmas.append(lemma)
		pos_tags.append(pos)
		
	# If new concept columns have been added, expand arrays as needed
	if new_concepts_added:
		if not lowMem:
			# Expand global feature neuron arrays
			if global_feature_neurons.shape[2] < c:
				extra_rows = c - global_feature_neurons.shape[2]
				extra_global_feature_neurons = SequenceObservedColumns.initialiseFeatureNeuronsSequence(extra_rows, f)
				global_feature_neurons = torch.cat([global_feature_neurons, extra_global_feature_neurons], dim=2)
				
	return concepts_found, words, lemmas, pos_tags

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
				observed_columns_sequence_word_index_dict[i] = concept_index
		else:
			concept_index = concept_columns_dict[lemma]
			# Load observed column from disk or create new one
			observed_column = load_or_create_observed_column(concept_index, lemma, i)
			observed_columns_dict[lemma] = observed_column
			observed_columns_sequence_word_index_dict[i] = concept_index
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
			extra_cols = f - global_feature_neurons_strength.shape[3]
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
	

def process_concept_words(doc, words, lemmas, pos_tags, sequence_observed_columns):
	"""
	For every concept word (lemma) in the sequence, identify every feature neuron in that column that occurs q words before or after the concept word in the sequence, including the concept neuron. This function has been parallelized using PyTorch array operations.
	"""
	global c, f, lowMem, global_feature_neurons

	if not usePOS:
		q = 5  # Fixed window size when not using POS tags

	# Identify all concept word indices
	concept_mask = torch.tensor([i in sequence_observed_columns.observed_columns_sequence_word_index_dict for i in range(len(lemmas))], dtype=torch.bool)
	concept_indices = torch.nonzero(concept_mask).squeeze(1)
	
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

	process_features(start_indices, end_indices, doc, words, lemmas, pos_tags, sequence_observed_columns, concept_indices)
   
def process_features(start_indices, end_indices, doc, words, lemmas, pos_tags, sequence_observed_columns, concept_indices):
	
	'''
	#not possible as sequence_observed_columns feature arrays are only defined for sequence features, not all sequence words;
	#convert start/end indices to active features arrays
	num_rows = len(concept_indices)
	num_cols = len(doc)
	feature_neurons_active = torch.zeros((num_rows, num_cols), dtype=torch.long)
	for i in range(num_rows):
		feature_neurons_active[i, start_indices[i]:end_indices[i]] = 1
	'''
 
	cs = sequence_observed_columns.cs #will be different than len(concept_indices) if there are multiple instances of a concept in a sequence
	fs = sequence_observed_columns.fs  #will be different than len(doc) as not every word in the sequence has a feature neuron assigned in the concept columns
	feature_neurons_active = torch.zeros((cs, fs), dtype=array_type)
	for i in range(concept_indices.shape[0]):
		concept_lemma = lemmas[concept_indices[i]]
		sequence_concept_index = sequence_observed_columns.concept_name_to_index[concept_lemma]
		for j in range(start_indices[i], end_indices[i]):	#sequence word index
			feature_word = words[j].lower()
			feature_lemma = lemmas[j]
			if(j in sequence_observed_columns.observed_columns_sequence_word_index_dict):	#test is required for concept neurons
				if(useDedicatedConceptNames2):
					sequence_feature_index = sequence_observed_columns.feature_word_to_index[variableConceptNeuronFeatureName]
				else:
					sequence_feature_index = sequence_observed_columns.feature_word_to_index[feature_lemma]
				feature_neurons_active[sequence_concept_index, sequence_feature_index] = 1
				#print("feature_lemma concept set active = ", feature_lemma)
			elif(feature_word in sequence_observed_columns.feature_word_to_index):
				sequence_feature_index = sequence_observed_columns.feature_word_to_index[feature_word]
				feature_neurons_active[sequence_concept_index, sequence_feature_index] = 1
	feature_neurons_inactive = 1 - feature_neurons_active
	 
	if lowMem:
		# Update feature neurons in sequence_observed_columns
		sequence_observed_columns.feature_neurons[array_index_properties_strength, array_index_type_all, :, :] += feature_neurons_active
		sequence_observed_columns.feature_neurons[array_index_properties_permanence, array_index_type_all, :, :] = feature_neurons_active*(sequence_observed_columns.feature_neurons[array_index_properties_permanence, array_index_type_all] ** 2) + feature_neurons_inactive*sequence_observed_columns.feature_neurons[array_index_properties_permanence, array_index_type_all]
		sequence_observed_columns.feature_neurons[array_index_properties_activation, array_index_type_all, :, :] = feature_neurons_active*j1
	else:
		pass  # Not lowMem mode

	feature_neurons_active_1d = feature_neurons_active.view(cs*fs)
	feature_connections_active = torch.matmul(feature_neurons_active_1d.unsqueeze(1), feature_neurons_active_1d.unsqueeze(0)).view(cs, fs, cs, fs)
	feature_connections_inactive = 1 - feature_connections_active

	sequence_observed_columns.feature_connections[array_index_properties_strength, array_index_type_all, :, :, :, :] += feature_connections_active
	sequence_observed_columns.feature_connections[array_index_properties_permanence, array_index_type_all, :, :, :, :] = feature_connections_active*(sequence_observed_columns.feature_connections[array_index_properties_permanence, array_index_type_all] ** 2) + feature_connections_inactive*sequence_observed_columns.feature_connections[array_index_properties_permanence, array_index_type_all]
	sequence_observed_columns.feature_connections[array_index_properties_activation, array_index_type_all, :, :, :, :] = feature_connections_active*j1

	decrease_permanence(sequence_observed_columns, feature_neurons_active, feature_neurons_inactive)
 
def decrease_permanence(sequence_observed_columns, feature_neurons_active, feature_neurons_inactive):
	cs = sequence_observed_columns.cs
	fs = sequence_observed_columns.fs 
	
	# Decrease permanence for feature neurons not activated
	sequence_observed_columns.feature_neurons[array_index_properties_permanence, array_index_type_all, :, :] -= z2
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
  
def update_activation(sequence_observed_columns):
	# Update activation traces for feature neurons
	active_indices = sequence_observed_columns.feature_neurons[array_index_properties_activation, array_index_type_all].nonzero(as_tuple=False)
	for idx in active_indices:
		c_idx = idx[0].item()
		f_idx = idx[1].item()
		if sequence_observed_columns.feature_neurons[array_index_properties_activation, array_index_type_all, c_idx, f_idx] > 0:
			sequence_observed_columns.feature_neurons[array_index_properties_activation, array_index_type_all, c_idx, f_idx] -= 1
			if sequence_observed_columns.feature_neurons[array_index_properties_activation, array_index_type_all, c_idx, f_idx] == 0:
				pass  # Activation trace expired

	# Update activation traces for connections
	active_indices = sequence_observed_columns.feature_connections[array_index_properties_activation, array_index_type_all].nonzero(as_tuple=False)
	for idx in active_indices:
		c_i = idx[0].item()
		f_i = idx[1].item()
		c_j = idx[2].item()
		f_j = idx[3].item()
		if sequence_observed_columns.feature_connections[array_index_properties_activation, array_index_type_all, c_i, f_i, c_j, f_j] > 0:
			sequence_observed_columns.feature_connections[array_index_properties_activation, array_index_type_all, c_i, f_i, c_j, f_j] -= 1
			if sequence_observed_columns.feature_connections[array_index_properties_activation, array_index_type_all, c_i, f_i, c_j, f_j] == 0:
				pass  # Activation trace expired

def visualize_graph(sequence_observed_columns):
	G.clear()

	if(drawSequenceObservedColumns):
		# Draw concept columns
		pos_dict = {}
		x_offset = 0
		for lemma, observed_column in sequence_observed_columns.observed_columns_dict.items():
			concept_index = observed_column.concept_index
			c_idx = sequence_observed_columns.concept_name_to_index[lemma]

			# Draw feature neurons
			y_offset = 1
			for feature_word, feature_index_in_observed_column in observed_column.feature_word_to_index.items():
				if(useDedicatedConceptNames2):
					if feature_word==variableConceptNeuronFeatureName:
						neuron_color = 'blue'
						neuron_name = observed_column.concept_name
					else:
						neuron_color = 'cyan'
						neuron_name = feature_word
				else:
					neuron_color = 'blue' if feature_index_in_observed_column == 0 else 'cyan'
					neuron_name = feature_word
				if(feature_word in sequence_observed_columns.feature_word_to_index):
					f_idx = sequence_observed_columns.feature_word_to_index[feature_word]
					if lowMem:
						if(f_idx < sequence_observed_columns.fs and sequence_observed_columns.feature_neurons[array_index_properties_strength, array_index_type_all, c_idx, f_idx] > 0 and sequence_observed_columns.feature_neurons[array_index_properties_permanence, array_index_type_all, c_idx, f_idx] > 0):
							feature_node = f"{lemma}_{feature_word}_{f_idx}"
							G.add_node(feature_node, pos=(x_offset, y_offset), color=neuron_color, label=neuron_name)
							y_offset += 1

			# Draw rectangle around the column
			plt.gca().add_patch(plt.Rectangle((x_offset - 0.5, -0.5), 1, max(y_offset, 1) + 0.5, fill=False, edgecolor='black'))
			x_offset += 2  # Adjust x_offset for the next column

		# Draw connections
		for lemma, observed_column in sequence_observed_columns.observed_columns_dict.items():
			concept_index = observed_column.concept_index
			c_idx = sequence_observed_columns.concept_name_to_index[lemma]

			# Internal connections (yellow)
			for feature_word, feature_index_in_observed_column in observed_column.feature_word_to_index.items():
				if(feature_word in sequence_observed_columns.feature_word_to_index):
					source_node = f"{lemma}_{feature_word}_{sequence_observed_columns.feature_word_to_index[feature_word]}"
					if G.has_node(source_node):
						for other_feature_word, other_feature_index_in_observed_column in observed_column.feature_word_to_index.items():
							if(other_feature_word in sequence_observed_columns.feature_word_to_index):
								target_node = f"{lemma}_{other_feature_word}_{sequence_observed_columns.feature_word_to_index[other_feature_word]}"
								if G.has_node(target_node):
									if feature_word != other_feature_word:
										f_idx = sequence_observed_columns.feature_word_to_index[feature_word]
										other_f_idx = sequence_observed_columns.feature_word_to_index[other_feature_word]
										if(f_idx < sequence_observed_columns.fs and other_f_idx < sequence_observed_columns.fs and sequence_observed_columns.feature_connections[array_index_properties_strength, array_index_type_all, c_idx, f_idx, c_idx, other_f_idx] > 0 and sequence_observed_columns.feature_connections[array_index_properties_permanence, array_index_type_all, c_idx, f_idx, c_idx, other_f_idx] > 0):
											G.add_edge(source_node, target_node, color='yellow')

			# External connections (orange)
			for feature_word, feature_index_in_observed_column in observed_column.feature_word_to_index.items():
				if(feature_word in sequence_observed_columns.feature_word_to_index):
					source_node = f"{lemma}_{feature_word}_{sequence_observed_columns.feature_word_to_index[feature_word]}"
					if G.has_node(source_node):
						for other_lemma, other_observed_column in sequence_observed_columns.observed_columns_dict.items():
							for other_feature_word, other_feature_index_in_observed_column in other_observed_column.feature_word_to_index.items():
								if(other_feature_word in sequence_observed_columns.feature_word_to_index):
									target_node = f"{other_lemma}_{other_feature_word}_{sequence_observed_columns.feature_word_to_index[other_feature_word]}"
									if G.has_node(target_node):
										other_c_idx = sequence_observed_columns.concept_name_to_index[other_lemma]
										if other_c_idx != c_idx:
											f_idx = sequence_observed_columns.feature_word_to_index[feature_word]
											other_f_idx = sequence_observed_columns.feature_word_to_index[other_feature_word]
											if(f_idx < sequence_observed_columns.fs and other_f_idx < sequence_observed_columns.fs and sequence_observed_columns.feature_connections[array_index_properties_strength, array_index_type_all, c_idx, f_idx, other_c_idx, other_f_idx] > 0 and sequence_observed_columns.feature_connections[array_index_properties_permanence, array_index_type_all, c_idx, f_idx, other_c_idx, other_f_idx] > 0):
												G.add_edge(source_node, target_node, color='orange')
	else:
		# Draw concept columns
		pos_dict = {}
		x_offset = 0
		for lemma, observed_column in sequence_observed_columns.observed_columns_dict.items():
			concept_index = observed_column.concept_index
			c_idx = concept_columns_dict[lemma]

			# Draw feature neurons
			y_offset = 1
			for feature_word, feature_index_in_observed_column in observed_column.feature_word_to_index.items():
				if(useDedicatedConceptNames2):
					if feature_word==variableConceptNeuronFeatureName:
						neuron_color = 'blue'
						neuron_name = observed_column.concept_name
					else:
						neuron_color = 'cyan'
						neuron_name = feature_word
				else:
					neuron_color = 'blue' if feature_index_in_observed_column == 0 else 'cyan'
					neuron_name = feature_word
				f_idx = feature_index_in_observed_column
				if(observed_column.feature_neurons[array_index_properties_strength, array_index_type_all, feature_index_in_observed_column] > 0 and observed_column.feature_neurons[array_index_properties_permanence, array_index_type_all, feature_index_in_observed_column] > 0):
					feature_node = f"{lemma}_{feature_word}_{f_idx}"
					G.add_node(feature_node, pos=(x_offset, y_offset), color=neuron_color, label=neuron_name)
					y_offset += 1

			# Draw rectangle around the column
			plt.gca().add_patch(plt.Rectangle((x_offset - 0.5, -0.5), 1, max(y_offset, 1) + 0.5, fill=False, edgecolor='black'))
			x_offset += 2  # Adjust x_offset for the next column

		# Draw connections
		for lemma, observed_column in sequence_observed_columns.observed_columns_dict.items():
			concept_index = observed_column.concept_index
			c_idx = concept_columns_dict[lemma]
		
			# Internal connections (yellow)
			for feature_word, feature_index_in_observed_column in observed_column.feature_word_to_index.items():
				source_node = f"{lemma}_{feature_word}_{observed_column.feature_word_to_index[feature_word]}"
				if G.has_node(source_node):
					for other_feature_word, other_feature_index_in_observed_column in observed_column.feature_word_to_index.items():
						target_node = f"{lemma}_{other_feature_word}_{observed_column.feature_word_to_index[other_feature_word]}"
						if G.has_node(target_node):
							if feature_word != other_feature_word:
								f_idx = observed_column.feature_word_to_index[feature_word]
								other_f_idx = observed_column.feature_word_to_index[other_feature_word]
								if(observed_column.feature_connections[array_index_properties_strength, array_index_type_all, f_idx, c_idx, other_f_idx] > 0 and observed_column.feature_connections[array_index_properties_permanence, array_index_type_all, f_idx, c_idx, other_f_idx] > 0):
									G.add_edge(source_node, target_node, color='yellow')

			# External connections (orange)
			for feature_word, feature_index_in_observed_column in observed_column.feature_word_to_index.items():
				source_node = f"{lemma}_{feature_word}_{observed_column.feature_word_to_index[feature_word]}"
				if G.has_node(source_node):
					for other_lemma, other_observed_column in sequence_observed_columns.observed_columns_dict.items():
						for other_feature_word, other_feature_index_in_observed_column in other_observed_column.feature_word_to_index.items():
							target_node = f"{other_lemma}_{other_feature_word}_{other_observed_column.feature_word_to_index[other_feature_word]}"
							if G.has_node(target_node):
								other_c_idx = concept_columns_dict[other_lemma]
								f_idx = observed_column.feature_word_to_index[feature_word]
								other_f_idx = other_observed_column.feature_word_to_index[other_feature_word]
								if lemma != other_lemma:	#if observed_column != other_observed_column:
									if(observed_column.feature_connections[array_index_properties_strength, array_index_type_all, f_idx, other_c_idx, other_f_idx] > 0 and observed_column.feature_connections[array_index_properties_permanence, array_index_type_all, f_idx, other_c_idx, other_f_idx] > 0):
										G.add_edge(source_node, target_node, color='orange')
								
	# Get positions and colors for drawing
	pos = nx.get_node_attributes(G, 'pos')
	colors = [data['color'] for node, data in G.nodes(data=True)]
	edge_colors = [data['color'] for u, v, data in G.edges(data=True)]
	labels = nx.get_node_attributes(G, 'label')

	# Draw the graph
	nx.draw(G, pos, with_labels=True, labels=labels, node_color=colors, edge_color=edge_colors, node_size=500, font_size=8)
	plt.axis('off')  # Hide the axes
	plt.show()

def save_data(observed_columns_dict, concept_features_dict):
	# Save observed columns to disk
	for observed_column in observed_columns_dict.values():
		observed_column.save_to_disk()

	# Save global feature neuron arrays if not lowMem
	if not lowMem:
		torch.save(global_feature_neurons, feature_neurons_file)

	# Save concept columns dictionary to disk
	with open(concept_columns_dict_file, 'wb') as f_out:
		pickle.dump(concept_columns_dict, f_out)
		
	# Save concept features dictionary to disk
	with open(concept_features_dict_file, 'wb') as f_out:
		pickle.dump(concept_features_dict, f_out)

# Load the Wikipedia dataset using Hugging Face datasets
dataset = load_dataset('wikipedia', '20220301.en', split='train', streaming=True)

# Start processing the dataset
process_dataset(dataset)
