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

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize

# Set boolean variables as per specification
useInference = False  # Disable useInference mode
lowMem = True		 # Enable lowMem mode (can only be used when useInference is disabled)
usePOS = True		 # Enable usePOS mode
useParallelProcessing = True	#mandatory (else restore original code pre-GIAANNproto1b3a)
useSaveData = False

# Paths for saving data
concept_columns_dict_file = 'concept_columns_dict.pkl'
concept_features_dict_file = 'concept_features_dict.pkl'
observed_columns_dir = 'observed_columns'
os.makedirs(observed_columns_dir, exist_ok=True)

if not lowMem:
	feature_neurons_strength_file = 'global_feature_neurons_strength.pt'
	feature_neurons_permanence_file = 'global_feature_neurons_permanence.pt'
	feature_neurons_activation_file = 'global_feature_neurons_activation.pt'

if usePOS and not lowMem:
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

	if usePOS and not lowMem:
		print("error: usePOS and not lowMem case not yet coded - need to set f and populate concept_features_list/concept_features_dict etc")
		exit()
		# f = max_num_non_nouns + 1  # Maximum number of non-nouns in an English dictionary, plus the concept neuron of each column
  
# Initialize global feature neuron arrays if lowMem is disabled
if not lowMem:
	if os.path.exists(feature_neurons_strength_file):
		global_feature_neurons_strength = torch.load(feature_neurons_strength_file)
		global_feature_neurons_permanence = torch.load(feature_neurons_permanence_file)
		global_feature_neurons_activation = torch.load(feature_neurons_activation_file)
	else:
		global_feature_neurons_strength = torch.zeros(c, f)
		global_feature_neurons_permanence = torch.full((c, f), 3, dtype=torch.int32)  # Initialize permanence to z1=3
		global_feature_neurons_activation = torch.zeros(c, f, dtype=torch.int32)  # Activation trace

# Initialize spaCy model
nlp = spacy.load('en_core_web_sm')

# Define POS tag sets for nouns and non-nouns
noun_pos_tags = {'NOUN', 'PROPN'}
non_noun_pos_tags = {'ADJ', 'ADV', 'VERB', 'ADP', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NUM', 'PART', 'PRON', 'SCONJ', 'SYM', 'X'}

# Define constants for permanence and activation trace
z1 = 3  # Initial permanence value
z2 = 1  # Decrement value when not activated
j1 = 5   # Activation trace duration

# Define the ObservedColumn class
class ObservedColumn:
	"""
	Create a class defining observed columns. The observed column class contains an index to the
	dataset concept column dictionary. The observed column class contains a list of feature
	connection arrays. The observed column class also contains a list of feature neuron arrays
	when lowMem mode is enabled.
	"""
	def __init__(self, concept_index, lemma):
		self.concept_index = concept_index  # Index to the concept columns dictionary

		if lowMem:
			# If lowMem is enabled, the observed columns contain a list of arrays (pytorch)
			# of f feature neurons, where f is the maximum number of feature neurons per column.
			self.feature_neurons_strength = torch.zeros(f)
			self.feature_neurons_permanence = torch.full((f,), z1, dtype=torch.int32)  # Initialize permanence to z1=3
			self.feature_neurons_activation = torch.zeros(f, dtype=torch.int32)  # Activation trace counters

		# Map from feature words to indices in feature neurons
		self.feature_word_to_index = {}  # Maps feature words to indices
		self.feature_index_to_word = {}  # Maps indices to feature words
		self.next_feature_index = 1  # Start from 1 since index 0 is reserved for concept neuron
		self.feature_word_to_index[lemma] = 0
		self.feature_index_to_word[0] = lemma

		# Store all connections for each source column in a list of integer feature connection arrays,
		# each of size f * c * f, where c is the length of the dictionary of columns, and f is the maximum
		# number of feature neurons.
		self.connection_strength = torch.zeros(f, c, f, dtype=torch.int32)
		self.connection_permanence = torch.full((f, c, f), z1, dtype=torch.int32)  # Initialize permanence to z1=3
		self.connection_activation = torch.zeros(f, c, f, dtype=torch.int32)  # Activation trace counters

		for feature_index in range(1, f, 1):
			feature_word = concept_features_list[feature_index]
			self.feature_word_to_index[feature_word] = feature_index
			self.feature_index_to_word[feature_index] = feature_word
			self.next_feature_index += 1

	def resize_concept_arrays(self, new_c):
		load_c = self.connection_strength.shape[1]
		if new_c > load_c:
			extra_cols = new_c - load_c
			# Expand along dimension 1 (columns)
			self.connection_strength = torch.cat([self.connection_strength, torch.zeros(self.connection_strength.shape[0], extra_cols, self.connection_strength.shape[2], dtype=torch.int32)], dim=1)
			self.connection_permanence = torch.cat([self.connection_permanence, torch.full((self.connection_permanence.shape[0], extra_cols, self.connection_permanence.shape[2]), z1, dtype=torch.int32)], dim=1)
			self.connection_activation = torch.cat([self.connection_activation, torch.zeros(self.connection_activation.shape[0], extra_cols, self.connection_activation.shape[2], dtype=torch.int32)], dim=1)

	def expand_feature_arrays(self, new_f):
		load_f = self.connection_strength.shape[0]	# or self.connection_strength.shape[2]	   
		if new_f > load_f:
			extra_features = new_f - load_f
			
			# Expand along dimension 0 (rows) and dimension 2
			self.connection_strength = torch.cat([self.connection_strength, torch.zeros(extra_features, self.connection_strength.shape[1], self.connection_strength.shape[2], dtype=torch.int32)], dim=0)
			self.connection_permanence = torch.cat([self.connection_permanence, torch.full((extra_features, self.connection_permanence.shape[1], self.connection_permanence.shape[2]), z1, dtype=torch.int32)], dim=0)
			self.connection_activation = torch.cat([self.connection_activation, torch.zeros(extra_features, self.connection_activation.shape[1], self.connection_activation.shape[2], dtype=torch.int32)], dim=0)

			# Also expand along dimension 2
			self.connection_strength = torch.cat([self.connection_strength, torch.zeros(self.connection_strength.shape[0], self.connection_strength.shape[1], extra_features, dtype=torch.int32)], dim=2)
			self.connection_permanence = torch.cat([self.connection_permanence, torch.full((self.connection_permanence.shape[0], self.connection_permanence.shape[1], extra_features), z1, dtype=torch.int32)], dim=2)
			self.connection_activation = torch.cat([self.connection_activation, torch.zeros(self.connection_activation.shape[0], self.connection_activation.shape[1], extra_features, dtype=torch.int32)], dim=2)

			if lowMem:
				self.feature_neurons_strength = torch.cat([self.feature_neurons_strength, torch.zeros(extra_features)], dim=0)
				self.feature_neurons_permanence = torch.cat([self.feature_neurons_permanence, torch.full((extra_features,), z1, dtype=torch.int32)], dim=0)
				self.feature_neurons_activation = torch.cat([self.feature_neurons_activation, torch.zeros(extra_features, dtype=torch.int32)], dim=0)
 
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
		torch.save(self.connection_strength, os.path.join(observed_columns_dir, f"{self.concept_index}_connection_strength.pt"))
		torch.save(self.connection_permanence, os.path.join(observed_columns_dir, f"{self.concept_index}_connection_permanence.pt"))
		torch.save(self.connection_activation, os.path.join(observed_columns_dir, f"{self.concept_index}_connection_activation.pt"))
		if lowMem:
			torch.save(self.feature_neurons_strength, os.path.join(observed_columns_dir, f"{self.concept_index}_feature_neurons_strength.pt"))
			torch.save(self.feature_neurons_permanence, os.path.join(observed_columns_dir, f"{self.concept_index}_feature_neurons_permanence.pt"))
			torch.save(self.feature_neurons_activation, os.path.join(observed_columns_dir, f"{self.concept_index}_feature_neurons_activation.pt"))

	@classmethod
	def load_from_disk(cls, concept_index, lemma):
		"""
		Load the observed column data from disk.
		"""
		# Load the data dictionary
		with open(os.path.join(observed_columns_dir, f"{concept_index}_data.pkl"), 'rb') as f:
			data = pickle.load(f)
		instance = cls(concept_index, lemma)
		instance.feature_word_to_index = data['feature_word_to_index']
		instance.feature_index_to_word = data['feature_index_to_word']
		instance.next_feature_index = data['next_feature_index']
		# Load the tensors
		instance.connection_strength = torch.load(os.path.join(observed_columns_dir, f"{concept_index}_connection_strength.pt"))
		instance.connection_permanence = torch.load(os.path.join(observed_columns_dir, f"{concept_index}_connection_permanence.pt"))
		instance.connection_activation = torch.load(os.path.join(observed_columns_dir, f"{concept_index}_connection_activation.pt"))
		if lowMem:
			instance.feature_neurons_strength = torch.load(os.path.join(observed_columns_dir, f"{concept_index}_feature_neurons_strength.pt"))
			instance.feature_neurons_permanence = torch.load(os.path.join(observed_columns_dir, f"{concept_index}_feature_neurons_permanence.pt"))
			instance.feature_neurons_activation = torch.load(os.path.join(observed_columns_dir, f"{concept_index}_feature_neurons_activation.pt"))
		return instance

# Define the SequenceObservedColumns class
class SequenceObservedColumns:
	"""
	Contains sequence observed columns object arrays which stack a feature subset of the observed columns object arrays for the current sequence.
	"""
	def __init__(self, observed_columns_dict):
		# Map from concept names to indices in sequence arrays
		self.concept_name_to_index = {}
		self.index_to_concept_name = {}
		self.cs = len(observed_columns_dict)
		for idx, (lemma, observed_column) in enumerate(observed_columns_dict.items()):
			self.concept_name_to_index[lemma] = idx
			self.index_to_concept_name[idx] = lemma

		# Collect all feature words from observed columns
		feature_words_set = set()
		for observed_column in observed_columns_dict.values():
			feature_words_set.update(observed_column.feature_word_to_index.keys())
		self.fs = len(feature_words_set)
		self.feature_word_to_index = {}
		self.index_to_feature_word = {}
		for idx, feature_word in enumerate(feature_words_set):
			self.feature_word_to_index[feature_word] = idx
			self.index_to_feature_word[idx] = feature_word

		# Map from (lemma, feature_word) to indices in sequence arrays
		self.feature_index_map = {}
		for lemma, observed_column in observed_columns_dict.items():
			for feature_word, feature_index_in_observed_column in observed_column.feature_word_to_index.items():
				f_idx = self.feature_word_to_index[feature_word]
				self.feature_index_map[(lemma, feature_word)] = f_idx

		# Initialize arrays
		self.feature_neurons_strength = torch.zeros(self.cs, self.fs)
		self.feature_neurons_permanence = torch.full((self.cs, self.fs), z1, dtype=torch.int32)
		self.feature_neurons_activation = torch.zeros(self.cs, self.fs, dtype=torch.int32)

		self.connection_strength = torch.zeros(self.cs, self.fs, self.cs, self.fs, dtype=torch.int32)
		self.connection_permanence = torch.full((self.cs, self.fs, self.cs, self.fs), z1, dtype=torch.int32)
		self.connection_activation = torch.zeros(self.cs, self.fs, self.cs, self.fs, dtype=torch.int32)

		# Populate arrays with data from observed_columns_dict
		self.populate_arrays(observed_columns_dict)

	def populate_arrays(self, observed_columns_dict):
		# Collect indices and data for feature neurons
		c_idx_list = []
		f_idx_list = []
		feature_strength_list = []
		feature_permanence_list = []
		feature_activation_list = []

		for lemma, observed_column in observed_columns_dict.items():
			c_idx = self.concept_name_to_index[lemma]
			feature_words = list(observed_column.feature_word_to_index.keys())
			feature_indices_in_observed = torch.tensor(list(observed_column.feature_word_to_index.values()), dtype=torch.long)
			f_idx_tensor = torch.tensor([self.feature_word_to_index[fw] for fw in feature_words], dtype=torch.long)

			num_features = len(f_idx_tensor)

			c_idx_list.append(torch.full((num_features,), c_idx, dtype=torch.long))
			f_idx_list.append(f_idx_tensor)

			feature_strength_list.append(observed_column.feature_neurons_strength[feature_indices_in_observed])
			feature_permanence_list.append(observed_column.feature_neurons_permanence[feature_indices_in_observed])
			feature_activation_list.append(observed_column.feature_neurons_activation[feature_indices_in_observed])

		# Concatenate lists to tensors
		c_idx_tensor = torch.cat(c_idx_list)
		f_idx_tensor = torch.cat(f_idx_list)
		feature_strength_tensor = torch.cat(feature_strength_list)
		feature_permanence_tensor = torch.cat(feature_permanence_list)
		feature_activation_tensor = torch.cat(feature_activation_list)

		if lowMem:
			# Use advanced indexing to assign values
			self.feature_neurons_strength[c_idx_tensor, f_idx_tensor] = feature_strength_tensor
			self.feature_neurons_permanence[c_idx_tensor, f_idx_tensor] = feature_permanence_tensor
			self.feature_neurons_activation[c_idx_tensor, f_idx_tensor] = feature_activation_tensor

		# Now handle connections
		connection_indices = []
		connection_strength_values = []
		connection_permanence_values = []
		connection_activation_values = []

		for lemma, observed_column in observed_columns_dict.items():
			c_idx = self.concept_name_to_index[lemma]
			feature_words = list(observed_column.feature_word_to_index.keys())
			feature_indices_in_observed = torch.tensor(list(observed_column.feature_word_to_index.values()), dtype=torch.long)
			f_idx_tensor = torch.tensor([self.feature_word_to_index[fw] for fw in feature_words], dtype=torch.long)

			for other_lemma, other_observed_column in observed_columns_dict.items():
				other_c_idx = self.concept_name_to_index[other_lemma]
				other_concept_index = other_observed_column.concept_index
				other_feature_words = list(other_observed_column.feature_word_to_index.keys())
				other_feature_indices_in_observed = torch.tensor(list(other_observed_column.feature_word_to_index.values()), dtype=torch.long)
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
				strength_values = observed_column.connection_strength[feature_idx_obs_flat, other_concept_index, other_feature_idx_obs_flat]
				permanence_values = observed_column.connection_permanence[feature_idx_obs_flat, other_concept_index, other_feature_idx_obs_flat]
				activation_values = observed_column.connection_activation[feature_idx_obs_flat, other_concept_index, other_feature_idx_obs_flat]

				# Append to lists
				connection_indices.append((c_idx_flat, f_idx_flat, other_c_idx_flat, other_f_idx_flat))
				connection_strength_values.append(strength_values)
				connection_permanence_values.append(permanence_values)
				connection_activation_values.append(activation_values)

		# Concatenate tensors
		c_idx_conn_tensor = torch.cat([idx[0] for idx in connection_indices])
		f_idx_conn_tensor = torch.cat([idx[1] for idx in connection_indices])
		other_c_idx_conn_tensor = torch.cat([idx[2] for idx in connection_indices])
		other_f_idx_conn_tensor = torch.cat([idx[3] for idx in connection_indices])

		connection_strength_tensor = torch.cat(connection_strength_values)
		connection_permanence_tensor = torch.cat(connection_permanence_values)
		connection_activation_tensor = torch.cat(connection_activation_values)

		# Use advanced indexing to assign connection values
		self.connection_strength[c_idx_conn_tensor, f_idx_conn_tensor, other_c_idx_conn_tensor, other_f_idx_conn_tensor] = connection_strength_tensor
		self.connection_permanence[c_idx_conn_tensor, f_idx_conn_tensor, other_c_idx_conn_tensor, other_f_idx_conn_tensor] = connection_permanence_tensor
		self.connection_activation[c_idx_conn_tensor, f_idx_conn_tensor, other_c_idx_conn_tensor, other_f_idx_conn_tensor] = connection_activation_tensor

	def update_observed_columns(self, observed_columns_dict):
		# Update observed columns with data from sequence arrays
		for lemma, observed_column in observed_columns_dict.items():
			c_idx = self.concept_name_to_index[lemma]
			feature_words = list(observed_column.feature_word_to_index.keys())
			feature_indices_in_observed = torch.tensor(list(observed_column.feature_word_to_index.values()), dtype=torch.long)
			f_idx_tensor = torch.tensor([self.feature_word_to_index[fw] for fw in feature_words], dtype=torch.long)

			if lowMem:
				# Use advanced indexing to get values from self.feature_neurons_*
				strength_values = self.feature_neurons_strength[c_idx, f_idx_tensor]
				permanence_values = self.feature_neurons_permanence[c_idx, f_idx_tensor]
				activation_values = self.feature_neurons_activation[c_idx, f_idx_tensor]

				# Assign values to observed_column's feature_neurons_* arrays
				observed_column.feature_neurons_strength[feature_indices_in_observed] = strength_values
				observed_column.feature_neurons_permanence[feature_indices_in_observed] = permanence_values
				observed_column.feature_neurons_activation[feature_indices_in_observed] = activation_values

			# Now handle connections
			conn_feature_indices_obs = []
			conn_other_concept_indices = []
			conn_other_feature_indices_obs = []
			conn_strength_values = []
			conn_permanence_values = []
			conn_activation_values = []

			for other_lemma, other_observed_column in observed_columns_dict.items():
				other_c_idx = self.concept_name_to_index[other_lemma]
				other_concept_index = other_observed_column.concept_index
				other_feature_words = list(other_observed_column.feature_word_to_index.keys())
				other_feature_indices_in_observed = torch.tensor(list(other_observed_column.feature_word_to_index.values()), dtype=torch.long)
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
				strength_values = self.connection_strength[c_idx, f_idx_flat, other_c_idx, other_f_idx_flat]
				permanence_values = self.connection_permanence[c_idx, f_idx_flat, other_c_idx, other_f_idx_flat]
				activation_values = self.connection_activation[c_idx, f_idx_flat, other_c_idx, other_f_idx_flat]

				# Append data to lists
				conn_feature_indices_obs.append(feature_idx_obs_flat)
				conn_other_concept_indices.append(torch.full_like(feature_idx_obs_flat, other_concept_index, dtype=torch.long))
				conn_other_feature_indices_obs.append(other_feature_idx_obs_flat)
				conn_strength_values.append(strength_values)
				conn_permanence_values.append(permanence_values)
				conn_activation_values.append(activation_values)

			# Concatenate lists to form tensors
			conn_feature_indices_obs = torch.cat(conn_feature_indices_obs)
			conn_other_concept_indices = torch.cat(conn_other_concept_indices)
			conn_other_feature_indices_obs = torch.cat(conn_other_feature_indices_obs)
			conn_strength_values = torch.cat(conn_strength_values)
			conn_permanence_values = torch.cat(conn_permanence_values)
			conn_activation_values = torch.cat(conn_activation_values)

			# Assign values to observed_column's connection_* arrays using advanced indexing
			observed_column.connection_strength[conn_feature_indices_obs, conn_other_concept_indices, conn_other_feature_indices_obs] = conn_strength_values
			observed_column.connection_permanence[conn_feature_indices_obs, conn_other_concept_indices, conn_other_feature_indices_obs] = conn_permanence_values
			observed_column.connection_activation[conn_feature_indices_obs, conn_other_concept_indices, conn_other_feature_indices_obs] = conn_activation_values

# Initialize NetworkX graph for visualization
G = nx.Graph()

# For the purpose of the example, process a limited number of sentences
sentence_count = 0
max_sentences = 1000  # Adjust as needed

def process_dataset(dataset):
	global sentence_count
	for article in dataset:
		process_article(article)
		if sentence_count >= max_sentences:
			break

def process_article(article):
	global sentence_count
	sentences = sent_tokenize(article['text'])
	for sentence in sentences:
		process_sentence(sentence)
		if sentence_count >= max_sentences:
			break

def process_sentence(sentence):
	global sentence_count, c, f, concept_columns_dict, concept_columns_list, concept_features_dict, concept_features_list
	print(f"Processing sentence: {sentence}")

	# Refresh the observed columns dictionary for each new sequence
	observed_columns_dict = {}  # key: lemma, value: ObservedColumn

	# Process the sentence with spaCy
	doc = nlp(sentence)

	# First pass: Extract words, lemmas, POS tags, and update concept_columns_dict and c
	words, lemmas, pos_tags = first_pass(doc)

	# When usePOS is enabled, detect all possible new features in the sequence
	if not (usePOS and not lowMem):
		detect_new_features(words, lemmas, pos_tags)
		
	# Second pass: Create observed_columns_dict
	observed_columns_dict = second_pass(lemmas, pos_tags)

	# Create the sequence observed columns object
	sequence_observed_columns = SequenceObservedColumns(observed_columns_dict)

	# Process each concept word in the sequence
	process_concept_words(doc, words, lemmas, pos_tags, observed_columns_dict, sequence_observed_columns)

	# Update activation traces for feature neurons and connections
	update_activation(observed_columns_dict, sequence_observed_columns)

	# Visualize the complete graph every time a new sentence is parsed by the application.
	visualize_graph(observed_columns_dict, sequence_observed_columns)

	# Update observed columns from sequence observed columns
	sequence_observed_columns.update_observed_columns(observed_columns_dict)

	# Save observed columns to disk
	if(useSaveData):
		save_data(observed_columns_dict, concept_features_dict)

	# Break if we've reached the maximum number of sentences
	global sentence_count
	sentence_count += 1

def first_pass(doc):
	global c, f, concept_columns_dict, concept_columns_list
	if not lowMem:
		global global_feature_neurons_strength, global_feature_neurons_permanence, global_feature_neurons_activation
	words = []
	lemmas = []
	pos_tags = []
	new_concepts_added = False

	for token in doc:
		word = token.text
		lemma = token.lemma_.lower()
		pos = token.pos_  # Part-of-speech tag

		words.append(word)
		lemmas.append(lemma)
		pos_tags.append(pos)

		if usePOS:
			if pos in noun_pos_tags:
				# Only assign unique concept columns for nouns
				if lemma not in concept_columns_dict:
					# Add to concept columns dictionary
					concept_columns_dict[lemma] = c
					concept_columns_list.append(lemma)
					c += 1
					new_concepts_added = True
		else:
			# When usePOS is disabled, assign concept columns for every new lemma encountered
			if lemma not in concept_columns_dict:
				concept_columns_dict[lemma] = c
				concept_columns_list.append(lemma)
				c += 1
				new_concepts_added = True

	# If new concept columns have been added, expand arrays as needed
	if new_concepts_added:
		if not lowMem:
			# Expand global feature neuron arrays
			if global_feature_neurons_strength.shape[0] < c:
				extra_rows = c - global_feature_neurons_strength.shape[0]
				global_feature_neurons_strength = torch.cat([global_feature_neurons_strength, torch.zeros(extra_rows, f)], dim=0)
				global_feature_neurons_permanence = torch.cat([global_feature_neurons_permanence, torch.full((extra_rows, f), z1, dtype=torch.int32)], dim=0)
				global_feature_neurons_activation = torch.cat([global_feature_neurons_activation, torch.zeros(extra_rows, f, dtype=torch.int32)], dim=0)

	return words, lemmas, pos_tags

def second_pass(lemmas, pos_tags):
	observed_columns_dict = {}
	for i, lemma in enumerate(lemmas):
		pos = pos_tags[i]
		if usePOS:
			if pos in noun_pos_tags:
				concept_index = concept_columns_dict[lemma]
				# Load observed column from disk or create new one
				observed_column = load_or_create_observed_column(concept_index, lemma)
				observed_columns_dict[lemma] = observed_column
		else:
			concept_index = concept_columns_dict[lemma]
			# Load observed column from disk or create new one
			observed_column = load_or_create_observed_column(concept_index, lemma)
			observed_columns_dict[lemma] = observed_column
	return observed_columns_dict

def load_or_create_observed_column(concept_index, lemma=None):
	observed_column_file = os.path.join(observed_columns_dir, f"{concept_index}_data.pkl")
	if os.path.exists(observed_column_file):
		observed_column = ObservedColumn.load_from_disk(concept_index, lemma)
		# Resize connection arrays if c has increased
		observed_column.resize_concept_arrays(c)
		# Also expand feature arrays if f has increased
		observed_column.expand_feature_arrays(f)
	else:
		observed_column = ObservedColumn(concept_index, lemma)
		# Initialize connection arrays with correct size
		observed_column.resize_concept_arrays(c)
		observed_column.expand_feature_arrays(f)
	return observed_column

def detect_new_features(words, lemmas, pos_tags):
	"""
	When usePOS mode is enabled, detect all possible new features in the sequence
	by searching for all new non-nouns in the sequence.
	"""
	global f, lowMem, global_feature_neurons_strength, global_feature_neurons_permanence, global_feature_neurons_activation

	num_new_features = 0
	for j, (word_j, pos_j) in enumerate(zip(words, pos_tags)):
		if(process_feature_detection(j, word_j, pos_tags)):
			num_new_features += 1

	# After processing all features, update f
	f += num_new_features

	# Now, expand arrays accordingly
	if not lowMem:
		if f > global_feature_neurons_strength.shape[1]:
			extra_cols = f - global_feature_neurons_strength.shape[1]
			global_feature_neurons_strength = torch.cat([global_feature_neurons_strength, torch.zeros(global_feature_neurons_strength.shape[0], extra_cols)], dim=1)
			global_feature_neurons_permanence = torch.cat([global_feature_neurons_permanence, torch.full((global_feature_neurons_permanence.shape[0], extra_cols), z1, dtype=torch.int32)], dim=1)
			global_feature_neurons_activation = torch.cat([global_feature_neurons_activation, torch.zeros(global_feature_neurons_activation.shape[0], extra_cols, dtype=torch.int32)], dim=1)

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
	

def process_concept_words(doc, words, lemmas, pos_tags, observed_columns_dict, sequence_observed_columns):
	"""
	For every concept word (lemma) in the sequence, identify every feature neuron in that column
	that occurs q words before or after the concept word in the sequence, including the concept neuron.
	This function has been parallelized using PyTorch array operations.
	"""
	global c, f, lowMem, global_feature_neurons_strength, global_feature_neurons_permanence, global_feature_neurons_activation

	if not usePOS:
		q = 5  # Fixed window size when not using POS tags

	# Identify all concept word indices where lemma is in observed_columns_dict
	concept_mask = torch.tensor([lemma in observed_columns_dict for lemma in lemmas], dtype=torch.bool)
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
		dist_to_prev_concept = torch.where(prev_concept_exists, concept_indices - prev_concept_indices, concept_indices) #If no previous concept, distance is the index itself
		
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
		start_indices = (concept_indices - dist_to_prev_concept).clamp(min=0)
		end_indices = (concept_indices + dist_to_next_concept).clamp(max=len(doc))
	else:
		start_indices = (concept_indices - q).clamp(min=0)
		end_indices = (concept_indices + q + 1).clamp(max=len(doc))

	process_features(start_indices, end_indices, doc, words, lemmas, pos_tags, observed_columns_dict, sequence_observed_columns, concept_indices)
   
def process_features(start_indices, end_indices, doc, words, lemmas, pos_tags, observed_columns_dict, sequence_observed_columns, concept_indices):

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
	feature_neurons_active = torch.zeros((cs, fs), dtype=torch.long)
	for i in range(concept_indices.shape[0]):
		concept_lemma = lemmas[concept_indices[i]]
		sequence_concept_index = sequence_observed_columns.concept_name_to_index[concept_lemma]
		for j in range(start_indices[i], end_indices[i]):	#sequence word index
			feature_word = words[j].lower()
			feature_lemma = lemmas[j]
			if(feature_word in sequence_observed_columns.feature_word_to_index):
				sequence_feature_index = sequence_observed_columns.feature_word_to_index[feature_word]
				feature_neurons_active[sequence_concept_index, sequence_feature_index] = 1
			elif(feature_lemma in sequence_observed_columns.feature_word_to_index):	#feature_lemma test is required for concept neurons
				sequence_feature_index = sequence_observed_columns.feature_word_to_index[feature_lemma]
				feature_neurons_active[sequence_concept_index, sequence_feature_index] = 1
	#feature_neurons_active[:, feature_index_concept_neuron] = 1	#always use the concept neuron feature of each column during training	#index 0 is not reserved
	feature_neurons_inactive = 1 - feature_neurons_active
	   
	if lowMem:
		# Update feature neurons in sequence_observed_columns
		sequence_observed_columns.feature_neurons_strength[:, :] += feature_neurons_active
		sequence_observed_columns.feature_neurons_permanence[:, :] = feature_neurons_active*(sequence_observed_columns.feature_neurons_permanence ** 2) + feature_neurons_inactive*sequence_observed_columns.feature_neurons_permanence
		sequence_observed_columns.feature_neurons_activation[:, :] = feature_neurons_active*j1
	else:
		pass  # Not lowMem mode

	feature_connections_active = feature_neurons_active.unsqueeze(2).unsqueeze(3).expand(cs, fs, cs, fs)
	feature_connections_inactive = 1 - feature_connections_active
	
	sequence_observed_columns.connection_strength[:, :, :, :] += feature_connections_active
	sequence_observed_columns.connection_permanence[:, :, :, :] = feature_connections_active*(sequence_observed_columns.connection_permanence ** 2) + feature_connections_inactive*sequence_observed_columns.connection_permanence
	sequence_observed_columns.connection_activation[:, :, :, :] = feature_connections_active*j1

	#decrease_permanence(observed_columns_dict, sequence_observed_columns, feature_neurons_active, feature_neurons_inactive, feature_connections_active, feature_connections_inactive)
 
def decrease_permanence(observed_columns_dict, sequence_observed_columns, feature_neurons_active, feature_neurons_inactive, feature_connections_active, feature_connections_inactive):
	# Decrease permanence for feature neurons not activated
	sequence_observed_columns.feature_neurons_permanence[:, :] -= z2
	sequence_observed_columns.feature_neurons_permanence = torch.clamp(sequence_observed_columns.feature_neurons_permanence, min=0)

	feature_neurons_all = torch.ones((sequence_observed_columns.cs, sequence_observed_columns.fs), dtype=torch.long)
	feature_neurons_all_expand1 = feature_neurons_all.unsqueeze(2).unsqueeze(3)	# Shape (c, f, 1, 1)
	feature_neurons_all_expand2 = feature_neurons_all.unsqueeze(0).unsqueeze(1)	# Shape (1, 1, c, f)
	feature_connections_active_expand1 = feature_neurons_active.unsqueeze(2).unsqueeze(3)	# Shape (c, f, 1, 1)
	feature_connections_inactive_expand1 = feature_neurons_inactive.unsqueeze(2).unsqueeze(3)	# Shape (c, f, 1, 1)
	feature_connections_active_expand2 = feature_neurons_active.unsqueeze(0).unsqueeze(1)	# Shape (1, 1, c, f)
	feature_connections_inactive_expand2 = feature_neurons_inactive.unsqueeze(0).unsqueeze(1)	# Shape (1, 1, c, f)
	 
	# Decrease permanence of connections from inactive feature neurons in column
	feature_connections_decrease1 = feature_connections_inactive_expand1 * feature_neurons_all_expand2	# Shape (c, f, c, f)
	sequence_observed_columns.connection_permanence[:, :, :, :] -= feature_connections_decrease1
	sequence_observed_columns.connection_permanence = torch.clamp(sequence_observed_columns.connection_permanence, min=0)
	
	# Decrease permanence of inactive connections for activated features in column 
	feature_connections_decrease2 = feature_connections_active_expand1 * feature_connections_inactive_expand2	# Shape (c, f, c, f)
	sequence_observed_columns.connection_permanence[:, :, :, :] -= feature_connections_decrease2
	sequence_observed_columns.connection_permanence = torch.clamp(sequence_observed_columns.connection_permanence, min=0)
  
def update_activation(observed_columns_dict, sequence_observed_columns):
	# Update activation traces for feature neurons
	active_indices = sequence_observed_columns.feature_neurons_activation.nonzero(as_tuple=False)
	for idx in active_indices:
		c_idx = idx[0].item()
		f_idx = idx[1].item()
		if sequence_observed_columns.feature_neurons_activation[c_idx, f_idx] > 0:
			sequence_observed_columns.feature_neurons_activation[c_idx, f_idx] -= 1
			if sequence_observed_columns.feature_neurons_activation[c_idx, f_idx] == 0:
				pass  # Activation trace expired

	# Update activation traces for connections
	active_indices = sequence_observed_columns.connection_activation.nonzero(as_tuple=False)
	for idx in active_indices:
		c_i = idx[0].item()
		f_i = idx[1].item()
		c_j = idx[2].item()
		f_j = idx[3].item()
		if sequence_observed_columns.connection_activation[c_i, f_i, c_j, f_j] > 0:
			sequence_observed_columns.connection_activation[c_i, f_i, c_j, f_j] -= 1
			if sequence_observed_columns.connection_activation[c_i, f_i, c_j, f_j] == 0:
				pass  # Activation trace expired

def visualize_graph(observed_columns_dict, sequence_observed_columns):
	G.clear()

	# Draw concept columns
	pos_dict = {}
	x_offset = 0
	for lemma, observed_column in observed_columns_dict.items():
		concept_index = observed_column.concept_index
		c_idx = sequence_observed_columns.concept_name_to_index[lemma]

		# Draw feature neurons
		y_offset = 1
		for feature_word, feature_index_in_observed_column in observed_column.feature_word_to_index.items():
			f_idx = sequence_observed_columns.feature_word_to_index[feature_word]
			neuron_color = 'blue' if feature_index_in_observed_column == 0 else 'cyan'
			if lowMem:
				if (f_idx < sequence_observed_columns.fs and
					sequence_observed_columns.feature_neurons_strength[c_idx, f_idx] > 0 and
					sequence_observed_columns.feature_neurons_permanence[c_idx, f_idx] > 0):
					feature_node = f"{lemma}_{feature_word}_{f_idx}"
					G.add_node(feature_node, pos=(x_offset, y_offset), color=neuron_color, label=feature_word)
					y_offset += 1

		# Draw rectangle around the column
		plt.gca().add_patch(plt.Rectangle((x_offset - 0.5, -0.5), 1, max(y_offset, 1) + 0.5, fill=False, edgecolor='black'))

		x_offset += 2  # Adjust x_offset for the next column

	# Draw connections
	for lemma, observed_column in observed_columns_dict.items():
		concept_index = observed_column.concept_index
		c_idx = sequence_observed_columns.concept_name_to_index[lemma]

		# Internal connections (yellow)
		for feature_word, feature_index_in_observed_column in observed_column.feature_word_to_index.items():
			source_node = f"{lemma}_{feature_word}_{sequence_observed_columns.feature_word_to_index[feature_word]}"
			if G.has_node(source_node):
				for other_feature_word, other_feature_index_in_observed_column in observed_column.feature_word_to_index.items():
					target_node = f"{lemma}_{other_feature_word}_{sequence_observed_columns.feature_word_to_index[other_feature_word]}"
					if G.has_node(target_node):
						if feature_word != other_feature_word:
							f_idx = sequence_observed_columns.feature_word_to_index[feature_word]
							other_f_idx = sequence_observed_columns.feature_word_to_index[other_feature_word]
							if (f_idx < sequence_observed_columns.fs and
								other_f_idx < sequence_observed_columns.fs and
								sequence_observed_columns.connection_strength[c_idx, f_idx, c_idx, other_f_idx] > 0 and
								sequence_observed_columns.connection_permanence[c_idx, f_idx, c_idx, other_f_idx] > 0):
								G.add_edge(source_node, target_node, color='yellow')

		# External connections (orange)
		for feature_word, feature_index_in_observed_column in observed_column.feature_word_to_index.items():
			source_node = f"{lemma}_{feature_word}_{sequence_observed_columns.feature_word_to_index[feature_word]}"
			if G.has_node(source_node):
				for other_lemma, other_observed_column in observed_columns_dict.items():
					other_c_idx = sequence_observed_columns.concept_name_to_index[other_lemma]
					if other_c_idx != c_idx:
						for other_feature_word, other_feature_index_in_observed_column in other_observed_column.feature_word_to_index.items():
							target_node = f"{other_lemma}_{other_feature_word}_{sequence_observed_columns.feature_word_to_index[other_feature_word]}"
							if G.has_node(target_node):
								f_idx = sequence_observed_columns.feature_word_to_index[feature_word]
								other_f_idx = sequence_observed_columns.feature_word_to_index[other_feature_word]
								if (f_idx < sequence_observed_columns.fs and
									other_f_idx < sequence_observed_columns.fs and
									sequence_observed_columns.connection_strength[c_idx, f_idx, other_c_idx, other_f_idx] > 0 and
									sequence_observed_columns.connection_permanence[c_idx, f_idx, other_c_idx, other_f_idx] > 0):
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
		torch.save(global_feature_neurons_strength, feature_neurons_strength_file)
		torch.save(global_feature_neurons_permanence, feature_neurons_permanence_file)
		torch.save(global_feature_neurons_activation, feature_neurons_activation_file)

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
