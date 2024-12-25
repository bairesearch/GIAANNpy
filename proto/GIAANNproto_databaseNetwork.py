"""GIAANNproto_databaseNetwork.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_main.py

# Usage:
see GIAANNproto_main.py

# Description:
GIA ANN proto database Network

"""

import torch as pt

from GIAANNproto_globalDefs import *
import GIAANNproto_databaseNetworkFiles
import GIAANNproto_sparseTensors

class DatabaseNetworkClass():
	def __init__(self, c, f, s, p, concept_columns_dict, concept_columns_list, concept_features_dict, concept_features_list, global_feature_neurons):
		self.c = c
		self.f = f
		self.s = s
		self.p = p
		self.concept_columns_dict = concept_columns_dict
		self.concept_columns_list = concept_columns_list
		self.concept_features_dict = concept_features_dict
		self.concept_features_list = concept_features_list
		self.global_feature_neurons = global_feature_neurons
		self.global_feature_connections = None #transformerUseInputConnections: initialised during prediction phase

def backupGlobalArrays(databaseNetworkObject):
	databaseNetworkObject.global_feature_neurons_backup = databaseNetworkObject.global_feature_neurons.clone()
	if(databaseNetworkObject.global_feature_connections is not None):
		databaseNetworkObject.global_feature_connections_backup = databaseNetworkObject.global_feature_connections.clone()
	else:
		databaseNetworkObject.global_feature_connections_backup = None
		
def restoreGlobalArrays(databaseNetworkObject):
	databaseNetworkObject.global_feature_neurons = databaseNetworkObject.global_feature_neurons_backup
	databaseNetworkObject.global_feature_connections = databaseNetworkObject.global_feature_connections_backup

# Initialize global feature neuron arrays if lowMem is disabled
if not lowMem:
	def initialiseFeatureNeuronsGlobal(c, f):
		global_feature_neurons = GIAANNproto_sparseTensors.createEmptySparseTensor((array_number_of_properties, array_number_of_segments, c, f))
		return global_feature_neurons
		
	def loadFeatureNeuronsGlobal(c, f):
		if GIAANNproto_databaseNetworkFiles.pathExists(global_feature_neurons_file+pytorch_tensor_file_extension):
			global_feature_neurons = GIAANNproto_databaseNetworkFiles.loadFeatureNeuronsGlobalFile()
		else:
			global_feature_neurons = initialiseFeatureNeuronsGlobal(c, f)
			#print("initialiseFeatureNeuronsGlobal: global_feature_neurons = ", global_feature_neurons)
		return global_feature_neurons
		
def initialiseDatabaseNetwork():

	concept_columns_dict = {}  # key: lemma, value: index
	concept_columns_list = []  # list of concept column names (lemmas)
	c = 0  # current number of concept columns
	concept_features_dict = {}  # key: word, value: index
	concept_features_list = []  # list of concept feature names (words)
	f = 0  # current number of concept features
	
	# Initialize the concept columns dictionary
	if(GIAANNproto_databaseNetworkFiles.pathExists(concept_columns_dict_file)):
		concept_columns_dict = GIAANNproto_databaseNetworkFiles.loadDictFile(concept_columns_dict_file)
		c = len(concept_columns_dict)
		concept_columns_list = list(concept_columns_dict.keys())
		concept_features_dict = GIAANNproto_databaseNetworkFiles.loadDictFile(concept_features_dict_file)
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

	if not lowMem:
		global_feature_neurons = loadFeatureNeuronsGlobal(c, f)
	else:
		global_feature_neurons = None

	s = array_number_of_segments
	p = array_number_of_properties
		
	databaseNetworkObject = DatabaseNetworkClass(c, f, s, p, concept_columns_dict, concept_columns_list, concept_features_dict, concept_features_list, global_feature_neurons)
	
	return databaseNetworkObject
	



# Define the ObservedColumn class
class ObservedColumn:
	"""
	Create a class defining observed columns. The observed column class contains an index to the dataset concept column dictionary. The observed column class contains a list of feature connection arrays. The observed column class also contains a list of feature neuron arrays when lowMem mode is enabled.
	"""
	def __init__(self, databaseNetworkObject, concept_index, lemma, i):
		self.databaseNetworkObject = databaseNetworkObject
		self.concept_index = concept_index  # Index to the concept columns dictionary
		self.concept_name = lemma
		self.concept_sequence_word_index = i	#not currently used (use SequenceObservedColumns observed_columns_sequence_word_index_dict instead)
		
		if lowMem:
			# If lowMem is enabled, the observed columns contain a list of arrays (pytorch) of f feature neurons, where f is the maximum number of feature neurons per column.
			self.feature_neurons = self.initialiseFeatureNeurons(databaseNetworkObject.f)

		# Map from feature words to indices in feature neurons
		self.feature_word_to_index = {}  # Maps feature words to indices
		self.feature_index_to_word = {}  # Maps indices to feature words
		if(useDedicatedConceptNames):
			self.next_feature_index = 1  # Start from 1 since index 0 is reserved for concept neuron
			if(useDedicatedConceptNames2):
				self.feature_word_to_index[variableConceptNeuronFeatureName] = feature_index_concept_neuron
				self.feature_index_to_word[feature_index_concept_neuron] = variableConceptNeuronFeatureName
			
		# Store all connections for each source column in a list of integer feature connection arrays, each of size f * c * f, where c is the length of the dictionary of columns, and f is the maximum number of feature neurons.
		self.feature_connections = self.initialiseFeatureConnections(databaseNetworkObject.c, databaseNetworkObject.f) 

		self.next_feature_index = 0
		for feature_index in range(1, databaseNetworkObject.f, 1):
			feature_word = databaseNetworkObject.concept_features_list[feature_index]
			self.feature_word_to_index[feature_word] = feature_index
			self.feature_index_to_word[feature_index] = feature_word
			self.next_feature_index += 1
					
	@staticmethod
	def initialiseFeatureNeurons(f):
		feature_neurons = GIAANNproto_sparseTensors.createEmptySparseTensor((array_number_of_properties, array_number_of_segments, f))
		return feature_neurons

	@staticmethod
	def initialiseFeatureConnections(c, f):
		feature_connections = GIAANNproto_sparseTensors.createEmptySparseTensor((array_number_of_properties, array_number_of_segments, f, c, f))
		return feature_connections
	
	def resize_concept_arrays(self, new_c):
		load_c = self.feature_connections.shape[3]
		if new_c > load_c:
			expanded_size = (self.feature_connections.shape[0], self.feature_connections.shape[1], self.feature_connections.shape[2], new_c, self.feature_connections.shape[4])
			self.feature_connections = pt.sparse_coo_tensor(self.feature_connections.indices(), self.feature_connections.values(), size=expanded_size, dtype=array_type, device=deviceSparse)
		
	def expand_feature_arrays(self, new_f):
		load_f = self.feature_connections.shape[2]  # or self.feature_connections.shape[4]
		if new_f > load_f:
			# Expand feature_connections along dimensions 2 and 4
			self.feature_connections = self.feature_connections.coalesce()
			expanded_size_connections = (self.feature_connections.shape[0], self.feature_connections.shape[1], new_f, self.feature_connections.shape[3], new_f)
			self.feature_connections = pt.sparse_coo_tensor(self.feature_connections.indices(), self.feature_connections.values(), size=expanded_size_connections, dtype=array_type, device=deviceSparse)
	
			if lowMem:
				expanded_size_neurons = (self.feature_neurons.shape[0], self.feature_neurons.shape[1], new_f)
				self.feature_neurons = self.feature_neurons.coalesce()
				self.feature_neurons = pt.sparse_coo_tensor(self.feature_neurons.indices(), self.feature_neurons.values(), size=expanded_size_neurons, dtype=array_type, device=deviceSparse)

			for feature_index in range(load_f, new_f):
				feature_word = self.databaseNetworkObject.concept_features_list[feature_index]
				self.feature_word_to_index[feature_word] = feature_index
				self.feature_index_to_word[feature_index] = feature_word
				self.next_feature_index += 1

	def save_to_disk(self):
		GIAANNproto_databaseNetworkFiles.observed_column_save_to_disk(self)

	@classmethod
	def load_from_disk(cls, databaseNetworkObject, concept_index, lemma, i):
		return GIAANNproto_databaseNetworkFiles.observed_column_load_from_disk(cls, databaseNetworkObject, concept_index, lemma, i)
		

def addConceptToConceptColumnsDict(databaseNetworkObject, lemma, concepts_found, new_concepts_added):
	concepts_found = True
	if lemma not in databaseNetworkObject.concept_columns_dict:
		# Add to concept columns dictionary
		#print("adding concept = ", lemma)
		databaseNetworkObject.concept_columns_dict[lemma] = databaseNetworkObject.c
		databaseNetworkObject.concept_columns_list.append(lemma)
		databaseNetworkObject.c += 1
		new_concepts_added = True
	return concepts_found, new_concepts_added
	
def load_or_create_observed_column(databaseNetworkObject, concept_index, lemma, i):
	observed_column_file = observed_columns_dir + '/' + f"{concept_index}_data.pkl"
	if GIAANNproto_databaseNetworkFiles.pathExists(observed_column_file):
		observed_column = ObservedColumn.load_from_disk(databaseNetworkObject, concept_index, lemma, i)
		# Resize connection arrays if c has increased
		observed_column.resize_concept_arrays(databaseNetworkObject.c)
		# Also expand feature arrays if f has increased
		observed_column.expand_feature_arrays(databaseNetworkObject.f)
	else:
		observed_column = ObservedColumn(databaseNetworkObject, concept_index, lemma, i)
		# Initialize connection arrays with correct size
		observed_column.resize_concept_arrays(databaseNetworkObject.c)
		observed_column.expand_feature_arrays(databaseNetworkObject.f)
	return observed_column

def generate_global_feature_connections(databaseNetworkObject):
	concept_columns_list = []
	for i, (lemma, concept_index) in enumerate(databaseNetworkObject.concept_columns_dict.items()):
		concept_column = load_or_create_observed_column(databaseNetworkObject, concept_index, lemma, i)
		concept_columns_list.append(concept_column)
	global_feature_connections_list = []
	for concept_column in concept_columns_list:
		global_feature_connections_list.append(concept_column.feature_connections)
	databaseNetworkObject.global_feature_connections = pt.stack(global_feature_connections_list, dim=2)
	print("generate_global_feature_connections: databaseNetworkObject.global_feature_connections.shape = ", databaseNetworkObject.global_feature_connections.shape)

def load_all_columns(databaseNetworkObject):
	observed_columns_dict = {}
	for i, (lemma, concept_index) in enumerate(databaseNetworkObject.concept_columns_dict.items()):
		concept_column = load_or_create_observed_column(databaseNetworkObject, concept_index, lemma, i)
		observed_columns_dict[lemma] = concept_column
	return observed_columns_dict

'''
def getTokenConceptFeatureIndexForSequenceConceptIndex(sequence_observed_columns, words_doc, concept_mask, sequence_concept_index, sequenceWordIndex):
	concept_index = sequence_observed_columns.sequence_observed_columns_dict[sequence_concept_index].concept_index
	if(concept_mask[sequenceWordIndex]):
		feature_index = feature_index_concept_neuron
	else:
		feature_index = sequence_observed_columns.feature_word_to_index[words_doc[sequenceWordIndex]]
	return concept_index, feature_index
'''

def getTokenConceptFeatureIndexTensor(sequence_observed_columns, words_doc, lemmas_doc, concept_mask, sequenceWordIndex, kcMax):
	foundNextColumnIndex, previousColumnIndex, nextColumnIndex, targetFeatureIndex = getTokenConceptFeatureIndex(sequence_observed_columns, words_doc, lemmas_doc, concept_mask, sequenceWordIndex)

	if(kcMax == 1 or not foundNextColumnIndex):
		concept_columns_indices = pt.tensor(previousColumnIndex).unsqueeze(0)
		concept_columns_feature_indices = pt.tensor(targetFeatureIndex).unsqueeze(0).unsqueeze(0)
		multiple_sources = False
	elif(kcMax == 2 and foundNextColumnIndex): 
		concept_columns_indices = pt.tensor([previousColumnIndex, nextColumnIndex])
		concept_columns_feature_indices = pt.stack([pt.tensor(targetFeatureIndex).unsqueeze(0), pt.tensor(targetFeatureIndex).unsqueeze(0)], dim=0)
		multiple_sources = True
	else:
		printe("getTokenConceptFeatureIndexTensor currently requires kcMax == 1 or 2; corresponding to the number of target columns per token; check debugConceptFeaturesOccurFirstInSubsequence/multipleTargets")

	return multiple_sources, previousColumnIndex, nextColumnIndex, targetFeatureIndex, concept_columns_indices, concept_columns_feature_indices

def getTokenConceptFeatureIndex(sequence_observed_columns, words_doc, lemmas_doc, concept_mask, sequenceWordIndex):
	databaseNetworkObject = sequence_observed_columns.databaseNetworkObject
	columns_index_sequence_word_index_dict = sequence_observed_columns.columns_index_sequence_word_index_dict
	
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
	
	return foundNextColumnIndex, previousColumnIndex, nextColumnIndex, targetFeatureIndex
