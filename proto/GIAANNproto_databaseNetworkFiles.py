"""GIAANNproto_databaseNetworkFiles.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_main.py

# Usage:
see GIAANNproto_main.py

# Description:
GIA ANN proto database Network Files

"""

import torch as pt
import pickle
import os

from GIAANNproto_globalDefs import *


def initialiseDatabaseFiles():
	os.makedirs(observed_columns_dir, exist_ok=True)

def pathExists(pathName):
	if(os.path.exists(pathName)):
		return True
	else:
		return False
	
def loadDictFile(dictFileName):
	with open(dictFileName, 'rb') as f_in:
		dictionary = pickle.load(f_in)
	return dictionary

def loadFeatureNeuronsGlobalFile():
	global_feature_neurons = load_tensor_list(databaseFolder, global_feature_neurons_file)
	return global_feature_neurons
	
def save_data(databaseNetworkObject, observed_columns_dict):
	# Save observed columns to disk
	for observed_column in observed_columns_dict.values():
		observed_column.save_to_disk()

	# Save global feature neuron arrays if not lowMem
	if not lowMem:
		if(performRedundantCoalesce):
			databaseNetworkObject.global_feature_neurons = databaseNetworkObject.global_feature_neurons.coalesce()
		save_tensor_list(databaseNetworkObject.global_feature_neurons, databaseFolder, global_feature_neurons_file)

	# Save concept columns dictionary to disk
	with open(concept_columns_dict_file, 'wb') as f_out:
		pickle.dump(databaseNetworkObject.concept_columns_dict, f_out)

	# Save concept features dictionary to disk
	with open(concept_features_dict_file, 'wb') as f_out:
		pickle.dump(databaseNetworkObject.concept_features_dict, f_out)


def observed_column_save_to_disk(self):
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
	# Save the tensors using pt.save
	if(performRedundantCoalesce):
		self.feature_connections = self.feature_connections.coalesce()
		print("self.feature_connections = ", self.feature_connections)
	save_tensor_list(self.feature_connections, observed_columns_dir, f"{self.concept_index}_feature_connections")
	if lowMem:
		if(performRedundantCoalesce):
			self.feature_neurons = self.feature_neurons.coalesce()
			print("self.feature_neurons = ", self.feature_neurons)
		save_tensor_list(self.feature_neurons, observed_columns_dir, f"{self.concept_index}_feature_neurons")

def observed_column_load_from_disk(cls, databaseNetworkObject, concept_index, lemma, i):
	"""
	Load the observed column data from disk.
	"""
	# Load the data dictionary
	with open(os.path.join(observed_columns_dir, f"{concept_index}_data.pkl"), 'rb') as f:
		data = pickle.load(f)
	instance = cls(databaseNetworkObject, concept_index, lemma, i)
	instance.feature_word_to_index = data['feature_word_to_index']
	instance.feature_index_to_word = data['feature_index_to_word']
	instance.next_feature_index = data['next_feature_index']
	# Load the tensors
	instance.feature_connections = load_tensor_list(observed_columns_dir, f"{concept_index}_feature_connections")
	if lowMem:
		instance.feature_neurons = load_tensor_list(observed_columns_dir, f"{concept_index}_feature_neurons")
	return instance

def save_tensor_list(tensor_list, folder_name, file_name):
	for i in range(array_number_of_properties):
		tensor = tensor_list[i]
		array_properties_string = "_property" + str(i)
		pt.save(tensor, os.path.join(folder_name, file_name+array_properties_string+pytorch_tensor_file_extension))

def load_tensor_list(folder_name, file_name):
	tensor_list = []
	for i in range(array_number_of_properties):
		array_properties_string = "_property" + str(i)
		tensor = pt.load(os.path.join(folder_name, file_name+array_properties_string+pytorch_tensor_file_extension))	#does not work: , map_location=deviceSparse
		tensor = tensor.to(deviceSparse)
		tensor_list.append(tensor)
	return tensor_list
