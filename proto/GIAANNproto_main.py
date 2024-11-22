"""GIAANNproto_main.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:

conda create -n pytorchsenv
source activate pytorchsenv
conda install python=3.12
pip install networkx
pip install matplotlib
pip install yattag
pip install torch
pip install torch_geometric
pip install nltk spacy
pip install datasets
python3 -m spacy download en_core_web_sm
pip install benepar

# Usage:
source activate pytorchsenv
python GIAANNproto_main.py

# Description:
GIA ANN proto main

"""

# Import necessary libraries
import torch as pt
import spacy
from datasets import load_dataset
pt.set_printoptions(threshold=float('inf'))

from GIAANNproto_globalDefs import *
import GIAANNproto_sparseTensors
import GIAANNproto_databaseNetwork
import GIAANNproto_databaseNetworkFiles
import GIAANNproto_databaseNetworkDraw
import GIAANNproto_databaseNetworkTrain
if(useInference):
	import GIAANNproto_predictiveNetwork


# Initialize spaCy model
nlp = spacy.load('en_core_web_sm')

databaseNetworkObject = GIAANNproto_databaseNetwork.initialiseDatabaseNetwork()
databaseNetworkObject.nlp = nlp	#used by pos_string_to_pos_int

def main():
	GIAANNproto_databaseNetworkFiles.initialiseDatabaseFiles()
	# Start processing the dataset
	if(useInference or debugSmallDataset):
		process_prompt()
	else:
		process_dataset(dataset)

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
	global sentence_count
	
	if(debugReloadGlobalFeatureNeuronsEverySentence):
		initialiseDatabaseNetwork()
		if(not lowMem):
			databaseNetworkObject.global_feature_neurons = GIAANNproto_databaseNetwork.initialiseFeatureNeuronsGlobal(databaseNetworkObject.c, databaseNetworkObject.f)

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
			detect_new_features(databaseNetworkObject, words, lemmas, pos_tags)

		# Second pass: Create observed_columns_dict
		observed_columns_dict, observed_columns_sequence_word_index_dict = second_pass(databaseNetworkObject, lemmas, pos_tags)

		# Create the sequence observed columns object
		sequence_observed_columns = GIAANNproto_databaseNetworkTrain.SequenceObservedColumns(databaseNetworkObject, words, lemmas, observed_columns_dict, observed_columns_sequence_word_index_dict)

		if(lastSentenceInPrompt):
			# Process each concept word in the sequence (predict)
			GIAANNproto_predictiveNetwork.process_concept_words_inference(sequence_observed_columns, sentence_count, doc, doc_seed, doc_predict, num_seed_tokens, num_prediction_tokens)
		else:
			# Process each concept word in the sequence (train)
			GIAANNproto_databaseNetworkTrain.process_concept_words(sequence_observed_columns, sentence_count, doc, words, lemmas, pos_tags)

			# Update observed columns from sequence observed columns
			sequence_observed_columns.update_observed_columns_wrapper()

			if(drawNetworkDuringTrain):
				# Visualize the complete graph every time a new sentence is parsed by the application.
				GIAANNproto_databaseNetworkDraw.visualize_graph(sequence_observed_columns)

			# Save observed columns to disk
			if(useSaveData):
				GIAANNproto_databaseNetworkFiles.save_data(databaseNetworkObject, observed_columns_dict)
			
			'''
			if(useActivationDecrement):
				#decrement activation after each train interval; not currently used
				databaseNetworkObject.global_feature_neurons[array_index_properties_activation, array_index_segment_first] -= activationDecrementPerPredictedSentence
				databaseNetworkObject.global_feature_neurons[array_index_properties_activation, array_index_segment_first] = pt.clamp(databaseNetworkObject.global_feature_neurons[array_index_properties_activation, array_index_segment_first], min=0)
			'''
			
	# Break if we've reached the maximum number of sentences
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
				concepts_found, new_concepts_added = GIAANNproto_databaseNetwork.addConceptToConceptColumnsDict(databaseNetworkObject, lemma, concepts_found, new_concepts_added)
		else:
			# When usePOS is disabled, assign concept columns for every new lemma encountered
			concepts_found, new_concepts_added = GIAANNproto_databaseNetwork.addConceptToConceptColumnsDict(databaseNetworkObject, lemma, concepts_found, new_concepts_added)

		words.append(word)
		lemmas.append(lemma)
		pos_tags.append(pos)

	# If new concept columns have been added, expand arrays as needed
	if new_concepts_added:
		if not lowMem:
			# Expand global feature neuron arrays
			if databaseNetworkObject.global_feature_neurons.shape[2] < databaseNetworkObject.c:
				new_shape = (databaseNetworkObject.global_feature_neurons.shape[0], databaseNetworkObject.global_feature_neurons.shape[1], databaseNetworkObject.c, databaseNetworkObject.global_feature_neurons.shape[3])
				if(performRedundantCoalesce):
					databaseNetworkObject.global_feature_neurons = databaseNetworkObject.global_feature_neurons.coalesce()
				databaseNetworkObject.global_feature_neurons = pt.sparse_coo_tensor(databaseNetworkObject.global_feature_neurons._indices(), databaseNetworkObject.global_feature_neurons._values(), size=new_shape, dtype=array_type)
				
	return concepts_found, words, lemmas, pos_tags

				
def second_pass(databaseNetworkObject, lemmas, pos_tags):
	observed_columns_dict = {}
	observed_columns_sequence_word_index_dict = {}
	for i, lemma in enumerate(lemmas):
		pos = pos_tags[i]
		if usePOS:
			if pos in noun_pos_tags:
				concept_index = databaseNetworkObject.concept_columns_dict[lemma]
				# Load observed column from disk or create new one
				observed_column = GIAANNproto_databaseNetwork.load_or_create_observed_column(databaseNetworkObject, concept_index, lemma, i)
				observed_columns_dict[lemma] = observed_column
				observed_columns_sequence_word_index_dict[i] = observed_column
		else:
			concept_index = databaseNetworkObject.concept_columns_dict[lemma]
			# Load observed column from disk or create new one
			observed_column = GIAANNproto_databaseNetwork.load_or_create_observed_column(databaseNetworkObject, concept_index, lemma, i)
			observed_columns_dict[lemma] = observed_column
			observed_columns_sequence_word_index_dict[i] = observed_column
	return observed_columns_dict, observed_columns_sequence_word_index_dict


def detect_new_features(databaseNetworkObject, words, lemmas, pos_tags):
	"""
	When usePOS mode is enabled, detect all possible new features in the sequence
	by searching for all new non-nouns in the sequence.
	"""

	num_new_features = 0
	for j, (word_j, pos_j) in enumerate(zip(words, pos_tags)):
		if(process_feature_detection(databaseNetworkObject, j, word_j, pos_tags)):
			num_new_features += 1

	# After processing all features, update f
	databaseNetworkObject.f += num_new_features

	# Now, expand arrays accordingly
	if not lowMem:
		if databaseNetworkObject.f > databaseNetworkObject.global_feature_neurons.shape[3]:
			extra_cols = databaseNetworkObject.f - databaseNetworkObject.global_feature_neurons.shape[3]
			new_shape = (databaseNetworkObject.global_feature_neurons.shape[0], databaseNetworkObject.global_feature_neurons.shape[1], databaseNetworkObject.global_feature_neurons.shape[2], databaseNetworkObject.f)
			databaseNetworkObject.global_feature_neurons = databaseNetworkObject.global_feature_neurons.coalesce()
			databaseNetworkObject.global_feature_neurons = pt.sparse_coo_tensor(databaseNetworkObject.global_feature_neurons.indices(), databaseNetworkObject.global_feature_neurons.values(), size=new_shape, dtype=array_type)

def process_feature_detection(databaseNetworkObject, j, word_j, pos_tags):
	"""
	Helper function to detect new features prior to processing concept words.
	"""
	
	pos_j = pos_tags[j]
	feature_word = word_j.lower()
	
	if usePOS:
		if pos_j in noun_pos_tags:
			return False  # Skip nouns as features

	if feature_word not in databaseNetworkObject.concept_features_dict:
		databaseNetworkObject.concept_features_dict[feature_word] = len(databaseNetworkObject.concept_features_dict)
		databaseNetworkObject.concept_features_list.append(feature_word)
		return True
	else:
		return False
	


# Load the Wikipedia dataset using Hugging Face datasets
dataset = load_dataset('wikipedia', '20220301.en', split='train', streaming=True)

if __name__ == "__main__":
	main()
	
