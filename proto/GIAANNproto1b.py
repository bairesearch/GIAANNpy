# Import necessary libraries
import torch
import networkx as nx
import matplotlib.pyplot as plt
import nltk
import spacy
from datasets import load_dataset
import os
import pickle

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize

# Set boolean variables as per specification
useInference = False  # Disable useInference mode
lowMem = True         # Enable lowMem mode (can only be used when useInference is disabled)
usePOS = True         # Enable usePOS mode

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
    c = len(concept_features_dict)
    concept_features_list = list(concept_features_dict.keys())
else:
    concept_columns_dict = {}  # key: lemma, value: index
    concept_columns_list = []  # list of concept column names (lemmas)
    c = 0  # current number of concept columns
    concept_features_dict = {}  # key: lemma, value: index
    concept_features_list = []  # list of concept column names (lemmas)
    f = 0  # current number of concept features
    
    # Set the size of the feature arrays (f)

    #add dummy feature for concept neuron (different per concept column)
    feature_index_concept_neuron = 0
    variableConceptNeuronFeatureName = "variableConceptNeuronFeature"
    concept_features_list.append(variableConceptNeuronFeatureName)
    concept_features_dict[variableConceptNeuronFeatureName] = len(concept_features_dict)
    f = 1 # Will be updated dynamically based on c

    if usePOS and not lowMem:
        print("error: usePOS and not lowMem case not yet coded - need to set f and populate concept_features_list/concept_features_dict etc")
        exit()
        #f = max_num_non_nouns + 1  # Maximum number of non-nouns in an English dictionary, plus the concept neuron of each column
  
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
non_noun_pos_tags = {'ADJ', 'ADV', 'VERB', 'ADP', 'AUX', 'CCONJ', 'DET', 'INTJ',
                     'NUM', 'PART', 'PRON', 'SCONJ', 'SYM', 'X'}

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
            #print("create class ObservedColumn: adding: feature_word = ", feature_word)

    def resize_concept_arrays(self, new_c):
        load_c = self.connection_strength.shape[1]
        if new_c > load_c:
            extra_cols = new_c - load_c
            # Expand along dimension 1 (columns)
            self.connection_strength = torch.cat([self.connection_strength, torch.zeros(self.connection_strength.shape[0], extra_cols, self.connection_strength.shape[2], dtype=torch.int32)], dim=1)
            self.connection_permanence = torch.cat([self.connection_permanence, torch.full((self.connection_permanence.shape[0], extra_cols, self.connection_permanence.shape[2]), z1, dtype=torch.int32)], dim=1)
            self.connection_activation = torch.cat([self.connection_activation, torch.zeros(self.connection_activation.shape[0], extra_cols, self.connection_activation.shape[2], dtype=torch.int32)], dim=1)

    def expand_feature_arrays(self, new_f):
        load_f = self.connection_strength.shape[0]    #or self.connection_strength.shape[2]       
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
                #print("expand_feature_arrays, adding: feature_word = ", feature_word)

            #ALREADYDONE: Expand global feature neuron arrays
            
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

    # Process each concept word in the sequence
    process_concept_words(doc, lemmas, pos_tags, observed_columns_dict)

    # Update activation traces for feature neurons and connections
    update_activation(observed_columns_dict)

    # Visualize the complete graph every time a new sentence is parsed by the application.
    visualize_graph(observed_columns_dict)

    # Save observed columns to disk
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
        #print("process_feature_detection, adding: feature_word = ", feature_word)
        return True
    else:
        return False
    
def process_concept_words(doc, lemmas, pos_tags, observed_columns_dict):
    """
    For every concept word (lemma) i in the sequence, identify every feature neuron in that column
    that occurs q words before or after the concept word in the sequence, including the concept neuron.
    If usePOS is disabled, set q to 5. If usePOS is enabled, set q to the distance to the previous/next noun
    (depending on whether the feature selected is before or after the current concept word in the sequence).
    Always ensure the feature neuron selected is not out of bounds of the sequence.
    """
    global c, f, lowMem, global_feature_neurons_strength, global_feature_neurons_permanence, global_feature_neurons_activation
    if usePOS:
        # Precompute noun indices
        noun_indices = [index for index, pos in enumerate(pos_tags) if pos in noun_pos_tags]
    else:
        q = 5

    for i, token in enumerate(doc):
        lemma_i = lemmas[i]
        pos_i = pos_tags[i]

        if lemma_i in observed_columns_dict:
            observed_column = observed_columns_dict[lemma_i]
            concept_index_i = observed_column.concept_index

            # Set to track feature neurons activated in this sequence for this concept
            activated_feature_indices = set()

            if usePOS:
                # Compute distances to previous and next nouns
                # Get index of previous noun before i
                prev_noun_index = None
                for idx in range(i - 1, -1, -1):
                    if pos_tags[idx] in noun_pos_tags:
                        prev_noun_index = idx
                        break
                dist_to_prev_noun = i - prev_noun_index - 1 if prev_noun_index is not None else i

                # Get index of next noun after i
                next_noun_index = None
                for idx in range(i + 1, len(doc)):
                    if pos_tags[idx] in noun_pos_tags:
                        next_noun_index = idx
                        break
                dist_to_next_noun = next_noun_index - i - 1 if next_noun_index is not None else len(doc) - i - 1
            else:
                q = 5

            # Process positions before i
            if usePOS:
                start = max(0, i - dist_to_prev_noun)
            else:
                start = max(0, i - q)

            for j in range(start, i):
                process_feature(observed_column, i, j, doc, lemmas, pos_tags, activated_feature_indices, observed_columns_dict)

            # Process positions including and after i
            if usePOS:
                end = min(len(doc), i + dist_to_next_noun + 1)
            else:
                end = min(len(doc), i + q + 1)

            for j in range(i, end):    #start at the concept neuron i in the feature array
                process_feature(observed_column, i, j, doc, lemmas, pos_tags, activated_feature_indices, observed_columns_dict)

            # Increment the strength of the concept neuron
            if lowMem:
                observed_column.feature_neurons_strength[feature_index_concept_neuron] += 1
                # Increase permanence exponentially
                observed_column.feature_neurons_permanence[feature_index_concept_neuron] = observed_column.feature_neurons_permanence[feature_index_concept_neuron] ** 2
                # Set activation trace to j1 sequences
                observed_column.feature_neurons_activation[feature_index_concept_neuron] = j1  # Overwrite with j1
            else:
                global_feature_neurons_strength[concept_index_i, feature_index_concept_neuron] += 1
                # Increase permanence exponentially
                global_feature_neurons_permanence[concept_index_i, feature_index_concept_neuron] = global_feature_neurons_permanence[concept_index_i, feature_index_concept_neuron] ** 2
                # Set activation trace to j1 sequences
                global_feature_neurons_activation[concept_index_i, feature_index_concept_neuron] = j1
            
            decrease_permanence(observed_column, concept_index_i, activated_feature_indices, observed_columns_dict)

def decrease_permanence(observed_column, concept_index_i, activated_feature_indices, observed_columns_dict):
    # Decrease permanence for feature neurons not activated
    all_feature_indices = set(observed_column.feature_word_to_index.values())
    inactive_feature_indices = all_feature_indices - activated_feature_indices
    for feature_index in inactive_feature_indices:
        # Decrease permanence linearly
        if lowMem:
            observed_column.feature_neurons_permanence[feature_index] -= z2
            # Remove feature neuron if permanence <= 0
            if observed_column.feature_neurons_permanence[feature_index] <= 0:
                # Set permanence and activation to zero
                observed_column.feature_neurons_permanence[feature_index] = 0
                observed_column.feature_neurons_activation[feature_index] = 0
        else:
            global_feature_neurons_permanence[concept_index_i, feature_index] -= z2
            # Remove feature neuron if permanence <= 0
            if global_feature_neurons_permanence[concept_index_i, feature_index] <= 0:
                # Set permanence and activation to zero
                global_feature_neurons_permanence[concept_index_i, feature_index] = 0
                global_feature_neurons_activation[concept_index_i, feature_index] = 0
    '''
    # Decrease permanence of connections from inactive feature neurons in column
    for feature_index in inactive_feature_indices:
        #Inefficient; Decrease permanence of inactive connections between all columns:
        #for other_concept_index in range(c):
        #   other_observed_column = load_or_create_observed_column(other_concept_index)
        #Decrease permanence of inactive connections between observed columns:
        for other_lemma, other_observed_column in observed_columns_dict.items():
            other_concept_index = other_observed_column.concept_index
            all_other_feature_indices = set(other_observed_column.feature_word_to_index.values())
            for other_feature_index in all_other_feature_indices:
                if observed_column.connection_permanence[feature_index, other_concept_index, other_feature_index] > 0:
                    observed_column.connection_permanence[feature_index, other_concept_index, other_feature_index] -= z2
                    # Remove connection if permanence <= 0
                    if observed_column.connection_permanence[feature_index, other_concept_index, other_feature_index] <= 0:
                        observed_column.connection_permanence[feature_index, other_concept_index, other_feature_index] = 0
                        observed_column.connection_activation[feature_index, other_concept_index, other_feature_index] = 0
    '''
    # Decrease permanence of inactive connections for activated features in column 
    for feature_index in activated_feature_indices:
        #Inefficient; Decrease permanence of inactive connections between all columns:
        #for other_concept_index in range(c):
        #   other_observed_column = load_or_create_observed_column(other_concept_index)
        #Decrease permanence of inactive connections between observed columns:
        for other_lemma, other_observed_column in observed_columns_dict.items():
            other_concept_index = other_observed_column.concept_index
            for other_feature_index in inactive_feature_indices:
                if observed_column.connection_permanence[feature_index, other_concept_index, other_feature_index] > 0:
                    observed_column.connection_permanence[feature_index, other_concept_index, other_feature_index] -= z2
                    # Remove connection if permanence <= 0
                    if observed_column.connection_permanence[feature_index, other_concept_index, other_feature_index] <= 0:
                        observed_column.connection_permanence[feature_index, other_concept_index, other_feature_index] = 0
                        observed_column.connection_activation[feature_index, other_concept_index, other_feature_index] = 0

                   
def process_feature(observed_column, i, j, doc, lemmas, pos_tags, activated_feature_indices, observed_columns_dict):
    """
    Helper function to process a feature at position j for the concept at position i.
    """
    global c, f, lowMem, global_feature_neurons_strength, global_feature_neurons_permanence, global_feature_neurons_activation
    lemma_i = lemmas[i]
    lemma_j = lemmas[j]
    pos_j = pos_tags[j]
    token_j = doc[j]
    word_j = token_j.text

    if(i == j):
        #concept neuron
        feature_word = lemma_j    #.lower() has already been executed
    else:
        #feature neuron
        # Assign feature neurons to words not lemmas
        feature_word = word_j.lower()
        # Skip nouns as features when usePOS is enabled
        if usePOS and pos_j in noun_pos_tags:
            return

    # Get feature neuron index for feature_word in this column
    if feature_word not in observed_column.feature_word_to_index:
        print("process_feature error: feature_word not in observed_column.feature_word_to_index: Should not happen as features are pre-detected: feature_word = ", feature_word)
        return
    else:
        feature_index = observed_column.feature_word_to_index[feature_word]

    # Add feature index to activated set
    activated_feature_indices.add(feature_index)

    # Increment the strength of the feature neuron
    if lowMem:
        observed_column.feature_neurons_strength[feature_index] += 1
        # Increase permanence exponentially
        observed_column.feature_neurons_permanence[feature_index] = observed_column.feature_neurons_permanence[feature_index] ** 2
        # Set activation trace to j1 sequences
        observed_column.feature_neurons_activation[feature_index] = j1  # Overwrite with j1
    else:
        concept_index_i = observed_column.concept_index
        global_feature_neurons_strength[concept_index_i, feature_index] += 1
        # Increase permanence exponentially
        global_feature_neurons_permanence[concept_index_i, feature_index] = global_feature_neurons_permanence[concept_index_i, feature_index] ** 2
        # Set activation trace to j1 sequences
        global_feature_neurons_activation[concept_index_i, feature_index] = j1

    # Create connections
    # Connect these feature neurons to every other identified feature neuron (observed in the current sequence) in every other concept column in the sequence
    for other_lemma, other_observed_column in observed_columns_dict.items():
        other_concept_index = other_observed_column.concept_index
        for other_feature_word, other_feature_index in other_observed_column.feature_word_to_index.items():
            # Only connect column features to future or present (not past) column features
            if other_concept_index >= observed_column.concept_index:
                if not(other_concept_index == observed_column.concept_index and other_feature_index == feature_index):
                    #print("creating connection; feature_index = ", feature_index, ", other_concept_index = ", other_concept_index, ", other_feature_index = ", other_feature_index)
                    # Update the connection arrays
                    observed_column.connection_strength[feature_index, other_concept_index, other_feature_index] += 1
                    # Increase permanence exponentially
                    observed_column.connection_permanence[feature_index, other_concept_index, other_feature_index] = observed_column.connection_permanence[feature_index, other_concept_index, other_feature_index] ** 2
                    # Set activation trace to j1 sequences
                    observed_column.connection_activation[feature_index, other_concept_index, other_feature_index] = j1

def update_activation(observed_columns_dict):
    # For each observed column, update activation traces
    for observed_column in observed_columns_dict.values():
        # Feature neurons
        if lowMem:
            active_indices = observed_column.feature_neurons_activation.nonzero(as_tuple=True)[0]
            for idx in active_indices:
                idx = idx.item()  # Convert tensor to integer
                if observed_column.feature_neurons_activation[idx] > 0:
                    observed_column.feature_neurons_activation[idx] -= 1
                    if observed_column.feature_neurons_activation[idx] == 0:
                        # Activation trace expired
                        pass  # Do nothing for now

        # Connections
        active_indices = observed_column.connection_activation.nonzero(as_tuple=False)
        for idx in active_indices:
            i = idx[0].item()
            j = idx[1].item()
            k = idx[2].item()
            if observed_column.connection_activation[i, j, k] > 0:
                observed_column.connection_activation[i, j, k] -= 1
                if observed_column.connection_activation[i, j, k] == 0:
                    # Activation trace expired
                    pass  # Do nothing for now

def visualize_graph(observed_columns_dict):
    G.clear()

    # Draw concept columns
    pos_dict = {}
    x_offset = 0
    for lemma, observed_column in observed_columns_dict.items():
        concept_index = observed_column.concept_index

        # Draw feature neurons
        y_offset = 1
        for feature_index, feature_word in observed_column.feature_index_to_word.items():
            if feature_index == 0:
                neuron_color = 'blue'
            else:
                neuron_color = 'cyan'
            if lowMem:
                # Only visualize feature neurons with strength > 0 and permanence > 0
                if (feature_index < f and
                    observed_column.feature_neurons_strength[feature_index] > 0 and
                    observed_column.feature_neurons_permanence[feature_index] > 0):
                    feature_node = f"{lemma}_{feature_word}_{feature_index}"
                    G.add_node(feature_node, pos=(x_offset, y_offset), color=neuron_color, label=feature_word)
                    y_offset += 1
            else:
                if (feature_index < f and
                    global_feature_neurons_strength[concept_index, feature_index] > 0 and
                    global_feature_neurons_permanence[concept_index, feature_index] > 0):
                    feature_node = f"{lemma}_{feature_word}_{feature_index}"
                    G.add_node(feature_node, pos=(x_offset, y_offset), color=neuron_color, label=feature_word)
                    y_offset += 1

        # Draw rectangle around the column
        plt.gca().add_patch(plt.Rectangle((x_offset - 0.5, -0.5), 1, max(y_offset, 1) + 0.5, fill=False, edgecolor='black'))

        x_offset += 2  # Adjust x_offset for the next column

    # Draw connections
    for lemma, observed_column in observed_columns_dict.items():
        concept_index = observed_column.concept_index

        # Internal connections (yellow)
        for feature_index, feature_word in observed_column.feature_index_to_word.items():
            source_node = f"{lemma}_{feature_word}_{feature_index}"
            if G.has_node(source_node):
                for other_feature_index, other_feature_word in observed_column.feature_index_to_word.items():
                    target_node = f"{lemma}_{other_feature_word}_{other_feature_index}"
                    if G.has_node(target_node):
                        if feature_index != other_feature_index:
                            # Only visualize connections with strength > 0 and permanence > 0
                            if (feature_index < f and
                                other_feature_index < f and
                                observed_column.connection_strength[feature_index, concept_index, other_feature_index] > 0 and
                                observed_column.connection_permanence[feature_index, concept_index, other_feature_index] > 0):
                                G.add_edge(source_node, target_node, color='yellow')

        # External connections (orange)
        for feature_index, feature_word in observed_column.feature_index_to_word.items():
            source_node = f"{lemma}_{feature_word}_{feature_index}"
            if G.has_node(source_node):
                for other_concept_index in range(c):
                    if other_concept_index != concept_index:
                        other_lemma = concept_columns_list[other_concept_index]
                        other_observed_column = observed_columns_dict.get(other_lemma)
                        if other_observed_column is not None:
                            for other_feature_index, other_feature_word in other_observed_column.feature_index_to_word.items():
                                target_node = f"{other_lemma}_{other_feature_word}_{other_feature_index}"
                                if G.has_node(target_node):
                                    # Only visualize connections with strength > 0 and permanence > 0
                                    if (feature_index < f and
                                        other_feature_index < f and
                                        observed_column.connection_strength[feature_index, other_concept_index, other_feature_index] > 0 and
                                        observed_column.connection_permanence[feature_index, other_concept_index, other_feature_index] > 0):
                                        G.add_edge(source_node, target_node, color='orange')

    # Get positions and colors for drawing
    pos = nx.get_node_attributes(G, 'pos')
    colors = [data['color'] for node, data in G.nodes(data=True)]
    edge_colors = [data['color'] for u, v, data in G.edges(data=True)]
    labels = nx.get_node_attributes(G, 'label')

    # Draw the graph
    #plt.figure(figsize=(12, 8))
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
