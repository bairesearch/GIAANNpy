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
observed_columns_dir = 'observed_columns'
os.makedirs(observed_columns_dir, exist_ok=True)

if not lowMem:
    feature_neurons_strength_file = 'global_feature_neurons_strength.pt'
    feature_neurons_permanence_file = 'global_feature_neurons_permanence.pt'
    feature_neurons_activation_file = 'global_feature_neurons_activation.pt'

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

# Set the size of the feature arrays (f)
if usePOS:
    f = max_num_non_nouns  # Maximum number of non-nouns in an English dictionary
else:
    f = 0  # Will be updated dynamically based on c

# Initialize the concept columns dictionary
if useInference and os.path.exists(concept_columns_dict_file):
    with open(concept_columns_dict_file, 'rb') as f_in:
        concept_columns_dict = pickle.load(f_in)
    c = len(concept_columns_dict)
    concept_columns_list = list(concept_columns_dict.keys())
else:
    concept_columns_dict = {}  # key: lemma, value: index
    concept_columns_list = []  # list of concept column names (lemmas)
    c = 0  # current number of concept columns

# Initialize global feature neuron arrays if lowMem is disabled
if not lowMem:
    if os.path.exists(feature_neurons_strength_file):
        global_feature_neurons_strength = torch.load(feature_neurons_strength_file)
        global_feature_neurons_permanence = torch.load(feature_neurons_permanence_file)
        global_feature_neurons_activation = torch.load(feature_neurons_activation_file)
    else:
        global_feature_neurons_strength = torch.zeros(c, f)
        global_feature_neurons_permanence = torch.full((c, f), 3)  # Initialize permanence to z1=3
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
    def __init__(self, concept_index):
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

        # Store all connections for each source column in a list of integer feature connection arrays,
        # each of size f * c, where c is the length of the dictionary of columns, and f is the maximum
        # number of feature neurons.
        self.connection_strength = torch.zeros(f, c, dtype=torch.int32)
        self.connection_permanence = torch.full((f, c), z1, dtype=torch.int32)  # Initialize permanence to z1=3
        self.connection_activation = torch.zeros(f, c, dtype=torch.int32)  # Activation trace counters

    def resize_connection_arrays(self, new_c):
        if new_c > self.connection_strength.shape[1]:
            extra_cols = new_c - self.connection_strength.shape[1]
            self.connection_strength = torch.cat([self.connection_strength, torch.zeros(self.connection_strength.shape[0], extra_cols, dtype=torch.int32)], dim=1)
            self.connection_permanence = torch.cat([self.connection_permanence, torch.full((self.connection_permanence.shape[0], extra_cols), z1, dtype=torch.int32)], dim=1)
            self.connection_activation = torch.cat([self.connection_activation, torch.zeros(self.connection_activation.shape[0], extra_cols, dtype=torch.int32)], dim=1)

    def expand_feature_arrays(self, new_f):
        if new_f > self.connection_strength.shape[0]:
            extra_rows = new_f - self.connection_strength.shape[0]
            self.connection_strength = torch.cat([self.connection_strength, torch.zeros(extra_rows, self.connection_strength.shape[1], dtype=torch.int32)], dim=0)
            self.connection_permanence = torch.cat([self.connection_permanence, torch.full((extra_rows, self.connection_permanence.shape[1]), z1, dtype=torch.int32)], dim=0)
            self.connection_activation = torch.cat([self.connection_activation, torch.zeros(extra_rows, self.connection_activation.shape[1], dtype=torch.int32)], dim=0)
            if lowMem:
                self.feature_neurons_strength = torch.cat([self.feature_neurons_strength, torch.zeros(extra_rows)], dim=0)
                self.feature_neurons_permanence = torch.cat([self.feature_neurons_permanence, torch.full((extra_rows,), z1, dtype=torch.int32)], dim=0)
                self.feature_neurons_activation = torch.cat([self.feature_neurons_activation, torch.zeros(extra_rows, dtype=torch.int32)], dim=0)

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
    def load_from_disk(cls, concept_index):
        """
        Load the observed column data from disk.
        """
        # Load the data dictionary
        with open(os.path.join(observed_columns_dir, f"{concept_index}_data.pkl"), 'rb') as f:
            data = pickle.load(f)
        instance = cls(concept_index)
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
max_sentences = 5  # Adjust as needed

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
    global sentence_count, c, f, concept_columns_dict, concept_columns_list
    print(f"Processing sentence: {sentence}")

    # Refresh the observed columns dictionary for each new sequence
    observed_columns_dict = {}  # key: lemma, value: ObservedColumn

    # Process the sentence with spaCy
    doc = nlp(sentence)

    # First pass: Extract words, lemmas, POS tags, and update concept_columns_dict and c
    words, lemmas, pos_tags = first_pass(doc)

    # Second pass: Create observed_columns_dict
    observed_columns_dict = second_pass(lemmas, pos_tags)

    # Process each observed column to ensure connection arrays are resized if needed
    for observed_column in observed_columns_dict.values():
        observed_column.resize_connection_arrays(c)

    # Process each concept word in the sequence
    process_concept_words(doc, lemmas, pos_tags, observed_columns_dict)

    # Update permanence and activation traces for feature neurons and connections
    update_permanence_and_activation(observed_columns_dict)

    # Visualize the complete graph every time a new sentence is parsed by the application.
    visualize_graph(observed_columns_dict)

    # Save observed columns to disk
    save_data(observed_columns_dict)

    # Break if we've reached the maximum number of sentences
    global sentence_count
    sentence_count += 1

def first_pass(doc):
    global c, f, concept_columns_dict, concept_columns_list
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
                observed_column = load_or_create_observed_column(concept_index)
                observed_columns_dict[lemma] = observed_column
        else:
            concept_index = concept_columns_dict[lemma]
            # Load observed column from disk or create new one
            observed_column = load_or_create_observed_column(concept_index)
            observed_columns_dict[lemma] = observed_column
    return observed_columns_dict

def load_or_create_observed_column(concept_index):
    observed_column_file = os.path.join(observed_columns_dir, f"{concept_index}_data.pkl")
    if os.path.exists(observed_column_file):
        observed_column = ObservedColumn.load_from_disk(concept_index)
        # Resize connection arrays if c has increased
        observed_column.resize_connection_arrays(c)
    else:
        observed_column = ObservedColumn(concept_index)
        # Initialize connection arrays with correct size
        observed_column.resize_connection_arrays(c)
    return observed_column

def process_concept_words(doc, lemmas, pos_tags, observed_columns_dict):
    global c, f
    q = len(doc)
    for i, token in enumerate(doc):
        lemma_i = lemmas[i]
        pos_i = pos_tags[i]

        if lemma_i in observed_columns_dict:
            observed_column = observed_columns_dict[lemma_i]
            concept_index_i = observed_column.concept_index

            # Set to track feature neurons activated in this sequence for this concept
            activated_feature_indices = set()

            # Include the concept neuron (index 0)
            # For positions j in [i - q, i + q], excluding i
            start = max(0, i - q)
            end = min(len(doc), i + q + 1)

            for j in range(start, end):
                if j == i:
                    continue  # Exclude the current word

                token_j = doc[j]
                word_j = token_j.text
                lemma_j = lemmas[j]
                pos_j = pos_tags[j]

                # Assign feature neurons to words not lemmas
                feature_word = word_j.lower()

                # Get or assign feature neuron index for feature_word in this column
                if feature_word not in observed_column.feature_word_to_index:
                    feature_index = observed_column.next_feature_index
                    observed_column.feature_word_to_index[feature_word] = feature_index
                    observed_column.feature_index_to_word[feature_index] = feature_word
                    observed_column.next_feature_index += 1
                    # Expand feature arrays if needed
                    if feature_index >= f:
                        f = feature_index + 1
                        if not lowMem:
                            # Expand global feature neuron arrays
                            global_feature_neurons_strength = torch.cat([global_feature_neurons_strength, torch.zeros(global_feature_neurons_strength.shape[0], 1)], dim=1)
                            global_feature_neurons_permanence = torch.cat([global_feature_neurons_permanence, torch.full((global_feature_neurons_permanence.shape[0], 1), z1, dtype=torch.int32)], dim=1)
                            global_feature_neurons_activation = torch.cat([global_feature_neurons_activation, torch.zeros(global_feature_neurons_activation.shape[0], 1, dtype=torch.int32)], dim=1)
                        # Expand connection and feature arrays for all observed columns
                        for obs_col in observed_columns_dict.values():
                            obs_col.expand_feature_arrays(f)
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
                    global_feature_neurons_strength[concept_index_i, feature_index] += 1
                    # Increase permanence exponentially
                    global_feature_neurons_permanence[concept_index_i, feature_index] = global_feature_neurons_permanence[concept_index_i, feature_index] ** 2
                    # Set activation trace to j1 sequences
                    global_feature_neurons_activation[concept_index_i, feature_index] = j1

                # Create connections
                # Connect these feature neurons to every other feature neuron in every other concept column in the sequence
                for other_lemma, other_observed_column in observed_columns_dict.items():
                    other_concept_index = other_observed_column.concept_index
                    if other_concept_index != concept_index_i:
                        observed_column.connection_strength[feature_index, other_concept_index] += 1
                        # Increase permanence exponentially
                        observed_column.connection_permanence[feature_index, other_concept_index] = observed_column.connection_permanence[feature_index, other_concept_index] ** 2
                        # Set activation trace to j1 sequences
                        observed_column.connection_activation[feature_index, other_concept_index] = j1

            # Decrease permanence for feature neurons not activated
            all_feature_indices = set(observed_column.feature_word_to_index.values())
            inactive_feature_indices = all_feature_indices - activated_feature_indices
            for feature_index in inactive_feature_indices:
                # Decrease permanence linearly
                if lowMem:
                    observed_column.feature_neurons_permanence[feature_index] -= z2
                    # Remove feature neuron if permanence <= 0
                    if observed_column.feature_neurons_permanence[feature_index] <= 0:
                        # Remove feature neuron from the column
                        del_word = observed_column.feature_index_to_word[feature_index]
                        del observed_column.feature_word_to_index[del_word]
                        del observed_column.feature_index_to_word[feature_index]
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

            # Decrease permanence for connections not activated
            for feature_index in inactive_feature_indices:
                for other_concept_index in range(c):
                    if other_concept_index != concept_index_i:
                        if observed_column.connection_permanence[feature_index, other_concept_index] > 0:
                            observed_column.connection_permanence[feature_index, other_concept_index] -= z2
                            # Remove connection if permanence <= 0
                            if observed_column.connection_permanence[feature_index, other_concept_index] <= 0:
                                observed_column.connection_permanence[feature_index, other_concept_index] = 0
                                observed_column.connection_activation[feature_index, other_concept_index] = 0

def update_permanence_and_activation(observed_columns_dict):
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
            if observed_column.connection_activation[i, j] > 0:
                observed_column.connection_activation[i, j] -= 1
                if observed_column.connection_activation[i, j] == 0:
                    # Activation trace expired
                    pass  # Do nothing for now


def visualize_graph(observed_columns_dict):
    G.clear()

    # Draw concept columns
    pos_dict = {}
    x_offset = 0
    for lemma, observed_column in observed_columns_dict.items():
        concept_index = observed_column.concept_index

        # Draw the concept neuron (blue)
        concept_node = f"{lemma}_concept"
        G.add_node(concept_node, pos=(x_offset, 0), color='blue', label=lemma)

        # Draw feature neurons (cyan)
        y_offset = 1
        for feature_index, feature_word in observed_column.feature_index_to_word.items():
            if lowMem:
                # Only visualize feature neurons with permanence > 0
                if observed_column.feature_neurons_permanence[feature_index] > 0:
                    feature_node = f"{lemma}_{feature_word}_{feature_index}"
                    G.add_node(feature_node, pos=(x_offset, y_offset), color='cyan', label=feature_word)
                    y_offset += 1
            else:
                feature_node = f"{lemma}_{feature_word}_{feature_index}"
                G.add_node(feature_node, pos=(x_offset, y_offset), color='cyan', label=feature_word)
                y_offset += 1

        # Draw rectangle around the column
        plt.gca().add_patch(plt.Rectangle((x_offset - 0.5, -0.5), 1, max(y_offset, 1) + 0.5, fill=False, edgecolor='black'))

        x_offset += 2  # Adjust x_offset for the next column

    # Draw connections
    for lemma, observed_column in observed_columns_dict.items():
        concept_index = observed_column.concept_index
        concept_node = f"{lemma}_concept"

        # Internal connections (yellow)
        for feature_index, feature_word in observed_column.feature_index_to_word.items():
            source_node = f"{lemma}_{feature_word}_{feature_index}"
            if G.has_node(source_node):
                G.add_edge(concept_node, source_node, color='yellow')

        # External connections (orange)
        for feature_index, feature_word in observed_column.feature_index_to_word.items():
            source_node = f"{lemma}_{feature_word}_{feature_index}"
            if G.has_node(source_node):
                for other_lemma, other_observed_column in observed_columns_dict.items():
                    other_concept_index = other_observed_column.concept_index
                    other_concept_node = f"{other_lemma}_concept"
                    if other_concept_index != concept_index:
                        if G.has_node(other_concept_node):
                            # Only visualize connections with permanence > 0
                            if observed_column.connection_permanence[feature_index, other_concept_index] > 0:
                                G.add_edge(source_node, other_concept_node, color='orange')

    # Get positions and colors for drawing
    pos = nx.get_node_attributes(G, 'pos')
    colors = [data['color'] for node, data in G.nodes(data=True)]
    edge_colors = [data['color'] for u, v, data in G.edges(data=True)]
    labels = nx.get_node_attributes(G, 'label')

    # Draw the graph
    #plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, labels=labels, node_color=colors, edge_color=edge_colors)
    plt.show()

def save_data(observed_columns_dict):
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

# Load the Wikipedia dataset using Hugging Face datasets
dataset = load_dataset('wikipedia', '20220301.en', split='train', streaming=True)

# Start processing the dataset
process_dataset(dataset)
