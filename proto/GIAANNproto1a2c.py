import torch
import networkx as nx
import matplotlib.pyplot as plt
import nltk
import spacy
from nltk.tokenize import sent_tokenize
from datasets import load_dataset

# Ensure necessary NLTK data packages are downloaded
nltk.download('punkt')

# Load spaCy English model
nlp = spacy.load('en_core_web_sm')

# Load dataset from Hugging Face
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

# Get text data
text = '\n'.join(dataset['text'])

# Tokenize text into sentences
sentences = sent_tokenize(text)

# Initialize the neural network representation
columns = {}  # key: word (concept), value: {'concept_neuron': neuron_id, 'relation_neurons': {relation_word: {'neuron_id': neuron_id, 'permanence': int}}}

# Initialize NetworkX graph
G = nx.Graph()

# Global neuron ID counter to ensure unique IDs
neuron_id_counter = 0

# Function to visualize the network
def visualize_network(G, columns):
    plt.figure(figsize=(12, 8))
    pos = {}  # positions of nodes
    labels = {}  # labels of nodes
    x_margin = 2  # margin between columns
    y_margin = 1  # margin between neurons in column
    neuron_size = 500
    x_positions = {}
    max_y = 0
    for i, (concept_word, neurons) in enumerate(columns.items()):
        # x position is i * x_margin
        x = i * x_margin * 2
        x_positions[concept_word] = x
        # Concept neuron at bottom (y=0)
        concept_neuron_id = neurons['concept_neuron']
        pos[concept_neuron_id] = (x, 0)
        labels[concept_neuron_id] = concept_word
        # Relation neurons above
        for j, (relation_word, relation_info) in enumerate(neurons['relation_neurons'].items()):
            relation_neuron_id = relation_info['neuron_id']
            y = (j + 1) * y_margin * 2  # y position for relation neurons
            pos[relation_neuron_id] = (x, y)
            labels[relation_neuron_id] = f"{relation_word} ({relation_info['permanence']})"
            if y > max_y:
                max_y = y
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=neuron_size, node_color='lightblue')
    # Draw edges
    nx.draw_networkx_edges(G, pos)
    # Draw labels
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    # Draw rectangles around columns
    for concept_word, x in x_positions.items():
        # Get y positions of neurons in this column
        neurons = columns[concept_word]
        y_positions = [pos[neurons['concept_neuron']][1]]
        for relation_info in neurons['relation_neurons'].values():
            y_positions.append(pos[relation_info['neuron_id']][1])
        y_min = min(y_positions) - y_margin
        y_max = max(y_positions) + y_margin
        # Draw rectangle
        plt.gca().add_patch(plt.Rectangle((x - x_margin, y_min), x_margin * 2, y_max - y_min + y_margin, fill=False, edgecolor='black'))
    plt.axis('off')
    plt.show()

# Main processing loop
for sentence in sentences:
    # Process sentence
    print(sentence)
    doc = nlp(sentence)
    words = [token.text for token in doc]
    pos_tags = [token.pos_ for token in doc]

    # Keep track of activated concept neurons and their relations in this sentence
    activated_concepts = {}
    activated_relations = {}

    # For each word in the sentence, identify concept words and their relations
    for idx, (word, pos_tag) in enumerate(zip(words, pos_tags)):
        if not word.isalpha():
            continue

        if pos_tag == 'NOUN' or pos_tag == 'PROPN':
            # This word is a concept word (noun)
            # If word not in columns, create a new column
            if word not in columns:
                # Assign a unique neuron ID for the concept neuron
                neuron_id_counter += 1
                concept_neuron_id = neuron_id_counter
                columns[word] = {'concept_neuron': concept_neuron_id, 'relation_neurons': {}}
                # Add the concept neuron to the graph
                G.add_node(concept_neuron_id)
            else:
                concept_neuron_id = columns[word]['concept_neuron']
            # Mark this concept as activated
            activated_concepts[word] = True

            # Process words after this concept word to find relations
            relations_found = []
            for next_idx in range(idx+1, len(words)):
                next_word = words[next_idx]
                next_pos_tag = pos_tags[next_idx]
                if not next_word.isalpha():
                    continue
                if next_pos_tag == 'VERB' or next_pos_tag == 'ADP':  # Verb or preposition
                    relations_found.append(next_word)
                    # Ensure the relation neuron exists in the column
                    if next_word not in columns[word]['relation_neurons']:
                        # Assign a unique neuron ID for the relation neuron
                        neuron_id_counter += 1
                        relation_neuron_id = neuron_id_counter
                        columns[word]['relation_neurons'][next_word] = {'neuron_id': relation_neuron_id, 'permanence': 0}
                        # Add the relation neuron to the graph
                        G.add_node(relation_neuron_id)
                    else:
                        relation_neuron_id = columns[word]['relation_neurons'][next_word]['neuron_id']
                    # Connect the relation neuron to its target(s)
                    # The target is the next concept word after the relation word
                    for target_idx in range(next_idx+1, len(words)):
                        target_word = words[target_idx]
                        target_pos_tag = pos_tags[target_idx]
                        if not target_word.isalpha():
                            continue
                        if target_pos_tag == 'NOUN' or target_pos_tag == 'PROPN':
                            # Ensure the target concept word has a concept neuron
                            if target_word not in columns:
                                neuron_id_counter += 1
                                target_concept_neuron_id = neuron_id_counter
                                columns[target_word] = {'concept_neuron': target_concept_neuron_id, 'relation_neurons': {}}
                                G.add_node(target_concept_neuron_id)
                            else:
                                target_concept_neuron_id = columns[target_word]['concept_neuron']
                            # Connect the relation neuron to the target concept neuron
                            G.add_edge(relation_neuron_id, target_concept_neuron_id)
                            break  # Only connect to the first concept word after relation word
                else:
                    continue
            # Store the relations found for this concept in this sentence
            activated_relations[word] = relations_found

    # Update permanence values
    for concept_word, neurons in columns.items():
        if concept_word in activated_concepts:
            # Concept neuron was activated in this sentence
            active_relations = activated_relations.get(concept_word, [])
            # Update permanence for each relation neuron
            relations_to_remove = []
            for relation_word, relation_info in list(neurons['relation_neurons'].items()):
                if relation_word in active_relations:
                    # Relation neuron was activated, increase permanence by 3
                    relation_info['permanence'] += 3
                else:
                    # Relation neuron was not activated, decrease permanence by 1
                    relation_info['permanence'] -= 1
                    if relation_info['permanence'] <= 0:
                        # Mark for removal
                        relations_to_remove.append(relation_word)
            # Remove relation neurons with permanence <= 0
            for relation_word in relations_to_remove:
                relation_neuron_id = neurons['relation_neurons'][relation_word]['neuron_id']
                if G.has_node(relation_neuron_id):
                    G.remove_node(relation_neuron_id)
                del neurons['relation_neurons'][relation_word]

    # Visualize the network
    visualize_network(G, columns)
'''
    # Pause the simulation after every sentence
    try:
        input("Press Enter to continue or Ctrl-D to exit...")
    except EOFError:
        break  # Exit the simulation
'''
