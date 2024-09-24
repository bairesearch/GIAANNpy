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
columns = {}  # key: word (concept), value: {'concept_neuron': neuron_id, 'permanence': int, 'relation_neurons': {...}, 'quality_neurons': {...}}

# Initialize NetworkX graph
G = nx.Graph()

# Global neuron ID counter to ensure unique IDs
neuron_id_counter = 0

# Lists for POS types
concept_pos_list = ['NOUN', 'PROPN', 'PRON', 'X']  # POS types for concept columns
relation_pos_list = ['VERB', 'ADP', 'CONJ']        # POS types for relation neurons
quality_pos_list = ['DET', 'ADV', 'ADJ']           # POS types for quality neurons

# Function to visualize the network
def visualize_network(G, columns):
    plt.figure(figsize=(14, 10))
    pos = {}  # positions of nodes
    labels = {}  # labels of nodes
    x_margin = 2  # margin between columns
    y_margin = 1  # margin between neurons in column
    neuron_size = 500
    x_positions = {}
    max_y = 0

    # Draw concept columns
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
            labels[relation_neuron_id] = relation_word  # We can include permanence if needed
            if y > max_y:
                max_y = y
        # Quality neurons above relation neurons
        for k, (quality_word, quality_info) in enumerate(neurons['quality_neurons'].items()):
            quality_neuron_id = quality_info['neuron_id']
            y = (len(neurons['relation_neurons']) + k + 1) * y_margin * 2
            pos[quality_neuron_id] = (x, y)
            labels[quality_neuron_id] = quality_word
            if y > max_y:
                max_y = y

    # Draw nodes
    node_colors = []
    for node_id in G.nodes():
        if node_id in [neurons['concept_neuron'] for neurons in columns.values()]:
            node_colors.append('blue')  # Concept neurons
        elif any(node_id in [info['neuron_id'] for info in neurons['relation_neurons'].values()] for neurons in columns.values()):
            # Relation neurons
            # Determine if action (green) or condition (red) or conjunction (we'll consider as action for coloring)
            for neurons in columns.values():
                for relation_word, relation_info in neurons['relation_neurons'].items():
                    if relation_info['neuron_id'] == node_id:
                        relation_pos = relation_info['pos']
                        if relation_pos == 'VERB':
                            node_colors.append('green')  # Action
                        elif relation_pos == 'ADP':
                            node_colors.append('red')    # Condition
                        elif relation_pos == 'CONJ':
                            node_colors.append('green')  # Conjunctions colored as action for simplicity
                        else:
                            node_colors.append('gray')   # Default color
                        break
        elif any(node_id in [info['neuron_id'] for info in neurons['quality_neurons'].values()] for neurons in columns.values()):
            node_colors.append('cyan')  # Quality neurons
        else:
            node_colors.append('gray')   # Default color

    nx.draw_networkx_nodes(G, pos, node_size=neuron_size, node_color=node_colors)

    # Draw edges
    edge_colors = []
    for u, v in G.edges():
        edge = G.get_edge_data(u, v)
        if edge['type'] == 'relation_target':
            if edge['pos'] == 'VERB':
                edge_colors.append('green')
            elif edge['pos'] == 'ADP':
                edge_colors.append('red')
            elif edge['pos'] == 'CONJ':
                edge_colors.append('green')  # Conjunctions colored as action for simplicity
            else:
                edge_colors.append('gray')
        elif edge['type'] == 'concept_source':
            edge_colors.append('blue')
        else:
            edge_colors.append('gray')  # Default color

    nx.draw_networkx_edges(G, pos, edge_color=edge_colors)

    # Draw labels
    nx.draw_networkx_labels(G, pos, labels, font_size=8)

    # Draw rectangles around columns
    for concept_word, x in x_positions.items():
        # Get y positions of neurons in this column
        neurons = columns[concept_word]
        y_positions = [pos[neurons['concept_neuron']][1]]
        for relation_info in neurons['relation_neurons'].values():
            y_positions.append(pos[relation_info['neuron_id']][1])
        for quality_info in neurons['quality_neurons'].values():
            y_positions.append(pos[quality_info['neuron_id']][1])
        y_min = min(y_positions) - y_margin
        y_max = max(y_positions) + y_margin
        # Draw rectangle
        plt.gca().add_patch(plt.Rectangle((x - x_margin, y_min), x_margin * 2, y_max - y_min + y_margin, fill=False, edgecolor='black'))
    plt.axis('off')
    plt.show()

# Main processing loop
for sentence in sentences:
    # Print the sentence
    print(sentence)

    # Process sentence
    doc = nlp(sentence)
    words = [token.text for token in doc]
    pos_tags = [token.pos_ for token in doc]

    # Keep track of activated concepts and their relations in this sentence
    activated_concepts = {}
    activated_relations = {}
    activated_qualities = {}

    # First, ensure all words are in columns
    for word, pos_tag in zip(words, pos_tags):
        if not word.isalpha():
            continue
        if word not in columns:
            neuron_id_counter += 1
            concept_neuron_id = neuron_id_counter
            columns[word] = {
                'concept_neuron': concept_neuron_id,
                'permanence': 0,  # Initialize permanence as 0
                'relation_neurons': {},
                'quality_neurons': {}
            }
            # Add the concept neuron to the graph
            G.add_node(concept_neuron_id)

    # For each word in the sentence, identify concept words and their relations and qualities
    for idx, (word, pos_tag) in enumerate(zip(words, pos_tags)):
        if not word.isalpha():
            continue

        concept_neuron_id = columns[word]['concept_neuron']

        # Mark this concept as activated
        activated_concepts[word] = True

        # Process relations
        relations_found = []
        for next_idx in range(idx+1, len(words)):
            next_word = words[next_idx]
            next_pos_tag = pos_tags[next_idx]
            if not next_word.isalpha():
                continue

            if next_pos_tag in relation_pos_list:
                relations_found.append(next_word)
                # Ensure the relation neuron exists in the column
                if next_word not in columns[word]['relation_neurons']:
                    neuron_id_counter += 1
                    relation_neuron_id = neuron_id_counter
                    columns[word]['relation_neurons'][next_word] = {
                        'neuron_id': relation_neuron_id,
                        'permanence': 0,
                        'pos': next_pos_tag  # Store POS type
                    }
                    # Add the relation neuron to the graph
                    G.add_node(relation_neuron_id)
                    # Draw concept source connection
                    relation_word_concept_id = columns[next_word]['concept_neuron']
                    G.add_edge(relation_word_concept_id, relation_neuron_id, type='concept_source')
                else:
                    relation_neuron_id = columns[word]['relation_neurons'][next_word]['neuron_id']

                # Connect the relation neuron to its target(s)
                # The target is the next concept word after the relation word
                for target_idx in range(next_idx+1, len(words)):
                    target_word = words[target_idx]
                    target_pos_tag = pos_tags[target_idx]
                    if not target_word.isalpha():
                        continue
                    if target_pos_tag in concept_pos_list:
                        target_concept_neuron_id = columns[target_word]['concept_neuron']
                        # Connect the relation neuron to the target concept neuron
                        G.add_edge(relation_neuron_id, target_concept_neuron_id, type='relation_target', pos=next_pos_tag)
                        break  # Only connect to the first concept word after relation word
            else:
                # Break if we encounter another concept word or quality word before a relation word
                if next_pos_tag in concept_pos_list or next_pos_tag in quality_pos_list:
                    break
                continue
        # Store the relations found for this concept in this sentence
        activated_relations[word] = relations_found

        # Process qualities
        qualities_found = []
        # Check previous word
        if idx > 0:
            prev_word = words[idx - 1]
            prev_pos_tag = pos_tags[idx - 1]
            if prev_pos_tag in quality_pos_list:
                # Ensure there is no relation word between them
                intervening_words = pos_tags[idx - 1:idx]
                if not any(pos in relation_pos_list for pos in intervening_words):
                    qualities_found.append(prev_word)
                    # Ensure the quality neuron exists in the column
                    if prev_word not in columns[word]['quality_neurons']:
                        neuron_id_counter += 1
                        quality_neuron_id = neuron_id_counter
                        columns[word]['quality_neurons'][prev_word] = {
                            'neuron_id': quality_neuron_id,
                            'permanence': 0,
                            'pos': prev_pos_tag
                        }
                        # Add the quality neuron to the graph
                        G.add_node(quality_neuron_id)
                        # Ensure prev_word has a concept neuron
                        prev_word_concept_id = columns[prev_word]['concept_neuron']
                        # Draw concept source connection
                        G.add_edge(prev_word_concept_id, quality_neuron_id, type='concept_source')
                    else:
                        quality_neuron_id = columns[word]['quality_neurons'][prev_word]['neuron_id']
        # Check next word
        if idx < len(words) - 1:
            next_word = words[idx + 1]
            next_pos_tag = pos_tags[idx + 1]
            if next_pos_tag in quality_pos_list:
                # Ensure there is no relation word between them
                intervening_words = pos_tags[idx + 1:idx + 2]
                if not any(pos in relation_pos_list for pos in intervening_words):
                    qualities_found.append(next_word)
                    # Ensure the quality neuron exists in the column
                    if next_word not in columns[word]['quality_neurons']:
                        neuron_id_counter += 1
                        quality_neuron_id = neuron_id_counter
                        columns[word]['quality_neurons'][next_word] = {
                            'neuron_id': quality_neuron_id,
                            'permanence': 0,
                            'pos': next_pos_tag
                        }
                        # Add the quality neuron to the graph
                        G.add_node(quality_neuron_id)
                        # Ensure next_word has a concept neuron
                        next_word_concept_id = columns[next_word]['concept_neuron']
                        # Draw concept source connection
                        G.add_edge(next_word_concept_id, quality_neuron_id, type='concept_source')
                    else:
                        quality_neuron_id = columns[word]['quality_neurons'][next_word]['neuron_id']

        # Store the qualities found for this concept in this sentence
        activated_qualities[word] = qualities_found

    # Update permanence values for concept neurons
    for concept_word, neurons in columns.items():
        if concept_word in activated_concepts:
            neurons['permanence'] += 1  # Increase permanence by 1 when activated
        # No decrease specified for concept neurons

    # Update permanence values for relation neurons
    for concept_word, neurons in columns.items():
        if concept_word in activated_concepts:
            active_relations = activated_relations.get(concept_word, [])
            # Update permanence for each relation neuron
            relations_to_remove = []
            for relation_word, relation_info in list(neurons['relation_neurons'].items()):
                if relation_word in active_relations:
                    # Relation neuron was activated, increase permanence by 1
                    relation_info['permanence'] += 1
                else:
                    # Relation neuron was not activated, decrease permanence by 1
                    relation_info['permanence'] -= 1
                    if relation_info['permanence'] <= 0:
                        # Remove the relation neuron
                        relation_neuron_id = relation_info['neuron_id']
                        if G.has_node(relation_neuron_id):
                            G.remove_node(relation_neuron_id)
                        del neurons['relation_neurons'][relation_word]
            # No decrease specified for relations not associated with activated concepts

    # Update permanence values for quality neurons
    for concept_word, neurons in columns.items():
        if concept_word in activated_concepts:
            active_qualities = activated_qualities.get(concept_word, [])
            # Update permanence for each quality neuron
            qualities_to_remove = []
            for quality_word, quality_info in list(neurons['quality_neurons'].items()):
                if quality_word in active_qualities:
                    # Quality neuron was activated, increase permanence by 1
                    quality_info['permanence'] += 1
                else:
                    # Quality neuron was not activated, decrease permanence by 1
                    quality_info['permanence'] -= 1
                    if quality_info['permanence'] <= 0:
                        # Remove the quality neuron
                        quality_neuron_id = quality_info['neuron_id']
                        if G.has_node(quality_neuron_id):
                            G.remove_node(quality_neuron_id)
                        del neurons['quality_neurons'][quality_word]
            # No decrease specified for qualities not associated with activated concepts

    # Visualize the network
    visualize_network(G, columns)

    # No pause; continue to next sentence
