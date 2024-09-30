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
columns = {}  # key: word (concept), value: {'concept_neuron', 'permanence', 'relation_neurons', 'quality_neurons', 'instance_connections'}

# Initialize NetworkX graph
G = nx.Graph()

# Global neuron ID counter to ensure unique IDs
neuron_id_counter = 0

# Sentence counter to manage activation traces
sentence_counter = 0

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
            labels[relation_neuron_id] = relation_word
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
            for neurons in columns.values():
                for relation_word, relation_info in neurons['relation_neurons'].items():
                    if relation_info['neuron_id'] == node_id:
                        relation_pos = relation_info['pos']
                        if relation_pos == 'VERB':
                            node_colors.append('green')  # Action
                        elif relation_pos == 'ADP':
                            node_colors.append('red')    # Condition
                        elif relation_pos == 'CONJ':
                            node_colors.append('green')  # Conjunctions colored as action
                        else:
                            node_colors.append('gray')
                        break
        elif any(node_id in [info['neuron_id'] for info in neurons['quality_neurons'].values()] for neurons in columns.values()):
            node_colors.append('cyan')  # Quality neurons
        else:
            node_colors.append('gray')

    # Draw edges
    edge_colors = []
    edge_styles = []
    for u, v in G.edges():
        edge = G.get_edge_data(u, v)
        if edge['type'] == 'relation_target':
            if edge['pos'] == 'VERB':
                edge_colors.append('green')
            elif edge['pos'] == 'ADP':
                edge_colors.append('red')
            elif edge['pos'] == 'CONJ':
                edge_colors.append('green')
            else:
                edge_colors.append('gray')
            edge_styles.append('solid')
        elif edge['type'] == 'concept_source':
            edge_colors.append('blue')
            edge_styles.append('solid')
        elif edge['type'] == 'instance_connection':
            edge_colors.append('yellow')
            edge_styles.append('solid')
        else:
            edge_colors.append('gray')
            edge_styles.append('solid')

    nx.draw_networkx_nodes(G, pos, node_size=neuron_size, node_color=node_colors)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, style=edge_styles)
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

def ensure_words_in_columns(words):
    global neuron_id_counter
    for word in words:
        if not word.isalpha():
            continue
        if word not in columns:
            neuron_id_counter += 1
            concept_neuron_id = neuron_id_counter
            columns[word] = {
                'concept_neuron': concept_neuron_id,
                'permanence': 3,  # Initialized to 3
                'concept_activation_trace_counter': 0,  # Activation trace counter
                'relation_neurons': {},
                'quality_neurons': {},
                'instance_connections': {}
            }
            # Add the concept neuron to the graph
            G.add_node(concept_neuron_id)

def collect_qualities_and_initialize_instances(words, pos_tags):
    global neuron_id_counter  # Moved to the top
    activated_concepts = {}
    activated_instances = {}
    activated_qualities = {}
    for idx, (word, pos_tag) in enumerate(zip(words, pos_tags)):
        if not word.isalpha():
            continue

        activated_concepts[word] = True

        # Set activation trace for concept neuron
        columns[word]['concept_activation_trace_counter'] = 5  # Reset to 5

        # Collect qualities for this word
        qualities_found = []
        # Check previous word
        if idx > 0:
            prev_word = words[idx - 1]
            prev_pos_tag = pos_tags[idx - 1]
            if prev_pos_tag in quality_pos_list:
                intervening_words = pos_tags[idx - 1:idx]
                if not any(pos in relation_pos_list for pos in intervening_words):
                    qualities_found.append(prev_word)
                    # Ensure the quality neuron exists in the column
                    if prev_word not in columns[word]['quality_neurons']:
                        neuron_id_counter += 1
                        quality_neuron_id = neuron_id_counter
                        columns[word]['quality_neurons'][prev_word] = {
                            'neuron_id': quality_neuron_id,
                            'permanence': 3,  # Initialized to 3
                            'activation_trace_counter': 5,  # Activation trace counter
                            'pos': prev_pos_tag,
                            'first_activation': True
                        }
                        # Add the quality neuron to the graph
                        G.add_node(quality_neuron_id)
                        # Ensure prev_word has a concept neuron
                        prev_word_concept_id = columns[prev_word]['concept_neuron']
                        # Draw concept source connection
                        G.add_edge(prev_word_concept_id, quality_neuron_id, type='concept_source')
                    else:
                        quality_neuron_id = columns[word]['quality_neurons'][prev_word]['neuron_id']
                        # Reset activation trace counter
                        columns[word]['quality_neurons'][prev_word]['activation_trace_counter'] = 5
        # Check next word
        if idx < len(words) - 1:
            next_word = words[idx + 1]
            next_pos_tag = pos_tags[idx + 1]
            if next_pos_tag in quality_pos_list:
                intervening_words = pos_tags[idx + 1:idx + 2]
                if not any(pos in relation_pos_list for pos in intervening_words):
                    qualities_found.append(next_word)
                    if next_word not in columns[word]['quality_neurons']:
                        neuron_id_counter += 1
                        quality_neuron_id = neuron_id_counter
                        columns[word]['quality_neurons'][next_word] = {
                            'neuron_id': quality_neuron_id,
                            'permanence': 3,  # Initialized to 3
                            'activation_trace_counter': 5,  # Activation trace counter
                            'pos': next_pos_tag,
                            'first_activation': True
                        }
                        # Add the quality neuron to the graph
                        G.add_node(quality_neuron_id)
                        # Ensure next_word has a concept neuron
                        next_word_concept_id = columns[next_word]['concept_neuron']
                        # Draw concept source connection
                        G.add_edge(next_word_concept_id, quality_neuron_id, type='concept_source')
                    else:
                        quality_neuron_id = columns[word]['quality_neurons'][next_word]['neuron_id']
                        # Reset activation trace counter
                        columns[word]['quality_neurons'][next_word]['activation_trace_counter'] = 5

        # Store the qualities found for this concept in this sentence
        activated_qualities[word] = qualities_found

        # Collect instance neurons for this word
        instance_neurons = []
        # Quality neurons
        for qual_word in qualities_found:
            if qual_word in columns[word]['quality_neurons']:
                qual_neuron_info = columns[word]['quality_neurons'][qual_word]
                qual_neuron_id = qual_neuron_info['neuron_id']
                instance_neurons.append(qual_neuron_id)
                # Reset activation trace counter
                qual_neuron_info['activation_trace_counter'] = 5

        # Store activated instance neurons for this concept
        activated_instances[word] = instance_neurons

    return activated_concepts, activated_instances, activated_qualities

def process_relations(words, pos_tags, activated_instances):
    global neuron_id_counter  # Moved to the top
    activated_relations = {}
    activated_relation_targets = {}
    for idx, (word, pos_tag) in enumerate(zip(words, pos_tags)):
        if not word.isalpha():
            continue

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
                        'permanence': 3,  # Initialized to 3
                        'activation_trace_counter': 5,  # Activation trace counter
                        'pos': next_pos_tag,
                        'target_connections': {},
                        'first_activation': True
                    }
                    # Add the relation neuron to the graph
                    G.add_node(relation_neuron_id)
                    # Draw concept source connection
                    relation_word_concept_id = columns[next_word]['concept_neuron']
                    # Reset activation trace counter for relation word concept neuron
                    columns[next_word]['concept_activation_trace_counter'] = 5
                    G.add_edge(relation_word_concept_id, relation_neuron_id, type='concept_source')
                else:
                    relation_neuron_info = columns[word]['relation_neurons'][next_word]
                    relation_neuron_id = relation_neuron_info['neuron_id']
                    # Reset activation trace counter
                    relation_neuron_info['activation_trace_counter'] = 5

                # Initialize activated relation targets
                if relation_neuron_id not in activated_relation_targets:
                    activated_relation_targets[relation_neuron_id] = []

                # Now find the target word
                for target_idx in range(next_idx+1, len(words)):
                    target_word = words[target_idx]
                    target_pos_tag = pos_tags[target_idx]
                    if not target_word.isalpha():
                        continue
                    if target_pos_tag in concept_pos_list:
                        target_concept_neuron_id = columns[target_word]['concept_neuron']
                        # Connect the relation neuron to the target concept neuron
                        G.add_edge(relation_neuron_id, target_concept_neuron_id, type='relation_target', pos=next_pos_tag)
                        # Store the permanence value for the connection
                        if 'target_connections' not in columns[word]['relation_neurons'][next_word]:
                            columns[word]['relation_neurons'][next_word]['target_connections'] = {}
                        if target_concept_neuron_id not in columns[word]['relation_neurons'][next_word]['target_connections']:
                            columns[word]['relation_neurons'][next_word]['target_connections'][target_concept_neuron_id] = {
                                'permanence': 3,  # Initialized to 3
                                'activation_trace_counter': 5,  # Activation trace counter
                                'first_activation': True
                            }
                        else:
                            # Reset activation trace counter
                            columns[word]['relation_neurons'][next_word]['target_connections'][target_concept_neuron_id]['activation_trace_counter'] = 5
                        # Add to activated targets
                        activated_relation_targets.setdefault(relation_neuron_id, []).append(target_concept_neuron_id)
                        # Now connect the relation neuron to activated instance neurons in the target column
                        if target_word in activated_instances:
                            for target_instance_neuron_id in activated_instances[target_word]:
                                G.add_edge(relation_neuron_id, target_instance_neuron_id, type='relation_target', pos=next_pos_tag)
                                if target_instance_neuron_id not in columns[word]['relation_neurons'][next_word]['target_connections']:
                                    columns[word]['relation_neurons'][next_word]['target_connections'][target_instance_neuron_id] = {
                                        'permanence': 3,  # Initialized to 3
                                        'activation_trace_counter': 5,  # Activation trace counter
                                        'first_activation': True
                                    }
                                else:
                                    # Reset activation trace counter
                                    columns[word]['relation_neurons'][next_word]['target_connections'][target_instance_neuron_id]['activation_trace_counter'] = 5
                                activated_relation_targets[relation_neuron_id].append(target_instance_neuron_id)
                        # Reset activation trace for target concept neuron
                        columns[target_word]['concept_activation_trace_counter'] = 5
                        break  # Only connect to the first concept word after relation word
            else:
                if next_pos_tag in concept_pos_list or next_pos_tag in quality_pos_list:
                    break
                continue

        # Store the relations found for this concept in this sentence
        activated_relations[word] = relations_found

        # Collect instance neurons for this word
        if word in activated_instances:
            instance_neurons = activated_instances[word]
        else:
            instance_neurons = []
        # Relation neurons
        for rel_word in relations_found:
            if rel_word in columns[word]['relation_neurons']:
                rel_neuron_info = columns[word]['relation_neurons'][rel_word]
                rel_neuron_id = rel_neuron_info['neuron_id']
                instance_neurons.append(rel_neuron_id)
                # Reset activation trace counter
                rel_neuron_info['activation_trace_counter'] = 5
        # Update activated_instances[word]
        activated_instances[word] = instance_neurons

    return activated_relations, activated_relation_targets

def update_instance_connections(activated_instances):
    for word, instance_neurons in activated_instances.items():
        if 'instance_connections' not in columns[word]:
            columns[word]['instance_connections'] = {}
        if len(instance_neurons) >= 2:
            for i in range(len(instance_neurons)):
                for j in range(i + 1, len(instance_neurons)):
                    neuron_pair = tuple(sorted((instance_neurons[i], instance_neurons[j])))
                    if neuron_pair not in columns[word]['instance_connections']:
                        columns[word]['instance_connections'][neuron_pair] = {
                            'permanence': 3,  # Initialized to 3
                            'activation_trace_counter': 5,  # Activation trace counter
                            'first_activation': True
                        }
                        # Add edge to graph if not already present
                        if not G.has_edge(neuron_pair[0], neuron_pair[1]):
                            G.add_edge(neuron_pair[0], neuron_pair[1], type='instance_connection')
                    else:
                        # Reset activation trace counter
                        columns[word]['instance_connections'][neuron_pair]['activation_trace_counter'] = 5

def update_permanence_values_concept_neurons(activated_concepts):
    for concept_word, neurons in columns.items():
        if concept_word in activated_concepts:
            neurons['permanence'] += 1  # No change required

def update_permanence_values_relation_neurons(activated_concepts, activated_relations, activated_relation_targets):
    for concept_word, neurons in columns.items():
        if concept_word in activated_concepts:
            active_relations = activated_relations.get(concept_word, [])
            for relation_word, relation_info in list(neurons['relation_neurons'].items()):
                relation_neuron_id = relation_info['neuron_id']
                if relation_word in active_relations:
                    if relation_info.get('first_activation', False):
                        relation_info['first_activation'] = False
                    else:
                        relation_info['permanence'] = relation_info['permanence'] ** 2
                    # Update target connections
                    if 'target_connections' in relation_info:
                        for target_neuron_id, target_conn_info in list(relation_info['target_connections'].items()):
                            if relation_neuron_id in activated_relation_targets and target_neuron_id in activated_relation_targets[relation_neuron_id]:
                                if target_conn_info.get('first_activation', False):
                                    target_conn_info['first_activation'] = False
                                else:
                                    target_conn_info['permanence'] = target_conn_info['permanence'] ** 2
                            else:
                                target_conn_info['permanence'] -= 1
                                if target_conn_info['permanence'] <= 0:
                                    # Remove edge
                                    if G.has_edge(relation_neuron_id, target_neuron_id):
                                        G.remove_edge(relation_neuron_id, target_neuron_id)
                                    del relation_info['target_connections'][target_neuron_id]
                else:
                    # Decrease permanence by 1
                    relation_info['permanence'] -= 1
                    if relation_info['permanence'] <= 0:
                        # Remove relation neuron
                        if G.has_node(relation_neuron_id):
                            G.remove_node(relation_neuron_id)
                        del neurons['relation_neurons'][relation_word]
                    else:
                        # Decrease permanence of target connections
                        if 'target_connections' in relation_info:
                            for target_neuron_id, target_conn_info in list(relation_info['target_connections'].items()):
                                target_conn_info['permanence'] -= 1
                                if target_conn_info['permanence'] <= 0:
                                    # Remove edge
                                    if G.has_edge(relation_neuron_id, target_neuron_id):
                                        G.remove_edge(relation_neuron_id, target_neuron_id)
                                    del relation_info['target_connections'][target_neuron_id]

def update_permanence_values_quality_neurons(activated_concepts, activated_qualities):
    for concept_word, neurons in columns.items():
        if concept_word in activated_concepts:
            active_qualities = activated_qualities.get(concept_word, [])
            for quality_word, quality_info in list(neurons['quality_neurons'].items()):
                if quality_word in active_qualities:
                    if quality_info.get('first_activation', False):
                        quality_info['first_activation'] = False
                    else:
                        quality_info['permanence'] = quality_info['permanence'] ** 2
                else:
                    quality_info['permanence'] -= 1
                    if quality_info['permanence'] <= 0:
                        quality_neuron_id = quality_info['neuron_id']
                        if G.has_node(quality_neuron_id):
                            G.remove_node(quality_neuron_id)
                        del neurons['quality_neurons'][quality_word]

def update_permanence_values_instance_connections(activated_concepts, activated_instances):
    for concept_word, neurons in columns.items():
        if concept_word in activated_concepts:
            active_pairs = set()
            if concept_word in activated_instances:
                instance_neurons = activated_instances[concept_word]
                if len(instance_neurons) >= 2:
                    active_pairs = set(
                        tuple(sorted((instance_neurons[i], instance_neurons[j])))
                        for i in range(len(instance_neurons))
                        for j in range(i + 1, len(instance_neurons))
                    )
            for neuron_pair, connection_info in list(neurons['instance_connections'].items()):
                if neuron_pair in active_pairs:
                    if connection_info.get('first_activation', False):
                        connection_info['first_activation'] = False
                    else:
                        connection_info['permanence'] = connection_info['permanence'] ** 2
                else:
                    # Decrease permanence by 1
                    connection_info['permanence'] -= 1
                    if connection_info['permanence'] <= 0:
                        # Remove the connection
                        if G.has_edge(neuron_pair[0], neuron_pair[1]):
                            G.remove_edge(neuron_pair[0], neuron_pair[1])
                        del neurons['instance_connections'][neuron_pair]

def decrease_activation_trace_counters():
    for concept_word, neurons in columns.items():
        # Concept neuron
        if neurons['concept_activation_trace_counter'] > 0:
            neurons['concept_activation_trace_counter'] -= 1

        # Relation neurons
        for relation_info in neurons['relation_neurons'].values():
            if relation_info['activation_trace_counter'] > 0:
                relation_info['activation_trace_counter'] -= 1
            # Target connections
            if 'target_connections' in relation_info:
                for target_conn_info in relation_info['target_connections'].values():
                    if target_conn_info['activation_trace_counter'] > 0:
                        target_conn_info['activation_trace_counter'] -= 1

        # Quality neurons
        for quality_info in neurons['quality_neurons'].values():
            if quality_info['activation_trace_counter'] > 0:
                quality_info['activation_trace_counter'] -= 1

        # Instance connections
        for connection_info in neurons['instance_connections'].values():
            if connection_info['activation_trace_counter'] > 0:
                connection_info['activation_trace_counter'] -= 1

def process_sentences(sentences):
    global sentence_counter
    for sentence in sentences:
        # Print the sentence
        print(sentence)

        # Process sentence
        doc = nlp(sentence)
        words = [token.text for token in doc]
        pos_tags = [token.pos_ for token in doc]

        # Increase the sentence counter
        sentence_counter += 1

        # Ensure all words are in columns
        ensure_words_in_columns(words)

        # Collect qualities and initialize instance neurons
        activated_concepts, activated_instances, activated_qualities = collect_qualities_and_initialize_instances(words, pos_tags)

        # Process relations
        activated_relations, activated_relation_targets = process_relations(words, pos_tags, activated_instances)

        # Update instance connections within the concept column
        update_instance_connections(activated_instances)

        # Update permanence values for concept neurons
        update_permanence_values_concept_neurons(activated_concepts)

        # Update permanence values for relation neurons and their target connections
        update_permanence_values_relation_neurons(activated_concepts, activated_relations, activated_relation_targets)

        # Update permanence values for quality neurons
        update_permanence_values_quality_neurons(activated_concepts, activated_qualities)

        # Update permanence values for instance connections
        update_permanence_values_instance_connections(activated_concepts, activated_instances)

        # Decrease activation trace counters
        decrease_activation_trace_counters()

        # Visualize the network
        visualize_network(G, columns)

# Call the main processing function
process_sentences(sentences)
