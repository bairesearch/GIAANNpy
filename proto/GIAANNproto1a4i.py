import torch
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches
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
columns = {}  # key: lemma (concept), value: {'concept_neuron', 'permanence', 'relation_neurons', 'quality_neurons', 'instance_connections'}

# Initialize NetworkX graph
G = nx.Graph()

# Global neuron ID counter to ensure unique IDs
neuron_id_counter = 0

# Sentence counter to manage activation traces
sentence_counter = 0

# Lists for POS types
concept_pos_list = ['NOUN', 'PROPN', 'PRON', 'X']  # POS types for concept columns
relation_pos_list = ['VERB', 'ADP', 'CONJ']        # POS types for relation neurons
quality_pos_list = ['DET', 'ADJ']                  # Removed 'ADV' from the list

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

    # Collect all nodes to ensure they get positions
    all_nodes = set()

    # Draw concept columns
    for i, (concept_lemma, neurons) in enumerate(columns.items()):
        # x position is i * x_margin
        x = i * x_margin * 2
        x_positions[concept_lemma] = x
        # Concept neuron at bottom (y=0)
        concept_neuron_id = neurons['concept_neuron']
        pos[concept_neuron_id] = (x, 0)
        labels[concept_neuron_id] = concept_lemma
        all_nodes.add(concept_neuron_id)
        # Collect neuron IDs for coloring
        neuron_ids_in_column = [concept_neuron_id]
        # Relation neurons above
        relation_y_positions = {}
        for j, (relation_word, relation_info) in enumerate(neurons['relation_neurons'].items()):
            relation_neuron_id = relation_info['neuron_id']
            y = (j + 1) * y_margin * 2  # y position for relation neurons
            pos[relation_neuron_id] = (x, y)
            labels[relation_neuron_id] = relation_word
            all_nodes.add(relation_neuron_id)
            relation_y_positions[relation_neuron_id] = y
            if y > max_y:
                max_y = y
            # Feature modifiers for relation neurons
            fm_offset = 0.5  # Offset for feature modifiers
            for k, (fm_lemma, fm_info) in enumerate(relation_info.get('feature_modifiers', {}).items()):
                fm_neuron_id = fm_info['neuron_id']
                y_fm = y + (k + 1) * fm_offset
                pos[fm_neuron_id] = (x, y_fm)
                labels[fm_neuron_id] = fm_lemma
                all_nodes.add(fm_neuron_id)
                if y_fm > max_y:
                    max_y = y_fm
        # Quality neurons above relation neurons
        for k, (quality_lemma, quality_info) in enumerate(neurons['quality_neurons'].items()):
            quality_neuron_id = quality_info['neuron_id']
            y = (len(neurons['relation_neurons']) + k + 1) * y_margin * 2
            pos[quality_neuron_id] = (x, y)
            labels[quality_neuron_id] = quality_lemma
            all_nodes.add(quality_neuron_id)
            if y > max_y:
                max_y = y
            # Feature modifiers for quality neurons
            fm_offset = 0.5  # Offset for feature modifiers
            for l, (fm_lemma, fm_info) in enumerate(quality_info.get('feature_modifiers', {}).items()):
                fm_neuron_id = fm_info['neuron_id']
                y_fm = y + (l + 1) * fm_offset
                pos[fm_neuron_id] = (x, y_fm)
                labels[fm_neuron_id] = fm_lemma
                all_nodes.add(fm_neuron_id)
                if y_fm > max_y:
                    max_y = y_fm

        # Instance connections are edges, so no positions needed for them

    # Ensure all nodes in G have positions
    missing_nodes = set(G.nodes()) - set(pos.keys())
    if missing_nodes:
        # For any nodes not assigned a position (e.g., concept neurons from other columns connected via definitions),
        # assign them default positions off to the side
        offset_x = (len(columns) + 1) * x_margin * 2
        y_offset = 0
        for node_id in missing_nodes:
            pos[node_id] = (offset_x, y_offset)
            labels[node_id] = f"Node {node_id}"
            y_offset += y_margin * 2
            all_nodes.add(node_id)

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
                        relation_word = relation_info['word']
                        if relation_word == 'have':
                            node_colors.append('cyan')  # 'have' auxiliary action relations
                        elif relation_pos == 'VERB':
                            node_colors.append('green')  # Action
                        elif relation_pos == 'ADP':
                            node_colors.append('red')    # Condition
                        elif relation_pos == 'CONJ':
                            node_colors.append('green')  # Conjunctions colored as action
                        else:
                            node_colors.append('gray')
                        break
        elif any(node_id in [info['neuron_id'] for info in neurons['quality_neurons'].values()] for neurons in columns.values()):
            node_colors.append('turquoise')  # Quality neurons
        elif any(node_id in [fm_info['neuron_id'] for relation_info in neurons['relation_neurons'].values() for fm_info in relation_info.get('feature_modifiers', {}).values()] for neurons in columns.values()):
            node_colors.append('lightblue')  # Feature modifiers for relations
        elif any(node_id in [fm_info['neuron_id'] for quality_info in neurons['quality_neurons'].values() for fm_info in quality_info.get('feature_modifiers', {}).values()] for neurons in columns.values()):
            node_colors.append('lightblue')  # Feature modifiers for qualities
        else:
            node_colors.append('gray')

    # Draw edges
    edge_colors = []
    edge_styles = []
    for u, v in G.edges():
        edge = G.get_edge_data(u, v)
        if edge['type'] == 'relation_target':
            if edge.get('word') == 'have':
                edge_colors.append('cyan')
            elif edge['pos'] == 'VERB':
                edge_colors.append('green')
            elif edge['pos'] == 'ADP':
                edge_colors.append('red')
            elif edge['pos'] == 'CONJ':
                edge_colors.append('green')
            else:
                edge_colors.append('gray')
            edge_styles.append('solid')
        elif edge['type'] == 'concept_source':
            edge_colors.append('blue')  # All concept_source edges are colored blue
            edge_styles.append('solid')
        elif edge['type'] == 'definition':
            edge_colors.append('darkblue')  # Definition connections
            edge_styles.append('solid')
        elif edge['type'] == 'instance_connection':
            edge_colors.append('yellow')
            edge_styles.append('solid')
        elif edge['type'] == 'internal_relation':
            if edge.get('pos') == 'VERB':
                edge_colors.append('green')
            elif edge.get('pos') == 'ADP':
                edge_colors.append('red')
            elif edge.get('pos') == 'CONJ':
                edge_colors.append('green')
            else:
                edge_colors.append('gray')
            edge_styles.append('dashed')
        elif edge['type'] == 'feature_modifier':
            edge_colors.append('lightblue')
            edge_styles.append('solid')
        else:
            edge_colors.append('gray')
            edge_styles.append('solid')

    nx.draw_networkx_nodes(G, pos, node_size=neuron_size, node_color=node_colors)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, style=edge_styles)
    nx.draw_networkx_labels(G, pos, labels, font_size=8)

    # Draw rectangles around columns
    for concept_lemma, x in x_positions.items():
        # Get y positions of neurons in this column
        neurons = columns[concept_lemma]
        y_positions = [pos[neurons['concept_neuron']][1]]
        for relation_info in neurons['relation_neurons'].values():
            if relation_info['column'] == concept_lemma:
                y_positions.append(pos[relation_info['neuron_id']][1])
                for fm_info in relation_info.get('feature_modifiers', {}).values():
                    y_positions.append(pos[fm_info['neuron_id']][1])
        for quality_info in neurons['quality_neurons'].values():
            y_positions.append(pos[quality_info['neuron_id']][1])
            for fm_info in quality_info.get('feature_modifiers', {}).values():
                y_positions.append(pos[fm_info['neuron_id']][1])
        y_min = min(y_positions) - y_margin
        y_max = max(y_positions) + y_margin
        # Draw rectangle
        plt.gca().add_patch(plt.Rectangle((x - x_margin, y_min), x_margin * 2, y_max - y_min + y_margin, fill=False, edgecolor='black'))
    plt.axis('off')
    plt.show()

def ensure_words_in_columns(lemmas, tokens):
    global neuron_id_counter
    for lemma, token in zip(lemmas, tokens):
        pos_tag = token.pos_
        dep_tag = token.dep_
        tag = token.tag_
        word = token.text

        # Ignore "be" and "do" auxiliaries
        if dep_tag == 'aux' and lemma in ['be', 'do']:
            continue

        # Convert possessive clitic to "have"
        if tag == 'POS':
            lemma = 'have'
            pos_tag = 'VERB'
            dep_tag = 'aux'

        if not lemma.isalpha():
            continue
        if lemma not in columns:
            neuron_id_counter += 1
            concept_neuron_id = neuron_id_counter
            columns[lemma] = {
                'concept_neuron': concept_neuron_id,
                'permanence': 3,  # Initialized to 3
                'concept_activation_trace_counter': 0,  # Activation trace counter
                'relation_neurons': {},
                'quality_neurons': {},
                'instance_connections': {}
            }
            # Add the concept neuron to the graph
            G.add_node(concept_neuron_id)

def collect_qualities_and_initialize_instances(lemmas, pos_tags, tokens):
    global neuron_id_counter
    activated_concepts = {}
    activated_instances = {}
    activated_qualities = {}
    for idx, (lemma, pos_tag) in enumerate(zip(lemmas, pos_tags)):
        if not lemma.isalpha():
            continue

        activated_concepts[lemma] = True

        # Set activation trace for concept neuron
        columns[lemma]['concept_activation_trace_counter'] = 5  # Reset to 5

        # Collect qualities for this lemma
        qualities_found = []
        # Check previous lemma
        if idx > 0:
            prev_lemma = lemmas[idx - 1]
            prev_pos_tag = pos_tags[idx - 1]
            prev_token = tokens[idx - 1]
            if prev_pos_tag in quality_pos_list:
                intervening_pos = pos_tags[idx - 1:idx]
                if not any(pos in relation_pos_list for pos in intervening_pos):
                    qualities_found.append(prev_lemma)
                    # Ensure the quality neuron exists in the column
                    if prev_lemma not in columns[lemma]['quality_neurons']:
                        neuron_id_counter += 1
                        quality_neuron_id = neuron_id_counter
                        columns[lemma]['quality_neurons'][prev_lemma] = {
                            'neuron_id': quality_neuron_id,
                            'permanence': 3,  # Initialized to 3
                            'activation_trace_counter': 5,  # Activation trace counter
                            'pos': prev_pos_tag,
                            'first_activation': True
                        }
                        # Add the quality neuron to the graph
                        G.add_node(quality_neuron_id)
                        # Ensure prev_lemma has a concept neuron
                        ensure_words_in_columns([prev_lemma], [prev_token])
                        prev_lemma_concept_id = columns[prev_lemma]['concept_neuron']
                        # Draw concept source connection
                        G.add_edge(prev_lemma_concept_id, quality_neuron_id, type='concept_source')
                    else:
                        quality_neuron_id = columns[lemma]['quality_neurons'][prev_lemma]['neuron_id']
                        # Reset activation trace counter
                        columns[lemma]['quality_neurons'][prev_lemma]['activation_trace_counter'] = 5
                    # Process feature modifiers for the quality neuron
                    process_feature_modifiers_for_quality(idx - 1, tokens, columns[lemma]['quality_neurons'][prev_lemma])

        # Check next lemma
        if idx < len(lemmas) - 1:
            next_lemma = lemmas[idx + 1]
            next_pos_tag = pos_tags[idx + 1]
            next_token = tokens[idx + 1]
            if next_pos_tag in quality_pos_list:
                intervening_pos = pos_tags[idx + 1:idx + 2]
                if not any(pos in relation_pos_list for pos in intervening_pos):
                    qualities_found.append(next_lemma)
                    if next_lemma not in columns[lemma]['quality_neurons']:
                        neuron_id_counter += 1
                        quality_neuron_id = neuron_id_counter
                        columns[lemma]['quality_neurons'][next_lemma] = {
                            'neuron_id': quality_neuron_id,
                            'permanence': 3,  # Initialized to 3
                            'activation_trace_counter': 5,  # Activation trace counter
                            'pos': next_pos_tag,
                            'first_activation': True
                        }
                        # Add the quality neuron to the graph
                        G.add_node(quality_neuron_id)
                        # Ensure next_lemma has a concept neuron
                        ensure_words_in_columns([next_lemma], [next_token])
                        next_lemma_concept_id = columns[next_lemma]['concept_neuron']
                        # Draw concept source connection
                        G.add_edge(next_lemma_concept_id, quality_neuron_id, type='concept_source')
                    else:
                        quality_neuron_id = columns[lemma]['quality_neurons'][next_lemma]['neuron_id']
                        # Reset activation trace counter
                        columns[lemma]['quality_neurons'][next_lemma]['activation_trace_counter'] = 5
                    # Process feature modifiers for the quality neuron
                    process_feature_modifiers_for_quality(idx + 1, tokens, columns[lemma]['quality_neurons'][next_lemma])

        # Store the qualities found for this concept in this sentence
        activated_qualities[lemma] = qualities_found

        # Collect instance neurons for this word
        instance_neurons = []
        # Quality neurons
        for qual_lemma in qualities_found:
            if qual_lemma in columns[lemma]['quality_neurons']:
                qual_neuron_info = columns[lemma]['quality_neurons'][qual_lemma]
                qual_neuron_id = qual_neuron_info['neuron_id']
                instance_neurons.append(qual_neuron_id)
                # Reset activation trace counter
                qual_neuron_info['activation_trace_counter'] = 5

        # Store activated instance neurons for this concept
        activated_instances[lemma] = instance_neurons

    return activated_concepts, activated_instances, activated_qualities

def process_feature_modifiers_for_quality(quality_idx, tokens, quality_info):
    global neuron_id_counter

    if 'feature_modifiers' not in quality_info:
        quality_info['feature_modifiers'] = {}

    # Check previous token
    if quality_idx > 0:
        prev_token = tokens[quality_idx - 1]
        if prev_token.pos_ == 'ADV':
            modifier_lemma = prev_token.lemma_
            modifier_pos = prev_token.pos_
            # Process feature modifier
            if modifier_lemma not in quality_info['feature_modifiers']:
                neuron_id_counter += 1
                feature_modifier_neuron_id = neuron_id_counter
                quality_info['feature_modifiers'][modifier_lemma] = {
                    'neuron_id': feature_modifier_neuron_id,
                    'permanence': 3,
                    'activation_trace_counter': 5,
                    'first_activation': True,
                    'pos': modifier_pos
                }
                G.add_node(feature_modifier_neuron_id)
                # Add edge from feature modifier neuron to quality neuron
                G.add_edge(feature_modifier_neuron_id, quality_info['neuron_id'], type='feature_modifier')
            else:
                feature_modifier_info = quality_info['feature_modifiers'][modifier_lemma]
                # Reset activation trace counter
                feature_modifier_info['activation_trace_counter'] = 5

    # Check next token
    if quality_idx < len(tokens) - 1:
        next_token = tokens[quality_idx + 1]
        if next_token.pos_ == 'ADV':
            modifier_lemma = next_token.lemma_
            modifier_pos = next_token.pos_
            # Process feature modifier
            if modifier_lemma not in quality_info['feature_modifiers']:
                neuron_id_counter += 1
                feature_modifier_neuron_id = neuron_id_counter
                quality_info['feature_modifiers'][modifier_lemma] = {
                    'neuron_id': feature_modifier_neuron_id,
                    'permanence': 3,
                    'activation_trace_counter': 5,
                    'first_activation': True,
                    'pos': modifier_pos
                }
                G.add_node(feature_modifier_neuron_id)
                # Add edge from feature modifier neuron to quality neuron
                G.add_edge(feature_modifier_neuron_id, quality_info['neuron_id'], type='feature_modifier')
            else:
                feature_modifier_info = quality_info['feature_modifiers'][modifier_lemma]
                # Reset activation trace counter
                feature_modifier_info['activation_trace_counter'] = 5

def process_relations(lemmas, pos_tags, tokens, activated_instances):
    global neuron_id_counter
    activated_relations = {}
    activated_relation_targets = {}
    for idx, (lemma, pos_tag) in enumerate(zip(lemmas, pos_tags)):
        if not lemma.isalpha():
            continue

        if pos_tag not in concept_pos_list:
            continue

        # Start processing relations after the concept word
        curr_idx = idx + 1
        relation_sequence = []
        while curr_idx < len(lemmas):
            next_lemma = lemmas[curr_idx]
            next_pos_tag = pos_tags[curr_idx]
            next_token = tokens[curr_idx]
            next_word = next_token.text  # Get the original word
            if next_pos_tag in relation_pos_list:
                # Ignore "be" and "do" auxiliaries
                if next_lemma in ['be', 'do'] and next_token.dep_ == 'aux':
                    curr_idx += 1
                    continue
                relation_sequence.append(curr_idx)
                curr_idx += 1
            else:
                break

        # Process the relation sequence
        if not relation_sequence:
            continue

        # Initialize current_column_lemma as the current concept lemma
        current_column_lemma = lemma

        for i in range(len(relation_sequence)):
            relation_idx = relation_sequence[i]
            relation_lemma = lemmas[relation_idx]
            relation_pos_tag = pos_tags[relation_idx]
            relation_token = tokens[relation_idx]
            relation_word = relation_token.text  # Use original word

            # Determine the concept column to which this relation neuron belongs
            if i == 0:
                # First relation word after the noun
                if len(relation_sequence) == 1:
                    # Only one relation word, connect to noun's concept neuron
                    concept_lemma = lemma
                    # Ensure the relation neuron exists in the noun's concept column
                    if relation_word not in columns[concept_lemma]['relation_neurons']:
                        neuron_id_counter += 1
                        relation_neuron_id = neuron_id_counter
                        columns[concept_lemma]['relation_neurons'][relation_word] = {
                            'neuron_id': relation_neuron_id,
                            'permanence': 3,
                            'activation_trace_counter': 5,
                            'pos': relation_pos_tag,
                            'word': relation_word,
                            'target_connections': {},
                            'first_activation': True,
                            'column': concept_lemma  # Store the column it belongs to
                        }
                        relation_info = columns[concept_lemma]['relation_neurons'][relation_word]
                        G.add_node(relation_neuron_id)
                        # Draw concept source connection
                        ensure_words_in_columns([relation_lemma], [relation_token])
                        relation_lemma_concept_id = columns[relation_lemma]['concept_neuron']
                        columns[relation_lemma]['concept_activation_trace_counter'] = 5
                        G.add_edge(relation_lemma_concept_id, relation_neuron_id, type='concept_source')
                        # Connect to noun's concept neuron
                        concept_neuron_id = columns[concept_lemma]['concept_neuron']
                        G.add_edge(concept_neuron_id, relation_neuron_id, type='relation_target', pos=relation_pos_tag, word=relation_word)
                    else:
                        relation_info = columns[concept_lemma]['relation_neurons'][relation_word]
                        relation_neuron_id = relation_info['neuron_id']
                        # Reset activation trace counter
                        relation_info['activation_trace_counter'] = 5
                        # Connect to noun's concept neuron
                        concept_neuron_id = columns[concept_lemma]['concept_neuron']
                        G.add_edge(concept_neuron_id, relation_neuron_id, type='relation_target', pos=relation_pos_tag, word=relation_word)
                    # Process feature modifiers for the relation neuron
                    process_feature_modifiers_for_relation(relation_idx, tokens, relation_info)
                else:
                    # More than one relation word, connect to next relation word
                    next_relation_idx = relation_sequence[i + 1]
                    next_relation_lemma = lemmas[next_relation_idx]
                    next_relation_pos_tag = pos_tags[next_relation_idx]
                    next_relation_token = tokens[next_relation_idx]
                    next_relation_word = next_relation_token.text

                    # Ensure current relation neuron exists in columns
                    if relation_word not in columns[current_column_lemma]['relation_neurons']:
                        neuron_id_counter += 1
                        relation_neuron_id = neuron_id_counter
                        columns[current_column_lemma]['relation_neurons'][relation_word] = {
                            'neuron_id': relation_neuron_id,
                            'permanence': 3,
                            'activation_trace_counter': 5,
                            'pos': relation_pos_tag,
                            'word': relation_word,
                            'target_connections': {},
                            'first_activation': True,
                            'column': current_column_lemma
                        }
                        relation_info = columns[current_column_lemma]['relation_neurons'][relation_word]
                        G.add_node(relation_neuron_id)
                        # Draw concept source connection
                        ensure_words_in_columns([relation_lemma], [relation_token])
                        relation_lemma_concept_id = columns[relation_lemma]['concept_neuron']
                        columns[relation_lemma]['concept_activation_trace_counter'] = 5
                        G.add_edge(relation_lemma_concept_id, relation_neuron_id, type='concept_source')
                    else:
                        relation_info = columns[current_column_lemma]['relation_neurons'][relation_word]
                        relation_neuron_id = relation_info['neuron_id']
                        relation_info['activation_trace_counter'] = 5

                    # Process feature modifiers for the relation neuron
                    process_feature_modifiers_for_relation(relation_idx, tokens, relation_info)

                    # Ensure next relation neuron exists in columns
                    if next_relation_word not in columns[current_column_lemma]['relation_neurons']:
                        neuron_id_counter += 1
                        next_relation_neuron_id = neuron_id_counter
                        columns[current_column_lemma]['relation_neurons'][next_relation_word] = {
                            'neuron_id': next_relation_neuron_id,
                            'permanence': 3,
                            'activation_trace_counter': 5,
                            'pos': next_relation_pos_tag,
                            'word': next_relation_word,
                            'target_connections': {},
                            'first_activation': True,
                            'column': current_column_lemma
                        }
                        next_relation_info = columns[current_column_lemma]['relation_neurons'][next_relation_word]
                        G.add_node(next_relation_neuron_id)
                        # Draw concept source connection
                        ensure_words_in_columns([next_relation_lemma], [next_relation_token])
                        next_relation_lemma_concept_id = columns[next_relation_lemma]['concept_neuron']
                        columns[next_relation_lemma]['concept_activation_trace_counter'] = 5
                        G.add_edge(next_relation_lemma_concept_id, next_relation_neuron_id, type='concept_source')
                    else:
                        next_relation_info = columns[current_column_lemma]['relation_neurons'][next_relation_word]
                        next_relation_neuron_id = next_relation_info['neuron_id']
                        next_relation_info['activation_trace_counter'] = 5

                    # Process feature modifiers for the next relation neuron
                    process_feature_modifiers_for_relation(next_relation_idx, tokens, next_relation_info)

                    # Connect current relation to next relation (internal action/condition relation)
                    G.add_edge(relation_neuron_id, next_relation_neuron_id, type='internal_relation', pos=relation_pos_tag)
                    # Update current_column_lemma to the current relation word
                    current_column_lemma = current_column_lemma
            else:
                # Subsequent relation words
                previous_relation_idx = relation_sequence[i - 1]
                previous_relation_lemma = lemmas[previous_relation_idx]
                previous_relation_pos_tag = pos_tags[previous_relation_idx]
                previous_relation_token = tokens[previous_relation_idx]
                previous_relation_word = previous_relation_token.text

                # Ensure current relation neuron exists in columns
                if relation_word not in columns[current_column_lemma]['relation_neurons']:
                    neuron_id_counter += 1
                    relation_neuron_id = neuron_id_counter
                    columns[current_column_lemma]['relation_neurons'][relation_word] = {
                        'neuron_id': relation_neuron_id,
                        'permanence': 3,
                        'activation_trace_counter': 5,
                        'pos': relation_pos_tag,
                        'word': relation_word,
                        'target_connections': {},
                        'first_activation': True,
                        'column': current_column_lemma
                    }
                    relation_info = columns[current_column_lemma]['relation_neurons'][relation_word]
                    G.add_node(relation_neuron_id)
                    # Draw concept source connection
                    ensure_words_in_columns([relation_lemma], [relation_token])
                    relation_lemma_concept_id = columns[relation_lemma]['concept_neuron']
                    columns[relation_lemma]['concept_activation_trace_counter'] = 5
                    G.add_edge(relation_lemma_concept_id, relation_neuron_id, type='concept_source')
                else:
                    relation_info = columns[current_column_lemma]['relation_neurons'][relation_word]
                    relation_neuron_id = relation_info['neuron_id']
                    relation_info['activation_trace_counter'] = 5

                # Process feature modifiers for the relation neuron
                process_feature_modifiers_for_relation(relation_idx, tokens, relation_info)

                # Ensure previous relation neuron exists
                previous_relation_info = columns[current_column_lemma]['relation_neurons'][previous_relation_word]
                previous_relation_neuron_id = previous_relation_info['neuron_id']

                # Connect previous relation to current relation (internal action/condition relation)
                G.add_edge(previous_relation_neuron_id, relation_neuron_id, type='internal_relation', pos=relation_pos_tag)

                if i == len(relation_sequence) - 1:
                    # Last relation word, connect to noun's concept neuron
                    concept_neuron_id = columns[lemma]['concept_neuron']
                    G.add_edge(concept_neuron_id, relation_neuron_id, type='relation_target', pos=relation_pos_tag, word=relation_word)
                else:
                    # Connect current relation to next relation
                    next_relation_idx = relation_sequence[i + 1]
                    next_relation_lemma = lemmas[next_relation_idx]
                    next_relation_pos_tag = pos_tags[next_relation_idx]
                    next_relation_token = tokens[next_relation_idx]
                    next_relation_word = next_relation_token.text

                    # Ensure next relation neuron exists in columns
                    if next_relation_word not in columns[current_column_lemma]['relation_neurons']:
                        neuron_id_counter += 1
                        next_relation_neuron_id = neuron_id_counter
                        columns[current_column_lemma]['relation_neurons'][next_relation_word] = {
                            'neuron_id': next_relation_neuron_id,
                            'permanence': 3,
                            'activation_trace_counter': 5,
                            'pos': next_relation_pos_tag,
                            'word': next_relation_word,
                            'target_connections': {},
                            'first_activation': True,
                            'column': current_column_lemma
                        }
                        next_relation_info = columns[current_column_lemma]['relation_neurons'][next_relation_word]
                        G.add_node(next_relation_neuron_id)
                        # Draw concept source connection
                        ensure_words_in_columns([next_relation_lemma], [next_relation_token])
                        next_relation_lemma_concept_id = columns[next_relation_lemma]['concept_neuron']
                        columns[next_relation_lemma]['concept_activation_trace_counter'] = 5
                        G.add_edge(next_relation_lemma_concept_id, next_relation_neuron_id, type='concept_source')
                    else:
                        next_relation_info = columns[current_column_lemma]['relation_neurons'][next_relation_word]
                        next_relation_neuron_id = next_relation_info['neuron_id']
                        next_relation_info['activation_trace_counter'] = 5

                    # Process feature modifiers for the next relation neuron
                    process_feature_modifiers_for_relation(next_relation_idx, tokens, next_relation_info)

                    # Connect current relation to next relation (internal action/condition relation)
                    G.add_edge(relation_neuron_id, next_relation_neuron_id, type='internal_relation', pos=relation_pos_tag)

    return activated_relations, activated_relation_targets

def process_feature_modifiers_for_relation(relation_idx, tokens, relation_info):
    global neuron_id_counter

    if 'feature_modifiers' not in relation_info:
        relation_info['feature_modifiers'] = {}

    # Check previous token
    if relation_idx > 0:
        prev_token = tokens[relation_idx - 1]
        if prev_token.pos_ == 'ADV':
            modifier_lemma = prev_token.lemma_
            modifier_pos = prev_token.pos_
            # Process feature modifier
            if modifier_lemma not in relation_info['feature_modifiers']:
                neuron_id_counter += 1
                feature_modifier_neuron_id = neuron_id_counter
                relation_info['feature_modifiers'][modifier_lemma] = {
                    'neuron_id': feature_modifier_neuron_id,
                    'permanence': 3,
                    'activation_trace_counter': 5,
                    'first_activation': True,
                    'pos': modifier_pos
                }
                G.add_node(feature_modifier_neuron_id)
                # Add edge from feature modifier neuron to relation neuron
                G.add_edge(feature_modifier_neuron_id, relation_info['neuron_id'], type='feature_modifier')
            else:
                feature_modifier_info = relation_info['feature_modifiers'][modifier_lemma]
                # Reset activation trace counter
                feature_modifier_info['activation_trace_counter'] = 5

    # Check next token
    if relation_idx < len(tokens) - 1:
        next_token = tokens[relation_idx + 1]
        if next_token.pos_ == 'ADV':
            modifier_lemma = next_token.lemma_
            modifier_pos = next_token.pos_
            # Process feature modifier
            if modifier_lemma not in relation_info['feature_modifiers']:
                neuron_id_counter += 1
                feature_modifier_neuron_id = neuron_id_counter
                relation_info['feature_modifiers'][modifier_lemma] = {
                    'neuron_id': feature_modifier_neuron_id,
                    'permanence': 3,
                    'activation_trace_counter': 5,
                    'first_activation': True,
                    'pos': modifier_pos
                }
                G.add_node(feature_modifier_neuron_id)
                # Add edge from feature modifier neuron to relation neuron
                G.add_edge(feature_modifier_neuron_id, relation_info['neuron_id'], type='feature_modifier')
            else:
                feature_modifier_info = relation_info['feature_modifiers'][modifier_lemma]
                # Reset activation trace counter
                feature_modifier_info['activation_trace_counter'] = 5

def process_definitions(tokens):
    lemmas = [token.lemma_ for token in tokens]
    pos_tags = [token.pos_ for token in tokens]
    dep_tags = [token.dep_ for token in tokens]

    for idx, token in enumerate(tokens):
        lemma = token.lemma_
        pos_tag = token.pos_
        if not lemma.isalpha():
            continue
        # Check if current lemma is a concept lemma
        if pos_tag in concept_pos_list:
            # Look ahead for 'be' auxiliary lemma
            for next_idx in range(idx+1, len(tokens)):
                next_token = tokens[next_idx]
                next_lemma = next_token.lemma_
                next_pos_tag = next_token.pos_
                if not next_lemma.isalpha():
                    continue
                if next_pos_tag in concept_pos_list:
                    # Found another concept lemma without an intermediary action/condition relation
                    # Now check if there was a 'be' auxiliary lemma in between
                    be_found = False
                    intermediary_found = False
                    for in_between_idx in range(idx+1, next_idx):
                        in_between_token = tokens[in_between_idx]
                        in_between_lemma = in_between_token.lemma_
                        in_between_pos = in_between_token.pos_
                        in_between_dep = in_between_token.dep_
                        if in_between_lemma == 'be' and in_between_dep == 'aux':
                            be_found = True
                        elif in_between_pos in relation_pos_list:
                            intermediary_found = True
                            break
                    if be_found and not intermediary_found:
                        # Create definition connection between lemma and next_lemma
                        concept_neuron_id_1 = columns[lemma]['concept_neuron']
                        concept_neuron_id_2 = columns[next_lemma]['concept_neuron']
                        # Add edge if not already present
                        if not G.has_edge(concept_neuron_id_1, concept_neuron_id_2):
                            G.add_edge(concept_neuron_id_1, concept_neuron_id_2, type='definition')
                        break  # Only connect to the first concept word satisfying the condition
                    elif intermediary_found:
                        break  # An intermediary relation is found, so break
                elif next_pos_tag in relation_pos_list:
                    break  # An intermediary action/condition relation is found, so break
                else:
                    continue

def update_instance_connections(activated_instances):
    for lemma, instance_neurons in activated_instances.items():
        if 'instance_connections' not in columns[lemma]:
            columns[lemma]['instance_connections'] = {}
        if len(instance_neurons) >= 2:
            for i in range(len(instance_neurons)):
                for j in range(i + 1, len(instance_neurons)):
                    neuron_pair = tuple(sorted((instance_neurons[i], instance_neurons[j])))
                    if neuron_pair not in columns[lemma]['instance_connections']:
                        columns[lemma]['instance_connections'][neuron_pair] = {
                            'permanence': 3,  # Initialized to 3
                            'activation_trace_counter': 5,  # Activation trace counter
                            'first_activation': True
                        }
                        # Add edge to graph if not already present
                        if not G.has_edge(neuron_pair[0], neuron_pair[1]):
                            G.add_edge(neuron_pair[0], neuron_pair[1], type='instance_connection')
                    else:
                        # Reset activation trace counter
                        columns[lemma]['instance_connections'][neuron_pair]['activation_trace_counter'] = 5

def update_permanence_values_concept_neurons(activated_concepts):
    for concept_lemma, neurons in columns.items():
        if concept_lemma in activated_concepts:
            neurons['permanence'] += 1  # No change required

def update_permanence_values_relation_neurons(activated_concepts, activated_relations, activated_relation_targets):
    for concept_lemma, neurons in columns.items():
        if concept_lemma in activated_concepts:
            active_relations = activated_relations.get(concept_lemma, [])
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
                    # Update feature modifiers
                    update_permanence_values_feature_modifiers(relation_info['feature_modifiers'])
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
                        # Decrease permanence of feature modifiers
                        update_permanence_values_feature_modifiers(relation_info['feature_modifiers'], decrease=True)

def update_permanence_values_quality_neurons(activated_concepts, activated_qualities):
    for concept_lemma, neurons in columns.items():
        if concept_lemma in activated_concepts:
            active_qualities = activated_qualities.get(concept_lemma, [])
            for quality_lemma, quality_info in list(neurons['quality_neurons'].items()):
                if quality_lemma in active_qualities:
                    if quality_info.get('first_activation', False):
                        quality_info['first_activation'] = False
                    else:
                        quality_info['permanence'] = quality_info['permanence'] ** 2
                    # Update feature modifiers
                    update_permanence_values_feature_modifiers(quality_info['feature_modifiers'])
                else:
                    quality_info['permanence'] -= 1
                    if quality_info['permanence'] <= 0:
                        quality_neuron_id = quality_info['neuron_id']
                        if G.has_node(quality_neuron_id):
                            G.remove_node(quality_neuron_id)
                        del neurons['quality_neurons'][quality_lemma]
                    else:
                        # Decrease permanence of feature modifiers
                        update_permanence_values_feature_modifiers(quality_info['feature_modifiers'], decrease=True)

def update_permanence_values_feature_modifiers(feature_modifiers, decrease=False):
    for fm_lemma, fm_info in list(feature_modifiers.items()):
        if decrease:
            fm_info['permanence'] -= 1
            if fm_info['permanence'] <= 0:
                # Remove node and edge
                fm_neuron_id = fm_info['neuron_id']
                if G.has_node(fm_neuron_id):
                    G.remove_node(fm_neuron_id)
                del feature_modifiers[fm_lemma]
        else:
            if fm_info.get('first_activation', False):
                fm_info['first_activation'] = False
            else:
                fm_info['permanence'] = fm_info['permanence'] ** 2

def update_permanence_values_instance_connections(activated_concepts, activated_instances):
    for concept_lemma, neurons in columns.items():
        if concept_lemma in activated_concepts:
            active_pairs = set()
            if concept_lemma in activated_instances:
                instance_neurons = activated_instances[concept_lemma]
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
    for concept_lemma, neurons in columns.items():
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
            # Feature modifiers
            for fm_info in relation_info.get('feature_modifiers', {}).values():
                if fm_info['activation_trace_counter'] > 0:
                    fm_info['activation_trace_counter'] -= 1

        # Quality neurons
        for quality_info in neurons['quality_neurons'].values():
            if quality_info['activation_trace_counter'] > 0:
                quality_info['activation_trace_counter'] -= 1
            # Feature modifiers
            for fm_info in quality_info.get('feature_modifiers', {}).values():
                if fm_info['activation_trace_counter'] > 0:
                    fm_info['activation_trace_counter'] -= 1

        # Instance connections
        for connection_info in neurons['instance_connections'].values():
            if connection_info['activation_trace_counter'] > 0:
                connection_info['activation_trace_counter'] -= 1

def draw_dependency_tree(sentence, tokens, columns, G):
    # Extract lemmas and positions
    lemmas = [token.lemma_ for token in tokens]
    pos_tags = [token.pos_ for token in tokens]
    dep_tags = [token.dep_ for token in tokens]
    tags = [token.tag_ for token in tokens]
    words = [token.text for token in tokens]

    # Prepare the plot
    plt.figure(figsize=(12, 6))

    # Positions for nodes on x-axis
    x_positions = list(range(len(lemmas)))
    y_position = 0  # All nodes on the same vertical height

    # Determine node colors
    node_colors = []
    for idx, lemma in enumerate(lemmas):
        pos_tag = pos_tags[idx]
        if pos_tag in concept_pos_list:
            node_colors.append('blue')  # Concept node
        elif pos_tag in relation_pos_list:
            if lemma == 'have':
                node_colors.append('cyan')
            elif pos_tag == 'VERB':
                node_colors.append('green')
            elif pos_tag == 'ADP':
                node_colors.append('red')
            elif pos_tag == 'CONJ':
                node_colors.append('green')
            else:
                node_colors.append('gray')
        elif pos_tag in quality_pos_list:
            node_colors.append('turquoise')
        else:
            node_colors.append('gray')

    # Draw nodes as circles without edge colors
    node_size = 750 / (1.5 ** 2)
    plt.scatter(x_positions, [y_position]*len(lemmas), s=node_size, c=node_colors)

    # Draw lemma text inside the nodes with reduced font size
    for idx, word in enumerate(words):
        plt.text(x_positions[idx], y_position, word, fontsize=8, ha='center', va='center', color='black')

    # Now determine the dependencies
    # We will store dependencies as tuples: (head_idx, dep_idx, dep_type, color)
    dependencies = []

    # For each concept lemma in the sentence
    for idx, lemma in enumerate(lemmas):
        token_pos = pos_tags[idx]
        if lemma not in columns:
            continue
        neurons = columns[lemma]
        # Process qualities
        for quality_lemma, quality_info in neurons.get('quality_neurons', {}).items():
            if quality_lemma in lemmas:
                quality_idx = lemmas.index(quality_lemma)
                # Check permanence value
                permanence = quality_info.get('permanence', 0)
                if permanence >= 3:
                    # Add dependency
                    dependencies.append((idx, quality_idx, 'quality', 'turquoise'))

        # Process relations
        for relation_word, relation_info in neurons.get('relation_neurons', {}).items():
            if relation_word in words:
                relation_idx = words.index(relation_word)
                # Check permanence value
                permanence = relation_info.get('permanence', 0)
                if permanence >= 3:
                    # Add dependency from concept to relation
                    # Determine color
                    pos = relation_info.get('pos', '')
                    if relation_word == 'have':
                        color = 'cyan'
                    elif pos == 'VERB':
                        color = 'green'
                    elif pos == 'ADP':
                        color = 'red'
                    elif pos == 'CONJ':
                        color = 'green'
                    else:
                        color = 'gray'
                    dependencies.append((idx, relation_idx, 'relation', color))
                    # Now check target connections
                    relation_neuron_id = relation_info['neuron_id']
                    if 'target_connections' in relation_info:
                        for target_neuron_id, target_conn_info in relation_info['target_connections'].items():
                            target_lemma = None
                            # Find the lemma corresponding to target_neuron_id
                            for i, l in enumerate(lemmas):
                                if l in columns and columns[l]['concept_neuron'] == target_neuron_id:
                                    target_lemma = l
                                    target_idx = i
                                    break
                            if target_lemma is not None:
                                # Check permanence
                                target_perm = target_conn_info.get('permanence', 0)
                                if target_perm >= 3:
                                    # Add dependency from relation to target concept
                                    dependencies.append((relation_idx, target_idx, 'relation_target', color))
        # Process definitions
        concept_neuron_id = neurons['concept_neuron']
        for other_idx, other_lemma in enumerate(lemmas):
            if other_lemma != lemma and other_lemma in columns:
                other_concept_neuron_id = columns[other_lemma]['concept_neuron']
                if G.has_edge(concept_neuron_id, other_concept_neuron_id):
                    edge_data = G.get_edge_data(concept_neuron_id, other_concept_neuron_id)
                    if edge_data.get('type') == 'definition':
                        dependencies.append((idx, other_idx, 'definition', 'darkblue'))

    # Now, draw the dependencies
    for head_idx, dep_idx, dep_type, color in dependencies:
        x1 = x_positions[head_idx]
        x2 = x_positions[dep_idx]
        xm = (x1 + x2) / 2
        width = abs(x2 - x1)
        if width == 0:
            width = 0.5  # Avoid zero width
        # Set height proportional to width to make arcs visible
        height = width / 2  # Increased height for better visibility
        # Draw an arc from x1 to x2
        arc = matplotlib.patches.Arc((xm, y_position), width=width, height=height, angle=0, theta1=0, theta2=180, color=color)
        plt.gca().add_patch(arc)

    plt.xlim(-1, len(lemmas))
    plt.ylim(-1, max(1.5, height))  # Adjusted ylim to ensure arcs are visible

    # Ensure the aspect ratio is equal to make nodes circular
    plt.gca().set_aspect('equal', adjustable='datalim')

    plt.axis('off')
    plt.show()

def process_sentences(sentences):
    global sentence_counter
    for sentence in sentences:
        # Print the sentence
        print(sentence)

        # Process sentence
        doc = nlp(sentence)
        tokens = [token for token in doc]
        # Prepare adjusted lists
        adjusted_lemmas = []
        adjusted_pos_tags = []
        adjusted_dep_tags = []
        adjusted_tags = []
        tokens_to_use = []

        for token in tokens:
            lemma = token.lemma_
            pos_tag = token.pos_
            dep_tag = token.dep_
            tag = token.tag_
            word = token.text

            # Ignore "be" and "do" auxiliaries
            if dep_tag == 'aux' and lemma in ['be', 'do']:
                continue

            # Convert possessive clitic to "have"
            if tag == 'POS':
                lemma = 'have'
                pos_tag = 'VERB'
                dep_tag = 'aux'

            adjusted_lemmas.append(lemma)
            adjusted_pos_tags.append(pos_tag)
            adjusted_dep_tags.append(dep_tag)
            adjusted_tags.append(tag)
            tokens_to_use.append(token)

        # Increase the sentence counter
        sentence_counter += 1

        # Ensure all lemmas are in columns
        ensure_words_in_columns(adjusted_lemmas, tokens_to_use)

        # Collect qualities and initialize instance neurons
        activated_concepts, activated_instances, activated_qualities = collect_qualities_and_initialize_instances(
            adjusted_lemmas, adjusted_pos_tags, tokens_to_use)

        # Process relations
        activated_relations, activated_relation_targets = process_relations(
            adjusted_lemmas, adjusted_pos_tags, tokens_to_use, activated_instances)

        # Process definition connections
        process_definitions(tokens_to_use)

        # Update instance connections within the concept column
        update_instance_connections(activated_instances)

        # Update permanence values for concept neurons
        update_permanence_values_concept_neurons(activated_concepts)

        # Update permanence values for relation neurons and their target connections
        update_permanence_values_relation_neurons(
            activated_concepts, activated_relations, activated_relation_targets)

        # Update permanence values for quality neurons
        update_permanence_values_quality_neurons(activated_concepts, activated_qualities)

        # Update permanence values for instance connections
        update_permanence_values_instance_connections(activated_concepts, activated_instances)

        # Decrease activation trace counters
        decrease_activation_trace_counters()

        # Visualize the network
        visualize_network(G, columns)

        # Draw the dependency tree
        draw_dependency_tree(sentence, tokens_to_use, columns, G)

# Call the main processing function
process_sentences(sentences)
