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
columns = {}  # key: lemma (concept), value: {'concept_neuron', 'permanence', 'relation_neurons', 'quality_neurons'}

# Initialize NetworkX graph
G = nx.Graph()

# Global neuron ID counter to ensure unique IDs
neuron_id_counter = 0

# Sentence counter to manage activation traces
sentence_counter = 0

# Global instance connections
global_instance_connections = {}

# Lists for POS types
concept_pos_list = ['NOUN', 'PROPN', 'PRON', 'X']  # POS types for concept columns
relation_pos_list = ['VERB', 'ADP', 'CONJ']        # POS types for relation neurons
quality_pos_list = ['DET', 'ADJ']                  # Removed 'ADV' to exclude adverbs from quality detection
modifier_pos_list = ['ADV']                        # POS types for modifier neurons

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
    for i, (concept_lemma, neurons) in enumerate(columns.items()):
        # x position is i * x_margin
        x = i * x_margin * 2
        x_positions[concept_lemma] = x
        # Concept neuron at bottom (y=0)
        concept_neuron_id = neurons['concept_neuron']
        pos[concept_neuron_id] = (x, 0)
        labels[concept_neuron_id] = concept_lemma
        current_y = y_margin  # Start placing neurons above the concept neuron

        # Relation neurons above
        for relation_lemma, relation_info in neurons['relation_neurons'].items():
            relation_neuron_id = relation_info['neuron_id']
            pos[relation_neuron_id] = (x, current_y)
            labels[relation_neuron_id] = relation_lemma
            relation_info['y_pos'] = current_y  # Store y position for internal relation drawing

            # Modifiers for relation neurons
            if 'modifiers' in relation_info:
                for modifier_lemma, modifier_info in relation_info['modifiers'].items():
                    modifier_neuron_id = modifier_info['neuron_id']
                    current_y += y_margin
                    pos[modifier_neuron_id] = (x, current_y)
                    labels[modifier_neuron_id] = modifier_lemma
            current_y += y_margin
            if current_y > max_y:
                max_y = current_y

        # Quality neurons above relation neurons
        for quality_lemma, quality_info in neurons['quality_neurons'].items():
            quality_neuron_id = quality_info['neuron_id']
            pos[quality_neuron_id] = (x, current_y)
            labels[quality_neuron_id] = quality_lemma

            # Modifiers for quality neurons
            if 'modifiers' in quality_info:
                for modifier_lemma, modifier_info in quality_info['modifiers'].items():
                    modifier_neuron_id = modifier_info['neuron_id']
                    current_y += y_margin
                    pos[modifier_neuron_id] = (x, current_y)
                    labels[modifier_neuron_id] = modifier_lemma
            current_y += y_margin
            if current_y > max_y:
                max_y = current_y

    # Assign positions to any remaining nodes that haven't been assigned yet
    unpositioned_nodes = set(G.nodes()) - set(pos.keys())
    if unpositioned_nodes:
        current_x = (len(columns) + 1) * x_margin * 2  # Start after the last column
        current_y = y_margin
        for node_id in unpositioned_nodes:
            pos[node_id] = (current_x, current_y)
            labels[node_id] = f"Node {node_id}"
            current_y += y_margin * 2
            if current_y > max_y:
                max_y = current_y

    # Draw nodes
    node_colors = []
    for node_id in G.nodes():
        if node_id in [neurons['concept_neuron'] for neurons in columns.values()]:
            node_colors.append('blue')  # Concept neurons
        elif any(node_id in [info['neuron_id'] for info in neurons['relation_neurons'].values()] for neurons in columns.values()):
            # Relation neurons
            for neurons in columns.values():
                for relation_lemma, relation_info in neurons['relation_neurons'].items():
                    if relation_info['neuron_id'] == node_id:
                        relation_pos = relation_info['pos']
                        relation_lemma = relation_info['lemma']
                        if relation_lemma == 'have':
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
        elif any(node_id in [modifier_info['neuron_id'] for neuron_info in neurons['relation_neurons'].values() if 'modifiers' in neuron_info for modifier_info in neuron_info['modifiers'].values()] for neurons in columns.values()):
            node_colors.append('lightblue')  # Modifier neurons for relations
        elif any(node_id in [modifier_info['neuron_id'] for neuron_info in neurons['quality_neurons'].values() if 'modifiers' in neuron_info for modifier_info in neuron_info['modifiers'].values()] for neurons in columns.values()):
            node_colors.append('lightblue')  # Modifier neurons for qualities
        else:
            node_colors.append('lightblue')  # Default color for unpositioned modifier nodes

    # Draw edges
    edge_colors = []
    edge_styles = []
    for u, v in G.edges():
        edge = G.get_edge_data(u, v)
        if edge['type'] == 'relation_target':
            if edge.get('lemma') == 'have':
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
            if edge.get('lemma') == 'have':
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
        elif edge['type'] == 'modifier':
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
            y_positions.append(pos[relation_info['neuron_id']][1])
            if 'modifiers' in relation_info:
                for modifier_info in relation_info['modifiers'].values():
                    y_positions.append(pos[modifier_info['neuron_id']][1])
        for quality_info in neurons['quality_neurons'].values():
            y_positions.append(pos[quality_info['neuron_id']][1])
            if 'modifiers' in quality_info:
                for modifier_info in quality_info['modifiers'].values():
                    y_positions.append(pos[modifier_info['neuron_id']][1])
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

        # Ignore specific determiners: 'the', 'a', 'an'
        if lemma.lower() in ['the', 'a', 'an']:
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
                'quality_neurons': {}
            }
            # Add the concept neuron to the graph
            G.add_node(concept_neuron_id)

def collect_modifiers(idx, lemmas, pos_tags):
    modifiers_found = []
    # Check previous lemma
    if idx > 0:
        prev_lemma = lemmas[idx - 1]
        prev_pos_tag = pos_tags[idx - 1]
        if prev_pos_tag in modifier_pos_list:
            # Ensure no intervening relation words
            intervening_pos = pos_tags[idx - 1:idx]
            if not any(pos in relation_pos_list for pos in intervening_pos):
                modifiers_found.append((prev_lemma, idx - 1))
    # Check next lemma
    if idx < len(lemmas) - 1:
        next_lemma = lemmas[idx + 1]
        next_pos_tag = pos_tags[idx + 1]
        if next_pos_tag in modifier_pos_list:
            # Ensure no intervening relation words
            intervening_pos = pos_tags[idx + 1:idx + 2]
            if not any(pos in relation_pos_list for pos in intervening_pos):
                modifiers_found.append((next_lemma, idx + 1))
    return modifiers_found

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
                            'first_activation': True,
                            'modifiers': {}
                        }
                        # Add the quality neuron to the graph
                        G.add_node(quality_neuron_id)
                        # Ensure prev_lemma has a concept neuron
                        ensure_words_in_columns([prev_lemma], [tokens[idx - 1]])
                        prev_lemma_concept_id = columns[prev_lemma]['concept_neuron']
                        # Draw concept source connection
                        G.add_edge(prev_lemma_concept_id, quality_neuron_id, type='concept_source')
                    else:
                        quality_neuron_id = columns[lemma]['quality_neurons'][prev_lemma]['neuron_id']
                        # Reset activation trace counter
                        columns[lemma]['quality_neurons'][prev_lemma]['activation_trace_counter'] = 5
                    # Now check for modifiers for this quality neuron
                    quality_neuron_info = columns[lemma]['quality_neurons'][prev_lemma]
                    modifiers_found = collect_modifiers(idx - 1, lemmas, pos_tags)
                    for modifier_lemma, modifier_idx in modifiers_found:
                        if modifier_lemma not in quality_neuron_info['modifiers']:
                            neuron_id_counter += 1
                            modifier_neuron_id = neuron_id_counter
                            quality_neuron_info['modifiers'][modifier_lemma] = {
                                'neuron_id': modifier_neuron_id,
                                'permanence': 3,
                                'activation_trace_counter': 5,
                                'first_activation': True
                            }
                            # Add the modifier neuron to the graph
                            G.add_node(modifier_neuron_id)
                            # Draw edge from modifier neuron to quality neuron
                            G.add_edge(modifier_neuron_id, quality_neuron_id, type='modifier')
                        else:
                            modifier_neuron_info = quality_neuron_info['modifiers'][modifier_lemma]
                            # Reset activation trace counter
                            modifier_neuron_info['activation_trace_counter'] = 5

            # Similar check for modifiers when quality is in the next lemma
        # Check next lemma
        if idx < len(lemmas) - 1:
            next_lemma = lemmas[idx + 1]
            next_pos_tag = pos_tags[idx + 1]
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
                            'first_activation': True,
                            'modifiers': {}
                        }
                        # Add the quality neuron to the graph
                        G.add_node(quality_neuron_id)
                        # Ensure next_lemma has a concept neuron
                        ensure_words_in_columns([next_lemma], [tokens[idx + 1]])
                        next_lemma_concept_id = columns[next_lemma]['concept_neuron']
                        # Draw concept source connection
                        G.add_edge(next_lemma_concept_id, quality_neuron_id, type='concept_source')
                    else:
                        quality_neuron_id = columns[lemma]['quality_neurons'][next_lemma]['neuron_id']
                        # Reset activation trace counter
                        columns[lemma]['quality_neurons'][next_lemma]['activation_trace_counter'] = 5
                    # Now check for modifiers for this quality neuron
                    quality_neuron_info = columns[lemma]['quality_neurons'][next_lemma]
                    modifiers_found = collect_modifiers(idx + 1, lemmas, pos_tags)
                    for modifier_lemma, modifier_idx in modifiers_found:
                        if modifier_lemma not in quality_neuron_info['modifiers']:
                            neuron_id_counter += 1
                            modifier_neuron_id = neuron_id_counter
                            quality_neuron_info['modifiers'][modifier_lemma] = {
                                'neuron_id': modifier_neuron_id,
                                'permanence': 3,
                                'activation_trace_counter': 5,
                                'first_activation': True
                            }
                            # Add the modifier neuron to the graph
                            G.add_node(modifier_neuron_id)
                            # Draw edge from modifier neuron to quality neuron
                            G.add_edge(modifier_neuron_id, quality_neuron_id, type='modifier')
                        else:
                            modifier_neuron_info = quality_neuron_info['modifiers'][modifier_lemma]
                            # Reset activation trace counter
                            modifier_neuron_info['activation_trace_counter'] = 5

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

def process_relations(lemmas, pos_tags, tokens, activated_instances):
    global neuron_id_counter
    activated_relations = {}
    activated_relation_targets = {}
    prev_concept_lemma = None
    prev_relation_neuron_id = None
    prev_relation_pos = None
    prev_relation_lemma = None
    for idx, (lemma, pos_tag) in enumerate(zip(lemmas, pos_tags)):
        if not lemma.isalpha():
            continue

        # Check if current token is a concept word
        if pos_tag in concept_pos_list:
            prev_concept_lemma = lemma
            prev_relation_neuron_id = None
            prev_relation_pos = None
            prev_relation_lemma = None
            # Reset activation trace for concept neuron
            columns[lemma]['concept_activation_trace_counter'] = 5
            continue

        # Check if current token is a relation word (verb/preposition/conjunction)
        elif pos_tag in relation_pos_list:
            if prev_concept_lemma is None:
                continue  # Skip if there's no preceding concept word

            # Create a relation neuron in the concept column
            if lemma not in columns[prev_concept_lemma]['relation_neurons']:
                neuron_id_counter += 1
                relation_neuron_id = neuron_id_counter
                columns[prev_concept_lemma]['relation_neurons'][lemma] = {
                    'neuron_id': relation_neuron_id,
                    'permanence': 3,  # Initialized to 3
                    'activation_trace_counter': 5,  # Activation trace counter
                    'pos': pos_tag,
                    'lemma': lemma,
                    'target_connections': {},
                    'internal_relations': {},
                    'modifiers': {},
                    'first_activation': True
                }
                # Add the relation neuron to the graph
                G.add_node(relation_neuron_id)
                # Draw concept source connection
                ensure_words_in_columns([lemma], [tokens[idx]])
                relation_lemma_concept_id = columns[lemma]['concept_neuron']
                # Reset activation trace counter for relation word concept neuron
                columns[lemma]['concept_activation_trace_counter'] = 5
                G.add_edge(relation_lemma_concept_id, relation_neuron_id, type='concept_source')
            else:
                relation_info = columns[prev_concept_lemma]['relation_neurons'][lemma]
                relation_neuron_id = relation_info['neuron_id']
                relation_info['activation_trace_counter'] = 5

            # Need to keep track of activated_relations
            if prev_concept_lemma not in activated_relations:
                activated_relations[prev_concept_lemma] = []
            activated_relations[prev_concept_lemma].append(lemma)

            # If previous token was also a relation word, create internal relation
            if prev_relation_neuron_id is not None:
                # Create internal relation between prev_relation_neuron_id and current relation_neuron_id
                edge_type = 'internal_relation'
                G.add_edge(prev_relation_neuron_id, relation_neuron_id, type=edge_type, pos=pos_tag, lemma=lemma)
                # Need to store permanence and activation trace for internal relations
                prev_relation_info = columns[prev_concept_lemma]['relation_neurons'][prev_relation_lemma]
                if relation_neuron_id not in prev_relation_info['internal_relations']:
                    prev_relation_info['internal_relations'][relation_neuron_id] = {
                        'permanence': 3,
                        'activation_trace_counter': 5,
                        'first_activation': True
                    }
                else:
                    # Reset activation trace counter
                    prev_relation_info['internal_relations'][relation_neuron_id]['activation_trace_counter'] = 5
            else:
                # This is the first relation word after a concept
                pass  # No internal relation to create

            # Collect modifiers for this relation neuron
            modifiers_found = collect_modifiers(idx, lemmas, pos_tags)
            for modifier_lemma, modifier_idx in modifiers_found:
                if modifier_lemma not in relation_info['modifiers']:
                    neuron_id_counter += 1
                    modifier_neuron_id = neuron_id_counter
                    relation_info['modifiers'][modifier_lemma] = {
                        'neuron_id': modifier_neuron_id,
                        'permanence': 3,
                        'activation_trace_counter': 5,
                        'first_activation': True
                    }
                    # Add the modifier neuron to the graph
                    G.add_node(modifier_neuron_id)
                    # Draw edge from modifier neuron to relation neuron
                    G.add_edge(modifier_neuron_id, relation_neuron_id, type='modifier')
                else:
                    modifier_neuron_info = relation_info['modifiers'][modifier_lemma]
                    # Reset activation trace counter
                    modifier_neuron_info['activation_trace_counter'] = 5

            # Update prev_relation variables
            prev_relation_neuron_id = relation_neuron_id
            prev_relation_pos = pos_tag
            prev_relation_lemma = lemma

            # Check if the next token is not a relation word, or this is the last token
            if idx + 1 >= len(lemmas) or pos_tags[idx + 1] not in relation_pos_list:
                # This is the last in sequence, process external relation
                # Now find the target concept word after this relation word
                for target_idx in range(idx + 1, len(lemmas)):
                    target_lemma = lemmas[target_idx]
                    target_pos_tag = pos_tags[target_idx]
                    if not target_lemma.isalpha():
                        continue
                    if target_pos_tag in concept_pos_list:
                        target_concept_neuron_id = columns[target_lemma]['concept_neuron']
                        # Connect the relation neuron to the target concept neuron
                        G.add_edge(relation_neuron_id, target_concept_neuron_id, type='relation_target', pos=pos_tag, lemma=lemma)
                        # Store the permanence value for the connection
                        relation_info = columns[prev_concept_lemma]['relation_neurons'][lemma]
                        if target_concept_neuron_id not in relation_info['target_connections']:
                            relation_info['target_connections'][target_concept_neuron_id] = {
                                'permanence': 3,  # Initialized to 3
                                'activation_trace_counter': 5,  # Activation trace counter
                                'first_activation': True
                            }
                        else:
                            # Reset activation trace counter
                            relation_info['target_connections'][target_concept_neuron_id]['activation_trace_counter'] = 5
                        # Add to activated targets
                        activated_relation_targets.setdefault(relation_neuron_id, []).append(target_concept_neuron_id)
                        # Now connect the relation neuron to activated instance neurons in the target column
                        if target_lemma in activated_instances:
                            for target_instance_neuron_id in activated_instances[target_lemma]:
                                G.add_edge(relation_neuron_id, target_instance_neuron_id, type='relation_target', pos=pos_tag, lemma=lemma)
                                if target_instance_neuron_id not in relation_info['target_connections']:
                                    relation_info['target_connections'][target_instance_neuron_id] = {
                                        'permanence': 3,  # Initialized to 3
                                        'activation_trace_counter': 5,  # Activation trace counter
                                        'first_activation': True
                                    }
                                else:
                                    # Reset activation trace counter
                                    relation_info['target_connections'][target_instance_neuron_id]['activation_trace_counter'] = 5
                                activated_relation_targets[relation_neuron_id].append(target_instance_neuron_id)
                        # Reset activation trace for target concept neuron
                        columns[target_lemma]['concept_activation_trace_counter'] = 5
                        break  # Only connect to the first concept word after relation word

                # Reset prev_relation variables since sequence has ended
                prev_relation_neuron_id = None
                prev_relation_pos = None
                prev_relation_lemma = None

        else:
            # Not a concept word or relation word, reset prev_relation variables
            prev_relation_neuron_id = None
            prev_relation_pos = None
            prev_relation_lemma = None

    return activated_relations, activated_relation_targets

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
    all_instance_neurons = []
    for instance_neurons in activated_instances.values():
        all_instance_neurons.extend(instance_neurons)
    if len(all_instance_neurons) >= 2:
        for i in range(len(all_instance_neurons)):
            for j in range(i + 1, len(all_instance_neurons)):
                neuron_pair = tuple(sorted((all_instance_neurons[i], all_instance_neurons[j])))
                if neuron_pair not in global_instance_connections:
                    global_instance_connections[neuron_pair] = {
                        'permanence': 3,  # Initialized to 3
                        'activation_trace_counter': 5,  # Activation trace counter
                        'first_activation': True
                    }
                    # Add edge to graph if not already present
                    if not G.has_edge(neuron_pair[0], neuron_pair[1]):
                        G.add_edge(neuron_pair[0], neuron_pair[1], type='instance_connection')
                else:
                    # Reset activation trace counter
                    global_instance_connections[neuron_pair]['activation_trace_counter'] = 5

def update_permanence_values_concept_neurons(activated_concepts):
    for concept_lemma, neurons in columns.items():
        if concept_lemma in activated_concepts:
            neurons['permanence'] += 1  # No change required

def update_permanence_values_relation_neurons(activated_concepts, activated_relations, activated_relation_targets):
    for concept_lemma, neurons in columns.items():
        if concept_lemma in activated_concepts:
            active_relations = activated_relations.get(concept_lemma, [])
            for relation_lemma, relation_info in list(neurons['relation_neurons'].items()):
                relation_neuron_id = relation_info['neuron_id']
                if relation_lemma in active_relations:
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
                    # Update internal relations
                    if 'internal_relations' in relation_info:
                        for internal_neuron_id, internal_conn_info in list(relation_info['internal_relations'].items()):
                            if internal_conn_info.get('first_activation', False):
                                internal_conn_info['first_activation'] = False
                            else:
                                internal_conn_info['permanence'] = internal_conn_info['permanence'] ** 2
                    # Update modifiers
                    if 'modifiers' in relation_info:
                        for modifier_lemma, modifier_info in list(relation_info['modifiers'].items()):
                            if modifier_info.get('first_activation', False):
                                modifier_info['first_activation'] = False
                            else:
                                modifier_info['permanence'] = modifier_info['permanence'] ** 2
                else:
                    # Decrease permanence by 1
                    relation_info['permanence'] -= 1
                    if relation_info['permanence'] <= 0:
                        # Remove relation neuron
                        if G.has_node(relation_neuron_id):
                            G.remove_node(relation_neuron_id)
                        del neurons['relation_neurons'][relation_lemma]
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
                        # Decrease permanence of internal relations
                        if 'internal_relations' in relation_info:
                            for internal_neuron_id, internal_conn_info in list(relation_info['internal_relations'].items()):
                                internal_conn_info['permanence'] -= 1
                                if internal_conn_info['permanence'] <= 0:
                                    # Remove edge
                                    if G.has_edge(relation_neuron_id, internal_neuron_id):
                                        G.remove_edge(relation_neuron_id, internal_neuron_id)
                                    del relation_info['internal_relations'][internal_neuron_id]
                        # Decrease permanence of modifiers
                        if 'modifiers' in relation_info:
                            for modifier_lemma, modifier_info in list(relation_info['modifiers'].items()):
                                modifier_info['permanence'] -= 1
                                if modifier_info['permanence'] <= 0:
                                    modifier_neuron_id = modifier_info['neuron_id']
                                    if G.has_node(modifier_neuron_id):
                                        G.remove_node(modifier_neuron_id)
                                    if G.has_edge(modifier_neuron_id, relation_neuron_id):
                                        G.remove_edge(modifier_neuron_id, relation_neuron_id)
                                    del relation_info['modifiers'][modifier_lemma]

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
                    # Update modifiers
                    if 'modifiers' in quality_info:
                        for modifier_lemma, modifier_info in list(quality_info['modifiers'].items()):
                            if modifier_info.get('first_activation', False):
                                modifier_info['first_activation'] = False
                            else:
                                modifier_info['permanence'] = modifier_info['permanence'] ** 2
                else:
                    quality_info['permanence'] -= 1
                    if quality_info['permanence'] <= 0:
                        quality_neuron_id = quality_info['neuron_id']
                        if G.has_node(quality_neuron_id):
                            G.remove_node(quality_neuron_id)
                        del neurons['quality_neurons'][quality_lemma]
                    else:
                        # Decrease permanence of modifiers
                        if 'modifiers' in quality_info:
                            for modifier_lemma, modifier_info in list(quality_info['modifiers'].items()):
                                modifier_info['permanence'] -= 1
                                if modifier_info['permanence'] <= 0:
                                    modifier_neuron_id = modifier_info['neuron_id']
                                    if G.has_node(modifier_neuron_id):
                                        G.remove_node(modifier_neuron_id)
                                    if G.has_edge(modifier_neuron_id, quality_neuron_id):
                                        G.remove_edge(modifier_neuron_id, quality_neuron_id)
                                    del quality_info['modifiers'][modifier_lemma]

def update_permanence_values_instance_connections(activated_instances):
    active_pairs = set()
    # Collect all activated instance neurons
    all_instance_neurons = []
    for instance_neurons in activated_instances.values():
        all_instance_neurons.extend(instance_neurons)
    if len(all_instance_neurons) >= 2:
        active_pairs = set(
            tuple(sorted((all_instance_neurons[i], all_instance_neurons[j])))
            for i in range(len(all_instance_neurons))
            for j in range(i + 1, len(all_instance_neurons))
        )
    # Now update the permanence values for all instance connections
    for neuron_pair, connection_info in list(global_instance_connections.items()):
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
                del global_instance_connections[neuron_pair]

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
            # Internal relations
            if 'internal_relations' in relation_info:
                for internal_conn_info in relation_info['internal_relations'].values():
                    if internal_conn_info['activation_trace_counter'] > 0:
                        internal_conn_info['activation_trace_counter'] -= 1
            # Modifiers
            if 'modifiers' in relation_info:
                for modifier_info in relation_info['modifiers'].values():
                    if modifier_info['activation_trace_counter'] > 0:
                        modifier_info['activation_trace_counter'] -= 1

        # Quality neurons
        for quality_info in neurons['quality_neurons'].values():
            if quality_info['activation_trace_counter'] > 0:
                quality_info['activation_trace_counter'] -= 1
            # Modifiers
            if 'modifiers' in quality_info:
                for modifier_info in quality_info['modifiers'].values():
                    if modifier_info['activation_trace_counter'] > 0:
                        modifier_info['activation_trace_counter'] -= 1

    # Decrease activation trace counters for global instance connections
    for connection_info in global_instance_connections.values():
        if connection_info['activation_trace_counter'] > 0:
            connection_info['activation_trace_counter'] -= 1

def draw_dependency_tree(sentence, tokens, columns, G):
    # Extract lemmas and positions
    lemmas = [token.lemma_ for token in tokens]
    pos_tags = [token.pos_ for token in tokens]
    dep_tags = [token.dep_ for token in tokens]
    tags = [token.tag_ for token in tokens]

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
                node_colors.append('cyan')  # 'have' auxiliary action relations
            elif pos_tag == 'VERB':
                node_colors.append('green')  # Action
            elif pos_tag == 'ADP':
                node_colors.append('red')    # Condition
            elif pos_tag == 'CONJ':
                node_colors.append('green')  # Conjunctions colored as action
            else:
                node_colors.append('gray')
        elif pos_tag in quality_pos_list:
            node_colors.append('turquoise')  # Quality node
        elif pos_tag in modifier_pos_list:
            node_colors.append('lightblue')  # Modifier node
        else:
            node_colors.append('gray')

    # Draw nodes as circles without edge colors
    node_size = 750 / (1.5 ** 2)  # Reduced size to decrease radius by a factor of 1.5
    plt.scatter(x_positions, [y_position]*len(lemmas), s=node_size, c=node_colors)

    # Draw lemma text inside the nodes with reduced font size
    for idx, lemma in enumerate(lemmas):
        plt.text(x_positions[idx], y_position, lemma, fontsize=8, ha='center', va='center', color='black')

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
                    # Process modifiers
                    if 'modifiers' in quality_info:
                        for modifier_lemma, modifier_info in quality_info['modifiers'].items():
                            if modifier_lemma in lemmas:
                                modifier_idx = lemmas.index(modifier_lemma)
                                dependencies.append((modifier_idx, quality_idx, 'modifier', 'lightblue'))

        # Process relations
        for relation_lemma, relation_info in neurons.get('relation_neurons', {}).items():
            if relation_lemma in lemmas:
                relation_idx = lemmas.index(relation_lemma)
                # Check permanence value
                permanence = relation_info.get('permanence', 0)
                if permanence >= 3:
                    # Add dependency from concept to relation
                    # Determine color
                    pos = relation_info.get('pos', '')
                    if relation_lemma == 'have':
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
                    # Process modifiers
                    if 'modifiers' in relation_info:
                        for modifier_lemma, modifier_info in relation_info['modifiers'].items():
                            if modifier_lemma in lemmas:
                                modifier_idx = lemmas.index(modifier_lemma)
                                dependencies.append((modifier_idx, relation_idx, 'modifier', 'lightblue'))
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

            # Ignore specific determiners: 'the', 'a', 'an'
            if lemma.lower() in ['the', 'a', 'an']:
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

        # Update instance connections across all instance neurons in the sentence
        update_instance_connections(activated_instances)

        # Update permanence values for concept neurons
        update_permanence_values_concept_neurons(activated_concepts)

        # Update permanence values for relation neurons and their target connections
        update_permanence_values_relation_neurons(
            activated_concepts, activated_relations, activated_relation_targets)

        # Update permanence values for quality neurons
        update_permanence_values_quality_neurons(activated_concepts, activated_qualities)

        # Update permanence values for instance connections
        update_permanence_values_instance_connections(activated_instances)

        # Decrease activation trace counters
        decrease_activation_trace_counters()

        # Visualize the network
        visualize_network(G, columns)

        # Draw the dependency tree
        draw_dependency_tree(sentence, tokens_to_use, columns, G)


# Call the main processing function
process_sentences(sentences)
