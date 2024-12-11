"""GIAANNproto_databaseNetworkDraw.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_main.py

# Usage:
see GIAANNproto_main.py

# Description:
GIA ANN proto database Network Draw

"""

import networkx as nx
import matplotlib.pyplot as plt
import random
import GIAANNproto_sparseTensors

from GIAANNproto_globalDefs import *
import GIAANNproto_databaseNetwork

if(drawRelationTypes):
	relation_type_concept_pos1 = 'NOUN'
	relation_type_concept_pos2 = 'PROPN'
	relation_type_action_pos = 'VERB'
	relation_type_condition_pos = 'ADP'
	relation_type_quality_pos = 'ADJ'
	relation_type_modifier_pos = 'ADV'
	
	relation_type_determiner_pos = 'DET'
	relation_type_conjunction_pos1 = 'CCONJ'
	relation_type_conjunction_pos2 = 'SCONJ'
	relation_type_quantity_pos1 = 'SYM'
	relation_type_quantity_pos2 = 'NUM'
	relation_type_aux_pos = 'AUX'

	neuron_pos_to_relation_type_dict = {}
	neuron_pos_to_relation_type_dict[relation_type_concept_pos1] = 'blue'
	neuron_pos_to_relation_type_dict[relation_type_concept_pos2] = 'blue'
	neuron_pos_to_relation_type_dict[relation_type_action_pos] = 'green'
	neuron_pos_to_relation_type_dict[relation_type_condition_pos] = 'red'
	neuron_pos_to_relation_type_dict[relation_type_quality_pos] = 'turquoise'
	neuron_pos_to_relation_type_dict[relation_type_modifier_pos] = 'lightskyblue'
	
	neuron_pos_to_relation_type_dict[relation_type_determiner_pos] = 'magenta'
	neuron_pos_to_relation_type_dict[relation_type_conjunction_pos1] = 'black'
	neuron_pos_to_relation_type_dict[relation_type_conjunction_pos2] = 'black'
	neuron_pos_to_relation_type_dict[relation_type_quantity_pos1] = 'purple'
	neuron_pos_to_relation_type_dict[relation_type_quantity_pos2] = 'purple'
	neuron_pos_to_relation_type_dict[relation_type_aux_pos] = 'lightskyblue'

	relation_type_part_property_col = 'cyan'
	relation_type_aux_definition_col = 'blue'
	relation_type_aux_quality_col = 'turquoise'
	relation_type_aux_action_col = 'green'
	relation_type_aux_property_col = 'cyan'
	
	relation_type_other_col = 'gray'	#INTJ, X, other AUX
	
	be_auxiliaries = ["am", "is", "are", "was", "were", "being", "been"]
	have_auxiliaries = ["have", "has", "had", "having"]
	do_auxiliaries = ["do", "does", "did", "doing"]

	def generateFeatureNeuronColour(databaseNetworkObject, pos_float_torch, word, internal_connection=False):
		#print("pos_float_torch = ", pos_float_torch)
		pos_int = pos_float_torch.int().item()
		pos_string = pos_int_to_pos_string(databaseNetworkObject.nlp, pos_int)
		if(pos_string):
			if(pos_string in neuron_pos_to_relation_type_dict):
				colour = neuron_pos_to_relation_type_dict[pos_string]
			else:
				colour = relation_type_other_col
				
			#special cases;
			if(pos_string == 'AUX'):
				if(word in have_auxiliaries):
					colour = relation_type_aux_property_col
				elif(word in be_auxiliaries):
					if(internal_connection):
						colour = relation_type_aux_quality_col
					else:
						colour = relation_type_aux_definition_col
				elif(word in do_auxiliaries):
					colour = relation_type_aux_action_col
			if(pos_string == 'PART'):
				if(word == "'s"):
					colour = relation_type_part_property_col
		else:
			colour = relation_type_other_col
			#print("generateFeatureNeuronColour error; pos int = 0")
			
		return colour
		
# Initialize NetworkX graph for visualization
G = nx.DiGraph()


def createNeuronLabelWithActivation(name, strength):
	label = name + "\n" + floatToString(strength)
	return label
	
def floatToString(value):
	result = str(round(value.item(), 2))
	return result
		
		
def visualize_graph(sequence_observed_columns, save=False, fileName=None):
	databaseNetworkObject = sequence_observed_columns.databaseNetworkObject
	G.clear()

	if(drawAllColumns):
		observed_columns_dict = GIAANNproto_databaseNetwork.load_all_columns(databaseNetworkObject)
	else:
		observed_columns_dict = sequence_observed_columns.observed_columns_dict
	
	if not lowMem:
		global global_feature_neurons
		if(performRedundantCoalesce):
			global_feature_neurons = global_feature_neurons.coalesce()

	# Draw concept columns
	pos_dict = {}
	x_offset = 0
	for lemma, observed_column in observed_columns_dict.items():
		concept_index = observed_column.concept_index
		
		if(performRedundantCoalesce):
			if lowMem:
				observed_column.feature_neurons = observed_column.feature_neurons.coalesce()
		
		if(drawSequenceObservedColumns):
			feature_word_to_index = sequence_observed_columns.feature_word_to_index
			y_offset = 1 + 1	#reserve space at bottom of column for feature concept neuron (as it will not appear first in sequence_observed_columns.feature_word_to_index, only observed_column.feature_word_to_index)
			c_idx = sequence_observed_columns.concept_name_to_index[lemma]
			feature_neurons = sequence_observed_columns.feature_neurons[:, :, c_idx]
		else:
			feature_word_to_index = observed_column.feature_word_to_index
			y_offset = 1
			if lowMem:
				feature_neurons = observed_column.feature_neurons
			else:
				feature_neurons = GIAANNproto_sparseTensors.slice_sparse_tensor(databaseNetworkObject.global_feature_neurons, 2, concept_index)
				#feature_neurons = databaseNetworkObject.global_feature_neurons[:, :, concept_index]	#operation not supported for sparse tensors
					
		# Draw feature neurons
		for feature_word, feature_index_in_observed_column in feature_word_to_index.items():
			conceptNeuronFeature = False
			if(useDedicatedConceptNames and useDedicatedConceptNames2):
				if feature_word==variableConceptNeuronFeatureName:
					neuron_color = 'blue'
					neuron_name = observed_column.concept_name
					conceptNeuronFeature = True
					#print("\nvisualize_graph: conceptNeuronFeature neuron_name = ", neuron_name)
				else:
					neuron_color = 'turquoise'
					neuron_name = feature_word
			else:
				neuron_color = 'turqoise'
				neuron_name = feature_word

			f_idx = feature_index_in_observed_column	#not used
		
			featurePresent = False
			featureActive = False
			if(feature_neurons[array_index_properties_strength, array_index_segment_internal_column, feature_index_in_observed_column] > 0 and feature_neurons[array_index_properties_permanence, array_index_segment_internal_column, feature_index_in_observed_column] > 0):
				featurePresent = True
			if(feature_neurons[array_index_properties_activation, array_index_segment_internal_column, feature_index_in_observed_column] > 0):
				featureActive = True
				
			if(featurePresent):
				if(drawRelationTypes):
					if not conceptNeuronFeature:
						neuron_color = generateFeatureNeuronColour(databaseNetworkObject, feature_neurons[array_index_properties_pos, array_index_segment_internal_column, feature_index_in_observed_column], feature_word)
				elif(featureActive):
					if(conceptNeuronFeature):
						neuron_color = 'lightskyblue'
					else:
						neuron_color = 'cyan'
						
				if(debugDrawNeuronActivations):
					neuron_name = createNeuronLabelWithActivation(neuron_name, feature_neurons[array_index_properties_activation, array_index_segment_internal_column, feature_index_in_observed_column])

				feature_node = f"{lemma}_{feature_word}_{f_idx}"
				if(randomiseColumnFeatureXposition and not conceptNeuronFeature):
					x_offset_shuffled = x_offset + random.uniform(-0.5, 0.5)
				else:
					x_offset_shuffled = x_offset
				if(drawSequenceObservedColumns and conceptNeuronFeature):
					y_offset_prev = y_offset
					y_offset = 1
				G.add_node(feature_node, pos=(x_offset_shuffled, y_offset), color=neuron_color, label=neuron_name)
				if(drawSequenceObservedColumns and conceptNeuronFeature):
					y_offset = y_offset_prev
				else:
					y_offset += 1

		# Draw rectangle around the column
		plt.gca().add_patch(plt.Rectangle((x_offset - 0.5, -0.5), 1, max(y_offset, 1) + 0.5, fill=False, edgecolor='black'))
		x_offset += 2  # Adjust x_offset for the next column

	# Draw connections
	for lemma, observed_column in observed_columns_dict.items():
	
		if(performRedundantCoalesce):
			observed_column.feature_connections = observed_column.feature_connections.coalesce()
					
		concept_index = observed_column.concept_index
		if(drawSequenceObservedColumns):
			feature_word_to_index = sequence_observed_columns.feature_word_to_index
			other_feature_word_to_index = sequence_observed_columns.feature_word_to_index
			c_idx = sequence_observed_columns.concept_name_to_index[lemma]
			feature_connections = sequence_observed_columns.feature_connections[:, :, c_idx]
		else:
			feature_word_to_index = observed_column.feature_word_to_index
			other_feature_word_to_index = observed_column.feature_word_to_index
			c_idx = databaseNetworkObject.concept_columns_dict[lemma]
			feature_connections = observed_column.feature_connections
		feature_connections = pt.sum(feature_connections, dim=1)	#sum along sequential segment index (draw connections to all segments)
	
		# Internal connections (yellow)
		for feature_word, feature_index_in_observed_column in feature_word_to_index.items():
			source_node = f"{lemma}_{feature_word}_{feature_index_in_observed_column}"
			if G.has_node(source_node):
				for other_feature_word, other_feature_index_in_observed_column in feature_word_to_index.items():
					target_node = f"{lemma}_{other_feature_word}_{other_feature_index_in_observed_column}"
					if G.has_node(target_node):
						if feature_word != other_feature_word:
							f_idx = feature_word_to_index[feature_word]
							other_f_idx = feature_word_to_index[other_feature_word]
							
							featurePresent = False
							if(feature_connections[array_index_properties_strength, f_idx, c_idx, other_f_idx] > 0 and feature_connections[array_index_properties_permanence, f_idx, c_idx, other_f_idx] > 0):
								featurePresent = True
								
							if(drawRelationTypes):
								connection_color = generateFeatureNeuronColour(databaseNetworkObject, feature_connections[array_index_properties_pos, f_idx, c_idx, other_f_idx], feature_word, internal_connection=True)
							else:
								connection_color = 'yellow'
								
							if(featurePresent):
								G.add_edge(source_node, target_node, color=connection_color)
		
		# External connections (orange)
		for feature_word, feature_index_in_observed_column in feature_word_to_index.items():
			source_node = f"{lemma}_{feature_word}_{feature_index_in_observed_column}"
			if G.has_node(source_node):
				for other_lemma, other_observed_column in observed_columns_dict.items():
					if(drawSequenceObservedColumns):
						other_feature_word_to_index = sequence_observed_columns.feature_word_to_index
					else:
						other_feature_word_to_index = other_observed_column.feature_word_to_index
					for other_feature_word, other_feature_index_in_observed_column in other_feature_word_to_index.items():
						target_node = f"{other_lemma}_{other_feature_word}_{other_feature_index_in_observed_column}"
						if G.has_node(target_node):
							f_idx = feature_word_to_index[feature_word]
							other_f_idx = other_feature_word_to_index[other_feature_word]
							
							externalConnection = False
							if(drawSequenceObservedColumns):
								other_c_idx = sequence_observed_columns.concept_name_to_index[other_lemma]
								if other_c_idx != c_idx:
									externalConnection = True
							else:
								other_c_idx = databaseNetworkObject.concept_columns_dict[other_lemma]
								if lemma != other_lemma:	#if observed_column != other_observed_column:
									externalConnection = True
					
							featurePresent = False
							if(externalConnection):
								#print("feature_connections[array_index_properties_strength, f_idx, other_c_idx, other_f_idx] = ", feature_connections[array_index_properties_strength, f_idx, other_c_idx, other_f_idx])
								#print("feature_connections[array_index_properties_permanence, f_idx, other_c_idx, other_f_idx] = ", feature_connections[array_index_properties_permanence, f_idx, other_c_idx, other_f_idx])
								if(feature_connections[array_index_properties_strength, f_idx, other_c_idx, other_f_idx] > 0 and feature_connections[array_index_properties_permanence, f_idx, other_c_idx, other_f_idx] > 0):
									featurePresent = True
									#print("\tfeaturePresent")

							if(drawRelationTypes):
								connection_color = generateFeatureNeuronColour(databaseNetworkObject, feature_connections[array_index_properties_pos, f_idx, other_c_idx, other_f_idx], feature_word, internal_connection=False)
							else:
								connection_color = 'orange'
								
							if(featurePresent):
								G.add_edge(source_node, target_node, color=connection_color)
								
	# Get positions and colors for drawing
	pos = nx.get_node_attributes(G, 'pos')
	colors = [data['color'] for node, data in G.nodes(data=True)]
	edge_colors = [data['color'] for u, v, data in G.edges(data=True)]
	labels = nx.get_node_attributes(G, 'label')

	if(save):
		highResolutionFigure = True
	else:
		highResolutionFigure = False
	if(highResolutionFigure):
		displayFigDPI = 100
		saveFigDPI = 300	#approx HD	#depth per inch
		saveFigSize = (16,9)	#(9,9)	#in inches
		figureWidth = 1920
		figureHeight = 1080
		plt.gcf().set_size_inches(figureWidth / displayFigDPI, figureHeight / displayFigDPI)

	# Draw the graph
	nx.draw(G, pos, with_labels=True, labels=labels, arrows=True, node_color=colors, edge_color=edge_colors, node_size=500, font_size=8)
	plt.axis('off')  # Hide the axes
	
	if(save):
		if(highResolutionFigure):
			plt.savefig(fileName, dpi=saveFigDPI)
		else:
			plt.savefig(fileName)
		plt.clf()	
	else:
		plt.show()
