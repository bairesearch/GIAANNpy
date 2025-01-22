"""GIAANNproto_databaseNetworkDraw.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

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
	relationTypeConceptPos1 = 'NOUN'
	relationTypeConceptPos2 = 'PROPN'
	relationTypeActionPos = 'VERB'
	relationTypeConditionPos = 'ADP'
	relationTypeQualityPos = 'ADJ'
	relationTypeModifierPos = 'ADV'
	
	relationTypeDeterminerPos = 'DET'
	relationTypeConjunctionPos1 = 'CCONJ'
	relationTypeConjunctionPos2 = 'SCONJ'
	relationTypeQuantityPos1 = 'SYM'
	relationTypeQuantityPos2 = 'NUM'
	relationTypeAuxPos = 'AUX'

	neuronPosToRelationTypeDict = {}
	neuronPosToRelationTypeDict[relationTypeConceptPos1] = 'blue'
	neuronPosToRelationTypeDict[relationTypeConceptPos2] = 'blue'
	neuronPosToRelationTypeDict[relationTypeActionPos] = 'green'
	neuronPosToRelationTypeDict[relationTypeConditionPos] = 'red'
	neuronPosToRelationTypeDict[relationTypeQualityPos] = 'turquoise'
	neuronPosToRelationTypeDict[relationTypeModifierPos] = 'lightskyblue'
	
	neuronPosToRelationTypeDict[relationTypeDeterminerPos] = 'magenta'
	neuronPosToRelationTypeDict[relationTypeConjunctionPos1] = 'black'
	neuronPosToRelationTypeDict[relationTypeConjunctionPos2] = 'black'
	neuronPosToRelationTypeDict[relationTypeQuantityPos1] = 'purple'
	neuronPosToRelationTypeDict[relationTypeQuantityPos2] = 'purple'
	neuronPosToRelationTypeDict[relationTypeAuxPos] = 'lightskyblue'

	relationTypePartPropertyCol = 'cyan'
	relationTypeAuxDefinitionCol = 'blue'
	relationTypeAuxQualityCol = 'turquoise'
	relationTypeAuxActionCol = 'green'
	relationTypeAuxPropertyCol = 'cyan'
	
	relationTypeOtherCol = 'gray'	#INTJ, X, other AUX
	
	beAuxiliaries = ["am", "is", "are", "was", "were", "being", "been"]
	haveAuxiliaries = ["have", "has", "had", "having"]
	doAuxiliaries = ["do", "does", "did", "doing"]

	def generateFeatureNeuronColour(databaseNetworkObject, posFloatTorch, word, internalConnection=False):
		#print("posFloatTorch = ", posFloatTorch)
		posInt = posFloatTorch.int().item()
		posString = posIntToPosString(databaseNetworkObject.nlp, posInt)
		if(posString):
			if(posString in neuronPosToRelationTypeDict):
				colour = neuronPosToRelationTypeDict[posString]
			else:
				colour = relationTypeOtherCol
				
			#special cases;
			if(posString == 'AUX'):
				if(word in haveAuxiliaries):
					colour = relationTypeAuxPropertyCol
				elif(word in beAuxiliaries):
					if(internalConnection):
						colour = relationTypeAuxQualityCol
					else:
						colour = relationTypeAuxDefinitionCol
				elif(word in doAuxiliaries):
					colour = relationTypeAuxActionCol
			if(posString == 'PART'):
				if(word == "'s"):
					colour = relationTypePartPropertyCol
		else:
			colour = relationTypeOtherCol
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
		
		
def visualizeGraph(sequenceObservedColumns, save=False, fileName=None):
	databaseNetworkObject = sequenceObservedColumns.databaseNetworkObject
	G.clear()

	if(drawAllColumns):
		observedColumnsDict = GIAANNproto_databaseNetwork.loadAllColumns(databaseNetworkObject)
	else:
		observedColumnsDict = sequenceObservedColumns.observedColumnsDict
	
	if not lowMem:
		global globalFeatureNeurons
		if(performRedundantCoalesce):
			globalFeatureNeurons = globalFeatureNeurons.coalesce()

	# Draw concept columns
	posDict = {}
	xOffset = 0
	for lemma, observedColumn in observedColumnsDict.items():
		conceptIndex = observedColumn.conceptIndex
		
		if(performRedundantCoalesce):
			if lowMem:
				observedColumn.featureNeurons = observedColumn.featureNeurons.coalesce()
		
		if(drawSequenceObservedColumns):
			featureWordToIndex = sequenceObservedColumns.featureWordToIndex
			yOffset = 1 + 1	#reserve space at bottom of column for feature concept neuron
			cIdx = sequenceObservedColumns.conceptNameToIndex[lemma]
			featureNeurons = sequenceObservedColumns.featureNeurons[:, :, cIdx]
		else:
			featureWordToIndex = observedColumn.featureWordToIndex
			yOffset = 1
			if lowMem:
				featureNeurons = observedColumn.featureNeurons
			else:
				featureNeurons = GIAANNproto_sparseTensors.sliceSparseTensor(databaseNetworkObject.globalFeatureNeurons, 2, conceptIndex)
				#featureNeurons = databaseNetworkObject.globalFeatureNeurons[:, :, conceptIndex]	#operation not supported for sparse tensors
					
		# Draw feature neurons
		for featureWord, featureIndexInObservedColumn in featureWordToIndex.items():
			conceptNeuronFeature = False
			if(useDedicatedConceptNames and useDedicatedConceptNames2):
				if featureWord == variableConceptNeuronFeatureName:
					neuronColor = 'blue'
					neuronName = observedColumn.conceptName
					conceptNeuronFeature = True
					#print("\nvisualizeGraph: conceptNeuronFeature neuronName = ", neuronName)
				else:
					neuronColor = 'turquoise'
					neuronName = featureWord
			else:
				neuronColor = 'turqoise'
				neuronName = featureWord

			fIdx = featureIndexInObservedColumn	#not used
		
			featurePresent = False
			featureActive = False
			if(neuronIsActive(featureNeurons, arrayIndexPropertiesStrength, featureIndexInObservedColumn) and neuronIsActive(featureNeurons, arrayIndexPropertiesPermanence, featureIndexInObservedColumn)):
				featurePresent = True
			if(neuronIsActive(featureNeurons, arrayIndexPropertiesActivation, featureIndexInObservedColumn)):
				featureActive = True
				
			if(featurePresent):
				if(drawRelationTypes):
					if not conceptNeuronFeature:
						if(useSANI):
							segmentIndex = arrayIndexSegmentAdjacentColumn #arrayIndexSegmentAdjacentColumn has a higher probability of being filled than arrayIndexSegmentInternalColumn
						else:
							segmentIndex = arrayIndexSegmentInternalColumn
						neuronColor = generateFeatureNeuronColour(databaseNetworkObject, featureNeurons[arrayIndexPropertiesPos, segmentIndex, featureIndexInObservedColumn], featureWord)	
				elif(featureActive):
					if(conceptNeuronFeature):
						neuronColor = 'lightskyblue'
					else:
						neuronColor = 'cyan'
						
				if(debugDrawNeuronActivations):
					neuronName = createNeuronLabelWithActivation(neuronName, neuronActivation(featureNeurons, arrayIndexPropertiesActivation, featureIndexInObservedColumn))

				featureNode = f"{lemma}_{featureWord}_{fIdx}"
				if(randomiseColumnFeatureXposition and not conceptNeuronFeature):
					xOffsetShuffled = xOffset + random.uniform(-0.5, 0.5)
				else:
					xOffsetShuffled = xOffset
				if(drawSequenceObservedColumns and conceptNeuronFeature):
					yOffsetPrev = yOffset
					yOffset = 1
				G.add_node(featureNode, pos=(xOffsetShuffled, yOffset), color=neuronColor, label=neuronName)
				if(drawSequenceObservedColumns and conceptNeuronFeature):
					yOffset = yOffsetPrev
				else:
					yOffset += 1

		# Draw rectangle around the column
		plt.gca().add_patch(plt.Rectangle((xOffset - 0.5, -0.5), 1, max(yOffset, 1) + 0.5, fill=False, edgecolor='black'))
		xOffset += 2  # Adjust xOffset for the next column

	# Draw connections
	for lemma, observedColumn in observedColumnsDict.items():
	
		if(performRedundantCoalesce):
			observedColumn.featureConnections = observedColumn.featureConnections.coalesce()
					
		conceptIndex = observedColumn.conceptIndex
		if(drawSequenceObservedColumns):
			featureWordToIndex = sequenceObservedColumns.featureWordToIndex
			otherFeatureWordToIndex = sequenceObservedColumns.featureWordToIndex
			cIdx = sequenceObservedColumns.conceptNameToIndex[lemma]
			featureConnections = sequenceObservedColumns.featureConnections[:, :, cIdx]
		else:
			featureWordToIndex = observedColumn.featureWordToIndex
			otherFeatureWordToIndex = observedColumn.featureWordToIndex
			cIdx = databaseNetworkObject.conceptColumnsDict[lemma]
			featureConnections = observedColumn.featureConnections
		featureConnections = pt.sum(featureConnections, dim=1)	#sum along sequential segment index (draw connections to all segments)
	
		# Internal connections (yellow)
		for featureWord, featureIndexInObservedColumn in featureWordToIndex.items():
			sourceNode = f"{lemma}_{featureWord}_{featureIndexInObservedColumn}"
			if G.has_node(sourceNode):
				for otherFeatureWord, otherFeatureIndexInObservedColumn in featureWordToIndex.items():
					targetNode = f"{lemma}_{otherFeatureWord}_{otherFeatureIndexInObservedColumn}"
					if G.has_node(targetNode):
						if featureWord != otherFeatureWord:
							fIdx = featureWordToIndex[featureWord]
							otherFIdx = featureWordToIndex[otherFeatureWord]
							
							featurePresent = False
							if(featureConnections[arrayIndexPropertiesStrength, fIdx, cIdx, otherFIdx] > 0 and featureConnections[arrayIndexPropertiesPermanence, fIdx, cIdx, otherFIdx] > 0):
								featurePresent = True
								
							if(drawRelationTypes):
								connectionColor = generateFeatureNeuronColour(databaseNetworkObject, featureConnections[arrayIndexPropertiesPos, fIdx, cIdx, otherFIdx], featureWord, internalConnection=True)
							else:
								connectionColor = 'yellow'
								
							if(featurePresent):
								G.add_edge(sourceNode, targetNode, color=connectionColor)
		
		# External connections (orange)
		for featureWord, featureIndexInObservedColumn in featureWordToIndex.items():
			sourceNode = f"{lemma}_{featureWord}_{featureIndexInObservedColumn}"
			if G.has_node(sourceNode):
				for otherLemma, otherObservedColumn in observedColumnsDict.items():
					if(drawSequenceObservedColumns):
						otherFeatureWordToIndex = sequenceObservedColumns.featureWordToIndex
					else:
						otherFeatureWordToIndex = otherObservedColumn.featureWordToIndex
					for otherFeatureWord, otherFeatureIndexInObservedColumn in otherFeatureWordToIndex.items():
						targetNode = f"{otherLemma}_{otherFeatureWord}_{otherFeatureIndexInObservedColumn}"
						if G.has_node(targetNode):
							fIdx = featureWordToIndex[featureWord]
							otherFIdx = otherFeatureWordToIndex[otherFeatureWord]
							
							externalConnection = False
							if(drawSequenceObservedColumns):
								otherCIdx = sequenceObservedColumns.conceptNameToIndex[otherLemma]
								if otherCIdx != cIdx:
									externalConnection = True
							else:
								otherCIdx = databaseNetworkObject.conceptColumnsDict[otherLemma]
								if lemma != otherLemma:
									externalConnection = True
					
							featurePresent = False
							if(externalConnection):
								if(featureConnections[arrayIndexPropertiesStrength, fIdx, otherCIdx, otherFIdx] > 0 and featureConnections[arrayIndexPropertiesPermanence, fIdx, otherCIdx, otherFIdx] > 0):
									featurePresent = True

							if(drawRelationTypes):
								connectionColor = generateFeatureNeuronColour(databaseNetworkObject, featureConnections[arrayIndexPropertiesPos, fIdx, otherCIdx, otherFIdx], featureWord, internalConnection=False)
							else:
								connectionColor = 'orange'
								
							if(featurePresent):
								G.add_edge(sourceNode, targetNode, color=connectionColor)
								
	# Get positions and colors for drawing
	pos = nx.get_node_attributes(G, 'pos')
	colors = [data['color'] for node, data in G.nodes(data=True)]
	edgeColors = [data['color'] for u, v, data in G.edges(data=True)]
	labels = nx.get_node_attributes(G, 'label')

	if(save):
		highResolutionFigure = True
	else:
		if(drawHighResolutionFigure):
			highResolutionFigure = True
		else:
			highResolutionFigure = False
	if(highResolutionFigure):
		displayFigDPI = 100
		saveFigDPI = 300	#approx HD
		saveFigSize = (16,9)
		figureWidth = 1920
		figureHeight = 1080
		plt.gcf().set_size_inches(figureWidth / displayFigDPI, figureHeight / displayFigDPI)

	# Draw the graph
	nx.draw(G, pos, with_labels=True, labels=labels, arrows=True, node_color=colors, edge_color=edgeColors, node_size=500, font_size=8)
	plt.axis('off')  # Hide the axes
	
	if(save):
		if(highResolutionFigure):
			plt.savefig(fileName, dpi=saveFigDPI)
		else:
			plt.savefig(fileName)
		plt.clf()	
	else:
		plt.show()

def neuronIsActive(featureNeurons, arrayIndexProperties, featureIndexInObservedColumn):
	featureNeuronsActive = neuronActivation(featureNeurons, arrayIndexProperties, featureIndexInObservedColumn)
	featureNeuronsActive = featureNeuronsActive.item() > 0
	return featureNeuronsActive

def neuronActivation(featureNeurons, arrayIndexProperties, featureIndexInObservedColumn):
	featureNeuronsActivation = featureNeurons[arrayIndexProperties]
	featureNeuronsActivation = GIAANNproto_sparseTensors.neuronActivationSparse(featureNeuronsActivation)
	featureNeuronsActivation = featureNeuronsActivation[featureIndexInObservedColumn]
	return featureNeuronsActivation
	
		
