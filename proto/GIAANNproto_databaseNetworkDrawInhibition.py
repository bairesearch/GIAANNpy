"""GIAANNproto_databaseNetworkDrawInhibition.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2025 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_main.py

# Usage:
see GIAANNproto_main.py

# Description:
GIA ANN proto database Network Draw Inhibition

"""

import random
import torch as pt
import GIAANNproto_sparseTensors

from GIAANNproto_globalDefs import *
import GIAANNproto_databaseNetworkFilesInhibition

if(drawSegmentsTrain):
	segmentColours = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan']	#len must be >= arrayNumberOfSegments

inhibitoryInputConnectionColor = '#8B0000'	# dark red
inhibitoryOutputConnectionColor = '#FFB6C1'	# light pink


def drawInhibitoryFeatureNeurons(plt, G, sequenceObservedColumns, observedColumnsDict, databaseNetworkObject, conceptIndexToLemma, drawSegments, excitatoryNodeMap):
	xOffset = 0
	for lemma, observedColumn in observedColumnsDict.items():
		inhibitoryColumn = GIAANNproto_databaseNetworkFilesInhibition.getInhibitoryObservedColumn(databaseNetworkObject, observedColumn.conceptIndex, lemma)
		if inhibitoryColumn is None:
			xOffset += 2
			continue
		featureNeuronsTensor = inhibitoryColumn.featureNeurons
		if featureNeuronsTensor.is_sparse:
			if(featureNeuronsTensor._nnz() == 0):
				xOffset += 2
				continue
			featureNeurons = featureNeuronsTensor.to_dense()
		else:
			if not pt.any(featureNeuronsTensor):
				xOffset += 2
				continue
			featureNeurons = featureNeuronsTensor.clone()
		yOffset = inhibitoryNeuronYoffset
		columnNodeMap = {}
		for featureIndex in range(featureNeurons.shape[2]):
			featurePresent = neuronIsActive(featureNeurons, arrayIndexPropertiesStrengthIndex, featureIndex, "doNotEnforceActivationAcrossSegments")
			if(not featurePresent):
				continue
			featureActive = neuronIsActive(featureNeurons, arrayIndexPropertiesActivationIndex, featureIndex, "doNotEnforceActivationAcrossSegments")
			neuronColor = 'crimson' if featureActive else 'lightcoral'
			featureName = getInhibitoryFeatureName(observedColumn, databaseNetworkObject, featureIndex, lemma)
			if(featureName is None):
				neuronLabel = f"Inhib {featureIndex}"
			else:
				neuronLabel = featureName
			featureNode = f"inhib_{lemma}_{featureIndex}"
			if(randomiseColumnFeatureXposition):
				xOffsetShuffled = xOffset + random.uniform(-0.5, 0.5)
			else:
				xOffsetShuffled = xOffset
			G.add_node(featureNode, pos=(xOffsetShuffled, yOffset), color=neuronColor, label=neuronLabel)
			columnNodeMap[featureIndex] = featureNode
			yOffset += 1
		if(columnNodeMap):
			plt.gca().add_patch(plt.Rectangle((xOffset - 0.5, inhibitoryNeuronYoffset - 0.5), 1, max(yOffset - inhibitoryNeuronYoffset, 1) + 0.5, fill=False, edgecolor='red', linestyle='--'))
			drawInhibitoryInputConnections(plt, G, inhibitoryColumn, columnNodeMap, excitatoryNodeMap, conceptIndexToLemma, drawSegments)
			drawInhibitoryOutputConnections(plt, G, inhibitoryColumn, columnNodeMap, excitatoryNodeMap, conceptIndexToLemma, drawSegments)
		xOffset += 2

def getInhibitoryFeatureName(observedColumn, databaseNetworkObject, featureIndex, lemma):
	if(observedColumn is not None):
		featureIndexToWord = getattr(observedColumn, "featureIndexToWord", None)
		if(featureIndexToWord is not None):
			name = featureIndexToWord.get(featureIndex)
			if(name):
				return name
	conceptFeaturesList = getattr(databaseNetworkObject, "conceptFeaturesList", None)
	if(conceptFeaturesList is not None and featureIndex >= 0 and featureIndex < len(conceptFeaturesList)):
		return conceptFeaturesList[featureIndex]
	return f"{lemma}:{featureIndex}"

def drawInhibitoryInputConnections(plt, G, inhibitoryColumn, columnNodeMap, excitatoryNodeMap, conceptIndexToLemma, drawSegments):
	if not columnNodeMap:
		return
	connectionsTensor = inhibitoryColumn.featureConnectionsInput
	if connectionsTensor.is_sparse:
		if(connectionsTensor._nnz() == 0):
			return
		connectionsTensor = connectionsTensor.to_dense()
	else:
		if not pt.any(connectionsTensor):
			return
	if(drawSegments):
		numberOfSegmentsToIterate = arrayNumberOfSegments
	else:
		numberOfSegmentsToIterate = 1
		connectionsTensor = connectionsTensor.sum(dim=1, keepdim=True)
	for segmentIndex in range(numberOfSegmentsToIterate):
		segmentSlice = connectionsTensor[:, segmentIndex]
		strengthSlice = segmentSlice[arrayIndexPropertiesStrengthIndex]
		permanenceSlice = segmentSlice[arrayIndexPropertiesPermanenceIndex]
		activeMask = (strengthSlice > 0) & (permanenceSlice > 0)
		activeIndices = pt.nonzero(activeMask, as_tuple=False)
		if(activeIndices.numel() == 0):
			continue
		for idx in activeIndices:
			inhibFeatureIndex = int(idx[0].item())
			sourceConceptIndex = int(idx[1].item())
			sourceFeatureIndex = int(idx[2].item())
			targetNode = columnNodeMap.get(inhibFeatureIndex)
			if targetNode is None or not G.has_node(targetNode):
				continue
			sourceLemma = conceptIndexToLemma.get(sourceConceptIndex)
			if sourceLemma is None:
				continue
			sourceNode = excitatoryNodeMap.get((sourceLemma, sourceFeatureIndex))
			if sourceNode is None or not G.has_node(sourceNode):
				continue
			connectionColor = inhibitoryInputConnectionColor
			G.add_edge(sourceNode, targetNode, color=connectionColor)

def drawInhibitoryOutputConnections(plt, G, inhibitoryColumn, columnNodeMap, excitatoryNodeMap, conceptIndexToLemma, drawSegments):
	if not columnNodeMap:
		return
	connectionsTensor = inhibitoryColumn.featureConnectionsOutput
	if connectionsTensor.is_sparse:
		if(connectionsTensor._nnz() == 0):
			return
		connectionsTensor = connectionsTensor.to_dense()
	else:
		if not pt.any(connectionsTensor):
			return
	if(drawSegments):
		numberOfSegmentsToIterate = arrayNumberOfSegments
	else:
		numberOfSegmentsToIterate = 1
		connectionsTensor = connectionsTensor.sum(dim=1, keepdim=True)
	for segmentIndex in range(numberOfSegmentsToIterate):
		segmentSlice = connectionsTensor[:, segmentIndex]
		strengthSlice = segmentSlice[arrayIndexPropertiesStrengthIndex]
		permanenceSlice = segmentSlice[arrayIndexPropertiesPermanenceIndex]
		activeMask = (strengthSlice > 0) & (permanenceSlice > 0)
		activeIndices = pt.nonzero(activeMask, as_tuple=False)
		if(activeIndices.numel() == 0):
			continue
		for idx in activeIndices:
			inhibFeatureIndex = int(idx[0].item())
			targetConceptIndex = int(idx[1].item())
			targetFeatureIndex = int(idx[2].item())
			sourceNode = columnNodeMap.get(inhibFeatureIndex)
			if sourceNode is None or not G.has_node(sourceNode):
				continue
			targetLemma = conceptIndexToLemma.get(targetConceptIndex)
			if targetLemma is None:
				continue
			targetNode = excitatoryNodeMap.get((targetLemma, targetFeatureIndex))
			if targetNode is None or not G.has_node(targetNode):
				continue
			connectionColor = inhibitoryOutputConnectionColor
			#print(f"draw inhibitory output connection: column {inhibitoryColumn.conceptName}, inhibitory feature {inhibFeatureIndex} -> {targetLemma}:{targetFeatureIndex}")
			G.add_edge(sourceNode, targetNode, color=connectionColor)

def neuronIsActive(featureNeurons, arrayIndexProperties, featureIndexInObservedColumn, algorithmMatrixSANImethod):
	featureNeuronsActive = neuronActivation(featureNeurons, arrayIndexProperties, featureIndexInObservedColumn, algorithmMatrixSANImethod)
	featureNeuronsActive = featureNeuronsActive.item() > 0
	return featureNeuronsActive

def neuronActivation(featureNeurons, arrayIndexProperties, featureIndexInObservedColumn, algorithmMatrixSANImethod):
	featureNeuronsActivation = featureNeurons[arrayIndexProperties]
	featureNeuronsActivation = GIAANNproto_sparseTensors.neuronActivationSparse(featureNeuronsActivation, algorithmMatrixSANImethod)	#algorithmMatrixSANImethod
	featureNeuronsActivation = featureNeuronsActivation[featureIndexInObservedColumn]
	return featureNeuronsActivation
