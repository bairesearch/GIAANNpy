"""GIAANNproto_databaseNetworkDrawExcitation.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_main.py

# Usage:
see GIAANNproto_main.py

# Description:
GIA ANN proto database Network Draw Excitation

"""

import networkx as nx
import matplotlib.pyplot as plt
import random
import torch as pt
import GIAANNproto_sparseTensors

from GIAANNproto_globalDefs import *
import GIAANNproto_databaseNetworkExcitation
if(trainInhibitoryNeurons):
	import GIAANNproto_databaseNetworkDrawInhibition

def selectDrawBranch(tensor, drawBranches):
	if(tensor is None):
		return tensor
	if(tensor.dim() > 1 and tensor.size(1) == numberOfDendriticBranches):
		if(drawBranches):
			return tensor
		if(tensor.is_sparse):
			return GIAANNproto_sparseTensors.sliceSparseTensor(tensor, 1, 0)
		return tensor[:, 0]
	return tensor

def collapseBranchDimensionForNodes(tensor, drawBranches):
	result = tensor
	if(drawBranches and multipleDendriticBranches):
		if(tensor is None):
			raise RuntimeError("collapseBranchDimensionForNodes error: tensor is None while drawBranches enabled")
		if(tensor.dim() <= 1):
			raise RuntimeError("collapseBranchDimensionForNodes error: tensor rank missing branch dimension")
		if(tensor.size(1) != numberOfDendriticBranches):
			raise RuntimeError("collapseBranchDimensionForNodes error: branch dimension mismatch")
		if(tensor.is_sparse):
			tensorCoalesced = tensor.coalesce()
			indices = tensorCoalesced.indices()
			values = tensorCoalesced.values()
			reorderedIndices = pt.cat([indices[1:2], indices[0:1], indices[2:]], dim=0)
			reorderedSize = (tensorCoalesced.size(1), tensorCoalesced.size(0)) + tensorCoalesced.size()[2:]
			reorderedTensor = pt.sparse_coo_tensor(reorderedIndices, values, size=reorderedSize, device=tensorCoalesced.device).coalesce()
			result = GIAANNproto_sparseTensors.reduceSparseBranchMax(reorderedTensor)
		else:
			result = tensor.max(dim=1).values
	return result

if(drawSegmentsTrain):
	segmentColoursBase = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan']
	segmentColours = segmentColoursBase
	if(arrayNumberOfSegments > len(segmentColoursBase)):
		repeats = (arrayNumberOfSegments + len(segmentColoursBase) - 1) // len(segmentColoursBase)
		segmentColours = (segmentColoursBase * repeats)[:arrayNumberOfSegments]
	else:
		segmentColours = segmentColoursBase[:arrayNumberOfSegments]
if(drawBranchesTrain):
	branchColoursBase = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan']
	branchColours = branchColoursBase
	if(numberOfDendriticBranches > len(branchColoursBase)):
		repeats = (numberOfDendriticBranches + len(branchColoursBase) - 1) // len(branchColoursBase)
		branchColours = (branchColoursBase * repeats)[:numberOfDendriticBranches]
	else:
		branchColours = branchColoursBase[:numberOfDendriticBranches]

if(drawRelationTypesTrain):
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
	relationTypePunctuationPos = 'PUNCT'
	relationTypePronounPos = 'PRON'

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
	neuronPosToRelationTypeDict[relationTypePunctuationPos] = 'brown'
	neuronPosToRelationTypeDict[relationTypePronounPos] = 'indigo'

	relationTypePartPropertyCol = 'cyan'
	relationTypeAuxDefinitionCol = 'blue'
	relationTypeAuxQualityCol = 'turquoise'
	relationTypeAuxActionCol = 'green'
	relationTypeAuxPropertyCol = 'cyan'
	
	relationTypeOtherCol = 'gray'	#INTJ, X, other AUX
	
	beAuxiliaries = ["am", "is", "are", "was", "were", "being", "been"]
	haveAuxiliaries = ["have", "has", "had", "having"]
	doAuxiliaries = ["do", "does", "did", "doing"]

	def generateFeatureNeuronColour(databaseNetworkObject, posFloatTorch, word, internalConnection=False, drawFeatureNodes=False):
		#print("posFloatTorch = ", posFloatTorch)
		posInt = posFloatTorch.int().item()
		posString = posIntToPosString(databaseNetworkObject.nlp, posInt)
		if(posString):
			if(posString in neuronPosToRelationTypeDict):
				if(debugDrawRelationTypesTrain and drawFeatureNodes):
					print("OK: word = ", word, ", posString = ", posString, ", posInt = ", posInt)
				colour = neuronPosToRelationTypeDict[posString]
			else:
				if(debugDrawRelationTypesTrain and drawFeatureNodes):
					print("FAIL: word = ", word, ", posString = ", posString, ", posInt = ", posInt)
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
			if(debugDrawRelationTypesTrain and drawFeatureNodes):
				print("FAIL: word = ", word, ", posString = ", posString, ", posInt = ", posInt)
			
		return colour

def getFeaturePosValue(featureNeurons, featureIndexInObservedColumn, segmentIndex):
	posValue = 0.0
	if(arrayIndexPropertiesPosIndex is not None):
		posTensor = featureNeurons[arrayIndexPropertiesPosIndex]
		if(useSANI):
			if(posTensor.is_sparse):
				posTensor = posTensor.coalesce()
				indices = posTensor.indices()
				values = posTensor.values()
				if(indices.numel() > 0):
					mask = indices[1] == featureIndexInObservedColumn
					if(mask.any()):
						posValue = float(values[mask].max().item())
			else:
				posValue = float(posTensor[:, featureIndexInObservedColumn].max().item())
		else:
			if(posTensor.is_sparse):
				posTensor = posTensor.coalesce()
				indices = posTensor.indices()
				values = posTensor.values()
				if(indices.numel() > 0):
					mask = (indices[0] == segmentIndex) & (indices[1] == featureIndexInObservedColumn)
					if(mask.any()):
						posValue = float(values[mask].max().item())
			else:
				posValue = float(posTensor[segmentIndex, featureIndexInObservedColumn].item())
	posTensorValue = pt.tensor(posValue, dtype=arrayType)
	return posTensorValue


# Initialize NetworkX graph for visualization
G = nx.DiGraph()


def createNeuronLabelWithActivation(name, value):
	label = name + "\n" + value
	return label

def intToString(value):
	result = str(int(value))
	return result
	
def floatToString(value):
	result = f"{value:.3f}"
	#result = str(round(value, 2))
	return result

def visualizeGraph(sequenceObservedColumns, inferenceMode, save=False, fileName=None):

	if(inferenceMode):
		drawRelationTypes = drawRelationTypesInference
		drawSegments = drawSegmentsInference
		drawBranches = drawBranchesInference
	else:
		drawRelationTypes = drawRelationTypesTrain
		drawSegments = drawSegmentsTrain
		drawBranches = drawBranchesTrain

	databaseNetworkObject = sequenceObservedColumns.databaseNetworkObject
	G.clear()

	if(drawAllColumns):
		observedColumnsDict = GIAANNproto_databaseNetworkExcitation.loadAllColumns(databaseNetworkObject)
	else:
		observedColumnsDict = sequenceObservedColumns.observedColumnsDict
	
	conceptIndexToLemma = {}
	for lemma, observedColumn in observedColumnsDict.items():
		conceptIndexToLemma[observedColumn.conceptIndex] = lemma

	if not lowMem:
		global globalFeatureNeurons
		if(performRedundantCoalesce):
			globalFeatureNeurons = globalFeatureNeurons.coalesce()

	excitatoryNodeMap = drawExcitatoryFeatureNeurons(sequenceObservedColumns, observedColumnsDict, databaseNetworkObject, drawRelationTypes, drawSegments, drawBranches, inferenceMode)
	if(trainInhibitoryNeurons):
		GIAANNproto_databaseNetworkDrawInhibition.drawInhibitoryFeatureNeurons(plt, G, sequenceObservedColumns, observedColumnsDict, databaseNetworkObject, conceptIndexToLemma, drawSegments, excitatoryNodeMap)

								
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

if(drawSparseArrays):
	def drawExcitatoryFeatureNeurons(sequenceObservedColumns, observedColumnsDict, databaseNetworkObject, drawRelationTypes, drawSegments, drawBranches, inferenceMode):

		# Draw concept columns
		posDict = {}
		nodeNameMap = {}
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
				featureNeurons = selectDrawBranch(sequenceObservedColumns.featureNeurons, drawBranches)
				if(drawBranches):
					featureNeurons = collapseBranchDimensionForNodes(featureNeurons, drawBranches)
				featureNeurons = featureNeurons[:, :, cIdx]
			else:
				featureWordToIndex = observedColumn.featureWordToIndex
				yOffset = 1
				if lowMem:
					featureNeurons = selectDrawBranch(observedColumn.featureNeurons, drawBranches)
					if(drawBranches):
						featureNeurons = collapseBranchDimensionForNodes(featureNeurons, drawBranches)
				else:
					featureNeurons = GIAANNproto_sparseTensors.sliceSparseTensor(databaseNetworkObject.globalFeatureNeurons, 3, conceptIndex)
					featureNeurons = selectDrawBranch(featureNeurons, drawBranches)
					if(drawBranches):
						featureNeurons = collapseBranchDimensionForNodes(featureNeurons, drawBranches)
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
				if(neuronIsActive(featureNeurons, arrayIndexPropertiesStrengthIndex, featureIndexInObservedColumn, "doNotEnforceActivationAcrossSegments")):	#if not useSANI: and neuronIsActive(featureNeurons, arrayIndexPropertiesPermanenceIndex, featureIndexInObservedColumn)
					featurePresent = True
				if(neuronIsActive(featureNeurons, arrayIndexPropertiesActivationIndex, featureIndexInObservedColumn, "doNotEnforceActivationAcrossSegments")):	#default: algorithmMatrixSANImethod
					featureActive = True
					
				if(featurePresent):
					if(drawRelationTypes):
						if not conceptNeuronFeature:
							if(useSANI and useSANIcolumns):
								segmentIndex = arrayIndexSegmentAdjacentColumn #arrayIndexSegmentAdjacentColumn has a higher probability of being filled than arrayIndexSegmentLast
							else:
								segmentIndex = arrayIndexSegmentLast
							if(arrayIndexPropertiesPosIndex is not None):
								posValue = getFeaturePosValue(featureNeurons, featureIndexInObservedColumn, segmentIndex)
								neuronColor = generateFeatureNeuronColour(databaseNetworkObject, posValue, featureWord, drawFeatureNodes=True)
					elif(featureActive):
						if(conceptNeuronFeature):
							neuronColor = 'lightskyblue'
						else:
							neuronColor = 'cyan'

					if(inferenceMode):	
						if(debugDrawNeuronActivations):
							neuronName = createNeuronLabelWithActivation(neuronName, neuronActivationString(featureNeurons, arrayIndexPropertiesActivationIndex, featureIndexInObservedColumn))

					featureNode = f"{lemma}_{featureWord}_{fIdx}"
					if(randomiseColumnFeatureXposition and not conceptNeuronFeature):
						xOffsetShuffled = xOffset + random.uniform(-0.5, 0.5)
					else:
						xOffsetShuffled = xOffset
					if(drawSequenceObservedColumns and conceptNeuronFeature):
						yOffsetPrev = yOffset
						yOffset = 1
					G.add_node(featureNode, pos=(xOffsetShuffled, yOffset), color=neuronColor, label=neuronName)
					nodeNameMap[(lemma, featureIndexInObservedColumn)] = featureNode
					if(drawSequenceObservedColumns and conceptNeuronFeature):
						yOffset = yOffsetPrev
					else:
						yOffset += 1

			# Draw rectangle around the column
			plt.gca().add_patch(plt.Rectangle((xOffset - 0.5, -0.5), 1, max(yOffset, 1) + 0.5, fill=False, edgecolor='black'))
			xOffset += 2  # Adjust xOffset for the next column

		def buildSparseConnectionLookup(featureConnectionsSparse, drawSegments, drawBranches):
			connectionLookup = {}
			featureConnectionsSparse = featureConnectionsSparse.coalesce()
			indices = featureConnectionsSparse.indices()
			values = featureConnectionsSparse.values()
			hasBranchDim = featureConnectionsSparse.dim() == 6
			useBranchKey = drawBranches and hasBranchDim
			for entryIndex in range(values.shape[0]):
				propertyIndex = int(indices[0, entryIndex].item())
				if(propertyIndex != arrayIndexPropertiesStrengthIndex and propertyIndex != arrayIndexPropertiesPermanenceIndex and propertyIndex != arrayIndexPropertiesPosIndex):
					continue
				if(hasBranchDim):
					branchIndex = int(indices[1, entryIndex].item())
					segmentIndex = int(indices[2, entryIndex].item())
					sourceFeatureIndex = int(indices[3, entryIndex].item())
					targetColumnIndex = int(indices[4, entryIndex].item())
					targetFeatureIndex = int(indices[5, entryIndex].item())
				else:
					branchIndex = 0
					segmentIndex = int(indices[1, entryIndex].item())
					sourceFeatureIndex = int(indices[2, entryIndex].item())
					targetColumnIndex = int(indices[3, entryIndex].item())
					targetFeatureIndex = int(indices[4, entryIndex].item())
				if(drawBranches and debugOnlyDrawBranchIndexConnections):
					if(branchIndex != debugOnlyDrawBranchIndexX):
						continue
				if(drawSegments):
					if(useBranchKey):
						key = (branchIndex, segmentIndex, sourceFeatureIndex, targetColumnIndex, targetFeatureIndex)
					else:
						key = (segmentIndex, sourceFeatureIndex, targetColumnIndex, targetFeatureIndex)
				else:
					if(useBranchKey):
						key = (branchIndex, sourceFeatureIndex, targetColumnIndex, targetFeatureIndex)
					else:
						key = (sourceFeatureIndex, targetColumnIndex, targetFeatureIndex)
				entry = connectionLookup.get(key)
				if(entry is None):
					entry = {"strength": 0.0, "permanence": 0.0, "pos": 0.0}
					connectionLookup[key] = entry
				value = float(values[entryIndex].item())
				if(propertyIndex == arrayIndexPropertiesStrengthIndex):
					entry["strength"] += value
				elif(propertyIndex == arrayIndexPropertiesPermanenceIndex):
					entry["permanence"] += value
				else:
					entry["pos"] += value
			return connectionLookup

		# Draw connections
		for lemma, observedColumn in observedColumnsDict.items():
		
			if(performRedundantCoalesce):
				observedColumn.featureConnections = observedColumn.featureConnections.coalesce()
						
			conceptIndex = observedColumn.conceptIndex
			if(drawSequenceObservedColumns):
				featureWordToIndex = sequenceObservedColumns.featureWordToIndex
				otherFeatureWordToIndex = sequenceObservedColumns.featureWordToIndex
				cIdx = sequenceObservedColumns.conceptNameToIndex[lemma]
				featureConnections = selectDrawBranch(sequenceObservedColumns.featureConnections, drawBranches)
				if(featureConnections.dim() == 7):
					featureConnections = featureConnections[:, :, :, cIdx]
				else:
					featureConnections = featureConnections[:, :, cIdx]
			else:
				featureWordToIndex = observedColumn.featureWordToIndex
				otherFeatureWordToIndex = observedColumn.featureWordToIndex
				cIdx = databaseNetworkObject.conceptColumnsDict[lemma]
				featureConnections = selectDrawBranch(observedColumn.featureConnections, drawBranches)
			
			useSparseConnections = featureConnections.is_sparse and not drawSequenceObservedColumns
			if(useSparseConnections):
				hasBranchDim = featureConnections.dim() == 6
				useBranchKey = drawBranches and hasBranchDim
				connectionLookup = buildSparseConnectionLookup(featureConnections, drawSegments, drawBranches)
				sourceFeatureIndexToWord = observedColumn.featureIndexToWord
				for key, entry in connectionLookup.items():
					branchIndex = 0
					if(drawSegments):
						if(useBranchKey):
							branchIndex, segmentIndex, fIdx, otherCIdx, otherFIdx = key
						else:
							segmentIndex, fIdx, otherCIdx, otherFIdx = key
					else:
						segmentIndex = None
						if(useBranchKey):
							branchIndex, fIdx, otherCIdx, otherFIdx = key
						else:
							fIdx, otherCIdx, otherFIdx = key
					if(entry["strength"] <= 0 or entry["permanence"] <= 0):
						continue
					sourceFeatureWord = sourceFeatureIndexToWord.get(fIdx)
					if(sourceFeatureWord is None):
						continue
					sourceNode = f"{lemma}_{sourceFeatureWord}_{fIdx}"
					if not G.has_node(sourceNode):
						continue
					internalConnection = (otherCIdx == cIdx)
					if(internalConnection and fIdx == otherFIdx):
						continue
					if(otherCIdx < 0 or otherCIdx >= len(databaseNetworkObject.conceptColumnsList)):
						continue
					otherLemma = databaseNetworkObject.conceptColumnsList[otherCIdx]
					otherObservedColumn = observedColumnsDict.get(otherLemma)
					if(otherObservedColumn is None):
						continue
					targetFeatureWord = otherObservedColumn.featureIndexToWord.get(otherFIdx)
					if(targetFeatureWord is None):
						continue
					targetNode = f"{otherLemma}_{targetFeatureWord}_{otherFIdx}"
					if not G.has_node(targetNode):
						continue
					if(drawSegments):
						connectionColor = segmentColours[segmentIndex]
					elif(drawBranches):
						connectionColor = branchColours[branchIndex]
					elif(drawRelationTypes):
						posValueTensor = pt.tensor(entry["pos"])
						connectionColor = generateFeatureNeuronColour(databaseNetworkObject, posValueTensor, sourceFeatureWord, internalConnection=internalConnection)
					else:
						connectionColor = 'yellow' if internalConnection else 'orange'
					G.add_edge(sourceNode, targetNode, color=connectionColor)
			else:
				hasBranchDim = featureConnections.dim() == 6
				if(drawSegments):
					if featureConnections.is_sparse:
						featureConnections = featureConnections.to_dense()
					numberOfSegmentsToIterate = arrayNumberOfSegments
				else:
					numberOfSegmentsToIterate = 1
					if(hasBranchDim):
						featureConnectionsCollapsed = pt.sum(featureConnections, dim=2)	#sum along sequential segment index (draw connections to all segments)
					else:
						featureConnectionsCollapsed = pt.sum(featureConnections, dim=1)	#sum along sequential segment index (draw connections to all segments)
				if(drawBranches and hasBranchDim):
					if(debugOnlyDrawBranchIndexConnections):
						if(debugOnlyDrawBranchIndexX >= 0 and debugOnlyDrawBranchIndexX < featureConnections.size(1)):
							branchIndices = [debugOnlyDrawBranchIndexX]
						else:
							branchIndices = []
					else:
						branchIndices = range(featureConnections.size(1))
				else:
					if(drawBranches and debugOnlyDrawBranchIndexConnections):
						if(debugOnlyDrawBranchIndexX == 0):
							branchIndices = [0]
						else:
							branchIndices = []
					else:
						branchIndices = [0]
		
				# Internal connections (yellow)
				for branchIndex in branchIndices:
					for segmentIndex in range(numberOfSegmentsToIterate):
						if(drawSegments):
							if(hasBranchDim):
								featureConnectionsSegment = featureConnections[:, branchIndex, segmentIndex]
							else:
								featureConnectionsSegment =	featureConnections[:, segmentIndex]
						else:
							if(hasBranchDim):
								if(featureConnectionsCollapsed.is_sparse):
									featureConnectionsSegment = GIAANNproto_sparseTensors.sliceSparseTensor(featureConnectionsCollapsed, 1, branchIndex)
								else:
									featureConnectionsSegment = featureConnectionsCollapsed[:, branchIndex]
							else:
								featureConnectionsSegment = featureConnectionsCollapsed

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
											if(arrayIndexPropertiesPermanenceIndex is not None):
												if(featureConnectionsSegment[arrayIndexPropertiesStrengthIndex, fIdx, cIdx, otherFIdx] > 0 and featureConnectionsSegment[arrayIndexPropertiesPermanenceIndex, fIdx, cIdx, otherFIdx] > 0):
													featurePresent = True
											else:
												if(featureConnectionsSegment[arrayIndexPropertiesStrengthIndex, fIdx, cIdx, otherFIdx] > 0):
													featurePresent = True
											
											if(drawSegments):
												connectionColor = segmentColours[segmentIndex]
											elif(drawBranches):
												connectionColor = branchColours[branchIndex]
											elif(drawRelationTypes):
												if(arrayIndexPropertiesPosIndex is not None):
													connectionColor = generateFeatureNeuronColour(databaseNetworkObject, featureConnectionsSegment[arrayIndexPropertiesPosIndex, fIdx, cIdx, otherFIdx], featureWord, internalConnection=True)
												else:
													connectionColor = 'orange'
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
												if(arrayIndexPropertiesPermanenceIndex is not None):
													if(featureConnectionsSegment[arrayIndexPropertiesStrengthIndex, fIdx, otherCIdx, otherFIdx] > 0 and featureConnectionsSegment[arrayIndexPropertiesPermanenceIndex, fIdx, otherCIdx, otherFIdx] > 0):
														featurePresent = True
												else:
													if(featureConnectionsSegment[arrayIndexPropertiesStrengthIndex, fIdx, otherCIdx, otherFIdx] > 0):
														featurePresent = True

											if(drawSegments):
												connectionColor = segmentColours[segmentIndex]
											elif(drawBranches):
												connectionColor = branchColours[branchIndex]
											elif(drawRelationTypes):
												if(arrayIndexPropertiesPosIndex is not None):
													connectionColor = generateFeatureNeuronColour(databaseNetworkObject, featureConnectionsSegment[arrayIndexPropertiesPosIndex, fIdx, otherCIdx, otherFIdx], featureWord, internalConnection=False)
												else:
													connectionColor = 'orange'
											else:
												connectionColor = 'orange'
												
											if(featurePresent):
												G.add_edge(sourceNode, targetNode, color=connectionColor)
		return nodeNameMap
else:
	def drawExcitatoryFeatureNeurons(sequenceObservedColumns, observedColumnsDict, databaseNetworkObject, drawRelationTypes, drawSegments, drawBranches, inferenceMode):

		# Draw concept columns
		posDict = {}
		nodeNameMap = {}
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
				featureNeurons = selectDrawBranch(sequenceObservedColumns.featureNeurons, drawBranches)
				if(drawBranches):
					featureNeurons = collapseBranchDimensionForNodes(featureNeurons, drawBranches)
				featureNeurons = featureNeurons[:, :, cIdx]
			else:
				featureWordToIndex = observedColumn.featureWordToIndex
				yOffset = 1
				if lowMem:
					featureNeurons = selectDrawBranch(observedColumn.featureNeurons, drawBranches)
					if(drawBranches):
						featureNeurons = collapseBranchDimensionForNodes(featureNeurons, drawBranches)
				else:
					featureNeurons = GIAANNproto_sparseTensors.sliceSparseTensor(databaseNetworkObject.globalFeatureNeurons, 3, conceptIndex)
					featureNeurons = selectDrawBranch(featureNeurons, drawBranches)
					if(drawBranches):
						featureNeurons = collapseBranchDimensionForNodes(featureNeurons, drawBranches)
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
				if(neuronIsActive(featureNeurons, arrayIndexPropertiesStrengthIndex, featureIndexInObservedColumn, "doNotEnforceActivationAcrossSegments")):	#if not useSANI: and neuronIsActive(featureNeurons, arrayIndexPropertiesPermanenceIndex, featureIndexInObservedColumn)
					featurePresent = True
				if(neuronIsActive(featureNeurons, arrayIndexPropertiesActivationIndex, featureIndexInObservedColumn, "doNotEnforceActivationAcrossSegments")):	#default: algorithmMatrixSANImethod
					featureActive = True
					
				if(featurePresent):
					if(drawRelationTypes):
						if not conceptNeuronFeature:
							if(useSANI and useSANIcolumns):
								segmentIndex = arrayIndexSegmentAdjacentColumn #arrayIndexSegmentAdjacentColumn has a higher probability of being filled than arrayIndexSegmentLast
							else:
								segmentIndex = arrayIndexSegmentLast
							if(arrayIndexPropertiesPosIndex is not None):
								posValue = getFeaturePosValue(featureNeurons, featureIndexInObservedColumn, segmentIndex)
								neuronColor = generateFeatureNeuronColour(databaseNetworkObject, posValue, featureWord, drawFeatureNodes=True)
					elif(featureActive):
						if(conceptNeuronFeature):
							neuronColor = 'lightskyblue'
						else:
							neuronColor = 'cyan'

					if(inferenceMode):	
						if(debugDrawNeuronActivations):
							neuronName = createNeuronLabelWithActivation(neuronName, neuronActivationString(featureNeurons, arrayIndexPropertiesActivationIndex, featureIndexInObservedColumn))

					featureNode = f"{lemma}_{featureWord}_{fIdx}"
					if(randomiseColumnFeatureXposition and not conceptNeuronFeature):
						xOffsetShuffled = xOffset + random.uniform(-0.5, 0.5)
					else:
						xOffsetShuffled = xOffset
					if(drawSequenceObservedColumns and conceptNeuronFeature):
						yOffsetPrev = yOffset
						yOffset = 1
					G.add_node(featureNode, pos=(xOffsetShuffled, yOffset), color=neuronColor, label=neuronName)
					nodeNameMap[(lemma, featureIndexInObservedColumn)] = featureNode
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
				featureConnections = selectDrawBranch(sequenceObservedColumns.featureConnections, drawBranches)
				if(featureConnections.dim() == 7):
					featureConnections = featureConnections[:, :, :, cIdx]
				else:
					featureConnections = featureConnections[:, :, cIdx]
			else:
				featureWordToIndex = observedColumn.featureWordToIndex
				otherFeatureWordToIndex = observedColumn.featureWordToIndex
				cIdx = databaseNetworkObject.conceptColumnsDict[lemma]
				featureConnections = selectDrawBranch(observedColumn.featureConnections, drawBranches)
			
			hasBranchDim = featureConnections.dim() == 6
			if(drawSegments):
				if featureConnections.is_sparse:
					featureConnections = featureConnections.to_dense()
				numberOfSegmentsToIterate = arrayNumberOfSegments
			else:
				numberOfSegmentsToIterate = 1
				if(hasBranchDim):
					featureConnectionsCollapsed = pt.sum(featureConnections, dim=2)	#sum along sequential segment index (draw connections to all segments)
				else:
					featureConnectionsCollapsed = pt.sum(featureConnections, dim=1)	#sum along sequential segment index (draw connections to all segments)
			if(drawBranches and hasBranchDim):
				if(debugOnlyDrawBranchIndexConnections):
					if(debugOnlyDrawBranchIndexX >= 0 and debugOnlyDrawBranchIndexX < featureConnections.size(1)):
						branchIndices = [debugOnlyDrawBranchIndexX]
					else:
						branchIndices = []
				else:
					branchIndices = range(featureConnections.size(1))
			else:
				if(drawBranches and debugOnlyDrawBranchIndexConnections):
					if(debugOnlyDrawBranchIndexX == 0):
						branchIndices = [0]
					else:
						branchIndices = []
				else:
					branchIndices = [0]
		
			# Internal connections (yellow)
			for branchIndex in branchIndices:
				for segmentIndex in range(numberOfSegmentsToIterate):
					if(drawSegments):
						if(hasBranchDim):
							featureConnectionsSegment = featureConnections[:, branchIndex, segmentIndex]
						else:
							featureConnectionsSegment =	featureConnections[:, segmentIndex]
					else:
						if(hasBranchDim):
							if(featureConnectionsCollapsed.is_sparse):
								featureConnectionsSegment = GIAANNproto_sparseTensors.sliceSparseTensor(featureConnectionsCollapsed, 1, branchIndex)
							else:
								featureConnectionsSegment = featureConnectionsCollapsed[:, branchIndex]
						else:
							featureConnectionsSegment = featureConnectionsCollapsed

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
										if(arrayIndexPropertiesPermanenceIndex is not None):
											if(featureConnectionsSegment[arrayIndexPropertiesStrengthIndex, fIdx, cIdx, otherFIdx] > 0 and featureConnectionsSegment[arrayIndexPropertiesPermanenceIndex, fIdx, cIdx, otherFIdx] > 0):
												featurePresent = True
										else:
											if(featureConnectionsSegment[arrayIndexPropertiesStrengthIndex, fIdx, cIdx, otherFIdx] > 0):
												featurePresent = True
										
										if(drawSegments):
											connectionColor = segmentColours[segmentIndex]
										elif(drawBranches):
											connectionColor = branchColours[branchIndex]
										elif(drawRelationTypes):
											if(arrayIndexPropertiesPosIndex is not None):
												connectionColor = generateFeatureNeuronColour(databaseNetworkObject, featureConnectionsSegment[arrayIndexPropertiesPosIndex, fIdx, cIdx, otherFIdx], featureWord, internalConnection=True)
											else:
												connectionColor = 'orange'
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
											if(arrayIndexPropertiesPermanenceIndex is not None):
												if(featureConnectionsSegment[arrayIndexPropertiesStrengthIndex, fIdx, otherCIdx, otherFIdx] > 0 and featureConnectionsSegment[arrayIndexPropertiesPermanenceIndex, fIdx, otherCIdx, otherFIdx] > 0):
													featurePresent = True
											else:
												if(featureConnectionsSegment[arrayIndexPropertiesStrengthIndex, fIdx, otherCIdx, otherFIdx] > 0):
													featurePresent = True

										if(drawSegments):
											connectionColor = segmentColours[segmentIndex]
										elif(drawBranches):
											connectionColor = branchColours[branchIndex]
										elif(drawRelationTypes):
											if(arrayIndexPropertiesPosIndex is not None):
												connectionColor = generateFeatureNeuronColour(databaseNetworkObject, featureConnectionsSegment[arrayIndexPropertiesPosIndex, fIdx, otherCIdx, otherFIdx], featureWord, internalConnection=False)
											else:
												connectionColor = 'orange'
										else:
											connectionColor = 'orange'
											
										if(featurePresent):
											G.add_edge(sourceNode, targetNode, color=connectionColor)
		return nodeNameMap



def neuronIsActive(featureNeurons, arrayIndexProperties, featureIndexInObservedColumn, algorithmMatrixSANImethod):
	if(arrayIndexProperties is None):
		return False
	featureNeuronsActive = neuronActivation(featureNeurons, arrayIndexProperties, featureIndexInObservedColumn, algorithmMatrixSANImethod)
	featureNeuronsActive = featureNeuronsActive.item() > 0
	return featureNeuronsActive

def neuronActivation(featureNeurons, arrayIndexProperties, featureIndexInObservedColumn, algorithmMatrixSANImethod):
	if(arrayIndexProperties is None):
		return None
	featureNeuronsActivation = featureNeurons[arrayIndexProperties]
	featureNeuronsActivation = GIAANNproto_sparseTensors.neuronActivationSparse(featureNeuronsActivation, algorithmMatrixSANImethod)	#algorithmMatrixSANImethod
	featureNeuronsActivation = featureNeuronsActivation[featureIndexInObservedColumn]
	return featureNeuronsActivation

def neuronActivationString(featureNeurons, arrayIndexProperties, featureIndexInObservedColumn):
	if(arrayIndexProperties is None):
		return ""
	string = ""
	featureNeuronsActivation = featureNeurons[arrayIndexProperties]
	for s in range(arrayNumberOfSegments):	#ignore internal column activation requirement
		value = featureNeuronsActivation[s, featureIndexInObservedColumn].item()
		if(inferenceSegmentActivationsBoolean):
			value = intToString(value)
		else:
			value = floatToString(value)
		if s != arrayNumberOfSegments-1:
			value += " "
		string += value
	return string
