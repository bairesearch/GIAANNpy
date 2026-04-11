"""GIAANNproto_databaseNetworkDrawLarge.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_main.py

# Usage:
see GIAANNproto_main.py

# Description:
GIA ANN proto database Network Draw Large

"""

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.backends.backend_svg import FigureCanvasSVG
import torch as pt

from GIAANNproto_globalDefs import *
import GIAANNproto_databaseNetworkDraw
import GIAANNproto_databaseNetworkFiles
from GIAANNproto_databaseNetworkObservedColumn import ObservedColumn


def drawDatabaseGraphStandalone(databaseNetworkObject, save=False, fileName=None, display=True):
	if(databaseNetworkObject is None):
		raise RuntimeError("drawDatabaseGraphStandalone error: databaseNetworkObject is None")
	if(not drawEfficient):
		printe("drawDatabaseGraphStandalone requires drawEfficient")
	sortedConceptColumns, activeFeatureSetsByConceptIndex, compactFeaturePositionsByConceptIndex, primeFeatureIndexByConceptIndex, nodeXs, nodeYs, nodeColors, columnRectangles, maxConceptIndex, maxFeaturePosition = prepareStandaloneDrawEfficientLayout(databaseNetworkObject)
	lineSegments, lineColours = prepareStandaloneDrawEfficientConnectionSegments(databaseNetworkObject, sortedConceptColumns, activeFeatureSetsByConceptIndex, compactFeaturePositionsByConceptIndex)

	figureWidth = max(16.0, (float(maxConceptIndex) * 0.08) + 4.0)
	figureHeight = max(9.0, (float(maxFeaturePosition) * 0.04) + 4.0)
	fig, ax = createStandaloneDrawEfficientFigure(figureWidth, figureHeight, save, display)
	if(len(lineSegments) > 0):
		lineCollection = LineCollection(lineSegments, colors=lineColours, linewidths=0.35, alpha=1.0, zorder=1)
		ax.add_collection(lineCollection)
	for rectangleX, rectangleY, rectangleWidth, rectangleHeight in columnRectangles:
		ax.add_patch(plt.Rectangle((rectangleX, rectangleY), rectangleWidth, rectangleHeight, fill=False, edgecolor='black', linewidth=0.5, zorder=2))
	if(len(nodeXs) > 0):
		ax.scatter(nodeXs, nodeYs, marker='s', s=9, c=nodeColors, edgecolors=nodeColors, linewidths=0.0, zorder=3)
	ax.set_xlim(-1.5, max(float(maxConceptIndex) + 1.5, 1.5))
	ax.set_ylim(-1.0, max(float(maxFeaturePosition) + 1.5, 1.5))
	ax.axis('off')
	ax.margins(0.0)

	if(save):
		if(fileName is None):
			fileName = drawNetworkIndependentSaveFilename
		outputFileName = GIAANNproto_databaseNetworkDraw.resolveGraphOutputFileName(fileName)
		if(drawNetworkSaveFormatVector):
			fig.savefig(outputFileName, format="svg", bbox_inches='tight', pad_inches=0.0)
		else:
			fig.savefig(outputFileName, bbox_inches='tight', pad_inches=0.0)
	if(display):
		plt.show()
	else:
		plt.close(fig)
	return

def createStandaloneDrawEfficientFigure(figureWidth, figureHeight, save, display):
	useStandaloneSvgCanvas = save and (not display) and drawNetworkSaveFormatVector
	if(useStandaloneSvgCanvas):
		fig = Figure(figsize=(figureWidth, figureHeight))
		FigureCanvasSVG(fig)
		ax = fig.add_subplot(111)
	else:
		fig, ax = plt.subplots(figsize=(figureWidth, figureHeight))
	result = (fig, ax)
	return result

def prepareStandaloneDrawEfficientLayout(databaseNetworkObject):
	if(databaseNetworkObject is None):
		raise RuntimeError("prepareStandaloneDrawEfficientLayout error: databaseNetworkObject is None")
	sortedConceptColumns = getSortedStandaloneDrawConceptColumns(databaseNetworkObject)
	useGlobalFeatureNeurons = determineStandaloneDrawNeuronSource(databaseNetworkObject)
	activeFeatureIndicesByConceptIndex, primeFeatureIndexByConceptIndex = buildStandaloneDrawColumnActivityMaps(databaseNetworkObject, sortedConceptColumns, useGlobalFeatureNeurons)
	activeFeatureSetsByConceptIndex = {}
	for conceptIndex, activeFeatureIndices in activeFeatureIndicesByConceptIndex.items():
		activeFeatureSetsByConceptIndex[conceptIndex] = set(activeFeatureIndices)
	if(drawEfficientCompact):
		compactFeaturePositionsByConceptIndex = createCompactFeaturePositionMap(sortedConceptColumns, activeFeatureIndicesByConceptIndex)
	else:
		compactFeaturePositionsByConceptIndex = None
	nodeXs = []
	nodeYs = []
	nodeColors = []
	columnRectangles = []
	maxConceptIndex = 0
	maxFeaturePosition = 0.0
	for lemma, conceptIndex, sequenceIndex in sortedConceptColumns:
		activeFeatureIndices = activeFeatureIndicesByConceptIndex[conceptIndex]
		topFeaturePosition = 0.0
		if(conceptIndex > maxConceptIndex):
			maxConceptIndex = conceptIndex
		primeFeatureIndex = primeFeatureIndexByConceptIndex[conceptIndex]
		for featureIndex in activeFeatureIndices:
			nodePosition = getNodePosition(conceptIndex, featureIndex, compactFeaturePositionsByConceptIndex)
			nodeXs.append(nodePosition[0])
			nodeYs.append(nodePosition[1])
			primeConceptNeuronFeature = useDedicatedConceptNames and useDedicatedConceptNames2 and (primeFeatureIndex is not None) and (featureIndex == primeFeatureIndex)
			if(primeConceptNeuronFeature):
				nodeColors.append(GIAANNproto_databaseNetworkDraw.defaultColourConceptFeature)
			else:
				nodeColors.append(GIAANNproto_databaseNetworkDraw.defaultColourFeature)
			if(nodePosition[1] > topFeaturePosition):
				topFeaturePosition = nodePosition[1]
			if(nodePosition[1] > maxFeaturePosition):
				maxFeaturePosition = nodePosition[1]
		columnRectangles.append((float(conceptIndex) - 0.5, -0.5, 1.0, topFeaturePosition + 1.0))
	result = (sortedConceptColumns, activeFeatureSetsByConceptIndex, compactFeaturePositionsByConceptIndex, primeFeatureIndexByConceptIndex, nodeXs, nodeYs, nodeColors, columnRectangles, maxConceptIndex, maxFeaturePosition)
	return result

def prepareStandaloneDrawEfficientConnectionSegments(databaseNetworkObject, sortedConceptColumns, activeFeatureSetsByConceptIndex, compactFeaturePositionsByConceptIndex):
	if(databaseNetworkObject is None):
		raise RuntimeError("prepareStandaloneDrawEfficientConnectionSegments error: databaseNetworkObject is None")
	if(sortedConceptColumns is None):
		raise RuntimeError("prepareStandaloneDrawEfficientConnectionSegments error: sortedConceptColumns is None")
	if(activeFeatureSetsByConceptIndex is None):
		raise RuntimeError("prepareStandaloneDrawEfficientConnectionSegments error: activeFeatureSetsByConceptIndex is None")
	lineSegments = []
	lineColours = []
	for lemma, conceptIndex, sequenceIndex in sortedConceptColumns:
		observedColumn = getStandaloneDrawObservedColumn(databaseNetworkObject, conceptIndex, lemma, sequenceIndex, loadFeatureNeurons=False)
		activeSourceFeatureIndices = activeFeatureSetsByConceptIndex[conceptIndex]
		sourceFeatureIndices = observedColumn.listStoredSourceFeatureIndices()
		for sourceFeatureIndex in sourceFeatureIndices:
			sourceFeatureIndex = int(sourceFeatureIndex)
			if(sourceFeatureIndex not in activeSourceFeatureIndices):
				continue
			sourceNodePosition = getNodePosition(conceptIndex, sourceFeatureIndex, compactFeaturePositionsByConceptIndex)
			sourceConnections = observedColumn.getFeatureConnectionsForSourceFeature(sourceFeatureIndex, targetDevice=deviceDatabase, createMissing=False)
			connectionLookup = buildStandaloneDrawConnectionLookup(databaseNetworkObject, sourceConnections)
			for (targetColumnIndex, targetFeatureIndex), connectionData in connectionLookup.items():
				targetColumnIndex = int(targetColumnIndex)
				targetFeatureIndex = int(targetFeatureIndex)
				if(connectionData["strength"] <= 0):
					continue
				if(databaseNetworkObject.arrayIndexPropertiesPermanenceIndex is not None and connectionData["permanence"] <= 0):
					continue
				if(targetColumnIndex == conceptIndex and targetFeatureIndex == sourceFeatureIndex):
					continue
				if(targetColumnIndex not in activeFeatureSetsByConceptIndex):
					continue
				if(targetFeatureIndex not in activeFeatureSetsByConceptIndex[targetColumnIndex]):
					continue
				targetNodePosition = getNodePosition(targetColumnIndex, targetFeatureIndex, compactFeaturePositionsByConceptIndex)
				lineSegments.append([sourceNodePosition, targetNodePosition])
				if(targetColumnIndex == conceptIndex):
					lineColours.append(GIAANNproto_databaseNetworkDraw.defaultConnectionColourInternal)
				else:
					lineColours.append(GIAANNproto_databaseNetworkDraw.defaultConnectionColourExternal)
			if(not storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam):
				observedColumn.unloadLoadedSourceFeatureConnections([sourceFeatureIndex])
	result = (lineSegments, lineColours)
	return result

def getSortedStandaloneDrawConceptColumns(databaseNetworkObject):
	if(databaseNetworkObject is None):
		raise RuntimeError("getSortedStandaloneDrawConceptColumns error: databaseNetworkObject is None")
	result = []
	sortedConceptItems = sorted(databaseNetworkObject.conceptColumnsDict.items(), key=lambda item: int(item[1]))
	for sequenceIndex, (lemma, conceptIndex) in enumerate(sortedConceptItems):
		result.append((lemma, int(conceptIndex), sequenceIndex))
	return result

def buildStandaloneDrawColumnActivityMaps(databaseNetworkObject, sortedConceptColumns, useGlobalFeatureNeurons):
	if(databaseNetworkObject is None):
		raise RuntimeError("buildStandaloneDrawColumnActivityMaps error: databaseNetworkObject is None")
	if(sortedConceptColumns is None):
		raise RuntimeError("buildStandaloneDrawColumnActivityMaps error: sortedConceptColumns is None")
	activeFeatureIndicesByConceptIndex = {}
	primeFeatureIndexByConceptIndex = {}
	if(storeDatabaseGlobalFeatureNeuronsInRam):
		if(not useGlobalFeatureNeurons):
			raise RuntimeError("buildStandaloneDrawColumnActivityMaps error: useGlobalFeatureNeurons must be True while storeDatabaseGlobalFeatureNeuronsInRam is True")
		activeFeatureIndicesByConceptIndex = getActiveFeatureIndicesByConceptIndexGlobal(databaseNetworkObject, sortedConceptColumns)
		for lemma, conceptIndex, sequenceIndex in sortedConceptColumns:
			observedColumn = getStandaloneDrawObservedColumn(databaseNetworkObject, conceptIndex, lemma, sequenceIndex, loadFeatureNeurons=False)
			primeFeatureIndexByConceptIndex[conceptIndex] = observedColumn.featureWordToIndex.get(variablePrimeConceptFeatureNeuronName)
	else:
		if(useGlobalFeatureNeurons):
			raise RuntimeError("buildStandaloneDrawColumnActivityMaps error: useGlobalFeatureNeurons must be False while storeDatabaseGlobalFeatureNeuronsInRam is False")
		for lemma, conceptIndex, sequenceIndex in sortedConceptColumns:
			observedColumn = getStandaloneDrawObservedColumn(databaseNetworkObject, conceptIndex, lemma, sequenceIndex, loadFeatureNeurons=True)
			if(not hasattr(observedColumn, "featureNeurons")):
				raise RuntimeError(f"buildStandaloneDrawColumnActivityMaps error: featureNeurons missing for conceptIndex {conceptIndex}")
			activeFeatureIndicesByConceptIndex[conceptIndex] = getActiveFeatureIndicesFromFeatureNeurons(databaseNetworkObject, observedColumn.featureNeurons)
			primeFeatureIndexByConceptIndex[conceptIndex] = observedColumn.featureWordToIndex.get(variablePrimeConceptFeatureNeuronName)
			if(not storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam):
				unloadStandaloneDrawObservedColumnFeatureNeurons(observedColumn)
	result = (activeFeatureIndicesByConceptIndex, primeFeatureIndexByConceptIndex)
	return result

def ensureStandaloneDrawObservedColumnFeatureState(observedColumn, newF, loadFeatureNeurons):
	if(observedColumn is None):
		raise RuntimeError("ensureStandaloneDrawObservedColumnFeatureState error: observedColumn is None")
	if(loadFeatureNeurons or storeDatabaseGlobalFeatureNeuronsInRam or hasattr(observedColumn, "featureNeurons")):
		observedColumn.ensureObservedColumnFeatureArraysFeatures(newF)
	else:
		if(observedColumn.requiresExpandFeatureMapsFeatures(newF)):
			observedColumn.expandFeatureMapsFeatures(newF)
	return

def getStandaloneDrawObservedColumn(databaseNetworkObject, conceptIndex, lemma, sequenceIndex, loadFeatureNeurons):
	if(databaseNetworkObject is None):
		raise RuntimeError("getStandaloneDrawObservedColumn error: databaseNetworkObject is None")
	if(storeDatabaseFeatureConnectionsAndColumnFeatureNeuronsInRam):
		if(not databaseNetworkObject.observedColumnsRAMLoaded):
			raise RuntimeError("getStandaloneDrawObservedColumn error: observedColumnsRAMLoaded is False")
		if(databaseNetworkObject.observedColumnsDictRAM is None):
			raise RuntimeError("getStandaloneDrawObservedColumn error: observedColumnsDictRAM is None")
		if(lemma not in databaseNetworkObject.observedColumnsDictRAM):
			raise RuntimeError(f"getStandaloneDrawObservedColumn error: missing RAM observed column for lemma {lemma}")
		observedColumn = databaseNetworkObject.observedColumnsDictRAM[lemma]
		ensureStandaloneDrawObservedColumnFeatureState(observedColumn, databaseNetworkObject.f, loadFeatureNeurons)
		if(loadFeatureNeurons and (not storeDatabaseGlobalFeatureNeuronsInRam) and (not hasattr(observedColumn, "featureNeurons"))):
			raise RuntimeError(f"getStandaloneDrawObservedColumn error: RAM observed column missing featureNeurons for lemma {lemma}")
	else:
		if(GIAANNproto_databaseNetworkFiles.observedColumnHasPersistedData(conceptIndex)):
			if(not GIAANNproto_databaseNetworkFiles.observedColumnHasConsistentPersistedMetadata(conceptIndex)):
				raise RuntimeError(f"getStandaloneDrawObservedColumn error: inconsistent observed column storage for conceptIndex {conceptIndex}, lemma {lemma}")
			observedColumn = ObservedColumn.loadFromDisk(databaseNetworkObject, conceptIndex, lemma, sequenceIndex, targetDevice=deviceDatabase, loadAllSourceFeatures=False, resizeFeatureTensorsToCurrentSize=False, loadFeatureNeurons=loadFeatureNeurons)
		else:
			observedColumn = ObservedColumn(databaseNetworkObject, conceptIndex, lemma, sequenceIndex)
			observedColumn.ensureObservedColumnFeatureArraysFeatures(databaseNetworkObject.f)
	result = observedColumn
	return result

def unloadStandaloneDrawObservedColumnFeatureNeurons(observedColumn):
	if(observedColumn is None):
		raise RuntimeError("unloadStandaloneDrawObservedColumnFeatureNeurons error: observedColumn is None")
	if(hasattr(observedColumn, "featureNeurons")):
		del observedColumn.featureNeurons
	return

def determineStandaloneDrawNeuronSource(databaseNetworkObject):
	if(databaseNetworkObject is None):
		raise RuntimeError("determineStandaloneDrawNeuronSource error: databaseNetworkObject is None")
	useGlobalFeatureNeurons = storeDatabaseGlobalFeatureNeuronsInRam
	if(useGlobalFeatureNeurons):
		if(databaseNetworkObject.globalFeatureNeurons is None):
			raise RuntimeError("determineStandaloneDrawNeuronSource error: globalFeatureNeurons is required but is None")
	result = useGlobalFeatureNeurons
	return result

def getPositiveSparseFeatureIndicesByConceptIndex(databaseNetworkObject, featureNeurons, featureIndexPosition):
	if(databaseNetworkObject is None):
		raise RuntimeError("getPositiveSparseFeatureIndicesByConceptIndex error: databaseNetworkObject is None")
	if(featureNeurons is None):
		raise RuntimeError("getPositiveSparseFeatureIndicesByConceptIndex error: featureNeurons is None")
	resultStrength = {}
	resultPermanence = {}
	featureNeurons = featureNeurons.coalesce()
	indices = featureNeurons.indices()
	values = featureNeurons.values()
	for entryIndex in range(values.shape[0]):
		value = float(values[entryIndex].item())
		if(value <= 0):
			continue
		propertyIndex = int(indices[0, entryIndex].item())
		featureIndex = int(indices[featureIndexPosition, entryIndex].item())
		if(featureIndexPosition == 4):
			conceptIndex = int(indices[3, entryIndex].item())
			if(propertyIndex == databaseNetworkObject.arrayIndexPropertiesStrengthIndex):
				if(conceptIndex not in resultStrength):
					resultStrength[conceptIndex] = set()
				resultStrength[conceptIndex].add(featureIndex)
			elif(databaseNetworkObject.arrayIndexPropertiesPermanenceIndex is not None and propertyIndex == databaseNetworkObject.arrayIndexPropertiesPermanenceIndex):
				if(conceptIndex not in resultPermanence):
					resultPermanence[conceptIndex] = set()
				resultPermanence[conceptIndex].add(featureIndex)
		else:
			if(propertyIndex == databaseNetworkObject.arrayIndexPropertiesStrengthIndex):
				resultStrength[featureIndex] = True
			elif(databaseNetworkObject.arrayIndexPropertiesPermanenceIndex is not None and propertyIndex == databaseNetworkObject.arrayIndexPropertiesPermanenceIndex):
				resultPermanence[featureIndex] = True
	return resultStrength, resultPermanence

def getActiveFeatureIndicesByConceptIndexGlobal(databaseNetworkObject, sortedConceptColumns):
	if(databaseNetworkObject is None):
		raise RuntimeError("getActiveFeatureIndicesByConceptIndexGlobal error: databaseNetworkObject is None")
	if(sortedConceptColumns is None):
		raise RuntimeError("getActiveFeatureIndicesByConceptIndexGlobal error: sortedConceptColumns is None")
	if(databaseNetworkObject.globalFeatureNeurons is None):
		raise RuntimeError("getActiveFeatureIndicesByConceptIndexGlobal error: globalFeatureNeurons is None")
	result = {}
	if(databaseNetworkObject.globalFeatureNeurons.is_sparse):
		globalFeatureNeurons = databaseNetworkObject.globalFeatureNeurons.coalesce()
		if(drawEfficientDrawDeadNeurons):
			indices = globalFeatureNeurons.indices()
			strengthFeatureIndicesByConceptIndex = {}
			permanenceFeatureIndicesByConceptIndex = {}
			for entryIndex in range(indices.shape[1]):
				propertyIndex = int(indices[0, entryIndex].item())
				conceptIndex = int(indices[3, entryIndex].item())
				featureIndex = int(indices[4, entryIndex].item())
				if(propertyIndex == databaseNetworkObject.arrayIndexPropertiesStrengthIndex):
					if(conceptIndex not in strengthFeatureIndicesByConceptIndex):
						strengthFeatureIndicesByConceptIndex[conceptIndex] = set()
					strengthFeatureIndicesByConceptIndex[conceptIndex].add(featureIndex)
				elif(databaseNetworkObject.arrayIndexPropertiesPermanenceIndex is not None and propertyIndex == databaseNetworkObject.arrayIndexPropertiesPermanenceIndex):
					if(conceptIndex not in permanenceFeatureIndicesByConceptIndex):
						permanenceFeatureIndicesByConceptIndex[conceptIndex] = set()
					permanenceFeatureIndicesByConceptIndex[conceptIndex].add(featureIndex)
		else:
			strengthFeatureIndicesByConceptIndex, permanenceFeatureIndicesByConceptIndex = getPositiveSparseFeatureIndicesByConceptIndex(databaseNetworkObject, globalFeatureNeurons, 4)
		for lemma, conceptIndex, sequenceIndex in sortedConceptColumns:
			activeFeatureIndices = strengthFeatureIndicesByConceptIndex.get(conceptIndex, set())
			if(databaseNetworkObject.arrayIndexPropertiesPermanenceIndex is not None):
				activeFeatureIndices = activeFeatureIndices.intersection(permanenceFeatureIndicesByConceptIndex.get(conceptIndex, set()))
			result[conceptIndex] = sorted(activeFeatureIndices)
	else:
		for lemma, conceptIndex, sequenceIndex in sortedConceptColumns:
			featureNeurons = databaseNetworkObject.globalFeatureNeurons[:, :, :, conceptIndex]
			result[conceptIndex] = getActiveFeatureIndicesFromFeatureNeurons(databaseNetworkObject, featureNeurons)
	return result

def getActiveFeatureIndicesFromFeatureNeurons(databaseNetworkObject, featureNeurons):
	if(databaseNetworkObject is None):
		raise RuntimeError("getActiveFeatureIndicesFromFeatureNeurons error: databaseNetworkObject is None")
	if(featureNeurons is None):
		raise RuntimeError("getActiveFeatureIndicesFromFeatureNeurons error: featureNeurons is None")
	result = []
	if(featureNeurons.is_sparse):
		featureNeurons = featureNeurons.coalesce()
		if(drawEfficientDrawDeadNeurons):
			indices = featureNeurons.indices()
			if(indices.numel() > 0):
				strengthMask = indices[0] == databaseNetworkObject.arrayIndexPropertiesStrengthIndex
				strengthFeatureIndices = set(int(featureIndex) for featureIndex in indices[3, strengthMask].detach().cpu().tolist())
				activeFeatureIndices = strengthFeatureIndices
				if(databaseNetworkObject.arrayIndexPropertiesPermanenceIndex is not None):
					permanenceMask = indices[0] == databaseNetworkObject.arrayIndexPropertiesPermanenceIndex
					permanenceFeatureIndices = set(int(featureIndex) for featureIndex in indices[3, permanenceMask].detach().cpu().tolist())
					activeFeatureIndices = strengthFeatureIndices.intersection(permanenceFeatureIndices)
				result = sorted(activeFeatureIndices)
		else:
			strengthFeatureIndices, permanenceFeatureIndices = getPositiveSparseFeatureIndicesByConceptIndex(databaseNetworkObject, featureNeurons, 3)
			activeFeatureIndices = set(strengthFeatureIndices.keys())
			if(databaseNetworkObject.arrayIndexPropertiesPermanenceIndex is not None):
				activeFeatureIndices = activeFeatureIndices.intersection(set(permanenceFeatureIndices.keys()))
			result = sorted(activeFeatureIndices)
	else:
		strengthTensor = featureNeurons[databaseNetworkObject.arrayIndexPropertiesStrengthIndex].sum(dim=0).sum(dim=0)
		if(databaseNetworkObject.arrayIndexPropertiesPermanenceIndex is not None):
			permanenceTensor = featureNeurons[databaseNetworkObject.arrayIndexPropertiesPermanenceIndex].sum(dim=0).sum(dim=0)
			activeMask = (strengthTensor > 0) & (permanenceTensor > 0)
		else:
			activeMask = strengthTensor > 0
		result = [int(featureIndex) for featureIndex in pt.nonzero(activeMask, as_tuple=False).view(-1).detach().cpu().tolist()]
	return result

def createCompactFeaturePositionMap(sortedConceptColumns, activeFeatureIndicesByConceptIndex):
	if(sortedConceptColumns is None):
		raise RuntimeError("createCompactFeaturePositionMap error: sortedConceptColumns is None")
	if(activeFeatureIndicesByConceptIndex is None):
		raise RuntimeError("createCompactFeaturePositionMap error: activeFeatureIndicesByConceptIndex is None")
	result = {}
	for lemma, conceptIndex, sequenceIndex in sortedConceptColumns:
		result[conceptIndex] = {}
		for compactFeatureIndex, featureIndex in enumerate(activeFeatureIndicesByConceptIndex[conceptIndex]):
			result[conceptIndex][int(featureIndex)] = int(compactFeatureIndex)
	return result

def getNodePosition(conceptIndex, featureIndex, compactFeaturePositionsByConceptIndex=None):
	if(not drawEfficient):
		raise RuntimeError("getNodePosition error: drawEfficient is False")
	if(drawEfficientGrid):
		xPosition = float(conceptIndex)
		yPosition = float(featureIndex)
	elif(drawEfficientCompact):
		if(compactFeaturePositionsByConceptIndex is None):
			raise RuntimeError("getNodePosition error: compactFeaturePositionsByConceptIndex is None while drawEfficientCompact is enabled")
		if(conceptIndex not in compactFeaturePositionsByConceptIndex):
			raise RuntimeError(f"getNodePosition error: missing compact feature positions for conceptIndex {conceptIndex}")
		if(featureIndex not in compactFeaturePositionsByConceptIndex[conceptIndex]):
			raise RuntimeError(f"getNodePosition error: missing compact feature position for conceptIndex {conceptIndex}, featureIndex {featureIndex}")
		xPosition = float(conceptIndex)
		yPosition = float(compactFeaturePositionsByConceptIndex[conceptIndex][featureIndex])
	else:
		raise RuntimeError("getNodePosition error: unsupported drawEfficient configuration")
	result = (xPosition, yPosition)
	return result

def buildStandaloneDrawConnectionLookup(databaseNetworkObject, sourceConnections):
	result = {}
	if(sourceConnections is None):
		raise RuntimeError("buildStandaloneDrawConnectionLookup error: sourceConnections is None")
	if(sourceConnections.dim() != 5):
		raise RuntimeError("buildStandaloneDrawConnectionLookup error: sourceConnections rank must be 5")
	if(sourceConnections.is_sparse):
		sourceConnections = sourceConnections.coalesce()
		indices = sourceConnections.indices()
		values = sourceConnections.values()
		for entryIndex in range(values.shape[0]):
			propertyIndex = int(indices[0, entryIndex].item())
			if(propertyIndex != databaseNetworkObject.arrayIndexPropertiesStrengthIndex and propertyIndex != databaseNetworkObject.arrayIndexPropertiesPermanenceIndex):
				continue
			targetColumnIndex = int(indices[3, entryIndex].item())
			targetFeatureIndex = int(indices[4, entryIndex].item())
			key = (targetColumnIndex, targetFeatureIndex)
			entry = result.get(key)
			if(entry is None):
				entry = {"strength": 0.0, "permanence": 0.0}
				result[key] = entry
			value = float(values[entryIndex].item())
			if(propertyIndex == databaseNetworkObject.arrayIndexPropertiesStrengthIndex):
				entry["strength"] = entry["strength"] + value
			elif(propertyIndex == databaseNetworkObject.arrayIndexPropertiesPermanenceIndex):
				entry["permanence"] = entry["permanence"] + value
	else:
		strengthTensor = sourceConnections[databaseNetworkObject.arrayIndexPropertiesStrengthIndex].sum(dim=0).sum(dim=0)
		if(databaseNetworkObject.arrayIndexPropertiesPermanenceIndex is not None):
			permanenceTensor = sourceConnections[databaseNetworkObject.arrayIndexPropertiesPermanenceIndex].sum(dim=0).sum(dim=0)
			connectionIndices = pt.nonzero((strengthTensor > 0) & (permanenceTensor > 0), as_tuple=False)
		else:
			connectionIndices = pt.nonzero(strengthTensor > 0, as_tuple=False)
		for targetColumnIndex, targetFeatureIndex in connectionIndices.tolist():
			key = (int(targetColumnIndex), int(targetFeatureIndex))
			entry = {"strength": float(strengthTensor[targetColumnIndex, targetFeatureIndex].item()), "permanence": 1.0}
			if(databaseNetworkObject.arrayIndexPropertiesPermanenceIndex is not None):
				entry["permanence"] = float(permanenceTensor[targetColumnIndex, targetFeatureIndex].item())
			result[key] = entry
	return result
