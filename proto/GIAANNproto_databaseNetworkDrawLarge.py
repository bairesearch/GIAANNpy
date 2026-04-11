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

import math
import os
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.backends.backend_svg import FigureCanvasSVG
import torch as pt

from GIAANNproto_globalDefs import *
import GIAANNproto_databaseNetworkDraw
import GIAANNproto_databaseNetworkFiles
from GIAANNproto_databaseNetworkObservedColumn import ObservedColumn

standaloneDrawEfficient3DLdrNodePartFile = "4-4CUBE.DAT"
standaloneDrawEfficient3DLdrNodeScale = 4.0
standaloneDrawEfficient3DLdrColumnSpacing = 40.0
standaloneDrawEfficient3DLdrFeatureSpacing = 20.0
standaloneDrawEfficient3DLdrColumnDepth = 20.0
standaloneDrawEfficient3DLdrColourConceptNode = 1
standaloneDrawEfficient3DLdrColourFeatureNode = 3
standaloneDrawEfficient3DLdrColourInternalConnection = 14
standaloneDrawEfficient3DLdrColourExternalConnection = 25
standaloneDrawEfficient3DLdrColourColumnBox = 7


def drawDatabaseGraphStandalone(databaseNetworkObject, save=False, fileName=None, display=True):
	if(databaseNetworkObject is None):
		raise RuntimeError("drawDatabaseGraphStandalone error: databaseNetworkObject is None")
	if(not drawEfficient):
		printe("drawDatabaseGraphStandalone requires drawEfficient")
	sortedConceptColumns, activeFeatureSetsByConceptIndex, compactFeaturePositionsByConceptIndex, primeFeatureIndexByConceptIndex, nodeXs, nodeYs, nodeColors, columnRectangles, maxConceptIndex, maxFeaturePosition = prepareStandaloneDrawEfficientLayout(databaseNetworkObject)
	lineSegments, lineColours = prepareStandaloneDrawEfficientConnectionSegments(databaseNetworkObject, sortedConceptColumns, activeFeatureSetsByConceptIndex, compactFeaturePositionsByConceptIndex)
	if(drawEfficientFormat3D):
		if(not save):
			raise RuntimeError("drawDatabaseGraphStandalone error: drawEfficientFormat3D requires save=True")
		if(display):
			raise RuntimeError("drawDatabaseGraphStandalone error: drawEfficientFormat3D requires display=False")
		if(fileName is None):
			fileName = drawNetworkIndependentSaveFilename
		outputFileName = resolveStandaloneDrawEfficient3DOutputFileName(fileName)
		saveStandaloneDrawEfficientLdrFile(outputFileName, sortedConceptColumns, nodeXs, nodeYs, nodeColors, columnRectangles, lineSegments, lineColours)
	else:
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

def saveStandaloneDrawEfficientLdrFile(outputFileName, sortedConceptColumns, nodeXs, nodeYs, nodeColors, columnRectangles, lineSegments, lineColours):
	if(outputFileName is None):
		raise RuntimeError("saveStandaloneDrawEfficientLdrFile error: outputFileName is None")
	ldrLines = generateStandaloneDrawEfficientLdrLines(outputFileName, sortedConceptColumns, nodeXs, nodeYs, nodeColors, columnRectangles, lineSegments, lineColours)
	with open(outputFileName, "w", encoding="ascii") as outputFile:
		outputFile.write("\n".join(ldrLines))
		outputFile.write("\n")
	return

def generateStandaloneDrawEfficientLdrLines(outputFileName, sortedConceptColumns, nodeXs, nodeYs, nodeColors, columnRectangles, lineSegments, lineColours):
	if(outputFileName is None):
		raise RuntimeError("generateStandaloneDrawEfficientLdrLines error: outputFileName is None")
	if(sortedConceptColumns is None):
		raise RuntimeError("generateStandaloneDrawEfficientLdrLines error: sortedConceptColumns is None")
	if(len(nodeXs) != len(nodeYs) or len(nodeXs) != len(nodeColors)):
		raise RuntimeError("generateStandaloneDrawEfficientLdrLines error: node coordinate arrays are inconsistent")
	if(len(lineSegments) != len(lineColours)):
		raise RuntimeError("generateStandaloneDrawEfficientLdrLines error: connection arrays are inconsistent")
	totalNumberColumns = calculateStandaloneDrawEfficientLdrTotalNumberColumns(sortedConceptColumns)
	totalNumberFeatures = calculateStandaloneDrawEfficientLdrTotalNumberFeatures(nodeYs)
	lines = []
	lines.append("0 GIAANN drawEfficientFormat3D")
	lines.append("0 Name: " + os.path.basename(outputFileName))
	lines.append("0 Author: Codex")
	appendStandaloneDrawEfficientLdrNodes(lines, totalNumberColumns, totalNumberFeatures, nodeXs, nodeYs, nodeColors)
	appendStandaloneDrawEfficientLdrConnections(lines, totalNumberColumns, totalNumberFeatures, lineSegments, lineColours)
	appendStandaloneDrawEfficientLdrColumnBoxes(lines, totalNumberColumns, totalNumberFeatures, columnRectangles)
	result = lines
	return result

def calculateStandaloneDrawEfficientLdrTotalNumberColumns(sortedConceptColumns):
	if(sortedConceptColumns is None):
		raise RuntimeError("calculateStandaloneDrawEfficientLdrTotalNumberColumns error: sortedConceptColumns is None")
	result = int(len(sortedConceptColumns))
	return result

def calculateStandaloneDrawEfficientLdrTotalNumberFeatures(nodeYs):
	if(nodeYs is None):
		raise RuntimeError("calculateStandaloneDrawEfficientLdrTotalNumberFeatures error: nodeYs is None")
	totalNumberFeatures = 0
	if(len(nodeYs) > 0):
		maxFeaturePosition = max(float(nodeY) for nodeY in nodeYs)
		totalNumberFeatures = int(math.floor(maxFeaturePosition)) + 1
	result = totalNumberFeatures
	return result

def appendStandaloneDrawEfficientLdrColumnBoxes(lines, totalNumberColumns, totalNumberFeatures, columnRectangles):
	if(lines is None):
		raise RuntimeError("appendStandaloneDrawEfficientLdrColumnBoxes error: lines is None")
	if(columnRectangles is None):
		raise RuntimeError("appendStandaloneDrawEfficientLdrColumnBoxes error: columnRectangles is None")
	for rectangle in columnRectangles:
		appendStandaloneDrawEfficientLdrBox(lines, totalNumberColumns, totalNumberFeatures, rectangle, standaloneDrawEfficient3DLdrColourColumnBox)
	return

def appendStandaloneDrawEfficientLdrConnections(lines, totalNumberColumns, totalNumberFeatures, lineSegments, lineColours):
	if(lines is None):
		raise RuntimeError("appendStandaloneDrawEfficientLdrConnections error: lines is None")
	if(lineSegments is None):
		raise RuntimeError("appendStandaloneDrawEfficientLdrConnections error: lineSegments is None")
	if(lineColours is None):
		raise RuntimeError("appendStandaloneDrawEfficientLdrConnections error: lineColours is None")
	for lineSegmentIndex, lineSegment in enumerate(lineSegments):
		lineColour = getStandaloneDrawEfficientLdrColourFromConnectionColour(lineColours[lineSegmentIndex])
		lineStart, lineEnd = getStandaloneDrawEfficientLdrLineEndpoints(totalNumberColumns, totalNumberFeatures, lineSegment)
		appendStandaloneDrawEfficientLdrLine(lines, lineColour, lineStart, lineEnd)
	return

def appendStandaloneDrawEfficientLdrNodes(lines, totalNumberColumns, totalNumberFeatures, nodeXs, nodeYs, nodeColors):
	if(lines is None):
		raise RuntimeError("appendStandaloneDrawEfficientLdrNodes error: lines is None")
	if(nodeXs is None or nodeYs is None or nodeColors is None):
		raise RuntimeError("appendStandaloneDrawEfficientLdrNodes error: node arrays are None")
	for nodeIndex in range(len(nodeXs)):
		nodePosition = getStandaloneDrawEfficientLdrNodePosition(totalNumberColumns, totalNumberFeatures, nodeXs[nodeIndex], nodeYs[nodeIndex])
		nodeColour = getStandaloneDrawEfficientLdrColourFromNodeColour(nodeColors[nodeIndex])
		appendStandaloneDrawEfficientLdrNode(lines, nodeColour, nodePosition)
	return

def appendStandaloneDrawEfficientLdrBox(lines, totalNumberColumns, totalNumberFeatures, rectangle, lineColour):
	if(lines is None):
		raise RuntimeError("appendStandaloneDrawEfficientLdrBox error: lines is None")
	if(rectangle is None):
		raise RuntimeError("appendStandaloneDrawEfficientLdrBox error: rectangle is None")
	frontPoints, backPoints = getStandaloneDrawEfficientLdrRectanglePoints(totalNumberColumns, totalNumberFeatures, rectangle)
	for pointIndex in range(4):
		nextPointIndex = (pointIndex + 1) % 4
		appendStandaloneDrawEfficientLdrLine(lines, lineColour, frontPoints[pointIndex], frontPoints[nextPointIndex])
		appendStandaloneDrawEfficientLdrLine(lines, lineColour, backPoints[pointIndex], backPoints[nextPointIndex])
		appendStandaloneDrawEfficientLdrLine(lines, lineColour, frontPoints[pointIndex], backPoints[pointIndex])
	return

def appendStandaloneDrawEfficientLdrLine(lines, lineColour, pointStart, pointEnd):
	if(lines is None):
		raise RuntimeError("appendStandaloneDrawEfficientLdrLine error: lines is None")
	if(pointStart is None or pointEnd is None):
		raise RuntimeError("appendStandaloneDrawEfficientLdrLine error: line endpoints are None")
	lineString = "2 " + str(int(lineColour)) + " " + " ".join(format(float(value), ".5f") for value in pointStart) + " " + " ".join(format(float(value), ".5f") for value in pointEnd)
	lines.append(lineString)
	return

def appendStandaloneDrawEfficientLdrNode(lines, nodeColour, nodePosition):
	if(lines is None):
		raise RuntimeError("appendStandaloneDrawEfficientLdrNode error: lines is None")
	if(nodePosition is None):
		raise RuntimeError("appendStandaloneDrawEfficientLdrNode error: nodePosition is None")
	scale = standaloneDrawEfficient3DLdrNodeScale
	nodeString = "1 " + str(int(nodeColour)) + " " + " ".join(format(float(value), ".5f") for value in nodePosition) + " " + format(scale, ".5f") + " 0.0 0.0 0.0 " + format(scale, ".5f") + " 0.0 0.0 0.0 " + format(scale, ".5f") + " " + standaloneDrawEfficient3DLdrNodePartFile
	lines.append(nodeString)
	return

def getStandaloneDrawEfficientLdrRectanglePoints(totalNumberColumns, totalNumberFeatures, rectangle):
	if(rectangle is None):
		raise RuntimeError("getStandaloneDrawEfficientLdrRectanglePoints error: rectangle is None")
	if(len(rectangle) != 4):
		raise RuntimeError("getStandaloneDrawEfficientLdrRectanglePoints error: rectangle must have four entries")
	rectangleX, rectangleY, rectangleWidth, rectangleHeight = rectangle
	conceptIndex = getStandaloneDrawEfficientLdrConceptIndex(float(rectangleX) + (float(rectangleWidth)/2.0))
	frontPoints = [generateGridCoordinates(conceptIndex, rectangleY, totalNumberColumns, totalNumberFeatures, pointXOffset=-(float(rectangleWidth)/2.0), pointZOffset=-(standaloneDrawEfficient3DLdrColumnDepth/2.0)), generateGridCoordinates(conceptIndex, rectangleY, totalNumberColumns, totalNumberFeatures, pointXOffset=(float(rectangleWidth)/2.0), pointZOffset=-(standaloneDrawEfficient3DLdrColumnDepth/2.0)), generateGridCoordinates(conceptIndex, float(rectangleY) + float(rectangleHeight), totalNumberColumns, totalNumberFeatures, pointXOffset=(float(rectangleWidth)/2.0), pointZOffset=-(standaloneDrawEfficient3DLdrColumnDepth/2.0)), generateGridCoordinates(conceptIndex, float(rectangleY) + float(rectangleHeight), totalNumberColumns, totalNumberFeatures, pointXOffset=-(float(rectangleWidth)/2.0), pointZOffset=-(standaloneDrawEfficient3DLdrColumnDepth/2.0))]
	backPoints = [generateGridCoordinates(conceptIndex, rectangleY, totalNumberColumns, totalNumberFeatures, pointXOffset=-(float(rectangleWidth)/2.0), pointZOffset=(standaloneDrawEfficient3DLdrColumnDepth/2.0)), generateGridCoordinates(conceptIndex, rectangleY, totalNumberColumns, totalNumberFeatures, pointXOffset=(float(rectangleWidth)/2.0), pointZOffset=(standaloneDrawEfficient3DLdrColumnDepth/2.0)), generateGridCoordinates(conceptIndex, float(rectangleY) + float(rectangleHeight), totalNumberColumns, totalNumberFeatures, pointXOffset=(float(rectangleWidth)/2.0), pointZOffset=(standaloneDrawEfficient3DLdrColumnDepth/2.0)), generateGridCoordinates(conceptIndex, float(rectangleY) + float(rectangleHeight), totalNumberColumns, totalNumberFeatures, pointXOffset=-(float(rectangleWidth)/2.0), pointZOffset=(standaloneDrawEfficient3DLdrColumnDepth/2.0))]
	result = (frontPoints, backPoints)
	return result

def getStandaloneDrawEfficientLdrLineEndpoints(totalNumberColumns, totalNumberFeatures, lineSegment):
	if(lineSegment is None):
		raise RuntimeError("getStandaloneDrawEfficientLdrLineEndpoints error: lineSegment is None")
	if(len(lineSegment) != 2):
		raise RuntimeError("getStandaloneDrawEfficientLdrLineEndpoints error: lineSegment must contain source and target positions")
	lineStart = convertStandaloneDrawEfficientPointToLdrPosition(totalNumberColumns, totalNumberFeatures, lineSegment[0][0], lineSegment[0][1], 0.0)
	lineEnd = convertStandaloneDrawEfficientPointToLdrPosition(totalNumberColumns, totalNumberFeatures, lineSegment[1][0], lineSegment[1][1], 0.0)
	result = (lineStart, lineEnd)
	return result

def getStandaloneDrawEfficientLdrNodePosition(totalNumberColumns, totalNumberFeatures, nodeX, nodeY):
	result = convertStandaloneDrawEfficientPointToLdrPosition(totalNumberColumns, totalNumberFeatures, nodeX, nodeY, 0.0)
	return result

def getStandaloneDrawEfficientLdrConceptIndex(pointX):
	pointXFloat = float(pointX)
	conceptIndex = int(round(pointXFloat))
	if(not math.isclose(pointXFloat, float(conceptIndex), abs_tol=1.0e-6)):
		raise RuntimeError(f"getStandaloneDrawEfficientLdrConceptIndex error: pointX {pointXFloat} does not identify a column centre")
	result = conceptIndex
	return result

def getStandaloneDrawEfficientLdrHeightPosition(pointY):
	result = -float(pointY) * standaloneDrawEfficient3DLdrFeatureSpacing
	return result

def calculateStandaloneDrawEfficientLdrPrismGridWidth(totalNumberColumns):
	if(totalNumberColumns <= 0):
		raise RuntimeError("calculateStandaloneDrawEfficientLdrPrismGridWidth error: totalNumberColumns must be > 0")
	result = int(math.ceil(math.sqrt(float(totalNumberColumns))))
	return result

def generateGridCoordinates(conceptIndex, featureIndex, totalNumberColumns, totalNumberFeatures, pointXOffset=0.0, pointZOffset=0.0):
	if(conceptIndex < 0):
		raise RuntimeError("generateGridCoordinates error: conceptIndex must be >= 0")
	if(totalNumberColumns <= 0):
		raise RuntimeError("generateGridCoordinates error: totalNumberColumns must be > 0")
	if(conceptIndex >= totalNumberColumns):
		raise RuntimeError(f"generateGridCoordinates error: conceptIndex {conceptIndex} >= totalNumberColumns {totalNumberColumns}")
	if(drawEfficientFormat3Dprism):
		gridWidth = calculateStandaloneDrawEfficientLdrPrismGridWidth(totalNumberColumns)
		gridColumnIndex = int(conceptIndex % gridWidth)
		gridRowIndex = int(conceptIndex // gridWidth)
		xPosition = (float(gridColumnIndex) + float(pointXOffset)) * standaloneDrawEfficient3DLdrColumnSpacing
		zPosition = (float(gridRowIndex) * standaloneDrawEfficient3DLdrColumnSpacing) + float(pointZOffset)
	else:
		xPosition = (float(conceptIndex) + float(pointXOffset)) * standaloneDrawEfficient3DLdrColumnSpacing
		zPosition = float(pointZOffset)
	yPosition = getStandaloneDrawEfficientLdrHeightPosition(featureIndex)
	result = (xPosition, yPosition, zPosition)
	return result

def convertStandaloneDrawEfficientPointToLdrPosition(totalNumberColumns, totalNumberFeatures, pointX, pointY, pointZ):
	conceptIndex = getStandaloneDrawEfficientLdrConceptIndex(pointX)
	result = generateGridCoordinates(conceptIndex, pointY, totalNumberColumns, totalNumberFeatures, pointXOffset=0.0, pointZOffset=pointZ)
	return result

def getStandaloneDrawEfficientLdrColourFromNodeColour(nodeColour):
	if(nodeColour == GIAANNproto_databaseNetworkDraw.defaultColourConceptFeature):
		result = standaloneDrawEfficient3DLdrColourConceptNode
	elif(nodeColour == GIAANNproto_databaseNetworkDraw.defaultColourFeature):
		result = standaloneDrawEfficient3DLdrColourFeatureNode
	else:
		raise RuntimeError(f"getStandaloneDrawEfficientLdrColourFromNodeColour error: unsupported node colour {nodeColour}")
	return result

def getStandaloneDrawEfficientLdrColourFromConnectionColour(connectionColour):
	if(connectionColour == GIAANNproto_databaseNetworkDraw.defaultConnectionColourInternal):
		result = standaloneDrawEfficient3DLdrColourInternalConnection
	elif(connectionColour == GIAANNproto_databaseNetworkDraw.defaultConnectionColourExternal):
		result = standaloneDrawEfficient3DLdrColourExternalConnection
	else:
		raise RuntimeError(f"getStandaloneDrawEfficientLdrColourFromConnectionColour error: unsupported connection colour {connectionColour}")
	return result

def resolveStandaloneDrawEfficient3DOutputFileName(fileName):
	result = fileName
	if(result is None):
		raise RuntimeError("resolveStandaloneDrawEfficient3DOutputFileName error: fileName is None")
	if(result.endswith(".svg") or result.endswith(".png")):
		result = result[:-4] + ".ldr"
	elif(not result.endswith(".ldr")):
		result = result + ".ldr"
	if(not os.path.isabs(result)):
		result = os.path.join(databaseFolder, result)
	return result

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
