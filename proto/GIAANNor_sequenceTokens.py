"""GIAANNor_sequenceTokens.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN OR sequence Tokens

"""

import torch as pt

from GIAANNcmn_globalDefs import *


def buildColumnConceptName(xIndex, yIndex, layerIndex):
	result = None
	result = "OR_L" + str(layerIndex) + "_X" + str(xIndex).zfill(3) + "_Y" + str(yIndex).zfill(3)
	return result


def buildColumnMetadataList(gridHeight, gridWidth):
	result = []
	layerIndex = modalityORtrainMaxLayerIndex
	if(layerIndex != 0):
		raise RuntimeError("buildColumnMetadataList error: the initial OR implementation only supports layerIndex 0")
	for yIndex in range(gridHeight):
		for xIndex in range(gridWidth):
			result.append({"conceptName": buildColumnConceptName(xIndex, yIndex, layerIndex), "xIndex": xIndex, "yIndex": yIndex, "layerIndex": layerIndex})
	return result


def tokeniseSnapshotsToColumns(snapshotTensor):
	# tokenise each snapshot into a series of columns using modalityORpixelsPerColumn (executed in parallel using pytorch):
	# each column in modalityName=="OR" represents a particular a) x token and b) y token in the snapshot (forming a 2D map like the visual cortex V1), and c) a particular layer l in the substrate (forming a hierarchical visual cortex).
	# for this initial implementation, only train layer l=0 (do not train the higher layers).
	result = None
	snapshotTensorTrimmed = None
	gridHeight = None
	gridWidth = None
	patchTensor = None
	columnTensor = None
	columnMetadataList = None
	if(not pt.is_tensor(snapshotTensor)):
		raise RuntimeError("tokeniseSnapshotsToColumns error: snapshotTensor must be a tensor")
	if(snapshotTensor.dim() != 4):
		raise RuntimeError("tokeniseSnapshotsToColumns error: snapshotTensor rank must be 4")
	if(snapshotTensor.shape[1] != 3):
		raise RuntimeError("tokeniseSnapshotsToColumns error: snapshotTensor channel count must be 3")
	if(tokensiationMethodOneColumnPerSnapshotPixel):
		result = tokeniseSnapshotsToPixelColumns(snapshotTensor)
	else:
		if(modalityORpixelsPerColumn <= 0):
			raise RuntimeError("tokeniseSnapshotsToColumns error: modalityORpixelsPerColumn must be > 0")
		gridHeight = int(snapshotTensor.shape[2]//modalityORpixelsPerColumn)
		gridWidth = int(snapshotTensor.shape[3]//modalityORpixelsPerColumn)
		if(gridHeight <= 0 or gridWidth <= 0):
			raise RuntimeError("tokeniseSnapshotsToColumns error: snapshotTensor is smaller than modalityORpixelsPerColumn")
		snapshotTensorTrimmed = snapshotTensor[:, :, :gridHeight*modalityORpixelsPerColumn, :gridWidth*modalityORpixelsPerColumn]
		if(modalityORsnapshotRetinotopicFieldBias):
			columnTensor, columnMetadataList = tokeniseSnapshotsToRetinotopicColumns(snapshotTensorTrimmed, gridHeight, gridWidth)
		else:
			patchTensor = snapshotTensorTrimmed.unfold(2, modalityORpixelsPerColumn, modalityORpixelsPerColumn).unfold(3, modalityORpixelsPerColumn, modalityORpixelsPerColumn)
			patchTensor = patchTensor.permute(0, 2, 3, 1, 4, 5).contiguous()
			columnTensor = patchTensor.view(patchTensor.shape[0], gridHeight*gridWidth, patchTensor.shape[3], patchTensor.shape[4], patchTensor.shape[5])
			columnMetadataList = buildColumnMetadataList(gridHeight, gridWidth)
		result = {"columnTensor": columnTensor, "columnMetadataList": columnMetadataList, "gridHeight": gridHeight, "gridWidth": gridWidth}
	return result


if(tokensiationMethodOneColumnPerSnapshotPixel):
	def tokeniseSnapshotsToPixelColumns(snapshotTensor):
		result = None
		gridHeight = None
		gridWidth = None
		columnTensor = None
		columnMetadataList = None
		validatePixelColumnTokenisationParameters(snapshotTensor)
		gridHeight, gridWidth = calculatePixelColumnGridDimensions(int(snapshotTensor.shape[2]), int(snapshotTensor.shape[3]))
		columnTensor = transformSnapshotsToPixelColumnTensor(snapshotTensor, gridHeight, gridWidth)
		if(int(columnTensor.shape[2])*int(columnTensor.shape[3]) != int(modalityORnumberOfColumnsV1)):
			raise RuntimeError("tokeniseSnapshotsToPixelColumns error: transformed snapshot pixel count must equal modalityORnumberOfColumnsV1")
		columnMetadataList = buildColumnMetadataList(gridHeight, gridWidth)
		result = {"columnTensor": columnTensor, "columnMetadataList": columnMetadataList, "gridHeight": gridHeight, "gridWidth": gridWidth}
		return result


	def validatePixelColumnTokenisationParameters(snapshotTensor):
		result = None
		if(tokensiationMethodOneColumnPerSnapshotPixel):
			if(not pt.is_tensor(snapshotTensor)):
				raise RuntimeError("validatePixelColumnTokenisationParameters error: snapshotTensor must be a tensor")
			if(snapshotTensor.dim() != 4):
				raise RuntimeError("validatePixelColumnTokenisationParameters error: snapshotTensor rank must be 4")
			if(int(snapshotTensor.shape[1]) != 3):
				raise RuntimeError("validatePixelColumnTokenisationParameters error: snapshotTensor channel count must be 3")
			if(not isinstance(modalityORnumberOfColumnsV1, int)):
				raise RuntimeError("validatePixelColumnTokenisationParameters error: modalityORnumberOfColumnsV1 must be an int")
			if(modalityORnumberOfColumnsV1 <= 0):
				raise RuntimeError("validatePixelColumnTokenisationParameters error: modalityORnumberOfColumnsV1 must be > 0")
			if(modalityORsnapshotRetinotopicFieldBias):
				validateRetinotopicFieldBiasParameters()
		else:
			raise RuntimeError("validatePixelColumnTokenisationParameters error: requires tokensiationMethodOneColumnPerSnapshotPixel")
		return result


	def calculatePixelColumnGridDimensions(snapshotHeight, snapshotWidth):
		result = None
		targetAspectRatio = None
		candidateHeight = None
		candidateWidth = None
		candidateAspectRatio = None
		candidateDifference = None
		bestDifference = None
		bestGridHeight = None
		bestGridWidth = None
		if(tokensiationMethodOneColumnPerSnapshotPixel):
			if(snapshotHeight <= 0 or snapshotWidth <= 0):
				raise RuntimeError("calculatePixelColumnGridDimensions error: snapshotHeight/snapshotWidth must be > 0")
			if(not isinstance(modalityORnumberOfColumnsV1, int)):
				raise RuntimeError("calculatePixelColumnGridDimensions error: modalityORnumberOfColumnsV1 must be an int")
			if(modalityORnumberOfColumnsV1 <= 0):
				raise RuntimeError("calculatePixelColumnGridDimensions error: modalityORnumberOfColumnsV1 must be > 0")
			targetAspectRatio = float(snapshotWidth)/float(snapshotHeight)
			candidateHeight = 1
			while(candidateHeight*candidateHeight <= int(modalityORnumberOfColumnsV1)):
				if(int(modalityORnumberOfColumnsV1)%candidateHeight == 0):
					candidateWidth = int(modalityORnumberOfColumnsV1)//candidateHeight
					candidateAspectRatio = float(candidateWidth)/float(candidateHeight)
					candidateDifference = abs(candidateAspectRatio - targetAspectRatio)/targetAspectRatio
					if(bestDifference is None or candidateDifference < bestDifference):
						bestDifference = candidateDifference
						bestGridHeight = candidateHeight
						bestGridWidth = candidateWidth
					candidateAspectRatio = float(candidateHeight)/float(candidateWidth)
					candidateDifference = abs(candidateAspectRatio - targetAspectRatio)/targetAspectRatio
					if(bestDifference is None or candidateDifference < bestDifference):
						bestDifference = candidateDifference
						bestGridHeight = candidateWidth
						bestGridWidth = candidateHeight
				candidateHeight = candidateHeight + 1
			if(bestGridHeight is None or bestGridWidth is None):
				raise RuntimeError("calculatePixelColumnGridDimensions error: failed to calculate grid dimensions")
			result = bestGridHeight, bestGridWidth
		else:
			raise RuntimeError("calculatePixelColumnGridDimensions error: requires tokensiationMethodOneColumnPerSnapshotPixel")
		return result


	def transformSnapshotsToPixelColumnTensor(snapshotTensor, gridHeight, gridWidth):
		result = None
		if(tokensiationMethodOneColumnPerSnapshotPixel):
			if(gridHeight <= 0 or gridWidth <= 0):
				raise RuntimeError("transformSnapshotsToPixelColumnTensor error: gridHeight/gridWidth must be > 0")
			if(int(gridHeight)*int(gridWidth) != int(modalityORnumberOfColumnsV1)):
				raise RuntimeError("transformSnapshotsToPixelColumnTensor error: gridHeight*gridWidth must equal modalityORnumberOfColumnsV1")
			if(modalityORsnapshotRetinotopicFieldBias):
				result = transformSnapshotsToRetinotopicPixelColumnTensor(snapshotTensor, gridHeight, gridWidth)
			else:
				result = pt.nn.functional.interpolate(snapshotTensor.to(dtype=arrayType), size=(gridHeight, gridWidth), mode="bilinear", align_corners=False)
			if(result.dim() != 4):
				raise RuntimeError("transformSnapshotsToPixelColumnTensor error: transformed snapshot rank must be 4")
			if(int(result.shape[1]) != 3 or int(result.shape[2]) != int(gridHeight) or int(result.shape[3]) != int(gridWidth)):
				raise RuntimeError("transformSnapshotsToPixelColumnTensor error: transformed snapshot shape mismatch")
		else:
			raise RuntimeError("transformSnapshotsToPixelColumnTensor error: requires tokensiationMethodOneColumnPerSnapshotPixel")
		return result


	def transformSnapshotsToRetinotopicPixelColumnTensor(snapshotTensor, gridHeight, gridWidth):
		result = None
		sourceCoordinateTensor = None
		samplingGrid = None
		if(tokensiationMethodOneColumnPerSnapshotPixel):
			if(not modalityORsnapshotRetinotopicFieldBias):
				raise RuntimeError("transformSnapshotsToRetinotopicPixelColumnTensor error: requires modalityORsnapshotRetinotopicFieldBias")
			sourceCoordinateTensor = calculateRetinotopicPixelColumnSourceCoordinateTensor(gridHeight, gridWidth, snapshotTensor.device)
			samplingGrid = sourceCoordinateTensor.view(gridHeight, gridWidth, 2).unsqueeze(0).expand(int(snapshotTensor.shape[0]), gridHeight, gridWidth, 2)
			result = pt.nn.functional.grid_sample(snapshotTensor.to(dtype=arrayType), samplingGrid, mode="bilinear", padding_mode="border", align_corners=True)
		else:
			raise RuntimeError("transformSnapshotsToRetinotopicPixelColumnTensor error: requires tokensiationMethodOneColumnPerSnapshotPixel")
		return result


	def calculateRetinotopicPixelColumnSourceCoordinateTensor(gridHeight, gridWidth, targetDevice):
		result = None
		numberOfColumns = None
		xGrid = None
		yGrid = None
		xFlat = None
		yFlat = None
		centreX = None
		centreY = None
		deltaX = None
		deltaY = None
		corticalDistanceTensor = None
		distanceSortIndices = None
		corticalColumnFractionTensor = None
		columnFractionValues = None
		visualRadiusFractionTensor = None
		directionX = None
		directionY = None
		nonzeroDistanceMask = None
		radiusLimitX = None
		radiusLimitY = None
		maxRadiusToRectangle = None
		sourceXCoordinate = None
		sourceYCoordinate = None
		if(tokensiationMethodOneColumnPerSnapshotPixel):
			if(not modalityORsnapshotRetinotopicFieldBias):
				raise RuntimeError("calculateRetinotopicPixelColumnSourceCoordinateTensor error: requires modalityORsnapshotRetinotopicFieldBias")
			if(gridHeight <= 0 or gridWidth <= 0):
				raise RuntimeError("calculateRetinotopicPixelColumnSourceCoordinateTensor error: gridHeight/gridWidth must be > 0")
			if(int(gridHeight)*int(gridWidth) != int(modalityORnumberOfColumnsV1)):
				raise RuntimeError("calculateRetinotopicPixelColumnSourceCoordinateTensor error: gridHeight*gridWidth must equal modalityORnumberOfColumnsV1")
			validateRetinotopicFieldBiasParameters()
			numberOfColumns = int(gridHeight)*int(gridWidth)
			yGrid, xGrid = pt.meshgrid(pt.arange(gridHeight, dtype=arrayType, device=targetDevice), pt.arange(gridWidth, dtype=arrayType, device=targetDevice), indexing="ij")
			xFlat = xGrid.reshape(-1)
			yFlat = yGrid.reshape(-1)
			centreX = (float(gridWidth) - 1.0)/2.0
			centreY = (float(gridHeight) - 1.0)/2.0
			deltaX = xFlat - centreX
			deltaY = yFlat - centreY
			corticalDistanceTensor = pt.sqrt((deltaX*deltaX) + (deltaY*deltaY))
			distanceSortIndices = pt.argsort(corticalDistanceTensor, descending=False)
			corticalColumnFractionTensor = pt.zeros(numberOfColumns, dtype=arrayType, device=targetDevice)
			if(numberOfColumns > 1):
				columnFractionValues = pt.arange(numberOfColumns, dtype=arrayType, device=targetDevice)/float(numberOfColumns - 1)
				corticalColumnFractionTensor[distanceSortIndices] = columnFractionValues
			visualRadiusFractionTensor = calculateRetinotopicVisualRadiusFractionTensor(corticalColumnFractionTensor, targetDevice)
			directionX = pt.zeros(numberOfColumns, dtype=arrayType, device=targetDevice)
			directionY = pt.zeros(numberOfColumns, dtype=arrayType, device=targetDevice)
			nonzeroDistanceMask = corticalDistanceTensor > 0.0
			directionX[nonzeroDistanceMask] = deltaX[nonzeroDistanceMask]/corticalDistanceTensor[nonzeroDistanceMask]
			directionY[nonzeroDistanceMask] = deltaY[nonzeroDistanceMask]/corticalDistanceTensor[nonzeroDistanceMask]
			radiusLimitX = pt.full((numberOfColumns,), float("inf"), dtype=arrayType, device=targetDevice)
			radiusLimitY = pt.full((numberOfColumns,), float("inf"), dtype=arrayType, device=targetDevice)
			radiusLimitX[directionX != 0.0] = 1.0/pt.abs(directionX[directionX != 0.0])
			radiusLimitY[directionY != 0.0] = 1.0/pt.abs(directionY[directionY != 0.0])
			maxRadiusToRectangle = pt.minimum(radiusLimitX, radiusLimitY)
			maxRadiusToRectangle[~nonzeroDistanceMask] = 0.0
			sourceXCoordinate = directionX*visualRadiusFractionTensor*maxRadiusToRectangle
			sourceYCoordinate = directionY*visualRadiusFractionTensor*maxRadiusToRectangle
			if(bool(pt.any(sourceXCoordinate < -1.0).item()) or bool(pt.any(sourceYCoordinate < -1.0).item())):
				raise RuntimeError("calculateRetinotopicPixelColumnSourceCoordinateTensor error: calculated source coordinate is < -1")
			if(bool(pt.any(sourceXCoordinate > 1.0).item()) or bool(pt.any(sourceYCoordinate > 1.0).item())):
				raise RuntimeError("calculateRetinotopicPixelColumnSourceCoordinateTensor error: calculated source coordinate is > 1")
			result = pt.stack((sourceXCoordinate, sourceYCoordinate), dim=1)
		else:
			raise RuntimeError("calculateRetinotopicPixelColumnSourceCoordinateTensor error: requires tokensiationMethodOneColumnPerSnapshotPixel")
		return result


def tokeniseSnapshotsToRetinotopicColumns(snapshotTensorTrimmed, gridHeight, gridWidth):
	result = None
	sourceStartCoordinateTensor = None
	columnTensor = None
	columnMetadataList = None
	if(modalityORsnapshotRetinotopicFieldBias):
		if(not pt.is_tensor(snapshotTensorTrimmed)):
			raise RuntimeError("tokeniseSnapshotsToRetinotopicColumns error: snapshotTensorTrimmed must be a tensor")
		if(snapshotTensorTrimmed.dim() != 4):
			raise RuntimeError("tokeniseSnapshotsToRetinotopicColumns error: snapshotTensorTrimmed rank must be 4")
		if(gridHeight <= 0 or gridWidth <= 0):
			raise RuntimeError("tokeniseSnapshotsToRetinotopicColumns error: gridHeight/gridWidth must be > 0")
		sourceStartCoordinateTensor = calculateRetinotopicSourceStartCoordinateTensor(gridHeight, gridWidth, int(snapshotTensorTrimmed.shape[2]), int(snapshotTensorTrimmed.shape[3]), snapshotTensorTrimmed.device)
		columnTensor = extractRetinotopicColumnTensor(snapshotTensorTrimmed, sourceStartCoordinateTensor)
		columnMetadataList = buildRetinotopicColumnMetadataList(gridHeight, gridWidth, sourceStartCoordinateTensor)
		result = columnTensor, columnMetadataList
	else:
		raise RuntimeError("tokeniseSnapshotsToRetinotopicColumns error: requires modalityORsnapshotRetinotopicFieldBias")
	return result


def calculateRetinotopicSourceStartCoordinateTensor(gridHeight, gridWidth, snapshotHeight, snapshotWidth, targetDevice):
	result = None
	numberOfColumns = None
	maxStartX = None
	maxStartY = None
	xGrid = None
	yGrid = None
	xFlat = None
	yFlat = None
	centreX = None
	centreY = None
	deltaX = None
	deltaY = None
	corticalDistanceTensor = None
	distanceSortIndices = None
	corticalColumnFractionTensor = None
	columnFractionValues = None
	visualRadiusFractionTensor = None
	directionX = None
	directionY = None
	nonzeroDistanceMask = None
	radiusLimitX = None
	radiusLimitY = None
	maxRadiusToRectangle = None
	sourceXFraction = None
	sourceYFraction = None
	sourceStartX = None
	sourceStartY = None
	if(modalityORsnapshotRetinotopicFieldBias):
		if(gridHeight <= 0 or gridWidth <= 0):
			raise RuntimeError("calculateRetinotopicSourceStartCoordinateTensor error: gridHeight/gridWidth must be > 0")
		if(snapshotHeight <= 0 or snapshotWidth <= 0):
			raise RuntimeError("calculateRetinotopicSourceStartCoordinateTensor error: snapshotHeight/snapshotWidth must be > 0")
		if(snapshotHeight < modalityORpixelsPerColumn or snapshotWidth < modalityORpixelsPerColumn):
			raise RuntimeError("calculateRetinotopicSourceStartCoordinateTensor error: snapshot dimensions must be >= modalityORpixelsPerColumn")
		validateRetinotopicFieldBiasParameters()
		numberOfColumns = int(gridHeight)*int(gridWidth)
		maxStartX = int(snapshotWidth) - int(modalityORpixelsPerColumn)
		maxStartY = int(snapshotHeight) - int(modalityORpixelsPerColumn)
		yGrid, xGrid = pt.meshgrid(pt.arange(gridHeight, dtype=arrayType, device=targetDevice), pt.arange(gridWidth, dtype=arrayType, device=targetDevice), indexing="ij")
		xFlat = xGrid.reshape(-1)
		yFlat = yGrid.reshape(-1)
		centreX = (float(gridWidth) - 1.0)/2.0
		centreY = (float(gridHeight) - 1.0)/2.0
		deltaX = xFlat - centreX
		deltaY = yFlat - centreY
		corticalDistanceTensor = pt.sqrt((deltaX*deltaX) + (deltaY*deltaY))
		distanceSortIndices = pt.argsort(corticalDistanceTensor, descending=False)
		corticalColumnFractionTensor = pt.zeros(numberOfColumns, dtype=arrayType, device=targetDevice)
		if(numberOfColumns > 1):
			columnFractionValues = pt.arange(numberOfColumns, dtype=arrayType, device=targetDevice)/float(numberOfColumns - 1)
			corticalColumnFractionTensor[distanceSortIndices] = columnFractionValues
		visualRadiusFractionTensor = calculateRetinotopicVisualRadiusFractionTensor(corticalColumnFractionTensor, targetDevice)
		directionX = pt.zeros(numberOfColumns, dtype=arrayType, device=targetDevice)
		directionY = pt.zeros(numberOfColumns, dtype=arrayType, device=targetDevice)
		nonzeroDistanceMask = corticalDistanceTensor > 0.0
		directionX[nonzeroDistanceMask] = deltaX[nonzeroDistanceMask]/corticalDistanceTensor[nonzeroDistanceMask]
		directionY[nonzeroDistanceMask] = deltaY[nonzeroDistanceMask]/corticalDistanceTensor[nonzeroDistanceMask]
		radiusLimitX = pt.full((numberOfColumns,), float("inf"), dtype=arrayType, device=targetDevice)
		radiusLimitY = pt.full((numberOfColumns,), float("inf"), dtype=arrayType, device=targetDevice)
		radiusLimitX[directionX != 0.0] = 0.5/pt.abs(directionX[directionX != 0.0])
		radiusLimitY[directionY != 0.0] = 0.5/pt.abs(directionY[directionY != 0.0])
		maxRadiusToRectangle = pt.minimum(radiusLimitX, radiusLimitY)
		maxRadiusToRectangle[~nonzeroDistanceMask] = 0.0
		sourceXFraction = 0.5 + (directionX*visualRadiusFractionTensor*maxRadiusToRectangle)
		sourceYFraction = 0.5 + (directionY*visualRadiusFractionTensor*maxRadiusToRectangle)
		sourceStartX = pt.round(sourceXFraction*float(maxStartX)).to(dtype=pt.long)
		sourceStartY = pt.round(sourceYFraction*float(maxStartY)).to(dtype=pt.long)
		if(bool(pt.any(sourceStartX < 0).item()) or bool(pt.any(sourceStartY < 0).item())):
			raise RuntimeError("calculateRetinotopicSourceStartCoordinateTensor error: calculated source start is < 0")
		if(bool(pt.any(sourceStartX > maxStartX).item()) or bool(pt.any(sourceStartY > maxStartY).item())):
			raise RuntimeError("calculateRetinotopicSourceStartCoordinateTensor error: calculated source start exceeds snapshot bounds")
		result = pt.stack((sourceStartX, sourceStartY), dim=1)
	else:
		raise RuntimeError("calculateRetinotopicSourceStartCoordinateTensor error: requires modalityORsnapshotRetinotopicFieldBias")
	return result


def extractRetinotopicColumnTensor(snapshotTensorTrimmed, sourceStartCoordinateTensor):
	result = None
	numberOfColumns = None
	yIndexTensor = None
	xIndexTensor = None
	rowIndexTensor = None
	columnIndexTensor = None
	patchTensor = None
	if(modalityORsnapshotRetinotopicFieldBias):
		if(not pt.is_tensor(snapshotTensorTrimmed)):
			raise RuntimeError("extractRetinotopicColumnTensor error: snapshotTensorTrimmed must be a tensor")
		if(not pt.is_tensor(sourceStartCoordinateTensor)):
			raise RuntimeError("extractRetinotopicColumnTensor error: sourceStartCoordinateTensor must be a tensor")
		if(snapshotTensorTrimmed.dim() != 4):
			raise RuntimeError("extractRetinotopicColumnTensor error: snapshotTensorTrimmed rank must be 4")
		if(sourceStartCoordinateTensor.dim() != 2):
			raise RuntimeError("extractRetinotopicColumnTensor error: sourceStartCoordinateTensor rank must be 2")
		if(int(sourceStartCoordinateTensor.shape[1]) != 2):
			raise RuntimeError("extractRetinotopicColumnTensor error: sourceStartCoordinateTensor last dimension must equal 2")
		numberOfColumns = int(sourceStartCoordinateTensor.shape[0])
		if(numberOfColumns <= 0):
			raise RuntimeError("extractRetinotopicColumnTensor error: numberOfColumns must be > 0")
		rowIndexTensor = pt.arange(modalityORpixelsPerColumn, device=snapshotTensorTrimmed.device).view(1, modalityORpixelsPerColumn, 1)
		columnIndexTensor = pt.arange(modalityORpixelsPerColumn, device=snapshotTensorTrimmed.device).view(1, 1, modalityORpixelsPerColumn)
		yIndexTensor = sourceStartCoordinateTensor[:, 1].view(numberOfColumns, 1, 1) + rowIndexTensor
		xIndexTensor = sourceStartCoordinateTensor[:, 0].view(numberOfColumns, 1, 1) + columnIndexTensor
		patchTensor = snapshotTensorTrimmed[:, :, yIndexTensor, xIndexTensor]
		result = patchTensor.permute(0, 2, 1, 3, 4).contiguous()
	else:
		raise RuntimeError("extractRetinotopicColumnTensor error: requires modalityORsnapshotRetinotopicFieldBias")
	return result


def buildRetinotopicColumnMetadataList(gridHeight, gridWidth, sourceStartCoordinateTensor):
	result = None
	columnMetadataList = None
	if(modalityORsnapshotRetinotopicFieldBias):
		if(not pt.is_tensor(sourceStartCoordinateTensor)):
			raise RuntimeError("buildRetinotopicColumnMetadataList error: sourceStartCoordinateTensor must be a tensor")
		if(sourceStartCoordinateTensor.dim() != 2):
			raise RuntimeError("buildRetinotopicColumnMetadataList error: sourceStartCoordinateTensor rank must be 2")
		if(int(sourceStartCoordinateTensor.shape[0]) != int(gridHeight)*int(gridWidth)):
			raise RuntimeError("buildRetinotopicColumnMetadataList error: sourceStartCoordinateTensor column count mismatch")
		columnMetadataList = buildColumnMetadataList(gridHeight, gridWidth)
		for columnIndex, columnMetadata in enumerate(columnMetadataList):
			columnMetadata["sourceStartX"] = int(sourceStartCoordinateTensor[columnIndex, 0].item())
			columnMetadata["sourceStartY"] = int(sourceStartCoordinateTensor[columnIndex, 1].item())
		result = columnMetadataList
	else:
		raise RuntimeError("buildRetinotopicColumnMetadataList error: requires modalityORsnapshotRetinotopicFieldBias")
	return result


def calculateRetinotopicVisualRadiusFractionTensor(corticalColumnFractionTensor, targetDevice):
	result = None
	visualRadiusFractionList = []
	visualRadiusDegrees = None
	snapshotRetinotopicFieldMaxDegrees = None
	if(modalityORsnapshotRetinotopicFieldBias):
		if(not pt.is_tensor(corticalColumnFractionTensor)):
			raise RuntimeError("calculateRetinotopicVisualRadiusFractionTensor error: corticalColumnFractionTensor must be a tensor")
		if(corticalColumnFractionTensor.dim() != 1):
			raise RuntimeError("calculateRetinotopicVisualRadiusFractionTensor error: corticalColumnFractionTensor rank must be 1")
		validateRetinotopicFieldBiasParameters()
		snapshotRetinotopicFieldMaxDegrees = calculateExpectedSnapshotRetinotopicFieldMaxDegrees()
		for corticalColumnFraction in corticalColumnFractionTensor:
			visualRadiusDegrees = calculateRetinotopicVisualFieldRadiusDegreesForColumnFraction(float(corticalColumnFraction.item()))
			visualRadiusFractionList.append(visualRadiusDegrees/snapshotRetinotopicFieldMaxDegrees)
		result = pt.tensor(visualRadiusFractionList, dtype=arrayType, device=targetDevice)
	else:
		raise RuntimeError("calculateRetinotopicVisualRadiusFractionTensor error: requires modalityORsnapshotRetinotopicFieldBias")
	return result


def calculateRetinotopicVisualFieldRadiusDegreesForColumnFraction(columnFraction):
	result = None
	table = None
	maxFieldPercentage = None
	snapshotRetinotopicFieldMaxDegrees = None
	targetPercentage = None
	lowerRadiusDegrees = None
	lowerPercentage = None
	upperRadiusDegrees = None
	upperPercentage = None
	interpolationFraction = None
	if(modalityORsnapshotRetinotopicFieldBias):
		if(columnFraction < 0.0 or columnFraction > 1.0):
			raise RuntimeError("calculateRetinotopicVisualFieldRadiusDegreesForColumnFraction error: columnFraction must be >= 0.0 and <= 1.0")
		validateRetinotopicFieldBiasParameters()
		table = getRetinotopicFieldBiasReferenceTable()
		snapshotRetinotopicFieldMaxDegrees = calculateExpectedSnapshotRetinotopicFieldMaxDegrees()
		maxFieldPercentage = calculateRetinotopicColumnPercentageForVisualFieldRadiusDegrees(snapshotRetinotopicFieldMaxDegrees)
		targetPercentage = float(columnFraction)*float(maxFieldPercentage)
		result = 0.0
		for tableIndex in range(1, len(table)):
			lowerRadiusDegrees = float(table[tableIndex - 1][0])
			lowerPercentage = float(table[tableIndex - 1][1])
			upperRadiusDegrees = float(table[tableIndex][0])
			upperPercentage = float(table[tableIndex][1])
			if(targetPercentage >= lowerPercentage and targetPercentage <= upperPercentage):
				if(upperPercentage <= lowerPercentage):
					raise RuntimeError("calculateRetinotopicVisualFieldRadiusDegreesForColumnFraction error: retinotopic table percentages must increase")
				interpolationFraction = (targetPercentage - lowerPercentage)/(upperPercentage - lowerPercentage)
				result = lowerRadiusDegrees + (interpolationFraction*(upperRadiusDegrees - lowerRadiusDegrees))
				break
		if(result < 0.0 or result > snapshotRetinotopicFieldMaxDegrees):
			raise RuntimeError("calculateRetinotopicVisualFieldRadiusDegreesForColumnFraction error: calculated radius is outside expected range")
	else:
		raise RuntimeError("calculateRetinotopicVisualFieldRadiusDegreesForColumnFraction error: requires modalityORsnapshotRetinotopicFieldBias")
	return result


def calculateRetinotopicColumnPercentageForVisualFieldRadiusDegrees(visualFieldRadiusDegrees):
	# Human V1 reference table from GIAANNOR-theoreticalReferenceGuide.txt:
	# Central visual field radius from fixation | Approx. % of V1
	# 1 deg                                      | ~5%
	# 2 deg                                      | ~10-15%
	# 5 deg                                      | ~30-40%
	# 10 deg                                     | ~50-60%
	# 15 deg                                     | ~60-70%
	# 20 deg                                     | ~65-75%
	# 30 deg                                     | ~75-85%
	result = None
	table = None
	snapshotRetinotopicFieldMaxDegrees = None
	lowerRadiusDegrees = None
	lowerPercentage = None
	upperRadiusDegrees = None
	upperPercentage = None
	interpolationFraction = None
	if(modalityORsnapshotRetinotopicFieldBias):
		validateRetinotopicFieldBiasParameters()
		if(visualFieldRadiusDegrees < 0.0):
			raise RuntimeError("calculateRetinotopicColumnPercentageForVisualFieldRadiusDegrees error: visualFieldRadiusDegrees must be >= 0.0")
		snapshotRetinotopicFieldMaxDegrees = calculateExpectedSnapshotRetinotopicFieldMaxDegrees()
		if(visualFieldRadiusDegrees > snapshotRetinotopicFieldMaxDegrees):
			raise RuntimeError("calculateRetinotopicColumnPercentageForVisualFieldRadiusDegrees error: visualFieldRadiusDegrees exceeds snapshot retinotopic FOV max degrees")
		table = getRetinotopicFieldBiasReferenceTable()
		if(visualFieldRadiusDegrees > float(table[-1][0])):
			raise RuntimeError("calculateRetinotopicColumnPercentageForVisualFieldRadiusDegrees error: visualFieldRadiusDegrees exceeds retinotopic table range")
		result = float(table[-1][1])
		for tableIndex in range(1, len(table)):
			lowerRadiusDegrees = float(table[tableIndex - 1][0])
			lowerPercentage = float(table[tableIndex - 1][1])
			upperRadiusDegrees = float(table[tableIndex][0])
			upperPercentage = float(table[tableIndex][1])
			if(visualFieldRadiusDegrees >= lowerRadiusDegrees and visualFieldRadiusDegrees <= upperRadiusDegrees):
				interpolationFraction = (float(visualFieldRadiusDegrees) - lowerRadiusDegrees)/(upperRadiusDegrees - lowerRadiusDegrees)
				result = lowerPercentage + (interpolationFraction*(upperPercentage - lowerPercentage))
				break
	else:
		raise RuntimeError("calculateRetinotopicColumnPercentageForVisualFieldRadiusDegrees error: requires modalityORsnapshotRetinotopicFieldBias")
	return result


def validateRetinotopicFieldBiasParameters():
	result = None
	table = None
	expectedSnapshotRetinotopicFieldMaxDegrees = None
	if(modalityORsnapshotRetinotopicFieldBias):
		if(not isinstance(datasetCameraFOV, int) and not isinstance(datasetCameraFOV, float)):
			raise RuntimeError("validateRetinotopicFieldBiasParameters error: datasetCameraFOV must be an int or float")
		if(datasetCameraFOV <= 0.0):
			raise RuntimeError("validateRetinotopicFieldBiasParameters error: datasetCameraFOV must be > 0.0")
		if(not isinstance(modalityORsnapshotFractionOfImage, int) and not isinstance(modalityORsnapshotFractionOfImage, float)):
			raise RuntimeError("validateRetinotopicFieldBiasParameters error: modalityORsnapshotFractionOfImage must be an int or float")
		if(modalityORsnapshotFractionOfImage <= 0.0 or modalityORsnapshotFractionOfImage > 1.0):
			raise RuntimeError("validateRetinotopicFieldBiasParameters error: modalityORsnapshotFractionOfImage must be > 0.0 and <= 1.0")
		if(not isinstance(modalityORsnapshotRetinotopicFieldMaxDegrees, int) and not isinstance(modalityORsnapshotRetinotopicFieldMaxDegrees, float)):
			raise RuntimeError("validateRetinotopicFieldBiasParameters error: modalityORsnapshotRetinotopicFieldMaxDegrees must be an int or float")
		if(modalityORsnapshotRetinotopicFieldMaxDegrees <= 0.0):
			raise RuntimeError("validateRetinotopicFieldBiasParameters error: modalityORsnapshotRetinotopicFieldMaxDegrees must be > 0.0")
		expectedSnapshotRetinotopicFieldMaxDegrees = calculateExpectedSnapshotRetinotopicFieldMaxDegrees()
		if(abs(float(modalityORsnapshotRetinotopicFieldMaxDegrees) - expectedSnapshotRetinotopicFieldMaxDegrees) > 0.000001):
			raise RuntimeError("validateRetinotopicFieldBiasParameters error: modalityORsnapshotRetinotopicFieldMaxDegrees must equal modalityORsnapshotFractionOfImage*datasetCameraFOV")
		table = getRetinotopicFieldBiasReferenceTable()
		if(float(modalityORsnapshotRetinotopicFieldMaxDegrees) > float(table[-1][0])):
			raise RuntimeError("validateRetinotopicFieldBiasParameters error: modalityORsnapshotRetinotopicFieldMaxDegrees exceeds retinotopic table range")
	else:
		raise RuntimeError("validateRetinotopicFieldBiasParameters error: requires modalityORsnapshotRetinotopicFieldBias")
	return result


def calculateExpectedSnapshotRetinotopicFieldMaxDegrees():
	result = None
	if(modalityORsnapshotRetinotopicFieldBias):
		if(not isinstance(datasetCameraFOV, int) and not isinstance(datasetCameraFOV, float)):
			raise RuntimeError("calculateExpectedSnapshotRetinotopicFieldMaxDegrees error: datasetCameraFOV must be an int or float")
		if(not isinstance(modalityORsnapshotFractionOfImage, int) and not isinstance(modalityORsnapshotFractionOfImage, float)):
			raise RuntimeError("calculateExpectedSnapshotRetinotopicFieldMaxDegrees error: modalityORsnapshotFractionOfImage must be an int or float")
		result = float(modalityORsnapshotFractionOfImage)*float(datasetCameraFOV)
	else:
		raise RuntimeError("calculateExpectedSnapshotRetinotopicFieldMaxDegrees error: requires modalityORsnapshotRetinotopicFieldBias")
	return result


def getRetinotopicFieldBiasReferenceTable():
	result = None
	if(modalityORsnapshotRetinotopicFieldBias):
		result = [(0.0, 0.0), (1.0, 5.0), (2.0, 12.5), (5.0, 35.0), (10.0, 55.0), (15.0, 65.0), (20.0, 70.0), (30.0, 80.0)]
	else:
		raise RuntimeError("getRetinotopicFieldBiasReferenceTable error: requires modalityORsnapshotRetinotopicFieldBias")
	return result
