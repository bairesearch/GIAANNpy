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
	if(modalityORpixelsPerColumn <= 0):
		raise RuntimeError("tokeniseSnapshotsToColumns error: modalityORpixelsPerColumn must be > 0")
	gridHeight = int(snapshotTensor.shape[2]//modalityORpixelsPerColumn)
	gridWidth = int(snapshotTensor.shape[3]//modalityORpixelsPerColumn)
	if(gridHeight <= 0 or gridWidth <= 0):
		raise RuntimeError("tokeniseSnapshotsToColumns error: snapshotTensor is smaller than modalityORpixelsPerColumn")
	snapshotTensorTrimmed = snapshotTensor[:, :, :gridHeight*modalityORpixelsPerColumn, :gridWidth*modalityORpixelsPerColumn]
	patchTensor = snapshotTensorTrimmed.unfold(2, modalityORpixelsPerColumn, modalityORpixelsPerColumn).unfold(3, modalityORpixelsPerColumn, modalityORpixelsPerColumn)
	patchTensor = patchTensor.permute(0, 2, 3, 1, 4, 5).contiguous()
	columnTensor = patchTensor.view(patchTensor.shape[0], gridHeight*gridWidth, patchTensor.shape[3], patchTensor.shape[4], patchTensor.shape[5])
	columnMetadataList = buildColumnMetadataList(gridHeight, gridWidth)
	result = {"columnTensor": columnTensor, "columnMetadataList": columnMetadataList, "gridHeight": gridHeight, "gridWidth": gridWidth}
	return result

