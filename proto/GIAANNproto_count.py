"""GIAANNproto_count.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_main.py

# Usage:
see GIAANNproto_main.py

# Description:
GIA ANN proto count helpers

"""

import os
import torch as pt

from GIAANNproto_globalDefs import *

def printCountTotalParametersRun(databaseNetworkObject):
	assert arrayIndexPropertiesEfficient 	#only databaseNetworkObject.arrayIndexPropertiesStrengthIndex stored in database, all tensors are coalesced
	if(databaseNetworkObject is None):
		raise RuntimeError("printCountTotalParametersRun error: databaseNetworkObject is None")
	if(databaseNetworkObject.arrayIndexPropertiesStrengthIndex is None):
		raise RuntimeError("printCountTotalParametersRun error: databaseNetworkObject.arrayIndexPropertiesStrengthIndex is None")
	totalColumns = len(databaseNetworkObject.conceptColumnsList)
	if(totalColumns <= 0):
		raise RuntimeError("printCountTotalParametersRun error: conceptColumnsList is empty")
	totalConnections = 0
	totalFeatureNeurons = 0
	if(storeDatabaseGlobalFeatureNeuronsInRam):
		if(databaseNetworkObject.globalFeatureNeurons is None):
			raise RuntimeError("printCountTotalParametersRun error: databaseNetworkObject.globalFeatureNeurons is None")
		totalFeatureNeurons = countAssignedFeatureNeuronsInTensor(databaseNetworkObject.globalFeatureNeurons, databaseNetworkObject.arrayIndexPropertiesStrengthIndex, "databaseNetworkObject.globalFeatureNeurons")
	for columnIndex, lemma in enumerate(databaseNetworkObject.conceptColumnsList):
		conceptIndex = databaseNetworkObject.conceptColumnsDict.get(lemma)
		if(conceptIndex is None):
			raise RuntimeError("printCountTotalParametersRun error: conceptIndex is None for lemma = " + lemma)
		columnConnections = debugCountObservedColumnConnections(databaseNetworkObject, conceptIndex, lemma, columnIndex)
		totalConnections += columnConnections
		if(not storeDatabaseGlobalFeatureNeuronsInRam):
			totalFeatureNeurons += loadAndCountObservedColumnFeatureNeurons(databaseNetworkObject, conceptIndex)
	databasePtSizeGb = debugCalculateDatabasePtSizeGiB()
	memoryGb = debugCalculateDatabaseSizeGiB()
	if(printCountTotalParameters):
		numberNeurons = totalFeatureNeurons
		numberConnections = totalConnections
		numberColumns = totalColumns
		numberFeatures = databaseNetworkObject.f
		databaseSizeGB = memoryGb
		#print(f"Total .pt size (uncompressed GB): {databasePtSizeGb:.3f}")
		print(f"neurons: {numberNeurons}")
		print(f"connections: {numberConnections}")
		print(f"columns: {numberColumns}")
		print(f"features: {numberFeatures}")
		print(f"database size (uncompressed GB): {databaseSizeGB:.3f}")
	return memoryGb

def debugCountObservedColumnConnections(databaseNetworkObject, conceptIndex, lemma, columnIndex):
	import GIAANNproto_databaseNetworkFiles
	columnConnections = 0
	if(GIAANNproto_databaseNetworkFiles.observedColumnMetadataExists(conceptIndex)):
		sourceFeatureIndices = GIAANNproto_databaseNetworkFiles.listObservedColumnSourceFeatureIndices(conceptIndex)
		for sourceFeatureIndex in sourceFeatureIndices:
			featureConnections = GIAANNproto_databaseNetworkFiles.loadObservedColumnSourceFeatureConnectionsTensor(databaseNetworkObject, conceptIndex, sourceFeatureIndex, deviceDatabase)
			if(featureConnections is None):
				raise RuntimeError("debugCountObservedColumnConnections error: featureConnections is None for conceptIndex = " + str(conceptIndex) + ", sourceFeatureIndex = " + str(sourceFeatureIndex))
			if(databaseNetworkObject.arrayIndexPropertiesStrengthIndex < 0 or databaseNetworkObject.arrayIndexPropertiesStrengthIndex >= featureConnections.shape[0]):
				raise RuntimeError("debugCountObservedColumnConnections error: databaseNetworkObject.arrayIndexPropertiesStrengthIndex out of range")
			if(featureConnections.is_sparse):
				columnConnections = columnConnections + int(featureConnections._nnz())
			else:
				columnConnections = columnConnections + int(pt.count_nonzero(featureConnections).item())
			del featureConnections
	return columnConnections

def loadAndCountObservedColumnFeatureNeurons(databaseNetworkObject, conceptIndex):
	import GIAANNproto_databaseNetworkFiles
	result = 0
	if(databaseNetworkObject is None):
		raise RuntimeError("loadAndCountObservedColumnFeatureNeurons error: databaseNetworkObject is None")
	if(conceptIndex is None):
		raise RuntimeError("loadAndCountObservedColumnFeatureNeurons error: conceptIndex is None")
	if(not GIAANNproto_databaseNetworkFiles.observedColumnMetadataExists(conceptIndex)):
		raise RuntimeError("loadAndCountObservedColumnFeatureNeurons error: observed column metadata does not exist for conceptIndex = " + str(conceptIndex))
	featureNeuronsTensorName = f"observedColumn.featureNeurons[{conceptIndex}]"
	featureNeurons = GIAANNproto_databaseNetworkFiles.loadTensor(GIAANNproto_databaseNetworkFiles.getObservedColumnFolder(conceptIndex), GIAANNproto_databaseNetworkFiles.getObservedColumnFeatureNeuronsFileBaseName(), targetDevice=deviceDatabase)
	featureNeurons = GIAANNproto_databaseNetworkFiles.adjustPropertyDimensions(databaseNetworkObject.inferenceMode, featureNeurons, featureNeuronsTensorName)
	featureNeurons = GIAANNproto_databaseNetworkFiles.adjustBranchDimensions(featureNeurons, featureNeuronsTensorName, expectedRank=4)
	if(debugLimitFeatures):
		featureNeurons = applyDebugLimitFeatureNeuronsTensor(featureNeurons, databaseNetworkObject.f, featureNeuronsTensorName)
	result = countAssignedFeatureNeuronsInTensor(featureNeurons, databaseNetworkObject.arrayIndexPropertiesStrengthIndex, featureNeuronsTensorName)
	return result

def debugCalculateDatabasePtSizeGiB():
	if(not os.path.isdir(databaseFolder)):
		raise RuntimeError("debugCalculateDatabasePtSizeGiB error: missing databaseFolder = " + databaseFolder)
	totalPtBytesUncompressed = 0
	for directoryPath, directoryNames, fileNames in os.walk(databaseFolder):
		if(directoryNames is None):
			raise RuntimeError("debugCalculateDatabasePtSizeGiB error: directoryNames is None")
		for fileName in fileNames:
			filePath = os.path.join(directoryPath, fileName)
			if(not os.path.isfile(filePath)):
				raise RuntimeError("debugCalculateDatabasePtSizeGiB error: path is not a file = " + filePath)
			if(fileName.lower().endswith(pytorchTensorFileExtension.lower())):
				totalPtBytesUncompressed = totalPtBytesUncompressed + os.path.getsize(filePath)
	totalPtGiB = totalPtBytesUncompressed / (1024 ** 3)
	return totalPtGiB

def debugCalculateDatabaseSizeGiB():
	if(not os.path.isdir(databaseFolder)):
		raise RuntimeError("debugCalculateDatabaseSizeGiB error: missing databaseFolder = " + databaseFolder)
	totalDatabaseBytesUncompressed = 0
	for directoryPath, directoryNames, fileNames in os.walk(databaseFolder):
		if(directoryNames is None):
			raise RuntimeError("debugCalculateDatabaseSizeGiB error: directoryNames is None")
		for fileName in fileNames:
			filePath = os.path.join(directoryPath, fileName)
			if(not os.path.isfile(filePath)):
				raise RuntimeError("debugCalculateDatabaseSizeGiB error: path is not a file = " + filePath)
			totalDatabaseBytesUncompressed = totalDatabaseBytesUncompressed + os.path.getsize(filePath)
	totalDatabaseGiB = totalDatabaseBytesUncompressed / (1024 ** 3)
	return totalDatabaseGiB

def countAssignedFeatureNeuronsInTensor(featureNeurons, strengthPropertyIndex, tensorName):
	result = 0
	if(featureNeurons is None):
		raise RuntimeError("countAssignedFeatureNeuronsInTensor error: featureNeurons is None for " + tensorName)
	if(strengthPropertyIndex is None):
		raise RuntimeError("countAssignedFeatureNeuronsInTensor error: strengthPropertyIndex is None for " + tensorName)
	if(featureNeurons.dim() < 1):
		raise RuntimeError("countAssignedFeatureNeuronsInTensor error: featureNeurons.dim() < 1 for " + tensorName)
	if(strengthPropertyIndex < 0 or strengthPropertyIndex >= featureNeurons.size(0)):
		raise RuntimeError("countAssignedFeatureNeuronsInTensor error: strengthPropertyIndex out of range for " + tensorName)
	if(featureNeurons.layout == pt.sparse_coo):
		featureNeurons = featureNeurons.coalesce()
		strengthMask = pt.logical_and(featureNeurons.indices()[0] == strengthPropertyIndex, featureNeurons.values() > 0)
		if(bool(pt.any(strengthMask).item())):
			if(featureNeurons.dim() == 5):
				neuronIndices = featureNeurons.indices()[3:5, strengthMask]
				result = int(pt.unique(neuronIndices, dim=1).shape[1])
			elif(featureNeurons.dim() == 4):
				neuronIndices = featureNeurons.indices()[3, strengthMask]
				result = int(pt.unique(neuronIndices).shape[0])
			else:
				raise RuntimeError("countAssignedFeatureNeuronsInTensor error: unsupported tensor rank for " + tensorName + ", dim = " + str(featureNeurons.dim()))
	elif(featureNeurons.layout == pt.strided):
		strengthTensor = featureNeurons[strengthPropertyIndex]
		if(featureNeurons.dim() == 5):
			positiveNeuronMask = pt.any(strengthTensor > 0, dim=0)
			positiveNeuronMask = pt.any(positiveNeuronMask, dim=0)
			result = int(pt.count_nonzero(positiveNeuronMask).item())
		elif(featureNeurons.dim() == 4):
			positiveNeuronMask = pt.any(strengthTensor > 0, dim=0)
			positiveNeuronMask = pt.any(positiveNeuronMask, dim=0)
			result = int(pt.count_nonzero(positiveNeuronMask).item())
		else:
			raise RuntimeError("countAssignedFeatureNeuronsInTensor error: unsupported tensor rank for " + tensorName + ", dim = " + str(featureNeurons.dim()))
	else:
		raise RuntimeError("countAssignedFeatureNeuronsInTensor error: unsupported tensor layout for " + tensorName + ", layout = " + str(featureNeurons.layout))
	return result

def applyDebugLimitFeatureNeuronsTensor(tensor, fLimit, tensorName):
	result = tensor
	if(debugLimitFeatures):
		if(fLimit <= 0):
			raise RuntimeError(f"{tensorName} debug limit requires positive limits")
		capF = tensor.size(3)
		if(capF > fLimit):
			capF = fLimit
		if(capF != tensor.size(3)):
			if(tensor.is_sparse):
				tensor = tensor.coalesce()
				indices = tensor.indices()
				values = tensor.values()
				mask = indices[3] < capF
				indices = indices[:, mask]
				values = values[mask]
				newSize = list(tensor.size())
				newSize[3] = capF
				result = pt.sparse_coo_tensor(indices, values, size=newSize, dtype=tensor.dtype, device=tensor.device).coalesce()
			else:
				result = tensor[:, :, :, :capF]
	return result
