"""GIAANNcmn_count.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 BAI Research Pty Ltd (bairesearch.com.au)

# License:
MIT License

# Installation:
see GIAANNcmn_main.py

# Usage:
see GIAANNcmn_main.py

# Description:
GIA ANN common count helpers

"""

import os
import torch as pt

from GIAANNcmn_globalDefs import *

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
	usePersistedGlobalFeatureNeurons = countUsesPersistedGlobalFeatureNeurons()
	if(usePersistedGlobalFeatureNeurons):
		if(databaseNetworkObject.globalFeatureNeurons is None):
			raise RuntimeError("printCountTotalParametersRun error: databaseNetworkObject.globalFeatureNeurons is None")
		totalFeatureNeurons = countAssignedFeatureNeuronsInTensor(databaseNetworkObject.globalFeatureNeurons, databaseNetworkObject.arrayIndexPropertiesStrengthIndex, "databaseNetworkObject.globalFeatureNeurons")
	for columnIndex, lemma in enumerate(databaseNetworkObject.conceptColumnsList):
		conceptIndex = databaseNetworkObject.conceptColumnsDict.get(lemma)
		if(conceptIndex is None):
			raise RuntimeError("printCountTotalParametersRun error: conceptIndex is None for lemma = " + lemma)
		columnConnections = debugCountObservedColumnConnections(databaseNetworkObject, conceptIndex, lemma, columnIndex)
		totalConnections += columnConnections
		if(not usePersistedGlobalFeatureNeurons):
			totalFeatureNeurons += loadAndCountObservedColumnFeatureNeurons(databaseNetworkObject, conceptIndex)
	databasePtSizeGb = debugCalculateDatabasePtSizeGiB()
	databaseMemoryGb = debugCalculateDatabaseSizeGiB()
	if(printCountTotalParameters):
		numberNeurons = totalFeatureNeurons
		numberConnections = totalConnections
		numberColumns = totalColumns
		numberFeatures = databaseNetworkObject.f
		databaseSizeGB = databaseMemoryGb
		#print(f"Total .pt size (uncompressed GB): {databasePtSizeGb:.3f}")
		print(f"neurons: {numberNeurons}")
		print(f"connections: {numberConnections}")
		print(f"columns: {numberColumns}")
		print(f"features: {numberFeatures}")
		print(f"database size (uncompressed GB): {databaseSizeGB:.3f}")
	return databaseMemoryGb

def countUsesPersistedGlobalFeatureNeurons():
	import GIAANNcmn_databaseNetworkFiles
	result = False
	if(storeDatabaseGlobalFeatureNeuronsInRam):
		result = GIAANNcmn_databaseNetworkFiles.pathExists(globalFeatureNeuronsFileFull)
	return result

def debugCountObservedColumnConnections(databaseNetworkObject, conceptIndex, lemma, columnIndex):
	import GIAANNcmn_databaseNetworkFiles
	columnConnections = 0
	if(GIAANNcmn_databaseNetworkFiles.observedColumnHasPersistedData(conceptIndex)):
		if(not GIAANNcmn_databaseNetworkFiles.observedColumnHasConsistentPersistedMetadata(conceptIndex)):
			raise RuntimeError("debugCountObservedColumnConnections error: inconsistent observed column storage for conceptIndex = " + str(conceptIndex) + ", lemma = " + lemma)
		sourceFeatureIndices = GIAANNcmn_databaseNetworkFiles.listObservedColumnSourceFeatureIndices(conceptIndex)
		for sourceFeatureIndex in sourceFeatureIndices:
			featureConnections = GIAANNcmn_databaseNetworkFiles.loadObservedColumnSourceFeatureConnectionsTensor(databaseNetworkObject, conceptIndex, sourceFeatureIndex, deviceDatabase)
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
	import GIAANNcmn_databaseNetworkFiles
	result = 0
	if(databaseNetworkObject is None):
		raise RuntimeError("loadAndCountObservedColumnFeatureNeurons error: databaseNetworkObject is None")
	if(conceptIndex is None):
		raise RuntimeError("loadAndCountObservedColumnFeatureNeurons error: conceptIndex is None")
	if(GIAANNcmn_databaseNetworkFiles.observedColumnHasPersistedData(conceptIndex)):
		if(not GIAANNcmn_databaseNetworkFiles.observedColumnHasConsistentPersistedMetadata(conceptIndex)):
			raise RuntimeError("loadAndCountObservedColumnFeatureNeurons error: inconsistent observed column storage for conceptIndex = " + str(conceptIndex))
		featureNeuronsTensorName = f"observedColumn.featureNeurons[{conceptIndex}]"
		featureNeurons = GIAANNcmn_databaseNetworkFiles.loadTensor(GIAANNcmn_databaseNetworkFiles.getObservedColumnFolder(conceptIndex), GIAANNcmn_databaseNetworkFiles.getObservedColumnFeatureNeuronsFileBaseName(), targetDevice=deviceDatabase)
		featureNeurons = GIAANNcmn_databaseNetworkFiles.adjustPropertyDimensions(databaseNetworkObject.inferenceMode, featureNeurons, featureNeuronsTensorName)
		featureNeurons = GIAANNcmn_databaseNetworkFiles.adjustBranchDimensions(featureNeurons, featureNeuronsTensorName, expectedRank=4)
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

def countResetGpuRamMaxUsage():
	if(printRamMaxUsage or useAutoresearch):
		gpuRamUsageDevice = countGetGpuRamUsageDevice()
		if(gpuRamUsageDevice is not None):
			pt.cuda.reset_peak_memory_stats(gpuRamUsageDevice)
	return

def countPrintGpuRamMaxUsageSummary():
	if(printRamMaxUsage):
		gpuRamMaxAllocatedUsageBytes = countGetGpuRamMaxAllocatedUsageBytes()
		gpuRamMaxReservedUsageBytes = countGetGpuRamMaxReservedUsageBytes()
		cpuRamMaxUsageBytes = countGetCpuRamMaxUsageBytes()
		if(useAutoresearch):
			raise RuntimeError("countPrintGpuRamMaxUsageSummary error: useAutoresearch path is not implemented")
		else:
			printText = "debugPrintGPUramUsage max: program, gpuRamMaxAllocatedUsageGb = " + countConvertRamUsageBytesToGigabytesText(gpuRamMaxAllocatedUsageBytes) + ", gpuRamMaxReservedUsageGb = " + countConvertRamUsageBytesToGigabytesText(gpuRamMaxReservedUsageBytes) + ", cpuRamMaxUsageGb = " + countConvertRamUsageBytesToGigabytesText(cpuRamMaxUsageBytes)
		print(printText)
	return

def countConvertRamUsageBytesToGigabytesText(ramUsageBytes):
	if(ramUsageBytes < 0):
		raise RuntimeError("countConvertRamUsageBytesToGigabytesText error: ramUsageBytes must be >= 0")
	result = f"{(float(ramUsageBytes) / (1024.0*1024.0*1024.0)):.4f}"
	return result
	
def countGetGpuRamMaxAllocatedUsageBytes():
	result = 0
	gpuRamUsageDevice = countGetGpuRamUsageDevice()
	if(gpuRamUsageDevice is not None):
		result = int(pt.cuda.max_memory_allocated(gpuRamUsageDevice))
	return result

def countGetGpuRamMaxReservedUsageBytes():
	result = 0
	gpuRamUsageDevice = countGetGpuRamUsageDevice()
	if(gpuRamUsageDevice is not None):
		result = int(pt.cuda.max_memory_reserved(gpuRamUsageDevice))
	return result

def countGetCpuRamMaxUsageBytes():
	result = countGetCpuRamUsageBytesByProcStatusField("VmHWM", "countGetCpuRamMaxUsageBytes")
	return result

def countGetCpuRamUsageBytesByProcStatusField(memoryFieldName, functionName):
	if(memoryFieldName is None or memoryFieldName == ""):
		raise RuntimeError("countGetCpuRamUsageBytesByProcStatusField error: memoryFieldName must not be empty")
	if(functionName is None or functionName == ""):
		raise RuntimeError("countGetCpuRamUsageBytesByProcStatusField error: functionName must not be empty")
	result = None
	processStatusFieldKey = memoryFieldName + ":"
	with open("/proc/self/status", "r") as processStatusFile:
		for processStatusLine in processStatusFile:
			if(processStatusLine.startswith(processStatusFieldKey)):
				processStatusFields = processStatusLine.split()
				if(len(processStatusFields) < 2):
					raise RuntimeError(functionName + " error: " + processStatusFieldKey + " line is malformed")
				result = int(processStatusFields[1]) * 1024
	if(result is None):
		raise RuntimeError(functionName + " error: " + processStatusFieldKey + " not found in /proc/self/status")
	return result

def countGetGpuRamUsageDevice():
	result = None
	if(deviceDense.type == "cuda"):
		result = deviceDense
	elif(deviceSparse.type == "cuda"):
		result = deviceSparse
	elif(deviceFileIO.type == "cuda"):
		result = deviceFileIO
	elif(deviceDatabase.type == "cuda"):
		result = deviceDatabase
	elif(pt.cuda.is_available()):
		result = pt.device("cuda")
	return result

def printCountPrintTimeDatabaseLoadSaveTimesSummary(summaryName, totalExecutionTimeSeconds, loadAllObservedColumnsToRamExecutionTimeSeconds, saveAllObservedColumnsToDiskExecutionTimeSeconds):
	processingExecutionTimeSeconds = totalExecutionTimeSeconds - loadAllObservedColumnsToRamExecutionTimeSeconds - saveAllObservedColumnsToDiskExecutionTimeSeconds
	if(processingExecutionTimeSeconds < 0):
		raise RuntimeError("printCountPrintTimeDatabaseLoadSaveTimesSummary error: processingExecutionTimeSeconds must be >= 0")
	print(summaryName)
	printCountPrintTimeDatabaseLoadSaveTimesEntry("total execution time", totalExecutionTimeSeconds)
	printCountPrintTimeDatabaseLoadSaveTimesEntry("loadAllObservedColumnsToRam execution time", loadAllObservedColumnsToRamExecutionTimeSeconds)
	printCountPrintTimeDatabaseLoadSaveTimesEntry("saveAllObservedColumnsToDisk execution time", saveAllObservedColumnsToDiskExecutionTimeSeconds)
	printCountPrintTimeDatabaseLoadSaveTimesEntry("processing time", processingExecutionTimeSeconds)
	return

def printCountPrintTimeDatabaseLoadSaveTimesEntry(executionTimeName, executionTimeSeconds):
	executionTimeText = getCountPrintTimeDatabaseLoadSaveTimesText(executionTimeSeconds)
	print(executionTimeName + ": " + executionTimeText)
	return

def getCountPrintTimeDatabaseLoadSaveTimesText(executionTimeSeconds):
	if(executionTimeSeconds < 0):
		raise RuntimeError("getCountPrintTimeDatabaseLoadSaveTimesText error: executionTimeSeconds must be >= 0")
	executionTimeHoursMinutesSecondsText = getCountPrintTimeDatabaseLoadSaveTimesHoursMinutesSecondsText(executionTimeSeconds)
	result = executionTimeHoursMinutesSecondsText + " [" + f"{executionTimeSeconds:.6f}" + " seconds]"
	return result

def getCountPrintTimeDatabaseLoadSaveTimesHoursMinutesSecondsText(executionTimeSeconds):
	if(executionTimeSeconds < 0):
		raise RuntimeError("getCountPrintTimeDatabaseLoadSaveTimesHoursMinutesSecondsText error: executionTimeSeconds must be >= 0")
	totalExecutionTimeSecondsInteger = int(executionTimeSeconds)
	executionHours = totalExecutionTimeSecondsInteger // 3600
	executionMinutes = (totalExecutionTimeSecondsInteger % 3600) // 60
	executionSeconds = totalExecutionTimeSecondsInteger % 60
	result = f"{executionHours:02d}:{executionMinutes:02d}:{executionSeconds:02d}"
	return result

def getCountPrintTimeDatabaseLoadSaveTimesExecutionModeCount():
	result = 0
	if(executionMode=="inference"):
		result = 1
	elif(executionMode=="trainAndInference"):
		result = 2
	elif(executionMode=="train"):
		result = 1
	else:
		raise RuntimeError("getCountPrintTimeDatabaseLoadSaveTimesExecutionModeCount error: unsupported executionMode = " + str(executionMode))
	return result
