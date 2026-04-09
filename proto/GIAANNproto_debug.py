"""GIAANNproto_debug.py

# Author:
Richard Bruce Baxter - Copyright (c) 2024-2026 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see GIAANNproto_main.py

# Usage:
see GIAANNproto_main.py

# Description:
GIA ANN proto debug helpers

"""

import os
import torch as pt

from GIAANNproto_globalDefs import *

debugPrintGPUramUsage = debugPrintRamCurrentUsage or debugPrintRamAverageUsage or debugPrintRamMaxUsage or debugPrintRamMaxUsagePhaseLocal
totalInferenceTokensSeed = 0
totalInferenceTokensPrediction = 0
totalInferenceTokensAll = 0

if(debugPrintGPUramUsage):
	debugPrintRamUsageStatistics = {}
	debugPrintGpuRamMaxUsageProgramAllocatedBytes = 0
	debugPrintGpuRamMaxUsageProgramReservedBytes = 0
	debugPrintGpuRamMaxUsagePhaseLocalStatistics = {}

	def debugPrintRamUsage(label, contextText=""):
		if(label is None or label == ""):
			raise RuntimeError("debugPrintRamUsage error: label must not be empty")
		if(contextText is None):
			raise RuntimeError("debugPrintRamUsage error: contextText must not be None")
		gpuRamCurrentAllocatedUsageBytes = debugGetGpuRamCurrentAllocatedUsageBytes()
		gpuRamCurrentReservedUsageBytes = debugGetGpuRamCurrentReservedUsageBytes()
		cpuRamCurrentUsageBytes = debugGetCpuRamCurrentUsageBytes()
		gpuRamAverageAllocatedUsageBytes, gpuRamAverageReservedUsageBytes, cpuRamAverageUsageBytes, sampleCount = debugGetRamUsageStatistics(label, gpuRamCurrentAllocatedUsageBytes, gpuRamCurrentReservedUsageBytes, cpuRamCurrentUsageBytes)
		printText = "debugPrintGPUramUsage: " + label
		if(contextText != ""):
			printText = printText + " (" + contextText + ")"
		if(debugPrintRamCurrentUsage):
			printText = printText + ", gpuRamCurrentAllocatedUsageGb = " + debugConvertRamUsageBytesToGigabytesText(gpuRamCurrentAllocatedUsageBytes) + ", gpuRamCurrentReservedUsageGb = " + debugConvertRamUsageBytesToGigabytesText(gpuRamCurrentReservedUsageBytes) + ", cpuRamCurrentUsageGb = " + debugConvertRamUsageBytesToGigabytesText(cpuRamCurrentUsageBytes)
			if(debugPrintRamAverageUsage):
				printText = printText + ", gpuRamAverageAllocatedUsageGb = " + debugConvertRamUsageBytesToGigabytesText(gpuRamAverageAllocatedUsageBytes) + ", gpuRamAverageReservedUsageGb = " + debugConvertRamUsageBytesToGigabytesText(gpuRamAverageReservedUsageBytes) + ", cpuRamAverageUsageGb = " + debugConvertRamUsageBytesToGigabytesText(cpuRamAverageUsageBytes) + ", sampleCount = " + str(sampleCount)
			print(printText)
		return

	def debugPrintRamUsageSummary():
		if(debugPrintRamAverageUsage):
			for label in sorted(debugPrintRamUsageStatistics.keys()):
				ramUsageStatistics = debugPrintRamUsageStatistics[label]
				sampleCount = ramUsageStatistics["sampleCount"]
				if(sampleCount <= 0):
					raise RuntimeError("debugPrintRamUsageSummary error: sampleCount must be > 0")
				gpuRamAverageAllocatedUsageBytes = ramUsageStatistics["gpuRamAllocatedUsageTotalBytes"] / sampleCount
				gpuRamAverageReservedUsageBytes = ramUsageStatistics["gpuRamReservedUsageTotalBytes"] / sampleCount
				cpuRamAverageUsageBytes = ramUsageStatistics["cpuRamUsageTotalBytes"] / sampleCount
				printText = "debugPrintGPUramUsage average: " + label + ", gpuRamAverageAllocatedUsageGb = " + debugConvertRamUsageBytesToGigabytesText(gpuRamAverageAllocatedUsageBytes) + ", gpuRamAverageReservedUsageGb = " + debugConvertRamUsageBytesToGigabytesText(gpuRamAverageReservedUsageBytes) + ", cpuRamAverageUsageGb = " + debugConvertRamUsageBytesToGigabytesText(cpuRamAverageUsageBytes) + ", sampleCount = " + str(sampleCount)
				print(printText)
		return

	def debugResetGpuRamMaxUsage():
		if(debugPrintRamMaxUsage or debugPrintRamMaxUsagePhaseLocal):
			global debugPrintGpuRamMaxUsageProgramAllocatedBytes
			global debugPrintGpuRamMaxUsageProgramReservedBytes
			debugPrintGpuRamMaxUsageProgramAllocatedBytes = 0
			debugPrintGpuRamMaxUsageProgramReservedBytes = 0
			gpuRamUsageDevice = debugGetGpuRamUsageDevice()
			if(gpuRamUsageDevice is not None):
				pt.cuda.reset_peak_memory_stats(gpuRamUsageDevice)
		return

	def debugResetGpuRamMaxUsagePhaseLocal(label):
		if(debugPrintRamMaxUsagePhaseLocal):
			if(label is None or label == ""):
				raise RuntimeError("debugResetGpuRamMaxUsagePhaseLocal error: label must not be empty")
			gpuRamUsageDevice = debugGetGpuRamUsageDevice()
			if(gpuRamUsageDevice is not None):
				pt.cuda.reset_peak_memory_stats(gpuRamUsageDevice)
		return

	def debugRecordGpuRamMaxUsagePhaseLocal(label):
		if(debugPrintRamMaxUsagePhaseLocal):
			if(label is None or label == ""):
				raise RuntimeError("debugRecordGpuRamMaxUsagePhaseLocal error: label must not be empty")
			gpuRamMaxAllocatedUsageBytes = debugGetGpuRamMaxAllocatedUsageBytes()
			gpuRamMaxReservedUsageBytes = debugGetGpuRamMaxReservedUsageBytes()
			debugRecordGpuRamMaxUsagePhaseLocalValue(label, gpuRamMaxAllocatedUsageBytes, gpuRamMaxReservedUsageBytes)
		return

	def debugRecordGpuRamMaxUsagePhaseLocalValue(label, gpuRamMaxAllocatedUsageBytes, gpuRamMaxReservedUsageBytes):
		if(debugPrintRamMaxUsagePhaseLocal):
			if(label is None or label == ""):
				raise RuntimeError("debugRecordGpuRamMaxUsagePhaseLocalValue error: label must not be empty")
			if(gpuRamMaxAllocatedUsageBytes < 0):
				raise RuntimeError("debugRecordGpuRamMaxUsagePhaseLocalValue error: gpuRamMaxAllocatedUsageBytes must be >= 0")
			if(gpuRamMaxReservedUsageBytes < 0):
				raise RuntimeError("debugRecordGpuRamMaxUsagePhaseLocalValue error: gpuRamMaxReservedUsageBytes must be >= 0")
			debugUpdateGpuRamMaxUsageProgramStatistics(gpuRamMaxAllocatedUsageBytes, gpuRamMaxReservedUsageBytes)
			if(label in debugPrintGpuRamMaxUsagePhaseLocalStatistics):
				phaseLocalStatistics = debugPrintGpuRamMaxUsagePhaseLocalStatistics[label]
			else:
				phaseLocalStatistics = {"gpuRamMaxAllocatedUsageBytes": 0, "gpuRamMaxReservedUsageBytes": 0, "sampleCount": 0}
				debugPrintGpuRamMaxUsagePhaseLocalStatistics[label] = phaseLocalStatistics
			if(gpuRamMaxAllocatedUsageBytes > phaseLocalStatistics["gpuRamMaxAllocatedUsageBytes"]):
				phaseLocalStatistics["gpuRamMaxAllocatedUsageBytes"] = gpuRamMaxAllocatedUsageBytes
			if(gpuRamMaxReservedUsageBytes > phaseLocalStatistics["gpuRamMaxReservedUsageBytes"]):
				phaseLocalStatistics["gpuRamMaxReservedUsageBytes"] = gpuRamMaxReservedUsageBytes
			phaseLocalStatistics["sampleCount"] = phaseLocalStatistics["sampleCount"] + 1
		return

	def debugRecordGpuRamMaxUsagePhaseLocalGrouped(label, aggregateLabel=None):
		if(debugPrintRamMaxUsagePhaseLocal):
			if(label is None or label == ""):
				raise RuntimeError("debugRecordGpuRamMaxUsagePhaseLocalGrouped error: label must not be empty")
			if(aggregateLabel is not None and aggregateLabel == ""):
				raise RuntimeError("debugRecordGpuRamMaxUsagePhaseLocalGrouped error: aggregateLabel must not be empty")
			gpuRamMaxAllocatedUsageBytes = debugGetGpuRamMaxAllocatedUsageBytes()
			gpuRamMaxReservedUsageBytes = debugGetGpuRamMaxReservedUsageBytes()
			debugRecordGpuRamMaxUsagePhaseLocalValue(label, gpuRamMaxAllocatedUsageBytes, gpuRamMaxReservedUsageBytes)
			if(aggregateLabel is not None):
				debugRecordGpuRamMaxUsagePhaseLocalValue(aggregateLabel, gpuRamMaxAllocatedUsageBytes, gpuRamMaxReservedUsageBytes)
		return

	def debugPrintGpuRamMaxUsagePhaseLocalSummary():
		if(debugPrintRamMaxUsagePhaseLocal):
			for label in sorted(debugPrintGpuRamMaxUsagePhaseLocalStatistics.keys()):
				phaseLocalStatistics = debugPrintGpuRamMaxUsagePhaseLocalStatistics[label]
				sampleCount = phaseLocalStatistics["sampleCount"]
				if(sampleCount <= 0):
					raise RuntimeError("debugPrintGpuRamMaxUsagePhaseLocalSummary error: sampleCount must be > 0")
				gpuRamMaxAllocatedUsageBytes = phaseLocalStatistics["gpuRamMaxAllocatedUsageBytes"]
				gpuRamMaxReservedUsageBytes = phaseLocalStatistics["gpuRamMaxReservedUsageBytes"]
				printText = "debugPrintGPUramUsage max phase-local: " + label + ", gpuRamMaxAllocatedUsageGb = " + debugConvertRamUsageBytesToGigabytesText(gpuRamMaxAllocatedUsageBytes) + ", gpuRamMaxReservedUsageGb = " + debugConvertRamUsageBytesToGigabytesText(gpuRamMaxReservedUsageBytes) + ", sampleCount = " + str(sampleCount)
				print(printText)
		return

	def debugPrintGpuRamMaxUsageSummary():
		if(debugPrintRamMaxUsage):
			gpuRamMaxAllocatedUsageBytes = debugGetGpuRamMaxAllocatedUsageBytes()
			gpuRamMaxReservedUsageBytes = debugGetGpuRamMaxReservedUsageBytes()
			if(debugPrintRamMaxUsagePhaseLocal):
				debugUpdateGpuRamMaxUsageProgramStatistics(gpuRamMaxAllocatedUsageBytes, gpuRamMaxReservedUsageBytes)
				gpuRamMaxAllocatedUsageBytes = debugPrintGpuRamMaxUsageProgramAllocatedBytes
				gpuRamMaxReservedUsageBytes = debugPrintGpuRamMaxUsageProgramReservedBytes
			printText = "debugPrintGPUramUsage max: program, gpuRamMaxAllocatedUsageGb = " + debugConvertRamUsageBytesToGigabytesText(gpuRamMaxAllocatedUsageBytes) + ", gpuRamMaxReservedUsageGb = " + debugConvertRamUsageBytesToGigabytesText(gpuRamMaxReservedUsageBytes)
			print(printText)
		return

	def debugUpdateGpuRamMaxUsageProgramStatistics(gpuRamMaxAllocatedUsageBytes, gpuRamMaxReservedUsageBytes):
		global debugPrintGpuRamMaxUsageProgramAllocatedBytes
		global debugPrintGpuRamMaxUsageProgramReservedBytes
		if(gpuRamMaxAllocatedUsageBytes < 0):
			raise RuntimeError("debugUpdateGpuRamMaxUsageProgramStatistics error: gpuRamMaxAllocatedUsageBytes must be >= 0")
		if(gpuRamMaxReservedUsageBytes < 0):
			raise RuntimeError("debugUpdateGpuRamMaxUsageProgramStatistics error: gpuRamMaxReservedUsageBytes must be >= 0")
		if(gpuRamMaxAllocatedUsageBytes > debugPrintGpuRamMaxUsageProgramAllocatedBytes):
			debugPrintGpuRamMaxUsageProgramAllocatedBytes = gpuRamMaxAllocatedUsageBytes
		if(gpuRamMaxReservedUsageBytes > debugPrintGpuRamMaxUsageProgramReservedBytes):
			debugPrintGpuRamMaxUsageProgramReservedBytes = gpuRamMaxReservedUsageBytes
		return

	def debugGetRamUsageStatistics(label, gpuRamCurrentAllocatedUsageBytes, gpuRamCurrentReservedUsageBytes, cpuRamCurrentUsageBytes):
		if(gpuRamCurrentAllocatedUsageBytes < 0):
			raise RuntimeError("debugGetRamUsageStatistics error: gpuRamCurrentAllocatedUsageBytes must be >= 0")
		if(gpuRamCurrentReservedUsageBytes < 0):
			raise RuntimeError("debugGetRamUsageStatistics error: gpuRamCurrentReservedUsageBytes must be >= 0")
		if(cpuRamCurrentUsageBytes < 0):
			raise RuntimeError("debugGetRamUsageStatistics error: cpuRamCurrentUsageBytes must be >= 0")
		if(label in debugPrintRamUsageStatistics):
			ramUsageStatistics = debugPrintRamUsageStatistics[label]
		else:
			ramUsageStatistics = {"gpuRamAllocatedUsageTotalBytes": 0, "gpuRamReservedUsageTotalBytes": 0, "cpuRamUsageTotalBytes": 0, "sampleCount": 0}
			debugPrintRamUsageStatistics[label] = ramUsageStatistics
		ramUsageStatistics["gpuRamAllocatedUsageTotalBytes"] = ramUsageStatistics["gpuRamAllocatedUsageTotalBytes"] + gpuRamCurrentAllocatedUsageBytes
		ramUsageStatistics["gpuRamReservedUsageTotalBytes"] = ramUsageStatistics["gpuRamReservedUsageTotalBytes"] + gpuRamCurrentReservedUsageBytes
		ramUsageStatistics["cpuRamUsageTotalBytes"] = ramUsageStatistics["cpuRamUsageTotalBytes"] + cpuRamCurrentUsageBytes
		ramUsageStatistics["sampleCount"] = ramUsageStatistics["sampleCount"] + 1
		sampleCount = ramUsageStatistics["sampleCount"]
		if(sampleCount <= 0):
			raise RuntimeError("debugGetRamUsageStatistics error: sampleCount must be > 0")
		gpuRamAverageAllocatedUsageBytes = ramUsageStatistics["gpuRamAllocatedUsageTotalBytes"] / sampleCount
		gpuRamAverageReservedUsageBytes = ramUsageStatistics["gpuRamReservedUsageTotalBytes"] / sampleCount
		cpuRamAverageUsageBytes = ramUsageStatistics["cpuRamUsageTotalBytes"] / sampleCount
		return gpuRamAverageAllocatedUsageBytes, gpuRamAverageReservedUsageBytes, cpuRamAverageUsageBytes, sampleCount

	def debugGetGpuRamCurrentAllocatedUsageBytes():
		result = 0
		gpuRamUsageDevice = debugGetGpuRamUsageDevice()
		if(gpuRamUsageDevice is not None):
			result = int(pt.cuda.memory_allocated(gpuRamUsageDevice))
		return result

	def debugGetGpuRamCurrentReservedUsageBytes():
		result = 0
		gpuRamUsageDevice = debugGetGpuRamUsageDevice()
		if(gpuRamUsageDevice is not None):
			result = int(pt.cuda.memory_reserved(gpuRamUsageDevice))
		return result

	def debugGetGpuRamMaxAllocatedUsageBytes():
		result = 0
		gpuRamUsageDevice = debugGetGpuRamUsageDevice()
		if(gpuRamUsageDevice is not None):
			result = int(pt.cuda.max_memory_allocated(gpuRamUsageDevice))
		return result

	def debugGetGpuRamMaxReservedUsageBytes():
		result = 0
		gpuRamUsageDevice = debugGetGpuRamUsageDevice()
		if(gpuRamUsageDevice is not None):
			result = int(pt.cuda.max_memory_reserved(gpuRamUsageDevice))
		return result

	def debugGetGpuRamUsageDevice():
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

	def debugGetCpuRamCurrentUsageBytes():
		result = None
		with open("/proc/self/status", "r") as processStatusFile:
			for processStatusLine in processStatusFile:
				if(processStatusLine.startswith("VmRSS:")):
					processStatusFields = processStatusLine.split()
					if(len(processStatusFields) < 2):
						raise RuntimeError("debugGetCpuRamCurrentUsageBytes error: VmRSS line is malformed")
					result = int(processStatusFields[1]) * 1024
		if(result is None):
			raise RuntimeError("debugGetCpuRamCurrentUsageBytes error: VmRSS not found in /proc/self/status")
		return result

	def debugConvertRamUsageBytesToGigabytesText(ramUsageBytes):
		if(ramUsageBytes < 0):
			raise RuntimeError("debugConvertRamUsageBytesToGigabytesText error: ramUsageBytes must be >= 0")
		result = f"{(float(ramUsageBytes) / (1024.0*1024.0*1024.0)):.4f}"
		return result

def resetTotalInferenceTokens():
	if(debugPrintTotalInferenceTokens):
		global totalInferenceTokensSeed
		global totalInferenceTokensPrediction
		global totalInferenceTokensAll
		totalInferenceTokensSeed = 0
		totalInferenceTokensPrediction = 0
		totalInferenceTokensAll = 0
	return

def addTotalInferenceTokens(seedTokensCount, predictionTokensCount):
	if(debugPrintTotalInferenceTokens):
		if(seedTokensCount is None or predictionTokensCount is None):
			raise RuntimeError("addTotalInferenceTokens error: token counts are None")
		if(seedTokensCount < 0 or predictionTokensCount < 0):
			raise RuntimeError("addTotalInferenceTokens error: token counts must be non-negative")
		global totalInferenceTokensSeed
		global totalInferenceTokensPrediction
		global totalInferenceTokensAll
		totalInferenceTokensSeed += int(seedTokensCount)
		totalInferenceTokensPrediction += int(predictionTokensCount)
		totalInferenceTokensAll += int(seedTokensCount) + int(predictionTokensCount)
	return

def printTotalInferenceTokens():
	if(debugPrintTotalInferenceTokens):
		print("debugPrintTotalInferenceTokens: seedPhaseTokens = ", totalInferenceTokensSeed, ", predictionPhaseTokens = ", totalInferenceTokensPrediction, ", totalInferenceTokens = ", totalInferenceTokensAll)
	return

def getDebugPrintTimeDatabaseLoadSaveTimesExecutionModeCount():
	result = 0
	if(executionMode=="inference"):
		result = 1
	elif(executionMode=="trainAndInference"):
		result = 2
	elif(executionMode=="train"):
		result = 1
	else:
		raise RuntimeError("getDebugPrintTimeDatabaseLoadSaveTimesExecutionModeCount error: unsupported executionMode = " + str(executionMode))
	return result

def getDebugPrintTimeDatabaseLoadSaveTimesHoursMinutesSecondsText(executionTimeSeconds):
	if(executionTimeSeconds < 0):
		raise RuntimeError("getDebugPrintTimeDatabaseLoadSaveTimesHoursMinutesSecondsText error: executionTimeSeconds must be >= 0")
	totalExecutionTimeSecondsInteger = int(executionTimeSeconds)
	executionHours = totalExecutionTimeSecondsInteger // 3600
	executionMinutes = (totalExecutionTimeSecondsInteger % 3600) // 60
	executionSeconds = totalExecutionTimeSecondsInteger % 60
	result = f"{executionHours:02d}:{executionMinutes:02d}:{executionSeconds:02d}"
	return result

def getDebugPrintTimeDatabaseLoadSaveTimesText(executionTimeSeconds):
	if(executionTimeSeconds < 0):
		raise RuntimeError("getDebugPrintTimeDatabaseLoadSaveTimesText error: executionTimeSeconds must be >= 0")
	executionTimeHoursMinutesSecondsText = getDebugPrintTimeDatabaseLoadSaveTimesHoursMinutesSecondsText(executionTimeSeconds)
	result = executionTimeHoursMinutesSecondsText + " [" + f"{executionTimeSeconds:.6f}" + " seconds]"
	return result

def printDebugPrintTimeDatabaseLoadSaveTimesEntry(executionTimeName, executionTimeSeconds):
	executionTimeText = getDebugPrintTimeDatabaseLoadSaveTimesText(executionTimeSeconds)
	print(executionTimeName + ": " + executionTimeText)
	return

def printDebugPrintTimeDatabaseLoadSaveTimesSummary(summaryName, totalExecutionTimeSeconds, loadAllObservedColumnsToRamExecutionTimeSeconds, saveAllObservedColumnsToDiskExecutionTimeSeconds):
	processingExecutionTimeSeconds = totalExecutionTimeSeconds - loadAllObservedColumnsToRamExecutionTimeSeconds - saveAllObservedColumnsToDiskExecutionTimeSeconds
	if(processingExecutionTimeSeconds < 0):
		raise RuntimeError("printDebugPrintTimeDatabaseLoadSaveTimesSummary error: processingExecutionTimeSeconds must be >= 0")
	print(summaryName)
	printDebugPrintTimeDatabaseLoadSaveTimesEntry("total execution time", totalExecutionTimeSeconds)
	printDebugPrintTimeDatabaseLoadSaveTimesEntry("loadAllObservedColumnsToRam execution time", loadAllObservedColumnsToRamExecutionTimeSeconds)
	printDebugPrintTimeDatabaseLoadSaveTimesEntry("saveAllObservedColumnsToDisk execution time", saveAllObservedColumnsToDiskExecutionTimeSeconds)
	printDebugPrintTimeDatabaseLoadSaveTimesEntry("processing time", processingExecutionTimeSeconds)
	return

def debugTrainSectionTimesReset(databaseNetworkObject, sequenceCount):
	if(debugPrintTrainSectionTimes):
		databaseNetworkObject.debugTrainSectionTimes = {}
		databaseNetworkObject.debugTrainSectionSequenceCount = sequenceCount
		if(debugPrintTrainSectionTimesSourceFeatureConnections):
			databaseNetworkObject.debugTrainSectionTimesContextStack = []
	return

def debugTrainSectionTimesAdd(databaseNetworkObject, sectionName, sectionDuration):
	if(debugPrintTrainSectionTimes):
		if(not hasattr(databaseNetworkObject, "debugTrainSectionTimes")):
			databaseNetworkObject.debugTrainSectionTimes = {}
		currentDuration = databaseNetworkObject.debugTrainSectionTimes.get(sectionName, 0.0)
		databaseNetworkObject.debugTrainSectionTimes[sectionName] = currentDuration + sectionDuration
	return

def debugTrainSectionTimesPrint(databaseNetworkObject):
	if(debugPrintTrainSectionTimes):
		sequenceCountDebug = getattr(databaseNetworkObject, "debugTrainSectionSequenceCount", -1)
		print(f"debugTrainSectionTimes: sequenceCount={sequenceCountDebug}")
		for sectionName, sectionDuration in databaseNetworkObject.debugTrainSectionTimes.items():
			print(f"\t{sectionName}: {sectionDuration:.6f}s")
	return

def debugTrainSectionTimesContextPush(databaseNetworkObject, contextName):
	if(debugPrintTrainSectionTimesSourceFeatureConnections):
		if(not debugPrintTrainSectionTimes):
			raise RuntimeError("debugTrainSectionTimesContextPush error: debugPrintTrainSectionTimes must be enabled")
		if(contextName is None or contextName == ""):
			raise RuntimeError("debugTrainSectionTimesContextPush error: contextName is invalid")
		if(not hasattr(databaseNetworkObject, "debugTrainSectionTimesContextStack")):
			databaseNetworkObject.debugTrainSectionTimesContextStack = []
		databaseNetworkObject.debugTrainSectionTimesContextStack.append(contextName)
	return

def debugTrainSectionTimesContextPop(databaseNetworkObject):
	result = None
	if(debugPrintTrainSectionTimesSourceFeatureConnections):
		if(not debugPrintTrainSectionTimes):
			raise RuntimeError("debugTrainSectionTimesContextPop error: debugPrintTrainSectionTimes must be enabled")
		if(not hasattr(databaseNetworkObject, "debugTrainSectionTimesContextStack")):
			raise RuntimeError("debugTrainSectionTimesContextPop error: debugTrainSectionTimesContextStack is not initialised")
		if(len(databaseNetworkObject.debugTrainSectionTimesContextStack) == 0):
			raise RuntimeError("debugTrainSectionTimesContextPop error: debugTrainSectionTimesContextStack is empty")
		result = databaseNetworkObject.debugTrainSectionTimesContextStack.pop()
	return result

def debugTrainSectionTimesContextGet(databaseNetworkObject):
	result = None
	if(debugPrintTrainSectionTimesSourceFeatureConnections):
		contextStack = getattr(databaseNetworkObject, "debugTrainSectionTimesContextStack", None)
		if(contextStack is not None):
			if(len(contextStack) > 0):
				result = contextStack[-1]
	return result

def getSourceFeatureConnectionsDebugSectionName(databaseNetworkObject, operationName):
	result = None
	if(debugPrintTrainSectionTimes and debugPrintTrainSectionTimesSourceFeatureConnections):
		debugSectionContext = debugTrainSectionTimesContextGet(databaseNetworkObject)
		if(debugSectionContext is not None):
			result = f"{debugSectionContext}/{operationName}"
	return result

def debugPrintConnectionSamples(label, indices, values, maxSamples=5):
	numEntries = indices.shape[1]
	print(f"\tsequenceObservedColumns debug: {label}: entries={numEntries}")
	if(numEntries > 0):
		sampleCount = min(numEntries, maxSamples)
		for entryIndex in range(sampleCount):
			indexTuple = indices[:, entryIndex].tolist()
			value = float(values[entryIndex].item())
			print(f"\tsequenceObservedColumns debug:\tindices={indexTuple}, value={value}")
	return

def debugPrintPersistedColumnSummary(sequenceObservedColumns, mode):
	databaseNetworkObject = sequenceObservedColumns.databaseNetworkObject
	debugEnabled = debugPrintSequenceObservedColumnsConnections
	if(debugEnabled):
		targetLemma = "movement"
		observedColumn = sequenceObservedColumns.observedColumnsDict.get(targetLemma)
		if(observedColumn is None):
			print(f"debugPersistedColumn ({mode}): lemma '{targetLemma}' not in observedColumnsDict")
		else:
			featureConnections = observedColumn.materialiseFeatureConnections(loadAllStored=True, targetDevice=deviceSparse)
			connectionsStrength = featureConnections[databaseNetworkObject.arrayIndexPropertiesStrengthIndex]
			if(connectionsStrength.is_sparse):
				connectionsStrength = connectionsStrength.to_dense()
			outgoingCount = 0
			maxStrength = 0.0
			internalCount = 0
			externalCount = 0
			segmentCounts = None
			lastSegmentCount = 0
			targetTop = []
			if(connectionsStrength.numel() > 0):
				outgoingMask = connectionsStrength > 0
				outgoingCount = int(outgoingMask.sum().item())
				maxStrength = float(connectionsStrength.max().item())
				internalMask = outgoingMask[:, :, :, observedColumn.conceptIndex, :]
				internalCount = int(internalMask.sum().item())
				externalCount = outgoingCount - internalCount
				segmentCounts = outgoingMask.sum(dim=(0, 2, 3, 4)).to("cpu").tolist()
				lastSegmentCount = int(outgoingMask[:, arrayIndexSegmentLast].sum().item())
				targetCounts = outgoingMask.sum(dim=(0, 1, 2, 4))
				if(targetCounts.numel() > 0):
					topkCount = min(3, int(targetCounts.shape[0]))
					topkValues, topkIndices = pt.topk(targetCounts, topkCount)
					targetTop = [(int(idx.item()), int(val.item())) for idx, val in zip(topkIndices, topkValues) if int(val.item()) > 0]
			print(f"debugPersistedColumn ({mode}): lemma={targetLemma}, outgoing>0={outgoingCount}, maxStrength={maxStrength}, internal>0={internalCount}, external>0={externalCount}, lastSegment>0={lastSegmentCount}")
			if(segmentCounts is not None):
				print(f"\tsegmentCounts>0={segmentCounts}")
			if(len(targetTop) > 0):
				targetLabels = []
				for targetIndex, targetCount in targetTop:
					targetLemma = "<unknown>"
					if(targetIndex < len(observedColumn.databaseNetworkObject.conceptColumnsList)):
						targetLemma = observedColumn.databaseNetworkObject.conceptColumnsList[targetIndex]
					targetLabels.append(f"{targetLemma}:{targetCount}")
				print(f"\ttopTargets={targetLabels}")
	return

def debugDescribeColumnFeatureName(databaseNetworkObject, columnIndex, featureIndex):
	if(0 <= columnIndex < len(databaseNetworkObject.conceptColumnsList)):
		columnName = databaseNetworkObject.conceptColumnsList[columnIndex]
	else:
		columnName = f"<invalid:{columnIndex}>"
	if(featureIndex == featureIndexPrimeConceptNeuron):
		featureName = "conceptNeuron"
	elif(0 <= featureIndex < len(databaseNetworkObject.conceptFeaturesList)):
		featureName = databaseNetworkObject.conceptFeaturesList[featureIndex]
	else:
		featureName = f"feature_{featureIndex}"
	return columnName, featureName

def debugDescribeAllowedBeamFeatures(databaseNetworkObject, connectedColumnsTensor, connectedColumnsFeatures):
	result = None
	if(connectedColumnsTensor is None):
		result = "<none>"
	elif(connectedColumnsTensor.numel() == 0):
		result = "[]"
	else:
		elements = []
		for columnValue in connectedColumnsTensor.cpu().tolist():
			columnName, _ = debugDescribeColumnFeatureName(databaseNetworkObject, columnValue, featureIndexPrimeConceptNeuron)
			if(connectedColumnsFeatures is not None and columnValue in connectedColumnsFeatures):
				featureList = []
				for featureIndex in sorted(connectedColumnsFeatures[columnValue]):
					_, featureName = debugDescribeColumnFeatureName(databaseNetworkObject, columnValue, featureIndex)
				featureList.append(f"{featureIndex}:{featureName}")
				elements.append(f"{columnName} -> [{', '.join(featureList)}]")
			else:
				elements.append(f"{columnName} -> <any feature>")
		result = "[" + "; ".join(elements) + "]"
	return result

def buildIndexListFromDict(indexDict, dictName):
	resultList = []
	if(debugLimitFeatures):
		maxIndex = -1
		for index in indexDict.values():
			if(index > maxIndex):
				maxIndex = index
		if(maxIndex >= 0):
			resultList = [None] * (maxIndex + 1)
			for name, index in indexDict.items():
				if(index < 0):
					raise RuntimeError(f"{dictName} index < 0")
				if(index >= len(resultList)):
					raise RuntimeError(f"{dictName} index out of bounds")
				if(resultList[index] is not None):
					raise RuntimeError(f"{dictName} duplicate index {index}")
				resultList[index] = name
			for name in resultList:
				if(name is None):
					raise RuntimeError(f"{dictName} missing index entry")
	return resultList

def applyDebugLimitIndexDict(indexDict, maxCount, dictName):
	resultDict = indexDict
	if(debugLimitFeatures):
		if(maxCount <= 0):
			raise RuntimeError(f"{dictName} maxCount must be > 0")
		if(len(indexDict) > maxCount):
			trimmedDict = {}
			for name, index in indexDict.items():
				if(index < 0):
					raise RuntimeError(f"{dictName} index < 0")
				if(index < maxCount):
					trimmedDict[name] = index
			resultDict = trimmedDict
	return resultDict

def applyDebugLimitList(listObject, maxCount, listName):
	resultList = listObject
	if(debugLimitFeatures):
		if(maxCount <= 0):
			raise RuntimeError(f"{listName} maxCount must be > 0")
		if(len(listObject) < maxCount):
			raise RuntimeError(f"{listName} length {len(listObject)} < expected {maxCount}")
		if(len(listObject) > maxCount):
			resultList = listObject[:maxCount]
	return resultList

def applyDebugLimitGlobalFeatureNeuronsTensor(tensor, cLimit, fLimit, tensorName):
	result = tensor
	if(debugLimitFeatures):
		if(cLimit <= 0 or fLimit <= 0):
			raise RuntimeError(f"{tensorName} debug limit requires positive limits")
		capC = tensor.size(3)
		capF = tensor.size(4)
		if(capC > cLimit):
			capC = cLimit
		if(capF > fLimit):
			capF = fLimit
		if(capC != tensor.size(3) or capF != tensor.size(4)):
			if(tensor.is_sparse):
				tensor = tensor.coalesce()
				indices = tensor.indices()
				values = tensor.values()
				mask = (indices[3] < capC) & (indices[4] < capF)
				indices = indices[:, mask]
				values = values[mask]
				newSize = list(tensor.size())
				newSize[3] = capC
				newSize[4] = capF
				result = pt.sparse_coo_tensor(indices, values, size=newSize, dtype=tensor.dtype, device=tensor.device).coalesce()
			else:
				result = tensor[:, :, :, :capC, :capF]
	return result

def applyDebugLimitFeatureConnectionsTensor(tensor, cLimit, fLimit, tensorName):
	result = tensor
	if(debugLimitFeatures):
		if(cLimit <= 0 or fLimit <= 0):
			raise RuntimeError(f"{tensorName} debug limit requires positive limits")
		if(tensor.size(3) != tensor.size(5)):
			raise RuntimeError(f"{tensorName} feature dimension mismatch: {tensor.size(3)} vs {tensor.size(5)}")
		capF = tensor.size(3)
		capC = tensor.size(4)
		if(capF > fLimit):
			capF = fLimit
		if(capC > cLimit):
			capC = cLimit
		if(capF != tensor.size(3) or capC != tensor.size(4)):
			if(tensor.is_sparse):
				tensor = tensor.coalesce()
				indices = tensor.indices()
				values = tensor.values()
				mask = (indices[3] < capF) & (indices[4] < capC) & (indices[5] < capF)
				indices = indices[:, mask]
				values = values[mask]
				newSize = list(tensor.size())
				newSize[3] = capF
				newSize[4] = capC
				newSize[5] = capF
				result = pt.sparse_coo_tensor(indices, values, size=newSize, dtype=tensor.dtype, device=tensor.device).coalesce()
			else:
				result = tensor[:, :, :, :capF, :capC, :capF]
	return result

def applyDebugLimitFeatureConnectionsSourceTensor(tensor, cLimit, fLimit, tensorName):
	result = tensor
	if(debugLimitFeatures):
		if(cLimit <= 0 or fLimit <= 0):
			raise RuntimeError(f"{tensorName} debug limit requires positive limits")
		capC = tensor.size(3)
		capF = tensor.size(4)
		if(capC > cLimit):
			capC = cLimit
		if(capF > fLimit):
			capF = fLimit
		if(capC != tensor.size(3) or capF != tensor.size(4)):
			if(tensor.is_sparse):
				tensor = tensor.coalesce()
				indices = tensor.indices()
				values = tensor.values()
				mask = (indices[3] < capC) & (indices[4] < capF)
				indices = indices[:, mask]
				values = values[mask]
				newSize = list(tensor.size())
				newSize[3] = capC
				newSize[4] = capF
				result = pt.sparse_coo_tensor(indices, values, size=newSize, dtype=tensor.dtype, device=tensor.device).coalesce()
			else:
				result = tensor[:, :, :, :capC, :capF]
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

def applyDebugLimitFeatureIndexMaps(featureWordToIndex, featureIndexToWord, fLimit, mapName):
	resultFeatureWordToIndex = featureWordToIndex
	resultFeatureIndexToWord = featureIndexToWord
	if(debugLimitFeatures):
		if(fLimit <= 0):
			raise RuntimeError(f"{mapName} debug limit requires positive limits")
		trimmedWordToIndex = {}
		trimmedIndexToWord = {}
		for word, index in featureWordToIndex.items():
			if(index < 0):
				raise RuntimeError(f"{mapName} index < 0")
			if(index < fLimit):
				trimmedWordToIndex[word] = index
		for index, word in featureIndexToWord.items():
			if(index < 0):
				raise RuntimeError(f"{mapName} index < 0")
			if(index < fLimit):
				trimmedIndexToWord[index] = word
		for word, index in trimmedWordToIndex.items():
			if(trimmedIndexToWord.get(index) != word):
				raise RuntimeError(f"{mapName} mismatch for index {index}")
		for index, word in trimmedIndexToWord.items():
			if(trimmedWordToIndex.get(word) != index):
				raise RuntimeError(f"{mapName} mismatch for word {word}")
		resultFeatureWordToIndex = trimmedWordToIndex
		resultFeatureIndexToWord = trimmedIndexToWord
	return resultFeatureWordToIndex, resultFeatureIndexToWord

def debugCountObservedColumnConnections(databaseNetworkObject, conceptIndex, lemma, columnIndex):
	import GIAANNproto_databaseNetworkFiles
	columnConnections = 0
	if(not GIAANNproto_databaseNetworkFiles.observedColumnMetadataExists(conceptIndex)):
		columnConnections = 0
	else:
		sourceFeatureIndices = GIAANNproto_databaseNetworkFiles.listObservedColumnSourceFeatureIndices(conceptIndex)
		for sourceFeatureIndex in sourceFeatureIndices:
			featureConnections = GIAANNproto_databaseNetworkFiles.loadObservedColumnSourceFeatureConnectionsTensor(databaseNetworkObject, conceptIndex, sourceFeatureIndex, deviceDatabase)
			if(featureConnections is None):
				raise RuntimeError("debugCountObservedColumnConnections error: featureConnections is None for conceptIndex = " + str(conceptIndex) + ", sourceFeatureIndex = " + str(sourceFeatureIndex))
			if(databaseNetworkObject.arrayIndexPropertiesStrengthIndex < 0 or databaseNetworkObject.arrayIndexPropertiesStrengthIndex >= featureConnections.shape[0]):
				raise RuntimeError("debugCountObservedColumnConnections error: databaseNetworkObject.arrayIndexPropertiesStrengthIndex out of range")
			columnConnections += debugCountNonZero(featureConnections)
			del featureConnections
	return columnConnections

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
				totalPtBytesUncompressed += os.path.getsize(filePath)
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
			totalDatabaseBytesUncompressed += os.path.getsize(filePath)
	totalDatabaseGiB = totalDatabaseBytesUncompressed / (1024 ** 3)
	return totalDatabaseGiB

def debugCountNonZero(t):
	result = 0
	if isinstance(t, pt.Tensor):
		if t.is_sparse or (hasattr(t, "layout") and t.layout in {pt.sparse_coo, pt.sparse_csr, pt.sparse_csc, pt.sparse_bsr, pt.sparse_bsc}):
			result = int(t._nnz())
		else:
			result = int(pt.count_nonzero(t).item())
	return result
