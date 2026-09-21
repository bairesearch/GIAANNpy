"""Opt-in CPU merges for RAM-resident training connections; neuron updates are unchanged."""

from GIAANNcmn_globalDefs import *

if(optimiseParallelisation1):
	import os
	from torch.utils.cpp_extension import load
	_cpuConnectionUpdateExtension = None
if(optimiseParallelisation2a or optimiseParallelisation2b or optimiseParallelisation2c or optimiseParallelisation2d or optimiseParallelisation2e):
	import GIAANNcmn_cpuParallelisation2


def updateConnectionSources(sequenceObservedColumns, observedColumnsByConceptIndex, connectionIndices, connectionValues, featureIndicesInObserved, conceptIndicesTensor):
	if(optimiseParallelisation1):
		if(connectionIndices.numel() > 0):
			database = sequenceObservedColumns.databaseNetworkObject
			if(optimiseParallelisation2b):
				sourceSize = (database.arrayNumberOfProperties, multipleDendriticBranchesNumber, arrayNumberOfSegments, database.c, database.f)
				sourceKeysUnique, updates = GIAANNcmn_cpuParallelisation2.prepareConnectionUpdates(connectionIndices, connectionValues, featureIndicesInObserved, conceptIndicesTensor, sourceSize, database.arrayIndexPropertiesStrengthIndex)
			else:
				sourceKeys = sequenceObservedColumns.buildConnectionSourceCombinedKeys(connectionIndices, featureIndicesInObserved, conceptIndicesTensor)
				sourceKeysUnique = pt.unique(sourceKeys, sorted=True)
				sourceSize = (database.arrayNumberOfProperties, multipleDendriticBranchesNumber, arrayNumberOfSegments, database.c, database.f)
				updateSize = sourceSize[:parallelisation1SourceBucketDimension] + (sourceKeysUnique.numel(),) + sourceSize[parallelisation1SourceBucketDimension:]
				updates = sequenceObservedColumns.buildConnectionSourceBucketUpdateSparse(connectionIndices, connectionValues, database.arrayIndexPropertiesStrengthIndex, featureIndicesInObserved, conceptIndicesTensor, sourceKeysUnique, updateSize).coalesce()
			sourceReferences = []
			existingSources = []
			for sourceKey in sourceKeysUnique.tolist():
				conceptIndex, sourceFeatureIndex = divmod(sourceKey, database.f)
				observedColumn = observedColumnsByConceptIndex[conceptIndex]
				sourceReferences.append((observedColumn, sourceFeatureIndex))
				existingSources.append(observedColumn.getFeatureConnectionsForSourceFeature(sourceFeatureIndex, targetDevice=updates.device, createMissing=False).coalesce())
			updatedSources = mergeConnectionSources(existingSources, updates, sourceSize)
			# Finish the entire native batch before publishing any updated connection tensor.
			for (observedColumn, sourceFeatureIndex), updatedSource in zip(sourceReferences, updatedSources):
				observedColumn.setFeatureConnectionsForSourceFeature(sourceFeatureIndex, updatedSource)
	return


def mergeConnectionSources(existingSources, updates, sourceSize):
	result = None
	if(optimiseParallelisation1):
		if(len(sourceSize) != parallelisation1SourceTensorRank):
			raise RuntimeError("mergeConnectionSources requires rank-five source tensors")
		if(optimiseParallelisation2c or optimiseParallelisation2d):
			result = GIAANNcmn_cpuParallelisation2.mergeSparseSources(existingSources, updates, sourceSize, parallelisation1SourceBucketDimension)
		else:
			extension = getCPUConnectionUpdateExtension()
			result = extension.mergeConnectionSources(optimiseParallelisation1, existingSources, updates, list(sourceSize), parallelisation1SourceBucketDimension, parallelisation1MergeGrainSize)
	return result


def getCPUConnectionUpdateExtension():
	global _cpuConnectionUpdateExtension
	result = None
	if(optimiseParallelisation1):
		if(_cpuConnectionUpdateExtension is None):
			_cpuConnectionUpdateExtension = load(name=parallelisation1ExtensionName, sources=[os.path.join(os.path.dirname(__file__), parallelisation1ExtensionSource)], extra_cflags=parallelisation1CompilerFlags, extra_ldflags=parallelisation1LinkerFlags, with_cuda=False, verbose=False)
		result = _cpuConnectionUpdateExtension
	return result
