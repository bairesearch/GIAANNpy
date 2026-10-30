"""Exact, sequence-ordered sparse connection updates over immutable historical COO tensors."""

from GIAANNcmn_globalDefs import *

if(optimiseTrainConnectionOverlay):
	import os
	from torch.utils.cpp_extension import load
	_trainConnectionOverlayExtension = None


def updateConnectionSources(sourceReferences, updates, sourceSize):
	if(optimiseTrainConnectionOverlay):
		extension = getExtension()
		overlays = []
		missingPositions = []
		missingSources = []
		for position, (observedColumn, sourceFeatureIndex) in enumerate(sourceReferences):
			if(not hasattr(observedColumn, "trainConnectionOverlays")):
				observedColumn.trainConnectionOverlays = {}
			overlay = observedColumn.trainConnectionOverlays.get(sourceFeatureIndex)
			if(overlay is None):
				source = observedColumn.getFeatureConnectionsForSourceFeature(sourceFeatureIndex, targetDevice=updates.device, createMissing=False).coalesce()
				missingPositions.append(position)
				missingSources.append(source)
			overlays.append(overlay)
		if(missingSources):
			created = extension.createConnectionOverlays(optimiseTrainConnectionOverlay, missingSources, trainConnectionOverlayMinimumEntries, trainConnectionOverlayCompactionFraction, trainConnectionOverlayGrainSize, trainConnectionOverlayInvalidInput)
			for position, overlay in zip(missingPositions, created):
				observedColumn, sourceFeatureIndex = sourceReferences[position]
				observedColumn.trainConnectionOverlays[sourceFeatureIndex] = overlay
				overlays[position] = overlay
		bases = extension.updateConnectionOverlays(optimiseTrainConnectionOverlay, overlays, updates, list(sourceSize), trainConnectionOverlayGrainSize, trainConnectionOverlayInvalidInput)
		for (observedColumn, sourceFeatureIndex), base in zip(sourceReferences, bases):
			observedColumn.featureConnectionsBySourceFeature[sourceFeatureIndex] = base
	return


def materialiseDatabase(databaseNetworkObject):
	if(optimiseTrainConnectionOverlay):
		sourceReferences = []
		for observedColumn in databaseNetworkObject.observedColumnsDictRAM.values():
			if(hasattr(observedColumn, "trainConnectionOverlays")):
				for sourceFeatureIndex in observedColumn.trainConnectionOverlays:
					sourceReferences.append((observedColumn, sourceFeatureIndex))
		if(sourceReferences):
			sourceSize = sourceReferences[0][0].getFeatureConnectionsTargetSize()
			extension = getExtension()
			for start in range(0, len(sourceReferences), trainConnectionOverlayMaterialiseBatchSize):
				batchReferences = sourceReferences[start:start + trainConnectionOverlayMaterialiseBatchSize]
				overlays = [observedColumn.trainConnectionOverlays[sourceFeatureIndex] for observedColumn, sourceFeatureIndex in batchReferences]
				tensors = extension.materialiseConnectionOverlays(optimiseTrainConnectionOverlay, overlays, list(sourceSize), trainConnectionOverlayGrainSize, trainConnectionOverlayInvalidInput)
				for (observedColumn, sourceFeatureIndex), tensor in zip(batchReferences, tensors):
					observedColumn.featureConnectionsBySourceFeature[sourceFeatureIndex] = tensor
					del observedColumn.trainConnectionOverlays[sourceFeatureIndex]
	return


def materialiseSource(observedColumn, sourceFeatureIndex):
	if(optimiseTrainConnectionOverlay):
		if(hasattr(observedColumn, "trainConnectionOverlays") and sourceFeatureIndex in observedColumn.trainConnectionOverlays):
			overlay = observedColumn.trainConnectionOverlays[sourceFeatureIndex]
			tensor = overlay.materialise(list(observedColumn.getFeatureConnectionsTargetSize()))
			observedColumn.featureConnectionsBySourceFeature[sourceFeatureIndex] = tensor
			del observedColumn.trainConnectionOverlays[sourceFeatureIndex]
	return


def discardSource(observedColumn, sourceFeatureIndex):
	if(optimiseTrainConnectionOverlay):
		if(hasattr(observedColumn, "trainConnectionOverlays")):
			observedColumn.trainConnectionOverlays.pop(sourceFeatureIndex, None)
	return


def getExtension():
	global _trainConnectionOverlayExtension
	result = None
	if(optimiseTrainConnectionOverlay):
		if(_trainConnectionOverlayExtension is None):
			_trainConnectionOverlayExtension = load(name=trainConnectionOverlayExtensionName, sources=[os.path.join(os.path.dirname(__file__), trainConnectionOverlayExtensionSource)], extra_cflags=trainConnectionOverlayCompilerFlags, extra_ldflags=trainConnectionOverlayLinkerFlags, with_cuda=False, verbose=False)
		result = _trainConnectionOverlayExtension
	return result
